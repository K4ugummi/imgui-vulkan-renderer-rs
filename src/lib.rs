//! Vulkan renderer backend for [imgui-rs](https://docs.rs/imgui) using
//! [ash](https://docs.rs/ash).
//!
//! # Usage
//!
//! ```rust,ignore
//! use imgui_vulkan_renderer_rs::{Renderer, RendererCreateInfo};
//!
//! let create_info = RendererCreateInfo {
//!     device: device.clone(),
//!     memory_properties: mem_props,
//!     render_pass,
//!     command_pool,
//!     queue,
//! };
//! let mut renderer = Renderer::new(&mut imgui, &create_info)?;
//!
//! // In your render loop, after building the UI:
//! let draw_data = imgui.render();
//! renderer.render(draw_data, command_buffer)?;
//! ```

use ash::util::read_spv;
use ash::vk;
use ash::vk::Handle;
use imgui::internal::RawWrapper;
use imgui::{Context, DrawCmd, DrawData, DrawIdx, DrawVert, TextureId};
use std::io::Cursor;
use std::mem;

/// Errors that can occur during renderer operations.
#[derive(Debug)]
pub enum RendererError {
    /// A Vulkan API call returned an error.
    Vulkan(vk::Result),
}

impl std::fmt::Display for RendererError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            RendererError::Vulkan(e) => write!(f, "Vulkan error: {e}"),
        }
    }
}

impl std::error::Error for RendererError {}

impl From<vk::Result> for RendererError {
    fn from(e: vk::Result) -> Self {
        RendererError::Vulkan(e)
    }
}

/// Parameters required to create a [`Renderer`].
pub struct RendererCreateInfo {
    /// A cloned `ash::Device` handle. The renderer stores its own clone.
    pub device: ash::Device,
    /// Physical device memory properties, used for buffer/image allocation.
    pub memory_properties: vk::PhysicalDeviceMemoryProperties,
    /// The render pass the imgui pipeline will be compatible with (subpass 0).
    pub render_pass: vk::RenderPass,
    /// A command pool for one-shot upload commands (font texture).
    pub command_pool: vk::CommandPool,
    /// A queue that supports graphics and transfer operations.
    pub queue: vk::Queue,
}

/// Vulkan renderer for Dear ImGui draw data.
pub struct Renderer {
    device: ash::Device,
    memory_properties: vk::PhysicalDeviceMemoryProperties,
    pipeline_layout: vk::PipelineLayout,
    pipeline: vk::Pipeline,
    descriptor_set_layout: vk::DescriptorSetLayout,
    descriptor_pool: vk::DescriptorPool,
    #[allow(dead_code)] // Kept alive; freed when descriptor_pool is destroyed.
    font_descriptor_set: vk::DescriptorSet,
    font_image: vk::Image,
    font_image_memory: vk::DeviceMemory,
    font_image_view: vk::ImageView,
    font_sampler: vk::Sampler,
    vertex_buffer: vk::Buffer,
    vertex_buffer_memory: vk::DeviceMemory,
    vertex_buffer_size: vk::DeviceSize,
    index_buffer: vk::Buffer,
    index_buffer_memory: vk::DeviceMemory,
    index_buffer_size: vk::DeviceSize,
    /// Old buffers waiting for the GPU to finish using them before destruction.
    retired_buffers: Vec<(vk::Buffer, vk::DeviceMemory)>,
}

const INITIAL_BUFFER_SIZE: vk::DeviceSize = 64 * 1024;

impl Renderer {
    /// Create a new Vulkan renderer.
    ///
    /// Builds the graphics pipeline, uploads the font atlas, and allocates
    /// initial vertex/index buffers. The `command_pool` and `queue` in
    /// `create_info` are used for a one-shot command buffer to upload the font
    /// texture; they are not stored.
    pub fn new(imgui: &mut Context, create_info: &RendererCreateInfo) -> Result<Self, RendererError> {
        let device = &create_info.device;
        let memory_properties = create_info.memory_properties;

        // --- Descriptor set layout ---
        let sampler_binding = vk::DescriptorSetLayoutBinding::default()
            .binding(0)
            .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::FRAGMENT);
        let ds_layout_info = vk::DescriptorSetLayoutCreateInfo::default()
            .bindings(std::slice::from_ref(&sampler_binding));
        let descriptor_set_layout =
            unsafe { device.create_descriptor_set_layout(&ds_layout_info, None)? };

        // --- Pipeline layout ---
        let push_constant_range = vk::PushConstantRange {
            stage_flags: vk::ShaderStageFlags::VERTEX,
            offset: 0,
            size: 64, // mat4
        };
        let pipeline_layout_info = vk::PipelineLayoutCreateInfo::default()
            .set_layouts(std::slice::from_ref(&descriptor_set_layout))
            .push_constant_ranges(std::slice::from_ref(&push_constant_range));
        let pipeline_layout = unsafe {
            device
                .create_pipeline_layout(&pipeline_layout_info, None)
                .inspect_err(|_| {
                    device.destroy_descriptor_set_layout(descriptor_set_layout, None);
                })?
        };

        // --- Shader modules ---
        let vert_spv = read_spv(&mut Cursor::new(
            &include_bytes!("shaders/imgui.vert.spv")[..],
        ))
        .expect("failed to read vertex shader SPIR-V");
        let frag_spv = read_spv(&mut Cursor::new(
            &include_bytes!("shaders/imgui.frag.spv")[..],
        ))
        .expect("failed to read fragment shader SPIR-V");

        let vert_module_info = vk::ShaderModuleCreateInfo::default().code(&vert_spv);
        let frag_module_info = vk::ShaderModuleCreateInfo::default().code(&frag_spv);

        let vert_module = unsafe {
            device
                .create_shader_module(&vert_module_info, None)
                .inspect_err(|_| {
                    device.destroy_pipeline_layout(pipeline_layout, None);
                    device.destroy_descriptor_set_layout(descriptor_set_layout, None);
                })?
        };
        let frag_module = unsafe {
            device
                .create_shader_module(&frag_module_info, None)
                .inspect_err(|_| {
                    device.destroy_shader_module(vert_module, None);
                    device.destroy_pipeline_layout(pipeline_layout, None);
                    device.destroy_descriptor_set_layout(descriptor_set_layout, None);
                })?
        };

        let entry_point = c"main";

        let shader_stages = [
            vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::VERTEX)
                .module(vert_module)
                .name(entry_point),
            vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::FRAGMENT)
                .module(frag_module)
                .name(entry_point),
        ];

        // --- Vertex input ---
        let binding_desc = vk::VertexInputBindingDescription {
            binding: 0,
            stride: mem::size_of::<DrawVert>() as u32,
            input_rate: vk::VertexInputRate::VERTEX,
        };
        let attribute_descs = [
            // Position: vec2 at offset 0
            vk::VertexInputAttributeDescription {
                location: 0,
                binding: 0,
                format: vk::Format::R32G32_SFLOAT,
                offset: 0,
            },
            // UV: vec2 at offset 8
            vk::VertexInputAttributeDescription {
                location: 1,
                binding: 0,
                format: vk::Format::R32G32_SFLOAT,
                offset: 8,
            },
            // Color: u32 packed RGBA at offset 16
            vk::VertexInputAttributeDescription {
                location: 2,
                binding: 0,
                format: vk::Format::R8G8B8A8_UNORM,
                offset: 16,
            },
        ];
        let vertex_input_info = vk::PipelineVertexInputStateCreateInfo::default()
            .vertex_binding_descriptions(std::slice::from_ref(&binding_desc))
            .vertex_attribute_descriptions(&attribute_descs);

        let input_assembly = vk::PipelineInputAssemblyStateCreateInfo::default()
            .topology(vk::PrimitiveTopology::TRIANGLE_LIST);

        let viewport_state = vk::PipelineViewportStateCreateInfo::default()
            .viewport_count(1)
            .scissor_count(1);

        let rasterization = vk::PipelineRasterizationStateCreateInfo::default()
            .polygon_mode(vk::PolygonMode::FILL)
            .cull_mode(vk::CullModeFlags::NONE)
            .front_face(vk::FrontFace::COUNTER_CLOCKWISE)
            .line_width(1.0);

        let multisample = vk::PipelineMultisampleStateCreateInfo::default()
            .rasterization_samples(vk::SampleCountFlags::TYPE_1);

        let color_blend_attachment = vk::PipelineColorBlendAttachmentState::default()
            .blend_enable(true)
            .src_color_blend_factor(vk::BlendFactor::SRC_ALPHA)
            .dst_color_blend_factor(vk::BlendFactor::ONE_MINUS_SRC_ALPHA)
            .color_blend_op(vk::BlendOp::ADD)
            .src_alpha_blend_factor(vk::BlendFactor::ONE)
            .dst_alpha_blend_factor(vk::BlendFactor::ONE_MINUS_SRC_ALPHA)
            .alpha_blend_op(vk::BlendOp::ADD)
            .color_write_mask(vk::ColorComponentFlags::RGBA);

        let color_blend = vk::PipelineColorBlendStateCreateInfo::default()
            .attachments(std::slice::from_ref(&color_blend_attachment));

        let dynamic_states = [vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR];
        let dynamic_state =
            vk::PipelineDynamicStateCreateInfo::default().dynamic_states(&dynamic_states);

        let depth_stencil = vk::PipelineDepthStencilStateCreateInfo::default();

        let pipeline_info = vk::GraphicsPipelineCreateInfo::default()
            .stages(&shader_stages)
            .vertex_input_state(&vertex_input_info)
            .input_assembly_state(&input_assembly)
            .viewport_state(&viewport_state)
            .rasterization_state(&rasterization)
            .multisample_state(&multisample)
            .depth_stencil_state(&depth_stencil)
            .color_blend_state(&color_blend)
            .dynamic_state(&dynamic_state)
            .layout(pipeline_layout)
            .render_pass(create_info.render_pass)
            .subpass(0);

        let pipeline = unsafe {
            let result = device
                .create_graphics_pipelines(vk::PipelineCache::null(), &[pipeline_info], None)
                .map_err(|(_, e)| e);

            // Shader modules are no longer needed regardless of success/failure.
            device.destroy_shader_module(vert_module, None);
            device.destroy_shader_module(frag_module, None);

            result.inspect_err(|_| {
                device.destroy_pipeline_layout(pipeline_layout, None);
                device.destroy_descriptor_set_layout(descriptor_set_layout, None);
            })?[0]
        };

        // From here on, if anything fails we need to clean up pipeline +
        // layout + descriptor set layout (plus whatever else was created).
        // Use a helper closure to avoid deep nesting.
        let cleanup_base = |device: &ash::Device| unsafe {
            device.destroy_pipeline(pipeline, None);
            device.destroy_pipeline_layout(pipeline_layout, None);
            device.destroy_descriptor_set_layout(descriptor_set_layout, None);
        };

        // --- Descriptor pool ---
        let pool_size = vk::DescriptorPoolSize {
            ty: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
            descriptor_count: 1,
        };
        let pool_info = vk::DescriptorPoolCreateInfo::default()
            .max_sets(1)
            .pool_sizes(std::slice::from_ref(&pool_size));
        let descriptor_pool = unsafe {
            device.create_descriptor_pool(&pool_info, None).inspect_err(|_| cleanup_base(device))?
        };

        let cleanup_pool = |device: &ash::Device| unsafe {
            device.destroy_descriptor_pool(descriptor_pool, None);
            cleanup_base(device);
        };

        // --- Font texture upload ---
        let fonts = imgui.fonts();
        let font_atlas = fonts.build_rgba32_texture();
        let (width, height) = (font_atlas.width, font_atlas.height);
        let upload_size = (width * height * 4) as vk::DeviceSize;

        // Create image
        let image_info = vk::ImageCreateInfo::default()
            .image_type(vk::ImageType::TYPE_2D)
            .format(vk::Format::R8G8B8A8_UNORM)
            .extent(vk::Extent3D {
                width,
                height,
                depth: 1,
            })
            .mip_levels(1)
            .array_layers(1)
            .samples(vk::SampleCountFlags::TYPE_1)
            .tiling(vk::ImageTiling::OPTIMAL)
            .usage(vk::ImageUsageFlags::TRANSFER_DST | vk::ImageUsageFlags::SAMPLED)
            .initial_layout(vk::ImageLayout::UNDEFINED);
        let font_image = unsafe {
            device.create_image(&image_info, None).inspect_err(|_| cleanup_pool(device))?
        };

        let mem_req = unsafe { device.get_image_memory_requirements(font_image) };
        let font_image_memory = unsafe {
            let alloc_info = vk::MemoryAllocateInfo::default()
                .allocation_size(mem_req.size)
                .memory_type_index(
                    find_memory_type(
                        &memory_properties,
                        mem_req.memory_type_bits,
                        vk::MemoryPropertyFlags::DEVICE_LOCAL,
                    )
                    .expect("no suitable memory type for font image"),
                );
            device.allocate_memory(&alloc_info, None).inspect_err(|_| {
                device.destroy_image(font_image, None);
                cleanup_pool(device);
            })?
        };
        unsafe {
            device.bind_image_memory(font_image, font_image_memory, 0).inspect_err(|_| {
                device.free_memory(font_image_memory, None);
                device.destroy_image(font_image, None);
                cleanup_pool(device);
            })?;
        }

        let cleanup_image = |device: &ash::Device| unsafe {
            device.free_memory(font_image_memory, None);
            device.destroy_image(font_image, None);
            cleanup_pool(device);
        };

        // Staging buffer
        let (staging_buffer, staging_memory) = create_buffer(
            device,
            &memory_properties,
            upload_size,
            vk::BufferUsageFlags::TRANSFER_SRC,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        )
        .inspect_err(|_| cleanup_image(device))?;

        unsafe {
            let ptr = device
                .map_memory(staging_memory, 0, upload_size, vk::MemoryMapFlags::empty())
                .inspect_err(|_| {
                    device.destroy_buffer(staging_buffer, None);
                    device.free_memory(staging_memory, None);
                    cleanup_image(device);
                })?;
            std::ptr::copy_nonoverlapping(
                font_atlas.data.as_ptr(),
                ptr as *mut u8,
                upload_size as usize,
            );
            device.unmap_memory(staging_memory);
        }

        // Upload via one-shot command buffer
        execute_one_time_commands(
            device,
            create_info.command_pool,
            create_info.queue,
            |cmd| {
                // Transition UNDEFINED -> TRANSFER_DST_OPTIMAL
                let barrier = vk::ImageMemoryBarrier::default()
                    .image(font_image)
                    .old_layout(vk::ImageLayout::UNDEFINED)
                    .new_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                    .src_access_mask(vk::AccessFlags::empty())
                    .dst_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                    .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                    .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                    .subresource_range(vk::ImageSubresourceRange {
                        aspect_mask: vk::ImageAspectFlags::COLOR,
                        base_mip_level: 0,
                        level_count: 1,
                        base_array_layer: 0,
                        layer_count: 1,
                    });
                unsafe {
                    device.cmd_pipeline_barrier(
                        cmd,
                        vk::PipelineStageFlags::TOP_OF_PIPE,
                        vk::PipelineStageFlags::TRANSFER,
                        vk::DependencyFlags::empty(),
                        &[],
                        &[],
                        &[barrier],
                    );
                }

                // Copy buffer to image
                let region = vk::BufferImageCopy {
                    buffer_offset: 0,
                    buffer_row_length: 0,
                    buffer_image_height: 0,
                    image_subresource: vk::ImageSubresourceLayers {
                        aspect_mask: vk::ImageAspectFlags::COLOR,
                        mip_level: 0,
                        base_array_layer: 0,
                        layer_count: 1,
                    },
                    image_offset: vk::Offset3D { x: 0, y: 0, z: 0 },
                    image_extent: vk::Extent3D {
                        width,
                        height,
                        depth: 1,
                    },
                };
                unsafe {
                    device.cmd_copy_buffer_to_image(
                        cmd,
                        staging_buffer,
                        font_image,
                        vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                        &[region],
                    );
                }

                // Transition TRANSFER_DST -> SHADER_READ_ONLY
                let barrier = vk::ImageMemoryBarrier::default()
                    .image(font_image)
                    .old_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                    .new_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                    .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                    .dst_access_mask(vk::AccessFlags::SHADER_READ)
                    .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                    .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                    .subresource_range(vk::ImageSubresourceRange {
                        aspect_mask: vk::ImageAspectFlags::COLOR,
                        base_mip_level: 0,
                        level_count: 1,
                        base_array_layer: 0,
                        layer_count: 1,
                    });
                unsafe {
                    device.cmd_pipeline_barrier(
                        cmd,
                        vk::PipelineStageFlags::TRANSFER,
                        vk::PipelineStageFlags::FRAGMENT_SHADER,
                        vk::DependencyFlags::empty(),
                        &[],
                        &[],
                        &[barrier],
                    );
                }
            },
        )
        .inspect_err(|_| {
            unsafe {
                device.destroy_buffer(staging_buffer, None);
                device.free_memory(staging_memory, None);
            }
            cleanup_image(device);
        })?;

        // Destroy staging buffer
        unsafe {
            device.destroy_buffer(staging_buffer, None);
            device.free_memory(staging_memory, None);
        }

        // Image view
        let view_info = vk::ImageViewCreateInfo::default()
            .image(font_image)
            .view_type(vk::ImageViewType::TYPE_2D)
            .format(vk::Format::R8G8B8A8_UNORM)
            .subresource_range(vk::ImageSubresourceRange {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                base_mip_level: 0,
                level_count: 1,
                base_array_layer: 0,
                layer_count: 1,
            });
        let font_image_view = unsafe {
            device.create_image_view(&view_info, None).inspect_err(|_| cleanup_image(device))?
        };

        let cleanup_view = |device: &ash::Device| unsafe {
            device.destroy_image_view(font_image_view, None);
            cleanup_image(device);
        };

        // Sampler
        let sampler_info = vk::SamplerCreateInfo::default()
            .mag_filter(vk::Filter::LINEAR)
            .min_filter(vk::Filter::LINEAR)
            .address_mode_u(vk::SamplerAddressMode::CLAMP_TO_EDGE)
            .address_mode_v(vk::SamplerAddressMode::CLAMP_TO_EDGE)
            .address_mode_w(vk::SamplerAddressMode::CLAMP_TO_EDGE);
        let font_sampler = unsafe {
            device.create_sampler(&sampler_info, None).inspect_err(|_| cleanup_view(device))?
        };

        let cleanup_sampler = |device: &ash::Device| unsafe {
            device.destroy_sampler(font_sampler, None);
            cleanup_view(device);
        };

        // --- Allocate font descriptor set ---
        let alloc_info = vk::DescriptorSetAllocateInfo::default()
            .descriptor_pool(descriptor_pool)
            .set_layouts(std::slice::from_ref(&descriptor_set_layout));
        let font_descriptor_set = unsafe {
            device
                .allocate_descriptor_sets(&alloc_info)
                .inspect_err(|_| cleanup_sampler(device))?[0]
        };

        let image_write = vk::DescriptorImageInfo::default()
            .sampler(font_sampler)
            .image_view(font_image_view)
            .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL);
        let write = vk::WriteDescriptorSet::default()
            .dst_set(font_descriptor_set)
            .dst_binding(0)
            .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .image_info(std::slice::from_ref(&image_write));
        unsafe { device.update_descriptor_sets(&[write], &[]) };

        // Set font texture ID to the descriptor set handle so we can look it up
        // during rendering.
        imgui
            .fonts()
            .tex_id = TextureId::new(font_descriptor_set.as_raw() as usize);

        // --- Initial vertex/index buffers ---
        let (vertex_buffer, vertex_buffer_memory) = create_buffer(
            device,
            &memory_properties,
            INITIAL_BUFFER_SIZE,
            vk::BufferUsageFlags::VERTEX_BUFFER,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        )
        .inspect_err(|_| cleanup_sampler(device))?;

        let (index_buffer, index_buffer_memory) = create_buffer(
            device,
            &memory_properties,
            INITIAL_BUFFER_SIZE,
            vk::BufferUsageFlags::INDEX_BUFFER,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        )
        .inspect_err(|_| unsafe {
            device.destroy_buffer(vertex_buffer, None);
            device.free_memory(vertex_buffer_memory, None);
            cleanup_sampler(device);
        })?;

        Ok(Renderer {
            device: device.clone(),
            memory_properties,
            pipeline_layout,
            pipeline,
            descriptor_set_layout,
            descriptor_pool,
            font_descriptor_set,
            font_image,
            font_image_memory,
            font_image_view,
            font_sampler,
            vertex_buffer,
            vertex_buffer_memory,
            vertex_buffer_size: INITIAL_BUFFER_SIZE,
            index_buffer,
            index_buffer_memory,
            index_buffer_size: INITIAL_BUFFER_SIZE,
            retired_buffers: Vec::new(),
        })
    }

    /// Record draw commands for the given imgui draw data into `command_buffer`.
    ///
    /// The command buffer must already be inside a compatible render pass.
    /// Vertex and index buffers are reallocated if the current ones are too
    /// small; old buffers are kept alive until the *next* call to `render()`
    /// so that any previously recorded command buffers still referencing them
    /// can finish executing.
    pub fn render(
        &mut self,
        draw_data: &DrawData,
        command_buffer: vk::CommandBuffer,
    ) -> Result<(), RendererError> {
        // Destroy buffers that were retired during the *previous* render call.
        // By now the user must have waited for the previous frame's command
        // buffer to finish (standard Vulkan frame pacing), so these are safe
        // to free.
        self.flush_retired_buffers();

        if draw_data.total_vtx_count <= 0 {
            return Ok(());
        }

        let device = &self.device;

        // --- Reallocate buffers if needed ---
        let vertex_size =
            (draw_data.total_vtx_count as usize * mem::size_of::<DrawVert>()) as vk::DeviceSize;
        let index_size =
            (draw_data.total_idx_count as usize * mem::size_of::<DrawIdx>()) as vk::DeviceSize;

        if vertex_size > self.vertex_buffer_size {
            // Retire the old buffer instead of destroying immediately -- a
            // previously submitted command buffer may still reference it.
            self.retired_buffers
                .push((self.vertex_buffer, self.vertex_buffer_memory));
            let (buf, mem) = create_buffer(
                device,
                &self.memory_properties,
                vertex_size,
                vk::BufferUsageFlags::VERTEX_BUFFER,
                vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
            )?;
            self.vertex_buffer = buf;
            self.vertex_buffer_memory = mem;
            self.vertex_buffer_size = vertex_size;
        }

        if index_size > self.index_buffer_size {
            self.retired_buffers
                .push((self.index_buffer, self.index_buffer_memory));
            let (buf, mem) = create_buffer(
                device,
                &self.memory_properties,
                index_size,
                vk::BufferUsageFlags::INDEX_BUFFER,
                vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
            )?;
            self.index_buffer = buf;
            self.index_buffer_memory = mem;
            self.index_buffer_size = index_size;
        }

        // --- Upload vertex/index data ---
        unsafe {
            let vtx_ptr = device.map_memory(
                self.vertex_buffer_memory,
                0,
                vertex_size,
                vk::MemoryMapFlags::empty(),
            )? as *mut DrawVert;
            let idx_ptr = device.map_memory(
                self.index_buffer_memory,
                0,
                index_size,
                vk::MemoryMapFlags::empty(),
            )? as *mut DrawIdx;

            let mut vtx_offset = 0usize;
            let mut idx_offset = 0usize;
            for draw_list in draw_data.draw_lists() {
                let vtx_buf = draw_list.vtx_buffer();
                let idx_buf = draw_list.idx_buffer();
                std::ptr::copy_nonoverlapping(
                    vtx_buf.as_ptr(),
                    vtx_ptr.add(vtx_offset),
                    vtx_buf.len(),
                );
                std::ptr::copy_nonoverlapping(
                    idx_buf.as_ptr(),
                    idx_ptr.add(idx_offset),
                    idx_buf.len(),
                );
                vtx_offset += vtx_buf.len();
                idx_offset += idx_buf.len();
            }

            device.unmap_memory(self.vertex_buffer_memory);
            device.unmap_memory(self.index_buffer_memory);
        }

        // --- Record commands ---
        let fb_scale = draw_data.framebuffer_scale;
        let display_pos = draw_data.display_pos;
        let display_size = draw_data.display_size;
        let fb_width = display_size[0] * fb_scale[0];
        let fb_height = display_size[1] * fb_scale[1];
        if fb_width <= 0.0 || fb_height <= 0.0 {
            return Ok(());
        }

        // Pre-compute values needed both for initial setup and ResetRenderState.
        let viewport = vk::Viewport {
            x: 0.0,
            y: 0.0,
            width: fb_width,
            height: fb_height,
            min_depth: 0.0,
            max_depth: 1.0,
        };

        let left = display_pos[0];
        let right = display_pos[0] + display_size[0];
        let top = display_pos[1];
        let bottom = display_pos[1] + display_size[1];

        #[rustfmt::skip]
        let proj_mtx: [f32; 16] = [
            2.0 / (right - left),              0.0,                                0.0, 0.0,
            0.0,                               2.0 / (bottom - top),               0.0, 0.0,
            0.0,                               0.0,                               -1.0, 0.0,
            (right + left) / (left - right),   (top + bottom) / (top - bottom),    0.0, 1.0,
        ];
        let proj_bytes = bytemuck_cast_slice(&proj_mtx);

        let idx_type = if mem::size_of::<DrawIdx>() == 2 {
            vk::IndexType::UINT16
        } else {
            vk::IndexType::UINT32
        };

        unsafe {
            self.setup_render_state(command_buffer, &viewport, proj_bytes, idx_type);
        }

        let clip_off = display_pos;
        let clip_scale = fb_scale;

        let mut global_vtx_offset = 0u32;
        let mut global_idx_offset = 0u32;

        for draw_list in draw_data.draw_lists() {
            for cmd in draw_list.commands() {
                match cmd {
                    DrawCmd::Elements { count, cmd_params } => {
                        let clip_min_x =
                            (cmd_params.clip_rect[0] - clip_off[0]) * clip_scale[0];
                        let clip_min_y =
                            (cmd_params.clip_rect[1] - clip_off[1]) * clip_scale[1];
                        let clip_max_x =
                            (cmd_params.clip_rect[2] - clip_off[0]) * clip_scale[0];
                        let clip_max_y =
                            (cmd_params.clip_rect[3] - clip_off[1]) * clip_scale[1];

                        if clip_max_x <= clip_min_x || clip_max_y <= clip_min_y {
                            continue;
                        }

                        let scissor = vk::Rect2D {
                            offset: vk::Offset2D {
                                x: clip_min_x.max(0.0) as i32,
                                y: clip_min_y.max(0.0) as i32,
                            },
                            extent: vk::Extent2D {
                                width: (clip_max_x - clip_min_x.max(0.0)) as u32,
                                height: (clip_max_y - clip_min_y.max(0.0)) as u32,
                            },
                        };

                        // The texture_id stores the raw VkDescriptorSet handle.
                        let descriptor_set = vk::DescriptorSet::from_raw(
                            cmd_params.texture_id.id() as u64,
                        );

                        unsafe {
                            self.device.cmd_set_scissor(command_buffer, 0, &[scissor]);
                            self.device.cmd_bind_descriptor_sets(
                                command_buffer,
                                vk::PipelineBindPoint::GRAPHICS,
                                self.pipeline_layout,
                                0,
                                &[descriptor_set],
                                &[],
                            );
                            self.device.cmd_draw_indexed(
                                command_buffer,
                                count as u32,
                                1,
                                cmd_params.idx_offset as u32 + global_idx_offset,
                                (cmd_params.vtx_offset as i32)
                                    + (global_vtx_offset as i32),
                                0,
                            );
                        }
                    }
                    DrawCmd::ResetRenderState => unsafe {
                        self.setup_render_state(
                            command_buffer,
                            &viewport,
                            proj_bytes,
                            idx_type,
                        );
                    },
                    DrawCmd::RawCallback { callback, raw_cmd } => unsafe {
                        callback(draw_list.raw(), raw_cmd);
                    },
                }
            }

            global_vtx_offset += draw_list.vtx_buffer().len() as u32;
            global_idx_offset += draw_list.idx_buffer().len() as u32;
        }

        Ok(())
    }

    /// Bind pipeline, buffers, viewport, scissor, and push constants.
    ///
    /// # Safety
    ///
    /// `command_buffer` must be in the recording state inside a compatible
    /// render pass.
    unsafe fn setup_render_state(
        &self,
        command_buffer: vk::CommandBuffer,
        viewport: &vk::Viewport,
        proj_bytes: &[u8],
        idx_type: vk::IndexType,
    ) {
        unsafe {
            self.device.cmd_bind_pipeline(
                command_buffer,
                vk::PipelineBindPoint::GRAPHICS,
                self.pipeline,
            );
            self.device
                .cmd_bind_vertex_buffers(command_buffer, 0, &[self.vertex_buffer], &[0]);
            self.device.cmd_bind_index_buffer(
                command_buffer,
                self.index_buffer,
                0,
                idx_type,
            );
            self.device
                .cmd_set_viewport(command_buffer, 0, std::slice::from_ref(viewport));
            self.device.cmd_push_constants(
                command_buffer,
                self.pipeline_layout,
                vk::ShaderStageFlags::VERTEX,
                0,
                proj_bytes,
            );
        }
    }

    /// Destroy buffers that were retired during a previous `render()` call.
    fn flush_retired_buffers(&mut self) {
        for (buffer, memory) in self.retired_buffers.drain(..) {
            unsafe {
                self.device.destroy_buffer(buffer, None);
                self.device.free_memory(memory, None);
            }
        }
    }
}

impl Drop for Renderer {
    fn drop(&mut self) {
        unsafe {
            let d = &self.device;
            d.device_wait_idle().ok();
            // Flush any buffers still waiting for retirement.
            for (buffer, memory) in self.retired_buffers.drain(..) {
                d.destroy_buffer(buffer, None);
                d.free_memory(memory, None);
            }
            d.destroy_buffer(self.index_buffer, None);
            d.free_memory(self.index_buffer_memory, None);
            d.destroy_buffer(self.vertex_buffer, None);
            d.free_memory(self.vertex_buffer_memory, None);
            d.destroy_sampler(self.font_sampler, None);
            d.destroy_image_view(self.font_image_view, None);
            d.free_memory(self.font_image_memory, None);
            d.destroy_image(self.font_image, None);
            d.destroy_descriptor_pool(self.descriptor_pool, None);
            d.destroy_pipeline(self.pipeline, None);
            d.destroy_pipeline_layout(self.pipeline_layout, None);
            d.destroy_descriptor_set_layout(self.descriptor_set_layout, None);
        }
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn find_memory_type(
    props: &vk::PhysicalDeviceMemoryProperties,
    type_filter: u32,
    flags: vk::MemoryPropertyFlags,
) -> Option<u32> {
    for i in 0..props.memory_type_count {
        if (type_filter & (1 << i)) != 0
            && props.memory_types[i as usize]
                .property_flags
                .contains(flags)
        {
            return Some(i);
        }
    }
    None
}

fn create_buffer(
    device: &ash::Device,
    memory_properties: &vk::PhysicalDeviceMemoryProperties,
    size: vk::DeviceSize,
    usage: vk::BufferUsageFlags,
    mem_flags: vk::MemoryPropertyFlags,
) -> Result<(vk::Buffer, vk::DeviceMemory), RendererError> {
    let buffer_info = vk::BufferCreateInfo::default()
        .size(size)
        .usage(usage)
        .sharing_mode(vk::SharingMode::EXCLUSIVE);
    let buffer = unsafe { device.create_buffer(&buffer_info, None)? };
    let mem_req = unsafe { device.get_buffer_memory_requirements(buffer) };
    let alloc_info = vk::MemoryAllocateInfo::default()
        .allocation_size(mem_req.size)
        .memory_type_index(
            find_memory_type(memory_properties, mem_req.memory_type_bits, mem_flags)
                .expect("no suitable memory type for buffer"),
        );
    let memory = unsafe {
        device.allocate_memory(&alloc_info, None).inspect_err(|_| {
            device.destroy_buffer(buffer, None);
        })?
    };
    unsafe {
        device
            .bind_buffer_memory(buffer, memory, 0)
            .inspect_err(|_| {
                device.free_memory(memory, None);
                device.destroy_buffer(buffer, None);
            })?;
    }
    Ok((buffer, memory))
}

fn execute_one_time_commands(
    device: &ash::Device,
    command_pool: vk::CommandPool,
    queue: vk::Queue,
    f: impl FnOnce(vk::CommandBuffer),
) -> Result<(), RendererError> {
    let alloc_info = vk::CommandBufferAllocateInfo::default()
        .command_pool(command_pool)
        .level(vk::CommandBufferLevel::PRIMARY)
        .command_buffer_count(1);
    let cmd = unsafe { device.allocate_command_buffers(&alloc_info)?[0] };

    let begin_info =
        vk::CommandBufferBeginInfo::default().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
    unsafe { device.begin_command_buffer(cmd, &begin_info)? };

    f(cmd);

    unsafe { device.end_command_buffer(cmd)? };

    let submit_info = vk::SubmitInfo::default().command_buffers(std::slice::from_ref(&cmd));
    let fence_info = vk::FenceCreateInfo::default();
    let fence = unsafe { device.create_fence(&fence_info, None)? };

    unsafe {
        device.queue_submit(queue, &[submit_info], fence)?;
        device.wait_for_fences(&[fence], true, u64::MAX)?;
        device.destroy_fence(fence, None);
        device.free_command_buffers(command_pool, &[cmd]);
    }

    Ok(())
}

/// Reinterpret a `&[f32]` slice as `&[u8]` for push constants.
fn bytemuck_cast_slice(data: &[f32]) -> &[u8] {
    unsafe {
        std::slice::from_raw_parts(data.as_ptr() as *const u8, data.len() * mem::size_of::<f32>())
    }
}
