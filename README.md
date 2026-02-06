# imgui-vulkan-renderer-rs

Vulkan renderer for [Dear ImGui](https://github.com/ocornut/imgui) via the
[imgui](https://crates.io/crates/imgui) Rust bindings and
[ash](https://crates.io/crates/ash).

Pair with [imgui-glfw-rs](https://crates.io/crates/imgui-glfw-rs) for a
complete GLFW + Vulkan + Dear ImGui integration.

## Usage

Add to your `Cargo.toml`:

```toml
[dependencies]
imgui-vulkan-renderer-rs = "0.1"
```

Then in your code:

```rust
use imgui::Context;
use imgui_vulkan_renderer_rs::{Renderer, RendererCreateInfo};

let mut imgui = Context::create();

let create_info = RendererCreateInfo {
    device: device.clone(),
    memory_properties: mem_props,
    render_pass,
    command_pool,
    queue,
};
let mut renderer = Renderer::new(&mut imgui, &create_info)
    .expect("Failed to initialize renderer");

// In your main loop, after building the imgui frame:
let draw_data = imgui.render();
renderer.render(draw_data, command_buffer).unwrap();
```

## Requirements

- Vulkan 1.0 or later
- A valid `ash::Device` and compatible `VkRenderPass` when calling `Renderer::new`
- A command buffer in the recording state inside a compatible render pass when calling `Renderer::render`

## License

Licensed under either of

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE_APACHE) or <http://www.apache.org/licenses/LICENSE-2.0>)
- MIT License ([LICENSE-MIT](LICENSE_MIT) or <http://opensource.org/licenses/MIT>)

at your option.
