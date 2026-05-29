/*
SPDX-License-Identifier:    GPL-3.0-only
Copyright (C) 2026 Hippolyte Audet-Lagacé
Author:         Hippolyte Audet-Lagacé
Description:    This program implements a vulkan rasterizing renderer meant to work on an existing vulkan context, the intent
                being to allow for the vulkan context to be used for a compute pipeline separately. The rendered object is in
                this implementation a Quadrilateralized spherical cube (QSC) refered to by the sphere variable, but any object
                implementing a vertices, indices and rgba array set would work fine with little adaptation needed.
*/

use crate::vulkan_context;
use crate::sphere::{Sphere, Vertex};

use ash::vk;
use winit::{
    application::ApplicationHandler,
    event::{WindowEvent, ElementState, MouseButton, MouseScrollDelta},
    event_loop::ActiveEventLoop,
    window::{Window, WindowId, WindowAttributes},
    dpi::LogicalSize,
    raw_window_handle::{HasDisplayHandle, HasWindowHandle},
    keyboard::Key,
};
use ash_window;
use nalgebra::{Matrix4, Vector3, Point3, Isometry3, UnitQuaternion};
use std::f32::consts;

// Section1: Core -------------------------------------------------------------------------------------------------------------

// Global constants
const WIDTH: u32 = 1280;
const HEIGHT: u32 = 720;
const FRAMES_IN_FLIGHT: u32 = 2;
const TURNTABLE_SENSITIVITY_H: f32 = 0.005; // radians per pixel
const TURNTABLE_SENSITIVITY_V: f32 = 0.005;

/* Buffer object */
struct GpuBuffer {
    buffer: vk::Buffer,
    memory: vk::DeviceMemory,
    size: vk::DeviceSize,
}
impl GpuBuffer {
    /* Destroy the buffer and free its memory. */
    fn new() -> Self {
        GpuBuffer {
            buffer: vk::Buffer::null(),
            memory: vk::DeviceMemory::null(),
            size: 0,
        }
    }
    fn destroy(&self, device: &ash::Device) {
        unsafe {
            if self.buffer != vk::Buffer::null() {
                device.destroy_buffer(self.buffer, None);
            }
            if self.memory != vk::DeviceMemory::null() {
                device.free_memory(self.memory, None);
            }
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
struct UniformBufferObject {
    model: Matrix4<f32>,
    view:  Matrix4<f32>,
    proj:  Matrix4<f32>,
}
impl UniformBufferObject {
    fn new(angle: f32, extent: vk::Extent2D) -> Self {
        let model = Matrix4::from_axis_angle(&Vector3::y_axis(), angle);

        let eye    = Point3::new(0.0, 0.0, 5.0);
        let target = Point3::origin();
        let view   = Isometry3::look_at_rh(&eye, &target, &Vector3::y())
                         .to_homogeneous();

        let aspect = extent.width as f32 / extent.height as f32;

        let f = 1.0 / (consts::FRAC_PI_3 / 2.0).tan();
        let proj = Matrix4::new(
            f / aspect,     0.0,    0.0,                    0.0,
            0.0,            -f,     0.0,                    0.0,
            0.0,            0.0,    100.0 / (0.1 - 100.0),  (100.0 * 0.1) / (0.1 - 100.0),
            0.0,            0.0,    -1.0,                   0.0,
        );

        Self { model, view, proj }
    }
}
impl Default for UniformBufferObject {
    fn default() -> Self {
        Self::new(0.0, vk::Extent2D { width: WIDTH, height: HEIGHT })
    }
}

/* Complete Vulkan rasterizing renderer */ 
pub struct Renderer {
    initialized:                bool,
    pub context:                vulkan_context::VulkanContext,
    pub sphere:                 Sphere,
    window:                     Option<Window>,
    surface:                    vk::SurfaceKHR,
    swapchain:                  vk::SwapchainKHR,
    swapchain_images:           Vec<vk::Image>,
    swapchain_image_views:      Vec<vk::ImageView>,
    swapchain_format:           vk::Format,
    swapchain_extent:           vk::Extent2D,
    depth_image:                vk::Image,
    depth_image_memory:         vk::DeviceMemory,
    depth_image_view:           vk::ImageView,
    render_pass:                vk::RenderPass,
    framebuffers:               Vec<vk::Framebuffer>,
    command_pool:               vk::CommandPool,
    command_buffers:            Vec<vk::CommandBuffer>,
    image_available_semaphores: Vec<vk::Semaphore>,
    render_finished_semaphores: Vec<vk::Semaphore>,
    in_flight_fences:           Vec<vk::Fence>,
    current_frame:              usize,
    graphics_queue:             vk::Queue,
    present_queue:              vk::Queue,
    descriptor_set_layout:      vk::DescriptorSetLayout,
    descriptor_pool:            vk::DescriptorPool,
    descriptor_set:             vk::DescriptorSet,
    pipeline_layout:            vk::PipelineLayout,
    pipeline:                   vk::Pipeline,
    vertex_buffer:              GpuBuffer,
    index_buffer:               GpuBuffer,
    ubo:                        UniformBufferObject,
    uniform_buffer:             GpuBuffer,
    uniform_buffer_mapped:      *mut UniformBufferObject,
    camera_drag:                bool,                           
    drag_last_pos:              Option<winit::dpi::PhysicalPosition<f64>>,  
    camera_rotation:            UnitQuaternion<f32>,            
    camera_distance:            f32,
    window_size:                winit::dpi::PhysicalSize<u32>,
    color_mode:                 i32,
}
impl Renderer {
    pub fn new(context: vulkan_context::VulkanContext, sphere: Sphere) -> Self {
        Renderer {
            initialized:                false,
            context,
            sphere,
            window: None,
            surface:                    vk::SurfaceKHR::null(),
            swapchain:                  vk::SwapchainKHR::null(),
            swapchain_images:           Vec::new(),
            swapchain_image_views:      Vec::new(),
            swapchain_format:           vk::Format::UNDEFINED,
            swapchain_extent:           vk::Extent2D::default(),
            depth_image:                vk::Image::null(),
            depth_image_memory:         vk::DeviceMemory::null(),
            depth_image_view:           vk::ImageView::null(),
            render_pass:                vk::RenderPass::null(),
            framebuffers:               Vec::new(),
            command_pool:               vk::CommandPool::null(),
            command_buffers:            Vec::new(),
            image_available_semaphores: Vec::new(),
            render_finished_semaphores: Vec::new(),
            in_flight_fences:           Vec::new(),
            current_frame:              0,
            graphics_queue:             vk::Queue::null(),
            present_queue:              vk::Queue::null(),
            descriptor_set_layout:      vk::DescriptorSetLayout::null(),
            descriptor_pool:            vk::DescriptorPool::null(),
            descriptor_set:             vk::DescriptorSet::null(),
            pipeline_layout:            vk::PipelineLayout::null(),
            pipeline:                   vk::Pipeline::null(),
            vertex_buffer:              GpuBuffer::new(),
            index_buffer:               GpuBuffer::new(),
            ubo:                        UniformBufferObject::default(),
            uniform_buffer:             GpuBuffer::new(),
            uniform_buffer_mapped:      std::ptr::null_mut(),
            camera_drag:                false,
            drag_last_pos:              None,
            camera_rotation:            UnitQuaternion::identity(),
            camera_distance:            3.0,
            window_size:                winit::dpi::PhysicalSize::new(WIDTH, HEIGHT),
            color_mode:                 0,
        }
    }

    fn initialize(&mut self, event_loop: &ActiveEventLoop) -> Result<(), Box<dyn std::error::Error>> {
        // Window
        let attributes = WindowAttributes::default()
            .with_title("Window McWindowface")
            .with_inner_size(LogicalSize::new(WIDTH, HEIGHT))
            .with_resizable(true);
        let window = event_loop.create_window(attributes)?;

        // Queues, retrieved as static handles to avoid a fetching overhead in draw_frame()
        self.graphics_queue = unsafe {
            self.context.device.as_ref().unwrap()
                .get_device_queue(self.context.queue_families.as_ref().unwrap().graphics, 0)
        };
        self.present_queue = unsafe {
            self.context.device.as_ref().unwrap()
                .get_device_queue(self.context.queue_families.as_ref().unwrap().graphics, 0)
        };

        // Surface
        self.surface = create_surface(
            self.context.entry.as_ref().unwrap(),
            self.context.instance.as_ref().unwrap(),
            &window,
        )?;

        // Swapchain
        let (swapchain, format, extent) = create_swap_chain(
            self.context.physical_device,
            self.surface,
            self.context.surface_loader.as_ref().unwrap(),
            self.context.swapchain_loader.as_ref().unwrap(),
            &window,
        )?;
        self.swapchain = swapchain;
        self.swapchain_format = format;
        self.swapchain_extent = extent;

        // Uniform buffer object
        self.ubo = UniformBufferObject::new(0.0, self.swapchain_extent);

        // Image view
        let (swapchain_images, swapchain_image_views) = create_image_views(
            self.context.device.as_ref().unwrap(),
            self.context.swapchain_loader.as_ref().unwrap(),
            self.swapchain,
            self.swapchain_format,
        )?;
        self.swapchain_images = swapchain_images;
        self.swapchain_image_views = swapchain_image_views;

        // Depth image
        let (depth_image, depth_memory, depth_image_view) = create_depth_image(
            self.context.instance.as_ref().unwrap(),
            self.context.physical_device,
            self.context.device.as_ref().unwrap(),
            self.swapchain_extent,
        )?;
        self.depth_image        = depth_image;
        self.depth_image_memory = depth_memory;
        self.depth_image_view   = depth_image_view;

        // Render pass
        self.render_pass = create_render_pass(
            self.context.device.as_ref().unwrap(),
            self.swapchain_format,
        )?;

        // Framebuffers
        self.framebuffers = create_framebuffers(
            self.context.device.as_ref().unwrap(),
            self.render_pass,
            &self.swapchain_image_views,
            self.depth_image_view,
            self.swapchain_extent,
        )?;

        // Command pool
        self.command_pool = vulkan_context::create_command_pool(
            self.context.device.as_ref().unwrap(),
            self.context.queue_families.as_ref().unwrap().graphics,
        )?;

        // Command buffers
        self.command_buffers = create_command_buffers(
            self.context.device.as_ref().unwrap(),
            self.command_pool,
        )?;

        // Descriptor set layout
        self.descriptor_set_layout = create_descriptor_set_layout(
            self.context.device.as_ref().unwrap(),
        )?;

        // pipeline
        let (pipeline_layout, pipeline) = create_graphics_pipeline(
            self.context.device.as_ref().unwrap(),
            self.render_pass,
            self.swapchain_extent,
            self.descriptor_set_layout,
        )?;
        self.pipeline_layout = pipeline_layout;
        self.pipeline = pipeline;

        // Vertex buffer
        self.vertex_buffer = upload_to_device_local_buffer(
            self.context.instance.as_ref().unwrap(),
            self.context.physical_device,
            self.context.device.as_ref().unwrap(),
            self.command_pool,
            self.graphics_queue,
            &self.sphere.vertices,
            vk::BufferUsageFlags::VERTEX_BUFFER,
        )?;

        // Index buffer
        self.index_buffer = upload_to_device_local_buffer(
            self.context.instance.as_ref().unwrap(),
            self.context.physical_device,
            self.context.device.as_ref().unwrap(),
            self.command_pool,
            self.graphics_queue,
            &self.sphere.indices,
            vk::BufferUsageFlags::INDEX_BUFFER,
        )?;

        // Uniform buffer
        let ubo_size = std::mem::size_of::<UniformBufferObject>() as vk::DeviceSize;
        self.uniform_buffer = create_raw_buffer(
            self.context.instance.as_ref().unwrap(),
            self.context.physical_device,
            self.context.device.as_ref().unwrap(),
            ubo_size,
            vk::BufferUsageFlags::UNIFORM_BUFFER,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        )?;
        self.uniform_buffer_mapped = unsafe {
            self.context.device.as_ref().unwrap().map_memory(
                self.uniform_buffer.memory,
                0,
                std::mem::size_of::<UniformBufferObject>() as vk::DeviceSize,
                vk::MemoryMapFlags::empty(),
            )? as *mut UniformBufferObject
        };

        // Descriptor pool and descriptor sets
        self.descriptor_pool = create_descriptor_pool(
            self.context.device.as_ref().unwrap(),
        )?;
        self.descriptor_set = allocate_descriptor_set(
            self.context.device.as_ref().unwrap(),
            self.descriptor_pool,
            self.descriptor_set_layout,
            &self.uniform_buffer,
        )?;

        // Sync objects
        let (image_available_semaphores, render_finished_semaphores, in_flight_fences) =
            create_sync_objects(self.context.device.as_ref().unwrap())?;
        self.image_available_semaphores = image_available_semaphores;
        self.render_finished_semaphores = render_finished_semaphores;
        self.in_flight_fences = in_flight_fences;

        // Window needs to be borrowed between creation and here
        self.window = Some(window);

        println!("Vertices: {}", self.sphere.vertices.len());

        return Ok(());
    }

    /* Rendering function */
    fn draw_frame(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        let device = self.context.device.as_ref().unwrap().clone();  // ash::Device is an Arc
        let current_frame = self.current_frame;

        // wait for previous frame to finish
        unsafe {
            device.wait_for_fences(
                &[self.in_flight_fences[current_frame]],
                true,
                u64::MAX,
            )?;
            device.reset_fences(&[self.in_flight_fences[current_frame]])?;
        }

        // Update UBO
        self.update_ubo();

        // acquire next swapchain image
        let (image_index, _) = unsafe {
            self.context.swapchain_loader.as_ref().unwrap()
                .acquire_next_image(
                    self.swapchain,
                    u64::MAX,
                    self.image_available_semaphores[current_frame],
                    vk::Fence::null(),
                )?
        };

        // record command buffer
        let command_buffer = self.command_buffers[current_frame];
        unsafe {
            device.reset_command_buffer(command_buffer, vk::CommandBufferResetFlags::empty())?;
        }
        self.record_command_buffer(command_buffer, image_index as usize)?;

        // submit
        let wait_semaphores = [self.image_available_semaphores[current_frame]];
        let wait_stages = [vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT];
        let signal_semaphores = [self.render_finished_semaphores[image_index as usize]];
        let command_buffers = [command_buffer];

        let submit_info = vk::SubmitInfo {
            wait_semaphore_count: wait_semaphores.len() as u32,
            p_wait_semaphores: wait_semaphores.as_ptr(),
            p_wait_dst_stage_mask: wait_stages.as_ptr(),
            command_buffer_count: command_buffers.len() as u32,
            p_command_buffers: command_buffers.as_ptr(),
            signal_semaphore_count: signal_semaphores.len() as u32,
            p_signal_semaphores: signal_semaphores.as_ptr(),
            ..Default::default()
        };

        unsafe {
            device.queue_submit(
                self.graphics_queue,
                &[submit_info],
                self.in_flight_fences[current_frame],
            )?;
        }

        // present
        let swapchains = [self.swapchain];
        let image_indices = [image_index];
        let present_info = vk::PresentInfoKHR {
            wait_semaphore_count: signal_semaphores.len() as u32,
            p_wait_semaphores: signal_semaphores.as_ptr(),
            swapchain_count: swapchains.len() as u32,
            p_swapchains: swapchains.as_ptr(),
            p_image_indices: image_indices.as_ptr(),
            ..Default::default()
        };

        unsafe {
            self.context.swapchain_loader.as_ref().unwrap()
                .queue_present(self.present_queue, &present_info)?;
        }

        self.current_frame = (current_frame + 1) % FRAMES_IN_FLIGHT as usize;

        return Ok(());
    }
    // Helper function for draw_frame(), fills the command buffer with commands
    fn record_command_buffer(
        &self,
        command_buffer: vk::CommandBuffer,
        image_index: usize,
    ) -> Result<(), Box<dyn std::error::Error>> {
        // Setup
        let device = self.context.device.as_ref().unwrap();

        let begin_info = vk::CommandBufferBeginInfo::default();
        unsafe { device.begin_command_buffer(command_buffer, &begin_info)?; }

        let clear_values = [
            vk::ClearValue {
                color: vk::ClearColorValue { float32: [0.0, 0.0, 0.0, 1.0] },
            },
            vk::ClearValue {
                depth_stencil: vk::ClearDepthStencilValue { depth: 1.0, stencil: 0 },
            },
        ];
        let render_pass_begin_info = vk::RenderPassBeginInfo {
            render_pass:    self.render_pass,
            framebuffer:    self.framebuffers[image_index],
            render_area:    vk::Rect2D {
                offset:     vk::Offset2D { x: 0, y: 0 },
                extent:     self.swapchain_extent,
            },
            clear_value_count: clear_values.len() as u32,
            p_clear_values:    clear_values.as_ptr(),
            ..Default::default()
        };

        // Drawing
        unsafe {
            device.cmd_begin_render_pass(
                command_buffer,
                &render_pass_begin_info,
                vk::SubpassContents::INLINE,
            );
            device.cmd_bind_pipeline(
                command_buffer,
                vk::PipelineBindPoint::GRAPHICS,
                self.pipeline,
            );
            let viewport = vk::Viewport {
                x: 0.0,
                y: 0.0,
                width:     self.swapchain_extent.width  as f32,
                height:    self.swapchain_extent.height as f32,
                min_depth: 0.0,
                max_depth: 1.0,
            };
            let scissor = vk::Rect2D {
                offset: vk::Offset2D::default(),
                extent: self.swapchain_extent,
            };
            device.cmd_set_viewport(command_buffer, 0, &[viewport]);
            device.cmd_set_scissor(command_buffer, 0, &[scissor]);
            device.cmd_bind_vertex_buffers(
                command_buffer,
                0,
                &[self.vertex_buffer.buffer],
                &[0],
            );
            device.cmd_bind_index_buffer(
                command_buffer,
                self.index_buffer.buffer,
                0,
                vk::IndexType::UINT32,
            );
            device.cmd_bind_descriptor_sets(
                command_buffer,
                vk::PipelineBindPoint::GRAPHICS,
                self.pipeline_layout,
                0,
                &[self.descriptor_set],
                &[],
            );
            device.cmd_push_constants(
                command_buffer,
                self.pipeline_layout,
                vk::ShaderStageFlags::VERTEX,
                0,
                &self.color_mode.to_ne_bytes(),  
            );
            device.cmd_draw_indexed(
                command_buffer,
                self.sphere.indices.len() as u32,
                1,
                0,
                0,
                0,
            );

            device.cmd_end_render_pass(command_buffer);
            device.end_command_buffer(command_buffer)?;
        }

        Ok(())
    }

    /* Camera control updates */
    fn update_ubo(&mut self) {
        // Camera orientation encoded in the unit quaternion
        let rotation    = self.camera_rotation.to_rotation_matrix();
        let rotation_m  = rotation.matrix();
        let eye = self.camera_rotation * Vector3::new(0.0, 0.0, self.camera_distance);
        
        // Update view matrix
        let view = Matrix4::new(
            rotation_m[(0,0)],  rotation_m[(1,0)],  rotation_m[(2,0)],  -eye.dot(&rotation.matrix().column(0)),
            rotation_m[(0,1)],  rotation_m[(1,1)],  rotation_m[(2,1)],  -eye.dot(&rotation.matrix().column(1)),
            rotation_m[(0,2)],  rotation_m[(1,2)],  rotation_m[(2,2)],  -eye.dot(&rotation.matrix().column(2)),
            0.0,                0.0,                0.0,                 1.0,
        );

        self.ubo.view = view;
        unsafe { self.uniform_buffer_mapped.write(self.ubo); }
    }
}
/* Vulkan renderer cleanup, in reverse order of creation */
impl Drop for Renderer {
    fn drop(&mut self) {
        unsafe {
            // Wait for GPU to stop executing
            self.context.device.as_ref().unwrap()
                .device_wait_idle().unwrap();
            // Destroy vulkan handles in reverse order of creation
            // 10. Sync objects
            for i in 0..FRAMES_IN_FLIGHT as usize {
                self.context.device.as_ref().unwrap()
                    .destroy_semaphore(self.image_available_semaphores[i], None);
                self.context.device.as_ref().unwrap()
                    .destroy_semaphore(self.render_finished_semaphores[i], None);
                self.context.device.as_ref().unwrap()
                    .destroy_fence(self.in_flight_fences[i], None);
            }
            // 9. Descriptors and their descripts
            self.context.device.as_ref().unwrap().destroy_descriptor_pool(self.descriptor_pool, None);
            self.context.device.as_ref().unwrap().destroy_descriptor_set_layout(self.descriptor_set_layout, None);
            self.uniform_buffer.destroy(&self.context.device.as_ref().unwrap());
            self.index_buffer.destroy(&self.context.device.as_ref().unwrap());
            self.vertex_buffer.destroy(&self.context.device.as_ref().unwrap());
            // 8. Pipeline
            self.context.device.as_ref().unwrap().destroy_pipeline(self.pipeline, None);
            self.context.device.as_ref().unwrap().destroy_pipeline_layout(self.pipeline_layout, None);
            // 7. Command pool
            if self.command_pool != vk::CommandPool::null() {
                self.context.device.as_ref().unwrap()
                    .destroy_command_pool(self.command_pool, None);
            }
            // 6. Framebuffers
            for &framebuffer in &self.framebuffers {
                self.context.device.as_ref().unwrap()
                    .destroy_framebuffer(framebuffer, None);
            }
            // 5. Render pass
            if self.render_pass != vk::RenderPass::null() {
                self.context.device.as_ref().unwrap()
                    .destroy_render_pass(self.render_pass, None);
            }
            // 4. Image view
            self.context.device.as_ref().unwrap().destroy_image_view(self.depth_image_view, None);
            self.context.device.as_ref().unwrap().destroy_image(self.depth_image, None);
            self.context.device.as_ref().unwrap().free_memory(self.depth_image_memory, None);
            for &image_view in &self.swapchain_image_views {
                if image_view != vk::ImageView::null() {
                    self.context.device.as_ref().unwrap()
                        .destroy_image_view(image_view, None);
                }
            }
            // 3. Swapchain
            if self.swapchain != vk::SwapchainKHR::null() {
                self.context.swapchain_loader.as_ref().unwrap()
                    .destroy_swapchain(self.swapchain, None);
            }
            // 2. Surface
            if self.surface != vk::SurfaceKHR::null() {
                self.context.surface_loader.as_ref().unwrap()
                    .destroy_surface(self.surface, None);
            }
            // 1. explicit Window drop
            self.window = None;
            // Un-initialize, unnecessary but we might as well return to the default state before destroying
            self.initialized = false;
        }
    }
}

/* Implementation of winit::application::ApplicationHandler to call it's run_app method on EventLoop. This is where draw_frame() is called */ 
impl ApplicationHandler for Renderer {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        // Initialize once
        if !self.initialized {
            if let Err(e) = self.initialize(event_loop) {
                eprintln!("Failed to initialize renderer: {}", e);
                std::process::exit(1);
            }
            self.initialized = true;
        }
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        match event {
            // Window
            WindowEvent::CloseRequested => {
                event_loop.exit();
            }
            WindowEvent::Resized(new_size) => {
                recreate_swapchain(
                    self.context.instance.as_ref().unwrap(),
                    self.context.device.as_ref().unwrap(),
                    self.context.physical_device,
                    self.surface,
                    self.context.surface_loader.as_ref().unwrap(),
                    self.context.swapchain_loader.as_ref().unwrap(),
                    self.window.as_ref().unwrap(),
                    &mut self.swapchain,
                    &mut self.swapchain_format,
                    &mut self.swapchain_extent,
                    &mut self.swapchain_image_views,
                    &mut self.framebuffers,
                    &mut self.depth_image,
                    &mut self.depth_image_memory,
                    &mut self.depth_image_view,
                    self.render_pass,
                    new_size,
                ).expect("swapchain recreation failed");
                // TODO Swapchain recreation
            }

            // Color mode swap
            WindowEvent::KeyboardInput { event, .. } => {
                if event.state == ElementState::Pressed {
                    match event.logical_key {
                        Key::Character(ref s) => match s.as_str() {
                            "0" => self.color_mode = 0,
                            "1" => self.color_mode = 1,
                            "2" => self.color_mode = 2,
                            _ => {}
                        },
                        _ => {}
                    }
                }
            }

            // Mouse control for the camera
            WindowEvent::MouseInput { state, button: MouseButton::Left, .. } => {
                match state {
                    ElementState::Pressed  => {
                        self.camera_drag = true;
                    }
                    ElementState::Released => {
                        self.camera_drag     = false;
                        self.drag_last_pos   = None;
                    }
                }
            }
            WindowEvent::CursorMoved { position, .. } => {
                if self.camera_drag {
                    if let Some(last_pos) = self.drag_last_pos {
                        let dx = (position.x - last_pos.x) as f32; 
                        let dy = (position.y - last_pos.y) as f32; 

                        // Horizontal delta
                        let rot_z = UnitQuaternion::from_axis_angle(
                            &Vector3::z_axis(),
                            - dx * TURNTABLE_SENSITIVITY_H,
                        );

                        // Vertical delta with Y flipped because Vulkan
                        let rot_x = UnitQuaternion::from_axis_angle(
                            &Vector3::x_axis(),
                            - dy * TURNTABLE_SENSITIVITY_V,
                        );
                        
                        self.camera_rotation = rot_z * self.camera_rotation * rot_x;
                    }
                    self.drag_last_pos = Some(position);
                }
            }
            WindowEvent::MouseWheel { delta, .. } => {
                let scroll = match delta {
                    MouseScrollDelta::LineDelta(_, y) => y,
                    MouseScrollDelta::PixelDelta(pos) => pos.y as f32 * 0.01,
                };
                self.camera_distance = (self.camera_distance - scroll * 0.2)
                    .clamp(1.15, 3.0);
            }

            _ => {}
        }
    }

    fn about_to_wait(&mut self, _event_loop: &ActiveEventLoop) {
        if self.initialized {
            if let Err(e) = self.draw_frame() {
                eprintln!("Failed to draw frame: {}", e);
                std::process::exit(1);
            }
        }
    }
}

// Section2: Vulkan pipeline initialization functions -------------------------------------------------------------------------

/* Creates a surface where visuals will be rendered, this surface will then be pushed to the window frame */
fn create_surface(
    entry: &ash::Entry,
    instance: &ash::Instance,
    window: &winit::window::Window,
) -> Result<vk::SurfaceKHR, Box<dyn std::error::Error>> {
    let surface = unsafe {
        let allocation_callbacks = None;
        ash_window::create_surface(
            entry,
            instance,
            window.display_handle()?.as_raw(),
            window.window_handle()?.as_raw(),
            allocation_callbacks,
        )?
    };
    return Ok(surface);
}

/* Creates the swap chain, image queue waiting to be presented */
fn create_swap_chain(
    physical_device: vk::PhysicalDevice,
    surface: vk::SurfaceKHR,
    surface_loader: &ash::khr::surface::Instance,
    swapchain_loader: &ash::khr::swapchain::Device,
    window: &Window,
) -> Result<(vk::SwapchainKHR, vk::Format, vk::Extent2D), Box<dyn std::error::Error>> {
    // query surface capabilities
    let capabilities = unsafe {
        surface_loader.get_physical_device_surface_capabilities(physical_device, surface)?
    };
    let formats = unsafe {
        surface_loader.get_physical_device_surface_formats(physical_device, surface)?
    };
    // FIFO is guaranteed to be available
    let present_mode = vk::PresentModeKHR::FIFO;

    // pick format - prefer BGRA8 SRGB with nonlinear colorspace
    let format = formats.iter()
        .find(|f| {
            f.format == vk::Format::B8G8R8A8_SRGB
            && f.color_space == vk::ColorSpaceKHR::SRGB_NONLINEAR
        })
        .unwrap_or(&formats[0]);

    // pick swap extent
    let extent = if capabilities.current_extent.width != u32::MAX {
        capabilities.current_extent
    } else {
        let window_size = window.inner_size();
        vk::Extent2D {
            width: window_size.width.clamp(
                capabilities.min_image_extent.width,
                capabilities.max_image_extent.width,
            ),
            height: window_size.height.clamp(
                capabilities.min_image_extent.height,
                capabilities.max_image_extent.height,
            ),
        }
    };

    // 2 images for double buffering
    let image_count = FRAMES_IN_FLIGHT.clamp(
        capabilities.min_image_count,
        if capabilities.max_image_count == 0 { u32::MAX } else { capabilities.max_image_count },
    );

    let allocation_callbacks = None;
    let create_info = vk::SwapchainCreateInfoKHR {
        surface,
        min_image_count: image_count,
        image_format: format.format,
        image_color_space: format.color_space,
        image_extent: extent,
        image_array_layers: 1,
        image_usage: vk::ImageUsageFlags::COLOR_ATTACHMENT,
        image_sharing_mode: vk::SharingMode::EXCLUSIVE,
        pre_transform: capabilities.current_transform,
        composite_alpha: vk::CompositeAlphaFlagsKHR::OPAQUE,
        present_mode,
        clipped: vk::TRUE,
        ..Default::default()
    };

    let swapchain = unsafe {
        swapchain_loader.create_swapchain(&create_info, allocation_callbacks)?
    };

    return Ok((swapchain, format.format, extent));
}
/* Recreate the swapchain on window size change */
fn recreate_swapchain(
    instance:               &ash::Instance,
    device:                 &ash::Device,
    physical_device:        vk::PhysicalDevice,
    surface:                vk::SurfaceKHR,
    surface_loader:         &ash::khr::surface::Instance,
    swapchain_loader:       &ash::khr::swapchain::Device,
    window:                 &Window,
    swapchain:              &mut vk::SwapchainKHR,
    swapchain_format:       &mut vk::Format,
    swapchain_extent:       &mut vk::Extent2D,
    swapchain_image_views:  &mut Vec<vk::ImageView>,
    framebuffers:           &mut Vec<vk::Framebuffer>,
    depth_image:            &mut vk::Image,
    depth_image_memory:     &mut vk::DeviceMemory,
    depth_image_view:       &mut vk::ImageView,
    render_pass:            vk::RenderPass,
    new_size:               winit::dpi::PhysicalSize<u32>,
) -> Result<(), Box<dyn std::error::Error>> {
    if new_size.width == 0 || new_size.height == 0 { return Ok(()); }

    unsafe { device.device_wait_idle()?; }

    // destroy old resources
    unsafe {
        for fb in framebuffers.drain(..) {
            device.destroy_framebuffer(fb, None);
        }
        device.destroy_image_view(*depth_image_view, None);
        device.destroy_image(*depth_image, None);
        device.free_memory(*depth_image_memory, None);
        for iv in swapchain_image_views.drain(..) {
            device.destroy_image_view(iv, None);
        }
        swapchain_loader.destroy_swapchain(*swapchain, None);
    }

    // recreate swapchain
    let (new_swapchain, new_format, new_extent) = create_swap_chain(
        physical_device,
        surface,
        surface_loader,
        swapchain_loader,
        window,
    )?;
    *swapchain        = new_swapchain;
    *swapchain_format = new_format;
    *swapchain_extent = new_extent;

    // Recreate image views
    let (_new_swapchain_images, new_image_views) = create_image_views(
        device,
        swapchain_loader,
        *swapchain,
        *swapchain_format,
    )?;
    *swapchain_image_views = new_image_views;

    // Recreate depth image
    let (new_depth_image, new_depth_memory, new_depth_image_view) = create_depth_image(
        instance,
        physical_device,
        device,
        *swapchain_extent,
    )?;
    *depth_image        = new_depth_image;
    *depth_image_memory = new_depth_memory;
    *depth_image_view   = new_depth_image_view;

    // Recreate framebuffers
    *framebuffers = create_framebuffers(
        device,
        render_pass,
        swapchain_image_views,
        *depth_image_view,
        *swapchain_extent,
    )?;

    Ok(())
}

/* Create the visible part of the image presented */
fn create_image_views(
    device: &ash::Device,
    swapchain_loader: &ash::khr::swapchain::Device,
    swapchain: vk::SwapchainKHR,
    format: vk::Format,
) -> Result<(Vec<vk::Image>, Vec<vk::ImageView>), Box<dyn std::error::Error>> {
    let swapchain_images = unsafe {
        swapchain_loader.get_swapchain_images(swapchain)?
    };

    let image_views = swapchain_images.iter()
        .map(|&image| {
            let create_info = vk::ImageViewCreateInfo {
                image,
                view_type: vk::ImageViewType::TYPE_2D,
                format,
                components: vk::ComponentMapping {
                    r: vk::ComponentSwizzle::IDENTITY,
                    g: vk::ComponentSwizzle::IDENTITY,
                    b: vk::ComponentSwizzle::IDENTITY,
                    a: vk::ComponentSwizzle::IDENTITY,
                },
                subresource_range: vk::ImageSubresourceRange {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    base_mip_level: 0,
                    level_count: 1,
                    base_array_layer: 0,
                    layer_count: 1,
                },
                ..Default::default()
            };
            let allocation_callbacks = None;
            unsafe { device.create_image_view(&create_info, allocation_callbacks) }
        })
        .collect::<Result<Vec<_>, _>>()?;

    return Ok((swapchain_images, image_views));
}

fn create_depth_image(
    instance:        &ash::Instance,
    physical_device: vk::PhysicalDevice,
    device:          &ash::Device,
    extent:          vk::Extent2D,
) -> Result<(vk::Image, vk::DeviceMemory, vk::ImageView), Box<dyn std::error::Error>> {
    let format = vk::Format::D32_SFLOAT;
    // Create image
    let image_info = vk::ImageCreateInfo {
        image_type:   vk::ImageType::TYPE_2D,
        format,
        extent: vk::Extent3D {
            width:  extent.width,
            height: extent.height,
            depth:  1,
        },
        mip_levels:   1,
        array_layers: 1,
        samples:      vk::SampleCountFlags::TYPE_1,
        tiling:       vk::ImageTiling::OPTIMAL,
        usage:        vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT,
        sharing_mode: vk::SharingMode::EXCLUSIVE,
        ..Default::default()
    };
    let image = unsafe { device.create_image(&image_info, None)? };
    // Allocate and bind memory
    let mem_reqs = unsafe { device.get_image_memory_requirements(image) };
    let mem_type = find_memory_type(
        instance, physical_device,
        mem_reqs.memory_type_bits,
        vk::MemoryPropertyFlags::DEVICE_LOCAL,
    );
    let alloc_info = vk::MemoryAllocateInfo {
        allocation_size:   mem_reqs.size,
        memory_type_index: mem_type,
        ..Default::default()
    };
    let memory = unsafe { device.allocate_memory(&alloc_info, None)? };
    unsafe { device.bind_image_memory(image, memory, 0)? };
    // Create image view
    let view_info = vk::ImageViewCreateInfo {
        image,
        view_type: vk::ImageViewType::TYPE_2D,
        format,
        subresource_range: vk::ImageSubresourceRange {
            aspect_mask:     vk::ImageAspectFlags::DEPTH,
            base_mip_level:  0,
            level_count:     1,
            base_array_layer:0,
            layer_count:     1,
        },
        ..Default::default()
    };
    let view = unsafe { device.create_image_view(&view_info, None)? };
    Ok((image, memory, view))
}

/* Establish rendering information */ 
fn create_render_pass(
    device: &ash::Device,
    format: vk::Format,
) -> Result<vk::RenderPass, Box<dyn std::error::Error>> {
    let color_attachment = vk::AttachmentDescription {
        format,
        samples:            vk::SampleCountFlags::TYPE_1,
        load_op:            vk::AttachmentLoadOp::CLEAR,
        store_op:           vk::AttachmentStoreOp::STORE,
        stencil_load_op:    vk::AttachmentLoadOp::DONT_CARE,
        stencil_store_op:   vk::AttachmentStoreOp::DONT_CARE,
        initial_layout:     vk::ImageLayout::UNDEFINED,
        final_layout:       vk::ImageLayout::PRESENT_SRC_KHR,
        ..Default::default()
    };
    let depth_attachment = vk::AttachmentDescription {
        format:             vk::Format::D32_SFLOAT,
        samples:            vk::SampleCountFlags::TYPE_1,
        load_op:            vk::AttachmentLoadOp::CLEAR,
        store_op:           vk::AttachmentStoreOp::DONT_CARE,
        stencil_load_op:    vk::AttachmentLoadOp::DONT_CARE,
        stencil_store_op:   vk::AttachmentStoreOp::DONT_CARE,
        initial_layout:     vk::ImageLayout::UNDEFINED,
        final_layout:       vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
        ..Default::default()
    };
    let color_attachment_ref = vk::AttachmentReference {
        attachment: 0,
        layout:     vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
    };
    let depth_attachment_ref = vk::AttachmentReference {
        attachment: 1,
        layout:     vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
    };
    let subpass = vk::SubpassDescription {
        pipeline_bind_point:        vk::PipelineBindPoint::GRAPHICS,
        color_attachment_count:     1,
        p_color_attachments:        &color_attachment_ref,
        p_depth_stencil_attachment: &depth_attachment_ref,
        ..Default::default()
    };
    let dependency = vk::SubpassDependency {
        src_subpass:        vk::SUBPASS_EXTERNAL,
        dst_subpass:        0,
        src_stage_mask:     vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT
                            | vk::PipelineStageFlags::EARLY_FRAGMENT_TESTS,
        src_access_mask:    vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE,
        dst_stage_mask:     vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT
                            | vk::PipelineStageFlags::EARLY_FRAGMENT_TESTS,
        dst_access_mask:    vk::AccessFlags::COLOR_ATTACHMENT_WRITE
                            | vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE,
        ..Default::default()
    };
    let attachments = [color_attachment, depth_attachment];
    let create_info = vk::RenderPassCreateInfo {
        attachment_count:   attachments.len() as u32,
        p_attachments:      attachments.as_ptr(),
        subpass_count:      1,
        p_subpasses:        &subpass,
        dependency_count:   1,
        p_dependencies:     &dependency,
        ..Default::default()
    };
    Ok(unsafe { device.create_render_pass(&create_info, None)? })
}

/* References image view to present them to the screen */
fn create_framebuffers(
    device: &ash::Device,
    render_pass: vk::RenderPass,
    image_views: &[vk::ImageView],
    depth_image_view: vk::ImageView,
    extent: vk::Extent2D,
) -> Result<Vec<vk::Framebuffer>, Box<dyn std::error::Error>> {
    let framebuffers = image_views.iter()
        .map(|&image_view| {
            let attachments = [image_view, depth_image_view];
            let allocation_callbacks = None;
            let create_info = vk::FramebufferCreateInfo {
                render_pass,
                attachment_count:   attachments.len() as u32,
                p_attachments:      attachments.as_ptr(),
                width:              extent.width,
                height:             extent.height,
                layers:             1,
                ..Default::default()
            };
            unsafe { device.create_framebuffer(&create_info, allocation_callbacks) }
        }).collect::<Result<Vec<_>, _>>()?;
    return Ok(framebuffers);
}

/* Creates the buffer containing commands to send to the GPU */ 
fn create_command_buffers(
    device: &ash::Device,
    command_pool: vk::CommandPool,
) -> Result<Vec<vk::CommandBuffer>, Box<dyn std::error::Error>> {
    let allocate_info = vk::CommandBufferAllocateInfo {
        command_pool,
        level: vk::CommandBufferLevel::PRIMARY,
        command_buffer_count: FRAMES_IN_FLIGHT,
        ..Default::default()
    };

    let command_buffers = unsafe {
        device.allocate_command_buffers(&allocate_info)?
    };

    return Ok(command_buffers);
}

/* Specifies the type of ressources that will be accessed by the pipeline */ 
pub fn create_descriptor_set_layout(
    device: &ash::Device,
) -> Result<vk::DescriptorSetLayout, Box<dyn std::error::Error>> {
    // Binding 0: uniform buffer (vertex stage)
    let ubo_binding = vk::DescriptorSetLayoutBinding {
        binding:                0,
        descriptor_type:        vk::DescriptorType::UNIFORM_BUFFER,
        descriptor_count:       1,
        stage_flags:            vk::ShaderStageFlags::VERTEX,
        p_immutable_samplers:   std::ptr::null(),
        ..Default::default()
    };
    let layout_info = vk::DescriptorSetLayoutCreateInfo {
        binding_count:  1,
        p_bindings:     &ubo_binding,
        ..Default::default()
    };
    Ok(unsafe { device.create_descriptor_set_layout(&layout_info, None)? })
}

/* Used to create descriptor sets */
fn create_descriptor_pool(
    device: &ash::Device,
) -> Result<vk::DescriptorPool, Box<dyn std::error::Error>> {
    let pool_size = vk::DescriptorPoolSize {
        ty:                 vk::DescriptorType::UNIFORM_BUFFER,
        descriptor_count:   1,
    };
    let pool_info = vk::DescriptorPoolCreateInfo {
        max_sets:           1,
        pool_size_count:    1,
        p_pool_sizes:       &pool_size,
        ..Default::default()
    };
    Ok(unsafe { device.create_descriptor_pool(&pool_info, None)? })
}

/* Self explanatory */
fn allocate_descriptor_set(
    device:         &ash::Device,
    pool:           vk::DescriptorPool,
    layout:         vk::DescriptorSetLayout,
    uniform_buffer: &GpuBuffer,
) -> Result<vk::DescriptorSet, Box<dyn std::error::Error>> {
    let alloc_info = vk::DescriptorSetAllocateInfo {
        descriptor_pool:        pool,
        descriptor_set_count:   1,
        p_set_layouts:          &layout,
        ..Default::default()
    };
    let set = unsafe { device.allocate_descriptor_sets(&alloc_info)?[0] };

    // Point the UBO binding at our uniform buffer
    let buf_info = vk::DescriptorBufferInfo {
        buffer: uniform_buffer.buffer,
        offset: 0,
        range:  uniform_buffer.size,
    };
    let write = vk::WriteDescriptorSet {
        dst_set:            set,
        dst_binding:        0,
        dst_array_element:  0,
        descriptor_count:   1,
        descriptor_type:    vk::DescriptorType::UNIFORM_BUFFER,
        p_buffer_info:      &buf_info,
        ..Default::default()
    };
    unsafe { device.update_descriptor_sets(&[write], &[]) };
    Ok(set)
}

pub fn create_graphics_pipeline(
    device:                 &ash::Device,
    render_pass:            vk::RenderPass,
    extent:                 vk::Extent2D,
    descriptor_set_layout:  vk::DescriptorSetLayout,
) -> Result<(vk::PipelineLayout, vk::Pipeline), Box<dyn std::error::Error>> {
    // Load SPIR-V — place your compiled .spv next to the binary or embed with include_bytes!
    let vert_spv = include_bytes!("../shaders/vert.spv");
    let frag_spv = include_bytes!("../shaders/frag.spv");

    let vert_module = create_shader_module(device, vert_spv)?;
    let frag_module = create_shader_module(device, frag_spv)?;
    let entry_point = std::ffi::CString::new("main").unwrap();

    let shader_stages = [
        vk::PipelineShaderStageCreateInfo {
            stage:  vk::ShaderStageFlags::VERTEX,
            module: vert_module,
            p_name: entry_point.as_ptr(),
            ..Default::default()
        },
        vk::PipelineShaderStageCreateInfo {
            stage:  vk::ShaderStageFlags::FRAGMENT,
            module: frag_module,
            p_name: entry_point.as_ptr(),
            ..Default::default()
        },
    ];

    let binding_desc   = vk::VertexInputBindingDescription {
        binding:    0,
        stride:     std::mem::size_of::<Vertex>() as u32,
        input_rate: vk::VertexInputRate::VERTEX,
    };
    let attribute_desc = [
        vk::VertexInputAttributeDescription {                           // position
            location: 0,
            binding:  0,
            format:   vk::Format::R32G32B32_SFLOAT,                     // vec3 in glsl
            offset:   0,
        },
        vk::VertexInputAttributeDescription {                           // elevation
            location: 1,
            binding:  0,
            format:   vk::Format::R32G32B32_SFLOAT,                     // vec3 in glsl
            offset:   std::mem::offset_of!(Vertex, elevation) as u32,
        },
        vk::VertexInputAttributeDescription {                           // rgba
            location: 2,
            binding:  0,
            format:   vk::Format::R8G8B8A8_UNORM,                       // vec4 in glsl
            offset:   std::mem::offset_of!(Vertex, rgba) as u32,
        },
    ];

    let vertex_input = vk::PipelineVertexInputStateCreateInfo {
        vertex_binding_description_count:   1,
        p_vertex_binding_descriptions:      &binding_desc,
        vertex_attribute_description_count: attribute_desc.len() as u32,
        p_vertex_attribute_descriptions:    attribute_desc.as_ptr(),
        ..Default::default()
    };

    let input_assembly = vk::PipelineInputAssemblyStateCreateInfo {
        topology:                 vk::PrimitiveTopology::TRIANGLE_LIST,
        primitive_restart_enable: vk::FALSE,
        ..Default::default()
    };

    // viewport and scissor
    let dynamic_states = [vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR];
    let dynamic_state = vk::PipelineDynamicStateCreateInfo {
        dynamic_state_count: dynamic_states.len() as u32,
        p_dynamic_states:    dynamic_states.as_ptr(),
        ..Default::default()
    };
    let viewport_state = vk::PipelineViewportStateCreateInfo {
        viewport_count: 1,
        scissor_count:  1,
        ..Default::default()
    };

    let rasterizer = vk::PipelineRasterizationStateCreateInfo {
        polygon_mode:              vk::PolygonMode::FILL,
        cull_mode:                 vk::CullModeFlags::BACK,
        front_face:                vk::FrontFace::COUNTER_CLOCKWISE,
        line_width:                1.0,
        depth_clamp_enable:        vk::FALSE,
        rasterizer_discard_enable: vk::FALSE,
        depth_bias_enable:         vk::FALSE,
        ..Default::default()
    };

    let multisampling = vk::PipelineMultisampleStateCreateInfo {
        rasterization_samples: vk::SampleCountFlags::TYPE_1,
        sample_shading_enable: vk::FALSE,
        ..Default::default()
    };

    let depth_stencil = vk::PipelineDepthStencilStateCreateInfo {
        depth_test_enable:        vk::TRUE,
        depth_write_enable:       vk::TRUE,
        depth_compare_op:         vk::CompareOp::LESS,
        depth_bounds_test_enable: vk::FALSE,
        stencil_test_enable:      vk::FALSE,
        ..Default::default()
    };
    
    let color_blend_attachment = vk::PipelineColorBlendAttachmentState {
        color_write_mask: vk::ColorComponentFlags::RGBA,
        blend_enable:     vk::FALSE,
        ..Default::default()
    };
    let color_blending = vk::PipelineColorBlendStateCreateInfo {
        logic_op_enable:  vk::FALSE,
        attachment_count: 1,
        p_attachments:    &color_blend_attachment,
        ..Default::default()
    };

    let push_constant_range = vk::PushConstantRange {
        stage_flags: vk::ShaderStageFlags::VERTEX,
        offset:      0,
        size:        std::mem::size_of::<i32>() as u32,
    };

    let layout_info = vk::PipelineLayoutCreateInfo {
        set_layout_count:           1,
        p_set_layouts:              &descriptor_set_layout,
        push_constant_range_count:  1,
        p_push_constant_ranges:     &push_constant_range,
        ..Default::default()
    };
    let pipeline_layout = unsafe { device.create_pipeline_layout(&layout_info, None)? };

    let pipeline_info = vk::GraphicsPipelineCreateInfo {
        stage_count:            shader_stages.len() as u32,
        p_stages:               shader_stages.as_ptr(),
        p_vertex_input_state:   &vertex_input,
        p_input_assembly_state: &input_assembly,
        p_viewport_state:       &viewport_state,
        p_rasterization_state:  &rasterizer,
        p_multisample_state:    &multisampling,
        p_depth_stencil_state:  &depth_stencil,
        p_color_blend_state:    &color_blending,
        p_dynamic_state:        &dynamic_state,
        layout:                 pipeline_layout,
        render_pass,
        subpass:                0,
        ..Default::default()
    };

    let pipeline = unsafe {
        device.create_graphics_pipelines(vk::PipelineCache::null(), &[pipeline_info], None)
              .map_err(|(_, e)| e)?[0]
    };

    unsafe {
        device.destroy_shader_module(vert_module, None);
        device.destroy_shader_module(frag_module, None);
    }

    Ok((pipeline_layout, pipeline))
}
/* Helper function that creates shader modules from compiled shaders (the shaders should be compiled in build.rs) */
fn create_shader_module(
    device: &ash::Device,
    spv:    &[u8],
) -> Result<vk::ShaderModule, Box<dyn std::error::Error>> {
    // SPIR-V words must be u32-aligned
    assert!(spv.len() % 4 == 0, "SPIR-V byte length not multiple of 4");
    let spv_u32: Vec<u32> = spv.chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    let info = vk::ShaderModuleCreateInfo {
        code_size:  spv.len(),
        p_code:     spv_u32.as_ptr(),
        ..Default::default()
    };
    Ok(unsafe { device.create_shader_module(&info, None)? })
}

/* Creates the synchronization primitives: semaphores and fences */
fn create_sync_objects(
    device: &ash::Device,
) -> Result<(Vec<vk::Semaphore>, Vec<vk::Semaphore>, Vec<vk::Fence>), Box<dyn std::error::Error>> {
    let semaphore_create_info = vk::SemaphoreCreateInfo::default();
    let fence_create_info = vk::FenceCreateInfo {
        flags: vk::FenceCreateFlags::SIGNALED,
        ..Default::default()
    };
    let allocation_callbacks = None;

    let mut image_available_semaphores = Vec::new();
    let mut render_finished_semaphores = Vec::new();
    let mut in_flight_fences = Vec::new();

    for _ in 0..FRAMES_IN_FLIGHT {
        unsafe {
            image_available_semaphores.push(
                device.create_semaphore(&semaphore_create_info, allocation_callbacks)?
            );
            render_finished_semaphores.push(
                device.create_semaphore(&semaphore_create_info, allocation_callbacks)?
            );
            in_flight_fences.push(
                device.create_fence(&fence_create_info, allocation_callbacks)?
            );
        }
    }

    return Ok((image_available_semaphores, render_finished_semaphores, in_flight_fences));
}



// Section3: General helper functions -------------------------------------------------------------------------

/* Find a memory type index that satisfies both the type filter and the required property flags. */
pub fn find_memory_type(
    instance:           &ash::Instance,
    physical_device:    vk::PhysicalDevice,
    type_filter:        u32,
    properties:         vk::MemoryPropertyFlags,
) -> u32 {
    unsafe { 
        let memory_properties = instance.get_physical_device_memory_properties(physical_device); 
        for i in 0..memory_properties.memory_type_count {
            let type_matches = (type_filter & (1 << i)) != 0;
            let prop_matches = memory_properties.memory_types[i as usize]
                .property_flags
                .contains(properties);
            if type_matches && prop_matches {
                return i;
            }
        }
        panic!("No suitable memory type found");
    }
}
 
/* Allocate a raw Vulkan buffer with the given usage and memory properties. */
fn create_raw_buffer(
    instance:        &ash::Instance,
    physical_device: vk::PhysicalDevice,
    device:          &ash::Device,
    size:            vk::DeviceSize,
    usage:           vk::BufferUsageFlags,
    mem_props:       vk::MemoryPropertyFlags,
) -> Result<GpuBuffer, Box<dyn std::error::Error>> {
    let buffer_info = vk::BufferCreateInfo {
        size,
        usage,
        sharing_mode: vk::SharingMode::EXCLUSIVE,
        ..Default::default()
    };
    let buffer = unsafe { device.create_buffer(&buffer_info, None)? };

    let mem_reqs   = unsafe { device.get_buffer_memory_requirements(buffer) };
    let mem_type   = find_memory_type(instance, physical_device, mem_reqs.memory_type_bits, mem_props);

    let alloc_info = vk::MemoryAllocateInfo {
        allocation_size:   mem_reqs.size,
        memory_type_index: mem_type,
        ..Default::default()
    };
    let memory = unsafe { device.allocate_memory(&alloc_info, None)? };
    unsafe { device.bind_buffer_memory(buffer, memory, 0)? };

    return Ok(GpuBuffer { buffer, memory, size });
}

/* Copies data from src to dst using a temporary command buffer */
fn copy_buffer(
    device:         &ash::Device,
    command_pool:   vk::CommandPool,
    graphics_queue: vk::Queue,
    src:            vk::Buffer,
    dst:            vk::Buffer,
    size:           vk::DeviceSize,
) -> Result<(), Box<dyn std::error::Error>> {
    // Allocate command buffer
    let alloc_info = vk::CommandBufferAllocateInfo {
        command_pool,
        level:                vk::CommandBufferLevel::PRIMARY,
        command_buffer_count: 1,
        ..Default::default()
    };
    let cmd_buffer = unsafe { device.allocate_command_buffers(&alloc_info)?[0] };

    // Record
    let begin_info = vk::CommandBufferBeginInfo {
        flags: vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT,
        ..Default::default()
    };
    unsafe {
        device.begin_command_buffer(cmd_buffer, &begin_info)?;
        let region = vk::BufferCopy { src_offset: 0, dst_offset: 0, size };
        device.cmd_copy_buffer(cmd_buffer, src, dst, &[region]);
        device.end_command_buffer(cmd_buffer)?;
    }

    // Submit and wait
    let submit_info = vk::SubmitInfo {
        command_buffer_count: 1,
        p_command_buffers: &cmd_buffer,
        ..Default::default()
    };
    unsafe {
        device.queue_submit(graphics_queue, &[submit_info], vk::Fence::null())?;
        device.queue_wait_idle(graphics_queue)?;
        device.free_command_buffers(command_pool, &[cmd_buffer]);
    }

    return Ok(());
}

/* Uses a staging buffer with CPU visible memory type (suboptimal for buffers used by the GPU) to write data to a GPU destined buffer */
fn upload_to_device_local_buffer<T: Copy>(
    instance:           &ash::Instance,
    physical_device:    vk::PhysicalDevice,
    device:             &ash::Device,
    command_pool:       vk::CommandPool,
    graphics_queue:     vk::Queue,
    data:               &[T],
    usage:              vk::BufferUsageFlags,
) -> Result<GpuBuffer, Box<dyn std::error::Error>> {
    let size = (std::mem::size_of::<T>() * data.len()) as vk::DeviceSize;

    // CPU visible staging buffer
    let staging_buffer = create_raw_buffer(
        instance, physical_device, device, size,
        vk::BufferUsageFlags::TRANSFER_SRC,
        vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
    )?;
    unsafe {
        let ptr = device.map_memory(staging_buffer.memory, 0, size, vk::MemoryMapFlags::empty())? as *mut T;
        ptr.copy_from_nonoverlapping(data.as_ptr(), data.len());
        device.unmap_memory(staging_buffer.memory);
    }

    // Device-local buffer
    let gpu_buffer = create_raw_buffer(
        instance, physical_device, device, size,
        vk::BufferUsageFlags::TRANSFER_DST | usage,
        vk::MemoryPropertyFlags::DEVICE_LOCAL,
    )?;

    copy_buffer(device, command_pool, graphics_queue, staging_buffer.buffer, gpu_buffer.buffer, size)?;
    staging_buffer.destroy(device);

    return Ok(gpu_buffer);
}