/*
SPDX-License-Identifier:    GPL-3.0-only
Copyright (C) 2026 Hippolyte Audet-Lagacé
Author:         Hippolyte Audet-Lagacé
Description:    This program implements a vulkan context that can be used to create a pipeline, also implements 
                a basic vulkan validation layer to report GPU-side bugs that aren't caught by rust.
                This approach keeps the informations directly related to hardware isolated and easily accessible.
*/

use ash::vk;
use std::ffi::{CString, CStr};

const VALIDATION_LAYERS: &[&str] = &["VK_LAYER_KHRONOS_validation"];

const REQUIRED_DEVICE_EXTENSIONS: &[*const i8] = &[
    ash::khr::swapchain::NAME.as_ptr(),
    ash::khr::fragment_shader_barycentric::NAME.as_ptr(),
];

// Vulkan context pipeline
pub struct VulkanContext {
    pub entry:              Option<ash::Entry>,
    pub instance:           Option<ash::Instance>,
    debug_messenger:        Option<vk::DebugUtilsMessengerEXT>,
    debug_utils:            Option<ash::ext::debug_utils::Instance>,
    pub physical_device:    vk::PhysicalDevice,
    pub device:             Option<ash::Device>,
    pub queue_families:     Option<QueueFamilyIndices>,
    pub surface_loader:     Option<ash::khr::surface::Instance>,
    pub swapchain_loader:   Option<ash::khr::swapchain::Device>,
}
pub struct QueueFamilyIndices {
    pub graphics: u32,
    pub compute: u32,
    pub transfer: u32,
}

/* Initialization of the vulkan handles to null() or None */
impl Default for VulkanContext {
    fn default() -> Self {
        VulkanContext {
            entry:              None,
            instance:           None,
            debug_messenger:    None,
            debug_utils:        None,
            physical_device:    vk::PhysicalDevice::null(),
            device:             None,
            queue_families:     None,
            surface_loader:     None,
            swapchain_loader:   None,
        }
    }
}

/* Vulkan context cleanup, in reverse order of creation */
impl Drop for VulkanContext {
    fn drop(&mut self) {
        unsafe {
            // Destroy vulkan handles in reverse order of creation
            // 3. Logical device 
            if let Some(device) = &self.device {
                device.destroy_device(None);
            }
            // 2. debug messenger 
            if let (Some(debug_utils), Some(messenger)) = 
                (&self.debug_utils, self.debug_messenger) {
                debug_utils.destroy_debug_utils_messenger(messenger, None);
            }
            // 1. instance
            if let Some(instance) = &self.instance {
                instance.destroy_instance(None);
            }
        }
    }
}

/* Fills the VulkanContext struct with usable values */ 
pub fn init_vulkan() -> Result<VulkanContext, Box<dyn std::error::Error>> {
    let mut vulkan_context = VulkanContext::default();

    // Load vulkan
    vulkan_context.entry = unsafe { Some(ash::Entry::load()?) };

    // Create instance
    vulkan_context.instance = Some(create_instance(vulkan_context.entry.as_ref().unwrap())?);

    // Initialize vulkan GPU debug messenger
    let debug_utils = ash::ext::debug_utils::Instance::new(
        vulkan_context.entry.as_ref().unwrap(),
        vulkan_context.instance.as_ref().unwrap()
    );
    vulkan_context.debug_messenger = Some(create_debug_messenger(
        vulkan_context.entry.as_ref().unwrap(),
        vulkan_context.instance.as_ref().unwrap()
    )?);
    vulkan_context.debug_utils = Some(debug_utils);

    // Pick a GPU
    vulkan_context.physical_device = pick_physical_device(vulkan_context.instance.as_ref().unwrap())?;

    // Create logical device
    let (device, indices) = create_logical_device(vulkan_context.instance.as_ref().unwrap(), vulkan_context.physical_device)?;
    vulkan_context.device = Some(device);
    vulkan_context.queue_families = Some(indices);

    // Loads extensions used throughout the program
    let (surface_loader, swapchain_loader) = load_extension_functions(
        vulkan_context.entry.as_ref().unwrap(),
        vulkan_context.instance.as_ref().unwrap(),
        vulkan_context.device.as_ref().unwrap(),
    );
    vulkan_context.surface_loader = Some(surface_loader);
    vulkan_context.swapchain_loader = Some(swapchain_loader);

    return Ok(vulkan_context);
}

/* Creates the connection between the application and the vulkan library */
fn create_instance(entry: &ash::Entry) -> Result<ash::Instance, Box<dyn std::error::Error>> {
    let app_name = CString::new("Placeholder Vulkan App Name")?;
    let engine_name = CString::new("Placeholder Engine Name")?;

    let app_info = vk::ApplicationInfo {
        p_application_name: app_name.as_ptr(),
        application_version: vk::make_api_version(0, 1, 0, 0),
        p_engine_name: engine_name.as_ptr(),
        engine_version: vk::make_api_version(0, 1, 0, 0),
        api_version: vk::API_VERSION_1_3,
        ..Default::default()
    };

    let extensions = vec![
        ash::khr::surface::NAME.as_ptr(),
        ash::khr::win32_surface::NAME.as_ptr(),
        ash::ext::debug_utils::NAME.as_ptr(),
    ];

    let layer_names: Vec<CString> = VALIDATION_LAYERS
        .iter()
        .map(|&s| CString::new(s).unwrap())
        .collect();
    let layer_ptrs: Vec<*const i8> = layer_names.iter().map(|s| s.as_ptr()).collect();

    let create_info = vk::InstanceCreateInfo {
        p_application_info: &app_info,
        enabled_extension_count: extensions.len() as u32,
        pp_enabled_extension_names: extensions.as_ptr(),
        enabled_layer_count: layer_ptrs.len() as u32,
        pp_enabled_layer_names: layer_ptrs.as_ptr(),
        ..Default::default()
    };

    let instance = unsafe { entry.create_instance(&create_info, None)? };
    return Ok(instance);
}

/* Handles GPU error callbacks */
fn create_debug_messenger(
    entry: &ash::Entry,
    instance: &ash::Instance,
) -> Result<vk::DebugUtilsMessengerEXT, Box<dyn std::error::Error>> {
    let create_info = vk::DebugUtilsMessengerCreateInfoEXT {
        message_severity: vk::DebugUtilsMessageSeverityFlagsEXT::ERROR
            | vk::DebugUtilsMessageSeverityFlagsEXT::WARNING,
        message_type: vk::DebugUtilsMessageTypeFlagsEXT::GENERAL
            | vk::DebugUtilsMessageTypeFlagsEXT::VALIDATION
            | vk::DebugUtilsMessageTypeFlagsEXT::PERFORMANCE,
        pfn_user_callback: Some(debug_callback),
        ..Default::default()
    };

    let debug_utils = ash::ext::debug_utils::Instance::new(entry, instance);
    let messenger = unsafe { debug_utils.create_debug_utils_messenger(&create_info, None)? };
    return Ok(messenger);
}
/* Helper function, handles the internal vulkan calls (extern "system") for debug, requires dereferencing raw pointers (unsafe) */
unsafe extern "system" fn debug_callback(
    message_severity: vk::DebugUtilsMessageSeverityFlagsEXT,
    message_type: vk::DebugUtilsMessageTypeFlagsEXT,
    p_callback_data: *const vk::DebugUtilsMessengerCallbackDataEXT,
    _user_data: *mut std::ffi::c_void,
) -> vk::Bool32 {
    let message = unsafe { CStr::from_ptr((*p_callback_data).p_message) };
    eprintln!("[Vulkan] [{:?}] [{:?}] {:?}", message_severity, message_type, message);
    return vk::FALSE;
}

/* Select a physical GPU, priorising discrete GPU and requiring all 3 queue families */
fn pick_physical_device(instance: &ash::Instance) -> Result<vk::PhysicalDevice, Box<dyn std::error::Error>> {
    let devices = unsafe { instance.enumerate_physical_devices()? };

    if devices.is_empty() {
        return Err("No Vulkan capable GPU found".into());
    }

    // prefer discrete, fall back to anything available
    let physical_device = devices.iter()
        .max_by_key(|&&device| {
            let properties = unsafe { instance.get_physical_device_properties(device) };
            match properties.device_type {
                vk::PhysicalDeviceType::DISCRETE_GPU => 2,
                vk::PhysicalDeviceType::INTEGRATED_GPU => 1,
                _ => 0,
            }
        })
        .copied()
        .ok_or("Failed to find suitable GPU")?;

    // Checks extension support
    let supported_extensions = unsafe {
        instance.enumerate_device_extension_properties(physical_device).unwrap()
    };
    let all_supported = REQUIRED_DEVICE_EXTENSIONS.iter().all(|&required| {
        let required = unsafe { std::ffi::CStr::from_ptr(required) };
        supported_extensions.iter().any(|ext| {
            ext.extension_name_as_c_str().unwrap() == required
        })
    });
    assert!(all_supported, "Required device extensions not supported on this GPU");

    // log which device was selected
    let properties = unsafe { instance.get_physical_device_properties(physical_device) };
    let device_name = unsafe { CStr::from_ptr(properties.device_name.as_ptr()) };
    println!("Selected GPU: {:?}", device_name);

    // verify required queue families exist
    let queue_families = unsafe { 
        instance.get_physical_device_queue_family_properties(physical_device) 
    };

    let has_graphics = queue_families.iter().any(|q| 
        q.queue_flags.contains(vk::QueueFlags::GRAPHICS));
    let has_compute = queue_families.iter().any(|q| 
        q.queue_flags.contains(vk::QueueFlags::COMPUTE));
    let has_transfer = queue_families.iter().any(|q| 
        q.queue_flags.contains(vk::QueueFlags::TRANSFER));

    if !has_graphics || !has_compute || !has_transfer {
        return Err("GPU does not support required queue families".into());
    }

    return Ok(physical_device);
}

/* Creates the interface between the program and the physical device (GPU) */
fn create_logical_device(
    instance: &ash::Instance,
    physical_device: vk::PhysicalDevice,
) -> Result<(ash::Device, QueueFamilyIndices), Box<dyn std::error::Error>> {
    let indices = find_queue_families(instance, physical_device)?;

    // deduplicate queue families since graphics/compute/transfer may share indices
    let mut unique_families = std::collections::HashSet::new();
    unique_families.insert(indices.graphics);
    unique_families.insert(indices.compute);
    unique_families.insert(indices.transfer);

    let queue_priority = 1.0f32;
    let queue_create_infos: Vec<vk::DeviceQueueCreateInfo> = unique_families.iter()
        .map(|&index| vk::DeviceQueueCreateInfo {
            queue_family_index: index,
            queue_count: 1,
            p_queue_priorities: &queue_priority,
            ..Default::default()
        })
        .collect();

    let device_features = vk::PhysicalDeviceFeatures::default();

    let mut barycentric_feature =
    vk::PhysicalDeviceFragmentShaderBarycentricFeaturesKHR::default()
        .fragment_shader_barycentric(true);

    let create_info = vk::DeviceCreateInfo {
        queue_create_info_count:    queue_create_infos.len() as u32,
        p_queue_create_infos:       queue_create_infos.as_ptr(),
        enabled_extension_count:    REQUIRED_DEVICE_EXTENSIONS.len() as u32,
        pp_enabled_extension_names: REQUIRED_DEVICE_EXTENSIONS.as_ptr(),
        p_enabled_features:         &device_features,
        p_next:                     &mut barycentric_feature
                                    as *mut vk::PhysicalDeviceFragmentShaderBarycentricFeaturesKHR
                                    as *mut std::ffi::c_void,
        ..Default::default()
    };

    let device = unsafe { instance.create_device(physical_device, &create_info, None)? };
    return Ok((device, indices));
}

/* Helper function for create_logical_device() to find appropriate queue families */
fn find_queue_families(
    instance: &ash::Instance,
    physical_device: vk::PhysicalDevice,
) -> Result<QueueFamilyIndices, Box<dyn std::error::Error>> {
    let queue_families = unsafe {
        instance.get_physical_device_queue_family_properties(physical_device)
    };

    let mut graphics = None;
    let mut compute = None;
    let mut transfer = None;

    for (i, queue) in queue_families.iter().enumerate() {
        if queue.queue_flags.contains(vk::QueueFlags::GRAPHICS) && graphics.is_none() {
            graphics = Some(i as u32);
        }
        if queue.queue_flags.contains(vk::QueueFlags::COMPUTE) && compute.is_none() {
            compute = Some(i as u32);
        }
        if queue.queue_flags.contains(vk::QueueFlags::TRANSFER) && transfer.is_none() {
            transfer = Some(i as u32);
        }
    }

    return Ok(QueueFamilyIndices {
        graphics: graphics.ok_or("No graphics queue family found")?,
        compute: compute.ok_or("No compute queue family found")?,
        transfer: transfer.ok_or("No transfer queue family found")?,
    });
}

/* Loads the extensions that will be required by other parts of the code */
fn load_extension_functions(
    entry: &ash::Entry,
    instance: &ash::Instance,
    device: &ash::Device,
) -> (ash::khr::surface::Instance, ash::khr::swapchain::Device) {
    let surface_loader = ash::khr::surface::Instance::new(entry, instance);
    let swapchain_loader = ash::khr::swapchain::Device::new(instance, device);
    return (surface_loader, swapchain_loader);
}

// Public functions -------------------------------------------------------------------------------------------------------------

/* Used to create command buffers */ 
pub fn create_command_pool(
    device: &ash::Device,
    queue_family_index: u32,
) -> Result<vk::CommandPool, Box<dyn std::error::Error>> {
    let create_info = vk::CommandPoolCreateInfo {
        queue_family_index,
        flags: vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER,
        ..Default::default()
    };

    let allocation_callbacks = None;
    let command_pool = unsafe {
        device.create_command_pool(&create_info, allocation_callbacks)?
    };

    return Ok(command_pool);
}