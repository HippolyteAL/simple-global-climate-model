mod vulkan_context;
mod renderer;
use renderer::Renderer;
mod sphere;
use sphere::Sphere;
use winit::event_loop::EventLoop;

fn main() {
    // Initialize the window frame and event loop
    let event_loop = EventLoop::new().expect("Failed to create event loop");

    // Initialize the Vulkan context
    let vulkan_context = match vulkan_context::init_vulkan() {
        Ok(vulkan_context) => vulkan_context,
        Err(e) => {
            eprintln!("Failed to initialize Vulkan: {}", e);
            std::process::exit(1);
        }
    };

    let mut sphere = match Sphere::new(9) {
        Ok(sphere) => sphere,
        Err(e) => {
            eprintln!("Failed to initialize the sphere: {}", e);
            std::process::exit(1);
        }
    };

    // make true to write sphere
    let write_sphere = false;
    if write_sphere {
        if let Err(e) = sphere.write_positions("locations.bin") {
            eprintln!("Failed to write sphere data: {}", e);
            std::process::exit(1);
        }
    }

    // Needs a readable elevations.bin file for the subdivision count chosen in Sphere::new()
    if let Err(e) = sphere.read_elevations("elevations.bin") {
        eprintln!("Failed to read sphere data: {}", e);
        std::process::exit(1);
    }

    let mut renderer = Renderer::new(vulkan_context, sphere);

    // Render loop
    event_loop.run_app(&mut renderer).expect("Event loop failed");

    println!("Exiting main")
}


