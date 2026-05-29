use std::process::Command;

fn compile_shader(src: &str, dst: &str, stage: &str) {
    let glslc = r"C:\Dev\VulkanSDK\1.4.321.1\Bin\glslc.exe";
    let stage_flag = format!("-fshader-stage={}", stage);
    let status = Command::new(glslc)
        .args(&[&stage_flag, src, "-o", dst])
        .status()
        .expect("Failed to run glslc");
    if !status.success() {
        panic!("Shader compilation failed: {}", src);
    }
    println!("cargo:rerun-if-changed={}", src);
}

fn main() {
    compile_shader("shaders/vert.glsl", "shaders/vert.spv", "vertex");
    compile_shader("shaders/frag.glsl", "shaders/frag.spv", "fragment");
}