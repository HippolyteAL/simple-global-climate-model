#version 450

// change to true to look at only vertices for low subdivision spheres
#define VERTICES_ONLY false
#if VERTICES_ONLY
#extension GL_EXT_fragment_shader_barycentric : require
#endif

layout(location = 0) in vec4 frag_color;

layout(location = 0) out vec4 out_color;

void main() {
    #if VERTICES_ONLY
        vec3  bary    = gl_BaryCoordEXT;
        float minBary = min(bary.x, min(bary.y, bary.z));
        if (minBary < 0.02)
            out_color = frag_color;
        else
            discard;
    #else
        out_color = frag_color;
    #endif
}