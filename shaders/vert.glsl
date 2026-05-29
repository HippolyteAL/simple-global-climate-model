#version 450

layout(location = 0) in vec3 position;  // [x, y, z]
layout(location = 1) in vec3 elevation; // [elevation, standard deviation, relief]
layout(location = 2) in vec4 rgba;      // debug colors (faces)

layout(location = 0) out vec4 frag_color;

layout(set = 0, binding = 0) uniform UniformBufferObject {
    mat4 model;
    mat4 view;
    mat4 proj;
} ubo;

layout(push_constant) uniform PushConstants {
    int colorMode;
} mode;

#define MIN_ELEV -7000.0
#define MAX_ELEV 6000.0
vec4 elevationColor(float e) {
    if (e >= 0.0) {
        // white to dark red (0.4 0.0 0.0)
        float t = pow(clamp(e / MAX_ELEV, 0.0, 1.0), 0.25);
        return vec4(1.0 - t * 0.6, 1.0 - t, 1.0 - t, 1.0);
    } else {
        // white to dark blue (0.0 0.0 0.4)
        float t = pow(clamp(-e / -MIN_ELEV, 0.0, 1.0), 0.25);
        return vec4(1.0 - t, 1.0 - t, 1.0 - t * 0.6, 1.0);
    }
}
vec4 terrainColor(float elevation, float stddev, float relief) {
    if (elevation <= 0.0) {
        if      (elevation > -400.0)    return vec4(0.42, 0.64, 0.71, 1.0); // coastline
        else if (elevation > -3000.0)   return vec4(0.25, 0.50, 0.64, 1.0); // coastal sea
        else                            return vec4(0.13, 0.33, 0.52, 1.0); // open sea
    }
    if      (relief > 800.0)                            return vec4(0.412, 0.094, 0.016, 1.0);  // mountains        #691804
    else if (relief > 400.0  && elevation <= 400.0)     return vec4(0.690, 0.506, 0.082, 1.0);  // large hills      #B08115
    else if (relief > 150.0  && elevation <= 400.0)     return vec4(0.949, 0.949, 0.435, 1.0);  // hills            #F2F26F
    else if (relief > 150.0  && elevation >  400.0)     return vec4(0.902, 0.725, 0.082, 1.0);  // highland hills   #E6B915
    else if (relief <= 150.0 && elevation >  400.0)     return vec4(0.161, 0.608, 0.086, 1.0);  // highland plains  #299B16
    else                                                return vec4(0.353, 0.922, 0.106, 1.0);  // plains           #5AEB1B
}

void main() {
    gl_Position = ubo.proj * ubo.view * ubo.model * vec4(position, 1.0);
    
    if      (mode.colorMode == 1)   frag_color = elevationColor(elevation.x);
    else if (mode.colorMode == 2)   frag_color = terrainColor(elevation.x, elevation.y, elevation.z);
    else                            frag_color = rgba;
}