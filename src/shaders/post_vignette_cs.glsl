    #version 430
//  #version 430
    layout(local_size_x = 16, local_size_y = 16) in;
//  layout(local_size_x = 16, local_size_y = 16) in;

    layout(binding = 0, rgba32f) uniform image2D textureOutput;
//  layout(binding = 0, rgba32f) uniform image2D textureOutput;
    layout(binding = 1, rgba32f) uniform image2D textureInput;
//  layout(binding = 1, rgba32f) uniform image2D textureInput;

    // Vignette intensity, smoothness, and color
    // Vignette intensity, smoothness, and color
    uniform float uIntensity = 1.0;
//  uniform float uIntensity = 1.0;
    uniform float uSmoothness = 0.5;
//  uniform float uSmoothness = 0.5;
    uniform vec3 uColor = vec3(0.0);
//  uniform vec3 uColor = vec3(0.0);

    void main() {
//  void main() {
        ivec2 coord = ivec2(gl_GlobalInvocationID.xy);
//      ivec2 coord = ivec2(gl_GlobalInvocationID.xy);
        ivec2 size = imageSize(textureOutput);
//      ivec2 size = imageSize(textureOutput);

        if (coord.x >= size.x || coord.y >= size.y) {
//      if (coord.x >= size.x || coord.y >= size.y) {
            return;
//          return;
        }
//      }

        vec2 uv = vec2(coord) / vec2(size);
//      vec2 uv = vec2(coord) / vec2(size);
        vec4 color = imageLoad(textureInput, coord);
//      vec4 color = imageLoad(textureInput, coord);

        // Enhance: aspect ratio correction for circular vignette
        // Enhance: aspect ratio correction for circular vignette
        vec2 dir = uv - 0.5;
//      vec2 dir = uv - 0.5;
        dir.x *= float(size.x) / float(size.y);
//      dir.x *= float(size.x) / float(size.y);

        // Calculate distance from center
        // Calculate distance from center
        float dist = length(dir);
//      float dist = length(dir);

        // Enhance: smoothstep and subtle power curve for filmic vignette
        // Enhance: smoothstep and subtle power curve for filmic vignette
        float vignette = 1.0 - smoothstep(uSmoothness * 0.5, 0.8 + uSmoothness, dist * uIntensity);
//      float vignette = 1.0 - smoothstep(uSmoothness * 0.5, 0.8 + uSmoothness, dist * uIntensity);
        vignette = pow(vignette, 1.2);
//      vignette = pow(vignette, 1.2);

        // Enhance: Mix with vignette color instead of pure multiply
        // Enhance: Mix with vignette color instead of pure multiply
        color.rgb = mix(uColor, color.rgb, vignette);
//      color.rgb = mix(uColor, color.rgb, vignette);

        imageStore(textureOutput, coord, vec4(color.rgb, 1.0));
//      imageStore(textureOutput, coord, vec4(color.rgb, 1.0));
    }
//  }