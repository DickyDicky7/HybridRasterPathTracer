    #version 430
//  #version 430
    layout(local_size_x = 16, local_size_y = 16) in;
//  layout(local_size_x = 16, local_size_y = 16) in;

    layout(binding = 0, rgba32f) uniform image2D textureOutput;
//  layout(binding = 0, rgba32f) uniform image2D textureOutput;
    layout(binding = 1, rgba32f) uniform image2D textureInput;
//  layout(binding = 1, rgba32f) uniform image2D textureInput;

    // Vignette Parameters: Dictates the core visual framing profile. The intensity scalar drives the darkening gradient, smoothness governs the geometric falloff transition from the center, and the color override allows for cinematic tinting of the attenuated border regions.
    // Vignette Parameters: Dictates the core visual framing profile. The intensity scalar drives the darkening gradient, smoothness governs the geometric falloff transition from the center, and the color override allows for cinematic tinting of the attenuated border regions.
    uniform float uIntensity = 1.0;
//  uniform float uIntensity = 1.0;
    uniform float uSmoothness = 0.5;
//  uniform float uSmoothness = 0.5;
    uniform vec3 uColor = vec3(0.0);
//  uniform vec3 uColor = vec3(0.0);

    // Constants
//  // Constants
    const float VIGNETTE_START = 0.8;
//  const float VIGNETTE_START = 0.8;
    const float VIGNETTE_CURVE = 1.2;
//  const float VIGNETTE_CURVE = 1.2;

    // Interleaved Gradient Dither: A deterministic high-frequency screen-space hash evaluated per pixel, returning a signed offset smaller than a single 8-bit display code value. Injecting this into the attenuated result perturbs the otherwise mathematically perfect falloff below the quantisation threshold, dissolving the concentric banding rings that low bit-depth output would expose across the gentle gradient.
    // Interleaved Gradient Dither: A deterministic high-frequency screen-space hash evaluated per pixel, returning a signed offset smaller than a single 8-bit display code value. Injecting this into the attenuated result perturbs the otherwise mathematically perfect falloff below the quantisation threshold, dissolving the concentric banding rings that low bit-depth output would expose across the gentle gradient.
    float ditherValue(vec2 p) {
//  float ditherValue(vec2 p) {
        float noise = fract(52.9829189 * fract(dot(p, vec2(0.06711056, 0.00583715))));
//      float noise = fract(52.9829189 * fract(dot(p, vec2(0.06711056, 0.00583715))));
        return (noise - 0.5) / 255.0;
//      return (noise - 0.5) / 255.0;
    }
//  }

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

        // Aspect Ratio Normalization: Modifies the radial distance coordinate vector by scaling the horizontal axis inversely to the viewport resolution. This mathematically preserves a perfectly circular vignette gradient structure irrespective of the physical window or screen dimensions.
        // Aspect Ratio Normalization: Modifies the radial distance coordinate vector by scaling the horizontal axis inversely to the viewport resolution. This mathematically preserves a perfectly circular vignette gradient structure irrespective of the physical window or screen dimensions.
        vec2 dir = uv - 0.5;
//      vec2 dir = uv - 0.5;
        dir.x *= float(size.x) / float(size.y);
//      dir.x *= float(size.x) / float(size.y);

        // Distance Evaluation: Determines the Euclidean magnitude from the normalized screen center to the current fragment coordinate.
        // Distance Evaluation: Determines the Euclidean magnitude from the normalized screen center to the current fragment coordinate.
        float dist = length(dir);
//      float dist = length(dir);

        // Filmic Attenuation Curve: Employs a non-linear smoothstep function to calculate the foundational vignette mask based on the spatial distance. A supplementary power curve is subsequently applied to sculpt a physically accurate light drop-off reminiscent of classic cinema lenses.
        // Filmic Attenuation Curve: Employs a non-linear smoothstep function to calculate the foundational vignette mask based on the spatial distance. A supplementary power curve is subsequently applied to sculpt a physically accurate light drop-off reminiscent of classic cinema lenses.
        float vignette = 1.0 - smoothstep(uSmoothness * 0.5, VIGNETTE_START + uSmoothness, dist * uIntensity);
//      float vignette = 1.0 - smoothstep(uSmoothness * 0.5, VIGNETTE_START + uSmoothness, dist * uIntensity);
        vignette = pow(vignette, VIGNETTE_CURVE);
//      vignette = pow(vignette, VIGNETTE_CURVE);

        // Color Blending Integration: Transitions smoothly between the custom vignette color value and the underlying rendered fragment color utilizing the calculated attenuation mask as a linear interpolant, avoiding the muddy results typical of rudimentary multiplicative darkening methods.
        // Color Blending Integration: Transitions smoothly between the custom vignette color value and the underlying rendered fragment color utilizing the calculated attenuation mask as a linear interpolant, avoiding the muddy results typical of rudimentary multiplicative darkening methods.
        color.rgb = mix(uColor, color.rgb, vignette);
//      color.rgb = mix(uColor, color.rgb, vignette);

        // Dither Injection: Add a sub-quantisation offset to break up the smoothstep banding before the result is written out for display.
        // Dither Injection: Add a sub-quantisation offset to break up the smoothstep banding before the result is written out for display.
        color.rgb += ditherValue(vec2(coord));
//      color.rgb += ditherValue(vec2(coord));

        imageStore(textureOutput, coord, vec4(color.rgb, 1.0));
//      imageStore(textureOutput, coord, vec4(color.rgb, 1.0));
    }
//  }