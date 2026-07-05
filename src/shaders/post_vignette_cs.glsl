    #version 430
//  #version 430
    layout(local_size_x = 16, local_size_y = 16) in;
//  layout(local_size_x = 16, local_size_y = 16) in;

    // rgba16f matches the f2 post-processing ping-pong pair (LDR data - no visible precision loss)
//  // rgba16f matches the f2 post-processing ping-pong pair (LDR data - no visible precision loss)
    layout(binding = 0, rgba16f) uniform image2D textureOutput;
//  layout(binding = 0, rgba16f) uniform image2D textureOutput;
    layout(binding = 1, rgba16f) uniform image2D textureInput;
//  layout(binding = 1, rgba16f) uniform image2D textureInput;

    // Vignette Parameters: Define the core framing profile. The intensity scalar drives the darkening gradient, the smoothness governs the falloff transition out from the centre, and the color override allows for cinematic tinting of the attenuated border regions.
//  // Vignette Parameters: Define the core framing profile. The intensity scalar drives the darkening gradient, the smoothness governs the falloff transition out from the centre, and the color override allows for cinematic tinting of the attenuated border regions.
    uniform float uVignetteIntensity = 1.0;
//  uniform float uVignetteIntensity = 1.0;
    uniform float uFalloffSmoothness = 0.5;
//  uniform float uFalloffSmoothness = 0.5;
    uniform vec3 uVignetteColor = vec3(0.0);
//  uniform vec3 uVignetteColor = vec3(0.0);

    // Constants
//  // Constants
    const float VIGNETTE_FALLOFF_START_RADIUS = 0.8;
//  const float VIGNETTE_FALLOFF_START_RADIUS = 0.8;
    const float VIGNETTE_FALLOFF_EXPONENT = 1.2;
//  const float VIGNETTE_FALLOFF_EXPONENT = 1.2;

    // Interleaved Gradient Dither: A deterministic high-frequency screen-space hash evaluated per pixel, returning a signed offset smaller than a single 8-bit display code value. Injecting this into the attenuated result perturbs the otherwise mathematically perfect falloff below the quantisation threshold, dissolving the concentric banding rings that low-bit-depth output would otherwise expose across the gentle gradient.
//  // Interleaved Gradient Dither: A deterministic high-frequency screen-space hash evaluated per pixel, returning a signed offset smaller than a single 8-bit display code value. Injecting this into the attenuated result perturbs the otherwise mathematically perfect falloff below the quantisation threshold, dissolving the concentric banding rings that low-bit-depth output would otherwise expose across the gentle gradient.
    float computeDitherOffset(vec2 pixelCoord) {
//  float computeDitherOffset(vec2 pixelCoord) {
        float interleavedGradientNoise = fract(52.9829189 * fract(dot(pixelCoord, vec2(0.06711056, 0.00583715))));
//      float interleavedGradientNoise = fract(52.9829189 * fract(dot(pixelCoord, vec2(0.06711056, 0.00583715))));
        return (interleavedGradientNoise - 0.5) / 255.0;
//      return (interleavedGradientNoise - 0.5) / 255.0;
    }
//  }

    void main() {
//  void main() {
        ivec2 destinationPixelCoord = ivec2(gl_GlobalInvocationID.xy);
//      ivec2 destinationPixelCoord = ivec2(gl_GlobalInvocationID.xy);
        ivec2 outputImageSize = imageSize(textureOutput);
//      ivec2 outputImageSize = imageSize(textureOutput);

        if (destinationPixelCoord.x >= outputImageSize.x || destinationPixelCoord.y >= outputImageSize.y) {
//      if (destinationPixelCoord.x >= outputImageSize.x || destinationPixelCoord.y >= outputImageSize.y) {
            return;
//          return;
        }
//      }

        vec2 pixelUv = vec2(destinationPixelCoord) / vec2(outputImageSize);
//      vec2 pixelUv = vec2(destinationPixelCoord) / vec2(outputImageSize);
        vec4 pixelColor = imageLoad(textureInput, destinationPixelCoord);
//      vec4 pixelColor = imageLoad(textureInput, destinationPixelCoord);

        // Aspect Ratio Correction: Scales the horizontal component of the direction vector by the viewport aspect ratio. This keeps the vignette gradient perfectly circular regardless of the window or screen dimensions.
//      // Aspect Ratio Correction: Scales the horizontal component of the direction vector by the viewport aspect ratio. This keeps the vignette gradient perfectly circular regardless of the window or screen dimensions.
        vec2 directionFromCenter = pixelUv - 0.5;
//      vec2 directionFromCenter = pixelUv - 0.5;
        directionFromCenter.x *= float(outputImageSize.x) / float(outputImageSize.y);
//      directionFromCenter.x *= float(outputImageSize.x) / float(outputImageSize.y);

        // Radial Distance: Computes the Euclidean distance from the normalized screen centre to the current fragment.
//      // Radial Distance: Computes the Euclidean distance from the normalized screen centre to the current fragment.
        float distanceFromCenter = length(directionFromCenter);
//      float distanceFromCenter = length(directionFromCenter);

        // Filmic Attenuation Curve: Uses a smoothstep function to build the base vignette mask from the radial distance. A supplementary power curve is then applied to sculpt a light drop-off reminiscent of classic cinema lenses.
//      // Filmic Attenuation Curve: Uses a smoothstep function to build the base vignette mask from the radial distance. A supplementary power curve is then applied to sculpt a light drop-off reminiscent of classic cinema lenses.
        float vignetteMask = 1.0 - smoothstep(uFalloffSmoothness * 0.5, VIGNETTE_FALLOFF_START_RADIUS + uFalloffSmoothness, distanceFromCenter * uVignetteIntensity);
//      float vignetteMask = 1.0 - smoothstep(uFalloffSmoothness * 0.5, VIGNETTE_FALLOFF_START_RADIUS + uFalloffSmoothness, distanceFromCenter * uVignetteIntensity);
        vignetteMask = pow(vignetteMask, VIGNETTE_FALLOFF_EXPONENT);
//      vignetteMask = pow(vignetteMask, VIGNETTE_FALLOFF_EXPONENT);

        // Color Blending: Interpolates between the custom vignette color and the underlying rendered fragment using the attenuation mask as the blend factor, avoiding the muddy results typical of naive multiplicative darkening.
//      // Color Blending: Interpolates between the custom vignette color and the underlying rendered fragment using the attenuation mask as the blend factor, avoiding the muddy results typical of naive multiplicative darkening.
        pixelColor.rgb = mix(uVignetteColor, pixelColor.rgb, vignetteMask);
//      pixelColor.rgb = mix(uVignetteColor, pixelColor.rgb, vignetteMask);

        // Dither Injection: Adds a sub-quantisation offset to break up the smoothstep banding before the result is written out for display.
//      // Dither Injection: Adds a sub-quantisation offset to break up the smoothstep banding before the result is written out for display.
        pixelColor.rgb += computeDitherOffset(vec2(destinationPixelCoord));
//      pixelColor.rgb += computeDitherOffset(vec2(destinationPixelCoord));

        imageStore(textureOutput, destinationPixelCoord, vec4(pixelColor.rgb, 1.0));
//      imageStore(textureOutput, destinationPixelCoord, vec4(pixelColor.rgb, 1.0));
    }
//  }
