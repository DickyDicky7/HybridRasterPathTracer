    // rgba16f matches the f2 post-processing ping-pong pair (LDR data - no visible precision loss)
//  // rgba16f matches the f2 post-processing ping-pong pair (LDR data - no visible precision loss)
    [format("rgba16f")] WTexture2D<float4> textureOutput;
//  [format("rgba16f")] WTexture2D<float4> textureOutput;
    Texture2D<float4> textureInput;
//  Texture2D<float4> textureInput;

    // Vignette Parameters: Define the core framing profile. The intensity scalar drives the darkening gradient, the smoothness governs the falloff transition out from the centre, and the color override allows for cinematic tinting of the attenuated border regions.
//  // Vignette Parameters: Define the core framing profile. The intensity scalar drives the darkening gradient, the smoothness governs the falloff transition out from the centre, and the color override allows for cinematic tinting of the attenuated border regions.
    // Slang ignores initialisers on uniforms, so the renderer uploads the defaults (intensity 1.0, smoothness 0.5, black tint) when it builds this program.
//  // Slang ignores initialisers on uniforms, so the renderer uploads the defaults (intensity 1.0, smoothness 0.5, black tint) when it builds this program.
    uniform float uVignetteIntensity;
//  uniform float uVignetteIntensity;
    uniform float uFalloffSmoothness;
//  uniform float uFalloffSmoothness;
    uniform float3 uVignetteColor;
//  uniform float3 uVignetteColor;

    // Constants
//  // Constants
    static const float VIGNETTE_FALLOFF_START_RADIUS = 0.8;
//  static const float VIGNETTE_FALLOFF_START_RADIUS = 0.8;
    static const float VIGNETTE_FALLOFF_EXPONENT = 1.2;
//  static const float VIGNETTE_FALLOFF_EXPONENT = 1.2;

    // Interleaved Gradient Dither: A deterministic high-frequency screen-space hash evaluated per pixel, returning a signed offset smaller than a single 8-bit display code value. Injecting this into the attenuated result perturbs the otherwise mathematically perfect falloff below the quantisation threshold, dissolving the concentric banding rings that low-bit-depth output would otherwise expose across the gentle gradient.
//  // Interleaved Gradient Dither: A deterministic high-frequency screen-space hash evaluated per pixel, returning a signed offset smaller than a single 8-bit display code value. Injecting this into the attenuated result perturbs the otherwise mathematically perfect falloff below the quantisation threshold, dissolving the concentric banding rings that low-bit-depth output would otherwise expose across the gentle gradient.
    float computeDitherOffset(float2 pixelCoord) {
//  float computeDitherOffset(float2 pixelCoord) {
        float interleavedGradientNoise = frac(52.9829189 * frac(dot(pixelCoord, float2(0.06711056, 0.00583715))));
//      float interleavedGradientNoise = frac(52.9829189 * frac(dot(pixelCoord, float2(0.06711056, 0.00583715))));
        return (interleavedGradientNoise - 0.5) / 255.0;
//      return (interleavedGradientNoise - 0.5) / 255.0;
    }
//  }

    [shader("compute")]
//  [shader("compute")]
    [numthreads(16, 16, 1)]
//  [numthreads(16, 16, 1)]
    void computeMain(uint3 globalInvocationID : SV_DispatchThreadID, uint localInvocationIndex : SV_GroupIndex) {
//  void computeMain(uint3 globalInvocationID : SV_DispatchThreadID, uint localInvocationIndex : SV_GroupIndex) {
        int2 destinationPixelCoord = int2(globalInvocationID.xy);
//      int2 destinationPixelCoord = int2(globalInvocationID.xy);
        uint2 outputImageSizeUnsigned;
//      uint2 outputImageSizeUnsigned;
        textureOutput.GetDimensions(outputImageSizeUnsigned.x, outputImageSizeUnsigned.y);
//      textureOutput.GetDimensions(outputImageSizeUnsigned.x, outputImageSizeUnsigned.y);
        int2 outputImageSize = int2(outputImageSizeUnsigned);
//      int2 outputImageSize = int2(outputImageSizeUnsigned);

        if (destinationPixelCoord.x >= outputImageSize.x || destinationPixelCoord.y >= outputImageSize.y) {
//      if (destinationPixelCoord.x >= outputImageSize.x || destinationPixelCoord.y >= outputImageSize.y) {
            return;
//          return;
        }
//      }

        float2 pixelUv = float2(destinationPixelCoord) / float2(outputImageSize);
//      float2 pixelUv = float2(destinationPixelCoord) / float2(outputImageSize);
        float4 pixelColor = textureInput.Load(int3(destinationPixelCoord, 0));
//      float4 pixelColor = textureInput.Load(int3(destinationPixelCoord, 0));

        // Aspect Ratio Correction: Scales the horizontal component of the direction vector by the viewport aspect ratio. This keeps the vignette gradient perfectly circular regardless of the window or screen dimensions.
//      // Aspect Ratio Correction: Scales the horizontal component of the direction vector by the viewport aspect ratio. This keeps the vignette gradient perfectly circular regardless of the window or screen dimensions.
        float2 directionFromCenter = pixelUv - 0.5;
//      float2 directionFromCenter = pixelUv - 0.5;
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
        pixelColor.rgb = lerp(uVignetteColor, pixelColor.rgb, vignetteMask);
//      pixelColor.rgb = lerp(uVignetteColor, pixelColor.rgb, vignetteMask);

        // Dither Injection: Adds a sub-quantisation offset to break up the smoothstep banding before the result is written out for display.
//      // Dither Injection: Adds a sub-quantisation offset to break up the smoothstep banding before the result is written out for display.
        pixelColor.rgb += computeDitherOffset(float2(destinationPixelCoord));
//      pixelColor.rgb += computeDitherOffset(float2(destinationPixelCoord));

        textureOutput.Store(destinationPixelCoord, float4(pixelColor.rgb, 1.0));
//      textureOutput.Store(destinationPixelCoord, float4(pixelColor.rgb, 1.0));
    }
//  }
