    // rgba16f matches the f2 post-processing ping-pong pair (LDR data - no visible precision loss)
//  // rgba16f matches the f2 post-processing ping-pong pair (LDR data - no visible precision loss)
    [format("rgba16f")] WTexture2D<float4> textureOutput;
//  [format("rgba16f")] WTexture2D<float4> textureOutput;
    Texture2D<float4> textureInput;
//  Texture2D<float4> textureInput;

    // Aberration Strength: Controls the fractional displacement applied to the red and blue color channels across the lens. This simulates the optical dispersion of physical camera lenses, where light of differing wavelengths refracts at slightly different angles.
//  // Aberration Strength: Controls the fractional displacement applied to the red and blue color channels across the lens. This simulates the optical dispersion of physical camera lenses, where light of differing wavelengths refracts at slightly different angles.
    // Slang ignores initialisers on uniforms, so the renderer uploads the default strength (0.01) when it builds this program.
//  // Slang ignores initialisers on uniforms, so the renderer uploads the default strength (0.01) when it builds this program.
    uniform float uAberrationStrength;
//  uniform float uAberrationStrength;

    // Constants
//  // Constants
    static const float RADIAL_DISTORTION_SCALE = 4.0;
//  static const float RADIAL_DISTORTION_SCALE = 4.0;
    static const int SPECTRAL_SAMPLE_COUNT = 8;
//  static const int SPECTRAL_SAMPLE_COUNT = 8;

    // Bilinear Texel Gather: The input is bound as a texture without a sampler, read through integer texel loads, so fractional sample positions must be resolved manually by fetching the four surrounding texels and blending them along both axes. This upgrades the previous nearest-neighbour integer reads to sub-pixel-accurate fetches, eliminating the stair-step aliasing that would otherwise crawl along the chromatic fringes.
//  // Bilinear Texel Gather: The input is bound as a texture without a sampler, read through integer texel loads, so fractional sample positions must be resolved manually by fetching the four surrounding texels and blending them along both axes. This upgrades the previous nearest-neighbour integer reads to sub-pixel-accurate fetches, eliminating the stair-step aliasing that would otherwise crawl along the chromatic fringes.
    float3 sampleInputBilinear(float2 samplePosition, int2 imageSizeInTexels) {
//  float3 sampleInputBilinear(float2 samplePosition, int2 imageSizeInTexels) {
        float2 continuousTexel = samplePosition - 0.5;
//      float2 continuousTexel = samplePosition - 0.5;
        float2 texelFraction = frac(continuousTexel);
//      float2 texelFraction = frac(continuousTexel);
        int2 baseTexel = int2(floor(continuousTexel));
//      int2 baseTexel = int2(floor(continuousTexel));

        int2 texelTopLeft     = clamp(baseTexel,                int2(0), imageSizeInTexels - int2(1));
//      int2 texelTopLeft     = clamp(baseTexel,                int2(0), imageSizeInTexels - int2(1));
        int2 texelTopRight    = clamp(baseTexel + int2(1, 0),  int2(0), imageSizeInTexels - int2(1));
//      int2 texelTopRight    = clamp(baseTexel + int2(1, 0),  int2(0), imageSizeInTexels - int2(1));
        int2 texelBottomLeft  = clamp(baseTexel + int2(0, 1),  int2(0), imageSizeInTexels - int2(1));
//      int2 texelBottomLeft  = clamp(baseTexel + int2(0, 1),  int2(0), imageSizeInTexels - int2(1));
        int2 texelBottomRight = clamp(baseTexel + int2(1, 1),  int2(0), imageSizeInTexels - int2(1));
//      int2 texelBottomRight = clamp(baseTexel + int2(1, 1),  int2(0), imageSizeInTexels - int2(1));

        float3 topRowColor    = lerp(textureInput.Load(int3(texelTopLeft, 0)).rgb,    textureInput.Load(int3(texelTopRight, 0)).rgb,    texelFraction.x);
//      float3 topRowColor    = lerp(textureInput.Load(int3(texelTopLeft, 0)).rgb,    textureInput.Load(int3(texelTopRight, 0)).rgb,    texelFraction.x);
        float3 bottomRowColor = lerp(textureInput.Load(int3(texelBottomLeft, 0)).rgb, textureInput.Load(int3(texelBottomRight, 0)).rgb, texelFraction.x);
//      float3 bottomRowColor = lerp(textureInput.Load(int3(texelBottomLeft, 0)).rgb, textureInput.Load(int3(texelBottomRight, 0)).rgb, texelFraction.x);
        return lerp(topRowColor, bottomRowColor, texelFraction.y);
//      return lerp(topRowColor, bottomRowColor, texelFraction.y);
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

        float2 centerUv = (float2(destinationPixelCoord) + 0.5) / float2(outputImageSize);
//      float2 centerUv = (float2(destinationPixelCoord) + 0.5) / float2(outputImageSize);
        float2 directionFromCenter = centerUv - 0.5;
//      float2 directionFromCenter = centerUv - 0.5;

        // Radial Lens Distortion: Applies a non-linear cubic radial distortion to skew the sampling coordinates. Multiplying the direction vector by its own squared magnitude makes the aberration scale cubically towards the outer fringes of the frame, leaving the centre almost untouched.
//      // Radial Lens Distortion: Applies a non-linear cubic radial distortion to skew the sampling coordinates. Multiplying the direction vector by its own squared magnitude makes the aberration scale cubically towards the outer fringes of the frame, leaving the centre almost untouched.
        float2 radialAberrationOffset = directionFromCenter * dot(directionFromCenter, directionFromCenter) * uAberrationStrength * RADIAL_DISTORTION_SCALE;
//      float2 radialAberrationOffset = directionFromCenter * dot(directionFromCenter, directionFromCenter) * uAberrationStrength * RADIAL_DISTORTION_SCALE;

        // Spectral Multi-Tap Integration: Replaces the basic per-channel sample shift with a continuous fractional integration loop across the spectrum. Blending adjacent shifted samples generates an organically smoothed color fringe, preventing the hard-edged stepping artifacts of a single discrete shift.
//      // Spectral Multi-Tap Integration: Replaces the basic per-channel sample shift with a continuous fractional integration loop across the spectrum. Blending adjacent shifted samples generates an organically smoothed color fringe, preventing the hard-edged stepping artifacts of a single discrete shift.
        float3 weightedColorSum = float3(0.0);
//      float3 weightedColorSum = float3(0.0);
        float3 channelWeightSum = float3(0.0);
//      float3 channelWeightSum = float3(0.0);

        for (int tapIndex = 0; tapIndex < SPECTRAL_SAMPLE_COUNT; tapIndex++) {
//      for (int tapIndex = 0; tapIndex < SPECTRAL_SAMPLE_COUNT; tapIndex++) {
            float normalizedTapPosition = float(tapIndex) / float(SPECTRAL_SAMPLE_COUNT - 1);
//          float normalizedTapPosition = float(tapIndex) / float(SPECTRAL_SAMPLE_COUNT - 1);
            float spectralShift = (normalizedTapPosition - 0.5) * 2.0;
//          float spectralShift = (normalizedTapPosition - 0.5) * 2.0;

            float2 samplePixelCoord = (centerUv + radialAberrationOffset * spectralShift) * float2(outputImageSize);
//          float2 samplePixelCoord = (centerUv + radialAberrationOffset * spectralShift) * float2(outputImageSize);
            float3 sampledColor = sampleInputBilinear(samplePixelCoord, outputImageSize);
//          float3 sampledColor = sampleInputBilinear(samplePixelCoord, outputImageSize);

            // Triangular spectral weights: red peaks at the +1 shift, green at the centre, blue at the -1 shift.
//          // Triangular spectral weights: red peaks at the +1 shift, green at the centre, blue at the -1 shift.
            float3 channelWeights = float3(max(0.0, 1.0 - abs(spectralShift - 1.0)), max(0.0, 1.0 - abs(spectralShift)), max(0.0, 1.0 - abs(spectralShift + 1.0)));
//          float3 channelWeights = float3(max(0.0, 1.0 - abs(spectralShift - 1.0)), max(0.0, 1.0 - abs(spectralShift)), max(0.0, 1.0 - abs(spectralShift + 1.0)));

            weightedColorSum += sampledColor * channelWeights;
//          weightedColorSum += sampledColor * channelWeights;
            channelWeightSum += channelWeights;
//          channelWeightSum += channelWeights;
        }
//      }

        textureOutput.Store(destinationPixelCoord, float4(weightedColorSum / channelWeightSum, 1.0));
//      textureOutput.Store(destinationPixelCoord, float4(weightedColorSum / channelWeightSum, 1.0));
    }
//  }
