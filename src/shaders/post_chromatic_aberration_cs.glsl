    #version 430
//  #version 430
    layout(local_size_x = 16, local_size_y = 16) in;
//  layout(local_size_x = 16, local_size_y = 16) in;

    layout(binding = 0, rgba32f) uniform image2D textureOutput;
//  layout(binding = 0, rgba32f) uniform image2D textureOutput;
    layout(binding = 1, rgba32f) uniform image2D textureInput;
//  layout(binding = 1, rgba32f) uniform image2D textureInput;

    // Aberration Strength: Controls the fractional displacement applied to the red and blue color channels across the lens. This simulates the optical dispersion of physical camera lenses, where light of differing wavelengths refracts at slightly different angles.
//  // Aberration Strength: Controls the fractional displacement applied to the red and blue color channels across the lens. This simulates the optical dispersion of physical camera lenses, where light of differing wavelengths refracts at slightly different angles.
    uniform float uAberrationStrength = 0.01;
//  uniform float uAberrationStrength = 0.01;

    // Constants
//  // Constants
    const float RADIAL_DISTORTION_SCALE = 4.0;
//  const float RADIAL_DISTORTION_SCALE = 4.0;
    const int SPECTRAL_SAMPLE_COUNT = 8;
//  const int SPECTRAL_SAMPLE_COUNT = 8;

    // Bilinear Texel Gather: The input is bound as a raw storage image rather than a hardware-filtered sampler, so fractional sample positions must be resolved manually by fetching the four surrounding texels and blending them along both axes. This upgrades the previous nearest-neighbour integer reads to sub-pixel-accurate fetches, eliminating the stair-step aliasing that would otherwise crawl along the chromatic fringes.
//  // Bilinear Texel Gather: The input is bound as a raw storage image rather than a hardware-filtered sampler, so fractional sample positions must be resolved manually by fetching the four surrounding texels and blending them along both axes. This upgrades the previous nearest-neighbour integer reads to sub-pixel-accurate fetches, eliminating the stair-step aliasing that would otherwise crawl along the chromatic fringes.
    vec3 sampleInputBilinear(vec2 samplePosition, ivec2 imageSizeInTexels) {
//  vec3 sampleInputBilinear(vec2 samplePosition, ivec2 imageSizeInTexels) {
        vec2 texelFraction = fract(samplePosition);
//      vec2 texelFraction = fract(samplePosition);
        ivec2 baseTexel = ivec2(floor(samplePosition));
//      ivec2 baseTexel = ivec2(floor(samplePosition));

        ivec2 texelTopLeft     = clamp(baseTexel,                ivec2(0), imageSizeInTexels - ivec2(1));
//      ivec2 texelTopLeft     = clamp(baseTexel,                ivec2(0), imageSizeInTexels - ivec2(1));
        ivec2 texelTopRight    = clamp(baseTexel + ivec2(1, 0),  ivec2(0), imageSizeInTexels - ivec2(1));
//      ivec2 texelTopRight    = clamp(baseTexel + ivec2(1, 0),  ivec2(0), imageSizeInTexels - ivec2(1));
        ivec2 texelBottomLeft  = clamp(baseTexel + ivec2(0, 1),  ivec2(0), imageSizeInTexels - ivec2(1));
//      ivec2 texelBottomLeft  = clamp(baseTexel + ivec2(0, 1),  ivec2(0), imageSizeInTexels - ivec2(1));
        ivec2 texelBottomRight = clamp(baseTexel + ivec2(1, 1),  ivec2(0), imageSizeInTexels - ivec2(1));
//      ivec2 texelBottomRight = clamp(baseTexel + ivec2(1, 1),  ivec2(0), imageSizeInTexels - ivec2(1));

        vec3 topRowColor    = mix(imageLoad(textureInput, texelTopLeft).rgb,    imageLoad(textureInput, texelTopRight).rgb,    texelFraction.x);
//      vec3 topRowColor    = mix(imageLoad(textureInput, texelTopLeft).rgb,    imageLoad(textureInput, texelTopRight).rgb,    texelFraction.x);
        vec3 bottomRowColor = mix(imageLoad(textureInput, texelBottomLeft).rgb, imageLoad(textureInput, texelBottomRight).rgb, texelFraction.x);
//      vec3 bottomRowColor = mix(imageLoad(textureInput, texelBottomLeft).rgb, imageLoad(textureInput, texelBottomRight).rgb, texelFraction.x);
        return mix(topRowColor, bottomRowColor, texelFraction.y);
//      return mix(topRowColor, bottomRowColor, texelFraction.y);
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

        vec2 centerUv = vec2(destinationPixelCoord) / vec2(outputImageSize);
//      vec2 centerUv = vec2(destinationPixelCoord) / vec2(outputImageSize);
        vec2 directionFromCenter = centerUv - 0.5;
//      vec2 directionFromCenter = centerUv - 0.5;

        // Radial Lens Distortion: Applies a non-linear cubic radial distortion to skew the sampling coordinates. Multiplying the direction vector by its own squared magnitude makes the aberration scale cubically towards the outer fringes of the frame, leaving the centre almost untouched.
//      // Radial Lens Distortion: Applies a non-linear cubic radial distortion to skew the sampling coordinates. Multiplying the direction vector by its own squared magnitude makes the aberration scale cubically towards the outer fringes of the frame, leaving the centre almost untouched.
        vec2 radialAberrationOffset = directionFromCenter * dot(directionFromCenter, directionFromCenter) * uAberrationStrength * RADIAL_DISTORTION_SCALE;
//      vec2 radialAberrationOffset = directionFromCenter * dot(directionFromCenter, directionFromCenter) * uAberrationStrength * RADIAL_DISTORTION_SCALE;

        // Spectral Multi-Tap Integration: Replaces the basic per-channel sample shift with a continuous fractional integration loop across the spectrum. Blending adjacent shifted samples generates an organically smoothed color fringe, preventing the hard-edged stepping artifacts of a single discrete shift.
//      // Spectral Multi-Tap Integration: Replaces the basic per-channel sample shift with a continuous fractional integration loop across the spectrum. Blending adjacent shifted samples generates an organically smoothed color fringe, preventing the hard-edged stepping artifacts of a single discrete shift.
        vec3 weightedColorSum = vec3(0.0);
//      vec3 weightedColorSum = vec3(0.0);
        vec3 channelWeightSum = vec3(0.0);
//      vec3 channelWeightSum = vec3(0.0);

        for (int tapIndex = 0; tapIndex < SPECTRAL_SAMPLE_COUNT; tapIndex++) {
//      for (int tapIndex = 0; tapIndex < SPECTRAL_SAMPLE_COUNT; tapIndex++) {
            float normalizedTapPosition = float(tapIndex) / float(SPECTRAL_SAMPLE_COUNT - 1);
//          float normalizedTapPosition = float(tapIndex) / float(SPECTRAL_SAMPLE_COUNT - 1);
            float spectralShift = (normalizedTapPosition - 0.5) * 2.0;
//          float spectralShift = (normalizedTapPosition - 0.5) * 2.0;

            vec2 samplePixelCoord = (centerUv + radialAberrationOffset * spectralShift) * vec2(outputImageSize);
//          vec2 samplePixelCoord = (centerUv + radialAberrationOffset * spectralShift) * vec2(outputImageSize);
            vec3 sampledColor = sampleInputBilinear(samplePixelCoord, outputImageSize);
//          vec3 sampledColor = sampleInputBilinear(samplePixelCoord, outputImageSize);

            // Triangular spectral weights: red peaks at the +1 shift, green at the centre, blue at the -1 shift.
//          // Triangular spectral weights: red peaks at the +1 shift, green at the centre, blue at the -1 shift.
            vec3 channelWeights = vec3(max(0.0, 1.0 - abs(spectralShift - 1.0)), max(0.0, 1.0 - abs(spectralShift)), max(0.0, 1.0 - abs(spectralShift + 1.0)));
//          vec3 channelWeights = vec3(max(0.0, 1.0 - abs(spectralShift - 1.0)), max(0.0, 1.0 - abs(spectralShift)), max(0.0, 1.0 - abs(spectralShift + 1.0)));

            weightedColorSum += sampledColor * channelWeights;
//          weightedColorSum += sampledColor * channelWeights;
            channelWeightSum += channelWeights;
//          channelWeightSum += channelWeights;
        }
//      }

        imageStore(textureOutput, destinationPixelCoord, vec4(weightedColorSum / channelWeightSum, 1.0));
//      imageStore(textureOutput, destinationPixelCoord, vec4(weightedColorSum / channelWeightSum, 1.0));
    }
//  }
