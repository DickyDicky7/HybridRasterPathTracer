    #version 430
//  #version 430
    layout(local_size_x = 16, local_size_y = 16) in;
//  layout(local_size_x = 16, local_size_y = 16) in;

    layout(binding = 0, rgba32f) uniform image2D textureOutput;
//  layout(binding = 0, rgba32f) uniform image2D textureOutput;
    layout(binding = 1, rgba32f) uniform image2D textureGeometryGlobalPosition;
//  layout(binding = 1, rgba32f) uniform image2D textureGeometryGlobalPosition;
    layout(binding = 2, rgba32f) uniform image2D textureGeometryGlobalNormal;
//  layout(binding = 2, rgba32f) uniform image2D textureGeometryGlobalNormal;
    layout(binding = 3, rgba32f) uniform image2D textureGeometryAlbedo;
//  layout(binding = 3, rgba32f) uniform image2D textureGeometryAlbedo;
    layout(binding = 5, rgba32f) uniform image2D textureInput;
//  layout(binding = 5, rgba32f) uniform image2D textureInput;

    // Configuration
    // Configuration
    const int KERNEL_RADIUS = 2;
//  const int KERNEL_RADIUS = 2;
    // B3-Spline weights for À-Trous (1/16, 1/4, 3/8, 1/4, 1/16)
//  // B3-Spline weights for À-Trous (1/16, 1/4, 3/8, 1/4, 1/16)
    const float KERNEL_WEIGHTS[5] = float[](0.0625, 0.25, 0.375, 0.25, 0.0625);
//  const float KERNEL_WEIGHTS[5] = float[](0.0625, 0.25, 0.375, 0.25, 0.0625);

    const float SIGMA_COLOR = 0.2;
//  const float SIGMA_COLOR = 0.2;
    const float SIGMA_NORMAL = 0.1;
//  const float SIGMA_NORMAL = 0.1;
    const float SIGMA_POSITION = 0.5;
//  const float SIGMA_POSITION = 0.5;

    uniform int uStepSize;
//  uniform int uStepSize;
    uniform int uFinalPass;
//  uniform int uFinalPass;

    #include "tonemap.glsl"

    void main() {
//  void main() {
        ivec2 centerCoord = ivec2(gl_GlobalInvocationID.xy);
//      ivec2 centerCoord = ivec2(gl_GlobalInvocationID.xy);
        ivec2 size = imageSize(textureOutput);
//      ivec2 size = imageSize(textureOutput);

        if (centerCoord.x >= size.x || centerCoord.y >= size.y) {
//      if (centerCoord.x >= size.x || centerCoord.y >= size.y) {
            return;
//          return;
        }
//      }

        // Load Center Data
        // Load Center Data
        vec4 centerPos = imageLoad(textureGeometryGlobalPosition, centerCoord);
//      vec4 centerPos = imageLoad(textureGeometryGlobalPosition, centerCoord);
        vec4 centerNorm = imageLoad(textureGeometryGlobalNormal, centerCoord);
//      vec4 centerNorm = imageLoad(textureGeometryGlobalNormal, centerCoord);
        vec4 centerColor = imageLoad(textureInput, centerCoord);
//      vec4 centerColor = imageLoad(textureInput, centerCoord);
        vec3 centerAlbedo = imageLoad(textureGeometryAlbedo, centerCoord).rgb;
//      vec3 centerAlbedo = imageLoad(textureGeometryAlbedo, centerCoord).rgb;


        // If background (sky), just pass through
        // If background (sky), just pass through
        if (centerPos.w == 0.0) {
//      if (centerPos.w == 0.0) {
            vec3 final = centerColor.rgb;
//          vec3 final = centerColor.rgb;

            if (uFinalPass == 1) {
//          if (uFinalPass == 1) {
                final = aces(final);
//              final = aces(final);
                /*
                final = pow(final, vec3(1.0/2.2));
//              final = pow(final, vec3(1.0/2.2));
                */
            }
//          }

            imageStore(textureOutput, centerCoord, vec4(final, 1.0));
//          imageStore(textureOutput, centerCoord, vec4(final, 1.0));
            return;
//          return;
        }
//      }

        // Demodulation (Pass 1 only)
//      // Demodulation (Pass 1 only)
        if (uStepSize == 1) {
//      if (uStepSize == 1) {
            centerColor.rgb /= max(centerAlbedo, vec3(0.001));
//          centerColor.rgb /= max(centerAlbedo, vec3(0.001));
        }
//      }

        vec3 sumColor = vec3(0.0);
//      vec3 sumColor = vec3(0.0);
        float sumWeight = 0.0;
//      float sumWeight = 0.0;

        // A-Trous / Bilateral Filter Loop
        // A-Trous / Bilateral Filter Loop
        for (int y = -KERNEL_RADIUS; y <= KERNEL_RADIUS; y++) {
//      for (int y = -KERNEL_RADIUS; y <= KERNEL_RADIUS; y++) {
            for (int x = -KERNEL_RADIUS; x <= KERNEL_RADIUS; x++) {
//          for (int x = -KERNEL_RADIUS; x <= KERNEL_RADIUS; x++) {
                // Apply Step Size
                // Apply Step Size
                ivec2 offset = ivec2(x, y) * uStepSize;
//              ivec2 offset = ivec2(x, y) * uStepSize;
                ivec2 tapCoord = centerCoord + offset;
//              ivec2 tapCoord = centerCoord + offset;

                // Clamp to screen
                // Clamp to screen
                tapCoord = clamp(tapCoord, ivec2(0), size - ivec2(1));
//              tapCoord = clamp(tapCoord, ivec2(0), size - ivec2(1));

                vec4 tapPos = imageLoad(textureGeometryGlobalPosition, tapCoord);
//              vec4 tapPos = imageLoad(textureGeometryGlobalPosition, tapCoord);
                vec4 tapNorm = imageLoad(textureGeometryGlobalNormal, tapCoord);
//              vec4 tapNorm = imageLoad(textureGeometryGlobalNormal, tapCoord);
                vec4 tapColor = imageLoad(textureInput, tapCoord);
//              vec4 tapColor = imageLoad(textureInput, tapCoord);
                vec3 tapAlbedo = imageLoad(textureGeometryAlbedo, tapCoord).rgb;
//              vec3 tapAlbedo = imageLoad(textureGeometryAlbedo, tapCoord).rgb;

                // Demodulation (Pass 1 only)
//              // Demodulation (Pass 1 only)
                if (uStepSize == 1) {
//              if (uStepSize == 1) {
                    tapColor.rgb /= max(tapAlbedo, vec3(0.001));
//                  tapColor.rgb /= max(tapAlbedo, vec3(0.001));
                }
//              }

                // Calculate Weights
                // Calculate Weights

                // 1. Spatial Weight (B3-Spline Kernel)
                // 1. Spatial Weight (B3-Spline Kernel)
                float wSpatial = KERNEL_WEIGHTS[x + KERNEL_RADIUS] * KERNEL_WEIGHTS[y + KERNEL_RADIUS];
//              float wSpatial = KERNEL_WEIGHTS[x + KERNEL_RADIUS] * KERNEL_WEIGHTS[y + KERNEL_RADIUS];

                // 2. Position Weight (Edge Stopping) - Distinguish objects
                // 2. Position Weight (Edge Stopping) - Distinguish objects
                float distPos = distance(centerPos.xyz, tapPos.xyz);
//              float distPos = distance(centerPos.xyz, tapPos.xyz);
                float wPos = exp(-(distPos * distPos) / (2.0 * SIGMA_POSITION * SIGMA_POSITION));
//              float wPos = exp(-(distPos * distPos) / (2.0 * SIGMA_POSITION * SIGMA_POSITION));

                // 3. Normal Weight (Edge Stopping) - Distinguish surfaces
                // 3. Normal Weight (Edge Stopping) - Distinguish surfaces
                float distNorm = 1.0 - max(0.0, dot(centerNorm.xyz, tapNorm.xyz));
//              float distNorm = 1.0 - max(0.0, dot(centerNorm.xyz, tapNorm.xyz));
                float wNorm = exp(-(distNorm * distNorm) / (2.0 * SIGMA_NORMAL * SIGMA_NORMAL));
//              float wNorm = exp(-(distNorm * distNorm) / (2.0 * SIGMA_NORMAL * SIGMA_NORMAL));

                // 4. Color Weight (Intensity Stopping) - Preserve texture detail
                // 4. Color Weight (Intensity Stopping) - Preserve texture detail
                // Relative Color Weighting
//              // Relative Color Weighting
                float centerLuma = dot(centerColor.rgb, vec3(0.2126, 0.7152, 0.0722));
//              float centerLuma = dot(centerColor.rgb, vec3(0.2126, 0.7152, 0.0722));
                float normFactor = max(centerLuma, 0.03);
//              float normFactor = max(centerLuma, 0.03);

                float distColor = distance(centerColor.rgb, tapColor.rgb);
//              float distColor = distance(centerColor.rgb, tapColor.rgb);
                float distColorRel = distColor / normFactor;
//              float distColorRel = distColor / normFactor;

                float wColor = exp(-(distColorRel * distColorRel) / (2.0 * SIGMA_COLOR * SIGMA_COLOR));
//              float wColor = exp(-(distColorRel * distColorRel) / (2.0 * SIGMA_COLOR * SIGMA_COLOR));

                // Combined Weight
                // Combined Weight
                float weight = wSpatial * wPos * wNorm * wColor;
//              float weight = wSpatial * wPos * wNorm * wColor;

                sumColor += tapColor.rgb * weight;
//              sumColor += tapColor.rgb * weight;
                sumWeight += weight;
//              sumWeight += weight;
            }
//          }
        }
//      }

        vec3 finalColor = sumColor / sumWeight;
//      vec3 finalColor = sumColor / sumWeight;

        if (uFinalPass == 1) {
//      if (uFinalPass == 1) {
            // Remodulation (Final Pass only)
//          // Remodulation (Final Pass only)
            finalColor *= centerAlbedo;
//          finalColor *= centerAlbedo;

            // Final Tone Mapping (Moved from Shading CS)
            // Final Tone Mapping (Moved from Shading CS)
            finalColor = aces(finalColor);
//          finalColor = aces(finalColor);
            /*
            finalColor = pow(finalColor, vec3(1.0/2.2));
//          finalColor = pow(finalColor, vec3(1.0/2.2));
            */
        }
//      }

        imageStore(textureOutput, centerCoord, vec4(finalColor, 1.0));
//      imageStore(textureOutput, centerCoord, vec4(finalColor, 1.0));
    }
//  }
