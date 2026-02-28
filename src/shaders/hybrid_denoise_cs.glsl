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

    // À-Trous Filter Configuration: Controls kernel radius and spatial/geometric edge-stopping sensitivity
    // À-Trous Filter Configuration: Controls kernel radius and spatial/geometric edge-stopping sensitivity
    const int KERNEL_RADIUS = 2;
//  const int KERNEL_RADIUS = 2;
    // B3-Spline weights for À-Trous (1/16, 1/4, 3/8, 1/4, 1/16)
//  // B3-Spline weights for À-Trous (1/16, 1/4, 3/8, 1/4, 1/16)
    const float KERNEL_WEIGHTS[5] = float[](0.0625, 0.25, 0.375, 0.25, 0.0625);
//  const float KERNEL_WEIGHTS[5] = float[](0.0625, 0.25, 0.375, 0.25, 0.0625);

    // SIGMA parameters: Tuning thresholds for edge-stopping functions (higher = more blurring/bleeding)
    // SIGMA parameters: Tuning thresholds for edge-stopping functions (higher = more blurring/bleeding)
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

        // Geometry Fetch: Retrieve world-space guidance data from G-Buffer for bilateral weighting
        // Geometry Fetch: Retrieve world-space guidance data from G-Buffer for bilateral weighting
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

            /*
            if (uFinalPass == 1) {
//          if (uFinalPass == 1) {
                final = aces(final);
//              final = aces(final);
                final = pow(final, vec3(1.0/2.2));
//              final = pow(final, vec3(1.0/2.2));
            }
//          }
            */

            imageStore(textureOutput, centerCoord, vec4(final, 1.0));
//          imageStore(textureOutput, centerCoord, vec4(final, 1.0));
            return;
//          return;
        }
//      }

        // Albedo Demodulation: Factor out surface reflectance to filter core illumination (low variance)
//      // Albedo Demodulation: Factor out surface reflectance to filter core illumination (low variance)
        if (uStepSize == 1) {
//      if (uStepSize == 1) {
            centerColor.rgb /= max(centerAlbedo, vec3(0.001));
//          centerColor.rgb /= max(centerAlbedo, vec3(0.001));
        }
//      }

        // Outlier Clamping: Suppress HDR 'fireflies' produced by extreme path contributions to stabilize the filter
//      // Outlier Clamping: Suppress HDR 'fireflies' produced by extreme path contributions to stabilize the filter
        // Clamp extremely high values to prevent fireflies from exploding the filter.
//      // Clamp extremely high values to prevent fireflies from exploding the filter.
        float maxRadiance = 20.0;
//      float maxRadiance = 20.0;
        centerColor.rgb = min(centerColor.rgb, vec3(maxRadiance));
//      centerColor.rgb = min(centerColor.rgb, vec3(maxRadiance));

//      // Calculate center luminance and tonemap to reduce firefly weight (Karis Average)
        // Calculate center luminance and tonemap to reduce firefly weight (Karis Average)
//      float centerLuma = dot(centerColor.rgb, vec3(0.2126, 0.7152, 0.0722));
        float centerLuma = dot(centerColor.rgb, vec3(0.2126, 0.7152, 0.0722));
//      vec3 centerColorTM = centerColor.rgb / (1.0 + centerLuma);
        vec3 centerColorTM = centerColor.rgb / (1.0 + centerLuma);

        // Pre-calculate center weight to avoid branch in the inner loop
//      // Pre-calculate center weight to avoid branch in the inner loop
        float centerWeight = KERNEL_WEIGHTS[2] * KERNEL_WEIGHTS[2];
//      float centerWeight = KERNEL_WEIGHTS[2] * KERNEL_WEIGHTS[2];
        vec3 sumColor = centerColorTM * centerWeight;
//      vec3 sumColor = centerColorTM * centerWeight;
        float sumWeight = centerWeight;
//      float sumWeight = centerWeight;

        // Joint Bilateral Filter Loop: Hierarchical gather using À-Trous step size for large-scale noise suppression
//      // Joint Bilateral Filter Loop: Hierarchical gather using À-Trous step size for large-scale noise suppression
        for (int y = -KERNEL_RADIUS; y <= KERNEL_RADIUS; y++) {
//      for (int y = -KERNEL_RADIUS; y <= KERNEL_RADIUS; y++) {
            for (int x = -KERNEL_RADIUS; x <= KERNEL_RADIUS; x++) {
//          for (int x = -KERNEL_RADIUS; x <= KERNEL_RADIUS; x++) {
                // Skip center pixel as it is already accumulated
//              // Skip center pixel as it is already accumulated
                if (x == 0 && y == 0) continue;
//              if (x == 0 && y == 0) continue;

                // Apply Step Size
//              // Apply Step Size
                ivec2 offset = ivec2(x, y) * uStepSize;
//              ivec2 offset = ivec2(x, y) * uStepSize;
                ivec2 tapCoord = centerCoord + offset;
//              ivec2 tapCoord = centerCoord + offset;

                // Clamp to screen
//              // Clamp to screen
                tapCoord = clamp(tapCoord, ivec2(0), size - ivec2(1));
//              tapCoord = clamp(tapCoord, ivec2(0), size - ivec2(1));

                vec4 tapPos = imageLoad(textureGeometryGlobalPosition, tapCoord);
//              vec4 tapPos = imageLoad(textureGeometryGlobalPosition, tapCoord);

                // Skip tap if it hits the background to prevent sky colors bleeding into geometry
//              // Skip tap if it hits the background to prevent sky colors bleeding into geometry
                if (tapPos.w == 0.0) continue;
//              if (tapPos.w == 0.0) continue;

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

                // Firefly Clamping for Tap
//              // Firefly Clamping for Tap
                tapColor.rgb = min(tapColor.rgb, vec3(maxRadiance));
//              tapColor.rgb = min(tapColor.rgb, vec3(maxRadiance));

//              float tapLuma = dot(tapColor.rgb, vec3(0.2126, 0.7152, 0.0722));
                float tapLuma = dot(tapColor.rgb, vec3(0.2126, 0.7152, 0.0722));

                // Geometric Edge-Stopping Functions: Weight the tap based on similarity to center geometry
//              // Geometric Edge-Stopping Functions: Weight the tap based on similarity to center geometry

                // 1. Spatial Weight (B3-Spline Kernel)
//              // 1. Spatial Weight (B3-Spline Kernel)
                float wSpatial = KERNEL_WEIGHTS[x + KERNEL_RADIUS] * KERNEL_WEIGHTS[y + KERNEL_RADIUS];
//              float wSpatial = KERNEL_WEIGHTS[x + KERNEL_RADIUS] * KERNEL_WEIGHTS[y + KERNEL_RADIUS];

                // 2. Position Weight (Plane Distance)
//              // 2. Position Weight (Plane Distance)
                // Project the neighbor onto the tangent plane of the center pixel.
//              // Project the neighbor onto the tangent plane of the center pixel.
                // This preserves oblique surfaces better than Euclidean distance.
//              // This preserves oblique surfaces better than Euclidean distance.
                vec3 diffPos = tapPos.xyz - centerPos.xyz;
//              vec3 diffPos = tapPos.xyz - centerPos.xyz;
                float distPos = abs(dot(centerNorm.xyz, diffPos));
//              float distPos = abs(dot(centerNorm.xyz, diffPos));
                // Scale SIGMA_POSITION by uStepSize to prevent curvature over-rejection at large steps
//              // Scale SIGMA_POSITION by uStepSize to prevent curvature over-rejection at large steps
                float currentSigmaPos = SIGMA_POSITION * float(uStepSize);
//              float currentSigmaPos = SIGMA_POSITION * float(uStepSize);
                float wPos = exp(-(distPos * distPos) / (2.0 * currentSigmaPos * currentSigmaPos));
//              float wPos = exp(-(distPos * distPos) / (2.0 * currentSigmaPos * currentSigmaPos));

                // 3. Normal Weight (Edge Stopping) - Distinguish surfaces
//              // 3. Normal Weight (Edge Stopping) - Distinguish surfaces
                // Optimized normal edge stopping using max() instead of clamp() to strongly reject backward normals
//              // Optimized normal edge stopping using max() instead of clamp() to strongly reject backward normals
                float distNorm = max(0.0, 1.0 - dot(centerNorm.xyz, tapNorm.xyz));
//              float distNorm = max(0.0, 1.0 - dot(centerNorm.xyz, tapNorm.xyz));
                float wNorm = exp(-(distNorm * distNorm) / (2.0 * SIGMA_NORMAL * SIGMA_NORMAL));
//              float wNorm = exp(-(distNorm * distNorm) / (2.0 * SIGMA_NORMAL * SIGMA_NORMAL));

                // 4. Color Weight (Intensity Stopping) - Preserve texture detail
                // 4. Color Weight (Intensity Stopping) - Preserve texture detail
                // Relative Color Weighting
//              // Relative Color Weighting
                // Made luminance scaling symmetric by comparing the max luminance of both center and tap
//              // Made luminance scaling symmetric by comparing the max luminance of both center and tap
                float normFactor = max(max(centerLuma, tapLuma), 0.03);
//              float normFactor = max(max(centerLuma, tapLuma), 0.03);

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

//              // Tonemap tap color for accumulation
                // Tonemap tap color for accumulation
//              vec3 tapColorTM = tapColor.rgb / (1.0 + tapLuma);
                vec3 tapColorTM = tapColor.rgb / (1.0 + tapLuma);

                sumColor += tapColorTM * weight;
//              sumColor += tapColorTM * weight;
                sumWeight += weight;
//              sumWeight += weight;
            }
//          }
        }
//      }

        vec3 finalColor = sumColor / sumWeight;
//      vec3 finalColor = sumColor / sumWeight;

//      // Inverse reversible tonemapping to restore energy
        // Inverse reversible tonemapping to restore energy
//      float finalLuma = dot(finalColor, vec3(0.2126, 0.7152, 0.0722));
        float finalLuma = dot(finalColor, vec3(0.2126, 0.7152, 0.0722));
//      finalColor = finalColor / max(1.0 - finalLuma, 0.0001);
        finalColor = finalColor / max(1.0 - finalLuma, 0.0001);

        if (uFinalPass == 1) {
//      if (uFinalPass == 1) {
            // Albedo Remodulation: Re-apply surface textures to the filtered irradiance signal
//          // Albedo Remodulation: Re-apply surface textures to the filtered irradiance signal
            // Matched the demodulation clamping curve exactly to fix the energy conservation bug
//          // Matched the demodulation clamping curve exactly to fix the energy conservation bug
            finalColor *= max(centerAlbedo, vec3(0.001));
//          finalColor *= max(centerAlbedo, vec3(0.001));

            /*
            // LDR Conversion: Apply ACES Tone Mapper and sRGB Gamma correction
//          // LDR Conversion: Apply ACES Tone Mapper and sRGB Gamma correction
            finalColor = aces(finalColor);
//          finalColor = aces(finalColor);
            finalColor = pow(finalColor, vec3(1.0/2.2));
//          finalColor = pow(finalColor, vec3(1.0/2.2));
            */
        }
//      }

        imageStore(textureOutput, centerCoord, vec4(finalColor, 1.0));
//      imageStore(textureOutput, centerCoord, vec4(finalColor, 1.0));
    }
//  }
