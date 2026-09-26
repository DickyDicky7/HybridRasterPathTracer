    // ============================================================================
//  // ============================================================================
    // Shared volumetric helpers.
//  // Shared volumetric helpers.
    //
//  //
    // Phase functions and the noise-volume lookups used by both the sky cloud layer
//  // Phase functions and the noise-volume lookups used by both the sky cloud layer
    // (sky_clouds.glsl) and the in-scene participating media (participating_media.glsl).
//  // (sky_clouds.glsl) and the in-scene participating media (participating_media.glsl).
    //
//  //
    // Requires the includer to have already defined: PI, uCloudNoiseTexture (sampler3D holding the
//  // Requires the includer to have already defined: PI, uCloudNoiseTexture (sampler3D holding the
    // tileable noise written by cloud_noise_cs.glsl: R = Perlin-Worley, GBA = Worley FBM octaves).
//  // tileable noise written by cloud_noise_cs.glsl: R = Perlin-Worley, GBA = Worley FBM octaves).
    // ============================================================================
//  // ============================================================================
    const vec3 CLOUD_NOISE_FBM_WEIGHTS = vec3(0.625, 0.25, 0.125);
//  const vec3 CLOUD_NOISE_FBM_WEIGHTS = vec3(0.625, 0.25, 0.125);
    // The dilated Perlin-Worley below only spans roughly [0.74, 0.92] over the generated volume (its
//  // The dilated Perlin-Worley below only spans roughly [0.74, 0.92] over the generated volume (its
    // 2nd to 98th percentile), which would leave every threshold against it all-or-nothing. Stretching
//  // 2nd to 98th percentile), which would leave every threshold against it all-or-nothing. Stretching
    // that range onto [0, 1] lets coverage and shape masks carve individual billows.
//  // that range onto [0, 1] lets coverage and shape masks carve individual billows.
    const vec2 CLOUD_BASE_SHAPE_RANGE = vec2(0.74, 0.92);
//  const vec2 CLOUD_BASE_SHAPE_RANGE = vec2(0.74, 0.92);

    float remapValue(float value, float oldMin, float oldMax, float newMin, float newMax) {
//  float remapValue(float value, float oldMin, float oldMax, float newMin, float newMax) {
        return newMin + (value - oldMin) / (oldMax - oldMin) * (newMax - newMin);
//      return newMin + (value - oldMin) / (oldMax - oldMin) * (newMax - newMin);
    }
//  }

    // Henyey-Greenstein phase function, normalized over the sphere. cosTheta is measured between the
//  // Henyey-Greenstein phase function, normalized over the sphere. cosTheta is measured between the
    // direction light was travelling and the direction it scatters into, so g > 0 is forward scattering.
//  // direction light was travelling and the direction it scatters into, so g > 0 is forward scattering.
    float henyeyGreenstein(float cosTheta, float anisotropy) {
//  float henyeyGreenstein(float cosTheta, float anisotropy) {
        float anisotropySquared = anisotropy * anisotropy;
//      float anisotropySquared = anisotropy * anisotropy;
        float denominator = 1.0 + anisotropySquared - 2.0 * anisotropy * cosTheta;
//      float denominator = 1.0 + anisotropySquared - 2.0 * anisotropy * cosTheta;
        return (1.0 - anisotropySquared) / (4.0 * PI * denominator * sqrt(max(denominator, 1.0e-6)));
//      return (1.0 - anisotropySquared) / (4.0 * PI * denominator * sqrt(max(denominator, 1.0e-6)));
    }
//  }

    // Draws a scattered direction distributed exactly as henyeyGreenstein(dot(travelDirection, result), g),
//  // Draws a scattered direction distributed exactly as henyeyGreenstein(dot(travelDirection, result), g),
    // so the phase-over-pdf ratio is 1 and the path throughput only picks up the medium albedo.
//  // so the phase-over-pdf ratio is 1 and the path throughput only picks up the medium albedo.
    vec3 sampleHenyeyGreenstein(vec3 travelDirection, float anisotropy, float randomA, float randomB) {
//  vec3 sampleHenyeyGreenstein(vec3 travelDirection, float anisotropy, float randomA, float randomB) {
        float cosTheta;
//      float cosTheta;
        if (abs(anisotropy) < 1.0e-3) {
//      if (abs(anisotropy) < 1.0e-3) {
            cosTheta = 1.0 - 2.0 * randomA;
//          cosTheta = 1.0 - 2.0 * randomA;
        } else {
//      } else {
            float squareTerm = (1.0 - anisotropy * anisotropy) / (1.0 - anisotropy + 2.0 * anisotropy * randomA);
//          float squareTerm = (1.0 - anisotropy * anisotropy) / (1.0 - anisotropy + 2.0 * anisotropy * randomA);
            cosTheta = (1.0 + anisotropy * anisotropy - squareTerm * squareTerm) / (2.0 * anisotropy);
//          cosTheta = (1.0 + anisotropy * anisotropy - squareTerm * squareTerm) / (2.0 * anisotropy);
        }
//      }
        cosTheta = clamp(cosTheta, -1.0, 1.0);
//      cosTheta = clamp(cosTheta, -1.0, 1.0);
        float sinTheta = sqrt(max(0.0, 1.0 - cosTheta * cosTheta));
//      float sinTheta = sqrt(max(0.0, 1.0 - cosTheta * cosTheta));
        float azimuth = 2.0 * PI * randomB;
//      float azimuth = 2.0 * PI * randomB;
        vec3 helperAxis = (abs(travelDirection.y) < 0.999) ? vec3(0.0, 1.0, 0.0) : vec3(1.0, 0.0, 0.0);
//      vec3 helperAxis = (abs(travelDirection.y) < 0.999) ? vec3(0.0, 1.0, 0.0) : vec3(1.0, 0.0, 0.0);
        vec3 tangentAxis = normalize(cross(helperAxis, travelDirection));
//      vec3 tangentAxis = normalize(cross(helperAxis, travelDirection));
        vec3 bitangentAxis = cross(travelDirection, tangentAxis);
//      vec3 bitangentAxis = cross(travelDirection, tangentAxis);
        return normalize(tangentAxis * (sinTheta * cos(azimuth)) + bitangentAxis * (sinTheta * sin(azimuth)) + travelDirection * cosTheta);
//      return normalize(tangentAxis * (sinTheta * cos(azimuth)) + bitangentAxis * (sinTheta * sin(azimuth)) + travelDirection * cosTheta);
    }
//  }

    // Billowy base shape: the Perlin-Worley channel, dilated by the low-frequency Worley FBM so
//  // Billowy base shape: the Perlin-Worley channel, dilated by the low-frequency Worley FBM so
    // cloud bodies grow rounded, cauliflower-like lobes instead of Perlin's stringy connectivity.
//  // cloud bodies grow rounded, cauliflower-like lobes instead of Perlin's stringy connectivity.
    float sampleCloudBaseShape(vec3 noiseCoordinates) {
//  float sampleCloudBaseShape(vec3 noiseCoordinates) {
        vec4 baseNoise = textureLod(uCloudNoiseTexture, noiseCoordinates, 0.0);
//      vec4 baseNoise = textureLod(uCloudNoiseTexture, noiseCoordinates, 0.0);
        float lowFrequencyFbm = dot(baseNoise.gba, CLOUD_NOISE_FBM_WEIGHTS);
//      float lowFrequencyFbm = dot(baseNoise.gba, CLOUD_NOISE_FBM_WEIGHTS);
        float dilatedShape = remapValue(baseNoise.r, lowFrequencyFbm - 1.0, 1.0, 0.0, 1.0);
//      float dilatedShape = remapValue(baseNoise.r, lowFrequencyFbm - 1.0, 1.0, 0.0, 1.0);
        return clamp(remapValue(dilatedShape, CLOUD_BASE_SHAPE_RANGE.x, CLOUD_BASE_SHAPE_RANGE.y, 0.0, 1.0), 0.0, 1.0);
//      return clamp(remapValue(dilatedShape, CLOUD_BASE_SHAPE_RANGE.x, CLOUD_BASE_SHAPE_RANGE.y, 0.0, 1.0), 0.0, 1.0);
    }
//  }

    // High-frequency Worley FBM used to erode the edges of an already-shaped density.
//  // High-frequency Worley FBM used to erode the edges of an already-shaped density.
    float sampleCloudDetail(vec3 noiseCoordinates) {
//  float sampleCloudDetail(vec3 noiseCoordinates) {
        return dot(textureLod(uCloudNoiseTexture, noiseCoordinates, 0.0).gba, CLOUD_NOISE_FBM_WEIGHTS);
//      return dot(textureLod(uCloudNoiseTexture, noiseCoordinates, 0.0).gba, CLOUD_NOISE_FBM_WEIGHTS);
    }
//  }
