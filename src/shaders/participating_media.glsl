    // ============================================================================
//  // ============================================================================
    // In-scene participating media.
//  // In-scene participating media.
    //
//  //
    // Axis-aligned volumes of fog (homogeneous) or cloud (procedural, heterogeneous) that the path
//  // Axis-aligned volumes of fog (homogeneous) or cloud (procedural, heterogeneous) that the path
    // tracer scatters inside, instead of treating the space between surfaces as vacuum.
//  // tracer scatters inside, instead of treating the space between surfaces as vacuum.
    //
//  //
    // Free-flight distances are sampled per volume -- analytically for fog, by delta tracking
//  // Free-flight distances are sampled per volume -- analytically for fog, by delta tracking
    // (Woodcock) against the volume's own majorant for clouds -- and the nearest collision wins.
//  // (Woodcock) against the volume's own majorant for clouds -- and the nearest collision wins.
    // Overlapping volumes stay unbiased this way: extinction coefficients add, and the first
//  // Overlapping volumes stay unbiased this way: extinction coefficients add, and the first
    // collision of a sum of independent media is the earliest of each medium's first collision.
//  // collision of a sum of independent media is the earliest of each medium's first collision.
    // Shadow rays use the matching transmittance estimators: Beer-Lambert for fog, ratio tracking
//  // Shadow rays use the matching transmittance estimators: Beer-Lambert for fog, ratio tracking
    // (Cramer 1978, Novak et al. 2014) for clouds.
//  // (Cramer 1978, Novak et al. 2014) for clouds.
    //
//  //
    // Extinction is grey (one scalar per volume) and colour lives in the single-scattering albedo, so
//  // Extinction is grey (one scalar per volume) and colour lives in the single-scattering albedo, so
    // a real collision always scatters and multiplies the throughput by the albedo, and passing
//  // a real collision always scatters and multiplies the throughput by the albedo, and passing
    // through leaves the throughput untouched.
//  // through leaves the throughput untouched.
    //
//  //
    // Requires the includer to have already defined: PI, INF, MEDIA_RAND (a function-like macro or
//  // Requires the includer to have already defined: PI, INF, MEDIA_RAND (a function-like macro or
    // function name yielding a uniform float in [0, 1)), uCloudNoiseTexture, and to have included
//  // function name yielding a uniform float in [0, 1)), uCloudNoiseTexture, and to have included
    // volumetric_common.glsl.
//  // volumetric_common.glsl.
    // ============================================================================
//  // ============================================================================
    const int MAX_MEDIUM_VOLUMES = 4;
//  const int MAX_MEDIUM_VOLUMES = 4;
    const int MEDIUM_KIND_FOG = 0;
//  const int MEDIUM_KIND_FOG = 0;
    const int MEDIUM_KIND_CLOUD = 1;
//  const int MEDIUM_KIND_CLOUD = 1;
    const int MEDIUM_MAX_TRACKING_STEPS = 256;
//  const int MEDIUM_MAX_TRACKING_STEPS = 256;
    const float MEDIUM_DETAIL_EROSION = 0.3;
//  const float MEDIUM_DETAIL_EROSION = 0.3;
    const float MEDIUM_RATIO_TRACKING_ROULETTE = 0.1;
//  const float MEDIUM_RATIO_TRACKING_ROULETTE = 0.1;

    struct MediumVolume {
//  struct MediumVolume {
        vec3 aabbMin;
//      vec3 aabbMin;
        float density;        // majorant extinction coefficient, per world unit
//      float density;        // majorant extinction coefficient, per world unit
        vec3 aabbMax;
//      vec3 aabbMax;
        float anisotropy;     // Henyey-Greenstein g
//      float anisotropy;     // Henyey-Greenstein g
        vec3 albedo;          // single-scattering albedo
//      vec3 albedo;          // single-scattering albedo
        float noiseFrequency; // noise tiles per world unit (clouds only)
//      float noiseFrequency; // noise tiles per world unit (clouds only)
        int kind;             // MEDIUM_KIND_FOG or MEDIUM_KIND_CLOUD
//      int kind;             // MEDIUM_KIND_FOG or MEDIUM_KIND_CLOUD
    };
//  };

    uniform bool uMediaEnabled;
//  uniform bool uMediaEnabled;
    uniform int uMediumVolumeCount;
//  uniform int uMediumVolumeCount;
    uniform MediumVolume uMediumVolumes[MAX_MEDIUM_VOLUMES];
//  uniform MediumVolume uMediumVolumes[MAX_MEDIUM_VOLUMES];

    bool isMediaActive() {
//  bool isMediaActive() {
        return uMediaEnabled && uMediumVolumeCount > 0;
//      return uMediaEnabled && uMediumVolumeCount > 0;
    }
//  }

    // Slab test clipped to [0, segmentEnd]; returns the overlap as (entry, exit).
//  // Slab test clipped to [0, segmentEnd]; returns the overlap as (entry, exit).
    bool intersectMediumVolume(int volumeIndex, vec3 rayOrigin, vec3 rayDirection, float segmentEnd, out float entryDistance, out float exitDistance) {
//  bool intersectMediumVolume(int volumeIndex, vec3 rayOrigin, vec3 rayDirection, float segmentEnd, out float entryDistance, out float exitDistance) {
        vec3 inverseDirection = 1.0 / rayDirection;
//      vec3 inverseDirection = 1.0 / rayDirection;
        vec3 slabNear = (uMediumVolumes[volumeIndex].aabbMin - rayOrigin) * inverseDirection;
//      vec3 slabNear = (uMediumVolumes[volumeIndex].aabbMin - rayOrigin) * inverseDirection;
        vec3 slabFar = (uMediumVolumes[volumeIndex].aabbMax - rayOrigin) * inverseDirection;
//      vec3 slabFar = (uMediumVolumes[volumeIndex].aabbMax - rayOrigin) * inverseDirection;
        vec3 slabMin = min(slabNear, slabFar);
//      vec3 slabMin = min(slabNear, slabFar);
        vec3 slabMax = max(slabNear, slabFar);
//      vec3 slabMax = max(slabNear, slabFar);
        entryDistance = max(max(max(slabMin.x, slabMin.y), slabMin.z), 0.0);
//      entryDistance = max(max(max(slabMin.x, slabMin.y), slabMin.z), 0.0);
        exitDistance = min(min(min(slabMax.x, slabMax.y), slabMax.z), segmentEnd);
//      exitDistance = min(min(min(slabMax.x, slabMax.y), slabMax.z), segmentEnd);
        return exitDistance > entryDistance;
//      return exitDistance > entryDistance;
    }
//  }

    // Normalized density in [0, 1], so the volume's density field doubles as its exact majorant.
//  // Normalized density in [0, 1], so the volume's density field doubles as its exact majorant.
    // Clouds are an ellipsoidal puff with a flattened base, carved by the shared Perlin-Worley noise.
//  // Clouds are an ellipsoidal puff with a flattened base, carved by the shared Perlin-Worley noise.
    float mediumVolumeDensity(int volumeIndex, vec3 position) {
//  float mediumVolumeDensity(int volumeIndex, vec3 position) {
        if (uMediumVolumes[volumeIndex].kind == MEDIUM_KIND_FOG) {
//      if (uMediumVolumes[volumeIndex].kind == MEDIUM_KIND_FOG) {
            return 1.0;
//          return 1.0;
        }
//      }
        vec3 volumeCentre = 0.5 * (uMediumVolumes[volumeIndex].aabbMin + uMediumVolumes[volumeIndex].aabbMax);
//      vec3 volumeCentre = 0.5 * (uMediumVolumes[volumeIndex].aabbMin + uMediumVolumes[volumeIndex].aabbMax);
        vec3 volumeHalfExtent = 0.5 * (uMediumVolumes[volumeIndex].aabbMax - uMediumVolumes[volumeIndex].aabbMin);
//      vec3 volumeHalfExtent = 0.5 * (uMediumVolumes[volumeIndex].aabbMax - uMediumVolumes[volumeIndex].aabbMin);
        vec3 localPosition = (position - volumeCentre) / volumeHalfExtent;
//      vec3 localPosition = (position - volumeCentre) / volumeHalfExtent;
        float shape = clamp(remapValue(length(localPosition), 1.0, 0.3, 0.0, 1.0), 0.0, 1.0);
//      float shape = clamp(remapValue(length(localPosition), 1.0, 0.3, 0.0, 1.0), 0.0, 1.0);
        shape *= clamp(remapValue(localPosition.y, -1.0, -0.6, 0.0, 1.0), 0.0, 1.0);
//      shape *= clamp(remapValue(localPosition.y, -1.0, -0.6, 0.0, 1.0), 0.0, 1.0);
        if (shape <= 0.0) {
//      if (shape <= 0.0) {
            return 0.0;
//          return 0.0;
        }
//      }
        vec3 noiseCoordinates = position * uMediumVolumes[volumeIndex].noiseFrequency;
//      vec3 noiseCoordinates = position * uMediumVolumes[volumeIndex].noiseFrequency;
        float baseShape = sampleCloudBaseShape(noiseCoordinates);
//      float baseShape = sampleCloudBaseShape(noiseCoordinates);
        float density = clamp(remapValue(baseShape, 1.0 - shape, 1.0, 0.0, 1.0), 0.0, 1.0);
//      float density = clamp(remapValue(baseShape, 1.0 - shape, 1.0, 0.0, 1.0), 0.0, 1.0);
        if (density <= 0.0) {
//      if (density <= 0.0) {
            return 0.0;
//          return 0.0;
        }
//      }
        float detail = sampleCloudDetail(noiseCoordinates * 4.0);
//      float detail = sampleCloudDetail(noiseCoordinates * 4.0);
        return clamp(remapValue(density, detail * MEDIUM_DETAIL_EROSION, 1.0, 0.0, 1.0), 0.0, 1.0);
//      return clamp(remapValue(density, detail * MEDIUM_DETAIL_EROSION, 1.0, 0.0, 1.0), 0.0, 1.0);
    }
//  }

    // Distance to this volume's first real collision inside [entryDistance, exitDistance], or INF.
//  // Distance to this volume's first real collision inside [entryDistance, exitDistance], or INF.
    float sampleMediumVolumeFreeFlight(int volumeIndex, vec3 rayOrigin, vec3 rayDirection, float entryDistance, float exitDistance) {
//  float sampleMediumVolumeFreeFlight(int volumeIndex, vec3 rayOrigin, vec3 rayDirection, float entryDistance, float exitDistance) {
        float majorant = uMediumVolumes[volumeIndex].density;
//      float majorant = uMediumVolumes[volumeIndex].density;
        if (majorant <= 0.0) {
//      if (majorant <= 0.0) {
            return INF;
//          return INF;
        }
//      }
        float trackedDistance = entryDistance;
//      float trackedDistance = entryDistance;
        if (uMediumVolumes[volumeIndex].kind == MEDIUM_KIND_FOG) {
//      if (uMediumVolumes[volumeIndex].kind == MEDIUM_KIND_FOG) {
            trackedDistance -= log(1.0 - MEDIA_RAND()) / majorant;
//          trackedDistance -= log(1.0 - MEDIA_RAND()) / majorant;
            return (trackedDistance < exitDistance) ? trackedDistance : INF;
//          return (trackedDistance < exitDistance) ? trackedDistance : INF;
        }
//      }
        for (int trackingStep = 0; trackingStep < MEDIUM_MAX_TRACKING_STEPS; trackingStep++) {
//      for (int trackingStep = 0; trackingStep < MEDIUM_MAX_TRACKING_STEPS; trackingStep++) {
            trackedDistance -= log(1.0 - MEDIA_RAND()) / majorant;
//          trackedDistance -= log(1.0 - MEDIA_RAND()) / majorant;
            if (trackedDistance >= exitDistance) {
//          if (trackedDistance >= exitDistance) {
                return INF;
//              return INF;
            }
//          }
            // Real collision with probability density / majorant; otherwise a null collision, keep going.
//          // Real collision with probability density / majorant; otherwise a null collision, keep going.
            if (MEDIA_RAND() < mediumVolumeDensity(volumeIndex, rayOrigin + rayDirection * trackedDistance)) {
//          if (MEDIA_RAND() < mediumVolumeDensity(volumeIndex, rayOrigin + rayDirection * trackedDistance)) {
                return trackedDistance;
//              return trackedDistance;
            }
//          }
        }
//      }
        return INF;
//      return INF;
    }
//  }

    // Samples where along [0, segmentEnd] the ray first scatters in any volume. Returns false when the
//  // Samples where along [0, segmentEnd] the ray first scatters in any volume. Returns false when the
    // ray crosses every volume unscattered, which (with grey extinction) needs no throughput change.
//  // ray crosses every volume unscattered, which (with grey extinction) needs no throughput change.
    bool sampleMediumScattering(vec3 rayOrigin, vec3 rayDirection, float segmentEnd, out float scatterDistance, out int scatterVolumeIndex) {
//  bool sampleMediumScattering(vec3 rayOrigin, vec3 rayDirection, float segmentEnd, out float scatterDistance, out int scatterVolumeIndex) {
        scatterDistance = INF;
//      scatterDistance = INF;
        scatterVolumeIndex = -1;
//      scatterVolumeIndex = -1;
        if (!isMediaActive()) {
//      if (!isMediaActive()) {
            return false;
//          return false;
        }
//      }
        for (int volumeIndex = 0; volumeIndex < min(uMediumVolumeCount, MAX_MEDIUM_VOLUMES); volumeIndex++) {
//      for (int volumeIndex = 0; volumeIndex < min(uMediumVolumeCount, MAX_MEDIUM_VOLUMES); volumeIndex++) {
            float entryDistance;
//          float entryDistance;
            float exitDistance;
//          float exitDistance;
            if (!intersectMediumVolume(volumeIndex, rayOrigin, rayDirection, min(segmentEnd, scatterDistance), entryDistance, exitDistance)) {
//          if (!intersectMediumVolume(volumeIndex, rayOrigin, rayDirection, min(segmentEnd, scatterDistance), entryDistance, exitDistance)) {
                continue;
//              continue;
            }
//          }
            float collisionDistance = sampleMediumVolumeFreeFlight(volumeIndex, rayOrigin, rayDirection, entryDistance, exitDistance);
//          float collisionDistance = sampleMediumVolumeFreeFlight(volumeIndex, rayOrigin, rayDirection, entryDistance, exitDistance);
            if (collisionDistance < scatterDistance) {
//          if (collisionDistance < scatterDistance) {
                scatterDistance = collisionDistance;
//              scatterDistance = collisionDistance;
                scatterVolumeIndex = volumeIndex;
//              scatterVolumeIndex = volumeIndex;
            }
//          }
        }
//      }
        return scatterVolumeIndex != -1;
//      return scatterVolumeIndex != -1;
    }
//  }

    // Unbiased estimate of the transmittance along [0, segmentEnd], for shadow rays.
//  // Unbiased estimate of the transmittance along [0, segmentEnd], for shadow rays.
    float estimateMediumTransmittance(vec3 rayOrigin, vec3 rayDirection, float segmentEnd) {
//  float estimateMediumTransmittance(vec3 rayOrigin, vec3 rayDirection, float segmentEnd) {
        if (!isMediaActive()) {
//      if (!isMediaActive()) {
            return 1.0;
//          return 1.0;
        }
//      }
        float transmittance = 1.0;
//      float transmittance = 1.0;
        for (int volumeIndex = 0; volumeIndex < min(uMediumVolumeCount, MAX_MEDIUM_VOLUMES); volumeIndex++) {
//      for (int volumeIndex = 0; volumeIndex < min(uMediumVolumeCount, MAX_MEDIUM_VOLUMES); volumeIndex++) {
            float entryDistance;
//          float entryDistance;
            float exitDistance;
//          float exitDistance;
            if (!intersectMediumVolume(volumeIndex, rayOrigin, rayDirection, segmentEnd, entryDistance, exitDistance)) {
//          if (!intersectMediumVolume(volumeIndex, rayOrigin, rayDirection, segmentEnd, entryDistance, exitDistance)) {
                continue;
//              continue;
            }
//          }
            float majorant = uMediumVolumes[volumeIndex].density;
//          float majorant = uMediumVolumes[volumeIndex].density;
            if (majorant <= 0.0) {
//          if (majorant <= 0.0) {
                continue;
//              continue;
            }
//          }
            if (uMediumVolumes[volumeIndex].kind == MEDIUM_KIND_FOG) {
//          if (uMediumVolumes[volumeIndex].kind == MEDIUM_KIND_FOG) {
                transmittance *= exp(-majorant * (exitDistance - entryDistance));
//              transmittance *= exp(-majorant * (exitDistance - entryDistance));
                continue;
//              continue;
            }
//          }
            float trackedDistance = entryDistance;
//          float trackedDistance = entryDistance;
            for (int trackingStep = 0; trackingStep < MEDIUM_MAX_TRACKING_STEPS; trackingStep++) {
//          for (int trackingStep = 0; trackingStep < MEDIUM_MAX_TRACKING_STEPS; trackingStep++) {
                trackedDistance -= log(1.0 - MEDIA_RAND()) / majorant;
//              trackedDistance -= log(1.0 - MEDIA_RAND()) / majorant;
                if (trackedDistance >= exitDistance) {
//              if (trackedDistance >= exitDistance) {
                    break;
//                  break;
                }
//              }
                transmittance *= 1.0 - mediumVolumeDensity(volumeIndex, rayOrigin + rayDirection * trackedDistance);
//              transmittance *= 1.0 - mediumVolumeDensity(volumeIndex, rayOrigin + rayDirection * trackedDistance);
                if (transmittance <= 0.0) {
//              if (transmittance <= 0.0) {
                    return 0.0;
//                  return 0.0;
                }
//              }
                // Russian roulette keeps deep shadow rays from marching on for vanishing returns.
//              // Russian roulette keeps deep shadow rays from marching on for vanishing returns.
                if (transmittance < MEDIUM_RATIO_TRACKING_ROULETTE) {
//              if (transmittance < MEDIUM_RATIO_TRACKING_ROULETTE) {
                    if (MEDIA_RAND() < 0.5) {
//                  if (MEDIA_RAND() < 0.5) {
                        return 0.0;
//                      return 0.0;
                    }
//                  }
                    transmittance *= 2.0;
//                  transmittance *= 2.0;
                }
//              }
            }
//          }
        }
//      }
        return transmittance;
//      return transmittance;
    }
//  }
