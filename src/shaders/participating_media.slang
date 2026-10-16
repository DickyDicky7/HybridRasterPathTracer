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
    // The trackers draw their random numbers through IMediumRandom: each pass passes its own generator
//  // The trackers draw their random numbers through IMediumRandom: each pass passes its own generator
    // as the generic argument (e.g. estimateMediumTransmittance<ShadingRandom>(...)), so the volumes
//  // as the generic argument (e.g. estimateMediumTransmittance<ShadingRandom>(...)), so the volumes
    // consume the same RNG stream as the rest of that pass.
//  // consume the same RNG stream as the rest of that pass.
    // ============================================================================
//  // ============================================================================
    module participating_media;
//  module participating_media;

    import math_constants;
//  import math_constants;
    import volumetric_common;
//  import volumetric_common;

    // A pass's random number generator: next() yields a uniform float in [0, 1).
//  // A pass's random number generator: next() yields a uniform float in [0, 1).
    public interface IMediumRandom {
//  public interface IMediumRandom {
        static float next();
//      static float next();
    }
//  }

    public static const int MAX_MEDIUM_VOLUMES = 4;
//  public static const int MAX_MEDIUM_VOLUMES = 4;
    public static const int MEDIUM_KIND_FOG = 0;
//  public static const int MEDIUM_KIND_FOG = 0;
    public static const int MEDIUM_KIND_CLOUD = 1;
//  public static const int MEDIUM_KIND_CLOUD = 1;
    public static const int MEDIUM_MAX_TRACKING_STEPS = 256;
//  public static const int MEDIUM_MAX_TRACKING_STEPS = 256;
    public static const float MEDIUM_DETAIL_EROSION = 0.3;
//  public static const float MEDIUM_DETAIL_EROSION = 0.3;
    public static const float MEDIUM_RATIO_TRACKING_ROULETTE = 0.1;
//  public static const float MEDIUM_RATIO_TRACKING_ROULETTE = 0.1;

    public struct MediumVolume {
//  public struct MediumVolume {
        public float3 aabbMin;
//      public float3 aabbMin;
        public float density;        // majorant extinction coefficient, per world unit
//      public float density;        // majorant extinction coefficient, per world unit
        public float3 aabbMax;
//      public float3 aabbMax;
        public float anisotropy;     // Henyey-Greenstein g
//      public float anisotropy;     // Henyey-Greenstein g
        public float3 albedo;          // single-scattering albedo
//      public float3 albedo;          // single-scattering albedo
        public float noiseFrequency; // noise tiles per world unit (clouds only)
//      public float noiseFrequency; // noise tiles per world unit (clouds only)
        public int kind;             // MEDIUM_KIND_FOG or MEDIUM_KIND_CLOUD
//      public int kind;             // MEDIUM_KIND_FOG or MEDIUM_KIND_CLOUD
    };
//  };

    public uniform int uMediaEnabled;
//  public uniform int uMediaEnabled;
    public uniform int uMediumVolumeCount;
//  public uniform int uMediumVolumeCount;
    public uniform MediumVolume uMediumVolumes[MAX_MEDIUM_VOLUMES];
//  public uniform MediumVolume uMediumVolumes[MAX_MEDIUM_VOLUMES];

    public bool isMediaActive() {
//  public bool isMediaActive() {
        return (uMediaEnabled != 0) && uMediumVolumeCount > 0;
//      return (uMediaEnabled != 0) && uMediumVolumeCount > 0;
    }
//  }

    // Slab test clipped to [0, segmentEnd]; returns the overlap as (entry, exit).
//  // Slab test clipped to [0, segmentEnd]; returns the overlap as (entry, exit).
    public bool intersectMediumVolume(int volumeIndex, float3 rayOrigin, float3 rayDirection, float segmentEnd, out float entryDistance, out float exitDistance) {
//  public bool intersectMediumVolume(int volumeIndex, float3 rayOrigin, float3 rayDirection, float segmentEnd, out float entryDistance, out float exitDistance) {
        float3 inverseDirection = 1.0 / rayDirection;
//      float3 inverseDirection = 1.0 / rayDirection;
        float3 slabNear = (uMediumVolumes[volumeIndex].aabbMin - rayOrigin) * inverseDirection;
//      float3 slabNear = (uMediumVolumes[volumeIndex].aabbMin - rayOrigin) * inverseDirection;
        float3 slabFar = (uMediumVolumes[volumeIndex].aabbMax - rayOrigin) * inverseDirection;
//      float3 slabFar = (uMediumVolumes[volumeIndex].aabbMax - rayOrigin) * inverseDirection;
        float3 slabMin = min(slabNear, slabFar);
//      float3 slabMin = min(slabNear, slabFar);
        float3 slabMax = max(slabNear, slabFar);
//      float3 slabMax = max(slabNear, slabFar);
        entryDistance = max(max(max(slabMin.x, slabMin.y), slabMin.z), 0.0);
//      entryDistance = max(max(max(slabMin.x, slabMin.y), slabMin.z), 0.0);
        exitDistance = min(min(min(slabMax.x, slabMax.y), slabMax.z), segmentEnd);
//      exitDistance = min(min(min(slabMax.x, slabMax.y), slabMax.z), segmentEnd);
        if (exitDistance > entryDistance && uMediumVolumes[volumeIndex].kind == MEDIUM_KIND_CLOUD) {
//      if (exitDistance > entryDistance && uMediumVolumes[volumeIndex].kind == MEDIUM_KIND_CLOUD) {
            // A cloud is further clipped to the ellipsoid inscribed in its box: mediumVolumeDensity's shape mask is exactly 0
//          // A cloud is further clipped to the ellipsoid inscribed in its box: mediumVolumeDensity's shape mask is exactly 0
            // wherever the box-normalized radius reaches 1, so the density outside that ellipsoid is 0. Walking only the
//          // wherever the box-normalized radius reaches 1, so the density outside that ellipsoid is 0. Walking only the
            // inner part is exact, not an approximation -- delta tracking's free flights are memoryless, so null collisions
//          // inner part is exact, not an approximation -- delta tracking's free flights are memoryless, so null collisions
            // skipped in empty space cannot move the first real one, and ratio tracking's factor there is 1 - 0 = 1 -- but it
//          // skipped in empty space cannot move the first real one, and ratio tracking's factor there is 1 - 0 = 1 -- but it
            // spares every tracking step (up to two noise fetches each) the box corners used to cost.
//          // spares every tracking step (up to two noise fetches each) the box corners used to cost.
            // Ray against the unit sphere in box-normalized space: |localOrigin + t * localDirection|^2 = 1.
//          // Ray against the unit sphere in box-normalized space: |localOrigin + t * localDirection|^2 = 1.
            float3 volumeCentre = 0.5 * (uMediumVolumes[volumeIndex].aabbMin + uMediumVolumes[volumeIndex].aabbMax);
//          float3 volumeCentre = 0.5 * (uMediumVolumes[volumeIndex].aabbMin + uMediumVolumes[volumeIndex].aabbMax);
            float3 volumeHalfExtent = 0.5 * (uMediumVolumes[volumeIndex].aabbMax - uMediumVolumes[volumeIndex].aabbMin);
//          float3 volumeHalfExtent = 0.5 * (uMediumVolumes[volumeIndex].aabbMax - uMediumVolumes[volumeIndex].aabbMin);
            float3 localOrigin = (rayOrigin - volumeCentre) / volumeHalfExtent;
//          float3 localOrigin = (rayOrigin - volumeCentre) / volumeHalfExtent;
            float3 localDirection = rayDirection / volumeHalfExtent;
//          float3 localDirection = rayDirection / volumeHalfExtent;
            float quadraticA = dot(localDirection, localDirection);
//          float quadraticA = dot(localDirection, localDirection);
            float quadraticHalfB = dot(localOrigin, localDirection);
//          float quadraticHalfB = dot(localOrigin, localDirection);
            float quadraticC = dot(localOrigin, localOrigin) - 1.0;
//          float quadraticC = dot(localOrigin, localOrigin) - 1.0;
            float discriminant = quadraticHalfB * quadraticHalfB - quadraticA * quadraticC;
//          float discriminant = quadraticHalfB * quadraticHalfB - quadraticA * quadraticC;
            if (discriminant <= 0.0) {
//          if (discriminant <= 0.0) {
                return false;
//              return false;
            }
//          }
            float rootSpan = sqrt(discriminant);
//          float rootSpan = sqrt(discriminant);
            entryDistance = max(entryDistance, (-quadraticHalfB - rootSpan) / quadraticA);
//          entryDistance = max(entryDistance, (-quadraticHalfB - rootSpan) / quadraticA);
            exitDistance = min(exitDistance, (-quadraticHalfB + rootSpan) / quadraticA);
//          exitDistance = min(exitDistance, (-quadraticHalfB + rootSpan) / quadraticA);
        }
//      }
        return exitDistance > entryDistance;
//      return exitDistance > entryDistance;
    }
//  }

    // Normalized density in [0, 1], so the volume's density field doubles as its exact majorant.
//  // Normalized density in [0, 1], so the volume's density field doubles as its exact majorant.
    // Clouds are an ellipsoidal puff with a flattened base, carved by the shared Perlin-Worley noise.
//  // Clouds are an ellipsoidal puff with a flattened base, carved by the shared Perlin-Worley noise.
    public float mediumVolumeDensity(int volumeIndex, float3 position) {
//  public float mediumVolumeDensity(int volumeIndex, float3 position) {
        if (uMediumVolumes[volumeIndex].kind == MEDIUM_KIND_FOG) {
//      if (uMediumVolumes[volumeIndex].kind == MEDIUM_KIND_FOG) {
            return 1.0;
//          return 1.0;
        }
//      }
        float3 volumeCentre = 0.5 * (uMediumVolumes[volumeIndex].aabbMin + uMediumVolumes[volumeIndex].aabbMax);
//      float3 volumeCentre = 0.5 * (uMediumVolumes[volumeIndex].aabbMin + uMediumVolumes[volumeIndex].aabbMax);
        float3 volumeHalfExtent = 0.5 * (uMediumVolumes[volumeIndex].aabbMax - uMediumVolumes[volumeIndex].aabbMin);
//      float3 volumeHalfExtent = 0.5 * (uMediumVolumes[volumeIndex].aabbMax - uMediumVolumes[volumeIndex].aabbMin);
        float3 localPosition = (position - volumeCentre) / volumeHalfExtent;
//      float3 localPosition = (position - volumeCentre) / volumeHalfExtent;
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
        float3 noiseCoordinates = position * uMediumVolumes[volumeIndex].noiseFrequency;
//      float3 noiseCoordinates = position * uMediumVolumes[volumeIndex].noiseFrequency;
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
    public float sampleMediumVolumeFreeFlight<Random : IMediumRandom>(int volumeIndex, float3 rayOrigin, float3 rayDirection, float entryDistance, float exitDistance) {
//  public float sampleMediumVolumeFreeFlight<Random : IMediumRandom>(int volumeIndex, float3 rayOrigin, float3 rayDirection, float entryDistance, float exitDistance) {
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
            trackedDistance -= log(1.0 - Random.next()) / majorant;
//          trackedDistance -= log(1.0 - Random.next()) / majorant;
            return (trackedDistance < exitDistance) ? trackedDistance : INF;
//          return (trackedDistance < exitDistance) ? trackedDistance : INF;
        }
//      }
        for (int trackingStep = 0; trackingStep < MEDIUM_MAX_TRACKING_STEPS; trackingStep++) {
//      for (int trackingStep = 0; trackingStep < MEDIUM_MAX_TRACKING_STEPS; trackingStep++) {
            trackedDistance -= log(1.0 - Random.next()) / majorant;
//          trackedDistance -= log(1.0 - Random.next()) / majorant;
            if (trackedDistance >= exitDistance) {
//          if (trackedDistance >= exitDistance) {
                return INF;
//              return INF;
            }
//          }
            // Real collision with probability density / majorant; otherwise a null collision, keep going.
//          // Real collision with probability density / majorant; otherwise a null collision, keep going.
            if (Random.next() < mediumVolumeDensity(volumeIndex, rayOrigin + rayDirection * trackedDistance)) {
//          if (Random.next() < mediumVolumeDensity(volumeIndex, rayOrigin + rayDirection * trackedDistance)) {
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
    public bool sampleMediumScattering<Random : IMediumRandom>(float3 rayOrigin, float3 rayDirection, float segmentEnd, out float scatterDistance, out int scatterVolumeIndex) {
//  public bool sampleMediumScattering<Random : IMediumRandom>(float3 rayOrigin, float3 rayDirection, float segmentEnd, out float scatterDistance, out int scatterVolumeIndex) {
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
            float collisionDistance = sampleMediumVolumeFreeFlight<Random>(volumeIndex, rayOrigin, rayDirection, entryDistance, exitDistance);
//          float collisionDistance = sampleMediumVolumeFreeFlight<Random>(volumeIndex, rayOrigin, rayDirection, entryDistance, exitDistance);
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
    public float estimateMediumTransmittance<Random : IMediumRandom>(float3 rayOrigin, float3 rayDirection, float segmentEnd) {
//  public float estimateMediumTransmittance<Random : IMediumRandom>(float3 rayOrigin, float3 rayDirection, float segmentEnd) {
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
                trackedDistance -= log(1.0 - Random.next()) / majorant;
//              trackedDistance -= log(1.0 - Random.next()) / majorant;
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
                    if (Random.next() < 0.5) {
//                  if (Random.next() < 0.5) {
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
