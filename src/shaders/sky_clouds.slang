    // ============================================================================
//  // ============================================================================
    // Volumetric sky cloud layer (ray marched).
//  // Volumetric sky cloud layer (ray marched).
    //
//  //
    // A cumulus layer on a spherical shell above a ground-level observer, ray marched through a
//  // A cumulus layer on a spherical shell above a ground-level observer, ray marched through a
    // Perlin-Worley density field in the style of Schneider's Nubis (Horizon Zero Dawn, SIGGRAPH 2015)
//  // Perlin-Worley density field in the style of Schneider's Nubis (Horizon Zero Dawn, SIGGRAPH 2015)
    // and lit with Hillaire's energy-conserving integration and multiple-scattering octaves
//  // and lit with Hillaire's energy-conserving integration and multiple-scattering octaves
    // ("Physically Based Sky, Atmosphere and Cloud Rendering in Frostbite", SIGGRAPH 2016).
//  // ("Physically Based Sky, Atmosphere and Cloud Rendering in Frostbite", SIGGRAPH 2016).
    //
//  //
    // The layer is kilometres away, so like the rest of the sky it is a function of direction only:
//  // The layer is kilometres away, so like the rest of the sky it is a function of direction only:
    // every camera position sees the same clouds, which is what lets sky_clouds_bake_cs.slang
//  // every camera position sees the same clouds, which is what lets sky_clouds_bake_cs.slang
    // pre-integrate it into an equirectangular texture for every non-primary lookup.
//  // pre-integrate it into an equirectangular texture for every non-primary lookup.
    //
//  //
    // Builds on sky_environment (procedural sky, sun direction) and volumetric_common (noise, phase),
//  // Builds on sky_environment (procedural sky, sun direction) and volumetric_common (noise, phase),
    // and owns the layer's coverage and density parameters.
//  // and owns the layer's coverage and density parameters.
    // ============================================================================
//  // ============================================================================
    module sky_clouds;
//  module sky_clouds;

    import math_constants;
//  import math_constants;
    import sky_environment;
//  import sky_environment;
    import volumetric_common;
//  import volumetric_common;

    // Fraction of the sky the weather map lets clouds form in, and the layer's peak extinction per metre.
//  // Fraction of the sky the weather map lets clouds form in, and the layer's peak extinction per metre.
    public uniform float uSkyCloudCoverage;
//  public uniform float uSkyCloudCoverage;
    public uniform float uSkyCloudDensity;
//  public uniform float uSkyCloudDensity;

    public static const float SKY_PLANET_RADIUS = 6360000.0;           // metres
//  public static const float SKY_PLANET_RADIUS = 6360000.0;           // metres
    public static const float SKY_CLOUD_LAYER_BOTTOM = 1500.0;         // metres above ground
//  public static const float SKY_CLOUD_LAYER_BOTTOM = 1500.0;         // metres above ground
    public static const float SKY_CLOUD_LAYER_TOP = 4000.0;            // metres above ground
//  public static const float SKY_CLOUD_LAYER_TOP = 4000.0;            // metres above ground
    public static const float SKY_CLOUD_MAX_MARCH_DISTANCE = 30000.0;  // caps the grazing march near the horizon
//  public static const float SKY_CLOUD_MAX_MARCH_DISTANCE = 30000.0;  // caps the grazing march near the horizon
    public static const float SKY_CLOUD_FADE_DISTANCE = 45000.0;       // aerial perspective: clouds dissolve into the haze
//  public static const float SKY_CLOUD_FADE_DISTANCE = 45000.0;       // aerial perspective: clouds dissolve into the haze
    public static const float SKY_CLOUD_WEATHER_SCALE = 1.0 / 36000.0; // noise tiles per metre (coverage map)
//  public static const float SKY_CLOUD_WEATHER_SCALE = 1.0 / 36000.0; // noise tiles per metre (coverage map)
    public static const float SKY_CLOUD_SHAPE_SCALE = 1.0 / 9000.0;    // noise tiles per metre (billow shape)
//  public static const float SKY_CLOUD_SHAPE_SCALE = 1.0 / 9000.0;    // noise tiles per metre (billow shape)
    public static const float SKY_CLOUD_DETAIL_SCALE = 1.0 / 1800.0;   // noise tiles per metre (edge erosion)
//  public static const float SKY_CLOUD_DETAIL_SCALE = 1.0 / 1800.0;   // noise tiles per metre (edge erosion)
    public static const float SKY_CLOUD_DETAIL_EROSION = 0.45;
//  public static const float SKY_CLOUD_DETAIL_EROSION = 0.45;
    public static const float3 SKY_CLOUD_SUN_RADIANCE = float3(32.0, 30.7, 28.6);
//  public static const float3 SKY_CLOUD_SUN_RADIANCE = float3(32.0, 30.7, 28.6);
    public static const int SKY_CLOUD_PRIMARY_STEPS = 48;              // live march for directly visible sky
//  public static const int SKY_CLOUD_PRIMARY_STEPS = 48;              // live march for directly visible sky
    public static const int SKY_CLOUD_BAKE_STEPS = 128;                // one-off bake, so it can afford many more
//  public static const int SKY_CLOUD_BAKE_STEPS = 128;                // one-off bake, so it can afford many more
    public static const int SKY_CLOUD_LIGHT_STEPS = 6;
//  public static const int SKY_CLOUD_LIGHT_STEPS = 6;
    public static const float SKY_CLOUD_LIGHT_FIRST_STEP = 80.0;       // metres, doubled every light-march step
//  public static const float SKY_CLOUD_LIGHT_FIRST_STEP = 80.0;       // metres, doubled every light-march step
    public static const float SKY_CLOUD_TRANSMITTANCE_CUTOFF = 0.01;
//  public static const float SKY_CLOUD_TRANSMITTANCE_CUTOFF = 0.01;
    public static const int SKY_CLOUD_SCATTERING_OCTAVES = 4;
//  public static const int SKY_CLOUD_SCATTERING_OCTAVES = 4;

    // Far intersection of a ray starting inside a sphere centred on the planet.
//  // Far intersection of a ray starting inside a sphere centred on the planet.
    public float raySphereExitDistance(float3 rayOrigin, float3 rayDirection, float sphereRadius) {
//  public float raySphereExitDistance(float3 rayOrigin, float3 rayDirection, float sphereRadius) {
        float projectedDistance = dot(rayOrigin, rayDirection);
//      float projectedDistance = dot(rayOrigin, rayDirection);
        float discriminant = projectedDistance * projectedDistance - (dot(rayOrigin, rayOrigin) - sphereRadius * sphereRadius);
//      float discriminant = projectedDistance * projectedDistance - (dot(rayOrigin, rayOrigin) - sphereRadius * sphereRadius);
        return -projectedDistance + sqrt(max(discriminant, 0.0));
//      return -projectedDistance + sqrt(max(discriminant, 0.0));
    }
//  }

    // Cumulus vertical profile: a sharp flat base, rounded tops.
//  // Cumulus vertical profile: a sharp flat base, rounded tops.
    public float skyCloudHeightGradient(float heightFraction) {
//  public float skyCloudHeightGradient(float heightFraction) {
        return clamp(remapValue(heightFraction, 0.0, 0.08, 0.0, 1.0), 0.0, 1.0)
//      return clamp(remapValue(heightFraction, 0.0, 0.08, 0.0, 1.0), 0.0, 1.0)
             * clamp(remapValue(heightFraction, 0.35, 1.0, 1.0, 0.0), 0.0, 1.0);
//           * clamp(remapValue(heightFraction, 0.35, 1.0, 1.0, 0.0), 0.0, 1.0);
    }
//  }

    // Normalized cloud density in [0, 1] at a point measured from the observer (the planet centre is
//  // Normalized cloud density in [0, 1] at a point measured from the observer (the planet centre is
    // SKY_PLANET_RADIUS straight down). Detail erosion is optional: the light march skips it.
//  // SKY_PLANET_RADIUS straight down). Detail erosion is optional: the light march skips it.
    public float skyCloudDensity(float3 observerRelativePosition, float heightFraction, bool erodeDetail) {
//  public float skyCloudDensity(float3 observerRelativePosition, float heightFraction, bool erodeDetail) {
        if (heightFraction <= 0.0 || heightFraction >= 1.0) {
//      if (heightFraction <= 0.0 || heightFraction >= 1.0) {
            return 0.0;
//          return 0.0;
        }
//      }
        // The low-frequency Worley channel doubles as the weather map: its cellular blobs read as
//      // The low-frequency Worley channel doubles as the weather map: its cellular blobs read as
        // separate cloud clusters, where the Perlin-Worley channel is too uniform and turns overcast.
//      // separate cloud clusters, where the Perlin-Worley channel is too uniform and turns overcast.
        float weather = uCloudNoiseTexture.SampleLevel(float3(observerRelativePosition.xz * SKY_CLOUD_WEATHER_SCALE, 0.5).xzy, 0.0).g;
//      float weather = uCloudNoiseTexture.SampleLevel(float3(observerRelativePosition.xz * SKY_CLOUD_WEATHER_SCALE, 0.5).xzy, 0.0).g;
        float coverageThreshold = 1.0 - uSkyCloudCoverage;
//      float coverageThreshold = 1.0 - uSkyCloudCoverage;
        float coverage = clamp(remapValue(weather, coverageThreshold - 0.1, coverageThreshold + 0.25, 0.0, 1.0), 0.0, 1.0);
//      float coverage = clamp(remapValue(weather, coverageThreshold - 0.1, coverageThreshold + 0.25, 0.0, 1.0), 0.0, 1.0);
        // Coverage also lifts the tops, so thick patches tower and thin patches stay flat.
//      // Coverage also lifts the tops, so thick patches tower and thin patches stay flat.
        float heightGradient = skyCloudHeightGradient(heightFraction / lerp(0.45, 1.0, coverage));
//      float heightGradient = skyCloudHeightGradient(heightFraction / lerp(0.45, 1.0, coverage));
        float cloudCoverage = coverage * heightGradient;
//      float cloudCoverage = coverage * heightGradient;
        if (cloudCoverage <= 0.0) {
//      if (cloudCoverage <= 0.0) {
            return 0.0;
//          return 0.0;
        }
//      }
        float baseShape = sampleCloudBaseShape(observerRelativePosition * SKY_CLOUD_SHAPE_SCALE);
//      float baseShape = sampleCloudBaseShape(observerRelativePosition * SKY_CLOUD_SHAPE_SCALE);
        float density = clamp(remapValue(baseShape, 1.0 - cloudCoverage, 1.0, 0.0, 1.0), 0.0, 1.0) * cloudCoverage;
//      float density = clamp(remapValue(baseShape, 1.0 - cloudCoverage, 1.0, 0.0, 1.0), 0.0, 1.0) * cloudCoverage;
        if (erodeDetail && density > 0.0) {
//      if (erodeDetail && density > 0.0) {
            float detail = sampleCloudDetail(observerRelativePosition * SKY_CLOUD_DETAIL_SCALE);
//          float detail = sampleCloudDetail(observerRelativePosition * SKY_CLOUD_DETAIL_SCALE);
            // Wispy at the base, billowy at the top.
//          // Wispy at the base, billowy at the top.
            float detailModifier = lerp(detail, 1.0 - detail, clamp(heightFraction * 5.0, 0.0, 1.0));
//          float detailModifier = lerp(detail, 1.0 - detail, clamp(heightFraction * 5.0, 0.0, 1.0));
            density = clamp(remapValue(density, detailModifier * SKY_CLOUD_DETAIL_EROSION, 1.0, 0.0, 1.0), 0.0, 1.0);
//          density = clamp(remapValue(density, detailModifier * SKY_CLOUD_DETAIL_EROSION, 1.0, 0.0, 1.0), 0.0, 1.0);
        }
//      }
        return density;
//      return density;
    }
//  }

    // Dual-lobe phase: a strong forward lobe for the silver lining, a weak back lobe so clouds facing
//  // Dual-lobe phase: a strong forward lobe for the silver lining, a weak back lobe so clouds facing
    // away from the sun are not flat. Octave attenuation shrinks the anisotropy, as Hillaire does.
//  // away from the sun are not flat. Octave attenuation shrinks the anisotropy, as Hillaire does.
    public float skyCloudPhase(float cosTheta, float anisotropyScale) {
//  public float skyCloudPhase(float cosTheta, float anisotropyScale) {
        return lerp(henyeyGreenstein(cosTheta, 0.8 * anisotropyScale), henyeyGreenstein(cosTheta, -0.25 * anisotropyScale), 0.4);
//      return lerp(henyeyGreenstein(cosTheta, 0.8 * anisotropyScale), henyeyGreenstein(cosTheta, -0.25 * anisotropyScale), 0.4);
    }
//  }

    // Marches the cloud layer along rayDirection and returns (in-scattered radiance, transmittance).
//  // Marches the cloud layer along rayDirection and returns (in-scattered radiance, transmittance).
    // stepJitter in [0, 1) offsets the samples so successive frames (or neighbouring bake texels)
//  // stepJitter in [0, 1) offsets the samples so successive frames (or neighbouring bake texels)
    // integrate between one another instead of banding.
//  // integrate between one another instead of banding.
    public float4 marchSkyClouds(float3 rayDirection, int stepCount, float stepJitter) {
//  public float4 marchSkyClouds(float3 rayDirection, int stepCount, float stepJitter) {
        if (rayDirection.y <= 0.0 || uSkyCloudCoverage <= 0.0) {
//      if (rayDirection.y <= 0.0 || uSkyCloudCoverage <= 0.0) {
            return float4(0.0, 0.0, 0.0, 1.0);
//          return float4(0.0, 0.0, 0.0, 1.0);
        }
//      }
        float3 planetCentreToObserver = float3(0.0, SKY_PLANET_RADIUS, 0.0);
//      float3 planetCentreToObserver = float3(0.0, SKY_PLANET_RADIUS, 0.0);
        float marchStart = raySphereExitDistance(planetCentreToObserver, rayDirection, SKY_PLANET_RADIUS + SKY_CLOUD_LAYER_BOTTOM);
//      float marchStart = raySphereExitDistance(planetCentreToObserver, rayDirection, SKY_PLANET_RADIUS + SKY_CLOUD_LAYER_BOTTOM);
        float marchEnd = raySphereExitDistance(planetCentreToObserver, rayDirection, SKY_PLANET_RADIUS + SKY_CLOUD_LAYER_TOP);
//      float marchEnd = raySphereExitDistance(planetCentreToObserver, rayDirection, SKY_PLANET_RADIUS + SKY_CLOUD_LAYER_TOP);
        marchEnd = min(marchEnd, marchStart + SKY_CLOUD_MAX_MARCH_DISTANCE);
//      marchEnd = min(marchEnd, marchStart + SKY_CLOUD_MAX_MARCH_DISTANCE);
        float stepLength = (marchEnd - marchStart) / float(stepCount);
//      float stepLength = (marchEnd - marchStart) / float(stepCount);

        float cosSunAngle = dot(rayDirection, SKY_SUN_DIRECTION);
//      float cosSunAngle = dot(rayDirection, SKY_SUN_DIRECTION);
        float octavePhase[SKY_CLOUD_SCATTERING_OCTAVES];
//      float octavePhase[SKY_CLOUD_SCATTERING_OCTAVES];
        for (int octave = 0; octave < SKY_CLOUD_SCATTERING_OCTAVES; octave++) {
//      for (int octave = 0; octave < SKY_CLOUD_SCATTERING_OCTAVES; octave++) {
            octavePhase[octave] = skyCloudPhase(cosSunAngle, pow(0.5, float(octave)));
//          octavePhase[octave] = skyCloudPhase(cosSunAngle, pow(0.5, float(octave)));
        }
//      }
        // Isotropic ambient from the sky dome: brighter towards the tops, which see more of it.
//      // Isotropic ambient from the sky dome: brighter towards the tops, which see more of it.
        // The dome is averaged towards its horizon haze, which dominates the irradiance a cloud receives.
//      // The dome is averaged towards its horizon haze, which dominates the irradiance a cloud receives.
        float3 skyDomeRadiance = lerp(getProceduralSkyBase(float3(0.0, 1.0, 0.0)), getProceduralSkyBase(normalize(float3(1.0, 0.15, 0.0))), 0.5);
//      float3 skyDomeRadiance = lerp(getProceduralSkyBase(float3(0.0, 1.0, 0.0)), getProceduralSkyBase(normalize(float3(1.0, 0.15, 0.0))), 0.5);
        float3 ambientTop = skyDomeRadiance * 1.4;
//      float3 ambientTop = skyDomeRadiance * 1.4;
        float3 ambientBottom = skyDomeRadiance * 0.5;
//      float3 ambientBottom = skyDomeRadiance * 0.5;

        float3 inScattered = float3(0.0);
//      float3 inScattered = float3(0.0);
        float transmittance = 1.0;
//      float transmittance = 1.0;
        float firstHitDistance = -1.0;
//      float firstHitDistance = -1.0;
        for (int stepIndex = 0; stepIndex < stepCount; stepIndex++) {
//      for (int stepIndex = 0; stepIndex < stepCount; stepIndex++) {
            float sampleDistance = marchStart + (float(stepIndex) + stepJitter) * stepLength;
//          float sampleDistance = marchStart + (float(stepIndex) + stepJitter) * stepLength;
            float3 samplePosition = rayDirection * sampleDistance;
//          float3 samplePosition = rayDirection * sampleDistance;
            float heightFraction = (length(samplePosition + planetCentreToObserver) - SKY_PLANET_RADIUS - SKY_CLOUD_LAYER_BOTTOM) / (SKY_CLOUD_LAYER_TOP - SKY_CLOUD_LAYER_BOTTOM);
//          float heightFraction = (length(samplePosition + planetCentreToObserver) - SKY_PLANET_RADIUS - SKY_CLOUD_LAYER_BOTTOM) / (SKY_CLOUD_LAYER_TOP - SKY_CLOUD_LAYER_BOTTOM);
            float density = skyCloudDensity(samplePosition, heightFraction, true);
//          float density = skyCloudDensity(samplePosition, heightFraction, true);
            if (density <= 0.0) {
//          if (density <= 0.0) {
                continue;
//              continue;
            }
//          }
            if (firstHitDistance < 0.0) {
//          if (firstHitDistance < 0.0) {
                firstHitDistance = sampleDistance;
//              firstHitDistance = sampleDistance;
            }
//          }
            float extinction = density * uSkyCloudDensity;
//          float extinction = density * uSkyCloudDensity;

            // Light march towards the sun with geometrically growing steps.
//          // Light march towards the sun with geometrically growing steps.
            float sunOpticalDepth = 0.0;
//          float sunOpticalDepth = 0.0;
            float lightStepLength = SKY_CLOUD_LIGHT_FIRST_STEP;
//          float lightStepLength = SKY_CLOUD_LIGHT_FIRST_STEP;
            float lightDistance = 0.0;
//          float lightDistance = 0.0;
            for (int lightStep = 0; lightStep < SKY_CLOUD_LIGHT_STEPS; lightStep++) {
//          for (int lightStep = 0; lightStep < SKY_CLOUD_LIGHT_STEPS; lightStep++) {
                float3 lightSamplePosition = samplePosition + SKY_SUN_DIRECTION * (lightDistance + 0.5 * lightStepLength);
//              float3 lightSamplePosition = samplePosition + SKY_SUN_DIRECTION * (lightDistance + 0.5 * lightStepLength);
                float lightHeightFraction = (length(lightSamplePosition + planetCentreToObserver) - SKY_PLANET_RADIUS - SKY_CLOUD_LAYER_BOTTOM) / (SKY_CLOUD_LAYER_TOP - SKY_CLOUD_LAYER_BOTTOM);
//              float lightHeightFraction = (length(lightSamplePosition + planetCentreToObserver) - SKY_PLANET_RADIUS - SKY_CLOUD_LAYER_BOTTOM) / (SKY_CLOUD_LAYER_TOP - SKY_CLOUD_LAYER_BOTTOM);
                sunOpticalDepth += skyCloudDensity(lightSamplePosition, lightHeightFraction, false) * lightStepLength;
//              sunOpticalDepth += skyCloudDensity(lightSamplePosition, lightHeightFraction, false) * lightStepLength;
                lightDistance += lightStepLength;
//              lightDistance += lightStepLength;
                lightStepLength *= 2.0;
//              lightStepLength *= 2.0;
            }
//          }
            sunOpticalDepth *= uSkyCloudDensity;
//          sunOpticalDepth *= uSkyCloudDensity;

            // Multiple-scattering approximation: each octave halves scattering, extinction and anisotropy.
//          // Multiple-scattering approximation: each octave halves scattering, extinction and anisotropy.
            float3 sunScattering = float3(0.0);
//          float3 sunScattering = float3(0.0);
            float octaveWeight = 1.0;
//          float octaveWeight = 1.0;
            for (int octave = 0; octave < SKY_CLOUD_SCATTERING_OCTAVES; octave++) {
//          for (int octave = 0; octave < SKY_CLOUD_SCATTERING_OCTAVES; octave++) {
                sunScattering += octaveWeight * octavePhase[octave] * exp(-octaveWeight * sunOpticalDepth);
//              sunScattering += octaveWeight * octavePhase[octave] * exp(-octaveWeight * sunOpticalDepth);
                octaveWeight *= 0.5;
//              octaveWeight *= 0.5;
            }
//          }
            // Powder term: darkens the sun-facing rims of thin, freshly entered cloud.
//          // Powder term: darkens the sun-facing rims of thin, freshly entered cloud.
            float powder = 1.0 - 0.35 * exp(-extinction * 2.0 * stepLength);
//          float powder = 1.0 - 0.35 * exp(-extinction * 2.0 * stepLength);
            float3 ambient = lerp(ambientBottom, ambientTop, clamp(heightFraction, 0.0, 1.0));
//          float3 ambient = lerp(ambientBottom, ambientTop, clamp(heightFraction, 0.0, 1.0));
            float3 sourceRadiance = SKY_CLOUD_SUN_RADIANCE * sunScattering * powder + ambient;
//          float3 sourceRadiance = SKY_CLOUD_SUN_RADIANCE * sunScattering * powder + ambient;

            // Energy-conserving integration of a constant source over the step (Hillaire 2016).
//          // Energy-conserving integration of a constant source over the step (Hillaire 2016).
            float stepTransmittance = exp(-extinction * stepLength);
//          float stepTransmittance = exp(-extinction * stepLength);
            inScattered += transmittance * sourceRadiance * (1.0 - stepTransmittance);
//          inScattered += transmittance * sourceRadiance * (1.0 - stepTransmittance);
            transmittance *= stepTransmittance;
//          transmittance *= stepTransmittance;
            if (transmittance < SKY_CLOUD_TRANSMITTANCE_CUTOFF) {
//          if (transmittance < SKY_CLOUD_TRANSMITTANCE_CUTOFF) {
                break;
//              break;
            }
//          }
        }
//      }
        if (firstHitDistance < 0.0) {
//      if (firstHitDistance < 0.0) {
            return float4(0.0, 0.0, 0.0, 1.0);
//          return float4(0.0, 0.0, 0.0, 1.0);
        }
//      }
        // Aerial perspective: distant clouds dissolve into the sky behind them rather than going black,
//      // Aerial perspective: distant clouds dissolve into the sky behind them rather than going black,
        // so the faded fraction is handed back to the sky as transmittance.
//      // so the faded fraction is handed back to the sky as transmittance.
        float visibility = exp(-firstHitDistance / SKY_CLOUD_FADE_DISTANCE);
//      float visibility = exp(-firstHitDistance / SKY_CLOUD_FADE_DISTANCE);
        // Fade in just above the horizon so the layer does not end in a hard line.
//      // Fade in just above the horizon so the layer does not end in a hard line.
        visibility *= clamp(rayDirection.y * 12.0, 0.0, 1.0);
//      visibility *= clamp(rayDirection.y * 12.0, 0.0, 1.0);
        return float4(inScattered * visibility, lerp(1.0, transmittance, visibility));
//      return float4(inScattered * visibility, lerp(1.0, transmittance, visibility));
    }
//  }

    // Rays that see the sky directly -- camera rays and pure mirror/glass chains -- march the layer live
//  // Rays that see the sky directly -- camera rays and pure mirror/glass chains -- march the layer live
    // at full resolution; the per-frame jitter is resolved by temporal accumulation. Every other lookup
//  // at full resolution; the per-frame jitter is resolved by temporal accumulation. Every other lookup
    // goes through getSkyColor and the baked texture, whose filtering is invisible in indirect light.
//  // goes through getSkyColor and the baked texture, whose filtering is invisible in indirect light.
    public float3 getSkyColorRayMarched(float3 rayDirection, float stepJitter) {
//  public float3 getSkyColorRayMarched(float3 rayDirection, float stepJitter) {
        if ((uUseHdri != 0) || (uSkyCloudsEnabled == 0)) {
//      if ((uUseHdri != 0) || (uSkyCloudsEnabled == 0)) {
            return getSkyColor(rayDirection);
//          return getSkyColor(rayDirection);
        }
//      }
        return composeSkyWithClouds(rayDirection, marchSkyClouds(rayDirection, SKY_CLOUD_PRIMARY_STEPS, stepJitter));
//      return composeSkyWithClouds(rayDirection, marchSkyClouds(rayDirection, SKY_CLOUD_PRIMARY_STEPS, stepJitter));
    }
//  }
