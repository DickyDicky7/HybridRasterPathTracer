    // ============================================================================
//  // ============================================================================
    // Shared environment lighting.
//  // Shared environment lighting.
    //
//  //
    // Extracted verbatim out of hybrid_shading_cs.slang so that nrc_gather_cs.slang can evaluate the
//  // Extracted verbatim out of hybrid_shading_cs.slang so that nrc_gather_cs.slang can evaluate the
    // same environment radiance the shading pass gathers. Keeping one copy is the point: the Neural
//  // same environment radiance the shading pass gathers. Keeping one copy is the point: the Neural
    // Radiance Cache regresses total outgoing radiance, so a training pass that cannot see the sky
//  // Radiance Cache regresses total outgoing radiance, so a training pass that cannot see the sky
    // teaches the network to omit environment light, and every path the cache terminates then loses
//  // teaches the network to omit environment light, and every path the cache terminates then loses
    // that energy with nothing in the image to show where it went.
//  // that energy with nothing in the image to show where it went.
    //
//  //
    // The procedural sky is split into its atmosphere (getProceduralSkyBase) and its sun
//  // The procedural sky is split into its atmosphere (getProceduralSkyBase) and its sun
    // (getProceduralSun) so the volumetric cloud layer can sit between them and the viewer. The cloud
//  // (getProceduralSun) so the volumetric cloud layer can sit between them and the viewer. The cloud
    // layer is pre-integrated into uSkyCloudTexture by sky_clouds_bake_cs.slang: rgb holds the cloud
//  // layer is pre-integrated into uSkyCloudTexture by sky_clouds_bake_cs.slang: rgb holds the cloud
    // in-scattering, a holds the transmittance through the layer. Composing the two here keeps every
//  // in-scattering, a holds the transmittance through the layer. Composing the two here keeps every
    // caller of getSkyColor -- secondary bounces and NRC training alike -- lit by the same cloudy sky.
//  // caller of getSkyColor -- secondary bounces and NRC training alike -- lit by the same cloudy sky.
    //
//  //
    // Owns the environment inputs shared by every importer: the HDRI and the baked cloud layer.
//  // Owns the environment inputs shared by every importer: the HDRI and the baked cloud layer.
    // ============================================================================
//  // ============================================================================
    module sky_environment;
//  module sky_environment;

    import math_constants;
//  import math_constants;

    public uniform int uUseHdri;
//  public uniform int uUseHdri;
    public Sampler2D<float4> uHdriTexture;
//  public Sampler2D<float4> uHdriTexture;
    // Volumetric sky cloud layer, pre-integrated into an equirectangular texture by sky_clouds_bake_cs.slang.
//  // Volumetric sky cloud layer, pre-integrated into an equirectangular texture by sky_clouds_bake_cs.slang.
    public uniform int uSkyCloudsEnabled;
//  public uniform int uSkyCloudsEnabled;
    public Sampler2D<float4> uSkyCloudTexture;
//  public Sampler2D<float4> uSkyCloudTexture;
    // Upper bound on HDRI radiance, suppressing fireflies seeded by a few extreme-intensity texels.
//  // Upper bound on HDRI radiance, suppressing fireflies seeded by a few extreme-intensity texels.
    public static const float HDRI_CLAMP = 20.0;
//  public static const float HDRI_CLAMP = 20.0;

    public static const float3 SKY_SUN_DIRECTION = float3(0.0, 0.70710678, 0.70710678);
//  public static const float3 SKY_SUN_DIRECTION = float3(0.0, 0.70710678, 0.70710678);

    // Equirectangular mapping shared by the HDRI lookup and the baked cloud layer.
//  // Equirectangular mapping shared by the HDRI lookup and the baked cloud layer.
    public float2 equirectUvFromDirection(float3 rayDirection) {
//  public float2 equirectUvFromDirection(float3 rayDirection) {
        float latitudeAngle = acos(clamp(-rayDirection.y, -1.0, 1.0)); // latitude: 0 at the nadir, PI at the zenith
//      float latitudeAngle = acos(clamp(-rayDirection.y, -1.0, 1.0)); // latitude: 0 at the nadir, PI at the zenith
        float longitudeAngle = atan2(-rayDirection.z, rayDirection.x) + PI; // longitude: 0 to 2*PI
//      float longitudeAngle = atan2(-rayDirection.z, rayDirection.x) + PI; // longitude: 0 to 2*PI
        float equirectU = clamp(longitudeAngle / (2.0 * PI), 0.0, 1.0);
//      float equirectU = clamp(longitudeAngle / (2.0 * PI), 0.0, 1.0);
        float equirectV = clamp(latitudeAngle / PI, 0.0, 1.0);
//      float equirectV = clamp(latitudeAngle / PI, 0.0, 1.0);
        return float2(equirectU, equirectV);
//      return float2(equirectU, equirectV);
    }
//  }

    // Exact inverse of equirectUvFromDirection, used to aim each texel of the baked cloud layer.
//  // Exact inverse of equirectUvFromDirection, used to aim each texel of the baked cloud layer.
    public float3 directionFromEquirectUv(float2 equirectUv) {
//  public float3 directionFromEquirectUv(float2 equirectUv) {
        float latitudeAngle = equirectUv.y * PI;
//      float latitudeAngle = equirectUv.y * PI;
        float longitudeAngle = equirectUv.x * 2.0 * PI - PI;
//      float longitudeAngle = equirectUv.x * 2.0 * PI - PI;
        float sinLatitude = sin(latitudeAngle);
//      float sinLatitude = sin(latitudeAngle);
        return float3(cos(longitudeAngle) * sinLatitude, -cos(latitudeAngle), -sin(longitudeAngle) * sinLatitude);
//      return float3(cos(longitudeAngle) * sinLatitude, -cos(latitudeAngle), -sin(longitudeAngle) * sinLatitude);
    }
//  }

    public float3 getProceduralSkyBase(float3 rayDirection) {
//  public float3 getProceduralSkyBase(float3 rayDirection) {
        /*
        // Fallback: procedural gradient sky
//      // Fallback: procedural gradient sky
        float verticalBlend = 0.5 * (rayDirection.y + 1.0);
//      float verticalBlend = 0.5 * (rayDirection.y + 1.0);
        return lerp(float3(0.1), float3(0.5, 0.7, 1.0), verticalBlend);
//      return lerp(float3(0.1), float3(0.5, 0.7, 1.0), verticalBlend);
        */
        // Fallback: procedural atmospheric sky
//      // Fallback: procedural atmospheric sky
        // Rayleigh Gradient: Approximates scattering of blue wavelengths, deeper at zenith
//      // Rayleigh Gradient: Approximates scattering of blue wavelengths, deeper at zenith
        float3 skyColor = float3(0.2, 0.45, 0.9) - rayDirection.y * 0.25 * float3(1.0, 0.5, 1.2) + 0.1 * float3(1.0);
//      float3 skyColor = float3(0.2, 0.45, 0.9) - rayDirection.y * 0.25 * float3(1.0, 0.5, 1.2) + 0.1 * float3(1.0);
        // Mie Scattering: Exponential horizon haze due to higher atmospheric density
//      // Mie Scattering: Exponential horizon haze due to higher atmospheric density
        skyColor = lerp(skyColor, float3(0.9, 0.95, 1.0), exp(-15.0 * max(rayDirection.y, 0.0)));
//      skyColor = lerp(skyColor, float3(0.9, 0.95, 1.0), exp(-15.0 * max(rayDirection.y, 0.0)));
        // Cloud-like ground: Replace the bottom with a bright, vivid white-blue deck
//      // Cloud-like ground: Replace the bottom with a bright, vivid white-blue deck
        if (rayDirection.y < 0.0) skyColor = lerp(float3(0.9, 0.95, 1.0), float3(0.98, 0.99, 1.0), pow(abs(rayDirection.y), 0.5));
//      if (rayDirection.y < 0.0) skyColor = lerp(float3(0.9, 0.95, 1.0), float3(0.98, 0.99, 1.0), pow(abs(rayDirection.y), 0.5));
        return skyColor;
//      return skyColor;
    }
//  }

    public float3 getProceduralSun(float3 rayDirection) {
//  public float3 getProceduralSun(float3 rayDirection) {
        // Sun Disk: Layered glows using increasing powers of the dot product (cosine of angle)
//      // Sun Disk: Layered glows using increasing powers of the dot product (cosine of angle)
        // Enhance sun intensity and direction for a more dramatic sky
//      // Enhance sun intensity and direction for a more dramatic sky
        float3 sunDirection = SKY_SUN_DIRECTION;
//      float3 sunDirection = SKY_SUN_DIRECTION;
        float sunCosine = clamp(dot(rayDirection, sunDirection), 0.0, 1.0);
//      float sunCosine = clamp(dot(rayDirection, sunDirection), 0.0, 1.0);
        float sunCosinePow2 = sunCosine * sunCosine;
//      float sunCosinePow2 = sunCosine * sunCosine;
        float sunCosinePow4 = sunCosinePow2 * sunCosinePow2;
//      float sunCosinePow4 = sunCosinePow2 * sunCosinePow2;
        float sunCosinePow8 = sunCosinePow4 * sunCosinePow4;
//      float sunCosinePow8 = sunCosinePow4 * sunCosinePow4;
        float sunCosinePow16 = sunCosinePow8 * sunCosinePow8;
//      float sunCosinePow16 = sunCosinePow8 * sunCosinePow8;
        float sunCosinePow32 = sunCosinePow16 * sunCosinePow16;
//      float sunCosinePow32 = sunCosinePow16 * sunCosinePow16;
        float sunCosinePow64 = sunCosinePow32 * sunCosinePow32;
//      float sunCosinePow64 = sunCosinePow32 * sunCosinePow32;
        float sunCosinePow128 = sunCosinePow64 * sunCosinePow64;
//      float sunCosinePow128 = sunCosinePow64 * sunCosinePow64;
        float sunCosinePow256 = sunCosinePow128 * sunCosinePow128;
//      float sunCosinePow256 = sunCosinePow128 * sunCosinePow128;
        float sunCosinePow512 = sunCosinePow256 * sunCosinePow256;
//      float sunCosinePow512 = sunCosinePow256 * sunCosinePow256;
        float3 sunColor = float3(0.0);
//      float3 sunColor = float3(0.0);
        sunColor += 0.4 * float3(10.0, 10.6, 10.3) * sunCosinePow8; // Wide soft orange glow
//      sunColor += 0.4 * float3(10.0, 10.6, 10.3) * sunCosinePow8; // Wide soft orange glow
        sunColor += 0.3 * float3(10.0, 10.8, 10.5) * sunCosinePow64; // Bright golden core
//      sunColor += 0.3 * float3(10.0, 10.8, 10.5) * sunCosinePow64; // Bright golden core
        sunColor += 0.5 * float3(10.0, 10.0, 10.0) * sunCosinePow512; // Intense white disk
//      sunColor += 0.5 * float3(10.0, 10.0, 10.0) * sunCosinePow512; // Intense white disk
        return sunColor;
//      return sunColor;
    }
//  }

    // Composites a cloud layer sample (rgb: in-scattered radiance, a: transmittance) over the sky.
//  // Composites a cloud layer sample (rgb: in-scattered radiance, a: transmittance) over the sky.
    public float3 composeSkyWithClouds(float3 rayDirection, float4 cloudLayer) {
//  public float3 composeSkyWithClouds(float3 rayDirection, float4 cloudLayer) {
        return (getProceduralSkyBase(rayDirection) + getProceduralSun(rayDirection)) * cloudLayer.a + cloudLayer.rgb;
//      return (getProceduralSkyBase(rayDirection) + getProceduralSun(rayDirection)) * cloudLayer.a + cloudLayer.rgb;
    }
//  }

    public float3 getSkyColor(float3 rayDirection) {
//  public float3 getSkyColor(float3 rayDirection) {
        if (uUseHdri != 0) {
//      if (uUseHdri != 0) {
            // Equirectangular mapping
//          // Equirectangular mapping
            float3 environmentColor = uHdriTexture.SampleLevel(equirectUvFromDirection(rayDirection), 0.0).rgb;
//          float3 environmentColor = uHdriTexture.SampleLevel(equirectUvFromDirection(rayDirection), 0.0).rgb;
            return min(environmentColor, float3(HDRI_CLAMP));
//          return min(environmentColor, float3(HDRI_CLAMP));
        }
//      }
        if (uSkyCloudsEnabled != 0) {
//      if (uSkyCloudsEnabled != 0) {
            return composeSkyWithClouds(rayDirection, uSkyCloudTexture.SampleLevel(equirectUvFromDirection(rayDirection), 0.0));
//          return composeSkyWithClouds(rayDirection, uSkyCloudTexture.SampleLevel(equirectUvFromDirection(rayDirection), 0.0));
        }
//      }
        return getProceduralSkyBase(rayDirection) + getProceduralSun(rayDirection);
//      return getProceduralSkyBase(rayDirection) + getProceduralSun(rayDirection);
    }
//  }
