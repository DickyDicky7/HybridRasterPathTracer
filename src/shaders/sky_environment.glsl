    // ============================================================================
//  // ============================================================================
    // Shared environment lighting.
//  // Shared environment lighting.
    //
//  //
    // Extracted verbatim out of hybrid_shading_cs.glsl so that nrc_gather_cs.glsl can evaluate the
//  // Extracted verbatim out of hybrid_shading_cs.glsl so that nrc_gather_cs.glsl can evaluate the
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
    // Requires the includer to have already defined: PI, HDRI_CLAMP, uUseHdri, uHdriTexture.
//  // Requires the includer to have already defined: PI, HDRI_CLAMP, uUseHdri, uHdriTexture.
    // ============================================================================
//  // ============================================================================
    vec3 getSkyColor(vec3 rayDirection) {
//  vec3 getSkyColor(vec3 rayDirection) {
        if (uUseHdri) {
//      if (uUseHdri) {
            // Equirectangular mapping
//          // Equirectangular mapping
            float latitudeAngle = acos(clamp(-rayDirection.y, -1.0, 1.0)); // latitude: 0 at top, PI at bottom
//          float latitudeAngle = acos(clamp(-rayDirection.y, -1.0, 1.0)); // latitude: 0 at top, PI at bottom
            float longitudeAngle = atan(-rayDirection.z, rayDirection.x) + PI; // longitude: 0 to 2*PI
//          float longitudeAngle = atan(-rayDirection.z, rayDirection.x) + PI; // longitude: 0 to 2*PI
            float equirectU = clamp(longitudeAngle / (2.0 * PI), 0.0, 1.0);
//          float equirectU = clamp(longitudeAngle / (2.0 * PI), 0.0, 1.0);
            float equirectV = clamp(latitudeAngle / PI, 0.0, 1.0);
//          float equirectV = clamp(latitudeAngle / PI, 0.0, 1.0);
            vec3 environmentColor = textureLod(uHdriTexture, vec2(equirectU, equirectV), 0.0).rgb;
//          vec3 environmentColor = textureLod(uHdriTexture, vec2(equirectU, equirectV), 0.0).rgb;
            return min(environmentColor, vec3(HDRI_CLAMP));
//          return min(environmentColor, vec3(HDRI_CLAMP));
        }
//      }
        /*
        // Fallback: procedural gradient sky
//      // Fallback: procedural gradient sky
        float verticalBlend = 0.5 * (rayDirection.y + 1.0);
//      float verticalBlend = 0.5 * (rayDirection.y + 1.0);
        return mix(vec3(0.1), vec3(0.5, 0.7, 1.0), verticalBlend);
//      return mix(vec3(0.1), vec3(0.5, 0.7, 1.0), verticalBlend);
        */
        // Fallback: procedural atmospheric sky
//      // Fallback: procedural atmospheric sky
        // Rayleigh Gradient: Approximates scattering of blue wavelengths, deeper at zenith
//      // Rayleigh Gradient: Approximates scattering of blue wavelengths, deeper at zenith
        vec3 skyColor = vec3(0.2, 0.45, 0.9) - rayDirection.y * 0.25 * vec3(1.0, 0.5, 1.2) + 0.1 * vec3(1.0);
//      vec3 skyColor = vec3(0.2, 0.45, 0.9) - rayDirection.y * 0.25 * vec3(1.0, 0.5, 1.2) + 0.1 * vec3(1.0);
        // Mie Scattering: Exponential horizon haze due to higher atmospheric density
//      // Mie Scattering: Exponential horizon haze due to higher atmospheric density
        skyColor = mix(skyColor, vec3(0.9, 0.95, 1.0), exp(-15.0 * max(rayDirection.y, 0.0)));
//      skyColor = mix(skyColor, vec3(0.9, 0.95, 1.0), exp(-15.0 * max(rayDirection.y, 0.0)));
        // Cloud-like ground: Replace the bottom with a bright, vivid white-blue deck
//      // Cloud-like ground: Replace the bottom with a bright, vivid white-blue deck
        if (rayDirection.y < 0.0) skyColor = mix(vec3(0.9, 0.95, 1.0), vec3(0.98, 0.99, 1.0), pow(abs(rayDirection.y), 0.5));
//      if (rayDirection.y < 0.0) skyColor = mix(vec3(0.9, 0.95, 1.0), vec3(0.98, 0.99, 1.0), pow(abs(rayDirection.y), 0.5));
        // Sun Disk: Layered glows using increasing powers of the dot product (cosine of angle)
//      // Sun Disk: Layered glows using increasing powers of the dot product (cosine of angle)
        // Enhance sun intensity and direction for a more dramatic sky
//      // Enhance sun intensity and direction for a more dramatic sky
        vec3 sunDirection = normalize(vec3(0.0, 0.5, 0.5));
//      vec3 sunDirection = normalize(vec3(0.0, 0.5, 0.5));
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
        skyColor += 0.4 * vec3(10.0, 10.6, 10.3) * sunCosinePow8; // Wide soft orange glow
//      skyColor += 0.4 * vec3(10.0, 10.6, 10.3) * sunCosinePow8; // Wide soft orange glow
        skyColor += 0.3 * vec3(10.0, 10.8, 10.5) * sunCosinePow64; // Bright golden core
//      skyColor += 0.3 * vec3(10.0, 10.8, 10.5) * sunCosinePow64; // Bright golden core
        skyColor += 0.5 * vec3(10.0, 10.0, 10.0) * sunCosinePow512; // Intense white disk
//      skyColor += 0.5 * vec3(10.0, 10.0, 10.0) * sunCosinePow512; // Intense white disk
        return skyColor;
//      return skyColor;
    }
//  }
