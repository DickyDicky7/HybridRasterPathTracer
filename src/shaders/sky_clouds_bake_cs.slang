    // ============================================================================
//  // ============================================================================
    // Sky cloud layer bake.
//  // Sky cloud layer bake.
    //
//  //
    // Pre-integrates the ray-marched cloud layer (sky_clouds.slang) into an equirectangular texture
//  // Pre-integrates the ray-marched cloud layer (sky_clouds.slang) into an equirectangular texture
    // using the same mapping as the HDRI lookup: rgb = in-scattered cloud radiance, a = transmittance.
//  // using the same mapping as the HDRI lookup: rgb = in-scattered cloud radiance, a = transmittance.
    // getSkyColor composites it over the procedural sky, so every secondary bounce and every NRC
//  // getSkyColor composites it over the procedural sky, so every secondary bounce and every NRC
    // training path sees the clouds for the price of one texture fetch instead of a full march.
//  // training path sees the clouds for the price of one texture fetch instead of a full march.
    // The clouds are static, so this runs once per scene load or cloud toggle, not per frame.
//  // The clouds are static, so this runs once per scene load or cloud toggle, not per frame.
    // ============================================================================
//  // ============================================================================

    [format("rgba16f")] WTexture2D<float4> textureSkyClouds;
//  [format("rgba16f")] WTexture2D<float4> textureSkyClouds;

    // The cloud noise volume and the coverage / density parameters come with the imported modules (the
//  // The cloud noise volume and the coverage / density parameters come with the imported modules (the
    // HDRI and baked-layer parameters of sky_environment are declared there too, though the bake never reads them).
//  // HDRI and baked-layer parameters of sky_environment are declared there too, though the bake never reads them).
    import math_constants;
//  import math_constants;
    import sky_environment;
//  import sky_environment;
    import volumetric_common;
//  import volumetric_common;
    import sky_clouds;
//  import sky_clouds;

    // Interleaved gradient noise (Jimenez 2014): decorrelates neighbouring texels' march offsets so
//  // Interleaved gradient noise (Jimenez 2014): decorrelates neighbouring texels' march offsets so
    // bilinear filtering averages the step pattern away instead of freezing it into bands.
//  // bilinear filtering averages the step pattern away instead of freezing it into bands.
    float interleavedGradientNoise(float2 pixel) {
//  float interleavedGradientNoise(float2 pixel) {
        return frac(52.9829189 * frac(dot(pixel, float2(0.06711056, 0.00583715))));
//      return frac(52.9829189 * frac(dot(pixel, float2(0.06711056, 0.00583715))));
    }
//  }

    [shader("compute")]
//  [shader("compute")]
    [numthreads(16, 16, 1)]
//  [numthreads(16, 16, 1)]
    void computeMain(uint3 globalInvocationID : SV_DispatchThreadID, uint localInvocationIndex : SV_GroupIndex) {
//  void computeMain(uint3 globalInvocationID : SV_DispatchThreadID, uint localInvocationIndex : SV_GroupIndex) {
        int2 texelCoordinates = int2(globalInvocationID.xy);
//      int2 texelCoordinates = int2(globalInvocationID.xy);
        uint2 textureSizeUnsigned;
//      uint2 textureSizeUnsigned;
        textureSkyClouds.GetDimensions(textureSizeUnsigned.x, textureSizeUnsigned.y);
//      textureSkyClouds.GetDimensions(textureSizeUnsigned.x, textureSizeUnsigned.y);
        int2 textureSize = int2(textureSizeUnsigned);
//      int2 textureSize = int2(textureSizeUnsigned);
        if (texelCoordinates.x >= textureSize.x || texelCoordinates.y >= textureSize.y) {
//      if (texelCoordinates.x >= textureSize.x || texelCoordinates.y >= textureSize.y) {
            return;
//          return;
        }
//      }
        float2 equirectUv = (float2(texelCoordinates) + 0.5) / float2(textureSize);
//      float2 equirectUv = (float2(texelCoordinates) + 0.5) / float2(textureSize);
        float3 rayDirection = directionFromEquirectUv(equirectUv);
//      float3 rayDirection = directionFromEquirectUv(equirectUv);
        float4 cloudLayer = marchSkyClouds(rayDirection, SKY_CLOUD_BAKE_STEPS, interleavedGradientNoise(float2(texelCoordinates)));
//      float4 cloudLayer = marchSkyClouds(rayDirection, SKY_CLOUD_BAKE_STEPS, interleavedGradientNoise(float2(texelCoordinates)));
        textureSkyClouds.Store(texelCoordinates, cloudLayer);
//      textureSkyClouds.Store(texelCoordinates, cloudLayer);
    }
//  }
