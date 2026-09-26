    #version 430
//  #version 430

    // ============================================================================
//  // ============================================================================
    // Sky cloud layer bake.
//  // Sky cloud layer bake.
    //
//  //
    // Pre-integrates the ray-marched cloud layer (sky_clouds.glsl) into an equirectangular texture
//  // Pre-integrates the ray-marched cloud layer (sky_clouds.glsl) into an equirectangular texture
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
    layout(local_size_x = 16, local_size_y = 16) in;
//  layout(local_size_x = 16, local_size_y = 16) in;

    layout(binding = 0, rgba16f) uniform writeonly image2D textureSkyClouds;
//  layout(binding = 0, rgba16f) uniform writeonly image2D textureSkyClouds;

    // getSkyColor (compiled in with sky_environment.glsl but unused here) needs these declared.
//  // getSkyColor (compiled in with sky_environment.glsl but unused here) needs these declared.
    uniform bool uUseHdri;
//  uniform bool uUseHdri;
    uniform sampler2D uHdriTexture;
//  uniform sampler2D uHdriTexture;
    uniform bool uSkyCloudsEnabled;
//  uniform bool uSkyCloudsEnabled;
    uniform sampler2D uSkyCloudTexture;
//  uniform sampler2D uSkyCloudTexture;

    uniform sampler3D uCloudNoiseTexture;
//  uniform sampler3D uCloudNoiseTexture;
    uniform float uSkyCloudCoverage;
//  uniform float uSkyCloudCoverage;
    uniform float uSkyCloudDensity;
//  uniform float uSkyCloudDensity;

    const float PI = 3.14159265359;
//  const float PI = 3.14159265359;
    const float HDRI_CLAMP = 20.0;
//  const float HDRI_CLAMP = 20.0;

    #include "sky_environment.glsl"

    #include "volumetric_common.glsl"

    #include "sky_clouds.glsl"

    // Interleaved gradient noise (Jimenez 2014): decorrelates neighbouring texels' march offsets so
//  // Interleaved gradient noise (Jimenez 2014): decorrelates neighbouring texels' march offsets so
    // bilinear filtering averages the step pattern away instead of freezing it into bands.
//  // bilinear filtering averages the step pattern away instead of freezing it into bands.
    float interleavedGradientNoise(vec2 pixel) {
//  float interleavedGradientNoise(vec2 pixel) {
        return fract(52.9829189 * fract(dot(pixel, vec2(0.06711056, 0.00583715))));
//      return fract(52.9829189 * fract(dot(pixel, vec2(0.06711056, 0.00583715))));
    }
//  }

    void main() {
//  void main() {
        ivec2 texelCoordinates = ivec2(gl_GlobalInvocationID.xy);
//      ivec2 texelCoordinates = ivec2(gl_GlobalInvocationID.xy);
        ivec2 textureSize = imageSize(textureSkyClouds);
//      ivec2 textureSize = imageSize(textureSkyClouds);
        if (texelCoordinates.x >= textureSize.x || texelCoordinates.y >= textureSize.y) {
//      if (texelCoordinates.x >= textureSize.x || texelCoordinates.y >= textureSize.y) {
            return;
//          return;
        }
//      }
        vec2 equirectUv = (vec2(texelCoordinates) + 0.5) / vec2(textureSize);
//      vec2 equirectUv = (vec2(texelCoordinates) + 0.5) / vec2(textureSize);
        vec3 rayDirection = directionFromEquirectUv(equirectUv);
//      vec3 rayDirection = directionFromEquirectUv(equirectUv);
        vec4 cloudLayer = marchSkyClouds(rayDirection, SKY_CLOUD_BAKE_STEPS, interleavedGradientNoise(vec2(texelCoordinates)));
//      vec4 cloudLayer = marchSkyClouds(rayDirection, SKY_CLOUD_BAKE_STEPS, interleavedGradientNoise(vec2(texelCoordinates)));
        imageStore(textureSkyClouds, texelCoordinates, cloudLayer);
//      imageStore(textureSkyClouds, texelCoordinates, cloudLayer);
    }
//  }
