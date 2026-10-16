    // Shared modules. Environment Lighting (sky_environment): evaluates the infinitely distant background radiance arriving along a ray's direction, serving as both the visible backdrop and the scene's image-based light source. When an HDR map is bound, the direction is projected into equirectangular (latitude-longitude) UVs and sampled, with the result clamped to suppress fireflies seeded by a few extreme-intensity texels. Otherwise it synthesizes a sky analytically — a Rayleigh-style zenith-to-horizon gradient, an exponential Mie horizon haze, and layered power-of-cosine glows that build a soft-to-sharp sun disk. Shared with nrc_gather_cs.slang so the Neural Radiance Cache trains against the same sky this pass gathers.
//  // Shared modules. Environment Lighting (sky_environment): evaluates the infinitely distant background radiance arriving along a ray's direction, serving as both the visible backdrop and the scene's image-based light source. When an HDR map is bound, the direction is projected into equirectangular (latitude-longitude) UVs and sampled, with the result clamped to suppress fireflies seeded by a few extreme-intensity texels. Otherwise it synthesizes a sky analytically — a Rayleigh-style zenith-to-horizon gradient, an exponential Mie horizon haze, and layered power-of-cosine glows that build a soft-to-sharp sun disk. Shared with nrc_gather_cs.slang so the Neural Radiance Cache trains against the same sky this pass gathers.
    // Volumetrics (volumetric_common, sky_clouds, participating_media): shared phase/noise helpers, the ray-marched sky cloud layer, and the in-scene participating media the path tracer scatters inside. Media sampling draws from this pass's RNG through ShadingRandom.
//  // Volumetrics (volumetric_common, sky_clouds, participating_media): shared phase/noise helpers, the ray-marched sky cloud layer, and the in-scene participating media the path tracer scatters inside. Media sampling draws from this pass's RNG through ShadingRandom.
    // Principled BSDF Components (principled_bsdf): the building blocks of a Disney-style physically-based surface model. These comprise Schlick's Fresnel approximation for view-dependent reflectivity, three interchangeable diffuse lobes (Lambert, Disney retro-reflective, Oren-Nayar), and the composed energy-conserving evaluation/PDF routine. Shared with nrc_gather_cs.slang so the Neural Radiance Cache trains against the same BSDF this pass shades with.
//  // Principled BSDF Components (principled_bsdf): the building blocks of a Disney-style physically-based surface model. These comprise Schlick's Fresnel approximation for view-dependent reflectivity, three interchangeable diffuse lobes (Lambert, Disney retro-reflective, Oren-Nayar), and the composed energy-conserving evaluation/PDF routine. Shared with nrc_gather_cs.slang so the Neural Radiance Cache trains against the same BSDF this pass shades with.
    import math_constants;
//  import math_constants;
    import sky_environment;
//  import sky_environment;
    import volumetric_common;
//  import volumetric_common;
    import sky_clouds;
//  import sky_clouds;
    import participating_media;
//  import participating_media;
    import principled_bsdf;
//  import principled_bsdf;

    Texture2D<float4> textureGeometryGlobalPosition;
//  Texture2D<float4> textureGeometryGlobalPosition;
    Texture2D<float4> textureGeometryGlobalNormal;
//  Texture2D<float4> textureGeometryGlobalNormal;
    Texture2D<float4> textureGeometryAlbedo;
//  Texture2D<float4> textureGeometryAlbedo;
    Texture2D<float4> textureGeometryGlobalTangent;
//  Texture2D<float4> textureGeometryGlobalTangent;
    // Progressive accumulation ping-pong pair: WebGPU only allows read-write storage access on single-channel 32-bit formats, so the running average is read from last frame's rgba32f target (sampled, integer texel loads) and blended into this frame's write-only target. The renderer swaps the pair after every dispatch.
//  // Progressive accumulation ping-pong pair: WebGPU only allows read-write storage access on single-channel 32-bit formats, so the running average is read from last frame's rgba32f target (sampled, integer texel loads) and blended into this frame's write-only target. The renderer swaps the pair after every dispatch.
    Texture2D<float4> textureAccumHistory;
//  Texture2D<float4> textureAccumHistory;
    [format("rgba32f")] WTexture2D<float4> textureAccum;
//  [format("rgba32f")] WTexture2D<float4> textureAccum;

    // Bounding Volume Hierarchy (BVH) acceleration structure: a binary tree of Axis-Aligned Bounding Boxes (AABBs) that recursively partitions scene primitives to rapidly cull empty space. Each 32-byte node packs its AABB extents into two vec4s, overloading the otherwise-unused w-channels to carry topology — inner nodes store left/right child indices, while leaf nodes encode a primitive triangle offset (signalled by a negative left-child sentinel). This compact, cache-coherent layout collapses ray-scene queries from O(n) brute force to an expected O(log n) walk of intersection tests.
//  // Bounding Volume Hierarchy (BVH) acceleration structure: a binary tree of Axis-Aligned Bounding Boxes (AABBs) that recursively partitions scene primitives to rapidly cull empty space. Each 32-byte node packs its AABB extents into two vec4s, overloading the otherwise-unused w-channels to carry topology — inner nodes store left/right child indices, while leaf nodes encode a primitive triangle offset (signalled by a negative left-child sentinel). This compact, cache-coherent layout collapses ray-scene queries from O(n) brute force to an expected O(log n) walk of intersection tests.
    struct Node {
//  struct Node {
        float4 aabbMinAndLeftChild; // xyz = AABB minimum corner; w = left-child node index, or a negative sentinel that marks this node as a leaf
//      float4 aabbMinAndLeftChild; // xyz = AABB minimum corner; w = left-child node index, or a negative sentinel that marks this node as a leaf
        float4 aabbMaxAndRightChild; // xyz = AABB maximum corner; w = right-child node index for inner nodes, reinterpreted as the primitive triangle index for leaves
//      float4 aabbMaxAndRightChild; // xyz = AABB maximum corner; w = right-child node index for inner nodes, reinterpreted as the primitive triangle index for leaves
    };
//  };

    StructuredBuffer<Node> nodes;
//  StructuredBuffer<Node> nodes;

    // Global Geometry Buffer: a single flat SSBO of interleaved per-vertex attributes (position, normal, tangent, UV) packed into three 16-byte-aligned vec4s for coalesced, cache-friendly GPU reads. Triangles are stored as contiguous vertex triplets addressed by triangleIndex * 3, and this layout is shared with the rasterizer's VBOs so the hybrid pipeline never duplicates geometry between its raster and ray-traced passes.
//  // Global Geometry Buffer: a single flat SSBO of interleaved per-vertex attributes (position, normal, tangent, UV) packed into three 16-byte-aligned vec4s for coalesced, cache-friendly GPU reads. Triangles are stored as contiguous vertex triplets addressed by triangleIndex * 3, and this layout is shared with the rasterizer's VBOs so the hybrid pipeline never duplicates geometry between its raster and ray-traced passes.
    struct VertexData {
//  struct VertexData {
        float4 positionAndTexcoordU; // xyz = world-space vertex position; w = texture coordinate U (packed to keep the vec4 fully utilized)
//      float4 positionAndTexcoordU; // xyz = world-space vertex position; w = texture coordinate U (packed to keep the vec4 fully utilized)
        float4 normalAndTexcoordV; // xyz = vertex shading normal; w = texture coordinate V
//      float4 normalAndTexcoordV; // xyz = vertex shading normal; w = texture coordinate V
        float4 tangentAndMaterialIndex; // xyz = vertex tangent (basis for normal mapping); w = float-encoded material index, read from the triangle's first vertex
//      float4 tangentAndMaterialIndex; // xyz = vertex tangent (basis for normal mapping); w = float-encoded material index, read from the triangle's first vertex
    };
//  };

    StructuredBuffer<VertexData> vertices;
//  StructuredBuffer<VertexData> vertices;

    // Material Properties Buffer: stores the Principled/PBR parameter set per material — base color/albedo, perceptual roughness, metallic mask, transmission and index-of-refraction (IOR) for dielectrics, plus emissive strength and a UV tiling scale. Each material also carries float-encoded layer indices into the shared Texture2DArray (a value below -0.5 acting as an "absent texture" sentinel), enabling spatially varying, texture-driven evaluation of every channel during shading.
//  // Material Properties Buffer: stores the Principled/PBR parameter set per material — base color/albedo, perceptual roughness, metallic mask, transmission and index-of-refraction (IOR) for dielectrics, plus emissive strength and a UV tiling scale. Each material also carries float-encoded layer indices into the shared Texture2DArray (a value below -0.5 acting as an "absent texture" sentinel), enabling spatially varying, texture-driven evaluation of every channel during shading.
    struct Material {
//  struct Material {
        float4 albedo;       // rgb = base color / diffuse reflectance (linear); w = unused padding for std430 16-byte alignment
//      float4 albedo;       // rgb = base color / diffuse reflectance (linear); w = unused padding for std430 16-byte alignment
        float roughness; // perceptual roughness in [0,1]; squared into the GGX alpha and clamped to MIN_ROUGHNESS during shading
//      float roughness; // perceptual roughness in [0,1]; squared into the GGX alpha and clamped to MIN_ROUGHNESS during shading
        float metallic; // metalness mask in [0,1]: 0 = dielectric (F0 = 0.04), 1 = conductor (F0 = albedo, diffuse suppressed)
//      float metallic; // metalness mask in [0,1]: 0 = dielectric (F0 = 0.04), 1 = conductor (F0 = albedo, diffuse suppressed)
        float transmission; // fraction of energy refracted through the surface; > 0 enables the glass/dielectric transmission lobe
//      float transmission; // fraction of energy refracted through the surface; > 0 enables the glass/dielectric transmission lobe
        float ior; // index of refraction for the transmission lobe (e.g. ~1.5 for glass), feeding the Snell/Fresnel terms
//      float ior; // index of refraction for the transmission lobe (e.g. ~1.5 for glass), feeding the Snell/Fresnel terms
        float textureIndexAlbedo; // Texture2DArray layer for the albedo map (sRGB-encoded, decoded to linear on read), or < -0.5 if absent
//      float textureIndexAlbedo; // Texture2DArray layer for the albedo map (sRGB-encoded, decoded to linear on read), or < -0.5 if absent
        float textureIndexRoughness; // Texture2DArray layer for roughness, sampled from the R channel of the packed ORM texture, or < -0.5 if absent
//      float textureIndexRoughness; // Texture2DArray layer for roughness, sampled from the R channel of the packed ORM texture, or < -0.5 if absent
        float textureIndexMetallic; // Texture2DArray layer for metallic, sampled from the G channel of the packed ORM texture, or < -0.5 if absent
//      float textureIndexMetallic; // Texture2DArray layer for metallic, sampled from the G channel of the packed ORM texture, or < -0.5 if absent
        float textureIndexNormal; // Texture2DArray layer for the tangent-space normal map, or < -0.5 if absent
//      float textureIndexNormal; // Texture2DArray layer for the tangent-space normal map, or < -0.5 if absent
        float emissive; // emissive intensity multiplier; scales the emissive texture, or the albedo when no emissive map is bound
//      float emissive; // emissive intensity multiplier; scales the emissive texture, or the albedo when no emissive map is bound
        float textureIndexEmissive; // Texture2DArray layer for the emissive map (sRGB-decoded), or < -0.5 if absent
//      float textureIndexEmissive; // Texture2DArray layer for the emissive map (sRGB-decoded), or < -0.5 if absent
        float textureIndexTransmission; // Texture2DArray layer for transmission, sampled from the B channel of the packed ORM texture, or < -0.5 if absent
//      float textureIndexTransmission; // Texture2DArray layer for transmission, sampled from the B channel of the packed ORM texture, or < -0.5 if absent
        float padding002; // explicit padding to satisfy std430 scalar/vec2 alignment rules
//      float padding002; // explicit padding to satisfy std430 scalar/vec2 alignment rules
        float2 uvScale; // per-material UV tiling factor applied to interpolated texture coordinates before sampling
//      float2 uvScale; // per-material UV tiling factor applied to interpolated texture coordinates before sampling
        float2 padding003; // explicit padding to pad the struct to a 16-byte multiple for std430
//      float2 padding003; // explicit padding to pad the struct to a 16-byte multiple for std430
    };
//  };

    StructuredBuffer<Material> materials;
//  StructuredBuffer<Material> materials;

    struct RadianceCacheEntry {
//  struct RadianceCacheEntry {
        float4 accumulatedRadianceAndSampleCount; // rgb = running radiance accumulator; w = sample count, saturating at CACHE_MAX_SAMPLES so the average keeps adapting
//      float4 accumulatedRadianceAndSampleCount; // rgb = running radiance accumulator; w = sample count, saturating at CACHE_MAX_SAMPLES so the average keeps adapting
        float4 worldPositionAndHashKey; // xyz = world position of the cached point (for spatial-proximity validation on read); w = bit-packed hash key identifying the occupying entry
//      float4 worldPositionAndHashKey; // xyz = world position of the cached point (for spatial-proximity validation on read); w = bit-packed hash key identifying the occupying entry
        float4 surfaceNormalAndLastWriteFrame; // xyz = surface normal, gated against the query normal to reject mismatched surfaces; w = frame index of the last write, used for stale-entry eviction
//      float4 surfaceNormalAndLastWriteFrame; // xyz = surface normal, gated against the query normal to reject mismatched surfaces; w = frame index of the last write, used for stale-entry eviction
        // The former reservedPadding vec4 was removed: three vec4s already satisfy std430
//      // The former reservedPadding vec4 was removed: three vec4s already satisfy std430
        // alignment, and dropping the dead 16 bytes cuts every cache entry read/write by 25%.
//      // alignment, and dropping the dead 16 bytes cuts every cache entry read/write by 25%.
    };
//  };

    RWStructuredBuffer<RadianceCacheEntry> cacheEntries;
//  RWStructuredBuffer<RadianceCacheEntry> cacheEntries;

    // ReSTIR Spatiotemporal Reservoir Buffer (double-buffered: 2 frames)
//  // ReSTIR Spatiotemporal Reservoir Buffer (double-buffered: 2 frames)
    struct LightSample {
//  struct LightSample {
        float3 position;
//      float3 position;
        float padding1;
//      float padding1;
        float3 normal;
//      float3 normal;
        float padding2;
//      float padding2;
        float3 emission;
//      float3 emission;
        float probability_density;
//      float probability_density;
    };
//  };

    struct Reservoir {
//  struct Reservoir {
        LightSample selected_sample; // 48 bytes
//      LightSample selected_sample; // 48 bytes
        float sum_of_weights;        // 4 bytes | w_sum
//      float sum_of_weights;        // 4 bytes | w_sum
        float sample_count;          // 4 bytes | M
//      float sample_count;          // 4 bytes | M
        float contribution_weight;   // 4 bytes | W
//      float contribution_weight;   // 4 bytes | W
        float surface_depth;         // 4 bytes | hit distance
//      float surface_depth;         // 4 bytes | hit distance
        float3 surface_normal;         // 12 bytes| world normal
//      float3 surface_normal;         // 12 bytes| world normal
        uint rate_and_flags;         // 4 bytes | VRT rate class in the low bits, 0x80 marks an actively traced pixel
//      uint rate_and_flags;         // 4 bytes | VRT rate class in the low bits, 0x80 marks an actively traced pixel
    };
//  };

    RWStructuredBuffer<Reservoir> sampleReservoirs;
//  RWStructuredBuffer<Reservoir> sampleReservoirs;

    // Neural Radiance Cache (NRC) Weights Buffer (4556 floats total)
//  // Neural Radiance Cache (NRC) Weights Buffer (4556 floats total)
    StructuredBuffer<float> neuralNetworkWeights;
//  StructuredBuffer<float> neuralNetworkWeights;

    // NRC Training Records Buffer (up to 8192 records of 9 floats)
//  // NRC Training Records Buffer (up to 8192 records of 9 floats)
    StructuredBuffer<float> nrcTrainingRecords;
//  StructuredBuffer<float> nrcTrainingRecords;

    // NRC Training Atomic Counter
//  // NRC Training Atomic Counter
    StructuredBuffer<uint> trainingRecordCounter;
//  StructuredBuffer<uint> trainingRecordCounter;

    struct Ray {
//  struct Ray {
        float3 origin; // ray start point in world space
//      float3 origin; // ray start point in world space
        float3 direction; // ray propagation direction in world space (kept normalized so hit distances read as true world units)
//      float3 direction; // ray propagation direction in world space (kept normalized so hit distances read as true world units)
    };
//  };

    struct RayHitResult {
//  struct RayHitResult {
        float3 hitSurfaceNormal; // interpolated surface normal at the hit, flipped to oppose the incoming ray for consistent shading
//      float3 hitSurfaceNormal; // interpolated surface normal at the hit, flipped to oppose the incoming ray for consistent shading
        float3 hitSurfaceTangent; // interpolated surface tangent at the hit, used to build the TBN basis for normal mapping
//      float3 hitSurfaceTangent; // interpolated surface tangent at the hit, used to build the TBN basis for normal mapping
        float2 hitSurfaceTexcoord; // interpolated UV at the hit; transiently holds the raw (u,v) barycentrics until the closest-hit post-pass resolves them
//      float2 hitSurfaceTexcoord; // interpolated UV at the hit; transiently holds the raw (u,v) barycentrics until the closest-hit post-pass resolves them
        float hitDistance; // parametric distance t from the ray origin to the hit point
//      float hitDistance; // parametric distance t from the ray origin to the hit point
        bool isFrontFaceHit; // true when the ray struck the geometric front face (surface normal opposes the ray direction)
//      bool isFrontFaceHit; // true when the ray struck the geometric front face (surface normal opposes the ray direction)
        int hitTriangleIndex; // index of the intersected triangle, or -1 to denote a miss
//      int hitTriangleIndex; // index of the intersected triangle, or -1 to denote a miss
    };
//  };

    uniform float uTime;
//  uniform float uTime;
    struct PointLight {
//  struct PointLight {
        float3 position; // light center in world space
//      float3 position; // light center in world space
        float radius; // sphere-light radius; the facing hemisphere area 2*PI*r*r drives the NEE solid-angle PDF
//      float radius; // sphere-light radius; the facing hemisphere area 2*PI*r*r drives the NEE solid-angle PDF
        float3 color; // emitted radiance (RGB), already scaled by the light's intensity
//      float3 color; // emitted radiance (RGB), already scaled by the light's intensity
        float cdf; // cumulative selection probability used to stochastically pick this light from the list
//      float cdf; // cumulative selection probability used to stochastically pick this light from the list
        float pdf; // probability of selecting this light, dividing out to keep the NEE estimator unbiased
//      float pdf; // probability of selecting this light, dividing out to keep the NEE estimator unbiased
        float padding; // explicit padding to satisfy std430 alignment after the vec3 + scalars
//      float padding; // explicit padding to satisfy std430 alignment after the vec3 + scalars
    };
//  };
    uniform int uPointLightCount;
//  uniform int uPointLightCount;
    uniform PointLight uPointLights[10];
//  uniform PointLight uPointLights[10];
    uniform float3 uCameraGlobalPosition;
//  uniform float3 uCameraGlobalPosition;
    uniform int uFrameCount;
//  uniform int uFrameCount;
    Sampler2DArray<float4> uSceneTextureArray;
//  Sampler2DArray<float4> uSceneTextureArray;
    // The HDRI, the volumetric sky cloud layer (baked equirectangular layer, noise volume, coverage and
//  // The HDRI, the volumetric sky cloud layer (baked equirectangular layer, noise volume, coverage and
    // density) and the participating media are parameters of the imported modules.
//  // density) and the participating media are parameters of the imported modules.

    uniform float3 uPixel00Coordinates;
//  uniform float3 uPixel00Coordinates;
    uniform float3 uPixelDeltaU;
//  uniform float3 uPixelDeltaU;
    uniform float3 uPixelDeltaV;
//  uniform float3 uPixelDeltaV;
    uniform float2 uJitter;
//  uniform float2 uJitter;
    uniform uint uCacheFrameCounter;
//  uniform uint uCacheFrameCounter;
    uniform float uCacheBlendFactor;
//  uniform float uCacheBlendFactor;
    uniform int uNRCEnabled;
//  uniform int uNRCEnabled;
    uniform int uReSTIREnabled;
//  uniform int uReSTIREnabled;
    uniform int uVRTEnabled;
//  uniform int uVRTEnabled;
    uniform int uVRTVisualize;
//  uniform int uVRTVisualize;
    uniform int2 uResolution;
//  uniform int2 uResolution;
    uniform float4x4 uPreviousTransformViewProjection;
//  uniform float4x4 uPreviousTransformViewProjection;
    uniform float3 uPreviousCameraPosition;
//  uniform float3 uPreviousCameraPosition;
    uniform int uResetHistory;
//  uniform int uResetHistory;
    // NRC input-domain normalisation. The frequency positional encoding below is only well conditioned
//  // NRC input-domain normalisation. The frequency positional encoding below is only well conditioned
    // when the encoded position spans roughly [-0.75, 0.75] (which is what Experiment001's unit-scale
//  // when the encoded position spans roughly [-0.75, 0.75] (which is what Experiment001's unit-scale
    // Cornell box gave it via a bare position * 0.5). These map this scene's world AABB onto that range,
//  // Cornell box gave it via a bare position * 0.5). These map this scene's world AABB onto that range,
    // otherwise sin/cos of the raw position aliases many times over and the MLP cannot fit the scene.
//  // otherwise sin/cos of the raw position aliases many times over and the MLP cannot fit the scene.
    uniform float3 uNRCPositionOffset;
//  uniform float3 uNRCPositionOffset;
    uniform float uNRCPositionScale;
//  uniform float uNRCPositionScale;

    // Workgroup-resident copy of the NRC Exponential Moving Average weights, filled once per
//  // Workgroup-resident copy of the NRC Exponential Moving Average weights, filled once per
    // workgroup at the top of computeMain() so every cache query reads shared memory instead of the SSBO.
//  // workgroup at the top of computeMain() so every cache query reads shared memory instead of the SSBO.
    groupshared float shared_network_weights[1139];
//  groupshared float shared_network_weights[1139];

    // Shading Tuning Constants (PI / TWO_PI / INF come from math_constants, MIN_ROUGHNESS and F0_DEFAULT from principled_bsdf)
//  // Shading Tuning Constants (PI / TWO_PI / INF come from math_constants, MIN_ROUGHNESS and F0_DEFAULT from principled_bsdf)
    static const float NEE_PDF_EPSILON = 0.0001;
//  static const float NEE_PDF_EPSILON = 0.0001;
    static const float NEE_DIRECT_LIGHT_CLAMP = 50.0;
//  static const float NEE_DIRECT_LIGHT_CLAMP = 50.0;
    static const float RR_MIN_PROBABILITY = 0.05;
//  static const float RR_MIN_PROBABILITY = 0.05;
    static const float RR_MAX_PROBABILITY = 0.95;
//  static const float RR_MAX_PROBABILITY = 0.95;
    static const uint RR_MIN_BOUNCES = 4u;
//  static const uint RR_MIN_BOUNCES = 4u;
    static const float ACCUM_CLAMP = 20.0;
//  static const float ACCUM_CLAMP = 20.0;
    static const float EPSILON_INTERSECT = 1.0e-7;
//  static const float EPSILON_INTERSECT = 1.0e-7;
    static const float EPSILON_OFFSET = 0.001;
//  static const float EPSILON_OFFSET = 0.001;
    static const float MISS_DISTANCE = INF;
//  static const float MISS_DISTANCE = INF;
    // Variable Rate Tracing (VRT) rate classes; the value is the number of pixels that share one traced ray.
//  // Variable Rate Tracing (VRT) rate classes; the value is the number of pixels that share one traced ray.
    static const uint VRT_RATE_1X1 = 0u; // 1 ray per 1x1 block (100% ray budget)
//  static const uint VRT_RATE_1X1 = 0u; // 1 ray per 1x1 block (100% ray budget)
    static const uint VRT_RATE_2X1 = 1u; // 1 ray per 2x1 block (50% ray budget)
//  static const uint VRT_RATE_2X1 = 1u; // 1 ray per 2x1 block (50% ray budget)
    static const uint VRT_RATE_1X2 = 2u; // 1 ray per 1x2 block (50% ray budget)
//  static const uint VRT_RATE_1X2 = 2u; // 1 ray per 1x2 block (50% ray budget)
    static const uint VRT_RATE_2X2 = 3u; // 1 ray per 2x2 block (25% ray budget)
//  static const uint VRT_RATE_2X2 = 3u; // 1 ray per 2x2 block (25% ray budget)
    static const uint VRT_RATE_4X4 = 4u; // 1 ray per 4x4 block (6.25% ray budget)
//  static const uint VRT_RATE_4X4 = 4u; // 1 ray per 4x4 block (6.25% ray budget)
    static const uint CACHE_SIZE = 131072u;
//  static const uint CACHE_SIZE = 131072u;
    static const float CACHE_CELL_SIZE = 0.5;
//  static const float CACHE_CELL_SIZE = 0.5;
    static const float CACHE_NORMAL_THRESHOLD = 0.7;
//  static const float CACHE_NORMAL_THRESHOLD = 0.7;
    static const float CACHE_MIN_SAMPLES = 4.0;
//  static const float CACHE_MIN_SAMPLES = 4.0;
    static const float CACHE_MAX_SAMPLES = 256.0;
//  static const float CACHE_MAX_SAMPLES = 256.0;
    static const float CACHE_EVICTION_AGE = 30.0;
//  static const float CACHE_EVICTION_AGE = 30.0;
    static const float CACHE_ROUGHNESS_GATE = 0.3;
//  static const float CACHE_ROUGHNESS_GATE = 0.3;

    // Hardcoded Math Constants (EPSILON_MATH / EPSILON_DOT come from principled_bsdf)
//  // Hardcoded Math Constants (EPSILON_MATH / EPSILON_DOT come from principled_bsdf)
    static const float EPSILON_RAND = 0.999;
//  static const float EPSILON_RAND = 0.999;
    static const int MAX_BOUNCES = 32;
//  static const int MAX_BOUNCES = 32;

    // Xorshift/Wang Hash Pseudo-Random Number Generator (PRNG): a fast integer-avalanche hash maintaining a single 32-bit state, reseeded per-pixel and per-frame so sample streams stay decorrelated across the image and over time (avoiding structured, fixed-pattern noise). Each call avalanches the state through shifts and multiplies, then normalizes it to a uniform float in [0, 1) — supplying the cheap, statistically robust variates that drive every Monte Carlo decision: hemisphere direction sampling, GGX lobe selection, Russian Roulette termination, and Next Event Estimation.
//  // Xorshift/Wang Hash Pseudo-Random Number Generator (PRNG): a fast integer-avalanche hash maintaining a single 32-bit state, reseeded per-pixel and per-frame so sample streams stay decorrelated across the image and over time (avoiding structured, fixed-pattern noise). Each call avalanches the state through shifts and multiplies, then normalizes it to a uniform float in [0, 1) — supplying the cheap, statistically robust variates that drive every Monte Carlo decision: hemisphere direction sampling, GGX lobe selection, Russian Roulette termination, and Next Event Estimation.
    static uint seed;
//  static uint seed;
    uint wang_hash(uint seed) {
//  uint wang_hash(uint seed) {
        seed = (seed ^ 61) ^ (seed >> 16);
//      seed = (seed ^ 61) ^ (seed >> 16);
        seed *= 9;
//      seed *= 9;
        seed = seed ^ (seed >> 4);
//      seed = seed ^ (seed >> 4);
        seed *= 0x27d4eb2d;
//      seed *= 0x27d4eb2d;
        seed = seed ^ (seed >> 15);
//      seed = seed ^ (seed >> 15);
        return seed;
//      return seed;
    }
//  }
    float rand() {
//  float rand() {
        seed = wang_hash(seed);
//      seed = wang_hash(seed);
        return min(float(seed) / 4294967296.0, EPSILON_RAND);
//      return min(float(seed) / 4294967296.0, EPSILON_RAND);
    }
//  }
    void initRNG(float2 pixel) {
//  void initRNG(float2 pixel) {
        // Use float coordinates and wang_hash for better random distribution
//      // Use float coordinates and wang_hash for better random distribution
        seed = uint(pixel.x * 1973.0 + pixel.y * 9277.0) + uint(uFrameCount) * 26699u;
//      seed = uint(pixel.x * 1973.0 + pixel.y * 9277.0) + uint(uFrameCount) * 26699u;
        seed = wang_hash(seed);
//      seed = wang_hash(seed);
    }
//  }

    uint hashCacheKey(float3 worldPosition, float3 surfaceNormal) {
//  uint hashCacheKey(float3 worldPosition, float3 surfaceNormal) {
        int3 cellCoordinate = int3(floor(worldPosition / CACHE_CELL_SIZE));
//      int3 cellCoordinate = int3(floor(worldPosition / CACHE_CELL_SIZE));
        uint normalDirectionIndex = 0u;
//      uint normalDirectionIndex = 0u;
        float3 absoluteNormal = abs(surfaceNormal);
//      float3 absoluteNormal = abs(surfaceNormal);
        if (absoluteNormal.x >= absoluteNormal.y && absoluteNormal.x >= absoluteNormal.z) normalDirectionIndex = surfaceNormal.x > 0.0 ? 0u : 1u;
//      if (absoluteNormal.x >= absoluteNormal.y && absoluteNormal.x >= absoluteNormal.z) normalDirectionIndex = surfaceNormal.x > 0.0 ? 0u : 1u;
        else if (absoluteNormal.y >= absoluteNormal.z) normalDirectionIndex = surfaceNormal.y > 0.0 ? 2u : 3u;
//      else if (absoluteNormal.y >= absoluteNormal.z) normalDirectionIndex = surfaceNormal.y > 0.0 ? 2u : 3u;
        else normalDirectionIndex = surfaceNormal.z > 0.0 ? 4u : 5u;
//      else normalDirectionIndex = surfaceNormal.z > 0.0 ? 4u : 5u;
        uint hashKey = uint(cellCoordinate.x) * 73856093u ^ uint(cellCoordinate.y) * 19349663u ^ uint(cellCoordinate.z) * 83492791u ^ normalDirectionIndex * 6291469u;
//      uint hashKey = uint(cellCoordinate.x) * 73856093u ^ uint(cellCoordinate.y) * 19349663u ^ uint(cellCoordinate.z) * 83492791u ^ normalDirectionIndex * 6291469u;
        return wang_hash(hashKey);
//      return wang_hash(hashKey);
    }
//  }

    uint cacheSlot(uint hashKey) {
//  uint cacheSlot(uint hashKey) {
        return hashKey & (CACHE_SIZE - 1u);
//      return hashKey & (CACHE_SIZE - 1u);
    }
//  }

    bool readCache(float3 worldPosition, float3 surfaceNormal, out float3 outRadiance) {
//  bool readCache(float3 worldPosition, float3 surfaceNormal, out float3 outRadiance) {
        uint hashKey = hashCacheKey(worldPosition, surfaceNormal);
//      uint hashKey = hashCacheKey(worldPosition, surfaceNormal);
        uint slotIndex = cacheSlot(hashKey);
//      uint slotIndex = cacheSlot(hashKey);
        RadianceCacheEntry entry = cacheEntries[slotIndex];
//      RadianceCacheEntry entry = cacheEntries[slotIndex];
        if (asuint(entry.worldPositionAndHashKey.w) != hashKey) return false;
//      if (asuint(entry.worldPositionAndHashKey.w) != hashKey) return false;
        if (entry.accumulatedRadianceAndSampleCount.w < CACHE_MIN_SAMPLES) return false;
//      if (entry.accumulatedRadianceAndSampleCount.w < CACHE_MIN_SAMPLES) return false;
        float distanceToEntry = length(entry.worldPositionAndHashKey.xyz - worldPosition);
//      float distanceToEntry = length(entry.worldPositionAndHashKey.xyz - worldPosition);
        if (distanceToEntry > CACHE_CELL_SIZE * 1.5) return false;
//      if (distanceToEntry > CACHE_CELL_SIZE * 1.5) return false;
        if (dot(entry.surfaceNormalAndLastWriteFrame.xyz, surfaceNormal) < CACHE_NORMAL_THRESHOLD) return false;
//      if (dot(entry.surfaceNormalAndLastWriteFrame.xyz, surfaceNormal) < CACHE_NORMAL_THRESHOLD) return false;
        outRadiance = entry.accumulatedRadianceAndSampleCount.rgb / entry.accumulatedRadianceAndSampleCount.w;
//      outRadiance = entry.accumulatedRadianceAndSampleCount.rgb / entry.accumulatedRadianceAndSampleCount.w;
        return true;
//      return true;
    }
//  }

    void writeCache(float3 worldPosition, float3 surfaceNormal, float3 radiance) {
//  void writeCache(float3 worldPosition, float3 surfaceNormal, float3 radiance) {
        uint hashKey = hashCacheKey(worldPosition, surfaceNormal);
//      uint hashKey = hashCacheKey(worldPosition, surfaceNormal);
        uint slotIndex = cacheSlot(hashKey);
//      uint slotIndex = cacheSlot(hashKey);
        RadianceCacheEntry entry = cacheEntries[slotIndex];
//      RadianceCacheEntry entry = cacheEntries[slotIndex];
        uint existingHashKey = asuint(entry.worldPositionAndHashKey.w);
//      uint existingHashKey = asuint(entry.worldPositionAndHashKey.w);
        if (existingHashKey == hashKey) {
//      if (existingHashKey == hashKey) {
            float sampleCount = min(entry.accumulatedRadianceAndSampleCount.w + 1.0, CACHE_MAX_SAMPLES);
//          float sampleCount = min(entry.accumulatedRadianceAndSampleCount.w + 1.0, CACHE_MAX_SAMPLES);
            float blendAlpha = 1.0 / sampleCount;
//          float blendAlpha = 1.0 / sampleCount;
            float3 previousAverageRadiance = entry.accumulatedRadianceAndSampleCount.rgb / max(entry.accumulatedRadianceAndSampleCount.w, 1.0);
//          float3 previousAverageRadiance = entry.accumulatedRadianceAndSampleCount.rgb / max(entry.accumulatedRadianceAndSampleCount.w, 1.0);
            cacheEntries[slotIndex].accumulatedRadianceAndSampleCount = float4(lerp(previousAverageRadiance, radiance, blendAlpha) * sampleCount, sampleCount);
//          cacheEntries[slotIndex].accumulatedRadianceAndSampleCount = float4(lerp(previousAverageRadiance, radiance, blendAlpha) * sampleCount, sampleCount);
            cacheEntries[slotIndex].surfaceNormalAndLastWriteFrame.w = float(uCacheFrameCounter);
//          cacheEntries[slotIndex].surfaceNormalAndLastWriteFrame.w = float(uCacheFrameCounter);
        } else {
//      } else {
            float entryAge = float(uCacheFrameCounter) - entry.surfaceNormalAndLastWriteFrame.w;
//          float entryAge = float(uCacheFrameCounter) - entry.surfaceNormalAndLastWriteFrame.w;
            if (existingHashKey == 0u || entryAge > CACHE_EVICTION_AGE || entry.accumulatedRadianceAndSampleCount.w < 1.0) {
//          if (existingHashKey == 0u || entryAge > CACHE_EVICTION_AGE || entry.accumulatedRadianceAndSampleCount.w < 1.0) {
                cacheEntries[slotIndex].accumulatedRadianceAndSampleCount = float4(radiance, 1.0);
//              cacheEntries[slotIndex].accumulatedRadianceAndSampleCount = float4(radiance, 1.0);
                cacheEntries[slotIndex].worldPositionAndHashKey = float4(worldPosition, asfloat(hashKey));
//              cacheEntries[slotIndex].worldPositionAndHashKey = float4(worldPosition, asfloat(hashKey));
                cacheEntries[slotIndex].surfaceNormalAndLastWriteFrame = float4(surfaceNormal, float(uCacheFrameCounter));
//              cacheEntries[slotIndex].surfaceNormalAndLastWriteFrame = float4(surfaceNormal, float(uCacheFrameCounter));
            }
//          }
        }
//      }
    }
//  }

    float3 randomUnitVector() {
//  float3 randomUnitVector() {
        float cosPolar = rand() * 2.0 - 1.0;
//      float cosPolar = rand() * 2.0 - 1.0;
        float azimuthAngle = rand() * 2.0 * PI;
//      float azimuthAngle = rand() * 2.0 * PI;
        float sinPolarRadius = sqrt(1.0 - cosPolar * cosPolar);
//      float sinPolarRadius = sqrt(1.0 - cosPolar * cosPolar);
        float directionX = sinPolarRadius * cos(azimuthAngle);
//      float directionX = sinPolarRadius * cos(azimuthAngle);
        float directionY = sinPolarRadius * sin(azimuthAngle);
//      float directionY = sinPolarRadius * sin(azimuthAngle);
        return float3(directionX, directionY, cosPolar);
//      return float3(directionX, directionY, cosPolar);
    }
//  }

    // This pass's random source for the participating media trackers.
//  // This pass's random source for the participating media trackers.
    struct ShadingRandom : IMediumRandom {
//  struct ShadingRandom : IMediumRandom {
        static float next() {
//      static float next() {
            return rand();
//          return rand();
        }
//      }
    }
//  }

    // New Structs and Helper Functions
//  // New Structs and Helper Functions

    struct Interval {
//  struct Interval {
        float min;
//      float min;
        float max;
//      float max;
    };
//  };

    bool intervalContain(Interval interval, float value) { return interval.min <= value && value <= interval.max; }
//  bool intervalContain(Interval interval, float value) { return interval.min <= value && value <= interval.max; }
    bool intervalSurround(Interval interval, float value) { return interval.min < value && value < interval.max; }
//  bool intervalSurround(Interval interval, float value) { return interval.min < value && value < interval.max; }

    // Vector Utilities: Helper functions for component-wise min/max operations
//  // Vector Utilities: Helper functions for component-wise min/max operations
    float maxVec3(float3 vector) { return max(vector.x, max(vector.y, vector.z)); }
//  float maxVec3(float3 vector) { return max(vector.x, max(vector.y, vector.z)); }
    float minVec3(float3 vector) { return min(vector.x, min(vector.y, vector.z)); }
//  float minVec3(float3 vector) { return min(vector.x, min(vector.y, vector.z)); }

    // Ray-AABB Intersection (Slab Method): a branchless overlap test against the three pairs of parallel "slab" planes that bound an AABB. Using the precomputed reciprocal ray direction (turning costly divides into multiplies), it derives per-axis entry/exit distances, then collapses them to the latest entry and earliest exit via component-wise min/max — the box is hit only if that interval stays non-empty within the ray's valid travel range. Robust against axis-aligned rays through the reciprocal's signed infinities, this is the per-node rejection test that prunes whole subtrees during BVH traversal.
//  // Ray-AABB Intersection (Slab Method): a branchless overlap test against the three pairs of parallel "slab" planes that bound an AABB. Using the precomputed reciprocal ray direction (turning costly divides into multiplies), it derives per-axis entry/exit distances, then collapses them to the latest entry and earliest exit via component-wise min/max — the box is hit only if that interval stays non-empty within the ray's valid travel range. Robust against axis-aligned rays through the reciprocal's signed infinities, this is the per-node rejection test that prunes whole subtrees during BVH traversal.
    bool intersectAABB_Interval(float3 inverseRayDirection, float3 rayOrigin, Interval rayTravelDistanceLimit, float3 boxMinBound, float3 boxMaxBound) {
//  bool intersectAABB_Interval(float3 inverseRayDirection, float3 rayOrigin, Interval rayTravelDistanceLimit, float3 boxMinBound, float3 boxMaxBound) {
        float3 distanceToMinPlanes = (boxMinBound - rayOrigin) * inverseRayDirection;
//      float3 distanceToMinPlanes = (boxMinBound - rayOrigin) * inverseRayDirection;
        float3 distanceToMaxPlanes = (boxMaxBound - rayOrigin) * inverseRayDirection;
//      float3 distanceToMaxPlanes = (boxMaxBound - rayOrigin) * inverseRayDirection;
        float3 perAxisNearDistance = min(distanceToMinPlanes, distanceToMaxPlanes);
//      float3 perAxisNearDistance = min(distanceToMinPlanes, distanceToMaxPlanes);
        float3 perAxisFarDistance = max(distanceToMinPlanes, distanceToMaxPlanes);
//      float3 perAxisFarDistance = max(distanceToMinPlanes, distanceToMaxPlanes);

        float nearestEntryDistance = maxVec3(perAxisNearDistance);
//      float nearestEntryDistance = maxVec3(perAxisNearDistance);
        float farthestExitDistance = minVec3(perAxisFarDistance);
//      float farthestExitDistance = minVec3(perAxisFarDistance);

        float clampedEntryDistance = max(rayTravelDistanceLimit.min, nearestEntryDistance);
//      float clampedEntryDistance = max(rayTravelDistanceLimit.min, nearestEntryDistance);
        float clampedExitDistance = min(rayTravelDistanceLimit.max, farthestExitDistance);
//      float clampedExitDistance = min(rayTravelDistanceLimit.max, farthestExitDistance);

        return clampedExitDistance >= clampedEntryDistance;
//      return clampedExitDistance >= clampedEntryDistance;
    }
//  }

    // Distance Estimation to AABB: returns the parametric entry distance (t) at which a ray first pierces a bounding box, or a MISS sentinel when the box is missed entirely or lies wholly behind the ray's current valid range. Unlike the boolean slab test, this scalar lets the traversal order a node's two children front-to-back — pushing the farther child first so the nearer one is popped and tested first — which tightens the closest-hit interval sooner and prunes the more distant subtree, sharply reducing the number of traversal steps.
//  // Distance Estimation to AABB: returns the parametric entry distance (t) at which a ray first pierces a bounding box, or a MISS sentinel when the box is missed entirely or lies wholly behind the ray's current valid range. Unlike the boolean slab test, this scalar lets the traversal order a node's two children front-to-back — pushing the farther child first so the nearer one is popped and tested first — which tightens the closest-hit interval sooner and prunes the more distant subtree, sharply reducing the number of traversal steps.
    float calculateDistanceToAABB3D(float3 inverseRayDirection, float3 rayOrigin, Interval rayTravelDistanceLimit, float3 boxMinBound, float3 boxMaxBound) {
//  float calculateDistanceToAABB3D(float3 inverseRayDirection, float3 rayOrigin, Interval rayTravelDistanceLimit, float3 boxMinBound, float3 boxMaxBound) {
        float3 distanceToMinPlanes = (boxMinBound - rayOrigin) * inverseRayDirection;
//      float3 distanceToMinPlanes = (boxMinBound - rayOrigin) * inverseRayDirection;
        float3 distanceToMaxPlanes = (boxMaxBound - rayOrigin) * inverseRayDirection;
//      float3 distanceToMaxPlanes = (boxMaxBound - rayOrigin) * inverseRayDirection;

        float3 clampedNearDistances = max(min(distanceToMinPlanes, distanceToMaxPlanes), float3(rayTravelDistanceLimit.min));
//      float3 clampedNearDistances = max(min(distanceToMinPlanes, distanceToMaxPlanes), float3(rayTravelDistanceLimit.min));
        float3 clampedFarDistances = min(max(distanceToMinPlanes, distanceToMaxPlanes), float3(rayTravelDistanceLimit.max));
//      float3 clampedFarDistances = min(max(distanceToMinPlanes, distanceToMaxPlanes), float3(rayTravelDistanceLimit.max));

        float nearestEntryDistance = maxVec3(clampedNearDistances);
//      float nearestEntryDistance = maxVec3(clampedNearDistances);
        float farthestExitDistance = minVec3(clampedFarDistances);
//      float farthestExitDistance = minVec3(clampedFarDistances);

        if (farthestExitDistance >= nearestEntryDistance && nearestEntryDistance < rayTravelDistanceLimit.max) {
//      if (farthestExitDistance >= nearestEntryDistance && nearestEntryDistance < rayTravelDistanceLimit.max) {
            return nearestEntryDistance;
//          return nearestEntryDistance;
        }
//      }
        return MISS_DISTANCE;
//      return MISS_DISTANCE;
    }
//  }

    // Ray-Triangle Fast Intersection (Any-Hit): a Möller-Trumbore test specialized for boolean occlusion queries. It computes only the barycentric coordinates and hit distance required to accept or reject — skipping all attribute interpolation — and deliberately treats strongly transmissive surfaces (transmission > 0.5) as non-occluding so glass and other refractive media cast no opaque shadow. Returning true on the first qualifying hit within the distance limits makes it the ideal primitive test for the binary shadow rays of Next Event Estimation.
//  // Ray-Triangle Fast Intersection (Any-Hit): a Möller-Trumbore test specialized for boolean occlusion queries. It computes only the barycentric coordinates and hit distance required to accept or reject — skipping all attribute interpolation — and deliberately treats strongly transmissive surfaces (transmission > 0.5) as non-occluding so glass and other refractive media cast no opaque shadow. Returning true on the first qualifying hit within the distance limits makes it the ideal primitive test for the binary shadow rays of Next Event Estimation.
    bool intersectTriangleAnyHit(Ray ray, int triangleIndex, Interval rayTravelDistanceLimit) {
//  bool intersectTriangleAnyHit(Ray ray, int triangleIndex, Interval rayTravelDistanceLimit) {
        int vertexBaseIndex = triangleIndex * 3;
//      int vertexBaseIndex = triangleIndex * 3;
        float3 vertexPosition0 = vertices[vertexBaseIndex + 0].positionAndTexcoordU.xyz;
//      float3 vertexPosition0 = vertices[vertexBaseIndex + 0].positionAndTexcoordU.xyz;
        float3 vertexPosition1 = vertices[vertexBaseIndex + 1].positionAndTexcoordU.xyz;
//      float3 vertexPosition1 = vertices[vertexBaseIndex + 1].positionAndTexcoordU.xyz;
        float3 vertexPosition2 = vertices[vertexBaseIndex + 2].positionAndTexcoordU.xyz;
//      float3 vertexPosition2 = vertices[vertexBaseIndex + 2].positionAndTexcoordU.xyz;

        const float intersectionEpsilon = EPSILON_INTERSECT;
//      const float intersectionEpsilon = EPSILON_INTERSECT;

        float3 edge01 = vertexPosition1 - vertexPosition0;
//      float3 edge01 = vertexPosition1 - vertexPosition0;
        float3 edge02 = vertexPosition2 - vertexPosition0;
//      float3 edge02 = vertexPosition2 - vertexPosition0;

        float3 directionCrossEdge02 = cross(ray.direction, edge02);
//      float3 directionCrossEdge02 = cross(ray.direction, edge02);
        float determinant = dot(edge01, directionCrossEdge02);
//      float determinant = dot(edge01, directionCrossEdge02);

        if (abs(determinant) < intersectionEpsilon) return false;
//      if (abs(determinant) < intersectionEpsilon) return false;

        float inverseDeterminant = 1.0 / determinant;
//      float inverseDeterminant = 1.0 / determinant;
        float3 originToVertex0 = ray.origin - vertexPosition0;
//      float3 originToVertex0 = ray.origin - vertexPosition0;
        float barycentricU = inverseDeterminant * dot(originToVertex0, directionCrossEdge02);
//      float barycentricU = inverseDeterminant * dot(originToVertex0, directionCrossEdge02);

        if (barycentricU < 0.0 || barycentricU > 1.0) return false;
//      if (barycentricU < 0.0 || barycentricU > 1.0) return false;

        float3 originCrossEdge01 = cross(originToVertex0, edge01);
//      float3 originCrossEdge01 = cross(originToVertex0, edge01);
        float barycentricV = inverseDeterminant * dot(ray.direction, originCrossEdge01);
//      float barycentricV = inverseDeterminant * dot(ray.direction, originCrossEdge01);

        if (barycentricV < 0.0 || barycentricU + barycentricV > 1.0) return false;
//      if (barycentricV < 0.0 || barycentricU + barycentricV > 1.0) return false;

        float hitDistanceAlongRay = inverseDeterminant * dot(edge02, originCrossEdge01);
//      float hitDistanceAlongRay = inverseDeterminant * dot(edge02, originCrossEdge01);

        if (!intervalSurround(rayTravelDistanceLimit, hitDistanceAlongRay)) return false;
//      if (!intervalSurround(rayTravelDistanceLimit, hitDistanceAlongRay)) return false;

        int materialIndex = int(vertices[vertexBaseIndex].tangentAndMaterialIndex.w);
//      int materialIndex = int(vertices[vertexBaseIndex].tangentAndMaterialIndex.w);
        if (materials[materialIndex].transmission > 0.5) return false;
//      if (materials[materialIndex].transmission > 0.5) return false;

        return true;
//      return true;
    }
//  }

    // Ray-Triangle Detailed Intersection (Closest-Hit): a full Möller-Trumbore test that, on a successful hit, records the hit distance, triangle index, and raw barycentric coordinates. Rather than interpolating attributes here, it defers that work to the traversal routine — so the per-vertex UVs, normals, and tangents are reconstructed from barycentrics exactly once, for the single winning surface — delivering the smooth, high-fidelity shading inputs of the closest visible hit while avoiding wasted interpolation on candidates that later prove occluded.
//  // Ray-Triangle Detailed Intersection (Closest-Hit): a full Möller-Trumbore test that, on a successful hit, records the hit distance, triangle index, and raw barycentric coordinates. Rather than interpolating attributes here, it defers that work to the traversal routine — so the per-vertex UVs, normals, and tangents are reconstructed from barycentrics exactly once, for the single winning surface — delivering the smooth, high-fidelity shading inputs of the closest visible hit while avoiding wasted interpolation on candidates that later prove occluded.
    RayHitResult intersectTriangleClosestHit(Ray ray, int triangleIndex, Interval rayTravelDistanceLimit) {
//  RayHitResult intersectTriangleClosestHit(Ray ray, int triangleIndex, Interval rayTravelDistanceLimit) {
        RayHitResult hitResult;
//      RayHitResult hitResult;
        hitResult.hitTriangleIndex = -1;
//      hitResult.hitTriangleIndex = -1;
        hitResult.hitDistance = rayTravelDistanceLimit.max;
//      hitResult.hitDistance = rayTravelDistanceLimit.max;

        int vertexBaseIndex = triangleIndex * 3;
//      int vertexBaseIndex = triangleIndex * 3;
        float3 vertexPosition0 = vertices[vertexBaseIndex + 0].positionAndTexcoordU.xyz;
//      float3 vertexPosition0 = vertices[vertexBaseIndex + 0].positionAndTexcoordU.xyz;
        float3 vertexPosition1 = vertices[vertexBaseIndex + 1].positionAndTexcoordU.xyz;
//      float3 vertexPosition1 = vertices[vertexBaseIndex + 1].positionAndTexcoordU.xyz;
        float3 vertexPosition2 = vertices[vertexBaseIndex + 2].positionAndTexcoordU.xyz;
//      float3 vertexPosition2 = vertices[vertexBaseIndex + 2].positionAndTexcoordU.xyz;

        const float intersectionEpsilon = EPSILON_INTERSECT;
//      const float intersectionEpsilon = EPSILON_INTERSECT;
        float3 edge01 = vertexPosition1 - vertexPosition0;
//      float3 edge01 = vertexPosition1 - vertexPosition0;
        float3 edge02 = vertexPosition2 - vertexPosition0;
//      float3 edge02 = vertexPosition2 - vertexPosition0;

        float3 directionCrossEdge02 = cross(ray.direction, edge02);
//      float3 directionCrossEdge02 = cross(ray.direction, edge02);
        float determinant = dot(edge01, directionCrossEdge02);
//      float determinant = dot(edge01, directionCrossEdge02);

        if (abs(determinant) < intersectionEpsilon) return hitResult;
//      if (abs(determinant) < intersectionEpsilon) return hitResult;

        float inverseDeterminant = 1.0 / determinant;
//      float inverseDeterminant = 1.0 / determinant;
        float3 originToVertex0 = ray.origin - vertexPosition0;
//      float3 originToVertex0 = ray.origin - vertexPosition0;
        float barycentricU = inverseDeterminant * dot(originToVertex0, directionCrossEdge02);
//      float barycentricU = inverseDeterminant * dot(originToVertex0, directionCrossEdge02);

        if (barycentricU < 0.0 || barycentricU > 1.0) return hitResult;
//      if (barycentricU < 0.0 || barycentricU > 1.0) return hitResult;

        float3 originCrossEdge01 = cross(originToVertex0, edge01);
//      float3 originCrossEdge01 = cross(originToVertex0, edge01);
        float barycentricV = inverseDeterminant * dot(ray.direction, originCrossEdge01);
//      float barycentricV = inverseDeterminant * dot(ray.direction, originCrossEdge01);

        if (barycentricV < 0.0 || barycentricU + barycentricV > 1.0) return hitResult;
//      if (barycentricV < 0.0 || barycentricU + barycentricV > 1.0) return hitResult;

        float hitDistanceAlongRay = inverseDeterminant * dot(edge02, originCrossEdge01);
//      float hitDistanceAlongRay = inverseDeterminant * dot(edge02, originCrossEdge01);

        if (!intervalSurround(rayTravelDistanceLimit, hitDistanceAlongRay)) return hitResult;
//      if (!intervalSurround(rayTravelDistanceLimit, hitDistanceAlongRay)) return hitResult;

        if (hitDistanceAlongRay > intersectionEpsilon) {
//      if (hitDistanceAlongRay > intersectionEpsilon) {
            hitResult.hitDistance = hitDistanceAlongRay;
//          hitResult.hitDistance = hitDistanceAlongRay;
            hitResult.hitTriangleIndex = triangleIndex;
//          hitResult.hitTriangleIndex = triangleIndex;
            hitResult.hitSurfaceTexcoord = float2(barycentricU, barycentricV);
//          hitResult.hitSurfaceTexcoord = float2(barycentricU, barycentricV);

            return hitResult;
//          return hitResult;
        }
//      }

        return hitResult;
//      return hitResult;
    }
//  }

    // BVH Occlusion Query (Any-Hit Traversal): a stack-based depth-first walk that answers a single boolean — does any primitive block the segment between two points? Because it needs only existence, not the nearest hit, it keeps a fixed [min, max] distance interval (never narrowing it) and returns the instant a blocker is found, abandoning the rest of the stack. This early-out, combined with the any-hit triangle test that skips attribute interpolation, makes shadow-ray visibility for Next Event Estimation dramatically cheaper than a full closest-hit search.
//  // BVH Occlusion Query (Any-Hit Traversal): a stack-based depth-first walk that answers a single boolean — does any primitive block the segment between two points? Because it needs only existence, not the nearest hit, it keeps a fixed [min, max] distance interval (never narrowing it) and returns the instant a blocker is found, abandoning the rest of the stack. This early-out, combined with the any-hit triangle test that skips attribute interpolation, makes shadow-ray visibility for Next Event Estimation dramatically cheaper than a full closest-hit search.
    bool traverseBVHAnyHit(Ray ray, Interval rayTravelDistanceLimit) {
//  bool traverseBVHAnyHit(Ray ray, Interval rayTravelDistanceLimit) {
        // A NaN ray fails every triangle test but passes every box test (min/max drop the NaN), so it would walk the whole tree and stall its warp; it cannot hit anything, so return immediately.
//      // A NaN ray fails every triangle test but passes every box test (min/max drop the NaN), so it would walk the whole tree and stall its warp; it cannot hit anything, so return immediately.
        if (any(isnan(ray.origin)) || any(isnan(ray.direction))) {
//      if (any(isnan(ray.origin)) || any(isnan(ray.direction))) {
            return false;
//          return false;
        }
//      }
        float3 directionSign = sign(ray.direction);
//      float3 directionSign = sign(ray.direction);
        directionSign += 1.0 - abs(directionSign);
//      directionSign += 1.0 - abs(directionSign);
        float3 safeRayDirection = ray.direction + step(abs(ray.direction), float3(1e-8)) * directionSign * 1e-8;
//      float3 safeRayDirection = ray.direction + step(abs(ray.direction), float3(1e-8)) * directionSign * 1e-8;
        float3 inverseRayDirection = 1.0 / safeRayDirection;
//      float3 inverseRayDirection = 1.0 / safeRayDirection;
        int traversalStack[32];
//      int traversalStack[32];
        int traversalStackPointer = 0;
//      int traversalStackPointer = 0;

        // Check Root
//      // Check Root
        Node rootNode = nodes[0];
//      Node rootNode = nodes[0];
        if (!intersectAABB_Interval(inverseRayDirection, ray.origin, rayTravelDistanceLimit, rootNode.aabbMinAndLeftChild.xyz, rootNode.aabbMaxAndRightChild.xyz)) {
//      if (!intersectAABB_Interval(inverseRayDirection, ray.origin, rayTravelDistanceLimit, rootNode.aabbMinAndLeftChild.xyz, rootNode.aabbMaxAndRightChild.xyz)) {
            return false;
//          return false;
        }
//      }

        traversalStack[traversalStackPointer++] = 0;
//      traversalStack[traversalStackPointer++] = 0;

        while (traversalStackPointer > 0) {
//      while (traversalStackPointer > 0) {
            int currentNodeIndex = traversalStack[--traversalStackPointer];
//          int currentNodeIndex = traversalStack[--traversalStackPointer];
            Node currentNode = nodes[currentNodeIndex];
//          Node currentNode = nodes[currentNodeIndex];

            if (currentNode.aabbMinAndLeftChild.w < 0.0) { // Leaf
//          if (currentNode.aabbMinAndLeftChild.w < 0.0) { // Leaf
                int leafTriangleIndex = int(currentNode.aabbMaxAndRightChild.w);
//              int leafTriangleIndex = int(currentNode.aabbMaxAndRightChild.w);
                if (intersectTriangleAnyHit(ray, leafTriangleIndex, rayTravelDistanceLimit)) {
//              if (intersectTriangleAnyHit(ray, leafTriangleIndex, rayTravelDistanceLimit)) {
                    return true;
//                  return true;
                }
//              }
                continue;
//              continue;
            }
//          }

            // Internal
//          // Internal
            int leftChildIndex = int(currentNode.aabbMinAndLeftChild.w);
//          int leftChildIndex = int(currentNode.aabbMinAndLeftChild.w);
            int rightChildIndex = int(currentNode.aabbMaxAndRightChild.w);
//          int rightChildIndex = int(currentNode.aabbMaxAndRightChild.w);

            Node leftChild = nodes[leftChildIndex];
//          Node leftChild = nodes[leftChildIndex];
            Node rightChild = nodes[rightChildIndex];
//          Node rightChild = nodes[rightChildIndex];

            float leftChildDistance = calculateDistanceToAABB3D(inverseRayDirection, ray.origin, rayTravelDistanceLimit, leftChild.aabbMinAndLeftChild.xyz, leftChild.aabbMaxAndRightChild.xyz);
//          float leftChildDistance = calculateDistanceToAABB3D(inverseRayDirection, ray.origin, rayTravelDistanceLimit, leftChild.aabbMinAndLeftChild.xyz, leftChild.aabbMaxAndRightChild.xyz);
            float rightChildDistance = calculateDistanceToAABB3D(inverseRayDirection, ray.origin, rayTravelDistanceLimit, rightChild.aabbMinAndLeftChild.xyz, rightChild.aabbMaxAndRightChild.xyz);
//          float rightChildDistance = calculateDistanceToAABB3D(inverseRayDirection, ray.origin, rayTravelDistanceLimit, rightChild.aabbMinAndLeftChild.xyz, rightChild.aabbMaxAndRightChild.xyz);

            if (leftChildDistance == MISS_DISTANCE && rightChildDistance == MISS_DISTANCE) continue;
//          if (leftChildDistance == MISS_DISTANCE && rightChildDistance == MISS_DISTANCE) continue;

            if (leftChildDistance < rightChildDistance) {
//          if (leftChildDistance < rightChildDistance) {
                if (rightChildDistance < MISS_DISTANCE) traversalStack[traversalStackPointer++] = rightChildIndex;
//              if (rightChildDistance < MISS_DISTANCE) traversalStack[traversalStackPointer++] = rightChildIndex;
                if (leftChildDistance < MISS_DISTANCE) traversalStack[traversalStackPointer++] = leftChildIndex;
//              if (leftChildDistance < MISS_DISTANCE) traversalStack[traversalStackPointer++] = leftChildIndex;
            } else {
//          } else {
                if (leftChildDistance < MISS_DISTANCE) traversalStack[traversalStackPointer++] = leftChildIndex;
//              if (leftChildDistance < MISS_DISTANCE) traversalStack[traversalStackPointer++] = leftChildIndex;
                if (rightChildDistance < MISS_DISTANCE) traversalStack[traversalStackPointer++] = rightChildIndex;
//              if (rightChildDistance < MISS_DISTANCE) traversalStack[traversalStackPointer++] = rightChildIndex;
            }
//          }
        }
//      }
        return false;
//      return false;
    }
//  }

    // BVH Nearest Hit Query (Closest-Hit Traversal): a stack-based search for the single closest intersection along a ray. As each nearer surface is discovered it shrinks the interval's upper bound to that hit distance, so every subsequent node-box and triangle test culls anything lying farther away; ordering child pushes by entry distance (nearest popped first) makes this tightening happen as early as possible. Once the stack drains, a final pass reconstructs the winning hit's interpolated UVs, normals, and tangents to feed the shading model.
//  // BVH Nearest Hit Query (Closest-Hit Traversal): a stack-based search for the single closest intersection along a ray. As each nearer surface is discovered it shrinks the interval's upper bound to that hit distance, so every subsequent node-box and triangle test culls anything lying farther away; ordering child pushes by entry distance (nearest popped first) makes this tightening happen as early as possible. Once the stack drains, a final pass reconstructs the winning hit's interpolated UVs, normals, and tangents to feed the shading model.
    RayHitResult traverseBVHClosestHit(Ray ray, Interval rayTravelDistanceLimit) {
//  RayHitResult traverseBVHClosestHit(Ray ray, Interval rayTravelDistanceLimit) {
        RayHitResult closestHitResult;
//      RayHitResult closestHitResult;
        closestHitResult.hitTriangleIndex = -1;
//      closestHitResult.hitTriangleIndex = -1;
        closestHitResult.hitDistance = rayTravelDistanceLimit.max; // Initialize with max
//      closestHitResult.hitDistance = rayTravelDistanceLimit.max; // Initialize with max
        float closestHitDistance = rayTravelDistanceLimit.max;
//      float closestHitDistance = rayTravelDistanceLimit.max;
        // A NaN ray fails every triangle test but passes every box test (min/max drop the NaN), so it would walk the whole tree and stall its warp; it cannot hit anything, so return immediately.
//      // A NaN ray fails every triangle test but passes every box test (min/max drop the NaN), so it would walk the whole tree and stall its warp; it cannot hit anything, so return immediately.
        if (any(isnan(ray.origin)) || any(isnan(ray.direction))) {
//      if (any(isnan(ray.origin)) || any(isnan(ray.direction))) {
            return closestHitResult;
//          return closestHitResult;
        }
//      }

        float3 directionSign = sign(ray.direction);
//      float3 directionSign = sign(ray.direction);
        directionSign += 1.0 - abs(directionSign);
//      directionSign += 1.0 - abs(directionSign);
        float3 safeRayDirection = ray.direction + step(abs(ray.direction), float3(1e-8)) * directionSign * 1e-8;
//      float3 safeRayDirection = ray.direction + step(abs(ray.direction), float3(1e-8)) * directionSign * 1e-8;
        float3 inverseRayDirection = 1.0 / safeRayDirection;
//      float3 inverseRayDirection = 1.0 / safeRayDirection;
        int traversalStack[32];
//      int traversalStack[32];
        int traversalStackPointer = 0;
//      int traversalStackPointer = 0;

        traversalStack[traversalStackPointer++] = 0;
//      traversalStack[traversalStackPointer++] = 0;

        while (traversalStackPointer > 0) {
//      while (traversalStackPointer > 0) {
            int currentNodeIndex = traversalStack[--traversalStackPointer];
//          int currentNodeIndex = traversalStack[--traversalStackPointer];
            Node currentNode = nodes[currentNodeIndex];
//          Node currentNode = nodes[currentNodeIndex];

            Interval currentDistanceLimit;
//          Interval currentDistanceLimit;
            currentDistanceLimit.min = rayTravelDistanceLimit.min;
//          currentDistanceLimit.min = rayTravelDistanceLimit.min;
            currentDistanceLimit.max = closestHitDistance;
//          currentDistanceLimit.max = closestHitDistance;

            if (!intersectAABB_Interval(inverseRayDirection, ray.origin, currentDistanceLimit, currentNode.aabbMinAndLeftChild.xyz, currentNode.aabbMaxAndRightChild.xyz)) continue;
//          if (!intersectAABB_Interval(inverseRayDirection, ray.origin, currentDistanceLimit, currentNode.aabbMinAndLeftChild.xyz, currentNode.aabbMaxAndRightChild.xyz)) continue;

            if (currentNode.aabbMinAndLeftChild.w < 0.0) { // Leaf
//          if (currentNode.aabbMinAndLeftChild.w < 0.0) { // Leaf
                int leafTriangleIndex = int(currentNode.aabbMaxAndRightChild.w);
//              int leafTriangleIndex = int(currentNode.aabbMaxAndRightChild.w);
                RayHitResult candidateHit = intersectTriangleClosestHit(ray, leafTriangleIndex, currentDistanceLimit);
//              RayHitResult candidateHit = intersectTriangleClosestHit(ray, leafTriangleIndex, currentDistanceLimit);
                if (candidateHit.hitTriangleIndex != -1) {
//              if (candidateHit.hitTriangleIndex != -1) {
                    closestHitResult = candidateHit;
//                  closestHitResult = candidateHit;
                    closestHitDistance = candidateHit.hitDistance;
//                  closestHitDistance = candidateHit.hitDistance;
                }
//              }
                continue;
//              continue;
            }
//          }

            // Internal
//          // Internal
            int leftChildIndex = int(currentNode.aabbMinAndLeftChild.w);
//          int leftChildIndex = int(currentNode.aabbMinAndLeftChild.w);
            int rightChildIndex = int(currentNode.aabbMaxAndRightChild.w);
//          int rightChildIndex = int(currentNode.aabbMaxAndRightChild.w);
            Node leftChild = nodes[leftChildIndex];
//          Node leftChild = nodes[leftChildIndex];
            Node rightChild = nodes[rightChildIndex];
//          Node rightChild = nodes[rightChildIndex];

            float leftChildDistance = calculateDistanceToAABB3D(inverseRayDirection, ray.origin, currentDistanceLimit, leftChild.aabbMinAndLeftChild.xyz, leftChild.aabbMaxAndRightChild.xyz);
//          float leftChildDistance = calculateDistanceToAABB3D(inverseRayDirection, ray.origin, currentDistanceLimit, leftChild.aabbMinAndLeftChild.xyz, leftChild.aabbMaxAndRightChild.xyz);
            float rightChildDistance = calculateDistanceToAABB3D(inverseRayDirection, ray.origin, currentDistanceLimit, rightChild.aabbMinAndLeftChild.xyz, rightChild.aabbMaxAndRightChild.xyz);
//          float rightChildDistance = calculateDistanceToAABB3D(inverseRayDirection, ray.origin, currentDistanceLimit, rightChild.aabbMinAndLeftChild.xyz, rightChild.aabbMaxAndRightChild.xyz);

            if (leftChildDistance == MISS_DISTANCE && rightChildDistance == MISS_DISTANCE) continue;
//          if (leftChildDistance == MISS_DISTANCE && rightChildDistance == MISS_DISTANCE) continue;

            if (leftChildDistance < rightChildDistance) {
//          if (leftChildDistance < rightChildDistance) {
                if (rightChildDistance < closestHitDistance) traversalStack[traversalStackPointer++] = rightChildIndex;
//              if (rightChildDistance < closestHitDistance) traversalStack[traversalStackPointer++] = rightChildIndex;
                if (leftChildDistance < closestHitDistance) traversalStack[traversalStackPointer++] = leftChildIndex;
//              if (leftChildDistance < closestHitDistance) traversalStack[traversalStackPointer++] = leftChildIndex;
            } else {
//          } else {
                if (leftChildDistance < closestHitDistance) traversalStack[traversalStackPointer++] = leftChildIndex;
//              if (leftChildDistance < closestHitDistance) traversalStack[traversalStackPointer++] = leftChildIndex;
                if (rightChildDistance < closestHitDistance) traversalStack[traversalStackPointer++] = rightChildIndex;
//              if (rightChildDistance < closestHitDistance) traversalStack[traversalStackPointer++] = rightChildIndex;
            }
//          }
        }
//      }

        if (closestHitResult.hitTriangleIndex != -1) {
//      if (closestHitResult.hitTriangleIndex != -1) {
            float barycentricU = closestHitResult.hitSurfaceTexcoord.x;
//          float barycentricU = closestHitResult.hitSurfaceTexcoord.x;
            float barycentricV = closestHitResult.hitSurfaceTexcoord.y;
//          float barycentricV = closestHitResult.hitSurfaceTexcoord.y;
            int hitTriangleIndex = closestHitResult.hitTriangleIndex;
//          int hitTriangleIndex = closestHitResult.hitTriangleIndex;
            int vertexBaseIndex = hitTriangleIndex * 3;
//          int vertexBaseIndex = hitTriangleIndex * 3;

            // Interpolate UVs
//          // Interpolate UVs
            float2 texcoord0 = float2(vertices[vertexBaseIndex + 0].positionAndTexcoordU.w, vertices[vertexBaseIndex + 0].normalAndTexcoordV.w);
//          float2 texcoord0 = float2(vertices[vertexBaseIndex + 0].positionAndTexcoordU.w, vertices[vertexBaseIndex + 0].normalAndTexcoordV.w);
            float2 texcoord1 = float2(vertices[vertexBaseIndex + 1].positionAndTexcoordU.w, vertices[vertexBaseIndex + 1].normalAndTexcoordV.w);
//          float2 texcoord1 = float2(vertices[vertexBaseIndex + 1].positionAndTexcoordU.w, vertices[vertexBaseIndex + 1].normalAndTexcoordV.w);
            float2 texcoord2 = float2(vertices[vertexBaseIndex + 2].positionAndTexcoordU.w, vertices[vertexBaseIndex + 2].normalAndTexcoordV.w);
//          float2 texcoord2 = float2(vertices[vertexBaseIndex + 2].positionAndTexcoordU.w, vertices[vertexBaseIndex + 2].normalAndTexcoordV.w);
            closestHitResult.hitSurfaceTexcoord = (1.0 - barycentricU - barycentricV) * texcoord0 + barycentricU * texcoord1 + barycentricV * texcoord2;
//          closestHitResult.hitSurfaceTexcoord = (1.0 - barycentricU - barycentricV) * texcoord0 + barycentricU * texcoord1 + barycentricV * texcoord2;

            // Interpolate Normals
//          // Interpolate Normals
            float3 normal0 = vertices[vertexBaseIndex + 0].normalAndTexcoordV.xyz;
//          float3 normal0 = vertices[vertexBaseIndex + 0].normalAndTexcoordV.xyz;
            float3 normal1 = vertices[vertexBaseIndex + 1].normalAndTexcoordV.xyz;
//          float3 normal1 = vertices[vertexBaseIndex + 1].normalAndTexcoordV.xyz;
            float3 normal2 = vertices[vertexBaseIndex + 2].normalAndTexcoordV.xyz;
//          float3 normal2 = vertices[vertexBaseIndex + 2].normalAndTexcoordV.xyz;
            closestHitResult.hitSurfaceNormal = normalize((1.0 - barycentricU - barycentricV) * normal0 + barycentricU * normal1 + barycentricV * normal2);
//          closestHitResult.hitSurfaceNormal = normalize((1.0 - barycentricU - barycentricV) * normal0 + barycentricU * normal1 + barycentricV * normal2);

            // Interpolate Tangents
//          // Interpolate Tangents
            float3 tangent0 = vertices[vertexBaseIndex + 0].tangentAndMaterialIndex.xyz;
//          float3 tangent0 = vertices[vertexBaseIndex + 0].tangentAndMaterialIndex.xyz;
            float3 tangent1 = vertices[vertexBaseIndex + 1].tangentAndMaterialIndex.xyz;
//          float3 tangent1 = vertices[vertexBaseIndex + 1].tangentAndMaterialIndex.xyz;
            float3 tangent2 = vertices[vertexBaseIndex + 2].tangentAndMaterialIndex.xyz;
//          float3 tangent2 = vertices[vertexBaseIndex + 2].tangentAndMaterialIndex.xyz;
            closestHitResult.hitSurfaceTangent = normalize((1.0 - barycentricU - barycentricV) * tangent0 + barycentricU * tangent1 + barycentricV * tangent2);
//          closestHitResult.hitSurfaceTangent = normalize((1.0 - barycentricU - barycentricV) * tangent0 + barycentricU * tangent1 + barycentricV * tangent2);
        }
//      }

        return closestHitResult;
//      return closestHitResult;
    }
//  }

    float3 sampleGGX(float3 normal, float roughness) {
//  float3 sampleGGX(float3 normal, float roughness) {
        float randomAzimuth = rand();
//      float randomAzimuth = rand();
        // Clamp randomElevation to prevent NaN/Inf in GGX sampling
//      // Clamp randomElevation to prevent NaN/Inf in GGX sampling
        float randomElevation = clamp(rand(), 0.0, EPSILON_RAND);
//      float randomElevation = clamp(rand(), 0.0, EPSILON_RAND);

        float alphaRoughness = roughness * roughness;
//      float alphaRoughness = roughness * roughness;
        float azimuthAngle = 2.0 * PI * randomAzimuth;
//      float azimuthAngle = 2.0 * PI * randomAzimuth;
        float cosElevation = sqrt((1.0 - randomElevation) / (1.0 + (alphaRoughness * alphaRoughness - 1.0) * randomElevation));
//      float cosElevation = sqrt((1.0 - randomElevation) / (1.0 + (alphaRoughness * alphaRoughness - 1.0) * randomElevation));
        // Rounding can push cosElevation one ulp past 1 on rough surfaces; clamp so sqrt never returns NaN (a NaN direction makes the BVH walk visit every node).
//      // Rounding can push cosElevation one ulp past 1 on rough surfaces; clamp so sqrt never returns NaN (a NaN direction makes the BVH walk visit every node).
        float sinElevation = sqrt(max(0.0, 1.0 - cosElevation * cosElevation));
//      float sinElevation = sqrt(max(0.0, 1.0 - cosElevation * cosElevation));

        float localX = sinElevation * cos(azimuthAngle);
//      float localX = sinElevation * cos(azimuthAngle);
        float localY = sinElevation * sin(azimuthAngle);
//      float localY = sinElevation * sin(azimuthAngle);
        float localZ = cosElevation;
//      float localZ = cosElevation;

        // Tangent space basis construction (orthonormal basis)
//      // Tangent space basis construction (orthonormal basis)
        float3 basisHelper;
//      float3 basisHelper;
        if (abs(normal.z) < EPSILON_RAND) {
//      if (abs(normal.z) < EPSILON_RAND) {
            basisHelper = float3(0.0, 0.0, 1.0);
//          basisHelper = float3(0.0, 0.0, 1.0);
        } else {
//      } else {
            basisHelper = float3(1.0, 0.0, 0.0);
//          basisHelper = float3(1.0, 0.0, 0.0);
        }
//      }
        float3 tangentAxis = normalize(cross(basisHelper, normal));
//      float3 tangentAxis = normalize(cross(basisHelper, normal));
        float3 bitangentAxis = cross(normal, tangentAxis);
//      float3 bitangentAxis = cross(normal, tangentAxis);

        // Transform the sampled half-vector to world space
//      // Transform the sampled half-vector to world space
        return normalize(tangentAxis * localX + bitangentAxis * localY + normal * localZ);
//      return normalize(tangentAxis * localX + bitangentAxis * localY + normal * localZ);
    }
//  }

    float3 reflectPrincipled(float3 incomingVector, float3 normal) {
//  float3 reflectPrincipled(float3 incomingVector, float3 normal) {
        return incomingVector - 2.0 * dot(incomingVector, normal) * normal;
//      return incomingVector - 2.0 * dot(incomingVector, normal) * normal;
    }
//  }

    float3 refractPrincipled(float3 incomingVector, float3 normal, float etaRatioOfIncidenceOverTransmission, float cosIncidentAngle, float sinTransmittedAngleSquared) {
//  float3 refractPrincipled(float3 incomingVector, float3 normal, float etaRatioOfIncidenceOverTransmission, float cosIncidentAngle, float sinTransmittedAngleSquared) {
        float cosTransmittedAngle = sqrt(max(0.0, 1.0 - sinTransmittedAngleSquared));
//      float cosTransmittedAngle = sqrt(max(0.0, 1.0 - sinTransmittedAngleSquared));
        float3 refractedDirection = normalize(etaRatioOfIncidenceOverTransmission * incomingVector + (etaRatioOfIncidenceOverTransmission * cosIncidentAngle - cosTransmittedAngle) * normal);
//      float3 refractedDirection = normalize(etaRatioOfIncidenceOverTransmission * incomingVector + (etaRatioOfIncidenceOverTransmission * cosIncidentAngle - cosTransmittedAngle) * normal);
        return refractedDirection;
//      return refractedDirection;
    }
//  }


    // Fused BSDF + PDF Evaluation: returns the bidirectional reflectance value f_r for a given view/light direction pair (the per-steradian throughput, before any cosine or PDF weighting) and, through the out parameter, the matching importance-sampling PDF. The BSDF sums a Disney diffuse lobe with a Cook-Torrance specular term — the GGX distribution D, the Smith geometry factor G, and the Schlick Fresnel F combined as D·F·G / (4·N·V·N·L) — where Fresnel partitions energy between the lobes and the metallic mask removes the diffuse component from conductors. Every scatter and NEE sample needs both the value and the PDF, so fusing them shares the half-vector, dot products, Fresnel and GGX D terms the previously separate routines each recomputed, halving the per-sample shading math without changing a single term of either result.
//  // Fused BSDF + PDF Evaluation: returns the bidirectional reflectance value f_r for a given view/light direction pair (the per-steradian throughput, before any cosine or PDF weighting) and, through the out parameter, the matching importance-sampling PDF. The BSDF sums a Disney diffuse lobe with a Cook-Torrance specular term — the GGX distribution D, the Smith geometry factor G, and the Schlick Fresnel F combined as D·F·G / (4·N·V·N·L) — where Fresnel partitions energy between the lobes and the metallic mask removes the diffuse component from conductors. Every scatter and NEE sample needs both the value and the PDF, so fusing them shares the half-vector, dot products, Fresnel and GGX D terms the previously separate routines each recomputed, halving the per-sample shading math without changing a single term of either result.

    float calculateLuminance(float3 color) {
//  float calculateLuminance(float3 color) {
        return dot(color, float3(0.2126, 0.7152, 0.0722));
//      return dot(color, float3(0.2126, 0.7152, 0.0722));
    }
//  }

    // Neural Radiance Cache forward pass.
//  // Neural Radiance Cache forward pass.
    // Multi-Layer Perceptron: 16 inputs -> 16 hidden (x4, ReLU) -> 3 outputs.
//  // Multi-Layer Perceptron: 16 inputs -> 16 hidden (x4, ReLU) -> 3 outputs.
    //
//  //
    // Reads the weights out of shared_network_weights, which computeMain() fills once per workgroup with the
//  // Reads the weights out of shared_network_weights, which computeMain() fills once per workgroup with the
    // Exponential Moving Average copy of the network. Two reasons this matters: it turns 1139 scalar
//  // Exponential Moving Average copy of the network. Two reasons this matters: it turns 1139 scalar
    // SSBO loads per query into shared-memory loads, and reading the weights straight from the SSBO
//  // SSBO loads per query into shared-memory loads, and reading the weights straight from the SSBO
    // inside this deeply inlined function made the NVIDIA GLSL compiler (the original OpenGL backend)
//  // inside this deeply inlined function made the NVIDIA GLSL compiler (the original OpenGL backend)
    // give up with "error C5025: lvalue in array access too complex", failing the whole program link.
//  // give up with "error C5025: lvalue in array access too complex", failing the whole program link.
    //
//  //
    // Activations are flat scalar arrays rather than vec4[4] blocks for the same reason: assigning a
//  // Activations are flat scalar arrays rather than vec4[4] blocks for the same reason: assigning a
    // dynamically indexed component of a dynamically indexed vec4 array ("h[i / 4u][i % 4u] = ...")
//  // dynamically indexed component of a dynamically indexed vec4 array ("h[i / 4u][i % 4u] = ...")
    // tripped the same compiler limit. This layout also matches nrc_train_cs.slang exactly, so inference
//  // tripped the same compiler limit. This layout also matches nrc_train_cs.slang exactly, so inference
    // and training cannot drift apart.
//  // and training cannot drift apart.
    float3 evaluateNeuralRadianceCache(float3 position, float3 normal) {
//  float3 evaluateNeuralRadianceCache(float3 position, float3 normal) {
        float3 scaledPos = (position - uNRCPositionOffset) * uNRCPositionScale;
//      float3 scaledPos = (position - uNRCPositionOffset) * uNRCPositionScale;

        // 16-dimensional input feature vector (position, normal, and frequency positional encoding)
//      // 16-dimensional input feature vector (position, normal, and frequency positional encoding)
        float inputFeatures[16];
//      float inputFeatures[16];
        inputFeatures[0] = scaledPos.x;
//      inputFeatures[0] = scaledPos.x;
        inputFeatures[1] = scaledPos.y;
//      inputFeatures[1] = scaledPos.y;
        inputFeatures[2] = scaledPos.z;
//      inputFeatures[2] = scaledPos.z;
        inputFeatures[3] = normal.x;
//      inputFeatures[3] = normal.x;
        inputFeatures[4] = normal.y;
//      inputFeatures[4] = normal.y;
        inputFeatures[5] = normal.z;
//      inputFeatures[5] = normal.z;
        inputFeatures[6] = sin(scaledPos.x * PI);
//      inputFeatures[6] = sin(scaledPos.x * PI);
        inputFeatures[7] = sin(scaledPos.y * PI);
//      inputFeatures[7] = sin(scaledPos.y * PI);
        inputFeatures[8] = sin(scaledPos.z * PI);
//      inputFeatures[8] = sin(scaledPos.z * PI);
        inputFeatures[9] = cos(scaledPos.x * PI);
//      inputFeatures[9] = cos(scaledPos.x * PI);
        inputFeatures[10] = cos(scaledPos.y * PI);
//      inputFeatures[10] = cos(scaledPos.y * PI);
        inputFeatures[11] = cos(scaledPos.z * PI);
//      inputFeatures[11] = cos(scaledPos.z * PI);
        inputFeatures[12] = sin(scaledPos.x * TWO_PI);
//      inputFeatures[12] = sin(scaledPos.x * TWO_PI);
        inputFeatures[13] = sin(scaledPos.y * TWO_PI);
//      inputFeatures[13] = sin(scaledPos.y * TWO_PI);
        inputFeatures[14] = sin(scaledPos.z * TWO_PI);
//      inputFeatures[14] = sin(scaledPos.z * TWO_PI);
        inputFeatures[15] = 1.0;
//      inputFeatures[15] = 1.0;

        // Hidden Layer 1 (weights offset 0, bias offset 256)
//      // Hidden Layer 1 (weights offset 0, bias offset 256)
        float hiddenLayer1[16];
//      float hiddenLayer1[16];
        [ForceUnroll]
//      [ForceUnroll]
        for (uint i = 0u; i < 16u; i++) {
//      for (uint i = 0u; i < 16u; i++) {
            float activationSum = shared_network_weights[256u + i];
//          float activationSum = shared_network_weights[256u + i];
            uint weightOffset = i * 16u;
//          uint weightOffset = i * 16u;
            [ForceUnroll]
//          [ForceUnroll]
            for (uint j = 0u; j < 16u; j++) {
//          for (uint j = 0u; j < 16u; j++) {
                activationSum += inputFeatures[j] * shared_network_weights[weightOffset + j];
//              activationSum += inputFeatures[j] * shared_network_weights[weightOffset + j];
            }
//          }
            hiddenLayer1[i] = max(0.0, activationSum); // ReLU
//          hiddenLayer1[i] = max(0.0, activationSum); // ReLU
        }
//      }

        // Hidden Layer 2 (weights offset 272, bias offset 528)
//      // Hidden Layer 2 (weights offset 272, bias offset 528)
        float hiddenLayer2[16];
//      float hiddenLayer2[16];
        [ForceUnroll]
//      [ForceUnroll]
        for (uint i = 0u; i < 16u; i++) {
//      for (uint i = 0u; i < 16u; i++) {
            float activationSum = shared_network_weights[528u + i];
//          float activationSum = shared_network_weights[528u + i];
            uint weightOffset = 272u + i * 16u;
//          uint weightOffset = 272u + i * 16u;
            [ForceUnroll]
//          [ForceUnroll]
            for (uint j = 0u; j < 16u; j++) {
//          for (uint j = 0u; j < 16u; j++) {
                activationSum += hiddenLayer1[j] * shared_network_weights[weightOffset + j];
//              activationSum += hiddenLayer1[j] * shared_network_weights[weightOffset + j];
            }
//          }
            hiddenLayer2[i] = max(0.0, activationSum); // ReLU
//          hiddenLayer2[i] = max(0.0, activationSum); // ReLU
        }
//      }

        // Hidden Layer 3 (weights offset 544, bias offset 800)
//      // Hidden Layer 3 (weights offset 544, bias offset 800)
        float hiddenLayer3[16];
//      float hiddenLayer3[16];
        [ForceUnroll]
//      [ForceUnroll]
        for (uint i = 0u; i < 16u; i++) {
//      for (uint i = 0u; i < 16u; i++) {
            float activationSum = shared_network_weights[800u + i];
//          float activationSum = shared_network_weights[800u + i];
            uint weightOffset = 544u + i * 16u;
//          uint weightOffset = 544u + i * 16u;
            [ForceUnroll]
//          [ForceUnroll]
            for (uint j = 0u; j < 16u; j++) {
//          for (uint j = 0u; j < 16u; j++) {
                activationSum += hiddenLayer2[j] * shared_network_weights[weightOffset + j];
//              activationSum += hiddenLayer2[j] * shared_network_weights[weightOffset + j];
            }
//          }
            hiddenLayer3[i] = max(0.0, activationSum); // ReLU
//          hiddenLayer3[i] = max(0.0, activationSum); // ReLU
        }
//      }

        // Hidden Layer 4 (weights offset 816, bias offset 1072)
//      // Hidden Layer 4 (weights offset 816, bias offset 1072)
        float hiddenLayer4[16];
//      float hiddenLayer4[16];
        [ForceUnroll]
//      [ForceUnroll]
        for (uint i = 0u; i < 16u; i++) {
//      for (uint i = 0u; i < 16u; i++) {
            float activationSum = shared_network_weights[1072u + i];
//          float activationSum = shared_network_weights[1072u + i];
            uint weightOffset = 816u + i * 16u;
//          uint weightOffset = 816u + i * 16u;
            [ForceUnroll]
//          [ForceUnroll]
            for (uint j = 0u; j < 16u; j++) {
//          for (uint j = 0u; j < 16u; j++) {
                activationSum += hiddenLayer3[j] * shared_network_weights[weightOffset + j];
//              activationSum += hiddenLayer3[j] * shared_network_weights[weightOffset + j];
            }
//          }
            hiddenLayer4[i] = max(0.0, activationSum); // ReLU
//          hiddenLayer4[i] = max(0.0, activationSum); // ReLU
        }
//      }

        // Output Layer (weights offset 1088, bias offset 1136)
//      // Output Layer (weights offset 1088, bias offset 1136)
        float3 outputRadiance = float3(0.0);
//      float3 outputRadiance = float3(0.0);
        [ForceUnroll]
//      [ForceUnroll]
        for (uint i = 0u; i < 3u; i++) {
//      for (uint i = 0u; i < 3u; i++) {
            float activationSum = shared_network_weights[1136u + i];
//          float activationSum = shared_network_weights[1136u + i];
            uint weightOffset = 1088u + i * 16u;
//          uint weightOffset = 1088u + i * 16u;
            [ForceUnroll]
//          [ForceUnroll]
            for (uint j = 0u; j < 16u; j++) {
//          for (uint j = 0u; j < 16u; j++) {
                activationSum += hiddenLayer4[j] * shared_network_weights[weightOffset + j];
//              activationSum += hiddenLayer4[j] * shared_network_weights[weightOffset + j];
            }
//          }
            outputRadiance[int(i)] = activationSum;
//          outputRadiance[int(i)] = activationSum;
        }
//      }

        // Clamp to prevent explosive feedback during early training
//      // Clamp to prevent explosive feedback during early training
        return clamp(outputRadiance, float3(0.0), float3(15.0));
//      return clamp(outputRadiance, float3(0.0), float3(15.0));
    }
//  }

    // Variable Rate Tracing (VRT) rate classification and temporal allocation.
//  // Variable Rate Tracing (VRT) rate classification and temporal allocation.
    // The rate encodes how many pixels share a single traced ray: 1x1 spends the full ray budget,
//  // The rate encodes how many pixels share a single traced ray: 1x1 spends the full ray budget,
    // 4x4 spends 1/16th of it, and the remaining pixels in a tile live off the ReSTIR history instead.
//  // 4x4 spends 1/16th of it, and the remaining pixels in a tile live off the ReSTIR history instead.
    uint classifyPixelTracingRate(float3 geometricNormal, float roughness, float metallic, float transmission, float emissive) {
//  uint classifyPixelTracingRate(float3 geometricNormal, float roughness, float metallic, float transmission, float emissive) {
        if (uVRTEnabled == 0) {
//      if (uVRTEnabled == 0) {
            return VRT_RATE_1X1;
//          return VRT_RATE_1X1;
        }
//      }

        // Specular reflections, glass transmission and smooth surfaces carry high-frequency detail: full rate.
//      // Specular reflections, glass transmission and smooth surfaces carry high-frequency detail: full rate.
        if (metallic > 0.1 || transmission > 0.0 || roughness < 0.25) {
//      if (metallic > 0.1 || transmission > 0.0 || roughness < 0.25) {
            return VRT_RATE_1X1;
//          return VRT_RATE_1X1;
        }
//      }

        // Direct emitters: full rate.
//      // Direct emitters: full rate.
        if (emissive > 0.0) {
//      if (emissive > 0.0) {
            return VRT_RATE_1X1;
//          return VRT_RATE_1X1;
        }
//      }

        // Planar diffuse geometry (floors, walls, ceilings) tolerates the coarsest rate.
//      // Planar diffuse geometry (floors, walls, ceilings) tolerates the coarsest rate.
        bool isCardinalPlane = (abs(geometricNormal.x) > 0.95 || abs(geometricNormal.y) > 0.95 || abs(geometricNormal.z) > 0.95);
//      bool isCardinalPlane = (abs(geometricNormal.x) > 0.95 || abs(geometricNormal.y) > 0.95 || abs(geometricNormal.z) > 0.95);

        if (roughness >= 0.5 && isCardinalPlane) {
//      if (roughness >= 0.5 && isCardinalPlane) {
            return VRT_RATE_4X4;
//          return VRT_RATE_4X4;
        }
//      }

        if (roughness >= 0.35) {
//      if (roughness >= 0.35) {
            return VRT_RATE_2X2;
//          return VRT_RATE_2X2;
        }
//      }

        if (abs(geometricNormal.y) > 0.9) {
//      if (abs(geometricNormal.y) > 0.9) {
            return VRT_RATE_2X1;
//          return VRT_RATE_2X1;
        }
//      }

        return VRT_RATE_2X2;
//      return VRT_RATE_2X2;
    }
//  }

    // Rotates which pixel inside a tile owns the ray this frame, so every pixel is refreshed over time.
//  // Rotates which pixel inside a tile owns the ray this frame, so every pixel is refreshed over time.
    bool isPixelActiveForVRT(uint pixelX, uint pixelY, uint rate, uint frameIndex) {
//  bool isPixelActiveForVRT(uint pixelX, uint pixelY, uint rate, uint frameIndex) {
        if (rate == VRT_RATE_2X1) {
//      if (rate == VRT_RATE_2X1) {
            uint tileId = pixelX / 2u;
//          uint tileId = pixelX / 2u;
            uint activeOffset = (frameIndex + tileId) % 2u;
//          uint activeOffset = (frameIndex + tileId) % 2u;
            return (pixelX % 2u) == activeOffset;
//          return (pixelX % 2u) == activeOffset;
        }
//      }
        if (rate == VRT_RATE_1X2) {
//      if (rate == VRT_RATE_1X2) {
            uint tileId = pixelY / 2u;
//          uint tileId = pixelY / 2u;
            uint activeOffset = (frameIndex + tileId) % 2u;
//          uint activeOffset = (frameIndex + tileId) % 2u;
            return (pixelY % 2u) == activeOffset;
//          return (pixelY % 2u) == activeOffset;
        }
//      }
        if (rate == VRT_RATE_2X2) {
//      if (rate == VRT_RATE_2X2) {
            uint tileX = pixelX / 2u;
//          uint tileX = pixelX / 2u;
            uint tileY = pixelY / 2u;
//          uint tileY = pixelY / 2u;
            uint subPixel = (pixelY % 2u) * 2u + (pixelX % 2u);
//          uint subPixel = (pixelY % 2u) * 2u + (pixelX % 2u);
            uint dither = (tileX * 3u + tileY * 7u) % 4u;
//          uint dither = (tileX * 3u + tileY * 7u) % 4u;
            uint activeSub = (frameIndex + dither) % 4u;
//          uint activeSub = (frameIndex + dither) % 4u;
            return subPixel == activeSub;
//          return subPixel == activeSub;
        }
//      }
        if (rate == VRT_RATE_4X4) {
//      if (rate == VRT_RATE_4X4) {
            uint tileX = pixelX / 4u;
//          uint tileX = pixelX / 4u;
            uint tileY = pixelY / 4u;
//          uint tileY = pixelY / 4u;
            uint subPixel = (pixelY % 4u) * 4u + (pixelX % 4u);
//          uint subPixel = (pixelY % 4u) * 4u + (pixelX % 4u);
            uint dither = (tileX * 7u + tileY * 11u) % 16u;
//          uint dither = (tileX * 7u + tileY * 11u) % 16u;
            uint activeSub = (frameIndex * 3u + dither) % 16u;
//          uint activeSub = (frameIndex * 3u + dither) % 16u;
            return subPixel == activeSub;
//          return subPixel == activeSub;
        }
//      }
        return true;
//      return true;
    }
//  }

    // Heatmap colours matching the Experiment001 legend: red 1x1, yellow 2x1/1x2, green 2x2, blue 4x4.
//  // Heatmap colours matching the Experiment001 legend: red 1x1, yellow 2x1/1x2, green 2x2, blue 4x4.
    float3 visualizeTracingRate(uint rate, bool isActiveTrace) {
//  float3 visualizeTracingRate(uint rate, bool isActiveTrace) {
        float3 heatColor = float3(0.9, 0.15, 0.15);
//      float3 heatColor = float3(0.9, 0.15, 0.15);
        if (rate == VRT_RATE_2X1 || rate == VRT_RATE_1X2) {
//      if (rate == VRT_RATE_2X1 || rate == VRT_RATE_1X2) {
            heatColor = float3(0.95, 0.85, 0.1);
//          heatColor = float3(0.95, 0.85, 0.1);
        } else if (rate == VRT_RATE_2X2) {
//      } else if (rate == VRT_RATE_2X2) {
            heatColor = float3(0.1, 0.85, 0.25);
//          heatColor = float3(0.1, 0.85, 0.25);
        } else if (rate == VRT_RATE_4X4) {
//      } else if (rate == VRT_RATE_4X4) {
            heatColor = float3(0.15, 0.55, 1.0);
//          heatColor = float3(0.15, 0.55, 1.0);
        }
//      }
        if (isActiveTrace) {
//      if (isActiveTrace) {
            heatColor = min(heatColor + float3(0.3, 0.3, 0.3), float3(1.0));
//          heatColor = min(heatColor + float3(0.3, 0.3, 0.3), float3(1.0));
        } else {
//      } else {
            heatColor *= 0.55;
//          heatColor *= 0.55;
        }
//      }
        return heatColor;
//      return heatColor;
    }
//  }
    LightSample generateLightSampleCandidate(float3 hitPoint) {
//  LightSample generateLightSampleCandidate(float3 hitPoint) {
        LightSample candidate;
//      LightSample candidate;
        candidate.position = float3(0.0);
//      candidate.position = float3(0.0);
        candidate.padding1 = 0.0;
//      candidate.padding1 = 0.0;
        candidate.normal = float3(0.0, 1.0, 0.0);
//      candidate.normal = float3(0.0, 1.0, 0.0);
        candidate.padding2 = 0.0;
//      candidate.padding2 = 0.0;
        candidate.emission = float3(0.0);
//      candidate.emission = float3(0.0);
        candidate.probability_density = 0.0;
//      candidate.probability_density = 0.0;

        if (uPointLightCount <= 0) return candidate;
//      if (uPointLightCount <= 0) return candidate;

        float lightSelectionRandom = rand();
//      float lightSelectionRandom = rand();
        int selectedLightIndex = max(0, uPointLightCount - 1);
//      int selectedLightIndex = max(0, uPointLightCount - 1);
        for (int lightSearchIndex = 0; lightSearchIndex < uPointLightCount; lightSearchIndex++) {
//      for (int lightSearchIndex = 0; lightSearchIndex < uPointLightCount; lightSearchIndex++) {
            if (lightSelectionRandom <= uPointLights[lightSearchIndex].cdf) {
//          if (lightSelectionRandom <= uPointLights[lightSearchIndex].cdf) {
                selectedLightIndex = lightSearchIndex;
//              selectedLightIndex = lightSearchIndex;
                break;
//              break;
            }
//          }
        }
//      }

        float3 lightPosition = uPointLights[selectedLightIndex].position;
//      float3 lightPosition = uPointLights[selectedLightIndex].position;
        float lightRadius = uPointLights[selectedLightIndex].radius;
//      float lightRadius = uPointLights[selectedLightIndex].radius;
        float lightPdf = uPointLights[selectedLightIndex].pdf;
//      float lightPdf = uPointLights[selectedLightIndex].pdf;

        float3 lightToSurfaceDirection = normalize(hitPoint - lightPosition);
//      float3 lightToSurfaceDirection = normalize(hitPoint - lightPosition);
        float3 lightSampleNormal = randomUnitVector();
//      float3 lightSampleNormal = randomUnitVector();
        if (dot(lightSampleNormal, lightToSurfaceDirection) < 0.0) lightSampleNormal = -lightSampleNormal;
//      if (dot(lightSampleNormal, lightToSurfaceDirection) < 0.0) lightSampleNormal = -lightSampleNormal;
        float3 sampledLightPoint = lightPosition + lightSampleNormal * lightRadius;
//      float3 sampledLightPoint = lightPosition + lightSampleNormal * lightRadius;

        float lightSurfaceArea = 2.0 * PI * lightRadius * lightRadius;
//      float lightSurfaceArea = 2.0 * PI * lightRadius * lightRadius;
        float pdfLightArea = lightPdf / max(lightSurfaceArea, 1.0e-5);
//      float pdfLightArea = lightPdf / max(lightSurfaceArea, 1.0e-5);

        candidate.position = sampledLightPoint;
//      candidate.position = sampledLightPoint;
        candidate.normal = lightSampleNormal;
//      candidate.normal = lightSampleNormal;
        candidate.emission = uPointLights[selectedLightIndex].color;
//      candidate.emission = uPointLights[selectedLightIndex].color;
        candidate.probability_density = pdfLightArea;
//      candidate.probability_density = pdfLightArea;
        return candidate;
//      return candidate;
    }
//  }

    float evaluateTargetPdf(LightSample lightSample, float3 hitPoint, float3 shadingNormal, float3 viewDirection, float3 albedo, float roughness, float metallic, float transmission) {
//  float evaluateTargetPdf(LightSample lightSample, float3 hitPoint, float3 shadingNormal, float3 viewDirection, float3 albedo, float roughness, float metallic, float transmission) {
        float3 toLight = lightSample.position - hitPoint;
//      float3 toLight = lightSample.position - hitPoint;
        float squaredDistance = dot(toLight, toLight);
//      float squaredDistance = dot(toLight, toLight);
        if (squaredDistance < 1.0e-6) return 0.0;
//      if (squaredDistance < 1.0e-6) return 0.0;

        float distanceToLight = sqrt(squaredDistance);
//      float distanceToLight = sqrt(squaredDistance);
        float3 lightDirection = toLight / distanceToLight;
//      float3 lightDirection = toLight / distanceToLight;
        float normalDotLight = max(0.0, dot(shadingNormal, lightDirection));
//      float normalDotLight = max(0.0, dot(shadingNormal, lightDirection));
        float lightNormalDotDirection = max(0.0, dot(lightSample.normal, -lightDirection));
//      float lightNormalDotDirection = max(0.0, dot(lightSample.normal, -lightDirection));

        if (normalDotLight > 0.0 && lightNormalDotDirection > 0.0) {
//      if (normalDotLight > 0.0 && lightNormalDotDirection > 0.0) {
            float unusedPdf;
//          float unusedPdf;
            float3 brdfValue = evalPrincipledBSDFAndPDF(-viewDirection, lightDirection, shadingNormal, albedo, roughness, metallic, transmission, unusedPdf);
//          float3 brdfValue = evalPrincipledBSDFAndPDF(-viewDirection, lightDirection, shadingNormal, albedo, roughness, metallic, transmission, unusedPdf);
            float3 targetColor = brdfValue * normalDotLight * lightSample.emission * (lightNormalDotDirection / squaredDistance);
//          float3 targetColor = brdfValue * normalDotLight * lightSample.emission * (lightNormalDotDirection / squaredDistance);
            return calculateLuminance(targetColor);
//          return calculateLuminance(targetColor);
        }
//      }
        return 0.0;
//      return 0.0;
    }
//  }

    void updateReservoirSample(inout Reservoir reservoir, LightSample candidate, float weight, float countMultiplier) {
//  void updateReservoirSample(inout Reservoir reservoir, LightSample candidate, float weight, float countMultiplier) {
        if (weight <= 0.0 || isnan(weight) || isinf(weight)) return;
//      if (weight <= 0.0 || isnan(weight) || isinf(weight)) return;
        reservoir.sum_of_weights += weight;
//      reservoir.sum_of_weights += weight;
        reservoir.sample_count += countMultiplier;
//      reservoir.sample_count += countMultiplier;
        if (rand() * reservoir.sum_of_weights < weight) {
//      if (rand() * reservoir.sum_of_weights < weight) {
            reservoir.selected_sample = candidate;
//          reservoir.selected_sample = candidate;
        }
//      }
    }
//  }

    // Stochastic Importance Sampling: chooses a secondary ray direction by first stochastically picking a lobe — Fresnel-weighted specular, diffuse, or transmission — then sampling that lobe according to its matching BSDF PDF: GGX half-vectors for specular reflection and refraction, a cosine-weighted hemisphere for diffuse. Concentrating samples where the integrand is large minimizes Monte Carlo variance. It also resolves Walter-style microfacet refraction with a total-internal-reflection fallback, returns the path throughput f·cos θ / pdf folded into a single attenuation, and flags delta (perfectly specular/transmissive) bounces so the integrator skips Next Event Estimation on them.
//  // Stochastic Importance Sampling: chooses a secondary ray direction by first stochastically picking a lobe — Fresnel-weighted specular, diffuse, or transmission — then sampling that lobe according to its matching BSDF PDF: GGX half-vectors for specular reflection and refraction, a cosine-weighted hemisphere for diffuse. Concentrating samples where the integrand is large minimizes Monte Carlo variance. It also resolves Walter-style microfacet refraction with a total-internal-reflection fallback, returns the path throughput f·cos θ / pdf folded into a single attenuation, and flags delta (perfectly specular/transmissive) bounces so the integrator skips Next Event Estimation on them.
    bool scatterPrincipled(Ray incomingRay, RayHitResult hitResult, Material material, float pathRoughnessFloor, out float3 outAlbedo, out float outRoughness, out float outMetallic, out float outTransmission, out float3 outShadingNormal, out float3 outEmission, out float3 outScatteredDirection, out float3 outAttenuation, out float outPdf, out bool outIsDelta) {
//  bool scatterPrincipled(Ray incomingRay, RayHitResult hitResult, Material material, float pathRoughnessFloor, out float3 outAlbedo, out float outRoughness, out float outMetallic, out float outTransmission, out float3 outShadingNormal, out float3 outEmission, out float3 outScatteredDirection, out float3 outAttenuation, out float outPdf, out bool outIsDelta) {
        outScatteredDirection = float3(0.0);
//      outScatteredDirection = float3(0.0);
        outAttenuation = float3(0.0);
//      outAttenuation = float3(0.0);
        outEmission = float3(0.0);
//      outEmission = float3(0.0);
        outPdf = 0.0;
//      outPdf = 0.0;
        outIsDelta = false;
//      outIsDelta = false;
        // Seeded from the raw scalar so the value is defined on every early return below; the packed
//      // Seeded from the raw scalar so the value is defined on every early return below; the packed
        // ORM fetch further down overwrites it with the texture-resolved value where a map exists.
//      // ORM fetch further down overwrites it with the texture-resolved value where a map exists.
        outTransmission = material.transmission;
//      outTransmission = material.transmission;

        // Check texture indices
//      // Check texture indices
        float3 albedo = material.albedo.rgb;
//      float3 albedo = material.albedo.rgb;
        float roughness = material.roughness;
//      float roughness = material.roughness;
        float metallic = material.metallic;
//      float metallic = material.metallic;

        // Texture Sampling (using textureLod for Compute Shader safety)
//      // Texture Sampling (using textureLod for Compute Shader safety)
        float textureLodLevel = max(0.0, log2(hitResult.hitDistance * 0.1));
//      float textureLodLevel = max(0.0, log2(hitResult.hitDistance * 0.1));
        float2 scaledTexcoord = hitResult.hitSurfaceTexcoord * material.uvScale;
//      float2 scaledTexcoord = hitResult.hitSurfaceTexcoord * material.uvScale;

        if (material.textureIndexAlbedo > -0.5) {
//      if (material.textureIndexAlbedo > -0.5) {
            // Albedo is stored sRGB-encoded in 8-bit; decode to linear here
//          // Albedo is stored sRGB-encoded in 8-bit; decode to linear here
            albedo *= pow(uSceneTextureArray.SampleLevel(float3(scaledTexcoord, material.textureIndexAlbedo), textureLodLevel).rgb, float3(2.2));
//          albedo *= pow(uSceneTextureArray.SampleLevel(float3(scaledTexcoord, material.textureIndexAlbedo), textureLodLevel).rgb, float3(2.2));
        }
//      }
        if (material.textureIndexRoughness > -0.5) {
//      if (material.textureIndexRoughness > -0.5) {
            // Packed ORM layer: roughness is in the R channel
//          // Packed ORM layer: roughness is in the R channel
            roughness = uSceneTextureArray.SampleLevel(float3(scaledTexcoord, material.textureIndexRoughness), textureLodLevel).r;
//          roughness = uSceneTextureArray.SampleLevel(float3(scaledTexcoord, material.textureIndexRoughness), textureLodLevel).r;
        }
//      }
        if (material.textureIndexMetallic > -0.5) {
//      if (material.textureIndexMetallic > -0.5) {
            // Packed ORM layer: metallic is in the G channel
//          // Packed ORM layer: metallic is in the G channel
            metallic = uSceneTextureArray.SampleLevel(float3(scaledTexcoord, material.textureIndexMetallic), textureLodLevel).g;
//          metallic = uSceneTextureArray.SampleLevel(float3(scaledTexcoord, material.textureIndexMetallic), textureLodLevel).g;
        }
//      }

        if (material.textureIndexTransmission > -0.5) {
//      if (material.textureIndexTransmission > -0.5) {
            // Packed ORM layer: transmission is in the B channel
//          // Packed ORM layer: transmission is in the B channel
            material.transmission *= uSceneTextureArray.SampleLevel(float3(scaledTexcoord, material.textureIndexTransmission), textureLodLevel).b;
//          material.transmission *= uSceneTextureArray.SampleLevel(float3(scaledTexcoord, material.textureIndexTransmission), textureLodLevel).b;
        }
//      }
        // Hand the resolved value back. computeMain() gates ReSTIR and the Neural Radiance Cache on it, and
//      // Hand the resolved value back. computeMain() gates ReSTIR and the Neural Radiance Cache on it, and
        // material.transmission there is only the raw scalar -- a 1.0 placeholder on any material that
//      // material.transmission there is only the raw scalar -- a 1.0 placeholder on any material that
        // carries a transmission map, exactly the trap already fixed for metallic and roughness.
//      // carries a transmission map, exactly the trap already fixed for metallic and roughness.
        outTransmission = material.transmission;
//      outTransmission = material.transmission;

        // Roughness annealing (SDS variance mitigation) is applied HERE, after the texture fetch: the old
//      // Roughness annealing (SDS variance mitigation) is applied HERE, after the texture fetch: the old
        // pre-scatter clamp in computeMain() was silently overwritten whenever a roughness texture was bound,
//      // pre-scatter clamp in computeMain() was silently overwritten whenever a roughness texture was bound,
        // so textured surfaces never annealed. Also clamp to MIN_ROUGHNESS to prevent division by zero
//      // so textured surfaces never annealed. Also clamp to MIN_ROUGHNESS to prevent division by zero
        // in the specular calculations.
//      // in the specular calculations.
        roughness = max(max(roughness, pathRoughnessFloor), MIN_ROUGHNESS);
//      roughness = max(max(roughness, pathRoughnessFloor), MIN_ROUGHNESS);

        if (material.textureIndexEmissive > -0.5) {
//      if (material.textureIndexEmissive > -0.5) {
            // Emissive is stored sRGB-encoded in 8-bit; decode to linear here
//          // Emissive is stored sRGB-encoded in 8-bit; decode to linear here
            outEmission = material.emissive * pow(uSceneTextureArray.SampleLevel(float3(scaledTexcoord, material.textureIndexEmissive), textureLodLevel).rgb, float3(2.2));
//          outEmission = material.emissive * pow(uSceneTextureArray.SampleLevel(float3(scaledTexcoord, material.textureIndexEmissive), textureLodLevel).rgb, float3(2.2));
        } else {
//      } else {
            outEmission = material.emissive * albedo;
//          outEmission = material.emissive * albedo;
        }
//      }

        if (outEmission.r > 0.0 || outEmission.g > 0.0 || outEmission.b > 0.0) {
//      if (outEmission.r > 0.0 || outEmission.g > 0.0 || outEmission.b > 0.0) {
            return false;
//          return false;
        }
//      }

        // Principled Logic
//      // Principled Logic

        // 1. Shading Normal Calculation (Normal Mapping)
//      // 1. Shading Normal Calculation (Normal Mapping)
        float3 shadingNormal = hitResult.hitSurfaceNormal;
//      float3 shadingNormal = hitResult.hitSurfaceNormal;

        if (material.textureIndexNormal > -0.5) {
//      if (material.textureIndexNormal > -0.5) {
            float3 tangentSpaceNormal = uSceneTextureArray.SampleLevel(float3(scaledTexcoord, material.textureIndexNormal), textureLodLevel).rgb;
//          float3 tangentSpaceNormal = uSceneTextureArray.SampleLevel(float3(scaledTexcoord, material.textureIndexNormal), textureLodLevel).rgb;
            tangentSpaceNormal = tangentSpaceNormal * 2.0 - 1.0;
//          tangentSpaceNormal = tangentSpaceNormal * 2.0 - 1.0;

            float3 normalizedShadingNormal = normalize(shadingNormal);
//          float3 normalizedShadingNormal = normalize(shadingNormal);
            float3 tangentVector = normalize(hitResult.hitSurfaceTangent);
//          float3 tangentVector = normalize(hitResult.hitSurfaceTangent);
            // Re-orthogonalize the tangent with respect to the shading normal
//          // Re-orthogonalize the tangent with respect to the shading normal
            float3 orthogonalizedTangent = tangentVector - dot(tangentVector, normalizedShadingNormal) * normalizedShadingNormal;
//          float3 orthogonalizedTangent = tangentVector - dot(tangentVector, normalizedShadingNormal) * normalizedShadingNormal;
            if (dot(orthogonalizedTangent, orthogonalizedTangent) > EPSILON_MATH) {
//          if (dot(orthogonalizedTangent, orthogonalizedTangent) > EPSILON_MATH) {
                tangentVector = normalize(orthogonalizedTangent);
//              tangentVector = normalize(orthogonalizedTangent);
            } else {
//          } else {
                // Fallback to avoid NaN if the tangent and normal are collinear
//              // Fallback to avoid NaN if the tangent and normal are collinear
                float3 basisHelper;
//              float3 basisHelper;
                if (abs(normalizedShadingNormal.z) < 0.999) {
//              if (abs(normalizedShadingNormal.z) < 0.999) {
                    basisHelper = float3(0.0, 0.0, 1.0);
//                  basisHelper = float3(0.0, 0.0, 1.0);
                } else {
//              } else {
                    basisHelper = float3(1.0, 0.0, 0.0);
//                  basisHelper = float3(1.0, 0.0, 0.0);
                }
//              }
                tangentVector = normalize(cross(basisHelper, normalizedShadingNormal));
//              tangentVector = normalize(cross(basisHelper, normalizedShadingNormal));
            }
//          }
            float3 bitangentVector = cross(normalizedShadingNormal, tangentVector);
//          float3 bitangentVector = cross(normalizedShadingNormal, tangentVector);
            // Slang matrix constructors take rows (GLSL mat3 takes columns), so the TBN basis vectors are laid in as rows and the tangent-space normal is multiplied from the left, which places them as the columns of the tangent-to-world transform.
//          // Slang matrix constructors take rows (GLSL mat3 takes columns), so the TBN basis vectors are laid in as rows and the tangent-space normal is multiplied from the left, which places them as the columns of the tangent-to-world transform.
            float3x3 tangentToWorldMatrix = float3x3(tangentVector, bitangentVector, normalizedShadingNormal);
//          float3x3 tangentToWorldMatrix = float3x3(tangentVector, bitangentVector, normalizedShadingNormal);

            shadingNormal = normalize(mul(tangentSpaceNormal, tangentToWorldMatrix));
//          shadingNormal = normalize(mul(tangentSpaceNormal, tangentToWorldMatrix));
        }
//      }

        // Store material properties for NEE
//      // Store material properties for NEE
        outAlbedo = albedo;
//      outAlbedo = albedo;
        outRoughness = roughness;
//      outRoughness = roughness;
        outMetallic = metallic;
//      outMetallic = metallic;
        // F0 calculation: 0.04 for dielectrics, albedo for metals
//      // F0 calculation: 0.04 for dielectrics, albedo for metals
        float3 reflectanceAtNormalIncidence = lerp(float3(F0_DEFAULT), albedo, metallic);
//      float3 reflectanceAtNormalIncidence = lerp(float3(F0_DEFAULT), albedo, metallic);

        // Fix normal map pointing away from incoming ray to prevent black fringes
//      // Fix normal map pointing away from incoming ray to prevent black fringes
        if (dot(shadingNormal, -incomingRay.direction) <= 0.0) {
//      if (dot(shadingNormal, -incomingRay.direction) <= 0.0) {
            shadingNormal = hitResult.hitSurfaceNormal;
//          shadingNormal = hitResult.hitSurfaceNormal;
        }
//      }

        outShadingNormal = shadingNormal;
//      outShadingNormal = shadingNormal;

        // Schlick's fresnel approximation at incident angle
//      // Schlick's fresnel approximation at incident angle
        float cosIncidentAngle = clamp(dot(-incomingRay.direction, shadingNormal), 0.0, 1.0);
//      float cosIncidentAngle = clamp(dot(-incomingRay.direction, shadingNormal), 0.0, 1.0);
        float3 fresnelReflectance = schlickFresnel(cosIncidentAngle, reflectanceAtNormalIncidence);
//      float3 fresnelReflectance = schlickFresnel(cosIncidentAngle, reflectanceAtNormalIncidence);
        // Use average fresnel for importance sampling probability
//      // Use average fresnel for importance sampling probability
        float averageFresnel = (fresnelReflectance.r + fresnelReflectance.g + fresnelReflectance.b) / 3.0;
//      float averageFresnel = (fresnelReflectance.r + fresnelReflectance.g + fresnelReflectance.b) / 3.0;

        float randomLobeSelector = rand();
//      float randomLobeSelector = rand();

        // --- PATH A: METALLIC REFLECTION ---
//      // --- PATH A: METALLIC REFLECTION ---
        // Clamp specular probability to minimum 15% to eliminate dielectric fireflies caused by extremely low 1/PDF divisions
//      // Clamp specular probability to minimum 15% to eliminate dielectric fireflies caused by extremely low 1/PDF divisions
        float specularSelectionProbability = max(lerp(averageFresnel, 1.0, metallic), 0.15);
//      float specularSelectionProbability = max(lerp(averageFresnel, 1.0, metallic), 0.15);

        if (randomLobeSelector < specularSelectionProbability) {
//      if (randomLobeSelector < specularSelectionProbability) {
            // SPECULAR REFLECTION (METAL OR DIELECTRIC COAT)
//          // SPECULAR REFLECTION (METAL OR DIELECTRIC COAT)
            float3 microfacetNormal = sampleGGX(shadingNormal, roughness);
//          float3 microfacetNormal = sampleGGX(shadingNormal, roughness);
            float3 specularReflectedDirection = reflectPrincipled(incomingRay.direction, microfacetNormal);
//          float3 specularReflectedDirection = reflectPrincipled(incomingRay.direction, microfacetNormal);

            if (dot(specularReflectedDirection, hitResult.hitSurfaceNormal) <= 0.0) {
//          if (dot(specularReflectedDirection, hitResult.hitSurfaceNormal) <= 0.0) {
                specularReflectedDirection = normalize(specularReflectedDirection - 2.0 * dot(specularReflectedDirection, hitResult.hitSurfaceNormal) * hitResult.hitSurfaceNormal);
//              specularReflectedDirection = normalize(specularReflectedDirection - 2.0 * dot(specularReflectedDirection, hitResult.hitSurfaceNormal) * hitResult.hitSurfaceNormal);
            }
//          }
            if (dot(specularReflectedDirection, shadingNormal) <= 0.0) {
//          if (dot(specularReflectedDirection, shadingNormal) <= 0.0) {
                specularReflectedDirection = normalize(specularReflectedDirection + hitResult.hitSurfaceNormal * 0.2);
//              specularReflectedDirection = normalize(specularReflectedDirection + hitResult.hitSurfaceNormal * 0.2);
            }
//          }
            outScatteredDirection = specularReflectedDirection;
//          outScatteredDirection = specularReflectedDirection;
            float pdfValue;
//          float pdfValue;
            float3 bsdfValue = evalPrincipledBSDFAndPDF(incomingRay.direction, specularReflectedDirection, shadingNormal, albedo, roughness, metallic, material.transmission, pdfValue);
//          float3 bsdfValue = evalPrincipledBSDFAndPDF(incomingRay.direction, specularReflectedDirection, shadingNormal, albedo, roughness, metallic, material.transmission, pdfValue);
            float cosOutgoingAngle = max(dot(shadingNormal, specularReflectedDirection), 0.0);
//          float cosOutgoingAngle = max(dot(shadingNormal, specularReflectedDirection), 0.0);
            outAttenuation = bsdfValue * cosOutgoingAngle / max(pdfValue, EPSILON_MATH);
//          outAttenuation = bsdfValue * cosOutgoingAngle / max(pdfValue, EPSILON_MATH);
            outIsDelta = false;
//          outIsDelta = false;
            outPdf = pdfValue;
//          outPdf = pdfValue;
        } else {
//      } else {
            // --- PATH B: DIELECTRIC (DIFFUSE OR TRANSMISSION) ---
//          // --- PATH B: DIELECTRIC (DIFFUSE OR TRANSMISSION) ---

            // Re-normalize random variable for the next choice
//          // Re-normalize random variable for the next choice
            float remappedRandom = (randomLobeSelector - specularSelectionProbability) / (1.0 - specularSelectionProbability);
//          float remappedRandom = (randomLobeSelector - specularSelectionProbability) / (1.0 - specularSelectionProbability);

            if (material.transmission > 0.0 && remappedRandom < material.transmission) {
//          if (material.transmission > 0.0 && remappedRandom < material.transmission) {
                // TRANSMISSION (REFRACTION)
//              // TRANSMISSION (REFRACTION)
                float etaRatioOfIncidenceOverTransmission = 1.0 / material.ior;
//              float etaRatioOfIncidenceOverTransmission = 1.0 / material.ior;
                if (!hitResult.isFrontFaceHit) {
//              if (!hitResult.isFrontFaceHit) {
                    etaRatioOfIncidenceOverTransmission = material.ior;
//                  etaRatioOfIncidenceOverTransmission = material.ior;
                }
//              }

                float3 microfacetNormal = sampleGGX(shadingNormal, roughness);
//              float3 microfacetNormal = sampleGGX(shadingNormal, roughness);

                float cosIncidentAngleMicro = dot(-incomingRay.direction, microfacetNormal);
//              float cosIncidentAngleMicro = dot(-incomingRay.direction, microfacetNormal);
                if (cosIncidentAngleMicro <= 0.0) return false;
//              if (cosIncidentAngleMicro <= 0.0) return false;
                float sinTransmittedAngleSquared = (1.0 - cosIncidentAngleMicro * cosIncidentAngleMicro) * (etaRatioOfIncidenceOverTransmission * etaRatioOfIncidenceOverTransmission);
//              float sinTransmittedAngleSquared = (1.0 - cosIncidentAngleMicro * cosIncidentAngleMicro) * (etaRatioOfIncidenceOverTransmission * etaRatioOfIncidenceOverTransmission);

                float3 refractedDirection = refractPrincipled(incomingRay.direction, microfacetNormal, etaRatioOfIncidenceOverTransmission, cosIncidentAngleMicro, sinTransmittedAngleSquared);
//              float3 refractedDirection = refractPrincipled(incomingRay.direction, microfacetNormal, etaRatioOfIncidenceOverTransmission, cosIncidentAngleMicro, sinTransmittedAngleSquared);
                float3 reflectedDirection = reflectPrincipled(incomingRay.direction, microfacetNormal);
//              float3 reflectedDirection = reflectPrincipled(incomingRay.direction, microfacetNormal);

                // When [ sinTransmittedAngleSquared <= 1.0 ] then Refraction happened else Total Internal Reflection happened
//              // When [ sinTransmittedAngleSquared <= 1.0 ] then Refraction happened else Total Internal Reflection happened

                // Fresnel evaluated at the sampled microfacet angle (not the macro shading normal)
//              // Fresnel evaluated at the sampled microfacet angle (not the macro shading normal)
                float3 microfacetFresnel = schlickFresnel(cosIncidentAngleMicro, reflectanceAtNormalIncidence);
//              float3 microfacetFresnel = schlickFresnel(cosIncidentAngleMicro, reflectanceAtNormalIncidence);

                // NDF-sampled microfacet weight: G * |V.Hm| / (|N.V| * |N.Hm|), with the same Smith
//              // NDF-sampled microfacet weight: G * |V.Hm| / (|N.V| * |N.Hm|), with the same Smith
                // convention as the reflection BSDF (k = roughness^2 / 2). Degrades to ~1.0 at MIN_ROUGHNESS.
//              // convention as the reflection BSDF (k = roughness^2 / 2). Degrades to ~1.0 at MIN_ROUGHNESS.
                float smithGeometryK = (roughness * roughness) / 2.0;
//              float smithGeometryK = (roughness * roughness) / 2.0;
                float normalDotViewMicro = max(dot(shadingNormal, -incomingRay.direction), EPSILON_DOT);
//              float normalDotViewMicro = max(dot(shadingNormal, -incomingRay.direction), EPSILON_DOT);
                float normalDotHalfMicro = max(dot(shadingNormal, microfacetNormal), EPSILON_DOT);
//              float normalDotHalfMicro = max(dot(shadingNormal, microfacetNormal), EPSILON_DOT);
                float geometrySmithView = normalDotViewMicro / (normalDotViewMicro * (1.0 - smithGeometryK) + smithGeometryK);
//              float geometrySmithView = normalDotViewMicro / (normalDotViewMicro * (1.0 - smithGeometryK) + smithGeometryK);

                float branchProbability = (1.0 - specularSelectionProbability) * material.transmission;
//              float branchProbability = (1.0 - specularSelectionProbability) * material.transmission;

                if (sinTransmittedAngleSquared <= 1.0) {
//              if (sinTransmittedAngleSquared <= 1.0) {
                    outScatteredDirection = refractedDirection;
//                  outScatteredDirection = refractedDirection;
                    // Reflection-form weight is exact for refraction too: Walter 2007's eta_t^2, D(Hm)
//                  // Reflection-form weight is exact for refraction too: Walter 2007's eta_t^2, D(Hm)
                    // and the transmission half-vector Jacobian all cancel against the full-NDF sampling
//                  // and the transmission half-vector Jacobian all cancel against the full-NDF sampling
                    // pdf, leaving G*|V.Hm|/(|N.V|*|N.Hm|). normalDotLightMicro uses abs() since transmitted L is far-side.
//                  // pdf, leaving G*|V.Hm|/(|N.V|*|N.Hm|). normalDotLightMicro uses abs() since transmitted L is far-side.
                    float normalDotLightMicro = max(abs(dot(shadingNormal, refractedDirection)), EPSILON_DOT);
//                  float normalDotLightMicro = max(abs(dot(shadingNormal, refractedDirection)), EPSILON_DOT);
                    float geometrySmithLight = normalDotLightMicro / (normalDotLightMicro * (1.0 - smithGeometryK) + smithGeometryK);
//                  float geometrySmithLight = normalDotLightMicro / (normalDotLightMicro * (1.0 - smithGeometryK) + smithGeometryK);
                    float microfacetWeight = (geometrySmithView * geometrySmithLight) * cosIncidentAngleMicro / (normalDotViewMicro * normalDotHalfMicro);
//                  float microfacetWeight = (geometrySmithView * geometrySmithLight) * cosIncidentAngleMicro / (normalDotViewMicro * normalDotHalfMicro);
                    outAttenuation = (float3(1.0) - microfacetFresnel) * albedo * microfacetWeight / max(branchProbability, EPSILON_MATH);
//                  outAttenuation = (float3(1.0) - microfacetFresnel) * albedo * microfacetWeight / max(branchProbability, EPSILON_MATH);
                } else {
//              } else {
                    outScatteredDirection = reflectedDirection;
//                  outScatteredDirection = reflectedDirection;
                    float normalDotLightMicro = max(dot(shadingNormal, reflectedDirection), EPSILON_DOT);
//                  float normalDotLightMicro = max(dot(shadingNormal, reflectedDirection), EPSILON_DOT);
                    float geometrySmithLight = normalDotLightMicro / (normalDotLightMicro * (1.0 - smithGeometryK) + smithGeometryK);
//                  float geometrySmithLight = normalDotLightMicro / (normalDotLightMicro * (1.0 - smithGeometryK) + smithGeometryK);
                    float microfacetWeight = (geometrySmithView * geometrySmithLight) * cosIncidentAngleMicro / (normalDotViewMicro * normalDotHalfMicro);
//                  float microfacetWeight = (geometrySmithView * geometrySmithLight) * cosIncidentAngleMicro / (normalDotViewMicro * normalDotHalfMicro);
                    outAttenuation = float3(microfacetWeight) / max(branchProbability, EPSILON_MATH);
//                  outAttenuation = float3(microfacetWeight) / max(branchProbability, EPSILON_MATH);
                }
//              }
                outIsDelta = true; // No NEE for transmission
//              outIsDelta = true; // No NEE for transmission
                outPdf = 1.0;
//              outPdf = 1.0;
            } else {
//          } else {
                // DIFFUSE (LAMBERTIAN)
//              // DIFFUSE (LAMBERTIAN)
                float3 diffuseDirectionUnnormalized = shadingNormal + randomUnitVector();
//              float3 diffuseDirectionUnnormalized = shadingNormal + randomUnitVector();
                float3 diffuseDirection;
//              float3 diffuseDirection;
                if (dot(diffuseDirectionUnnormalized, diffuseDirectionUnnormalized) > EPSILON_MATH) {
//              if (dot(diffuseDirectionUnnormalized, diffuseDirectionUnnormalized) > EPSILON_MATH) {
                    diffuseDirection = normalize(diffuseDirectionUnnormalized);
//                  diffuseDirection = normalize(diffuseDirectionUnnormalized);
                } else {
//              } else {
                    diffuseDirection = shadingNormal;
//                  diffuseDirection = shadingNormal;
                }
//              }

                if (dot(diffuseDirection, hitResult.hitSurfaceNormal) <= 0.0) {
//              if (dot(diffuseDirection, hitResult.hitSurfaceNormal) <= 0.0) {
                    diffuseDirection = normalize(diffuseDirection - 2.0 * dot(diffuseDirection, hitResult.hitSurfaceNormal) * hitResult.hitSurfaceNormal);
//                  diffuseDirection = normalize(diffuseDirection - 2.0 * dot(diffuseDirection, hitResult.hitSurfaceNormal) * hitResult.hitSurfaceNormal);
                }
//              }
                if (dot(diffuseDirection, shadingNormal) <= 0.0) {
//              if (dot(diffuseDirection, shadingNormal) <= 0.0) {
                    diffuseDirection = normalize(diffuseDirection + hitResult.hitSurfaceNormal * 0.2);
//                  diffuseDirection = normalize(diffuseDirection + hitResult.hitSurfaceNormal * 0.2);
                }
//              }

                outScatteredDirection = diffuseDirection;
//              outScatteredDirection = diffuseDirection;
                float pdfValue;
//              float pdfValue;
                float3 bsdfValue = evalPrincipledBSDFAndPDF(incomingRay.direction, diffuseDirection, shadingNormal, albedo, roughness, metallic, material.transmission, pdfValue);
//              float3 bsdfValue = evalPrincipledBSDFAndPDF(incomingRay.direction, diffuseDirection, shadingNormal, albedo, roughness, metallic, material.transmission, pdfValue);
                float cosOutgoingAngle = max(dot(shadingNormal, diffuseDirection), 0.0);
//              float cosOutgoingAngle = max(dot(shadingNormal, diffuseDirection), 0.0);
                outAttenuation = bsdfValue * cosOutgoingAngle / max(pdfValue, EPSILON_MATH);
//              outAttenuation = bsdfValue * cosOutgoingAngle / max(pdfValue, EPSILON_MATH);
                outIsDelta = false;
//              outIsDelta = false;
                outPdf = pdfValue;
//              outPdf = pdfValue;
            }
//          }
        }
//      }
        return true;
//      return true;
    }
//  }

    struct SphereLightHit {
//  struct SphereLightHit {
        float distanceToHit;
//      float distanceToHit;
        int lightIndex;
//      int lightIndex;
    };
//  };

    SphereLightHit intersectSphereLights(Ray ray) {
//  SphereLightHit intersectSphereLights(Ray ray) {
        SphereLightHit closestLightHit;
//      SphereLightHit closestLightHit;
        closestLightHit.distanceToHit = INF;
//      closestLightHit.distanceToHit = INF;
        closestLightHit.lightIndex = -1;
//      closestLightHit.lightIndex = -1;

        for (int lightLoopIndex = 0; lightLoopIndex < uPointLightCount; lightLoopIndex++) {
//      for (int lightLoopIndex = 0; lightLoopIndex < uPointLightCount; lightLoopIndex++) {
            float3 originToCenter = ray.origin - uPointLights[lightLoopIndex].position;
//          float3 originToCenter = ray.origin - uPointLights[lightLoopIndex].position;
            float halfBCoefficient = dot(originToCenter, ray.direction);
//          float halfBCoefficient = dot(originToCenter, ray.direction);
            float centerDistanceTerm = dot(originToCenter, originToCenter) - uPointLights[lightLoopIndex].radius * uPointLights[lightLoopIndex].radius;
//          float centerDistanceTerm = dot(originToCenter, originToCenter) - uPointLights[lightLoopIndex].radius * uPointLights[lightLoopIndex].radius;
            float discriminant = halfBCoefficient * halfBCoefficient - centerDistanceTerm;
//          float discriminant = halfBCoefficient * halfBCoefficient - centerDistanceTerm;
            if (discriminant > 0.0) {
//          if (discriminant > 0.0) {
                float rootDistance = -halfBCoefficient - sqrt(discriminant);
//              float rootDistance = -halfBCoefficient - sqrt(discriminant);
                if (rootDistance <= EPSILON_OFFSET) {
//              if (rootDistance <= EPSILON_OFFSET) {
                    rootDistance = -halfBCoefficient + sqrt(discriminant);
//                  rootDistance = -halfBCoefficient + sqrt(discriminant);
                }
//              }
                if (rootDistance > EPSILON_OFFSET && rootDistance < closestLightHit.distanceToHit) {
//              if (rootDistance > EPSILON_OFFSET && rootDistance < closestLightHit.distanceToHit) {
                    closestLightHit.distanceToHit = rootDistance;
//                  closestLightHit.distanceToHit = rootDistance;
                    closestLightHit.lightIndex = lightLoopIndex;
//                  closestLightHit.lightIndex = lightLoopIndex;
                }
//              }
            }
//          }
        }
//      }
        return closestLightHit;
//      return closestLightHit;
    }
//  }

    [shader("compute")]
//  [shader("compute")]
    [numthreads(16, 16, 1)]
//  [numthreads(16, 16, 1)]
    void computeMain(uint3 globalInvocationID : SV_DispatchThreadID, uint localInvocationIndex : SV_GroupIndex) {
//  void computeMain(uint3 globalInvocationID : SV_DispatchThreadID, uint localInvocationIndex : SV_GroupIndex) {
        // Stage the NRC EMA weights in shared memory. This runs before the bounds check on purpose:
//      // Stage the NRC EMA weights in shared memory. This runs before the bounds check on purpose:
        // barrier() must be reached by every invocation in the workgroup, so no thread may return first.
//      // barrier() must be reached by every invocation in the workgroup, so no thread may return first.
        //
//      //
        // The copy is skipped outright when the cache is off. uNRCEnabled is a uniform, so this is
//      // The copy is skipped outright when the cache is off. uNRCEnabled is a uniform, so this is
        // uniform control flow across the workgroup and the barrier below stays legal; the barrier is
//      // uniform control flow across the workgroup and the barrier below stays legal; the barrier is
        // kept outside the guard so it is reached unconditionally either way. This saves 1139 scalar
//      // kept outside the guard so it is reached unconditionally either way. This saves 1139 scalar
        // SSBO loads per workgroup per frame whenever the Neural Radiance Cache is disabled, and it is
//      // SSBO loads per workgroup per frame whenever the Neural Radiance Cache is disabled, and it is
        // safe because the sole reader, evaluateNeuralRadianceCache, is only reached through
//      // safe because the sole reader, evaluateNeuralRadianceCache, is only reached through
        // nrcTerminatesPath, which already requires uNRCEnabled.
//      // nrcTerminatesPath, which already requires uNRCEnabled.
        if (uNRCEnabled != 0) {
//      if (uNRCEnabled != 0) {
            for (uint weightIndex = localInvocationIndex; weightIndex < 1139u; weightIndex += 256u) {
//          for (uint weightIndex = localInvocationIndex; weightIndex < 1139u; weightIndex += 256u) {
                shared_network_weights[weightIndex] = neuralNetworkWeights[1139u + weightIndex];
//              shared_network_weights[weightIndex] = neuralNetworkWeights[1139u + weightIndex];
            }
//          }
        }
//      }
        GroupMemoryBarrierWithGroupSync();
//      GroupMemoryBarrierWithGroupSync();

        int2 pixelCoordinates = int2(globalInvocationID.xy);
//      int2 pixelCoordinates = int2(globalInvocationID.xy);
        // Bounds-check against uResolution rather than the accumulation texture's dimensions. Both describe the
//      // Bounds-check against uResolution rather than the accumulation texture's dimensions. Both describe the
        // same extent and agree today only because the window is a fixed size, but they are two
//      // same extent and agree today only because the window is a fixed size, but they are two
        // independent sources of truth and only one of them is the one that matters here: every
//      // independent sources of truth and only one of them is the one that matters here: every
        // reservoir index below is built from uResolution, and ssbo_sample_reservoirs is sized for
//      // reservoir index below is built from uResolution, and ssbo_sample_reservoirs is sized for
        // exactly that many pixels. An out-of-range texture store is a defined no-op, whereas an
//      // exactly that many pixels. An out-of-range texture store is a defined no-op, whereas an
        // out-of-range SSBO write corrupts memory, so the guard is derived from the buffer's extent.
//      // out-of-range SSBO write corrupts memory, so the guard is derived from the buffer's extent.
        int2 dimensions = uResolution;
//      int2 dimensions = uResolution;

        if (pixelCoordinates.x >= dimensions.x || pixelCoordinates.y >= dimensions.y) {
//      if (pixelCoordinates.x >= dimensions.x || pixelCoordinates.y >= dimensions.y) {
            return;
//          return;
        }
//      }

        initRNG(float2(pixelCoordinates));
//      initRNG(float2(pixelCoordinates));

        float3 accumulatedColor = float3(0.0);
//      float3 accumulatedColor = float3(0.0);
        float3 attenuation = float3(1.0);
//      float3 attenuation = float3(1.0);

        // Primary Ray Initialization: reconstructs this pixel's world-space camera ray from the precomputed sensor basis — the top-left sample position plus per-pixel horizontal and vertical step vectors scaled by the integer pixel coordinate — then normalizes the vector from the camera origin through that point. A per-frame sub-pixel jitter nudges the sample within the pixel's footprint, so temporal accumulation effectively integrates over the entire pixel area, yielding anti-aliasing and smoother convergence without an explicit per-pixel multi-sample loop.
//      // Primary Ray Initialization: reconstructs this pixel's world-space camera ray from the precomputed sensor basis — the top-left sample position plus per-pixel horizontal and vertical step vectors scaled by the integer pixel coordinate — then normalizes the vector from the camera origin through that point. A per-frame sub-pixel jitter nudges the sample within the pixel's footprint, so temporal accumulation effectively integrates over the entire pixel area, yielding anti-aliasing and smoother convergence without an explicit per-pixel multi-sample loop.
        float3 pixelSampleCenter = uPixel00Coordinates + uPixelDeltaU * (float(pixelCoordinates.x) + uJitter.x) + uPixelDeltaV * (float(pixelCoordinates.y) + uJitter.y);
//      float3 pixelSampleCenter = uPixel00Coordinates + uPixelDeltaU * (float(pixelCoordinates.x) + uJitter.x) + uPixelDeltaV * (float(pixelCoordinates.y) + uJitter.y);

        Ray currentRay;
//      Ray currentRay;
        currentRay.origin = uCameraGlobalPosition;
//      currentRay.origin = uCameraGlobalPosition;
        currentRay.direction = normalize(pixelSampleCenter - uCameraGlobalPosition);
//      currentRay.direction = normalize(pixelSampleCenter - uCameraGlobalPosition);

        int maxDepth = MAX_BOUNCES;
//      int maxDepth = MAX_BOUNCES;

        float lastBrdfPdfSolidAngle = 1.0;
//      float lastBrdfPdfSolidAngle = 1.0;
        bool lastWasDelta = true;
//      bool lastWasDelta = true;
        // Tracker for roughness annealing to mitigate Specular-Diffuse-Specular (SDS) variance explosions
//      // Tracker for roughness annealing to mitigate Specular-Diffuse-Specular (SDS) variance explosions
        float pathRoughness = 0.0;
//      float pathRoughness = 0.0;
        float3 hitPoint = float3(0.0);
//      float3 hitPoint = float3(0.0);
        bool lastSkippedNEE = false;
//      bool lastSkippedNEE = false;
        bool lastRanReSTIR = false;
//      bool lastRanReSTIR = false;
        int diffuseBounceCount = 0;
//      int diffuseBounceCount = 0;

        for (int depth = 0; depth < maxDepth; depth++) {
//      for (int depth = 0; depth < maxDepth; depth++) {
            RayHitResult rayHitResult;
//          RayHitResult rayHitResult;

            if (depth == 0) {
//          if (depth == 0) {
                if (uReSTIREnabled != 0) {
//              if (uReSTIREnabled != 0) {
                    uint totalPixels = uint(uResolution.x * uResolution.y);
//                  uint totalPixels = uint(uResolution.x * uResolution.y);
                    uint currentFrameOffset = (uCacheFrameCounter % 2u) * totalPixels;
//                  uint currentFrameOffset = (uCacheFrameCounter % 2u) * totalPixels;
                    uint pixelIndex = uint(pixelCoordinates.y * uResolution.x + pixelCoordinates.x);
//                  uint pixelIndex = uint(pixelCoordinates.y * uResolution.x + pixelCoordinates.x);
                    sampleReservoirs[currentFrameOffset + pixelIndex].sample_count = 0.0;
//                  sampleReservoirs[currentFrameOffset + pixelIndex].sample_count = 0.0;
                }
//              }
                // Hybrid Start: Fetch base surface properties from G-Buffer to avoid first-hit traversal overhead
//              // Hybrid Start: Fetch base surface properties from G-Buffer to avoid first-hit traversal overhead
                float4 sampleGlobalPosition = textureGeometryGlobalPosition.Load(int3(pixelCoordinates, 0));
//              float4 sampleGlobalPosition = textureGeometryGlobalPosition.Load(int3(pixelCoordinates, 0));
                float4 sampleGlobalNormal = textureGeometryGlobalNormal.Load(int3(pixelCoordinates, 0));
//              float4 sampleGlobalNormal = textureGeometryGlobalNormal.Load(int3(pixelCoordinates, 0));
                float4 sampleGlobalTangent = textureGeometryGlobalTangent.Load(int3(pixelCoordinates, 0));
//              float4 sampleGlobalTangent = textureGeometryGlobalTangent.Load(int3(pixelCoordinates, 0));

                if (sampleGlobalPosition.w == 0.0) {
//              if (sampleGlobalPosition.w == 0.0) {
                    rayHitResult.hitTriangleIndex = -1;
//                  rayHitResult.hitTriangleIndex = -1;
                } else {
//              } else {
                    hitPoint = sampleGlobalPosition.xyz;
//                  hitPoint = sampleGlobalPosition.xyz;
                    rayHitResult.hitSurfaceNormal = normalize(sampleGlobalNormal.xyz);
//                  rayHitResult.hitSurfaceNormal = normalize(sampleGlobalNormal.xyz);
                    rayHitResult.hitSurfaceTangent = normalize(sampleGlobalTangent.xyz);
//                  rayHitResult.hitSurfaceTangent = normalize(sampleGlobalTangent.xyz);
                    rayHitResult.hitSurfaceTexcoord = float2(sampleGlobalNormal.w, sampleGlobalTangent.w);
//                  rayHitResult.hitSurfaceTexcoord = float2(sampleGlobalNormal.w, sampleGlobalTangent.w);
                    rayHitResult.isFrontFaceHit = dot(currentRay.direction, rayHitResult.hitSurfaceNormal) < 0.0;
//                  rayHitResult.isFrontFaceHit = dot(currentRay.direction, rayHitResult.hitSurfaceNormal) < 0.0;
                    if (!rayHitResult.isFrontFaceHit) {
//                  if (!rayHitResult.isFrontFaceHit) {
                        rayHitResult.hitSurfaceNormal = -rayHitResult.hitSurfaceNormal;
//                      rayHitResult.hitSurfaceNormal = -rayHitResult.hitSurfaceNormal;
                    }
//                  }
                    rayHitResult.hitTriangleIndex = int(sampleGlobalPosition.w) - 1;
//                  rayHitResult.hitTriangleIndex = int(sampleGlobalPosition.w) - 1;
                    rayHitResult.hitDistance = length(hitPoint - currentRay.origin);
//                  rayHitResult.hitDistance = length(hitPoint - currentRay.origin);
                }
//              }
            } else {
//          } else {
                Interval hitInterval;
//              Interval hitInterval;
                hitInterval.min = EPSILON_OFFSET;
//              hitInterval.min = EPSILON_OFFSET;
                hitInterval.max = INF;
//              hitInterval.max = INF;

                rayHitResult = traverseBVHClosestHit(currentRay, hitInterval);
//              rayHitResult = traverseBVHClosestHit(currentRay, hitInterval);

                if (rayHitResult.hitTriangleIndex != -1) {
//              if (rayHitResult.hitTriangleIndex != -1) {
                    hitPoint = currentRay.origin + currentRay.direction * rayHitResult.hitDistance;
//                  hitPoint = currentRay.origin + currentRay.direction * rayHitResult.hitDistance;

                    // Fix front face normal
//                  // Fix front face normal
                    bool isFrontFace = dot(currentRay.direction, rayHitResult.hitSurfaceNormal) < 0.0;
//                  bool isFrontFace = dot(currentRay.direction, rayHitResult.hitSurfaceNormal) < 0.0;
                    rayHitResult.isFrontFaceHit = isFrontFace;
//                  rayHitResult.isFrontFaceHit = isFrontFace;
                    if (!isFrontFace) {
//                  if (!isFrontFace) {
                        rayHitResult.hitSurfaceNormal = -rayHitResult.hitSurfaceNormal;
//                      rayHitResult.hitSurfaceNormal = -rayHitResult.hitSurfaceNormal;
                    }
//                  }
                }
//              }
            }
//          }

            SphereLightHit lightHit = intersectSphereLights(currentRay);
//          SphereLightHit lightHit = intersectSphereLights(currentRay);

            bool hitLight = false;
//          bool hitLight = false;
            if (lightHit.lightIndex != -1) {
//          if (lightHit.lightIndex != -1) {
                if (rayHitResult.hitTriangleIndex == -1 || lightHit.distanceToHit < rayHitResult.hitDistance) {
//              if (rayHitResult.hitTriangleIndex == -1 || lightHit.distanceToHit < rayHitResult.hitDistance) {
                    hitLight = true;
//                  hitLight = true;
                }
//              }
            }
//          }

            // Participating media: sample a free flight along the segment up to whatever the ray hits
//          // Participating media: sample a free flight along the segment up to whatever the ray hits
            // (surface, light, or nothing). A real collision turns this bounce into a medium vertex:
//          // (surface, light, or nothing). A real collision turns this bounce into a medium vertex:
            // next event estimation through the phase function, then a phase-sampled continuation.
//          // next event estimation through the phase function, then a phase-sampled continuation.
            // This runs at depth 0 too, so a medium in front of the G-Buffer surface pre-empts it; the
//          // This runs at depth 0 too, so a medium in front of the G-Buffer surface pre-empts it; the
            // pixel's ReSTIR reservoir was already cleared at the top of depth 0 and stays empty.
//          // pixel's ReSTIR reservoir was already cleared at the top of depth 0 and stays empty.
            float mediumScatterDistance;
//          float mediumScatterDistance;
            int mediumVolumeIndex;
//          int mediumVolumeIndex;
            float mediumSegmentEnd = hitLight ? lightHit.distanceToHit : ((rayHitResult.hitTriangleIndex != -1) ? rayHitResult.hitDistance : INF);
//          float mediumSegmentEnd = hitLight ? lightHit.distanceToHit : ((rayHitResult.hitTriangleIndex != -1) ? rayHitResult.hitDistance : INF);
            if (sampleMediumScattering<ShadingRandom>(currentRay.origin, currentRay.direction, mediumSegmentEnd, mediumScatterDistance, mediumVolumeIndex)) {
//          if (sampleMediumScattering<ShadingRandom>(currentRay.origin, currentRay.direction, mediumSegmentEnd, mediumScatterDistance, mediumVolumeIndex)) {
                hitPoint = currentRay.origin + currentRay.direction * mediumScatterDistance;
//              hitPoint = currentRay.origin + currentRay.direction * mediumScatterDistance;
                float3 mediumAlbedo = uMediumVolumes[mediumVolumeIndex].albedo;
//              float3 mediumAlbedo = uMediumVolumes[mediumVolumeIndex].albedo;
                float mediumAnisotropy = uMediumVolumes[mediumVolumeIndex].anisotropy;
//              float mediumAnisotropy = uMediumVolumes[mediumVolumeIndex].anisotropy;

                if (uPointLightCount > 0) {
//              if (uPointLightCount > 0) {
                    float lightSelectionRandom = rand();
//                  float lightSelectionRandom = rand();
                    int selectedLightIndex = max(0, uPointLightCount - 1);
//                  int selectedLightIndex = max(0, uPointLightCount - 1);
                    for (int lightSearchIndex = 0; lightSearchIndex < uPointLightCount; lightSearchIndex++) {
//                  for (int lightSearchIndex = 0; lightSearchIndex < uPointLightCount; lightSearchIndex++) {
                        if (lightSelectionRandom <= uPointLights[lightSearchIndex].cdf) {
//                      if (lightSelectionRandom <= uPointLights[lightSearchIndex].cdf) {
                            selectedLightIndex = lightSearchIndex;
//                          selectedLightIndex = lightSearchIndex;
                            break;
//                          break;
                        }
//                      }
                    }
//                  }
                    float3 lightPosition = uPointLights[selectedLightIndex].position;
//                  float3 lightPosition = uPointLights[selectedLightIndex].position;
                    float lightRadius = uPointLights[selectedLightIndex].radius;
//                  float lightRadius = uPointLights[selectedLightIndex].radius;
                    // Same facing-hemisphere light sampling as the surface NEE below, so the BRDF-hit MIS
//                  // Same facing-hemisphere light sampling as the surface NEE below, so the BRDF-hit MIS
                    // branch above weighs medium and surface vertices against one light pdf.
//                  // branch above weighs medium and surface vertices against one light pdf.
                    float3 lightSampleNormal = randomUnitVector();
//                  float3 lightSampleNormal = randomUnitVector();
                    if (dot(lightSampleNormal, normalize(hitPoint - lightPosition)) < 0.0) lightSampleNormal = -lightSampleNormal;
//                  if (dot(lightSampleNormal, normalize(hitPoint - lightPosition)) < 0.0) lightSampleNormal = -lightSampleNormal;
                    float3 sampledLightPoint = lightPosition + lightSampleNormal * lightRadius;
//                  float3 sampledLightPoint = lightPosition + lightSampleNormal * lightRadius;
                    float3 shadowRayDirection = sampledLightPoint - hitPoint;
//                  float3 shadowRayDirection = sampledLightPoint - hitPoint;
                    float actualDistanceToLight = length(shadowRayDirection);
//                  float actualDistanceToLight = length(shadowRayDirection);
                    float clampedDistanceToLight = max(actualDistanceToLight, lightRadius + 0.05);
//                  float clampedDistanceToLight = max(actualDistanceToLight, lightRadius + 0.05);
                    shadowRayDirection = normalize(shadowRayDirection);
//                  shadowRayDirection = normalize(shadowRayDirection);
                    float cosLightAngle = dot(lightSampleNormal, -shadowRayDirection);
//                  float cosLightAngle = dot(lightSampleNormal, -shadowRayDirection);
                    if (cosLightAngle > 0.0) {
//                  if (cosLightAngle > 0.0) {
                        Ray shadowRay;
//                      Ray shadowRay;
                        shadowRay.origin = hitPoint;
//                      shadowRay.origin = hitPoint;
                        shadowRay.direction = shadowRayDirection;
//                      shadowRay.direction = shadowRayDirection;
                        Interval shadowInterval;
//                      Interval shadowInterval;
                        shadowInterval.min = EPSILON_OFFSET;
//                      shadowInterval.min = EPSILON_OFFSET;
                        shadowInterval.max = actualDistanceToLight - EPSILON_OFFSET;
//                      shadowInterval.max = actualDistanceToLight - EPSILON_OFFSET;
                        if (!traverseBVHAnyHit(shadowRay, shadowInterval)) {
//                      if (!traverseBVHAnyHit(shadowRay, shadowInterval)) {
                            float shadowTransmittance = estimateMediumTransmittance<ShadingRandom>(hitPoint, shadowRayDirection, actualDistanceToLight);
//                          float shadowTransmittance = estimateMediumTransmittance<ShadingRandom>(hitPoint, shadowRayDirection, actualDistanceToLight);
                            float lightSurfaceArea = 2.0 * PI * lightRadius * lightRadius;
//                          float lightSurfaceArea = 2.0 * PI * lightRadius * lightRadius;
                            float pdfLightArea = uPointLights[selectedLightIndex].pdf / lightSurfaceArea;
//                          float pdfLightArea = uPointLights[selectedLightIndex].pdf / lightSurfaceArea;
                            float lightPdfSolidAngle = (pdfLightArea * clampedDistanceToLight * clampedDistanceToLight) / max(cosLightAngle, 1.0e-5);
//                          float lightPdfSolidAngle = (pdfLightArea * clampedDistanceToLight * clampedDistanceToLight) / max(cosLightAngle, 1.0e-5);
                            float phaseValue = henyeyGreenstein(dot(currentRay.direction, shadowRayDirection), mediumAnisotropy);
//                          float phaseValue = henyeyGreenstein(dot(currentRay.direction, shadowRayDirection), mediumAnisotropy);
                            float sumOfSquaredPdfs = lightPdfSolidAngle * lightPdfSolidAngle + phaseValue * phaseValue;
//                          float sumOfSquaredPdfs = lightPdfSolidAngle * lightPdfSolidAngle + phaseValue * phaseValue;
                            float misWeightNee = (sumOfSquaredPdfs > 0.0) ? (lightPdfSolidAngle * lightPdfSolidAngle) / sumOfSquaredPdfs : 0.0;
//                          float misWeightNee = (sumOfSquaredPdfs > 0.0) ? (lightPdfSolidAngle * lightPdfSolidAngle) / sumOfSquaredPdfs : 0.0;
                            float3 directLight = uPointLights[selectedLightIndex].color * phaseValue * shadowTransmittance / max(lightPdfSolidAngle, NEE_PDF_EPSILON);
//                          float3 directLight = uPointLights[selectedLightIndex].color * phaseValue * shadowTransmittance / max(lightPdfSolidAngle, NEE_PDF_EPSILON);
                            directLight = min(directLight, float3(NEE_DIRECT_LIGHT_CLAMP));
//                          directLight = min(directLight, float3(NEE_DIRECT_LIGHT_CLAMP));
                            accumulatedColor += attenuation * mediumAlbedo * directLight * misWeightNee;
//                          accumulatedColor += attenuation * mediumAlbedo * directLight * misWeightNee;
                        }
//                      }
                    }
//                  }
                }
//              }

                // Phase sampling is exact, so phase / pdf = 1 and only the albedo enters the throughput.
//              // Phase sampling is exact, so phase / pdf = 1 and only the albedo enters the throughput.
                float3 mediumScatteredDirection = sampleHenyeyGreenstein(currentRay.direction, mediumAnisotropy, rand(), rand());
//              float3 mediumScatteredDirection = sampleHenyeyGreenstein(currentRay.direction, mediumAnisotropy, rand(), rand());
                float phasePdf = henyeyGreenstein(dot(currentRay.direction, mediumScatteredDirection), mediumAnisotropy);
//              float phasePdf = henyeyGreenstein(dot(currentRay.direction, mediumScatteredDirection), mediumAnisotropy);
                attenuation *= mediumAlbedo;
//              attenuation *= mediumAlbedo;

                if (depth >= int(RR_MIN_BOUNCES)) {
//              if (depth >= int(RR_MIN_BOUNCES)) {
                    float survivalProbability = clamp(maxVec3(attenuation), RR_MIN_PROBABILITY, RR_MAX_PROBABILITY);
//                  float survivalProbability = clamp(maxVec3(attenuation), RR_MIN_PROBABILITY, RR_MAX_PROBABILITY);
                    if (rand() > survivalProbability) break;
//                  if (rand() > survivalProbability) break;
                    attenuation *= 1.0 / survivalProbability;
//                  attenuation *= 1.0 / survivalProbability;
                }
//              }

                currentRay.origin = hitPoint;
//              currentRay.origin = hitPoint;
                currentRay.direction = mediumScatteredDirection;
//              currentRay.direction = mediumScatteredDirection;
                lastBrdfPdfSolidAngle = phasePdf;
//              lastBrdfPdfSolidAngle = phasePdf;
                lastWasDelta = false;
//              lastWasDelta = false;
                lastSkippedNEE = false;
//              lastSkippedNEE = false;
                lastRanReSTIR = false;
//              lastRanReSTIR = false;
                // Counts as a diffuse bounce: a scattered path is already blurred enough that the
//              // Counts as a diffuse bounce: a scattered path is already blurred enough that the
                // Neural Radiance Cache may terminate it at the next surface.
//              // Neural Radiance Cache may terminate it at the next surface.
                diffuseBounceCount++;
//              diffuseBounceCount++;
                continue;
//              continue;
            }
//          }

            if (hitLight) {
//          if (hitLight) {
                float3 emittedRadiance = uPointLights[lightHit.lightIndex].color;
//              float3 emittedRadiance = uPointLights[lightHit.lightIndex].color;

                if (depth == 0 || lastWasDelta) {
//              if (depth == 0 || lastWasDelta) {
                    accumulatedColor += attenuation * emittedRadiance;
//                  accumulatedColor += attenuation * emittedRadiance;
                } else if (!lastRanReSTIR) {
//              } else if (!lastRanReSTIR) {
                    float3 lightHitNormal = normalize((currentRay.origin + currentRay.direction * lightHit.distanceToHit) - uPointLights[lightHit.lightIndex].position);
//                  float3 lightHitNormal = normalize((currentRay.origin + currentRay.direction * lightHit.distanceToHit) - uPointLights[lightHit.lightIndex].position);
                    float cosLightAngle = max(0.0, dot(lightHitNormal, -currentRay.direction));
//                  float cosLightAngle = max(0.0, dot(lightHitNormal, -currentRay.direction));
                    float lightRadius = uPointLights[lightHit.lightIndex].radius;
//                  float lightRadius = uPointLights[lightHit.lightIndex].radius;
                    float lightPdf = uPointLights[lightHit.lightIndex].pdf;
//                  float lightPdf = uPointLights[lightHit.lightIndex].pdf;
                    // NEE samples only the hemisphere facing the shading point, so the light area is 2*PI*r*r
//                  // NEE samples only the hemisphere facing the shading point, so the light area is 2*PI*r*r
                    float lightSurfaceArea = 2.0 * PI * lightRadius * lightRadius;
//                  float lightSurfaceArea = 2.0 * PI * lightRadius * lightRadius;
                    float pdfLightArea = lightPdf / lightSurfaceArea;
//                  float pdfLightArea = lightPdf / lightSurfaceArea;

                    float lightPdfSolidAngle = 0.0;
//                  float lightPdfSolidAngle = 0.0;
                    if (cosLightAngle > 0.0) {
//                  if (cosLightAngle > 0.0) {
                        lightPdfSolidAngle = (pdfLightArea * lightHit.distanceToHit * lightHit.distanceToHit) / max(cosLightAngle, 1.0e-5);
//                      lightPdfSolidAngle = (pdfLightArea * lightHit.distanceToHit * lightHit.distanceToHit) / max(cosLightAngle, 1.0e-5);
                    }
//                  }

                    float misWeightBrdf = 1.0;
//                  float misWeightBrdf = 1.0;
                    if (lightPdfSolidAngle > 0.0) {
//                  if (lightPdfSolidAngle > 0.0) {
                        // Replaced the Balance Heuristic with the robust Power Heuristic for calculating misWeightBrdf
//                      // Replaced the Balance Heuristic with the robust Power Heuristic for calculating misWeightBrdf
                        float sumOfSquaredPdfs = lastBrdfPdfSolidAngle * lastBrdfPdfSolidAngle + lightPdfSolidAngle * lightPdfSolidAngle;
//                      float sumOfSquaredPdfs = lastBrdfPdfSolidAngle * lastBrdfPdfSolidAngle + lightPdfSolidAngle * lightPdfSolidAngle;
                        misWeightBrdf = 0.0;
//                      misWeightBrdf = 0.0;
                        if (sumOfSquaredPdfs > 0.0) {
//                      if (sumOfSquaredPdfs > 0.0) {
                            misWeightBrdf = (lastBrdfPdfSolidAngle * lastBrdfPdfSolidAngle) / sumOfSquaredPdfs;
//                          misWeightBrdf = (lastBrdfPdfSolidAngle * lastBrdfPdfSolidAngle) / sumOfSquaredPdfs;
                        }
//                      }
                    }
//                  }

                    if (lastSkippedNEE) {
//                  if (lastSkippedNEE) {
                        misWeightBrdf = 1.0;
//                      misWeightBrdf = 1.0;
                    }
//                  }

                    accumulatedColor += attenuation * emittedRadiance * misWeightBrdf;
//                  accumulatedColor += attenuation * emittedRadiance * misWeightBrdf;
                }
//              }
                break;
//              break;
            }
//          }

            if (rayHitResult.hitTriangleIndex == -1) {
//          if (rayHitResult.hitTriangleIndex == -1) {
                // Sky
//              // Sky
                if (depth == 0 && (uReSTIREnabled != 0)) {
//              if (depth == 0 && (uReSTIREnabled != 0)) {
                    uint totalPixels = uint(uResolution.x * uResolution.y);
//                  uint totalPixels = uint(uResolution.x * uResolution.y);
                    uint currentFrameOffset = (uCacheFrameCounter % 2u) * totalPixels;
//                  uint currentFrameOffset = (uCacheFrameCounter % 2u) * totalPixels;
                    uint pixelIndex = uint(pixelCoordinates.y * uResolution.x + pixelCoordinates.x);
//                  uint pixelIndex = uint(pixelCoordinates.y * uResolution.x + pixelCoordinates.x);
                    sampleReservoirs[currentFrameOffset + pixelIndex].sample_count = 0.0;
//                  sampleReservoirs[currentFrameOffset + pixelIndex].sample_count = 0.0;
                }
//              }
                // Directly visible sky (camera ray, or a chain of perfect mirrors/glass) marches the cloud
//              // Directly visible sky (camera ray, or a chain of perfect mirrors/glass) marches the cloud
                // layer live at full resolution; blurrier bounces read the baked layer instead.
//              // layer live at full resolution; blurrier bounces read the baked layer instead.
                float3 skyRadiance = (depth == 0 || lastWasDelta) ? getSkyColorRayMarched(currentRay.direction, rand()) : getSkyColor(currentRay.direction);
//              float3 skyRadiance = (depth == 0 || lastWasDelta) ? getSkyColorRayMarched(currentRay.direction, rand()) : getSkyColor(currentRay.direction);
                accumulatedColor += attenuation * skyRadiance;
//              accumulatedColor += attenuation * skyRadiance;
                break;
//              break;
            }
//          }

            // Retrieve actual material index from the packed vertex buffer (stored in tangent w-component of the first vertex of the triangle)
//          // Retrieve actual material index from the packed vertex buffer (stored in tangent w-component of the first vertex of the triangle)
            int materialIndex = int(vertices[rayHitResult.hitTriangleIndex * 3].tangentAndMaterialIndex.w);
//          int materialIndex = int(vertices[rayHitResult.hitTriangleIndex * 3].tangentAndMaterialIndex.w);
            Material material = materials[materialIndex];
//          Material material = materials[materialIndex];

            // Optimize first-hit albedo by using the pre-evaluated G-Buffer, bypassing a redundant texture fetch
//          // Optimize first-hit albedo by using the pre-evaluated G-Buffer, bypassing a redundant texture fetch
            if (depth == 0) {
//          if (depth == 0) {
                material.albedo = textureGeometryAlbedo.Load(int3(pixelCoordinates, 0));
//              material.albedo = textureGeometryAlbedo.Load(int3(pixelCoordinates, 0));
                material.textureIndexAlbedo = -1.0;
//              material.textureIndexAlbedo = -1.0;
            }
//          }

            // As the ray depth increases, the surface roughness is artificially increased based on previous bounces to blur secondary reflections.
//          // As the ray depth increases, the surface roughness is artificially increased based on previous bounces to blur secondary reflections.
            // The clamp itself lives inside scatterPrincipled (after its texture fetches) via the pathRoughness floor;
//          // The clamp itself lives inside scatterPrincipled (after its texture fetches) via the pathRoughness floor;
            // pathRoughness is updated below from the true post-texture sampled roughness, not the base value
//          // pathRoughness is updated below from the true post-texture sampled roughness, not the base value
            // (which is often a 1.0 placeholder on textured materials and used to poison the anneal).
//          // (which is often a 1.0 placeholder on textured materials and used to poison the anneal).

            if (depth >= 1 && uCacheBlendFactor > 0.0 && max(material.roughness, pathRoughness) > CACHE_ROUGHNESS_GATE && material.transmission < 0.5) {
//          if (depth >= 1 && uCacheBlendFactor > 0.0 && max(material.roughness, pathRoughness) > CACHE_ROUGHNESS_GATE && material.transmission < 0.5) {
                float3 cachedRadiance;
//              float3 cachedRadiance;
                if (readCache(hitPoint, rayHitResult.hitSurfaceNormal, cachedRadiance)) {
//              if (readCache(hitPoint, rayHitResult.hitSurfaceNormal, cachedRadiance)) {
                    float depthFactor = clamp(float(depth - 1) / 3.0, 0.0, 1.0);
//                  float depthFactor = clamp(float(depth - 1) / 3.0, 0.0, 1.0);
                    float blendWeight = uCacheBlendFactor * lerp(0.3, 0.9, depthFactor);
//                  float blendWeight = uCacheBlendFactor * lerp(0.3, 0.9, depthFactor);
                    if (rand() < blendWeight) {
//                  if (rand() < blendWeight) {
                        accumulatedColor += attenuation * cachedRadiance;
//                      accumulatedColor += attenuation * cachedRadiance;
                        break;
//                      break;
                    }
//                  }
                }
//              }
            }
//          }

            float3 albedo;
//          float3 albedo;
            float roughness;
//          float roughness;
            float metallic;
//          float metallic;
            float transmission;
//          float transmission;
            float3 shadingNormal;
//          float3 shadingNormal;
            float3 emission;
//          float3 emission;
            float3 scatteredDirection;
//          float3 scatteredDirection;
            float3 scatterAttenuation;
//          float3 scatterAttenuation;
            float scatterPdf;
//          float scatterPdf;
            bool scatterIsDelta;
//          bool scatterIsDelta;

            bool isScattered = scatterPrincipled(currentRay, rayHitResult, material, (depth > 0) ? pathRoughness : 0.0, albedo, roughness, metallic, transmission, shadingNormal, emission, scatteredDirection, scatterAttenuation, scatterPdf, scatterIsDelta);
//          bool isScattered = scatterPrincipled(currentRay, rayHitResult, material, (depth > 0) ? pathRoughness : 0.0, albedo, roughness, metallic, transmission, shadingNormal, emission, scatteredDirection, scatterAttenuation, scatterPdf, scatterIsDelta);

            accumulatedColor += attenuation * emission;
//          accumulatedColor += attenuation * emission;
            float3 bounceDirectRadiance = emission;
//          float3 bounceDirectRadiance = emission;

            if (!isScattered) {
//          if (!isScattered) {
                break;
//              break;
            }
//          }

            // Track the true (post-texture) sampled roughness so deeper bounces anneal correctly
//          // Track the true (post-texture) sampled roughness so deeper bounces anneal correctly
            pathRoughness = max(pathRoughness, roughness);
//          pathRoughness = max(pathRoughness, roughness);

            // Spatiotemporal ReSTIR Direct Illumination (ST-ReSTIR DI)
//          // Spatiotemporal ReSTIR Direct Illumination (ST-ReSTIR DI)
            bool isSpecularDelta = roughness < 0.05 && (metallic > 0.99 || transmission > 0.99);
//          bool isSpecularDelta = roughness < 0.05 && (metallic > 0.99 || transmission > 0.99);
            // The Neural Radiance Cache regresses total outgoing radiance, direct illumination included.
//          // The Neural Radiance Cache regresses total outgoing radiance, direct illumination included.
            // Deciding here -- before Next Event Estimation runs -- whether the cache will terminate the
//          // Deciding here -- before Next Event Estimation runs -- whether the cache will terminate the
            // path lets NEE be suppressed at that same vertex. Experiment001 runs both and counts the
//          // path lets NEE be suppressed at that same vertex. Experiment001 runs both and counts the
            // vertex's direct lighting twice; that error is louder here because the spurious base_color
//          // vertex's direct lighting twice; that error is louder here because the spurious base_color
            // that used to damp its cache term has (correctly) been removed.
//          // that used to damp its cache term has (correctly) been removed.
            //
//          //
            // The predicate deliberately depends only on pre-scatter state. Gating NEE on one condition
//          // The predicate deliberately depends only on pre-scatter state. Gating NEE on one condition
            // and the cache read on another would risk dropping the vertex's direct light entirely
//          // and the cache read on another would risk dropping the vertex's direct light entirely
            // instead of double counting it, so both now consult this single flag.
//          // instead of double counting it, so both now consult this single flag.
            bool nrcTerminatesPath = (uNRCEnabled != 0) && diffuseBounceCount >= 1 && !isSpecularDelta
//          bool nrcTerminatesPath = (uNRCEnabled != 0) && diffuseBounceCount >= 1 && !isSpecularDelta
                                  && metallic < 0.99 && transmission < 0.01 && roughness >= 0.05;
//                                && metallic < 0.99 && transmission < 0.01 && roughness >= 0.05;
            bool skipNEE = isSpecularDelta || nrcTerminatesPath;
//          bool skipNEE = isSpecularDelta || nrcTerminatesPath;
            bool ranReSTIR = false;
//          bool ranReSTIR = false;
            // Gate on the texture-resolved metallic, not material.metallic: the raw scalar is a 1.0
//          // Gate on the texture-resolved metallic, not material.metallic: the raw scalar is a 1.0
            // placeholder on every material that carries a metallic map, which would switch ReSTIR
//          // placeholder on every material that carries a metallic map, which would switch ReSTIR
            // off across most of the scene.
//          // off across most of the scene.
            bool executeReSTIR = (uReSTIREnabled != 0) && (uPointLightCount > 0) && (depth == 0) && !isSpecularDelta && metallic < 0.99 && transmission < 0.01;
//          bool executeReSTIR = (uReSTIREnabled != 0) && (uPointLightCount > 0) && (depth == 0) && !isSpecularDelta && metallic < 0.99 && transmission < 0.01;

            // Variable Rate Tracing rate class for this pixel, and whether it owns the ray this frame.
//          // Variable Rate Tracing rate class for this pixel, and whether it owns the ray this frame.
            // Classified outside the ReSTIR gate on purpose. The rate only changes tracing behaviour
//          // Classified outside the ReSTIR gate on purpose. The rate only changes tracing behaviour
            // inside that block -- skipping a pixel's own light sample is only safe when reservoir reuse
//          // inside that block -- skipping a pixel's own light sample is only safe when reservoir reuse
            // can cover for it -- but the heatmap is a pure visualization and must still draw when
//          // can cover for it -- but the heatmap is a pure visualization and must still draw when
            // reuse is switched off, otherwise the uVRTVisualize toggle is a silent no-op.
//          // reuse is switched off, otherwise the uVRTVisualize toggle is a silent no-op.
            uint vrtRate = VRT_RATE_1X1;
//          uint vrtRate = VRT_RATE_1X1;
            bool isActiveTrace = true;
//          bool isActiveTrace = true;
            if (depth == 0) {
//          if (depth == 0) {
                // Pass the texture-resolved emission, not the raw material.emissive scalar, which is a
//              // Pass the texture-resolved emission, not the raw material.emissive scalar, which is a
                // placeholder on any material carrying an emissive map (same caveat as metallic above).
//              // placeholder on any material carrying an emissive map (same caveat as metallic above).
                // In practice this argument is always 0.0: scatterPrincipled returns false on an emissive
//              // In practice this argument is always 0.0: scatterPrincipled returns false on an emissive
                // surface, so the loop breaks before reaching here and the classifier's emitter branch is
//              // surface, so the loop breaks before reaching here and the classifier's emitter branch is
                // unreachable -- Experiment001's "emitters get full rate" rule holds trivially. Kept
//              // unreachable -- Experiment001's "emitters get full rate" rule holds trivially. Kept
                // correct-by-construction so the branch behaves if emitters ever stop terminating a path.
//              // correct-by-construction so the branch behaves if emitters ever stop terminating a path.
                vrtRate = classifyPixelTracingRate(rayHitResult.hitSurfaceNormal, roughness, metallic, transmission, maxVec3(emission));
//              vrtRate = classifyPixelTracingRate(rayHitResult.hitSurfaceNormal, roughness, metallic, transmission, maxVec3(emission));
                isActiveTrace = isPixelActiveForVRT(uint(pixelCoordinates.x), uint(pixelCoordinates.y), vrtRate, uint(uFrameCount));
//              isActiveTrace = isPixelActiveForVRT(uint(pixelCoordinates.x), uint(pixelCoordinates.y), vrtRate, uint(uFrameCount));
            }
//          }
            if (executeReSTIR) {
//          if (executeReSTIR) {
                uint totalPixels = uint(uResolution.x * uResolution.y);
//              uint totalPixels = uint(uResolution.x * uResolution.y);
                uint currentFrameOffset = (uCacheFrameCounter % 2u) * totalPixels;
//              uint currentFrameOffset = (uCacheFrameCounter % 2u) * totalPixels;
                uint previousFrameOffset = ((uCacheFrameCounter + 1u) % 2u) * totalPixels;
//              uint previousFrameOffset = ((uCacheFrameCounter + 1u) % 2u) * totalPixels;
                uint pixelIndex = uint(pixelCoordinates.y * uResolution.x + pixelCoordinates.x);
//              uint pixelIndex = uint(pixelCoordinates.y * uResolution.x + pixelCoordinates.x);


                Reservoir pixelReservoir;
//              Reservoir pixelReservoir;
                pixelReservoir.sum_of_weights = 0.0;
//              pixelReservoir.sum_of_weights = 0.0;
                pixelReservoir.sample_count = 0.0;
//              pixelReservoir.sample_count = 0.0;
                pixelReservoir.contribution_weight = 0.0;
//              pixelReservoir.contribution_weight = 0.0;
                pixelReservoir.surface_depth = rayHitResult.hitDistance;
//              pixelReservoir.surface_depth = rayHitResult.hitDistance;
                pixelReservoir.surface_normal = rayHitResult.hitSurfaceNormal;
//              pixelReservoir.surface_normal = rayHitResult.hitSurfaceNormal;
                pixelReservoir.rate_and_flags = vrtRate | (isActiveTrace ? 0x80u : 0u);
//              pixelReservoir.rate_and_flags = vrtRate | (isActiveTrace ? 0x80u : 0u);
                pixelReservoir.selected_sample.position = float3(0.0);
//              pixelReservoir.selected_sample.position = float3(0.0);
                pixelReservoir.selected_sample.padding1 = 0.0;
//              pixelReservoir.selected_sample.padding1 = 0.0;
                pixelReservoir.selected_sample.normal = float3(0.0, 1.0, 0.0);
//              pixelReservoir.selected_sample.normal = float3(0.0, 1.0, 0.0);
                pixelReservoir.selected_sample.padding2 = 0.0;
//              pixelReservoir.selected_sample.padding2 = 0.0;
                pixelReservoir.selected_sample.emission = float3(0.0);
//              pixelReservoir.selected_sample.emission = float3(0.0);
                pixelReservoir.selected_sample.probability_density = 0.0;
//              pixelReservoir.selected_sample.probability_density = 0.0;

                float3 viewDirection = -currentRay.direction;
//              float3 viewDirection = -currentRay.direction;

                // Step 1: Initial Local Candidate Generation (only for the pixel that owns this frame's ray)
//              // Step 1: Initial Local Candidate Generation (only for the pixel that owns this frame's ray)
                if (isActiveTrace) {
//              if (isActiveTrace) {
                    LightSample candidate = generateLightSampleCandidate(hitPoint);
//                  LightSample candidate = generateLightSampleCandidate(hitPoint);
                    float targetPdfCandidate = evaluateTargetPdf(candidate, hitPoint, shadingNormal, viewDirection, albedo, roughness, metallic, transmission);
//                  float targetPdfCandidate = evaluateTargetPdf(candidate, hitPoint, shadingNormal, viewDirection, albedo, roughness, metallic, transmission);
                    float candidateWeight = (candidate.probability_density > 0.0) ? (targetPdfCandidate / candidate.probability_density) : 0.0;
//                  float candidateWeight = (candidate.probability_density > 0.0) ? (targetPdfCandidate / candidate.probability_density) : 0.0;
                    updateReservoirSample(pixelReservoir, candidate, candidateWeight, 1.0);
//                  updateReservoirSample(pixelReservoir, candidate, candidateWeight, 1.0);
                }
//              }

                // Step 2: Spatiotemporal Reuse
//              // Step 2: Spatiotemporal Reuse
                if (uCacheFrameCounter > 1u && (uResetHistory == 0)) {
//              if (uCacheFrameCounter > 1u && (uResetHistory == 0)) {
                    float4 prevClipPos = mul(uPreviousTransformViewProjection, float4(hitPoint, 1.0));
//                  float4 prevClipPos = mul(uPreviousTransformViewProjection, float4(hitPoint, 1.0));
                    if (prevClipPos.w > 0.0) {
//                  if (prevClipPos.w > 0.0) {
                        float3 prevNdc = prevClipPos.xyz / prevClipPos.w;
//                      float3 prevNdc = prevClipPos.xyz / prevClipPos.w;
                        float2 prevScreenUv = prevNdc.xy * 0.5 + 0.5;
//                      float2 prevScreenUv = prevNdc.xy * 0.5 + 0.5;
                        int2 prevPixelCoord = int2(prevScreenUv * float2(uResolution));
//                      int2 prevPixelCoord = int2(prevScreenUv * float2(uResolution));
                        if (prevPixelCoord.x >= 0 && prevPixelCoord.x < uResolution.x && prevPixelCoord.y >= 0 && prevPixelCoord.y < uResolution.y) {
//                      if (prevPixelCoord.x >= 0 && prevPixelCoord.x < uResolution.x && prevPixelCoord.y >= 0 && prevPixelCoord.y < uResolution.y) {
                            uint prevPixelIndex = uint(prevPixelCoord.y * uResolution.x + prevPixelCoord.x);
//                          uint prevPixelIndex = uint(prevPixelCoord.y * uResolution.x + prevPixelCoord.x);
                            Reservoir previousReservoir = sampleReservoirs[previousFrameOffset + prevPixelIndex];
//                          Reservoir previousReservoir = sampleReservoirs[previousFrameOffset + prevPixelIndex];
                            if (previousReservoir.sample_count > 0.0) {
//                          if (previousReservoir.sample_count > 0.0) {
                                float prevHitDistance = length(hitPoint - uPreviousCameraPosition);
//                              float prevHitDistance = length(hitPoint - uPreviousCameraPosition);
                                float depthDiff = abs(previousReservoir.surface_depth - prevHitDistance) / max(prevHitDistance, 1.0e-4);
//                              float depthDiff = abs(previousReservoir.surface_depth - prevHitDistance) / max(prevHitDistance, 1.0e-4);
                                float normalDot = dot(previousReservoir.surface_normal, rayHitResult.hitSurfaceNormal);
//                              float normalDot = dot(previousReservoir.surface_normal, rayHitResult.hitSurfaceNormal);
                                if (depthDiff < 0.1 && normalDot > 0.9) {
//                              if (depthDiff < 0.1 && normalDot > 0.9) {
                                    // Pixels that skipped their own trace this frame are allowed a longer history.
//                                  // Pixels that skipped their own trace this frame are allowed a longer history.
                                    float maxHistoryM = isActiveTrace ? 20.0 : 30.0;
//                                  float maxHistoryM = isActiveTrace ? 20.0 : 30.0;
                                    previousReservoir.sample_count = min(previousReservoir.sample_count, maxHistoryM);
//                                  previousReservoir.sample_count = min(previousReservoir.sample_count, maxHistoryM);
                                    float targetPdfTemporal = evaluateTargetPdf(previousReservoir.selected_sample, hitPoint, shadingNormal, viewDirection, albedo, roughness, metallic, transmission);
//                                  float targetPdfTemporal = evaluateTargetPdf(previousReservoir.selected_sample, hitPoint, shadingNormal, viewDirection, albedo, roughness, metallic, transmission);
                                    updateReservoirSample(pixelReservoir, previousReservoir.selected_sample, targetPdfTemporal * previousReservoir.contribution_weight * previousReservoir.sample_count, previousReservoir.sample_count);
//                                  updateReservoirSample(pixelReservoir, previousReservoir.selected_sample, targetPdfTemporal * previousReservoir.contribution_weight * previousReservoir.sample_count, previousReservoir.sample_count);
                                }
//                              }
                            }
//                          }
                        }
//                      }
                    }
//                  }

                    // Spatial reuse compensates for the missing local candidate on non-traced pixels.
//                  // Spatial reuse compensates for the missing local candidate on non-traced pixels.
                    int spatialTapCount = isActiveTrace ? 3 : 5;
//                  int spatialTapCount = isActiveTrace ? 3 : 5;
                    for (int tap = 0; tap < spatialTapCount; tap++) {
//                  for (int tap = 0; tap < spatialTapCount; tap++) {
                        float radius = sqrt(rand()) * 30.0;
//                      float radius = sqrt(rand()) * 30.0;
                        float theta = rand() * TWO_PI;
//                      float theta = rand() * TWO_PI;
                        int neighborX = clamp(pixelCoordinates.x + int(radius * cos(theta)), 0, uResolution.x - 1);
//                      int neighborX = clamp(pixelCoordinates.x + int(radius * cos(theta)), 0, uResolution.x - 1);
                        int neighborY = clamp(pixelCoordinates.y + int(radius * sin(theta)), 0, uResolution.y - 1);
//                      int neighborY = clamp(pixelCoordinates.y + int(radius * sin(theta)), 0, uResolution.y - 1);
                        uint neighborIndex = uint(neighborY * uResolution.x + neighborX);
//                      uint neighborIndex = uint(neighborY * uResolution.x + neighborX);
                        Reservoir spatialReservoir = sampleReservoirs[previousFrameOffset + neighborIndex];
//                      Reservoir spatialReservoir = sampleReservoirs[previousFrameOffset + neighborIndex];
                        if (spatialReservoir.sample_count > 0.0) {
//                      if (spatialReservoir.sample_count > 0.0) {
                            float depthDiff = abs(spatialReservoir.surface_depth - rayHitResult.hitDistance) / max(rayHitResult.hitDistance, 1.0e-4);
//                          float depthDiff = abs(spatialReservoir.surface_depth - rayHitResult.hitDistance) / max(rayHitResult.hitDistance, 1.0e-4);
                            float normalDot = dot(spatialReservoir.surface_normal, rayHitResult.hitSurfaceNormal);
//                          float normalDot = dot(spatialReservoir.surface_normal, rayHitResult.hitSurfaceNormal);
                            if (depthDiff < 0.1 && normalDot > 0.9) {
//                          if (depthDiff < 0.1 && normalDot > 0.9) {
                                spatialReservoir.sample_count = min(spatialReservoir.sample_count, 20.0);
//                              spatialReservoir.sample_count = min(spatialReservoir.sample_count, 20.0);
                                float targetPdfSpatial = evaluateTargetPdf(spatialReservoir.selected_sample, hitPoint, shadingNormal, viewDirection, albedo, roughness, metallic, transmission);
//                              float targetPdfSpatial = evaluateTargetPdf(spatialReservoir.selected_sample, hitPoint, shadingNormal, viewDirection, albedo, roughness, metallic, transmission);
                                updateReservoirSample(pixelReservoir, spatialReservoir.selected_sample, targetPdfSpatial * spatialReservoir.contribution_weight * spatialReservoir.sample_count, spatialReservoir.sample_count);
//                              updateReservoirSample(pixelReservoir, spatialReservoir.selected_sample, targetPdfSpatial * spatialReservoir.contribution_weight * spatialReservoir.sample_count, spatialReservoir.sample_count);
                            }
//                          }
                        }
//                      }
                    }
//                  }
                }
//              }

                // Fallback candidate: a non-traced pixel with no usable history would otherwise stay black.
//              // Fallback candidate: a non-traced pixel with no usable history would otherwise stay black.
                if (pixelReservoir.sample_count == 0.0) {
//              if (pixelReservoir.sample_count == 0.0) {
                    LightSample fallbackCandidate = generateLightSampleCandidate(hitPoint);
//                  LightSample fallbackCandidate = generateLightSampleCandidate(hitPoint);
                    float targetPdfFallback = evaluateTargetPdf(fallbackCandidate, hitPoint, shadingNormal, viewDirection, albedo, roughness, metallic, transmission);
//                  float targetPdfFallback = evaluateTargetPdf(fallbackCandidate, hitPoint, shadingNormal, viewDirection, albedo, roughness, metallic, transmission);
                    float fallbackWeight = (fallbackCandidate.probability_density > 0.0) ? (targetPdfFallback / fallbackCandidate.probability_density) : 0.0;
//                  float fallbackWeight = (fallbackCandidate.probability_density > 0.0) ? (targetPdfFallback / fallbackCandidate.probability_density) : 0.0;
                    updateReservoirSample(pixelReservoir, fallbackCandidate, fallbackWeight, 1.0);
//                  updateReservoirSample(pixelReservoir, fallbackCandidate, fallbackWeight, 1.0);
                }
//              }

                // Step 3: Finalize Weights
//              // Step 3: Finalize Weights
                float targetPdfFinal = evaluateTargetPdf(pixelReservoir.selected_sample, hitPoint, shadingNormal, viewDirection, albedo, roughness, metallic, transmission);
//              float targetPdfFinal = evaluateTargetPdf(pixelReservoir.selected_sample, hitPoint, shadingNormal, viewDirection, albedo, roughness, metallic, transmission);
                float denom = pixelReservoir.sample_count * targetPdfFinal;
//              float denom = pixelReservoir.sample_count * targetPdfFinal;
                pixelReservoir.contribution_weight = (denom > 0.0) ? (pixelReservoir.sum_of_weights / denom) : 0.0;
//              pixelReservoir.contribution_weight = (denom > 0.0) ? (pixelReservoir.sum_of_weights / denom) : 0.0;
                // Quarantine non-finite weights before the reservoir is published. A NaN here would be
//              // Quarantine non-finite weights before the reservoir is published. A NaN here would be
                // written into the buffer, then picked up by every neighbour that samples this pixel as a
//              // written into the buffer, then picked up by every neighbour that samples this pixel as a
                // spatial tap, and the NaN guard at the end of computeMain() would black out the whole spread.
//              // spatial tap, and the NaN guard at the end of computeMain() would black out the whole spread.
                // Zeroing sample_count is what makes the next frame skip this entry outright.
//              // Zeroing sample_count is what makes the next frame skip this entry outright.
                if (isnan(pixelReservoir.sum_of_weights) || isinf(pixelReservoir.sum_of_weights)
//              if (isnan(pixelReservoir.sum_of_weights) || isinf(pixelReservoir.sum_of_weights)
                 || isnan(pixelReservoir.contribution_weight) || isinf(pixelReservoir.contribution_weight)) {
//               || isnan(pixelReservoir.contribution_weight) || isinf(pixelReservoir.contribution_weight)) {
                    pixelReservoir.sum_of_weights = 0.0;
//                  pixelReservoir.sum_of_weights = 0.0;
                    pixelReservoir.contribution_weight = 0.0;
//                  pixelReservoir.contribution_weight = 0.0;
                    pixelReservoir.sample_count = 0.0;
//                  pixelReservoir.sample_count = 0.0;
                }
//              }
                sampleReservoirs[currentFrameOffset + pixelIndex] = pixelReservoir;
//              sampleReservoirs[currentFrameOffset + pixelIndex] = pixelReservoir;

                // Step 4: Resolve Visibility & Direct Illumination
//              // Step 4: Resolve Visibility & Direct Illumination
                if (targetPdfFinal > 0.0) {
//              if (targetPdfFinal > 0.0) {
                    float3 toLight = pixelReservoir.selected_sample.position - hitPoint;
//                  float3 toLight = pixelReservoir.selected_sample.position - hitPoint;
                    float distanceToLight = length(toLight);
//                  float distanceToLight = length(toLight);
                    float3 lightDirection = toLight / max(distanceToLight, 1.0e-4);
//                  float3 lightDirection = toLight / max(distanceToLight, 1.0e-4);
                    float surfaceCosine = max(0.0, dot(shadingNormal, lightDirection));
//                  float surfaceCosine = max(0.0, dot(shadingNormal, lightDirection));
                    float lightCosine = max(0.0, dot(pixelReservoir.selected_sample.normal, -lightDirection));
//                  float lightCosine = max(0.0, dot(pixelReservoir.selected_sample.normal, -lightDirection));
                    if (surfaceCosine > 0.0 && lightCosine > 0.0 && dot(rayHitResult.hitSurfaceNormal, lightDirection) > 0.0) {
//                  if (surfaceCosine > 0.0 && lightCosine > 0.0 && dot(rayHitResult.hitSurfaceNormal, lightDirection) > 0.0) {
                        Ray shadowRay;
//                      Ray shadowRay;
                        shadowRay.origin = hitPoint;
//                      shadowRay.origin = hitPoint;
                        shadowRay.direction = lightDirection;
//                      shadowRay.direction = lightDirection;
                        Interval shadowInterval;
//                      Interval shadowInterval;
                        shadowInterval.min = EPSILON_OFFSET;
//                      shadowInterval.min = EPSILON_OFFSET;
                        shadowInterval.max = distanceToLight - EPSILON_OFFSET;
//                      shadowInterval.max = distanceToLight - EPSILON_OFFSET;

                        if (!traverseBVHAnyHit(shadowRay, shadowInterval)) {
//                      if (!traverseBVHAnyHit(shadowRay, shadowInterval)) {
                            float unusedPdf;
//                          float unusedPdf;
                            float3 bsdfValue = evalPrincipledBSDFAndPDF(currentRay.direction, lightDirection, shadingNormal, albedo, roughness, metallic, transmission, unusedPdf);
//                          float3 bsdfValue = evalPrincipledBSDFAndPDF(currentRay.direction, lightDirection, shadingNormal, albedo, roughness, metallic, transmission, unusedPdf);
                            float3 directLight = bsdfValue * surfaceCosine * pixelReservoir.selected_sample.emission * (lightCosine / max(distanceToLight * distanceToLight, 1.0e-4)) * pixelReservoir.contribution_weight;
//                          float3 directLight = bsdfValue * surfaceCosine * pixelReservoir.selected_sample.emission * (lightCosine / max(distanceToLight * distanceToLight, 1.0e-4)) * pixelReservoir.contribution_weight;
                            // Media shadowing enters at resolve only: the target pdf stays visibility-free,
//                          // Media shadowing enters at resolve only: the target pdf stays visibility-free,
                            // exactly like the BVH occlusion test above.
//                          // exactly like the BVH occlusion test above.
                            directLight *= estimateMediumTransmittance<ShadingRandom>(hitPoint, lightDirection, distanceToLight);
//                          directLight *= estimateMediumTransmittance<ShadingRandom>(hitPoint, lightDirection, distanceToLight);
                            directLight = min(directLight, float3(NEE_DIRECT_LIGHT_CLAMP));
//                          directLight = min(directLight, float3(NEE_DIRECT_LIGHT_CLAMP));
                            accumulatedColor += attenuation * directLight;
//                          accumulatedColor += attenuation * directLight;
                            bounceDirectRadiance += directLight;
//                          bounceDirectRadiance += directLight;
                        }
//                      }
                    }
//                  }
                }
//              }

                skipNEE = true;
//              skipNEE = true;
                ranReSTIR = true;
//              ranReSTIR = true;
            } else if (depth == 0 && (uReSTIREnabled != 0)) {
//          } else if (depth == 0 && (uReSTIREnabled != 0)) {
                uint totalPixels = uint(uResolution.x * uResolution.y);
//              uint totalPixels = uint(uResolution.x * uResolution.y);
                uint currentFrameOffset = (uCacheFrameCounter % 2u) * totalPixels;
//              uint currentFrameOffset = (uCacheFrameCounter % 2u) * totalPixels;
                uint pixelIndex = uint(pixelCoordinates.y * uResolution.x + pixelCoordinates.x);
//              uint pixelIndex = uint(pixelCoordinates.y * uResolution.x + pixelCoordinates.x);
                sampleReservoirs[currentFrameOffset + pixelIndex].sample_count = 0.0;
//              sampleReservoirs[currentFrameOffset + pixelIndex].sample_count = 0.0;
            }
//          }

            // Variable Rate Tracing heatmap overlay
//          // Variable Rate Tracing heatmap overlay
            if (depth == 0 && (uVRTVisualize != 0)) {
//          if (depth == 0 && (uVRTVisualize != 0)) {
                accumulatedColor = lerp(accumulatedColor, visualizeTracingRate(vrtRate, isActiveTrace) * 2.5, 0.75);
//              accumulatedColor = lerp(accumulatedColor, visualizeTracingRate(vrtRate, isActiveTrace) * 2.5, 0.75);
            }
//          }

            // NRC training records are collected by the dedicated nrc_gather_cs.slang pass.
//          // NRC training records are collected by the dedicated nrc_gather_cs.slang pass.

            // Direct Illumination (Next Event Estimation) with MIS
//          // Direct Illumination (Next Event Estimation) with MIS
            if (uPointLightCount > 0 && !skipNEE) {
//          if (uPointLightCount > 0 && !skipNEE) {
                float lightSelectionRandom = rand();
//              float lightSelectionRandom = rand();
                int selectedLightIndex = max(0, uPointLightCount - 1);
//              int selectedLightIndex = max(0, uPointLightCount - 1);
                for (int lightSearchIndex = 0; lightSearchIndex < uPointLightCount; lightSearchIndex++) {
//              for (int lightSearchIndex = 0; lightSearchIndex < uPointLightCount; lightSearchIndex++) {
                    if (lightSelectionRandom <= uPointLights[lightSearchIndex].cdf) {
//                  if (lightSelectionRandom <= uPointLights[lightSearchIndex].cdf) {
                        selectedLightIndex = lightSearchIndex;
//                      selectedLightIndex = lightSearchIndex;
                        break;
//                      break;
                    }
//                  }
                }
//              }

                float3 lightPosition = uPointLights[selectedLightIndex].position;
//              float3 lightPosition = uPointLights[selectedLightIndex].position;
                float lightRadius = uPointLights[selectedLightIndex].radius;
//              float lightRadius = uPointLights[selectedLightIndex].radius;
                float lightPdf = uPointLights[selectedLightIndex].pdf;
//              float lightPdf = uPointLights[selectedLightIndex].pdf;
                // Sample only the hemisphere of the light sphere facing the shading point to halve wasted (culled) samples
//              // Sample only the hemisphere of the light sphere facing the shading point to halve wasted (culled) samples
                float3 lightToSurfaceDirection = normalize(hitPoint - lightPosition);
//              float3 lightToSurfaceDirection = normalize(hitPoint - lightPosition);
                float3 lightSampleNormal = randomUnitVector();
//              float3 lightSampleNormal = randomUnitVector();
                if (dot(lightSampleNormal, lightToSurfaceDirection) < 0.0) lightSampleNormal = -lightSampleNormal;
//              if (dot(lightSampleNormal, lightToSurfaceDirection) < 0.0) lightSampleNormal = -lightSampleNormal;
                float3 sampledLightPoint = lightPosition + lightSampleNormal * lightRadius;
//              float3 sampledLightPoint = lightPosition + lightSampleNormal * lightRadius;

                float3 shadowRayDirection = sampledLightPoint - hitPoint;
//              float3 shadowRayDirection = sampledLightPoint - hitPoint;
                float actualDistanceToLight = length(shadowRayDirection);
//              float actualDistanceToLight = length(shadowRayDirection);
                // Clamp to slightly above the light's radius to prevent division-by-zero singularities
//              // Clamp to slightly above the light's radius to prevent division-by-zero singularities
                float clampedDistanceToLight = max(actualDistanceToLight, lightRadius + 0.05);
//              float clampedDistanceToLight = max(actualDistanceToLight, lightRadius + 0.05);
                shadowRayDirection = normalize(shadowRayDirection);
//              shadowRayDirection = normalize(shadowRayDirection);

                float cosSurfaceAngle = max(0.0, dot(shadingNormal, shadowRayDirection));
//              float cosSurfaceAngle = max(0.0, dot(shadingNormal, shadowRayDirection));
                float3 sampledLightNormal = normalize(sampledLightPoint - lightPosition);
//              float3 sampledLightNormal = normalize(sampledLightPoint - lightPosition);
                // Removed clamp to correctly cull occluded samples
//              // Removed clamp to correctly cull occluded samples
                float cosLightAngle = dot(sampledLightNormal, -shadowRayDirection);
//              float cosLightAngle = dot(sampledLightNormal, -shadowRayDirection);

                if (cosSurfaceAngle > 0.0 && cosLightAngle > 0.0 && dot(rayHitResult.hitSurfaceNormal, shadowRayDirection) > 0.0) {
//              if (cosSurfaceAngle > 0.0 && cosLightAngle > 0.0 && dot(rayHitResult.hitSurfaceNormal, shadowRayDirection) > 0.0) {
                    cosLightAngle = max(cosLightAngle, 1.0e-5);
//                  cosLightAngle = max(cosLightAngle, 1.0e-5);
                    Ray shadowRay;
//                  Ray shadowRay;
                    shadowRay.origin = hitPoint;
//                  shadowRay.origin = hitPoint;
                    shadowRay.direction = shadowRayDirection;
//                  shadowRay.direction = shadowRayDirection;

                    Interval shadowInterval;
//                  Interval shadowInterval;
                    shadowInterval.min = EPSILON_OFFSET;
//                  shadowInterval.min = EPSILON_OFFSET;
                    shadowInterval.max = actualDistanceToLight - EPSILON_OFFSET;
//                  shadowInterval.max = actualDistanceToLight - EPSILON_OFFSET;

                    if (!traverseBVHAnyHit(shadowRay, shadowInterval)) {
//                  if (!traverseBVHAnyHit(shadowRay, shadowInterval)) {
                        // Hemisphere sampling (see above): the sampleable light area is 2*PI*r*r, matching the BRDF-hit MIS area
//                      // Hemisphere sampling (see above): the sampleable light area is 2*PI*r*r, matching the BRDF-hit MIS area
                        float lightSurfaceArea = 2.0 * PI * lightRadius * lightRadius;
//                      float lightSurfaceArea = 2.0 * PI * lightRadius * lightRadius;
                        float pdfLightArea = lightPdf / lightSurfaceArea;
//                      float pdfLightArea = lightPdf / lightSurfaceArea;
                        float lightPdfSolidAngle = (pdfLightArea * clampedDistanceToLight * clampedDistanceToLight) / cosLightAngle;
//                      float lightPdfSolidAngle = (pdfLightArea * clampedDistanceToLight * clampedDistanceToLight) / cosLightAngle;

                        // Single fused evaluation supplies both the MIS counter-pdf and the BSDF value
//                      // Single fused evaluation supplies both the MIS counter-pdf and the BSDF value
                        float brdfPdfSolidAngle;
//                      float brdfPdfSolidAngle;
                        float3 bsdfValue = evalPrincipledBSDFAndPDF(currentRay.direction, shadowRayDirection, shadingNormal, albedo, roughness, metallic, transmission, brdfPdfSolidAngle);
//                      float3 bsdfValue = evalPrincipledBSDFAndPDF(currentRay.direction, shadowRayDirection, shadingNormal, albedo, roughness, metallic, transmission, brdfPdfSolidAngle);

                        // Replaced the Balance Heuristic with the robust Power Heuristic for calculating misWeightNee
//                      // Replaced the Balance Heuristic with the robust Power Heuristic for calculating misWeightNee
                        float sumOfSquaredPdfs = lightPdfSolidAngle * lightPdfSolidAngle + brdfPdfSolidAngle * brdfPdfSolidAngle;
//                      float sumOfSquaredPdfs = lightPdfSolidAngle * lightPdfSolidAngle + brdfPdfSolidAngle * brdfPdfSolidAngle;
                        float misWeightNee = 0.0;
//                      float misWeightNee = 0.0;
                        if (sumOfSquaredPdfs > 0.0) {
//                      if (sumOfSquaredPdfs > 0.0) {
                            misWeightNee = (lightPdfSolidAngle * lightPdfSolidAngle) / sumOfSquaredPdfs;
//                          misWeightNee = (lightPdfSolidAngle * lightPdfSolidAngle) / sumOfSquaredPdfs;
                        }
//                      }

                        // Radiance emitted by the point light's surface
//                      // Radiance emitted by the point light's surface
                        float3 directLight = uPointLights[selectedLightIndex].color * bsdfValue * cosSurfaceAngle / max(lightPdfSolidAngle, NEE_PDF_EPSILON);
//                      float3 directLight = uPointLights[selectedLightIndex].color * bsdfValue * cosSurfaceAngle / max(lightPdfSolidAngle, NEE_PDF_EPSILON);
                        directLight *= estimateMediumTransmittance<ShadingRandom>(hitPoint, shadowRayDirection, actualDistanceToLight);
//                      directLight *= estimateMediumTransmittance<ShadingRandom>(hitPoint, shadowRayDirection, actualDistanceToLight);
                        // Clamp direct light to prevent fireflies from extreme NEE contributions
//                      // Clamp direct light to prevent fireflies from extreme NEE contributions
                        directLight = min(directLight, float3(NEE_DIRECT_LIGHT_CLAMP));
//                      directLight = min(directLight, float3(NEE_DIRECT_LIGHT_CLAMP));
                        accumulatedColor += attenuation * directLight * misWeightNee;
//                      accumulatedColor += attenuation * directLight * misWeightNee;
                        bounceDirectRadiance += directLight * misWeightNee;
//                      bounceDirectRadiance += directLight * misWeightNee;
                    }
//                  }
                }
//              }
            }
//          }

            // Neural Radiance Cache: terminate the path and read the remaining indirect illumination out
//          // Neural Radiance Cache: terminate the path and read the remaining indirect illumination out
            // of the cache. The network regresses outgoing radiance, so it is added without re-applying
//          // of the cache. The network regresses outgoing radiance, so it is added without re-applying
            // the surface albedo. Experiment001 multiplies by base_color here, which double counts the
//          // the surface albedo. Experiment001 multiplies by base_color here, which double counts the
            // albedo its own training target already baked in. nrcTerminatesPath was decided before NEE
//          // albedo its own training target already baked in. nrcTerminatesPath was decided before NEE
            // so that this vertex's direct lighting is contributed by the cache alone.
//          // so that this vertex's direct lighting is contributed by the cache alone.
            if (nrcTerminatesPath) {
//          if (nrcTerminatesPath) {
                accumulatedColor += attenuation * evaluateNeuralRadianceCache(hitPoint, rayHitResult.hitSurfaceNormal);
//              accumulatedColor += attenuation * evaluateNeuralRadianceCache(hitPoint, rayHitResult.hitSurfaceNormal);
                break;
//              break;
            }
//          }
            if (depth >= 1 && uCacheBlendFactor > 0.0 && !scatterIsDelta) {
//          if (depth >= 1 && uCacheBlendFactor > 0.0 && !scatterIsDelta) {
                writeCache(hitPoint, rayHitResult.hitSurfaceNormal, bounceDirectRadiance);
//              writeCache(hitPoint, rayHitResult.hitSurfaceNormal, bounceDirectRadiance);
            }
//          }

            attenuation *= scatterAttenuation;
//          attenuation *= scatterAttenuation;

            // Russian Roulette: Stochastically terminate low-contribution paths to improve performance
//          // Russian Roulette: Stochastically terminate low-contribution paths to improve performance
            uint minBouncesForRR = RR_MIN_BOUNCES;
//          uint minBouncesForRR = RR_MIN_BOUNCES;
            if (depth >= minBouncesForRR) {
//          if (depth >= minBouncesForRR) {
                // Clamp probability to avoid extreme weights and premature termination
//              // Clamp probability to avoid extreme weights and premature termination
                float survivalProbability = clamp(maxVec3(attenuation), RR_MIN_PROBABILITY, RR_MAX_PROBABILITY);
//              float survivalProbability = clamp(maxVec3(attenuation), RR_MIN_PROBABILITY, RR_MAX_PROBABILITY);
                if (rand() > survivalProbability) break;
//              if (rand() > survivalProbability) break;
                attenuation *= 1.0 / survivalProbability;
//              attenuation *= 1.0 / survivalProbability;
            }
//          }

            currentRay.origin = hitPoint;
//          currentRay.origin = hitPoint;
            currentRay.direction = scatteredDirection;
//          currentRay.direction = scatteredDirection;
            lastBrdfPdfSolidAngle = scatterPdf;
//          lastBrdfPdfSolidAngle = scatterPdf;
            lastWasDelta = scatterIsDelta;
//          lastWasDelta = scatterIsDelta;
            if (!scatterIsDelta) {
//          if (!scatterIsDelta) {
                diffuseBounceCount++;
//              diffuseBounceCount++;
            }
//          }
            lastSkippedNEE = isSpecularDelta;
//          lastSkippedNEE = isSpecularDelta;
            lastRanReSTIR = ranReSTIR;
//          lastRanReSTIR = ranReSTIR;
        }
//      }

        accumulatedColor = min(accumulatedColor, float3(ACCUM_CLAMP));
//      accumulatedColor = min(accumulatedColor, float3(ACCUM_CLAMP));

        // Prevent NaN propagation in the accumulated frame buffer
//      // Prevent NaN propagation in the accumulated frame buffer
        if (isnan(accumulatedColor.r) || isnan(accumulatedColor.g) || isnan(accumulatedColor.b)) {
//      if (isnan(accumulatedColor.r) || isnan(accumulatedColor.g) || isnan(accumulatedColor.b)) {
            accumulatedColor = float3(0.0);
//          accumulatedColor = float3(0.0);
        }
//      }

        // Temporal Reprojection Accumulation: folds this frame's noisy radiance estimate into a persistent history buffer as a cumulative moving average, blending with weight 1/uFrameCount so each new sample contributes its equal share of the running mean. On the first frame (or after a camera/scene reset) the history is discarded and the buffer reseeded with full weight; across many stationary frames the variance of the mean falls as ~1/N, progressively resolving the stochastic noise into a clean, converged image. Tone mapping and final output are deferred to the separate denoise pass.
//      // Temporal Reprojection Accumulation: folds this frame's noisy radiance estimate into a persistent history buffer as a cumulative moving average, blending with weight 1/uFrameCount so each new sample contributes its equal share of the running mean. On the first frame (or after a camera/scene reset) the history is discarded and the buffer reseeded with full weight; across many stationary frames the variance of the mean falls as ~1/N, progressively resolving the stochastic noise into a clean, converged image. Tone mapping and final output are deferred to the separate denoise pass.
        float4 previousAccumulation = textureAccumHistory.Load(int3(pixelCoordinates, 0));
//      float4 previousAccumulation = textureAccumHistory.Load(int3(pixelCoordinates, 0));
        float3 historyColor = previousAccumulation.rgb;
//      float3 historyColor = previousAccumulation.rgb;
        float blendAlpha = 1.0 / float(uFrameCount);
//      float blendAlpha = 1.0 / float(uFrameCount);
        blendAlpha = max(blendAlpha, 0.001 /* 0.01 */);
//      blendAlpha = max(blendAlpha, 0.001 /* 0.01 */);
        if (uFrameCount == 1) {
//      if (uFrameCount == 1) {
            historyColor = float3(0.0);
//          historyColor = float3(0.0);
            blendAlpha = 1.0;
//          blendAlpha = 1.0;
        }
//      }
        float3 blendedColor = lerp(historyColor, accumulatedColor, blendAlpha);
//      float3 blendedColor = lerp(historyColor, accumulatedColor, blendAlpha);
        textureAccum.Store(pixelCoordinates, float4(blendedColor, 1.0));
//      textureAccum.Store(pixelCoordinates, float4(blendedColor, 1.0));

        // Note: Tone mapping and output moved to hybrid_denoise_cs.slang
//      // Note: Tone mapping and output moved to hybrid_denoise_cs.slang
    }
//  }
