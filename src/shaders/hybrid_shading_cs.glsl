    #version 430
//  #version 430

    layout(local_size_x = 16, local_size_y = 16) in;
//  layout(local_size_x = 16, local_size_y = 16) in;

    layout(binding = 0, rgba32f) uniform image2D textureOutput;
//  layout(binding = 0, rgba32f) uniform image2D textureOutput;
    layout(binding = 1, rgba32f) uniform image2D textureGeometryGlobalPosition;
//  layout(binding = 1, rgba32f) uniform image2D textureGeometryGlobalPosition;
    layout(binding = 2, rgba16f) uniform image2D textureGeometryGlobalNormal;
//  layout(binding = 2, rgba16f) uniform image2D textureGeometryGlobalNormal;
    layout(binding = 3, rgba8) uniform image2D textureGeometryAlbedo;
//  layout(binding = 3, rgba8) uniform image2D textureGeometryAlbedo;
    layout(binding = 4, rgba16f) uniform image2D textureGeometryGlobalTangent;
//  layout(binding = 4, rgba16f) uniform image2D textureGeometryGlobalTangent;
    layout(binding = 5, rgba32f) uniform image2D textureAccum;
//  layout(binding = 5, rgba32f) uniform image2D textureAccum;

    // Bounding Volume Hierarchy (BVH) acceleration structure: a binary tree of Axis-Aligned Bounding Boxes (AABBs) that recursively partitions scene primitives to rapidly cull empty space. Each 32-byte node packs its AABB extents into two vec4s, overloading the otherwise-unused w-channels to carry topology — inner nodes store left/right child indices, while leaf nodes encode a primitive triangle offset (signalled by a negative left-child sentinel). This compact, cache-coherent layout collapses ray-scene queries from O(n) brute force to an expected O(log n) walk of intersection tests.
//  // Bounding Volume Hierarchy (BVH) acceleration structure: a binary tree of Axis-Aligned Bounding Boxes (AABBs) that recursively partitions scene primitives to rapidly cull empty space. Each 32-byte node packs its AABB extents into two vec4s, overloading the otherwise-unused w-channels to carry topology — inner nodes store left/right child indices, while leaf nodes encode a primitive triangle offset (signalled by a negative left-child sentinel). This compact, cache-coherent layout collapses ray-scene queries from O(n) brute force to an expected O(log n) walk of intersection tests.
    struct Node {
//  struct Node {
        vec4 aabbMinAndLeftChild; // xyz = AABB minimum corner; w = left-child node index, or a negative sentinel that marks this node as a leaf
//      vec4 aabbMinAndLeftChild; // xyz = AABB minimum corner; w = left-child node index, or a negative sentinel that marks this node as a leaf
        vec4 aabbMaxAndRightChild; // xyz = AABB maximum corner; w = right-child node index for inner nodes, reinterpreted as the primitive triangle index for leaves
//      vec4 aabbMaxAndRightChild; // xyz = AABB maximum corner; w = right-child node index for inner nodes, reinterpreted as the primitive triangle index for leaves
    };
//  };

    layout(std430, binding = 6) buffer BVHNodes {
//  layout(std430, binding = 6) buffer BVHNodes {
        Node nodes[];
//      Node nodes[];
    };
//  };

    // Global Geometry Buffer: a single flat SSBO of interleaved per-vertex attributes (position, normal, tangent, UV) packed into three 16-byte-aligned vec4s for coalesced, cache-friendly GPU reads. Triangles are stored as contiguous vertex triplets addressed by triangleIndex * 3, and this layout is shared with the rasterizer's VBOs so the hybrid pipeline never duplicates geometry between its raster and ray-traced passes.
//  // Global Geometry Buffer: a single flat SSBO of interleaved per-vertex attributes (position, normal, tangent, UV) packed into three 16-byte-aligned vec4s for coalesced, cache-friendly GPU reads. Triangles are stored as contiguous vertex triplets addressed by triangleIndex * 3, and this layout is shared with the rasterizer's VBOs so the hybrid pipeline never duplicates geometry between its raster and ray-traced passes.
    struct VertexData {
//  struct VertexData {
        vec4 positionAndTexcoordU; // xyz = world-space vertex position; w = texture coordinate U (packed to keep the vec4 fully utilized)
//      vec4 positionAndTexcoordU; // xyz = world-space vertex position; w = texture coordinate U (packed to keep the vec4 fully utilized)
        vec4 normalAndTexcoordV; // xyz = vertex shading normal; w = texture coordinate V
//      vec4 normalAndTexcoordV; // xyz = vertex shading normal; w = texture coordinate V
        vec4 tangentAndMaterialIndex; // xyz = vertex tangent (basis for normal mapping); w = float-encoded material index, read from the triangle's first vertex
//      vec4 tangentAndMaterialIndex; // xyz = vertex tangent (basis for normal mapping); w = float-encoded material index, read from the triangle's first vertex
    };
//  };

    layout(std430, binding = 7) buffer SceneVertices {
//  layout(std430, binding = 7) buffer SceneVertices {
        VertexData vertices[];
//      VertexData vertices[];
    };
//  };

    // Material Properties Buffer: stores the Principled/PBR parameter set per material — base color/albedo, perceptual roughness, metallic mask, transmission and index-of-refraction (IOR) for dielectrics, plus emissive strength and a UV tiling scale. Each material also carries float-encoded layer indices into the shared sampler2DArray (a value below -0.5 acting as an "absent texture" sentinel), enabling spatially varying, texture-driven evaluation of every channel during shading.
//  // Material Properties Buffer: stores the Principled/PBR parameter set per material — base color/albedo, perceptual roughness, metallic mask, transmission and index-of-refraction (IOR) for dielectrics, plus emissive strength and a UV tiling scale. Each material also carries float-encoded layer indices into the shared sampler2DArray (a value below -0.5 acting as an "absent texture" sentinel), enabling spatially varying, texture-driven evaluation of every channel during shading.
    struct Material {
//  struct Material {
        vec4 albedo;       // rgb = base color / diffuse reflectance (linear); w = unused padding for std430 16-byte alignment
//      vec4 albedo;       // rgb = base color / diffuse reflectance (linear); w = unused padding for std430 16-byte alignment
        float roughness; // perceptual roughness in [0,1]; squared into the GGX alpha and clamped to MIN_ROUGHNESS during shading
//      float roughness; // perceptual roughness in [0,1]; squared into the GGX alpha and clamped to MIN_ROUGHNESS during shading
        float metallic; // metalness mask in [0,1]: 0 = dielectric (F0 = 0.04), 1 = conductor (F0 = albedo, diffuse suppressed)
//      float metallic; // metalness mask in [0,1]: 0 = dielectric (F0 = 0.04), 1 = conductor (F0 = albedo, diffuse suppressed)
        float transmission; // fraction of energy refracted through the surface; > 0 enables the glass/dielectric transmission lobe
//      float transmission; // fraction of energy refracted through the surface; > 0 enables the glass/dielectric transmission lobe
        float ior; // index of refraction for the transmission lobe (e.g. ~1.5 for glass), feeding the Snell/Fresnel terms
//      float ior; // index of refraction for the transmission lobe (e.g. ~1.5 for glass), feeding the Snell/Fresnel terms
        float textureIndexAlbedo; // sampler2DArray layer for the albedo map (sRGB-encoded, decoded to linear on read), or < -0.5 if absent
//      float textureIndexAlbedo; // sampler2DArray layer for the albedo map (sRGB-encoded, decoded to linear on read), or < -0.5 if absent
        float textureIndexRoughness; // sampler2DArray layer for roughness, sampled from the R channel of the packed ORM texture, or < -0.5 if absent
//      float textureIndexRoughness; // sampler2DArray layer for roughness, sampled from the R channel of the packed ORM texture, or < -0.5 if absent
        float textureIndexMetallic; // sampler2DArray layer for metallic, sampled from the G channel of the packed ORM texture, or < -0.5 if absent
//      float textureIndexMetallic; // sampler2DArray layer for metallic, sampled from the G channel of the packed ORM texture, or < -0.5 if absent
        float textureIndexNormal; // sampler2DArray layer for the tangent-space normal map, or < -0.5 if absent
//      float textureIndexNormal; // sampler2DArray layer for the tangent-space normal map, or < -0.5 if absent
        float emissive; // emissive intensity multiplier; scales the emissive texture, or the albedo when no emissive map is bound
//      float emissive; // emissive intensity multiplier; scales the emissive texture, or the albedo when no emissive map is bound
        float textureIndexEmissive; // sampler2DArray layer for the emissive map (sRGB-decoded), or < -0.5 if absent
//      float textureIndexEmissive; // sampler2DArray layer for the emissive map (sRGB-decoded), or < -0.5 if absent
        float textureIndexTransmission; // sampler2DArray layer for transmission, sampled from the B channel of the packed ORM texture, or < -0.5 if absent
//      float textureIndexTransmission; // sampler2DArray layer for transmission, sampled from the B channel of the packed ORM texture, or < -0.5 if absent
        float padding002; // explicit padding to satisfy std430 scalar/vec2 alignment rules
//      float padding002; // explicit padding to satisfy std430 scalar/vec2 alignment rules
        vec2 uvScale; // per-material UV tiling factor applied to interpolated texture coordinates before sampling
//      vec2 uvScale; // per-material UV tiling factor applied to interpolated texture coordinates before sampling
        vec2 padding003; // explicit padding to pad the struct to a 16-byte multiple for std430
//      vec2 padding003; // explicit padding to pad the struct to a 16-byte multiple for std430
    };
//  };

    layout(std430, binding = 8) buffer SceneMaterials {
//  layout(std430, binding = 8) buffer SceneMaterials {
        Material materials[];
//      Material materials[];
    };
//  };

    struct RadianceCacheEntry {
//  struct RadianceCacheEntry {
        vec4 accumulatedRadianceAndSampleCount; // rgb = running radiance accumulator; w = sample count, saturating at CACHE_MAX_SAMPLES so the average keeps adapting
//      vec4 accumulatedRadianceAndSampleCount; // rgb = running radiance accumulator; w = sample count, saturating at CACHE_MAX_SAMPLES so the average keeps adapting
        vec4 worldPositionAndHashKey; // xyz = world position of the cached point (for spatial-proximity validation on read); w = bit-packed hash key identifying the occupying entry
//      vec4 worldPositionAndHashKey; // xyz = world position of the cached point (for spatial-proximity validation on read); w = bit-packed hash key identifying the occupying entry
        vec4 surfaceNormalAndLastWriteFrame; // xyz = surface normal, gated against the query normal to reject mismatched surfaces; w = frame index of the last write, used for stale-entry eviction
//      vec4 surfaceNormalAndLastWriteFrame; // xyz = surface normal, gated against the query normal to reject mismatched surfaces; w = frame index of the last write, used for stale-entry eviction
        vec4 reservedPadding; // reserved 16-byte slot for future per-entry data; also keeps the struct std430 16-byte aligned
//      vec4 reservedPadding; // reserved 16-byte slot for future per-entry data; also keeps the struct std430 16-byte aligned
    };
//  };

    layout(std430, binding = 9) buffer RadianceCache {
//  layout(std430, binding = 9) buffer RadianceCache {
        RadianceCacheEntry cacheEntries[];
//      RadianceCacheEntry cacheEntries[];
    };
//  };

    struct Ray {
//  struct Ray {
        vec3 origin; // ray start point in world space
//      vec3 origin; // ray start point in world space
        vec3 direction; // ray propagation direction in world space (kept normalized so hit distances read as true world units)
//      vec3 direction; // ray propagation direction in world space (kept normalized so hit distances read as true world units)
    };
//  };

    struct RayHitResult {
//  struct RayHitResult {
        vec3 hitSurfaceNormal; // interpolated surface normal at the hit, flipped to oppose the incoming ray for consistent shading
//      vec3 hitSurfaceNormal; // interpolated surface normal at the hit, flipped to oppose the incoming ray for consistent shading
        vec3 hitSurfaceTangent; // interpolated surface tangent at the hit, used to build the TBN basis for normal mapping
//      vec3 hitSurfaceTangent; // interpolated surface tangent at the hit, used to build the TBN basis for normal mapping
        vec2 hitSurfaceTexcoord; // interpolated UV at the hit; transiently holds the raw (u,v) barycentrics until the closest-hit post-pass resolves them
//      vec2 hitSurfaceTexcoord; // interpolated UV at the hit; transiently holds the raw (u,v) barycentrics until the closest-hit post-pass resolves them
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
        vec3 position; // light center in world space
//      vec3 position; // light center in world space
        float radius; // sphere-light radius; the facing hemisphere area 2*PI*r*r drives the NEE solid-angle PDF
//      float radius; // sphere-light radius; the facing hemisphere area 2*PI*r*r drives the NEE solid-angle PDF
        vec3 color; // emitted radiance (RGB), already scaled by the light's intensity
//      vec3 color; // emitted radiance (RGB), already scaled by the light's intensity
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
    uniform vec3 uCameraGlobalPosition;
//  uniform vec3 uCameraGlobalPosition;
    uniform int uFrameCount;
//  uniform int uFrameCount;
    uniform sampler2D uHdriTexture;
//  uniform sampler2D uHdriTexture;
    uniform sampler2DArray uSceneTextureArray;
//  uniform sampler2DArray uSceneTextureArray;
    uniform bool uUseHdri;
//  uniform bool uUseHdri;

    uniform vec3 uPixel00Coordinates;
//  uniform vec3 uPixel00Coordinates;
    uniform vec3 uPixelDeltaU;
//  uniform vec3 uPixelDeltaU;
    uniform vec3 uPixelDeltaV;
//  uniform vec3 uPixelDeltaV;
    uniform vec2 uJitter;
//  uniform vec2 uJitter;
    uniform uint uCacheFrameCounter;
//  uniform uint uCacheFrameCounter;
    uniform float uCacheBlendFactor;
//  uniform float uCacheBlendFactor;

    const float PI = 3.14159265359;
//  const float PI = 3.14159265359;
    const float INF = 1e30;
//  const float INF = 1e30;

    // Shading Tuning Constants
//  // Shading Tuning Constants
    const float HDRI_CLAMP = 20.0;
//  const float HDRI_CLAMP = 20.0;
    const float MIN_ROUGHNESS = 0.04;
//  const float MIN_ROUGHNESS = 0.04;
    const float NEE_PDF_EPSILON = 0.0001;
//  const float NEE_PDF_EPSILON = 0.0001;
    const float NEE_DIRECT_LIGHT_CLAMP = 50.0;
//  const float NEE_DIRECT_LIGHT_CLAMP = 50.0;
    const float RR_MIN_PROBABILITY = 0.05;
//  const float RR_MIN_PROBABILITY = 0.05;
    const float RR_MAX_PROBABILITY = 0.95;
//  const float RR_MAX_PROBABILITY = 0.95;
    const uint RR_MIN_BOUNCES = 4u;
//  const uint RR_MIN_BOUNCES = 4u;
    const float ACCUM_CLAMP = 20.0;
//  const float ACCUM_CLAMP = 20.0;
    const float F0_DEFAULT = 0.04;
//  const float F0_DEFAULT = 0.04;
    const float EPSILON_INTERSECT = 1.0e-7;
//  const float EPSILON_INTERSECT = 1.0e-7;
    const float EPSILON_OFFSET = 0.001;
//  const float EPSILON_OFFSET = 0.001;
    const float MISS_DISTANCE = INF;
//  const float MISS_DISTANCE = INF;
    const uint CACHE_SIZE = 131072u;
//  const uint CACHE_SIZE = 131072u;
    const float CACHE_CELL_SIZE = 0.5;
//  const float CACHE_CELL_SIZE = 0.5;
    const float CACHE_NORMAL_THRESHOLD = 0.7;
//  const float CACHE_NORMAL_THRESHOLD = 0.7;
    const float CACHE_MIN_SAMPLES = 4.0;
//  const float CACHE_MIN_SAMPLES = 4.0;
    const float CACHE_MAX_SAMPLES = 256.0;
//  const float CACHE_MAX_SAMPLES = 256.0;
    const float CACHE_EVICTION_AGE = 30.0;
//  const float CACHE_EVICTION_AGE = 30.0;
    const float CACHE_ROUGHNESS_GATE = 0.3;
//  const float CACHE_ROUGHNESS_GATE = 0.3;

    // Hardcoded Math Constants
//  // Hardcoded Math Constants
    const float EPSILON_MATH = 0.0001;
//  const float EPSILON_MATH = 0.0001;
    const float EPSILON_DOT = 0.001;
//  const float EPSILON_DOT = 0.001;
    const float EPSILON_RAND = 0.999;
//  const float EPSILON_RAND = 0.999;
    const int MAX_BOUNCES = 32;
//  const int MAX_BOUNCES = 32;

    // Xorshift/Wang Hash Pseudo-Random Number Generator (PRNG): a fast integer-avalanche hash maintaining a single 32-bit state, reseeded per-pixel and per-frame so sample streams stay decorrelated across the image and over time (avoiding structured, fixed-pattern noise). Each call avalanches the state through shifts and multiplies, then normalizes it to a uniform float in [0, 1) — supplying the cheap, statistically robust variates that drive every Monte Carlo decision: hemisphere direction sampling, GGX lobe selection, Russian Roulette termination, and Next Event Estimation.
//  // Xorshift/Wang Hash Pseudo-Random Number Generator (PRNG): a fast integer-avalanche hash maintaining a single 32-bit state, reseeded per-pixel and per-frame so sample streams stay decorrelated across the image and over time (avoiding structured, fixed-pattern noise). Each call avalanches the state through shifts and multiplies, then normalizes it to a uniform float in [0, 1) — supplying the cheap, statistically robust variates that drive every Monte Carlo decision: hemisphere direction sampling, GGX lobe selection, Russian Roulette termination, and Next Event Estimation.
    uint seed;
//  uint seed;
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
    void initRNG(vec2 pixel) {
//  void initRNG(vec2 pixel) {
        // Use float coordinates and wang_hash for better random distribution
//      // Use float coordinates and wang_hash for better random distribution
        seed = uint(pixel.x * 1973.0 + pixel.y * 9277.0) + uint(uFrameCount) * 26699u;
//      seed = uint(pixel.x * 1973.0 + pixel.y * 9277.0) + uint(uFrameCount) * 26699u;
        seed = wang_hash(seed);
//      seed = wang_hash(seed);
    }
//  }

    uint hashCacheKey(vec3 worldPosition, vec3 surfaceNormal) {
//  uint hashCacheKey(vec3 worldPosition, vec3 surfaceNormal) {
        ivec3 cellCoordinate = ivec3(floor(worldPosition / CACHE_CELL_SIZE));
//      ivec3 cellCoordinate = ivec3(floor(worldPosition / CACHE_CELL_SIZE));
        uint normalDirectionIndex = 0u;
//      uint normalDirectionIndex = 0u;
        vec3 absoluteNormal = abs(surfaceNormal);
//      vec3 absoluteNormal = abs(surfaceNormal);
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

    bool readCache(vec3 worldPosition, vec3 surfaceNormal, out vec3 outRadiance) {
//  bool readCache(vec3 worldPosition, vec3 surfaceNormal, out vec3 outRadiance) {
        uint hashKey = hashCacheKey(worldPosition, surfaceNormal);
//      uint hashKey = hashCacheKey(worldPosition, surfaceNormal);
        uint slotIndex = cacheSlot(hashKey);
//      uint slotIndex = cacheSlot(hashKey);
        RadianceCacheEntry entry = cacheEntries[slotIndex];
//      RadianceCacheEntry entry = cacheEntries[slotIndex];
        if (floatBitsToUint(entry.worldPositionAndHashKey.w) != hashKey) return false;
//      if (floatBitsToUint(entry.worldPositionAndHashKey.w) != hashKey) return false;
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

    void writeCache(vec3 worldPosition, vec3 surfaceNormal, vec3 radiance) {
//  void writeCache(vec3 worldPosition, vec3 surfaceNormal, vec3 radiance) {
        uint hashKey = hashCacheKey(worldPosition, surfaceNormal);
//      uint hashKey = hashCacheKey(worldPosition, surfaceNormal);
        uint slotIndex = cacheSlot(hashKey);
//      uint slotIndex = cacheSlot(hashKey);
        RadianceCacheEntry entry = cacheEntries[slotIndex];
//      RadianceCacheEntry entry = cacheEntries[slotIndex];
        uint existingHashKey = floatBitsToUint(entry.worldPositionAndHashKey.w);
//      uint existingHashKey = floatBitsToUint(entry.worldPositionAndHashKey.w);
        if (existingHashKey == hashKey) {
//      if (existingHashKey == hashKey) {
            float sampleCount = min(entry.accumulatedRadianceAndSampleCount.w + 1.0, CACHE_MAX_SAMPLES);
//          float sampleCount = min(entry.accumulatedRadianceAndSampleCount.w + 1.0, CACHE_MAX_SAMPLES);
            float blendAlpha = 1.0 / sampleCount;
//          float blendAlpha = 1.0 / sampleCount;
            vec3 previousAverageRadiance = entry.accumulatedRadianceAndSampleCount.rgb / max(entry.accumulatedRadianceAndSampleCount.w, 1.0);
//          vec3 previousAverageRadiance = entry.accumulatedRadianceAndSampleCount.rgb / max(entry.accumulatedRadianceAndSampleCount.w, 1.0);
            cacheEntries[slotIndex].accumulatedRadianceAndSampleCount = vec4(mix(previousAverageRadiance, radiance, blendAlpha) * sampleCount, sampleCount);
//          cacheEntries[slotIndex].accumulatedRadianceAndSampleCount = vec4(mix(previousAverageRadiance, radiance, blendAlpha) * sampleCount, sampleCount);
            cacheEntries[slotIndex].surfaceNormalAndLastWriteFrame.w = float(uCacheFrameCounter);
//          cacheEntries[slotIndex].surfaceNormalAndLastWriteFrame.w = float(uCacheFrameCounter);
        } else {
//      } else {
            float entryAge = float(uCacheFrameCounter) - entry.surfaceNormalAndLastWriteFrame.w;
//          float entryAge = float(uCacheFrameCounter) - entry.surfaceNormalAndLastWriteFrame.w;
            if (existingHashKey == 0u || entryAge > CACHE_EVICTION_AGE || entry.accumulatedRadianceAndSampleCount.w < 1.0) {
//          if (existingHashKey == 0u || entryAge > CACHE_EVICTION_AGE || entry.accumulatedRadianceAndSampleCount.w < 1.0) {
                cacheEntries[slotIndex].accumulatedRadianceAndSampleCount = vec4(radiance, 1.0);
//              cacheEntries[slotIndex].accumulatedRadianceAndSampleCount = vec4(radiance, 1.0);
                cacheEntries[slotIndex].worldPositionAndHashKey = vec4(worldPosition, uintBitsToFloat(hashKey));
//              cacheEntries[slotIndex].worldPositionAndHashKey = vec4(worldPosition, uintBitsToFloat(hashKey));
                cacheEntries[slotIndex].surfaceNormalAndLastWriteFrame = vec4(surfaceNormal, float(uCacheFrameCounter));
//              cacheEntries[slotIndex].surfaceNormalAndLastWriteFrame = vec4(surfaceNormal, float(uCacheFrameCounter));
                cacheEntries[slotIndex].reservedPadding = vec4(0.0);
//              cacheEntries[slotIndex].reservedPadding = vec4(0.0);
            }
//          }
        }
//      }
    }
//  }

    vec3 randomUnitVector() {
//  vec3 randomUnitVector() {
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
        return vec3(directionX, directionY, cosPolar);
//      return vec3(directionX, directionY, cosPolar);
    }
//  }

    // Environment Lighting: evaluates the infinitely distant background radiance arriving along a ray's direction, serving as both the visible backdrop and the scene's image-based light source. When an HDR map is bound, the direction is projected into equirectangular (latitude-longitude) UVs and sampled, with the result clamped to suppress fireflies seeded by a few extreme-intensity texels. Otherwise it synthesizes a sky analytically — a Rayleigh-style zenith-to-horizon gradient, an exponential Mie horizon haze, and layered power-of-cosine glows that build a soft-to-sharp sun disk.
//  // Environment Lighting: evaluates the infinitely distant background radiance arriving along a ray's direction, serving as both the visible backdrop and the scene's image-based light source. When an HDR map is bound, the direction is projected into equirectangular (latitude-longitude) UVs and sampled, with the result clamped to suppress fireflies seeded by a few extreme-intensity texels. Otherwise it synthesizes a sky analytically — a Rayleigh-style zenith-to-horizon gradient, an exponential Mie horizon haze, and layered power-of-cosine glows that build a soft-to-sharp sun disk.
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
    float maxVec3(vec3 vector) { return max(vector.x, max(vector.y, vector.z)); }
//  float maxVec3(vec3 vector) { return max(vector.x, max(vector.y, vector.z)); }
    float minVec3(vec3 vector) { return min(vector.x, min(vector.y, vector.z)); }
//  float minVec3(vec3 vector) { return min(vector.x, min(vector.y, vector.z)); }

    // Ray-AABB Intersection (Slab Method): a branchless overlap test against the three pairs of parallel "slab" planes that bound an AABB. Using the precomputed reciprocal ray direction (turning costly divides into multiplies), it derives per-axis entry/exit distances, then collapses them to the latest entry and earliest exit via component-wise min/max — the box is hit only if that interval stays non-empty within the ray's valid travel range. Robust against axis-aligned rays through the reciprocal's signed infinities, this is the per-node rejection test that prunes whole subtrees during BVH traversal.
//  // Ray-AABB Intersection (Slab Method): a branchless overlap test against the three pairs of parallel "slab" planes that bound an AABB. Using the precomputed reciprocal ray direction (turning costly divides into multiplies), it derives per-axis entry/exit distances, then collapses them to the latest entry and earliest exit via component-wise min/max — the box is hit only if that interval stays non-empty within the ray's valid travel range. Robust against axis-aligned rays through the reciprocal's signed infinities, this is the per-node rejection test that prunes whole subtrees during BVH traversal.
    bool intersectAABB_Interval(vec3 inverseRayDirection, vec3 rayOrigin, Interval rayTravelDistanceLimit, vec3 boxMinBound, vec3 boxMaxBound) {
//  bool intersectAABB_Interval(vec3 inverseRayDirection, vec3 rayOrigin, Interval rayTravelDistanceLimit, vec3 boxMinBound, vec3 boxMaxBound) {
        vec3 distanceToMinPlanes = (boxMinBound - rayOrigin) * inverseRayDirection;
//      vec3 distanceToMinPlanes = (boxMinBound - rayOrigin) * inverseRayDirection;
        vec3 distanceToMaxPlanes = (boxMaxBound - rayOrigin) * inverseRayDirection;
//      vec3 distanceToMaxPlanes = (boxMaxBound - rayOrigin) * inverseRayDirection;
        vec3 perAxisNearDistance = min(distanceToMinPlanes, distanceToMaxPlanes);
//      vec3 perAxisNearDistance = min(distanceToMinPlanes, distanceToMaxPlanes);
        vec3 perAxisFarDistance = max(distanceToMinPlanes, distanceToMaxPlanes);
//      vec3 perAxisFarDistance = max(distanceToMinPlanes, distanceToMaxPlanes);

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
    float calculateDistanceToAABB3D(vec3 inverseRayDirection, vec3 rayOrigin, Interval rayTravelDistanceLimit, vec3 boxMinBound, vec3 boxMaxBound) {
//  float calculateDistanceToAABB3D(vec3 inverseRayDirection, vec3 rayOrigin, Interval rayTravelDistanceLimit, vec3 boxMinBound, vec3 boxMaxBound) {
        vec3 distanceToMinPlanes = (boxMinBound - rayOrigin) * inverseRayDirection;
//      vec3 distanceToMinPlanes = (boxMinBound - rayOrigin) * inverseRayDirection;
        vec3 distanceToMaxPlanes = (boxMaxBound - rayOrigin) * inverseRayDirection;
//      vec3 distanceToMaxPlanes = (boxMaxBound - rayOrigin) * inverseRayDirection;

        vec3 clampedNearDistances = max(min(distanceToMinPlanes, distanceToMaxPlanes), vec3(rayTravelDistanceLimit.min));
//      vec3 clampedNearDistances = max(min(distanceToMinPlanes, distanceToMaxPlanes), vec3(rayTravelDistanceLimit.min));
        vec3 clampedFarDistances = min(max(distanceToMinPlanes, distanceToMaxPlanes), vec3(rayTravelDistanceLimit.max));
//      vec3 clampedFarDistances = min(max(distanceToMinPlanes, distanceToMaxPlanes), vec3(rayTravelDistanceLimit.max));

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
        vec3 vertexPosition0 = vertices[vertexBaseIndex + 0].positionAndTexcoordU.xyz;
//      vec3 vertexPosition0 = vertices[vertexBaseIndex + 0].positionAndTexcoordU.xyz;
        vec3 vertexPosition1 = vertices[vertexBaseIndex + 1].positionAndTexcoordU.xyz;
//      vec3 vertexPosition1 = vertices[vertexBaseIndex + 1].positionAndTexcoordU.xyz;
        vec3 vertexPosition2 = vertices[vertexBaseIndex + 2].positionAndTexcoordU.xyz;
//      vec3 vertexPosition2 = vertices[vertexBaseIndex + 2].positionAndTexcoordU.xyz;

        const float intersectionEpsilon = EPSILON_INTERSECT;
//      const float intersectionEpsilon = EPSILON_INTERSECT;

        vec3 edge01 = vertexPosition1 - vertexPosition0;
//      vec3 edge01 = vertexPosition1 - vertexPosition0;
        vec3 edge02 = vertexPosition2 - vertexPosition0;
//      vec3 edge02 = vertexPosition2 - vertexPosition0;

        vec3 directionCrossEdge02 = cross(ray.direction, edge02);
//      vec3 directionCrossEdge02 = cross(ray.direction, edge02);
        float determinant = dot(edge01, directionCrossEdge02);
//      float determinant = dot(edge01, directionCrossEdge02);

        if (abs(determinant) < intersectionEpsilon) return false;
//      if (abs(determinant) < intersectionEpsilon) return false;

        float inverseDeterminant = 1.0 / determinant;
//      float inverseDeterminant = 1.0 / determinant;
        vec3 originToVertex0 = ray.origin - vertexPosition0;
//      vec3 originToVertex0 = ray.origin - vertexPosition0;
        float barycentricU = inverseDeterminant * dot(originToVertex0, directionCrossEdge02);
//      float barycentricU = inverseDeterminant * dot(originToVertex0, directionCrossEdge02);

        if (barycentricU < 0.0 || barycentricU > 1.0) return false;
//      if (barycentricU < 0.0 || barycentricU > 1.0) return false;

        vec3 originCrossEdge01 = cross(originToVertex0, edge01);
//      vec3 originCrossEdge01 = cross(originToVertex0, edge01);
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
        vec3 vertexPosition0 = vertices[vertexBaseIndex + 0].positionAndTexcoordU.xyz;
//      vec3 vertexPosition0 = vertices[vertexBaseIndex + 0].positionAndTexcoordU.xyz;
        vec3 vertexPosition1 = vertices[vertexBaseIndex + 1].positionAndTexcoordU.xyz;
//      vec3 vertexPosition1 = vertices[vertexBaseIndex + 1].positionAndTexcoordU.xyz;
        vec3 vertexPosition2 = vertices[vertexBaseIndex + 2].positionAndTexcoordU.xyz;
//      vec3 vertexPosition2 = vertices[vertexBaseIndex + 2].positionAndTexcoordU.xyz;

        const float intersectionEpsilon = EPSILON_INTERSECT;
//      const float intersectionEpsilon = EPSILON_INTERSECT;
        vec3 edge01 = vertexPosition1 - vertexPosition0;
//      vec3 edge01 = vertexPosition1 - vertexPosition0;
        vec3 edge02 = vertexPosition2 - vertexPosition0;
//      vec3 edge02 = vertexPosition2 - vertexPosition0;

        vec3 directionCrossEdge02 = cross(ray.direction, edge02);
//      vec3 directionCrossEdge02 = cross(ray.direction, edge02);
        float determinant = dot(edge01, directionCrossEdge02);
//      float determinant = dot(edge01, directionCrossEdge02);

        if (abs(determinant) < intersectionEpsilon) return hitResult;
//      if (abs(determinant) < intersectionEpsilon) return hitResult;

        float inverseDeterminant = 1.0 / determinant;
//      float inverseDeterminant = 1.0 / determinant;
        vec3 originToVertex0 = ray.origin - vertexPosition0;
//      vec3 originToVertex0 = ray.origin - vertexPosition0;
        float barycentricU = inverseDeterminant * dot(originToVertex0, directionCrossEdge02);
//      float barycentricU = inverseDeterminant * dot(originToVertex0, directionCrossEdge02);

        if (barycentricU < 0.0 || barycentricU > 1.0) return hitResult;
//      if (barycentricU < 0.0 || barycentricU > 1.0) return hitResult;

        vec3 originCrossEdge01 = cross(originToVertex0, edge01);
//      vec3 originCrossEdge01 = cross(originToVertex0, edge01);
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
            hitResult.hitSurfaceTexcoord = vec2(barycentricU, barycentricV);
//          hitResult.hitSurfaceTexcoord = vec2(barycentricU, barycentricV);

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
        vec3 directionSign = sign(ray.direction);
//      vec3 directionSign = sign(ray.direction);
        directionSign += 1.0 - abs(directionSign);
//      directionSign += 1.0 - abs(directionSign);
        vec3 safeRayDirection = ray.direction + step(abs(ray.direction), vec3(1e-8)) * directionSign * 1e-8;
//      vec3 safeRayDirection = ray.direction + step(abs(ray.direction), vec3(1e-8)) * directionSign * 1e-8;
        vec3 inverseRayDirection = 1.0 / safeRayDirection;
//      vec3 inverseRayDirection = 1.0 / safeRayDirection;
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

        vec3 directionSign = sign(ray.direction);
//      vec3 directionSign = sign(ray.direction);
        directionSign += 1.0 - abs(directionSign);
//      directionSign += 1.0 - abs(directionSign);
        vec3 safeRayDirection = ray.direction + step(abs(ray.direction), vec3(1e-8)) * directionSign * 1e-8;
//      vec3 safeRayDirection = ray.direction + step(abs(ray.direction), vec3(1e-8)) * directionSign * 1e-8;
        vec3 inverseRayDirection = 1.0 / safeRayDirection;
//      vec3 inverseRayDirection = 1.0 / safeRayDirection;
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
            vec2 texcoord0 = vec2(vertices[vertexBaseIndex + 0].positionAndTexcoordU.w, vertices[vertexBaseIndex + 0].normalAndTexcoordV.w);
//          vec2 texcoord0 = vec2(vertices[vertexBaseIndex + 0].positionAndTexcoordU.w, vertices[vertexBaseIndex + 0].normalAndTexcoordV.w);
            vec2 texcoord1 = vec2(vertices[vertexBaseIndex + 1].positionAndTexcoordU.w, vertices[vertexBaseIndex + 1].normalAndTexcoordV.w);
//          vec2 texcoord1 = vec2(vertices[vertexBaseIndex + 1].positionAndTexcoordU.w, vertices[vertexBaseIndex + 1].normalAndTexcoordV.w);
            vec2 texcoord2 = vec2(vertices[vertexBaseIndex + 2].positionAndTexcoordU.w, vertices[vertexBaseIndex + 2].normalAndTexcoordV.w);
//          vec2 texcoord2 = vec2(vertices[vertexBaseIndex + 2].positionAndTexcoordU.w, vertices[vertexBaseIndex + 2].normalAndTexcoordV.w);
            closestHitResult.hitSurfaceTexcoord = (1.0 - barycentricU - barycentricV) * texcoord0 + barycentricU * texcoord1 + barycentricV * texcoord2;
//          closestHitResult.hitSurfaceTexcoord = (1.0 - barycentricU - barycentricV) * texcoord0 + barycentricU * texcoord1 + barycentricV * texcoord2;

            // Interpolate Normals
//          // Interpolate Normals
            vec3 normal0 = vertices[vertexBaseIndex + 0].normalAndTexcoordV.xyz;
//          vec3 normal0 = vertices[vertexBaseIndex + 0].normalAndTexcoordV.xyz;
            vec3 normal1 = vertices[vertexBaseIndex + 1].normalAndTexcoordV.xyz;
//          vec3 normal1 = vertices[vertexBaseIndex + 1].normalAndTexcoordV.xyz;
            vec3 normal2 = vertices[vertexBaseIndex + 2].normalAndTexcoordV.xyz;
//          vec3 normal2 = vertices[vertexBaseIndex + 2].normalAndTexcoordV.xyz;
            closestHitResult.hitSurfaceNormal = normalize((1.0 - barycentricU - barycentricV) * normal0 + barycentricU * normal1 + barycentricV * normal2);
//          closestHitResult.hitSurfaceNormal = normalize((1.0 - barycentricU - barycentricV) * normal0 + barycentricU * normal1 + barycentricV * normal2);

            // Interpolate Tangents
//          // Interpolate Tangents
            vec3 tangent0 = vertices[vertexBaseIndex + 0].tangentAndMaterialIndex.xyz;
//          vec3 tangent0 = vertices[vertexBaseIndex + 0].tangentAndMaterialIndex.xyz;
            vec3 tangent1 = vertices[vertexBaseIndex + 1].tangentAndMaterialIndex.xyz;
//          vec3 tangent1 = vertices[vertexBaseIndex + 1].tangentAndMaterialIndex.xyz;
            vec3 tangent2 = vertices[vertexBaseIndex + 2].tangentAndMaterialIndex.xyz;
//          vec3 tangent2 = vertices[vertexBaseIndex + 2].tangentAndMaterialIndex.xyz;
            closestHitResult.hitSurfaceTangent = normalize((1.0 - barycentricU - barycentricV) * tangent0 + barycentricU * tangent1 + barycentricV * tangent2);
//          closestHitResult.hitSurfaceTangent = normalize((1.0 - barycentricU - barycentricV) * tangent0 + barycentricU * tangent1 + barycentricV * tangent2);
        }
//      }

        return closestHitResult;
//      return closestHitResult;
    }
//  }

    // Principled BSDF Components: the building blocks of a Disney-style physically-based surface model. These comprise Schlick's Fresnel approximation for view-dependent reflectivity, the GGX (Trowbridge-Reitz) microfacet Normal Distribution Function together with cosine-of-elevation importance sampling of its half-vectors, three interchangeable diffuse lobes (Lambert, Disney retro-reflective, Oren-Nayar), and the perfect reflect/refract operators. The full evaluation and PDF routines below compose these into a single energy-conserving model for realistic rough-surface scattering.
//  // Principled BSDF Components: the building blocks of a Disney-style physically-based surface model. These comprise Schlick's Fresnel approximation for view-dependent reflectivity, the GGX (Trowbridge-Reitz) microfacet Normal Distribution Function together with cosine-of-elevation importance sampling of its half-vectors, three interchangeable diffuse lobes (Lambert, Disney retro-reflective, Oren-Nayar), and the perfect reflect/refract operators. The full evaluation and PDF routines below compose these into a single energy-conserving model for realistic rough-surface scattering.
    vec3 schlickFresnel(float cosineIncidentAngle, vec3 reflectanceAtNormalIncidence) {
//  vec3 schlickFresnel(float cosineIncidentAngle, vec3 reflectanceAtNormalIncidence) {
        float oneMinusCosine = 1.0 - cosineIncidentAngle;
//      float oneMinusCosine = 1.0 - cosineIncidentAngle;
        float oneMinusCosinePow2 = oneMinusCosine * oneMinusCosine;
//      float oneMinusCosinePow2 = oneMinusCosine * oneMinusCosine;
        float oneMinusCosinePow4 = oneMinusCosinePow2 * oneMinusCosinePow2;
//      float oneMinusCosinePow4 = oneMinusCosinePow2 * oneMinusCosinePow2;
        float oneMinusCosinePow5 = oneMinusCosinePow4 * oneMinusCosine;
//      float oneMinusCosinePow5 = oneMinusCosinePow4 * oneMinusCosine;
        return reflectanceAtNormalIncidence + (1.0 - reflectanceAtNormalIncidence) * oneMinusCosinePow5;
//      return reflectanceAtNormalIncidence + (1.0 - reflectanceAtNormalIncidence) * oneMinusCosinePow5;
    }
//  }

    vec3 sampleGGX(vec3 normal, float roughness) {
//  vec3 sampleGGX(vec3 normal, float roughness) {
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
        float sinElevation = sqrt(1.0 - cosElevation * cosElevation);
//      float sinElevation = sqrt(1.0 - cosElevation * cosElevation);

        float localX = sinElevation * cos(azimuthAngle);
//      float localX = sinElevation * cos(azimuthAngle);
        float localY = sinElevation * sin(azimuthAngle);
//      float localY = sinElevation * sin(azimuthAngle);
        float localZ = cosElevation;
//      float localZ = cosElevation;

        // Tangent space basis construction (orthonormal basis)
//      // Tangent space basis construction (orthonormal basis)
        vec3 basisHelper;
//      vec3 basisHelper;
        if (abs(normal.z) < EPSILON_RAND) {
//      if (abs(normal.z) < EPSILON_RAND) {
            basisHelper = vec3(0.0, 0.0, 1.0);
//          basisHelper = vec3(0.0, 0.0, 1.0);
        } else {
//      } else {
            basisHelper = vec3(1.0, 0.0, 0.0);
//          basisHelper = vec3(1.0, 0.0, 0.0);
        }
//      }
        vec3 tangentAxis = normalize(cross(basisHelper, normal));
//      vec3 tangentAxis = normalize(cross(basisHelper, normal));
        vec3 bitangentAxis = cross(normal, tangentAxis);
//      vec3 bitangentAxis = cross(normal, tangentAxis);

        // Transform the sampled half-vector to world space
//      // Transform the sampled half-vector to world space
        return normalize(tangentAxis * localX + bitangentAxis * localY + normal * localZ);
//      return normalize(tangentAxis * localX + bitangentAxis * localY + normal * localZ);
    }
//  }

    vec3 reflectPrincipled(vec3 incomingVector, vec3 normal) {
//  vec3 reflectPrincipled(vec3 incomingVector, vec3 normal) {
        return incomingVector - 2.0 * dot(incomingVector, normal) * normal;
//      return incomingVector - 2.0 * dot(incomingVector, normal) * normal;
    }
//  }

    vec3 refractPrincipled(vec3 incomingVector, vec3 normal, float etaRatioOfIncidenceOverTransmission, float cosIncidentAngle, float sinTransmittedAngleSquared) {
//  vec3 refractPrincipled(vec3 incomingVector, vec3 normal, float etaRatioOfIncidenceOverTransmission, float cosIncidentAngle, float sinTransmittedAngleSquared) {
        float cosTransmittedAngle = sqrt(max(0.0, 1.0 - sinTransmittedAngleSquared));
//      float cosTransmittedAngle = sqrt(max(0.0, 1.0 - sinTransmittedAngleSquared));
        vec3 refractedDirection = normalize(etaRatioOfIncidenceOverTransmission * incomingVector + (etaRatioOfIncidenceOverTransmission * cosIncidentAngle - cosTransmittedAngle) * normal);
//      vec3 refractedDirection = normalize(etaRatioOfIncidenceOverTransmission * incomingVector + (etaRatioOfIncidenceOverTransmission * cosIncidentAngle - cosTransmittedAngle) * normal);
        return refractedDirection;
//      return refractedDirection;
    }
//  }

    vec3 evalDisneyDiffuse(vec3 surfaceNormal, vec3 viewDirection, vec3 lightDirection, vec3 albedo, float roughness) {
//  vec3 evalDisneyDiffuse(vec3 surfaceNormal, vec3 viewDirection, vec3 lightDirection, vec3 albedo, float roughness) {
        vec3 halfVectorUnnormalized = viewDirection + lightDirection;
//      vec3 halfVectorUnnormalized = viewDirection + lightDirection;
        vec3 halfVector;
//      vec3 halfVector;
        if (dot(halfVectorUnnormalized, halfVectorUnnormalized) > EPSILON_MATH) {
//      if (dot(halfVectorUnnormalized, halfVectorUnnormalized) > EPSILON_MATH) {
            halfVector = normalize(halfVectorUnnormalized);
//          halfVector = normalize(halfVectorUnnormalized);
        } else {
//      } else {
            halfVector = surfaceNormal;
//          halfVector = surfaceNormal;
        }
//      }
        float normalDotLight = max(dot(surfaceNormal, lightDirection), 0.0);
//      float normalDotLight = max(dot(surfaceNormal, lightDirection), 0.0);
        float normalDotView = max(dot(surfaceNormal, viewDirection), 0.0);
//      float normalDotView = max(dot(surfaceNormal, viewDirection), 0.0);
        float lightDotHalf = max(dot(lightDirection, halfVector), 0.0);
//      float lightDotHalf = max(dot(lightDirection, halfVector), 0.0);

        // Schlick weight for grazing angles
//      // Schlick weight for grazing angles
        float fresnelDiffuse90 = 0.5 + 2.0 * roughness * lightDotHalf * lightDotHalf;
//      float fresnelDiffuse90 = 0.5 + 2.0 * roughness * lightDotHalf * lightDotHalf;

        float oneMinusNormalDotLight = 1.0 - normalDotLight;
//      float oneMinusNormalDotLight = 1.0 - normalDotLight;
        float oneMinusNormalDotLightPow2 = oneMinusNormalDotLight * oneMinusNormalDotLight;
//      float oneMinusNormalDotLightPow2 = oneMinusNormalDotLight * oneMinusNormalDotLight;
        float oneMinusNormalDotLightPow5 = oneMinusNormalDotLightPow2 * oneMinusNormalDotLightPow2 * oneMinusNormalDotLight;
//      float oneMinusNormalDotLightPow5 = oneMinusNormalDotLightPow2 * oneMinusNormalDotLightPow2 * oneMinusNormalDotLight;
        float lightScatterFactor = 1.0 + (fresnelDiffuse90 - 1.0) * oneMinusNormalDotLightPow5;
//      float lightScatterFactor = 1.0 + (fresnelDiffuse90 - 1.0) * oneMinusNormalDotLightPow5;

        float oneMinusNormalDotView = 1.0 - normalDotView;
//      float oneMinusNormalDotView = 1.0 - normalDotView;
        float oneMinusNormalDotViewPow2 = oneMinusNormalDotView * oneMinusNormalDotView;
//      float oneMinusNormalDotViewPow2 = oneMinusNormalDotView * oneMinusNormalDotView;
        float oneMinusNormalDotViewPow5 = oneMinusNormalDotViewPow2 * oneMinusNormalDotViewPow2 * oneMinusNormalDotView;
//      float oneMinusNormalDotViewPow5 = oneMinusNormalDotViewPow2 * oneMinusNormalDotViewPow2 * oneMinusNormalDotView;
        float viewScatterFactor = 1.0 + (fresnelDiffuse90 - 1.0) * oneMinusNormalDotViewPow5;
//      float viewScatterFactor = 1.0 + (fresnelDiffuse90 - 1.0) * oneMinusNormalDotViewPow5;

        return (albedo / PI) * lightScatterFactor * viewScatterFactor;
//      return (albedo / PI) * lightScatterFactor * viewScatterFactor;
    }
//  }

    vec3 evalOrenNayarDiffuse(vec3 surfaceNormal, vec3 viewDirection, vec3 lightDirection, vec3 albedo, float roughness) {
//  vec3 evalOrenNayarDiffuse(vec3 surfaceNormal, vec3 viewDirection, vec3 lightDirection, vec3 albedo, float roughness) {
        float normalDotLight = max(dot(surfaceNormal, lightDirection), 0.0);
//      float normalDotLight = max(dot(surfaceNormal, lightDirection), 0.0);
        float normalDotView = max(dot(surfaceNormal, viewDirection), 0.0);
//      float normalDotView = max(dot(surfaceNormal, viewDirection), 0.0);

        float lightDotView = dot(lightDirection, viewDirection);
//      float lightDotView = dot(lightDirection, viewDirection);
        float geometricNumerator = lightDotView - normalDotLight * normalDotView;
//      float geometricNumerator = lightDotView - normalDotLight * normalDotView;
        float geometricDenominator = mix(1.0, max(normalDotLight, normalDotView), step(0.0, geometricNumerator));
//      float geometricDenominator = mix(1.0, max(normalDotLight, normalDotView), step(0.0, geometricNumerator));

        float sigmaSquared = roughness * roughness;
//      float sigmaSquared = roughness * roughness;
        float orenNayarTermA = 1.0 - 0.5 * (sigmaSquared / (sigmaSquared + 0.33));
//      float orenNayarTermA = 1.0 - 0.5 * (sigmaSquared / (sigmaSquared + 0.33));
        float orenNayarTermB = 0.45 * (sigmaSquared / (sigmaSquared + 0.09));
//      float orenNayarTermB = 0.45 * (sigmaSquared / (sigmaSquared + 0.09));

        return (albedo / PI) * (orenNayarTermA + orenNayarTermB * (geometricNumerator / (geometricDenominator + EPSILON_MATH)));
//      return (albedo / PI) * (orenNayarTermA + orenNayarTermB * (geometricNumerator / (geometricDenominator + EPSILON_MATH)));
    }
//  }

    vec3 evalLambertDiffuse(vec3 albedo) {
//  vec3 evalLambertDiffuse(vec3 albedo) {
        return albedo / PI;
//      return albedo / PI;
    }
//  }

    // Fused BSDF + PDF Evaluation: returns the bidirectional reflectance value f_r for a given view/light direction pair (the per-steradian throughput, before any cosine or PDF weighting) and, through the out parameter, the matching importance-sampling PDF. The BSDF sums a Disney diffuse lobe with a Cook-Torrance specular term — the GGX distribution D, the Smith geometry factor G, and the Schlick Fresnel F combined as D·F·G / (4·N·V·N·L) — where Fresnel partitions energy between the lobes and the metallic mask removes the diffuse component from conductors. Every scatter and NEE sample needs both the value and the PDF, so fusing them shares the half-vector, dot products, Fresnel and GGX D terms the previously separate routines each recomputed, halving the per-sample shading math without changing a single term of either result.
//  // Fused BSDF + PDF Evaluation: returns the bidirectional reflectance value f_r for a given view/light direction pair (the per-steradian throughput, before any cosine or PDF weighting) and, through the out parameter, the matching importance-sampling PDF. The BSDF sums a Disney diffuse lobe with a Cook-Torrance specular term — the GGX distribution D, the Smith geometry factor G, and the Schlick Fresnel F combined as D·F·G / (4·N·V·N·L) — where Fresnel partitions energy between the lobes and the metallic mask removes the diffuse component from conductors. Every scatter and NEE sample needs both the value and the PDF, so fusing them shares the half-vector, dot products, Fresnel and GGX D terms the previously separate routines each recomputed, halving the per-sample shading math without changing a single term of either result.
    vec3 evalPrincipledBSDFAndPDF(vec3 incomingDirection, vec3 outgoingDirection, vec3 normal, vec3 albedo, float roughness, float metallic, float transmission, out float outPdf) {
//  vec3 evalPrincipledBSDFAndPDF(vec3 incomingDirection, vec3 outgoingDirection, vec3 normal, vec3 albedo, float roughness, float metallic, float transmission, out float outPdf) {
        outPdf = 0.0;
//      outPdf = 0.0;
        vec3 surfaceNormal = normal;
//      vec3 surfaceNormal = normal;
        vec3 viewDirection = -incomingDirection;
//      vec3 viewDirection = -incomingDirection;
        vec3 lightDirection = outgoingDirection;
//      vec3 lightDirection = outgoingDirection;
        vec3 halfVectorUnnormalized = viewDirection + lightDirection;
//      vec3 halfVectorUnnormalized = viewDirection + lightDirection;
        vec3 halfVector;
//      vec3 halfVector;
        if (dot(halfVectorUnnormalized, halfVectorUnnormalized) > EPSILON_MATH) {
//      if (dot(halfVectorUnnormalized, halfVectorUnnormalized) > EPSILON_MATH) {
            halfVector = normalize(halfVectorUnnormalized);
//          halfVector = normalize(halfVectorUnnormalized);
        } else {
//      } else {
            halfVector = surfaceNormal;
//          halfVector = surfaceNormal;
        }
//      }

        float normalDotLight = max(dot(surfaceNormal, lightDirection), 0.0);
//      float normalDotLight = max(dot(surfaceNormal, lightDirection), 0.0);
        float normalDotView = max(dot(surfaceNormal, viewDirection), 0.0);
//      float normalDotView = max(dot(surfaceNormal, viewDirection), 0.0);

        if (normalDotLight <= 0.0 || normalDotView <= 0.0) return vec3(0.0);
//      if (normalDotLight <= 0.0 || normalDotView <= 0.0) return vec3(0.0);

        vec3 reflectanceAtNormalIncidence = mix(vec3(F0_DEFAULT), albedo, metallic);
//      vec3 reflectanceAtNormalIncidence = mix(vec3(F0_DEFAULT), albedo, metallic);
        // BSDF energy split uses Fresnel at the half-vector angle
//      // BSDF energy split uses Fresnel at the half-vector angle
        float halfDotView = max(dot(halfVector, viewDirection), 0.0);
//      float halfDotView = max(dot(halfVector, viewDirection), 0.0);
        vec3 fresnelReflectance = schlickFresnel(halfDotView, reflectanceAtNormalIncidence);
//      vec3 fresnelReflectance = schlickFresnel(halfDotView, reflectanceAtNormalIncidence);

        // Diffuse
//      // Diffuse
        vec3 specularWeight = fresnelReflectance;
//      vec3 specularWeight = fresnelReflectance;
        vec3 diffuseWeight = (vec3(1.0) - specularWeight) * (1.0 - metallic);
//      vec3 diffuseWeight = (vec3(1.0) - specularWeight) * (1.0 - metallic);

        // --- Try uncommenting one of these ---
//      // --- Try uncommenting one of these ---

        // 1. Original Lambert
//      // 1. Original Lambert
        // vec3 diffuseContribution = diffuseWeight * evalLambertDiffuse(albedo) * (1.0 - transmission);
//      // vec3 diffuseContribution = diffuseWeight * evalLambertDiffuse(albedo) * (1.0 - transmission);

        // 2. Disney Diffuse (Recommended)
//      // 2. Disney Diffuse (Recommended)
        vec3 diffuseContribution = diffuseWeight * evalDisneyDiffuse(surfaceNormal, viewDirection, lightDirection, albedo, roughness) * (1.0 - transmission);
//      vec3 diffuseContribution = diffuseWeight * evalDisneyDiffuse(surfaceNormal, viewDirection, lightDirection, albedo, roughness) * (1.0 - transmission);

        // 3. Oren-Nayar Diffuse
//      // 3. Oren-Nayar Diffuse
        // vec3 diffuseContribution = diffuseWeight * evalOrenNayarDiffuse(surfaceNormal, viewDirection, lightDirection, albedo, roughness) * (1.0 - transmission);
//      // vec3 diffuseContribution = diffuseWeight * evalOrenNayarDiffuse(surfaceNormal, viewDirection, lightDirection, albedo, roughness) * (1.0 - transmission);

        // Specular (GGX distribution D is shared between the BSDF value and the specular PDF)
//      // Specular (GGX distribution D is shared between the BSDF value and the specular PDF)
        float ggxAlpha = roughness * roughness;
//      float ggxAlpha = roughness * roughness;
        float ggxAlphaSquared = ggxAlpha * ggxAlpha;
//      float ggxAlphaSquared = ggxAlpha * ggxAlpha;
        float normalDotHalf = max(dot(surfaceNormal, halfVector), 0.0);
//      float normalDotHalf = max(dot(surfaceNormal, halfVector), 0.0);
        float distributionDenominator = (normalDotHalf * normalDotHalf * (ggxAlphaSquared - 1.0) + 1.0);
//      float distributionDenominator = (normalDotHalf * normalDotHalf * (ggxAlphaSquared - 1.0) + 1.0);
        float normalDistribution = ggxAlphaSquared / (PI * distributionDenominator * distributionDenominator);
//      float normalDistribution = ggxAlphaSquared / (PI * distributionDenominator * distributionDenominator);

        float smithGeometryK = (roughness * roughness) / 2.0;
//      float smithGeometryK = (roughness * roughness) / 2.0;
        // Optimize G term calculation by factoring out normalDotView and normalDotLight
//      // Optimize G term calculation by factoring out normalDotView and normalDotLight
        float geometryViewTerm = normalDotView * (1.0 - smithGeometryK) + smithGeometryK;
//      float geometryViewTerm = normalDotView * (1.0 - smithGeometryK) + smithGeometryK;
        float geometryLightTerm = normalDotLight * (1.0 - smithGeometryK) + smithGeometryK;
//      float geometryLightTerm = normalDotLight * (1.0 - smithGeometryK) + smithGeometryK;

        vec3 specularContribution = (normalDistribution * fresnelReflectance) / (4.0 * geometryViewTerm * geometryLightTerm + EPSILON_MATH);
//      vec3 specularContribution = (normalDistribution * fresnelReflectance) / (4.0 * geometryViewTerm * geometryLightTerm + EPSILON_MATH);

        // PDF: the lobe-selection probability uses Fresnel at the view angle (matching the sampler)
//      // PDF: the lobe-selection probability uses Fresnel at the view angle (matching the sampler)
        float diffusePdf = normalDotLight / PI;
//      float diffusePdf = normalDotLight / PI;
        float specularPdf = (normalDistribution * normalDotHalf) / (4.0 * halfDotView + EPSILON_DOT);
//      float specularPdf = (normalDistribution * normalDotHalf) / (4.0 * halfDotView + EPSILON_DOT);
        vec3 lobeSelectionFresnel = schlickFresnel(normalDotView, reflectanceAtNormalIncidence);
//      vec3 lobeSelectionFresnel = schlickFresnel(normalDotView, reflectanceAtNormalIncidence);
        float averageFresnel = (lobeSelectionFresnel.r + lobeSelectionFresnel.g + lobeSelectionFresnel.b) / 3.0;
//      float averageFresnel = (lobeSelectionFresnel.r + lobeSelectionFresnel.g + lobeSelectionFresnel.b) / 3.0;
        float specularSelectionProbability = max(mix(averageFresnel, 1.0, metallic), 0.15);
//      float specularSelectionProbability = max(mix(averageFresnel, 1.0, metallic), 0.15);
        outPdf = mix(diffusePdf * (1.0 - transmission), specularPdf, specularSelectionProbability);
//      outPdf = mix(diffusePdf * (1.0 - transmission), specularPdf, specularSelectionProbability);

        return diffuseContribution + specularContribution;
//      return diffuseContribution + specularContribution;
    }
//  }

    // Stochastic Importance Sampling: chooses a secondary ray direction by first stochastically picking a lobe — Fresnel-weighted specular, diffuse, or transmission — then sampling that lobe according to its matching BSDF PDF: GGX half-vectors for specular reflection and refraction, a cosine-weighted hemisphere for diffuse. Concentrating samples where the integrand is large minimizes Monte Carlo variance. It also resolves Walter-style microfacet refraction with a total-internal-reflection fallback, returns the path throughput f·cos θ / pdf folded into a single attenuation, and flags delta (perfectly specular/transmissive) bounces so the integrator skips Next Event Estimation on them.
//  // Stochastic Importance Sampling: chooses a secondary ray direction by first stochastically picking a lobe — Fresnel-weighted specular, diffuse, or transmission — then sampling that lobe according to its matching BSDF PDF: GGX half-vectors for specular reflection and refraction, a cosine-weighted hemisphere for diffuse. Concentrating samples where the integrand is large minimizes Monte Carlo variance. It also resolves Walter-style microfacet refraction with a total-internal-reflection fallback, returns the path throughput f·cos θ / pdf folded into a single attenuation, and flags delta (perfectly specular/transmissive) bounces so the integrator skips Next Event Estimation on them.
    bool scatterPrincipled(Ray incomingRay, RayHitResult hitResult, Material material, float pathRoughnessFloor, out vec3 outAlbedo, out float outRoughness, out float outMetallic, out vec3 outShadingNormal, out vec3 outEmission, out vec3 outScatteredDirection, out vec3 outAttenuation, out float outPdf, out bool outIsDelta) {
//  bool scatterPrincipled(Ray incomingRay, RayHitResult hitResult, Material material, float pathRoughnessFloor, out vec3 outAlbedo, out float outRoughness, out float outMetallic, out vec3 outShadingNormal, out vec3 outEmission, out vec3 outScatteredDirection, out vec3 outAttenuation, out float outPdf, out bool outIsDelta) {
        outScatteredDirection = vec3(0.0);
//      outScatteredDirection = vec3(0.0);
        outAttenuation = vec3(0.0);
//      outAttenuation = vec3(0.0);
        outEmission = vec3(0.0);
//      outEmission = vec3(0.0);
        outPdf = 0.0;
//      outPdf = 0.0;
        outIsDelta = false;
//      outIsDelta = false;

        // Check texture indices
//      // Check texture indices
        vec3 albedo = material.albedo.rgb;
//      vec3 albedo = material.albedo.rgb;
        float roughness = material.roughness;
//      float roughness = material.roughness;
        float metallic = material.metallic;
//      float metallic = material.metallic;

        // Texture Sampling (using textureLod for Compute Shader safety)
//      // Texture Sampling (using textureLod for Compute Shader safety)
        float textureLodLevel = max(0.0, log2(hitResult.hitDistance * 0.1));
//      float textureLodLevel = max(0.0, log2(hitResult.hitDistance * 0.1));
        vec2 scaledTexcoord = hitResult.hitSurfaceTexcoord * material.uvScale;
//      vec2 scaledTexcoord = hitResult.hitSurfaceTexcoord * material.uvScale;

        if (material.textureIndexAlbedo > -0.5) {
//      if (material.textureIndexAlbedo > -0.5) {
            // Albedo is stored sRGB-encoded in 8-bit; decode to linear here
//          // Albedo is stored sRGB-encoded in 8-bit; decode to linear here
            albedo *= pow(textureLod(uSceneTextureArray, vec3(scaledTexcoord, material.textureIndexAlbedo), textureLodLevel).rgb, vec3(2.2));
//          albedo *= pow(textureLod(uSceneTextureArray, vec3(scaledTexcoord, material.textureIndexAlbedo), textureLodLevel).rgb, vec3(2.2));
        }
//      }
        if (material.textureIndexRoughness > -0.5) {
//      if (material.textureIndexRoughness > -0.5) {
            // Packed ORM layer: roughness is in the R channel
//          // Packed ORM layer: roughness is in the R channel
            roughness = textureLod(uSceneTextureArray, vec3(scaledTexcoord, material.textureIndexRoughness), textureLodLevel).r;
//          roughness = textureLod(uSceneTextureArray, vec3(scaledTexcoord, material.textureIndexRoughness), textureLodLevel).r;
        }
//      }
        if (material.textureIndexMetallic > -0.5) {
//      if (material.textureIndexMetallic > -0.5) {
            // Packed ORM layer: metallic is in the G channel
//          // Packed ORM layer: metallic is in the G channel
            metallic = textureLod(uSceneTextureArray, vec3(scaledTexcoord, material.textureIndexMetallic), textureLodLevel).g;
//          metallic = textureLod(uSceneTextureArray, vec3(scaledTexcoord, material.textureIndexMetallic), textureLodLevel).g;
        }
//      }

        if (material.textureIndexTransmission > -0.5) {
//      if (material.textureIndexTransmission > -0.5) {
            // Packed ORM layer: transmission is in the B channel
//          // Packed ORM layer: transmission is in the B channel
            material.transmission *= textureLod(uSceneTextureArray, vec3(scaledTexcoord, material.textureIndexTransmission), textureLodLevel).b;
//          material.transmission *= textureLod(uSceneTextureArray, vec3(scaledTexcoord, material.textureIndexTransmission), textureLodLevel).b;
        }
//      }

        // Roughness annealing (SDS variance mitigation) is applied HERE, after the texture fetch: the old
//      // Roughness annealing (SDS variance mitigation) is applied HERE, after the texture fetch: the old
        // pre-scatter clamp in main() was silently overwritten whenever a roughness texture was bound,
//      // pre-scatter clamp in main() was silently overwritten whenever a roughness texture was bound,
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
            outEmission = material.emissive * pow(textureLod(uSceneTextureArray, vec3(scaledTexcoord, material.textureIndexEmissive), textureLodLevel).rgb, vec3(2.2));
//          outEmission = material.emissive * pow(textureLod(uSceneTextureArray, vec3(scaledTexcoord, material.textureIndexEmissive), textureLodLevel).rgb, vec3(2.2));
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
        vec3 shadingNormal = hitResult.hitSurfaceNormal;
//      vec3 shadingNormal = hitResult.hitSurfaceNormal;

        if (material.textureIndexNormal > -0.5) {
//      if (material.textureIndexNormal > -0.5) {
            vec3 tangentSpaceNormal = textureLod(uSceneTextureArray, vec3(scaledTexcoord, material.textureIndexNormal), textureLodLevel).rgb;
//          vec3 tangentSpaceNormal = textureLod(uSceneTextureArray, vec3(scaledTexcoord, material.textureIndexNormal), textureLodLevel).rgb;
            tangentSpaceNormal = tangentSpaceNormal * 2.0 - 1.0;
//          tangentSpaceNormal = tangentSpaceNormal * 2.0 - 1.0;

            vec3 normalizedShadingNormal = normalize(shadingNormal);
//          vec3 normalizedShadingNormal = normalize(shadingNormal);
            vec3 tangentVector = normalize(hitResult.hitSurfaceTangent);
//          vec3 tangentVector = normalize(hitResult.hitSurfaceTangent);
            // Re-orthogonalize the tangent with respect to the shading normal
//          // Re-orthogonalize the tangent with respect to the shading normal
            vec3 orthogonalizedTangent = tangentVector - dot(tangentVector, normalizedShadingNormal) * normalizedShadingNormal;
//          vec3 orthogonalizedTangent = tangentVector - dot(tangentVector, normalizedShadingNormal) * normalizedShadingNormal;
            if (dot(orthogonalizedTangent, orthogonalizedTangent) > EPSILON_MATH) {
//          if (dot(orthogonalizedTangent, orthogonalizedTangent) > EPSILON_MATH) {
                tangentVector = normalize(orthogonalizedTangent);
//              tangentVector = normalize(orthogonalizedTangent);
            } else {
//          } else {
                // Fallback to avoid NaN if the tangent and normal are collinear
//              // Fallback to avoid NaN if the tangent and normal are collinear
                vec3 basisHelper;
//              vec3 basisHelper;
                if (abs(normalizedShadingNormal.z) < 0.999) {
//              if (abs(normalizedShadingNormal.z) < 0.999) {
                    basisHelper = vec3(0.0, 0.0, 1.0);
//                  basisHelper = vec3(0.0, 0.0, 1.0);
                } else {
//              } else {
                    basisHelper = vec3(1.0, 0.0, 0.0);
//                  basisHelper = vec3(1.0, 0.0, 0.0);
                }
//              }
                tangentVector = normalize(cross(basisHelper, normalizedShadingNormal));
//              tangentVector = normalize(cross(basisHelper, normalizedShadingNormal));
            }
//          }
            vec3 bitangentVector = cross(normalizedShadingNormal, tangentVector);
//          vec3 bitangentVector = cross(normalizedShadingNormal, tangentVector);
            mat3 tangentToWorldMatrix = mat3(tangentVector, bitangentVector, normalizedShadingNormal);
//          mat3 tangentToWorldMatrix = mat3(tangentVector, bitangentVector, normalizedShadingNormal);

            shadingNormal = normalize(tangentToWorldMatrix * tangentSpaceNormal);
//          shadingNormal = normalize(tangentToWorldMatrix * tangentSpaceNormal);
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
        vec3 reflectanceAtNormalIncidence = mix(vec3(F0_DEFAULT), albedo, metallic);
//      vec3 reflectanceAtNormalIncidence = mix(vec3(F0_DEFAULT), albedo, metallic);

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
        vec3 fresnelReflectance = schlickFresnel(cosIncidentAngle, reflectanceAtNormalIncidence);
//      vec3 fresnelReflectance = schlickFresnel(cosIncidentAngle, reflectanceAtNormalIncidence);
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
        float specularSelectionProbability = max(mix(averageFresnel, 1.0, metallic), 0.15);
//      float specularSelectionProbability = max(mix(averageFresnel, 1.0, metallic), 0.15);

        if (randomLobeSelector < specularSelectionProbability) {
//      if (randomLobeSelector < specularSelectionProbability) {
            // SPECULAR REFLECTION (METAL OR DIELECTRIC COAT)
//          // SPECULAR REFLECTION (METAL OR DIELECTRIC COAT)
            vec3 microfacetNormal = sampleGGX(shadingNormal, roughness);
//          vec3 microfacetNormal = sampleGGX(shadingNormal, roughness);
            vec3 specularReflectedDirection = reflectPrincipled(incomingRay.direction, microfacetNormal);
//          vec3 specularReflectedDirection = reflectPrincipled(incomingRay.direction, microfacetNormal);

            if (dot(specularReflectedDirection, shadingNormal) > 0.0 && dot(specularReflectedDirection, hitResult.hitSurfaceNormal) > 0.0) {
//          if (dot(specularReflectedDirection, shadingNormal) > 0.0 && dot(specularReflectedDirection, hitResult.hitSurfaceNormal) > 0.0) {
                outScatteredDirection = specularReflectedDirection;
//              outScatteredDirection = specularReflectedDirection;
                float pdfValue;
//              float pdfValue;
                vec3 bsdfValue = evalPrincipledBSDFAndPDF(incomingRay.direction, specularReflectedDirection, shadingNormal, albedo, roughness, metallic, material.transmission, pdfValue);
//              vec3 bsdfValue = evalPrincipledBSDFAndPDF(incomingRay.direction, specularReflectedDirection, shadingNormal, albedo, roughness, metallic, material.transmission, pdfValue);
                float cosOutgoingAngle = max(dot(shadingNormal, specularReflectedDirection), 0.0);
//              float cosOutgoingAngle = max(dot(shadingNormal, specularReflectedDirection), 0.0);
                outAttenuation = bsdfValue * cosOutgoingAngle / max(pdfValue, EPSILON_MATH);
//              outAttenuation = bsdfValue * cosOutgoingAngle / max(pdfValue, EPSILON_MATH);
                outIsDelta = false;
//              outIsDelta = false;
                outPdf = pdfValue;
//              outPdf = pdfValue;
            } else {
//          } else {
                // Current/Recent ray is/was absorbed (next ray is scattering into surface)
//              // Current/Recent ray is/was absorbed (next ray is scattering into surface)
                return false;
//              return false;
            }
//          }
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

                vec3 microfacetNormal = sampleGGX(shadingNormal, roughness);
//              vec3 microfacetNormal = sampleGGX(shadingNormal, roughness);

                float cosIncidentAngleMicro = dot(-incomingRay.direction, microfacetNormal);
//              float cosIncidentAngleMicro = dot(-incomingRay.direction, microfacetNormal);
                if (cosIncidentAngleMicro <= 0.0) return false;
//              if (cosIncidentAngleMicro <= 0.0) return false;
                float sinTransmittedAngleSquared = (1.0 - cosIncidentAngleMicro * cosIncidentAngleMicro) * (etaRatioOfIncidenceOverTransmission * etaRatioOfIncidenceOverTransmission);
//              float sinTransmittedAngleSquared = (1.0 - cosIncidentAngleMicro * cosIncidentAngleMicro) * (etaRatioOfIncidenceOverTransmission * etaRatioOfIncidenceOverTransmission);

                vec3 refractedDirection = refractPrincipled(incomingRay.direction, microfacetNormal, etaRatioOfIncidenceOverTransmission, cosIncidentAngleMicro, sinTransmittedAngleSquared);
//              vec3 refractedDirection = refractPrincipled(incomingRay.direction, microfacetNormal, etaRatioOfIncidenceOverTransmission, cosIncidentAngleMicro, sinTransmittedAngleSquared);
                vec3 reflectedDirection = reflectPrincipled(incomingRay.direction, microfacetNormal);
//              vec3 reflectedDirection = reflectPrincipled(incomingRay.direction, microfacetNormal);

                // When [ sinTransmittedAngleSquared <= 1.0 ] then Refraction happened else Total Internal Reflection happened
//              // When [ sinTransmittedAngleSquared <= 1.0 ] then Refraction happened else Total Internal Reflection happened

                // Fresnel evaluated at the sampled microfacet angle (not the macro shading normal)
//              // Fresnel evaluated at the sampled microfacet angle (not the macro shading normal)
                vec3 microfacetFresnel = schlickFresnel(cosIncidentAngleMicro, reflectanceAtNormalIncidence);
//              vec3 microfacetFresnel = schlickFresnel(cosIncidentAngleMicro, reflectanceAtNormalIncidence);

                // NDF-sampled microfacet weight: G * |V.Hm| / (|N.V| * |N.Hm|), with the same Smith
                // convention as the reflection BSDF (k = roughness^2 / 2). Degrades to ~1.0 at MIN_ROUGHNESS.
//              // NDF-sampled microfacet weight: G * |V.Hm| / (|N.V| * |N.Hm|), with the same Smith
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
                    outAttenuation = (vec3(1.0) - microfacetFresnel) * albedo * microfacetWeight / max(branchProbability, EPSILON_MATH);
//                  outAttenuation = (vec3(1.0) - microfacetFresnel) * albedo * microfacetWeight / max(branchProbability, EPSILON_MATH);
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
                    outAttenuation = vec3(microfacetWeight) / max(branchProbability, EPSILON_MATH);
//                  outAttenuation = vec3(microfacetWeight) / max(branchProbability, EPSILON_MATH);
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
                vec3 diffuseDirectionUnnormalized = shadingNormal + randomUnitVector();
//              vec3 diffuseDirectionUnnormalized = shadingNormal + randomUnitVector();
                vec3 diffuseDirection;
//              vec3 diffuseDirection;
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
                    return false;
//                  return false;
                }
//              }

                outScatteredDirection = diffuseDirection;
//              outScatteredDirection = diffuseDirection;
                float pdfValue;
//              float pdfValue;
                vec3 bsdfValue = evalPrincipledBSDFAndPDF(incomingRay.direction, diffuseDirection, shadingNormal, albedo, roughness, metallic, material.transmission, pdfValue);
//              vec3 bsdfValue = evalPrincipledBSDFAndPDF(incomingRay.direction, diffuseDirection, shadingNormal, albedo, roughness, metallic, material.transmission, pdfValue);
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
            vec3 originToCenter = ray.origin - uPointLights[lightLoopIndex].position;
//          vec3 originToCenter = ray.origin - uPointLights[lightLoopIndex].position;
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

    void main() {
//  void main() {
        ivec2 pixelCoordinates = ivec2(gl_GlobalInvocationID.xy);
//      ivec2 pixelCoordinates = ivec2(gl_GlobalInvocationID.xy);
        ivec2 dimensions = imageSize(textureOutput);
//      ivec2 dimensions = imageSize(textureOutput);

        if (pixelCoordinates.x >= dimensions.x || pixelCoordinates.y >= dimensions.y) {
//      if (pixelCoordinates.x >= dimensions.x || pixelCoordinates.y >= dimensions.y) {
            return;
//          return;
        }
//      }

        initRNG(vec2(pixelCoordinates));
//      initRNG(vec2(pixelCoordinates));

        vec3 accumulatedColor = vec3(0.0);
//      vec3 accumulatedColor = vec3(0.0);
        vec3 attenuation = vec3(1.0);
//      vec3 attenuation = vec3(1.0);

        // Primary Ray Initialization: reconstructs this pixel's world-space camera ray from the precomputed sensor basis — the top-left sample position plus per-pixel horizontal and vertical step vectors scaled by the integer pixel coordinate — then normalizes the vector from the camera origin through that point. A per-frame sub-pixel jitter nudges the sample within the pixel's footprint, so temporal accumulation effectively integrates over the entire pixel area, yielding anti-aliasing and smoother convergence without an explicit per-pixel multi-sample loop.
//      // Primary Ray Initialization: reconstructs this pixel's world-space camera ray from the precomputed sensor basis — the top-left sample position plus per-pixel horizontal and vertical step vectors scaled by the integer pixel coordinate — then normalizes the vector from the camera origin through that point. A per-frame sub-pixel jitter nudges the sample within the pixel's footprint, so temporal accumulation effectively integrates over the entire pixel area, yielding anti-aliasing and smoother convergence without an explicit per-pixel multi-sample loop.
        vec3 pixelSampleCenter = uPixel00Coordinates + uPixelDeltaU * (float(pixelCoordinates.x) + uJitter.x) + uPixelDeltaV * (float(pixelCoordinates.y) + uJitter.y);
//      vec3 pixelSampleCenter = uPixel00Coordinates + uPixelDeltaU * (float(pixelCoordinates.x) + uJitter.x) + uPixelDeltaV * (float(pixelCoordinates.y) + uJitter.y);

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
        vec3 hitPoint = vec3(0.0);
//      vec3 hitPoint = vec3(0.0);
        bool lastSkippedNEE = false;
//      bool lastSkippedNEE = false;

        for (int depth = 0; depth < maxDepth; depth++) {
//      for (int depth = 0; depth < maxDepth; depth++) {
            RayHitResult rayHitResult;
//          RayHitResult rayHitResult;

            if (depth == 0) {
//          if (depth == 0) {
                // Hybrid Start: Fetch base surface properties from G-Buffer to avoid first-hit traversal overhead
//              // Hybrid Start: Fetch base surface properties from G-Buffer to avoid first-hit traversal overhead
                vec4 sampleGlobalPosition = imageLoad(textureGeometryGlobalPosition, pixelCoordinates);
//              vec4 sampleGlobalPosition = imageLoad(textureGeometryGlobalPosition, pixelCoordinates);
                vec4 sampleGlobalNormal = imageLoad(textureGeometryGlobalNormal, pixelCoordinates);
//              vec4 sampleGlobalNormal = imageLoad(textureGeometryGlobalNormal, pixelCoordinates);
                vec4 sampleGlobalTangent = imageLoad(textureGeometryGlobalTangent, pixelCoordinates);
//              vec4 sampleGlobalTangent = imageLoad(textureGeometryGlobalTangent, pixelCoordinates);

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
                    rayHitResult.hitSurfaceTexcoord = vec2(sampleGlobalNormal.w, sampleGlobalTangent.w);
//                  rayHitResult.hitSurfaceTexcoord = vec2(sampleGlobalNormal.w, sampleGlobalTangent.w);
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

            if (hitLight) {
//          if (hitLight) {
                vec3 emittedRadiance = uPointLights[lightHit.lightIndex].color;
//              vec3 emittedRadiance = uPointLights[lightHit.lightIndex].color;

                if (depth == 0 || lastWasDelta) {
//              if (depth == 0 || lastWasDelta) {
                    accumulatedColor += attenuation * emittedRadiance;
//                  accumulatedColor += attenuation * emittedRadiance;
                } else {
//              } else {
                    vec3 lightHitNormal = normalize((currentRay.origin + currentRay.direction * lightHit.distanceToHit) - uPointLights[lightHit.lightIndex].position);
//                  vec3 lightHitNormal = normalize((currentRay.origin + currentRay.direction * lightHit.distanceToHit) - uPointLights[lightHit.lightIndex].position);
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
                vec3 skyRadiance = getSkyColor(currentRay.direction);
//              vec3 skyRadiance = getSkyColor(currentRay.direction);
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
                material.albedo = imageLoad(textureGeometryAlbedo, pixelCoordinates);
//              material.albedo = imageLoad(textureGeometryAlbedo, pixelCoordinates);
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
                vec3 cachedRadiance;
//              vec3 cachedRadiance;
                if (readCache(hitPoint, rayHitResult.hitSurfaceNormal, cachedRadiance)) {
//              if (readCache(hitPoint, rayHitResult.hitSurfaceNormal, cachedRadiance)) {
                    float depthFactor = clamp(float(depth - 1) / 3.0, 0.0, 1.0);
//                  float depthFactor = clamp(float(depth - 1) / 3.0, 0.0, 1.0);
                    float blendWeight = uCacheBlendFactor * mix(0.3, 0.9, depthFactor);
//                  float blendWeight = uCacheBlendFactor * mix(0.3, 0.9, depthFactor);
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

            vec3 albedo;
//          vec3 albedo;
            float roughness;
//          float roughness;
            float metallic;
//          float metallic;
            vec3 shadingNormal;
//          vec3 shadingNormal;
            vec3 emission;
//          vec3 emission;
            vec3 scatteredDirection;
//          vec3 scatteredDirection;
            vec3 scatterAttenuation;
//          vec3 scatterAttenuation;
            float scatterPdf;
//          float scatterPdf;
            bool scatterIsDelta;
//          bool scatterIsDelta;

            bool isScattered = scatterPrincipled(currentRay, rayHitResult, material, (depth > 0) ? pathRoughness : 0.0, albedo, roughness, metallic, shadingNormal, emission, scatteredDirection, scatterAttenuation, scatterPdf, scatterIsDelta);
//          bool isScattered = scatterPrincipled(currentRay, rayHitResult, material, (depth > 0) ? pathRoughness : 0.0, albedo, roughness, metallic, shadingNormal, emission, scatteredDirection, scatterAttenuation, scatterPdf, scatterIsDelta);

            accumulatedColor += attenuation * emission;
//          accumulatedColor += attenuation * emission;
            vec3 bounceDirectRadiance = emission;
//          vec3 bounceDirectRadiance = emission;

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

            // Direct Illumination (Next Event Estimation) with MIS
//          // Direct Illumination (Next Event Estimation) with MIS
            bool skipNEE = roughness < 0.05 && (metallic > 0.99 || material.transmission > 0.99);
//          bool skipNEE = roughness < 0.05 && (metallic > 0.99 || material.transmission > 0.99);
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

                vec3 lightPosition = uPointLights[selectedLightIndex].position;
//              vec3 lightPosition = uPointLights[selectedLightIndex].position;
                float lightRadius = uPointLights[selectedLightIndex].radius;
//              float lightRadius = uPointLights[selectedLightIndex].radius;
                float lightPdf = uPointLights[selectedLightIndex].pdf;
//              float lightPdf = uPointLights[selectedLightIndex].pdf;
                // Sample only the hemisphere of the light sphere facing the shading point to halve wasted (culled) samples
//              // Sample only the hemisphere of the light sphere facing the shading point to halve wasted (culled) samples
                vec3 lightToSurfaceDirection = normalize(hitPoint - lightPosition);
//              vec3 lightToSurfaceDirection = normalize(hitPoint - lightPosition);
                vec3 lightSampleNormal = randomUnitVector();
//              vec3 lightSampleNormal = randomUnitVector();
                if (dot(lightSampleNormal, lightToSurfaceDirection) < 0.0) lightSampleNormal = -lightSampleNormal;
//              if (dot(lightSampleNormal, lightToSurfaceDirection) < 0.0) lightSampleNormal = -lightSampleNormal;
                vec3 sampledLightPoint = lightPosition + lightSampleNormal * lightRadius;
//              vec3 sampledLightPoint = lightPosition + lightSampleNormal * lightRadius;

                vec3 shadowRayDirection = sampledLightPoint - hitPoint;
//              vec3 shadowRayDirection = sampledLightPoint - hitPoint;
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
                vec3 sampledLightNormal = normalize(sampledLightPoint - lightPosition);
//              vec3 sampledLightNormal = normalize(sampledLightPoint - lightPosition);
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
                        vec3 bsdfValue = evalPrincipledBSDFAndPDF(currentRay.direction, shadowRayDirection, shadingNormal, albedo, roughness, metallic, material.transmission, brdfPdfSolidAngle);
//                      vec3 bsdfValue = evalPrincipledBSDFAndPDF(currentRay.direction, shadowRayDirection, shadingNormal, albedo, roughness, metallic, material.transmission, brdfPdfSolidAngle);

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
                        vec3 directLight = uPointLights[selectedLightIndex].color * bsdfValue * cosSurfaceAngle / max(lightPdfSolidAngle, NEE_PDF_EPSILON);
//                      vec3 directLight = uPointLights[selectedLightIndex].color * bsdfValue * cosSurfaceAngle / max(lightPdfSolidAngle, NEE_PDF_EPSILON);
                        // Clamp direct light to prevent fireflies from extreme NEE contributions
//                      // Clamp direct light to prevent fireflies from extreme NEE contributions
                        directLight = min(directLight, vec3(NEE_DIRECT_LIGHT_CLAMP));
//                      directLight = min(directLight, vec3(NEE_DIRECT_LIGHT_CLAMP));
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

            if (depth >= 1 && uCacheBlendFactor > 0.0 && !scatterIsDelta) {
//          if (depth >= 1 && uCacheBlendFactor > 0.0 && !scatterIsDelta) {
                writeCache(hitPoint, rayHitResult.hitSurfaceNormal, bounceDirectRadiance);
//              writeCache(hitPoint, rayHitResult.hitSurfaceNormal, bounceDirectRadiance);
            }
//          }

            /*
            if (!isScattered) {
//          if (!isScattered) {
                break;
//              break;
            }
//          }
            */

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
            lastSkippedNEE = skipNEE;
//          lastSkippedNEE = skipNEE;
        }
//      }

        accumulatedColor = min(accumulatedColor, vec3(ACCUM_CLAMP));
//      accumulatedColor = min(accumulatedColor, vec3(ACCUM_CLAMP));

        // Prevent NaN propagation in the accumulated frame buffer
//      // Prevent NaN propagation in the accumulated frame buffer
        if (isnan(accumulatedColor.r) || isnan(accumulatedColor.g) || isnan(accumulatedColor.b)) {
//      if (isnan(accumulatedColor.r) || isnan(accumulatedColor.g) || isnan(accumulatedColor.b)) {
            accumulatedColor = vec3(0.0);
//          accumulatedColor = vec3(0.0);
        }
//      }

        // Temporal Reprojection Accumulation: folds this frame's noisy radiance estimate into a persistent history buffer as a cumulative moving average, blending with weight 1/uFrameCount so each new sample contributes its equal share of the running mean. On the first frame (or after a camera/scene reset) the history is discarded and the buffer reseeded with full weight; across many stationary frames the variance of the mean falls as ~1/N, progressively resolving the stochastic noise into a clean, converged image. Tone mapping and final output are deferred to the separate denoise pass.
//      // Temporal Reprojection Accumulation: folds this frame's noisy radiance estimate into a persistent history buffer as a cumulative moving average, blending with weight 1/uFrameCount so each new sample contributes its equal share of the running mean. On the first frame (or after a camera/scene reset) the history is discarded and the buffer reseeded with full weight; across many stationary frames the variance of the mean falls as ~1/N, progressively resolving the stochastic noise into a clean, converged image. Tone mapping and final output are deferred to the separate denoise pass.
        vec4 previousAccumulation = imageLoad(textureAccum, pixelCoordinates);
//      vec4 previousAccumulation = imageLoad(textureAccum, pixelCoordinates);
        vec3 historyColor = previousAccumulation.rgb;
//      vec3 historyColor = previousAccumulation.rgb;
        float blendAlpha = 1.0 / float(uFrameCount);
//      float blendAlpha = 1.0 / float(uFrameCount);
        blendAlpha = max(blendAlpha, 0.001 /* 0.01 */);
//      blendAlpha = max(blendAlpha, 0.001 /* 0.01 */);
        if (uFrameCount == 1) {
//      if (uFrameCount == 1) {
            historyColor = vec3(0.0);
//          historyColor = vec3(0.0);
            blendAlpha = 1.0;
//          blendAlpha = 1.0;
        }
//      }
        vec3 blendedColor = mix(historyColor, accumulatedColor, blendAlpha);
//      vec3 blendedColor = mix(historyColor, accumulatedColor, blendAlpha);
        imageStore(textureAccum, pixelCoordinates, vec4(blendedColor, 1.0));
//      imageStore(textureAccum, pixelCoordinates, vec4(blendedColor, 1.0));

        // Note: Tone mapping and output moved to hybrid_denoise_cs.glsl
//      // Note: Tone mapping and output moved to hybrid_denoise_cs.glsl
    }
//  }
