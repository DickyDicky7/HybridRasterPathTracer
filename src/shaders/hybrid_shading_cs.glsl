    #version 430
//  #version 430

    layout(local_size_x = 16, local_size_y = 16) in;
//  layout(local_size_x = 16, local_size_y = 16) in;

    layout(binding = 0, rgba32f) uniform image2D textureOutput;
//  layout(binding = 0, rgba32f) uniform image2D textureOutput;
    layout(binding = 1, rgba32f) uniform image2D textureGeometryGlobalPosition;
//  layout(binding = 1, rgba32f) uniform image2D textureGeometryGlobalPosition;
    layout(binding = 2, rgba32f) uniform image2D textureGeometryGlobalNormal;
//  layout(binding = 2, rgba32f) uniform image2D textureGeometryGlobalNormal;
    layout(binding = 3, rgba32f) uniform image2D textureGeometryAlbedo;
//  layout(binding = 3, rgba32f) uniform image2D textureGeometryAlbedo;
    layout(binding = 4, rgba32f) uniform image2D textureGeometryGlobalTangent;
//  layout(binding = 4, rgba32f) uniform image2D textureGeometryGlobalTangent;
    layout(binding = 5, rgba32f) uniform image2D textureAccum;
//  layout(binding = 5, rgba32f) uniform image2D textureAccum;

    // Bounding Volume Hierarchy (BVH) acceleration structure: Nodes store Axis-Aligned Bounding Box (AABB) extents to rapidly cull empty space. Inner nodes store pointers to child nodes, while leaf nodes store offsets to primitive triangle data, enabling logarithmic time-complexity for ray intersection tests.
    // Bounding Volume Hierarchy (BVH) acceleration structure: Nodes store Axis-Aligned Bounding Box (AABB) extents to rapidly cull empty space. Inner nodes store pointers to child nodes, while leaf nodes store offsets to primitive triangle data, enabling logarithmic time-complexity for ray intersection tests.
    struct Node {
//  struct Node {
        vec4 min_left; // min.xyz, left_child_index (or -1.0 if leaf)
//      vec4 min_left; // min.xyz, left_child_index (or -1.0 if leaf)
        vec4 max_right; // max.xyz, right_child_index (or tri_index if leaf)
//      vec4 max_right; // max.xyz, right_child_index (or tri_index if leaf)
    };
//  };

    layout(std430, binding = 6) buffer BVHNodes {
//  layout(std430, binding = 6) buffer BVHNodes {
        Node nodes[];
//      Node nodes[];
    };
//  };

    // Global Geometry Buffer: A contiguous, tightly-packed flat array of world-space vertex positions (3 floats per vertex, 9 floats per triangle). This linearly mapped buffer minimizes cache misses during the computationally heavy closest-hit and any-hit ray-triangle intersection routines.
    // Global Geometry Buffer: A contiguous, tightly-packed flat array of world-space vertex positions (3 floats per vertex, 9 floats per triangle). This linearly mapped buffer minimizes cache misses during the computationally heavy closest-hit and any-hit ray-triangle intersection routines.
    layout(std430, binding = 7) buffer SceneTriangles {
//  layout(std430, binding = 7) buffer SceneTriangles {
        float triangles[];
//      float triangles[];
    };
//  };

    // Material Properties Buffer: Stores physically-based rendering (PBR) parameters including Base Color/Albedo, perceptual Roughness, Metallic masking, and Transmission coefficients. It also holds indexed pointers to bindless texture arrays for spatially varying material evaluations during shading.
    // Material Properties Buffer: Stores physically-based rendering (PBR) parameters including Base Color/Albedo, perceptual Roughness, Metallic masking, and Transmission coefficients. It also holds indexed pointers to bindless texture arrays for spatially varying material evaluations during shading.
    struct Material {
//  struct Material {
        vec4 albedo;       // r, g, b, padding
//      vec4 albedo;       // r, g, b, padding
        float roughness;
//      float roughness;
        float metallic;
//      float metallic;
        float transmission;
//      float transmission;
        float ior;
//      float ior;
        float textureIndexAlbedo;
//      float textureIndexAlbedo;
        float textureIndexRoughness;
//      float textureIndexRoughness;
        float textureIndexMetallic;
//      float textureIndexMetallic;
        float textureIndexNormal;
//      float textureIndexNormal;
        float emissive;
//      float emissive;
        float textureIndexEmissive;
//      float textureIndexEmissive;
        float padding001;
//      float padding001;
        float padding002;
//      float padding002;
    };
//  };

    layout(std430, binding = 8) buffer SceneMaterials {
//  layout(std430, binding = 8) buffer SceneMaterials {
        Material materials[];
//      Material materials[];
    };
//  };

    // UVs Buffer
//  // UVs Buffer
    layout(std430, binding = 9) buffer SceneUVs {
//  layout(std430, binding = 9) buffer SceneUVs {
        vec2 uvs[]; // 3 per triangle
//      vec2 uvs[]; // 3 per triangle
    };
//  };

    // Normals Buffer
//  // Normals Buffer
    layout(std430, binding = 10) buffer SceneNormals {
//  layout(std430, binding = 10) buffer SceneNormals {
        float normals[]; // 3 floats per vertex -> 9 floats per triangle
//      float normals[]; // 3 floats per vertex -> 9 floats per triangle
    };
//  };

    // Tangents Buffer
//  // Tangents Buffer
    layout(std430, binding = 11) buffer SceneTangents {
//  layout(std430, binding = 11) buffer SceneTangents {
        float tangents[]; // 3 floats per vertex -> 9 floats per triangle
//      float tangents[]; // 3 floats per vertex -> 9 floats per triangle
    };
//  };

    struct Ray {
//  struct Ray {
        vec3 origin;
//      vec3 origin;
        vec3 direction;
//      vec3 direction;
    };
//  };

    struct RayHitResult {
//  struct RayHitResult {
        vec3 at;
//      vec3 at;
        vec3 hittedSideNormal;
//      vec3 hittedSideNormal;
        vec3 hittedSideTangent;
//      vec3 hittedSideTangent;
        vec2 uvSurfaceCoordinate;
//      vec2 uvSurfaceCoordinate;
        float minDistance;
//      float minDistance;
        bool isHitted;
//      bool isHitted;
        bool isFrontFaceHitted;
//      bool isFrontFaceHitted;
        int materialIndex;
//      int materialIndex;
        int triangleIndex;
//      int triangleIndex;
    };
//  };

    struct MaterialLightScatteringResult {
//  struct MaterialLightScatteringResult {
        Ray scatteredRay;
//      Ray scatteredRay;
        vec3 attenuation;
//      vec3 attenuation;
        vec3 emission;
//      vec3 emission;
        bool isScattered;
//      bool isScattered;
        vec3 albedo;
//      vec3 albedo;
        float roughness;
//      float roughness;
        float metallic;
//      float metallic;
        vec3 shadingNormal;
//      vec3 shadingNormal;
        float pdf;
//      float pdf;
        bool isDelta;
//      bool isDelta;
    };
//  };

    uniform float uTime;
//  uniform float uTime;
    struct PointLight {
//  struct PointLight {
        vec3 position;
//      vec3 position;
        float radius;
//      float radius;
        vec3 color;
//      vec3 color;
        float cdf;
//      float cdf;
        float pdf;
//      float pdf;
        float padding;
//      float padding;
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
    const float EPSILON_INTERSECT = 1.0e-4;
//  const float EPSILON_INTERSECT = 1.0e-4;
    const float EPSILON_OFFSET = 0.001;
//  const float EPSILON_OFFSET = 0.001;
    const float MISS_DISTANCE = 9999999.0;
//  const float MISS_DISTANCE = 9999999.0;

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

    // Xorshift/Wang Hash Pseudo-Random Number Generator (PRNG): Provides fast, statistically robust uniform random number sequences essential for Monte Carlo integration. This ensures unbiased stochastic sampling across hemisphere distributions, Russian Roulette termination, and Next Event Estimation.
    // Xorshift/Wang Hash Pseudo-Random Number Generator (PRNG): Provides fast, statistically robust uniform random number sequences essential for Monte Carlo integration. This ensures unbiased stochastic sampling across hemisphere distributions, Russian Roulette termination, and Next Event Estimation.
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
        return float(seed) / 4294967296.0;
//      return float(seed) / 4294967296.0;
    }
//  }
    void initRNG(vec2 pixel) {
//  void initRNG(vec2 pixel) {
        // Use float coordinates and wang_hash for better random distribution
//      // Use float coordinates and wang_hash for better random distribution
        seed = uint(pixel.x * 1973.0 + pixel.y * 9277.0 + float(uFrameCount) * 26699.0);
//      seed = uint(pixel.x * 1973.0 + pixel.y * 9277.0 + float(uFrameCount) * 26699.0);
        seed = wang_hash(seed);
//      seed = wang_hash(seed);
    }
//  }

    vec3 randomUnitVector() {
//  vec3 randomUnitVector() {
        float z = rand() * 2.0 - 1.0;
//      float z = rand() * 2.0 - 1.0;
        float a = rand() * 2.0 * PI;
//      float a = rand() * 2.0 * PI;
        float r = sqrt(1.0 - z * z);
//      float r = sqrt(1.0 - z * z);
        float x = r * cos(a);
//      float x = r * cos(a);
        float y = r * sin(a);
//      float y = r * sin(a);
        return vec3(x, y, z);
//      return vec3(x, y, z);
    }
//  }

    // Environment Lighting: Evaluates infinite background illumination by either sampling a High Dynamic Range (HDR) environment map via equirectangular (latitude-longitude) projection, or by falling back to a computationally derived procedural atmospheric scattering sky model.
    // Environment Lighting: Evaluates infinite background illumination by either sampling a High Dynamic Range (HDR) environment map via equirectangular (latitude-longitude) projection, or by falling back to a computationally derived procedural atmospheric scattering sky model.
    vec3 getSkyColor(vec3 dir) {
//  vec3 getSkyColor(vec3 dir) {
        if (uUseHdri) {
//      if (uUseHdri) {
            // Equirectangular mapping
            // Equirectangular mapping
            float theta = acos(-dir.y);           // latitude: 0 at top, PI at bottom
//          float theta = acos(-dir.y);           // latitude: 0 at top, PI at bottom
            float phi = atan(-dir.z, dir.x) + PI; // longitude: 0 to 2*PI
//          float phi = atan(-dir.z, dir.x) + PI; // longitude: 0 to 2*PI
            float u = clamp(phi / (2.0 * PI), 0.0, 1.0);
//          float u = clamp(phi / (2.0 * PI), 0.0, 1.0);
            float v = clamp(theta / PI, 0.0, 1.0);
//          float v = clamp(theta / PI, 0.0, 1.0);
            vec3 hdriColor = texture(uHdriTexture, vec2(u, v)).rgb;
//          vec3 hdriColor = texture(uHdriTexture, vec2(u, v)).rgb;
            return min(hdriColor, vec3(HDRI_CLAMP));
//          return min(hdriColor, vec3(HDRI_CLAMP));
        }
//      }
        /*
        // Fallback: procedural gradient sky
        // Fallback: procedural gradient sky
        float t = 0.5 * (dir.y + 1.0);
//      float t = 0.5 * (dir.y + 1.0);
        return mix(vec3(0.1), vec3(0.5, 0.7, 1.0), t);
//      return mix(vec3(0.1), vec3(0.5, 0.7, 1.0), t);
        */
        // Fallback: procedural atmospheric sky
//      // Fallback: procedural atmospheric sky
        // Rayleigh Gradient: Approximates scattering of blue wavelengths, deeper at zenith
//      // Rayleigh Gradient: Approximates scattering of blue wavelengths, deeper at zenith
        vec3 col = vec3(0.2, 0.45, 0.9) - dir.y * 0.25 * vec3(1.0, 0.5, 1.2) + 0.1 * vec3(1.0);
//      vec3 col = vec3(0.2, 0.45, 0.9) - dir.y * 0.25 * vec3(1.0, 0.5, 1.2) + 0.1 * vec3(1.0);
        // Mie Scattering: Exponential horizon haze due to higher atmospheric density
//      // Mie Scattering: Exponential horizon haze due to higher atmospheric density
        col = mix(col, vec3(0.9, 0.95, 1.0), exp(-15.0 * max(dir.y, 0.0)));
//      col = mix(col, vec3(0.9, 0.95, 1.0), exp(-15.0 * max(dir.y, 0.0)));
        // Cloud-like ground: Replace the bottom with a bright, vivid white-blue deck
//      // Cloud-like ground: Replace the bottom with a bright, vivid white-blue deck
        if (dir.y < 0.0) col = mix(vec3(0.9, 0.95, 1.0), vec3(0.98, 0.99, 1.0), pow(abs(dir.y), 0.5));
//      if (dir.y < 0.0) col = mix(vec3(0.9, 0.95, 1.0), vec3(0.98, 0.99, 1.0), pow(abs(dir.y), 0.5));
        // Sun Disk: Layered glows using increasing powers of the dot product (cosine of angle)
//      // Sun Disk: Layered glows using increasing powers of the dot product (cosine of angle)
        // Enhance sun intensity and direction for a more dramatic sky
//      // Enhance sun intensity and direction for a more dramatic sky
        vec3 sunDir = normalize(vec3(0.0, 0.5, 0.5));
//      vec3 sunDir = normalize(vec3(0.0, 0.5, 0.5));
        float sun = clamp(dot(dir, sunDir), 0.0, 1.0);
//      float sun = clamp(dot(dir, sunDir), 0.0, 1.0);
        col += 0.4 * vec3(10.0, 10.6, 10.3) * pow(sun, 8.0); // Wide soft orange glow
//      col += 0.4 * vec3(10.0, 10.6, 10.3) * pow(sun, 8.0); // Wide soft orange glow
        col += 0.3 * vec3(10.0, 10.8, 10.5) * pow(sun, 64.0); // Bright golden core
//      col += 0.3 * vec3(10.0, 10.8, 10.5) * pow(sun, 64.0); // Bright golden core
        col += 0.5 * vec3(10.0, 10.0, 10.0) * pow(sun, 512.0); // Intense white disk
//      col += 0.5 * vec3(10.0, 10.0, 10.0) * pow(sun, 512.0); // Intense white disk
        return col;
//      return col;
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
    float maxVec3(vec3 v) { return max(v.x, max(v.y, v.z)); }
//  float maxVec3(vec3 v) { return max(v.x, max(v.y, v.z)); }
    float minVec3(vec3 v) { return min(v.x, min(v.y, v.z)); }
//  float minVec3(vec3 v) { return min(v.x, min(v.y, v.z)); }

    // Ray-AABB Intersection (Slab Method): An optimized branchless algorithm testing ray intersections against the 3 pairs of parallel planes defining an AABB. This robust implementation quickly culls rays that completely miss the bounding volume limits during BVH traversal.
//  // Ray-AABB Intersection (Slab Method): An optimized branchless algorithm testing ray intersections against the 3 pairs of parallel planes defining an AABB. This robust implementation quickly culls rays that completely miss the bounding volume limits during BVH traversal.
    bool intersectAABB_Interval(Ray ray, vec3 invDir, Interval rayTravelDistanceLimit, vec3 boxMin, vec3 boxMax) {
//  bool intersectAABB_Interval(Ray ray, vec3 invDir, Interval rayTravelDistanceLimit, vec3 boxMin, vec3 boxMax) {
        vec3 t1 = (boxMin - ray.origin) * invDir;
//      vec3 t1 = (boxMin - ray.origin) * invDir;
        vec3 t2 = (boxMax - ray.origin) * invDir;
//      vec3 t2 = (boxMax - ray.origin) * invDir;
        vec3 tMin = min(t1, t2);
//      vec3 tMin = min(t1, t2);
        vec3 tMaxVec = max(t1, t2);
//      vec3 tMaxVec = max(t1, t2);

        float overallDistanceMin = maxVec3(tMin);
//      float overallDistanceMin = maxVec3(tMin);
        float overallDistanceMax = minVec3(tMaxVec);
//      float overallDistanceMax = minVec3(tMaxVec);

        float finalDistanceMin = max(rayTravelDistanceLimit.min, overallDistanceMin);
//      float finalDistanceMin = max(rayTravelDistanceLimit.min, overallDistanceMin);
        float finalDistanceMax = min(rayTravelDistanceLimit.max, overallDistanceMax);
//      float finalDistanceMax = min(rayTravelDistanceLimit.max, overallDistanceMax);

        return finalDistanceMax >= finalDistanceMin;
//      return finalDistanceMax >= finalDistanceMin;
    }
//  }

    // Distance Estimation to AABB: Computes the precise parametric distance to the entry point of a bounding box. This enables crucial front-to-back heuristic sorting of BVH child nodes, drastically reducing traversal steps by prioritizing the closest spatial regions.
//  // Distance Estimation to AABB: Computes the precise parametric distance to the entry point of a bounding box. This enables crucial front-to-back heuristic sorting of BVH child nodes, drastically reducing traversal steps by prioritizing the closest spatial regions.
    float calculateDistanceToAABB3D(Ray ray, vec3 invDir, Interval rayTravelDistanceLimit, vec3 boxMin, vec3 boxMax) {
//  float calculateDistanceToAABB3D(Ray ray, vec3 invDir, Interval rayTravelDistanceLimit, vec3 boxMin, vec3 boxMax) {
        vec3 t1 = (boxMin - ray.origin) * invDir;
//      vec3 t1 = (boxMin - ray.origin) * invDir;
        vec3 t2 = (boxMax - ray.origin) * invDir;
//      vec3 t2 = (boxMax - ray.origin) * invDir;

        vec3 distancesMin = max(min(t1, t2), vec3(rayTravelDistanceLimit.min));
//      vec3 distancesMin = max(min(t1, t2), vec3(rayTravelDistanceLimit.min));
        vec3 distancesMax = min(max(t1, t2), vec3(rayTravelDistanceLimit.max));
//      vec3 distancesMax = min(max(t1, t2), vec3(rayTravelDistanceLimit.max));

        float distanceToBoxMin = maxVec3(distancesMin);
//      float distanceToBoxMin = maxVec3(distancesMin);
        float distanceToBoxMax = minVec3(distancesMax);
//      float distanceToBoxMax = minVec3(distancesMax);

        if (distanceToBoxMax >= distanceToBoxMin && distanceToBoxMin < rayTravelDistanceLimit.max) {
//      if (distanceToBoxMax >= distanceToBoxMin && distanceToBoxMin < rayTravelDistanceLimit.max) {
            return distanceToBoxMin;
//          return distanceToBoxMin;
        }
//      }
        return MISS_DISTANCE;
//      return MISS_DISTANCE;
    }
//  }

    // Ray-Triangle Fast Intersection (Any-Hit): A specialized Möller-Trumbore intersection algorithm optimized for boolean occlusion queries. It terminates immediately upon finding any valid surface within the distance limits, making it highly efficient for evaluating binary shadow rays in Next Event Estimation.
//  // Ray-Triangle Fast Intersection (Any-Hit): A specialized Möller-Trumbore intersection algorithm optimized for boolean occlusion queries. It terminates immediately upon finding any valid surface within the distance limits, making it highly efficient for evaluating binary shadow rays in Next Event Estimation.
    bool intersectTriangleAnyHit(Ray ray, int triangleIndex, Interval rayTravelDistanceLimit) {
//  bool intersectTriangleAnyHit(Ray ray, int triangleIndex, Interval rayTravelDistanceLimit) {
        int offset = triangleIndex * 9;
//      int offset = triangleIndex * 9;
        vec3 v0 = vec3(triangles[offset + 0], triangles[offset + 1], triangles[offset + 2]);
//      vec3 v0 = vec3(triangles[offset + 0], triangles[offset + 1], triangles[offset + 2]);
        vec3 v1 = vec3(triangles[offset + 3], triangles[offset + 4], triangles[offset + 5]);
//      vec3 v1 = vec3(triangles[offset + 3], triangles[offset + 4], triangles[offset + 5]);
        vec3 v2 = vec3(triangles[offset + 6], triangles[offset + 7], triangles[offset + 8]);
//      vec3 v2 = vec3(triangles[offset + 6], triangles[offset + 7], triangles[offset + 8]);

        const float EPSILON = EPSILON_INTERSECT;
//      const float EPSILON = EPSILON_INTERSECT;

        vec3 e1 = v1 - v0;
//      vec3 e1 = v1 - v0;
        vec3 e2 = v2 - v0;
//      vec3 e2 = v2 - v0;

        vec3 h = cross(ray.direction, e2);
//      vec3 h = cross(ray.direction, e2);
        float a = dot(e1, h);
//      float a = dot(e1, h);

        if (abs(a) < EPSILON) return false;
//      if (abs(a) < EPSILON) return false;

        float f = 1.0 / a;
//      float f = 1.0 / a;
        vec3 s = ray.origin - v0;
//      vec3 s = ray.origin - v0;
        float u = f * dot(s, h);
//      float u = f * dot(s, h);

        if (u < 0.0 || u > 1.0) return false;
//      if (u < 0.0 || u > 1.0) return false;

        vec3 q = cross(s, e1);
//      vec3 q = cross(s, e1);
        float v = f * dot(ray.direction, q);
//      float v = f * dot(ray.direction, q);

        if (v < 0.0 || u + v > 1.0) return false;
//      if (v < 0.0 || u + v > 1.0) return false;

        float t = f * dot(e2, q);
//      float t = f * dot(e2, q);

        if (!intervalSurround(rayTravelDistanceLimit, t)) return false;
//      if (!intervalSurround(rayTravelDistanceLimit, t)) return false;

        return true;
//      return true;
    }
//  }

    // Ray-Triangle Detailed Intersection (Closest-Hit): A comprehensive Möller-Trumbore implementation that computes barycentric coordinates upon a successful hit. It leverages these coordinates to seamlessly interpolate per-vertex attributes such as UVs, normals, and tangents for high-fidelity shading.
//  // Ray-Triangle Detailed Intersection (Closest-Hit): A comprehensive Möller-Trumbore implementation that computes barycentric coordinates upon a successful hit. It leverages these coordinates to seamlessly interpolate per-vertex attributes such as UVs, normals, and tangents for high-fidelity shading.
    RayHitResult intersectTriangleClosestHit(Ray ray, int triangleIndex, Interval rayTravelDistanceLimit) {
//  RayHitResult intersectTriangleClosestHit(Ray ray, int triangleIndex, Interval rayTravelDistanceLimit) {
        RayHitResult result;
//      RayHitResult result;
        result.materialIndex = triangleIndex;
//      result.materialIndex = triangleIndex;

        result.isHitted = false;
//      result.isHitted = false;
        result.triangleIndex = triangleIndex;
//      result.triangleIndex = triangleIndex;
        result.minDistance = rayTravelDistanceLimit.max;
//      result.minDistance = rayTravelDistanceLimit.max;

        int offset = triangleIndex * 9;
//      int offset = triangleIndex * 9;
        vec3 v0 = vec3(triangles[offset + 0], triangles[offset + 1], triangles[offset + 2]);
//      vec3 v0 = vec3(triangles[offset + 0], triangles[offset + 1], triangles[offset + 2]);
        vec3 v1 = vec3(triangles[offset + 3], triangles[offset + 4], triangles[offset + 5]);
//      vec3 v1 = vec3(triangles[offset + 3], triangles[offset + 4], triangles[offset + 5]);
        vec3 v2 = vec3(triangles[offset + 6], triangles[offset + 7], triangles[offset + 8]);
//      vec3 v2 = vec3(triangles[offset + 6], triangles[offset + 7], triangles[offset + 8]);

        const float EPSILON = EPSILON_INTERSECT;
//      const float EPSILON = EPSILON_INTERSECT;
        vec3 e1 = v1 - v0;
//      vec3 e1 = v1 - v0;
        vec3 e2 = v2 - v0;
//      vec3 e2 = v2 - v0;

        vec3 h = cross(ray.direction, e2);
//      vec3 h = cross(ray.direction, e2);
        float a = dot(e1, h);
//      float a = dot(e1, h);

        if (abs(a) < EPSILON) return result;
//      if (abs(a) < EPSILON) return result;

        float f = 1.0 / a;
//      float f = 1.0 / a;
        vec3 s = ray.origin - v0;
//      vec3 s = ray.origin - v0;
        float u = f * dot(s, h);
//      float u = f * dot(s, h);

        if (u < 0.0 || u > 1.0) return result;
//      if (u < 0.0 || u > 1.0) return result;

        vec3 q = cross(s, e1);
//      vec3 q = cross(s, e1);
        float v = f * dot(ray.direction, q);
//      float v = f * dot(ray.direction, q);

        if (v < 0.0 || u + v > 1.0) return result;
//      if (v < 0.0 || u + v > 1.0) return result;

        float t = f * dot(e2, q);
//      float t = f * dot(e2, q);

        if (!intervalSurround(rayTravelDistanceLimit, t)) return result;
//      if (!intervalSurround(rayTravelDistanceLimit, t)) return result;

        if (t > EPSILON) {
//      if (t > EPSILON) {
            result.isHitted = true;
//          result.isHitted = true;
            result.minDistance = t;
//          result.minDistance = t;
            result.triangleIndex = triangleIndex;
//          result.triangleIndex = triangleIndex;
            // Interpolate UVs
//          // Interpolate UVs
            int uvOffset = triangleIndex * 3;
//          int uvOffset = triangleIndex * 3;
            vec2 uv0 = uvs[uvOffset + 0];
//          vec2 uv0 = uvs[uvOffset + 0];
            vec2 uv1 = uvs[uvOffset + 1];
//          vec2 uv1 = uvs[uvOffset + 1];
            vec2 uv2 = uvs[uvOffset + 2];
//          vec2 uv2 = uvs[uvOffset + 2];
            vec2 interpolatedUV = (1.0 - u - v) * uv0 + u * uv1 + v * uv2;
//          vec2 interpolatedUV = (1.0 - u - v) * uv0 + u * uv1 + v * uv2;
            result.uvSurfaceCoordinate = interpolatedUV;
//          result.uvSurfaceCoordinate = interpolatedUV;
            // Interpolate Normals
//          // Interpolate Normals
            int normalOffsetIdx = triangleIndex * 9;
//          int normalOffsetIdx = triangleIndex * 9;
            vec3 n0 = vec3(normals[normalOffsetIdx + 0], normals[normalOffsetIdx + 1], normals[normalOffsetIdx + 2]);
//          vec3 n0 = vec3(normals[normalOffsetIdx + 0], normals[normalOffsetIdx + 1], normals[normalOffsetIdx + 2]);
            vec3 n1 = vec3(normals[normalOffsetIdx + 3], normals[normalOffsetIdx + 4], normals[normalOffsetIdx + 5]);
//          vec3 n1 = vec3(normals[normalOffsetIdx + 3], normals[normalOffsetIdx + 4], normals[normalOffsetIdx + 5]);
            vec3 n2 = vec3(normals[normalOffsetIdx + 6], normals[normalOffsetIdx + 7], normals[normalOffsetIdx + 8]);
//          vec3 n2 = vec3(normals[normalOffsetIdx + 6], normals[normalOffsetIdx + 7], normals[normalOffsetIdx + 8]);
            vec3 interpolatedNormal = normalize((1.0 - u - v) * n0 + u * n1 + v * n2);
//          vec3 interpolatedNormal = normalize((1.0 - u - v) * n0 + u * n1 + v * n2);
            result.hittedSideNormal = interpolatedNormal;
//          result.hittedSideNormal = interpolatedNormal;

            // Interpolate Tangents
//          // Interpolate Tangents
            vec3 t0 = vec3(tangents[normalOffsetIdx + 0], tangents[normalOffsetIdx + 1], tangents[normalOffsetIdx + 2]);
//          vec3 t0 = vec3(tangents[normalOffsetIdx + 0], tangents[normalOffsetIdx + 1], tangents[normalOffsetIdx + 2]);
            vec3 t1 = vec3(tangents[normalOffsetIdx + 3], tangents[normalOffsetIdx + 4], tangents[normalOffsetIdx + 5]);
//          vec3 t1 = vec3(tangents[normalOffsetIdx + 3], tangents[normalOffsetIdx + 4], tangents[normalOffsetIdx + 5]);
            vec3 t2 = vec3(tangents[normalOffsetIdx + 6], tangents[normalOffsetIdx + 7], tangents[normalOffsetIdx + 8]);
//          vec3 t2 = vec3(tangents[normalOffsetIdx + 6], tangents[normalOffsetIdx + 7], tangents[normalOffsetIdx + 8]);
            vec3 interpolatedTangent = normalize((1.0 - u - v) * t0 + u * t1 + v * t2);
//          vec3 interpolatedTangent = normalize((1.0 - u - v) * t0 + u * t1 + v * t2);
            result.hittedSideTangent = interpolatedTangent;
//          result.hittedSideTangent = interpolatedTangent;

            return result;
//          return result;
        }
//      }

        return result;
//      return result;
    }
//  }

    // BVH Occlusion Query (Any-Hit Traversal): Rapidly traverses the hierarchy to determine if any unoccluded geometry exists between two points. Designed specifically for shadow rays, it aborts traversal the moment a blocking primitive is found, saving unnecessary distance calculations.
//  // BVH Occlusion Query (Any-Hit Traversal): Rapidly traverses the hierarchy to determine if any unoccluded geometry exists between two points. Designed specifically for shadow rays, it aborts traversal the moment a blocking primitive is found, saving unnecessary distance calculations.
    bool traverseBVHAnyHit(Ray ray, Interval rayTravelDistanceLimit) {
//  bool traverseBVHAnyHit(Ray ray, Interval rayTravelDistanceLimit) {
        vec3 invDir = 1.0 / ray.direction;
//      vec3 invDir = 1.0 / ray.direction;
        int stack[32];
//      int stack[32];
        int stackPtr = 0;
//      int stackPtr = 0;

        // Check Root
//      // Check Root
        Node rootNode = nodes[0];
//      Node rootNode = nodes[0];
        if (!intersectAABB_Interval(ray, invDir, rayTravelDistanceLimit, rootNode.min_left.xyz, rootNode.max_right.xyz)) {
//      if (!intersectAABB_Interval(ray, invDir, rayTravelDistanceLimit, rootNode.min_left.xyz, rootNode.max_right.xyz)) {
            return false;
//          return false;
        }
//      }

        stack[stackPtr++] = 0;
//      stack[stackPtr++] = 0;

        while (stackPtr > 0) {
//      while (stackPtr > 0) {
            int nodeIdx = stack[--stackPtr];
//          int nodeIdx = stack[--stackPtr];
            Node node = nodes[nodeIdx];
//          Node node = nodes[nodeIdx];

            if (node.min_left.w < 0.0) { // Leaf
//          if (node.min_left.w < 0.0) { // Leaf
                int triIdx = int(node.max_right.w);
//              int triIdx = int(node.max_right.w);
                if (intersectTriangleAnyHit(ray, triIdx, rayTravelDistanceLimit)) {
//              if (intersectTriangleAnyHit(ray, triIdx, rayTravelDistanceLimit)) {
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
            int childIndexL = int(node.min_left.w);
//          int childIndexL = int(node.min_left.w);
            int childIndexR = int(node.max_right.w);
//          int childIndexR = int(node.max_right.w);

            Node childL = nodes[childIndexL];
//          Node childL = nodes[childIndexL];
            Node childR = nodes[childIndexR];
//          Node childR = nodes[childIndexR];

            float distL = calculateDistanceToAABB3D(ray, invDir, rayTravelDistanceLimit, childL.min_left.xyz, childL.max_right.xyz);
//          float distL = calculateDistanceToAABB3D(ray, invDir, rayTravelDistanceLimit, childL.min_left.xyz, childL.max_right.xyz);
            float distR = calculateDistanceToAABB3D(ray, invDir, rayTravelDistanceLimit, childR.min_left.xyz, childR.max_right.xyz);
//          float distR = calculateDistanceToAABB3D(ray, invDir, rayTravelDistanceLimit, childR.min_left.xyz, childR.max_right.xyz);

            if (distL == MISS_DISTANCE && distR == MISS_DISTANCE) continue;
//          if (distL == MISS_DISTANCE && distR == MISS_DISTANCE) continue;

            if (distL < distR) {
//          if (distL < distR) {
                if (distR < MISS_DISTANCE) stack[stackPtr++] = childIndexR;
//              if (distR < MISS_DISTANCE) stack[stackPtr++] = childIndexR;
                if (distL < MISS_DISTANCE) stack[stackPtr++] = childIndexL;
//              if (distL < MISS_DISTANCE) stack[stackPtr++] = childIndexL;
            } else {
//          } else {
                if (distL < MISS_DISTANCE) stack[stackPtr++] = childIndexL;
//              if (distL < MISS_DISTANCE) stack[stackPtr++] = childIndexL;
                if (distR < MISS_DISTANCE) stack[stackPtr++] = childIndexR;
//              if (distR < MISS_DISTANCE) stack[stackPtr++] = childIndexR;
            }
//          }
        }
//      }
        return false;
//      return false;
    }
//  }

    // BVH Nearest Hit Query (Closest-Hit Traversal): Rigorously searches the bounding volume hierarchy to find the single closest intersection point along a ray's path. It continuously narrows the valid depth interval as closer hits are discovered, discarding occluded geometry behind the nearest surface.
//  // BVH Nearest Hit Query (Closest-Hit Traversal): Rigorously searches the bounding volume hierarchy to find the single closest intersection point along a ray's path. It continuously narrows the valid depth interval as closer hits are discovered, discarding occluded geometry behind the nearest surface.
    RayHitResult traverseBVHClosestHit(Ray ray, Interval rayTravelDistanceLimit) {
//  RayHitResult traverseBVHClosestHit(Ray ray, Interval rayTravelDistanceLimit) {
        RayHitResult finalResult;
//      RayHitResult finalResult;
        finalResult.isHitted = false;
//      finalResult.isHitted = false;
        finalResult.minDistance = rayTravelDistanceLimit.max; // Initialize with max
//      finalResult.minDistance = rayTravelDistanceLimit.max; // Initialize with max
        float maxDist = rayTravelDistanceLimit.max;
//      float maxDist = rayTravelDistanceLimit.max;

        vec3 invDir = 1.0 / ray.direction;
//      vec3 invDir = 1.0 / ray.direction;
        int stack[32];
//      int stack[32];
        int stackPtr = 0;
//      int stackPtr = 0;

        stack[stackPtr++] = 0;
//      stack[stackPtr++] = 0;

        while (stackPtr > 0) {
//      while (stackPtr > 0) {
            int nodeIdx = stack[--stackPtr];
//          int nodeIdx = stack[--stackPtr];
            Node node = nodes[nodeIdx];
//          Node node = nodes[nodeIdx];

            Interval currentLimit;
//          Interval currentLimit;
            currentLimit.min = rayTravelDistanceLimit.min;
//          currentLimit.min = rayTravelDistanceLimit.min;
            currentLimit.max = maxDist;
//          currentLimit.max = maxDist;

            if (!intersectAABB_Interval(ray, invDir, currentLimit, node.min_left.xyz, node.max_right.xyz)) continue;
//          if (!intersectAABB_Interval(ray, invDir, currentLimit, node.min_left.xyz, node.max_right.xyz)) continue;

            if (node.min_left.w < 0.0) { // Leaf
//          if (node.min_left.w < 0.0) { // Leaf
                int triIdx = int(node.max_right.w);
//              int triIdx = int(node.max_right.w);
                RayHitResult temp = intersectTriangleClosestHit(ray, triIdx, currentLimit);
//              RayHitResult temp = intersectTriangleClosestHit(ray, triIdx, currentLimit);
                if (temp.isHitted && temp.minDistance < maxDist) {
//              if (temp.isHitted && temp.minDistance < maxDist) {
                    finalResult = temp;
//                  finalResult = temp;
                    maxDist = temp.minDistance;
//                  maxDist = temp.minDistance;
                }
//              }
                continue;
//              continue;
            }
//          }

            // Internal
//          // Internal
            int childIndexL = int(node.min_left.w);
//          int childIndexL = int(node.min_left.w);
            int childIndexR = int(node.max_right.w);
//          int childIndexR = int(node.max_right.w);
            Node childL = nodes[childIndexL];
//          Node childL = nodes[childIndexL];
            Node childR = nodes[childIndexR];
//          Node childR = nodes[childIndexR];

            float distL = calculateDistanceToAABB3D(ray, invDir, currentLimit, childL.min_left.xyz, childL.max_right.xyz);
//          float distL = calculateDistanceToAABB3D(ray, invDir, currentLimit, childL.min_left.xyz, childL.max_right.xyz);
            float distR = calculateDistanceToAABB3D(ray, invDir, currentLimit, childR.min_left.xyz, childR.max_right.xyz);
//          float distR = calculateDistanceToAABB3D(ray, invDir, currentLimit, childR.min_left.xyz, childR.max_right.xyz);

            if (distL == MISS_DISTANCE && distR == MISS_DISTANCE) continue;
//          if (distL == MISS_DISTANCE && distR == MISS_DISTANCE) continue;

            if (distL < distR) {
//          if (distL < distR) {
                if (distR < maxDist) stack[stackPtr++] = childIndexR;
//              if (distR < maxDist) stack[stackPtr++] = childIndexR;
                if (distL < maxDist) stack[stackPtr++] = childIndexL;
//              if (distL < maxDist) stack[stackPtr++] = childIndexL;
            } else {
//          } else {
                if (distL < maxDist) stack[stackPtr++] = childIndexL;
//              if (distL < maxDist) stack[stackPtr++] = childIndexL;
                if (distR < maxDist) stack[stackPtr++] = childIndexR;
//              if (distR < maxDist) stack[stackPtr++] = childIndexR;
            }
//          }
        }
//      }

        return finalResult;
//      return finalResult;
    }
//  }

    // Principled BSDF Components: A collection of physically-based Bidirectional Scattering Distribution Function routines. Includes Schlick's Fresnel approximation for view-dependent reflectivity, and the GGX (Trowbridge-Reitz) microfacet distribution model for realistic rough surface scattering.
    // Principled BSDF Components: A collection of physically-based Bidirectional Scattering Distribution Function routines. Includes Schlick's Fresnel approximation for view-dependent reflectivity, and the GGX (Trowbridge-Reitz) microfacet distribution model for realistic rough surface scattering.
    vec3 schlickFresnel(float cosine, vec3 f0) {
//  vec3 schlickFresnel(float cosine, vec3 f0) {
        float oneMinusCos = 1.0 - cosine;
//      float oneMinusCos = 1.0 - cosine;
        float oneMinusCos2 = oneMinusCos * oneMinusCos;
//      float oneMinusCos2 = oneMinusCos * oneMinusCos;
        float oneMinusCos4 = oneMinusCos2 * oneMinusCos2;
//      float oneMinusCos4 = oneMinusCos2 * oneMinusCos2;
        float oneMinusCos5 = oneMinusCos4 * oneMinusCos;
//      float oneMinusCos5 = oneMinusCos4 * oneMinusCos;
        return f0 + (1.0 - f0) * oneMinusCos5;
//      return f0 + (1.0 - f0) * oneMinusCos5;
    }
//  }

    vec3 sampleGGX(vec3 normal, float roughness) {
//  vec3 sampleGGX(vec3 normal, float roughness) {
        float random1 = rand();
//      float random1 = rand();
        // Clamp random2 to prevent NaN/Inf in GGX sampling
//      // Clamp random2 to prevent NaN/Inf in GGX sampling
        float random2 = clamp(rand(), 0.0, EPSILON_RAND);
//      float random2 = clamp(rand(), 0.0, EPSILON_RAND);

        float term = roughness * roughness;
//      float term = roughness * roughness;
        float phi = 2.0 * PI * random1;
//      float phi = 2.0 * PI * random1;
        float cosTheta = sqrt((1.0 - random2) / (1.0 + (term * term - 1.0) * random2));
//      float cosTheta = sqrt((1.0 - random2) / (1.0 + (term * term - 1.0) * random2));
        float sinTheta = sqrt(1.0 - cosTheta * cosTheta);
//      float sinTheta = sqrt(1.0 - cosTheta * cosTheta);

        float x = sinTheta * cos(phi);
//      float x = sinTheta * cos(phi);
        float y = sinTheta * sin(phi);
//      float y = sinTheta * sin(phi);
        float z = cosTheta;
//      float z = cosTheta;

        // Tangent space basis construction (orthonormal basis)
//      // Tangent space basis construction (orthonormal basis)
        vec3 up = abs(normal.z) < EPSILON_RAND ? vec3(0.0, 0.0, 1.0) : vec3(1.0, 0.0, 0.0);
//      vec3 up = abs(normal.z) < EPSILON_RAND ? vec3(0.0, 0.0, 1.0) : vec3(1.0, 0.0, 0.0);
        vec3 tangent = normalize(cross(up, normal));
//      vec3 tangent = normalize(cross(up, normal));
        vec3 bitangent = cross(normal, tangent);
//      vec3 bitangent = cross(normal, tangent);

        // Transform H to world space
//      // Transform H to world space
        return normalize(tangent * x + bitangent * y + normal * z);
//      return normalize(tangent * x + bitangent * y + normal * z);
    }
//  }

    vec3 reflectPrincipled(vec3 incomingVector, vec3 normal) {
//  vec3 reflectPrincipled(vec3 incomingVector, vec3 normal) {
        return incomingVector - 2.0 * dot(incomingVector, normal) * normal;
//      return incomingVector - 2.0 * dot(incomingVector, normal) * normal;
    }
//  }

    vec3 refractPrincipled(vec3 incomingVector, vec3 normal, float etaRatioOfIncidenceOverTransmission, float cosThetaIncidence, float sinThetaTransmission) {
//  vec3 refractPrincipled(vec3 incomingVector, vec3 normal, float etaRatioOfIncidenceOverTransmission, float cosThetaIncidence, float sinThetaTransmission) {
        float cosThetaTransmission = sqrt(1.0 - sinThetaTransmission);
//      float cosThetaTransmission = sqrt(1.0 - sinThetaTransmission);
        vec3 refractedDirection = normalize(etaRatioOfIncidenceOverTransmission * incomingVector + (etaRatioOfIncidenceOverTransmission * cosThetaIncidence - cosThetaTransmission) * normal);
//      vec3 refractedDirection = normalize(etaRatioOfIncidenceOverTransmission * incomingVector + (etaRatioOfIncidenceOverTransmission * cosThetaIncidence - cosThetaTransmission) * normal);
        return refractedDirection;
//      return refractedDirection;
    }
//  }

    vec3 evalDisneyDiffuse(vec3 N, vec3 V, vec3 L, vec3 albedo, float roughness) {
//  vec3 evalDisneyDiffuse(vec3 N, vec3 V, vec3 L, vec3 albedo, float roughness) {
        vec3 H = normalize(V + L);
//      vec3 H = normalize(V + L);
        float NdotL = max(dot(N, L), 0.0);
//      float NdotL = max(dot(N, L), 0.0);
        float NdotV = max(dot(N, V), 0.0);
//      float NdotV = max(dot(N, V), 0.0);
        float LdotH = max(dot(L, H), 0.0);
//      float LdotH = max(dot(L, H), 0.0);

        // Schlick weight for grazing angles
//      // Schlick weight for grazing angles
        float fd90 = 0.5 + 2.0 * roughness * LdotH * LdotH;
//      float fd90 = 0.5 + 2.0 * roughness * LdotH * LdotH;
        float lightScatter = 1.0 + (fd90 - 1.0) * pow(1.0 - NdotL, 5.0);
//      float lightScatter = 1.0 + (fd90 - 1.0) * pow(1.0 - NdotL, 5.0);
        float viewScatter = 1.0 + (fd90 - 1.0) * pow(1.0 - NdotV, 5.0);
//      float viewScatter = 1.0 + (fd90 - 1.0) * pow(1.0 - NdotV, 5.0);

        return (albedo / PI) * lightScatter * viewScatter;
//      return (albedo / PI) * lightScatter * viewScatter;
    }
//  }

    vec3 evalOrenNayarDiffuse(vec3 N, vec3 V, vec3 L, vec3 albedo, float roughness) {
//  vec3 evalOrenNayarDiffuse(vec3 N, vec3 V, vec3 L, vec3 albedo, float roughness) {
        float NdotL = max(dot(N, L), 0.0);
//      float NdotL = max(dot(N, L), 0.0);
        float NdotV = max(dot(N, V), 0.0);
//      float NdotV = max(dot(N, V), 0.0);

        float LdotV = dot(L, V);
//      float LdotV = dot(L, V);
        float s = LdotV - NdotL * NdotV;
//      float s = LdotV - NdotL * NdotV;
        float t = mix(1.0, max(NdotL, NdotV), step(0.0, s));
//      float t = mix(1.0, max(NdotL, NdotV), step(0.0, s));

        float sigma2 = roughness * roughness;
//      float sigma2 = roughness * roughness;
        float A = 1.0 - 0.5 * (sigma2 / (sigma2 + 0.33));
//      float A = 1.0 - 0.5 * (sigma2 / (sigma2 + 0.33));
        float B = 0.45 * (sigma2 / (sigma2 + 0.09));
//      float B = 0.45 * (sigma2 / (sigma2 + 0.09));

        return (albedo / PI) * (A + B * (s / (t + EPSILON_MATH)));
//      return (albedo / PI) * (A + B * (s / (t + EPSILON_MATH)));
    }
//  }

    // BSDF Evaluation: Computes the precise ratio of scattered radiance for a specific incoming/outgoing direction pair at a shading point. This combines both the lambertian diffuse and GGX specular microfacet models, scaled by the calculated energy-conserving Fresnel and Geometry terms.
//  // BSDF Evaluation: Computes the precise ratio of scattered radiance for a specific incoming/outgoing direction pair at a shading point. This combines both the lambertian diffuse and GGX specular microfacet models, scaled by the calculated energy-conserving Fresnel and Geometry terms.
    vec3 evalPrincipledBSDF(vec3 incomingDir, vec3 outgoingDir, vec3 normal, vec3 albedo, float roughness, float metallic) {
//  vec3 evalPrincipledBSDF(vec3 incomingDir, vec3 outgoingDir, vec3 normal, vec3 albedo, float roughness, float metallic) {
        vec3 N = normal;
//      vec3 N = normal;
        vec3 V = -incomingDir;
//      vec3 V = -incomingDir;
        vec3 L = outgoingDir;
//      vec3 L = outgoingDir;
        vec3 H = normalize(V + L);
//      vec3 H = normalize(V + L);

        float NdotL = max(dot(N, L), 0.0);
//      float NdotL = max(dot(N, L), 0.0);
        float NdotV = max(dot(N, V), 0.0);
//      float NdotV = max(dot(N, V), 0.0);

        if (NdotL <= 0.0 || NdotV <= 0.0) return vec3(0.0);
//      if (NdotL <= 0.0 || NdotV <= 0.0) return vec3(0.0);

        vec3 F0 = mix(vec3(F0_DEFAULT), albedo, metallic);
//      vec3 F0 = mix(vec3(F0_DEFAULT), albedo, metallic);
        float HdotV = max(dot(H, V), 0.0);
//      float HdotV = max(dot(H, V), 0.0);
        vec3 F = schlickFresnel(HdotV, F0);
//      vec3 F = schlickFresnel(HdotV, F0);

        // Diffuse
//      // Diffuse
        vec3 kS = F;
//      vec3 kS = F;
        vec3 kD = (vec3(1.0) - kS) * (1.0 - metallic);
//      vec3 kD = (vec3(1.0) - kS) * (1.0 - metallic);

        // --- Try uncommenting one of these ---
//      // --- Try uncommenting one of these ---

        // 1. Original Lambert
//      // 1. Original Lambert
        // vec3 diffuse = kD * albedo / PI;
//      // vec3 diffuse = kD * albedo / PI;

        // 2. Disney Diffuse (Recommended)
//      // 2. Disney Diffuse (Recommended)
        vec3 diffuse = kD * evalDisneyDiffuse(N, V, L, albedo, roughness);
//      vec3 diffuse = kD * evalDisneyDiffuse(N, V, L, albedo, roughness);

        // 3. Oren-Nayar Diffuse
//      // 3. Oren-Nayar Diffuse
        // vec3 diffuse = kD * evalOrenNayarDiffuse(N, V, L, albedo, roughness);
//      // vec3 diffuse = kD * evalOrenNayarDiffuse(N, V, L, albedo, roughness);

        // Specular
//      // Specular
        float alpha = roughness * roughness;
//      float alpha = roughness * roughness;
        float alpha2 = alpha * alpha;
//      float alpha2 = alpha * alpha;
        float NdotH = max(dot(N, H), 0.0);
//      float NdotH = max(dot(N, H), 0.0);
        float denom = (NdotH * NdotH * (alpha2 - 1.0) + 1.0);
//      float denom = (NdotH * NdotH * (alpha2 - 1.0) + 1.0);
        float D = alpha2 / (PI * denom * denom);
//      float D = alpha2 / (PI * denom * denom);

        float k = (roughness + 1.0) * (roughness + 1.0) / 8.0;
//      float k = (roughness + 1.0) * (roughness + 1.0) / 8.0;
        // Optimize G term calculation by factoring out NdotV and NdotL
//      // Optimize G term calculation by factoring out NdotV and NdotL
        float NdotV_G = NdotV * (1.0 - k) + k;
//      float NdotV_G = NdotV * (1.0 - k) + k;
        float NdotL_G = NdotL * (1.0 - k) + k;
//      float NdotL_G = NdotL * (1.0 - k) + k;

        vec3 specular = (D * F) / (4.0 * NdotV_G * NdotL_G + EPSILON_MATH);
//      vec3 specular = (D * F) / (4.0 * NdotV_G * NdotL_G + EPSILON_MATH);

        return diffuse + specular;
//      return diffuse + specular;
    }
//  }

    float evalPrincipledPDF(vec3 incomingDir, vec3 outgoingDir, vec3 normal, vec3 albedo, float roughness, float metallic) {
//  float evalPrincipledPDF(vec3 incomingDir, vec3 outgoingDir, vec3 normal, vec3 albedo, float roughness, float metallic) {
        vec3 N = normal;
//      vec3 N = normal;
        vec3 V = -incomingDir;
//      vec3 V = -incomingDir;
        vec3 L = outgoingDir;
//      vec3 L = outgoingDir;
        vec3 H = normalize(V + L);
//      vec3 H = normalize(V + L);

        float NdotL = max(dot(N, L), 0.0);
//      float NdotL = max(dot(N, L), 0.0);
        float NdotV = max(dot(N, V), 0.0);
//      float NdotV = max(dot(N, V), 0.0);
        if (NdotL <= 0.0 || NdotV <= 0.0) return 0.0;
//      if (NdotL <= 0.0 || NdotV <= 0.0) return 0.0;

        float pdfDiffuse = NdotL / PI;
//      float pdfDiffuse = NdotL / PI;

        float alpha = roughness * roughness;
//      float alpha = roughness * roughness;
        float alpha2 = max(alpha * alpha, EPSILON_MATH);
//      float alpha2 = max(alpha * alpha, EPSILON_MATH);
        float NdotH = max(dot(N, H), 0.0);
//      float NdotH = max(dot(N, H), 0.0);
        float denom = (NdotH * NdotH * (alpha2 - 1.0) + 1.0);
//      float denom = (NdotH * NdotH * (alpha2 - 1.0) + 1.0);
        float D = alpha2 / (PI * denom * denom);
//      float D = alpha2 / (PI * denom * denom);
        float VdotH = max(dot(V, H), 0.0);
//      float VdotH = max(dot(V, H), 0.0);
        float pdfSpecular = (D * NdotH) / (4.0 * VdotH + EPSILON_DOT);
//      float pdfSpecular = (D * NdotH) / (4.0 * VdotH + EPSILON_DOT);

        vec3 F0 = mix(vec3(F0_DEFAULT), albedo, metallic);
//      vec3 F0 = mix(vec3(F0_DEFAULT), albedo, metallic);
        vec3 F = schlickFresnel(VdotH, F0);
//      vec3 F = schlickFresnel(VdotH, F0);
        float fresnelProb = (F.r + F.g + F.b) / 3.0;
//      float fresnelProb = (F.r + F.g + F.b) / 3.0;
        float specularProbability = mix(fresnelProb, 1.0, metallic);
//      float specularProbability = mix(fresnelProb, 1.0, metallic);

        return mix(pdfDiffuse, pdfSpecular, specularProbability);
//      return mix(pdfDiffuse, pdfSpecular, specularProbability);
    }
//  }

    // Stochastic Importance Sampling: Generates optimal secondary ray directions guided by the material's underlying BSDF probability density functions (PDF). Specular paths follow a GGX-mapped distribution, while diffuse paths follow a cosine-weighted hemisphere, minimizing Monte Carlo variance.
//  // Stochastic Importance Sampling: Generates optimal secondary ray directions guided by the material's underlying BSDF probability density functions (PDF). Specular paths follow a GGX-mapped distribution, while diffuse paths follow a cosine-weighted hemisphere, minimizing Monte Carlo variance.
    MaterialLightScatteringResult scatterPrincipled(Ray incomingRay, RayHitResult recentRayHitResult, Material material) {
//  MaterialLightScatteringResult scatterPrincipled(Ray incomingRay, RayHitResult recentRayHitResult, Material material) {
        MaterialLightScatteringResult materialLightScatteringResult;
//      MaterialLightScatteringResult materialLightScatteringResult;
        materialLightScatteringResult.scatteredRay.origin = vec3(0.0);
//      materialLightScatteringResult.scatteredRay.origin = vec3(0.0);
        materialLightScatteringResult.scatteredRay.direction = vec3(0.0);
//      materialLightScatteringResult.scatteredRay.direction = vec3(0.0);
        materialLightScatteringResult.attenuation = vec3(0.0);
//      materialLightScatteringResult.attenuation = vec3(0.0);
        materialLightScatteringResult.emission = vec3(0.0);
//      materialLightScatteringResult.emission = vec3(0.0);
        materialLightScatteringResult.isScattered = false;
//      materialLightScatteringResult.isScattered = false;
        materialLightScatteringResult.albedo = vec3(0.0);
//      materialLightScatteringResult.albedo = vec3(0.0);
        materialLightScatteringResult.roughness = 0.0;
//      materialLightScatteringResult.roughness = 0.0;
        materialLightScatteringResult.metallic = 0.0;
//      materialLightScatteringResult.metallic = 0.0;
        materialLightScatteringResult.shadingNormal = vec3(0.0);
//      materialLightScatteringResult.shadingNormal = vec3(0.0);
        materialLightScatteringResult.pdf = 0.0;
//      materialLightScatteringResult.pdf = 0.0;
        materialLightScatteringResult.isDelta = false;
//      materialLightScatteringResult.isDelta = false;

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
        if (material.textureIndexAlbedo > -0.5) {
//      if (material.textureIndexAlbedo > -0.5) {
            albedo = textureLod(uSceneTextureArray, vec3(recentRayHitResult.uvSurfaceCoordinate, material.textureIndexAlbedo), 0.0).rgb;
//          albedo = textureLod(uSceneTextureArray, vec3(recentRayHitResult.uvSurfaceCoordinate, material.textureIndexAlbedo), 0.0).rgb;
        }
//      }
        if (material.textureIndexRoughness > -0.5) {
//      if (material.textureIndexRoughness > -0.5) {
            // Assume roughness is in R channel or grayscale
//          // Assume roughness is in R channel or grayscale
            roughness = textureLod(uSceneTextureArray, vec3(recentRayHitResult.uvSurfaceCoordinate, material.textureIndexRoughness), 0.0).r;
//          roughness = textureLod(uSceneTextureArray, vec3(recentRayHitResult.uvSurfaceCoordinate, material.textureIndexRoughness), 0.0).r;
        }
//      }
        if (material.textureIndexMetallic > -0.5) {
//      if (material.textureIndexMetallic > -0.5) {
            // Assume metallic is in R channel or grayscale
//          // Assume metallic is in R channel or grayscale
            metallic = textureLod(uSceneTextureArray, vec3(recentRayHitResult.uvSurfaceCoordinate, material.textureIndexMetallic), 0.0).r;
//          metallic = textureLod(uSceneTextureArray, vec3(recentRayHitResult.uvSurfaceCoordinate, material.textureIndexMetallic), 0.0).r;
        }
//      }

        // Clamp roughness to prevent division by zero in specular calculations
//      // Clamp roughness to prevent division by zero in specular calculations
        roughness = max(roughness, MIN_ROUGHNESS);
//      roughness = max(roughness, MIN_ROUGHNESS);

        if (material.textureIndexEmissive > -0.5) {
//      if (material.textureIndexEmissive > -0.5) {
            materialLightScatteringResult.emission = material.emissive * textureLod(uSceneTextureArray, vec3(recentRayHitResult.uvSurfaceCoordinate, material.textureIndexEmissive), 0.0).rgb;
//          materialLightScatteringResult.emission = material.emissive * textureLod(uSceneTextureArray, vec3(recentRayHitResult.uvSurfaceCoordinate, material.textureIndexEmissive), 0.0).rgb;
        } else {
//      } else {
            materialLightScatteringResult.emission = material.emissive * albedo;
//          materialLightScatteringResult.emission = material.emissive * albedo;
        }
//      }

        if (materialLightScatteringResult.emission.r > 0.0 || materialLightScatteringResult.emission.g > 0.0 || materialLightScatteringResult.emission.b > 0.0) {
//      if (materialLightScatteringResult.emission.r > 0.0 || materialLightScatteringResult.emission.g > 0.0 || materialLightScatteringResult.emission.b > 0.0) {
            return materialLightScatteringResult;
//          return materialLightScatteringResult;
        }
//      }

        // Principled Logic
//      // Principled Logic

        // 1. Shading Normal Calculation (Normal Mapping)
//      // 1. Shading Normal Calculation (Normal Mapping)
        vec3 shadingNormal = recentRayHitResult.hittedSideNormal;
//      vec3 shadingNormal = recentRayHitResult.hittedSideNormal;

        if (material.textureIndexNormal > -0.5) {
//      if (material.textureIndexNormal > -0.5) {
            vec3 mapN = textureLod(uSceneTextureArray, vec3(recentRayHitResult.uvSurfaceCoordinate, material.textureIndexNormal), 0.0).rgb;
//          vec3 mapN = textureLod(uSceneTextureArray, vec3(recentRayHitResult.uvSurfaceCoordinate, material.textureIndexNormal), 0.0).rgb;
            mapN = mapN * 2.0 - 1.0;
//          mapN = mapN * 2.0 - 1.0;

            vec3 N = normalize(shadingNormal);
//          vec3 N = normalize(shadingNormal);
            vec3 T = normalize(recentRayHitResult.hittedSideTangent);
//          vec3 T = normalize(recentRayHitResult.hittedSideTangent);
            // Re-orthogonalize T with respect to N
//          // Re-orthogonalize T with respect to N
            T = normalize(T - dot(T, N) * N);
//          T = normalize(T - dot(T, N) * N);
            vec3 B = cross(N, T);
//          vec3 B = cross(N, T);
            mat3 TBN = mat3(T, B, N);
//          mat3 TBN = mat3(T, B, N);

            shadingNormal = normalize(TBN * mapN);
//          shadingNormal = normalize(TBN * mapN);
        }
//      }

        // Store material properties for NEE
//      // Store material properties for NEE
        materialLightScatteringResult.albedo = albedo;
//      materialLightScatteringResult.albedo = albedo;
        materialLightScatteringResult.roughness = roughness;
//      materialLightScatteringResult.roughness = roughness;
        materialLightScatteringResult.metallic = metallic;
//      materialLightScatteringResult.metallic = metallic;
        materialLightScatteringResult.shadingNormal = shadingNormal;
//      materialLightScatteringResult.shadingNormal = shadingNormal;

        // F0 calculation: 0.04 for dielectrics, albedo for metals
//      // F0 calculation: 0.04 for dielectrics, albedo for metals
        vec3 f0 = mix(vec3(F0_DEFAULT), albedo, metallic);
//      vec3 f0 = mix(vec3(F0_DEFAULT), albedo, metallic);

        // Early exit for back-facing rays to prevent shading artifacts
//      // Early exit for back-facing rays to prevent shading artifacts
        if (dot(shadingNormal, -incomingRay.direction) <= 0.0) {
//      if (dot(shadingNormal, -incomingRay.direction) <= 0.0) {
            materialLightScatteringResult.isScattered = false;
//          materialLightScatteringResult.isScattered = false;
            return materialLightScatteringResult;
//          return materialLightScatteringResult;
        }
//      }

        // Schlick's fresnel approximation at incident angle
//      // Schlick's fresnel approximation at incident angle
        float cosTheta = clamp(dot(-incomingRay.direction, shadingNormal), 0.0, 1.0);
//      float cosTheta = clamp(dot(-incomingRay.direction, shadingNormal), 0.0, 1.0);
        vec3 fresnel = schlickFresnel(cosTheta, f0);
//      vec3 fresnel = schlickFresnel(cosTheta, f0);
        // Use average fresnel for importance sampling probability
//      // Use average fresnel for importance sampling probability
        float fresnelProb = (fresnel.r + fresnel.g + fresnel.b) / 3.0;
//      float fresnelProb = (fresnel.r + fresnel.g + fresnel.b) / 3.0;

        float randomChoice = rand();
//      float randomChoice = rand();

        // --- PATH A: METALLIC REFLECTION ---
//      // --- PATH A: METALLIC REFLECTION ---
        // Clamp specular probability to minimum 15% to eliminate dielectric fireflies caused by extremely low 1/PDF divisions
//      // Clamp specular probability to minimum 15% to eliminate dielectric fireflies caused by extremely low 1/PDF divisions
        float specularProbability = max(mix(fresnelProb, 1.0, metallic), 0.15);
//      float specularProbability = max(mix(fresnelProb, 1.0, metallic), 0.15);

        if (randomChoice < specularProbability) {
//      if (randomChoice < specularProbability) {
            // SPECULAR REFLECTION (METAL OR DIELECTRIC COAT)
//          // SPECULAR REFLECTION (METAL OR DIELECTRIC COAT)
            vec3 microfacetNormal = sampleGGX(shadingNormal, roughness);
//          vec3 microfacetNormal = sampleGGX(shadingNormal, roughness);
            vec3 specularReflectedDirection = reflectPrincipled(incomingRay.direction, microfacetNormal);
//          vec3 specularReflectedDirection = reflectPrincipled(incomingRay.direction, microfacetNormal);

            if (dot(specularReflectedDirection, shadingNormal) > 0.0) {
//          if (dot(specularReflectedDirection, shadingNormal) > 0.0) {
                materialLightScatteringResult.scatteredRay.origin = recentRayHitResult.at;
//              materialLightScatteringResult.scatteredRay.origin = recentRayHitResult.at;
                materialLightScatteringResult.scatteredRay.direction = specularReflectedDirection;
//              materialLightScatteringResult.scatteredRay.direction = specularReflectedDirection;
                float NdotV = max(dot(shadingNormal, -incomingRay.direction), EPSILON_DOT);
//              float NdotV = max(dot(shadingNormal, -incomingRay.direction), EPSILON_DOT);
                float NdotH = max(dot(shadingNormal, microfacetNormal), EPSILON_DOT);
//              float NdotH = max(dot(shadingNormal, microfacetNormal), EPSILON_DOT);
                float VdotH = max(dot(-incomingRay.direction, microfacetNormal), EPSILON_DOT);
//              float VdotH = max(dot(-incomingRay.direction, microfacetNormal), EPSILON_DOT);
                float NdotL = max(dot(shadingNormal, specularReflectedDirection), EPSILON_DOT);
//              float NdotL = max(dot(shadingNormal, specularReflectedDirection), EPSILON_DOT);
                vec3 F = schlickFresnel(VdotH, f0);
//              vec3 F = schlickFresnel(VdotH, f0);
                float k = (roughness + 1.0) * (roughness + 1.0) / 8.0;
//              float k = (roughness + 1.0) * (roughness + 1.0) / 8.0;
                // Optimize G term calculation by factoring out NdotV and NdotL
//              // Optimize G term calculation by factoring out NdotV and NdotL
                float NdotV_G = NdotV * (1.0 - k) + k;
//              float NdotV_G = NdotV * (1.0 - k) + k;
                float NdotL_G = NdotL * (1.0 - k) + k;
//              float NdotL_G = NdotL * (1.0 - k) + k;
                materialLightScatteringResult.attenuation = (F * VdotH * NdotL) / (NdotH * NdotV_G * NdotL_G * specularProbability + EPSILON_MATH);
//              materialLightScatteringResult.attenuation = (F * VdotH * NdotL) / (NdotH * NdotV_G * NdotL_G * specularProbability + EPSILON_MATH);
                materialLightScatteringResult.isScattered = true;
//              materialLightScatteringResult.isScattered = true;
                materialLightScatteringResult.isDelta = roughness < 0.05;
//              materialLightScatteringResult.isDelta = roughness < 0.05;
                materialLightScatteringResult.pdf = evalPrincipledPDF(incomingRay.direction, specularReflectedDirection, shadingNormal, albedo, roughness, metallic);
//              materialLightScatteringResult.pdf = evalPrincipledPDF(incomingRay.direction, specularReflectedDirection, shadingNormal, albedo, roughness, metallic);
            } else {
//          } else {
                // Current/Recent ray is/was absorbed (next ray is scattering into surface)
//              // Current/Recent ray is/was absorbed (next ray is scattering into surface)
                materialLightScatteringResult.isScattered = false;
//              materialLightScatteringResult.isScattered = false;
            }
//          }
        } else {
//      } else {
            // --- PATH B: DIELECTRIC (DIFFUSE OR TRANSMISSION) ---
//          // --- PATH B: DIELECTRIC (DIFFUSE OR TRANSMISSION) ---

            // Re-normalize random variable for the next choice
//          // Re-normalize random variable for the next choice
            float randomNextChoice = (randomChoice - specularProbability) / (1.0 - specularProbability);
//          float randomNextChoice = (randomChoice - specularProbability) / (1.0 - specularProbability);

            if (material.transmission > 0.0 && randomNextChoice < material.transmission) {
//          if (material.transmission > 0.0 && randomNextChoice < material.transmission) {
                // TRANSMISSION (REFRACTION)
//              // TRANSMISSION (REFRACTION)
                float etaRatioOfIncidenceOverTransmission = 1.0 / material.ior;
//              float etaRatioOfIncidenceOverTransmission = 1.0 / material.ior;
                if (!recentRayHitResult.isFrontFaceHitted) {
//              if (!recentRayHitResult.isFrontFaceHitted) {
                    etaRatioOfIncidenceOverTransmission = material.ior;
//                  etaRatioOfIncidenceOverTransmission = material.ior;
                }
//              }

                vec3 microfacetNormal = sampleGGX(shadingNormal, material.roughness);
//              vec3 microfacetNormal = sampleGGX(shadingNormal, material.roughness);

                float cosThetaIncidence = min(dot(-incomingRay.direction, microfacetNormal), 1.0);
//              float cosThetaIncidence = min(dot(-incomingRay.direction, microfacetNormal), 1.0);
                float sinThetaTransmission = (1.0 - cosThetaIncidence * cosThetaIncidence) * (etaRatioOfIncidenceOverTransmission * etaRatioOfIncidenceOverTransmission);
//              float sinThetaTransmission = (1.0 - cosThetaIncidence * cosThetaIncidence) * (etaRatioOfIncidenceOverTransmission * etaRatioOfIncidenceOverTransmission);

                vec3 refractedDirection = refractPrincipled(incomingRay.direction, microfacetNormal, etaRatioOfIncidenceOverTransmission, cosThetaIncidence, sinThetaTransmission);
//              vec3 refractedDirection = refractPrincipled(incomingRay.direction, microfacetNormal, etaRatioOfIncidenceOverTransmission, cosThetaIncidence, sinThetaTransmission);
                vec3 reflectedDirection = reflectPrincipled(incomingRay.direction, microfacetNormal);
//              vec3 reflectedDirection = reflectPrincipled(incomingRay.direction, microfacetNormal);

                // When [ sinThetaTransmission <= 1.0 ] then Refraction happened else Total Internal Reflection happened
//              // When [ sinThetaTransmission <= 1.0 ] then Refraction happened else Total Internal Reflection happened
                materialLightScatteringResult.scatteredRay.origin = recentRayHitResult.at;
//              materialLightScatteringResult.scatteredRay.origin = recentRayHitResult.at;

                if (sinThetaTransmission <= 1.0) {
//              if (sinThetaTransmission <= 1.0) {
                    materialLightScatteringResult.scatteredRay.direction = refractedDirection;
//                  materialLightScatteringResult.scatteredRay.direction = refractedDirection;
                    materialLightScatteringResult.attenuation = albedo;
//                  materialLightScatteringResult.attenuation = albedo;
                } else {
//              } else {
                    materialLightScatteringResult.scatteredRay.direction = reflectedDirection;
//                  materialLightScatteringResult.scatteredRay.direction = reflectedDirection;
                    materialLightScatteringResult.attenuation = vec3(1.0);
//                  materialLightScatteringResult.attenuation = vec3(1.0);
                }
//              }
                materialLightScatteringResult.isScattered = true;
//              materialLightScatteringResult.isScattered = true;
                materialLightScatteringResult.isDelta = true; // No NEE for transmission
//              materialLightScatteringResult.isDelta = true; // No NEE for transmission
                materialLightScatteringResult.pdf = 1.0;
//              materialLightScatteringResult.pdf = 1.0;
            } else {
//          } else {
                // DIFFUSE (LAMBERTIAN)
//              // DIFFUSE (LAMBERTIAN)
                vec3 diffuseDirection = normalize(shadingNormal + randomUnitVector());
//              vec3 diffuseDirection = normalize(shadingNormal + randomUnitVector());

                materialLightScatteringResult.scatteredRay.origin = recentRayHitResult.at;
//              materialLightScatteringResult.scatteredRay.origin = recentRayHitResult.at;
                materialLightScatteringResult.scatteredRay.direction = diffuseDirection;
//              materialLightScatteringResult.scatteredRay.direction = diffuseDirection;
                materialLightScatteringResult.attenuation = albedo;
//              materialLightScatteringResult.attenuation = albedo;
                materialLightScatteringResult.isScattered = true;
//              materialLightScatteringResult.isScattered = true;
                materialLightScatteringResult.isDelta = false;
//              materialLightScatteringResult.isDelta = false;
                materialLightScatteringResult.pdf = evalPrincipledPDF(incomingRay.direction, diffuseDirection, shadingNormal, albedo, roughness, metallic);
//              materialLightScatteringResult.pdf = evalPrincipledPDF(incomingRay.direction, diffuseDirection, shadingNormal, albedo, roughness, metallic);
            }
//          }
        }
//      }
        return materialLightScatteringResult;
//      return materialLightScatteringResult;
    }
//  }

    struct SphereLightHit {
//  struct SphereLightHit {
        bool isHitted;
//      bool isHitted;
        float dist;
//      float dist;
        int lightIndex;
//      int lightIndex;
        vec3 normal;
//      vec3 normal;
    };
//  };

    SphereLightHit intersectSphereLights(Ray ray) {
//  SphereLightHit intersectSphereLights(Ray ray) {
        SphereLightHit hit;
//      SphereLightHit hit;
        hit.isHitted = false;
//      hit.isHitted = false;
        hit.dist = INF;
//      hit.dist = INF;
        hit.lightIndex = -1;
//      hit.lightIndex = -1;

        for (int i = 0; i < uPointLightCount; i++) {
//      for (int i = 0; i < uPointLightCount; i++) {
            vec3 oc = ray.origin - uPointLights[i].position;
//          vec3 oc = ray.origin - uPointLights[i].position;
            float b = dot(oc, ray.direction);
//          float b = dot(oc, ray.direction);
            float c = dot(oc, oc) - uPointLights[i].radius * uPointLights[i].radius;
//          float c = dot(oc, oc) - uPointLights[i].radius * uPointLights[i].radius;
            float h = b * b - c;
//          float h = b * b - c;
            if (h > 0.0) {
//          if (h > 0.0) {
                float d = -b - sqrt(h);
//              float d = -b - sqrt(h);
                if (d > EPSILON_OFFSET && d < hit.dist) {
//              if (d > EPSILON_OFFSET && d < hit.dist) {
                    hit.isHitted = true;
//                  hit.isHitted = true;
                    hit.dist = d;
//                  hit.dist = d;
                    hit.lightIndex = i;
//                  hit.lightIndex = i;
                    hit.normal = normalize((ray.origin + ray.direction * d) - uPointLights[i].position);
//                  hit.normal = normalize((ray.origin + ray.direction * d) - uPointLights[i].position);
                }
//              }
            }
//          }
        }
//      }
        return hit;
//      return hit;
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

        // Primary Ray Initialization: Computes the initial world-space ray vector projecting from the virtual camera origin through the respective screen-space pixel coordinate. Applies sub-pixel jittering matrices for spatio-temporal anti-aliasing (TAA) and smoother sampling convergence.
//      // Primary Ray Initialization: Computes the initial world-space ray vector projecting from the virtual camera origin through the respective screen-space pixel coordinate. Applies sub-pixel jittering matrices for spatio-temporal anti-aliasing (TAA) and smoother sampling convergence.
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

        float lastPdfW = 1.0;
//      float lastPdfW = 1.0;
        bool lastWasDelta = true;
//      bool lastWasDelta = true;
        // Tracker for roughness annealing to mitigate Specular-Diffuse-Specular (SDS) variance explosions
//      // Tracker for roughness annealing to mitigate Specular-Diffuse-Specular (SDS) variance explosions
        float pathRoughness = 0.0;
//      float pathRoughness = 0.0;

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
                    rayHitResult.isHitted = false;
//                  rayHitResult.isHitted = false;
                } else {
//              } else {
                    rayHitResult.isHitted = true;
//                  rayHitResult.isHitted = true;
                    rayHitResult.at = sampleGlobalPosition.xyz;
//                  rayHitResult.at = sampleGlobalPosition.xyz;
                    rayHitResult.hittedSideNormal = normalize(sampleGlobalNormal.xyz);
//                  rayHitResult.hittedSideNormal = normalize(sampleGlobalNormal.xyz);
                    rayHitResult.hittedSideTangent = normalize(sampleGlobalTangent.xyz);
//                  rayHitResult.hittedSideTangent = normalize(sampleGlobalTangent.xyz);
                    rayHitResult.uvSurfaceCoordinate = vec2(sampleGlobalNormal.w, sampleGlobalTangent.w);
//                  rayHitResult.uvSurfaceCoordinate = vec2(sampleGlobalNormal.w, sampleGlobalTangent.w);
                    rayHitResult.isFrontFaceHitted = dot(currentRay.direction, rayHitResult.hittedSideNormal) < 0.0;
//                  rayHitResult.isFrontFaceHitted = dot(currentRay.direction, rayHitResult.hittedSideNormal) < 0.0;
                    rayHitResult.materialIndex = int(sampleGlobalPosition.w) - 1;
//                  rayHitResult.materialIndex = int(sampleGlobalPosition.w) - 1;
                    rayHitResult.triangleIndex = int(sampleGlobalPosition.w) - 1;
//                  rayHitResult.triangleIndex = int(sampleGlobalPosition.w) - 1;
                    rayHitResult.minDistance = length(rayHitResult.at - currentRay.origin);
//                  rayHitResult.minDistance = length(rayHitResult.at - currentRay.origin);
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

                if (rayHitResult.isHitted) {
//              if (rayHitResult.isHitted) {
                    // Fix front face normal
//                  // Fix front face normal
                    bool frontFace = dot(currentRay.direction, rayHitResult.hittedSideNormal) < 0.0;
//                  bool frontFace = dot(currentRay.direction, rayHitResult.hittedSideNormal) < 0.0;
                    rayHitResult.isFrontFaceHitted = frontFace;
//                  rayHitResult.isFrontFaceHitted = frontFace;
                    if (!frontFace) {
//                  if (!frontFace) {
                        rayHitResult.hittedSideNormal = -rayHitResult.hittedSideNormal;
//                      rayHitResult.hittedSideNormal = -rayHitResult.hittedSideNormal;
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
            if (lightHit.isHitted) {
//          if (lightHit.isHitted) {
                if (!rayHitResult.isHitted || lightHit.dist < rayHitResult.minDistance) {
//              if (!rayHitResult.isHitted || lightHit.dist < rayHitResult.minDistance) {
                    hitLight = true;
//                  hitLight = true;
                }
//              }
            }
//          }

            if (hitLight) {
//          if (hitLight) {
                vec3 Le = uPointLights[lightHit.lightIndex].color;
//              vec3 Le = uPointLights[lightHit.lightIndex].color;

                if (depth == 0 || lastWasDelta) {
//              if (depth == 0 || lastWasDelta) {
                    accumulatedColor += attenuation * Le;
//                  accumulatedColor += attenuation * Le;
                } else {
//              } else {
                    float cosThetaLight = max(0.0, dot(lightHit.normal, -currentRay.direction));
//                  float cosThetaLight = max(0.0, dot(lightHit.normal, -currentRay.direction));
                    float r = uPointLights[lightHit.lightIndex].radius;
//                  float r = uPointLights[lightHit.lightIndex].radius;
                    float lightPdf = uPointLights[lightHit.lightIndex].pdf;
//                  float lightPdf = uPointLights[lightHit.lightIndex].pdf;
                    float area = 4.0 * PI * r * r;
//                  float area = 4.0 * PI * r * r;
                    float pdfLightArea = lightPdf / area;
//                  float pdfLightArea = lightPdf / area;

                    float pdfLightW = 0.0;
//                  float pdfLightW = 0.0;
                    if (cosThetaLight > 0.0) {
//                  if (cosThetaLight > 0.0) {
                        pdfLightW = (pdfLightArea * lightHit.dist * lightHit.dist) / cosThetaLight;
//                      pdfLightW = (pdfLightArea * lightHit.dist * lightHit.dist) / cosThetaLight;
                    }
//                  }

                    float weightBrdf = 1.0;
//                  float weightBrdf = 1.0;
                    if (pdfLightW > 0.0) {
//                  if (pdfLightW > 0.0) {
                        // Replaced the Balance Heuristic with the robust Power Heuristic for calculating weightBrdf
//                      // Replaced the Balance Heuristic with the robust Power Heuristic for calculating weightBrdf
                        weightBrdf = (lastPdfW * lastPdfW) / max(lastPdfW * lastPdfW + pdfLightW * pdfLightW, 1e-8);
//                      weightBrdf = (lastPdfW * lastPdfW) / max(lastPdfW * lastPdfW + pdfLightW * pdfLightW, 1e-8);
                    }
//                  }

                    accumulatedColor += attenuation * Le * weightBrdf;
//                  accumulatedColor += attenuation * Le * weightBrdf;
                }
//              }
                break;
//              break;
            }
//          }

            if (!rayHitResult.isHitted) {
//          if (!rayHitResult.isHitted) {
                // Sky
//              // Sky
                vec3 sky = getSkyColor(currentRay.direction);
//              vec3 sky = getSkyColor(currentRay.direction);
                accumulatedColor += attenuation * sky;
//              accumulatedColor += attenuation * sky;
                break;
//              break;
            }
//          }

            Material material = materials[rayHitResult.materialIndex];
//          Material material = materials[rayHitResult.materialIndex];

            // As the ray depth increases, the surface roughness is artificially increased based on previous bounces to blur secondary reflections
//          // As the ray depth increases, the surface roughness is artificially increased based on previous bounces to blur secondary reflections
            if (depth > 0) {
//          if (depth > 0) {
                material.roughness = max(material.roughness, pathRoughness);
//              material.roughness = max(material.roughness, pathRoughness);
            }
//          }
            pathRoughness = max(pathRoughness, material.roughness);
//          pathRoughness = max(pathRoughness, material.roughness);

            MaterialLightScatteringResult scatterResult = scatterPrincipled(currentRay, rayHitResult, material);
//          MaterialLightScatteringResult scatterResult = scatterPrincipled(currentRay, rayHitResult, material);

            accumulatedColor += attenuation * scatterResult.emission;
//          accumulatedColor += attenuation * scatterResult.emission;

            // Direct Illumination (Next Event Estimation) with MIS
//          // Direct Illumination (Next Event Estimation) with MIS
            if (uPointLightCount > 0 && !scatterResult.isDelta) {
//          if (uPointLightCount > 0 && !scatterResult.isDelta) {
                float rnd = rand();
//              float rnd = rand();
                int lightIdx = 0;
//              int lightIdx = 0;
                for (int i = 0; i < uPointLightCount; i++) {
//              for (int i = 0; i < uPointLightCount; i++) {
                    if (rnd <= uPointLights[i].cdf) {
//                  if (rnd <= uPointLights[i].cdf) {
                        lightIdx = i;
//                      lightIdx = i;
                        break;
//                      break;
                    }
//                  }
                }
//              }

                vec3 lightPos = uPointLights[lightIdx].position;
//              vec3 lightPos = uPointLights[lightIdx].position;
                float r = uPointLights[lightIdx].radius;
//              float r = uPointLights[lightIdx].radius;
                float lightPdf = uPointLights[lightIdx].pdf;
//              float lightPdf = uPointLights[lightIdx].pdf;
                vec3 sampledPoint = lightPos + randomUnitVector() * r;
//              vec3 sampledPoint = lightPos + randomUnitVector() * r;

                vec3 shadowDir = sampledPoint - rayHitResult.at;
//              vec3 shadowDir = sampledPoint - rayHitResult.at;
                float distLight = length(shadowDir);
//              float distLight = length(shadowDir);
                // Clamped distLight to slightly above the light's radius to prevent division-by-zero singularities
//              // Clamped distLight to slightly above the light's radius to prevent division-by-zero singularities
                distLight = max(distLight, r + 0.05);
//              distLight = max(distLight, r + 0.05);
                shadowDir = normalize(shadowDir);
//              shadowDir = normalize(shadowDir);

                float cosTheta = max(0.0, dot(scatterResult.shadingNormal, shadowDir));
//              float cosTheta = max(0.0, dot(scatterResult.shadingNormal, shadowDir));
                vec3 lightNormal = normalize(sampledPoint - lightPos);
//              vec3 lightNormal = normalize(sampledPoint - lightPos);
                // Clamped cosThetaLight to 0.01 to prevent division-by-zero singularities
//              // Clamped cosThetaLight to 0.01 to prevent division-by-zero singularities
                float cosThetaLight = max(0.01, dot(lightNormal, -shadowDir));
//              float cosThetaLight = max(0.01, dot(lightNormal, -shadowDir));

                if (cosTheta > 0.0 && cosThetaLight > 0.0) {
//              if (cosTheta > 0.0 && cosThetaLight > 0.0) {
                    Ray shadowRay;
//                  Ray shadowRay;
                    shadowRay.origin = rayHitResult.at + rayHitResult.hittedSideNormal * EPSILON_OFFSET;
//                  shadowRay.origin = rayHitResult.at + rayHitResult.hittedSideNormal * EPSILON_OFFSET;
                    shadowRay.direction = shadowDir;
//                  shadowRay.direction = shadowDir;

                    Interval shadowInterval;
//                  Interval shadowInterval;
                    shadowInterval.min = EPSILON_OFFSET;
//                  shadowInterval.min = EPSILON_OFFSET;
                    shadowInterval.max = distLight - EPSILON_OFFSET;
//                  shadowInterval.max = distLight - EPSILON_OFFSET;

                    if (!traverseBVHAnyHit(shadowRay, shadowInterval)) {
//                  if (!traverseBVHAnyHit(shadowRay, shadowInterval)) {
                        float area = 4.0 * PI * r * r;
//                      float area = 4.0 * PI * r * r;
                        float pdfLightArea = lightPdf / area;
//                      float pdfLightArea = lightPdf / area;
                        float pdfLightW = (pdfLightArea * distLight * distLight) / cosThetaLight;
//                      float pdfLightW = (pdfLightArea * distLight * distLight) / cosThetaLight;

                        float pdfBrdfW = evalPrincipledPDF(currentRay.direction, shadowDir, scatterResult.shadingNormal, scatterResult.albedo, scatterResult.roughness, scatterResult.metallic);
//                      float pdfBrdfW = evalPrincipledPDF(currentRay.direction, shadowDir, scatterResult.shadingNormal, scatterResult.albedo, scatterResult.roughness, scatterResult.metallic);

                        // Replaced the Balance Heuristic with the robust Power Heuristic for calculating weightNee
//                      // Replaced the Balance Heuristic with the robust Power Heuristic for calculating weightNee
                        float weightNee = (pdfLightW * pdfLightW) / max(pdfLightW * pdfLightW + pdfBrdfW * pdfBrdfW, 1e-8);
//                      float weightNee = (pdfLightW * pdfLightW) / max(pdfLightW * pdfLightW + pdfBrdfW * pdfBrdfW, 1e-8);

                        vec3 bsdf = evalPrincipledBSDF(currentRay.direction, shadowDir, scatterResult.shadingNormal, scatterResult.albedo, scatterResult.roughness, scatterResult.metallic);
//                      vec3 bsdf = evalPrincipledBSDF(currentRay.direction, shadowDir, scatterResult.shadingNormal, scatterResult.albedo, scatterResult.roughness, scatterResult.metallic);

                        // Radiance emitted by the point light's surface
//                      // Radiance emitted by the point light's surface
                        vec3 directLight = uPointLights[lightIdx].color * bsdf * cosTheta / max(pdfLightW, NEE_PDF_EPSILON);
//                      vec3 directLight = uPointLights[lightIdx].color * bsdf * cosTheta / max(pdfLightW, NEE_PDF_EPSILON);
                        // Clamp direct light to prevent fireflies from extreme NEE contributions
//                      // Clamp direct light to prevent fireflies from extreme NEE contributions
                        directLight = min(directLight, vec3(NEE_DIRECT_LIGHT_CLAMP));
//                      directLight = min(directLight, vec3(NEE_DIRECT_LIGHT_CLAMP));
                        accumulatedColor += attenuation * directLight * weightNee;
//                      accumulatedColor += attenuation * directLight * weightNee;
                    }
//                  }
                }
//              }
            }
//          }

            if (!scatterResult.isScattered) {
//          if (!scatterResult.isScattered) {
                break;
//              break;
            }
//          }

            attenuation *= scatterResult.attenuation;
//          attenuation *= scatterResult.attenuation;

            // Russian Roulette: Stochastically terminate low-contribution paths to improve performance
//          // Russian Roulette: Stochastically terminate low-contribution paths to improve performance
            uint minBouncesForRR = RR_MIN_BOUNCES;
//          uint minBouncesForRR = RR_MIN_BOUNCES;
            if (depth >= minBouncesForRR) {
//          if (depth >= minBouncesForRR) {
                // Clamp probability to avoid extreme weights and premature termination
//              // Clamp probability to avoid extreme weights and premature termination
                float p = clamp(maxVec3(attenuation), RR_MIN_PROBABILITY, RR_MAX_PROBABILITY);
//              float p = clamp(maxVec3(attenuation), RR_MIN_PROBABILITY, RR_MAX_PROBABILITY);
                if (rand() > p) break;
//              if (rand() > p) break;
                attenuation *= 1.0 / p;
//              attenuation *= 1.0 / p;
            }
//          }

            currentRay = scatterResult.scatteredRay;
//          currentRay = scatterResult.scatteredRay;
            lastPdfW = scatterResult.pdf;
//          lastPdfW = scatterResult.pdf;
            lastWasDelta = scatterResult.isDelta;
//          lastWasDelta = scatterResult.isDelta;
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

        // Temporal Reprojection Accumulation: Blends the stochastically sampled noisy irradiance of the current frame into the persistent history buffer. This progressive refinement algorithm geometrically decreases variance over consecutive stationary frames to synthesize a pristine, converged image.
//      // Temporal Reprojection Accumulation: Blends the stochastically sampled noisy irradiance of the current frame into the persistent history buffer. This progressive refinement algorithm geometrically decreases variance over consecutive stationary frames to synthesize a pristine, converged image.
        vec4 prevAccum = imageLoad(textureAccum, pixelCoordinates);
//      vec4 prevAccum = imageLoad(textureAccum, pixelCoordinates);
        vec3 history = prevAccum.rgb;
//      vec3 history = prevAccum.rgb;
        float alpha = 1.0 / float(uFrameCount);
//      float alpha = 1.0 / float(uFrameCount);
        alpha = max(alpha, 0.01 /* 0.01 */);
//      alpha = max(alpha, 0.01 /* 0.01 */);
        if (uFrameCount == 1) {
//      if (uFrameCount == 1) {
             history = vec3(0.0);
//           history = vec3(0.0);
             alpha = 1.0;
//           alpha = 1.0;
        }
//      }
        vec3 newAverage = mix(history, accumulatedColor, alpha);
//      vec3 newAverage = mix(history, accumulatedColor, alpha);
        imageStore(textureAccum, pixelCoordinates, vec4(newAverage, 1.0));
//      imageStore(textureAccum, pixelCoordinates, vec4(newAverage, 1.0));

        // Note: Tone mapping and output moved to hybrid_denoise_cs.glsl
        // Note: Tone mapping and output moved to hybrid_denoise_cs.glsl
    }
//  }