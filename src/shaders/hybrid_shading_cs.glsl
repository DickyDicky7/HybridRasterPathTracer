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

    // BVH Nodes Buffer
    // BVH Nodes Buffer
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

    // Triangles Buffer (flat floats)
    // Triangles Buffer (flat floats)
    layout(std430, binding = 7) buffer SceneTriangles {
//  layout(std430, binding = 7) buffer SceneTriangles {
        float triangles[];
//      float triangles[];
    };
//  };

    // Materials Buffer
    // Materials Buffer
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
        float sampledRoughness;
//      float sampledRoughness;
        vec3 albedo;
//      vec3 albedo;
        float roughness;
//      float roughness;
        float metallic;
//      float metallic;
        vec3 shadingNormal;
//      vec3 shadingNormal;
    };
//  };

    uniform float uTime;
//  uniform float uTime;
    uniform vec3 uPointLight001GlobalPosition;
//  uniform vec3 uPointLight001GlobalPosition;
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

    // Random Number Generator
    // Random Number Generator
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
        seed = uint(pixel.x * 1973 + pixel.y * 9277 + uFrameCount * 26699) | 1u;
//      seed = uint(pixel.x * 1973 + pixel.y * 9277 + uFrameCount * 26699) | 1u;
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

    // Sky Color
    // Sky Color
    vec3 getSkyColor(vec3 dir, float roughness) {
//  vec3 getSkyColor(vec3 dir, float roughness) {
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

            float maxLod = 12.0;
//          float maxLod = 12.0;
            float lod = roughness * maxLod;
//          float lod = roughness * maxLod;

            vec3 hdriColor = textureLod(uHdriTexture, vec2(u, v), lod).rgb;
//          vec3 hdriColor = textureLod(uHdriTexture, vec2(u, v), lod).rgb;
            // Radiance clamping to reduce fireflies
            // Radiance clamping to reduce fireflies
            return min(hdriColor, vec3(10.0));
//          return min(hdriColor, vec3(10.0));
        }
//      }
        // Fallback: procedural gradient sky
        // Fallback: procedural gradient sky
        float t = 0.5 * (dir.y + 1.0);
//      float t = 0.5 * (dir.y + 1.0);
        return mix(vec3(0.1), vec3(0.5, 0.7, 1.0), t);
//      return mix(vec3(0.1), vec3(0.5, 0.7, 1.0), t);
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

    // Helper to get component
//  // Helper to get component
    float maxVec3(vec3 v) { return max(v.x, max(v.y, v.z)); }
//  float maxVec3(vec3 v) { return max(v.x, max(v.y, v.z)); }
    float minVec3(vec3 v) { return min(v.x, min(v.y, v.z)); }
//  float minVec3(vec3 v) { return min(v.x, min(v.y, v.z)); }

    // Intersect AABB
//  // Intersect AABB
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

    // Calculate Distance to AABB (for sorting)
//  // Calculate Distance to AABB (for sorting)
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
        return 9999999.0;
//      return 9999999.0;
    }
//  }

    // intersectTriangleAnyHit
//  // intersectTriangleAnyHit
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

        const float EPSILON = 1.0e-4;
//      const float EPSILON = 1.0e-4;

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

    // intersectTriangleClosestHit
//  // intersectTriangleClosestHit
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

        const float EPSILON = 1.0e-4;
//      const float EPSILON = 1.0e-4;
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

    // traverseBVHAnyHit
//  // traverseBVHAnyHit
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

            if (distL == 9999999.0 && distR == 9999999.0) continue;
//          if (distL == 9999999.0 && distR == 9999999.0) continue;

            if (distL < distR) {
//          if (distL < distR) {
                if (distR < 9999999.0) stack[stackPtr++] = childIndexR;
//              if (distR < 9999999.0) stack[stackPtr++] = childIndexR;
                if (distL < 9999999.0) stack[stackPtr++] = childIndexL;
//              if (distL < 9999999.0) stack[stackPtr++] = childIndexL;
            } else {
//          } else {
                if (distL < 9999999.0) stack[stackPtr++] = childIndexL;
//              if (distL < 9999999.0) stack[stackPtr++] = childIndexL;
                if (distR < 9999999.0) stack[stackPtr++] = childIndexR;
//              if (distR < 9999999.0) stack[stackPtr++] = childIndexR;
            }
//          }
        }
//      }
        return false;
//      return false;
    }
//  }

    // traverseBVHClosestHit
//  // traverseBVHClosestHit
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

            if (distL == 9999999.0 && distR == 9999999.0) continue;
//          if (distL == 9999999.0 && distR == 9999999.0) continue;

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

    // Principled BSDF Helpers
    // Principled BSDF Helpers
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
        float random2 = rand();
//      float random2 = rand();

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
        vec3 up = abs(normal.z) < 0.999 ? vec3(0.0, 0.0, 1.0) : vec3(1.0, 0.0, 0.0);
//      vec3 up = abs(normal.z) < 0.999 ? vec3(0.0, 0.0, 1.0) : vec3(1.0, 0.0, 0.0);
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

    // Evaluates the Principled BSDF for a given pair of directions
//  // Evaluates the Principled BSDF for a given pair of directions
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

        vec3 F0 = mix(vec3(0.04), albedo, metallic);
//      vec3 F0 = mix(vec3(0.04), albedo, metallic);
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
        vec3 diffuse = kD * albedo / PI;
//      vec3 diffuse = kD * albedo / PI;

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
        float G1 = NdotV / (NdotV * (1.0 - k) + k);
//      float G1 = NdotV / (NdotV * (1.0 - k) + k);
        float G2 = NdotL / (NdotL * (1.0 - k) + k);
//      float G2 = NdotL / (NdotL * (1.0 - k) + k);
        float G = G1 * G2;
//      float G = G1 * G2;

        vec3 specular = (D * F * G) / (4.0 * NdotV * NdotL + 0.001);
//      vec3 specular = (D * F * G) / (4.0 * NdotV * NdotL + 0.001);

        return diffuse + specular;
//      return diffuse + specular;
    }
//  }

    // Principled BSDF Scatter Function
//  // Principled BSDF Scatter Function
    MaterialLightScatteringResult scatterPrincipled(Ray incomingRay, RayHitResult recentRayHitResult, Material material) {
//  MaterialLightScatteringResult scatterPrincipled(Ray incomingRay, RayHitResult recentRayHitResult, Material material) {
        MaterialLightScatteringResult materialLightScatteringResult;
//      MaterialLightScatteringResult materialLightScatteringResult;
        materialLightScatteringResult.isScattered = false;
//      materialLightScatteringResult.isScattered = false;
        materialLightScatteringResult.attenuation = vec3(1.0);
//      materialLightScatteringResult.attenuation = vec3(1.0);
        materialLightScatteringResult.emission = vec3(0.0);
//      materialLightScatteringResult.emission = vec3(0.0);

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
        vec3 f0 = mix(vec3(0.04), albedo, metallic);
//      vec3 f0 = mix(vec3(0.04), albedo, metallic);

        // Schlick's fresnel approximation at incident angle
//      // Schlick's fresnel approximation at incident angle
        float cosTheta = min(dot(-incomingRay.direction, shadingNormal), 1.0);
//      float cosTheta = min(dot(-incomingRay.direction, shadingNormal), 1.0);
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
        float specularProbability = mix(fresnelProb, 1.0, metallic);
//      float specularProbability = mix(fresnelProb, 1.0, metallic);

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
                materialLightScatteringResult.attenuation = mix(vec3(1.0), albedo, metallic);
//              materialLightScatteringResult.attenuation = mix(vec3(1.0), albedo, metallic);
                materialLightScatteringResult.isScattered = true;
//              materialLightScatteringResult.isScattered = true;
                materialLightScatteringResult.sampledRoughness = roughness;
//              materialLightScatteringResult.sampledRoughness = roughness;
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
//           // --- PATH B: DIELECTRIC (DIFFUSE OR TRANSMISSION) ---

             // Re-normalize random variable for the next choice
//           // Re-normalize random variable for the next choice
             float randomNextChoice = (randomChoice - specularProbability) / (1.0 - specularProbability);
//           float randomNextChoice = (randomChoice - specularProbability) / (1.0 - specularProbability);

             if (material.transmission > 0.0 && randomNextChoice < material.transmission) {
//           if (material.transmission > 0.0 && randomNextChoice < material.transmission) {
                 // TRANSMISSION (REFRACTION)
//               // TRANSMISSION (REFRACTION)
                 float etaRatioOfIncidenceOverTransmission = 1.0 / material.ior;
//               float etaRatioOfIncidenceOverTransmission = 1.0 / material.ior;
                 if (!recentRayHitResult.isFrontFaceHitted) {
//               if (!recentRayHitResult.isFrontFaceHitted) {
                     etaRatioOfIncidenceOverTransmission = material.ior;
//                   etaRatioOfIncidenceOverTransmission = material.ior;
                 }
//               }

                 vec3 microfacetNormal = sampleGGX(shadingNormal, material.roughness);
//               vec3 microfacetNormal = sampleGGX(shadingNormal, material.roughness);

                 float cosThetaIncidence = min(dot(-incomingRay.direction, microfacetNormal), 1.0);
//               float cosThetaIncidence = min(dot(-incomingRay.direction, microfacetNormal), 1.0);
                 float sinThetaTransmission = (1.0 - cosThetaIncidence * cosThetaIncidence) * (etaRatioOfIncidenceOverTransmission * etaRatioOfIncidenceOverTransmission);
//               float sinThetaTransmission = (1.0 - cosThetaIncidence * cosThetaIncidence) * (etaRatioOfIncidenceOverTransmission * etaRatioOfIncidenceOverTransmission);

                 vec3 refractedDirection = refractPrincipled(incomingRay.direction, microfacetNormal, etaRatioOfIncidenceOverTransmission, cosThetaIncidence, sinThetaTransmission);
//               vec3 refractedDirection = refractPrincipled(incomingRay.direction, microfacetNormal, etaRatioOfIncidenceOverTransmission, cosThetaIncidence, sinThetaTransmission);
                 vec3 reflectedDirection = reflectPrincipled(incomingRay.direction, microfacetNormal);
//               vec3 reflectedDirection = reflectPrincipled(incomingRay.direction, microfacetNormal);

                 // When [ sinThetaTransmission <= 1.0 ] then Refraction happened else Total Internal Reflection happened
//               // When [ sinThetaTransmission <= 1.0 ] then Refraction happened else Total Internal Reflection happened
                 materialLightScatteringResult.scatteredRay.origin = recentRayHitResult.at;
//               materialLightScatteringResult.scatteredRay.origin = recentRayHitResult.at;

                 if (sinThetaTransmission <= 1.0) {
//               if (sinThetaTransmission <= 1.0) {
                     materialLightScatteringResult.scatteredRay.direction = refractedDirection;
//                   materialLightScatteringResult.scatteredRay.direction = refractedDirection;
                     materialLightScatteringResult.attenuation = albedo;
//                   materialLightScatteringResult.attenuation = albedo;
                 } else {
//               } else {
                     materialLightScatteringResult.scatteredRay.direction = reflectedDirection;
//                   materialLightScatteringResult.scatteredRay.direction = reflectedDirection;
                     materialLightScatteringResult.attenuation = vec3(1.0);
//                   materialLightScatteringResult.attenuation = vec3(1.0);
                 }
//               }
                 materialLightScatteringResult.isScattered = true;
//               materialLightScatteringResult.isScattered = true;
                 materialLightScatteringResult.sampledRoughness = material.roughness;
//               materialLightScatteringResult.sampledRoughness = material.roughness;
             } else {
//           } else {
                 // DIFFUSE (LAMBERTIAN)
//               // DIFFUSE (LAMBERTIAN)
                 vec3 diffuseDirection = normalize(shadingNormal + randomUnitVector());
//               vec3 diffuseDirection = normalize(shadingNormal + randomUnitVector());

                 materialLightScatteringResult.scatteredRay.origin = recentRayHitResult.at;
//               materialLightScatteringResult.scatteredRay.origin = recentRayHitResult.at;
                 materialLightScatteringResult.scatteredRay.direction = diffuseDirection;
//               materialLightScatteringResult.scatteredRay.direction = diffuseDirection;
                 materialLightScatteringResult.attenuation = albedo;
//               materialLightScatteringResult.attenuation = albedo;
                 materialLightScatteringResult.isScattered = true;
//               materialLightScatteringResult.isScattered = true;
                 materialLightScatteringResult.sampledRoughness = 1.0;
//               materialLightScatteringResult.sampledRoughness = 1.0;
             }
//           }
        }
//      }
        return materialLightScatteringResult;
//      return materialLightScatteringResult;
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

        // Calculate Primary Ray
//      // Calculate Primary Ray
        vec3 pixelSampleCenter = uPixel00Coordinates + uPixelDeltaU * (float(pixelCoordinates.x) + uJitter.x) + uPixelDeltaV * (float(pixelCoordinates.y) + uJitter.y);
//      vec3 pixelSampleCenter = uPixel00Coordinates + uPixelDeltaU * (float(pixelCoordinates.x) + uJitter.x) + uPixelDeltaV * (float(pixelCoordinates.y) + uJitter.y);

        Ray currentRay;
//      Ray currentRay;
        currentRay.origin = uCameraGlobalPosition;
//      currentRay.origin = uCameraGlobalPosition;
        currentRay.direction = normalize(pixelSampleCenter - uCameraGlobalPosition);
//      currentRay.direction = normalize(pixelSampleCenter - uCameraGlobalPosition);

        int maxDepth = 4;
//      int maxDepth = 4;

        for (int depth = 0; depth < maxDepth; depth++) {
//      for (int depth = 0; depth < maxDepth; depth++) {
            RayHitResult rayHitResult;
//          RayHitResult rayHitResult;

            if (depth == 0) {
//          if (depth == 0) {
                // Read from G-Buffer
//              // Read from G-Buffer
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
                hitInterval.min = 0.001;
//              hitInterval.min = 0.001;
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

            if (!rayHitResult.isHitted) {
//          if (!rayHitResult.isHitted) {
                // Sky
//              // Sky
                vec3 sky = getSkyColor(currentRay.direction, 0.0);
//              vec3 sky = getSkyColor(currentRay.direction, 0.0);
                accumulatedColor += attenuation * sky;
//              accumulatedColor += attenuation * sky;
                break;
//              break;
            }
//          }

            Material material = materials[rayHitResult.materialIndex];
//          Material material = materials[rayHitResult.materialIndex];

            MaterialLightScatteringResult scatterResult = scatterPrincipled(currentRay, rayHitResult, material);
//          MaterialLightScatteringResult scatterResult = scatterPrincipled(currentRay, rayHitResult, material);

            accumulatedColor += attenuation * scatterResult.emission;
//          accumulatedColor += attenuation * scatterResult.emission;

            // NEE
//          // NEE
            vec3 sunColor = vec3(300.0);
//          vec3 sunColor = vec3(300.0);
            float sunRadius = 0.05;
//          float sunRadius = 0.05;
            vec3 jitteredSunPos = uPointLight001GlobalPosition + randomUnitVector() * sunRadius;
//          vec3 jitteredSunPos = uPointLight001GlobalPosition + randomUnitVector() * sunRadius;
            vec3 jitteredSunDir = normalize(jitteredSunPos - rayHitResult.at);
//          vec3 jitteredSunDir = normalize(jitteredSunPos - rayHitResult.at);
            float distLight = length(jitteredSunPos - rayHitResult.at);
//          float distLight = length(jitteredSunPos - rayHitResult.at);

            Ray shadowRay;
//          Ray shadowRay;
            shadowRay.origin = rayHitResult.at + rayHitResult.hittedSideNormal * 0.001;
//          shadowRay.origin = rayHitResult.at + rayHitResult.hittedSideNormal * 0.001;
            shadowRay.direction = jitteredSunDir;
//          shadowRay.direction = jitteredSunDir;

            Interval shadowInterval;
//          Interval shadowInterval;
            shadowInterval.min = 0.001;
//          shadowInterval.min = 0.001;
            shadowInterval.max = distLight;
//          shadowInterval.max = distLight;

            if (!traverseBVHAnyHit(shadowRay, shadowInterval)) {
//          if (!traverseBVHAnyHit(shadowRay, shadowInterval)) {
                vec3 N = scatterResult.shadingNormal;
//              vec3 N = scatterResult.shadingNormal;
                float cosTheta = max(0.0, dot(N, jitteredSunDir));
//              float cosTheta = max(0.0, dot(N, jitteredSunDir));

                vec3 bsdf = evalPrincipledBSDF(currentRay.direction, jitteredSunDir, N, scatterResult.albedo, scatterResult.roughness, scatterResult.metallic);
//              vec3 bsdf = evalPrincipledBSDF(currentRay.direction, jitteredSunDir, N, scatterResult.albedo, scatterResult.roughness, scatterResult.metallic);

                float distSq = max(distLight * distLight, 0.1);
//              float distSq = max(distLight * distLight, 0.1);
                float falloff = 1.0 / distSq;
//              float falloff = 1.0 / distSq;
                vec3 directLight = bsdf * sunColor * cosTheta * falloff;
//              vec3 directLight = bsdf * sunColor * cosTheta * falloff;
                accumulatedColor += attenuation * directLight;
//              accumulatedColor += attenuation * directLight;
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

            // RR
//          // RR
            uint minBouncesForRR = 4u;
//          uint minBouncesForRR = 4u;
            if (depth >= minBouncesForRR) {
//          if (depth >= minBouncesForRR) {
                float p = maxVec3(attenuation);
//              float p = maxVec3(attenuation);
                if (rand() > p) break;
//              if (rand() > p) break;
                attenuation *= 1.0 / p;
//              attenuation *= 1.0 / p;
            }
//          }

            currentRay = scatterResult.scatteredRay;
//          currentRay = scatterResult.scatteredRay;
        }
//      }

        // Accumulation
//      // Accumulation
        vec4 prevAccum = imageLoad(textureAccum, pixelCoordinates);
//      vec4 prevAccum = imageLoad(textureAccum, pixelCoordinates);
        vec3 history = prevAccum.rgb;
//      vec3 history = prevAccum.rgb;
        float alpha = 1.0 / float(uFrameCount);
//      float alpha = 1.0 / float(uFrameCount);
        alpha = max(alpha, 0.1 /* 0.01 */);
//      alpha = max(alpha, 0.1 /* 0.01 */);
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