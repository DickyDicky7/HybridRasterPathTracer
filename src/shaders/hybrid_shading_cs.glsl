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
    layout(binding = 6, rgba32f) uniform image2D textureAccum;
//  layout(binding = 6, rgba32f) uniform image2D textureAccum;

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

    layout(std430, binding = 4) buffer BVHNodes {
//  layout(std430, binding = 4) buffer BVHNodes {
        Node nodes[];
//      Node nodes[];
    };
//  };

    // Triangles Buffer (flat floats)
    // Triangles Buffer (flat floats)
    layout(std430, binding = 5) buffer SceneTriangles {
//  layout(std430, binding = 5) buffer SceneTriangles {
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
    };
//  };

    layout(std430, binding = 7) buffer SceneMaterials {
//  layout(std430, binding = 7) buffer SceneMaterials {
        Material materials[];
//      Material materials[];
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
            result.uvSurfaceCoordinate = vec2(u, v); // Barycentrics
//          result.uvSurfaceCoordinate = vec2(u, v); // Barycentrics
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
        
        vec3 albedo = material.albedo.rgb;
//      vec3 albedo = material.albedo.rgb;

        // Principled Logic
//      // Principled Logic

        // F0 calculation: 0.04 for dielectrics, albedo for metals
//      // F0 calculation: 0.04 for dielectrics, albedo for metals
        vec3 f0 = mix(vec3(0.04), albedo, material.metallic);
//      vec3 f0 = mix(vec3(0.04), albedo, material.metallic);

        // Schlick's fresnel approximation at incident angle
//      // Schlick's fresnel approximation at incident angle
        float cosTheta = min(dot(-incomingRay.direction, recentRayHitResult.hittedSideNormal), 1.0);
//      float cosTheta = min(dot(-incomingRay.direction, recentRayHitResult.hittedSideNormal), 1.0);
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
        float specularProbability = mix(fresnelProb, 1.0, material.metallic);
//      float specularProbability = mix(fresnelProb, 1.0, material.metallic);

        if (randomChoice < specularProbability) {
//      if (randomChoice < specularProbability) {
             // SPECULAR REFLECTION (METAL OR DIELECTRIC COAT)
//           // SPECULAR REFLECTION (METAL OR DIELECTRIC COAT)
             vec3 microfacetNormal = sampleGGX(recentRayHitResult.hittedSideNormal, material.roughness);
//           vec3 microfacetNormal = sampleGGX(recentRayHitResult.hittedSideNormal, material.roughness);
             vec3 specularReflectedDirection = reflectPrincipled(incomingRay.direction, microfacetNormal);
//           vec3 specularReflectedDirection = reflectPrincipled(incomingRay.direction, microfacetNormal);

             if (dot(specularReflectedDirection, recentRayHitResult.hittedSideNormal) > 0.0) {
//           if (dot(specularReflectedDirection, recentRayHitResult.hittedSideNormal) > 0.0) {
                 materialLightScatteringResult.scatteredRay.origin = recentRayHitResult.at;
//               materialLightScatteringResult.scatteredRay.origin = recentRayHitResult.at;
                 materialLightScatteringResult.scatteredRay.direction = specularReflectedDirection;
//               materialLightScatteringResult.scatteredRay.direction = specularReflectedDirection;
                 materialLightScatteringResult.attenuation = mix(vec3(1.0), albedo, material.metallic);
//               materialLightScatteringResult.attenuation = mix(vec3(1.0), albedo, material.metallic);
                 materialLightScatteringResult.isScattered = true;
//               materialLightScatteringResult.isScattered = true;
                 materialLightScatteringResult.sampledRoughness = material.roughness;
//               materialLightScatteringResult.sampledRoughness = material.roughness;
             } else {
//           } else {
                 // Current/Recent ray is/was absorbed (next ray is scattering into surface)
//               // Current/Recent ray is/was absorbed (next ray is scattering into surface)
                 materialLightScatteringResult.isScattered = false;
//               materialLightScatteringResult.isScattered = false;
             }
//           }
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

                 vec3 microfacetNormal = sampleGGX(recentRayHitResult.hittedSideNormal, material.roughness);
//               vec3 microfacetNormal = sampleGGX(recentRayHitResult.hittedSideNormal, material.roughness);

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
                 vec3 diffuseDirection = normalize(recentRayHitResult.hittedSideNormal + randomUnitVector());
//               vec3 diffuseDirection = normalize(recentRayHitResult.hittedSideNormal + randomUnitVector());

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

        // Read First hit from G-Buffer (Depth 0)
//      // Read First hit from G-Buffer (Depth 0)
        vec4 sampleGlobalPosition = imageLoad(textureGeometryGlobalPosition, pixelCoordinates);
//      vec4 sampleGlobalPosition = imageLoad(textureGeometryGlobalPosition, pixelCoordinates);
        vec4 sampleGlobalNormal = imageLoad(textureGeometryGlobalNormal, pixelCoordinates);
//      vec4 sampleGlobalNormal = imageLoad(textureGeometryGlobalNormal, pixelCoordinates);

        // State variables
//      // State variables
        vec3 accumulatedColor = vec3(0.0);
//      vec3 accumulatedColor = vec3(0.0);
        vec3 attenuation = vec3(1.0);
//      vec3 attenuation = vec3(1.0);
        vec3 rayOrigin;
//      vec3 rayOrigin;
        vec3 rayDir;
//      vec3 rayDir;
        float currentRayRoughness = 0.0;
//      float currentRayRoughness = 0.0;

        // Incoming direction (for BSDF)
//      // Incoming direction (for BSDF)
        vec3 V;
//      vec3 V;

        bool hitSky = false;
//      bool hitSky = false;

        // Depth 0: G-Buffer Data
//      // Depth 0: G-Buffer Data
        if (sampleGlobalPosition.w == 0.0) {
//      if (sampleGlobalPosition.w == 0.0) {
            // Missed geometry in raster pass -> Sky
//          // Missed geometry in raster pass -> Sky
            // Manual Ray Construction (Path Tracing Pinhole Camera)
//          // Manual Ray Construction (Path Tracing Pinhole Camera)

            // Note: uPixel00Coordinates is the center of the top-left pixel (0,0).
//          // Note: uPixel00Coordinates is the center of the top-left pixel (0,0).
            // We use uJitter to apply TAA/sub-pixel jitter.
//          // We use uJitter to apply TAA/sub-pixel jitter.
            vec3 pixelSampleCenter = uPixel00Coordinates + uPixelDeltaU * (float(pixelCoordinates.x) + uJitter.x) + uPixelDeltaV * (float(pixelCoordinates.y) + uJitter.y);
//          vec3 pixelSampleCenter = uPixel00Coordinates + uPixelDeltaU * (float(pixelCoordinates.x) + uJitter.x) + uPixelDeltaV * (float(pixelCoordinates.y) + uJitter.y);

            rayOrigin = uCameraGlobalPosition;
//          rayOrigin = uCameraGlobalPosition;
            rayDir = normalize(pixelSampleCenter - rayOrigin);
//          rayDir = normalize(pixelSampleCenter - rayOrigin);

            accumulatedColor = getSkyColor(rayDir, 0.0);
//          accumulatedColor = getSkyColor(rayDir, 0.0);
            hitSky = true;
//          hitSky = true;
        } else {
//      } else {
            // Hit geometry
//          // Hit geometry
            rayOrigin = sampleGlobalPosition.xyz;
//          rayOrigin = sampleGlobalPosition.xyz;
            vec3 N = normalize(sampleGlobalNormal.xyz);
//          vec3 N = normalize(sampleGlobalNormal.xyz);

            // Recover Triangle Index
//          // Recover Triangle Index
            int triIdx = int(sampleGlobalPosition.w) - 1;
//          int triIdx = int(sampleGlobalPosition.w) - 1;
            Material mat = materials[triIdx];
//          Material mat = materials[triIdx];

            V = normalize(rayOrigin - uCameraGlobalPosition); // Incoming ray direction
//          V = normalize(rayOrigin - uCameraGlobalPosition); // Incoming ray direction

            // Apply Emission (If any - currently none, assuming objects are not emissive)
//          // Apply Emission (If any - currently none, assuming objects are not emissive)
            // accumulatedColor += attenuation * emission;
//          // accumulatedColor += attenuation * emission;

            // --- Multiple Importance Sampling: Next Event Estimation (Direct Light) ---
//          // --- Multiple Importance Sampling: Next Event Estimation (Direct Light) ---

            // Soft Shadow (Area Light approximation)
//          // Soft Shadow (Area Light approximation)
            vec3 sunDirectionCenter = normalize(uPointLight001GlobalPosition - rayOrigin);
//          vec3 sunDirectionCenter = normalize(uPointLight001GlobalPosition - rayOrigin);
            vec3 sunColor = vec3(300.0);
//          vec3 sunColor = vec3(300.0);
            float distToLight = length(uPointLight001GlobalPosition - rayOrigin);
//          float distToLight = length(uPointLight001GlobalPosition - rayOrigin);
            float sunRadius = 0.05;
//          float sunRadius = 0.05;

            vec3 jitteredSunPos = uPointLight001GlobalPosition + randomUnitVector() * sunRadius;
//          vec3 jitteredSunPos = uPointLight001GlobalPosition + randomUnitVector() * sunRadius;
            vec3 jitteredSunDir = normalize(jitteredSunPos - rayOrigin);
//          vec3 jitteredSunDir = normalize(jitteredSunPos - rayOrigin);
            float jitteredDist = length(jitteredSunPos - rayOrigin);
//          float jitteredDist = length(jitteredSunPos - rayOrigin);

            vec3 shadowRayOrigin = rayOrigin + N * 0.001;
//          vec3 shadowRayOrigin = rayOrigin + N * 0.001;

            Ray shadowRay;
//          Ray shadowRay;
            shadowRay.origin = shadowRayOrigin;
//          shadowRay.origin = shadowRayOrigin;
            shadowRay.direction = jitteredSunDir;
//          shadowRay.direction = jitteredSunDir;
            
            Interval shadowInterval;
//          Interval shadowInterval;
            shadowInterval.min = 0.001;
//          shadowInterval.min = 0.001;
            shadowInterval.max = jitteredDist;
//          shadowInterval.max = jitteredDist;

            if (!traverseBVHAnyHit(shadowRay, shadowInterval)) {
//          if (!traverseBVHAnyHit(shadowRay, shadowInterval)) {
                float cosTheta = max(0.0, dot(N, jitteredSunDir));
//              float cosTheta = max(0.0, dot(N, jitteredSunDir));
                // Inverse square falloff for point light
//              // Inverse square falloff for point light
                float falloff = 1.0 / (jitteredDist * jitteredDist);
//              float falloff = 1.0 / (jitteredDist * jitteredDist);
                // Approximate NEE using albedo (Diffuse assumption for now)
//              // Approximate NEE using albedo (Diffuse assumption for now)
                vec3 directLight = mat.albedo.rgb * (1.0 - mat.metallic) * sunColor * cosTheta * falloff;
//              vec3 directLight = mat.albedo.rgb * (1.0 - mat.metallic) * sunColor * cosTheta * falloff;
                accumulatedColor += attenuation * directLight;
//              accumulatedColor += attenuation * directLight;
            }
//          }

            // --- Principled Scatter ---
//          // --- Principled Scatter ---
            Ray incomingRay;
//          Ray incomingRay;
            incomingRay.origin = uCameraGlobalPosition;
//          incomingRay.origin = uCameraGlobalPosition;
            incomingRay.direction = V;
//          incomingRay.direction = V;

            RayHitResult hitRes;
//          RayHitResult hitRes;
            hitRes.at = rayOrigin;
//          hitRes.at = rayOrigin;
            hitRes.hittedSideNormal = N;
//          hitRes.hittedSideNormal = N;
            hitRes.isFrontFaceHitted = dot(V, N) < 0.0;
//          hitRes.isFrontFaceHitted = dot(V, N) < 0.0;
            hitRes.materialIndex = triIdx;
//          hitRes.materialIndex = triIdx;
            hitRes.triangleIndex = triIdx; // Assuming this is correct
//          hitRes.triangleIndex = triIdx; // Assuming this is correct

            MaterialLightScatteringResult res = scatterPrincipled(incomingRay, hitRes, mat);
//          MaterialLightScatteringResult res = scatterPrincipled(incomingRay, hitRes, mat);

            currentRayRoughness = max(currentRayRoughness, res.sampledRoughness);
//          currentRayRoughness = max(currentRayRoughness, res.sampledRoughness);

            if (res.isScattered) {
//          if (res.isScattered) {
                attenuation *= res.attenuation;
//              attenuation *= res.attenuation;
                rayOrigin = res.scatteredRay.origin;
//              rayOrigin = res.scatteredRay.origin;
                rayDir = res.scatteredRay.direction;
//              rayDir = res.scatteredRay.direction;
            } else {
//          } else {
                hitSky = true;
//              hitSky = true;
            }
//          }
        }
//      }

        if (!hitSky) {
//      if (!hitSky) {
            // Path Trace loop for bounces 1..Max
//          // Path Trace loop for bounces 1..Max
            int maxDepth = 3;
//          int maxDepth = 3;
            for (int depth = 1; depth < maxDepth; depth++) {
//          for (int depth = 1; depth < maxDepth; depth++) {
                
                Interval hitInterval;
//              Interval hitInterval;
                hitInterval.min = 0.001;
//              hitInterval.min = 0.001;
                hitInterval.max = INF;
//              hitInterval.max = INF;
                
                Ray ray;
//              Ray ray;
                ray.origin = rayOrigin;
//              ray.origin = rayOrigin;
                ray.direction = rayDir;
//              ray.direction = rayDir;

                RayHitResult hit = traverseBVHClosestHit(ray, hitInterval);
//              RayHitResult hit = traverseBVHClosestHit(ray, hitInterval);

                if (hit.isHitted) {
//              if (hit.isHitted) {
                    // Hit surface
//                  // Hit surface
                    V = rayDir; // Incoming direction for next bounce
//                  V = rayDir; // Incoming direction for next bounce
                    vec3 prevOrigin = rayOrigin;
//                  vec3 prevOrigin = rayOrigin;
                    rayOrigin = rayOrigin + rayDir * hit.minDistance;
//                  rayOrigin = rayOrigin + rayDir * hit.minDistance;

                    // Reconstruct Normal
//                  // Reconstruct Normal
                    int offset = hit.triangleIndex * 9;
//                  int offset = hit.triangleIndex * 9;
                    vec3 v0 = vec3(triangles[offset], triangles[offset+1], triangles[offset+2]);
//                  vec3 v0 = vec3(triangles[offset], triangles[offset+1], triangles[offset+2]);
                    vec3 v1 = vec3(triangles[offset+3], triangles[offset+4], triangles[offset+5]);
//                  vec3 v1 = vec3(triangles[offset+3], triangles[offset+4], triangles[offset+5]);
                    vec3 v2 = vec3(triangles[offset+6], triangles[offset+7], triangles[offset+8]);
//                  vec3 v2 = vec3(triangles[offset+6], triangles[offset+7], triangles[offset+8]);
                    vec3 hitN = normalize(cross(v1 - v0, v2 - v0));
//                  vec3 hitN = normalize(cross(v1 - v0, v2 - v0));
                    bool frontFace = dot(V, hitN) < 0.0;
//                  bool frontFace = dot(V, hitN) < 0.0;
                    if (!frontFace) hitN = -hitN;
//                  if (!frontFace) hitN = -hitN;

                    Material mat = materials[hit.triangleIndex];
//                  Material mat = materials[hit.triangleIndex];

                    // Direct Light (NEE)
//                  // Direct Light (NEE)
                    // Soft Shadow (Area Light approximation)
//                  // Soft Shadow (Area Light approximation)
                    vec3 sunDirectionCenter = normalize(uPointLight001GlobalPosition - rayOrigin);
//                  vec3 sunDirectionCenter = normalize(uPointLight001GlobalPosition - rayOrigin);
                    vec3 sunColor = vec3(300.0);
//                  vec3 sunColor = vec3(300.0);
                    float sunRadius = 0.05;
//                  float sunRadius = 0.05;
                    vec3 jitteredSunPos = uPointLight001GlobalPosition + randomUnitVector() * sunRadius;
//                  vec3 jitteredSunPos = uPointLight001GlobalPosition + randomUnitVector() * sunRadius;
                    vec3 jitteredSunDir = normalize(jitteredSunPos - rayOrigin);
//                  vec3 jitteredSunDir = normalize(jitteredSunPos - rayOrigin);
                    float jitteredDist = length(jitteredSunPos - rayOrigin);
//                  float jitteredDist = length(jitteredSunPos - rayOrigin);

                    vec3 shadowRayOrigin = rayOrigin + hitN * 0.001; // Offset outward for shadow
//                  vec3 shadowRayOrigin = rayOrigin + hitN * 0.001; // Offset outward for shadow
                    
                    Ray shadowRay;
//                  Ray shadowRay;
                    shadowRay.origin = shadowRayOrigin;
//                  shadowRay.origin = shadowRayOrigin;
                    shadowRay.direction = jitteredSunDir;
//                  shadowRay.direction = jitteredSunDir;
                    
                    Interval shadowInterval;
//                  Interval shadowInterval;
                    shadowInterval.min = 0.001;
//                  shadowInterval.min = 0.001;
                    shadowInterval.max = jitteredDist;
//                  shadowInterval.max = jitteredDist;

                    if (!traverseBVHAnyHit(shadowRay, shadowInterval)) {
//                  if (!traverseBVHAnyHit(shadowRay, shadowInterval)) {
                        float cosTheta = max(0.0, dot(hitN, jitteredSunDir));
//                      float cosTheta = max(0.0, dot(hitN, jitteredSunDir));
                        // Inverse square falloff for point light
//                      // Inverse square falloff for point light
                        float falloff = 1.0 / (jitteredDist * jitteredDist);
//                      float falloff = 1.0 / (jitteredDist * jitteredDist);
                        vec3 directLight = mat.albedo.rgb * (1.0 - mat.metallic) * sunColor * cosTheta * falloff;
//                      vec3 directLight = mat.albedo.rgb * (1.0 - mat.metallic) * sunColor * cosTheta * falloff;
                        accumulatedColor += attenuation * directLight;
//                      accumulatedColor += attenuation * directLight;
                    }
//                  }

                    // --- Principled Scatter ---
//                  // --- Principled Scatter ---
                    Ray incomingRay;
//                  Ray incomingRay;
                    incomingRay.origin = prevOrigin;
//                  incomingRay.origin = prevOrigin;
                    incomingRay.direction = V;
//                  incomingRay.direction = V;

                    RayHitResult hitRes;
//                  RayHitResult hitRes;
                    hitRes.at = rayOrigin;
//                  hitRes.at = rayOrigin;
                    hitRes.hittedSideNormal = hitN;
//                  hitRes.hittedSideNormal = hitN;
                    hitRes.isFrontFaceHitted = frontFace;
//                  hitRes.isFrontFaceHitted = frontFace;
                    hitRes.materialIndex = hit.triangleIndex;
//                  hitRes.materialIndex = hit.triangleIndex;
                    hitRes.minDistance = hit.minDistance;
//                  hitRes.minDistance = hit.minDistance;
                    hitRes.isHitted = true;
//                  hitRes.isHitted = true;
                    hitRes.triangleIndex = hit.triangleIndex;
//                  hitRes.triangleIndex = hit.triangleIndex;
                    hitRes.uvSurfaceCoordinate = hit.uvSurfaceCoordinate;
//                  hitRes.uvSurfaceCoordinate = hit.uvSurfaceCoordinate;

                    MaterialLightScatteringResult res = scatterPrincipled(incomingRay, hitRes, mat);
//                  MaterialLightScatteringResult res = scatterPrincipled(incomingRay, hitRes, mat);

                    currentRayRoughness = max(currentRayRoughness, res.sampledRoughness);
//                  currentRayRoughness = max(currentRayRoughness, res.sampledRoughness);

                    if (res.isScattered) {
//                  if (res.isScattered) {
                        attenuation *= res.attenuation;
//                      attenuation *= res.attenuation;
                        rayOrigin = res.scatteredRay.origin;
//                      rayOrigin = res.scatteredRay.origin;
                        rayDir = res.scatteredRay.direction;
//                      rayDir = res.scatteredRay.direction;
                    } else {
//                  } else {
                        break;
//                      break;
                    }
//                  }

                    // Russian Roulette
//                  // Russian Roulette
                    if (depth >= 2) {
//                  if (depth >= 2) {
                        float pr = max(attenuation.r, max(attenuation.g, attenuation.b));
//                      float pr = max(attenuation.r, max(attenuation.g, attenuation.b));
                        if (rand() > pr) break;
//                      if (rand() > pr) break;
                        attenuation *= 1.0 / pr;
//                      attenuation *= 1.0 / pr;
                    }
//                  }

                } else {
//              } else {
                    // Hit Sky
//                  // Hit Sky
                    vec3 sky = getSkyColor(rayDir, currentRayRoughness);
//                  vec3 sky = getSkyColor(rayDir, currentRayRoughness);
                    accumulatedColor += attenuation * sky;
//                  accumulatedColor += attenuation * sky;
                    break;
//                  break;
                }
//              }
            }
//          }
        }
//      }

        // Accumulation
//      // Accumulation
        vec4 prevAccum = imageLoad(textureAccum, pixelCoordinates);
//      vec4 prevAccum = imageLoad(textureAccum, pixelCoordinates);

        vec3 history = prevAccum.rgb;
//      vec3 history = prevAccum.rgb;

        // Temporal Accumulation (Exponential Moving Average)
//      // Temporal Accumulation (Exponential Moving Average)
        // Alpha determines how much the current frame contributes.
//      // Alpha determines how much the current frame contributes.
        // Start with 1.0 (overwrite) and decay to a minimum (e.g., 0.1) for dynamic stability.
//      // Start with 1.0 (overwrite) and decay to a minimum (e.g., 0.1) for dynamic stability.
        float alpha = 1.0 / float(uFrameCount);
//      float alpha = 1.0 / float(uFrameCount);
        alpha = max(alpha, 0.1);
//      alpha = max(alpha, 0.1);

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