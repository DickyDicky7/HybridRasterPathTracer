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

    uniform float uTime;
//  uniform float uTime;
    uniform vec3 uPointLight001GlobalPosition;
//  uniform vec3 uPointLight001GlobalPosition;
    uniform vec3 uCameraGlobalPosition;
//  uniform vec3 uCameraGlobalPosition;
    uniform int uFrameCount;
//  uniform int uFrameCount;

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
    vec3 getSkyColor(vec3 dir) {
//  vec3 getSkyColor(vec3 dir) {
        float t = 0.5 * (dir.y + 1.0);
//      float t = 0.5 * (dir.y + 1.0);
        return mix(vec3(0.1), vec3(0.5, 0.7, 1.0), t);
//      return mix(vec3(0.1), vec3(0.5, 0.7, 1.0), t);
    }
//  }

    // Ray-AABB Intersection
    // Ray-AABB Intersection
    bool intersectAABB(vec3 origin, vec3 invDir, vec3 boxMin, vec3 boxMax, float tMax) {
//  bool intersectAABB(vec3 origin, vec3 invDir, vec3 boxMin, vec3 boxMax, float tMax) {
        vec3 t1 = (boxMin - origin) * invDir;
//      vec3 t1 = (boxMin - origin) * invDir;
        vec3 t2 = (boxMax - origin) * invDir;
//      vec3 t2 = (boxMax - origin) * invDir;
        vec3 tMin = min(t1, t2);
//      vec3 tMin = min(t1, t2);
        vec3 tMaxVec = max(t1, t2);
//      vec3 tMaxVec = max(t1, t2);
        float tEnter = max(max(tMin.x, tMin.y), tMin.z);
//      float tEnter = max(max(tMin.x, tMin.y), tMin.z);
        float tExit = min(min(tMaxVec.x, tMaxVec.y), tMaxVec.z);
//      float tExit = min(min(tMaxVec.x, tMaxVec.y), tMaxVec.z);
        return tEnter <= tExit && tEnter < tMax && tExit > 0.0;
//      return tEnter <= tExit && tEnter < tMax && tExit > 0.0;
    }
//  }

    struct Hit {
//  struct Hit {
        float t;
//      float t;
        float u;
//      float u;
        float v;
//      float v;
        int triIdx;
//      int triIdx;
    };
//  };

    // Ray-Triangle Intersection (Moller-Trumbore) - Closest Hit
    // Ray-Triangle Intersection (Moller-Trumbore) - Closest Hit
    bool intersectTriangleClosest(vec3 origin, vec3 dir, int triIdx, inout Hit closestHit) {
//  bool intersectTriangleClosest(vec3 origin, vec3 dir, int triIdx, inout Hit closestHit) {
        int offset = triIdx * 9;
//      int offset = triIdx * 9;
        vec3 v0 = vec3(triangles[offset + 0], triangles[offset + 1], triangles[offset + 2]);
//      vec3 v0 = vec3(triangles[offset + 0], triangles[offset + 1], triangles[offset + 2]);
        vec3 v1 = vec3(triangles[offset + 3], triangles[offset + 4], triangles[offset + 5]);
//      vec3 v1 = vec3(triangles[offset + 3], triangles[offset + 4], triangles[offset + 5]);
        vec3 v2 = vec3(triangles[offset + 6], triangles[offset + 7], triangles[offset + 8]);
//      vec3 v2 = vec3(triangles[offset + 6], triangles[offset + 7], triangles[offset + 8]);

        vec3 e1 = v1 - v0;
//      vec3 e1 = v1 - v0;
        vec3 e2 = v2 - v0;
//      vec3 e2 = v2 - v0;
        vec3 h = cross(dir, e2);
//      vec3 h = cross(dir, e2);
        float a = dot(e1, h);
//      float a = dot(e1, h);

        if (a > -0.00001 && a < 0.00001) return false;
//      if (a > -0.00001 && a < 0.00001) return false;

        float f = 1.0 / a;
//      float f = 1.0 / a;
        vec3 s = origin - v0;
//      vec3 s = origin - v0;
        float u = f * dot(s, h);
//      float u = f * dot(s, h);
        if (u < 0.0 || u > 1.0) return false;
//      if (u < 0.0 || u > 1.0) return false;

        vec3 q = cross(s, e1);
//      vec3 q = cross(s, e1);
        float v = f * dot(dir, q);
//      float v = f * dot(dir, q);
        if (v < 0.0 || u + v > 1.0) return false;
//      if (v < 0.0 || u + v > 1.0) return false;

        float t = f * dot(e2, q);
//      float t = f * dot(e2, q);

        if (t > 0.001 && t < closestHit.t) {
//      if (t > 0.001 && t < closestHit.t) {
            closestHit.t = t;
//          closestHit.t = t;
            closestHit.u = u;
//          closestHit.u = u;
            closestHit.v = v;
//          closestHit.v = v;
            closestHit.triIdx = triIdx;
//          closestHit.triIdx = triIdx;
            return true;
//          return true;
        }
//      }
        return false;
//      return false;
    }
//  }

    // Shadow Ray Traversal (Any Hit)
    // Shadow Ray Traversal (Any Hit)
    bool traverseBVHShadow(vec3 origin, vec3 dir, float tMax) {
//  bool traverseBVHShadow(vec3 origin, vec3 dir, float tMax) {
        vec3 invDir = 1.0 / dir;
//      vec3 invDir = 1.0 / dir;
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

            if (!intersectAABB(origin, invDir, node.min_left.xyz, node.max_right.xyz, tMax)) continue;
//          if (!intersectAABB(origin, invDir, node.min_left.xyz, node.max_right.xyz, tMax)) continue;

            if (node.min_left.w < 0.0) { // Leaf
//          if (node.min_left.w < 0.0) { // Leaf
                int triIdx = int(node.max_right.w);
//              int triIdx = int(node.max_right.w);
                int offset = triIdx * 9;
//              int offset = triIdx * 9;
                vec3 v0 = vec3(triangles[offset], triangles[offset+1], triangles[offset+2]);
//              vec3 v0 = vec3(triangles[offset], triangles[offset+1], triangles[offset+2]);
                vec3 v1 = vec3(triangles[offset+3], triangles[offset+4], triangles[offset+5]);
//              vec3 v1 = vec3(triangles[offset+3], triangles[offset+4], triangles[offset+5]);
                vec3 v2 = vec3(triangles[offset+6], triangles[offset+7], triangles[offset+8]);
//              vec3 v2 = vec3(triangles[offset+6], triangles[offset+7], triangles[offset+8]);
                vec3 e1 = v1 - v0;
//              vec3 e1 = v1 - v0;
                vec3 e2 = v2 - v0;
//              vec3 e2 = v2 - v0;
                vec3 h = cross(dir, e2);
//              vec3 h = cross(dir, e2);
                float a = dot(e1, h);
//              float a = dot(e1, h);
                if (abs(a) < 1e-5) continue;
//              if (abs(a) < 1e-5) continue;
                float f = 1.0/a;
//              float f = 1.0/a;
                vec3 s = origin - v0;
//              vec3 s = origin - v0;
                float u = f * dot(s, h);
//              float u = f * dot(s, h);
                if (u < 0.0 || u > 1.0) continue;
//              if (u < 0.0 || u > 1.0) continue;
                vec3 q = cross(s, e1);
//              vec3 q = cross(s, e1);
                float v = f * dot(dir, q);
//              float v = f * dot(dir, q);
                if (v < 0.0 || u + v > 1.0) continue;
//              if (v < 0.0 || u + v > 1.0) continue;
                float t = f * dot(e2, q);
//              float t = f * dot(e2, q);
                if (t > 0.001 && t < tMax) return true; // Occluded
//              if (t > 0.001 && t < tMax) return true; // Occluded
            } else { // Internal
//          } else { // Internal
                if (stackPtr < 32) stack[stackPtr++] = int(node.max_right.w);
//              if (stackPtr < 32) stack[stackPtr++] = int(node.max_right.w);
                if (stackPtr < 32) stack[stackPtr++] = int(node.min_left.w);
//              if (stackPtr < 32) stack[stackPtr++] = int(node.min_left.w);
            }
//          }
        }
//      }
        return false; // Not occluded
//      return false; // Not occluded
    }
//  }

    // Closest Hit Traversal
    // Closest Hit Traversal
    bool traverseBVHClosest(vec3 origin, vec3 dir, inout Hit hit) {
//  bool traverseBVHClosest(vec3 origin, vec3 dir, inout Hit hit) {
        vec3 invDir = 1.0 / dir;
//      vec3 invDir = 1.0 / dir;
        int stack[32];
//      int stack[32];
        int stackPtr = 0;
//      int stackPtr = 0;
        stack[stackPtr++] = 0;
//      stack[stackPtr++] = 0;
        bool hitSomething = false;
//      bool hitSomething = false;

        while (stackPtr > 0) {
//      while (stackPtr > 0) {
            int nodeIdx = stack[--stackPtr];
//          int nodeIdx = stack[--stackPtr];
            Node node = nodes[nodeIdx];
//          Node node = nodes[nodeIdx];

            if (!intersectAABB(origin, invDir, node.min_left.xyz, node.max_right.xyz, hit.t)) continue;
//          if (!intersectAABB(origin, invDir, node.min_left.xyz, node.max_right.xyz, hit.t)) continue;

            if (node.min_left.w < 0.0) { // Leaf
//          if (node.min_left.w < 0.0) { // Leaf
                int triIdx = int(node.max_right.w);
//              int triIdx = int(node.max_right.w);
                if (intersectTriangleClosest(origin, dir, triIdx, hit)) {
//              if (intersectTriangleClosest(origin, dir, triIdx, hit)) {
                    hitSomething = true;
//                  hitSomething = true;
                }
//              }
            } else {
//          } else {
                if (stackPtr < 32) stack[stackPtr++] = int(node.max_right.w);
//              if (stackPtr < 32) stack[stackPtr++] = int(node.max_right.w);
                if (stackPtr < 32) stack[stackPtr++] = int(node.min_left.w);
//              if (stackPtr < 32) stack[stackPtr++] = int(node.min_left.w);
            }
//          }
        }
//      }
        return hitSomething;
//      return hitSomething;
    }
//  }

    // Sampling
    // Sampling
    vec3 sampleHemisphere(vec3 n) {
//  vec3 sampleHemisphere(vec3 n) {
        float z = rand(); // cos(theta)
//      float z = rand(); // cos(theta)
        float r = sqrt(max(0.0, 1.0 - z * z));
//      float r = sqrt(max(0.0, 1.0 - z * z));
        float phi = 2.0 * PI * rand();
//      float phi = 2.0 * PI * rand();
        vec3 local = vec3(r * cos(phi), r * sin(phi), z);
//      vec3 local = vec3(r * cos(phi), r * sin(phi), z);

        vec3 up = abs(n.z) < 0.999 ? vec3(0.0, 0.0, 1.0) : vec3(1.0, 0.0, 0.0);
//      vec3 up = abs(n.z) < 0.999 ? vec3(0.0, 0.0, 1.0) : vec3(1.0, 0.0, 0.0);
        vec3 x = normalize(cross(up, n));
//      vec3 x = normalize(cross(up, n));
        vec3 y = cross(n, x);
//      vec3 y = cross(n, x);

        return x * local.x + y * local.y + n * local.z;
//      return x * local.x + y * local.y + n * local.z;
    }
//  }

    vec3 aces(vec3 x) {
//  vec3 aces(vec3 x) {
        const float a = 2.51;
//      const float a = 2.51;
        const float b = 0.03;
//      const float b = 0.03;
        const float c = 2.43;
//      const float c = 2.43;
        const float d = 0.59;
//      const float d = 0.59;
        const float e = 0.14;
//      const float e = 0.14;
        return clamp((x * (a * x + b)) / (x * (c * x + d) + e), 0.0, 1.0);
//      return clamp((x * (a * x + b)) / (x * (c * x + d) + e), 0.0, 1.0);
    }
//  }

    vec3 reinhard(vec3 x) {
//  vec3 reinhard(vec3 x) {
        return x / (x + vec3(1.0));
//      return x / (x + vec3(1.0));
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
        // Read First hit from G-Buffer (Depth 0)
        vec4 sampleGlobalPosition = imageLoad(textureGeometryGlobalPosition, pixelCoordinates);
//      vec4 sampleGlobalPosition = imageLoad(textureGeometryGlobalPosition, pixelCoordinates);
        vec4 sampleGlobalNormal = imageLoad(textureGeometryGlobalNormal, pixelCoordinates);
//      vec4 sampleGlobalNormal = imageLoad(textureGeometryGlobalNormal, pixelCoordinates);
        vec4 sampleAlbedo = imageLoad(textureGeometryAlbedo, pixelCoordinates);
//      vec4 sampleAlbedo = imageLoad(textureGeometryAlbedo, pixelCoordinates);

        // State variables
        // State variables
        vec3 accumulatedColor = vec3(0.0);
//      vec3 accumulatedColor = vec3(0.0);
        vec3 attenuation = vec3(1.0);
//      vec3 attenuation = vec3(1.0);
        vec3 rayOrigin;
//      vec3 rayOrigin;
        vec3 rayDir;
//      vec3 rayDir;

        bool hitSky = false;
//      bool hitSky = false;

        // Depth 0: G-Buffer Data
        // Depth 0: G-Buffer Data
        if (sampleGlobalPosition.w == 0.0) {
//      if (sampleGlobalPosition.w == 0.0) {
            // Missed geometry in raster pass -> Sky
            // Missed geometry in raster pass -> Sky
            vec2 uv = (vec2(pixelCoordinates) + 0.5) / vec2(dimensions);
//          vec2 uv = (vec2(pixelCoordinates) + 0.5) / vec2(dimensions);
            // Reconstruct view ray from camera for correct sky (simplified here)
            // Reconstruct view ray from camera for correct sky (simplified here)
            // We don't have camera vectors here easily without uniforms or matrix.
            // We don't have camera vectors here easily without uniforms or matrix.
            // Fallback to simple gradient based on screen Y
            // Fallback to simple gradient based on screen Y
            vec3 sky = mix(vec3(0.7, 0.8, 0.9), vec3(0.4, 0.7, 1.0), uv.y);
//          vec3 sky = mix(vec3(0.7, 0.8, 0.9), vec3(0.4, 0.7, 1.0), uv.y);
            accumulatedColor = sky;
//          accumulatedColor = sky;
            hitSky = true;
//          hitSky = true;
        } else {
//      } else {
            // Hit geometry
            // Hit geometry
            rayOrigin = sampleGlobalPosition.xyz;
//          rayOrigin = sampleGlobalPosition.xyz;
            vec3 N = normalize(sampleGlobalNormal.xyz);
//          vec3 N = normalize(sampleGlobalNormal.xyz);
            vec3 albedo = sampleAlbedo.rgb;
//          vec3 albedo = sampleAlbedo.rgb;

            // Apply Emission (If any - currently none, assuming objects are not emissive)
            // Apply Emission (If any - currently none, assuming objects are not emissive)
            // accumulatedColor += attenuation * emission;
            // accumulatedColor += attenuation * emission;

            // --- Multiple Importance Sampling: Next Event Estimation (Direct Light) ---
            // --- Multiple Importance Sampling: Next Event Estimation (Direct Light) ---

            // Soft Shadow (Area Light approximation)
            // Soft Shadow (Area Light approximation)
            vec3 sunDirectionCenter = normalize(uPointLight001GlobalPosition - rayOrigin);
//          vec3 sunDirectionCenter = normalize(uPointLight001GlobalPosition - rayOrigin);
            vec3 sunColor = vec3(300.0); // Increased intensity
//          vec3 sunColor = vec3(300.0); // Increased intensity
            float distToLight = length(uPointLight001GlobalPosition - rayOrigin);
//          float distToLight = length(uPointLight001GlobalPosition - rayOrigin);
            float sunRadius = 0.05; // Sharper shadows
//          float sunRadius = 0.05; // Sharper shadows

            vec3 jitteredSunPos = uPointLight001GlobalPosition + randomUnitVector() * sunRadius;
//          vec3 jitteredSunPos = uPointLight001GlobalPosition + randomUnitVector() * sunRadius;
            vec3 jitteredSunDir = normalize(jitteredSunPos - rayOrigin);
//          vec3 jitteredSunDir = normalize(jitteredSunPos - rayOrigin);
            float jitteredDist = length(jitteredSunPos - rayOrigin);
//          float jitteredDist = length(jitteredSunPos - rayOrigin);

            vec3 shadowRayOrigin = rayOrigin + N * 0.001;
//          vec3 shadowRayOrigin = rayOrigin + N * 0.001;

            if (!traverseBVHShadow(shadowRayOrigin, jitteredSunDir, jitteredDist)) {
//          if (!traverseBVHShadow(shadowRayOrigin, jitteredSunDir, jitteredDist)) {
                // Not occluded
                // Not occluded
                float cosTheta = max(0.0, dot(N, jitteredSunDir));
//              float cosTheta = max(0.0, dot(N, jitteredSunDir));
                // Inverse square falloff for point light
                // Inverse square falloff for point light
                float falloff = 1.0 / (jitteredDist * jitteredDist);
//              float falloff = 1.0 / (jitteredDist * jitteredDist);
                vec3 directLight = albedo * sunColor * cosTheta * falloff; // Missing 1/PI? Diffuse BRDF is albedo/PI.
//              vec3 directLight = albedo * sunColor * cosTheta * falloff; // Missing 1/PI? Diffuse BRDF is albedo/PI.
                // We keep it simpler or physically correct?
                // We keep it simpler or physically correct?
                // Example used: albedo * sunColor * cosTheta / PI
                // Example used: albedo * sunColor * cosTheta / PI
                accumulatedColor += attenuation * directLight; // / PI;
//              accumulatedColor += attenuation * directLight; // / PI;
            }

            // Prepare for next bounce (Indirect)
            // Prepare for next bounce (Indirect)
            attenuation *= albedo;
//          attenuation *= albedo;
            rayOrigin = rayOrigin + N * 0.001;
//          rayOrigin = rayOrigin + N * 0.001;
            rayDir = sampleHemisphere(N);
//          rayDir = sampleHemisphere(N);
        }
//      }

        if (!hitSky) {
//      if (!hitSky) {
            // Path Trace loop for bounces 1..Max
            // Path Trace loop for bounces 1..Max
            int maxDepth = 3;
//          int maxDepth = 3;
            for (int depth = 1; depth < maxDepth; depth++) {
//          for (int depth = 1; depth < maxDepth; depth++) {
                Hit hit;
//              Hit hit;
                hit.t = INF;
//              hit.t = INF;
                hit.triIdx = -1;
//              hit.triIdx = -1;

                if (traverseBVHClosest(rayOrigin, rayDir, hit)) {
//              if (traverseBVHClosest(rayOrigin, rayDir, hit)) {
                    // Hit surface
                    // Hit surface
                    rayOrigin = rayOrigin + rayDir * hit.t;
//                  rayOrigin = rayOrigin + rayDir * hit.t;

                    // Reconstruct Normal
                    // Reconstruct Normal
                    int offset = hit.triIdx * 9;
//                  int offset = hit.triIdx * 9;
                    vec3 v0 = vec3(triangles[offset], triangles[offset+1], triangles[offset+2]);
//                  vec3 v0 = vec3(triangles[offset], triangles[offset+1], triangles[offset+2]);
                    vec3 v1 = vec3(triangles[offset+3], triangles[offset+4], triangles[offset+5]);
//                  vec3 v1 = vec3(triangles[offset+3], triangles[offset+4], triangles[offset+5]);
                    vec3 v2 = vec3(triangles[offset+6], triangles[offset+7], triangles[offset+8]);
//                  vec3 v2 = vec3(triangles[offset+6], triangles[offset+7], triangles[offset+8]);
                    vec3 hitN = normalize(cross(v1 - v0, v2 - v0));
//                  vec3 hitN = normalize(cross(v1 - v0, v2 - v0));
                    if (dot(rayDir, hitN) > 0.0) hitN = -hitN;
//                  if (dot(rayDir, hitN) > 0.0) hitN = -hitN;

                    vec3 hitAlbedo = vec3(0.7); // Default diffuse gray
//                  vec3 hitAlbedo = vec3(0.7); // Default diffuse gray

                    // Direct Light (NEE)
                    // Direct Light (NEE)
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

                    vec3 shadowRayOrigin = rayOrigin + hitN * 0.001;
//                  vec3 shadowRayOrigin = rayOrigin + hitN * 0.001;

                    if (!traverseBVHShadow(shadowRayOrigin, jitteredSunDir, jitteredDist)) {
//                  if (!traverseBVHShadow(shadowRayOrigin, jitteredSunDir, jitteredDist)) {
                        float cosTheta = max(0.0, dot(hitN, jitteredSunDir));
//                      float cosTheta = max(0.0, dot(hitN, jitteredSunDir));
                        float falloff = 1.0 / (jitteredDist * jitteredDist);
//                      float falloff = 1.0 / (jitteredDist * jitteredDist);
                        vec3 directLight = hitAlbedo * sunColor * cosTheta * falloff;
//                      vec3 directLight = hitAlbedo * sunColor * cosTheta * falloff;
                        accumulatedColor += attenuation * directLight;
//                      accumulatedColor += attenuation * directLight;
                    }

                    // Scatter
                    // Scatter
                    attenuation *= hitAlbedo;
//                  attenuation *= hitAlbedo;
                    rayOrigin = shadowRayOrigin; // Already offset
//                  rayOrigin = shadowRayOrigin; // Already offset
                    rayDir = sampleHemisphere(hitN);
//                  rayDir = sampleHemisphere(hitN);

                    // Russian Roulette
                    // Russian Roulette
                    if (depth >= 2) {
//                  if (depth >= 2) {
                        float p = max(attenuation.r, max(attenuation.g, attenuation.b));
//                      float p = max(attenuation.r, max(attenuation.g, attenuation.b));
                        if (rand() > p) break;
//                      if (rand() > p) break;
                        attenuation *= 1.0 / p;
//                      attenuation *= 1.0 / p;
                    }

                } else {
//              } else {
                    // Hit Sky
                    // Hit Sky
                    vec3 sky = getSkyColor(rayDir);
//                  vec3 sky = getSkyColor(rayDir);
                    accumulatedColor += attenuation * sky;
//                  accumulatedColor += attenuation * sky;
                    break;
//                  break;
                }
            }
//          }
        }
//      }

        // Accumulation
        // Accumulation
        vec4 prevAccum = imageLoad(textureAccum, pixelCoordinates);
//      vec4 prevAccum = imageLoad(textureAccum, pixelCoordinates);
        
        vec3 history = prevAccum.rgb;
//      vec3 history = prevAccum.rgb;

        // Temporal Accumulation (Exponential Moving Average)
        // Temporal Accumulation (Exponential Moving Average)
        // Alpha determines how much the current frame contributes.
        // Alpha determines how much the current frame contributes.
        // Start with 1.0 (overwrite) and decay to a minimum (e.g., 0.1) for dynamic stability.
        // Start with 1.0 (overwrite) and decay to a minimum (e.g., 0.1) for dynamic stability.
        float alpha = 1.0 / float(uFrameCount);
//      float alpha = 1.0 / float(uFrameCount);
        alpha = max(alpha, 0.1); // Keep 10% contribution from new frames to handle motion/light changes
//      alpha = max(alpha, 0.1); // Keep 10% contribution from new frames to handle motion/light changes
        
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
