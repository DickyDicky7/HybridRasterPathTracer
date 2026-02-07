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

    // Principled BSDF Helpers
    // Principled BSDF Helpers
    vec3 schlickFresnel(float cosine, vec3 f0) {
//  vec3 schlickFresnel(float cosine, vec3 f0) {
        return f0 + (1.0 - f0) * pow(1.0 - cosine, 5.0);
//      return f0 + (1.0 - f0) * pow(1.0 - cosine, 5.0);
    }
//  }

    vec3 sampleGGX(vec3 n, float roughness) {
//  vec3 sampleGGX(vec3 n, float roughness) {
        float r1 = rand();
//      float r1 = rand();
        float r2 = rand();
//      float r2 = rand();
        float a = roughness * roughness;
//      float a = roughness * roughness;
        float phi = 2.0 * PI * r1;
//      float phi = 2.0 * PI * r1;
        float cosTheta = sqrt((1.0 - r2) / (1.0 + (a * a - 1.0) * r2));
//      float cosTheta = sqrt((1.0 - r2) / (1.0 + (a * a - 1.0) * r2));
        float sinTheta = sqrt(1.0 - cosTheta * cosTheta);
//      float sinTheta = sqrt(1.0 - cosTheta * cosTheta);

        vec3 h;
//      vec3 h;
        h.x = sinTheta * cos(phi);
//      h.x = sinTheta * cos(phi);
        h.y = sinTheta * sin(phi);
//      h.y = sinTheta * sin(phi);
        h.z = cosTheta;
//      h.z = cosTheta;

        vec3 up = abs(n.z) < 0.999 ? vec3(0.0, 0.0, 1.0) : vec3(1.0, 0.0, 0.0);
//      vec3 up = abs(n.z) < 0.999 ? vec3(0.0, 0.0, 1.0) : vec3(1.0, 0.0, 0.0);
        vec3 tangent = normalize(cross(up, n));
//      vec3 tangent = normalize(cross(up, n));
        vec3 bitangent = cross(n, tangent);
//      vec3 bitangent = cross(n, tangent);

        return normalize(tangent * h.x + bitangent * h.y + n * h.z);
//      return normalize(tangent * h.x + bitangent * h.y + n * h.z);
    }
//  }

    vec3 reflectPrincipled(vec3 i, vec3 n) {
//  vec3 reflectPrincipled(vec3 i, vec3 n) {
        return i - 2.0 * dot(i, n) * n;
//      return i - 2.0 * dot(i, n) * n;
    }
//  }

    vec3 refractPrincipled(vec3 i, vec3 n, float eta, float cosThetaI, float sinThetaT2) {
//  vec3 refractPrincipled(vec3 i, vec3 n, float eta, float cosThetaI, float sinThetaT2) {
        float cosThetaT = sqrt(1.0 - sinThetaT2);
//      float cosThetaT = sqrt(1.0 - sinThetaT2);
        return normalize(eta * i + (eta * cosThetaI - cosThetaT) * n);
//      return normalize(eta * i + (eta * cosThetaI - cosThetaT) * n);
    }
//  }

    // Principled BSDF Scatter Function
//  // Principled BSDF Scatter Function
    bool scatterPrincipled(vec3 V, vec3 N, Material mat, inout vec3 rayOrigin, inout vec3 rayDir, inout vec3 attenuation, inout bool hitSky) {
//  bool scatterPrincipled(vec3 V, vec3 N, Material mat, inout vec3 rayOrigin, inout vec3 rayDir, inout vec3 attenuation, inout bool hitSky) {
        vec3 albedo = mat.albedo.rgb;
//      vec3 albedo = mat.albedo.rgb;
        float roughness = mat.roughness;
//      float roughness = mat.roughness;
        float metallic = mat.metallic;
//      float metallic = mat.metallic;
        float transmission = mat.transmission;
//      float transmission = mat.transmission;
        float ior = mat.ior;
//      float ior = mat.ior;

        // F0 calculation: 0.04 for dielectrics, albedo for metals
//      // F0 calculation: 0.04 for dielectrics, albedo for metals
        vec3 f0 = mix(vec3(0.04), albedo, metallic);
//      vec3 f0 = mix(vec3(0.04), albedo, metallic);

        // Schlick's fresnel approximation at incident angle
//      // Schlick's fresnel approximation at incident angle
        float cosTheta = min(dot(-V, N), 1.0);
//      float cosTheta = min(dot(-V, N), 1.0);
        vec3 fresnel = schlickFresnel(cosTheta, f0);
//      vec3 fresnel = schlickFresnel(cosTheta, f0);

        // Use average fresnel for importance sampling probability
//      // Use average fresnel for importance sampling probability
        float fresnelProb = (fresnel.r + fresnel.g + fresnel.b) / 3.0;
//      float fresnelProb = (fresnel.r + fresnel.g + fresnel.b) / 3.0;

        // --- PATH A: METALLIC REFLECTION ---
//      // --- PATH A: METALLIC REFLECTION ---
        // Metals always reflect. They have no diffuse and no transmission.
//      // Metals always reflect. They have no diffuse and no transmission.
        // We also use fresnelProbability to decide between Specular Reflection and Diffuse/Transmission for Dielectrics.
//      // We also use fresnelProbability to decide between Specular Reflection and Diffuse/Transmission for Dielectrics.
        // [ Probability to Reflect = Metallic + (1.0 - Metallic) * Fresnel ]
//      // [ Probability to Reflect = Metallic + (1.0 - Metallic) * Fresnel ]

        float specularProb = mix(fresnelProb, 1.0, metallic);
//      float specularProb = mix(fresnelProb, 1.0, metallic);
        float p = rand();
//      float p = rand();

        if (p < specularProb) {
//      if (p < specularProb) {
            // SPECULAR REFLECTION (METAL OR DIELECTRIC COAT)
//          // SPECULAR REFLECTION (METAL OR DIELECTRIC COAT)
            vec3 microfacetN = sampleGGX(N, roughness);
//          vec3 microfacetN = sampleGGX(N, roughness);
            vec3 reflected = reflectPrincipled(V, microfacetN);
//          vec3 reflected = reflectPrincipled(V, microfacetN);

            if (dot(reflected, N) > 0.0) {
//          if (dot(reflected, N) > 0.0) {
                attenuation *= mix(vec3(1.0), albedo, metallic);
//              attenuation *= mix(vec3(1.0), albedo, metallic);
                rayDir = reflected;
//              rayDir = reflected;
                rayOrigin = rayOrigin + N * 0.001;
//              rayOrigin = rayOrigin + N * 0.001;
                return true;
//              return true;
            } else {
//          } else {
                // Current/Recent ray is/was absorbed (next ray is scattering into surface)
//              // Current/Recent ray is/was absorbed (next ray is scattering into surface)
                hitSky = true; // Absorb/Terminate (treat as sky hit for now to stop?) Or just false return?
//              hitSky = true; // Absorb/Terminate (treat as sky hit for now to stop?) Or just false return?
                // Actually returning false usually stops tracing.
//              // Actually returning false usually stops tracing.
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
            float p2 = (p - specularProb) / (1.0 - specularProb);
//          float p2 = (p - specularProb) / (1.0 - specularProb);

            if (transmission > 0.0 && p2 < transmission) {
//          if (transmission > 0.0 && p2 < transmission) {
                // TRANSMISSION (REFRACTION)
//              // TRANSMISSION (REFRACTION)
                float eta = 1.0 / ior;
//              float eta = 1.0 / ior;
                vec3 N_eff = N;
//              vec3 N_eff = N;
                bool entering = dot(V, N) < 0.0;
//              bool entering = dot(V, N) < 0.0;
                if (!entering) {
//              if (!entering) {
                    eta = ior; // Material -> Air
//                  eta = ior; // Material -> Air
                    N_eff = -N;
//                  N_eff = -N;
                }
//              }

                vec3 microfacetN = sampleGGX(N_eff, roughness);
//              vec3 microfacetN = sampleGGX(N_eff, roughness);
                float cosThetaIncidence = min(dot(-V, microfacetN), 1.0);
//              float cosThetaIncidence = min(dot(-V, microfacetN), 1.0);
                float sinThetaTransmission = (1.0 - cosThetaIncidence * cosThetaIncidence) * (eta * eta);
//              float sinThetaTransmission = (1.0 - cosThetaIncidence * cosThetaIncidence) * (eta * eta);

                // When [ sinThetaTransmission <= 1.0 ] then Refraction happened else Total Internal Reflection happened
//              // When [ sinThetaTransmission <= 1.0 ] then Refraction happened else Total Internal Reflection happened
                if (sinThetaTransmission <= 1.0) {
//              if (sinThetaTransmission <= 1.0) {
                     vec3 refracted = refractPrincipled(V, microfacetN, eta, cosThetaIncidence, sinThetaTransmission);
//                   vec3 refracted = refractPrincipled(V, microfacetN, eta, cosThetaIncidence, sinThetaTransmission);
                     attenuation *= albedo;
//                   attenuation *= albedo;
                     rayDir = refracted;
//                   rayDir = refracted;
                     rayOrigin = rayOrigin - N_eff * 0.001; // Offset INSIDE (relative to effective normal)
//                   rayOrigin = rayOrigin - N_eff * 0.001; // Offset INSIDE (relative to effective normal)
                     return true;
//                   return true;
                } else {
//              } else {
                    // TIR -> Reflect
//                  // TIR -> Reflect
                    vec3 reflected = reflectPrincipled(V, microfacetN);
//                  vec3 reflected = reflectPrincipled(V, microfacetN);
                    attenuation *= vec3(1.0);
//                  attenuation *= vec3(1.0);
                    rayDir = reflected;
//                  rayDir = reflected;
                    rayOrigin = rayOrigin + N_eff * 0.001;
//                  rayOrigin = rayOrigin + N_eff * 0.001;
                    return true;
//                  return true;
                }
            } else {
//          } else {
                // DIFFUSE (LAMBERTIAN)
//              // DIFFUSE (LAMBERTIAN)
                rayDir = sampleHemisphere(N);
//              rayDir = sampleHemisphere(N);
                attenuation *= albedo;
//              attenuation *= albedo;
                rayOrigin = rayOrigin + N * 0.001;
//              rayOrigin = rayOrigin + N * 0.001;
                return true;
//              return true;
            }
//          }
        }
//      }
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

            accumulatedColor = getSkyColor(rayDir);
//          accumulatedColor = getSkyColor(rayDir);
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

            if (!traverseBVHShadow(shadowRayOrigin, jitteredSunDir, jitteredDist)) {
//          if (!traverseBVHShadow(shadowRayOrigin, jitteredSunDir, jitteredDist)) {
                float cosTheta = max(0.0, dot(N, jitteredSunDir));
//              float cosTheta = max(0.0, dot(N, jitteredSunDir));
                // Inverse square falloff for point light
//              // Inverse square falloff for point light
                float falloff = 1.0 / (jitteredDist * jitteredDist);
//              float falloff = 1.0 / (jitteredDist * jitteredDist);
                // Approximate NEE using albedo (Diffuse assumption for now)
//              // Approximate NEE using albedo (Diffuse assumption for now)
                vec3 directLight = mat.albedo.rgb * sunColor * cosTheta * falloff;
//              vec3 directLight = mat.albedo.rgb * sunColor * cosTheta * falloff;
                accumulatedColor += attenuation * directLight;
//              accumulatedColor += attenuation * directLight;
            }
//          }

            // --- Principled Scatter ---
//          // --- Principled Scatter ---
            bool scattered = scatterPrincipled(V, N, mat, rayOrigin, rayDir, attenuation, hitSky);
//          bool scattered = scatterPrincipled(V, N, mat, rayOrigin, rayDir, attenuation, hitSky);
            if (!scattered && hitSky) {
//          if (!scattered && hitSky) {
                // Absorbed? If hitSky was set to true in scatterPrincipled it means absorption/termination
//              // Absorbed? If hitSky was set to true in scatterPrincipled it means absorption/termination
                // accumulatedColor += vec3(0.0);
//              // accumulatedColor += vec3(0.0);
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
                Hit hit;
//              Hit hit;
                hit.t = INF;
//              hit.t = INF;
                hit.triIdx = -1;
//              hit.triIdx = -1;

                if (traverseBVHClosest(rayOrigin, rayDir, hit)) {
//              if (traverseBVHClosest(rayOrigin, rayDir, hit)) {
                    // Hit surface
//                  // Hit surface
                    V = rayDir; // Incoming direction for next bounce
//                  V = rayDir; // Incoming direction for next bounce
                    rayOrigin = rayOrigin + rayDir * hit.t;
//                  rayOrigin = rayOrigin + rayDir * hit.t;

                    // Reconstruct Normal
//                  // Reconstruct Normal
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
                    bool frontFace = dot(V, hitN) < 0.0;
//                  bool frontFace = dot(V, hitN) < 0.0;
                    if (!frontFace) hitN = -hitN;
//                  if (!frontFace) hitN = -hitN;

                    Material mat = materials[hit.triIdx];
//                  Material mat = materials[hit.triIdx];

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

                    if (!traverseBVHShadow(shadowRayOrigin, jitteredSunDir, jitteredDist)) {
//                  if (!traverseBVHShadow(shadowRayOrigin, jitteredSunDir, jitteredDist)) {
                        float cosTheta = max(0.0, dot(hitN, jitteredSunDir));
//                      float cosTheta = max(0.0, dot(hitN, jitteredSunDir));
                        // Inverse square falloff for point light
//                      // Inverse square falloff for point light
                        float falloff = 1.0 / (jitteredDist * jitteredDist);
//                      float falloff = 1.0 / (jitteredDist * jitteredDist);
                        vec3 directLight = mat.albedo.rgb * sunColor * cosTheta * falloff;
//                      vec3 directLight = mat.albedo.rgb * sunColor * cosTheta * falloff;
                        accumulatedColor += attenuation * directLight;
//                      accumulatedColor += attenuation * directLight;
                    }
//                  }

                    // --- Principled Scatter ---
//                  // --- Principled Scatter ---
                    // Prepare for next bounce (Indirect)
//                  // Prepare for next bounce (Indirect)
                    bool scattered = scatterPrincipled(V, hitN, mat, rayOrigin, rayDir, attenuation, hitSky);
//                  bool scattered = scatterPrincipled(V, hitN, mat, rayOrigin, rayDir, attenuation, hitSky);
                    if (!scattered) break;
//                  if (!scattered) break;

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
                    vec3 sky = getSkyColor(rayDir);
//                  vec3 sky = getSkyColor(rayDir);
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
