    #version 430
//  #version 430

    layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;
//  layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;

    layout(binding = 1, rgba32f) uniform image2D textureGeometryGlobalPosition;
//  layout(binding = 1, rgba32f) uniform image2D textureGeometryGlobalPosition;
    layout(binding = 2, rgba16f) uniform image2D textureGeometryGlobalNormal;
//  layout(binding = 2, rgba16f) uniform image2D textureGeometryGlobalNormal;
    layout(binding = 3, rgba8) uniform image2D textureGeometryAlbedo;
//  layout(binding = 3, rgba8) uniform image2D textureGeometryAlbedo;

    struct Node {
//  struct Node {
        vec4 aabbMinAndLeftChild;
//      vec4 aabbMinAndLeftChild;
        vec4 aabbMaxAndRightChild;
//      vec4 aabbMaxAndRightChild;
    };
//  };

    layout(std430, binding = 6) buffer BVHNodes {
//  layout(std430, binding = 6) buffer BVHNodes {
        Node nodes[];
//      Node nodes[];
    };
//  };

    struct VertexData {
//  struct VertexData {
        vec4 positionAndTexcoordU;
//      vec4 positionAndTexcoordU;
        vec4 normalAndTexcoordV;
//      vec4 normalAndTexcoordV;
        vec4 tangentAndMaterialIndex;
//      vec4 tangentAndMaterialIndex;
    };
//  };

    layout(std430, binding = 7) buffer SceneVertices {
//  layout(std430, binding = 7) buffer SceneVertices {
        VertexData vertices[];
//      VertexData vertices[];
    };
//  };

    struct Material {
//  struct Material {
        vec4 albedo;
//      vec4 albedo;
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
        float textureIndexTransmission;
//      float textureIndexTransmission;
        float padding002;
//      float padding002;
        vec2 uvScale;
//      vec2 uvScale;
        vec2 padding003;
//      vec2 padding003;
    };
//  };

    layout(std430, binding = 8) buffer SceneMaterials {
//  layout(std430, binding = 8) buffer SceneMaterials {
        Material materials[];
//      Material materials[];
    };
//  };

    layout(std430, binding = 11) buffer NRCWeights {
//  layout(std430, binding = 11) buffer NRCWeights {
        float neuralNetworkWeights[];
//      float neuralNetworkWeights[];
    };
//  };

    layout(std430, binding = 12) buffer NRCTrainingRecords {
//  layout(std430, binding = 12) buffer NRCTrainingRecords {
        float nrcTrainingRecords[];
//      float nrcTrainingRecords[];
    };
//  };

    layout(std430, binding = 13) buffer NRCTrainingCounter {
//  layout(std430, binding = 13) buffer NRCTrainingCounter {
        uint uTrainingRecordCount;
//      uint uTrainingRecordCount;
    };
//  };

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
    uniform uint uTrainingStep;
//  uniform uint uTrainingStep;
    uniform ivec2 uResolution;
//  uniform ivec2 uResolution;
    // NRC input-domain normalisation; must match hybrid_shading_cs.glsl and nrc_train_cs.glsl exactly.
//  // NRC input-domain normalisation; must match hybrid_shading_cs.glsl and nrc_train_cs.glsl exactly.
    uniform vec3 uNRCPositionOffset;
//  uniform vec3 uNRCPositionOffset;
    uniform float uNRCPositionScale;
//  uniform float uNRCPositionScale;

    // Workgroup-resident copy of the NRC Exponential Moving Average weights, filled once per
//  // Workgroup-resident copy of the NRC Exponential Moving Average weights, filled once per
    // workgroup at the top of main() so every cache query reads shared memory instead of the SSBO.
//  // workgroup at the top of main() so every cache query reads shared memory instead of the SSBO.
    shared float shared_network_weights[1139];
//  shared float shared_network_weights[1139];

    const float PI = 3.14159265359;
//  const float PI = 3.14159265359;
    const float TWO_PI = 6.28318530718;
//  const float TWO_PI = 6.28318530718;
    const float EPSILON_OFFSET = 0.001;
//  const float EPSILON_OFFSET = 0.001;
    const float INF = 1e30;
//  const float INF = 1e30;

    uint gRngState;
//  uint gRngState;

    uint randUint() {
//  uint randUint() {
        gRngState = gRngState * 747796405u + 2891336453u;
//      gRngState = gRngState * 747796405u + 2891336453u;
        uint word = ((gRngState >> ((gRngState >> 28u) + 4u)) ^ gRngState) * 277803737u;
//      uint word = ((gRngState >> ((gRngState >> 28u) + 4u)) ^ gRngState) * 277803737u;
        return (word >> 22u) ^ word;
//      return (word >> 22u) ^ word;
    }
//  }

    float randFloat() {
//  float randFloat() {
        return float(randUint() >> 8u) / 16777216.0;
//      return float(randUint() >> 8u) / 16777216.0;
    }
//  }

    vec3 randomUnitVector() {
//  vec3 randomUnitVector() {
        float cosPolar = randFloat() * 2.0 - 1.0;
//      float cosPolar = randFloat() * 2.0 - 1.0;
        float azimuthAngle = randFloat() * TWO_PI;
//      float azimuthAngle = randFloat() * TWO_PI;
        float sinPolar = sqrt(max(0.0, 1.0 - cosPolar * cosPolar));
//      float sinPolar = sqrt(max(0.0, 1.0 - cosPolar * cosPolar));
        return vec3(sinPolar * cos(azimuthAngle), sinPolar * sin(azimuthAngle), cosPolar);
//      return vec3(sinPolar * cos(azimuthAngle), sinPolar * sin(azimuthAngle), cosPolar);
    }
//  }

    float calculateLuminance(vec3 color) {
//  float calculateLuminance(vec3 color) {
        return dot(color, vec3(0.2126, 0.7152, 0.0722));
//      return dot(color, vec3(0.2126, 0.7152, 0.0722));
    }
//  }

    // Neural Radiance Cache forward pass.
//  // Neural Radiance Cache forward pass.
    // Multi-Layer Perceptron: 16 inputs -> 16 hidden (x4, ReLU) -> 3 outputs.
//  // Multi-Layer Perceptron: 16 inputs -> 16 hidden (x4, ReLU) -> 3 outputs.
    //
//  //
    // Reads the weights out of shared_network_weights, which main() fills once per workgroup with the
//  // Reads the weights out of shared_network_weights, which main() fills once per workgroup with the
    // Exponential Moving Average copy of the network. Two reasons this matters: it turns 1139 scalar
//  // Exponential Moving Average copy of the network. Two reasons this matters: it turns 1139 scalar
    // SSBO loads per query into shared-memory loads, and reading the weights straight from the SSBO
//  // SSBO loads per query into shared-memory loads, and reading the weights straight from the SSBO
    // inside this deeply inlined function makes the NVIDIA GLSL compiler give up with
//  // inside this deeply inlined function makes the NVIDIA GLSL compiler give up with
    // "error C5025: lvalue in array access too complex", failing the whole program link.
//  // "error C5025: lvalue in array access too complex", failing the whole program link.
    //
//  //
    // Activations are flat scalar arrays rather than vec4[4] blocks for the same reason: assigning a
//  // Activations are flat scalar arrays rather than vec4[4] blocks for the same reason: assigning a
    // dynamically indexed component of a dynamically indexed vec4 array ("h[i / 4u][i % 4u] = ...")
//  // dynamically indexed component of a dynamically indexed vec4 array ("h[i / 4u][i % 4u] = ...")
    // trips the same compiler limit. This layout also matches nrc_train_cs.glsl exactly, so inference
//  // trips the same compiler limit. This layout also matches nrc_train_cs.glsl exactly, so inference
    // and training cannot drift apart.
//  // and training cannot drift apart.
    vec3 evaluateNeuralRadianceCache(vec3 position, vec3 normal) {
//  vec3 evaluateNeuralRadianceCache(vec3 position, vec3 normal) {
        vec3 scaledPos = (position - uNRCPositionOffset) * uNRCPositionScale;
//      vec3 scaledPos = (position - uNRCPositionOffset) * uNRCPositionScale;

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
        for (uint i = 0u; i < 16u; i++) {
//      for (uint i = 0u; i < 16u; i++) {
            float activationSum = shared_network_weights[256u + i];
//          float activationSum = shared_network_weights[256u + i];
            uint weightOffset = i * 16u;
//          uint weightOffset = i * 16u;
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
        for (uint i = 0u; i < 16u; i++) {
//      for (uint i = 0u; i < 16u; i++) {
            float activationSum = shared_network_weights[528u + i];
//          float activationSum = shared_network_weights[528u + i];
            uint weightOffset = 272u + i * 16u;
//          uint weightOffset = 272u + i * 16u;
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
        for (uint i = 0u; i < 16u; i++) {
//      for (uint i = 0u; i < 16u; i++) {
            float activationSum = shared_network_weights[800u + i];
//          float activationSum = shared_network_weights[800u + i];
            uint weightOffset = 544u + i * 16u;
//          uint weightOffset = 544u + i * 16u;
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
        for (uint i = 0u; i < 16u; i++) {
//      for (uint i = 0u; i < 16u; i++) {
            float activationSum = shared_network_weights[1072u + i];
//          float activationSum = shared_network_weights[1072u + i];
            uint weightOffset = 816u + i * 16u;
//          uint weightOffset = 816u + i * 16u;
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
        vec3 outputRadiance = vec3(0.0);
//      vec3 outputRadiance = vec3(0.0);
        for (uint i = 0u; i < 3u; i++) {
//      for (uint i = 0u; i < 3u; i++) {
            float activationSum = shared_network_weights[1136u + i];
//          float activationSum = shared_network_weights[1136u + i];
            uint weightOffset = 1088u + i * 16u;
//          uint weightOffset = 1088u + i * 16u;
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
        return clamp(outputRadiance, vec3(0.0), vec3(15.0));
//      return clamp(outputRadiance, vec3(0.0), vec3(15.0));
    }
//  }

    bool intersectAABB(vec3 rayOrigin, vec3 rayInverseDirection, vec3 aabbMin, vec3 aabbMax, float maxDistance) {
//  bool intersectAABB(vec3 rayOrigin, vec3 rayInverseDirection, vec3 aabbMin, vec3 aabbMax, float maxDistance) {
        vec3 t0 = (aabbMin - rayOrigin) * rayInverseDirection;
//      vec3 t0 = (aabbMin - rayOrigin) * rayInverseDirection;
        vec3 t1 = (aabbMax - rayOrigin) * rayInverseDirection;
//      vec3 t1 = (aabbMax - rayOrigin) * rayInverseDirection;
        vec3 tmin = min(t0, t1);
//      vec3 tmin = min(t0, t1);
        vec3 tmax = max(t0, t1);
//      vec3 tmax = max(t0, t1);
        float enterDistance = max(max(tmin.x, tmin.y), max(tmin.z, 0.001));
//      float enterDistance = max(max(tmin.x, tmin.y), max(tmin.z, 0.001));
        float exitDistance = min(min(tmax.x, tmax.y), min(tmax.z, maxDistance));
//      float exitDistance = min(min(tmax.x, tmax.y), min(tmax.z, maxDistance));
        return enterDistance <= exitDistance;
//      return enterDistance <= exitDistance;
    }
//  }

    bool traverseShadowRay(vec3 origin, vec3 direction, float maxDistance) {
//  bool traverseShadowRay(vec3 origin, vec3 direction, float maxDistance) {
        vec3 directionSign = sign(direction);
//      vec3 directionSign = sign(direction);
        directionSign += 1.0 - abs(directionSign);
//      directionSign += 1.0 - abs(directionSign);
        vec3 safeDirection = direction + step(abs(direction), vec3(1e-8)) * directionSign * 1e-8;
//      vec3 safeDirection = direction + step(abs(direction), vec3(1e-8)) * directionSign * 1e-8;
        vec3 invDir = 1.0 / safeDirection;
//      vec3 invDir = 1.0 / safeDirection;
        int stack[64];
//      int stack[64];
        int stackPointer = 0;
//      int stackPointer = 0;
        stack[stackPointer++] = 0;
//      stack[stackPointer++] = 0;

        while (stackPointer > 0) {
//      while (stackPointer > 0) {
            int nodeIndex = stack[--stackPointer];
//          int nodeIndex = stack[--stackPointer];
            Node node = nodes[nodeIndex];
//          Node node = nodes[nodeIndex];

            if (!intersectAABB(origin, invDir, node.aabbMinAndLeftChild.xyz, node.aabbMaxAndRightChild.xyz, maxDistance)) {
//          if (!intersectAABB(origin, invDir, node.aabbMinAndLeftChild.xyz, node.aabbMaxAndRightChild.xyz, maxDistance)) {
                continue;
//              continue;
            }
//          }

            if (node.aabbMinAndLeftChild.w < 0.0) {
//          if (node.aabbMinAndLeftChild.w < 0.0) {
                int triangleIndex = int(node.aabbMaxAndRightChild.w);
//              int triangleIndex = int(node.aabbMaxAndRightChild.w);
                vec3 v0 = vertices[triangleIndex * 3 + 0].positionAndTexcoordU.xyz;
//              vec3 v0 = vertices[triangleIndex * 3 + 0].positionAndTexcoordU.xyz;
                vec3 v1 = vertices[triangleIndex * 3 + 1].positionAndTexcoordU.xyz;
//              vec3 v1 = vertices[triangleIndex * 3 + 1].positionAndTexcoordU.xyz;
                vec3 v2 = vertices[triangleIndex * 3 + 2].positionAndTexcoordU.xyz;
//              vec3 v2 = vertices[triangleIndex * 3 + 2].positionAndTexcoordU.xyz;

                vec3 edge1 = v1 - v0;
//              vec3 edge1 = v1 - v0;
                vec3 edge2 = v2 - v0;
//              vec3 edge2 = v2 - v0;
                vec3 rayCrossEdge2 = cross(direction, edge2);
//              vec3 rayCrossEdge2 = cross(direction, edge2);
                float determinant = dot(edge1, rayCrossEdge2);
//              float determinant = dot(edge1, rayCrossEdge2);

                if (abs(determinant) > 1.0e-7) {
//              if (abs(determinant) > 1.0e-7) {
                    float inverseDeterminant = 1.0 / determinant;
//                  float inverseDeterminant = 1.0 / determinant;
                    vec3 originToVertex0 = origin - v0;
//                  vec3 originToVertex0 = origin - v0;
                    float u = dot(originToVertex0, rayCrossEdge2) * inverseDeterminant;
//                  float u = dot(originToVertex0, rayCrossEdge2) * inverseDeterminant;
                    if (u >= 0.0 && u <= 1.0) {
//                  if (u >= 0.0 && u <= 1.0) {
                        vec3 originCrossEdge1 = cross(originToVertex0, edge1);
//                      vec3 originCrossEdge1 = cross(originToVertex0, edge1);
                        float v = dot(direction, originCrossEdge1) * inverseDeterminant;
//                      float v = dot(direction, originCrossEdge1) * inverseDeterminant;
                        if (v >= 0.0 && (u + v) <= 1.0) {
//                      if (v >= 0.0 && (u + v) <= 1.0) {
                            float hitDist = dot(edge2, originCrossEdge1) * inverseDeterminant;
//                          float hitDist = dot(edge2, originCrossEdge1) * inverseDeterminant;
                            if (hitDist > 0.001 && hitDist < maxDistance - 0.001) {
//                          if (hitDist > 0.001 && hitDist < maxDistance - 0.001) {
                                int matIdx = int(vertices[triangleIndex * 3].tangentAndMaterialIndex.w);
//                              int matIdx = int(vertices[triangleIndex * 3].tangentAndMaterialIndex.w);
                                if (materials[matIdx].transmission <= 0.5) {
//                              if (materials[matIdx].transmission <= 0.5) {
                                    return true;
//                                  return true;
                                }
//                              }
                            }
//                          }
                        }
//                      }
                    }
//                  }
                }
//              }
            } else {
//          } else {
                if (stackPointer < 62) {
//              if (stackPointer < 62) {
                    stack[stackPointer++] = int(node.aabbMinAndLeftChild.w);
//                  stack[stackPointer++] = int(node.aabbMinAndLeftChild.w);
                    stack[stackPointer++] = int(node.aabbMaxAndRightChild.w);
//                  stack[stackPointer++] = int(node.aabbMaxAndRightChild.w);
                }
//              }
            }
//          }
        }
//      }
        return false;
//      return false;
    }
//  }

    bool traverseClosestHit(vec3 origin, vec3 direction, float maxDistance, out vec3 outHitPoint, out vec3 outNormal, out int outTriangleIndex) {
//  bool traverseClosestHit(vec3 origin, vec3 direction, float maxDistance, out vec3 outHitPoint, out vec3 outNormal, out int outTriangleIndex) {
        vec3 directionSign = sign(direction);
//      vec3 directionSign = sign(direction);
        directionSign += 1.0 - abs(directionSign);
//      directionSign += 1.0 - abs(directionSign);
        vec3 safeDirection = direction + step(abs(direction), vec3(1e-8)) * directionSign * 1e-8;
//      vec3 safeDirection = direction + step(abs(direction), vec3(1e-8)) * directionSign * 1e-8;
        vec3 invDir = 1.0 / safeDirection;
//      vec3 invDir = 1.0 / safeDirection;
        int stack[64];
//      int stack[64];
        int stackPointer = 0;
//      int stackPointer = 0;
        stack[stackPointer++] = 0;
//      stack[stackPointer++] = 0;

        float closestDist = maxDistance;
//      float closestDist = maxDistance;
        int hitTriangle = -1;
//      int hitTriangle = -1;
        float bestU = 0.0;
//      float bestU = 0.0;
        float bestV = 0.0;
//      float bestV = 0.0;
        vec3 bestNormal = vec3(0.0);
//      vec3 bestNormal = vec3(0.0);

        while (stackPointer > 0) {
//      while (stackPointer > 0) {
            int nodeIndex = stack[--stackPointer];
//          int nodeIndex = stack[--stackPointer];
            Node node = nodes[nodeIndex];
//          Node node = nodes[nodeIndex];

            if (!intersectAABB(origin, invDir, node.aabbMinAndLeftChild.xyz, node.aabbMaxAndRightChild.xyz, closestDist)) {
//          if (!intersectAABB(origin, invDir, node.aabbMinAndLeftChild.xyz, node.aabbMaxAndRightChild.xyz, closestDist)) {
                continue;
//              continue;
            }
//          }

            if (node.aabbMinAndLeftChild.w < 0.0) {
//          if (node.aabbMinAndLeftChild.w < 0.0) {
                int triangleIndex = int(node.aabbMaxAndRightChild.w);
//              int triangleIndex = int(node.aabbMaxAndRightChild.w);
                vec3 v0 = vertices[triangleIndex * 3 + 0].positionAndTexcoordU.xyz;
//              vec3 v0 = vertices[triangleIndex * 3 + 0].positionAndTexcoordU.xyz;
                vec3 v1 = vertices[triangleIndex * 3 + 1].positionAndTexcoordU.xyz;
//              vec3 v1 = vertices[triangleIndex * 3 + 1].positionAndTexcoordU.xyz;
                vec3 v2 = vertices[triangleIndex * 3 + 2].positionAndTexcoordU.xyz;
//              vec3 v2 = vertices[triangleIndex * 3 + 2].positionAndTexcoordU.xyz;

                vec3 edge1 = v1 - v0;
//              vec3 edge1 = v1 - v0;
                vec3 edge2 = v2 - v0;
//              vec3 edge2 = v2 - v0;
                vec3 rayCrossEdge2 = cross(direction, edge2);
//              vec3 rayCrossEdge2 = cross(direction, edge2);
                float determinant = dot(edge1, rayCrossEdge2);
//              float determinant = dot(edge1, rayCrossEdge2);

                if (abs(determinant) > 1.0e-7) {
//              if (abs(determinant) > 1.0e-7) {
                    float inverseDeterminant = 1.0 / determinant;
//                  float inverseDeterminant = 1.0 / determinant;
                    vec3 originToVertex0 = origin - v0;
//                  vec3 originToVertex0 = origin - v0;
                    float u = dot(originToVertex0, rayCrossEdge2) * inverseDeterminant;
//                  float u = dot(originToVertex0, rayCrossEdge2) * inverseDeterminant;
                    if (u >= 0.0 && u <= 1.0) {
//                  if (u >= 0.0 && u <= 1.0) {
                        vec3 originCrossEdge1 = cross(originToVertex0, edge1);
//                      vec3 originCrossEdge1 = cross(originToVertex0, edge1);
                        float v = dot(direction, originCrossEdge1) * inverseDeterminant;
//                      float v = dot(direction, originCrossEdge1) * inverseDeterminant;
                        if (v >= 0.0 && (u + v) <= 1.0) {
//                      if (v >= 0.0 && (u + v) <= 1.0) {
                            float hitDist = dot(edge2, originCrossEdge1) * inverseDeterminant;
//                          float hitDist = dot(edge2, originCrossEdge1) * inverseDeterminant;
                            if (hitDist > 0.001 && hitDist < closestDist) {
//                          if (hitDist > 0.001 && hitDist < closestDist) {
                                closestDist = hitDist;
//                              closestDist = hitDist;
                                hitTriangle = triangleIndex;
//                              hitTriangle = triangleIndex;
                                bestU = u;
//                              bestU = u;
                                bestV = v;
//                              bestV = v;
                                vec3 geomNorm = normalize(cross(edge1, edge2));
//                              vec3 geomNorm = normalize(cross(edge1, edge2));
                                bestNormal = (dot(geomNorm, -direction) > 0.0) ? geomNorm : -geomNorm;
//                              bestNormal = (dot(geomNorm, -direction) > 0.0) ? geomNorm : -geomNorm;
                            }
//                          }
                        }
//                      }
                    }
//                  }
                }
//              }
            } else {
//          } else {
                if (stackPointer < 62) {
//              if (stackPointer < 62) {
                    stack[stackPointer++] = int(node.aabbMinAndLeftChild.w);
//                  stack[stackPointer++] = int(node.aabbMinAndLeftChild.w);
                    stack[stackPointer++] = int(node.aabbMaxAndRightChild.w);
//                  stack[stackPointer++] = int(node.aabbMaxAndRightChild.w);
                }
//              }
            }
//          }
        }
//      }

        outTriangleIndex = hitTriangle;
//      outTriangleIndex = hitTriangle;

        if (hitTriangle != -1) {
//      if (hitTriangle != -1) {
            outHitPoint = origin + direction * closestDist;
//          outHitPoint = origin + direction * closestDist;
            vec3 n0 = vertices[hitTriangle * 3 + 0].normalAndTexcoordV.xyz;
//          vec3 n0 = vertices[hitTriangle * 3 + 0].normalAndTexcoordV.xyz;
            vec3 n1 = vertices[hitTriangle * 3 + 1].normalAndTexcoordV.xyz;
//          vec3 n1 = vertices[hitTriangle * 3 + 1].normalAndTexcoordV.xyz;
            vec3 n2 = vertices[hitTriangle * 3 + 2].normalAndTexcoordV.xyz;
//          vec3 n2 = vertices[hitTriangle * 3 + 2].normalAndTexcoordV.xyz;
            vec3 interpolatedNormal = (1.0 - bestU - bestV) * n0 + bestU * n1 + bestV * n2;
//          vec3 interpolatedNormal = (1.0 - bestU - bestV) * n0 + bestU * n1 + bestV * n2;
            if (dot(interpolatedNormal, interpolatedNormal) > 1.0e-6) {
//          if (dot(interpolatedNormal, interpolatedNormal) > 1.0e-6) {
                vec3 smoothNormal = normalize(interpolatedNormal);
//              vec3 smoothNormal = normalize(interpolatedNormal);
                outNormal = (dot(smoothNormal, -direction) > 0.0) ? smoothNormal : -smoothNormal;
//              outNormal = (dot(smoothNormal, -direction) > 0.0) ? smoothNormal : -smoothNormal;
            } else {
//          } else {
                outNormal = bestNormal;
//              outNormal = bestNormal;
            }
//          }
            return true;
//          return true;
        }
//      }
        return false;
//      return false;
    }
//  }

    // Direct + one-bounce-into-the-cache estimate of the outgoing radiance leaving a surface point.
//  // Direct + one-bounce-into-the-cache estimate of the outgoing radiance leaving a surface point.
    // This is the regression target: the network learns L_out, so callers add it as-is without
//  // This is the regression target: the network learns L_out, so callers add it as-is without
    // re-applying the surface albedo.
//  // re-applying the surface albedo.
    vec3 computeTrainingTarget(vec3 surfacePoint, vec3 surfaceNormal, vec3 albedo) {
//  vec3 computeTrainingTarget(vec3 surfacePoint, vec3 surfaceNormal, vec3 albedo) {
        vec3 targetRadiance = vec3(0.0);
//      vec3 targetRadiance = vec3(0.0);

        // 1. Direct illumination estimate (single stochastic light, Lambertian lobe)
//      // 1. Direct illumination estimate (single stochastic light, Lambertian lobe)
        if (uPointLightCount > 0) {
//      if (uPointLightCount > 0) {
            float lightRand = randFloat();
//          float lightRand = randFloat();
            int lightIndex = max(0, uPointLightCount - 1);
//          int lightIndex = max(0, uPointLightCount - 1);
            for (int i = 0; i < uPointLightCount; i++) {
//          for (int i = 0; i < uPointLightCount; i++) {
                if (lightRand <= uPointLights[i].cdf) {
//              if (lightRand <= uPointLights[i].cdf) {
                    lightIndex = i;
//                  lightIndex = i;
                    break;
//                  break;
                }
//              }
            }
//          }

            vec3 lightPos = uPointLights[lightIndex].position;
//          vec3 lightPos = uPointLights[lightIndex].position;
            float lightRadius = uPointLights[lightIndex].radius;
//          float lightRadius = uPointLights[lightIndex].radius;
            vec3 toLight = lightPos - surfacePoint;
//          vec3 toLight = lightPos - surfacePoint;
            float distToLight = length(toLight);
//          float distToLight = length(toLight);
            vec3 lightDir = toLight / max(distToLight, 1.0e-4);
//          vec3 lightDir = toLight / max(distToLight, 1.0e-4);

            float cosTheta = max(0.0, dot(surfaceNormal, lightDir));
//          float cosTheta = max(0.0, dot(surfaceNormal, lightDir));
            if (cosTheta > 0.0) {
//          if (cosTheta > 0.0) {
                if (!traverseShadowRay(surfacePoint + surfaceNormal * EPSILON_OFFSET, lightDir, distToLight)) {
//              if (!traverseShadowRay(surfacePoint + surfaceNormal * EPSILON_OFFSET, lightDir, distToLight)) {
                    float lightPdf = max(uPointLights[lightIndex].pdf, 1.0e-4);
//                  float lightPdf = max(uPointLights[lightIndex].pdf, 1.0e-4);
                    float lightProjectedArea = PI * lightRadius * lightRadius;
//                  float lightProjectedArea = PI * lightRadius * lightRadius;
                    float clampedDist = max(distToLight, lightRadius + 0.05);
//                  float clampedDist = max(distToLight, lightRadius + 0.05);
                    float distSq = clampedDist * clampedDist;
//                  float distSq = clampedDist * clampedDist;
                    targetRadiance += (albedo / PI) * uPointLights[lightIndex].color * lightProjectedArea * (cosTheta / distSq) / lightPdf;
//                  targetRadiance += (albedo / PI) * uPointLights[lightIndex].color * lightProjectedArea * (cosTheta / distSq) / lightPdf;
                }
//              }
            }
//          }
        }
//      }

        // 2. Indirect illumination bootstrapped from the current cache through one cosine-weighted bounce.
//      // 2. Indirect illumination bootstrapped from the current cache through one cosine-weighted bounce.
        // With a cosine-weighted pdf the (albedo / PI) * cos / pdf factor collapses to a bare albedo.
//      // With a cosine-weighted pdf the (albedo / PI) * cos / pdf factor collapses to a bare albedo.
        vec3 bounceDirUnnormalized = surfaceNormal + randomUnitVector();
//      vec3 bounceDirUnnormalized = surfaceNormal + randomUnitVector();
        vec3 randomBounceDir = (dot(bounceDirUnnormalized, bounceDirUnnormalized) > 1.0e-6) ? normalize(bounceDirUnnormalized) : surfaceNormal;
//      vec3 randomBounceDir = (dot(bounceDirUnnormalized, bounceDirUnnormalized) > 1.0e-6) ? normalize(bounceDirUnnormalized) : surfaceNormal;
        vec3 nextHitPoint;
//      vec3 nextHitPoint;
        vec3 nextHitNormal;
//      vec3 nextHitNormal;
        int nextTriangleIndex;
//      int nextTriangleIndex;
        if (traverseClosestHit(surfacePoint + surfaceNormal * EPSILON_OFFSET, randomBounceDir, INF, nextHitPoint, nextHitNormal, nextTriangleIndex)) {
//      if (traverseClosestHit(surfacePoint + surfaceNormal * EPSILON_OFFSET, randomBounceDir, INF, nextHitPoint, nextHitNormal, nextTriangleIndex)) {
            targetRadiance += albedo * evaluateNeuralRadianceCache(nextHitPoint, nextHitNormal);
//          targetRadiance += albedo * evaluateNeuralRadianceCache(nextHitPoint, nextHitNormal);
        }
//      }

        return min(targetRadiance, vec3(15.0));
//      return min(targetRadiance, vec3(15.0));
    }
//  }

    void main() {
//  void main() {
        // Stage the NRC EMA weights in shared memory before any early return, so that every
//      // Stage the NRC EMA weights in shared memory before any early return, so that every
        // invocation in the workgroup reaches barrier().
//      // invocation in the workgroup reaches barrier().
        for (uint weightIndex = gl_LocalInvocationIndex; weightIndex < 1139u; weightIndex += 64u) {
//      for (uint weightIndex = gl_LocalInvocationIndex; weightIndex < 1139u; weightIndex += 64u) {
            shared_network_weights[weightIndex] = neuralNetworkWeights[1139u + weightIndex];
//          shared_network_weights[weightIndex] = neuralNetworkWeights[1139u + weightIndex];
        }
//      }
        barrier();
//      barrier();

        gRngState = gl_GlobalInvocationID.x + uTrainingStep * 1000000u + 1337u;
//      gRngState = gl_GlobalInvocationID.x + uTrainingStep * 1000000u + 1337u;
        randUint();
//      randUint();

        // Start from a random primary-visible surface taken out of the G-Buffer.
//      // Start from a random primary-visible surface taken out of the G-Buffer.
        ivec2 pixelCoord = ivec2(
//      ivec2 pixelCoord = ivec2(
            clamp(int(randFloat() * float(uResolution.x)), 0, uResolution.x - 1),
//          clamp(int(randFloat() * float(uResolution.x)), 0, uResolution.x - 1),
            clamp(int(randFloat() * float(uResolution.y)), 0, uResolution.y - 1)
//          clamp(int(randFloat() * float(uResolution.y)), 0, uResolution.y - 1)
        );
//      );

        vec4 posSample = imageLoad(textureGeometryGlobalPosition, pixelCoord);
//      vec4 posSample = imageLoad(textureGeometryGlobalPosition, pixelCoord);
        if (posSample.w == 0.0) return;
//      if (posSample.w == 0.0) return;

        int triangleIndex = int(posSample.w) - 1;
//      int triangleIndex = int(posSample.w) - 1;
        if (triangleIndex < 0) return;
//      if (triangleIndex < 0) return;

        vec4 normSample = imageLoad(textureGeometryGlobalNormal, pixelCoord);
//      vec4 normSample = imageLoad(textureGeometryGlobalNormal, pixelCoord);
        if (dot(normSample.xyz, normSample.xyz) < 1.0e-6) return;
//      if (dot(normSample.xyz, normSample.xyz) < 1.0e-6) return;

        vec3 currentPoint = posSample.xyz;
//      vec3 currentPoint = posSample.xyz;
        vec3 currentNormal = normalize(normSample.xyz);
//      vec3 currentNormal = normalize(normSample.xyz);
        vec3 currentAlbedo = imageLoad(textureGeometryAlbedo, pixelCoord).rgb;
//      vec3 currentAlbedo = imageLoad(textureGeometryAlbedo, pixelCoord).rgb;
        int currentTriangle = triangleIndex;
//      int currentTriangle = triangleIndex;

        // Walk the path and record a single training vertex along it. Recording at deeper vertices is what
//      // Walk the path and record a single training vertex along it. Recording at deeper vertices is what
        // keeps the training distribution aligned with where hybrid_shading_cs.glsl queries the cache,
//      // keeps the training distribution aligned with where hybrid_shading_cs.glsl queries the cache,
        // which is at secondary bounces that are frequently not visible from the camera at all.
//      // which is at secondary bounces that are frequently not visible from the camera at all.
        for (uint bounce = 0u; bounce < 6u; bounce++) {
//      for (uint bounce = 0u; bounce < 6u; bounce++) {
            int materialIndex = int(vertices[currentTriangle * 3].tangentAndMaterialIndex.w);
//          int materialIndex = int(vertices[currentTriangle * 3].tangentAndMaterialIndex.w);
            Material material = materials[materialIndex];
//          Material material = materials[materialIndex];

            // Emitters are seen directly by the path tracer; caching them would double count.
//          // Emitters are seen directly by the path tracer; caching them would double count.
            if (material.emissive > 0.0 || material.textureIndexEmissive >= 0.0) break;
//          if (material.emissive > 0.0 || material.textureIndexEmissive >= 0.0) break;

            // This pass has no texture sampler, so a material whose metallic/roughness comes from a map
//          // This pass has no texture sampler, so a material whose metallic/roughness comes from a map
            // only exposes the 1.0 placeholder scalar. Trusting that placeholder would classify most of
//          // only exposes the 1.0 placeholder scalar. Trusting that placeholder would classify most of
            // the scene as a mirror and collect no training records at all, so fall back to neutral
//          // the scene as a mirror and collect no training records at all, so fall back to neutral
            // values whenever the scalar is not authoritative.
//          // values whenever the scalar is not authoritative.
            bool metallicIsAuthoritative = (material.textureIndexMetallic < -0.5);
//          bool metallicIsAuthoritative = (material.textureIndexMetallic < -0.5);
            bool roughnessIsAuthoritative = (material.textureIndexRoughness < -0.5);
//          bool roughnessIsAuthoritative = (material.textureIndexRoughness < -0.5);
            float effectiveMetallic = metallicIsAuthoritative ? material.metallic : 0.0;
//          float effectiveMetallic = metallicIsAuthoritative ? material.metallic : 0.0;
            float effectiveRoughness = roughnessIsAuthoritative ? material.roughness : 0.5;
//          float effectiveRoughness = roughnessIsAuthoritative ? material.roughness : 0.5;

            // Only cache broadly diffuse dielectrics; mirrors and glass are view dependent.
//          // Only cache broadly diffuse dielectrics; mirrors and glass are view dependent.
            bool isCacheable = (effectiveMetallic <= 0.8 && material.transmission <= 0.5 && effectiveRoughness >= 0.05);
//          bool isCacheable = (effectiveMetallic <= 0.8 && material.transmission <= 0.5 && effectiveRoughness >= 0.05);

            if (isCacheable) {
//          if (isCacheable) {
                vec3 targetRadiance = computeTrainingTarget(currentPoint, currentNormal, currentAlbedo);
//              vec3 targetRadiance = computeTrainingTarget(currentPoint, currentNormal, currentAlbedo);

                float acceptanceProb = 1.0;
//              float acceptanceProb = 1.0;

                if (bounce == 0u) {
//              if (bounce == 0u) {
                    // 1. Material-aware adaptive sampling: smooth surfaces act as mirrors and need the
//                  // 1. Material-aware adaptive sampling: smooth surfaces act as mirrors and need the
                    //    cache for screen-space reflections, rough ones are already served well by ReSTIR DI.
//                  //    cache for screen-space reflections, rough ones are already served well by ReSTIR DI.
                    float materialProb = mix(0.80, 0.10, effectiveRoughness);
//                  float materialProb = mix(0.80, 0.10, effectiveRoughness);

                    // 2. Luminance-driven sampling: high-energy areas dominate the variance.
//                  // 2. Luminance-driven sampling: high-energy areas dominate the variance.
                    float targetLum = calculateLuminance(targetRadiance);
//                  float targetLum = calculateLuminance(targetRadiance);
                    acceptanceProb = clamp(materialProb + targetLum * 0.15, 0.10, 1.0);
//                  acceptanceProb = clamp(materialProb + targetLum * 0.15, 0.10, 1.0);
                }
//              }

                // Stratified sampling (adaptive Russian-roulette acceptance): rejecting here pushes the
//              // Stratified sampling (adaptive Russian-roulette acceptance): rejecting here pushes the
                // record deeper along the path instead of throwing the whole path away.
//              // record deeper along the path instead of throwing the whole path away.
                if (randFloat() < acceptanceProb) {
//              if (randFloat() < acceptanceProb) {
                    uint storeIndex = atomicAdd(uTrainingRecordCount, 1u);
//                  uint storeIndex = atomicAdd(uTrainingRecordCount, 1u);

                    if (storeIndex < 8192u) {
//                  if (storeIndex < 8192u) {
                        uint baseOffset = storeIndex * 9u;
//                      uint baseOffset = storeIndex * 9u;
                        nrcTrainingRecords[baseOffset + 0u] = currentPoint.x;
//                      nrcTrainingRecords[baseOffset + 0u] = currentPoint.x;
                        nrcTrainingRecords[baseOffset + 1u] = currentPoint.y;
//                      nrcTrainingRecords[baseOffset + 1u] = currentPoint.y;
                        nrcTrainingRecords[baseOffset + 2u] = currentPoint.z;
//                      nrcTrainingRecords[baseOffset + 2u] = currentPoint.z;
                        nrcTrainingRecords[baseOffset + 3u] = currentNormal.x;
//                      nrcTrainingRecords[baseOffset + 3u] = currentNormal.x;
                        nrcTrainingRecords[baseOffset + 4u] = currentNormal.y;
//                      nrcTrainingRecords[baseOffset + 4u] = currentNormal.y;
                        nrcTrainingRecords[baseOffset + 5u] = currentNormal.z;
//                      nrcTrainingRecords[baseOffset + 5u] = currentNormal.z;
                        nrcTrainingRecords[baseOffset + 6u] = targetRadiance.x;
//                      nrcTrainingRecords[baseOffset + 6u] = targetRadiance.x;
                        nrcTrainingRecords[baseOffset + 7u] = targetRadiance.y;
//                      nrcTrainingRecords[baseOffset + 7u] = targetRadiance.y;
                        nrcTrainingRecords[baseOffset + 8u] = targetRadiance.z;
//                      nrcTrainingRecords[baseOffset + 8u] = targetRadiance.z;
                    }
//                  }
                    break; // One record per path
//                  break; // One record per path
                }
//              }
            }
//          }

            // Continue the path with a cosine-weighted bounce off the current surface.
//          // Continue the path with a cosine-weighted bounce off the current surface.
            vec3 continueDirUnnormalized = currentNormal + randomUnitVector();
//          vec3 continueDirUnnormalized = currentNormal + randomUnitVector();
            vec3 continueDir = (dot(continueDirUnnormalized, continueDirUnnormalized) > 1.0e-6) ? normalize(continueDirUnnormalized) : currentNormal;
//          vec3 continueDir = (dot(continueDirUnnormalized, continueDirUnnormalized) > 1.0e-6) ? normalize(continueDirUnnormalized) : currentNormal;

            vec3 nextPoint;
//          vec3 nextPoint;
            vec3 nextNormal;
//          vec3 nextNormal;
            int nextTriangle;
//          int nextTriangle;
            if (!traverseClosestHit(currentPoint + currentNormal * EPSILON_OFFSET, continueDir, INF, nextPoint, nextNormal, nextTriangle)) break;
//          if (!traverseClosestHit(currentPoint + currentNormal * EPSILON_OFFSET, continueDir, INF, nextPoint, nextNormal, nextTriangle)) break;
            if (nextTriangle < 0) break;
//          if (nextTriangle < 0) break;

            currentPoint = nextPoint;
//          currentPoint = nextPoint;
            currentNormal = nextNormal;
//          currentNormal = nextNormal;
            currentTriangle = nextTriangle;
//          currentTriangle = nextTriangle;
            currentAlbedo = materials[int(vertices[nextTriangle * 3].tangentAndMaterialIndex.w)].albedo.rgb;
//          currentAlbedo = materials[int(vertices[nextTriangle * 3].tangentAndMaterialIndex.w)].albedo.rgb;
        }
//      }
    }
//  }
