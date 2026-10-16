    // Shared modules: principled_bsdf so the cache is trained against the same BSDF hybrid_shading_cs.slang
//  // Shared modules: principled_bsdf so the cache is trained against the same BSDF hybrid_shading_cs.slang
    // later evaluates it with, sky_environment for the same environment radiance the shading pass gathers,
//  // later evaluates it with, sky_environment for the same environment radiance the shading pass gathers,
    // and the participating media for shadow-ray transmittance.
//  // and the participating media for shadow-ray transmittance.
    import math_constants;
//  import math_constants;
    import principled_bsdf;
//  import principled_bsdf;
    import sky_environment;
//  import sky_environment;
    import volumetric_common;
//  import volumetric_common;
    import participating_media;
//  import participating_media;

    Texture2D<float4> textureGeometryGlobalPosition;
//  Texture2D<float4> textureGeometryGlobalPosition;
    Texture2D<float4> textureGeometryGlobalNormal;
//  Texture2D<float4> textureGeometryGlobalNormal;
    Texture2D<float4> textureGeometryAlbedo;
//  Texture2D<float4> textureGeometryAlbedo;
    // Only the w channel is used, and only for the primary vertex: hybrid_shading_cs.slang packs the
//  // Only the w channel is used, and only for the primary vertex: hybrid_shading_cs.slang packs the
    // interpolated texcoord V there (U lives in the normal target), which is what the material
//  // interpolated texcoord V there (U lives in the normal target), which is what the material
    // resolution below needs to sample the same maps the shading pass shades that pixel with.
//  // resolution below needs to sample the same maps the shading pass shades that pixel with.
    Texture2D<float4> textureGeometryGlobalTangent;
//  Texture2D<float4> textureGeometryGlobalTangent;

    struct Node {
//  struct Node {
        float4 aabbMinAndLeftChild;
//      float4 aabbMinAndLeftChild;
        float4 aabbMaxAndRightChild;
//      float4 aabbMaxAndRightChild;
    };
//  };

    StructuredBuffer<Node> nodes;
//  StructuredBuffer<Node> nodes;

    struct VertexData {
//  struct VertexData {
        float4 positionAndTexcoordU;
//      float4 positionAndTexcoordU;
        float4 normalAndTexcoordV;
//      float4 normalAndTexcoordV;
        float4 tangentAndMaterialIndex;
//      float4 tangentAndMaterialIndex;
    };
//  };

    StructuredBuffer<VertexData> vertices;
//  StructuredBuffer<VertexData> vertices;

    struct Material {
//  struct Material {
        float4 albedo;
//      float4 albedo;
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
        float2 uvScale;
//      float2 uvScale;
        float2 padding003;
//      float2 padding003;
    };
//  };

    StructuredBuffer<Material> materials;
//  StructuredBuffer<Material> materials;

    StructuredBuffer<float> neuralNetworkWeights;
//  StructuredBuffer<float> neuralNetworkWeights;

    RWStructuredBuffer<float> nrcTrainingRecords;
//  RWStructuredBuffer<float> nrcTrainingRecords;

    RWStructuredBuffer<Atomic<uint>> trainingRecordCounter;
//  RWStructuredBuffer<Atomic<uint>> trainingRecordCounter;

    struct PointLight {
//  struct PointLight {
        float3 position;
//      float3 position;
        float radius;
//      float radius;
        float3 color;
//      float3 color;
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
    uniform int2 uResolution;
//  uniform int2 uResolution;
    // The training target evaluates a view-dependent BSDF, so this pass needs the direction each path
//  // The training target evaluates a view-dependent BSDF, so this pass needs the direction each path
    // vertex was reached along. For the first vertex, taken from the G-Buffer, that is the camera ray.
//  // vertex was reached along. For the first vertex, taken from the G-Buffer, that is the camera ray.
    uniform float3 uCameraGlobalPosition;
//  uniform float3 uCameraGlobalPosition;
    // NRC input-domain normalisation; must match hybrid_shading_cs.slang and nrc_train_cs.slang exactly.
//  // NRC input-domain normalisation; must match hybrid_shading_cs.slang and nrc_train_cs.slang exactly.
    uniform float3 uNRCPositionOffset;
//  uniform float3 uNRCPositionOffset;
    uniform float uNRCPositionScale;
//  uniform float uNRCPositionScale;

    // Environment lighting inputs (HDRI, baked cloud layer) and the cloud noise volume belong to the imported
//  // Environment lighting inputs (HDRI, baked cloud layer) and the cloud noise volume belong to the imported
    // sky_environment / volumetric_common modules. The cache regresses total outgoing radiance, so the
//  // sky_environment / volumetric_common modules. The cache regresses total outgoing radiance, so the
    // training target has to see the same cloudy sky hybrid_shading_cs.slang gathers on its secondary bounces.
//  // training target has to see the same cloudy sky hybrid_shading_cs.slang gathers on its secondary bounces.

    // The scene texture array, shared with hybrid_shading_cs.slang. Without it this pass could only see
//  // The scene texture array, shared with hybrid_shading_cs.slang. Without it this pass could only see
    // a material's scalar roughness/metallic/albedo, which on any textured material is a placeholder
//  // a material's scalar roughness/metallic/albedo, which on any textured material is a placeholder
    // rather than the value the shading pass actually shades with. That mismatch used to be masked by
//  // rather than the value the shading pass actually shades with. That mismatch used to be masked by
    // the exact Next Event Estimation term at the vertex the cache terminates a path on; since that
//  // the exact Next Event Estimation term at the vertex the cache terminates a path on; since that
    // term is (correctly) suppressed there, the network is the sole source of direct light at that
//  // term is (correctly) suppressed there, the network is the sole source of direct light at that
    // vertex and the target has to be built from the same material the renderer will ask about.
//  // vertex and the target has to be built from the same material the renderer will ask about.
    Sampler2DArray<float4> uSceneTextureArray;
//  Sampler2DArray<float4> uSceneTextureArray;

    // Workgroup-resident copy of the NRC Exponential Moving Average weights, filled once per
//  // Workgroup-resident copy of the NRC Exponential Moving Average weights, filled once per
    // workgroup at the top of computeMain() so every cache query reads shared memory instead of the SSBO.
//  // workgroup at the top of computeMain() so every cache query reads shared memory instead of the SSBO.
    groupshared float shared_network_weights[1139];
//  groupshared float shared_network_weights[1139];

    static const float EPSILON_OFFSET = 0.001;
//  static const float EPSILON_OFFSET = 0.001;

    static uint gRngState;
//  static uint gRngState;

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

    float3 randomUnitVector() {
//  float3 randomUnitVector() {
        float cosPolar = randFloat() * 2.0 - 1.0;
//      float cosPolar = randFloat() * 2.0 - 1.0;
        float azimuthAngle = randFloat() * TWO_PI;
//      float azimuthAngle = randFloat() * TWO_PI;
        float sinPolar = sqrt(max(0.0, 1.0 - cosPolar * cosPolar));
//      float sinPolar = sqrt(max(0.0, 1.0 - cosPolar * cosPolar));
        return float3(sinPolar * cos(azimuthAngle), sinPolar * sin(azimuthAngle), cosPolar);
//      return float3(sinPolar * cos(azimuthAngle), sinPolar * sin(azimuthAngle), cosPolar);
    }
//  }

    // This pass's random source for the participating media trackers. Only the shadow-ray transmittance
//  // This pass's random source for the participating media trackers. Only the shadow-ray transmittance
    // is used: the training target then carries media shadows on the direct light, while its bounce rays
//  // is used: the training target then carries media shadows on the direct light, while its bounce rays
    // still cross media unscattered.
//  // still cross media unscattered.
    struct GatherRandom : IMediumRandom {
//  struct GatherRandom : IMediumRandom {
        static float next() {
//      static float next() {
            return randFloat();
//          return randFloat();
        }
//      }
    }
//  }

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

    bool intersectAABB(float3 rayOrigin, float3 rayInverseDirection, float3 aabbMin, float3 aabbMax, float maxDistance) {
//  bool intersectAABB(float3 rayOrigin, float3 rayInverseDirection, float3 aabbMin, float3 aabbMax, float maxDistance) {
        float3 t0 = (aabbMin - rayOrigin) * rayInverseDirection;
//      float3 t0 = (aabbMin - rayOrigin) * rayInverseDirection;
        float3 t1 = (aabbMax - rayOrigin) * rayInverseDirection;
//      float3 t1 = (aabbMax - rayOrigin) * rayInverseDirection;
        float3 tmin = min(t0, t1);
//      float3 tmin = min(t0, t1);
        float3 tmax = max(t0, t1);
//      float3 tmax = max(t0, t1);
        float enterDistance = max(max(tmin.x, tmin.y), max(tmin.z, 0.001));
//      float enterDistance = max(max(tmin.x, tmin.y), max(tmin.z, 0.001));
        float exitDistance = min(min(tmax.x, tmax.y), min(tmax.z, maxDistance));
//      float exitDistance = min(min(tmax.x, tmax.y), min(tmax.z, maxDistance));
        return enterDistance <= exitDistance;
//      return enterDistance <= exitDistance;
    }
//  }

    bool traverseShadowRay(float3 origin, float3 direction, float maxDistance) {
//  bool traverseShadowRay(float3 origin, float3 direction, float maxDistance) {
        float3 directionSign = sign(direction);
//      float3 directionSign = sign(direction);
        directionSign += 1.0 - abs(directionSign);
//      directionSign += 1.0 - abs(directionSign);
        float3 safeDirection = direction + step(abs(direction), float3(1e-8)) * directionSign * 1e-8;
//      float3 safeDirection = direction + step(abs(direction), float3(1e-8)) * directionSign * 1e-8;
        float3 invDir = 1.0 / safeDirection;
//      float3 invDir = 1.0 / safeDirection;
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
                float3 v0 = vertices[triangleIndex * 3 + 0].positionAndTexcoordU.xyz;
//              float3 v0 = vertices[triangleIndex * 3 + 0].positionAndTexcoordU.xyz;
                float3 v1 = vertices[triangleIndex * 3 + 1].positionAndTexcoordU.xyz;
//              float3 v1 = vertices[triangleIndex * 3 + 1].positionAndTexcoordU.xyz;
                float3 v2 = vertices[triangleIndex * 3 + 2].positionAndTexcoordU.xyz;
//              float3 v2 = vertices[triangleIndex * 3 + 2].positionAndTexcoordU.xyz;

                float3 edge1 = v1 - v0;
//              float3 edge1 = v1 - v0;
                float3 edge2 = v2 - v0;
//              float3 edge2 = v2 - v0;
                float3 rayCrossEdge2 = cross(direction, edge2);
//              float3 rayCrossEdge2 = cross(direction, edge2);
                float determinant = dot(edge1, rayCrossEdge2);
//              float determinant = dot(edge1, rayCrossEdge2);

                if (abs(determinant) > 1.0e-7) {
//              if (abs(determinant) > 1.0e-7) {
                    float inverseDeterminant = 1.0 / determinant;
//                  float inverseDeterminant = 1.0 / determinant;
                    float3 originToVertex0 = origin - v0;
//                  float3 originToVertex0 = origin - v0;
                    float u = dot(originToVertex0, rayCrossEdge2) * inverseDeterminant;
//                  float u = dot(originToVertex0, rayCrossEdge2) * inverseDeterminant;
                    if (u >= 0.0 && u <= 1.0) {
//                  if (u >= 0.0 && u <= 1.0) {
                        float3 originCrossEdge1 = cross(originToVertex0, edge1);
//                      float3 originCrossEdge1 = cross(originToVertex0, edge1);
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

    bool traverseClosestHit(float3 origin, float3 direction, float maxDistance, out float3 outHitPoint, out float3 outNormal, out int outTriangleIndex, out float2 outTexcoord, out float outDistance) {
//  bool traverseClosestHit(float3 origin, float3 direction, float maxDistance, out float3 outHitPoint, out float3 outNormal, out int outTriangleIndex, out float2 outTexcoord, out float outDistance) {
        float3 directionSign = sign(direction);
//      float3 directionSign = sign(direction);
        directionSign += 1.0 - abs(directionSign);
//      directionSign += 1.0 - abs(directionSign);
        float3 safeDirection = direction + step(abs(direction), float3(1e-8)) * directionSign * 1e-8;
//      float3 safeDirection = direction + step(abs(direction), float3(1e-8)) * directionSign * 1e-8;
        float3 invDir = 1.0 / safeDirection;
//      float3 invDir = 1.0 / safeDirection;
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
        float3 bestNormal = float3(0.0);
//      float3 bestNormal = float3(0.0);

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
                float3 v0 = vertices[triangleIndex * 3 + 0].positionAndTexcoordU.xyz;
//              float3 v0 = vertices[triangleIndex * 3 + 0].positionAndTexcoordU.xyz;
                float3 v1 = vertices[triangleIndex * 3 + 1].positionAndTexcoordU.xyz;
//              float3 v1 = vertices[triangleIndex * 3 + 1].positionAndTexcoordU.xyz;
                float3 v2 = vertices[triangleIndex * 3 + 2].positionAndTexcoordU.xyz;
//              float3 v2 = vertices[triangleIndex * 3 + 2].positionAndTexcoordU.xyz;

                float3 edge1 = v1 - v0;
//              float3 edge1 = v1 - v0;
                float3 edge2 = v2 - v0;
//              float3 edge2 = v2 - v0;
                float3 rayCrossEdge2 = cross(direction, edge2);
//              float3 rayCrossEdge2 = cross(direction, edge2);
                float determinant = dot(edge1, rayCrossEdge2);
//              float determinant = dot(edge1, rayCrossEdge2);

                if (abs(determinant) > 1.0e-7) {
//              if (abs(determinant) > 1.0e-7) {
                    float inverseDeterminant = 1.0 / determinant;
//                  float inverseDeterminant = 1.0 / determinant;
                    float3 originToVertex0 = origin - v0;
//                  float3 originToVertex0 = origin - v0;
                    float u = dot(originToVertex0, rayCrossEdge2) * inverseDeterminant;
//                  float u = dot(originToVertex0, rayCrossEdge2) * inverseDeterminant;
                    if (u >= 0.0 && u <= 1.0) {
//                  if (u >= 0.0 && u <= 1.0) {
                        float3 originCrossEdge1 = cross(originToVertex0, edge1);
//                      float3 originCrossEdge1 = cross(originToVertex0, edge1);
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
                                float3 geomNorm = normalize(cross(edge1, edge2));
//                              float3 geomNorm = normalize(cross(edge1, edge2));
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
        outTexcoord = float2(0.0);
//      outTexcoord = float2(0.0);
        outDistance = closestDist;
//      outDistance = closestDist;

        if (hitTriangle != -1) {
//      if (hitTriangle != -1) {
            outHitPoint = origin + direction * closestDist;
//          outHitPoint = origin + direction * closestDist;
            float3 n0 = vertices[hitTriangle * 3 + 0].normalAndTexcoordV.xyz;
//          float3 n0 = vertices[hitTriangle * 3 + 0].normalAndTexcoordV.xyz;
            float3 n1 = vertices[hitTriangle * 3 + 1].normalAndTexcoordV.xyz;
//          float3 n1 = vertices[hitTriangle * 3 + 1].normalAndTexcoordV.xyz;
            float3 n2 = vertices[hitTriangle * 3 + 2].normalAndTexcoordV.xyz;
//          float3 n2 = vertices[hitTriangle * 3 + 2].normalAndTexcoordV.xyz;
            // Texcoords ride in the w channels of the two position/normal vec4s, barycentric
//          // Texcoords ride in the w channels of the two position/normal vec4s, barycentric
            // interpolated with the same weights as the shading normal above.
//          // interpolated with the same weights as the shading normal above.
            float2 t0 = float2(vertices[hitTriangle * 3 + 0].positionAndTexcoordU.w, vertices[hitTriangle * 3 + 0].normalAndTexcoordV.w);
//          float2 t0 = float2(vertices[hitTriangle * 3 + 0].positionAndTexcoordU.w, vertices[hitTriangle * 3 + 0].normalAndTexcoordV.w);
            float2 t1 = float2(vertices[hitTriangle * 3 + 1].positionAndTexcoordU.w, vertices[hitTriangle * 3 + 1].normalAndTexcoordV.w);
//          float2 t1 = float2(vertices[hitTriangle * 3 + 1].positionAndTexcoordU.w, vertices[hitTriangle * 3 + 1].normalAndTexcoordV.w);
            float2 t2 = float2(vertices[hitTriangle * 3 + 2].positionAndTexcoordU.w, vertices[hitTriangle * 3 + 2].normalAndTexcoordV.w);
//          float2 t2 = float2(vertices[hitTriangle * 3 + 2].positionAndTexcoordU.w, vertices[hitTriangle * 3 + 2].normalAndTexcoordV.w);
            outTexcoord = (1.0 - bestU - bestV) * t0 + bestU * t1 + bestV * t2;
//          outTexcoord = (1.0 - bestU - bestV) * t0 + bestU * t1 + bestV * t2;
            float3 interpolatedNormal = (1.0 - bestU - bestV) * n0 + bestU * n1 + bestV * n2;
//          float3 interpolatedNormal = (1.0 - bestU - bestV) * n0 + bestU * n1 + bestV * n2;
            if (dot(interpolatedNormal, interpolatedNormal) > 1.0e-6) {
//          if (dot(interpolatedNormal, interpolatedNormal) > 1.0e-6) {
                float3 smoothNormal = normalize(interpolatedNormal);
//              float3 smoothNormal = normalize(interpolatedNormal);
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

    // Single-light direct illumination estimate at a surface point, using the same principled BSDF the
//  // Single-light direct illumination estimate at a surface point, using the same principled BSDF the
    // shading pass evaluates the cache with (principled_bsdf.slang). A Lambert-only lobe here would train
//  // shading pass evaluates the cache with (principled_bsdf.slang). A Lambert-only lobe here would train
    // the network on a target the renderer never asks for on anything glossy.
//  // the network on a target the renderer never asks for on anything glossy.
    //
//  //
    // The light is sampled analytically rather than stochastically: hybrid_shading_cs.slang samples the
//  // The light is sampled analytically rather than stochastically: hybrid_shading_cs.slang samples the
    // facing hemisphere (area 2*PI*r*r) and weights by cos at the light, whose expectation over that
//  // facing hemisphere (area 2*PI*r*r) and weights by cos at the light, whose expectation over that
    // hemisphere is 1/2, so the projected area PI*r*r used here is the same estimator with less variance.
//  // hemisphere is 1/2, so the projected area PI*r*r used here is the same estimator with less variance.
    // Texture-resolved surface parameters at a hit, mirroring scatterPrincipled in
//  // Texture-resolved surface parameters at a hit, mirroring scatterPrincipled in
    // hybrid_shading_cs.slang term for term: same packed-ORM channel assignment, same sRGB decode on
//  // hybrid_shading_cs.slang term for term: same packed-ORM channel assignment, same sRGB decode on
    // albedo and emissive, same distance-derived LOD, same MIN_ROUGHNESS floor. The two have to agree,
//  // albedo and emissive, same distance-derived LOD, same MIN_ROUGHNESS floor. The two have to agree,
    // because the network is trained here and evaluated there; a divergence would make it regress a
//  // because the network is trained here and evaluated there; a divergence would make it regress a
    // target the renderer never asks for, and would be invisible in either file on its own.
//  // target the renderer never asks for, and would be invisible in either file on its own.
    //
//  //
    // The only term deliberately not mirrored is the roughness anneal: pathRoughnessFloor is a
//  // The only term deliberately not mirrored is the roughness anneal: pathRoughnessFloor is a
    // property of the camera path being traced, and this pass traces its own short paths.
//  // property of the camera path being traced, and this pass traces its own short paths.
    void resolveSurfaceMaterial(Material material, float2 texcoord, float hitDistance,
//  void resolveSurfaceMaterial(Material material, float2 texcoord, float hitDistance,
                                out float3 outAlbedo, out float outRoughness, out float outMetallic,
//                              out float3 outAlbedo, out float outRoughness, out float outMetallic,
                                out float outTransmission, out float3 outEmission) {
//                              out float outTransmission, out float3 outEmission) {
        outAlbedo = material.albedo.rgb;
//      outAlbedo = material.albedo.rgb;
        outRoughness = material.roughness;
//      outRoughness = material.roughness;
        outMetallic = material.metallic;
//      outMetallic = material.metallic;
        outTransmission = material.transmission;
//      outTransmission = material.transmission;

        float textureLodLevel = max(0.0, log2(hitDistance * 0.1));
//      float textureLodLevel = max(0.0, log2(hitDistance * 0.1));
        float2 scaledTexcoord = texcoord * material.uvScale;
//      float2 scaledTexcoord = texcoord * material.uvScale;

        if (material.textureIndexAlbedo > -0.5) {
//      if (material.textureIndexAlbedo > -0.5) {
            outAlbedo *= pow(uSceneTextureArray.SampleLevel(float3(scaledTexcoord, material.textureIndexAlbedo), textureLodLevel).rgb, float3(2.2));
//          outAlbedo *= pow(uSceneTextureArray.SampleLevel(float3(scaledTexcoord, material.textureIndexAlbedo), textureLodLevel).rgb, float3(2.2));
        }
//      }
        if (material.textureIndexRoughness > -0.5) {
//      if (material.textureIndexRoughness > -0.5) {
            outRoughness = uSceneTextureArray.SampleLevel(float3(scaledTexcoord, material.textureIndexRoughness), textureLodLevel).r;
//          outRoughness = uSceneTextureArray.SampleLevel(float3(scaledTexcoord, material.textureIndexRoughness), textureLodLevel).r;
        }
//      }
        if (material.textureIndexMetallic > -0.5) {
//      if (material.textureIndexMetallic > -0.5) {
            outMetallic = uSceneTextureArray.SampleLevel(float3(scaledTexcoord, material.textureIndexMetallic), textureLodLevel).g;
//          outMetallic = uSceneTextureArray.SampleLevel(float3(scaledTexcoord, material.textureIndexMetallic), textureLodLevel).g;
        }
//      }
        if (material.textureIndexTransmission > -0.5) {
//      if (material.textureIndexTransmission > -0.5) {
            outTransmission *= uSceneTextureArray.SampleLevel(float3(scaledTexcoord, material.textureIndexTransmission), textureLodLevel).b;
//          outTransmission *= uSceneTextureArray.SampleLevel(float3(scaledTexcoord, material.textureIndexTransmission), textureLodLevel).b;
        }
//      }
        outRoughness = max(outRoughness, MIN_ROUGHNESS);
//      outRoughness = max(outRoughness, MIN_ROUGHNESS);

        if (material.textureIndexEmissive > -0.5) {
//      if (material.textureIndexEmissive > -0.5) {
            outEmission = material.emissive * pow(uSceneTextureArray.SampleLevel(float3(scaledTexcoord, material.textureIndexEmissive), textureLodLevel).rgb, float3(2.2));
//          outEmission = material.emissive * pow(uSceneTextureArray.SampleLevel(float3(scaledTexcoord, material.textureIndexEmissive), textureLodLevel).rgb, float3(2.2));
        } else {
//      } else {
            outEmission = material.emissive * outAlbedo;
//          outEmission = material.emissive * outAlbedo;
        }
//      }
    }
//  }

    float3 estimateDirectLighting(float3 surfacePoint, float3 surfaceNormal, float3 incomingDirection,
//  float3 estimateDirectLighting(float3 surfacePoint, float3 surfaceNormal, float3 incomingDirection,
                                float3 albedo, float roughness, float metallic, float transmission) {
//                              float3 albedo, float roughness, float metallic, float transmission) {
        if (uPointLightCount <= 0) return float3(0.0);
//      if (uPointLightCount <= 0) return float3(0.0);

        float lightRand = randFloat();
//      float lightRand = randFloat();
        int lightIndex = max(0, uPointLightCount - 1);
//      int lightIndex = max(0, uPointLightCount - 1);
        for (int i = 0; i < uPointLightCount; i++) {
//      for (int i = 0; i < uPointLightCount; i++) {
            if (lightRand <= uPointLights[i].cdf) {
//          if (lightRand <= uPointLights[i].cdf) {
                lightIndex = i;
//              lightIndex = i;
                break;
//              break;
            }
//          }
        }
//      }

        float3 lightPos = uPointLights[lightIndex].position;
//      float3 lightPos = uPointLights[lightIndex].position;
        float lightRadius = uPointLights[lightIndex].radius;
//      float lightRadius = uPointLights[lightIndex].radius;
        float3 toLight = lightPos - surfacePoint;
//      float3 toLight = lightPos - surfacePoint;
        float distToLight = length(toLight);
//      float distToLight = length(toLight);
        float3 lightDir = toLight / max(distToLight, 1.0e-4);
//      float3 lightDir = toLight / max(distToLight, 1.0e-4);

        float cosTheta = max(0.0, dot(surfaceNormal, lightDir));
//      float cosTheta = max(0.0, dot(surfaceNormal, lightDir));
        if (cosTheta <= 0.0) return float3(0.0);
//      if (cosTheta <= 0.0) return float3(0.0);
        if (traverseShadowRay(surfacePoint + surfaceNormal * EPSILON_OFFSET, lightDir, distToLight)) return float3(0.0);
//      if (traverseShadowRay(surfacePoint + surfaceNormal * EPSILON_OFFSET, lightDir, distToLight)) return float3(0.0);

        float lightPdf = max(uPointLights[lightIndex].pdf, 1.0e-4);
//      float lightPdf = max(uPointLights[lightIndex].pdf, 1.0e-4);
        float lightProjectedArea = PI * lightRadius * lightRadius;
//      float lightProjectedArea = PI * lightRadius * lightRadius;
        float clampedDist = max(distToLight, lightRadius + 0.05);
//      float clampedDist = max(distToLight, lightRadius + 0.05);
        float distSq = clampedDist * clampedDist;
//      float distSq = clampedDist * clampedDist;

        float unusedPdf;
//      float unusedPdf;
        float3 bsdfValue = evalPrincipledBSDFAndPDF(incomingDirection, lightDir, surfaceNormal,
//      float3 bsdfValue = evalPrincipledBSDFAndPDF(incomingDirection, lightDir, surfaceNormal,
                                                  albedo, roughness, metallic, transmission, unusedPdf);
//                                                albedo, roughness, metallic, transmission, unusedPdf);
        float shadowTransmittance = estimateMediumTransmittance<GatherRandom>(surfacePoint + surfaceNormal * EPSILON_OFFSET, lightDir, distToLight);
//      float shadowTransmittance = estimateMediumTransmittance<GatherRandom>(surfacePoint + surfaceNormal * EPSILON_OFFSET, lightDir, distToLight);
        return bsdfValue * uPointLights[lightIndex].color * lightProjectedArea * (cosTheta / distSq) / lightPdf * shadowTransmittance;
//      return bsdfValue * uPointLights[lightIndex].color * lightProjectedArea * (cosTheta / distSq) / lightPdf * shadowTransmittance;
    }
//  }

    // Estimate of the outgoing radiance leaving a surface point, and the regression target for the
//  // Estimate of the outgoing radiance leaving a surface point, and the regression target for the
    // network: direct illumination at this vertex, plus one cosine-weighted bounce that terminates at
//  // network: direct illumination at this vertex, plus one cosine-weighted bounce that terminates at
    // the first cacheable vertex with a bootstrap read of the current cache. The network learns L_out,
//  // the first cacheable vertex with a bootstrap read of the current cache. The network learns L_out,
    // so callers add the result as-is without re-applying the surface albedo.
//  // so callers add the result as-is without re-applying the surface albedo.
    //
//  //
    // No next event estimation runs at the bounce vertices, and that is deliberate: the cache already
//  // No next event estimation runs at the bounce vertices, and that is deliberate: the cache already
    // represents the full outgoing radiance at a vertex, direct light included, so an explicit
//  // represents the full outgoing radiance at a vertex, direct light included, so an explicit
    // direct-lighting estimate there would count one bounce of direct light twice (see the bootstrap
//  // direct-lighting estimate there would count one bounce of direct light twice (see the bootstrap
    // branch below). The loop runs up to three iterations only to trace through mirror and glass
//  // branch below). The loop runs up to three iterations only to trace through mirror and glass
    // vertices, where the cache is view dependent and cannot be read.
//  // vertices, where the cache is view dependent and cannot be read.
    //
//  //
    // A bounce that escapes the geometry ends on the environment, which is a genuine light source in
//  // A bounce that escapes the geometry ends on the environment, which is a genuine light source in
    // an open scene, so its radiance is added rather than dropped. Omitting it trains the network to
//  // an open scene, so its radiance is added rather than dropped. Omitting it trains the network to
    // ignore sky lighting, and every path hybrid_shading_cs.slang terminates on the cache then loses
//  // ignore sky lighting, and every path hybrid_shading_cs.slang terminates on the cache then loses
    // that contribution -- a bias that grows with how much of the scene sees the sky.
//  // that contribution -- a bias that grows with how much of the scene sees the sky.
    float3 computeTrainingTarget(float3 surfacePoint, float3 surfaceNormal, float3 incomingDirection,
//  float3 computeTrainingTarget(float3 surfacePoint, float3 surfaceNormal, float3 incomingDirection,
                               float3 albedo, float roughness, float metallic, float transmission) {
//                             float3 albedo, float roughness, float metallic, float transmission) {
        float3 targetRadiance = estimateDirectLighting(surfacePoint, surfaceNormal, incomingDirection,
//      float3 targetRadiance = estimateDirectLighting(surfacePoint, surfaceNormal, incomingDirection,
                                                     albedo, roughness, metallic, transmission);
//                                                   albedo, roughness, metallic, transmission);

        // With a cosine-weighted pdf the (albedo / PI) * cos / pdf factor collapses to a bare albedo.
//      // With a cosine-weighted pdf the (albedo / PI) * cos / pdf factor collapses to a bare albedo.
        float3 throughput = albedo;
//      float3 throughput = albedo;
        float3 rayOrigin = surfacePoint + surfaceNormal * EPSILON_OFFSET;
//      float3 rayOrigin = surfacePoint + surfaceNormal * EPSILON_OFFSET;
        float3 bounceDirUnnormalized = surfaceNormal + randomUnitVector();
//      float3 bounceDirUnnormalized = surfaceNormal + randomUnitVector();
        float3 rayDirection = (dot(bounceDirUnnormalized, bounceDirUnnormalized) > 1.0e-6)
//      float3 rayDirection = (dot(bounceDirUnnormalized, bounceDirUnnormalized) > 1.0e-6)
                          ? normalize(bounceDirUnnormalized) : surfaceNormal;
//                        ? normalize(bounceDirUnnormalized) : surfaceNormal;

        for (uint bounce = 0u; bounce < 3u; bounce++) {
//      for (uint bounce = 0u; bounce < 3u; bounce++) {
            float3 nextHitPoint;
//          float3 nextHitPoint;
            float3 nextHitNormal;
//          float3 nextHitNormal;
            int nextTriangleIndex;
//          int nextTriangleIndex;
            float2 nextTexcoord;
//          float2 nextTexcoord;
            float nextDistance;
//          float nextDistance;
            if (!traverseClosestHit(rayOrigin, rayDirection, INF, nextHitPoint, nextHitNormal, nextTriangleIndex, nextTexcoord, nextDistance)) {
//          if (!traverseClosestHit(rayOrigin, rayDirection, INF, nextHitPoint, nextHitNormal, nextTriangleIndex, nextTexcoord, nextDistance)) {
                // Escaped the geometry: the radiance left along this direction is the environment. The
//              // Escaped the geometry: the radiance left along this direction is the environment. The
                // bounce is cosine sampled, so throughput already carries albedo * cos / pdf.
//              // bounce is cosine sampled, so throughput already carries albedo * cos / pdf.
                targetRadiance += throughput * getSkyColor(rayDirection);
//              targetRadiance += throughput * getSkyColor(rayDirection);
                break;
//              break;
            }
//          }
            if (nextTriangleIndex < 0) break;
//          if (nextTriangleIndex < 0) break;

            int hitMaterialIndex = int(vertices[nextTriangleIndex * 3].tangentAndMaterialIndex.w);
//          int hitMaterialIndex = int(vertices[nextTriangleIndex * 3].tangentAndMaterialIndex.w);
            Material hitMaterial = materials[hitMaterialIndex];
//          Material hitMaterial = materials[hitMaterialIndex];

            // Emissive geometry is a light source for this target, not something to skip. The reasoning
//          // Emissive geometry is a light source for this target, not something to skip. The reasoning
            // that emitters are "seen directly by the path tracer" only holds for emitters the camera
//          // that emitters are "seen directly by the path tracer" only holds for emitters the camera
            // can reach: hybrid_shading_cs.slang terminates a path at the vertex it reads the cache on,
//          // can reach: hybrid_shading_cs.slang terminates a path at the vertex it reads the cache on,
            // so nothing downstream ever reaches an emitter beyond it. Dropping the emission here
//          // so nothing downstream ever reaches an emitter beyond it. Dropping the emission here
            // deletes it from the render's indirect lighting outright, and estimateDirectLighting above
//          // deletes it from the render's indirect lighting outright, and estimateDirectLighting above
            // cannot cover for it either -- that samples uPointLights only, so mesh emitters are never
//          // cannot cover for it either -- that samples uPointLights only, so mesh emitters are never
            // next-event sampled at all. This scene lights four cubes purely by material emission.
//          // next-event sampled at all. This scene lights four cubes purely by material emission.
            //
//          //
            // The colour is whatever scatterPrincipled would have emitted at this vertex, mapped
//          // The colour is whatever scatterPrincipled would have emitted at this vertex, mapped
            // emitters included -- resolveSurfaceMaterial reproduces both of its branches.
//          // emitters included -- resolveSurfaceMaterial reproduces both of its branches.
            float3 hitAlbedo;
//          float3 hitAlbedo;
            float hitRoughness;
//          float hitRoughness;
            float hitMetallic;
//          float hitMetallic;
            float hitTransmission;
//          float hitTransmission;
            float3 hitEmission;
//          float3 hitEmission;
            resolveSurfaceMaterial(hitMaterial, nextTexcoord, nextDistance,
//          resolveSurfaceMaterial(hitMaterial, nextTexcoord, nextDistance,
                                   hitAlbedo, hitRoughness, hitMetallic, hitTransmission, hitEmission);
//                                 hitAlbedo, hitRoughness, hitMetallic, hitTransmission, hitEmission);

            if (hitEmission.r > 0.0 || hitEmission.g > 0.0 || hitEmission.b > 0.0) {
//          if (hitEmission.r > 0.0 || hitEmission.g > 0.0 || hitEmission.b > 0.0) {
                targetRadiance += throughput * hitEmission;
//              targetRadiance += throughput * hitEmission;
                break;
//              break;
            }
//          }

            bool isCacheableVertex = (hitMetallic <= 0.8 && hitTransmission <= 0.5 && hitRoughness >= 0.05);
//          bool isCacheableVertex = (hitMetallic <= 0.8 && hitTransmission <= 0.5 && hitRoughness >= 0.05);
            if (isCacheableVertex) {
//          if (isCacheableVertex) {
                // The cache already represents the full outgoing radiance at a vertex (its own target
//              // The cache already represents the full outgoing radiance at a vertex (its own target
                // is direct + indirect), which is exactly why hybrid_shading_cs.slang skips next event
//              // is direct + indirect), which is exactly why hybrid_shading_cs.slang skips next event
                // estimation on the vertex it terminates a path at. So the bootstrap adds the cache
//              // estimation on the vertex it terminates a path at. So the bootstrap adds the cache
                // lookup alone: adding an explicit direct-lighting estimate here as well would count
//              // lookup alone: adding an explicit direct-lighting estimate here as well would count
                // one bounce of direct light twice, and the surplus compounds through the recursion
//              // one bounce of direct light twice, and the surplus compounds through the recursion
                // the next training pass builds on top of this target.
//              // the next training pass builds on top of this target.
                targetRadiance += throughput * evaluateNeuralRadianceCache(nextHitPoint, nextHitNormal);
//              targetRadiance += throughput * evaluateNeuralRadianceCache(nextHitPoint, nextHitNormal);
                break;
//              break;
            }
//          }

            // Mirror or glass: the cache is view dependent there, so carry the path on instead. The
//          // Mirror or glass: the cache is view dependent there, so carry the path on instead. The
            // continuation lobe stays cosine rather than BSDF-sampled -- this is a bootstrap estimate
//          // continuation lobe stays cosine rather than BSDF-sampled -- this is a bootstrap estimate
            // for a vertex the network is not asked about, and the extra variance of a proper
//          // for a vertex the network is not asked about, and the extra variance of a proper
            // specular lobe here would cost more than the bias it removes.
//          // specular lobe here would cost more than the bias it removes.
            throughput *= hitAlbedo;
//          throughput *= hitAlbedo;
            rayOrigin = nextHitPoint + nextHitNormal * EPSILON_OFFSET;
//          rayOrigin = nextHitPoint + nextHitNormal * EPSILON_OFFSET;
            float3 continueDirUnnormalized = nextHitNormal + randomUnitVector();
//          float3 continueDirUnnormalized = nextHitNormal + randomUnitVector();
            rayDirection = (dot(continueDirUnnormalized, continueDirUnnormalized) > 1.0e-6)
//          rayDirection = (dot(continueDirUnnormalized, continueDirUnnormalized) > 1.0e-6)
                         ? normalize(continueDirUnnormalized) : nextHitNormal;
//                       ? normalize(continueDirUnnormalized) : nextHitNormal;

            if (bounce > 0u && bounce + 1u < 3u) {
//          if (bounce > 0u && bounce + 1u < 3u) {
                float survivalProbability = clamp(max(throughput.r, max(throughput.g, throughput.b)), 0.05, 1.0);
//              float survivalProbability = clamp(max(throughput.r, max(throughput.g, throughput.b)), 0.05, 1.0);
                if (randFloat() > survivalProbability) break;
//              if (randFloat() > survivalProbability) break;
                throughput /= survivalProbability;
//              throughput /= survivalProbability;
            }
//          }
        }
//      }

        return min(targetRadiance, float3(15.0));
//      return min(targetRadiance, float3(15.0));
    }
//  }


    [shader("compute")]
//  [shader("compute")]
    [numthreads(64, 1, 1)]
//  [numthreads(64, 1, 1)]
    void computeMain(uint3 globalInvocationID : SV_DispatchThreadID, uint localInvocationIndex : SV_GroupIndex) {
//  void computeMain(uint3 globalInvocationID : SV_DispatchThreadID, uint localInvocationIndex : SV_GroupIndex) {
        // Stage the NRC EMA weights in shared memory before any early return, so that every
//      // Stage the NRC EMA weights in shared memory before any early return, so that every
        // invocation in the workgroup reaches barrier().
//      // invocation in the workgroup reaches barrier().
        for (uint weightIndex = localInvocationIndex; weightIndex < 1139u; weightIndex += 64u) {
//      for (uint weightIndex = localInvocationIndex; weightIndex < 1139u; weightIndex += 64u) {
            shared_network_weights[weightIndex] = neuralNetworkWeights[1139u + weightIndex];
//          shared_network_weights[weightIndex] = neuralNetworkWeights[1139u + weightIndex];
        }
//      }
        GroupMemoryBarrierWithGroupSync();
//      GroupMemoryBarrierWithGroupSync();

        gRngState = globalInvocationID.x + uTrainingStep * 1000000u + 1337u;
//      gRngState = globalInvocationID.x + uTrainingStep * 1000000u + 1337u;
        randUint();
//      randUint();

        // Start from a random primary-visible surface taken out of the G-Buffer.
//      // Start from a random primary-visible surface taken out of the G-Buffer.
        int2 pixelCoord = int2(
//      int2 pixelCoord = int2(
            clamp(int(randFloat() * float(uResolution.x)), 0, uResolution.x - 1),
//          clamp(int(randFloat() * float(uResolution.x)), 0, uResolution.x - 1),
            clamp(int(randFloat() * float(uResolution.y)), 0, uResolution.y - 1)
//          clamp(int(randFloat() * float(uResolution.y)), 0, uResolution.y - 1)
        );
//      );

        float4 posSample = textureGeometryGlobalPosition.Load(int3(pixelCoord, 0));
//      float4 posSample = textureGeometryGlobalPosition.Load(int3(pixelCoord, 0));
        if (posSample.w == 0.0) return;
//      if (posSample.w == 0.0) return;

        int triangleIndex = int(posSample.w) - 1;
//      int triangleIndex = int(posSample.w) - 1;
        if (triangleIndex < 0) return;
//      if (triangleIndex < 0) return;

        float4 normSample = textureGeometryGlobalNormal.Load(int3(pixelCoord, 0));
//      float4 normSample = textureGeometryGlobalNormal.Load(int3(pixelCoord, 0));
        if (dot(normSample.xyz, normSample.xyz) < 1.0e-6) return;
//      if (dot(normSample.xyz, normSample.xyz) < 1.0e-6) return;

        float4 tangentSample = textureGeometryGlobalTangent.Load(int3(pixelCoord, 0));
//      float4 tangentSample = textureGeometryGlobalTangent.Load(int3(pixelCoord, 0));

        float3 currentPoint = posSample.xyz;
//      float3 currentPoint = posSample.xyz;
        float3 currentNormal = normalize(normSample.xyz);
//      float3 currentNormal = normalize(normSample.xyz);
        int currentTriangle = triangleIndex;
//      int currentTriangle = triangleIndex;
        float3 toSurface = currentPoint - uCameraGlobalPosition;
//      float3 toSurface = currentPoint - uCameraGlobalPosition;
        float currentDistance = length(toSurface);
//      float currentDistance = length(toSurface);
        float3 currentIncomingDir = (dot(toSurface, toSurface) > 1.0e-12) ? normalize(toSurface) : -currentNormal;
//      float3 currentIncomingDir = (dot(toSurface, toSurface) > 1.0e-12) ? normalize(toSurface) : -currentNormal;
        // U and V are packed into the w channels of the normal and tangent targets, matching the
//      // U and V are packed into the w channels of the normal and tangent targets, matching the
        // reconstruction hybrid_shading_cs.slang does for its own depth-0 vertex.
//      // reconstruction hybrid_shading_cs.slang does for its own depth-0 vertex.
        float2 currentTexcoord = float2(normSample.w, tangentSample.w);
//      float2 currentTexcoord = float2(normSample.w, tangentSample.w);
        // The albedo target already holds the resolved, sRGB-decoded base colour for this pixel, and
//      // The albedo target already holds the resolved, sRGB-decoded base colour for this pixel, and
        // the shading pass shades depth 0 from exactly this value, so prefer it over a re-fetch.
//      // the shading pass shades depth 0 from exactly this value, so prefer it over a re-fetch.
        float3 currentAlbedo = textureGeometryAlbedo.Load(int3(pixelCoord, 0)).rgb;
//      float3 currentAlbedo = textureGeometryAlbedo.Load(int3(pixelCoord, 0)).rgb;

        // Walk the path and record a single training vertex along it. Recording at deeper vertices is what
//      // Walk the path and record a single training vertex along it. Recording at deeper vertices is what
        // keeps the training distribution aligned with where hybrid_shading_cs.slang queries the cache,
//      // keeps the training distribution aligned with where hybrid_shading_cs.slang queries the cache,
        // which is at secondary bounces that are frequently not visible from the camera at all.
//      // which is at secondary bounces that are frequently not visible from the camera at all.
        for (uint bounce = 0u; bounce < 6u; bounce++) {
//      for (uint bounce = 0u; bounce < 6u; bounce++) {
            int materialIndex = int(vertices[currentTriangle * 3].tangentAndMaterialIndex.w);
//          int materialIndex = int(vertices[currentTriangle * 3].tangentAndMaterialIndex.w);
            Material material = materials[materialIndex];
//          Material material = materials[materialIndex];

            // The same maps the shading pass shades this surface with, so the recorded vertex is
//          // The same maps the shading pass shades this surface with, so the recorded vertex is
            // classified and its target built from the material the renderer will actually query about.
//          // classified and its target built from the material the renderer will actually query about.
            // The primary vertex keeps its G-Buffer albedo; deeper vertices take the resolved one.
//          // The primary vertex keeps its G-Buffer albedo; deeper vertices take the resolved one.
            float3 resolvedAlbedo;
//          float3 resolvedAlbedo;
            float effectiveRoughness;
//          float effectiveRoughness;
            float effectiveMetallic;
//          float effectiveMetallic;
            float effectiveTransmission;
//          float effectiveTransmission;
            float3 resolvedEmission;
//          float3 resolvedEmission;
            resolveSurfaceMaterial(material, currentTexcoord, currentDistance,
//          resolveSurfaceMaterial(material, currentTexcoord, currentDistance,
                                   resolvedAlbedo, effectiveRoughness, effectiveMetallic,
//                                 resolvedAlbedo, effectiveRoughness, effectiveMetallic,
                                   effectiveTransmission, resolvedEmission);
//                                 effectiveTransmission, resolvedEmission);
            if (bounce > 0u) {
//          if (bounce > 0u) {
                currentAlbedo = resolvedAlbedo;
//              currentAlbedo = resolvedAlbedo;
            }
//          }

            // Emitters are seen directly by the path tracer; caching them would double count. Tested on
//          // Emitters are seen directly by the path tracer; caching them would double count. Tested on
            // the resolved emission, so an emissive map that is black here no longer disqualifies the
//          // the resolved emission, so an emissive map that is black here no longer disqualifies the
            // whole material -- the same test scatterPrincipled applies before it refuses to scatter.
//          // whole material -- the same test scatterPrincipled applies before it refuses to scatter.
            if (resolvedEmission.r > 0.0 || resolvedEmission.g > 0.0 || resolvedEmission.b > 0.0) break;
//          if (resolvedEmission.r > 0.0 || resolvedEmission.g > 0.0 || resolvedEmission.b > 0.0) break;

            // Only cache broadly diffuse dielectrics; mirrors and glass are view dependent.
//          // Only cache broadly diffuse dielectrics; mirrors and glass are view dependent.
            bool isCacheable = (effectiveMetallic <= 0.8 && effectiveTransmission <= 0.5 && effectiveRoughness >= 0.05);
//          bool isCacheable = (effectiveMetallic <= 0.8 && effectiveTransmission <= 0.5 && effectiveRoughness >= 0.05);

            if (isCacheable) {
//          if (isCacheable) {
                float3 targetRadiance = computeTrainingTarget(currentPoint, currentNormal, currentIncomingDir,
//              float3 targetRadiance = computeTrainingTarget(currentPoint, currentNormal, currentIncomingDir,
                                                            currentAlbedo, effectiveRoughness, effectiveMetallic,
//                                                          currentAlbedo, effectiveRoughness, effectiveMetallic,
                                                            effectiveTransmission);
//                                                          effectiveTransmission);

                float acceptanceProb = 1.0;
//              float acceptanceProb = 1.0;

                if (bounce == 0u) {
//              if (bounce == 0u) {
                    // 1. Material-aware adaptive sampling: smooth surfaces act as mirrors and need the
//                  // 1. Material-aware adaptive sampling: smooth surfaces act as mirrors and need the
                    //    cache for screen-space reflections, rough ones are already served well by ReSTIR DI.
//                  //    cache for screen-space reflections, rough ones are already served well by ReSTIR DI.
                    float materialProb = lerp(0.80, 0.10, effectiveRoughness);
//                  float materialProb = lerp(0.80, 0.10, effectiveRoughness);

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
                    // Capacity invariant: this pass is dispatched as 128 workgroups of 64 invocations
//                  // Capacity invariant: this pass is dispatched as 128 workgroups of 64 invocations
                    // (8192 total) and each invocation stores at most one record before breaking out of
//                  // (8192 total) and each invocation stores at most one record before breaking out of
                    // the path loop, so storeIndex never reaches the 8192-record capacity of
//                  // the path loop, so storeIndex never reaches the 8192-record capacity of
                    // nrcTrainingRecords. The bounds check below is the guard that keeps that true if
//                  // nrcTrainingRecords. The bounds check below is the guard that keeps that true if
                    // the dispatch size in hybrid_renderer.py is ever raised without resizing the buffer.
//                  // the dispatch size in hybrid_renderer.py is ever raised without resizing the buffer.
                    uint storeIndex = trainingRecordCounter[0].add(1u);
//                  uint storeIndex = trainingRecordCounter[0].add(1u);

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
            float3 continueDirUnnormalized = currentNormal + randomUnitVector();
//          float3 continueDirUnnormalized = currentNormal + randomUnitVector();
            float3 continueDir = (dot(continueDirUnnormalized, continueDirUnnormalized) > 1.0e-6) ? normalize(continueDirUnnormalized) : currentNormal;
//          float3 continueDir = (dot(continueDirUnnormalized, continueDirUnnormalized) > 1.0e-6) ? normalize(continueDirUnnormalized) : currentNormal;

            float3 nextPoint;
//          float3 nextPoint;
            float3 nextNormal;
//          float3 nextNormal;
            int nextTriangle;
//          int nextTriangle;
            float2 nextTexcoord;
//          float2 nextTexcoord;
            float nextDistance;
//          float nextDistance;
            if (!traverseClosestHit(currentPoint + currentNormal * EPSILON_OFFSET, continueDir, INF, nextPoint, nextNormal, nextTriangle, nextTexcoord, nextDistance)) break;
//          if (!traverseClosestHit(currentPoint + currentNormal * EPSILON_OFFSET, continueDir, INF, nextPoint, nextNormal, nextTriangle, nextTexcoord, nextDistance)) break;
            if (nextTriangle < 0) break;
//          if (nextTriangle < 0) break;

            currentPoint = nextPoint;
//          currentPoint = nextPoint;
            currentNormal = nextNormal;
//          currentNormal = nextNormal;
            currentTriangle = nextTriangle;
//          currentTriangle = nextTriangle;
            currentIncomingDir = continueDir;
//          currentIncomingDir = continueDir;
            currentTexcoord = nextTexcoord;
//          currentTexcoord = nextTexcoord;
            currentDistance = nextDistance;
//          currentDistance = nextDistance;
        }
//      }
    }
//  }
