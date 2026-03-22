    #version 430
//  #version 430

    layout(location = 0) in vec3 inGeometryGlobalPosition;
//  layout(location = 0) in vec3 inGeometryGlobalPosition;
    layout(location = 1) in vec3 inGeometryGlobalNormal;
//  layout(location = 1) in vec3 inGeometryGlobalNormal;
    layout(location = 2) in vec3 inGeometryGlobalTangent;
//  layout(location = 2) in vec3 inGeometryGlobalTangent;
    layout(location = 3) in vec2 inGeometryUV;
//  layout(location = 3) in vec2 inGeometryUV;
    flat in int vMaterialIndex;
//  flat in int vMaterialIndex;

    layout(location = 0) out vec4 outGeometryGlobalPosition;
//  layout(location = 0) out vec4 outGeometryGlobalPosition;
    layout(location = 1) out vec4 outGeometryGlobalNormal;
//  layout(location = 1) out vec4 outGeometryGlobalNormal;
    layout(location = 2) out vec4 outGeometryAlbedo;
//  layout(location = 2) out vec4 outGeometryAlbedo;
    layout(location = 3) out vec4 outGeometryGlobalTangent;
//  layout(location = 3) out vec4 outGeometryGlobalTangent;

    struct Material {
//  struct Material {
        vec4 albedo; // r, g, b, padding
//      vec4 albedo; // r, g, b, padding
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
    };
//  };

    layout(std430, binding = 8) buffer SceneMaterials {
//  layout(std430, binding = 8) buffer SceneMaterials {
        Material materials[];
//      Material materials[];
    };
//  };

    uniform sampler2DArray uSceneTextureArray;
//  uniform sampler2DArray uSceneTextureArray;

    void main() {
//  void main() {
        int globalTriIdx = gl_PrimitiveID;
//      int globalTriIdx = gl_PrimitiveID;
        // Primitive Identity Packing: Stores the global triangle index in the 'w' component. We offset the index by +1.0 during encoding so that a 0.0 value definitively represents an empty background/sky-miss, eliminating potential ambiguity with index 0 while retrieving geometry hits later.
//      // Primitive Identity Packing: Stores the global triangle index in the 'w' component. We offset the index by +1.0 during encoding so that a 0.0 value definitively represents an empty background/sky-miss, eliminating potential ambiguity with index 0 while retrieving geometry hits later.
        outGeometryGlobalPosition = vec4(inGeometryGlobalPosition, float(globalTriIdx + 1));
//      outGeometryGlobalPosition = vec4(inGeometryGlobalPosition, float(globalTriIdx + 1));

        outGeometryGlobalNormal = vec4(normalize(inGeometryGlobalNormal), inGeometryUV.x);
//      outGeometryGlobalNormal = vec4(normalize(inGeometryGlobalNormal), inGeometryUV.x);

        Material mat = materials[vMaterialIndex];
//      Material mat = materials[vMaterialIndex];
        vec3 finalAlbedo = mat.albedo.rgb;
//      vec3 finalAlbedo = mat.albedo.rgb;
        if (mat.textureIndexAlbedo > -0.5) {
//      if (mat.textureIndexAlbedo > -0.5) {
            finalAlbedo *= texture(uSceneTextureArray, vec3(inGeometryUV, mat.textureIndexAlbedo)).rgb;
//          finalAlbedo *= texture(uSceneTextureArray, vec3(inGeometryUV, mat.textureIndexAlbedo)).rgb;
        }
//      }
        outGeometryAlbedo = vec4(finalAlbedo, 1.0);
//      outGeometryAlbedo = vec4(finalAlbedo, 1.0);

        outGeometryGlobalTangent = vec4(normalize(inGeometryGlobalTangent), inGeometryUV.y);
//      outGeometryGlobalTangent = vec4(normalize(inGeometryGlobalTangent), inGeometryUV.y);
    }
//  }
