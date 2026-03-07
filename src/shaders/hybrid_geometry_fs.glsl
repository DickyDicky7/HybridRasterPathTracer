    #version 430
//  #version 430

    layout(location = 0) in vec3 inGeometryGlobalPosition;
//  layout(location = 0) in vec3 inGeometryGlobalPosition;
    layout(location = 1) in vec3 inGeometryGlobalNormal;
//  layout(location = 1) in vec3 inGeometryGlobalNormal;
    layout(location = 2) in vec3 inGeometryAlbedo;
//  layout(location = 2) in vec3 inGeometryAlbedo;
    layout(location = 3) in vec3 inGeometryGlobalTangent;
//  layout(location = 3) in vec3 inGeometryGlobalTangent;
    layout(location = 4) in vec2 inGeometryUV;
//  layout(location = 4) in vec2 inGeometryUV;
    flat in int vInstanceID;
//  flat in int vInstanceID;

    layout(location = 0) out vec4 outGeometryGlobalPosition;
//  layout(location = 0) out vec4 outGeometryGlobalPosition;
    layout(location = 1) out vec4 outGeometryGlobalNormal;
//  layout(location = 1) out vec4 outGeometryGlobalNormal;
    layout(location = 2) out vec4 outGeometryAlbedo;
//  layout(location = 2) out vec4 outGeometryAlbedo;
    layout(location = 3) out vec4 outGeometryGlobalTangent;
//  layout(location = 3) out vec4 outGeometryGlobalTangent;

    uniform int uBaseTriangleIndexOffset;
//  uniform int uBaseTriangleIndexOffset;
    uniform int uTriangleCountPerInstance;
//  uniform int uTriangleCountPerInstance;

    uniform sampler2DArray uSceneTextureArray;
//  uniform sampler2DArray uSceneTextureArray;

    void main() {
//  void main() {
        int globalTriIdx = uBaseTriangleIndexOffset + vInstanceID * uTriangleCountPerInstance + gl_PrimitiveID;
//      int globalTriIdx = uBaseTriangleIndexOffset + vInstanceID * uTriangleCountPerInstance + gl_PrimitiveID;
        // Primitive Identity Packing: Stores the global triangle index in the 'w' component. We offset the index by +1.0 during encoding so that a 0.0 value definitively represents an empty background/sky-miss, eliminating potential ambiguity with index 0 while retrieving geometry hits later.
//      // Primitive Identity Packing: Stores the global triangle index in the 'w' component. We offset the index by +1.0 during encoding so that a 0.0 value definitively represents an empty background/sky-miss, eliminating potential ambiguity with index 0 while retrieving geometry hits later.
        outGeometryGlobalPosition = vec4(inGeometryGlobalPosition, float(globalTriIdx + 1));
//      outGeometryGlobalPosition = vec4(inGeometryGlobalPosition, float(globalTriIdx + 1));

        outGeometryGlobalNormal = vec4(normalize(inGeometryGlobalNormal), inGeometryUV.x);
//      outGeometryGlobalNormal = vec4(normalize(inGeometryGlobalNormal), inGeometryUV.x);

        vec3 finalAlbedo = inGeometryAlbedo;
//      vec3 finalAlbedo = inGeometryAlbedo;
        if (inGeometryAlbedo.y < 0.0) {
//      if (inGeometryAlbedo.y < 0.0) {
            finalAlbedo = texture(uSceneTextureArray, vec3(inGeometryUV, inGeometryAlbedo.x)).rgb;
//          finalAlbedo = texture(uSceneTextureArray, vec3(inGeometryUV, inGeometryAlbedo.x)).rgb;
        }
//      }
        outGeometryAlbedo = vec4(finalAlbedo, 1.0);
//      outGeometryAlbedo = vec4(finalAlbedo, 1.0);

        outGeometryGlobalTangent = vec4(normalize(inGeometryGlobalTangent), inGeometryUV.y);
//      outGeometryGlobalTangent = vec4(normalize(inGeometryGlobalTangent), inGeometryUV.y);
    }
//  }
