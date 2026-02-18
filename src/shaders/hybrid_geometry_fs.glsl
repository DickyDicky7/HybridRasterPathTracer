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
    layout(location = 4) in vec2 inVertexUV;
//  layout(location = 4) in vec2 inVertexUV;
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

    void main() {
//  void main() {
        int globalTriIdx = uBaseTriangleIndexOffset + vInstanceID * uTriangleCountPerInstance + gl_PrimitiveID;
//      int globalTriIdx = uBaseTriangleIndexOffset + vInstanceID * uTriangleCountPerInstance + gl_PrimitiveID;
        // outGeometryGlobalPosition = vec4(inGeometryGlobalPosition, float(globalTriIdx) + 0.5); // .w stores index. +0.5 to avoid precision issues when rounding? 
//      // outGeometryGlobalPosition = vec4(inGeometryGlobalPosition, float(globalTriIdx) + 0.5); // .w stores index. +0.5 to avoid precision issues when rounding? 
        // Logic: if w > 0, it's a hit.
//      // Logic: if w > 0, it's a hit.
        // float(idx) could be 0. So float(idx + 1) is safer?
//      // float(idx) could be 0. So float(idx + 1) is safer?
        // But if I store float(idx), 0 is a valid index.
//      // But if I store float(idx), 0 is a valid index.
        // But usually background is 0.0.
//      // But usually background is 0.0.
        // If I use float(idx + 1), then 0.0 means miss.
//      // If I use float(idx + 1), then 0.0 means miss.
        // Let's use float(idx + 1).
//      // Let's use float(idx + 1).
        outGeometryGlobalPosition = vec4(inGeometryGlobalPosition, float(globalTriIdx + 1));
//      outGeometryGlobalPosition = vec4(inGeometryGlobalPosition, float(globalTriIdx + 1));

        outGeometryGlobalNormal = vec4(normalize(inGeometryGlobalNormal), inVertexUV.x);
//      outGeometryGlobalNormal = vec4(normalize(inGeometryGlobalNormal), inVertexUV.x);
        outGeometryAlbedo = vec4(inGeometryAlbedo, 1.0);
//      outGeometryAlbedo = vec4(inGeometryAlbedo, 1.0);
        outGeometryGlobalTangent = vec4(normalize(inGeometryGlobalTangent), inVertexUV.y);
//      outGeometryGlobalTangent = vec4(normalize(inGeometryGlobalTangent), inVertexUV.y);
    }
//  }
