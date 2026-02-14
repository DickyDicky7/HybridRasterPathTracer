    #version 430
//  #version 430

    layout(location = 0) in vec3 inGeometryGlobalPosition;
//  layout(location = 0) in vec3 inGeometryGlobalPosition;
    layout(location = 1) in vec3 inGeometryGlobalNormal;
//  layout(location = 1) in vec3 inGeometryGlobalNormal;
    layout(location = 2) in vec3 inGeometryAlbedo;
//  layout(location = 2) in vec3 inGeometryAlbedo;
    flat in int vInstanceID;
//  flat in int vInstanceID;

    out vec4 outGeometryGlobalPosition;
//  out vec4 outGeometryGlobalPosition;
    out vec4 outGeometryGlobalNormal;
//  out vec4 outGeometryGlobalNormal;
    out vec4 outGeometryAlbedo;
//  out vec4 outGeometryAlbedo;

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

        outGeometryGlobalNormal = vec4(normalize(inGeometryGlobalNormal), 1.0);
//      outGeometryGlobalNormal = vec4(normalize(inGeometryGlobalNormal), 1.0);
        outGeometryAlbedo = vec4(inGeometryAlbedo, 1.0);
//      outGeometryAlbedo = vec4(inGeometryAlbedo, 1.0);
    }
//  }
