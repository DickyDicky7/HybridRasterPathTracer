    #version 430
//  #version 430

    layout(location = 0) in vec3 inGeometryGlobalPosition;
//  layout(location = 0) in vec3 inGeometryGlobalPosition;
    layout(location = 1) in vec3 inGeometryGlobalNormal;
//  layout(location = 1) in vec3 inGeometryGlobalNormal;
    layout(location = 2) in vec3 inGeometryAlbedo;
//  layout(location = 2) in vec3 inGeometryAlbedo;

    out vec4 outGeometryGlobalPosition;
//  out vec4 outGeometryGlobalPosition;
    out vec4 outGeometryGlobalNormal;
//  out vec4 outGeometryGlobalNormal;
    out vec4 outGeometryAlbedo;
//  out vec4 outGeometryAlbedo;

    void main() {
//  void main() {
        outGeometryGlobalPosition = vec4(inGeometryGlobalPosition, 1.0); // .w = 1 indicates geometry exists
//      outGeometryGlobalPosition = vec4(inGeometryGlobalPosition, 1.0); // .w = 1 indicates geometry exists
        outGeometryGlobalNormal = vec4(normalize(inGeometryGlobalNormal), 1.0);
//      outGeometryGlobalNormal = vec4(normalize(inGeometryGlobalNormal), 1.0);
        outGeometryAlbedo = vec4(inGeometryAlbedo, 1.0);
//      outGeometryAlbedo = vec4(inGeometryAlbedo, 1.0);
    }
//  }
