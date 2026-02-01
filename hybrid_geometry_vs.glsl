    #version 430
//  #version 430

    in vec3 inVertexLocalPosition;
//  in vec3 inVertexLocalPosition;
    in vec3 inVertexLocalNormal;
//  in vec3 inVertexLocalNormal;
    in vec3 inVertexAlbedo;
//  in vec3 inVertexAlbedo;

    layout(location = 0) out vec3 outVertexGlobalPosition;
//  layout(location = 0) out vec3 outVertexGlobalPosition;
    layout(location = 1) out vec3 outVertexGlobalNormal;
//  layout(location = 1) out vec3 outVertexGlobalNormal;
    layout(location = 2) out vec3 outVertexAlbedo;
//  layout(location = 2) out vec3 outVertexAlbedo;

    uniform mat4 uTransformModel;
//  uniform mat4 uTransformModel;
    uniform mat4 uTransformView;
//  uniform mat4 uTransformView;
    uniform mat4 uTransformProjection;
//  uniform mat4 uTransformProjection;

    void main() {
//  void main() {
        vec4 vertexGlobalPosition = uTransformModel * vec4(inVertexLocalPosition, 1.0);
//      vec4 vertexGlobalPosition = uTransformModel * vec4(inVertexLocalPosition, 1.0);
        outVertexGlobalPosition = vertexGlobalPosition.xyz;
//      outVertexGlobalPosition = vertexGlobalPosition.xyz;
        outVertexGlobalNormal = mat3(uTransformModel) * inVertexLocalNormal;
//      outVertexGlobalNormal = mat3(uTransformModel) * inVertexLocalNormal;
        outVertexAlbedo = inVertexAlbedo;
//      outVertexAlbedo = inVertexAlbedo;
        gl_Position = uTransformProjection * uTransformView * vertexGlobalPosition;
//      gl_Position = uTransformProjection * uTransformView * vertexGlobalPosition;
    }
//  }
