    #version 430
//  #version 430

    in vec3 inVertexLocalPosition;
//  in vec3 inVertexLocalPosition;
    in vec3 inVertexLocalNormal;
//  in vec3 inVertexLocalNormal;
    in mat4 inInstanceTransformModel;
//  in mat4 inInstanceTransformModel;
    in vec3 inInstanceAlbedo;
//  in vec3 inInstanceAlbedo;

    layout(location = 0) out vec3 outVertexGlobalPosition;
//  layout(location = 0) out vec3 outVertexGlobalPosition;
    layout(location = 1) out vec3 outVertexGlobalNormal;
//  layout(location = 1) out vec3 outVertexGlobalNormal;
    layout(location = 2) out vec3 outInstanceAlbedo;
//  layout(location = 2) out vec3 outInstanceAlbedo;

    uniform mat4 uTransformView;
//  uniform mat4 uTransformView;
    uniform mat4 uTransformProjection;
//  uniform mat4 uTransformProjection;

    void main() {
//  void main() {
        vec4 vertexGlobalPosition = inInstanceTransformModel * vec4(inVertexLocalPosition, 1.0);
//      vec4 vertexGlobalPosition = inInstanceTransformModel * vec4(inVertexLocalPosition, 1.0);
        outVertexGlobalPosition = vertexGlobalPosition.xyz;
//      outVertexGlobalPosition = vertexGlobalPosition.xyz;
        outVertexGlobalNormal = mat3(inInstanceTransformModel) * inVertexLocalNormal;
//      outVertexGlobalNormal = mat3(inInstanceTransformModel) * inVertexLocalNormal;
        outInstanceAlbedo = inInstanceAlbedo;
//      outInstanceAlbedo = inInstanceAlbedo;
        gl_Position = uTransformProjection * uTransformView * vertexGlobalPosition;
//      gl_Position = uTransformProjection * uTransformView * vertexGlobalPosition;
    }
//  }
