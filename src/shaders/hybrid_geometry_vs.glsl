    #version 430
//  #version 430

    in vec3 inVertexLocalPosition;
//  in vec3 inVertexLocalPosition;
    in vec3 inVertexLocalNormal;
//  in vec3 inVertexLocalNormal;
    in vec3 inVertexLocalTangent;
//  in vec3 inVertexLocalTangent;
    in vec2 inVertexLocalUV;
//  in vec2 inVertexLocalUV;
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
    layout(location = 3) out vec3 outVertexGlobalTangent;
//  layout(location = 3) out vec3 outVertexGlobalTangent;
    layout(location = 4) out vec2 outVertexUV;
//  layout(location = 4) out vec2 outVertexUV;
    flat out int vInstanceID;
//  flat out int vInstanceID;

    uniform mat4 uTransformView;
//  uniform mat4 uTransformView;
    uniform mat4 uTransformProjection;
//  uniform mat4 uTransformProjection;

    void main() {
//  void main() {
        vInstanceID = gl_InstanceID;
//      vInstanceID = gl_InstanceID;
        vec4 vertexGlobalPosition = inInstanceTransformModel * vec4(inVertexLocalPosition, 1.0);
//      vec4 vertexGlobalPosition = inInstanceTransformModel * vec4(inVertexLocalPosition, 1.0);
        outVertexGlobalPosition = vertexGlobalPosition.xyz;
//      outVertexGlobalPosition = vertexGlobalPosition.xyz;
        outVertexGlobalNormal = mat3(inInstanceTransformModel) * inVertexLocalNormal;
//      outVertexGlobalNormal = mat3(inInstanceTransformModel) * inVertexLocalNormal;
        outVertexGlobalTangent = mat3(inInstanceTransformModel) * inVertexLocalTangent;
//      outVertexGlobalTangent = mat3(inInstanceTransformModel) * inVertexLocalTangent;
        outInstanceAlbedo = inInstanceAlbedo;
//      outInstanceAlbedo = inInstanceAlbedo;
        outVertexUV = inVertexLocalUV;
//      outVertexUV = inVertexLocalUV;
        gl_Position = uTransformProjection * uTransformView * vertexGlobalPosition;
//      gl_Position = uTransformProjection * uTransformView * vertexGlobalPosition;
    }
//  }
