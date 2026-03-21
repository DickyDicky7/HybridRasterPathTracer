    #version 430
//  #version 430

    in vec4 inVertexGlobalPositionU;
//  in vec4 inVertexGlobalPositionU;
    in vec4 inVertexGlobalNormalV;
//  in vec4 inVertexGlobalNormalV;
    in vec4 inVertexGlobalTangentMat;
//  in vec4 inVertexGlobalTangentMat;

    layout(location = 0) out vec3 outVertexGlobalPosition;
//  layout(location = 0) out vec3 outVertexGlobalPosition;
    layout(location = 1) out vec3 outVertexGlobalNormal;
//  layout(location = 1) out vec3 outVertexGlobalNormal;
    layout(location = 2) out vec3 outVertexGlobalTangent;
//  layout(location = 2) out vec3 outVertexGlobalTangent;
    layout(location = 3) out vec2 outVertexUV;
//  layout(location = 3) out vec2 outVertexUV;
    flat out int vMaterialIndex;
//  flat out int vMaterialIndex;

    uniform mat4 uTransformView;
//  uniform mat4 uTransformView;
    uniform mat4 uTransformProjection;
//  uniform mat4 uTransformProjection;

    void main() {
//  void main() {
        outVertexGlobalPosition = inVertexGlobalPositionU.xyz;
//      outVertexGlobalPosition = inVertexGlobalPositionU.xyz;
        outVertexGlobalNormal = inVertexGlobalNormalV.xyz;
//      outVertexGlobalNormal = inVertexGlobalNormalV.xyz;
        outVertexGlobalTangent = inVertexGlobalTangentMat.xyz;
//      outVertexGlobalTangent = inVertexGlobalTangentMat.xyz;
        outVertexUV = vec2(inVertexGlobalPositionU.w, inVertexGlobalNormalV.w);
//      outVertexUV = vec2(inVertexGlobalPositionU.w, inVertexGlobalNormalV.w);
        vMaterialIndex = int(inVertexGlobalTangentMat.w);
//      vMaterialIndex = int(inVertexGlobalTangentMat.w);
        gl_Position = uTransformProjection * uTransformView * vec4(outVertexGlobalPosition, 1.0);
//      gl_Position = uTransformProjection * uTransformView * vec4(outVertexGlobalPosition, 1.0);
    }
//  }
