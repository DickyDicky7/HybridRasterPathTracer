    #version 430
//  #version 430

    in vec2 inScreenVertexPosition;
//  in vec2 inScreenVertexPosition;
    in vec2 inScreenVertexUV;
//  in vec2 inScreenVertexUV;

    layout(location = 0) out vec2 outScreenVertexUV;
//  layout(location = 0) out vec2 outScreenVertexUV;

    void main() {
//  void main() {
        outScreenVertexUV = inScreenVertexUV;
//      outScreenVertexUV = inScreenVertexUV;
        gl_Position = vec4(inScreenVertexPosition, 0.0, 1.0);
//      gl_Position = vec4(inScreenVertexPosition, 0.0, 1.0);
    }
//  }
