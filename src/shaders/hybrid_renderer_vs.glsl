    #version 430
//  #version 430

    layout(location = 0) in vec2 inScreenVertexPosition;
//  layout(location = 0) in vec2 inScreenVertexPosition;
    layout(location = 1) in vec2 inScreenVertexUV;
//  layout(location = 1) in vec2 inScreenVertexUV;

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
