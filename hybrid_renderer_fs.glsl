    #version 430
//  #version 430

    layout(location = 0) in vec2 inScreenFragmentUV;
//  layout(location = 0) in vec2 inScreenFragmentUV;

    out vec4 fragmentColor;
//  out vec4 fragmentColor;

    uniform sampler2D uTextureOutput;
//  uniform sampler2D uTextureOutput;

    void main() {
//  void main() {
        fragmentColor = texture(uTextureOutput, inScreenFragmentUV);
//      fragmentColor = texture(uTextureOutput, inScreenFragmentUV);
    }
//  }
