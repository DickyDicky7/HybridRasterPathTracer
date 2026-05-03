    #version 430
//  #version 430

    layout(location = 0) in vec2 inScreenFragmentUV;
//  layout(location = 0) in vec2 inScreenFragmentUV;
    layout(location = 0) out vec4 fragmentColor;
//  layout(location = 0) out vec4 fragmentColor;

    uniform sampler2D uTextureOutput;
//  uniform sampler2D uTextureOutput;
    uniform int uRenderMode;
//  uniform int uRenderMode;

    const float DISPLAY_GAMMA = 2.2;
//  const float DISPLAY_GAMMA = 2.2;

    void main() {
//  void main() {
        vec4 val = texture(uTextureOutput, inScreenFragmentUV);
//      vec4 val = texture(uTextureOutput, inScreenFragmentUV);
        if (uRenderMode != 0) {
//      if (uRenderMode != 0) {
            if (uRenderMode == 1) {
//          if (uRenderMode == 1) {
                fragmentColor = vec4(val.rgb, 1.0);
//              fragmentColor = vec4(val.rgb, 1.0);
            } else if (uRenderMode == 2) {
//          } else if (uRenderMode == 2) {
                fragmentColor = vec4(val.rgb * 0.5 + 0.5, 1.0);
//              fragmentColor = vec4(val.rgb * 0.5 + 0.5, 1.0);
            } else if (uRenderMode == 3) {
//          } else if (uRenderMode == 3) {
                fragmentColor = vec4(fract(val.rgb), 1.0);
//              fragmentColor = vec4(fract(val.rgb), 1.0);
            } else if (uRenderMode == 4) {
//          } else if (uRenderMode == 4) {
                fragmentColor = vec4(val.rgb * 0.5 + 0.5, 1.0);
//              fragmentColor = vec4(val.rgb * 0.5 + 0.5, 1.0);
            } else {
//          } else {
                fragmentColor = vec4(1.0, 0.0, 1.0, 1.0);
//              fragmentColor = vec4(1.0, 0.0, 1.0, 1.0);
            }
//          }
            fragmentColor.rgb = pow(fragmentColor.rgb, vec3(1.0 / DISPLAY_GAMMA));
//          fragmentColor.rgb = pow(fragmentColor.rgb, vec3(1.0 / DISPLAY_GAMMA));
        } else {
//      } else {
            fragmentColor = vec4(val.rgb, 1.0);
//          fragmentColor = vec4(val.rgb, 1.0);
        }
//      }
    }
//  }
