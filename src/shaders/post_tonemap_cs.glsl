    #version 430
//  #version 430
    layout(local_size_x = 16, local_size_y = 16) in;
//  layout(local_size_x = 16, local_size_y = 16) in;

    layout(binding = 0, rgba32f) uniform image2D textureOutput;
//  layout(binding = 0, rgba32f) uniform image2D textureOutput;
    layout(binding = 1, rgba32f) uniform image2D textureInput;
//  layout(binding = 1, rgba32f) uniform image2D textureInput;

    const float EXPOSURE = 0.5;
//  const float EXPOSURE = 0.5;

    #include "tonemap.glsl"

    const float DISPLAY_GAMMA = 2.2;
//  const float DISPLAY_GAMMA = 2.2;

    void main() {
//  void main() {
        ivec2 coord = ivec2(gl_GlobalInvocationID.xy);
//      ivec2 coord = ivec2(gl_GlobalInvocationID.xy);
        ivec2 size = imageSize(textureOutput);
//      ivec2 size = imageSize(textureOutput);

        if (coord.x >= size.x || coord.y >= size.y) {
//      if (coord.x >= size.x || coord.y >= size.y) {
            return;
//          return;
        }
//      }

        vec4 color = imageLoad(textureInput, coord);
//      vec4 color = imageLoad(textureInput, coord);

        color.rgb *= EXPOSURE;
//      color.rgb *= EXPOSURE;
        color.rgb = pbrNeutral(color.rgb);
//      color.rgb = pbrNeutral(color.rgb);
        color.rgb = pow(color.rgb, vec3(1.0 / DISPLAY_GAMMA));
//      color.rgb = pow(color.rgb, vec3(1.0 / DISPLAY_GAMMA));

        imageStore(textureOutput, coord, color);
//      imageStore(textureOutput, coord, color);
    }
//  }
