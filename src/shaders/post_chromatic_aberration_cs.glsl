    #version 430
//  #version 430
    layout(local_size_x = 16, local_size_y = 16) in;
//  layout(local_size_x = 16, local_size_y = 16) in;

    layout(binding = 0, rgba32f) uniform image2D textureOutput;
//  layout(binding = 0, rgba32f) uniform image2D textureOutput;
    layout(binding = 1, rgba32f) uniform image2D textureInput;
//  layout(binding = 1, rgba32f) uniform image2D textureInput;

    // Amount of shift for red and blue channels
    // Amount of shift for red and blue channels
    uniform float uAmount = 0.01;
//  uniform float uAmount = 0.01;

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

        vec2 uv = vec2(coord) / vec2(size);
//      vec2 uv = vec2(coord) / vec2(size);
        vec2 dir = uv - 0.5;
//      vec2 dir = uv - 0.5;

        // Enhance: use cubic radial distortion for more realistic lens effect
        // Enhance: use cubic radial distortion for more realistic lens effect
        vec2 offset = dir * dot(dir, dir) * uAmount * 4.0;
//      vec2 offset = dir * dot(dir, dir) * uAmount * 4.0;

        // Enhance: Spectral multi-tap blur for smooth chromatic aberration
        // Enhance: Spectral multi-tap blur for smooth chromatic aberration
        vec3 sum = vec3(0.0);
//      vec3 sum = vec3(0.0);
        vec3 wsum = vec3(0.0);
//      vec3 wsum = vec3(0.0);
        const int TAPS = 8;
//      const int TAPS = 8;

        for (int i = 0; i < TAPS; i++) {
//      for (int i = 0; i < TAPS; i++) {
            float t = float(i) / float(TAPS - 1);
//          float t = float(i) / float(TAPS - 1);
            float shift = (t - 0.5) * 2.0;
//          float shift = (t - 0.5) * 2.0;

            ivec2 c = clamp(ivec2((uv + offset * shift) * vec2(size)), ivec2(0), size - ivec2(1));
//          ivec2 c = clamp(ivec2((uv + offset * shift) * vec2(size)), ivec2(0), size - ivec2(1));
            vec3 smp = imageLoad(textureInput, c).rgb;
//          vec3 smp = imageLoad(textureInput, c).rgb;

            vec3 w = vec3(max(0.0, 1.0 - abs(shift - 1.0)), max(0.0, 1.0 - abs(shift)), max(0.0, 1.0 - abs(shift + 1.0)));
//          vec3 w = vec3(max(0.0, 1.0 - abs(shift - 1.0)), max(0.0, 1.0 - abs(shift)), max(0.0, 1.0 - abs(shift + 1.0)));

            sum += smp * w;
//          sum += smp * w;
            wsum += w;
//          wsum += w;
        }
//      }

        imageStore(textureOutput, coord, vec4(sum / wsum, 1.0));
//      imageStore(textureOutput, coord, vec4(sum / wsum, 1.0));
    }
//  }