    #version 430
//  #version 430
    layout(local_size_x = 16, local_size_y = 16) in;
//  layout(local_size_x = 16, local_size_y = 16) in;

    layout(binding = 0, rgba32f) uniform image2D textureOutput;
//  layout(binding = 0, rgba32f) uniform image2D textureOutput;
    layout(binding = 1, rgba32f) uniform image2D textureInput;
//  layout(binding = 1, rgba32f) uniform image2D textureInput;

    // Chromatic Aberration Shift: Controls the fractional displacement severity applied specifically to the red and blue color wavelengths across the lens. This simulates the optical dispersion artifacts of physical camera lenses where varying light frequencies refract at slightly different angles.
    // Chromatic Aberration Shift: Controls the fractional displacement severity applied specifically to the red and blue color wavelengths across the lens. This simulates the optical dispersion artifacts of physical camera lenses where varying light frequencies refract at slightly different angles.
    uniform float uAmount = 0.01;
//  uniform float uAmount = 0.01;

    // Constants
//  // Constants
    const float LENS_DISTORTION = 4.0;
//  const float LENS_DISTORTION = 4.0;
    const int CHROMA_TAPS = 8;
//  const int CHROMA_TAPS = 8;

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

        // Lens Distortion Mapping: Applies a non-linear cubic radial distortion algorithm to realistically skew the screen coordinates. By multiplying the directional vector by its squared magnitude, aberration intensity correctly scales exponentially towards the outer fringes of the frame.
        // Lens Distortion Mapping: Applies a non-linear cubic radial distortion algorithm to realistically skew the screen coordinates. By multiplying the directional vector by its squared magnitude, aberration intensity correctly scales exponentially towards the outer fringes of the frame.
        vec2 offset = dir * dot(dir, dir) * uAmount * LENS_DISTORTION;
//      vec2 offset = dir * dot(dir, dir) * uAmount * LENS_DISTORTION;

        // Spectral Multi-Tap Integration: Replaces the basic multi-channel sample shift with a continuous fractional integration loop across a spectrum curve. This blends adjacent shifted samples to generate an organically smoothed color fringe, effectively preventing hard edge artifact stepping.
        // Spectral Multi-Tap Integration: Replaces the basic multi-channel sample shift with a continuous fractional integration loop across a spectrum curve. This blends adjacent shifted samples to generate an organically smoothed color fringe, effectively preventing hard edge artifact stepping.
        vec3 sum = vec3(0.0);
//      vec3 sum = vec3(0.0);
        vec3 wsum = vec3(0.0);
//      vec3 wsum = vec3(0.0);

        for (int i = 0; i < CHROMA_TAPS; i++) {
//      for (int i = 0; i < CHROMA_TAPS; i++) {
            float t = float(i) / float(CHROMA_TAPS - 1);
//          float t = float(i) / float(CHROMA_TAPS - 1);
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