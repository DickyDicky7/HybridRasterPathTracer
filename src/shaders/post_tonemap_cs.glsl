    #version 430
//  #version 430
    layout(local_size_x = 16, local_size_y = 16) in;
//  layout(local_size_x = 16, local_size_y = 16) in;

    layout(binding = 0, rgba32f) uniform image2D textureOutput;
//  layout(binding = 0, rgba32f) uniform image2D textureOutput;
    layout(binding = 1, rgba32f) uniform image2D textureInput;
//  layout(binding = 1, rgba32f) uniform image2D textureInput;

    // Physically-based camera exposure (Lagarde & de Rousiers, "Moving Frostbite to PBR").
//  // Physically-based camera exposure (Lagarde & de Rousiers, "Moving Frostbite to PBR").
    // Exposure is derived from a real camera triangle instead of a hand-tuned gain. The
//  // Exposure is derived from a real camera triangle instead of a hand-tuned gain. The
    // scene's radiance is in arbitrary (non-physical) units, so EXPOSURE_CALIBRATION rescales
//  // scene's radiance is in arbitrary (non-physical) units, so EXPOSURE_CALIBRATION rescales
    // the physical result; its default makes the settings below reproduce the legacy 0.5 gain.
//  // the physical result; its default makes the settings below reproduce the legacy 0.5 gain.
    const float APERTURE             = 16.0;    // f-number N    (smaller = brighter)
//  const float APERTURE             = 16.0;    // f-number N    (smaller = brighter)
    const float SHUTTER_TIME         = 0.008;   // seconds  t    (~1/125 s)
//  const float SHUTTER_TIME         = 0.008;   // seconds  t    (~1/125 s)
    const float ISO                  = 100.0;   // sensitivity S (larger = brighter)
//  const float ISO                  = 100.0;   // sensitivity S (larger = brighter)
    const float EXPOSURE_COMP        = 0.0;     // artistic bias in stops (+ brighter / - darker)
//  const float EXPOSURE_COMP        = 0.0;     // artistic bias in stops (+ brighter / - darker)
    const float EXPOSURE_CALIBRATION = 19200.0; // arbitrary-units -> physical-luminance scale
//  const float EXPOSURE_CALIBRATION = 19200.0; // arbitrary-units -> physical-luminance scale

    #include "tonemap.glsl"

    const float DISPLAY_GAMMA = 2.2;
//  const float DISPLAY_GAMMA = 2.2;

    // EV100 from the camera triangle, then the Saturation-Based Sensitivity exposure
//  // EV100 from the camera triangle, then the Saturation-Based Sensitivity exposure
    // (maxLuminance = 1.2 * 2^EV100). Subtracting EXPOSURE_COMP brightens; the calibration
//  // (maxLuminance = 1.2 * 2^EV100). Subtracting EXPOSURE_COMP brightens; the calibration
    // scalar maps the renderer's arbitrary radiance units onto that physical exposure.
//  // scalar maps the renderer's arbitrary radiance units onto that physical exposure.
    float computeExposure() {
//  float computeExposure() {
        float ev100 = log2((APERTURE * APERTURE) / SHUTTER_TIME * 100.0 / ISO) - EXPOSURE_COMP;
//      float ev100 = log2((APERTURE * APERTURE) / SHUTTER_TIME * 100.0 / ISO) - EXPOSURE_COMP;
        float maxLuminance = 1.2 * exp2(ev100);
//      float maxLuminance = 1.2 * exp2(ev100);
        return EXPOSURE_CALIBRATION / max(maxLuminance, 1e-4);
//      return EXPOSURE_CALIBRATION / max(maxLuminance, 1e-4);
    }
//  }

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

        float exposure = computeExposure();
//      float exposure = computeExposure();
        color.rgb *= exposure;
//      color.rgb *= exposure;
        color.rgb = gtTonemap(color.rgb);
//      color.rgb = gtTonemap(color.rgb);
        color.rgb = pow(color.rgb, vec3(1.0 / DISPLAY_GAMMA));
//      color.rgb = pow(color.rgb, vec3(1.0 / DISPLAY_GAMMA));

        imageStore(textureOutput, coord, color);
//      imageStore(textureOutput, coord, color);
    }
//  }
