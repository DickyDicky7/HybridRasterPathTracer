    import tonemap;
//  import tonemap;

    // Output is the f2 ping-pong pair: the tonemapped result is gamma-encoded LDR in [0, 1],
//  // Output is the f2 ping-pong pair: the tonemapped result is gamma-encoded LDR in [0, 1],
    // where rgba16f resolves ~8x finer than an 8-bit display at half the memory traffic.
//  // where rgba16f resolves ~8x finer than an 8-bit display at half the memory traffic.
    [format("rgba16f")] WTexture2D<float4> textureOutput;
//  [format("rgba16f")] WTexture2D<float4> textureOutput;
    // Input stays rgba32f: this pass MUST run first in the pipeline, reading the f4 HDR
//  // Input stays rgba32f: this pass MUST run first in the pipeline, reading the f4 HDR
    // accumulation buffer directly (all later passes declare their input rgba16f).
//  // accumulation buffer directly (all later passes declare their input rgba16f).
    Texture2D<float4> textureInput;
//  Texture2D<float4> textureInput;

    // Physically-based camera exposure (Lagarde & de Rousiers, "Moving Frostbite to PBR").
//  // Physically-based camera exposure (Lagarde & de Rousiers, "Moving Frostbite to PBR").
    // Exposure is derived from a real camera triangle instead of a hand-tuned gain. The
//  // Exposure is derived from a real camera triangle instead of a hand-tuned gain. The
    // scene's radiance is in arbitrary (non-physical) units, so EXPOSURE_CALIBRATION rescales
//  // scene's radiance is in arbitrary (non-physical) units, so EXPOSURE_CALIBRATION rescales
    // the physical result; its default makes the settings below reproduce the legacy 0.5 gain.
//  // the physical result; its default makes the settings below reproduce the legacy 0.5 gain.
    static const float APERTURE             = 16.0;    // f-number N    (smaller = brighter)
//  static const float APERTURE             = 16.0;    // f-number N    (smaller = brighter)
    static const float SHUTTER_TIME         = 0.008;   // seconds  t    (~1/125 s)
//  static const float SHUTTER_TIME         = 0.008;   // seconds  t    (~1/125 s)
    static const float ISO                  = 100.0;   // sensitivity S (larger = brighter)
//  static const float ISO                  = 100.0;   // sensitivity S (larger = brighter)
    static const float EXPOSURE_COMP        = 0.0;     // artistic bias in stops (+ brighter / - darker)
//  static const float EXPOSURE_COMP        = 0.0;     // artistic bias in stops (+ brighter / - darker)
    static const float EXPOSURE_CALIBRATION = 19200.0; // arbitrary-units -> physical-luminance scale
//  static const float EXPOSURE_CALIBRATION = 19200.0; // arbitrary-units -> physical-luminance scale

    static const float DISPLAY_GAMMA = 2.2;
//  static const float DISPLAY_GAMMA = 2.2;

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

    [shader("compute")]
//  [shader("compute")]
    [numthreads(16, 16, 1)]
//  [numthreads(16, 16, 1)]
    void computeMain(uint3 globalInvocationID : SV_DispatchThreadID, uint localInvocationIndex : SV_GroupIndex) {
//  void computeMain(uint3 globalInvocationID : SV_DispatchThreadID, uint localInvocationIndex : SV_GroupIndex) {
        int2 coord = int2(globalInvocationID.xy);
//      int2 coord = int2(globalInvocationID.xy);
        uint2 sizeUnsigned;
//      uint2 sizeUnsigned;
        textureOutput.GetDimensions(sizeUnsigned.x, sizeUnsigned.y);
//      textureOutput.GetDimensions(sizeUnsigned.x, sizeUnsigned.y);
        int2 size = int2(sizeUnsigned);
//      int2 size = int2(sizeUnsigned);

        if (coord.x >= size.x || coord.y >= size.y) {
//      if (coord.x >= size.x || coord.y >= size.y) {
            return;
//          return;
        }
//      }

        float4 color = textureInput.Load(int3(coord, 0));
//      float4 color = textureInput.Load(int3(coord, 0));

        float exposure = computeExposure();
//      float exposure = computeExposure();
        color.rgb *= exposure;
//      color.rgb *= exposure;
        color.rgb = gtTonemap(color.rgb);
//      color.rgb = gtTonemap(color.rgb);
        color.rgb = pow(color.rgb, float3(1.0 / DISPLAY_GAMMA));
//      color.rgb = pow(color.rgb, float3(1.0 / DISPLAY_GAMMA));

        textureOutput.Store(coord, color);
//      textureOutput.Store(coord, color);
    }
//  }
