    // Interpolated texture coordinate written by hybrid_renderer_vs.slang, matched by semantic.
//  // Interpolated texture coordinate written by hybrid_renderer_vs.slang, matched by semantic.
    struct ScreenFragmentInput {
//  struct ScreenFragmentInput {
        float2 inScreenFragmentUV : TEXCOORD0;
//      float2 inScreenFragmentUV : TEXCOORD0;
    };
//  };

    Sampler2D<float4> uTextureOutput;
//  Sampler2D<float4> uTextureOutput;
    uniform int uRenderMode;
//  uniform int uRenderMode;

    static const float DISPLAY_GAMMA = 2.2;
//  static const float DISPLAY_GAMMA = 2.2;

    [shader("fragment")]
//  [shader("fragment")]
    float4 fragmentMain(ScreenFragmentInput input) : SV_Target0 {
//  float4 fragmentMain(ScreenFragmentInput input) : SV_Target0 {
        float4 fragmentColor;
//      float4 fragmentColor;
        float4 val = uTextureOutput.Sample(input.inScreenFragmentUV);
//      float4 val = uTextureOutput.Sample(input.inScreenFragmentUV);
        if (uRenderMode != 0) {
//      if (uRenderMode != 0) {
            if (uRenderMode == 1) {
//          if (uRenderMode == 1) {
                fragmentColor = float4(val.rgb, 1.0);
//              fragmentColor = float4(val.rgb, 1.0);
            } else if (uRenderMode == 2) {
//          } else if (uRenderMode == 2) {
                fragmentColor = float4(val.rgb * 0.5 + 0.5, 1.0);
//              fragmentColor = float4(val.rgb * 0.5 + 0.5, 1.0);
            } else if (uRenderMode == 3) {
//          } else if (uRenderMode == 3) {
                fragmentColor = float4(frac(val.rgb), 1.0);
//              fragmentColor = float4(frac(val.rgb), 1.0);
            } else if (uRenderMode == 4) {
//          } else if (uRenderMode == 4) {
                fragmentColor = float4(val.rgb * 0.5 + 0.5, 1.0);
//              fragmentColor = float4(val.rgb * 0.5 + 0.5, 1.0);
            } else {
//          } else {
                fragmentColor = float4(1.0, 0.0, 1.0, 1.0);
//              fragmentColor = float4(1.0, 0.0, 1.0, 1.0);
            }
//          }
            fragmentColor.rgb = pow(fragmentColor.rgb, float3(1.0 / DISPLAY_GAMMA));
//          fragmentColor.rgb = pow(fragmentColor.rgb, float3(1.0 / DISPLAY_GAMMA));
        } else {
//      } else {
            fragmentColor = float4(val.rgb, 1.0);
//          fragmentColor = float4(val.rgb, 1.0);
        }
//      }
        return fragmentColor;
//      return fragmentColor;
    }
//  }
