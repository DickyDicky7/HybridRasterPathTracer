    // Fullscreen-quad vertex stream: clip-space corner position followed by its texture coordinate.
//  // Fullscreen-quad vertex stream: clip-space corner position followed by its texture coordinate.
    struct ScreenVertexInput {
//  struct ScreenVertexInput {
        float2 inScreenVertexPosition : POSITION0;
//      float2 inScreenVertexPosition : POSITION0;
        float2 inScreenVertexUV : TEXCOORD0;
//      float2 inScreenVertexUV : TEXCOORD0;
    };
//  };

    struct ScreenVertexOutput {
//  struct ScreenVertexOutput {
        float4 position : SV_Position;
//      float4 position : SV_Position;
        float2 outScreenVertexUV : TEXCOORD0;
//      float2 outScreenVertexUV : TEXCOORD0;
    };
//  };

    [shader("vertex")]
//  [shader("vertex")]
    ScreenVertexOutput vertexMain(ScreenVertexInput input) {
//  ScreenVertexOutput vertexMain(ScreenVertexInput input) {
        ScreenVertexOutput output;
//      ScreenVertexOutput output;
        output.outScreenVertexUV = input.inScreenVertexUV;
//      output.outScreenVertexUV = input.inScreenVertexUV;
        output.position = float4(input.inScreenVertexPosition, 0.0, 1.0);
//      output.position = float4(input.inScreenVertexPosition, 0.0, 1.0);
        return output;
//      return output;
    }
//  }
