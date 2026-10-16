    // Interleaved scene vertex stream (three float4 per vertex, see SceneBuilder.build): the texture
//  // Interleaved scene vertex stream (three float4 per vertex, see SceneBuilder.build): the texture
    // coordinate rides in the .w lanes of position and normal, the material index in the tangent's .w.
//  // coordinate rides in the .w lanes of position and normal, the material index in the tangent's .w.
    struct GeometryVertexInput {
//  struct GeometryVertexInput {
        float4 inVertexGlobalPositionU : POSITION0;
//      float4 inVertexGlobalPositionU : POSITION0;
        float4 inVertexGlobalNormalV : NORMAL0;
//      float4 inVertexGlobalNormalV : NORMAL0;
        float4 inVertexGlobalTangentMat : TANGENT0;
//      float4 inVertexGlobalTangentMat : TANGENT0;
        uint vertexIndex : SV_VertexID;
//      uint vertexIndex : SV_VertexID;
    };
//  };

    struct GeometryVertexOutput {
//  struct GeometryVertexOutput {
        float4 position : SV_Position;
//      float4 position : SV_Position;
        float3 outVertexGlobalPosition : TEXCOORD0;
//      float3 outVertexGlobalPosition : TEXCOORD0;
        float3 outVertexGlobalNormal : TEXCOORD1;
//      float3 outVertexGlobalNormal : TEXCOORD1;
        float3 outVertexGlobalTangent : TEXCOORD2;
//      float3 outVertexGlobalTangent : TEXCOORD2;
        float2 outVertexUV : TEXCOORD3;
//      float2 outVertexUV : TEXCOORD3;
        nointerpolation int vMaterialIndex : TEXCOORD4;
//      nointerpolation int vMaterialIndex : TEXCOORD4;
        nointerpolation int vTriangleIndex : TEXCOORD5;
//      nointerpolation int vTriangleIndex : TEXCOORD5;
    };
//  };

    uniform float4x4 uTransformView;
//  uniform float4x4 uTransformView;
    uniform float4x4 uTransformProjection;
//  uniform float4x4 uTransformProjection;

    [shader("vertex")]
//  [shader("vertex")]
    GeometryVertexOutput vertexMain(GeometryVertexInput input) {
//  GeometryVertexOutput vertexMain(GeometryVertexInput input) {
        GeometryVertexOutput output;
//      GeometryVertexOutput output;
        output.outVertexGlobalPosition = input.inVertexGlobalPositionU.xyz;
//      output.outVertexGlobalPosition = input.inVertexGlobalPositionU.xyz;
        output.outVertexGlobalNormal = input.inVertexGlobalNormalV.xyz;
//      output.outVertexGlobalNormal = input.inVertexGlobalNormalV.xyz;
        output.outVertexGlobalTangent = input.inVertexGlobalTangentMat.xyz;
//      output.outVertexGlobalTangent = input.inVertexGlobalTangentMat.xyz;
        output.outVertexUV = float2(input.inVertexGlobalPositionU.w, input.inVertexGlobalNormalV.w);
//      output.outVertexUV = float2(input.inVertexGlobalPositionU.w, input.inVertexGlobalNormalV.w);
        output.vMaterialIndex = int(input.inVertexGlobalTangentMat.w);
//      output.vMaterialIndex = int(input.inVertexGlobalTangentMat.w);
        // The scene is drawn as one non-indexed triangle list starting at vertex 0, so every three consecutive vertices form one triangle. This reproduces gl_PrimitiveID without requiring the optional WebGPU primitive-index feature.
//      // The scene is drawn as one non-indexed triangle list starting at vertex 0, so every three consecutive vertices form one triangle. This reproduces gl_PrimitiveID without requiring the optional WebGPU primitive-index feature.
        output.vTriangleIndex = int(input.vertexIndex / 3u);
//      output.vTriangleIndex = int(input.vertexIndex / 3u);
        float4 clipPosition = mul(uTransformProjection, mul(uTransformView, float4(output.outVertexGlobalPosition, 1.0)));
//      float4 clipPosition = mul(uTransformProjection, mul(uTransformView, float4(output.outVertexGlobalPosition, 1.0)));
        // Clip-Space Convention: The view / projection matrices follow the OpenGL convention (pyrr), but WebGPU clips depth to [0, w] instead of [-w, w] and stores framebuffer row 0 at the top instead of the bottom. Remap z, and mirror y so the G-buffer keeps the bottom-up row order the compute passes (and the uPixel00Coordinates camera basis) assume. The mirror reverses the triangle winding, which the renderer compensates for by declaring clockwise front faces.
//      // Clip-Space Convention: The view / projection matrices follow the OpenGL convention (pyrr), but WebGPU clips depth to [0, w] instead of [-w, w] and stores framebuffer row 0 at the top instead of the bottom. Remap z, and mirror y so the G-buffer keeps the bottom-up row order the compute passes (and the uPixel00Coordinates camera basis) assume. The mirror reverses the triangle winding, which the renderer compensates for by declaring clockwise front faces.
        clipPosition.y = -clipPosition.y;
//      clipPosition.y = -clipPosition.y;
        clipPosition.z = (clipPosition.z + clipPosition.w) * 0.5;
//      clipPosition.z = (clipPosition.z + clipPosition.w) * 0.5;
        output.position = clipPosition;
//      output.position = clipPosition;
        return output;
//      return output;
    }
//  }
