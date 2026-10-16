    // Interpolated attributes written by hybrid_geometry_vs.slang, matched by semantic.
//  // Interpolated attributes written by hybrid_geometry_vs.slang, matched by semantic.
    struct GeometryFragmentInput {
//  struct GeometryFragmentInput {
        float3 inGeometryGlobalPosition : TEXCOORD0;
//      float3 inGeometryGlobalPosition : TEXCOORD0;
        float3 inGeometryGlobalNormal : TEXCOORD1;
//      float3 inGeometryGlobalNormal : TEXCOORD1;
        float3 inGeometryGlobalTangent : TEXCOORD2;
//      float3 inGeometryGlobalTangent : TEXCOORD2;
        float2 inGeometryUV : TEXCOORD3;
//      float2 inGeometryUV : TEXCOORD3;
        nointerpolation int vMaterialIndex : TEXCOORD4;
//      nointerpolation int vMaterialIndex : TEXCOORD4;
        nointerpolation int vTriangleIndex : TEXCOORD5;
//      nointerpolation int vTriangleIndex : TEXCOORD5;
    };
//  };

    struct GeometryFragmentOutput {
//  struct GeometryFragmentOutput {
        float4 outGeometryGlobalPosition : SV_Target0;
//      float4 outGeometryGlobalPosition : SV_Target0;
        float4 outGeometryGlobalNormal : SV_Target1;
//      float4 outGeometryGlobalNormal : SV_Target1;
        float4 outGeometryAlbedo : SV_Target2;
//      float4 outGeometryAlbedo : SV_Target2;
        float4 outGeometryGlobalTangent : SV_Target3;
//      float4 outGeometryGlobalTangent : SV_Target3;
    };
//  };

    struct Material {
//  struct Material {
        float4 albedo; // r, g, b, padding
//      float4 albedo; // r, g, b, padding
        float roughness;
//      float roughness;
        float metallic;
//      float metallic;
        float transmission;
//      float transmission;
        float ior;
//      float ior;
        float textureIndexAlbedo;
//      float textureIndexAlbedo;
        float textureIndexRoughness;
//      float textureIndexRoughness;
        float textureIndexMetallic;
//      float textureIndexMetallic;
        float textureIndexNormal;
//      float textureIndexNormal;
        float emissive;
//      float emissive;
        float textureIndexEmissive;
//      float textureIndexEmissive;
        float textureIndexTransmission;
//      float textureIndexTransmission;
        float padding002;
//      float padding002;
        float2 uvScale;
//      float2 uvScale;
        float2 padding003;
//      float2 padding003;
    };
//  };

    StructuredBuffer<Material> materials;
//  StructuredBuffer<Material> materials;

    Sampler2DArray<float4> uSceneTextureArray;
//  Sampler2DArray<float4> uSceneTextureArray;

    [shader("fragment")]
//  [shader("fragment")]
    GeometryFragmentOutput fragmentMain(GeometryFragmentInput input) {
//  GeometryFragmentOutput fragmentMain(GeometryFragmentInput input) {
        GeometryFragmentOutput output;
//      GeometryFragmentOutput output;
        int globalTriIdx = input.vTriangleIndex;
//      int globalTriIdx = input.vTriangleIndex;
        // Primitive Identity Packing: Stores the global triangle index in the 'w' component. We offset the index by +1.0 during encoding so that a 0.0 value definitively represents an empty background/sky-miss, eliminating potential ambiguity with index 0 while retrieving geometry hits later.
//      // Primitive Identity Packing: Stores the global triangle index in the 'w' component. We offset the index by +1.0 during encoding so that a 0.0 value definitively represents an empty background/sky-miss, eliminating potential ambiguity with index 0 while retrieving geometry hits later.
        output.outGeometryGlobalPosition = float4(input.inGeometryGlobalPosition, float(globalTriIdx + 1));
//      output.outGeometryGlobalPosition = float4(input.inGeometryGlobalPosition, float(globalTriIdx + 1));

        output.outGeometryGlobalNormal = float4(normalize(input.inGeometryGlobalNormal), input.inGeometryUV.x);
//      output.outGeometryGlobalNormal = float4(normalize(input.inGeometryGlobalNormal), input.inGeometryUV.x);

        Material mat = materials[input.vMaterialIndex];
//      Material mat = materials[input.vMaterialIndex];
        float3 finalAlbedo = mat.albedo.rgb;
//      float3 finalAlbedo = mat.albedo.rgb;
        // Explicit Gradients: WGSL only permits implicit-derivative sampling in uniform control flow, and the branch below depends on the per-triangle material. Taking the screen-space UV derivatives up front and sampling with them selects the same mip level and anisotropic footprint an implicit-LOD lookup would.
//      // Explicit Gradients: WGSL only permits implicit-derivative sampling in uniform control flow, and the branch below depends on the per-triangle material. Taking the screen-space UV derivatives up front and sampling with them selects the same mip level and anisotropic footprint an implicit-LOD lookup would.
        float2 scaledUV = input.inGeometryUV * mat.uvScale;
//      float2 scaledUV = input.inGeometryUV * mat.uvScale;
        float2 scaledUVDdx = ddx(scaledUV);
//      float2 scaledUVDdx = ddx(scaledUV);
        float2 scaledUVDdy = ddy(scaledUV);
//      float2 scaledUVDdy = ddy(scaledUV);
        if (mat.textureIndexAlbedo > -0.5) {
//      if (mat.textureIndexAlbedo > -0.5) {
            // Albedo is stored sRGB-encoded in 8-bit; decode to linear here
//          // Albedo is stored sRGB-encoded in 8-bit; decode to linear here
            finalAlbedo *= pow(uSceneTextureArray.SampleGrad(float3(scaledUV, mat.textureIndexAlbedo), scaledUVDdx, scaledUVDdy).rgb, float3(2.2));
//          finalAlbedo *= pow(uSceneTextureArray.SampleGrad(float3(scaledUV, mat.textureIndexAlbedo), scaledUVDdx, scaledUVDdy).rgb, float3(2.2));
        }
//      }
        output.outGeometryAlbedo = float4(finalAlbedo, 1.0);
//      output.outGeometryAlbedo = float4(finalAlbedo, 1.0);

        output.outGeometryGlobalTangent = float4(normalize(input.inGeometryGlobalTangent), input.inGeometryUV.y);
//      output.outGeometryGlobalTangent = float4(normalize(input.inGeometryGlobalTangent), input.inGeometryUV.y);
        return output;
//      return output;
    }
//  }
