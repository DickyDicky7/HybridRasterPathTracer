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

    #include "tonemap.glsl"

    // Fast Approximate Anti-Aliasing (FXAA) Configuration: Defines algorithm thresholds for edge detection span lengths and sub-pixel blending multipliers to eliminate spatial aliasing and jagged geometry lines efficiently.
//  // Fast Approximate Anti-Aliasing (FXAA) Configuration: Defines algorithm thresholds for edge detection span lengths and sub-pixel blending multipliers to eliminate spatial aliasing and jagged geometry lines efficiently.
    #define FXAA_SPAN_MAX 8.0
//  #define FXAA_SPAN_MAX 8.0
    #define FXAA_REDUCE_MUL   (1.0/8.0)
//  #define FXAA_REDUCE_MUL   (1.0/8.0)
    #define FXAA_REDUCE_MIN   (1.0/128.0)
//  #define FXAA_REDUCE_MIN   (1.0/128.0)

    const vec3 LUMA_WEIGHTS = vec3(0.299, 0.587, 0.114);
//  const vec3 LUMA_WEIGHTS = vec3(0.299, 0.587, 0.114);
    const float DISPLAY_GAMMA = 2.2;
//  const float DISPLAY_GAMMA = 2.2;

    void main() {
//  void main() {
        if (uRenderMode != 0) {
//      if (uRenderMode != 0) {
            vec4 val = texture(uTextureOutput, inScreenFragmentUV);
//          vec4 val = texture(uTextureOutput, inScreenFragmentUV);
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
            fragmentColor.rgb = pow(fragmentColor.rgb, vec3(1.0/DISPLAY_GAMMA));
//          fragmentColor.rgb = pow(fragmentColor.rgb, vec3(1.0/DISPLAY_GAMMA));
            return;
//          return;
        }
//      }

        // Texel Dimension Normalization: Calculates the fractional step size of a single texture element in normalized coordinate space, essential for precision neighbor-sampling when fetching surrounding kernel pixels during post-processing analysis.
//      // Texel Dimension Normalization: Calculates the fractional step size of a single texture element in normalized coordinate space, essential for precision neighbor-sampling when fetching surrounding kernel pixels during post-processing analysis.
        vec2 texCoordOffset = 1.0 / vec2(textureSize(uTextureOutput, 0));
//      vec2 texCoordOffset = 1.0 / vec2(textureSize(uTextureOutput, 0));

        vec3 luma = LUMA_WEIGHTS;
//      vec3 luma = LUMA_WEIGHTS;
        float lumaTL = dot(luma, texture(uTextureOutput, inScreenFragmentUV + (vec2(-1.0, -1.0) * texCoordOffset)).rgb);
//      float lumaTL = dot(luma, texture(uTextureOutput, inScreenFragmentUV + (vec2(-1.0, -1.0) * texCoordOffset)).rgb);
        float lumaTR = dot(luma, texture(uTextureOutput, inScreenFragmentUV + (vec2(1.0, -1.0) * texCoordOffset)).rgb);
//      float lumaTR = dot(luma, texture(uTextureOutput, inScreenFragmentUV + (vec2(1.0, -1.0) * texCoordOffset)).rgb);
        float lumaBL = dot(luma, texture(uTextureOutput, inScreenFragmentUV + (vec2(-1.0, 1.0) * texCoordOffset)).rgb);
//      float lumaBL = dot(luma, texture(uTextureOutput, inScreenFragmentUV + (vec2(-1.0, 1.0) * texCoordOffset)).rgb);
        float lumaBR = dot(luma, texture(uTextureOutput, inScreenFragmentUV + (vec2(1.0, 1.0) * texCoordOffset)).rgb);
//      float lumaBR = dot(luma, texture(uTextureOutput, inScreenFragmentUV + (vec2(1.0, 1.0) * texCoordOffset)).rgb);
        float lumaM  = dot(luma, texture(uTextureOutput, inScreenFragmentUV).rgb);
//      float lumaM  = dot(luma, texture(uTextureOutput, inScreenFragmentUV).rgb);

        float lumaMin = min(lumaM, min(min(lumaTL, lumaTR), min(lumaBL, lumaBR)));
//      float lumaMin = min(lumaM, min(min(lumaTL, lumaTR), min(lumaBL, lumaBR)));
        float lumaMax = max(lumaM, max(max(lumaTL, lumaTR), max(lumaBL, lumaBR)));
//      float lumaMax = max(lumaM, max(max(lumaTL, lumaTR), max(lumaBL, lumaBR)));

        vec2 dir;
//      vec2 dir;
        dir.x = -((lumaTL + lumaTR) - (lumaBL + lumaBR));
//      dir.x = -((lumaTL + lumaTR) - (lumaBL + lumaBR));
        dir.y = ((lumaTL + lumaBL) - (lumaTR + lumaBR));
//      dir.y = ((lumaTL + lumaBL) - (lumaTR + lumaBR));

        float dirReduce = max(
//      float dirReduce = max(
            (lumaTL + lumaTR + lumaBL + lumaBR) * (0.25 * FXAA_REDUCE_MUL),
//          (lumaTL + lumaTR + lumaBL + lumaBR) * (0.25 * FXAA_REDUCE_MUL),
            FXAA_REDUCE_MIN);
//          FXAA_REDUCE_MIN);

        float rcpDirMin = 1.0/(min(abs(dir.x), abs(dir.y)) + dirReduce);
//      float rcpDirMin = 1.0/(min(abs(dir.x), abs(dir.y)) + dirReduce);

        dir = min(vec2( FXAA_SPAN_MAX,  FXAA_SPAN_MAX),
//      dir = min(vec2( FXAA_SPAN_MAX,  FXAA_SPAN_MAX),
            max(vec2(-FXAA_SPAN_MAX, -FXAA_SPAN_MAX),
//          max(vec2(-FXAA_SPAN_MAX, -FXAA_SPAN_MAX),
            dir * rcpDirMin)) * texCoordOffset;
//          dir * rcpDirMin)) * texCoordOffset;

        vec3 rgbA = (1.0/2.0) * (
//      vec3 rgbA = (1.0/2.0) * (
            texture(uTextureOutput, inScreenFragmentUV.xy + dir * (1.0/3.0 - 0.5)).xyz +
//          texture(uTextureOutput, inScreenFragmentUV.xy + dir * (1.0/3.0 - 0.5)).xyz +
            texture(uTextureOutput, inScreenFragmentUV.xy + dir * (2.0/3.0 - 0.5)).xyz);
//          texture(uTextureOutput, inScreenFragmentUV.xy + dir * (2.0/3.0 - 0.5)).xyz);
        vec3 rgbB = rgbA * (1.0/2.0) + (1.0/4.0) * (
//      vec3 rgbB = rgbA * (1.0/2.0) + (1.0/4.0) * (
            texture(uTextureOutput, inScreenFragmentUV.xy + dir * (0.0/3.0 - 0.5)).xyz +
//          texture(uTextureOutput, inScreenFragmentUV.xy + dir * (0.0/3.0 - 0.5)).xyz +
            texture(uTextureOutput, inScreenFragmentUV.xy + dir * (3.0/3.0 - 0.5)).xyz);
//          texture(uTextureOutput, inScreenFragmentUV.xy + dir * (3.0/3.0 - 0.5)).xyz);
        float lumaB = dot(rgbB, luma);
//      float lumaB = dot(rgbB, luma);

        if((lumaB < lumaMin) || (lumaB > lumaMax)){
//      if((lumaB < lumaMin) || (lumaB > lumaMax)){
            fragmentColor = vec4(rgbA, 1.0);
//          fragmentColor = vec4(rgbA, 1.0);
        } else {
//      } else {
            fragmentColor = vec4(rgbB, 1.0);
//          fragmentColor = vec4(rgbB, 1.0);
        }
//      }

        fragmentColor.rgb = pbrNeutral(fragmentColor.rgb);
//      fragmentColor.rgb = pbrNeutral(fragmentColor.rgb);
        fragmentColor.rgb = pow(fragmentColor.rgb, vec3(1.0/DISPLAY_GAMMA));
//      fragmentColor.rgb = pow(fragmentColor.rgb, vec3(1.0/DISPLAY_GAMMA));
    }
//  }
