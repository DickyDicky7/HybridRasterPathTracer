    #version 430
//  #version 430

    layout(local_size_x = 16, local_size_y = 16) in;
//  layout(local_size_x = 16, local_size_y = 16) in;

    layout(binding = 0, rgba32f) uniform image2D textureOutput;
//  layout(binding = 0, rgba32f) uniform image2D textureOutput;
    layout(binding = 1) uniform sampler2D textureInput;
//  layout(binding = 1) uniform sampler2D textureInput;

    // High-Fidelity FXAA Configuration
//  // High-Fidelity FXAA Configuration
    #define FXAA_SPAN_MAX 16.0
//  #define FXAA_SPAN_MAX 16.0
    #define FXAA_REDUCE_MUL (1.0 / 16.0)
//  #define FXAA_REDUCE_MUL (1.0 / 16.0)
    #define FXAA_REDUCE_MIN (1.0 / 256.0)
//  #define FXAA_REDUCE_MIN (1.0 / 256.0)
    #define FXAA_EDGE_THRESHOLD (1.0 / 8.0)
//  #define FXAA_EDGE_THRESHOLD (1.0 / 8.0)
    #define FXAA_EDGE_THRESHOLD_MIN (1.0 / 32.0)
//  #define FXAA_EDGE_THRESHOLD_MIN (1.0 / 32.0)
    #define FXAA_SUBPIX_QUALITY 0.75
//  #define FXAA_SUBPIX_QUALITY 0.75

    const vec3 LUMA_WEIGHTS = vec3(0.299, 0.587, 0.114);
//  const vec3 LUMA_WEIGHTS = vec3(0.299, 0.587, 0.114);

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

        vec2 texCoordOffset = 1.0 / vec2(size);
//      vec2 texCoordOffset = 1.0 / vec2(size);
        vec2 uv = (vec2(coord) + 0.5) * texCoordOffset;
//      vec2 uv = (vec2(coord) + 0.5) * texCoordOffset;

        vec3 luma = LUMA_WEIGHTS;
//      vec3 luma = LUMA_WEIGHTS;
        vec3 rgbM = texture(textureInput, uv).rgb;
//      vec3 rgbM = texture(textureInput, uv).rgb;
        float lumaM  = dot(luma, rgbM);
//      float lumaM  = dot(luma, rgbM);
        float lumaN  = dot(luma, texture(textureInput, uv + vec2( 0.0, -1.0) * texCoordOffset).rgb);
//      float lumaN  = dot(luma, texture(textureInput, uv + vec2( 0.0, -1.0) * texCoordOffset).rgb);
        float lumaS  = dot(luma, texture(textureInput, uv + vec2( 0.0,  1.0) * texCoordOffset).rgb);
//      float lumaS  = dot(luma, texture(textureInput, uv + vec2( 0.0,  1.0) * texCoordOffset).rgb);
        float lumaW  = dot(luma, texture(textureInput, uv + vec2(-1.0,  0.0) * texCoordOffset).rgb);
//      float lumaW  = dot(luma, texture(textureInput, uv + vec2(-1.0,  0.0) * texCoordOffset).rgb);
        float lumaE  = dot(luma, texture(textureInput, uv + vec2( 1.0,  0.0) * texCoordOffset).rgb);
//      float lumaE  = dot(luma, texture(textureInput, uv + vec2( 1.0,  0.0) * texCoordOffset).rgb);
        float lumaNW = dot(luma, texture(textureInput, uv + vec2(-1.0, -1.0) * texCoordOffset).rgb);
//      float lumaNW = dot(luma, texture(textureInput, uv + vec2(-1.0, -1.0) * texCoordOffset).rgb);
        float lumaNE = dot(luma, texture(textureInput, uv + vec2( 1.0, -1.0) * texCoordOffset).rgb);
//      float lumaNE = dot(luma, texture(textureInput, uv + vec2( 1.0, -1.0) * texCoordOffset).rgb);
        float lumaSW = dot(luma, texture(textureInput, uv + vec2(-1.0,  1.0) * texCoordOffset).rgb);
//      float lumaSW = dot(luma, texture(textureInput, uv + vec2(-1.0,  1.0) * texCoordOffset).rgb);
        float lumaSE = dot(luma, texture(textureInput, uv + vec2( 1.0,  1.0) * texCoordOffset).rgb);
//      float lumaSE = dot(luma, texture(textureInput, uv + vec2( 1.0,  1.0) * texCoordOffset).rgb);

        float lumaMin = min(lumaM, min(min(lumaN, lumaS), min(lumaW, lumaE)));
//      float lumaMin = min(lumaM, min(min(lumaN, lumaS), min(lumaW, lumaE)));
        float lumaMax = max(lumaM, max(max(lumaN, lumaS), max(lumaW, lumaE)));
//      float lumaMax = max(lumaM, max(max(lumaN, lumaS), max(lumaW, lumaE)));
        float lumaRange = lumaMax - lumaMin;
//      float lumaRange = lumaMax - lumaMin;

        if (lumaRange < max(FXAA_EDGE_THRESHOLD_MIN, lumaMax * FXAA_EDGE_THRESHOLD)) {
//      if (lumaRange < max(FXAA_EDGE_THRESHOLD_MIN, lumaMax * FXAA_EDGE_THRESHOLD)) {
            imageStore(textureOutput, coord, vec4(rgbM, 1.0));
//          imageStore(textureOutput, coord, vec4(rgbM, 1.0));
            return;
//          return;
        }
//      }

        float lumaAvg = (lumaN + lumaS + lumaW + lumaE) * 0.25;
//      float lumaAvg = (lumaN + lumaS + lumaW + lumaE) * 0.25;
        float subpixAlias = clamp(abs(lumaAvg - lumaM) / lumaRange, 0.0, 1.0);
//      float subpixAlias = clamp(abs(lumaAvg - lumaM) / lumaRange, 0.0, 1.0);
        float subpixBlend = smoothstep(0.0, 1.0, subpixAlias) * smoothstep(0.0, 1.0, subpixAlias) * FXAA_SUBPIX_QUALITY;
//      float subpixBlend = smoothstep(0.0, 1.0, subpixAlias) * smoothstep(0.0, 1.0, subpixAlias) * FXAA_SUBPIX_QUALITY;

        vec2 dir;
//      vec2 dir;
        dir.x = -((lumaNW + lumaNE) - (lumaSW + lumaSE));
//      dir.x = -((lumaNW + lumaNE) - (lumaSW + lumaSE));
        dir.y =  ((lumaNW + lumaSW) - (lumaNE + lumaSE));
//      dir.y =  ((lumaNW + lumaSW) - (lumaNE + lumaSE));

        float dirReduce = max(
//      float dirReduce = max(
            (lumaNW + lumaNE + lumaSW + lumaSE) * (0.25 * FXAA_REDUCE_MUL),
//          (lumaNW + lumaNE + lumaSW + lumaSE) * (0.25 * FXAA_REDUCE_MUL),
            FXAA_REDUCE_MIN);
//          FXAA_REDUCE_MIN);

        float rcpDirMin = 1.0 / (min(abs(dir.x), abs(dir.y)) + dirReduce);
//      float rcpDirMin = 1.0 / (min(abs(dir.x), abs(dir.y)) + dirReduce);

        dir = min(vec2(FXAA_SPAN_MAX, FXAA_SPAN_MAX),
//      dir = min(vec2(FXAA_SPAN_MAX, FXAA_SPAN_MAX),
            max(vec2(-FXAA_SPAN_MAX, -FXAA_SPAN_MAX),
//          max(vec2(-FXAA_SPAN_MAX, -FXAA_SPAN_MAX),
                dir * rcpDirMin)) * texCoordOffset;
//              dir * rcpDirMin)) * texCoordOffset;

        vec3 rgbA = (1.0 / 2.0) * (
//      vec3 rgbA = (1.0 / 2.0) * (
            texture(textureInput, uv + dir * (1.0 / 3.0 - 0.5)).xyz +
//          texture(textureInput, uv + dir * (1.0 / 3.0 - 0.5)).xyz +
            texture(textureInput, uv + dir * (2.0 / 3.0 - 0.5)).xyz);
//          texture(textureInput, uv + dir * (2.0 / 3.0 - 0.5)).xyz);
        vec3 rgbB = rgbA * (1.0 / 2.0) + (1.0 / 4.0) * (
//      vec3 rgbB = rgbA * (1.0 / 2.0) + (1.0 / 4.0) * (
            texture(textureInput, uv + dir * (0.0 / 3.0 - 0.5)).xyz +
//          texture(textureInput, uv + dir * (0.0 / 3.0 - 0.5)).xyz +
            texture(textureInput, uv + dir * (3.0 / 3.0 - 0.5)).xyz);
//          texture(textureInput, uv + dir * (3.0 / 3.0 - 0.5)).xyz);
        float lumaB = dot(rgbB, luma);
//      float lumaB = dot(rgbB, luma);

        vec3 edgeResult;
//      vec3 edgeResult;
        if ((lumaB < lumaMin) || (lumaB > lumaMax)) {
//      if ((lumaB < lumaMin) || (lumaB > lumaMax)) {
            edgeResult = rgbA;
//          edgeResult = rgbA;
        } else {
//      } else {
            edgeResult = rgbB;
//          edgeResult = rgbB;
        }
//      }

        vec3 finalColor = mix(rgbM, edgeResult, subpixBlend);
//      vec3 finalColor = mix(rgbM, edgeResult, subpixBlend);
        imageStore(textureOutput, coord, vec4(finalColor, 1.0));
//      imageStore(textureOutput, coord, vec4(finalColor, 1.0));
    }
//  }
