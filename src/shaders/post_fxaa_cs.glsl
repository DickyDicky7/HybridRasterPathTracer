    #version 430
//  #version 430

    layout(local_size_x = 16, local_size_y = 16) in;
//  layout(local_size_x = 16, local_size_y = 16) in;

    layout(binding = 0, rgba32f) uniform image2D textureOutput;
//  layout(binding = 0, rgba32f) uniform image2D textureOutput;
    layout(binding = 1) uniform sampler2D textureInput;
//  layout(binding = 1) uniform sampler2D textureInput;

    // Fast Approximate Anti-Aliasing (FXAA) Configuration
//  // Fast Approximate Anti-Aliasing (FXAA) Configuration
    #define FXAA_SPAN_MAX 8.0
//  #define FXAA_SPAN_MAX 8.0
    #define FXAA_REDUCE_MUL (1.0 / 8.0)
//  #define FXAA_REDUCE_MUL (1.0 / 8.0)
    #define FXAA_REDUCE_MIN (1.0 / 128.0)
//  #define FXAA_REDUCE_MIN (1.0 / 128.0)

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
        float lumaTL = dot(luma, texture(textureInput, uv + (vec2(-1.0, -1.0) * texCoordOffset)).rgb);
//      float lumaTL = dot(luma, texture(textureInput, uv + (vec2(-1.0, -1.0) * texCoordOffset)).rgb);
        float lumaTR = dot(luma, texture(textureInput, uv + (vec2( 1.0, -1.0) * texCoordOffset)).rgb);
//      float lumaTR = dot(luma, texture(textureInput, uv + (vec2( 1.0, -1.0) * texCoordOffset)).rgb);
        float lumaBL = dot(luma, texture(textureInput, uv + (vec2(-1.0,  1.0) * texCoordOffset)).rgb);
//      float lumaBL = dot(luma, texture(textureInput, uv + (vec2(-1.0,  1.0) * texCoordOffset)).rgb);
        float lumaBR = dot(luma, texture(textureInput, uv + (vec2( 1.0,  1.0) * texCoordOffset)).rgb);
//      float lumaBR = dot(luma, texture(textureInput, uv + (vec2( 1.0,  1.0) * texCoordOffset)).rgb);
        float lumaM  = dot(luma, texture(textureInput, uv).rgb);
//      float lumaM  = dot(luma, texture(textureInput, uv).rgb);

        float lumaMin = min(lumaM, min(min(lumaTL, lumaTR), min(lumaBL, lumaBR)));
//      float lumaMin = min(lumaM, min(min(lumaTL, lumaTR), min(lumaBL, lumaBR)));
        float lumaMax = max(lumaM, max(max(lumaTL, lumaTR), max(lumaBL, lumaBR)));
//      float lumaMax = max(lumaM, max(max(lumaTL, lumaTR), max(lumaBL, lumaBR)));

        vec2 dir;
//      vec2 dir;
        dir.x = -((lumaTL + lumaTR) - (lumaBL + lumaBR));
//      dir.x = -((lumaTL + lumaTR) - (lumaBL + lumaBR));
        dir.y =  ((lumaTL + lumaBL) - (lumaTR + lumaBR));
//      dir.y =  ((lumaTL + lumaBL) - (lumaTR + lumaBR));

        float dirReduce = max(
//      float dirReduce = max(
            (lumaTL + lumaTR + lumaBL + lumaBR) * (0.25 * FXAA_REDUCE_MUL),
//          (lumaTL + lumaTR + lumaBL + lumaBR) * (0.25 * FXAA_REDUCE_MUL),
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

        if ((lumaB < lumaMin) || (lumaB > lumaMax)) {
//      if ((lumaB < lumaMin) || (lumaB > lumaMax)) {
            imageStore(textureOutput, coord, vec4(rgbA, 1.0));
//          imageStore(textureOutput, coord, vec4(rgbA, 1.0));
        } else {
//      } else {
            imageStore(textureOutput, coord, vec4(rgbB, 1.0));
//          imageStore(textureOutput, coord, vec4(rgbB, 1.0));
        }
//      }
    }
//  }
