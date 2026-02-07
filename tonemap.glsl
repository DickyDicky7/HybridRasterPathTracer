
    vec3 aces(vec3 x) {
//  vec3 aces(vec3 x) {
        const float a = 2.51;
//      const float a = 2.51;
        const float b = 0.03;
//      const float b = 0.03;
        const float c = 2.43;
//      const float c = 2.43;
        const float d = 0.59;
//      const float d = 0.59;
        const float e = 0.14;
//      const float e = 0.14;
        return clamp((x * (a * x + b)) / (x * (c * x + d) + e), 0.0, 1.0);
//      return clamp((x * (a * x + b)) / (x * (c * x + d) + e), 0.0, 1.0);
    }
//  }

    vec3 reinhard(vec3 x) {
//  vec3 reinhard(vec3 x) {
        return x / (x + vec3(1.0));
//      return x / (x + vec3(1.0));
    }
//  }

    // Khronos PBR Neutral Tone Mapper
    // Khronos PBR Neutral Tone Mapper
    vec3 pbrNeutral(vec3 color) {
//  vec3 pbrNeutral(vec3 color) {
        const float startCompression = 0.8 - 0.04;
//      const float startCompression = 0.8 - 0.04;
        const float desaturation = 0.15;
//      const float desaturation = 0.15;

        float x = min(color.r, min(color.g, color.b));
//      float x = min(color.r, min(color.g, color.b));
        float offset = x < 0.08 ? x - 6.25 * x * x : 0.04;
//      float offset = x < 0.08 ? x - 6.25 * x * x : 0.04;
        color -= offset;
//      color -= offset;

        float peak = max(color.r, max(color.g, color.b));
//      float peak = max(color.r, max(color.g, color.b));
        if (peak < startCompression) return color;
//      if (peak < startCompression) return color;

        const float d = 1.0 - startCompression;
//      const float d = 1.0 - startCompression;
        float newPeak = 1.0 - d * d / (peak + d - startCompression);
//      float newPeak = 1.0 - d * d / (peak + d - startCompression);
        color *= newPeak / peak;
//      color *= newPeak / peak;

        float g = 1.0 - 1.0 / (desaturation * (peak - newPeak) + 1.0);
//      float g = 1.0 - 1.0 / (desaturation * (peak - newPeak) + 1.0);
        return mix(color, newPeak * vec3(1.0, 1.0, 1.0), g);
//      return mix(color, newPeak * vec3(1.0, 1.0, 1.0), g);
    }
//  }
