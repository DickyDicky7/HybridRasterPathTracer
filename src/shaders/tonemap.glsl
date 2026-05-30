
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

    // Khronos PBR Neutral Tone Mapper: An advanced curve that gracefully compresses high dynamic range energy into standard display parameters while meticulously preserving the perceived hue and saturation of extreme light sources, preventing the common "yellow shift" seen in Reinhard or basic ACES.
    // Khronos PBR Neutral Tone Mapper: An advanced curve that gracefully compresses high dynamic range energy into standard display parameters while meticulously preserving the perceived hue and saturation of extreme light sources, preventing the common "yellow shift" seen in Reinhard or basic ACES.
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

    // Gran Turismo (Uchimura 2017) Tone Mapper: a configurable filmic curve stitched from an explicit toe, a straight linear mid-section and an exponential shoulder. Unlike ACES it exposes direct artistic control over contrast, the linear range and black crush, and unlike Reinhard it firmly anchors the maximum brightness so highlights roll off to white instead of washing out.
    // Gran Turismo (Uchimura 2017) Tone Mapper: a configurable filmic curve stitched from an explicit toe, a straight linear mid-section and an exponential shoulder. Unlike ACES it exposes direct artistic control over contrast, the linear range and black crush, and unlike Reinhard it firmly anchors the maximum brightness so highlights roll off to white instead of washing out.
    vec3 gtTonemap(vec3 x) {
//  vec3 gtTonemap(vec3 x) {
        const float P = 1.0;   // maximum brightness the curve maps to
//      const float P = 1.0;   // maximum brightness the curve maps to
        const float a = 1.0;   // contrast of the linear mid-section
//      const float a = 1.0;   // contrast of the linear mid-section
        const float m = 0.22;  // start of the linear section
//      const float m = 0.22;  // start of the linear section
        const float l = 0.4;   // length of the linear section
//      const float l = 0.4;   // length of the linear section
        const float c = 1.33;  // toe curvature (black tightness)
//      const float c = 1.33;  // toe curvature (black tightness)
        const float b = 0.0;   // pedestal (black lift)
//      const float b = 0.0;   // pedestal (black lift)

        // Derive the segment boundaries and the shoulder's exponential coefficients.
//      // Derive the segment boundaries and the shoulder's exponential coefficients.
        float l0 = ((P - m) * l) / a;
//      float l0 = ((P - m) * l) / a;
        float S0 = m + l0;
//      float S0 = m + l0;
        float S1 = m + a * l0;
//      float S1 = m + a * l0;
        float C2 = (a * P) / (P - S1);
//      float C2 = (a * P) / (P - S1);
        float CP = -C2 / P;
//      float CP = -C2 / P;

        // Per-channel masks select which of the three segments each value falls into.
//      // Per-channel masks select which of the three segments each value falls into.
        vec3 w0 = 1.0 - smoothstep(vec3(0.0), vec3(m), x);
//      vec3 w0 = 1.0 - smoothstep(vec3(0.0), vec3(m), x);
        vec3 w2 = step(vec3(m + l0), x);
//      vec3 w2 = step(vec3(m + l0), x);
        vec3 w1 = 1.0 - w0 - w2;
//      vec3 w1 = 1.0 - w0 - w2;

        vec3 T = m * pow(x / m, vec3(c)) + b;              // toe
//      vec3 T = m * pow(x / m, vec3(c)) + b;              // toe
        vec3 S = P - (P - S1) * exp(CP * (x - S0));        // shoulder
//      vec3 S = P - (P - S1) * exp(CP * (x - S0));        // shoulder
        vec3 L = m + a * (x - m);                          // linear mid-section
//      vec3 L = m + a * (x - m);                          // linear mid-section

        return T * w0 + L * w1 + S * w2;
//      return T * w0 + L * w1 + S * w2;
    }
//  }
