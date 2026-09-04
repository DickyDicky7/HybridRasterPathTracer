    // ============================================================================
//  // ============================================================================
    // Shared Principled BSDF evaluation.
//  // Shared Principled BSDF evaluation.
    //
//  //
    // Extracted verbatim out of hybrid_shading_cs.glsl so that nrc_gather_cs.glsl trains the Neural
//  // Extracted verbatim out of hybrid_shading_cs.glsl so that nrc_gather_cs.glsl trains the Neural
    // Radiance Cache against exactly the BSDF the shading pass later evaluates the cache with. Keeping
//  // Radiance Cache against exactly the BSDF the shading pass later evaluates the cache with. Keeping
    // one copy is the point: a divergent diffuse lobe here would make the network regress a target the
//  // one copy is the point: a divergent diffuse lobe here would make the network regress a target the
    // renderer never asks for, and the error would be invisible in both files on their own.
//  // renderer never asks for, and the error would be invisible in both files on their own.
    //
//  //
    // Requires the includer to have already defined: PI, F0_DEFAULT, EPSILON_MATH, EPSILON_DOT.
//  // Requires the includer to have already defined: PI, F0_DEFAULT, EPSILON_MATH, EPSILON_DOT.
    // ============================================================================
//  // ============================================================================
    vec3 schlickFresnel(float cosineIncidentAngle, vec3 reflectanceAtNormalIncidence) {
//  vec3 schlickFresnel(float cosineIncidentAngle, vec3 reflectanceAtNormalIncidence) {
        float oneMinusCosine = 1.0 - cosineIncidentAngle;
//      float oneMinusCosine = 1.0 - cosineIncidentAngle;
        float oneMinusCosinePow2 = oneMinusCosine * oneMinusCosine;
//      float oneMinusCosinePow2 = oneMinusCosine * oneMinusCosine;
        float oneMinusCosinePow4 = oneMinusCosinePow2 * oneMinusCosinePow2;
//      float oneMinusCosinePow4 = oneMinusCosinePow2 * oneMinusCosinePow2;
        float oneMinusCosinePow5 = oneMinusCosinePow4 * oneMinusCosine;
//      float oneMinusCosinePow5 = oneMinusCosinePow4 * oneMinusCosine;
        return reflectanceAtNormalIncidence + (1.0 - reflectanceAtNormalIncidence) * oneMinusCosinePow5;
//      return reflectanceAtNormalIncidence + (1.0 - reflectanceAtNormalIncidence) * oneMinusCosinePow5;
    }
//  }

    vec3 evalDisneyDiffuse(vec3 surfaceNormal, vec3 viewDirection, vec3 lightDirection, vec3 albedo, float roughness) {
//  vec3 evalDisneyDiffuse(vec3 surfaceNormal, vec3 viewDirection, vec3 lightDirection, vec3 albedo, float roughness) {
        vec3 halfVectorUnnormalized = viewDirection + lightDirection;
//      vec3 halfVectorUnnormalized = viewDirection + lightDirection;
        vec3 halfVector;
//      vec3 halfVector;
        if (dot(halfVectorUnnormalized, halfVectorUnnormalized) > EPSILON_MATH) {
//      if (dot(halfVectorUnnormalized, halfVectorUnnormalized) > EPSILON_MATH) {
            halfVector = normalize(halfVectorUnnormalized);
//          halfVector = normalize(halfVectorUnnormalized);
        } else {
//      } else {
            halfVector = surfaceNormal;
//          halfVector = surfaceNormal;
        }
//      }
        float normalDotLight = max(dot(surfaceNormal, lightDirection), 0.0);
//      float normalDotLight = max(dot(surfaceNormal, lightDirection), 0.0);
        float normalDotView = max(dot(surfaceNormal, viewDirection), 0.0);
//      float normalDotView = max(dot(surfaceNormal, viewDirection), 0.0);
        float lightDotHalf = max(dot(lightDirection, halfVector), 0.0);
//      float lightDotHalf = max(dot(lightDirection, halfVector), 0.0);

        // Schlick weight for grazing angles
//      // Schlick weight for grazing angles
        float fresnelDiffuse90 = 0.5 + 2.0 * roughness * lightDotHalf * lightDotHalf;
//      float fresnelDiffuse90 = 0.5 + 2.0 * roughness * lightDotHalf * lightDotHalf;

        float oneMinusNormalDotLight = 1.0 - normalDotLight;
//      float oneMinusNormalDotLight = 1.0 - normalDotLight;
        float oneMinusNormalDotLightPow2 = oneMinusNormalDotLight * oneMinusNormalDotLight;
//      float oneMinusNormalDotLightPow2 = oneMinusNormalDotLight * oneMinusNormalDotLight;
        float oneMinusNormalDotLightPow5 = oneMinusNormalDotLightPow2 * oneMinusNormalDotLightPow2 * oneMinusNormalDotLight;
//      float oneMinusNormalDotLightPow5 = oneMinusNormalDotLightPow2 * oneMinusNormalDotLightPow2 * oneMinusNormalDotLight;
        float lightScatterFactor = 1.0 + (fresnelDiffuse90 - 1.0) * oneMinusNormalDotLightPow5;
//      float lightScatterFactor = 1.0 + (fresnelDiffuse90 - 1.0) * oneMinusNormalDotLightPow5;

        float oneMinusNormalDotView = 1.0 - normalDotView;
//      float oneMinusNormalDotView = 1.0 - normalDotView;
        float oneMinusNormalDotViewPow2 = oneMinusNormalDotView * oneMinusNormalDotView;
//      float oneMinusNormalDotViewPow2 = oneMinusNormalDotView * oneMinusNormalDotView;
        float oneMinusNormalDotViewPow5 = oneMinusNormalDotViewPow2 * oneMinusNormalDotViewPow2 * oneMinusNormalDotView;
//      float oneMinusNormalDotViewPow5 = oneMinusNormalDotViewPow2 * oneMinusNormalDotViewPow2 * oneMinusNormalDotView;
        float viewScatterFactor = 1.0 + (fresnelDiffuse90 - 1.0) * oneMinusNormalDotViewPow5;
//      float viewScatterFactor = 1.0 + (fresnelDiffuse90 - 1.0) * oneMinusNormalDotViewPow5;

        return (albedo / PI) * lightScatterFactor * viewScatterFactor;
//      return (albedo / PI) * lightScatterFactor * viewScatterFactor;
    }
//  }

    vec3 evalOrenNayarDiffuse(vec3 surfaceNormal, vec3 viewDirection, vec3 lightDirection, vec3 albedo, float roughness) {
//  vec3 evalOrenNayarDiffuse(vec3 surfaceNormal, vec3 viewDirection, vec3 lightDirection, vec3 albedo, float roughness) {
        float normalDotLight = max(dot(surfaceNormal, lightDirection), 0.0);
//      float normalDotLight = max(dot(surfaceNormal, lightDirection), 0.0);
        float normalDotView = max(dot(surfaceNormal, viewDirection), 0.0);
//      float normalDotView = max(dot(surfaceNormal, viewDirection), 0.0);

        float lightDotView = dot(lightDirection, viewDirection);
//      float lightDotView = dot(lightDirection, viewDirection);
        float geometricNumerator = lightDotView - normalDotLight * normalDotView;
//      float geometricNumerator = lightDotView - normalDotLight * normalDotView;
        float geometricDenominator = mix(1.0, max(normalDotLight, normalDotView), step(0.0, geometricNumerator));
//      float geometricDenominator = mix(1.0, max(normalDotLight, normalDotView), step(0.0, geometricNumerator));

        float sigmaSquared = roughness * roughness;
//      float sigmaSquared = roughness * roughness;
        float orenNayarTermA = 1.0 - 0.5 * (sigmaSquared / (sigmaSquared + 0.33));
//      float orenNayarTermA = 1.0 - 0.5 * (sigmaSquared / (sigmaSquared + 0.33));
        float orenNayarTermB = 0.45 * (sigmaSquared / (sigmaSquared + 0.09));
//      float orenNayarTermB = 0.45 * (sigmaSquared / (sigmaSquared + 0.09));

        return (albedo / PI) * (orenNayarTermA + orenNayarTermB * (geometricNumerator / (geometricDenominator + EPSILON_MATH)));
//      return (albedo / PI) * (orenNayarTermA + orenNayarTermB * (geometricNumerator / (geometricDenominator + EPSILON_MATH)));
    }
//  }

    vec3 evalLambertDiffuse(vec3 albedo) {
//  vec3 evalLambertDiffuse(vec3 albedo) {
        return albedo / PI;
//      return albedo / PI;
    }
//  }

    vec3 evalPrincipledBSDFAndPDF(vec3 incomingDirection, vec3 outgoingDirection, vec3 normal, vec3 albedo, float roughness, float metallic, float transmission, out float outPdf) {
//  vec3 evalPrincipledBSDFAndPDF(vec3 incomingDirection, vec3 outgoingDirection, vec3 normal, vec3 albedo, float roughness, float metallic, float transmission, out float outPdf) {
        outPdf = 0.0;
//      outPdf = 0.0;
        vec3 surfaceNormal = normal;
//      vec3 surfaceNormal = normal;
        vec3 viewDirection = -incomingDirection;
//      vec3 viewDirection = -incomingDirection;
        vec3 lightDirection = outgoingDirection;
//      vec3 lightDirection = outgoingDirection;
        vec3 halfVectorUnnormalized = viewDirection + lightDirection;
//      vec3 halfVectorUnnormalized = viewDirection + lightDirection;
        vec3 halfVector;
//      vec3 halfVector;
        if (dot(halfVectorUnnormalized, halfVectorUnnormalized) > EPSILON_MATH) {
//      if (dot(halfVectorUnnormalized, halfVectorUnnormalized) > EPSILON_MATH) {
            halfVector = normalize(halfVectorUnnormalized);
//          halfVector = normalize(halfVectorUnnormalized);
        } else {
//      } else {
            halfVector = surfaceNormal;
//          halfVector = surfaceNormal;
        }
//      }

        float normalDotLight = max(dot(surfaceNormal, lightDirection), 0.0);
//      float normalDotLight = max(dot(surfaceNormal, lightDirection), 0.0);
        float normalDotView = max(dot(surfaceNormal, viewDirection), 0.0);
//      float normalDotView = max(dot(surfaceNormal, viewDirection), 0.0);

        if (normalDotLight <= 0.0 || normalDotView <= 0.0) return vec3(0.0);
//      if (normalDotLight <= 0.0 || normalDotView <= 0.0) return vec3(0.0);

        vec3 reflectanceAtNormalIncidence = mix(vec3(F0_DEFAULT), albedo, metallic);
//      vec3 reflectanceAtNormalIncidence = mix(vec3(F0_DEFAULT), albedo, metallic);
        // BSDF energy split uses Fresnel at the half-vector angle
//      // BSDF energy split uses Fresnel at the half-vector angle
        float halfDotView = max(dot(halfVector, viewDirection), 0.0);
//      float halfDotView = max(dot(halfVector, viewDirection), 0.0);
        vec3 fresnelReflectance = schlickFresnel(halfDotView, reflectanceAtNormalIncidence);
//      vec3 fresnelReflectance = schlickFresnel(halfDotView, reflectanceAtNormalIncidence);

        // Diffuse
//      // Diffuse
        vec3 specularWeight = fresnelReflectance;
//      vec3 specularWeight = fresnelReflectance;
        vec3 diffuseWeight = (vec3(1.0) - specularWeight) * (1.0 - metallic);
//      vec3 diffuseWeight = (vec3(1.0) - specularWeight) * (1.0 - metallic);

        // --- Try uncommenting one of these ---
//      // --- Try uncommenting one of these ---

        // 1. Original Lambert
//      // 1. Original Lambert
        // vec3 diffuseContribution = diffuseWeight * evalLambertDiffuse(albedo) * (1.0 - transmission);
//      // vec3 diffuseContribution = diffuseWeight * evalLambertDiffuse(albedo) * (1.0 - transmission);

        // 2. Disney Diffuse (Recommended)
//      // 2. Disney Diffuse (Recommended)
        vec3 diffuseContribution = diffuseWeight * evalDisneyDiffuse(surfaceNormal, viewDirection, lightDirection, albedo, roughness) * (1.0 - transmission);
//      vec3 diffuseContribution = diffuseWeight * evalDisneyDiffuse(surfaceNormal, viewDirection, lightDirection, albedo, roughness) * (1.0 - transmission);

        // 3. Oren-Nayar Diffuse
//      // 3. Oren-Nayar Diffuse
        // vec3 diffuseContribution = diffuseWeight * evalOrenNayarDiffuse(surfaceNormal, viewDirection, lightDirection, albedo, roughness) * (1.0 - transmission);
//      // vec3 diffuseContribution = diffuseWeight * evalOrenNayarDiffuse(surfaceNormal, viewDirection, lightDirection, albedo, roughness) * (1.0 - transmission);

        // Specular (GGX distribution D is shared between the BSDF value and the specular PDF)
//      // Specular (GGX distribution D is shared between the BSDF value and the specular PDF)
        float ggxAlpha = roughness * roughness;
//      float ggxAlpha = roughness * roughness;
        float ggxAlphaSquared = ggxAlpha * ggxAlpha;
//      float ggxAlphaSquared = ggxAlpha * ggxAlpha;
        float normalDotHalf = max(dot(surfaceNormal, halfVector), 0.0);
//      float normalDotHalf = max(dot(surfaceNormal, halfVector), 0.0);
        float distributionDenominator = (normalDotHalf * normalDotHalf * (ggxAlphaSquared - 1.0) + 1.0);
//      float distributionDenominator = (normalDotHalf * normalDotHalf * (ggxAlphaSquared - 1.0) + 1.0);
        float normalDistribution = ggxAlphaSquared / (PI * distributionDenominator * distributionDenominator);
//      float normalDistribution = ggxAlphaSquared / (PI * distributionDenominator * distributionDenominator);

        float smithGeometryK = (roughness * roughness) / 2.0;
//      float smithGeometryK = (roughness * roughness) / 2.0;
        // Optimize G term calculation by factoring out normalDotView and normalDotLight
//      // Optimize G term calculation by factoring out normalDotView and normalDotLight
        float geometryViewTerm = normalDotView * (1.0 - smithGeometryK) + smithGeometryK;
//      float geometryViewTerm = normalDotView * (1.0 - smithGeometryK) + smithGeometryK;
        float geometryLightTerm = normalDotLight * (1.0 - smithGeometryK) + smithGeometryK;
//      float geometryLightTerm = normalDotLight * (1.0 - smithGeometryK) + smithGeometryK;

        vec3 specularContribution = (normalDistribution * fresnelReflectance) / (4.0 * geometryViewTerm * geometryLightTerm + EPSILON_MATH);
//      vec3 specularContribution = (normalDistribution * fresnelReflectance) / (4.0 * geometryViewTerm * geometryLightTerm + EPSILON_MATH);

        // PDF: the lobe-selection probability uses Fresnel at the view angle (matching the sampler)
//      // PDF: the lobe-selection probability uses Fresnel at the view angle (matching the sampler)
        float diffusePdf = normalDotLight / PI;
//      float diffusePdf = normalDotLight / PI;
        float specularPdf = (normalDistribution * normalDotHalf) / (4.0 * halfDotView + EPSILON_DOT);
//      float specularPdf = (normalDistribution * normalDotHalf) / (4.0 * halfDotView + EPSILON_DOT);
        vec3 lobeSelectionFresnel = schlickFresnel(normalDotView, reflectanceAtNormalIncidence);
//      vec3 lobeSelectionFresnel = schlickFresnel(normalDotView, reflectanceAtNormalIncidence);
        float averageFresnel = (lobeSelectionFresnel.r + lobeSelectionFresnel.g + lobeSelectionFresnel.b) / 3.0;
//      float averageFresnel = (lobeSelectionFresnel.r + lobeSelectionFresnel.g + lobeSelectionFresnel.b) / 3.0;
        float specularSelectionProbability = max(mix(averageFresnel, 1.0, metallic), 0.15);
//      float specularSelectionProbability = max(mix(averageFresnel, 1.0, metallic), 0.15);
        outPdf = mix(diffusePdf * (1.0 - transmission), specularPdf, specularSelectionProbability);
//      outPdf = mix(diffusePdf * (1.0 - transmission), specularPdf, specularSelectionProbability);

        return diffuseContribution + specularContribution;
//      return diffuseContribution + specularContribution;
    }
//  }
