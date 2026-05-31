    #version 430
//  #version 430

    layout(local_size_x = 16, local_size_y = 16) in;
//  layout(local_size_x = 16, local_size_y = 16) in;

    layout(binding = 0, rgba32f) uniform image2D textureOutput;
//  layout(binding = 0, rgba32f) uniform image2D textureOutput;
    layout(binding = 1) uniform sampler2D textureInput;
//  layout(binding = 1) uniform sampler2D textureInput;

    // FXAA 3.11 - the PC Quality path (Lottes), cranked to a SUPER-ULTRA profile
//  // FXAA 3.11 - the PC Quality path (Lottes), cranked to a SUPER-ULTRA profile
    // (beyond the stock preset 39: 20 search taps and maxed-out edge sensitivity).
//  // (beyond the stock preset 39: 20 search taps and maxed-out edge sensitivity).
    // The input is already tone-mapped and gamma-encoded LDR - the perceptual
//  // The input is already tone-mapped and gamma-encoded LDR - the perceptual
    // space FXAA expects - so luminance is read directly from the encoded signal.
//  // space FXAA expects - so luminance is read directly from the encoded signal.

    // Strength of the sub-pixel aliasing removal (1.0 = full strength, i.e. the softest result).
//  // Strength of the sub-pixel aliasing removal (1.0 = full strength, i.e. the softest result).
    #define SUBPIXEL_ALIASING_REMOVAL_STRENGTH   1.0
//  #define SUBPIXEL_ALIASING_REMOVAL_STRENGTH   1.0
    // Relative edge-contrast sensitivity, pushed about as low as is sane before flat detail starts to soften.
//  // Relative edge-contrast sensitivity, pushed about as low as is sane before flat detail starts to soften.
    #define RELATIVE_EDGE_CONTRAST_THRESHOLD     0.0234
//  #define RELATIVE_EDGE_CONTRAST_THRESHOLD     0.0234
    // Absolute contrast floor, so that even dark regions are still anti-aliased.
//  // Absolute contrast floor, so that even dark regions are still anti-aliased.
    #define MINIMUM_ABSOLUTE_EDGE_CONTRAST       0.0117
//  #define MINIMUM_ABSOLUTE_EDGE_CONTRAST       0.0117
    // Number of edge-end search taps; more taps reach further along grazing-angle edges.
//  // Number of edge-end search taps; more taps reach further along grazing-angle edges.
    #define EDGE_END_SEARCH_STEP_COUNT           20
//  #define EDGE_END_SEARCH_STEP_COUNT           20

    const vec3 PERCEPTUAL_LUMINANCE_COEFFICIENTS = vec3(0.299, 0.587, 0.114);
//  const vec3 PERCEPTUAL_LUMINANCE_COEFFICIENTS = vec3(0.299, 0.587, 0.114);
    const float EDGE_END_SEARCH_STEP_SIZES[20] = float[20](1.0, 1.0, 1.0, 1.0, 1.0, 1.5, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 4.0, 4.0, 4.0, 4.0, 8.0, 8.0, 8.0, 8.0);
//  const float EDGE_END_SEARCH_STEP_SIZES[20] = float[20](1.0, 1.0, 1.0, 1.0, 1.0, 1.5, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 4.0, 4.0, 4.0, 4.0, 8.0, 8.0, 8.0, 8.0);

    // Perceptual luminance from the gamma-encoded RGB; colored emissive edges still register as contrast.
//  // Perceptual luminance from the gamma-encoded RGB; colored emissive edges still register as contrast.
    float computePerceptualLuminance(vec3 color) {
//  float computePerceptualLuminance(vec3 color) {
        return dot(color, PERCEPTUAL_LUMINANCE_COEFFICIENTS);
//      return dot(color, PERCEPTUAL_LUMINANCE_COEFFICIENTS);
    }
//  }

    // Sample the input and convert it to luminance; used for both the gathers and the edge walk.
//  // Sample the input and convert it to luminance; used for both the gathers and the edge walk.
    float samplePerceptualLuminanceAt(vec2 uv) {
//  float samplePerceptualLuminanceAt(vec2 uv) {
        return computePerceptualLuminance(texture(textureInput, uv).rgb);
//      return computePerceptualLuminance(texture(textureInput, uv).rgb);
    }
//  }

    void main() {
//  void main() {
        ivec2 destinationPixelCoord = ivec2(gl_GlobalInvocationID.xy);
//      ivec2 destinationPixelCoord = ivec2(gl_GlobalInvocationID.xy);
        ivec2 outputImageSize = imageSize(textureOutput);
//      ivec2 outputImageSize = imageSize(textureOutput);

        if (destinationPixelCoord.x >= outputImageSize.x || destinationPixelCoord.y >= outputImageSize.y) {
//      if (destinationPixelCoord.x >= outputImageSize.x || destinationPixelCoord.y >= outputImageSize.y) {
            return;
//          return;
        }
//      }

        vec2 texelSizeInUv = 1.0 / vec2(outputImageSize);
//      vec2 texelSizeInUv = 1.0 / vec2(outputImageSize);
        vec2 centerSampleUv = (vec2(destinationPixelCoord) + 0.5) * texelSizeInUv;
//      vec2 centerSampleUv = (vec2(destinationPixelCoord) + 0.5) * texelSizeInUv;

        // --- Gather local contrast across the center pixel and its north/south/east/west neighbors ---
//      // --- Gather local contrast across the center pixel and its north/south/east/west neighbors ---
        vec3 centerPixelColor = texture(textureInput, centerSampleUv).rgb;
//      vec3 centerPixelColor = texture(textureInput, centerSampleUv).rgb;
        float luminanceCenter = computePerceptualLuminance(centerPixelColor);
//      float luminanceCenter = computePerceptualLuminance(centerPixelColor);
        float luminanceNorth = samplePerceptualLuminanceAt(centerSampleUv + vec2( 0.0, -1.0) * texelSizeInUv);
//      float luminanceNorth = samplePerceptualLuminanceAt(centerSampleUv + vec2( 0.0, -1.0) * texelSizeInUv);
        float luminanceSouth = samplePerceptualLuminanceAt(centerSampleUv + vec2( 0.0,  1.0) * texelSizeInUv);
//      float luminanceSouth = samplePerceptualLuminanceAt(centerSampleUv + vec2( 0.0,  1.0) * texelSizeInUv);
        float luminanceEast  = samplePerceptualLuminanceAt(centerSampleUv + vec2( 1.0,  0.0) * texelSizeInUv);
//      float luminanceEast  = samplePerceptualLuminanceAt(centerSampleUv + vec2( 1.0,  0.0) * texelSizeInUv);
        float luminanceWest  = samplePerceptualLuminanceAt(centerSampleUv + vec2(-1.0,  0.0) * texelSizeInUv);
//      float luminanceWest  = samplePerceptualLuminanceAt(centerSampleUv + vec2(-1.0,  0.0) * texelSizeInUv);

        float maximumLuminance = max(luminanceCenter, max(max(luminanceNorth, luminanceSouth), max(luminanceEast, luminanceWest)));
//      float maximumLuminance = max(luminanceCenter, max(max(luminanceNorth, luminanceSouth), max(luminanceEast, luminanceWest)));
        float minimumLuminance = min(luminanceCenter, min(min(luminanceNorth, luminanceSouth), min(luminanceEast, luminanceWest)));
//      float minimumLuminance = min(luminanceCenter, min(min(luminanceNorth, luminanceSouth), min(luminanceEast, luminanceWest)));
        float luminanceContrastRange = maximumLuminance - minimumLuminance;
//      float luminanceContrastRange = maximumLuminance - minimumLuminance;

        // --- Early out: when local contrast is below threshold, pass the pixel through unchanged ---
//      // --- Early out: when local contrast is below threshold, pass the pixel through unchanged ---
        if (luminanceContrastRange < max(MINIMUM_ABSOLUTE_EDGE_CONTRAST, maximumLuminance * RELATIVE_EDGE_CONTRAST_THRESHOLD)) {
//      if (luminanceContrastRange < max(MINIMUM_ABSOLUTE_EDGE_CONTRAST, maximumLuminance * RELATIVE_EDGE_CONTRAST_THRESHOLD)) {
            imageStore(textureOutput, destinationPixelCoord, vec4(centerPixelColor, 1.0));
//          imageStore(textureOutput, destinationPixelCoord, vec4(centerPixelColor, 1.0));
            return;
//          return;
        }
//      }

        // --- Gather the diagonal neighbors, used for edge orientation and the low-pass term ---
//      // --- Gather the diagonal neighbors, used for edge orientation and the low-pass term ---
        float luminanceNorthWest = samplePerceptualLuminanceAt(centerSampleUv + vec2(-1.0, -1.0) * texelSizeInUv);
//      float luminanceNorthWest = samplePerceptualLuminanceAt(centerSampleUv + vec2(-1.0, -1.0) * texelSizeInUv);
        float luminanceNorthEast = samplePerceptualLuminanceAt(centerSampleUv + vec2( 1.0, -1.0) * texelSizeInUv);
//      float luminanceNorthEast = samplePerceptualLuminanceAt(centerSampleUv + vec2( 1.0, -1.0) * texelSizeInUv);
        float luminanceSouthWest = samplePerceptualLuminanceAt(centerSampleUv + vec2(-1.0,  1.0) * texelSizeInUv);
//      float luminanceSouthWest = samplePerceptualLuminanceAt(centerSampleUv + vec2(-1.0,  1.0) * texelSizeInUv);
        float luminanceSouthEast = samplePerceptualLuminanceAt(centerSampleUv + vec2( 1.0,  1.0) * texelSizeInUv);
//      float luminanceSouthEast = samplePerceptualLuminanceAt(centerSampleUv + vec2( 1.0,  1.0) * texelSizeInUv);

        // --- Estimate edge orientation: horizontal vs. vertical contrast magnitude ---
//      // --- Estimate edge orientation: horizontal vs. vertical contrast magnitude ---
        float luminanceNorthSouthSum = luminanceNorth + luminanceSouth;
//      float luminanceNorthSouthSum = luminanceNorth + luminanceSouth;
        float luminanceWestEastSum = luminanceWest + luminanceEast;
//      float luminanceWestEastSum = luminanceWest + luminanceEast;
        float inverseLuminanceContrastRange = 1.0 / luminanceContrastRange;
//      float inverseLuminanceContrastRange = 1.0 / luminanceContrastRange;
        float luminanceCrossNeighborSum = luminanceNorthSouthSum + luminanceWestEastSum;
//      float luminanceCrossNeighborSum = luminanceNorthSouthSum + luminanceWestEastSum;
        float horizontalEdgeTerm1 = (-2.0 * luminanceCenter) + luminanceNorthSouthSum;
//      float horizontalEdgeTerm1 = (-2.0 * luminanceCenter) + luminanceNorthSouthSum;
        float verticalEdgeTerm1 = (-2.0 * luminanceCenter) + luminanceWestEastSum;
//      float verticalEdgeTerm1 = (-2.0 * luminanceCenter) + luminanceWestEastSum;

        float luminanceEastCornerSum = luminanceNorthEast + luminanceSouthEast;
//      float luminanceEastCornerSum = luminanceNorthEast + luminanceSouthEast;
        float luminanceNorthCornerSum = luminanceNorthWest + luminanceNorthEast;
//      float luminanceNorthCornerSum = luminanceNorthWest + luminanceNorthEast;
        float horizontalEdgeTerm2 = (-2.0 * luminanceEast) + luminanceEastCornerSum;
//      float horizontalEdgeTerm2 = (-2.0 * luminanceEast) + luminanceEastCornerSum;
        float verticalEdgeTerm3 = (-2.0 * luminanceNorth) + luminanceNorthCornerSum;
//      float verticalEdgeTerm3 = (-2.0 * luminanceNorth) + luminanceNorthCornerSum;

        float luminanceWestCornerSum = luminanceNorthWest + luminanceSouthWest;
//      float luminanceWestCornerSum = luminanceNorthWest + luminanceSouthWest;
        float luminanceSouthCornerSum = luminanceSouthWest + luminanceSouthEast;
//      float luminanceSouthCornerSum = luminanceSouthWest + luminanceSouthEast;
        float horizontalEdgeTerm4 = (abs(horizontalEdgeTerm1) * 2.0) + abs(horizontalEdgeTerm2);
//      float horizontalEdgeTerm4 = (abs(horizontalEdgeTerm1) * 2.0) + abs(horizontalEdgeTerm2);
        float verticalEdgeTerm4 = (abs(verticalEdgeTerm1) * 2.0) + abs(verticalEdgeTerm3);
//      float verticalEdgeTerm4 = (abs(verticalEdgeTerm1) * 2.0) + abs(verticalEdgeTerm3);
        float horizontalEdgeTerm3 = (-2.0 * luminanceWest) + luminanceWestCornerSum;
//      float horizontalEdgeTerm3 = (-2.0 * luminanceWest) + luminanceWestCornerSum;
        float verticalEdgeTerm2 = (-2.0 * luminanceSouth) + luminanceSouthCornerSum;
//      float verticalEdgeTerm2 = (-2.0 * luminanceSouth) + luminanceSouthCornerSum;
        float horizontalEdgeMagnitude = abs(horizontalEdgeTerm3) + horizontalEdgeTerm4;
//      float horizontalEdgeMagnitude = abs(horizontalEdgeTerm3) + horizontalEdgeTerm4;
        float verticalEdgeMagnitude = abs(verticalEdgeTerm2) + verticalEdgeTerm4;
//      float verticalEdgeMagnitude = abs(verticalEdgeTerm2) + verticalEdgeTerm4;

        float luminanceAllCornersSum = luminanceWestCornerSum + luminanceEastCornerSum;
//      float luminanceAllCornersSum = luminanceWestCornerSum + luminanceEastCornerSum;
        float signedPerpendicularStep = texelSizeInUv.x;
//      float signedPerpendicularStep = texelSizeInUv.x;
        bool edgeIsHorizontal = horizontalEdgeMagnitude >= verticalEdgeMagnitude;
//      bool edgeIsHorizontal = horizontalEdgeMagnitude >= verticalEdgeMagnitude;
        float weightedLowpassLuminanceSum = luminanceCrossNeighborSum * 2.0 + luminanceAllCornersSum;
//      float weightedLowpassLuminanceSum = luminanceCrossNeighborSum * 2.0 + luminanceAllCornersSum;

        // --- Select the two neighbors that straddle the edge (the perpendicular axis) ---
//      // --- Select the two neighbors that straddle the edge (the perpendicular axis) ---
        // A horizontal span has a vertical gradient (north/south); a vertical span has a horizontal one (west/east).
//      // A horizontal span has a vertical gradient (north/south); a vertical span has a horizontal one (west/east).
        float perpendicularNeighborLuminance1;
//      float perpendicularNeighborLuminance1;
        float perpendicularNeighborLuminance2;
//      float perpendicularNeighborLuminance2;
        if (edgeIsHorizontal) {
//      if (edgeIsHorizontal) {
            perpendicularNeighborLuminance1 = luminanceNorth;
//          perpendicularNeighborLuminance1 = luminanceNorth;
            perpendicularNeighborLuminance2 = luminanceSouth;
//          perpendicularNeighborLuminance2 = luminanceSouth;
            signedPerpendicularStep = texelSizeInUv.y;
//          signedPerpendicularStep = texelSizeInUv.y;
        } else {
//      } else {
            perpendicularNeighborLuminance1 = luminanceWest;
//          perpendicularNeighborLuminance1 = luminanceWest;
            perpendicularNeighborLuminance2 = luminanceEast;
//          perpendicularNeighborLuminance2 = luminanceEast;
        }
//      }
        float lowpassLuminanceDelta = (weightedLowpassLuminanceSum * (1.0 / 12.0)) - luminanceCenter;
//      float lowpassLuminanceDelta = (weightedLowpassLuminanceSum * (1.0 / 12.0)) - luminanceCenter;

        // --- Pick the steeper side and build the sub-pixel blend factor ---
//      // --- Pick the steeper side and build the sub-pixel blend factor ---
        float luminanceGradientSide1 = perpendicularNeighborLuminance1 - luminanceCenter;
//      float luminanceGradientSide1 = perpendicularNeighborLuminance1 - luminanceCenter;
        float luminanceGradientSide2 = perpendicularNeighborLuminance2 - luminanceCenter;
//      float luminanceGradientSide2 = perpendicularNeighborLuminance2 - luminanceCenter;
        float pairedLuminanceSum1 = perpendicularNeighborLuminance1 + luminanceCenter;
//      float pairedLuminanceSum1 = perpendicularNeighborLuminance1 + luminanceCenter;
        float pairedLuminanceSum2 = perpendicularNeighborLuminance2 + luminanceCenter;
//      float pairedLuminanceSum2 = perpendicularNeighborLuminance2 + luminanceCenter;
        bool side1HasSteeperGradient = abs(luminanceGradientSide1) >= abs(luminanceGradientSide2);
//      bool side1HasSteeperGradient = abs(luminanceGradientSide1) >= abs(luminanceGradientSide2);
        float steepestGradientMagnitude = max(abs(luminanceGradientSide1), abs(luminanceGradientSide2));
//      float steepestGradientMagnitude = max(abs(luminanceGradientSide1), abs(luminanceGradientSide2));
        if (side1HasSteeperGradient) {
//      if (side1HasSteeperGradient) {
            signedPerpendicularStep = -signedPerpendicularStep;
//          signedPerpendicularStep = -signedPerpendicularStep;
        }
//      }
        float subpixelBlendAmountClamped = clamp(abs(lowpassLuminanceDelta) * inverseLuminanceContrastRange, 0.0, 1.0);
//      float subpixelBlendAmountClamped = clamp(abs(lowpassLuminanceDelta) * inverseLuminanceContrastRange, 0.0, 1.0);

        // --- Step half a texel onto the edge and seed the two search anchors ---
//      // --- Step half a texel onto the edge and seed the two search anchors ---
        vec2 edgeMidpointUv = centerSampleUv;
//      vec2 edgeMidpointUv = centerSampleUv;
        vec2 alongEdgeStepUv;
//      vec2 alongEdgeStepUv;
        if (edgeIsHorizontal) {
//      if (edgeIsHorizontal) {
            alongEdgeStepUv.x = texelSizeInUv.x;
//          alongEdgeStepUv.x = texelSizeInUv.x;
            alongEdgeStepUv.y = 0.0;
//          alongEdgeStepUv.y = 0.0;
            edgeMidpointUv.y += signedPerpendicularStep * 0.5;
//          edgeMidpointUv.y += signedPerpendicularStep * 0.5;
        } else {
//      } else {
            alongEdgeStepUv.x = 0.0;
//          alongEdgeStepUv.x = 0.0;
            alongEdgeStepUv.y = texelSizeInUv.y;
//          alongEdgeStepUv.y = texelSizeInUv.y;
            edgeMidpointUv.x += signedPerpendicularStep * 0.5;
//          edgeMidpointUv.x += signedPerpendicularStep * 0.5;
        }
//      }

        vec2 negativeSearchUv = edgeMidpointUv - alongEdgeStepUv * EDGE_END_SEARCH_STEP_SIZES[0];
//      vec2 negativeSearchUv = edgeMidpointUv - alongEdgeStepUv * EDGE_END_SEARCH_STEP_SIZES[0];
        vec2 positiveSearchUv = edgeMidpointUv + alongEdgeStepUv * EDGE_END_SEARCH_STEP_SIZES[0];
//      vec2 positiveSearchUv = edgeMidpointUv + alongEdgeStepUv * EDGE_END_SEARCH_STEP_SIZES[0];
        float subpixelBlendCubicSlope = (-2.0 * subpixelBlendAmountClamped) + 3.0;
//      float subpixelBlendCubicSlope = (-2.0 * subpixelBlendAmountClamped) + 3.0;
        float negativeSearchLuminance = samplePerceptualLuminanceAt(negativeSearchUv);
//      float negativeSearchLuminance = samplePerceptualLuminanceAt(negativeSearchUv);
        float subpixelBlendAmountSquared = subpixelBlendAmountClamped * subpixelBlendAmountClamped;
//      float subpixelBlendAmountSquared = subpixelBlendAmountClamped * subpixelBlendAmountClamped;
        float positiveSearchLuminance = samplePerceptualLuminanceAt(positiveSearchUv);
//      float positiveSearchLuminance = samplePerceptualLuminanceAt(positiveSearchUv);

        if (!side1HasSteeperGradient) {
//      if (!side1HasSteeperGradient) {
            pairedLuminanceSum1 = pairedLuminanceSum2;
//          pairedLuminanceSum1 = pairedLuminanceSum2;
        }
//      }
        float edgeEndGradientThreshold = steepestGradientMagnitude * (1.0 / 4.0);
//      float edgeEndGradientThreshold = steepestGradientMagnitude * (1.0 / 4.0);
        float centerLuminanceMinusBaseline = luminanceCenter - pairedLuminanceSum1 * 0.5;
//      float centerLuminanceMinusBaseline = luminanceCenter - pairedLuminanceSum1 * 0.5;
        float subpixelBlendSmoothstep = subpixelBlendCubicSlope * subpixelBlendAmountSquared;
//      float subpixelBlendSmoothstep = subpixelBlendCubicSlope * subpixelBlendAmountSquared;
        bool centerIsBelowPairAverage = centerLuminanceMinusBaseline < 0.0;
//      bool centerIsBelowPairAverage = centerLuminanceMinusBaseline < 0.0;

        negativeSearchLuminance -= pairedLuminanceSum1 * 0.5;
//      negativeSearchLuminance -= pairedLuminanceSum1 * 0.5;
        positiveSearchLuminance -= pairedLuminanceSum1 * 0.5;
//      positiveSearchLuminance -= pairedLuminanceSum1 * 0.5;
        bool negativeSideReachedEnd = abs(negativeSearchLuminance) >= edgeEndGradientThreshold;
//      bool negativeSideReachedEnd = abs(negativeSearchLuminance) >= edgeEndGradientThreshold;
        bool positiveSideReachedEnd = abs(positiveSearchLuminance) >= edgeEndGradientThreshold;
//      bool positiveSideReachedEnd = abs(positiveSearchLuminance) >= edgeEndGradientThreshold;
        if (!negativeSideReachedEnd) {
//      if (!negativeSideReachedEnd) {
            negativeSearchUv -= alongEdgeStepUv * EDGE_END_SEARCH_STEP_SIZES[1];
//          negativeSearchUv -= alongEdgeStepUv * EDGE_END_SEARCH_STEP_SIZES[1];
        }
//      }
        if (!positiveSideReachedEnd) {
//      if (!positiveSideReachedEnd) {
            positiveSearchUv += alongEdgeStepUv * EDGE_END_SEARCH_STEP_SIZES[1];
//          positiveSearchUv += alongEdgeStepUv * EDGE_END_SEARCH_STEP_SIZES[1];
        }
//      }
        bool searchStillActive = (!negativeSideReachedEnd) || (!positiveSideReachedEnd);
//      bool searchStillActive = (!negativeSideReachedEnd) || (!positiveSideReachedEnd);

        // --- Edge-end search: walk both directions until the edge fades out ---
//      // --- Edge-end search: walk both directions until the edge fades out ---
        if (searchStillActive) {
//      if (searchStillActive) {
            for (int searchStepIndex = 2; searchStepIndex < EDGE_END_SEARCH_STEP_COUNT; searchStepIndex++) {
//          for (int searchStepIndex = 2; searchStepIndex < EDGE_END_SEARCH_STEP_COUNT; searchStepIndex++) {
                if (!negativeSideReachedEnd) {
//              if (!negativeSideReachedEnd) {
                    negativeSearchLuminance = samplePerceptualLuminanceAt(negativeSearchUv) - pairedLuminanceSum1 * 0.5;
//                  negativeSearchLuminance = samplePerceptualLuminanceAt(negativeSearchUv) - pairedLuminanceSum1 * 0.5;
                }
//              }
                if (!positiveSideReachedEnd) {
//              if (!positiveSideReachedEnd) {
                    positiveSearchLuminance = samplePerceptualLuminanceAt(positiveSearchUv) - pairedLuminanceSum1 * 0.5;
//                  positiveSearchLuminance = samplePerceptualLuminanceAt(positiveSearchUv) - pairedLuminanceSum1 * 0.5;
                }
//              }
                negativeSideReachedEnd = abs(negativeSearchLuminance) >= edgeEndGradientThreshold;
//              negativeSideReachedEnd = abs(negativeSearchLuminance) >= edgeEndGradientThreshold;
                positiveSideReachedEnd = abs(positiveSearchLuminance) >= edgeEndGradientThreshold;
//              positiveSideReachedEnd = abs(positiveSearchLuminance) >= edgeEndGradientThreshold;
                if (!negativeSideReachedEnd) {
//              if (!negativeSideReachedEnd) {
                    negativeSearchUv -= alongEdgeStepUv * EDGE_END_SEARCH_STEP_SIZES[searchStepIndex];
//                  negativeSearchUv -= alongEdgeStepUv * EDGE_END_SEARCH_STEP_SIZES[searchStepIndex];
                }
//              }
                if (!positiveSideReachedEnd) {
//              if (!positiveSideReachedEnd) {
                    positiveSearchUv += alongEdgeStepUv * EDGE_END_SEARCH_STEP_SIZES[searchStepIndex];
//                  positiveSearchUv += alongEdgeStepUv * EDGE_END_SEARCH_STEP_SIZES[searchStepIndex];
                }
//              }
                searchStillActive = (!negativeSideReachedEnd) || (!positiveSideReachedEnd);
//              searchStillActive = (!negativeSideReachedEnd) || (!positiveSideReachedEnd);
                if (!searchStillActive) {
//              if (!searchStillActive) {
                    break;
//                  break;
                }
//              }
            }
//          }
        }
//      }

        // --- Measure the distance to each edge end; the nearer end drives the blend offset ---
//      // --- Measure the distance to each edge end; the nearer end drives the blend offset ---
        float distanceToNegativeEnd;
//      float distanceToNegativeEnd;
        float distanceToPositiveEnd;
//      float distanceToPositiveEnd;
        if (edgeIsHorizontal) {
//      if (edgeIsHorizontal) {
            distanceToNegativeEnd = centerSampleUv.x - negativeSearchUv.x;
//          distanceToNegativeEnd = centerSampleUv.x - negativeSearchUv.x;
            distanceToPositiveEnd = positiveSearchUv.x - centerSampleUv.x;
//          distanceToPositiveEnd = positiveSearchUv.x - centerSampleUv.x;
        } else {
//      } else {
            distanceToNegativeEnd = centerSampleUv.y - negativeSearchUv.y;
//          distanceToNegativeEnd = centerSampleUv.y - negativeSearchUv.y;
            distanceToPositiveEnd = positiveSearchUv.y - centerSampleUv.y;
//          distanceToPositiveEnd = positiveSearchUv.y - centerSampleUv.y;
        }
//      }
        bool negativeEndSpanValid = (negativeSearchLuminance < 0.0) != centerIsBelowPairAverage;
//      bool negativeEndSpanValid = (negativeSearchLuminance < 0.0) != centerIsBelowPairAverage;
        float totalEdgeSpanLength = (distanceToPositiveEnd + distanceToNegativeEnd);
//      float totalEdgeSpanLength = (distanceToPositiveEnd + distanceToNegativeEnd);
        bool positiveEndSpanValid = (positiveSearchLuminance < 0.0) != centerIsBelowPairAverage;
//      bool positiveEndSpanValid = (positiveSearchLuminance < 0.0) != centerIsBelowPairAverage;
        float inverseEdgeSpanLength = 1.0 / totalEdgeSpanLength;
//      float inverseEdgeSpanLength = 1.0 / totalEdgeSpanLength;

        bool nearestEndIsNegative = distanceToNegativeEnd < distanceToPositiveEnd;
//      bool nearestEndIsNegative = distanceToNegativeEnd < distanceToPositiveEnd;
        float distanceToNearestEnd = min(distanceToNegativeEnd, distanceToPositiveEnd);
//      float distanceToNearestEnd = min(distanceToNegativeEnd, distanceToPositiveEnd);
        bool nearestEndSpanValid;
//      bool nearestEndSpanValid;
        if (nearestEndIsNegative) {
//      if (nearestEndIsNegative) {
            nearestEndSpanValid = negativeEndSpanValid;
//          nearestEndSpanValid = negativeEndSpanValid;
        } else {
//      } else {
            nearestEndSpanValid = positiveEndSpanValid;
//          nearestEndSpanValid = positiveEndSpanValid;
        }
//      }
        float subpixelBlendSmoothstepSquared = subpixelBlendSmoothstep * subpixelBlendSmoothstep;
//      float subpixelBlendSmoothstepSquared = subpixelBlendSmoothstep * subpixelBlendSmoothstep;
        float edgeEndBlendOffset = (distanceToNearestEnd * (-inverseEdgeSpanLength)) + 0.5;
//      float edgeEndBlendOffset = (distanceToNearestEnd * (-inverseEdgeSpanLength)) + 0.5;
        float finalSubpixelBlendAmount = subpixelBlendSmoothstepSquared * SUBPIXEL_ALIASING_REMOVAL_STRENGTH;
//      float finalSubpixelBlendAmount = subpixelBlendSmoothstepSquared * SUBPIXEL_ALIASING_REMOVAL_STRENGTH;

        // --- Final offset = max(edge-end blend, sub-pixel blend), applied perpendicular to the edge ---
//      // --- Final offset = max(edge-end blend, sub-pixel blend), applied perpendicular to the edge ---
        float validatedEdgeEndBlendOffset;
//      float validatedEdgeEndBlendOffset;
        if (nearestEndSpanValid) {
//      if (nearestEndSpanValid) {
            validatedEdgeEndBlendOffset = edgeEndBlendOffset;
//          validatedEdgeEndBlendOffset = edgeEndBlendOffset;
        } else {
//      } else {
            validatedEdgeEndBlendOffset = 0.0;
//          validatedEdgeEndBlendOffset = 0.0;
        }
//      }
        float finalPerpendicularBlendOffset = max(validatedEdgeEndBlendOffset, finalSubpixelBlendAmount);
//      float finalPerpendicularBlendOffset = max(validatedEdgeEndBlendOffset, finalSubpixelBlendAmount);
        if (edgeIsHorizontal) {
//      if (edgeIsHorizontal) {
            centerSampleUv.y += finalPerpendicularBlendOffset * signedPerpendicularStep;
//          centerSampleUv.y += finalPerpendicularBlendOffset * signedPerpendicularStep;
        } else {
//      } else {
            centerSampleUv.x += finalPerpendicularBlendOffset * signedPerpendicularStep;
//          centerSampleUv.x += finalPerpendicularBlendOffset * signedPerpendicularStep;
        }
//      }

        // --- Take a single resolved tap through the bilinear sampler ---
//      // --- Take a single resolved tap through the bilinear sampler ---
        vec3 antialiasedColor = texture(textureInput, centerSampleUv).rgb;
//      vec3 antialiasedColor = texture(textureInput, centerSampleUv).rgb;
        imageStore(textureOutput, destinationPixelCoord, vec4(antialiasedColor, 1.0));
//      imageStore(textureOutput, destinationPixelCoord, vec4(antialiasedColor, 1.0));
    }
//  }
