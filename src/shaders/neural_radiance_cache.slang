    // ============================================================================
//  // ============================================================================
    // Neural Radiance Cache (NRC) network, shared by every pass that touches it.
//  // Neural Radiance Cache (NRC) network, shared by every pass that touches it.
    //
//  //
    // Multi-Layer Perceptron: 16 inputs -> 16 hidden (x4, ReLU) -> 3 outputs, 1139 parameters. hybrid_shading_cs.slang
//  // Multi-Layer Perceptron: 16 inputs -> 16 hidden (x4, ReLU) -> 3 outputs, 1139 parameters. hybrid_shading_cs.slang
    // and nrc_gather_cs.slang evaluate it (the gather pass to bootstrap its training targets), nrc_train_cs.slang trains
//  // and nrc_gather_cs.slang evaluate it (the gather pass to bootstrap its training targets), nrc_train_cs.slang trains
    // it; all three read the network through this one module, so inference and training cannot drift apart.
//  // it; all three read the network through this one module, so inference and training cannot drift apart.
    //
//  //
    // Inside one network copy every 16-wide weight row, bias block and layer starts on a multiple of four floats
//  // Inside one network copy every 16-wide weight row, bias block and layer starts on a multiple of four floats
    // (hidden layer L: weights at L * 272, biases at L * 272 + 256; output: weights at 1088, biases at 1136), so the
//  // (hidden layer L: weights at L * 272, biases at L * 272 + 256; output: weights at 1088, biases at 1136), so the
    // workgroup keeps the network as float4s: a row is four 128-bit shared-memory loads instead of sixteen scalar ones.
//  // workgroup keeps the network as float4s: a row is four 128-bit shared-memory loads instead of sixteen scalar ones.
    // Each pass stages the copy it needs (training: the active weights; inference: the EMA weights at
//  // Each pass stages the copy it needs (training: the active weights; inference: the EMA weights at
    // NRC_EMA_WEIGHTS_OFFSET) through nrcStoreSharedWeights at the top of its entry point, before any early return.
//  // NRC_EMA_WEIGHTS_OFFSET) through nrcStoreSharedWeights at the top of its entry point, before any early return.
    // ============================================================================
//  // ============================================================================
    module neural_radiance_cache;
//  module neural_radiance_cache;

    import math_constants;
//  import math_constants;

    // NRC input-domain normalisation. The frequency positional encoding below is only well conditioned
//  // NRC input-domain normalisation. The frequency positional encoding below is only well conditioned
    // when the encoded position spans roughly [-0.75, 0.75] (which is what Experiment001's unit-scale
//  // when the encoded position spans roughly [-0.75, 0.75] (which is what Experiment001's unit-scale
    // Cornell box gave it via a bare position * 0.5). These map this scene's world AABB onto that range,
//  // Cornell box gave it via a bare position * 0.5). These map this scene's world AABB onto that range,
    // otherwise sin/cos of the raw position aliases many times over and the MLP cannot fit the scene.
//  // otherwise sin/cos of the raw position aliases many times over and the MLP cannot fit the scene.
    public uniform float3 uNRCPositionOffset;
//  public uniform float3 uNRCPositionOffset;
    public uniform float uNRCPositionScale;
//  public uniform float uNRCPositionScale;

    public static const uint NRC_NETWORK_FLOAT_COUNT = 1139u;
//  public static const uint NRC_NETWORK_FLOAT_COUNT = 1139u;
    public static const uint NRC_NETWORK_FLOAT4_COUNT = 285u; // 1139 floats rounded up to whole float4s; the last w is padding
//  public static const uint NRC_NETWORK_FLOAT4_COUNT = 285u; // 1139 floats rounded up to whole float4s; the last w is padding
    public static const uint NRC_EMA_WEIGHTS_OFFSET = 1139u; // the Exponential Moving Average copy inference reads
//  public static const uint NRC_EMA_WEIGHTS_OFFSET = 1139u; // the Exponential Moving Average copy inference reads
    public static const uint NRC_HIDDEN_LAYER_FLOAT4_STRIDE = 68u; // 16 rows of 4 float4s, then 4 float4s of biases
//  public static const uint NRC_HIDDEN_LAYER_FLOAT4_STRIDE = 68u; // 16 rows of 4 float4s, then 4 float4s of biases
    public static const uint NRC_HIDDEN_BIAS_FLOAT4_OFFSET = 64u;
//  public static const uint NRC_HIDDEN_BIAS_FLOAT4_OFFSET = 64u;
    public static const uint NRC_OUTPUT_WEIGHTS_FLOAT4_OFFSET = 272u;
//  public static const uint NRC_OUTPUT_WEIGHTS_FLOAT4_OFFSET = 272u;
    public static const uint NRC_OUTPUT_BIAS_FLOAT4_OFFSET = 284u;
//  public static const uint NRC_OUTPUT_BIAS_FLOAT4_OFFSET = 284u;

    // Workgroup-resident copy of the network, so every query reads shared memory instead of the SSBO.
//  // Workgroup-resident copy of the network, so every query reads shared memory instead of the SSBO.
    groupshared float4 nrc_shared_weights[NRC_NETWORK_FLOAT4_COUNT];
//  groupshared float4 nrc_shared_weights[NRC_NETWORK_FLOAT4_COUNT];

    public float4 nrcLoadSharedWeights(uint segment) {
//  public float4 nrcLoadSharedWeights(uint segment) {
        return nrc_shared_weights[segment];
//      return nrc_shared_weights[segment];
    }
//  }

    public void nrcStoreSharedWeights(uint segment, float4 segmentWeights) {
//  public void nrcStoreSharedWeights(uint segment, float4 segmentWeights) {
        nrc_shared_weights[segment] = segmentWeights;
//      nrc_shared_weights[segment] = segmentWeights;
    }
//  }

    // 16-dimensional input feature vector (position, normal, and frequency positional encoding)
//  // 16-dimensional input feature vector (position, normal, and frequency positional encoding)
    public void nrcEncodeInput(float3 position, float3 normal, out float inputFeatures[16]) {
//  public void nrcEncodeInput(float3 position, float3 normal, out float inputFeatures[16]) {
        float3 scaledPos = (position - uNRCPositionOffset) * uNRCPositionScale;
//      float3 scaledPos = (position - uNRCPositionOffset) * uNRCPositionScale;
        inputFeatures[0] = scaledPos.x;
//      inputFeatures[0] = scaledPos.x;
        inputFeatures[1] = scaledPos.y;
//      inputFeatures[1] = scaledPos.y;
        inputFeatures[2] = scaledPos.z;
//      inputFeatures[2] = scaledPos.z;
        inputFeatures[3] = normal.x;
//      inputFeatures[3] = normal.x;
        inputFeatures[4] = normal.y;
//      inputFeatures[4] = normal.y;
        inputFeatures[5] = normal.z;
//      inputFeatures[5] = normal.z;
        inputFeatures[6] = sin(scaledPos.x * PI);
//      inputFeatures[6] = sin(scaledPos.x * PI);
        inputFeatures[7] = sin(scaledPos.y * PI);
//      inputFeatures[7] = sin(scaledPos.y * PI);
        inputFeatures[8] = sin(scaledPos.z * PI);
//      inputFeatures[8] = sin(scaledPos.z * PI);
        inputFeatures[9] = cos(scaledPos.x * PI);
//      inputFeatures[9] = cos(scaledPos.x * PI);
        inputFeatures[10] = cos(scaledPos.y * PI);
//      inputFeatures[10] = cos(scaledPos.y * PI);
        inputFeatures[11] = cos(scaledPos.z * PI);
//      inputFeatures[11] = cos(scaledPos.z * PI);
        inputFeatures[12] = sin(scaledPos.x * TWO_PI);
//      inputFeatures[12] = sin(scaledPos.x * TWO_PI);
        inputFeatures[13] = sin(scaledPos.y * TWO_PI);
//      inputFeatures[13] = sin(scaledPos.y * TWO_PI);
        inputFeatures[14] = sin(scaledPos.z * TWO_PI);
//      inputFeatures[14] = sin(scaledPos.z * TWO_PI);
        inputFeatures[15] = 1.0;
//      inputFeatures[15] = 1.0;
    }
//  }

    // One fully connected row: bias + dot(row, layerInput), accumulated in ascending input order.
//  // One fully connected row: bias + dot(row, layerInput), accumulated in ascending input order.
    [ForceInline]
//  [ForceInline]
    public float nrcDenseRow(uint rowFloat4Offset, float bias, float layerInput[16]) {
//  public float nrcDenseRow(uint rowFloat4Offset, float bias, float layerInput[16]) {
        float sum = bias;
//      float sum = bias;
        [ForceUnroll]
//      [ForceUnroll]
        for (uint quad = 0u; quad < 4u; quad++) {
//      for (uint quad = 0u; quad < 4u; quad++) {
            float4 rowWeights = nrc_shared_weights[rowFloat4Offset + quad];
//          float4 rowWeights = nrc_shared_weights[rowFloat4Offset + quad];
            sum += layerInput[quad * 4u + 0u] * rowWeights.x;
//          sum += layerInput[quad * 4u + 0u] * rowWeights.x;
            sum += layerInput[quad * 4u + 1u] * rowWeights.y;
//          sum += layerInput[quad * 4u + 1u] * rowWeights.y;
            sum += layerInput[quad * 4u + 2u] * rowWeights.z;
//          sum += layerInput[quad * 4u + 2u] * rowWeights.z;
            sum += layerInput[quad * 4u + 3u] * rowWeights.w;
//          sum += layerInput[quad * 4u + 3u] * rowWeights.w;
        }
//      }
        return sum;
//      return sum;
    }
//  }

    // Hidden layer forward pass with ReLU.
//  // Hidden layer forward pass with ReLU.
    [ForceInline]
//  [ForceInline]
    public void nrcForwardHiddenLayer(uint layer, float layerInput[16], out float layerOutput[16]) {
//  public void nrcForwardHiddenLayer(uint layer, float layerInput[16], out float layerOutput[16]) {
        uint layerFloat4Offset = layer * NRC_HIDDEN_LAYER_FLOAT4_STRIDE;
//      uint layerFloat4Offset = layer * NRC_HIDDEN_LAYER_FLOAT4_STRIDE;
        [ForceUnroll]
//      [ForceUnroll]
        for (uint unit = 0u; unit < 16u; unit++) {
//      for (uint unit = 0u; unit < 16u; unit++) {
            float bias = nrc_shared_weights[layerFloat4Offset + NRC_HIDDEN_BIAS_FLOAT4_OFFSET + unit / 4u][unit % 4u];
//          float bias = nrc_shared_weights[layerFloat4Offset + NRC_HIDDEN_BIAS_FLOAT4_OFFSET + unit / 4u][unit % 4u];
            layerOutput[unit] = max(0.0, nrcDenseRow(layerFloat4Offset + unit * 4u, bias, layerInput));
//          layerOutput[unit] = max(0.0, nrcDenseRow(layerFloat4Offset + unit * 4u, bias, layerInput));
        }
//      }
    }
//  }

    // Output layer, before any clamping: training differentiates the raw value, inference clamps it.
//  // Output layer, before any clamping: training differentiates the raw value, inference clamps it.
    [ForceInline]
//  [ForceInline]
    public float nrcOutputChannel(uint channel, float hiddenLayer4[16]) {
//  public float nrcOutputChannel(uint channel, float hiddenLayer4[16]) {
        return nrcDenseRow(NRC_OUTPUT_WEIGHTS_FLOAT4_OFFSET + channel * 4u, nrc_shared_weights[NRC_OUTPUT_BIAS_FLOAT4_OFFSET][channel], hiddenLayer4);
//      return nrcDenseRow(NRC_OUTPUT_WEIGHTS_FLOAT4_OFFSET + channel * 4u, nrc_shared_weights[NRC_OUTPUT_BIAS_FLOAT4_OFFSET][channel], hiddenLayer4);
    }
//  }

    // Neural Radiance Cache forward pass: the outgoing radiance the network predicts at a surface point.
//  // Neural Radiance Cache forward pass: the outgoing radiance the network predicts at a surface point.
    public float3 evaluateNeuralRadianceCache(float3 position, float3 normal) {
//  public float3 evaluateNeuralRadianceCache(float3 position, float3 normal) {
        float inputFeatures[16];
//      float inputFeatures[16];
        nrcEncodeInput(position, normal, inputFeatures);
//      nrcEncodeInput(position, normal, inputFeatures);

        float hiddenLayer1[16];
//      float hiddenLayer1[16];
        float hiddenLayer2[16];
//      float hiddenLayer2[16];
        float hiddenLayer3[16];
//      float hiddenLayer3[16];
        float hiddenLayer4[16];
//      float hiddenLayer4[16];
        nrcForwardHiddenLayer(0u, inputFeatures, hiddenLayer1);
//      nrcForwardHiddenLayer(0u, inputFeatures, hiddenLayer1);
        nrcForwardHiddenLayer(1u, hiddenLayer1, hiddenLayer2);
//      nrcForwardHiddenLayer(1u, hiddenLayer1, hiddenLayer2);
        nrcForwardHiddenLayer(2u, hiddenLayer2, hiddenLayer3);
//      nrcForwardHiddenLayer(2u, hiddenLayer2, hiddenLayer3);
        nrcForwardHiddenLayer(3u, hiddenLayer3, hiddenLayer4);
//      nrcForwardHiddenLayer(3u, hiddenLayer3, hiddenLayer4);

        float3 outputRadiance = float3(0.0);
//      float3 outputRadiance = float3(0.0);
        [ForceUnroll]
//      [ForceUnroll]
        for (uint i = 0u; i < 3u; i++) {
//      for (uint i = 0u; i < 3u; i++) {
            outputRadiance[int(i)] = nrcOutputChannel(i, hiddenLayer4);
//          outputRadiance[int(i)] = nrcOutputChannel(i, hiddenLayer4);
        }
//      }

        // Clamp to prevent explosive feedback during early training
//      // Clamp to prevent explosive feedback during early training
        return clamp(outputRadiance, float3(0.0), float3(15.0));
//      return clamp(outputRadiance, float3(0.0), float3(15.0));
    }
//  }
