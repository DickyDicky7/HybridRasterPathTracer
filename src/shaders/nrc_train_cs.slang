    // Memory Layout for the NRC weights buffer (4556 floats total):
//  // Memory Layout for the NRC weights buffer (4556 floats total):
    // [0, 1138]:       Active weights being trained
//  // [0, 1138]:       Active weights being trained
    // [1139, 2277]:    Exponential Moving Average (EMA) weights for inference
//  // [1139, 2277]:    Exponential Moving Average (EMA) weights for inference
    // [2278, 3416]:    Adam Optimizer Momentum (m)
//  // [2278, 3416]:    Adam Optimizer Momentum (m)
    // [3417, 4555]:    Adam Optimizer Velocity (v)
//  // [3417, 4555]:    Adam Optimizer Velocity (v)
    //
//  //
    // The network layout, its float4 workgroup copy, the input encoding and the forward layers live in
//  // The network layout, its float4 workgroup copy, the input encoding and the forward layers live in
    // neural_radiance_cache.slang, shared with the passes that evaluate the network.
//  // neural_radiance_cache.slang, shared with the passes that evaluate the network.
    import neural_radiance_cache;
//  import neural_radiance_cache;

    RWStructuredBuffer<float> neuralNetworkWeights;
//  RWStructuredBuffer<float> neuralNetworkWeights;

    StructuredBuffer<float> nrcTrainingRecords;
//  StructuredBuffer<float> nrcTrainingRecords;

    RWStructuredBuffer<uint> trainingRecordCounter;
//  RWStructuredBuffer<uint> trainingRecordCounter;

    uniform uint uTrainingStep;
//  uniform uint uTrainingStep;
    // The NRC input-domain normalisation (uNRCPositionOffset / uNRCPositionScale) is a parameter of neural_radiance_cache.
//  // The NRC input-domain normalisation (uNRCPositionOffset / uNRCPositionScale) is a parameter of neural_radiance_cache.
    static const uint THREAD_COUNT = 256u; // one record per thread per step, so also the mini-batch size
//  static const uint THREAD_COUNT = 256u; // one record per thread per step, so also the mini-batch size
    static const uint TRAINING_STEPS = 32u;
//  static const uint TRAINING_STEPS = 32u;
    static const uint MAX_TRAINING_RECORDS = 8192u;
//  static const uint MAX_TRAINING_RECORDS = 8192u;
    // Records staged to shared memory per round of the gradient reduction. 16 keeps the workgroup inside WebGPU's
//  // Records staged to shared memory per round of the gradient reduction. 16 keeps the workgroup inside WebGPU's
    // default 16 KiB of workgroup storage; the reduction adds each chunk's partial sum in turn, exactly as before.
//  // default 16 KiB of workgroup storage; the reduction adds each chunk's partial sum in turn, exactly as before.
    static const uint RECORDS_PER_CHUNK = 16u;
//  static const uint RECORDS_PER_CHUNK = 16u;
    static const uint CHUNK_COUNT = THREAD_COUNT / RECORDS_PER_CHUNK;
//  static const uint CHUNK_COUNT = THREAD_COUNT / RECORDS_PER_CHUNK;
    // Weight segments (float4s of the four hidden weight matrices) reduced one per thread, plus the 29 bias and
//  // Weight segments (float4s of the four hidden weight matrices) reduced one per thread, plus the 29 bias and
    // output-layer segments that threads 0..28 reduce as a second job.
//  // output-layer segments that threads 0..28 reduce as a second job.
    static const uint EXTRA_SEGMENT_COUNT = 29u;
//  static const uint EXTRA_SEGMENT_COUNT = 29u;

    // Staged chunk of records. Activations as float4 quads: [layer input (0 = features, 1..4 = hidden 1..4)][record][quad].
//  // Staged chunk of records. Activations as float4 quads: [layer input (0 = features, 1..4 = hidden 1..4)][record][quad].
    groupshared float4 batch_activations[5u * RECORDS_PER_CHUNK * 4u];
//  groupshared float4 batch_activations[5u * RECORDS_PER_CHUNK * 4u];
    // Back-propagated gradients as scalars: hidden layer l at [l * 256 + record * 16 + unit], output at [1024 + record * 4 + channel].
//  // Back-propagated gradients as scalars: hidden layer l at [l * 256 + record * 16 + unit], output at [1024 + record * 4 + channel].
    groupshared float batch_gradients[4u * RECORDS_PER_CHUNK * 16u + RECORDS_PER_CHUNK * 4u];
//  groupshared float batch_gradients[4u * RECORDS_PER_CHUNK * 16u + RECORDS_PER_CHUNK * 4u];
    groupshared uint shared_max_records;
//  groupshared uint shared_max_records;

    static const uint OUTPUT_GRADIENT_OFFSET = 4u * RECORDS_PER_CHUNK * 16u;
//  static const uint OUTPUT_GRADIENT_OFFSET = 4u * RECORDS_PER_CHUNK * 16u;

    // Back-propagates through ROWS rows of weights: gradientInput[i] = sum_j gradientOutput[j] * W[j][i], walked
//  // Back-propagates through ROWS rows of weights: gradientInput[i] = sum_j gradientOutput[j] * W[j][i], walked
    // row by row (ascending j, the order of the scalar column walk it replaces) so each row is read as four float4s,
//  // row by row (ascending j, the order of the scalar column walk it replaces) so each row is read as four float4s,
    // then masked by the ReLU derivative of the activation the row consumed.
//  // then masked by the ReLU derivative of the activation the row consumed.
    [ForceInline]
//  [ForceInline]
    void backwardLayer<let ROWS : uint>(uint weightsFloat4Offset, float gradientOutput[ROWS], float activation[16], out float gradientInput[16]) {
//  void backwardLayer<let ROWS : uint>(uint weightsFloat4Offset, float gradientOutput[ROWS], float activation[16], out float gradientInput[16]) {
        float sums[16];
//      float sums[16];
        [ForceUnroll]
//      [ForceUnroll]
        for (uint unit = 0u; unit < 16u; unit++) {
//      for (uint unit = 0u; unit < 16u; unit++) {
            sums[unit] = 0.0;
//          sums[unit] = 0.0;
        }
//      }
        [ForceUnroll]
//      [ForceUnroll]
        for (uint row = 0u; row < ROWS; row++) {
//      for (uint row = 0u; row < ROWS; row++) {
            [ForceUnroll]
//          [ForceUnroll]
            for (uint quad = 0u; quad < 4u; quad++) {
//          for (uint quad = 0u; quad < 4u; quad++) {
                float4 rowWeights = nrcLoadSharedWeights(weightsFloat4Offset + row * 4u + quad);
//              float4 rowWeights = nrcLoadSharedWeights(weightsFloat4Offset + row * 4u + quad);
                sums[quad * 4u + 0u] += gradientOutput[row] * rowWeights.x;
//              sums[quad * 4u + 0u] += gradientOutput[row] * rowWeights.x;
                sums[quad * 4u + 1u] += gradientOutput[row] * rowWeights.y;
//              sums[quad * 4u + 1u] += gradientOutput[row] * rowWeights.y;
                sums[quad * 4u + 2u] += gradientOutput[row] * rowWeights.z;
//              sums[quad * 4u + 2u] += gradientOutput[row] * rowWeights.z;
                sums[quad * 4u + 3u] += gradientOutput[row] * rowWeights.w;
//              sums[quad * 4u + 3u] += gradientOutput[row] * rowWeights.w;
            }
//          }
        }
//      }
        [ForceUnroll]
//      [ForceUnroll]
        for (uint unit = 0u; unit < 16u; unit++) {
//      for (uint unit = 0u; unit < 16u; unit++) {
            gradientInput[unit] = (activation[unit] > 0.0) ? sums[unit] : 0.0;
//          gradientInput[unit] = (activation[unit] > 0.0) ? sums[unit] : 0.0;
        }
//      }
    }
//  }

    // Copies a 16-wide activation vector into the staged chunk as four float4 quads.
//  // Copies a 16-wide activation vector into the staged chunk as four float4 quads.
    [ForceInline]
//  [ForceInline]
    void stageActivations(uint layerInput, uint record, float activation[16]) {
//  void stageActivations(uint layerInput, uint record, float activation[16]) {
        [ForceUnroll]
//      [ForceUnroll]
        for (uint quad = 0u; quad < 4u; quad++) {
//      for (uint quad = 0u; quad < 4u; quad++) {
            batch_activations[(layerInput * RECORDS_PER_CHUNK + record) * 4u + quad] = float4(activation[quad * 4u + 0u], activation[quad * 4u + 1u], activation[quad * 4u + 2u], activation[quad * 4u + 3u]);
//          batch_activations[(layerInput * RECORDS_PER_CHUNK + record) * 4u + quad] = float4(activation[quad * 4u + 0u], activation[quad * 4u + 1u], activation[quad * 4u + 2u], activation[quad * 4u + 3u]);
        }
//      }
    }
//  }

    [ForceInline]
//  [ForceInline]
    void stageHiddenGradients(uint layer, uint record, float gradient[16]) {
//  void stageHiddenGradients(uint layer, uint record, float gradient[16]) {
        [ForceUnroll]
//      [ForceUnroll]
        for (uint unit = 0u; unit < 16u; unit++) {
//      for (uint unit = 0u; unit < 16u; unit++) {
            batch_gradients[(layer * RECORDS_PER_CHUNK + record) * 16u + unit] = gradient[unit];
//          batch_gradients[(layer * RECORDS_PER_CHUNK + record) * 16u + unit] = gradient[unit];
        }
//      }
    }
//  }

    // Float4 segment of the network a thread owns in the gradient reduction and the Adam step. Threads 0..255 each
//  // Float4 segment of the network a thread owns in the gradient reduction and the Adam step. Threads 0..255 each
    // own one segment of the four hidden weight matrices (warps 0-1 layer 1, 2-3 layer 2, ...), so the reduction
//  // own one segment of the four hidden weight matrices (warps 0-1 layer 1, 2-3 layer 2, ...), so the reduction
    // branch is uniform across every warp; threads 0..28 additionally own the hidden biases (0..15), the output
//  // branch is uniform across every warp; threads 0..28 additionally own the hidden biases (0..15), the output
    // weights (16..27) and the output biases (28).
//  // weights (16..27) and the output biases (28).
    uint mainSegment(uint threadIndex) {
//  uint mainSegment(uint threadIndex) {
        return (threadIndex / 64u) * NRC_HIDDEN_LAYER_FLOAT4_STRIDE + threadIndex % 64u;
//      return (threadIndex / 64u) * NRC_HIDDEN_LAYER_FLOAT4_STRIDE + threadIndex % 64u;
    }
//  }

    uint extraSegment(uint threadIndex) {
//  uint extraSegment(uint threadIndex) {
        if (threadIndex < 16u) {
//      if (threadIndex < 16u) {
            return (threadIndex / 4u) * NRC_HIDDEN_LAYER_FLOAT4_STRIDE + NRC_HIDDEN_BIAS_FLOAT4_OFFSET + threadIndex % 4u;
//          return (threadIndex / 4u) * NRC_HIDDEN_LAYER_FLOAT4_STRIDE + NRC_HIDDEN_BIAS_FLOAT4_OFFSET + threadIndex % 4u;
        }
//      }
        return NRC_OUTPUT_WEIGHTS_FLOAT4_OFFSET + (threadIndex - 16u);
//      return NRC_OUTPUT_WEIGHTS_FLOAT4_OFFSET + (threadIndex - 16u);
    }
//  }

    // Gradient of a hidden weight segment summed over the staged chunk, ascending record order.
//  // Gradient of a hidden weight segment summed over the staged chunk, ascending record order.
    float4 reduceMainSegment(uint threadIndex) {
//  float4 reduceMainSegment(uint threadIndex) {
        uint layer = threadIndex / 64u;
//      uint layer = threadIndex / 64u;
        uint row = (threadIndex % 64u) / 4u;
//      uint row = (threadIndex % 64u) / 4u;
        uint quad = threadIndex % 4u;
//      uint quad = threadIndex % 4u;
        float4 partialGradient = float4(0.0);
//      float4 partialGradient = float4(0.0);
        [ForceUnroll]
//      [ForceUnroll]
        for (uint record = 0u; record < RECORDS_PER_CHUNK; record++) {
//      for (uint record = 0u; record < RECORDS_PER_CHUNK; record++) {
            float rowGradient = batch_gradients[(layer * RECORDS_PER_CHUNK + record) * 16u + row];
//          float rowGradient = batch_gradients[(layer * RECORDS_PER_CHUNK + record) * 16u + row];
            partialGradient += rowGradient * batch_activations[(layer * RECORDS_PER_CHUNK + record) * 4u + quad];
//          partialGradient += rowGradient * batch_activations[(layer * RECORDS_PER_CHUNK + record) * 4u + quad];
        }
//      }
        return partialGradient;
//      return partialGradient;
    }
//  }

    // Gradient of a bias or output-layer segment summed over the staged chunk, ascending record order.
//  // Gradient of a bias or output-layer segment summed over the staged chunk, ascending record order.
    float4 reduceExtraSegment(uint threadIndex) {
//  float4 reduceExtraSegment(uint threadIndex) {
        float4 partialGradient = float4(0.0);
//      float4 partialGradient = float4(0.0);
        if (threadIndex < 16u) {
//      if (threadIndex < 16u) {
            uint layer = threadIndex / 4u;
//          uint layer = threadIndex / 4u;
            uint quad = threadIndex % 4u;
//          uint quad = threadIndex % 4u;
            [ForceUnroll]
//          [ForceUnroll]
            for (uint record = 0u; record < RECORDS_PER_CHUNK; record++) {
//          for (uint record = 0u; record < RECORDS_PER_CHUNK; record++) {
                uint gradientOffset = (layer * RECORDS_PER_CHUNK + record) * 16u + quad * 4u;
//              uint gradientOffset = (layer * RECORDS_PER_CHUNK + record) * 16u + quad * 4u;
                partialGradient += float4(batch_gradients[gradientOffset + 0u], batch_gradients[gradientOffset + 1u], batch_gradients[gradientOffset + 2u], batch_gradients[gradientOffset + 3u]);
//              partialGradient += float4(batch_gradients[gradientOffset + 0u], batch_gradients[gradientOffset + 1u], batch_gradients[gradientOffset + 2u], batch_gradients[gradientOffset + 3u]);
            }
//          }
        } else if (threadIndex < 28u) {
//      } else if (threadIndex < 28u) {
            uint row = (threadIndex - 16u) / 4u;
//          uint row = (threadIndex - 16u) / 4u;
            uint quad = threadIndex % 4u;
//          uint quad = threadIndex % 4u;
            [ForceUnroll]
//          [ForceUnroll]
            for (uint record = 0u; record < RECORDS_PER_CHUNK; record++) {
//          for (uint record = 0u; record < RECORDS_PER_CHUNK; record++) {
                float rowGradient = batch_gradients[OUTPUT_GRADIENT_OFFSET + record * 4u + row];
//              float rowGradient = batch_gradients[OUTPUT_GRADIENT_OFFSET + record * 4u + row];
                partialGradient += rowGradient * batch_activations[(4u * RECORDS_PER_CHUNK + record) * 4u + quad];
//              partialGradient += rowGradient * batch_activations[(4u * RECORDS_PER_CHUNK + record) * 4u + quad];
            }
//          }
        } else {
//      } else {
            [ForceUnroll]
//          [ForceUnroll]
            for (uint record = 0u; record < RECORDS_PER_CHUNK; record++) {
//          for (uint record = 0u; record < RECORDS_PER_CHUNK; record++) {
                uint gradientOffset = OUTPUT_GRADIENT_OFFSET + record * 4u;
//              uint gradientOffset = OUTPUT_GRADIENT_OFFSET + record * 4u;
                partialGradient += float4(batch_gradients[gradientOffset + 0u], batch_gradients[gradientOffset + 1u], batch_gradients[gradientOffset + 2u], 0.0);
//              partialGradient += float4(batch_gradients[gradientOffset + 0u], batch_gradients[gradientOffset + 1u], batch_gradients[gradientOffset + 2u], 0.0);
            }
//          }
        }
//      }
        return partialGradient;
//      return partialGradient;
    }
//  }

    // Adam step for one float4 segment of the network; the padding lane past the last parameter is left alone.
//  // Adam step for one float4 segment of the network; the padding lane past the last parameter is left alone.
    void applyAdamSegment(uint segment, float4 accumulatedGradient, float biasCorrection1, float biasCorrection2) {
//  void applyAdamSegment(uint segment, float4 accumulatedGradient, float biasCorrection1, float biasCorrection2) {
        float4 segmentWeights = nrcLoadSharedWeights(segment);
//      float4 segmentWeights = nrcLoadSharedWeights(segment);
        [ForceUnroll]
//      [ForceUnroll]
        for (uint lane = 0u; lane < 4u; lane++) {
//      for (uint lane = 0u; lane < 4u; lane++) {
            uint parameterIndex = segment * 4u + lane;
//          uint parameterIndex = segment * 4u + lane;
            if (parameterIndex < NRC_NETWORK_FLOAT_COUNT) {
//          if (parameterIndex < NRC_NETWORK_FLOAT_COUNT) {
                float averagedGradient = accumulatedGradient[lane] / float(THREAD_COUNT);
//              float averagedGradient = accumulatedGradient[lane] / float(THREAD_COUNT);
                averagedGradient = clamp(averagedGradient, -1.0, 1.0);
//              averagedGradient = clamp(averagedGradient, -1.0, 1.0);

                uint momentumIndex = 2278u + parameterIndex;
//              uint momentumIndex = 2278u + parameterIndex;
                uint velocityIndex = 3417u + parameterIndex;
//              uint velocityIndex = 3417u + parameterIndex;

                float momentum = neuralNetworkWeights[momentumIndex] * 0.9 + 0.1 * averagedGradient;
//              float momentum = neuralNetworkWeights[momentumIndex] * 0.9 + 0.1 * averagedGradient;
                float velocity = neuralNetworkWeights[velocityIndex] * 0.99 + 0.01 * averagedGradient * averagedGradient;
//              float velocity = neuralNetworkWeights[velocityIndex] * 0.99 + 0.01 * averagedGradient * averagedGradient;

                neuralNetworkWeights[momentumIndex] = momentum;
//              neuralNetworkWeights[momentumIndex] = momentum;
                neuralNetworkWeights[velocityIndex] = velocity;
//              neuralNetworkWeights[velocityIndex] = velocity;

                float learningRate = 0.002;
//              float learningRate = 0.002;
                float update = learningRate * (momentum / biasCorrection1) / (sqrt(velocity / biasCorrection2) + 1.0e-8);
//              float update = learningRate * (momentum / biasCorrection1) / (sqrt(velocity / biasCorrection2) + 1.0e-8);

                segmentWeights[lane] -= update;
//              segmentWeights[lane] -= update;
            }
//          }
        }
//      }
        nrcStoreSharedWeights(segment, segmentWeights);
//      nrcStoreSharedWeights(segment, segmentWeights);
    }
//  }

    [shader("compute")]
//  [shader("compute")]
    [numthreads(256, 1, 1)]
//  [numthreads(256, 1, 1)]
    void computeMain(uint3 globalInvocationID : SV_DispatchThreadID, uint localInvocationIndex : SV_GroupIndex) {
//  void computeMain(uint3 globalInvocationID : SV_DispatchThreadID, uint localInvocationIndex : SV_GroupIndex) {
        uint localIndex = localInvocationIndex;
//      uint localIndex = localInvocationIndex;

        if (localIndex == 0u) {
//      if (localIndex == 0u) {
            shared_max_records = min(trainingRecordCounter[0], MAX_TRAINING_RECORDS);
//          shared_max_records = min(trainingRecordCounter[0], MAX_TRAINING_RECORDS);
        }
//      }

        // Load Active training weights into shared memory
//      // Load Active training weights into shared memory
        for (uint segment = localIndex; segment < NRC_NETWORK_FLOAT4_COUNT; segment += THREAD_COUNT) {
//      for (uint segment = localIndex; segment < NRC_NETWORK_FLOAT4_COUNT; segment += THREAD_COUNT) {
            float4 segmentWeights = float4(0.0);
//          float4 segmentWeights = float4(0.0);
            [ForceUnroll]
//          [ForceUnroll]
            for (uint lane = 0u; lane < 4u; lane++) {
//          for (uint lane = 0u; lane < 4u; lane++) {
                if (segment * 4u + lane < NRC_NETWORK_FLOAT_COUNT) {
//              if (segment * 4u + lane < NRC_NETWORK_FLOAT_COUNT) {
                    segmentWeights[lane] = neuralNetworkWeights[segment * 4u + lane];
//                  segmentWeights[lane] = neuralNetworkWeights[segment * 4u + lane];
                }
//              }
            }
//          }
            nrcStoreSharedWeights(segment, segmentWeights);
//          nrcStoreSharedWeights(segment, segmentWeights);
        }
//      }
        GroupMemoryBarrierWithGroupSync();
//      GroupMemoryBarrierWithGroupSync();

        uint maxRecords = shared_max_records;
//      uint maxRecords = shared_max_records;
        // No records gathered: there is nothing to train on, and the network and its EMA copy stay exactly as they are.
//      // No records gathered: there is nothing to train on, and the network and its EMA copy stay exactly as they are.
        // maxRecords comes from shared memory, so this exit is uniform across the workgroup.
//      // maxRecords comes from shared memory, so this exit is uniform across the workgroup.
        if (maxRecords == 0u) {
//      if (maxRecords == 0u) {
            if (localIndex == 0u) {
//          if (localIndex == 0u) {
                trainingRecordCounter[0] = 0u;
//              trainingRecordCounter[0] = 0u;
            }
//          }
            return;
//          return;
        }
//      }

        uint ownedMainSegment = mainSegment(localIndex);
//      uint ownedMainSegment = mainSegment(localIndex);
        bool ownsExtraSegment = localIndex < EXTRA_SEGMENT_COUNT;
//      bool ownsExtraSegment = localIndex < EXTRA_SEGMENT_COUNT;
        uint ownedExtraSegment = ownsExtraSegment ? extraSegment(localIndex) : 0u;
//      uint ownedExtraSegment = ownsExtraSegment ? extraSegment(localIndex) : 0u;

        // Mini-batch SGD (32 steps of batch size 256)
//      // Mini-batch SGD (32 steps of batch size 256)
        for (uint step = 0u; step < TRAINING_STEPS; step++) {
//      for (uint step = 0u; step < TRAINING_STEPS; step++) {
            uint recordIndex = (step * THREAD_COUNT + localIndex) % maxRecords;
//          uint recordIndex = (step * THREAD_COUNT + localIndex) % maxRecords;

            float localInput[16];
//          float localInput[16];
            float localHidden1[16];
//          float localHidden1[16];
            float localHidden2[16];
//          float localHidden2[16];
            float localHidden3[16];
//          float localHidden3[16];
            float localHidden4[16];
//          float localHidden4[16];
            float localGradOutput[3];
//          float localGradOutput[3];
            float localGradHidden4[16];
//          float localGradHidden4[16];
            float localGradHidden3[16];
//          float localGradHidden3[16];
            float localGradHidden2[16];
//          float localGradHidden2[16];
            float localGradHidden1[16];
//          float localGradHidden1[16];

            uint baseOffset = recordIndex * 9u;
//          uint baseOffset = recordIndex * 9u;
            float3 targetRadiance = float3(nrcTrainingRecords[baseOffset + 6u], nrcTrainingRecords[baseOffset + 7u], nrcTrainingRecords[baseOffset + 8u]);
//          float3 targetRadiance = float3(nrcTrainingRecords[baseOffset + 6u], nrcTrainingRecords[baseOffset + 7u], nrcTrainingRecords[baseOffset + 8u]);
            float3 position = float3(nrcTrainingRecords[baseOffset + 0u], nrcTrainingRecords[baseOffset + 1u], nrcTrainingRecords[baseOffset + 2u]);
//          float3 position = float3(nrcTrainingRecords[baseOffset + 0u], nrcTrainingRecords[baseOffset + 1u], nrcTrainingRecords[baseOffset + 2u]);
            float3 normal = float3(nrcTrainingRecords[baseOffset + 3u], nrcTrainingRecords[baseOffset + 4u], nrcTrainingRecords[baseOffset + 5u]);
//          float3 normal = float3(nrcTrainingRecords[baseOffset + 3u], nrcTrainingRecords[baseOffset + 4u], nrcTrainingRecords[baseOffset + 5u]);

            // Reconstruct 16 Input Features
//          // Reconstruct 16 Input Features
            nrcEncodeInput(position, normal, localInput);
//          nrcEncodeInput(position, normal, localInput);

            // Forward Pass
//          // Forward Pass
            nrcForwardHiddenLayer(0u, localInput, localHidden1);
//          nrcForwardHiddenLayer(0u, localInput, localHidden1);
            nrcForwardHiddenLayer(1u, localHidden1, localHidden2);
//          nrcForwardHiddenLayer(1u, localHidden1, localHidden2);
            nrcForwardHiddenLayer(2u, localHidden2, localHidden3);
//          nrcForwardHiddenLayer(2u, localHidden2, localHidden3);
            nrcForwardHiddenLayer(3u, localHidden3, localHidden4);
//          nrcForwardHiddenLayer(3u, localHidden3, localHidden4);

            // Output Error Calculation (L2 Loss Gradient)
//          // Output Error Calculation (L2 Loss Gradient)
            [ForceUnroll]
//          [ForceUnroll]
            for (uint i = 0u; i < 3u; i++) {
//          for (uint i = 0u; i < 3u; i++) {
                float sum = nrcOutputChannel(i, localHidden4);
//              float sum = nrcOutputChannel(i, localHidden4);
                localGradOutput[i] = clamp(sum - targetRadiance[int(i)], -5.0, 5.0);
//              localGradOutput[i] = clamp(sum - targetRadiance[int(i)], -5.0, 5.0);
            }
//          }

            // Backward Pass (Backpropagation with ReLU derivatives)
//          // Backward Pass (Backpropagation with ReLU derivatives)
            backwardLayer<3u>(NRC_OUTPUT_WEIGHTS_FLOAT4_OFFSET, localGradOutput, localHidden4, localGradHidden4);
//          backwardLayer<3u>(NRC_OUTPUT_WEIGHTS_FLOAT4_OFFSET, localGradOutput, localHidden4, localGradHidden4);
            backwardLayer<16u>(3u * NRC_HIDDEN_LAYER_FLOAT4_STRIDE, localGradHidden4, localHidden3, localGradHidden3);
//          backwardLayer<16u>(3u * NRC_HIDDEN_LAYER_FLOAT4_STRIDE, localGradHidden4, localHidden3, localGradHidden3);
            backwardLayer<16u>(2u * NRC_HIDDEN_LAYER_FLOAT4_STRIDE, localGradHidden3, localHidden2, localGradHidden2);
//          backwardLayer<16u>(2u * NRC_HIDDEN_LAYER_FLOAT4_STRIDE, localGradHidden3, localHidden2, localGradHidden2);
            backwardLayer<16u>(1u * NRC_HIDDEN_LAYER_FLOAT4_STRIDE, localGradHidden2, localHidden1, localGradHidden1);
//          backwardLayer<16u>(1u * NRC_HIDDEN_LAYER_FLOAT4_STRIDE, localGradHidden2, localHidden1, localGradHidden1);

            // Collaborative Gradient Accumulation via Shared Memory: the batch is staged RECORDS_PER_CHUNK records at
//          // Collaborative Gradient Accumulation via Shared Memory: the batch is staged RECORDS_PER_CHUNK records at
            // a time, and each thread adds the chunk's partial sum for the segments it owns.
//          // a time, and each thread adds the chunk's partial sum for the segments it owns.
            float4 accumulatedMainGradient = float4(0.0);
//          float4 accumulatedMainGradient = float4(0.0);
            float4 accumulatedExtraGradient = float4(0.0);
//          float4 accumulatedExtraGradient = float4(0.0);

            for (uint chunk = 0u; chunk < CHUNK_COUNT; chunk++) {
//          for (uint chunk = 0u; chunk < CHUNK_COUNT; chunk++) {
                if (localIndex / RECORDS_PER_CHUNK == chunk) {
//              if (localIndex / RECORDS_PER_CHUNK == chunk) {
                    uint record = localIndex % RECORDS_PER_CHUNK;
//                  uint record = localIndex % RECORDS_PER_CHUNK;
                    stageActivations(0u, record, localInput);
//                  stageActivations(0u, record, localInput);
                    stageActivations(1u, record, localHidden1);
//                  stageActivations(1u, record, localHidden1);
                    stageActivations(2u, record, localHidden2);
//                  stageActivations(2u, record, localHidden2);
                    stageActivations(3u, record, localHidden3);
//                  stageActivations(3u, record, localHidden3);
                    stageActivations(4u, record, localHidden4);
//                  stageActivations(4u, record, localHidden4);
                    stageHiddenGradients(0u, record, localGradHidden1);
//                  stageHiddenGradients(0u, record, localGradHidden1);
                    stageHiddenGradients(1u, record, localGradHidden2);
//                  stageHiddenGradients(1u, record, localGradHidden2);
                    stageHiddenGradients(2u, record, localGradHidden3);
//                  stageHiddenGradients(2u, record, localGradHidden3);
                    stageHiddenGradients(3u, record, localGradHidden4);
//                  stageHiddenGradients(3u, record, localGradHidden4);
                    [ForceUnroll]
//                  [ForceUnroll]
                    for (uint i = 0u; i < 3u; i++) {
//                  for (uint i = 0u; i < 3u; i++) {
                        batch_gradients[OUTPUT_GRADIENT_OFFSET + record * 4u + i] = localGradOutput[i];
//                      batch_gradients[OUTPUT_GRADIENT_OFFSET + record * 4u + i] = localGradOutput[i];
                    }
//                  }
                }
//              }
                GroupMemoryBarrierWithGroupSync();
//              GroupMemoryBarrierWithGroupSync();

                accumulatedMainGradient += reduceMainSegment(localIndex);
//              accumulatedMainGradient += reduceMainSegment(localIndex);
                if (ownsExtraSegment) {
//              if (ownsExtraSegment) {
                    accumulatedExtraGradient += reduceExtraSegment(localIndex);
//                  accumulatedExtraGradient += reduceExtraSegment(localIndex);
                }
//              }
                GroupMemoryBarrierWithGroupSync();
//              GroupMemoryBarrierWithGroupSync();
            }
//          }

            // Apply Adam Optimizer step for all aggregated parameters
//          // Apply Adam Optimizer step for all aggregated parameters
            float globalTrainingStep = min(float(uTrainingStep * TRAINING_STEPS + step) + 1.0, 50000.0);
//          float globalTrainingStep = min(float(uTrainingStep * TRAINING_STEPS + step) + 1.0, 50000.0);
            float biasCorrection1 = 1.0 - pow(0.9, globalTrainingStep);
//          float biasCorrection1 = 1.0 - pow(0.9, globalTrainingStep);
            float biasCorrection2 = 1.0 - pow(0.99, globalTrainingStep);
//          float biasCorrection2 = 1.0 - pow(0.99, globalTrainingStep);
            applyAdamSegment(ownedMainSegment, accumulatedMainGradient, biasCorrection1, biasCorrection2);
//          applyAdamSegment(ownedMainSegment, accumulatedMainGradient, biasCorrection1, biasCorrection2);
            if (ownsExtraSegment) {
//          if (ownsExtraSegment) {
                applyAdamSegment(ownedExtraSegment, accumulatedExtraGradient, biasCorrection1, biasCorrection2);
//              applyAdamSegment(ownedExtraSegment, accumulatedExtraGradient, biasCorrection1, biasCorrection2);
            }
//          }
            GroupMemoryBarrierWithGroupSync();
//          GroupMemoryBarrierWithGroupSync();
        }
//      }

        // Store trained active weights and update Exponential Moving Average (EMA) weights
//      // Store trained active weights and update Exponential Moving Average (EMA) weights
        for (uint segment = localIndex; segment < NRC_NETWORK_FLOAT4_COUNT; segment += THREAD_COUNT) {
//      for (uint segment = localIndex; segment < NRC_NETWORK_FLOAT4_COUNT; segment += THREAD_COUNT) {
            float4 segmentWeights = nrcLoadSharedWeights(segment);
//          float4 segmentWeights = nrcLoadSharedWeights(segment);
            [ForceUnroll]
//          [ForceUnroll]
            for (uint lane = 0u; lane < 4u; lane++) {
//          for (uint lane = 0u; lane < 4u; lane++) {
                uint parameterIndex = segment * 4u + lane;
//              uint parameterIndex = segment * 4u + lane;
                if (parameterIndex < NRC_NETWORK_FLOAT_COUNT) {
//              if (parameterIndex < NRC_NETWORK_FLOAT_COUNT) {
                    neuralNetworkWeights[parameterIndex] = segmentWeights[lane];
//                  neuralNetworkWeights[parameterIndex] = segmentWeights[lane];
                    neuralNetworkWeights[NRC_EMA_WEIGHTS_OFFSET + parameterIndex] = 0.95 * neuralNetworkWeights[NRC_EMA_WEIGHTS_OFFSET + parameterIndex] + 0.05 * segmentWeights[lane];
//                  neuralNetworkWeights[NRC_EMA_WEIGHTS_OFFSET + parameterIndex] = 0.95 * neuralNetworkWeights[NRC_EMA_WEIGHTS_OFFSET + parameterIndex] + 0.05 * segmentWeights[lane];
                }
//              }
            }
//          }
        }
//      }

        if (localIndex == 0u) {
//      if (localIndex == 0u) {
            trainingRecordCounter[0] = 0u;
//          trainingRecordCounter[0] = 0u;
        }
//      }
    }
//  }
