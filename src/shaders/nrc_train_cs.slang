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
    //
//  //
    // Training runs as a single workgroup (every step depends on the one before), so it occupies one SM. To keep that
//  // Training runs as a single workgroup (every step depends on the one before), so it occupies one SM. To keep that
    // SM busy, each record of the mini-batch is shared by a quad of four neighbouring subgroup lanes: lane q of the
//  // SM busy, each record of the mini-batch is shared by a quad of four neighbouring subgroup lanes: lane q of the
    // quad owns units 4q..4q+3 of every 16-wide layer, and the quad trades the rest of each layer through subgroup
//  // quad owns units 4q..4q+3 of every 16-wide layer, and the quad trades the rest of each layer through subgroup
    // shuffles. That is four times the warps of one-record-per-thread for the same work, which hides the shared-memory
//  // shuffles. That is four times the warps of one-record-per-thread for the same work, which hides the shared-memory
    // and FMA latency the per-record dependency chains used to stall on, and stages each record four times faster.
//  // and FMA latency the per-record dependency chains used to stall on, and stages each record four times faster.
    // Every unit is still summed in the same order as the scalar version, so the arithmetic is unchanged.
//  // Every unit is still summed in the same order as the scalar version, so the arithmetic is unchanged.
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
    static const uint BATCH_SIZE = 256u; // records per mini-batch step
//  static const uint BATCH_SIZE = 256u; // records per mini-batch step
    static const uint LANES_PER_RECORD = 4u; // one quad of subgroup lanes per record, each owning four units of a layer
//  static const uint LANES_PER_RECORD = 4u; // one quad of subgroup lanes per record, each owning four units of a layer
    static const uint THREAD_COUNT = BATCH_SIZE * LANES_PER_RECORD;
//  static const uint THREAD_COUNT = BATCH_SIZE * LANES_PER_RECORD;
    static const uint TRAINING_STEPS = 32u;
//  static const uint TRAINING_STEPS = 32u;
    static const uint MAX_TRAINING_RECORDS = 8192u;
//  static const uint MAX_TRAINING_RECORDS = 8192u;
    // Records staged to shared memory per round of the gradient reduction. 16 keeps the staging buffers inside WebGPU's
//  // Records staged to shared memory per round of the gradient reduction. 16 keeps the staging buffers inside WebGPU's
    // default 16 KiB of workgroup storage; the reduction adds each chunk's partial sum in turn.
//  // default 16 KiB of workgroup storage; the reduction adds each chunk's partial sum in turn.
    static const uint RECORDS_PER_CHUNK = 16u;
//  static const uint RECORDS_PER_CHUNK = 16u;
    static const uint CHUNK_COUNT = BATCH_SIZE / RECORDS_PER_CHUNK;
//  static const uint CHUNK_COUNT = BATCH_SIZE / RECORDS_PER_CHUNK;
    // Float4 segments of the network: threads 0..255 own the four hidden weight matrices (64 segments each), threads
//  // Float4 segments of the network: threads 0..255 own the four hidden weight matrices (64 segments each), threads
    // 256..284 the hidden biases (16), the output weights (12) and the output biases (1).
//  // 256..284 the hidden biases (16), the output weights (12) and the output biases (1).
    static const uint MAIN_SEGMENT_COUNT = 256u;
//  static const uint MAIN_SEGMENT_COUNT = 256u;
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

    // Value of another lane of this subgroup. Slang's WaveReadLaneAt hands WGSL's subgroupShuffle a signed lane index,
//  // Value of another lane of this subgroup. Slang's WaveReadLaneAt hands WGSL's subgroupShuffle a signed lane index,
    // which this wgpu's WGSL front end (naga) rejects, so the WGSL target spells the call out with the unsigned index.
//  // which this wgpu's WGSL front end (naga) rejects, so the WGSL target spells the call out with the unsigned index.
    float4 readSubgroupLane(float4 value, uint lane) {
//  float4 readSubgroupLane(float4 value, uint lane) {
        __target_switch {
//      __target_switch {
        case wgsl: __intrinsic_asm "subgroupShuffle($0, $1)";
//      case wgsl: __intrinsic_asm "subgroupShuffle($0, $1)";
        default: return WaveReadLaneAt(value, lane);
//      default: return WaveReadLaneAt(value, lane);
        }
//      }
    }
//  }

    // Reassembles a full 16-wide vector from the four quads the lanes of this thread's quad each hold.
//  // Reassembles a full 16-wide vector from the four quads the lanes of this thread's quad each hold.
    [ForceInline]
//  [ForceInline]
    void gatherQuadVector(float4 ownQuad, uint quadFirstLane, out float vector[16]) {
//  void gatherQuadVector(float4 ownQuad, uint quadFirstLane, out float vector[16]) {
        [ForceUnroll]
//      [ForceUnroll]
        for (uint quad = 0u; quad < 4u; quad++) {
//      for (uint quad = 0u; quad < 4u; quad++) {
            float4 quadValues = readSubgroupLane(ownQuad, quadFirstLane + quad);
//          float4 quadValues = readSubgroupLane(ownQuad, quadFirstLane + quad);
            vector[quad * 4u + 0u] = quadValues.x;
//          vector[quad * 4u + 0u] = quadValues.x;
            vector[quad * 4u + 1u] = quadValues.y;
//          vector[quad * 4u + 1u] = quadValues.y;
            vector[quad * 4u + 2u] = quadValues.z;
//          vector[quad * 4u + 2u] = quadValues.z;
            vector[quad * 4u + 3u] = quadValues.w;
//          vector[quad * 4u + 3u] = quadValues.w;
        }
//      }
    }
//  }

    // Units 4 * quad .. 4 * quad + 3 of a 16-wide vector, picked with selects rather than a dynamically indexed array
//  // Units 4 * quad .. 4 * quad + 3 of a 16-wide vector, picked with selects rather than a dynamically indexed array
    // (which would push the whole array out of registers).
//  // (which would push the whole array out of registers).
    [ForceInline]
//  [ForceInline]
    float4 pickQuad(float vector[16], uint quad) {
//  float4 pickQuad(float vector[16], uint quad) {
        float4 picked = float4(vector[0], vector[1], vector[2], vector[3]);
//      float4 picked = float4(vector[0], vector[1], vector[2], vector[3]);
        picked = (quad == 1u) ? float4(vector[4], vector[5], vector[6], vector[7]) : picked;
//      picked = (quad == 1u) ? float4(vector[4], vector[5], vector[6], vector[7]) : picked;
        picked = (quad == 2u) ? float4(vector[8], vector[9], vector[10], vector[11]) : picked;
//      picked = (quad == 2u) ? float4(vector[8], vector[9], vector[10], vector[11]) : picked;
        picked = (quad == 3u) ? float4(vector[12], vector[13], vector[14], vector[15]) : picked;
//      picked = (quad == 3u) ? float4(vector[12], vector[13], vector[14], vector[15]) : picked;
        return picked;
//      return picked;
    }
//  }

    // This lane's four units of a hidden layer's forward pass with ReLU (nrcForwardHiddenLayer, one quad of it).
//  // This lane's four units of a hidden layer's forward pass with ReLU (nrcForwardHiddenLayer, one quad of it).
    [ForceInline]
//  [ForceInline]
    float4 forwardHiddenQuad(uint layer, uint quad, float layerInput[16]) {
//  float4 forwardHiddenQuad(uint layer, uint quad, float layerInput[16]) {
        uint layerFloat4Offset = layer * NRC_HIDDEN_LAYER_FLOAT4_STRIDE;
//      uint layerFloat4Offset = layer * NRC_HIDDEN_LAYER_FLOAT4_STRIDE;
        float4 bias = nrcLoadSharedWeights(layerFloat4Offset + NRC_HIDDEN_BIAS_FLOAT4_OFFSET + quad);
//      float4 bias = nrcLoadSharedWeights(layerFloat4Offset + NRC_HIDDEN_BIAS_FLOAT4_OFFSET + quad);
        float4 layerOutput;
//      float4 layerOutput;
        [ForceUnroll]
//      [ForceUnroll]
        for (uint lane = 0u; lane < 4u; lane++) {
//      for (uint lane = 0u; lane < 4u; lane++) {
            layerOutput[lane] = max(0.0, nrcDenseRow(layerFloat4Offset + (quad * 4u + lane) * 4u, bias[lane], layerInput));
//          layerOutput[lane] = max(0.0, nrcDenseRow(layerFloat4Offset + (quad * 4u + lane) * 4u, bias[lane], layerInput));
        }
//      }
        return layerOutput;
//      return layerOutput;
    }
//  }

    // This lane's four units of the gradient back-propagated through ROWS rows of weights:
//  // This lane's four units of the gradient back-propagated through ROWS rows of weights:
    // gradientInput[i] = sum_j gradientOutput[j] * W[j][i] (ascending j), masked by the ReLU derivative of the
//  // gradientInput[i] = sum_j gradientOutput[j] * W[j][i] (ascending j), masked by the ReLU derivative of the
    // activation the row consumed. Column quad `quad` of each row is one float4 load.
//  // activation the row consumed. Column quad `quad` of each row is one float4 load.
    [ForceInline]
//  [ForceInline]
    float4 backwardQuad<let ROWS : uint>(uint weightsFloat4Offset, uint quad, float gradientOutput[ROWS], float4 activationQuad) {
//  float4 backwardQuad<let ROWS : uint>(uint weightsFloat4Offset, uint quad, float gradientOutput[ROWS], float4 activationQuad) {
        float4 sums = float4(0.0);
//      float4 sums = float4(0.0);
        [ForceUnroll]
//      [ForceUnroll]
        for (uint row = 0u; row < ROWS; row++) {
//      for (uint row = 0u; row < ROWS; row++) {
            sums += gradientOutput[row] * nrcLoadSharedWeights(weightsFloat4Offset + row * 4u + quad);
//          sums += gradientOutput[row] * nrcLoadSharedWeights(weightsFloat4Offset + row * 4u + quad);
        }
//      }
        return select(activationQuad > float4(0.0), sums, float4(0.0));
//      return select(activationQuad > float4(0.0), sums, float4(0.0));
    }
//  }

    [ForceInline]
//  [ForceInline]
    void stageHiddenGradientQuad(uint layer, uint record, uint quad, float4 gradientQuad) {
//  void stageHiddenGradientQuad(uint layer, uint record, uint quad, float4 gradientQuad) {
        uint gradientOffset = (layer * RECORDS_PER_CHUNK + record) * 16u + quad * 4u;
//      uint gradientOffset = (layer * RECORDS_PER_CHUNK + record) * 16u + quad * 4u;
        batch_gradients[gradientOffset + 0u] = gradientQuad.x;
//      batch_gradients[gradientOffset + 0u] = gradientQuad.x;
        batch_gradients[gradientOffset + 1u] = gradientQuad.y;
//      batch_gradients[gradientOffset + 1u] = gradientQuad.y;
        batch_gradients[gradientOffset + 2u] = gradientQuad.z;
//      batch_gradients[gradientOffset + 2u] = gradientQuad.z;
        batch_gradients[gradientOffset + 3u] = gradientQuad.w;
//      batch_gradients[gradientOffset + 3u] = gradientQuad.w;
    }
//  }

    // Float4 segment of the network a thread owns in the gradient reduction and the Adam step. Threads 0..255 each
//  // Float4 segment of the network a thread owns in the gradient reduction and the Adam step. Threads 0..255 each
    // own one segment of the four hidden weight matrices (warps 0-1 layer 1, 2-3 layer 2, ...), so the reduction
//  // own one segment of the four hidden weight matrices (warps 0-1 layer 1, 2-3 layer 2, ...), so the reduction
    // branch is uniform across every warp; extra segment e (thread 256 + e) is a hidden bias (0..15), an output
//  // branch is uniform across every warp; extra segment e (thread 256 + e) is a hidden bias (0..15), an output
    // weight (16..27) or the output biases (28).
//  // weight (16..27) or the output biases (28).
    uint mainSegment(uint threadIndex) {
//  uint mainSegment(uint threadIndex) {
        return (threadIndex / 64u) * NRC_HIDDEN_LAYER_FLOAT4_STRIDE + threadIndex % 64u;
//      return (threadIndex / 64u) * NRC_HIDDEN_LAYER_FLOAT4_STRIDE + threadIndex % 64u;
    }
//  }

    uint extraSegment(uint extraIndex) {
//  uint extraSegment(uint extraIndex) {
        if (extraIndex < 16u) {
//      if (extraIndex < 16u) {
            return (extraIndex / 4u) * NRC_HIDDEN_LAYER_FLOAT4_STRIDE + NRC_HIDDEN_BIAS_FLOAT4_OFFSET + extraIndex % 4u;
//          return (extraIndex / 4u) * NRC_HIDDEN_LAYER_FLOAT4_STRIDE + NRC_HIDDEN_BIAS_FLOAT4_OFFSET + extraIndex % 4u;
        }
//      }
        return NRC_OUTPUT_WEIGHTS_FLOAT4_OFFSET + (extraIndex - 16u);
//      return NRC_OUTPUT_WEIGHTS_FLOAT4_OFFSET + (extraIndex - 16u);
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
    float4 reduceExtraSegment(uint extraIndex) {
//  float4 reduceExtraSegment(uint extraIndex) {
        float4 partialGradient = float4(0.0);
//      float4 partialGradient = float4(0.0);
        if (extraIndex < 16u) {
//      if (extraIndex < 16u) {
            uint layer = extraIndex / 4u;
//          uint layer = extraIndex / 4u;
            uint quad = extraIndex % 4u;
//          uint quad = extraIndex % 4u;
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
        } else if (extraIndex < 28u) {
//      } else if (extraIndex < 28u) {
            uint row = (extraIndex - 16u) / 4u;
//          uint row = (extraIndex - 16u) / 4u;
            uint quad = extraIndex % 4u;
//          uint quad = extraIndex % 4u;
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
                float averagedGradient = accumulatedGradient[lane] / float(BATCH_SIZE);
//              float averagedGradient = accumulatedGradient[lane] / float(BATCH_SIZE);
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
    [numthreads(1024, 1, 1)]
//  [numthreads(1024, 1, 1)]
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

        // A 1D workgroup fills its subgroups with consecutive invocations, so the four invocations of a record are
//      // A 1D workgroup fills its subgroups with consecutive invocations, so the four invocations of a record are
        // also four consecutive lanes of one subgroup, starting at a multiple of four.
//      // also four consecutive lanes of one subgroup, starting at a multiple of four.
        uint record = localIndex / LANES_PER_RECORD;
//      uint record = localIndex / LANES_PER_RECORD;
        uint quad = localIndex % LANES_PER_RECORD;
//      uint quad = localIndex % LANES_PER_RECORD;
        uint quadFirstLane = WaveGetLaneIndex() & ~(LANES_PER_RECORD - 1u);
//      uint quadFirstLane = WaveGetLaneIndex() & ~(LANES_PER_RECORD - 1u);
        bool ownsMainSegment = localIndex < MAIN_SEGMENT_COUNT;
//      bool ownsMainSegment = localIndex < MAIN_SEGMENT_COUNT;
        bool ownsExtraSegment = localIndex >= MAIN_SEGMENT_COUNT && localIndex < MAIN_SEGMENT_COUNT + EXTRA_SEGMENT_COUNT;
//      bool ownsExtraSegment = localIndex >= MAIN_SEGMENT_COUNT && localIndex < MAIN_SEGMENT_COUNT + EXTRA_SEGMENT_COUNT;
        uint extraIndex = localIndex - MAIN_SEGMENT_COUNT;
//      uint extraIndex = localIndex - MAIN_SEGMENT_COUNT;
        uint ownedSegment = ownsMainSegment ? mainSegment(localIndex) : (ownsExtraSegment ? extraSegment(extraIndex) : 0u);
//      uint ownedSegment = ownsMainSegment ? mainSegment(localIndex) : (ownsExtraSegment ? extraSegment(extraIndex) : 0u);

        // Mini-batch SGD (32 steps of batch size 256)
//      // Mini-batch SGD (32 steps of batch size 256)
        for (uint step = 0u; step < TRAINING_STEPS; step++) {
//      for (uint step = 0u; step < TRAINING_STEPS; step++) {
            uint recordIndex = (step * BATCH_SIZE + record) % maxRecords;
//          uint recordIndex = (step * BATCH_SIZE + record) % maxRecords;

            uint baseOffset = recordIndex * 9u;
//          uint baseOffset = recordIndex * 9u;
            float3 targetRadiance = float3(nrcTrainingRecords[baseOffset + 6u], nrcTrainingRecords[baseOffset + 7u], nrcTrainingRecords[baseOffset + 8u]);
//          float3 targetRadiance = float3(nrcTrainingRecords[baseOffset + 6u], nrcTrainingRecords[baseOffset + 7u], nrcTrainingRecords[baseOffset + 8u]);
            float3 position = float3(nrcTrainingRecords[baseOffset + 0u], nrcTrainingRecords[baseOffset + 1u], nrcTrainingRecords[baseOffset + 2u]);
//          float3 position = float3(nrcTrainingRecords[baseOffset + 0u], nrcTrainingRecords[baseOffset + 1u], nrcTrainingRecords[baseOffset + 2u]);
            float3 normal = float3(nrcTrainingRecords[baseOffset + 3u], nrcTrainingRecords[baseOffset + 4u], nrcTrainingRecords[baseOffset + 5u]);
//          float3 normal = float3(nrcTrainingRecords[baseOffset + 3u], nrcTrainingRecords[baseOffset + 4u], nrcTrainingRecords[baseOffset + 5u]);

            // Reconstruct 16 Input Features (every lane of the quad encodes the whole record; it is cheaper than a shuffle)
//          // Reconstruct 16 Input Features (every lane of the quad encodes the whole record; it is cheaper than a shuffle)
            float layerInput[16];
//          float layerInput[16];
            nrcEncodeInput(position, normal, layerInput);
//          nrcEncodeInput(position, normal, layerInput);
            float4 inputQuad = pickQuad(layerInput, quad);
//          float4 inputQuad = pickQuad(layerInput, quad);

            // Forward Pass: each lane computes its quad of a layer, then the quad trades quads for the next layer's input.
//          // Forward Pass: each lane computes its quad of a layer, then the quad trades quads for the next layer's input.
            float4 hidden1Quad = forwardHiddenQuad(0u, quad, layerInput);
//          float4 hidden1Quad = forwardHiddenQuad(0u, quad, layerInput);
            gatherQuadVector(hidden1Quad, quadFirstLane, layerInput);
//          gatherQuadVector(hidden1Quad, quadFirstLane, layerInput);
            float4 hidden2Quad = forwardHiddenQuad(1u, quad, layerInput);
//          float4 hidden2Quad = forwardHiddenQuad(1u, quad, layerInput);
            gatherQuadVector(hidden2Quad, quadFirstLane, layerInput);
//          gatherQuadVector(hidden2Quad, quadFirstLane, layerInput);
            float4 hidden3Quad = forwardHiddenQuad(2u, quad, layerInput);
//          float4 hidden3Quad = forwardHiddenQuad(2u, quad, layerInput);
            gatherQuadVector(hidden3Quad, quadFirstLane, layerInput);
//          gatherQuadVector(hidden3Quad, quadFirstLane, layerInput);
            float4 hidden4Quad = forwardHiddenQuad(3u, quad, layerInput);
//          float4 hidden4Quad = forwardHiddenQuad(3u, quad, layerInput);
            gatherQuadVector(hidden4Quad, quadFirstLane, layerInput);
//          gatherQuadVector(hidden4Quad, quadFirstLane, layerInput);

            // Output Error Calculation (L2 Loss Gradient), evaluated by every lane of the quad
//          // Output Error Calculation (L2 Loss Gradient), evaluated by every lane of the quad
            float localGradOutput[3];
//          float localGradOutput[3];
            [ForceUnroll]
//          [ForceUnroll]
            for (uint i = 0u; i < 3u; i++) {
//          for (uint i = 0u; i < 3u; i++) {
                float sum = nrcOutputChannel(i, layerInput);
//              float sum = nrcOutputChannel(i, layerInput);
                localGradOutput[i] = clamp(sum - targetRadiance[int(i)], -5.0, 5.0);
//              localGradOutput[i] = clamp(sum - targetRadiance[int(i)], -5.0, 5.0);
            }
//          }

            // Backward Pass (Backpropagation with ReLU derivatives), one quad of each gradient per lane
//          // Backward Pass (Backpropagation with ReLU derivatives), one quad of each gradient per lane
            float layerGradient[16];
//          float layerGradient[16];
            float4 gradHidden4Quad = backwardQuad<3u>(NRC_OUTPUT_WEIGHTS_FLOAT4_OFFSET, quad, localGradOutput, hidden4Quad);
//          float4 gradHidden4Quad = backwardQuad<3u>(NRC_OUTPUT_WEIGHTS_FLOAT4_OFFSET, quad, localGradOutput, hidden4Quad);
            gatherQuadVector(gradHidden4Quad, quadFirstLane, layerGradient);
//          gatherQuadVector(gradHidden4Quad, quadFirstLane, layerGradient);
            float4 gradHidden3Quad = backwardQuad<16u>(3u * NRC_HIDDEN_LAYER_FLOAT4_STRIDE, quad, layerGradient, hidden3Quad);
//          float4 gradHidden3Quad = backwardQuad<16u>(3u * NRC_HIDDEN_LAYER_FLOAT4_STRIDE, quad, layerGradient, hidden3Quad);
            gatherQuadVector(gradHidden3Quad, quadFirstLane, layerGradient);
//          gatherQuadVector(gradHidden3Quad, quadFirstLane, layerGradient);
            float4 gradHidden2Quad = backwardQuad<16u>(2u * NRC_HIDDEN_LAYER_FLOAT4_STRIDE, quad, layerGradient, hidden2Quad);
//          float4 gradHidden2Quad = backwardQuad<16u>(2u * NRC_HIDDEN_LAYER_FLOAT4_STRIDE, quad, layerGradient, hidden2Quad);
            gatherQuadVector(gradHidden2Quad, quadFirstLane, layerGradient);
//          gatherQuadVector(gradHidden2Quad, quadFirstLane, layerGradient);
            float4 gradHidden1Quad = backwardQuad<16u>(1u * NRC_HIDDEN_LAYER_FLOAT4_STRIDE, quad, layerGradient, hidden1Quad);
//          float4 gradHidden1Quad = backwardQuad<16u>(1u * NRC_HIDDEN_LAYER_FLOAT4_STRIDE, quad, layerGradient, hidden1Quad);

            // Collaborative Gradient Accumulation via Shared Memory: the batch is staged RECORDS_PER_CHUNK records at
//          // Collaborative Gradient Accumulation via Shared Memory: the batch is staged RECORDS_PER_CHUNK records at
            // a time (each lane writing its own quads), and each owning thread adds the chunk's partial sum in turn.
//          // a time (each lane writing its own quads), and each owning thread adds the chunk's partial sum in turn.
            float4 accumulatedGradient = float4(0.0);
//          float4 accumulatedGradient = float4(0.0);

            for (uint chunk = 0u; chunk < CHUNK_COUNT; chunk++) {
//          for (uint chunk = 0u; chunk < CHUNK_COUNT; chunk++) {
                if (record / RECORDS_PER_CHUNK == chunk) {
//              if (record / RECORDS_PER_CHUNK == chunk) {
                    uint chunkRecord = record % RECORDS_PER_CHUNK;
//                  uint chunkRecord = record % RECORDS_PER_CHUNK;
                    batch_activations[(0u * RECORDS_PER_CHUNK + chunkRecord) * 4u + quad] = inputQuad;
//                  batch_activations[(0u * RECORDS_PER_CHUNK + chunkRecord) * 4u + quad] = inputQuad;
                    batch_activations[(1u * RECORDS_PER_CHUNK + chunkRecord) * 4u + quad] = hidden1Quad;
//                  batch_activations[(1u * RECORDS_PER_CHUNK + chunkRecord) * 4u + quad] = hidden1Quad;
                    batch_activations[(2u * RECORDS_PER_CHUNK + chunkRecord) * 4u + quad] = hidden2Quad;
//                  batch_activations[(2u * RECORDS_PER_CHUNK + chunkRecord) * 4u + quad] = hidden2Quad;
                    batch_activations[(3u * RECORDS_PER_CHUNK + chunkRecord) * 4u + quad] = hidden3Quad;
//                  batch_activations[(3u * RECORDS_PER_CHUNK + chunkRecord) * 4u + quad] = hidden3Quad;
                    batch_activations[(4u * RECORDS_PER_CHUNK + chunkRecord) * 4u + quad] = hidden4Quad;
//                  batch_activations[(4u * RECORDS_PER_CHUNK + chunkRecord) * 4u + quad] = hidden4Quad;
                    stageHiddenGradientQuad(0u, chunkRecord, quad, gradHidden1Quad);
//                  stageHiddenGradientQuad(0u, chunkRecord, quad, gradHidden1Quad);
                    stageHiddenGradientQuad(1u, chunkRecord, quad, gradHidden2Quad);
//                  stageHiddenGradientQuad(1u, chunkRecord, quad, gradHidden2Quad);
                    stageHiddenGradientQuad(2u, chunkRecord, quad, gradHidden3Quad);
//                  stageHiddenGradientQuad(2u, chunkRecord, quad, gradHidden3Quad);
                    stageHiddenGradientQuad(3u, chunkRecord, quad, gradHidden4Quad);
//                  stageHiddenGradientQuad(3u, chunkRecord, quad, gradHidden4Quad);
                    if (quad == 0u) {
//                  if (quad == 0u) {
                        [ForceUnroll]
//                      [ForceUnroll]
                        for (uint i = 0u; i < 3u; i++) {
//                      for (uint i = 0u; i < 3u; i++) {
                            batch_gradients[OUTPUT_GRADIENT_OFFSET + chunkRecord * 4u + i] = localGradOutput[i];
//                          batch_gradients[OUTPUT_GRADIENT_OFFSET + chunkRecord * 4u + i] = localGradOutput[i];
                        }
//                      }
                    }
//                  }
                }
//              }
                GroupMemoryBarrierWithGroupSync();
//              GroupMemoryBarrierWithGroupSync();

                if (ownsMainSegment) {
//              if (ownsMainSegment) {
                    accumulatedGradient += reduceMainSegment(localIndex);
//                  accumulatedGradient += reduceMainSegment(localIndex);
                } else if (ownsExtraSegment) {
//              } else if (ownsExtraSegment) {
                    accumulatedGradient += reduceExtraSegment(extraIndex);
//                  accumulatedGradient += reduceExtraSegment(extraIndex);
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
            if (ownsMainSegment || ownsExtraSegment) {
//          if (ownsMainSegment || ownsExtraSegment) {
                applyAdamSegment(ownedSegment, accumulatedGradient, biasCorrection1, biasCorrection2);
//              applyAdamSegment(ownedSegment, accumulatedGradient, biasCorrection1, biasCorrection2);
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
