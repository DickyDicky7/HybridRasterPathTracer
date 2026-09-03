    #version 430
//  #version 430

    layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;
//  layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

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
    layout(std430, binding = 11) buffer NRCWeights {
//  layout(std430, binding = 11) buffer NRCWeights {
        float neuralNetworkWeights[];
//      float neuralNetworkWeights[];
    };
//  };

    layout(std430, binding = 12) buffer NRCTrainingRecords {
//  layout(std430, binding = 12) buffer NRCTrainingRecords {
        float nrcTrainingRecords[];
//      float nrcTrainingRecords[];
    };
//  };

    layout(std430, binding = 13) buffer NRCTrainingCounter {
//  layout(std430, binding = 13) buffer NRCTrainingCounter {
        uint uTrainingRecordCount;
//      uint uTrainingRecordCount;
    };
//  };

    uniform uint uTrainingStep;
//  uniform uint uTrainingStep;
    // NRC input-domain normalisation; must match hybrid_shading_cs.glsl and nrc_gather_cs.glsl exactly.
//  // NRC input-domain normalisation; must match hybrid_shading_cs.glsl and nrc_gather_cs.glsl exactly.
    uniform vec3 uNRCPositionOffset;
//  uniform vec3 uNRCPositionOffset;
    uniform float uNRCPositionScale;
//  uniform float uNRCPositionScale;

    const float PI = 3.14159265359;
//  const float PI = 3.14159265359;
    const float TWO_PI = 6.28318530718;
//  const float TWO_PI = 6.28318530718;

    shared float shared_network_weights[1139];
//  shared float shared_network_weights[1139];
    shared float batch_input_features[256];
//  shared float batch_input_features[256];
    shared float batch_hidden_layer_1[256];
//  shared float batch_hidden_layer_1[256];
    shared float batch_hidden_layer_2[256];
//  shared float batch_hidden_layer_2[256];
    shared float batch_hidden_layer_3[256];
//  shared float batch_hidden_layer_3[256];
    shared float batch_hidden_layer_4[256];
//  shared float batch_hidden_layer_4[256];
    shared float batch_gradient_output_layer[48];
//  shared float batch_gradient_output_layer[48];
    shared float batch_gradient_hidden_4[256];
//  shared float batch_gradient_hidden_4[256];
    shared float batch_gradient_hidden_3[256];
//  shared float batch_gradient_hidden_3[256];
    shared float batch_gradient_hidden_2[256];
//  shared float batch_gradient_hidden_2[256];
    shared float batch_gradient_hidden_1[256];
//  shared float batch_gradient_hidden_1[256];
    shared uint shared_max_records;
//  shared uint shared_max_records;

    void main() {
//  void main() {
        uint localIndex = gl_LocalInvocationIndex;
//      uint localIndex = gl_LocalInvocationIndex;

        if (localIndex == 0u) {
//      if (localIndex == 0u) {
            shared_max_records = min(uTrainingRecordCount, 8192u);
//          shared_max_records = min(uTrainingRecordCount, 8192u);
        }
//      }

        // Load Active training weights into shared memory
//      // Load Active training weights into shared memory
        for (uint i = localIndex; i < 1139u; i += 256u) {
//      for (uint i = localIndex; i < 1139u; i += 256u) {
            shared_network_weights[i] = neuralNetworkWeights[i];
//          shared_network_weights[i] = neuralNetworkWeights[i];
        }
//      }
        barrier();
//      barrier();

        uint maxRecords = shared_max_records;
//      uint maxRecords = shared_max_records;
        bool isActiveThread = maxRecords > 0u;
//      bool isActiveThread = maxRecords > 0u;

        // Mini-batch SGD (32 steps of batch size 256)
//      // Mini-batch SGD (32 steps of batch size 256)
        for (uint step = 0u; step < 32u; step++) {
//      for (uint step = 0u; step < 32u; step++) {
            uint recordIndex = isActiveThread ? ((step * 256u + localIndex) % max(maxRecords, 1u)) : 0u;
//          uint recordIndex = isActiveThread ? ((step * 256u + localIndex) % max(maxRecords, 1u)) : 0u;

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

            for (uint k = 0u; k < 16u; k++) {
//          for (uint k = 0u; k < 16u; k++) {
                localInput[k] = 0.0;
//              localInput[k] = 0.0;
                localHidden1[k] = 0.0;
//              localHidden1[k] = 0.0;
                localHidden2[k] = 0.0;
//              localHidden2[k] = 0.0;
                localHidden3[k] = 0.0;
//              localHidden3[k] = 0.0;
                localHidden4[k] = 0.0;
//              localHidden4[k] = 0.0;
                localGradHidden4[k] = 0.0;
//              localGradHidden4[k] = 0.0;
                localGradHidden3[k] = 0.0;
//              localGradHidden3[k] = 0.0;
                localGradHidden2[k] = 0.0;
//              localGradHidden2[k] = 0.0;
                localGradHidden1[k] = 0.0;
//              localGradHidden1[k] = 0.0;
            }
//          }
            localGradOutput[0] = 0.0;
//          localGradOutput[0] = 0.0;
            localGradOutput[1] = 0.0;
//          localGradOutput[1] = 0.0;
            localGradOutput[2] = 0.0;
//          localGradOutput[2] = 0.0;

            if (isActiveThread) {
//          if (isActiveThread) {
                uint baseOffset = recordIndex * 9u;
//              uint baseOffset = recordIndex * 9u;
                vec3 targetRadiance = vec3(nrcTrainingRecords[baseOffset + 6u], nrcTrainingRecords[baseOffset + 7u], nrcTrainingRecords[baseOffset + 8u]);
//              vec3 targetRadiance = vec3(nrcTrainingRecords[baseOffset + 6u], nrcTrainingRecords[baseOffset + 7u], nrcTrainingRecords[baseOffset + 8u]);
                vec3 position = (vec3(nrcTrainingRecords[baseOffset + 0u], nrcTrainingRecords[baseOffset + 1u], nrcTrainingRecords[baseOffset + 2u]) - uNRCPositionOffset) * uNRCPositionScale;
//              vec3 position = (vec3(nrcTrainingRecords[baseOffset + 0u], nrcTrainingRecords[baseOffset + 1u], nrcTrainingRecords[baseOffset + 2u]) - uNRCPositionOffset) * uNRCPositionScale;
                vec3 normal = vec3(nrcTrainingRecords[baseOffset + 3u], nrcTrainingRecords[baseOffset + 4u], nrcTrainingRecords[baseOffset + 5u]);
//              vec3 normal = vec3(nrcTrainingRecords[baseOffset + 3u], nrcTrainingRecords[baseOffset + 4u], nrcTrainingRecords[baseOffset + 5u]);

                // Reconstruct 16 Input Features
//              // Reconstruct 16 Input Features
                localInput[0] = position.x;
//              localInput[0] = position.x;
                localInput[1] = position.y;
//              localInput[1] = position.y;
                localInput[2] = position.z;
//              localInput[2] = position.z;
                localInput[3] = normal.x;
//              localInput[3] = normal.x;
                localInput[4] = normal.y;
//              localInput[4] = normal.y;
                localInput[5] = normal.z;
//              localInput[5] = normal.z;
                localInput[6] = sin(position.x * PI);
//              localInput[6] = sin(position.x * PI);
                localInput[7] = sin(position.y * PI);
//              localInput[7] = sin(position.y * PI);
                localInput[8] = sin(position.z * PI);
//              localInput[8] = sin(position.z * PI);
                localInput[9] = cos(position.x * PI);
//              localInput[9] = cos(position.x * PI);
                localInput[10] = cos(position.y * PI);
//              localInput[10] = cos(position.y * PI);
                localInput[11] = cos(position.z * PI);
//              localInput[11] = cos(position.z * PI);
                localInput[12] = sin(position.x * TWO_PI);
//              localInput[12] = sin(position.x * TWO_PI);
                localInput[13] = sin(position.y * TWO_PI);
//              localInput[13] = sin(position.y * TWO_PI);
                localInput[14] = sin(position.z * TWO_PI);
//              localInput[14] = sin(position.z * TWO_PI);
                localInput[15] = 1.0;
//              localInput[15] = 1.0;

                // Forward Pass
//              // Forward Pass
                for (uint i = 0u; i < 16u; i++) {
//              for (uint i = 0u; i < 16u; i++) {
                    float sum = shared_network_weights[256u + i];
//                  float sum = shared_network_weights[256u + i];
                    for (uint j = 0u; j < 16u; j++) {
//                  for (uint j = 0u; j < 16u; j++) {
                        sum += localInput[j] * shared_network_weights[i * 16u + j];
//                      sum += localInput[j] * shared_network_weights[i * 16u + j];
                    }
//                  }
                    localHidden1[i] = max(0.0, sum);
//                  localHidden1[i] = max(0.0, sum);
                }
//              }

                for (uint i = 0u; i < 16u; i++) {
//              for (uint i = 0u; i < 16u; i++) {
                    float sum = shared_network_weights[528u + i];
//                  float sum = shared_network_weights[528u + i];
                    for (uint j = 0u; j < 16u; j++) {
//                  for (uint j = 0u; j < 16u; j++) {
                        sum += localHidden1[j] * shared_network_weights[272u + i * 16u + j];
//                      sum += localHidden1[j] * shared_network_weights[272u + i * 16u + j];
                    }
//                  }
                    localHidden2[i] = max(0.0, sum);
//                  localHidden2[i] = max(0.0, sum);
                }
//              }

                for (uint i = 0u; i < 16u; i++) {
//              for (uint i = 0u; i < 16u; i++) {
                    float sum = shared_network_weights[800u + i];
//                  float sum = shared_network_weights[800u + i];
                    for (uint j = 0u; j < 16u; j++) {
//                  for (uint j = 0u; j < 16u; j++) {
                        sum += localHidden2[j] * shared_network_weights[544u + i * 16u + j];
//                      sum += localHidden2[j] * shared_network_weights[544u + i * 16u + j];
                    }
//                  }
                    localHidden3[i] = max(0.0, sum);
//                  localHidden3[i] = max(0.0, sum);
                }
//              }

                for (uint i = 0u; i < 16u; i++) {
//              for (uint i = 0u; i < 16u; i++) {
                    float sum = shared_network_weights[1072u + i];
//                  float sum = shared_network_weights[1072u + i];
                    for (uint j = 0u; j < 16u; j++) {
//                  for (uint j = 0u; j < 16u; j++) {
                        sum += localHidden3[j] * shared_network_weights[816u + i * 16u + j];
//                      sum += localHidden3[j] * shared_network_weights[816u + i * 16u + j];
                    }
//                  }
                    localHidden4[i] = max(0.0, sum);
//                  localHidden4[i] = max(0.0, sum);
                }
//              }

                // Output Error Calculation (L2 Loss Gradient)
//              // Output Error Calculation (L2 Loss Gradient)
                for (uint i = 0u; i < 3u; i++) {
//              for (uint i = 0u; i < 3u; i++) {
                    float sum = shared_network_weights[1136u + i];
//                  float sum = shared_network_weights[1136u + i];
                    for (uint j = 0u; j < 16u; j++) {
//                  for (uint j = 0u; j < 16u; j++) {
                        sum += localHidden4[j] * shared_network_weights[1088u + i * 16u + j];
//                      sum += localHidden4[j] * shared_network_weights[1088u + i * 16u + j];
                    }
//                  }
                    localGradOutput[i] = clamp(sum - targetRadiance[int(i)], -5.0, 5.0);
//                  localGradOutput[i] = clamp(sum - targetRadiance[int(i)], -5.0, 5.0);
                }
//              }

                // Backward Pass (Backpropagation with ReLU derivatives)
//              // Backward Pass (Backpropagation with ReLU derivatives)
                for (uint i = 0u; i < 16u; i++) {
//              for (uint i = 0u; i < 16u; i++) {
                    float sum = 0.0;
//                  float sum = 0.0;
                    for (uint j = 0u; j < 3u; j++) {
//                  for (uint j = 0u; j < 3u; j++) {
                        sum += localGradOutput[j] * shared_network_weights[1088u + j * 16u + i];
//                      sum += localGradOutput[j] * shared_network_weights[1088u + j * 16u + i];
                    }
//                  }
                    localGradHidden4[i] = (localHidden4[i] > 0.0) ? sum : 0.0;
//                  localGradHidden4[i] = (localHidden4[i] > 0.0) ? sum : 0.0;
                }
//              }

                for (uint i = 0u; i < 16u; i++) {
//              for (uint i = 0u; i < 16u; i++) {
                    float sum = 0.0;
//                  float sum = 0.0;
                    for (uint j = 0u; j < 16u; j++) {
//                  for (uint j = 0u; j < 16u; j++) {
                        sum += localGradHidden4[j] * shared_network_weights[816u + j * 16u + i];
//                      sum += localGradHidden4[j] * shared_network_weights[816u + j * 16u + i];
                    }
//                  }
                    localGradHidden3[i] = (localHidden3[i] > 0.0) ? sum : 0.0;
//                  localGradHidden3[i] = (localHidden3[i] > 0.0) ? sum : 0.0;
                }
//              }

                for (uint i = 0u; i < 16u; i++) {
//              for (uint i = 0u; i < 16u; i++) {
                    float sum = 0.0;
//                  float sum = 0.0;
                    for (uint j = 0u; j < 16u; j++) {
//                  for (uint j = 0u; j < 16u; j++) {
                        sum += localGradHidden3[j] * shared_network_weights[544u + j * 16u + i];
//                      sum += localGradHidden3[j] * shared_network_weights[544u + j * 16u + i];
                    }
//                  }
                    localGradHidden2[i] = (localHidden2[i] > 0.0) ? sum : 0.0;
//                  localGradHidden2[i] = (localHidden2[i] > 0.0) ? sum : 0.0;
                }
//              }

                for (uint i = 0u; i < 16u; i++) {
//              for (uint i = 0u; i < 16u; i++) {
                    float sum = 0.0;
//                  float sum = 0.0;
                    for (uint j = 0u; j < 16u; j++) {
//                  for (uint j = 0u; j < 16u; j++) {
                        sum += localGradHidden2[j] * shared_network_weights[272u + j * 16u + i];
//                      sum += localGradHidden2[j] * shared_network_weights[272u + j * 16u + i];
                    }
//                  }
                    localGradHidden1[i] = (localHidden1[i] > 0.0) ? sum : 0.0;
//                  localGradHidden1[i] = (localHidden1[i] > 0.0) ? sum : 0.0;
                }
//              }
            }
//          }

            // Collaborative Gradient Accumulation via Shared Memory
//          // Collaborative Gradient Accumulation via Shared Memory
            float accumulatedGradients[5];
//          float accumulatedGradients[5];
            for (uint p = 0u; p < 5u; p++) {
//          for (uint p = 0u; p < 5u; p++) {
                accumulatedGradients[p] = 0.0;
//              accumulatedGradients[p] = 0.0;
            }
//          }

            for (uint chunk = 0u; chunk < 16u; chunk++) {
//          for (uint chunk = 0u; chunk < 16u; chunk++) {
                if (localIndex / 16u == chunk) {
//              if (localIndex / 16u == chunk) {
                    uint subId = localIndex % 16u;
//                  uint subId = localIndex % 16u;
                    for (uint i = 0u; i < 16u; i++) {
//                  for (uint i = 0u; i < 16u; i++) {
                        batch_input_features[subId * 16u + i] = localInput[i];
//                      batch_input_features[subId * 16u + i] = localInput[i];
                        batch_hidden_layer_1[subId * 16u + i] = localHidden1[i];
//                      batch_hidden_layer_1[subId * 16u + i] = localHidden1[i];
                        batch_hidden_layer_2[subId * 16u + i] = localHidden2[i];
//                      batch_hidden_layer_2[subId * 16u + i] = localHidden2[i];
                        batch_hidden_layer_3[subId * 16u + i] = localHidden3[i];
//                      batch_hidden_layer_3[subId * 16u + i] = localHidden3[i];
                        batch_hidden_layer_4[subId * 16u + i] = localHidden4[i];
//                      batch_hidden_layer_4[subId * 16u + i] = localHidden4[i];
                    }
//                  }
                    for (uint i = 0u; i < 3u; i++) {
//                  for (uint i = 0u; i < 3u; i++) {
                        batch_gradient_output_layer[subId * 3u + i] = localGradOutput[i];
//                      batch_gradient_output_layer[subId * 3u + i] = localGradOutput[i];
                    }
//                  }
                    for (uint i = 0u; i < 16u; i++) {
//                  for (uint i = 0u; i < 16u; i++) {
                        batch_gradient_hidden_4[subId * 16u + i] = localGradHidden4[i];
//                      batch_gradient_hidden_4[subId * 16u + i] = localGradHidden4[i];
                        batch_gradient_hidden_3[subId * 16u + i] = localGradHidden3[i];
//                      batch_gradient_hidden_3[subId * 16u + i] = localGradHidden3[i];
                        batch_gradient_hidden_2[subId * 16u + i] = localGradHidden2[i];
//                      batch_gradient_hidden_2[subId * 16u + i] = localGradHidden2[i];
                        batch_gradient_hidden_1[subId * 16u + i] = localGradHidden1[i];
//                      batch_gradient_hidden_1[subId * 16u + i] = localGradHidden1[i];
                    }
//                  }
                }
//              }
                barrier();
//              barrier();

                uint activeBatchSize = isActiveThread ? 16u : 0u;
//              uint activeBatchSize = isActiveThread ? 16u : 0u;
                if (activeBatchSize > 0u) {
//              if (activeBatchSize > 0u) {
                    for (uint p = 0u; p < 5u; p++) {
//                  for (uint p = 0u; p < 5u; p++) {
                        uint parameterIndex = localIndex + p * 256u;
//                      uint parameterIndex = localIndex + p * 256u;
                        if (parameterIndex < 1139u) {
//                      if (parameterIndex < 1139u) {
                            float paramGradient = 0.0;
//                          float paramGradient = 0.0;
                            if (parameterIndex < 256u) {
//                          if (parameterIndex < 256u) {
                                uint row = parameterIndex / 16u;
//                              uint row = parameterIndex / 16u;
                                uint col = parameterIndex % 16u;
//                              uint col = parameterIndex % 16u;
                                for (uint b = 0u; b < activeBatchSize; b++) {
//                              for (uint b = 0u; b < activeBatchSize; b++) {
                                    paramGradient += batch_gradient_hidden_1[b * 16u + row] * batch_input_features[b * 16u + col];
//                                  paramGradient += batch_gradient_hidden_1[b * 16u + row] * batch_input_features[b * 16u + col];
                                }
//                              }
                            } else if (parameterIndex < 272u) {
//                          } else if (parameterIndex < 272u) {
                                uint row = parameterIndex - 256u;
//                              uint row = parameterIndex - 256u;
                                for (uint b = 0u; b < activeBatchSize; b++) {
//                              for (uint b = 0u; b < activeBatchSize; b++) {
                                    paramGradient += batch_gradient_hidden_1[b * 16u + row];
//                                  paramGradient += batch_gradient_hidden_1[b * 16u + row];
                                }
//                              }
                            } else if (parameterIndex < 528u) {
//                          } else if (parameterIndex < 528u) {
                                uint row = (parameterIndex - 272u) / 16u;
//                              uint row = (parameterIndex - 272u) / 16u;
                                uint col = (parameterIndex - 272u) % 16u;
//                              uint col = (parameterIndex - 272u) % 16u;
                                for (uint b = 0u; b < activeBatchSize; b++) {
//                              for (uint b = 0u; b < activeBatchSize; b++) {
                                    paramGradient += batch_gradient_hidden_2[b * 16u + row] * batch_hidden_layer_1[b * 16u + col];
//                                  paramGradient += batch_gradient_hidden_2[b * 16u + row] * batch_hidden_layer_1[b * 16u + col];
                                }
//                              }
                            } else if (parameterIndex < 544u) {
//                          } else if (parameterIndex < 544u) {
                                uint row = parameterIndex - 528u;
//                              uint row = parameterIndex - 528u;
                                for (uint b = 0u; b < activeBatchSize; b++) {
//                              for (uint b = 0u; b < activeBatchSize; b++) {
                                    paramGradient += batch_gradient_hidden_2[b * 16u + row];
//                                  paramGradient += batch_gradient_hidden_2[b * 16u + row];
                                }
//                              }
                            } else if (parameterIndex < 800u) {
//                          } else if (parameterIndex < 800u) {
                                uint row = (parameterIndex - 544u) / 16u;
//                              uint row = (parameterIndex - 544u) / 16u;
                                uint col = (parameterIndex - 544u) % 16u;
//                              uint col = (parameterIndex - 544u) % 16u;
                                for (uint b = 0u; b < activeBatchSize; b++) {
//                              for (uint b = 0u; b < activeBatchSize; b++) {
                                    paramGradient += batch_gradient_hidden_3[b * 16u + row] * batch_hidden_layer_2[b * 16u + col];
//                                  paramGradient += batch_gradient_hidden_3[b * 16u + row] * batch_hidden_layer_2[b * 16u + col];
                                }
//                              }
                            } else if (parameterIndex < 816u) {
//                          } else if (parameterIndex < 816u) {
                                uint row = parameterIndex - 800u;
//                              uint row = parameterIndex - 800u;
                                for (uint b = 0u; b < activeBatchSize; b++) {
//                              for (uint b = 0u; b < activeBatchSize; b++) {
                                    paramGradient += batch_gradient_hidden_3[b * 16u + row];
//                                  paramGradient += batch_gradient_hidden_3[b * 16u + row];
                                }
//                              }
                            } else if (parameterIndex < 1072u) {
//                          } else if (parameterIndex < 1072u) {
                                uint row = (parameterIndex - 816u) / 16u;
//                              uint row = (parameterIndex - 816u) / 16u;
                                uint col = (parameterIndex - 816u) % 16u;
//                              uint col = (parameterIndex - 816u) % 16u;
                                for (uint b = 0u; b < activeBatchSize; b++) {
//                              for (uint b = 0u; b < activeBatchSize; b++) {
                                    paramGradient += batch_gradient_hidden_4[b * 16u + row] * batch_hidden_layer_3[b * 16u + col];
//                                  paramGradient += batch_gradient_hidden_4[b * 16u + row] * batch_hidden_layer_3[b * 16u + col];
                                }
//                              }
                            } else if (parameterIndex < 1088u) {
//                          } else if (parameterIndex < 1088u) {
                                uint row = parameterIndex - 1072u;
//                              uint row = parameterIndex - 1072u;
                                for (uint b = 0u; b < activeBatchSize; b++) {
//                              for (uint b = 0u; b < activeBatchSize; b++) {
                                    paramGradient += batch_gradient_hidden_4[b * 16u + row];
//                                  paramGradient += batch_gradient_hidden_4[b * 16u + row];
                                }
//                              }
                            } else if (parameterIndex < 1136u) {
//                          } else if (parameterIndex < 1136u) {
                                uint row = (parameterIndex - 1088u) / 16u;
//                              uint row = (parameterIndex - 1088u) / 16u;
                                uint col = (parameterIndex - 1088u) % 16u;
//                              uint col = (parameterIndex - 1088u) % 16u;
                                for (uint b = 0u; b < activeBatchSize; b++) {
//                              for (uint b = 0u; b < activeBatchSize; b++) {
                                    paramGradient += batch_gradient_output_layer[b * 3u + row] * batch_hidden_layer_4[b * 16u + col];
//                                  paramGradient += batch_gradient_output_layer[b * 3u + row] * batch_hidden_layer_4[b * 16u + col];
                                }
//                              }
                            } else {
//                          } else {
                                uint row = parameterIndex - 1136u;
//                              uint row = parameterIndex - 1136u;
                                for (uint b = 0u; b < activeBatchSize; b++) {
//                              for (uint b = 0u; b < activeBatchSize; b++) {
                                    paramGradient += batch_gradient_output_layer[b * 3u + row];
//                                  paramGradient += batch_gradient_output_layer[b * 3u + row];
                                }
//                              }
                            }
//                          }
                            accumulatedGradients[p] += paramGradient;
//                          accumulatedGradients[p] += paramGradient;
                        }
//                      }
                    }
//                  }
                }
//              }
                barrier();
//              barrier();
            }
//          }

            // Apply Adam Optimizer step for all aggregated parameters
//          // Apply Adam Optimizer step for all aggregated parameters
            uint totalBatchSize = isActiveThread ? 256u : 0u;
//          uint totalBatchSize = isActiveThread ? 256u : 0u;
            if (totalBatchSize > 0u) {
//          if (totalBatchSize > 0u) {
                float globalTrainingStep = min(float(uTrainingStep * 32u + step) + 1.0, 50000.0);
//              float globalTrainingStep = min(float(uTrainingStep * 32u + step) + 1.0, 50000.0);

                for (uint p = 0u; p < 5u; p++) {
//              for (uint p = 0u; p < 5u; p++) {
                    uint parameterIndex = localIndex + p * 256u;
//                  uint parameterIndex = localIndex + p * 256u;
                    if (parameterIndex < 1139u) {
//                  if (parameterIndex < 1139u) {
                        float averagedGradient = accumulatedGradients[p] / float(totalBatchSize);
//                      float averagedGradient = accumulatedGradients[p] / float(totalBatchSize);
                        averagedGradient = clamp(averagedGradient, -1.0, 1.0);
//                      averagedGradient = clamp(averagedGradient, -1.0, 1.0);

                        uint momentumIndex = 2278u + parameterIndex;
//                      uint momentumIndex = 2278u + parameterIndex;
                        uint velocityIndex = 3417u + parameterIndex;
//                      uint velocityIndex = 3417u + parameterIndex;

                        float momentum = neuralNetworkWeights[momentumIndex] * 0.9 + 0.1 * averagedGradient;
//                      float momentum = neuralNetworkWeights[momentumIndex] * 0.9 + 0.1 * averagedGradient;
                        float velocity = neuralNetworkWeights[velocityIndex] * 0.99 + 0.01 * averagedGradient * averagedGradient;
//                      float velocity = neuralNetworkWeights[velocityIndex] * 0.99 + 0.01 * averagedGradient * averagedGradient;

                        neuralNetworkWeights[momentumIndex] = momentum;
//                      neuralNetworkWeights[momentumIndex] = momentum;
                        neuralNetworkWeights[velocityIndex] = velocity;
//                      neuralNetworkWeights[velocityIndex] = velocity;

                        float biasCorrection1 = 1.0 - pow(0.9, globalTrainingStep);
//                      float biasCorrection1 = 1.0 - pow(0.9, globalTrainingStep);
                        float biasCorrection2 = 1.0 - pow(0.99, globalTrainingStep);
//                      float biasCorrection2 = 1.0 - pow(0.99, globalTrainingStep);

                        float learningRate = 0.002;
//                      float learningRate = 0.002;
                        float update = learningRate * (momentum / biasCorrection1) / (sqrt(velocity / biasCorrection2) + 1.0e-8);
//                      float update = learningRate * (momentum / biasCorrection1) / (sqrt(velocity / biasCorrection2) + 1.0e-8);

                        shared_network_weights[parameterIndex] -= update;
//                      shared_network_weights[parameterIndex] -= update;
                    }
//                  }
                }
//              }
            }
//          }
            barrier();
//          barrier();
        }
//      }

        // Store trained active weights and update Exponential Moving Average (EMA) weights
//      // Store trained active weights and update Exponential Moving Average (EMA) weights
        for (uint i = localIndex; i < 1139u; i += 256u) {
//      for (uint i = localIndex; i < 1139u; i += 256u) {
            neuralNetworkWeights[i] = shared_network_weights[i];
//          neuralNetworkWeights[i] = shared_network_weights[i];
            if (isActiveThread) {
//          if (isActiveThread) {
                neuralNetworkWeights[1139u + i] = 0.95 * neuralNetworkWeights[1139u + i] + 0.05 * shared_network_weights[i];
//              neuralNetworkWeights[1139u + i] = 0.95 * neuralNetworkWeights[1139u + i] + 0.05 * shared_network_weights[i];
            }
//          }
        }
//      }

        if (localIndex == 0u) {
//      if (localIndex == 0u) {
            uTrainingRecordCount = 0u;
//          uTrainingRecordCount = 0u;
        }
//      }
    }
//  }
