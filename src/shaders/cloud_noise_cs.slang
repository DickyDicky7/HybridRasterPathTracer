    // ============================================================================
//  // ============================================================================
    // Tileable 3D cloud noise generator (runs once, at renderer start-up).
//  // Tileable 3D cloud noise generator (runs once, at renderer start-up).
    //
//  //
    // Fills the RGBA8 noise volume every volumetric pass samples with hardware trilinear filtering,
//  // Fills the RGBA8 noise volume every volumetric pass samples with hardware trilinear filtering,
    // so no ray-march step ever evaluates procedural noise on the fly:
//  // so no ray-march step ever evaluates procedural noise on the fly:
    //   R: Perlin-Worley               -> billowy base shape
//  //   R: Perlin-Worley               -> billowy base shape
    //   G: Worley FBM,  4 cells / tile -> low-frequency erosion
//  //   G: Worley FBM,  4 cells / tile -> low-frequency erosion
    //   B: Worley FBM,  8 cells / tile
//  //   B: Worley FBM,  8 cells / tile
    //   A: Worley FBM, 16 cells / tile -> high-frequency edge detail
//  //   A: Worley FBM, 16 cells / tile -> high-frequency edge detail
    // Every octave's period is an integer number of cells per tile, so the volume wraps seamlessly
//  // Every octave's period is an integer number of cells per tile, so the volume wraps seamlessly
    // under GL_REPEAT on all three axes. Channel layout follows Schneider, "The Real-Time Volumetric
//  // under GL_REPEAT on all three axes. Channel layout follows Schneider, "The Real-Time Volumetric
    // Cloudscapes of Horizon Zero Dawn" (SIGGRAPH 2015).
//  // Cloudscapes of Horizon Zero Dawn" (SIGGRAPH 2015).
    // ============================================================================
//  // ============================================================================

    [format("rgba8")] WTexture3D<float4> textureCloudNoise;
//  [format("rgba8")] WTexture3D<float4> textureCloudNoise;

    uniform int uNoiseSize;
//  uniform int uNoiseSize;

    // 3D PCG hash (Jarzynski & Olano 2020): three independent 32-bit outputs per lattice cell.
//  // 3D PCG hash (Jarzynski & Olano 2020): three independent 32-bit outputs per lattice cell.
    uint3 pcg3d(uint3 value) {
//  uint3 pcg3d(uint3 value) {
        value = value * 1664525u + 1013904223u;
//      value = value * 1664525u + 1013904223u;
        value.x += value.y * value.z;
//      value.x += value.y * value.z;
        value.y += value.z * value.x;
//      value.y += value.z * value.x;
        value.z += value.x * value.y;
//      value.z += value.x * value.y;
        value ^= value >> 16u;
//      value ^= value >> 16u;
        value.x += value.y * value.z;
//      value.x += value.y * value.z;
        value.y += value.z * value.x;
//      value.y += value.z * value.x;
        value.z += value.x * value.y;
//      value.z += value.x * value.y;
        return value;
//      return value;
    }
//  }

    // Wrapping the lattice cell by the octave's period before hashing is what makes the octave tile.
//  // Wrapping the lattice cell by the octave's period before hashing is what makes the octave tile.
    float3 hashLatticeCell(int3 cell, int period, uint seed) {
//  float3 hashLatticeCell(int3 cell, int period, uint seed) {
        uint3 wrappedCell = uint3((cell % period + period) % period);
//      uint3 wrappedCell = uint3((cell % period + period) % period);
        return float3(pcg3d(wrappedCell + uint3(seed, seed * 7u, seed * 13u))) / 4294967295.0;
//      return float3(pcg3d(wrappedCell + uint3(seed, seed * 7u, seed * 13u))) / 4294967295.0;
    }
//  }

    // Inverted cellular noise: 1 at a feature point, falling to 0 one cell away.
//  // Inverted cellular noise: 1 at a feature point, falling to 0 one cell away.
    float worleyTileable(float3 tileCoordinates, int cellsPerTile, uint seed) {
//  float worleyTileable(float3 tileCoordinates, int cellsPerTile, uint seed) {
        float3 cellSpacePosition = tileCoordinates * float(cellsPerTile);
//      float3 cellSpacePosition = tileCoordinates * float(cellsPerTile);
        int3 baseCell = int3(floor(cellSpacePosition));
//      int3 baseCell = int3(floor(cellSpacePosition));
        float3 cellFraction = frac(cellSpacePosition);
//      float3 cellFraction = frac(cellSpacePosition);
        float nearestDistanceSquared = 1.0e10;
//      float nearestDistanceSquared = 1.0e10;
        for (int offsetZ = -1; offsetZ <= 1; offsetZ++) {
//      for (int offsetZ = -1; offsetZ <= 1; offsetZ++) {
            for (int offsetY = -1; offsetY <= 1; offsetY++) {
//          for (int offsetY = -1; offsetY <= 1; offsetY++) {
                for (int offsetX = -1; offsetX <= 1; offsetX++) {
//              for (int offsetX = -1; offsetX <= 1; offsetX++) {
                    int3 cellOffset = int3(offsetX, offsetY, offsetZ);
//                  int3 cellOffset = int3(offsetX, offsetY, offsetZ);
                    float3 featurePoint = float3(cellOffset) + hashLatticeCell(baseCell + cellOffset, cellsPerTile, seed);
//                  float3 featurePoint = float3(cellOffset) + hashLatticeCell(baseCell + cellOffset, cellsPerTile, seed);
                    float3 toFeature = featurePoint - cellFraction;
//                  float3 toFeature = featurePoint - cellFraction;
                    nearestDistanceSquared = min(nearestDistanceSquared, dot(toFeature, toFeature));
//                  nearestDistanceSquared = min(nearestDistanceSquared, dot(toFeature, toFeature));
                }
//              }
            }
//          }
        }
//      }
        return 1.0 - clamp(sqrt(nearestDistanceSquared), 0.0, 1.0);
//      return 1.0 - clamp(sqrt(nearestDistanceSquared), 0.0, 1.0);
    }
//  }

    float worleyFbmTileable(float3 tileCoordinates, int cellsPerTile, uint seed) {
//  float worleyFbmTileable(float3 tileCoordinates, int cellsPerTile, uint seed) {
        return worleyTileable(tileCoordinates, cellsPerTile, seed) * 0.625
//      return worleyTileable(tileCoordinates, cellsPerTile, seed) * 0.625
             + worleyTileable(tileCoordinates, cellsPerTile * 2, seed + 17u) * 0.25
//           + worleyTileable(tileCoordinates, cellsPerTile * 2, seed + 17u) * 0.25
             + worleyTileable(tileCoordinates, cellsPerTile * 4, seed + 43u) * 0.125;
//           + worleyTileable(tileCoordinates, cellsPerTile * 4, seed + 43u) * 0.125;
    }
//  }

    // Gradient (Perlin) noise with quintic fade, returning roughly [-1, 1].
//  // Gradient (Perlin) noise with quintic fade, returning roughly [-1, 1].
    float perlinTileable(float3 tileCoordinates, int period, uint seed) {
//  float perlinTileable(float3 tileCoordinates, int period, uint seed) {
        float3 latticePosition = tileCoordinates * float(period);
//      float3 latticePosition = tileCoordinates * float(period);
        int3 baseCell = int3(floor(latticePosition));
//      int3 baseCell = int3(floor(latticePosition));
        float3 cellFraction = frac(latticePosition);
//      float3 cellFraction = frac(latticePosition);
        float3 fade = cellFraction * cellFraction * cellFraction * (cellFraction * (cellFraction * 6.0 - 15.0) + 10.0);
//      float3 fade = cellFraction * cellFraction * cellFraction * (cellFraction * (cellFraction * 6.0 - 15.0) + 10.0);
        float cornerValues[8];
//      float cornerValues[8];
        for (int corner = 0; corner < 8; corner++) {
//      for (int corner = 0; corner < 8; corner++) {
            int3 cornerOffset = int3(corner & 1, (corner >> 1) & 1, (corner >> 2) & 1);
//          int3 cornerOffset = int3(corner & 1, (corner >> 1) & 1, (corner >> 2) & 1);
            float3 gradient = normalize(hashLatticeCell(baseCell + cornerOffset, period, seed) * 2.0 - 1.0 + 1.0e-5);
//          float3 gradient = normalize(hashLatticeCell(baseCell + cornerOffset, period, seed) * 2.0 - 1.0 + 1.0e-5);
            cornerValues[corner] = dot(gradient, cellFraction - float3(cornerOffset));
//          cornerValues[corner] = dot(gradient, cellFraction - float3(cornerOffset));
        }
//      }
        float edgeX00 = lerp(cornerValues[0], cornerValues[1], fade.x);
//      float edgeX00 = lerp(cornerValues[0], cornerValues[1], fade.x);
        float edgeX10 = lerp(cornerValues[2], cornerValues[3], fade.x);
//      float edgeX10 = lerp(cornerValues[2], cornerValues[3], fade.x);
        float edgeX01 = lerp(cornerValues[4], cornerValues[5], fade.x);
//      float edgeX01 = lerp(cornerValues[4], cornerValues[5], fade.x);
        float edgeX11 = lerp(cornerValues[6], cornerValues[7], fade.x);
//      float edgeX11 = lerp(cornerValues[6], cornerValues[7], fade.x);
        return lerp(lerp(edgeX00, edgeX10, fade.y), lerp(edgeX01, edgeX11, fade.y), fade.z);
//      return lerp(lerp(edgeX00, edgeX10, fade.y), lerp(edgeX01, edgeX11, fade.y), fade.z);
    }
//  }

    float perlinFbmTileable(float3 tileCoordinates, int basePeriod, uint seed) {
//  float perlinFbmTileable(float3 tileCoordinates, int basePeriod, uint seed) {
        float noiseSum = 0.0;
//      float noiseSum = 0.0;
        float amplitude = 1.0;
//      float amplitude = 1.0;
        float amplitudeSum = 0.0;
//      float amplitudeSum = 0.0;
        int period = basePeriod;
//      int period = basePeriod;
        for (int octave = 0; octave < 5; octave++) {
//      for (int octave = 0; octave < 5; octave++) {
            noiseSum += perlinTileable(tileCoordinates, period, seed + uint(octave) * 101u) * amplitude;
//          noiseSum += perlinTileable(tileCoordinates, period, seed + uint(octave) * 101u) * amplitude;
            amplitudeSum += amplitude;
//          amplitudeSum += amplitude;
            amplitude *= 0.5;
//          amplitude *= 0.5;
            period *= 2;
//          period *= 2;
        }
//      }
        return noiseSum / amplitudeSum;
//      return noiseSum / amplitudeSum;
    }
//  }

    [shader("compute")]
//  [shader("compute")]
    [numthreads(8, 8, 8)]
//  [numthreads(8, 8, 8)]
    void computeMain(uint3 globalInvocationID : SV_DispatchThreadID, uint localInvocationIndex : SV_GroupIndex) {
//  void computeMain(uint3 globalInvocationID : SV_DispatchThreadID, uint localInvocationIndex : SV_GroupIndex) {
        int3 voxelCoordinates = int3(globalInvocationID.xyz);
//      int3 voxelCoordinates = int3(globalInvocationID.xyz);
        if (any(voxelCoordinates >= int3(uNoiseSize))) {
//      if (any(voxelCoordinates >= int3(uNoiseSize))) {
            return;
//          return;
        }
//      }
        float3 tileCoordinates = (float3(voxelCoordinates) + 0.5) / float(uNoiseSize);
//      float3 tileCoordinates = (float3(voxelCoordinates) + 0.5) / float(uNoiseSize);

        // Perlin FBM remapped into [0, 1] with a little extra contrast, then floored by the Worley FBM
//      // Perlin FBM remapped into [0, 1] with a little extra contrast, then floored by the Worley FBM
        // (Schneider's remap(perlin, 0, 1, worley, 1)): Worley carves the billows, Perlin keeps them connected.
//      // (Schneider's remap(perlin, 0, 1, worley, 1)): Worley carves the billows, Perlin keeps them connected.
        float perlinNoise = clamp(perlinFbmTileable(tileCoordinates, 4, 7u) * 0.75 + 0.5, 0.0, 1.0);
//      float perlinNoise = clamp(perlinFbmTileable(tileCoordinates, 4, 7u) * 0.75 + 0.5, 0.0, 1.0);
        float baseWorley = worleyFbmTileable(tileCoordinates, 4, 3u);
//      float baseWorley = worleyFbmTileable(tileCoordinates, 4, 3u);
        float perlinWorley = clamp(baseWorley + perlinNoise * (1.0 - baseWorley), 0.0, 1.0);
//      float perlinWorley = clamp(baseWorley + perlinNoise * (1.0 - baseWorley), 0.0, 1.0);

        float4 noiseTexel = float4(
//      float4 noiseTexel = float4(
            perlinWorley,
//          perlinWorley,
            baseWorley,
//          baseWorley,
            worleyFbmTileable(tileCoordinates, 8, 29u),
//          worleyFbmTileable(tileCoordinates, 8, 29u),
            worleyFbmTileable(tileCoordinates, 16, 61u)
//          worleyFbmTileable(tileCoordinates, 16, 61u)
        );
//      );
        textureCloudNoise.Store(voxelCoordinates, noiseTexel);
//      textureCloudNoise.Store(voxelCoordinates, noiseTexel);
    }
//  }
