    #version 430
//  #version 430

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
    layout(local_size_x = 8, local_size_y = 8, local_size_z = 8) in;
//  layout(local_size_x = 8, local_size_y = 8, local_size_z = 8) in;

    layout(binding = 0, rgba8) uniform writeonly image3D textureCloudNoise;
//  layout(binding = 0, rgba8) uniform writeonly image3D textureCloudNoise;

    uniform int uNoiseSize;
//  uniform int uNoiseSize;

    // 3D PCG hash (Jarzynski & Olano 2020): three independent 32-bit outputs per lattice cell.
//  // 3D PCG hash (Jarzynski & Olano 2020): three independent 32-bit outputs per lattice cell.
    uvec3 pcg3d(uvec3 value) {
//  uvec3 pcg3d(uvec3 value) {
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
    vec3 hashLatticeCell(ivec3 cell, int period, uint seed) {
//  vec3 hashLatticeCell(ivec3 cell, int period, uint seed) {
        uvec3 wrappedCell = uvec3((cell % period + period) % period);
//      uvec3 wrappedCell = uvec3((cell % period + period) % period);
        return vec3(pcg3d(wrappedCell + uvec3(seed, seed * 7u, seed * 13u))) / 4294967295.0;
//      return vec3(pcg3d(wrappedCell + uvec3(seed, seed * 7u, seed * 13u))) / 4294967295.0;
    }
//  }

    // Inverted cellular noise: 1 at a feature point, falling to 0 one cell away.
//  // Inverted cellular noise: 1 at a feature point, falling to 0 one cell away.
    float worleyTileable(vec3 tileCoordinates, int cellsPerTile, uint seed) {
//  float worleyTileable(vec3 tileCoordinates, int cellsPerTile, uint seed) {
        vec3 cellSpacePosition = tileCoordinates * float(cellsPerTile);
//      vec3 cellSpacePosition = tileCoordinates * float(cellsPerTile);
        ivec3 baseCell = ivec3(floor(cellSpacePosition));
//      ivec3 baseCell = ivec3(floor(cellSpacePosition));
        vec3 cellFraction = fract(cellSpacePosition);
//      vec3 cellFraction = fract(cellSpacePosition);
        float nearestDistanceSquared = 1.0e10;
//      float nearestDistanceSquared = 1.0e10;
        for (int offsetZ = -1; offsetZ <= 1; offsetZ++) {
//      for (int offsetZ = -1; offsetZ <= 1; offsetZ++) {
            for (int offsetY = -1; offsetY <= 1; offsetY++) {
//          for (int offsetY = -1; offsetY <= 1; offsetY++) {
                for (int offsetX = -1; offsetX <= 1; offsetX++) {
//              for (int offsetX = -1; offsetX <= 1; offsetX++) {
                    ivec3 cellOffset = ivec3(offsetX, offsetY, offsetZ);
//                  ivec3 cellOffset = ivec3(offsetX, offsetY, offsetZ);
                    vec3 featurePoint = vec3(cellOffset) + hashLatticeCell(baseCell + cellOffset, cellsPerTile, seed);
//                  vec3 featurePoint = vec3(cellOffset) + hashLatticeCell(baseCell + cellOffset, cellsPerTile, seed);
                    vec3 toFeature = featurePoint - cellFraction;
//                  vec3 toFeature = featurePoint - cellFraction;
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

    float worleyFbmTileable(vec3 tileCoordinates, int cellsPerTile, uint seed) {
//  float worleyFbmTileable(vec3 tileCoordinates, int cellsPerTile, uint seed) {
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
    float perlinTileable(vec3 tileCoordinates, int period, uint seed) {
//  float perlinTileable(vec3 tileCoordinates, int period, uint seed) {
        vec3 latticePosition = tileCoordinates * float(period);
//      vec3 latticePosition = tileCoordinates * float(period);
        ivec3 baseCell = ivec3(floor(latticePosition));
//      ivec3 baseCell = ivec3(floor(latticePosition));
        vec3 cellFraction = fract(latticePosition);
//      vec3 cellFraction = fract(latticePosition);
        vec3 fade = cellFraction * cellFraction * cellFraction * (cellFraction * (cellFraction * 6.0 - 15.0) + 10.0);
//      vec3 fade = cellFraction * cellFraction * cellFraction * (cellFraction * (cellFraction * 6.0 - 15.0) + 10.0);
        float cornerValues[8];
//      float cornerValues[8];
        for (int corner = 0; corner < 8; corner++) {
//      for (int corner = 0; corner < 8; corner++) {
            ivec3 cornerOffset = ivec3(corner & 1, (corner >> 1) & 1, (corner >> 2) & 1);
//          ivec3 cornerOffset = ivec3(corner & 1, (corner >> 1) & 1, (corner >> 2) & 1);
            vec3 gradient = normalize(hashLatticeCell(baseCell + cornerOffset, period, seed) * 2.0 - 1.0 + 1.0e-5);
//          vec3 gradient = normalize(hashLatticeCell(baseCell + cornerOffset, period, seed) * 2.0 - 1.0 + 1.0e-5);
            cornerValues[corner] = dot(gradient, cellFraction - vec3(cornerOffset));
//          cornerValues[corner] = dot(gradient, cellFraction - vec3(cornerOffset));
        }
//      }
        float edgeX00 = mix(cornerValues[0], cornerValues[1], fade.x);
//      float edgeX00 = mix(cornerValues[0], cornerValues[1], fade.x);
        float edgeX10 = mix(cornerValues[2], cornerValues[3], fade.x);
//      float edgeX10 = mix(cornerValues[2], cornerValues[3], fade.x);
        float edgeX01 = mix(cornerValues[4], cornerValues[5], fade.x);
//      float edgeX01 = mix(cornerValues[4], cornerValues[5], fade.x);
        float edgeX11 = mix(cornerValues[6], cornerValues[7], fade.x);
//      float edgeX11 = mix(cornerValues[6], cornerValues[7], fade.x);
        return mix(mix(edgeX00, edgeX10, fade.y), mix(edgeX01, edgeX11, fade.y), fade.z);
//      return mix(mix(edgeX00, edgeX10, fade.y), mix(edgeX01, edgeX11, fade.y), fade.z);
    }
//  }

    float perlinFbmTileable(vec3 tileCoordinates, int basePeriod, uint seed) {
//  float perlinFbmTileable(vec3 tileCoordinates, int basePeriod, uint seed) {
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

    void main() {
//  void main() {
        ivec3 voxelCoordinates = ivec3(gl_GlobalInvocationID.xyz);
//      ivec3 voxelCoordinates = ivec3(gl_GlobalInvocationID.xyz);
        if (any(greaterThanEqual(voxelCoordinates, ivec3(uNoiseSize)))) {
//      if (any(greaterThanEqual(voxelCoordinates, ivec3(uNoiseSize)))) {
            return;
//          return;
        }
//      }
        vec3 tileCoordinates = (vec3(voxelCoordinates) + 0.5) / float(uNoiseSize);
//      vec3 tileCoordinates = (vec3(voxelCoordinates) + 0.5) / float(uNoiseSize);

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

        vec4 noiseTexel = vec4(
//      vec4 noiseTexel = vec4(
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
        imageStore(textureCloudNoise, voxelCoordinates, noiseTexel);
//      imageStore(textureCloudNoise, voxelCoordinates, noiseTexel);
    }
//  }
