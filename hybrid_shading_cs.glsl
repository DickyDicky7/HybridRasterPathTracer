    #version 430
//  #version 430

    layout(local_size_x = 16, local_size_y = 16) in;
//  layout(local_size_x = 16, local_size_y = 16) in;

    layout(binding = 0, rgba32f) uniform image2D textureOutput;
//  layout(binding = 0, rgba32f) uniform image2D textureOutput;
    layout(binding = 1, rgba32f) uniform image2D textureGeometryGlobalPosition;
//  layout(binding = 1, rgba32f) uniform image2D textureGeometryGlobalPosition;
    layout(binding = 2, rgba32f) uniform image2D textureGeometryGlobalNormal;
//  layout(binding = 2, rgba32f) uniform image2D textureGeometryGlobalNormal;
    layout(binding = 3, rgba32f) uniform image2D textureGeometryAlbedo;
//  layout(binding = 3, rgba32f) uniform image2D textureGeometryAlbedo;

    uniform float uTime;
//  uniform float uTime;
    uniform vec3 uPointLight001GlobalPosition;
//  uniform vec3 uPointLight001GlobalPosition;
    uniform vec3 uCameraGlobalPosition;
//  uniform vec3 uCameraGlobalPosition;

    void main() {
//  void main() {
        ivec2 pixelCoordinates = ivec2(gl_GlobalInvocationID.xy);
//      ivec2 pixelCoordinates = ivec2(gl_GlobalInvocationID.xy);
        ivec2 dimensions = imageSize(textureOutput);
//      ivec2 dimensions = imageSize(textureOutput);

        if (pixelCoordinates.x >= dimensions.x || pixelCoordinates.y >= dimensions.y) {
//      if (pixelCoordinates.x >= dimensions.x || pixelCoordinates.y >= dimensions.y) {
            return;
//          return;
        }
//      }

        // 1. Read G-Buffer
        // 1. Read G-Buffer
        // We used .w = 1.0 for valid geometry in the fragment shader
        // We used .w = 1.0 for valid geometry in the fragment shader
        vec4 sampleGlobalPosition = imageLoad(textureGeometryGlobalPosition, pixelCoordinates);
//      vec4 sampleGlobalPosition = imageLoad(textureGeometryGlobalPosition, pixelCoordinates);
        vec4 sampleGlobalNormal = imageLoad(textureGeometryGlobalNormal, pixelCoordinates);
//      vec4 sampleGlobalNormal = imageLoad(textureGeometryGlobalNormal, pixelCoordinates);
        vec4 sampleAlbedo = imageLoad(textureGeometryAlbedo, pixelCoordinates);
//      vec4 sampleAlbedo = imageLoad(textureGeometryAlbedo, pixelCoordinates);

        // 2. Background Check
        // 2. Background Check
        if (sampleGlobalPosition.w == 0.0) {
//      if (sampleGlobalPosition.w == 0.0) {
            // Simple gradient background
            // Simple gradient background
            float ratio = float(pixelCoordinates.y) / float(dimensions.y);
//          float ratio = float(pixelCoordinates.y) / float(dimensions.y);
            vec3 background = mix(vec3(0.7, 0.8, 0.9), vec3(0.4, 0.7, 1.0), ratio);
//          vec3 background = mix(vec3(0.7, 0.8, 0.9), vec3(0.4, 0.7, 1.0), ratio);
            imageStore(textureOutput, pixelCoordinates, vec4(background, 1.0));
//          imageStore(textureOutput, pixelCoordinates, vec4(background, 1.0));
            return;
//          return;
        }
//      }

        vec3 P = sampleGlobalPosition.xyz;
//      vec3 P = sampleGlobalPosition.xyz;
        vec3 N = normalize(sampleGlobalNormal.xyz);
//      vec3 N = normalize(sampleGlobalNormal.xyz);
        vec3 albedo = sampleAlbedo.rgb;
//      vec3 albedo = sampleAlbedo.rgb;

        // 3. Lighting Calculation (Hybrid / Deferred)
        // 3. Lighting Calculation (Hybrid / Deferred)

        vec3 L = normalize(uPointLight001GlobalPosition - P);
//      vec3 L = normalize(uPointLight001GlobalPosition - P);
        vec3 V = normalize(uCameraGlobalPosition - P);
//      vec3 V = normalize(uCameraGlobalPosition - P);

        // Diffuse
        // Diffuse
        float diff = max(dot(N, L), 0.0);
//      float diff = max(dot(N, L), 0.0);

        // Specular (Blinn-Phong)
        // Specular (Blinn-Phong)
        vec3 H = normalize(L + V);
//      vec3 H = normalize(L + V);
        float spec = pow(max(dot(N, H), 0.0), 32.0);
//      float spec = pow(max(dot(N, H), 0.0), 32.0);

        // Shadow Ray (Mockup for Path Tracing)
        // Shadow Ray (Mockup for Path Tracing)
        // In a full hybrid engine, you would traverse an acceleration structure (TLAS) here.
        // In a full hybrid engine, you would traverse an acceleration structure (TLAS) here.
        // For this example, we'll just check if the normal faces the light to simulate self-shadowing basics
        // For this example, we'll just check if the normal faces the light to simulate self-shadowing basics
        float shadow = 1.0;
//      float shadow = 1.0;
        if (dot(N, L) < 0.0) {
//      if (dot(N, L) < 0.0) {
            shadow = 0.1;
//          shadow = 0.1;
        }
//      }

        // Ambient
        // Ambient
        vec3 ambient = vec3(0.03) * albedo;
//      vec3 ambient = vec3(0.03) * albedo;

        // Combine
        // Combine
        vec3 finalColor = ambient + (diff * albedo + spec * vec3(1.0)) * shadow;
//      vec3 finalColor = ambient + (diff * albedo + spec * vec3(1.0)) * shadow;

        // 4. "Path Tracing" elements (optional expansion)
        // 4. "Path Tracing" elements (optional expansion)
        // Here you could shoot secondary rays for reflection/GI using P and N
        // Here you could shoot secondary rays for reflection/GI using P and N
        // vec3 reflectionDirection = reflect(-V, N);
        // vec3 reflectionDirection = reflect(-V, N);
        // ... Trace(P + N * 0.001, reflectionDirection) ...
        // ... Trace(P + N * 0.001, reflectionDirection) ...

        imageStore(textureOutput, pixelCoordinates, vec4(finalColor, 1.0));
//      imageStore(textureOutput, pixelCoordinates, vec4(finalColor, 1.0));
    }
//  }
