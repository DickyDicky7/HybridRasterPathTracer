<div align="center"><h1>HybridRasterPathTracer</h1></div>
<!--
<div align="center"><h1>HybridRasterPathTracer</h1></div>
-->

<div align="center"><p>An experimental real-time hybrid renderer leveraging a traditional rasterization pipeline for primary visibility alongside a Monte Carlo path-tracing compute pipeline for global illumination and advanced shading.</p></div>
<!--
<div align="center"><p>An experimental real-time hybrid renderer leveraging a traditional rasterization pipeline for primary visibility alongside a Monte Carlo path-tracing compute pipeline for global illumination and advanced shading.</p></div>
-->

<div align="center">

[![Demo Video](https://github.com/DickyDicky7/HybridRasterPathTracer/blob/main/gallery/t001.png?raw=true)](https://www.youtube.com/watch?v=lNo-WZ8jKRc)

[![Demo Video](https://github.com/DickyDicky7/HybridRasterPathTracer/blob/main/gallery/t002.png?raw=true)](https://youtu.be/hwP4Qbc7ths)

[![Demo Video](https://github.com/DickyDicky7/HybridRasterPathTracer/blob/main/gallery/t002.png?raw=true)](https://youtu.be/NZQR-cDmdLo)

</div>
<!--
<div align="center">

[![Demo Video](https://github.com/DickyDicky7/HybridRasterPathTracer/blob/main/gallery/t001.png?raw=true)](https://www.youtube.com/watch?v=lNo-WZ8jKRc)

[![Demo Video](https://github.com/DickyDicky7/HybridRasterPathTracer/blob/main/gallery/t002.png?raw=true)](https://youtu.be/hwP4Qbc7ths)

[![Demo Video](https://github.com/DickyDicky7/HybridRasterPathTracer/blob/main/gallery/t002.png?raw=true)](https://youtu.be/NZQR-cDmdLo)

</div>
-->

<div align="center"><h2>Architecture & Pipeline</h2></div>
<!--
<div align="center"><h2>Architecture & Pipeline</h2></div>
-->

<div align="center">
  <p>The renderer implements a 5-stage hybrid graphics pipeline powered by <b>ModernGL</b>. In computer graphics, rendering 3D scenes generally falls into two primary paradigms: <b>Rasterization</b> and <b>Ray Tracing</b>. A hybrid approach seeks to unify the strengths of both methodologies.</p>
  <p><b>Rasterization</b> is exceptionally fast, though its lighting and shading models rely on fundamental approximations. While advanced techniques—such as Disney's BRDF/BSDF models and texture sampling—enable rasterization to achieve Physically Based Rendering (PBR), reflections, refractions, and transmission, it cannot inherently compute Global Illumination (GI). Achieving GI in a purely rasterized pipeline necessitates precomputed lighting methods (e.g., Lightmaps, Light Probes, Irradiance Volumes, Voxel Cone Tracing, Radiosity) or screen-space approximations like Screen Space Global Illumination (SSGI).</p>
  <p><b>Ray Tracing</b>, conversely, models the physical behavior of light more accurately. By simulating the trajectory of light rays, ray tracing naturally achieves GI alongside numerous physically accurate optical effects. Advancing beyond basic ray tracing introduces <b>Spectral Rendering</b>, an even more precise simulation of light physics. This project synthesizes the performance of rasterization for primary visibility with the physical accuracy of ray tracing for illumination.</p>
</div>
<!--
<div align="center">
  <p>The renderer implements a 5-stage hybrid graphics pipeline powered by <b>ModernGL</b>. In computer graphics, rendering 3D scenes generally falls into two primary paradigms: <b>Rasterization</b> and <b>Ray Tracing</b>. A hybrid approach seeks to unify the strengths of both methodologies.</p>
  <p><b>Rasterization</b> is exceptionally fast, though its lighting and shading models rely on fundamental approximations. While advanced techniques—such as Disney's BRDF/BSDF models and texture sampling—enable rasterization to achieve Physically Based Rendering (PBR), reflections, refractions, and transmission, it cannot inherently compute Global Illumination (GI). Achieving GI in a purely rasterized pipeline necessitates precomputed lighting methods (e.g., Lightmaps, Light Probes, Irradiance Volumes, Voxel Cone Tracing, Radiosity) or screen-space approximations like Screen Space Global Illumination (SSGI).</p>
  <p><b>Ray Tracing</b>, conversely, models the physical behavior of light more accurately. By simulating the trajectory of light rays, ray tracing naturally achieves GI alongside numerous physically accurate optical effects. Advancing beyond basic ray tracing introduces <b>Spectral Rendering</b>, an even more precise simulation of light physics. This project synthesizes the performance of rasterization for primary visibility with the physical accuracy of ray tracing for illumination.</p>
</div>
-->

<div align="center">
  <ul>
    <li><b>1. Geometry Pass (Rasterization):</b> Rather than calculating primary ray intersections—a computationally expensive process—standard rasterization is employed to efficiently render the scene into a <b>G-Buffer</b> (Geometry Buffer). This G-Buffer functions as a multi-layered render target, storing precise surface attributes such as Global Position, Normal vectors, Albedo (base color), and Tangents, instead of final pixel colors.</li>
    <li><b>2. Shading Pass (Compute Shader Ray Tracing):</b> Leveraging the surface data preserved in the G-Buffer, <b>Monte Carlo Ray Tracing</b> is utilized to compute scene illumination. Rays are traced through the environment using a Shader Storage Buffer Object (SSBO)-backed Bounding Volume Hierarchy (BVH). While the camera remains stationary, the renderer accumulates sequential frames of these traced rays to resolve physically accurate soft shadows, reflections, and global illumination.</li>
    <li><b>3. Denoising Pass:</b> Due to real-time constraints limiting the number of rays traced per pixel per frame, the raw output exhibits significant stochastic noise. This is mitigated through a fast, multi-pass compute <b>À-Trous edge-avoiding filter</b>, which intelligently smooths the noise while preserving high-frequency geometric details and sharp edges.</li>
    <li><b>4. Post-Processing Pass:</b> To emulate real-world optical characteristics, several post-processing effects are applied. <b>Temporal Anti-Aliasing (TAA)</b> mitigates aliasing artifacts by accumulating historical frame data, <b>Chromatic Aberration</b> simulates lens color fringing, and <b>Khronos PBR Neutral Tonemapping</b> compresses the High Dynamic Range (HDR) lighting into a standard display color space.</li>
    <li><b>5. Composite Pass:</b> The finalized, post-processed frame is subsequently presented to the display.</li>
  </ul>
</div>
<!--
<div align="center">
  <ul>
    <li><b>1. Geometry Pass (Rasterization):</b> Rather than calculating primary ray intersections—a computationally expensive process—standard rasterization is employed to efficiently render the scene into a <b>G-Buffer</b> (Geometry Buffer). This G-Buffer functions as a multi-layered render target, storing precise surface attributes such as Global Position, Normal vectors, Albedo (base color), and Tangents, instead of final pixel colors.</li>
    <li><b>2. Shading Pass (Compute Shader Ray Tracing):</b> Leveraging the surface data preserved in the G-Buffer, <b>Monte Carlo Ray Tracing</b> is utilized to compute scene illumination. Rays are traced through the environment using a Shader Storage Buffer Object (SSBO)-backed Bounding Volume Hierarchy (BVH). While the camera remains stationary, the renderer accumulates sequential frames of these traced rays to resolve physically accurate soft shadows, reflections, and global illumination.</li>
    <li><b>3. Denoising Pass:</b> Due to real-time constraints limiting the number of rays traced per pixel per frame, the raw output exhibits significant stochastic noise. This is mitigated through a fast, multi-pass compute <b>À-Trous edge-avoiding filter</b>, which intelligently smooths the noise while preserving high-frequency geometric details and sharp edges.</li>
    <li><b>4. Post-Processing Pass:</b> To emulate real-world optical characteristics, several post-processing effects are applied. <b>Temporal Anti-Aliasing (TAA)</b> mitigates aliasing artifacts by accumulating historical frame data, <b>Chromatic Aberration</b> simulates lens color fringing, and <b>Khronos PBR Neutral Tonemapping</b> compresses the High Dynamic Range (HDR) lighting into a standard display color space.</li>
    <li><b>5. Composite Pass:</b> The finalized, post-processed frame is subsequently presented to the display.</li>
  </ul>
</div>
-->

<div align="center"><h2>Features & Technical Details</h2></div>
<!--
<div align="center"><h2>Features & Technical Details</h2></div>
-->

<div align="center">
  <ul>
    <li><b>Advanced PBR Shading:</b> The renderer implements a <b>Principled BSDF</b> material model (inspired by Disney's shading research) to accurately simulate light-matter interactions. This Physically Based Rendering (PBR) approach ensures that materials react realistically to lighting based on intuitive parameters such as Roughness, Metallic, Transmission (for dielectrics like glass), and Emissivity.</li>
    <li><b>Lighting System:</b> Scenes are illuminated using a combination of analytic <b>Point Lights</b> and image-based <b>HDRI Environment Maps</b>. To optimize performance, the engine employs <i>Importance Sampling</i>. By calculating Probability Density and Cumulative Distribution Functions (PDF/CDF), the renderer prioritizes ray allocation toward dominant light sources, significantly reducing variance and computation time.</li>
    <li><b>Acceleration Structure (LBVH):</b> As evaluating ray intersections against millions of triangles is computationally prohibitive for real-time applications, a <b>Linear Bounding Volume Hierarchy (LBVH)</b> is utilized. By assigning 30-bit <b>Morton Codes</b> (Z-Order curves) to each geometric primitive, 3D spatial data is mapped into a sorted 1D array. This enables rapid CPU-side construction of the BVH, which the GPU subsequently traverses to minimize ray-primitive intersection tests.</li>
    <li><b>Scene & Geometry Processing:</b> The engine leverages <code>PyAssimp</code> for parsing standard 3D asset formats (e.g., FBX, OBJ). The ingestion pipeline automatically transforms meshes into world-space, computes missing Tangent vectors (essential for detailed normal mapping), and packs the geometry into an interleaved, <code>std430</code>-aligned Shader Storage Buffer Object (SSBO) for optimal GPU memory access.</li>
    <li><b>AI-Powered High-Quality Denoising:</b> In addition to the real-time À-Trous filter, users can invoke <b>Intel Open Image Denoise (OIDN)</b>. This integration feeds the unresolved noisy image—supplemented with Albedo and Normal feature buffers—into a machine learning model to generate a pristine, artifact-free final render.</li>
    <li><b>Hardware Optimizations:</b> On Windows systems equipped with hybrid graphics architectures (e.g., NVIDIA Optimus), Python applications frequently default to the integrated GPU. To circumvent this, the engine explicitly interfaces with the <code>nvapi64.dll</code> driver, heuristically prompting the operating system to allocate the high-performance discrete NVIDIA GPU.</li>
  </ul>
</div>
<!--
<div align="center">
  <ul>
    <li><b>Advanced PBR Shading:</b> The renderer implements a <b>Principled BSDF</b> material model (inspired by Disney's shading research) to accurately simulate light-matter interactions. This Physically Based Rendering (PBR) approach ensures that materials react realistically to lighting based on intuitive parameters such as Roughness, Metallic, Transmission (for dielectrics like glass), and Emissivity.</li>
    <li><b>Lighting System:</b> Scenes are illuminated using a combination of analytic <b>Point Lights</b> and image-based <b>HDRI Environment Maps</b>. To optimize performance, the engine employs <i>Importance Sampling</i>. By calculating Probability Density and Cumulative Distribution Functions (PDF/CDF), the renderer prioritizes ray allocation toward dominant light sources, significantly reducing variance and computation time.</li>
    <li><b>Acceleration Structure (LBVH):</b> As evaluating ray intersections against millions of triangles is computationally prohibitive for real-time applications, a <b>Linear Bounding Volume Hierarchy (LBVH)</b> is utilized. By assigning 30-bit <b>Morton Codes</b> (Z-Order curves) to each geometric primitive, 3D spatial data is mapped into a sorted 1D array. This enables rapid CPU-side construction of the BVH, which the GPU subsequently traverses to minimize ray-primitive intersection tests.</li>
    <li><b>Scene & Geometry Processing:</b> The engine leverages <code>PyAssimp</code> for parsing standard 3D asset formats (e.g., FBX, OBJ). The ingestion pipeline automatically transforms meshes into world-space, computes missing Tangent vectors (essential for detailed normal mapping), and packs the geometry into an interleaved, <code>std430</code>-aligned Shader Storage Buffer Object (SSBO) for optimal GPU memory access.</li>
    <li><b>AI-Powered High-Quality Denoising:</b> In addition to the real-time À-Trous filter, users can invoke <b>Intel Open Image Denoise (OIDN)</b>. This integration feeds the unresolved noisy image—supplemented with Albedo and Normal feature buffers—into a machine learning model to generate a pristine, artifact-free final render.</li>
    <li><b>Hardware Optimizations:</b> On Windows systems equipped with hybrid graphics architectures (e.g., NVIDIA Optimus), Python applications frequently default to the integrated GPU. To circumvent this, the engine explicitly interfaces with the <code>nvapi64.dll</code> driver, heuristically prompting the operating system to allocate the high-performance discrete NVIDIA GPU.</li>
  </ul>
</div>
-->

<div align="center"><h2>Controls</h2></div>
<!--
<div align="center"><h2>Controls</h2></div>
-->

<div align="center">
  <ul>
    <li><b>W, A, S, D, Q, E:</b> Move Camera</li>
    <li><b>Arrow Keys:</b> Rotate Camera</li>
    <li><b>0:</b> Path Tracing Mode (Default)</li>
    <li><b>1-4:</b> G-Buffer Debug Modes (Albedo, Normal, Position, Tangent)</li>
    <li><b>I:</b> Denoise current frame with Intel OIDN</li>
  </ul>
</div>
<!--
<div align="center">
  <ul>
    <li><b>W, A, S, D, Q, E:</b> Move Camera</li>
    <li><b>Arrow Keys:</b> Rotate Camera</li>
    <li><b>0:</b> Path Tracing Mode (Default)</li>
    <li><b>1-4:</b> G-Buffer Debug Modes (Albedo, Normal, Position, Tangent)</li>
    <li><b>I:</b> Denoise current frame with Intel OIDN</li>
  </ul>
</div>
-->

<div align="center"><h2>Requirements</h2></div>
<!--
<div align="center"><h2>Requirements</h2></div>
-->

<div align="center">
  <ul>
    <li>Python 3.10+</li>
    <li>ModernGL, ModernGL Window, NumPy, Pyrr, OpenCV, PyOIDN, PyAssimp</li>
  </ul>
</div>
<!--
<div align="center">
  <ul>
    <li>Python 3.10+</li>
    <li>ModernGL, ModernGL Window, NumPy, Pyrr, OpenCV, PyOIDN, PyAssimp</li>
  </ul>
</div>
-->