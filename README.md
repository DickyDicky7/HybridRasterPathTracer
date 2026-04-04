<div align="center"><h1>HybridRasterPathTracer</h1></div>
<!--
<div align="center"><h1>HybridRasterPathTracer</h1></div>
-->

<div align="center"><p>An experimental real-time hybrid renderer that uses a traditional rasterization pipeline for primary visibility and a Monte Carlo path tracing compute pipeline for global illumination and shading.</p></div>
<!--
<div align="center"><p>An experimental real-time hybrid renderer that uses a traditional rasterization pipeline for primary visibility and a Monte Carlo path tracing compute pipeline for global illumination and shading.</p></div>
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
  <p>The renderer utilizes a 5-stage hybrid graphics pipeline via <b>ModernGL</b>. But what does "hybrid" mean? In computer graphics, there are generally two main ways to draw 3D scenes: <b>Rasterization</b> (super fast, but lighting is "faked") and <b>Ray Tracing</b> (physically accurate lighting, but mathematically expensive). This project combines the advantages of both approaches.</p>
</div>
<!--
<div align="center">
  <p>The renderer utilizes a 5-stage hybrid graphics pipeline via <b>ModernGL</b>. But what does "hybrid" mean? In computer graphics, there are generally two main ways to draw 3D scenes: <b>Rasterization</b> (super fast, but lighting is "faked") and <b>Ray Tracing</b> (physically accurate lighting, but mathematically expensive). This project combines the advantages of both approaches.</p>
</div>
-->

<div align="center">
  <ul>
    <li><b>1. Geometry Pass (Rasterization):</b> Instead of trying to find the first object a ray hits (which is computationally expensive), standard rasterization is used to quickly draw the scene into a <b>G-Buffer</b> (Geometry Buffer). A G-Buffer acts as a multi-layered image that stores information like Global Position, Normal (the direction the surface faces), Albedo (base color), and Tangent data, rather than just final colors.</li>
    <li><b>2. Shading Pass (Compute Shader Ray Tracing):</b> Utilizing the exact surface information from the G-Buffer, <b>Monte Carlo Ray Tracing</b> is employed to calculate the lighting. Mathematical "rays" of light are traced into the scene using an SSBO-backed BVH (a 3D search tree). As the camera stays still, the renderer accumulates multiple frames of these bouncing rays to create physically accurate soft shadows, reflections, and global illumination.</li>
    <li><b>3. Denoise Pass:</b> Because only a limited number of rays can be traced per pixel every frame, the raw output appears very grainy. This is resolved using a fast multi-pass compute <b>À-Trous edge-avoiding filter</b>. This technique intelligently smooths out the noise without blurring the sharp edges of the 3D models.</li>
    <li><b>4. Post-Processing Pass:</b> To emulate the look of a real-world camera, several effects are applied: <b>Temporal Anti-Aliasing (TAA)</b> smooths out jagged pixel edges by combining past frames, <b>Chromatic Aberration</b> simulates camera lens color fringing, and <b>Khronos PBR Neutral Tonemapping</b> maps the High Dynamic Range (HDR) lighting into a standard color space suitable for display.</li>
    <li><b>5. Composite Pass:</b> Finally, the processed, polished texture is rendered to the screen.</li>
  </ul>
</div>
<!--
<div align="center">
  <ul>
    <li><b>1. Geometry Pass (Rasterization):</b> Instead of trying to find the first object a ray hits (which is computationally expensive), standard rasterization is used to quickly draw the scene into a <b>G-Buffer</b> (Geometry Buffer). A G-Buffer acts as a multi-layered image that stores information like Global Position, Normal (the direction the surface faces), Albedo (base color), and Tangent data, rather than just final colors.</li>
    <li><b>2. Shading Pass (Compute Shader Ray Tracing):</b> Utilizing the exact surface information from the G-Buffer, <b>Monte Carlo Ray Tracing</b> is employed to calculate the lighting. Mathematical "rays" of light are traced into the scene using an SSBO-backed BVH (a 3D search tree). As the camera stays still, the renderer accumulates multiple frames of these bouncing rays to create physically accurate soft shadows, reflections, and global illumination.</li>
    <li><b>3. Denoise Pass:</b> Because only a limited number of rays can be traced per pixel every frame, the raw output appears very grainy. This is resolved using a fast multi-pass compute <b>À-Trous edge-avoiding filter</b>. This technique intelligently smooths out the noise without blurring the sharp edges of the 3D models.</li>
    <li><b>4. Post-Processing Pass:</b> To emulate the look of a real-world camera, several effects are applied: <b>Temporal Anti-Aliasing (TAA)</b> smooths out jagged pixel edges by combining past frames, <b>Chromatic Aberration</b> simulates camera lens color fringing, and <b>Khronos PBR Neutral Tonemapping</b> maps the High Dynamic Range (HDR) lighting into a standard color space suitable for display.</li>
    <li><b>5. Composite Pass:</b> Finally, the processed, polished texture is rendered to the screen.</li>
  </ul>
</div>
-->

<div align="center"><h2>Features & Technical Details</h2></div>
<!--
<div align="center"><h2>Features & Technical Details</h2></div>
-->

<div align="center">
  <ul>
    <li><b>Advanced PBR Shading:</b> "PBR" stands for Physically Based Rendering. A <b>Principled BSDF</b> material model (inspired by Disney's research) is used to mathematically simulate how light interacts with different types of matter. This ensures materials automatically look realistic based on parameters like Roughness, Metallic, Transmission (for glass), and Emissivity (for glowing objects).</li>
    <li><b>Lighting System:</b> The scene can be lit by both analytic <b>Point Lights</b> and a 360-degree <b>HDRI Environment Map</b>. To maintain high performance, the engine uses <i>Importance Sampling</i>—calculating PDF/CDF probabilities to direct more processing power towards casting rays at bright lights rather than spending time on dim ones.</li>
    <li><b>Acceleration Structure (LBVH):</b> Ray tracing millions of triangles directly is not feasible in real-time. Instead, a <b>Linear Bounding Volume Hierarchy (LBVH)</b> is constructed. By assigning 30-bit <b>Morton Codes</b> (Z-Order curves) to each triangle, their 3D positions are mapped into a 1D sorted list. This allows the CPU to build a search tree extremely fast, which the GPU then uses to drastically reduce ray collision checks.</li>
    <li><b>Scene & Geometry Processing:</b> The system uses <code>PyAssimp</code> to load standard 3D models (FBX/OBJ). The pipeline automatically pre-transforms meshes into world-space, generates missing Tangent vectors (crucial for normal mapping bumps and scratches), and packs everything into an interleaved <code>std430</code> aligned Global Geometry Buffer (SSBO) for efficient GPU consumption.</li>
    <li><b>AI-Powered High-Quality Denoising:</b> Besides the real-time À-Trous filter, pressing the 'I' key invokes <b>Intel Open Image Denoise (OIDN)</b>. This feeds the noisy image, along with the Albedo and Normal data, into a machine learning model to produce a remarkably clean final render.</li>
    <li><b>Hardware Optimizations:</b> On Windows laptops with both integrated and discrete GPUs (Optimus), Python applications often default to the less powerful integrated graphics. The engine explicitly loads the <code>nvapi64.dll</code> driver to heuristically trigger the OS to switch to the High-Performance NVIDIA GPU.</li>
  </ul>
</div>
<!--
<div align="center">
  <ul>
    <li><b>Advanced PBR Shading:</b> "PBR" stands for Physically Based Rendering. A <b>Principled BSDF</b> material model (inspired by Disney's research) is used to mathematically simulate how light interacts with different types of matter. This ensures materials automatically look realistic based on parameters like Roughness, Metallic, Transmission (for glass), and Emissivity (for glowing objects).</li>
    <li><b>Lighting System:</b> The scene can be lit by both analytic <b>Point Lights</b> and a 360-degree <b>HDRI Environment Map</b>. To maintain high performance, the engine uses <i>Importance Sampling</i>—calculating PDF/CDF probabilities to direct more processing power towards casting rays at bright lights rather than spending time on dim ones.</li>
    <li><b>Acceleration Structure (LBVH):</b> Ray tracing millions of triangles directly is not feasible in real-time. Instead, a <b>Linear Bounding Volume Hierarchy (LBVH)</b> is constructed. By assigning 30-bit <b>Morton Codes</b> (Z-Order curves) to each triangle, their 3D positions are mapped into a 1D sorted list. This allows the CPU to build a search tree extremely fast, which the GPU then uses to drastically reduce ray collision checks.</li>
    <li><b>Scene & Geometry Processing:</b> The system uses <code>PyAssimp</code> to load standard 3D models (FBX/OBJ). The pipeline automatically pre-transforms meshes into world-space, generates missing Tangent vectors (crucial for normal mapping bumps and scratches), and packs everything into an interleaved <code>std430</code> aligned Global Geometry Buffer (SSBO) for efficient GPU consumption.</li>
    <li><b>AI-Powered High-Quality Denoising:</b> Besides the real-time À-Trous filter, pressing the 'I' key invokes <b>Intel Open Image Denoise (OIDN)</b>. This feeds the noisy image, along with the Albedo and Normal data, into a machine learning model to produce a remarkably clean final render.</li>
    <li><b>Hardware Optimizations:</b> On Windows laptops with both integrated and discrete GPUs (Optimus), Python applications often default to the less powerful integrated graphics. The engine explicitly loads the <code>nvapi64.dll</code> driver to heuristically trigger the OS to switch to the High-Performance NVIDIA GPU.</li>
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