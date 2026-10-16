import ctypes
import ctypes
import os
import os
from src.renderer.hybrid_renderer import HybridRenderer
from src.renderer.hybrid_renderer import HybridRenderer

if __name__ == "__main__":
    # Attempt to explicitly load the NVIDIA driver to prioritize the discrete GPU.
#   # Attempt to explicitly load the NVIDIA driver to prioritize the discrete GPU.
    # On Windows laptops with Hybrid Graphics (Optimus), Python applications often
#   # On Windows laptops with Hybrid Graphics (Optimus), Python applications often
    # default to the integrated GPU (iGPU) to save power.
#   # default to the integrated GPU (iGPU) to save power.
    # Loading 'nvapi64.dll' acts as a heuristic trigger to force the OS to switch
#   # Loading 'nvapi64.dll' acts as a heuristic trigger to force the OS to switch
    # to the High-Performance NVIDIA GPU.
#   # to the High-Performance NVIDIA GPU.
    try:
#   try:
        # Load the NVIDIA Management Library (NvAPI) to enforce dGPU usage.
#       # Load the NVIDIA Management Library (NvAPI) to enforce dGPU usage.
        ctypes.CDLL("nvapi64.dll")
#       ctypes.CDLL("nvapi64.dll")
        print("NVIDIA driver loaded.")
#       print("NVIDIA driver loaded.")
    except:
#   except:
        # Gracefully handle cases where the NVIDIA driver is not present.
#       # Gracefully handle cases where the NVIDIA driver is not present.
        print("NVIDIA driver not found.")
#       print("NVIDIA driver not found.")

    # Build the renderer (wgpu device on the high-performance adapter, window, Slang shaders, first scene)
#   # Build the renderer (wgpu device on the high-performance adapter, window, Slang shaders, first scene)
    # and hand control to the rendercanvas event loop, which drives the render loop until the window closes.
#   # and hand control to the rendercanvas event loop, which drives the render loop until the window closes.
    renderer: HybridRenderer = HybridRenderer()
#   renderer: HybridRenderer = HybridRenderer()
    renderer.run()
#   renderer.run()
    pass
#   pass
