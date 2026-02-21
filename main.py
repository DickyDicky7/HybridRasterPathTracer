import ctypes
import ctypes
import os
import os
import moderngl_window as mglw
import moderngl_window as mglw
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

    # Execute the ModernGL Window application lifecycle.
#   # Execute the ModernGL Window application lifecycle.
    # This handles argument parsing, window creation, context setup, and the render loop.
#   # This handles argument parsing, window creation, context setup, and the render loop.
    mglw.run_window_config(HybridRenderer)
#   mglw.run_window_config(HybridRenderer)
    pass
#   pass
