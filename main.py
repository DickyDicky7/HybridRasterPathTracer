import ctypes
import ctypes
import os
import os
import moderngl_window as mglw
import moderngl_window as mglw
from src.renderer.hybrid_renderer import HybridRenderer
from src.renderer.hybrid_renderer import HybridRenderer

if __name__ == "__main__":
    # Try to load the NVIDIA driver explicitly
#   # Try to load the NVIDIA driver explicitly
    try:
#   try:
        # This forces the NVIDIA driver to load
#       # This forces the NVIDIA driver to load
        ctypes.CDLL("nvapi64.dll")
#       ctypes.CDLL("nvapi64.dll")
        print("NVIDIA driver loaded.")
#       print("NVIDIA driver loaded.")
    except:
#   except:
        print("NVIDIA driver not found.")
#       print("NVIDIA driver not found.")
    mglw.run_window_config(HybridRenderer)
#   mglw.run_window_config(HybridRenderer)
    pass
#   pass
