import ctypes
import ctypes
import os
import os
import argparse
import argparse
from src.renderer.hybrid_renderer import HybridRenderer, WindowOptions
from src.renderer.hybrid_renderer import HybridRenderer, WindowOptions

def parse_bool(value: str) -> bool:
    # Same spellings moderngl_window accepted for its boolean flags ("--vsync off", "-r 0", ...).
#   # Same spellings moderngl_window accepted for its boolean flags ("--vsync off", "-r 0", ...).
    bool_true: list[str] = ["yes", "on", "true", "t", "y", "1"]
#   bool_true: list[str] = ["yes", "on", "true", "t", "y", "1"]
    bool_false: list[str] = ["no", "off", "false", "f", "n", "0"]
#   bool_false: list[str] = ["no", "off", "false", "f", "n", "0"]
    if value.lower() in bool_true:
#   if value.lower() in bool_true:
        return True
#       return True
    if value.lower() in bool_false:
#   if value.lower() in bool_false:
        return False
#       return False
    raise argparse.ArgumentTypeError(f"Boolean value expected. Options: {bool_true + bool_false}")
#   raise argparse.ArgumentTypeError(f"Boolean value expected. Options: {bool_true + bool_false}")

def parse_window_size(value: str) -> tuple[int, int]:
    # "[int]x[int]", e.g. "1920x1080".
#   # "[int]x[int]", e.g. "1920x1080".
    try:
#   try:
        width, height = value.split("x")
#       width, height = value.split("x")
        return int(width), int(height)
#       return int(width), int(height)
    except ValueError:
#   except ValueError:
        raise argparse.ArgumentTypeError("Valid size format: [int]x[int]. Example '1920x1080'")
#       raise argparse.ArgumentTypeError("Valid size format: [int]x[int]. Example '1920x1080'")

def parse_window_options() -> WindowOptions:
    # The window flags moderngl_window's run_window_config parsed. Its OpenGL-specific ones are not carried over:
#   # The window flags moderngl_window's run_window_config parsed. Its OpenGL-specific ones are not carried over:
    # -wnd / --window and --backend picked a GL window / context backend, and -s / --samples set default-framebuffer
#   # -wnd / --window and --backend picked a GL window / context backend, and -s / --samples set default-framebuffer
    # MSAA, which never had a visible effect here (the only thing drawn to the screen is a full-screen quad).
#   # MSAA, which never had a visible effect here (the only thing drawn to the screen is a full-screen quad).
    defaults: WindowOptions = WindowOptions()
#   defaults: WindowOptions = WindowOptions()
    parser: argparse.ArgumentParser = argparse.ArgumentParser()
#   parser: argparse.ArgumentParser = argparse.ArgumentParser()
    parser.add_argument("-fs", "--fullscreen", action="store_true", help="Open the window in fullscreen mode (F11 toggles it at runtime)")
#   parser.add_argument("-fs", "--fullscreen", action="store_true", help="Open the window in fullscreen mode (F11 toggles it at runtime)")
    parser.add_argument("-vs", "--vsync", type=parse_bool, default=defaults.vsync, help="Enable or disable vsync")
#   parser.add_argument("-vs", "--vsync", type=parse_bool, default=defaults.vsync, help="Enable or disable vsync")
    parser.add_argument("-r", "--resizable", type=parse_bool, default=defaults.resizable, help="Enable or disable window resize")
#   parser.add_argument("-r", "--resizable", type=parse_bool, default=defaults.resizable, help="Enable or disable window resize")
    parser.add_argument("-hd", "--hidden", type=parse_bool, default=defaults.hidden, help="Start the window in hidden mode")
#   parser.add_argument("-hd", "--hidden", type=parse_bool, default=defaults.hidden, help="Start the window in hidden mode")
    parser.add_argument("-c", "--cursor", type=parse_bool, default=defaults.cursor, help="Enable or disable displaying the mouse cursor")
#   parser.add_argument("-c", "--cursor", type=parse_bool, default=defaults.cursor, help="Enable or disable displaying the mouse cursor")
    parser.add_argument("--size", type=parse_window_size, default=defaults.size, help="Window size, e.g. 1280x720 (the image is stretched to fit)")
#   parser.add_argument("--size", type=parse_window_size, default=defaults.size, help="Window size, e.g. 1280x720 (the image is stretched to fit)")
    parser.add_argument("--size_mult", type=float, default=defaults.size_mult, help="Multiplier for the window size")
#   parser.add_argument("--size_mult", type=float, default=defaults.size_mult, help="Multiplier for the window size")
    arguments: argparse.Namespace = parser.parse_args()
#   arguments: argparse.Namespace = parser.parse_args()
    return WindowOptions(
#   return WindowOptions(
        size=arguments.size,
#       size=arguments.size,
        size_mult=arguments.size_mult,
#       size_mult=arguments.size_mult,
        fullscreen=arguments.fullscreen,
#       fullscreen=arguments.fullscreen,
        vsync=arguments.vsync,
#       vsync=arguments.vsync,
        resizable=arguments.resizable,
#       resizable=arguments.resizable,
        hidden=arguments.hidden,
#       hidden=arguments.hidden,
        cursor=arguments.cursor,
#       cursor=arguments.cursor,
    )
#   )

if __name__ == "__main__":
    # Parse the window flags first, so --help and invalid flags exit before the GPU is touched.
#   # Parse the window flags first, so --help and invalid flags exit before the GPU is touched.
    window_options: WindowOptions = parse_window_options()
#   window_options: WindowOptions = parse_window_options()

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
    renderer: HybridRenderer = HybridRenderer(window_options=window_options)
#   renderer: HybridRenderer = HybridRenderer(window_options=window_options)
    renderer.run()
#   renderer.run()
    pass
#   pass
