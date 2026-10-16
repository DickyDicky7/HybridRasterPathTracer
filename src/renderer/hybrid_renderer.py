import wgpu # type: ignore[import-untyped]
import wgpu
from rendercanvas.glfw import RenderCanvas, loop # type: ignore[import-untyped]
from rendercanvas.glfw import RenderCanvas, loop
import glfw
import glfw
import numpy as np
import numpy as np
import numpy.typing as npt
import numpy.typing as npt
import pyrr as rr # type: ignore[import-untyped]
import pyrr as rr
import pathlib as pl
import pathlib as pl
import typing
import typing
import functools
import functools
import concurrent.futures
import concurrent.futures
import enum
import enum
import time as tm
import time as tm
import os
import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
import cv2
import cv2
import pyoidn     # type: ignore[import-untyped]
import pyoidn
from src.renderer.shader_compiler import ComputeProgram, RenderProgram, prefetch_slang, compute_stages, render_stages
from src.renderer.shader_compiler import ComputeProgram, RenderProgram, prefetch_slang, compute_stages, render_stages
from src.scene.scene_builder import SceneBuilder
from src.scene.scene_builder import SceneBuilder
from src.scene.camera import Camera
from src.scene.camera import Camera
from src.scene import BaseScene, ShowcaseScene, CornellBoxScene, OfficeScene
from src.scene import BaseScene, ShowcaseScene, CornellBoxScene, OfficeScene
from src.core.common_types import vec2i32, vec3i32, vec4i32, vec2f32, vec3f32, vec4f32, Material, PointLight, MediumVolume
from src.core.common_types import vec2i32, vec3i32, vec4i32, vec2f32, vec3f32, vec4f32, Material, PointLight, MediumVolume

class RenderMode(enum.IntEnum):
    PATH_TRACE = 0
#   PATH_TRACE = 0
    ALBEDO = 1
#   ALBEDO = 1
    NORMAL = 2
#   NORMAL = 2
    POSITION = 3
#   POSITION = 3
    TANGENT = 4
#   TANGENT = 4

class WindowOptions(typing.NamedTuple):
    # Window settings parsed from the command line by main.py; the same flags moderngl_window's
#   # Window settings parsed from the command line by main.py; the same flags moderngl_window's
    # run_window_config accepted. size / size_mult only change the window: the render targets stay
#   # run_window_config accepted. size / size_mult only change the window: the render targets stay
    # at HybridRenderer.window_size and are stretched to fit, as with moderngl_window.
#   # at HybridRenderer.window_size and are stretched to fit, as with moderngl_window.
    size: vec2i32 | None = None
#   size: vec2i32 | None = None
    size_mult: float = 1.0
#   size_mult: float = 1.0
    fullscreen: bool = False
#   fullscreen: bool = False
    vsync: bool = True
#   vsync: bool = True
    resizable: bool = True
#   resizable: bool = True
    hidden: bool = False
#   hidden: bool = False
    cursor: bool = True
#   cursor: bool = True

class HybridRenderer:
    # Main Renderer Class implementing a Hybrid Pipeline:
#   # Main Renderer Class implementing a Hybrid Pipeline:
    # 1. Rasterization Pass: Renders scene geometry to G-Buffer (Position, Normal, Albedo, etc.).
#   # 1. Rasterization Pass: Renders scene geometry to G-Buffer (Position, Normal, Albedo, etc.).
    # 2. Compute Pass (Ray Tracing): Uses G-Buffer + Ray Tracing (BVH) to calculate lighting/reflections.
#   # 2. Compute Pass (Ray Tracing): Uses G-Buffer + Ray Tracing (BVH) to calculate lighting/reflections.
    # 3. Post Processing Pass: Applies effects like Chromatic Aberration and Vignette.
#   # 3. Post Processing Pass: Applies effects like Chromatic Aberration and Vignette.
    # 4. Composite Pass: Displays the final result to the screen.
#   # 4. Composite Pass: Displays the final result to the screen.
    # GPU work runs on wgpu (WebGPU) with every shader written in Slang and compiled to WGSL at start-up.
#   # GPU work runs on wgpu (WebGPU) with every shader written in Slang and compiled to WGSL at start-up.
    title: str = "Hybrid Rendering: Rasterization + Path Tracing"
#   title: str = "Hybrid Rendering: Rasterization + Path Tracing"
    window_size: vec2i32 = (800, 600)
#   window_size: vec2i32 = (800, 600)
    aspect_ratio: float = window_size[0] / window_size[1]
#   aspect_ratio: float = window_size[0] / window_size[1]
    resource_dir: pl.Path = pl.Path(__file__).parent.resolve(strict=False)
#   resource_dir: pl.Path = pl.Path(__file__).parent.resolve(strict=False)
    shader_dir: pl.Path = (resource_dir / "../shaders").resolve(strict=False)
#   shader_dir: pl.Path = (resource_dir / "../shaders").resolve(strict=False)
    # 8-bit texel -> normalised float32, plain and with the sRGB (gamma 2.2) decode applied (see load_texture_layer).
#   # 8-bit texel -> normalised float32, plain and with the sRGB (gamma 2.2) decode applied (see load_texture_layer).
    unorm8_to_float_lut: npt.NDArray[np.float32] = np.arange(256, dtype=np.float32) / 255.0
#   unorm8_to_float_lut: npt.NDArray[np.float32] = np.arange(256, dtype=np.float32) / 255.0
    srgb8_to_linear_lut: npt.NDArray[np.float32] = np.power(np.arange(256, dtype=np.float32) / 255.0, 2.2)
#   srgb8_to_linear_lut: npt.NDArray[np.float32] = np.power(np.arange(256, dtype=np.float32) / 255.0, 2.2)
    # Worker threads preparing texture array layers (see upload_pending_textures).
#   # Worker threads preparing texture array layers (see upload_pending_textures).
    texture_worker_count: int = 4
#   texture_worker_count: int = 4
    # rendercanvas key names -> the names key_state and on_key_event use (letters and digits map to themselves, upper-cased).
#   # rendercanvas key names -> the names key_state and on_key_event use (letters and digits map to themselves, upper-cased).
    key_names: dict[str, str] = {
#   key_names: dict[str, str] = {
        "ArrowUp": "UP", "ArrowDown": "DOWN", "ArrowLeft": "LEFT", "ArrowRight": "RIGHT",
#       "ArrowUp": "UP", "ArrowDown": "DOWN", "ArrowLeft": "LEFT", "ArrowRight": "RIGHT",
        "Tab": "TAB",
#       "Tab": "TAB",
    }
#   }

    def __init__(self, window_options: WindowOptions = WindowOptions()) -> None:
#   def __init__(self, window_options: WindowOptions = WindowOptions()) -> None:
        self.window_options: WindowOptions = window_options
#       self.window_options: WindowOptions = window_options
        self.frame_count: int = 0
#       self.frame_count: int = 0
        self.cache_frame_counter: int = 0
#       self.cache_frame_counter: int = 0
        self.last_view_matrix: rr.Matrix44 = None
#       self.last_view_matrix: rr.Matrix44 = None
        self.last_view_proj_matrix: typing.Optional[rr.Matrix44] = None
#       self.last_view_proj_matrix: typing.Optional[rr.Matrix44] = None
        self.last_camera_position: typing.Optional[tuple[float, float, float]] = None
#       self.last_camera_position: typing.Optional[tuple[float, float, float]] = None
        self.is_scene_reset: bool = True
#       self.is_scene_reset: bool = True
        self.render_mode: RenderMode = RenderMode.PATH_TRACE
#       self.render_mode: RenderMode = RenderMode.PATH_TRACE
        self.is_nrc_enabled: bool = True
#       self.is_nrc_enabled: bool = True
        self.is_restir_enabled: bool = True
#       self.is_restir_enabled: bool = True
        self.is_vrt_enabled: bool = True
#       self.is_vrt_enabled: bool = True
        self.is_vrt_visualize: bool = False
#       self.is_vrt_visualize: bool = False
        # Volumetrics: the ray-marched sky cloud layer and the in-scene participating media. Each only
#       # Volumetrics: the ray-marched sky cloud layer and the in-scene participating media. Each only
        # takes effect in scenes that define it (BaseScene.sky_clouds_enabled / get_medium_volumes).
#       # takes effect in scenes that define it (BaseScene.sky_clouds_enabled / get_medium_volumes).
        self.is_sky_clouds_enabled: bool = True
#       self.is_sky_clouds_enabled: bool = True
        self.is_media_enabled: bool = True
#       self.is_media_enabled: bool = True
        self.nrc_training_step: int = 0
#       self.nrc_training_step: int = 0

        # -----------------------------
#       # -----------------------------
        # 0. Device and Window Setup
#       # 0. Device and Window Setup
        # -----------------------------
#       # -----------------------------
        # Start compiling every Slang program right away: the slangc processes run in parallel, on background threads,
#       # Start compiling every Slang program right away: the slangc processes run in parallel, on background threads,
        # while the adapter, device and window are created below; each program constructor then collects its result.
#       # while the adapter, device and window are created below; each program constructor then collects its result.
        prefetch_slang(shader_dir=self.shader_dir, programs=[
#       prefetch_slang(shader_dir=self.shader_dir, programs=[
            render_stages(vertex_file_name="hybrid_geometry_vs.slang", fragment_file_name="hybrid_geometry_fs.slang"),
#           render_stages(vertex_file_name="hybrid_geometry_vs.slang", fragment_file_name="hybrid_geometry_fs.slang"),
            render_stages(vertex_file_name="hybrid_renderer_vs.slang", fragment_file_name="hybrid_renderer_fs.slang"),
#           render_stages(vertex_file_name="hybrid_renderer_vs.slang", fragment_file_name="hybrid_renderer_fs.slang"),
            *[compute_stages(file_name=file_name) for file_name in (
#           *[compute_stages(file_name=file_name) for file_name in (
                "hybrid_shading_cs.slang", "nrc_gather_cs.slang", "nrc_train_cs.slang", "cloud_noise_cs.slang", "sky_clouds_bake_cs.slang",
#               "hybrid_shading_cs.slang", "nrc_gather_cs.slang", "nrc_train_cs.slang", "cloud_noise_cs.slang", "sky_clouds_bake_cs.slang",
                "post_chromatic_aberration_cs.slang", "post_vignette_cs.slang", "post_tonemap_cs.slang", "post_fxaa_cs.slang",
#               "post_chromatic_aberration_cs.slang", "post_vignette_cs.slang", "post_tonemap_cs.slang", "post_fxaa_cs.slang",
            )],
#           )],
        ])
#       ])
        # Ask for the high-performance adapter so hybrid-graphics laptops render on the discrete GPU.
#       # Ask for the high-performance adapter so hybrid-graphics laptops render on the discrete GPU.
        self.adapter: wgpu.GPUAdapter = wgpu.gpu.request_adapter_sync(power_preference="high-performance")
#       self.adapter: wgpu.GPUAdapter = wgpu.gpu.request_adapter_sync(power_preference="high-performance")
        print(f"[GPU] {self.adapter.info['device']} ({self.adapter.info['backend_type']})")
#       print(f"[GPU] {self.adapter.info['device']} ({self.adapter.info['backend_type']})")
        # float32-filterable: the HDRI and the debug views sample rgba32float textures through linear samplers.
#       # float32-filterable: the HDRI and the debug views sample rgba32float textures through linear samplers.
        # The shading pass binds eight storage buffers (the WebGPU default ceiling) and large scenes outgrow
#       # The shading pass binds eight storage buffers (the WebGPU default ceiling) and large scenes outgrow
        # the default buffer sizes, so those limits are raised to whatever the adapter supports.
#       # the default buffer sizes, so those limits are raised to whatever the adapter supports.
        required_limits: dict[str, int] = {
#       required_limits: dict[str, int] = {
            limit_name: self.adapter.limits[limit_name]
#           limit_name: self.adapter.limits[limit_name]
            for limit_name in ("max-storage-buffers-per-shader-stage", "max-storage-buffer-binding-size", "max-buffer-size", "max-texture-array-layers")
#           for limit_name in ("max-storage-buffers-per-shader-stage", "max-storage-buffer-binding-size", "max-buffer-size", "max-texture-array-layers")
        }
#       }
        self.device: wgpu.GPUDevice = self.adapter.request_device_sync(required_features=["float32-filterable"], required_limits=required_limits)
#       self.device: wgpu.GPUDevice = self.adapter.request_device_sync(required_features=["float32-filterable"], required_limits=required_limits)

        canvas_size: vec2i32 = window_options.size if window_options.size is not None else self.window_size
#       canvas_size: vec2i32 = window_options.size if window_options.size is not None else self.window_size
        canvas_size = (int(canvas_size[0] * window_options.size_mult), int(canvas_size[1] * window_options.size_mult))
#       canvas_size = (int(canvas_size[0] * window_options.size_mult), int(canvas_size[1] * window_options.size_mult))
        self.canvas: RenderCanvas = RenderCanvas(size=canvas_size, title=self.title, update_mode="continuous", max_fps=1000.0, vsync=window_options.vsync)
#       self.canvas: RenderCanvas = RenderCanvas(size=canvas_size, title=self.title, update_mode="continuous", max_fps=1000.0, vsync=window_options.vsync)
        # rendercanvas has no API for resizability, visibility or fullscreen, so those go straight to its glfw window handle.
#       # rendercanvas has no API for resizability, visibility or fullscreen, so those go straight to its glfw window handle.
        self.glfw_window: typing.Any = self.canvas._window
#       self.glfw_window: typing.Any = self.canvas._window
        if not window_options.resizable:
#       if not window_options.resizable:
            glfw.set_window_attrib(self.glfw_window, glfw.RESIZABLE, glfw.FALSE)
#           glfw.set_window_attrib(self.glfw_window, glfw.RESIZABLE, glfw.FALSE)
        if not window_options.cursor:
#       if not window_options.cursor:
            self.canvas.set_cursor("none")
#           self.canvas.set_cursor("none")
        if window_options.hidden:
#       if window_options.hidden:
            glfw.hide_window(self.glfw_window)
#           glfw.hide_window(self.glfw_window)
        self.is_fullscreen: bool = False
#       self.is_fullscreen: bool = False
        self.windowed_rect: vec4i32 = (0, 0, canvas_size[0], canvas_size[1])
#       self.windowed_rect: vec4i32 = (0, 0, canvas_size[0], canvas_size[1])
        if window_options.fullscreen:
#       if window_options.fullscreen:
            self.set_fullscreen(is_fullscreen=True)
#           self.set_fullscreen(is_fullscreen=True)
        self.present_context: typing.Any = self.canvas.get_context("wgpu")
#       self.present_context: typing.Any = self.canvas.get_context("wgpu")
        # A non-sRGB swapchain: the post-processing chain already gamma-encodes, like the GL default framebuffer did.
#       # A non-sRGB swapchain: the post-processing chain already gamma-encodes, like the GL default framebuffer did.
        self.present_format: str = self.present_context.get_preferred_format(self.adapter).removesuffix("-srgb")
#       self.present_format: str = self.present_context.get_preferred_format(self.adapter).removesuffix("-srgb")
        self.present_context.configure(device=self.device, format=self.present_format)
#       self.present_context.configure(device=self.device, format=self.present_format)
        self.canvas.add_event_handler(self.on_key_down, "key_down")
#       self.canvas.add_event_handler(self.on_key_down, "key_down")
        self.canvas.add_event_handler(self.on_key_up, "key_up")
#       self.canvas.add_event_handler(self.on_key_up, "key_up")
        self.canvas.add_event_handler(self.on_close, "close")
#       self.canvas.add_event_handler(self.on_close, "close")
        self.start_time: float = tm.perf_counter()
#       self.start_time: float = tm.perf_counter()
        self.last_frame_time: float = self.start_time
#       self.last_frame_time: float = self.start_time

        # Sampler states. OpenGL attached filtering / wrapping to each texture; WebGPU keeps them in sampler
#       # Sampler states. OpenGL attached filtering / wrapping to each texture; WebGPU keeps them in sampler
        # objects, bound together with the texture for every Slang combined texture-sampler (Sampler2D, ...).
#       # objects, bound together with the texture for every Slang combined texture-sampler (Sampler2D, ...).
        # Clamp-to-edge on every screen-sized target: FXAA's edge-end search taps (and any filtered read
#       # Clamp-to-edge on every screen-sized target: FXAA's edge-end search taps (and any filtered read
        # near the border) would otherwise wrap around and bleed the opposite side of the screen into the edges.
#       # near the border) would otherwise wrap around and bleed the opposite side of the screen into the edges.
        self.sampler_screen: wgpu.GPUSampler = self.device.create_sampler(
#       self.sampler_screen: wgpu.GPUSampler = self.device.create_sampler(
            mag_filter=wgpu.FilterMode.linear, min_filter=wgpu.FilterMode.linear,
#           mag_filter=wgpu.FilterMode.linear, min_filter=wgpu.FilterMode.linear,
            address_mode_u=wgpu.AddressMode.clamp_to_edge, address_mode_v=wgpu.AddressMode.clamp_to_edge,
#           address_mode_u=wgpu.AddressMode.clamp_to_edge, address_mode_v=wgpu.AddressMode.clamp_to_edge,
        )
#       )
        # Equirectangular maps (HDRI, baked sky clouds) wrap around in longitude and clamp at the poles.
#       # Equirectangular maps (HDRI, baked sky clouds) wrap around in longitude and clamp at the poles.
        self.sampler_equirect: wgpu.GPUSampler = self.device.create_sampler(
#       self.sampler_equirect: wgpu.GPUSampler = self.device.create_sampler(
            mag_filter=wgpu.FilterMode.linear, min_filter=wgpu.FilterMode.linear,
#           mag_filter=wgpu.FilterMode.linear, min_filter=wgpu.FilterMode.linear,
            address_mode_u=wgpu.AddressMode.repeat, address_mode_v=wgpu.AddressMode.clamp_to_edge,
#           address_mode_u=wgpu.AddressMode.repeat, address_mode_v=wgpu.AddressMode.clamp_to_edge,
        )
#       )
        self.sampler_cloud_noise: wgpu.GPUSampler = self.device.create_sampler(
#       self.sampler_cloud_noise: wgpu.GPUSampler = self.device.create_sampler(
            mag_filter=wgpu.FilterMode.linear, min_filter=wgpu.FilterMode.linear,
#           mag_filter=wgpu.FilterMode.linear, min_filter=wgpu.FilterMode.linear,
            address_mode_u=wgpu.AddressMode.repeat, address_mode_v=wgpu.AddressMode.repeat, address_mode_w=wgpu.AddressMode.repeat,
#           address_mode_u=wgpu.AddressMode.repeat, address_mode_v=wgpu.AddressMode.repeat, address_mode_w=wgpu.AddressMode.repeat,
        )
#       )
        # Trilinear + anisotropic filtering keeps textures sharp at grazing angles.
#       # Trilinear + anisotropic filtering keeps textures sharp at grazing angles.
        self.sampler_texture_array: wgpu.GPUSampler = self.device.create_sampler(
#       self.sampler_texture_array: wgpu.GPUSampler = self.device.create_sampler(
            mag_filter=wgpu.FilterMode.linear, min_filter=wgpu.FilterMode.linear, mipmap_filter=wgpu.MipmapFilterMode.linear,
#           mag_filter=wgpu.FilterMode.linear, min_filter=wgpu.FilterMode.linear, mipmap_filter=wgpu.MipmapFilterMode.linear,
            address_mode_u=wgpu.AddressMode.repeat, address_mode_v=wgpu.AddressMode.repeat,
#           address_mode_u=wgpu.AddressMode.repeat, address_mode_v=wgpu.AddressMode.repeat,
            max_anisotropy=16,
#           max_anisotropy=16,
        )
#       )

        # -----------------------------
#       # -----------------------------
        # 1. G-Buffer Setup
#       # 1. G-Buffer Setup
        # -----------------------------
#       # -----------------------------
        # Create textures for the Geometric Buffer (G-Buffer).
#       # Create textures for the Geometric Buffer (G-Buffer).
        # These store the geometric properties of the visible surface at each pixel.
#       # These store the geometric properties of the visible surface at each pixel.
        # COPY_SRC on albedo / normal lets denoise_and_show read them back as OIDN auxiliary images.
#       # COPY_SRC on albedo / normal lets denoise_and_show read them back as OIDN auxiliary images.
        gbuffer_usage: int = wgpu.TextureUsage.RENDER_ATTACHMENT | wgpu.TextureUsage.TEXTURE_BINDING | wgpu.TextureUsage.COPY_SRC
#       gbuffer_usage: int = wgpu.TextureUsage.RENDER_ATTACHMENT | wgpu.TextureUsage.TEXTURE_BINDING | wgpu.TextureUsage.COPY_SRC
        self.texture_geometry_global_position: wgpu.GPUTexture = self.create_texture(size=self.window_size, texture_format=wgpu.TextureFormat.rgba32float, usage=gbuffer_usage, label="gbuffer position")
#       self.texture_geometry_global_position: wgpu.GPUTexture = self.create_texture(size=self.window_size, texture_format=wgpu.TextureFormat.rgba32float, usage=gbuffer_usage, label="gbuffer position")
        self.texture_geometry_global_normal: wgpu.GPUTexture = self.create_texture(size=self.window_size, texture_format=wgpu.TextureFormat.rgba16float, usage=gbuffer_usage, label="gbuffer normal")
#       self.texture_geometry_global_normal: wgpu.GPUTexture = self.create_texture(size=self.window_size, texture_format=wgpu.TextureFormat.rgba16float, usage=gbuffer_usage, label="gbuffer normal")
        self.texture_geometry_albedo: wgpu.GPUTexture = self.create_texture(size=self.window_size, texture_format=wgpu.TextureFormat.rgba8unorm, usage=gbuffer_usage, label="gbuffer albedo")
#       self.texture_geometry_albedo: wgpu.GPUTexture = self.create_texture(size=self.window_size, texture_format=wgpu.TextureFormat.rgba8unorm, usage=gbuffer_usage, label="gbuffer albedo")
        self.texture_geometry_global_tangent: wgpu.GPUTexture = self.create_texture(size=self.window_size, texture_format=wgpu.TextureFormat.rgba16float, usage=gbuffer_usage, label="gbuffer tangent")
#       self.texture_geometry_global_tangent: wgpu.GPUTexture = self.create_texture(size=self.window_size, texture_format=wgpu.TextureFormat.rgba16float, usage=gbuffer_usage, label="gbuffer tangent")
        self.texture_depth: wgpu.GPUTexture = self.create_texture(size=self.window_size, texture_format=wgpu.TextureFormat.depth24plus, usage=wgpu.TextureUsage.RENDER_ATTACHMENT, label="gbuffer depth")
#       self.texture_depth: wgpu.GPUTexture = self.create_texture(size=self.window_size, texture_format=wgpu.TextureFormat.depth24plus, usage=wgpu.TextureUsage.RENDER_ATTACHMENT, label="gbuffer depth")

        # Render targets of the geometry pass, in the SV_Target order of hybrid_geometry_fs.slang.
#       # Render targets of the geometry pass, in the SV_Target order of hybrid_geometry_fs.slang.
        self.gbuffer: list[wgpu.GPUTexture] = [
#       self.gbuffer: list[wgpu.GPUTexture] = [
            self.texture_geometry_global_position,
#           self.texture_geometry_global_position,
            self.texture_geometry_global_normal,
#           self.texture_geometry_global_normal,
            self.texture_geometry_albedo,
#           self.texture_geometry_albedo,
            self.texture_geometry_global_tangent,
#           self.texture_geometry_global_tangent,
        ]
#       ]

        # Post-processing ping-pong pair in rgba16f (f2): everything after the tonemap pass is
#       # Post-processing ping-pong pair in rgba16f (f2): everything after the tonemap pass is
        # gamma-encoded LDR in [0, 1], where half floats resolve ~1/2048 steps vs the display's
#       # gamma-encoded LDR in [0, 1], where half floats resolve ~1/2048 steps vs the display's
        # 1/255 - visually identical, at half the memory traffic for every post pass and blit.
#       # 1/255 - visually identical, at half the memory traffic for every post pass and blit.
        screen_usage: int = wgpu.TextureUsage.STORAGE_BINDING | wgpu.TextureUsage.TEXTURE_BINDING
#       screen_usage: int = wgpu.TextureUsage.STORAGE_BINDING | wgpu.TextureUsage.TEXTURE_BINDING
        self.texture_output: wgpu.GPUTexture = self.create_texture(size=self.window_size, texture_format=wgpu.TextureFormat.rgba16float, usage=screen_usage, label="post output")
#       self.texture_output: wgpu.GPUTexture = self.create_texture(size=self.window_size, texture_format=wgpu.TextureFormat.rgba16float, usage=screen_usage, label="post output")
        self.texture_ping: wgpu.GPUTexture = self.create_texture(size=self.window_size, texture_format=wgpu.TextureFormat.rgba16float, usage=screen_usage, label="post ping")
#       self.texture_ping: wgpu.GPUTexture = self.create_texture(size=self.window_size, texture_format=wgpu.TextureFormat.rgba16float, usage=screen_usage, label="post ping")

        # The accumulation history MUST stay f4: it averages thousands of HDR samples with blend
#       # The accumulation history MUST stay f4: it averages thousands of HDR samples with blend
        # weights down to 0.001, which half-float precision would visibly stall and band.
#       # weights down to 0.001, which half-float precision would visibly stall and band.
        # WebGPU only grants read-write storage access to single-channel 32-bit formats, so the history is
#       # WebGPU only grants read-write storage access to single-channel 32-bit formats, so the history is
        # a ping-pong pair: the shading pass reads last frame's result and writes this frame's into the other
#       # a ping-pong pair: the shading pass reads last frame's result and writes this frame's into the other
        # texture. texture_accum always names the most recently written one.
#       # texture. texture_accum always names the most recently written one.
        self.texture_accum_pair: list[wgpu.GPUTexture] = [
#       self.texture_accum_pair: list[wgpu.GPUTexture] = [
            self.create_texture(size=self.window_size, texture_format=wgpu.TextureFormat.rgba32float, usage=screen_usage | wgpu.TextureUsage.COPY_SRC, label=f"accumulation {accum_index}")
#           self.create_texture(size=self.window_size, texture_format=wgpu.TextureFormat.rgba32float, usage=screen_usage | wgpu.TextureUsage.COPY_SRC, label=f"accumulation {accum_index}")
            for accum_index in range(2)
#           for accum_index in range(2)
        ]
#       ]
        self.texture_accum: wgpu.GPUTexture = self.texture_accum_pair[0]
#       self.texture_accum: wgpu.GPUTexture = self.texture_accum_pair[0]

        # 1x1 stand-in for sampled parameters a pass declares but must not alias with its own output
#       # 1x1 stand-in for sampled parameters a pass declares but must not alias with its own output
        # (the sky cloud bake writes texture_sky_clouds, so it cannot also sample it).
#       # (the sky cloud bake writes texture_sky_clouds, so it cannot also sample it).
        self.texture_dummy: wgpu.GPUTexture = self.create_texture(size=(1, 1), texture_format=wgpu.TextureFormat.rgba16float, usage=wgpu.TextureUsage.TEXTURE_BINDING, label="dummy")
#       self.texture_dummy: wgpu.GPUTexture = self.create_texture(size=(1, 1), texture_format=wgpu.TextureFormat.rgba16float, usage=wgpu.TextureUsage.TEXTURE_BINDING, label="dummy")

        # OIDN Setup
#       # OIDN Setup
        self.oidn_device: pyoidn.Device = pyoidn.Device(device_type=pyoidn.OIDN_DEVICE_TYPE_CPU)
#       self.oidn_device: pyoidn.Device = pyoidn.Device(device_type=pyoidn.OIDN_DEVICE_TYPE_CPU)
        self.oidn_device.commit()
#       self.oidn_device.commit()

        # HDRI Texture Loading
#       # HDRI Texture Loading
        self.texture_hdri: wgpu.GPUTexture = typing.cast(wgpu.GPUTexture, None)
#       self.texture_hdri: wgpu.GPUTexture = typing.cast(wgpu.GPUTexture, None)
        self.use_hdri: bool = False
#       self.use_hdri: bool = False

        # Texture Array for Scene Materials
#       # Texture Array for Scene Materials
        # The array is created lazily in upload_pending_textures() once we know exactly
#       # The array is created lazily in upload_pending_textures() once we know exactly
        # how many layers are needed, so no VRAM is wasted on unused layers.
#       # how many layers are needed, so no VRAM is wasted on unused layers.
        self.texture_array_size: int = 2048
#       self.texture_array_size: int = 2048
        # Full mip chain down to 1x1; WebGPU has no glGenerateMipmap, so the levels are built on the CPU per layer.
#       # Full mip chain down to 1x1; WebGPU has no glGenerateMipmap, so the levels are built on the CPU per layer.
        self.texture_array_mip_count: int = int(np.log2(self.texture_array_size)) + 1
#       self.texture_array_mip_count: int = int(np.log2(self.texture_array_size)) + 1
        self.texture_array: wgpu.GPUTexture = typing.cast(wgpu.GPUTexture, None)
#       self.texture_array: wgpu.GPUTexture = typing.cast(wgpu.GPUTexture, None)

        self.texture_cache: dict[str, float] = {}
#       self.texture_cache: dict[str, float] = {}
        self.next_texture_layer: int = 0
#       self.next_texture_layer: int = 0
        # Deferred uploads: (path, layer_index, is_srgb) registered by load_texture()
#       # Deferred uploads: (path, layer_index, is_srgb) registered by load_texture()
        self.pending_textures: list[tuple[pl.Path, int, bool]] = []
#       self.pending_textures: list[tuple[pl.Path, int, bool]] = []
        # Deferred packed uploads: (layer_index, roughness_path, metallic_path, transmission_path)
#       # Deferred packed uploads: (layer_index, roughness_path, metallic_path, transmission_path)
        self.pending_packed_textures: list[tuple[int, pl.Path | None, pl.Path | None, pl.Path | None]] = []
#       self.pending_packed_textures: list[tuple[int, pl.Path | None, pl.Path | None, pl.Path | None]] = []

        # -----------------------------
#       # -----------------------------
        # 2. Rasterization Shader (Geometry Pass)
#       # 2. Rasterization Shader (Geometry Pass)
        # -----------------------------
#       # -----------------------------
        # One interleaved vertex stream of three float4 per vertex (see SceneBuilder.build).
#       # One interleaved vertex stream of three float4 per vertex (see SceneBuilder.build).
        # Depth test + back-face culling as in GL; the front face is clockwise because hybrid_geometry_vs.slang
#       # Depth test + back-face culling as in GL; the front face is clockwise because hybrid_geometry_vs.slang
        # mirrors clip-space y to keep OpenGL's bottom-up G-Buffer row order, which reverses the winding.
#       # mirrors clip-space y to keep OpenGL's bottom-up G-Buffer row order, which reverses the winding.
        self.program_geometry: RenderProgram = RenderProgram(
#       self.program_geometry: RenderProgram = RenderProgram(
            device=self.device,
#           device=self.device,
            shader_dir=self.shader_dir,
#           shader_dir=self.shader_dir,
            vertex_file_name="hybrid_geometry_vs.slang",
#           vertex_file_name="hybrid_geometry_vs.slang",
            fragment_file_name="hybrid_geometry_fs.slang",
#           fragment_file_name="hybrid_geometry_fs.slang",
            vertex_buffers=[{
#           vertex_buffers=[{
                "array_stride": 48,
#               "array_stride": 48,
                "step_mode": wgpu.VertexStepMode.vertex,
#               "step_mode": wgpu.VertexStepMode.vertex,
                "attributes": [
#               "attributes": [
                    {"format": wgpu.VertexFormat.float32x4, "offset": 0, "shader_location": 0},
#                   {"format": wgpu.VertexFormat.float32x4, "offset": 0, "shader_location": 0},
                    {"format": wgpu.VertexFormat.float32x4, "offset": 16, "shader_location": 1},
#                   {"format": wgpu.VertexFormat.float32x4, "offset": 16, "shader_location": 1},
                    {"format": wgpu.VertexFormat.float32x4, "offset": 32, "shader_location": 2},
#                   {"format": wgpu.VertexFormat.float32x4, "offset": 32, "shader_location": 2},
                ],
#               ],
            }],
#           }],
            color_formats=[texture.format for texture in self.gbuffer],
#           color_formats=[texture.format for texture in self.gbuffer],
            primitive={"topology": wgpu.PrimitiveTopology.triangle_list, "front_face": wgpu.FrontFace.cw, "cull_mode": wgpu.CullMode.back},
#           primitive={"topology": wgpu.PrimitiveTopology.triangle_list, "front_face": wgpu.FrontFace.cw, "cull_mode": wgpu.CullMode.back},
            depth_stencil={"format": wgpu.TextureFormat.depth24plus, "depth_write_enabled": True, "depth_compare": wgpu.CompareFunction.less},
#           depth_stencil={"format": wgpu.TextureFormat.depth24plus, "depth_write_enabled": True, "depth_compare": wgpu.CompareFunction.less},
        )
#       )

        # -----------------------------
#       # -----------------------------
        # 3. Compute Shader (Shading Pass)
#       # 3. Compute Shader (Shading Pass)
        # -----------------------------
#       # -----------------------------
        self.program_shading: ComputeProgram = ComputeProgram(device=self.device, shader_dir=self.shader_dir, file_name="hybrid_shading_cs.slang")
#       self.program_shading: ComputeProgram = ComputeProgram(device=self.device, shader_dir=self.shader_dir, file_name="hybrid_shading_cs.slang")

        # -----------------------------
#       # -----------------------------
        # 3b. Compute Shader (NRC Training Passes)
#       # 3b. Compute Shader (NRC Training Passes)
        # -----------------------------
#       # -----------------------------
        self.program_nrc_gather: ComputeProgram = ComputeProgram(device=self.device, shader_dir=self.shader_dir, file_name="nrc_gather_cs.slang")
#       self.program_nrc_gather: ComputeProgram = ComputeProgram(device=self.device, shader_dir=self.shader_dir, file_name="nrc_gather_cs.slang")

        self.program_nrc_train: ComputeProgram = ComputeProgram(device=self.device, shader_dir=self.shader_dir, file_name="nrc_train_cs.slang")
#       self.program_nrc_train: ComputeProgram = ComputeProgram(device=self.device, shader_dir=self.shader_dir, file_name="nrc_train_cs.slang")

        # -----------------------------
#       # -----------------------------
        # 3c. Compute Shaders (Volumetrics)
#       # 3c. Compute Shaders (Volumetrics)
        # -----------------------------
#       # -----------------------------
        self.program_cloud_noise: ComputeProgram = ComputeProgram(device=self.device, shader_dir=self.shader_dir, file_name="cloud_noise_cs.slang")
#       self.program_cloud_noise: ComputeProgram = ComputeProgram(device=self.device, shader_dir=self.shader_dir, file_name="cloud_noise_cs.slang")

        self.program_sky_clouds_bake: ComputeProgram = ComputeProgram(device=self.device, shader_dir=self.shader_dir, file_name="sky_clouds_bake_cs.slang")
#       self.program_sky_clouds_bake: ComputeProgram = ComputeProgram(device=self.device, shader_dir=self.shader_dir, file_name="sky_clouds_bake_cs.slang")

        # Tileable Perlin-Worley / Worley noise volume shared by the sky cloud layer and every
#       # Tileable Perlin-Worley / Worley noise volume shared by the sky cloud layer and every
        # cloud-shaped scene volume. It never changes, so it is generated once, here, and every
#       # cloud-shaped scene volume. It never changes, so it is generated once, here, and every
        # ray-march step afterwards is a single hardware-filtered fetch instead of procedural noise.
#       # ray-march step afterwards is a single hardware-filtered fetch instead of procedural noise.
        cloud_noise_size: int = 128
#       cloud_noise_size: int = 128
        self.texture_cloud_noise: wgpu.GPUTexture = self.device.create_texture(
#       self.texture_cloud_noise: wgpu.GPUTexture = self.device.create_texture(
            label="cloud noise",
#           label="cloud noise",
            size=(cloud_noise_size, cloud_noise_size, cloud_noise_size),
#           size=(cloud_noise_size, cloud_noise_size, cloud_noise_size),
            dimension=wgpu.TextureDimension.d3,
#           dimension=wgpu.TextureDimension.d3,
            format=wgpu.TextureFormat.rgba8unorm,
#           format=wgpu.TextureFormat.rgba8unorm,
            usage=wgpu.TextureUsage.STORAGE_BINDING | wgpu.TextureUsage.TEXTURE_BINDING,
#           usage=wgpu.TextureUsage.STORAGE_BINDING | wgpu.TextureUsage.TEXTURE_BINDING,
        )
#       )
        self.program_cloud_noise["uNoiseSize"] = cloud_noise_size
#       self.program_cloud_noise["uNoiseSize"] = cloud_noise_size
        cloud_noise_groups: int = (cloud_noise_size + 7) // 8
#       cloud_noise_groups: int = (cloud_noise_size + 7) // 8
        command_encoder: wgpu.GPUCommandEncoder = self.device.create_command_encoder(label="cloud noise")
#       command_encoder: wgpu.GPUCommandEncoder = self.device.create_command_encoder(label="cloud noise")
        self.program_cloud_noise.run(command_encoder=command_encoder, resources={"textureCloudNoise": self.texture_cloud_noise}, group_x=cloud_noise_groups, group_y=cloud_noise_groups, group_z=cloud_noise_groups)
#       self.program_cloud_noise.run(command_encoder=command_encoder, resources={"textureCloudNoise": self.texture_cloud_noise}, group_x=cloud_noise_groups, group_y=cloud_noise_groups, group_z=cloud_noise_groups)
        self.device.queue.submit([command_encoder.finish()])
#       self.device.queue.submit([command_encoder.finish()])

        # Baked sky cloud layer, equirectangular like the HDRI (rgb: in-scattering, a: transmittance).
#       # Baked sky cloud layer, equirectangular like the HDRI (rgb: in-scattering, a: transmittance).
        # Only secondary bounces and NRC training read it, so half the HDRI's typical resolution is plenty.
#       # Only secondary bounces and NRC training read it, so half the HDRI's typical resolution is plenty.
        self.texture_sky_clouds: wgpu.GPUTexture = self.create_texture(size=(1024, 512), texture_format=wgpu.TextureFormat.rgba16float, usage=screen_usage, label="sky clouds")
#       self.texture_sky_clouds: wgpu.GPUTexture = self.create_texture(size=(1024, 512), texture_format=wgpu.TextureFormat.rgba16float, usage=screen_usage, label="sky clouds")
        # Whether the current frame composites the sky cloud layer (toggle, scene, and no HDRI all agree).
#       # Whether the current frame composites the sky cloud layer (toggle, scene, and no HDRI all agree).
        self.is_sky_clouds_active: bool = False
#       self.is_sky_clouds_active: bool = False

        # -----------------------------
#       # -----------------------------
        # 4. Rasterization Shader (Renderer Pass)
#       # 4. Rasterization Shader (Renderer Pass)
        # -----------------------------
#       # -----------------------------
        self.program_renderer: RenderProgram = RenderProgram(
#       self.program_renderer: RenderProgram = RenderProgram(
            device=self.device,
#           device=self.device,
            shader_dir=self.shader_dir,
#           shader_dir=self.shader_dir,
            vertex_file_name="hybrid_renderer_vs.slang",
#           vertex_file_name="hybrid_renderer_vs.slang",
            fragment_file_name="hybrid_renderer_fs.slang",
#           fragment_file_name="hybrid_renderer_fs.slang",
            vertex_buffers=[{
#           vertex_buffers=[{
                "array_stride": 16,
#               "array_stride": 16,
                "step_mode": wgpu.VertexStepMode.vertex,
#               "step_mode": wgpu.VertexStepMode.vertex,
                "attributes": [
#               "attributes": [
                    {"format": wgpu.VertexFormat.float32x2, "offset": 0, "shader_location": 0},
#                   {"format": wgpu.VertexFormat.float32x2, "offset": 0, "shader_location": 0},
                    {"format": wgpu.VertexFormat.float32x2, "offset": 8, "shader_location": 1},
#                   {"format": wgpu.VertexFormat.float32x2, "offset": 8, "shader_location": 1},
                ],
#               ],
            }],
#           }],
            color_formats=[self.present_format],
#           color_formats=[self.present_format],
            primitive={"topology": wgpu.PrimitiveTopology.triangle_strip, "cull_mode": wgpu.CullMode.none},
#           primitive={"topology": wgpu.PrimitiveTopology.triangle_strip, "cull_mode": wgpu.CullMode.none},
        )
#       )
        # Screen data (x, y, u, v)
#       # Screen data (x, y, u, v)
        screen_data: npt.NDArray[np.float32] = np.array([
#       screen_data: npt.NDArray[np.float32] = np.array([
            -1.0, -1.0,  0.0,  0.0,
#           -1.0, -1.0,  0.0,  0.0,
             1.0, -1.0,  1.0,  0.0,
#            1.0, -1.0,  1.0,  0.0,
            -1.0,  1.0,  0.0,  1.0,
#           -1.0,  1.0,  0.0,  1.0,
             1.0,  1.0,  1.0,  1.0,
#            1.0,  1.0,  1.0,  1.0,
        ], dtype=np.float32)
#       ], dtype=np.float32)
        self.vbo_screen: wgpu.GPUBuffer = self.create_buffer(data=screen_data.tobytes(), usage=wgpu.BufferUsage.VERTEX, label="screen quad")
#       self.vbo_screen: wgpu.GPUBuffer = self.create_buffer(data=screen_data.tobytes(), usage=wgpu.BufferUsage.VERTEX, label="screen quad")

        # -----------------------------
#       # -----------------------------
        # 5. Scene Management
#       # 5. Scene Management
        # -----------------------------
#       # -----------------------------
        self.materials: list[Material] = []
#       self.materials: list[Material] = []
        self.scene_builder: SceneBuilder = typing.cast(SceneBuilder, None)
#       self.scene_builder: SceneBuilder = typing.cast(SceneBuilder, None)
        self.num_triangles: int = 0
#       self.num_triangles: int = 0
        self.ssbo_bvh_nodes: wgpu.GPUBuffer = typing.cast(wgpu.GPUBuffer, None)
#       self.ssbo_bvh_nodes: wgpu.GPUBuffer = typing.cast(wgpu.GPUBuffer, None)
        self.buffer_global_vertices: wgpu.GPUBuffer = typing.cast(wgpu.GPUBuffer, None)
#       self.buffer_global_vertices: wgpu.GPUBuffer = typing.cast(wgpu.GPUBuffer, None)
        self.ssbo_materials: wgpu.GPUBuffer = typing.cast(wgpu.GPUBuffer, None)
#       self.ssbo_materials: wgpu.GPUBuffer = typing.cast(wgpu.GPUBuffer, None)
        self.nrc_position_offset: vec3f32 = (0.0, 0.0, 0.0)
#       self.nrc_position_offset: vec3f32 = (0.0, 0.0, 0.0)
        self.nrc_position_scale: float = 1.0
#       self.nrc_position_scale: float = 1.0
        cache_entry_count: int = 131072
#       cache_entry_count: int = 131072
        # 3 vec4s per entry (radiance+count, position+key, normal+frame): the reserved 4th vec4
#       # 3 vec4s per entry (radiance+count, position+key, normal+frame): the reserved 4th vec4
        # was never read or needed for std430 alignment, so dropping it cuts every cache
#       # was never read or needed for std430 alignment, so dropping it cuts every cache
        # read/write in the bounce loop by 25% and shrinks the buffer 8MB -> 6MB.
#       # read/write in the bounce loop by 25% and shrinks the buffer 8MB -> 6MB.
        cache_floats_per_entry: int = 12
#       cache_floats_per_entry: int = 12
        self.ssbo_radiance_cache: wgpu.GPUBuffer = self.create_buffer(data=np.zeros(cache_entry_count * cache_floats_per_entry, dtype=np.float32).tobytes(), label="radiance cache")
#       self.ssbo_radiance_cache: wgpu.GPUBuffer = self.create_buffer(data=np.zeros(cache_entry_count * cache_floats_per_entry, dtype=np.float32).tobytes(), label="radiance cache")

        # 4. Spatiotemporal ReSTIR Reservoirs Buffer (double-buffered: 2 frames)
#       # 4. Spatiotemporal ReSTIR Reservoirs Buffer (double-buffered: 2 frames)
        reservoir_floats_per_item: int = 20
#       reservoir_floats_per_item: int = 20
        w_res, h_res = self.window_size
#       w_res, h_res = self.window_size
        self.ssbo_sample_reservoirs: wgpu.GPUBuffer = self.create_buffer(data=np.zeros(w_res * h_res * 2 * reservoir_floats_per_item, dtype=np.float32).tobytes(), label="restir reservoirs")
#       self.ssbo_sample_reservoirs: wgpu.GPUBuffer = self.create_buffer(data=np.zeros(w_res * h_res * 2 * reservoir_floats_per_item, dtype=np.float32).tobytes(), label="restir reservoirs")

        # 5. Neural Radiance Cache (NRC) Weights Buffer: 4556 floats total
#       # 5. Neural Radiance Cache (NRC) Weights Buffer: 4556 floats total
        self.ssbo_nrc_weights: wgpu.GPUBuffer = self.create_buffer(data=np.zeros(4556, dtype=np.float32).tobytes(), label="nrc weights")
#       self.ssbo_nrc_weights: wgpu.GPUBuffer = self.create_buffer(data=np.zeros(4556, dtype=np.float32).tobytes(), label="nrc weights")

        # 6. NRC Training Records Buffer: 8192 records * 9 floats, sized to match the 8192 paths the
#       # 6. NRC Training Records Buffer: 8192 records * 9 floats, sized to match the 8192 paths the
        #    gather pass dispatches (see program_nrc_gather.run below); one record per path maximum.
#       #    gather pass dispatches (see program_nrc_gather.run below); one record per path maximum.
        self.ssbo_nrc_training_records: wgpu.GPUBuffer = self.create_buffer(data=np.zeros(8192 * 9, dtype=np.float32).tobytes(), label="nrc training records")
#       self.ssbo_nrc_training_records: wgpu.GPUBuffer = self.create_buffer(data=np.zeros(8192 * 9, dtype=np.float32).tobytes(), label="nrc training records")

        # 7. NRC Training Counter Buffer: 1 uint32 atomic counter
#       # 7. NRC Training Counter Buffer: 1 uint32 atomic counter
        self.ssbo_nrc_training_count: wgpu.GPUBuffer = self.create_buffer(data=np.zeros(1, dtype=np.uint32).tobytes(), label="nrc training count")
#       self.ssbo_nrc_training_count: wgpu.GPUBuffer = self.create_buffer(data=np.zeros(1, dtype=np.uint32).tobytes(), label="nrc training count")

        self.reset_neural_cache_weights()
#       self.reset_neural_cache_weights()

        # -----------------------------
#       # -----------------------------
        # 6. Post Processing System
#       # 6. Post Processing System
        # -----------------------------
#       # -----------------------------
        self.program_post_chromatic_aberration: ComputeProgram = ComputeProgram(device=self.device, shader_dir=self.shader_dir, file_name="post_chromatic_aberration_cs.slang")
#       self.program_post_chromatic_aberration: ComputeProgram = ComputeProgram(device=self.device, shader_dir=self.shader_dir, file_name="post_chromatic_aberration_cs.slang")

        self.program_post_vignette: ComputeProgram = ComputeProgram(device=self.device, shader_dir=self.shader_dir, file_name="post_vignette_cs.slang")
#       self.program_post_vignette: ComputeProgram = ComputeProgram(device=self.device, shader_dir=self.shader_dir, file_name="post_vignette_cs.slang")

        self.program_post_tonemap: ComputeProgram = ComputeProgram(device=self.device, shader_dir=self.shader_dir, file_name="post_tonemap_cs.slang")
#       self.program_post_tonemap: ComputeProgram = ComputeProgram(device=self.device, shader_dir=self.shader_dir, file_name="post_tonemap_cs.slang")

        self.program_post_fxaa: ComputeProgram = ComputeProgram(device=self.device, shader_dir=self.shader_dir, file_name="post_fxaa_cs.slang")
#       self.program_post_fxaa: ComputeProgram = ComputeProgram(device=self.device, shader_dir=self.shader_dir, file_name="post_fxaa_cs.slang")

        # Slang ignores initialisers on uniforms, so the defaults the GLSL declarations carried are staged here once.
#       # Slang ignores initialisers on uniforms, so the defaults the GLSL declarations carried are staged here once.
        self.program_post_chromatic_aberration["uAberrationStrength"] = 0.01
#       self.program_post_chromatic_aberration["uAberrationStrength"] = 0.01
        self.program_post_vignette["uVignetteIntensity"] = 1.0
#       self.program_post_vignette["uVignetteIntensity"] = 1.0
        self.program_post_vignette["uFalloffSmoothness"] = 0.5
#       self.program_post_vignette["uFalloffSmoothness"] = 0.5
        self.program_post_vignette["uVignetteColor"] = (0.0, 0.0, 0.0)
#       self.program_post_vignette["uVignetteColor"] = (0.0, 0.0, 0.0)

        # Pipeline of active post-processing shaders
#       # Pipeline of active post-processing shaders
        # ORDER CONSTRAINT: tonemap must stay FIRST. It is the only pass whose input is the f4
#       # ORDER CONSTRAINT: tonemap must stay FIRST. It is the only pass whose input is the f4
        # accumulation buffer; every later pass reads the f2 ping-pong pair.
#       # accumulation buffer; every later pass reads the f2 ping-pong pair.
        self.post_processing_pipeline: list[ComputeProgram] = [
#       self.post_processing_pipeline: list[ComputeProgram] = [
            self.program_post_tonemap,
#           self.program_post_tonemap,
            self.program_post_fxaa,
#           self.program_post_fxaa,
            self.program_post_chromatic_aberration,
#           self.program_post_chromatic_aberration,
            self.program_post_vignette,
#           self.program_post_vignette,
        ]
#       ]

        # Key states for first-person camera movement, kept current by on_key_down / on_key_up
#       # Key states for first-person camera movement, kept current by on_key_down / on_key_up
        self.key_state: dict[str, bool] = {
#       self.key_state: dict[str, bool] = {
            "W": False, "A": False, "S": False, "D": False,
#           "W": False, "A": False, "S": False, "D": False,
            "Q": False, "E": False,
#           "Q": False, "E": False,
            "UP": False, "DOWN": False, "LEFT": False, "RIGHT": False,
#           "UP": False, "DOWN": False, "LEFT": False, "RIGHT": False,
        }
#       }

        # Camera instance (initialized by set_scene)
#       # Camera instance (initialized by set_scene)
        self.camera: Camera = typing.cast(Camera, None)
#       self.camera: Camera = typing.cast(Camera, None)

        # Scenes Collection
#       # Scenes Collection
        self.scenes: list[BaseScene] = [
#       self.scenes: list[BaseScene] = [
            OfficeScene(resource_dir=self.resource_dir),
#           OfficeScene(resource_dir=self.resource_dir),
            CornellBoxScene(resource_dir=self.resource_dir),
#           CornellBoxScene(resource_dir=self.resource_dir),
            ShowcaseScene(resource_dir=self.resource_dir),
#           ShowcaseScene(resource_dir=self.resource_dir),
        ]
#       ]
        self.current_scene_index: int = 0
#       self.current_scene_index: int = 0
        self.current_scene: BaseScene = typing.cast(BaseScene, None)
#       self.current_scene: BaseScene = typing.cast(BaseScene, None)
        self.set_scene(scene=self.scenes[self.current_scene_index])
#       self.set_scene(scene=self.scenes[self.current_scene_index])

        self.canvas.request_draw(self.on_draw)
#       self.canvas.request_draw(self.on_draw)
        pass
#       pass

    def run(self) -> None:
#   def run(self) -> None:
        # Hand control to the rendercanvas event loop, which calls on_draw every frame until the window closes.
#       # Hand control to the rendercanvas event loop, which calls on_draw every frame until the window closes.
        # The clock starts here rather than in __init__, so shader compilation and the first scene load
#       # The clock starts here rather than in __init__, so shader compilation and the first scene load
        # count neither toward the first frame_time nor toward time (as moderngl_window's timer did).
#       # count neither toward the first frame_time nor toward time (as moderngl_window's timer did).
        self.start_time = tm.perf_counter()
#       self.start_time = tm.perf_counter()
        self.last_frame_time = self.start_time
#       self.last_frame_time = self.start_time
        loop.run()
#       loop.run()
        pass
#       pass

    def create_texture(self, size: vec2i32, texture_format: str, usage: int, label: str) -> wgpu.GPUTexture:
#   def create_texture(self, size: vec2i32, texture_format: str, usage: int, label: str) -> wgpu.GPUTexture:
        return self.device.create_texture(label=label, size=(size[0], size[1], 1), format=texture_format, usage=usage)
#       return self.device.create_texture(label=label, size=(size[0], size[1], 1), format=texture_format, usage=usage)

    def create_buffer(self, data: bytes, label: str, usage: int = wgpu.BufferUsage.STORAGE) -> wgpu.GPUBuffer:
#   def create_buffer(self, data: bytes, label: str, usage: int = wgpu.BufferUsage.STORAGE) -> wgpu.GPUBuffer:
        # Storage buffer initialised with data. COPY_DST lets the scene / NRC resets rewrite it in place, and
#       # Storage buffer initialised with data. COPY_DST lets the scene / NRC resets rewrite it in place, and
        # COPY_SRC keeps it readable for debugging. Empty data still gets a 16-byte buffer, since WebGPU
#       # COPY_SRC keeps it readable for debugging. Empty data still gets a 16-byte buffer, since WebGPU
        # rejects zero-sized bindings.
#       # rejects zero-sized bindings.
        padded_data: bytes = data if len(data) >= 16 else data + bytes(16 - len(data))
#       padded_data: bytes = data if len(data) >= 16 else data + bytes(16 - len(data))
        return self.device.create_buffer_with_data(label=label, data=padded_data, usage=usage | wgpu.BufferUsage.COPY_DST | wgpu.BufferUsage.COPY_SRC)
#       return self.device.create_buffer_with_data(label=label, data=padded_data, usage=usage | wgpu.BufferUsage.COPY_DST | wgpu.BufferUsage.COPY_SRC)

    def get_shader_resources(self) -> dict[str, typing.Any]:
#   def get_shader_resources(self) -> dict[str, typing.Any]:
        # Every scene-wide buffer, texture and sampler, keyed by its Slang parameter name. Each program binds
#       # Every scene-wide buffer, texture and sampler, keyed by its Slang parameter name. Each program binds
        # the subset it declares, which plays the role GL's global binding points / texture units used to.
#       # the subset it declares, which plays the role GL's global binding points / texture units used to.
        return {
#       return {
            "textureGeometryGlobalPosition": self.texture_geometry_global_position,
#           "textureGeometryGlobalPosition": self.texture_geometry_global_position,
            "textureGeometryGlobalNormal": self.texture_geometry_global_normal,
#           "textureGeometryGlobalNormal": self.texture_geometry_global_normal,
            "textureGeometryAlbedo": self.texture_geometry_albedo,
#           "textureGeometryAlbedo": self.texture_geometry_albedo,
            "textureGeometryGlobalTangent": self.texture_geometry_global_tangent,
#           "textureGeometryGlobalTangent": self.texture_geometry_global_tangent,
            "nodes": self.ssbo_bvh_nodes,
#           "nodes": self.ssbo_bvh_nodes,
            "vertices": self.buffer_global_vertices,
#           "vertices": self.buffer_global_vertices,
            "materials": self.ssbo_materials,
#           "materials": self.ssbo_materials,
            "cacheEntries": self.ssbo_radiance_cache,
#           "cacheEntries": self.ssbo_radiance_cache,
            "sampleReservoirs": self.ssbo_sample_reservoirs,
#           "sampleReservoirs": self.ssbo_sample_reservoirs,
            "neuralNetworkWeights": self.ssbo_nrc_weights,
#           "neuralNetworkWeights": self.ssbo_nrc_weights,
            "nrcTrainingRecords": self.ssbo_nrc_training_records,
#           "nrcTrainingRecords": self.ssbo_nrc_training_records,
            "trainingRecordCounter": self.ssbo_nrc_training_count,
#           "trainingRecordCounter": self.ssbo_nrc_training_count,
            "uHdriTexture": (self.texture_hdri, self.sampler_equirect),
#           "uHdriTexture": (self.texture_hdri, self.sampler_equirect),
            "uSceneTextureArray": (self.texture_array, self.sampler_texture_array),
#           "uSceneTextureArray": (self.texture_array, self.sampler_texture_array),
            "uSkyCloudTexture": (self.texture_sky_clouds, self.sampler_equirect),
#           "uSkyCloudTexture": (self.texture_sky_clouds, self.sampler_equirect),
            "uCloudNoiseTexture": (self.texture_cloud_noise, self.sampler_cloud_noise),
#           "uCloudNoiseTexture": (self.texture_cloud_noise, self.sampler_cloud_noise),
        }
#       }

    def read_texture(self, texture: wgpu.GPUTexture, bytes_per_pixel: int) -> bytes:
#   def read_texture(self, texture: wgpu.GPUTexture, bytes_per_pixel: int) -> bytes:
        # Synchronous read-back of mip 0, rows in memory order (row 0 is the bottom of the image, as in GL).
#       # Synchronous read-back of mip 0, rows in memory order (row 0 is the bottom of the image, as in GL).
        width, height = texture.size[0], texture.size[1]
#       width, height = texture.size[0], texture.size[1]
        texture_data: typing.Any = self.device.queue.read_texture(
#       texture_data: typing.Any = self.device.queue.read_texture(
            source={"texture": texture, "origin": (0, 0, 0), "mip_level": 0},
#           source={"texture": texture, "origin": (0, 0, 0), "mip_level": 0},
            data_layout={"offset": 0, "bytes_per_row": width * bytes_per_pixel, "rows_per_image": height},
#           data_layout={"offset": 0, "bytes_per_row": width * bytes_per_pixel, "rows_per_image": height},
            size=(width, height, 1),
#           size=(width, height, 1),
        )
#       )
        return bytes(texture_data)
#       return bytes(texture_data)

    def get_halton_jitter(self, index: int, base: int) -> float:
#   def get_halton_jitter(self, index: int, base: int) -> float:
        result: float = 0.0
#       result: float = 0.0
        f: float = 1.0 / base
#       f: float = 1.0 / base
        i: int = index
#       i: int = index
        while i > 0:
#       while i > 0:
            result += f * (i % base)
#           result += f * (i % base)
            i //= base
#           i //= base
            f /= base
#           f /= base
        return result
#       return result

    def load_texture(self, path: pl.Path, is_srgb: bool = False) -> float:
#   def load_texture(self, path: pl.Path, is_srgb: bool = False) -> float:
        # Registers a texture for deferred upload and returns its layer index.
#       # Registers a texture for deferred upload and returns its layer index.
        # The actual GPU upload happens later in upload_pending_textures().
#       # The actual GPU upload happens later in upload_pending_textures().
        path_str = str(path.resolve())
#       path_str = str(path.resolve())
        if path_str in self.texture_cache:
#       if path_str in self.texture_cache:
            return self.texture_cache[path_str]
#           return self.texture_cache[path_str]

        if not path.exists():
#       if not path.exists():
            print(f"Warning: Texture not found: {path}")
#           print(f"Warning: Texture not found: {path}")
            return -1.0
#           return -1.0

        layer_index = self.next_texture_layer
#       layer_index = self.next_texture_layer
        self.texture_cache[path_str] = float(layer_index)
#       self.texture_cache[path_str] = float(layer_index)
        self.next_texture_layer += 1
#       self.next_texture_layer += 1
        self.pending_textures.append((path, layer_index, is_srgb))
#       self.pending_textures.append((path, layer_index, is_srgb))
        return float(layer_index)
#       return float(layer_index)

    def load_packed_orm(self, roughness_path: pl.Path | None = None, metallic_path: pl.Path | None = None, transmission_path: pl.Path | None = None) -> tuple[float, float, float]:
#   def load_packed_orm(self, roughness_path: pl.Path | None = None, metallic_path: pl.Path | None = None, transmission_path: pl.Path | None = None) -> tuple[float, float, float]:
        # Packs up to three single-channel data maps into ONE layer: roughness->R, metallic->G, transmission->B.
#       # Packs up to three single-channel data maps into ONE layer: roughness->R, metallic->G, transmission->B.
        # Returns (roughness_index, metallic_index, transmission_index): each is the shared layer index if that
#       # Returns (roughness_index, metallic_index, transmission_index): each is the shared layer index if that
        # map is present, else -1.0. Saves up to two layers per material vs. one layer per map.
#       # map is present, else -1.0. Saves up to two layers per material vs. one layer per map.
        def _present(p: pl.Path | None) -> pl.Path | None:
#       def _present(p: pl.Path | None) -> pl.Path | None:
            if p is None:
#           if p is None:
                return None
#               return None
            if not p.exists():
#           if not p.exists():
                print(f"Warning: Texture not found: {p}")
#               print(f"Warning: Texture not found: {p}")
                return None
#               return None
            return p
#           return p

        roughness_path = _present(roughness_path)
#       roughness_path = _present(roughness_path)
        metallic_path = _present(metallic_path)
#       metallic_path = _present(metallic_path)
        transmission_path = _present(transmission_path)
#       transmission_path = _present(transmission_path)
        if roughness_path is None and metallic_path is None and transmission_path is None:
#       if roughness_path is None and metallic_path is None and transmission_path is None:
            return (-1.0, -1.0, -1.0)
#           return (-1.0, -1.0, -1.0)

        cache_key = "orm:" + "|".join(str(p.resolve()) if p is not None else "-" for p in (roughness_path, metallic_path, transmission_path))
#       cache_key = "orm:" + "|".join(str(p.resolve()) if p is not None else "-" for p in (roughness_path, metallic_path, transmission_path))
        if cache_key in self.texture_cache:
#       if cache_key in self.texture_cache:
            shared = self.texture_cache[cache_key]
#           shared = self.texture_cache[cache_key]
        else:
#       else:
            shared = float(self.next_texture_layer)
#           shared = float(self.next_texture_layer)
            self.texture_cache[cache_key] = shared
#           self.texture_cache[cache_key] = shared
            self.next_texture_layer += 1
#           self.next_texture_layer += 1
            self.pending_packed_textures.append((int(shared), roughness_path, metallic_path, transmission_path))
#           self.pending_packed_textures.append((int(shared), roughness_path, metallic_path, transmission_path))

        return (
#       return (
            shared if roughness_path is not None else -1.0,
#           shared if roughness_path is not None else -1.0,
            shared if metallic_path is not None else -1.0,
#           shared if metallic_path is not None else -1.0,
            shared if transmission_path is not None else -1.0,
#           shared if transmission_path is not None else -1.0,
        )
#       )

    def _load_data_channel(self, path: pl.Path) -> npt.NDArray[np.float32] | None:
#   def _load_data_channel(self, path: pl.Path) -> npt.NDArray[np.float32] | None:
        # Loads a single-channel linear data map, returns it resized to the array size in [0, 1].
#       # Loads a single-channel linear data map, returns it resized to the array size in [0, 1].
        loaded_data = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
#       loaded_data = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if loaded_data is None:
#       if loaded_data is None:
            print(f"Warning: Failed to load texture: {path}")
#           print(f"Warning: Failed to load texture: {path}")
            return None
#           return None

        # Grayscale maps are 2D; for color-stored maps take the red channel (OpenCV is BGR, so index 2)
#       # Grayscale maps are 2D; for color-stored maps take the red channel (OpenCV is BGR, so index 2)
        channel = loaded_data if len(loaded_data.shape) == 2 else loaded_data[..., 2]
#       channel = loaded_data if len(loaded_data.shape) == 2 else loaded_data[..., 2]

        channel_float: npt.NDArray[np.float32]
#       channel_float: npt.NDArray[np.float32]
        if channel.dtype == np.uint8:
#       if channel.dtype == np.uint8:
            channel_float = channel.astype(dtype=np.float32) / 255.0
#           channel_float = channel.astype(dtype=np.float32) / 255.0
        elif channel.dtype == np.uint16:
#       elif channel.dtype == np.uint16:
            channel_float = channel.astype(dtype=np.float32) / 65535.0
#           channel_float = channel.astype(dtype=np.float32) / 65535.0
        else:
#       else:
            channel_float = channel.astype(dtype=np.float32)
#           channel_float = channel.astype(dtype=np.float32)

        if channel_float.shape[0] != self.texture_array_size or channel_float.shape[1] != self.texture_array_size:
#       if channel_float.shape[0] != self.texture_array_size or channel_float.shape[1] != self.texture_array_size:
            src_height, src_width = channel_float.shape[:2]
#           src_height, src_width = channel_float.shape[:2]
            is_shrinking: bool = self.texture_array_size < src_width or self.texture_array_size < src_height
#           is_shrinking: bool = self.texture_array_size < src_width or self.texture_array_size < src_height
            interpolation: int = cv2.INTER_AREA if is_shrinking else cv2.INTER_LINEAR
#           interpolation: int = cv2.INTER_AREA if is_shrinking else cv2.INTER_LINEAR
            channel_float = cv2.resize(channel_float, (self.texture_array_size, self.texture_array_size), interpolation=interpolation).astype(dtype=np.float32)
#           channel_float = cv2.resize(channel_float, (self.texture_array_size, self.texture_array_size), interpolation=interpolation).astype(dtype=np.float32)

        return channel_float
#       return channel_float

    def build_mip_chain(self, data_uint8: npt.NDArray[np.uint8]) -> list[npt.NDArray[np.uint8]]:
#   def build_mip_chain(self, data_uint8: npt.NDArray[np.uint8]) -> list[npt.NDArray[np.uint8]]:
        # One RGBA8 layer and its full mip chain. Each level is the 2x2 box average of the one above it, the same filter
#       # One RGBA8 layer and its full mip chain. Each level is the 2x2 box average of the one above it, the same filter
        # glGenerateMipmap applied to the 8-bit data. The average runs in 16-bit integers: (a + b + c + d + 2) >> 2 is
#       # glGenerateMipmap applied to the 8-bit data. The average runs in 16-bit integers: (a + b + c + d + 2) >> 2 is
        # exactly the old float32 floor((a + b + c + d) * 0.25 + 0.5) for 8-bit inputs (every intermediate is exact
#       # exactly the old float32 floor((a + b + c + d) * 0.25 + 0.5) for 8-bit inputs (every intermediate is exact
        # there), without converting each level to float32 and back.
#       # there), without converting each level to float32 and back.
        mip_chain: list[npt.NDArray[np.uint8]] = [np.ascontiguousarray(data_uint8)]
#       mip_chain: list[npt.NDArray[np.uint8]] = [np.ascontiguousarray(data_uint8)]
        for _ in range(1, self.texture_array_mip_count):
#       for _ in range(1, self.texture_array_mip_count):
            level_data: npt.NDArray[np.uint8] = mip_chain[-1]
#           level_data: npt.NDArray[np.uint8] = mip_chain[-1]
            if level_data.shape[0] == 1 or level_data.shape[1] == 1:
#           if level_data.shape[0] == 1 or level_data.shape[1] == 1:
                break
#               break
            level_sum: npt.NDArray[np.uint16] = level_data[0::2, 0::2].astype(np.uint16)
#           level_sum: npt.NDArray[np.uint16] = level_data[0::2, 0::2].astype(np.uint16)
            level_sum += level_data[1::2, 0::2]
#           level_sum += level_data[1::2, 0::2]
            level_sum += level_data[0::2, 1::2]
#           level_sum += level_data[0::2, 1::2]
            level_sum += level_data[1::2, 1::2]
#           level_sum += level_data[1::2, 1::2]
            level_sum += 2
#           level_sum += 2
            mip_chain.append(np.ascontiguousarray((level_sum >> 2).astype(np.uint8)))
#           mip_chain.append(np.ascontiguousarray((level_sum >> 2).astype(np.uint8)))
        return mip_chain
#       return mip_chain

    def write_texture_array_layer(self, mip_chain: list[npt.NDArray[np.uint8]], layer_index: int) -> None:
#   def write_texture_array_layer(self, mip_chain: list[npt.NDArray[np.uint8]], layer_index: int) -> None:
        # Uploads one layer of the texture array together with its mip chain (see build_mip_chain).
#       # Uploads one layer of the texture array together with its mip chain (see build_mip_chain).
        for mip_level, level_data in enumerate(mip_chain):
#       for mip_level, level_data in enumerate(mip_chain):
            level_height, level_width = level_data.shape[:2]
#           level_height, level_width = level_data.shape[:2]
            self.device.queue.write_texture(
#           self.device.queue.write_texture(
                destination={"texture": self.texture_array, "mip_level": mip_level, "origin": (0, 0, layer_index)},
#               destination={"texture": self.texture_array, "mip_level": mip_level, "origin": (0, 0, layer_index)},
                data=level_data,
#               data=level_data,
                data_layout={"offset": 0, "bytes_per_row": level_width * 4, "rows_per_image": level_height},
#               data_layout={"offset": 0, "bytes_per_row": level_width * 4, "rows_per_image": level_height},
                size=(level_width, level_height, 1),
#               size=(level_width, level_height, 1),
            )
#           )
        pass
#       pass

    def upload_pending_textures(self) -> None:
#   def upload_pending_textures(self) -> None:
        # Creates the texture array at exactly the size needed, then uploads every
#       # Creates the texture array at exactly the size needed, then uploads every
        # registered texture. Called once after all materials have been set up.
#       # registered texture. Called once after all materials have been set up.
        layer_count: int = max(1, self.next_texture_layer)
#       layer_count: int = max(1, self.next_texture_layer)
        self.texture_array = self.device.create_texture(
#       self.texture_array = self.device.create_texture(
            label="scene texture array",
#           label="scene texture array",
            size=(self.texture_array_size, self.texture_array_size, layer_count),
#           size=(self.texture_array_size, self.texture_array_size, layer_count),
            format=wgpu.TextureFormat.rgba8unorm,
#           format=wgpu.TextureFormat.rgba8unorm,
            mip_level_count=self.texture_array_mip_count,
#           mip_level_count=self.texture_array_mip_count,
            usage=wgpu.TextureUsage.TEXTURE_BINDING | wgpu.TextureUsage.COPY_DST,
#           usage=wgpu.TextureUsage.TEXTURE_BINDING | wgpu.TextureUsage.COPY_DST,
        )
#       )

        # Decoding, resizing, quantizing and mip building are CPU work that OpenCV and numpy run without holding the
#       # Decoding, resizing, quantizing and mip building are CPU work that OpenCV and numpy run without holding the
        # GIL, so the layers are prepared on a small thread pool while this thread uploads the finished ones in
#       # GIL, so the layers are prepared on a small thread pool while this thread uploads the finished ones in
        # registration order. The worker count bounds the transient memory: a layer from a 4096^2 source passes
#       # registration order. The worker count bounds the transient memory: a layer from a 4096^2 source passes
        # through a 256 MB float32 intermediate (an 8192^2 source, 1 GB).
#       # through a 256 MB float32 intermediate (an 8192^2 source, 1 GB).
        layer_jobs: list[tuple[int, typing.Callable[[], npt.NDArray[np.uint8] | None]]] = []
#       layer_jobs: list[tuple[int, typing.Callable[[], npt.NDArray[np.uint8] | None]]] = []
        for path, layer_index, is_srgb in self.pending_textures:
#       for path, layer_index, is_srgb in self.pending_textures:
            layer_jobs.append((layer_index, functools.partial(self.load_texture_layer, path, is_srgb)))
#           layer_jobs.append((layer_index, functools.partial(self.load_texture_layer, path, is_srgb)))
        # Packed data maps: roughness->R, metallic->G, transmission->B in a single layer
#       # Packed data maps: roughness->R, metallic->G, transmission->B in a single layer
        for layer_index, roughness_path, metallic_path, transmission_path in self.pending_packed_textures:
#       for layer_index, roughness_path, metallic_path, transmission_path in self.pending_packed_textures:
            layer_jobs.append((layer_index, functools.partial(self.pack_orm_layer, roughness_path, metallic_path, transmission_path)))
#           layer_jobs.append((layer_index, functools.partial(self.pack_orm_layer, roughness_path, metallic_path, transmission_path)))

        def prepare_layer(layer_job: tuple[int, typing.Callable[[], npt.NDArray[np.uint8] | None]]) -> tuple[int, list[npt.NDArray[np.uint8]] | None]:
#       def prepare_layer(layer_job: tuple[int, typing.Callable[[], npt.NDArray[np.uint8] | None]]) -> tuple[int, list[npt.NDArray[np.uint8]] | None]:
            layer_index, load_layer = layer_job
#           layer_index, load_layer = layer_job
            data_uint8: npt.NDArray[np.uint8] | None = load_layer()
#           data_uint8: npt.NDArray[np.uint8] | None = load_layer()
            return layer_index, (self.build_mip_chain(data_uint8=data_uint8) if data_uint8 is not None else None)
#           return layer_index, (self.build_mip_chain(data_uint8=data_uint8) if data_uint8 is not None else None)

        with concurrent.futures.ThreadPoolExecutor(max_workers=self.texture_worker_count) as executor:
#       with concurrent.futures.ThreadPoolExecutor(max_workers=self.texture_worker_count) as executor:
            for layer_index, mip_chain in executor.map(prepare_layer, layer_jobs):
#           for layer_index, mip_chain in executor.map(prepare_layer, layer_jobs):
                if mip_chain is not None:
#               if mip_chain is not None:
                    self.write_texture_array_layer(mip_chain=mip_chain, layer_index=layer_index)
#                   self.write_texture_array_layer(mip_chain=mip_chain, layer_index=layer_index)
        pass
#       pass

    def pack_orm_layer(self, roughness_path: pl.Path | None, metallic_path: pl.Path | None, transmission_path: pl.Path | None) -> npt.NDArray[np.uint8]:
#   def pack_orm_layer(self, roughness_path: pl.Path | None, metallic_path: pl.Path | None, transmission_path: pl.Path | None) -> npt.NDArray[np.uint8]:
        # Quantize each channel straight into the 8-bit layer: identical bytes to the old
#       # Quantize each channel straight into the 8-bit layer: identical bytes to the old
        # float32 intermediate (same per-element clip*255+0.5 math) at 1/4 the transient RAM
#       # float32 intermediate (same per-element clip*255+0.5 math) at 1/4 the transient RAM
        # (a 2048^2 rgba float32 buffer was 67MB per packed layer).
#       # (a 2048^2 rgba float32 buffer was 67MB per packed layer).
        packed_uint8: npt.NDArray[np.uint8] = np.zeros((self.texture_array_size, self.texture_array_size, 4), dtype=np.uint8)
#       packed_uint8: npt.NDArray[np.uint8] = np.zeros((self.texture_array_size, self.texture_array_size, 4), dtype=np.uint8)
        packed_uint8[..., 3] = 255
#       packed_uint8[..., 3] = 255
        for source_path, channel_index in ((roughness_path, 0), (metallic_path, 1), (transmission_path, 2)):
#       for source_path, channel_index in ((roughness_path, 0), (metallic_path, 1), (transmission_path, 2)):
            if source_path is None:
#           if source_path is None:
                continue
#               continue
            channel = self._load_data_channel(source_path)
#           channel = self._load_data_channel(source_path)
            if channel is not None:
#           if channel is not None:
                packed_uint8[..., channel_index] = (np.clip(channel, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
#               packed_uint8[..., channel_index] = (np.clip(channel, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
        return np.flipud(packed_uint8)
#       return np.flipud(packed_uint8)

    def setup_hdri(self, hdri_path: pl.Path | None = None, use_hdri: bool = False) -> None:
#   def setup_hdri(self, hdri_path: pl.Path | None = None, use_hdri: bool = False) -> None:
        self.use_hdri = use_hdri and hdri_path is not None and hdri_path.exists()
#       self.use_hdri = use_hdri and hdri_path is not None and hdri_path.exists()
        if self.texture_hdri is not None:
#       if self.texture_hdri is not None:
            self.texture_hdri.destroy()
#           self.texture_hdri.destroy()
        # WebGPU has no three-channel float format, so the HDRI is stored as rgba32float with alpha = 1.
#       # WebGPU has no three-channel float format, so the HDRI is stored as rgba32float with alpha = 1.
        hdri_data: npt.NDArray[np.float32]
#       hdri_data: npt.NDArray[np.float32]
        if self.use_hdri and hdri_path is not None:
#       if self.use_hdri and hdri_path is not None:
            loaded_data = cv2.imread(str(hdri_path), cv2.IMREAD_UNCHANGED)
#           loaded_data = cv2.imread(str(hdri_path), cv2.IMREAD_UNCHANGED)
            if loaded_data is None:
#           if loaded_data is None:
                raise RuntimeError(f"Failed to load HDRI: {hdri_path}")
#               raise RuntimeError(f"Failed to load HDRI: {hdri_path}")
            hdri_data = loaded_data.astype(dtype=np.float32)
#           hdri_data = loaded_data.astype(dtype=np.float32)
            hdri_data = typing.cast(npt.NDArray[np.float32], cv2.cvtColor(hdri_data, cv2.COLOR_BGR2RGBA))
#           hdri_data = typing.cast(npt.NDArray[np.float32], cv2.cvtColor(hdri_data, cv2.COLOR_BGR2RGBA))
            hdri_data[..., 3] = 1.0
#           hdri_data[..., 3] = 1.0
            hdri_data = np.ascontiguousarray(np.flipud(hdri_data))
#           hdri_data = np.ascontiguousarray(np.flipud(hdri_data))
        else:
#       else:
            hdri_data = np.zeros((1, 1, 4), dtype=np.float32)
#           hdri_data = np.zeros((1, 1, 4), dtype=np.float32)
        h, w, c = hdri_data.shape
#       h, w, c = hdri_data.shape
        self.texture_hdri = self.create_texture(size=(w, h), texture_format=wgpu.TextureFormat.rgba32float, usage=wgpu.TextureUsage.TEXTURE_BINDING | wgpu.TextureUsage.COPY_DST, label="hdri")
#       self.texture_hdri = self.create_texture(size=(w, h), texture_format=wgpu.TextureFormat.rgba32float, usage=wgpu.TextureUsage.TEXTURE_BINDING | wgpu.TextureUsage.COPY_DST, label="hdri")
        self.device.queue.write_texture(
#       self.device.queue.write_texture(
            destination={"texture": self.texture_hdri, "mip_level": 0, "origin": (0, 0, 0)},
#           destination={"texture": self.texture_hdri, "mip_level": 0, "origin": (0, 0, 0)},
            data=hdri_data,
#           data=hdri_data,
            data_layout={"offset": 0, "bytes_per_row": w * c * 4, "rows_per_image": h},
#           data_layout={"offset": 0, "bytes_per_row": w * c * 4, "rows_per_image": h},
            size=(w, h, 1),
#           size=(w, h, 1),
        )
#       )
        pass
#       pass

    def bake_sky_clouds(self) -> None:
#   def bake_sky_clouds(self) -> None:
        # Re-integrate the sky cloud layer into texture_sky_clouds. The layer is static, so this only
#       # Re-integrate the sky cloud layer into texture_sky_clouds. The layer is static, so this only
        # runs when the scene or the cloud toggle changes, never per frame.
#       # runs when the scene or the cloud toggle changes, never per frame.
        self.is_sky_clouds_active = self.is_sky_clouds_enabled and self.current_scene.sky_clouds_enabled and not self.use_hdri
#       self.is_sky_clouds_active = self.is_sky_clouds_enabled and self.current_scene.sky_clouds_enabled and not self.use_hdri
        if not self.is_sky_clouds_active:
#       if not self.is_sky_clouds_active:
            return
#           return
        if "uSkyCloudCoverage" in self.program_sky_clouds_bake:
#       if "uSkyCloudCoverage" in self.program_sky_clouds_bake:
            self.program_sky_clouds_bake["uSkyCloudCoverage"] = self.current_scene.sky_cloud_coverage
#           self.program_sky_clouds_bake["uSkyCloudCoverage"] = self.current_scene.sky_cloud_coverage
        if "uSkyCloudDensity" in self.program_sky_clouds_bake:
#       if "uSkyCloudDensity" in self.program_sky_clouds_bake:
            self.program_sky_clouds_bake["uSkyCloudDensity"] = self.current_scene.sky_cloud_density
#           self.program_sky_clouds_bake["uSkyCloudDensity"] = self.current_scene.sky_cloud_density
        bake_resources: dict[str, typing.Any] = self.get_shader_resources()
#       bake_resources: dict[str, typing.Any] = self.get_shader_resources()
        bake_resources["textureSkyClouds"] = self.texture_sky_clouds
#       bake_resources["textureSkyClouds"] = self.texture_sky_clouds
        # The bake writes texture_sky_clouds, so the sampled uSkyCloudTexture the shared sky code declares
#       # The bake writes texture_sky_clouds, so the sampled uSkyCloudTexture the shared sky code declares
        # (but the bake never reads) is pointed at the dummy texture instead.
#       # (but the bake never reads) is pointed at the dummy texture instead.
        bake_resources["uSkyCloudTexture"] = (self.texture_dummy, self.sampler_equirect)
#       bake_resources["uSkyCloudTexture"] = (self.texture_dummy, self.sampler_equirect)
        bake_width, bake_height = self.texture_sky_clouds.size[0], self.texture_sky_clouds.size[1]
#       bake_width, bake_height = self.texture_sky_clouds.size[0], self.texture_sky_clouds.size[1]
        command_encoder: wgpu.GPUCommandEncoder = self.device.create_command_encoder(label="sky clouds bake")
#       command_encoder: wgpu.GPUCommandEncoder = self.device.create_command_encoder(label="sky clouds bake")
        self.program_sky_clouds_bake.run(command_encoder=command_encoder, resources=bake_resources, group_x=(bake_width + 15) // 16, group_y=(bake_height + 15) // 16, group_z=1)
#       self.program_sky_clouds_bake.run(command_encoder=command_encoder, resources=bake_resources, group_x=(bake_width + 15) // 16, group_y=(bake_height + 15) // 16, group_z=1)
        self.device.queue.submit([command_encoder.finish()])
#       self.device.queue.submit([command_encoder.finish()])
        pass
#       pass

    def upload_volumetric_uniforms(self, program: ComputeProgram) -> None:
#   def upload_volumetric_uniforms(self, program: ComputeProgram) -> None:
        # Sky cloud layer and participating media uniforms, shared by the shading and NRC gather passes.
#       # Sky cloud layer and participating media uniforms, shared by the shading and NRC gather passes.
        # texture_sky_clouds / texture_cloud_noise are bound through get_shader_resources.
#       # texture_sky_clouds / texture_cloud_noise are bound through get_shader_resources.
        if "uSkyCloudsEnabled" in program:
#       if "uSkyCloudsEnabled" in program:
            program["uSkyCloudsEnabled"] = self.is_sky_clouds_active
#           program["uSkyCloudsEnabled"] = self.is_sky_clouds_active
        if "uSkyCloudCoverage" in program:
#       if "uSkyCloudCoverage" in program:
            program["uSkyCloudCoverage"] = self.current_scene.sky_cloud_coverage
#           program["uSkyCloudCoverage"] = self.current_scene.sky_cloud_coverage
        if "uSkyCloudDensity" in program:
#       if "uSkyCloudDensity" in program:
            program["uSkyCloudDensity"] = self.current_scene.sky_cloud_density
#           program["uSkyCloudDensity"] = self.current_scene.sky_cloud_density
        # Must match MAX_MEDIUM_VOLUMES in participating_media.slang.
#       # Must match MAX_MEDIUM_VOLUMES in participating_media.slang.
        medium_volumes: list[MediumVolume] = self.current_scene.get_medium_volumes()[:4]
#       medium_volumes: list[MediumVolume] = self.current_scene.get_medium_volumes()[:4]
        if "uMediaEnabled" in program:
#       if "uMediaEnabled" in program:
            program["uMediaEnabled"] = self.is_media_enabled and len(medium_volumes) > 0
#           program["uMediaEnabled"] = self.is_media_enabled and len(medium_volumes) > 0
        if "uMediumVolumeCount" in program:
#       if "uMediumVolumeCount" in program:
            program["uMediumVolumeCount"] = len(medium_volumes)
#           program["uMediumVolumeCount"] = len(medium_volumes)
        for i, volume in enumerate(medium_volumes):
#       for i, volume in enumerate(medium_volumes):
            # Checked member by member, so a program whose shader omits part of the volume layout still
#           # Checked member by member, so a program whose shader omits part of the volume layout still
            # accepts the upload (the GLSL compiler used to strip the scattering members from NRC gather).
#           # accepts the upload (the GLSL compiler used to strip the scattering members from NRC gather).
            volume_members: dict[str, typing.Any] = {
#           volume_members: dict[str, typing.Any] = {
                "aabbMin": volume["aabb_min"],
#               "aabbMin": volume["aabb_min"],
                "aabbMax": volume["aabb_max"],
#               "aabbMax": volume["aabb_max"],
                "density": volume["density"],
#               "density": volume["density"],
                "albedo": volume["albedo"],
#               "albedo": volume["albedo"],
                "anisotropy": volume["anisotropy"],
#               "anisotropy": volume["anisotropy"],
                "noiseFrequency": volume["noise_frequency"],
#               "noiseFrequency": volume["noise_frequency"],
                "kind": 0 if volume["kind"] == "fog" else 1,
#               "kind": 0 if volume["kind"] == "fog" else 1,
            }
#           }
            for member_name, member_value in volume_members.items():
#           for member_name, member_value in volume_members.items():
                if f"uMediumVolumes[{i}].{member_name}" in program:
#               if f"uMediumVolumes[{i}].{member_name}" in program:
                    program[f"uMediumVolumes[{i}].{member_name}"] = member_value
#                   program[f"uMediumVolumes[{i}].{member_name}"] = member_value
        pass
#       pass

    def set_scene(self, scene: BaseScene) -> None:
#   def set_scene(self, scene: BaseScene) -> None:
        # Load and activate a scene, uploading its geometry, materials, textures,
#       # Load and activate a scene, uploading its geometry, materials, textures,
        # and configuring the camera, lights, and NRC bounding box.
#       # and configuring the camera, lights, and NRC bounding box.
        print(f"[SCENE] Loading scene: {scene.name} (Index: {self.current_scene_index})")
#       print(f"[SCENE] Loading scene: {scene.name} (Index: {self.current_scene_index})")
        self.current_scene = scene
#       self.current_scene = scene

        # Reset texture registration state
#       # Reset texture registration state
        self.texture_cache = {}
#       self.texture_cache = {}
        self.pending_textures = []
#       self.pending_textures = []
        self.pending_packed_textures = []
#       self.pending_packed_textures = []
        self.next_texture_layer = 0
#       self.next_texture_layer = 0

        # Build scene materials and geometry
#       # Build scene materials and geometry
        self.materials = []
#       self.materials = []
        self.scene_builder = SceneBuilder(device=self.device, program_geometry=self.program_geometry, materials=self.materials)
#       self.scene_builder = SceneBuilder(device=self.device, program_geometry=self.program_geometry, materials=self.materials)
        scene.build(scene_builder=self.scene_builder, renderer=self)
#       scene.build(scene_builder=self.scene_builder, renderer=self)

        # Build BVH and pack vertex / material buffers, on a background thread: the build is numpy work that runs
#       # Build BVH and pack vertex / material buffers, on a background thread: the build is numpy work that runs
        # outside the GIL, and so is the texture decoding upload_pending_textures does meanwhile on this thread and
#       # outside the GIL, and so is the texture decoding upload_pending_textures does meanwhile on this thread and
        # its pool, so the two largest pieces of a scene load overlap instead of adding up.
#       # its pool, so the two largest pieces of a scene load overlap instead of adding up.
        with concurrent.futures.ThreadPoolExecutor(max_workers=1, thread_name_prefix="scene build") as scene_build_executor:
#       with concurrent.futures.ThreadPoolExecutor(max_workers=1, thread_name_prefix="scene build") as scene_build_executor:
            scene_build: concurrent.futures.Future[tuple[int, bytes, bytes, bytes]] = scene_build_executor.submit(self.scene_builder.build)
#           scene_build: concurrent.futures.Future[tuple[int, bytes, bytes, bytes]] = scene_build_executor.submit(self.scene_builder.build)

            # Upload all registered textures to the texture array
#           # Upload all registered textures to the texture array
            if self.texture_array is not None:
#           if self.texture_array is not None:
                self.texture_array.destroy()
#               self.texture_array.destroy()
            self.upload_pending_textures()
#           self.upload_pending_textures()

            self.num_triangles, bvh_data, vertices_data, materials_data = scene_build.result()
#           self.num_triangles, bvh_data, vertices_data, materials_data = scene_build.result()

        # Upload to GPU buffers
#       # Upload to GPU buffers
        if self.ssbo_bvh_nodes is not None:
#       if self.ssbo_bvh_nodes is not None:
            self.ssbo_bvh_nodes.destroy()
#           self.ssbo_bvh_nodes.destroy()
        self.ssbo_bvh_nodes = self.create_buffer(data=bvh_data, label="bvh nodes")
#       self.ssbo_bvh_nodes = self.create_buffer(data=bvh_data, label="bvh nodes")

        # The interleaved vertices feed both the rasterizer (vertex buffer) and the ray tracer (storage buffer).
#       # The interleaved vertices feed both the rasterizer (vertex buffer) and the ray tracer (storage buffer).
        if self.buffer_global_vertices is not None:
#       if self.buffer_global_vertices is not None:
            self.buffer_global_vertices.destroy()
#           self.buffer_global_vertices.destroy()
        self.buffer_global_vertices = self.create_buffer(data=vertices_data, label="scene vertices", usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.VERTEX)
#       self.buffer_global_vertices = self.create_buffer(data=vertices_data, label="scene vertices", usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.VERTEX)

        if self.ssbo_materials is not None:
#       if self.ssbo_materials is not None:
            self.ssbo_materials.destroy()
#           self.ssbo_materials.destroy()
        self.ssbo_materials = self.create_buffer(data=materials_data, label="scene materials")
#       self.ssbo_materials = self.create_buffer(data=materials_data, label="scene materials")

        # NRC input-domain normalisation, derived from the BVH root AABB
#       # NRC input-domain normalisation, derived from the BVH root AABB
        if len(bvh_data) >= 32:
#       if len(bvh_data) >= 32:
            bvh_root: npt.NDArray[np.float32] = np.frombuffer(bvh_data, dtype=np.float32, count=8)
#           bvh_root: npt.NDArray[np.float32] = np.frombuffer(bvh_data, dtype=np.float32, count=8)
            scene_aabb_min: npt.NDArray[np.float32] = bvh_root[0:3]
#           scene_aabb_min: npt.NDArray[np.float32] = bvh_root[0:3]
            scene_aabb_max: npt.NDArray[np.float32] = bvh_root[4:7]
#           scene_aabb_max: npt.NDArray[np.float32] = bvh_root[4:7]
        else:
#       else:
            scene_aabb_min = np.array([-1.0, -1.0, -1.0], dtype=np.float32)
#           scene_aabb_min = np.array([-1.0, -1.0, -1.0], dtype=np.float32)
            scene_aabb_max = np.array([1.0, 1.0, 1.0], dtype=np.float32)
#           scene_aabb_max = np.array([1.0, 1.0, 1.0], dtype=np.float32)
        scene_center: npt.NDArray[np.float32] = (scene_aabb_min + scene_aabb_max) * 0.5
#       scene_center: npt.NDArray[np.float32] = (scene_aabb_min + scene_aabb_max) * 0.5
        scene_max_extent: float = float(max(np.max(scene_aabb_max - scene_aabb_min), 1.0e-3))
#       scene_max_extent: float = float(max(np.max(scene_aabb_max - scene_aabb_min), 1.0e-3))
        self.nrc_position_offset = (float(scene_center[0]), float(scene_center[1]), float(scene_center[2]))
#       self.nrc_position_offset = (float(scene_center[0]), float(scene_center[1]), float(scene_center[2]))
        self.nrc_position_scale = 1.5 / scene_max_extent
#       self.nrc_position_scale = 1.5 / scene_max_extent
        print(f"[NRC] scene aabb min={scene_aabb_min} max={scene_aabb_max} offset={self.nrc_position_offset} scale={self.nrc_position_scale:.5f}")
#       print(f"[NRC] scene aabb min={scene_aabb_min} max={scene_aabb_max} offset={self.nrc_position_offset} scale={self.nrc_position_scale:.5f}")

        # Reset Neural Radiance Cache weights
#       # Reset Neural Radiance Cache weights
        self.reset_neural_cache_weights()
#       self.reset_neural_cache_weights()

        # Flush reservoirs and cache
#       # Flush reservoirs and cache
        w_res, h_res = self.window_size
#       w_res, h_res = self.window_size
        reservoir_floats_per_item: int = 20
#       reservoir_floats_per_item: int = 20
        self.device.queue.write_buffer(self.ssbo_sample_reservoirs, 0, np.zeros(w_res * h_res * 2 * reservoir_floats_per_item, dtype=np.float32).tobytes())
#       self.device.queue.write_buffer(self.ssbo_sample_reservoirs, 0, np.zeros(w_res * h_res * 2 * reservoir_floats_per_item, dtype=np.float32).tobytes())
        cache_entry_count: int = 131072
#       cache_entry_count: int = 131072
        cache_floats_per_entry: int = 12
#       cache_floats_per_entry: int = 12
        self.device.queue.write_buffer(self.ssbo_radiance_cache, 0, np.zeros(cache_entry_count * cache_floats_per_entry, dtype=np.float32).tobytes())
#       self.device.queue.write_buffer(self.ssbo_radiance_cache, 0, np.zeros(cache_entry_count * cache_floats_per_entry, dtype=np.float32).tobytes())

        # Setup HDRI / Environment
#       # Setup HDRI / Environment
        self.setup_hdri(hdri_path=scene.hdri_path, use_hdri=scene.use_hdri)
#       self.setup_hdri(hdri_path=scene.hdri_path, use_hdri=scene.use_hdri)
        # After setup_hdri: an HDRI scene skips the procedural cloud layer entirely.
#       # After setup_hdri: an HDRI scene skips the procedural cloud layer entirely.
        self.bake_sky_clouds()
#       self.bake_sky_clouds()

        # Setup Camera
#       # Setup Camera
        self.camera = Camera(
#       self.camera = Camera(
            position=scene.camera_position,
#           position=scene.camera_position,
            look_at=scene.camera_look_at,
#           look_at=scene.camera_look_at,
            up=scene.camera_up,
#           up=scene.camera_up,
            aspect_ratio=self.aspect_ratio,
#           aspect_ratio=self.aspect_ratio,
            fov=scene.camera_fov,
#           fov=scene.camera_fov,
            speed=scene.camera_speed,
#           speed=scene.camera_speed,
        )
#       )

        # Reset history and frame counters
#       # Reset history and frame counters
        self.frame_count = 0
#       self.frame_count = 0
        self.cache_frame_counter = 0
#       self.cache_frame_counter = 0
        self.is_scene_reset = True
#       self.is_scene_reset = True
        self.last_view_matrix = None
#       self.last_view_matrix = None
        self.last_view_proj_matrix = None
#       self.last_view_proj_matrix = None
        self.last_camera_position = None
#       self.last_camera_position = None
        pass
#       pass

    def set_scene_by_index(self, index: int) -> None:
#   def set_scene_by_index(self, index: int) -> None:
        self.current_scene_index = index % len(self.scenes)
#       self.current_scene_index = index % len(self.scenes)
        self.set_scene(scene=self.scenes[self.current_scene_index])
#       self.set_scene(scene=self.scenes[self.current_scene_index])
        pass
#       pass

    def reset_neural_cache_weights(self) -> None:
#   def reset_neural_cache_weights(self) -> None:
        initial_weights: npt.NDArray[np.float32] = np.zeros(4556, dtype=np.float32)
#       initial_weights: npt.NDArray[np.float32] = np.zeros(4556, dtype=np.float32)
        def initialize_layer(start_offset: int, inputs: int, outputs: int) -> None:
#       def initialize_layer(start_offset: int, inputs: int, outputs: int) -> None:
            distribution_limit: float = float(np.sqrt(2.0 / float(inputs)))
#           distribution_limit: float = float(np.sqrt(2.0 / float(inputs)))
            for j in range(inputs * outputs):
#           for j in range(inputs * outputs):
                random_val: float = float((np.random.rand() * 2.0 - 1.0) * distribution_limit)
#               random_val: float = float((np.random.rand() * 2.0 - 1.0) * distribution_limit)
                initial_weights[start_offset + j] = random_val
#               initial_weights[start_offset + j] = random_val
                initial_weights[1139 + start_offset + j] = random_val
#               initial_weights[1139 + start_offset + j] = random_val
        initialize_layer(0, 16, 16)
#       initialize_layer(0, 16, 16)
        initialize_layer(272, 16, 16)
#       initialize_layer(272, 16, 16)
        initialize_layer(544, 16, 16)
#       initialize_layer(544, 16, 16)
        initialize_layer(816, 16, 16)
#       initialize_layer(816, 16, 16)
        initialize_layer(1088, 16, 3)
#       initialize_layer(1088, 16, 3)
        self.device.queue.write_buffer(self.ssbo_nrc_weights, 0, initial_weights.tobytes())
#       self.device.queue.write_buffer(self.ssbo_nrc_weights, 0, initial_weights.tobytes())
        self.device.queue.write_buffer(self.ssbo_nrc_training_count, 0, np.zeros(1, dtype=np.uint32).tobytes())
#       self.device.queue.write_buffer(self.ssbo_nrc_training_count, 0, np.zeros(1, dtype=np.uint32).tobytes())
        self.nrc_training_step = 0
#       self.nrc_training_step = 0

    def load_texture_layer(self, path: pl.Path, is_srgb: bool = False) -> npt.NDArray[np.uint8] | None:
#   def load_texture_layer(self, path: pl.Path, is_srgb: bool = False) -> npt.NDArray[np.uint8] | None:
        # Loads a texture from disk and converts it into one RGBA8 layer of the 2D Texture Array (None on failure).
#       # Loads a texture from disk and converts it into one RGBA8 layer of the 2D Texture Array (None on failure).
        # Texture Arrays allow the shader to access many different textures using a single sampler + index.
#       # Texture Arrays allow the shader to access many different textures using a single sampler + index.
        # Runs on upload_pending_textures' worker threads, so it only does CPU work; the upload happens on the caller.
#       # Runs on upload_pending_textures' worker threads, so it only does CPU work; the upload happens on the caller.
        if not path.exists():
#       if not path.exists():
            print(f"Warning: Texture not found: {path}")
#           print(f"Warning: Texture not found: {path}")
            return None
#           return None

        loaded_data = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
#       loaded_data = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if loaded_data is None:
#       if loaded_data is None:
            print(f"Warning: Failed to load texture: {path}")
#           print(f"Warning: Failed to load texture: {path}")
            return None
#           return None

        # Ensure RGBA
#       # Ensure RGBA
        print(f"path: {path} loaded_data.shape: {loaded_data.shape}")
#       print(f"path: {path} loaded_data.shape: {loaded_data.shape}")
        if len(loaded_data.shape) == 2:
#       if len(loaded_data.shape) == 2:
            loaded_data = cv2.cvtColor(loaded_data, cv2.COLOR_GRAY2RGBA)
#           loaded_data = cv2.cvtColor(loaded_data, cv2.COLOR_GRAY2RGBA)
        elif loaded_data.shape[2] == 3:
#       elif loaded_data.shape[2] == 3:
            loaded_data = cv2.cvtColor(loaded_data, cv2.COLOR_BGR2RGBA)
#           loaded_data = cv2.cvtColor(loaded_data, cv2.COLOR_BGR2RGBA)
        elif loaded_data.shape[2] == 4:
#       elif loaded_data.shape[2] == 4:
            loaded_data = cv2.cvtColor(loaded_data, cv2.COLOR_BGRA2RGBA)
#           loaded_data = cv2.cvtColor(loaded_data, cv2.COLOR_BGRA2RGBA)

        # Fast path: an 8-bit source already at the array size needs no resize, so the
#       # Fast path: an 8-bit source already at the array size needs no resize, so the
        # sRGB decode -> resize -> encode round-trip below is an exact identity and both
#       # sRGB decode -> resize -> encode round-trip below is an exact identity and both
        # full-resolution pow() passes (plus the float conversion) can be skipped entirely.
#       # full-resolution pow() passes (plus the float conversion) can be skipped entirely.
        if loaded_data.dtype == np.uint8 and loaded_data.shape[0] == self.texture_array_size and loaded_data.shape[1] == self.texture_array_size:
#       if loaded_data.dtype == np.uint8 and loaded_data.shape[0] == self.texture_array_size and loaded_data.shape[1] == self.texture_array_size:
            # Flip vertically: the shaders keep OpenGL's convention that texture row 0 is the bottom of the image.
#           # Flip vertically: the shaders keep OpenGL's convention that texture row 0 is the bottom of the image.
            return typing.cast(npt.NDArray[np.uint8], np.flipud(loaded_data))
#           return typing.cast(npt.NDArray[np.uint8], np.flipud(loaded_data))

        # Convert to normalized float first so resizing is done on linear data
#       # Convert to normalized float first so resizing is done on linear data
        data_float: npt.NDArray[np.float32]
#       data_float: npt.NDArray[np.float32]

        if loaded_data.dtype == np.uint8:
#       if loaded_data.dtype == np.uint8:
            # An 8-bit texel has only 256 possible values, so normalising (and for colour maps the sRGB -> Linear decode,
#           # An 8-bit texel has only 256 possible values, so normalising (and for colour maps the sRGB -> Linear decode,
            # Approximate Gamma 2.2, applied before resize so filtering happens in linear space) is a table lookup
#           # Approximate Gamma 2.2, applied before resize so filtering happens in linear space) is a table lookup
            # holding exactly the float32 values the per-texel divide and pow() produced -- one gather over the full
#           # holding exactly the float32 values the per-texel divide and pow() produced -- one gather over the full
            # resolution source instead of a conversion, a divide and a transcendental pass.
#           # resolution source instead of a conversion, a divide and a transcendental pass.
            data_float = self.unorm8_to_float_lut[loaded_data]
#           data_float = self.unorm8_to_float_lut[loaded_data]
            if is_srgb:
#           if is_srgb:
                data_float[..., :3] = self.srgb8_to_linear_lut[loaded_data[..., :3]]
#               data_float[..., :3] = self.srgb8_to_linear_lut[loaded_data[..., :3]]
        else:
#       else:
            if loaded_data.dtype == np.uint16:
#           if loaded_data.dtype == np.uint16:
                data_float = (loaded_data.astype(dtype=np.float32) / 65535.0)
#               data_float = (loaded_data.astype(dtype=np.float32) / 65535.0)
            else:
#           else:
                data_float = loaded_data.astype(dtype=np.float32)
#               data_float = loaded_data.astype(dtype=np.float32)

            # Apply sRGB -> Linear conversion if requested (before resize so filtering happens in linear space)
#           # Apply sRGB -> Linear conversion if requested (before resize so filtering happens in linear space)
            if is_srgb:
#           if is_srgb:
                # Approximate Gamma 2.2
#               # Approximate Gamma 2.2
                data_float[..., :3] = np.power(data_float[..., :3], 2.2)
#               data_float[..., :3] = np.power(data_float[..., :3], 2.2)

        # Resize to the texture array size. INTER_AREA avoids aliasing when shrinking; INTER_LINEAR when enlarging.
#       # Resize to the texture array size. INTER_AREA avoids aliasing when shrinking; INTER_LINEAR when enlarging.
        if data_float.shape[0] != self.texture_array_size or data_float.shape[1] != self.texture_array_size:
#       if data_float.shape[0] != self.texture_array_size or data_float.shape[1] != self.texture_array_size:
            src_height, src_width = data_float.shape[:2]
#           src_height, src_width = data_float.shape[:2]
            is_shrinking: bool = self.texture_array_size < src_width or self.texture_array_size < src_height
#           is_shrinking: bool = self.texture_array_size < src_width or self.texture_array_size < src_height
            interpolation: int = cv2.INTER_AREA if is_shrinking else cv2.INTER_LINEAR
#           interpolation: int = cv2.INTER_AREA if is_shrinking else cv2.INTER_LINEAR
            data_float = cv2.resize(data_float, (self.texture_array_size, self.texture_array_size), interpolation=interpolation).astype(dtype=np.float32)
#           data_float = cv2.resize(data_float, (self.texture_array_size, self.texture_array_size), interpolation=interpolation).astype(dtype=np.float32)

        # Re-encode color back to sRGB so 8-bit storage is perceptually uniform (shader decodes on sample)
#       # Re-encode color back to sRGB so 8-bit storage is perceptually uniform (shader decodes on sample)
        if is_srgb:
#       if is_srgb:
            data_float[..., :3] = np.power(np.clip(data_float[..., :3], 0.0, 1.0), 1.0 / 2.2)
#           data_float[..., :3] = np.power(np.clip(data_float[..., :3], 0.0, 1.0), 1.0 / 2.2)

        # Flip vertically: the shaders keep OpenGL's convention that texture row 0 is the bottom of the image.
#       # Flip vertically: the shaders keep OpenGL's convention that texture row 0 is the bottom of the image.
        data_float = np.flipud(data_float)
#       data_float = np.flipud(data_float)

        # Quantize to 8-bit (f1) for the texture array: 4x less VRAM than f4
#       # Quantize to 8-bit (f1) for the texture array: 4x less VRAM than f4
        data_uint8: npt.NDArray[np.uint8] = (np.clip(data_float, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
#       data_uint8: npt.NDArray[np.uint8] = (np.clip(data_float, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)

        return data_uint8
#       return data_uint8


    """
    def on_key_event(self, key: typing.Any, action: typing.Any, modifiers: KeyModifiers) -> None:
#   def on_key_event(self, key: typing.Any, action: typing.Any, modifiers: KeyModifiers) -> None:
        keys: BaseKeys = self.wnd.keys
#       keys: BaseKeys = self.wnd.keys
        if action == keys.ACTION_PRESS:
#       if action == keys.ACTION_PRESS:
            if key == keys.W: self.key_state["W"] = True
#           if key == keys.W: self.key_state["W"] = True
            elif key == keys.S: self.key_state["S"] = True
#           elif key == keys.S: self.key_state["S"] = True
            elif key == keys.A: self.key_state["A"] = True
#           elif key == keys.A: self.key_state["A"] = True
            elif key == keys.D: self.key_state["D"] = True
#           elif key == keys.D: self.key_state["D"] = True
            elif key == keys.Q: self.key_state["Q"] = True
#           elif key == keys.Q: self.key_state["Q"] = True
            elif key == keys.E: self.key_state["E"] = True
#           elif key == keys.E: self.key_state["E"] = True
            elif key == keys.UP: self.key_state["UP"] = True
#           elif key == keys.UP: self.key_state["UP"] = True
            elif key == keys.DOWN: self.key_state["DOWN"] = True
#           elif key == keys.DOWN: self.key_state["DOWN"] = True
            elif key == keys.LEFT: self.key_state["LEFT"] = True
#           elif key == keys.LEFT: self.key_state["LEFT"] = True
            elif key == keys.RIGHT: self.key_state["RIGHT"] = True
#           elif key == keys.RIGHT: self.key_state["RIGHT"] = True
        elif action == keys.ACTION_RELEASE:
#       elif action == keys.ACTION_RELEASE:
            if key == keys.W: self.key_state["W"] = False
#           if key == keys.W: self.key_state["W"] = False
            elif key == keys.S: self.key_state["S"] = False
#           elif key == keys.S: self.key_state["S"] = False
            elif key == keys.A: self.key_state["A"] = False
#           elif key == keys.A: self.key_state["A"] = False
            elif key == keys.D: self.key_state["D"] = False
#           elif key == keys.D: self.key_state["D"] = False
            elif key == keys.Q: self.key_state["Q"] = False
#           elif key == keys.Q: self.key_state["Q"] = False
            elif key == keys.E: self.key_state["E"] = False
#           elif key == keys.E: self.key_state["E"] = False
            elif key == keys.UP: self.key_state["UP"] = False
#           elif key == keys.UP: self.key_state["UP"] = False
            elif key == keys.DOWN: self.key_state["DOWN"] = False
#           elif key == keys.DOWN: self.key_state["DOWN"] = False
            elif key == keys.LEFT: self.key_state["LEFT"] = False
#           elif key == keys.LEFT: self.key_state["LEFT"] = False
            elif key == keys.RIGHT: self.key_state["RIGHT"] = False
#           elif key == keys.RIGHT: self.key_state["RIGHT"] = False
    """

    def denoise_and_show(self) -> None:
#   def denoise_and_show(self) -> None:
        print("Denoising with OIDN...")
#       print("Denoising with OIDN...")

        w, h = self.window_size
#       w, h = self.window_size

        # Read textures (Read 4 components to be safe, then slice)
#       # Read textures (Read 4 components to be safe, then slice)
        # Accumulator (Color)
#       # Accumulator (Color)
        color_data: bytes = self.read_texture(texture=self.texture_accum, bytes_per_pixel=16)
#       color_data: bytes = self.read_texture(texture=self.texture_accum, bytes_per_pixel=16)
        color_arr: npt.NDArray[np.float32] = np.frombuffer(color_data, dtype=np.float32).reshape((h, w, 4))[:, :, :3]
#       color_arr: npt.NDArray[np.float32] = np.frombuffer(color_data, dtype=np.float32).reshape((h, w, 4))[:, :, :3]
        color_arr = np.ascontiguousarray(color_arr)
#       color_arr = np.ascontiguousarray(color_arr)

        # Check inputs
#       # Check inputs
        # print(f"OIDN Input Color Range: {np.min(color_arr)} - {np.max(color_arr)}")
#       # print(f"OIDN Input Color Range: {np.min(color_arr)} - {np.max(color_arr)}")

        # Albedo
#       # Albedo
        albedo_data: bytes = self.read_texture(texture=self.texture_geometry_albedo, bytes_per_pixel=4)
#       albedo_data: bytes = self.read_texture(texture=self.texture_geometry_albedo, bytes_per_pixel=4)
        albedo_arr_uint8: npt.NDArray[np.uint8] = np.frombuffer(albedo_data, dtype=np.uint8).reshape((h, w, 4))[:, :, :3]
#       albedo_arr_uint8: npt.NDArray[np.uint8] = np.frombuffer(albedo_data, dtype=np.uint8).reshape((h, w, 4))[:, :, :3]
        albedo_arr: npt.NDArray[np.float32] = albedo_arr_uint8.astype(np.float32) / 255.0
#       albedo_arr: npt.NDArray[np.float32] = albedo_arr_uint8.astype(np.float32) / 255.0
        albedo_arr = np.ascontiguousarray(albedo_arr)
#       albedo_arr = np.ascontiguousarray(albedo_arr)

        # Normal
#       # Normal
        normal_data: bytes = self.read_texture(texture=self.texture_geometry_global_normal, bytes_per_pixel=8)
#       normal_data: bytes = self.read_texture(texture=self.texture_geometry_global_normal, bytes_per_pixel=8)
        normal_arr_f16: npt.NDArray[np.float16] = np.frombuffer(normal_data, dtype=np.float16).reshape((h, w, 4))[:, :, :3]
#       normal_arr_f16: npt.NDArray[np.float16] = np.frombuffer(normal_data, dtype=np.float16).reshape((h, w, 4))[:, :, :3]
        normal_arr: npt.NDArray[np.float32] = normal_arr_f16.astype(np.float32)
#       normal_arr: npt.NDArray[np.float32] = normal_arr_f16.astype(np.float32)
        normal_arr = np.ascontiguousarray(normal_arr)
#       normal_arr = np.ascontiguousarray(normal_arr)

        # Output
#       # Output
        output_arr = np.zeros((h, w, 3), dtype=np.float32)
#       output_arr = np.zeros((h, w, 3), dtype=np.float32)

        # OIDN Filter
#       # OIDN Filter
        filter: pyoidn.Filter = pyoidn.Filter(device=self.oidn_device, filter_type=pyoidn.OIDN_FILTER_TYPE_RT)
#       filter: pyoidn.Filter = pyoidn.Filter(device=self.oidn_device, filter_type=pyoidn.OIDN_FILTER_TYPE_RT)

        # We rely on default strides (0) which OIDN interprets as packed
#       # We rely on default strides (0) which OIDN interprets as packed
        filter.set_image(name=pyoidn.OIDN_IMAGE_COLOR, data=color_arr, data_format=pyoidn.OIDN_FORMAT_FLOAT3, width=w, height=h)
#       filter.set_image(name=pyoidn.OIDN_IMAGE_COLOR, data=color_arr, data_format=pyoidn.OIDN_FORMAT_FLOAT3, width=w, height=h)
        filter.set_image(name=pyoidn.OIDN_IMAGE_ALBEDO, data=albedo_arr, data_format=pyoidn.OIDN_FORMAT_FLOAT3, width=w, height=h)
#       filter.set_image(name=pyoidn.OIDN_IMAGE_ALBEDO, data=albedo_arr, data_format=pyoidn.OIDN_FORMAT_FLOAT3, width=w, height=h)
        filter.set_image(name=pyoidn.OIDN_IMAGE_NORMAL, data=normal_arr, data_format=pyoidn.OIDN_FORMAT_FLOAT3, width=w, height=h)
#       filter.set_image(name=pyoidn.OIDN_IMAGE_NORMAL, data=normal_arr, data_format=pyoidn.OIDN_FORMAT_FLOAT3, width=w, height=h)
        filter.set_image(name=pyoidn.OIDN_IMAGE_OUTPUT, data=output_arr, data_format=pyoidn.OIDN_FORMAT_FLOAT3, width=w, height=h)
#       filter.set_image(name=pyoidn.OIDN_IMAGE_OUTPUT, data=output_arr, data_format=pyoidn.OIDN_FORMAT_FLOAT3, width=w, height=h)

        filter.set_quality(quality=pyoidn.OIDN_QUALITY_HIGH)
#       filter.set_quality(quality=pyoidn.OIDN_QUALITY_HIGH)

        filter.commit()
#       filter.commit()
        filter.execute()
#       filter.execute()

        # Check errors
#       # Check errors
        error_msg = self.oidn_device.get_error()
#       error_msg = self.oidn_device.get_error()
        if error_msg:
#       if error_msg:
            print(f"OIDN Error: {error_msg}")
#           print(f"OIDN Error: {error_msg}")

        # print(f"OIDN Output Range: {np.min(output_arr)} - {np.max(output_arr)}")
#       # print(f"OIDN Output Range: {np.min(output_arr)} - {np.max(output_arr)}")

        # OIDN can emit slightly negative values; clamp so the tonemap toe and gamma pow stay well-defined
#       # OIDN can emit slightly negative values; clamp so the tonemap toe and gamma pow stay well-defined
        output_arr = np.maximum(output_arr, 0.0)
#       output_arr = np.maximum(output_arr, 0.0)

        # if np.allclose(output_arr, 0.5):
#       # if np.allclose(output_arr, 0.5):
        #     print("ERROR: OIDN did not write to output array!")
#       #     print("ERROR: OIDN did not write to output array!")

        """
        # Apply ACES Tonemap
#       # Apply ACES Tonemap
        a = 2.51
#       a = 2.51
        b = 0.03
#       b = 0.03
        c = 2.43
#       c = 2.43
        d = 0.59
#       d = 0.59
        e = 0.14
#       e = 0.14
        output_arr = (output_arr * (a * output_arr + b)) / (output_arr * (c * output_arr + d) + e)
#       output_arr = (output_arr * (a * output_arr + b)) / (output_arr * (c * output_arr + d) + e)
        """

        # Apply Exposure - physically-based camera EV (matches computeExposure in post_tonemap_cs.slang)
#       # Apply Exposure - physically-based camera EV (matches computeExposure in post_tonemap_cs.slang)
        APERTURE = 16.0                 # f-number N
#       APERTURE = 16.0                 # f-number N
        SHUTTER_TIME = 0.008            # seconds t (~1/125 s)
#       SHUTTER_TIME = 0.008            # seconds t (~1/125 s)
        ISO = 100.0                     # sensitivity S
#       ISO = 100.0                     # sensitivity S
        EXPOSURE_COMP = 0.0             # artistic bias in stops
#       EXPOSURE_COMP = 0.0             # artistic bias in stops
        EXPOSURE_CALIBRATION = 19200.0  # arbitrary-units -> physical-luminance scale
#       EXPOSURE_CALIBRATION = 19200.0  # arbitrary-units -> physical-luminance scale
        ev100 = np.log2((APERTURE * APERTURE) / SHUTTER_TIME * 100.0 / ISO) - EXPOSURE_COMP
#       ev100 = np.log2((APERTURE * APERTURE) / SHUTTER_TIME * 100.0 / ISO) - EXPOSURE_COMP
        exposure = EXPOSURE_CALIBRATION / max(1.2 * (2.0 ** ev100), 1e-4)
#       exposure = EXPOSURE_CALIBRATION / max(1.2 * (2.0 ** ev100), 1e-4)
        output_arr = output_arr * exposure
#       output_arr = output_arr * exposure

        # Apply Gran Turismo (Uchimura 2017) Tonemap (matches gtTonemap in tonemap.slang)
#       # Apply Gran Turismo (Uchimura 2017) Tonemap (matches gtTonemap in tonemap.slang)
        P = 1.0    # maximum brightness the curve maps to
#       P = 1.0    # maximum brightness the curve maps to
        a = 1.0    # contrast of the linear mid-section
#       a = 1.0    # contrast of the linear mid-section
        m = 0.22   # start of the linear section
#       m = 0.22   # start of the linear section
        l = 0.4    # length of the linear section
#       l = 0.4    # length of the linear section
        c = 1.33   # toe curvature (black tightness)
#       c = 1.33   # toe curvature (black tightness)
        b = 0.0    # pedestal (black lift)
#       b = 0.0    # pedestal (black lift)

        l0 = ((P - m) * l) / a
#       l0 = ((P - m) * l) / a
        S0 = m + l0
#       S0 = m + l0
        S1 = m + a * l0
#       S1 = m + a * l0
        C2 = (a * P) / (P - S1)
#       C2 = (a * P) / (P - S1)
        CP = -C2 / P
#       CP = -C2 / P

        # Three weighted segments: toe (T), linear mid (L) and exponential shoulder (S).
#       # Three weighted segments: toe (T), linear mid (L) and exponential shoulder (S).
        t = np.clip(output_arr / m, 0.0, 1.0)
#       t = np.clip(output_arr / m, 0.0, 1.0)
        w0 = 1.0 - (t * t * (3.0 - 2.0 * t))
#       w0 = 1.0 - (t * t * (3.0 - 2.0 * t))
        w2 = np.where(output_arr >= (m + l0), 1.0, 0.0)
#       w2 = np.where(output_arr >= (m + l0), 1.0, 0.0)
        w1 = 1.0 - w0 - w2
#       w1 = 1.0 - w0 - w2

        T = m * np.power(np.maximum(output_arr / m, 0.0), c) + b
#       T = m * np.power(np.maximum(output_arr / m, 0.0), c) + b
        S = P - (P - S1) * np.exp(CP * (output_arr - S0))
#       S = P - (P - S1) * np.exp(CP * (output_arr - S0))
        L = m + a * (output_arr - m)
#       L = m + a * (output_arr - m)

        output_arr = T * w0 + L * w1 + S * w2
#       output_arr = T * w0 + L * w1 + S * w2

        output_arr = np.clip(output_arr, 0.0, 1.0)
#       output_arr = np.clip(output_arr, 0.0, 1.0)

        # Apply Gamma Correction
#       # Apply Gamma Correction
        output_arr = np.power(output_arr, 1.0 / 2.2)
#       output_arr = np.power(output_arr, 1.0 / 2.2)

        # Show result
#       # Show result
        # Convert RGB to BGR for OpenCV and Flip for correct orientation
#       # Convert RGB to BGR for OpenCV and Flip for correct orientation
        # The float64 np.where weights above promote the array to float64, which cvtColor rejects; cast back to float32.
#       # The float64 np.where weights above promote the array to float64, which cvtColor rejects; cast back to float32.
        display_img = cv2.cvtColor(output_arr.astype(np.float32), cv2.COLOR_RGB2BGR)
#       display_img = cv2.cvtColor(output_arr.astype(np.float32), cv2.COLOR_RGB2BGR)
        display_img = cv2.flip(display_img, 0)
#       display_img = cv2.flip(display_img, 0)
        cv2.imshow("OIDN Denoised Result", display_img)
#       cv2.imshow("OIDN Denoised Result", display_img)
        cv2.waitKey(1)
#       cv2.waitKey(1)

        filter.release()
#       filter.release()

        print("Denoise complete. Check the window.")
#       print("Denoise complete. Check the window.")

    def on_draw(self) -> None:
#   def on_draw(self) -> None:
        # rendercanvas draw callback: derives the elapsed / frame time moderngl_window used to pass in.
#       # rendercanvas draw callback: derives the elapsed / frame time moderngl_window used to pass in.
        current_time: float = tm.perf_counter()
#       current_time: float = tm.perf_counter()
        frame_time: float = current_time - self.last_frame_time
#       frame_time: float = current_time - self.last_frame_time
        self.last_frame_time = current_time
#       self.last_frame_time = current_time
        self.on_render(time=current_time - self.start_time, frame_time=frame_time)
#       self.on_render(time=current_time - self.start_time, frame_time=frame_time)
        pass
#       pass

    def render_to_screen(self, command_encoder: wgpu.GPUCommandEncoder, texture: wgpu.GPUTexture) -> None:
#   def render_to_screen(self, command_encoder: wgpu.GPUCommandEncoder, texture: wgpu.GPUTexture) -> None:
        # Composite Pass: draws texture over the window through hybrid_renderer_fs.slang, then submits the frame.
#       # Composite Pass: draws texture over the window through hybrid_renderer_fs.slang, then submits the frame.
        if "uRenderMode" in self.program_renderer:
#       if "uRenderMode" in self.program_renderer:
            self.program_renderer["uRenderMode"] = self.render_mode.value
#           self.program_renderer["uRenderMode"] = self.render_mode.value
        screen_view: wgpu.GPUTextureView = self.present_context.get_current_texture().create_view()
#       screen_view: wgpu.GPUTextureView = self.present_context.get_current_texture().create_view()
        self.program_renderer.render(
#       self.program_renderer.render(
            command_encoder=command_encoder,
#           command_encoder=command_encoder,
            resources={"uTextureOutput": (texture, self.sampler_screen)},
#           resources={"uTextureOutput": (texture, self.sampler_screen)},
            color_attachments=[{"view": screen_view, "clear_value": (0.0, 0.0, 0.0, 1.0), "load_op": wgpu.LoadOp.clear, "store_op": wgpu.StoreOp.store}],
#           color_attachments=[{"view": screen_view, "clear_value": (0.0, 0.0, 0.0, 1.0), "load_op": wgpu.LoadOp.clear, "store_op": wgpu.StoreOp.store}],
            vertex_buffer=self.vbo_screen,
#           vertex_buffer=self.vbo_screen,
            vertex_count=4,
#           vertex_count=4,
        )
#       )
        self.device.queue.submit([command_encoder.finish()])
#       self.device.queue.submit([command_encoder.finish()])
        pass
#       pass

    def on_render(self, time: float, frame_time: float) -> None:
#   def on_render(self, time: float, frame_time: float) -> None:
        # Main Render Loop
#       # Main Render Loop
        # 1. Update Game State (Camera, Input)
#       # 1. Update Game State (Camera, Input)
        # 2. Rasterize Geometry to G-Buffer (Position, Normal, etc.)
#       # 2. Rasterize Geometry to G-Buffer (Position, Normal, etc.)
        # 3. Ray Trace Lighting using Compute Shader (reads G-Buffer & BVH)
#       # 3. Ray Trace Lighting using Compute Shader (reads G-Buffer & BVH)
        # 4. Denoise the output
#       # 4. Denoise the output
        # 5. Post Processing Pipeline
#       # 5. Post Processing Pipeline
        # 6. Display to Screen
#       # 6. Display to Screen
        # Every pass of the frame is recorded into one command encoder and submitted at the end; WebGPU
#       # Every pass of the frame is recorded into one command encoder and submitted at the end; WebGPU
        # tracks the resource hazards between passes, so the explicit GL memory barriers have no equivalent.
#       # tracks the resource hazards between passes, so the explicit GL memory barriers have no equivalent.
        self.frame_count += 1
#       self.frame_count += 1

        self.camera.update(frame_time=frame_time, key_state=self.key_state)
#       self.camera.update(frame_time=frame_time, key_state=self.key_state)

        # Check for movement (Reset Accumulation)
#       # Check for movement (Reset Accumulation)
        current_view_matrix = self.camera.get_view_matrix()
#       current_view_matrix = self.camera.get_view_matrix()
        if self.last_view_matrix is not None:
#       if self.last_view_matrix is not None:
            if not np.allclose(current_view_matrix, self.last_view_matrix, atol=1e-5):
#           if not np.allclose(current_view_matrix, self.last_view_matrix, atol=1e-5):
                self.frame_count = 1
#               self.frame_count = 1
        self.last_view_matrix = current_view_matrix
#       self.last_view_matrix = current_view_matrix

        # TAA Jitter
#       # TAA Jitter
        # Halton sequence for x (base 2) and y (base 3)
#       # Halton sequence for x (base 2) and y (base 3)
        # Scale to [-0.5, 0.5] pixels
#       # Scale to [-0.5, 0.5] pixels
        if self.frame_count <= 1:
#       if self.frame_count <= 1:
            jitter_x: float = 0.0
#           jitter_x: float = 0.0
            jitter_y: float = 0.0
#           jitter_y: float = 0.0
        else:
#       else:
            jitter_x = (self.get_halton_jitter(index=self.frame_count, base=2) - 0.5)
#           jitter_x = (self.get_halton_jitter(index=self.frame_count, base=2) - 0.5)
            jitter_y = (self.get_halton_jitter(index=self.frame_count, base=3) - 0.5)
#           jitter_y = (self.get_halton_jitter(index=self.frame_count, base=3) - 0.5)

        # Reuse the matrix computed for the movement check above instead of rebuilding it
#       # Reuse the matrix computed for the movement check above instead of rebuilding it
        transform_view: rr.Matrix44 = current_view_matrix
#       transform_view: rr.Matrix44 = current_view_matrix
        transform_projection: rr.Matrix44 = self.camera.get_projection_matrix(jitter=(jitter_x, jitter_y), window_size=self.window_size)
#       transform_projection: rr.Matrix44 = self.camera.get_projection_matrix(jitter=(jitter_x, jitter_y), window_size=self.window_size)
        current_view_proj: rr.Matrix44 = np.array(transform_view, dtype=np.float32) @ np.array(transform_projection, dtype=np.float32)
#       current_view_proj: rr.Matrix44 = np.array(transform_view, dtype=np.float32) @ np.array(transform_projection, dtype=np.float32)
        if self.last_view_proj_matrix is None:
#       if self.last_view_proj_matrix is None:
            self.last_view_proj_matrix = current_view_proj
#           self.last_view_proj_matrix = current_view_proj
            self.last_camera_position = tuple(self.camera.look_from)
#           self.last_camera_position = tuple(self.camera.look_from)

        command_encoder: wgpu.GPUCommandEncoder = self.device.create_command_encoder(label="frame")
#       command_encoder: wgpu.GPUCommandEncoder = self.device.create_command_encoder(label="frame")

        # 1. Rasterize to G-Buffer
#       # 1. Rasterize to G-Buffer
        self.program_geometry.write("uTransformView", transform_view.astype(dtype=np.float32).tobytes())
#       self.program_geometry.write("uTransformView", transform_view.astype(dtype=np.float32).tobytes())
        self.program_geometry.write("uTransformProjection", transform_projection.astype(dtype=np.float32).tobytes())
#       self.program_geometry.write("uTransformProjection", transform_projection.astype(dtype=np.float32).tobytes())

        self.program_geometry.render(
#       self.program_geometry.render(
            command_encoder=command_encoder,
#           command_encoder=command_encoder,
            resources=self.get_shader_resources(),
#           resources=self.get_shader_resources(),
            # Zero out, .w = 0 means background
#           # Zero out, .w = 0 means background
            color_attachments=[
#           color_attachments=[
                {"view": gbuffer_texture.create_view(), "clear_value": (0.0, 0.0, 0.0, 0.0), "load_op": wgpu.LoadOp.clear, "store_op": wgpu.StoreOp.store}
#               {"view": gbuffer_texture.create_view(), "clear_value": (0.0, 0.0, 0.0, 0.0), "load_op": wgpu.LoadOp.clear, "store_op": wgpu.StoreOp.store}
                for gbuffer_texture in self.gbuffer
#               for gbuffer_texture in self.gbuffer
            ],
#           ],
            depth_stencil_attachment={"view": self.texture_depth.create_view(), "depth_clear_value": 1.0, "depth_load_op": wgpu.LoadOp.clear, "depth_store_op": wgpu.StoreOp.store},
#           depth_stencil_attachment={"view": self.texture_depth.create_view(), "depth_clear_value": 1.0, "depth_load_op": wgpu.LoadOp.clear, "depth_store_op": wgpu.StoreOp.store},
            vertex_buffer=self.buffer_global_vertices,
#           vertex_buffer=self.buffer_global_vertices,
            vertex_count=self.num_triangles * 3,
#           vertex_count=self.num_triangles * 3,
        )
#       )

        if self.render_mode != RenderMode.PATH_TRACE:
#       if self.render_mode != RenderMode.PATH_TRACE:
            debug_texture: wgpu.GPUTexture = self.texture_geometry_albedo
#           debug_texture: wgpu.GPUTexture = self.texture_geometry_albedo
            if self.render_mode == RenderMode.ALBEDO:
#           if self.render_mode == RenderMode.ALBEDO:
                debug_texture = self.texture_geometry_albedo
#               debug_texture = self.texture_geometry_albedo
            elif self.render_mode == RenderMode.NORMAL:
#           elif self.render_mode == RenderMode.NORMAL:
                debug_texture = self.texture_geometry_global_normal
#               debug_texture = self.texture_geometry_global_normal
            elif self.render_mode == RenderMode.POSITION:
#           elif self.render_mode == RenderMode.POSITION:
                debug_texture = self.texture_geometry_global_position
#               debug_texture = self.texture_geometry_global_position
            elif self.render_mode == RenderMode.TANGENT:
#           elif self.render_mode == RenderMode.TANGENT:
                debug_texture = self.texture_geometry_global_tangent
#               debug_texture = self.texture_geometry_global_tangent

            self.render_to_screen(command_encoder=command_encoder, texture=debug_texture)
#           self.render_to_screen(command_encoder=command_encoder, texture=debug_texture)
            return
#           return

        # 2. Compute Shade
#       # 2. Compute Shade
        # Update Uniforms
#       # Update Uniforms
        if "uTime" in self.program_shading:
#       if "uTime" in self.program_shading:
            self.program_shading["uTime"] = time
#           self.program_shading["uTime"] = time

        # Point Lights Setup
#       # Point Lights Setup
        point_lights: list[PointLight] = self.current_scene.get_point_lights(time=time)
#       point_lights: list[PointLight] = self.current_scene.get_point_lights(time=time)

        # Optimize by sorting lights by emission strength
#       # Optimize by sorting lights by emission strength
        def get_light_power(light: PointLight) -> float:
#       def get_light_power(light: PointLight) -> float:
            c = light["color"]
#           c = light["color"]
            r = light["radius"]
#           r = light["radius"]
            luminance = 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2]
#           luminance = 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2]
            return luminance * 4.0 * np.pi * r * r
#           return luminance * 4.0 * np.pi * r * r

        point_lights.sort(key=get_light_power, reverse=True)
#       point_lights.sort(key=get_light_power, reverse=True)
        total_power = sum(get_light_power(light) for light in point_lights)
#       total_power = sum(get_light_power(light) for light in point_lights)

        if "uPointLightCount" in self.program_shading:
#       if "uPointLightCount" in self.program_shading:
            self.program_shading["uPointLightCount"] = len(point_lights)
#           self.program_shading["uPointLightCount"] = len(point_lights)
            cumulative_prob = 0.0
#           cumulative_prob = 0.0
            for i, light in enumerate(point_lights):
#           for i, light in enumerate(point_lights):
                pos = light["position"]
#               pos = light["position"]
                color = light["color"]
#               color = light["color"]
                r = light["radius"]
#               r = light["radius"]
                power = get_light_power(light)
#               power = get_light_power(light)
                prob = power / total_power if total_power > 0.0 else 1.0 / len(point_lights)
#               prob = power / total_power if total_power > 0.0 else 1.0 / len(point_lights)
                cumulative_prob += prob
#               cumulative_prob += prob

                if f"uPointLights[{i}].position" in self.program_shading:
#               if f"uPointLights[{i}].position" in self.program_shading:
                    self.program_shading[f"uPointLights[{i}].position"] = pos
#                   self.program_shading[f"uPointLights[{i}].position"] = pos
                    self.program_shading[f"uPointLights[{i}].color"] = color
#                   self.program_shading[f"uPointLights[{i}].color"] = color
                    self.program_shading[f"uPointLights[{i}].radius"] = r
#                   self.program_shading[f"uPointLights[{i}].radius"] = r
                    if f"uPointLights[{i}].cdf" in self.program_shading:
#                   if f"uPointLights[{i}].cdf" in self.program_shading:
                        self.program_shading[f"uPointLights[{i}].cdf"] = cumulative_prob
#                       self.program_shading[f"uPointLights[{i}].cdf"] = cumulative_prob
                        self.program_shading[f"uPointLights[{i}].pdf"] = prob
#                       self.program_shading[f"uPointLights[{i}].pdf"] = prob

        if "uCameraGlobalPosition" in self.program_shading:
#       if "uCameraGlobalPosition" in self.program_shading:
            self.program_shading["uCameraGlobalPosition"] = tuple(self.camera.look_from)
#           self.program_shading["uCameraGlobalPosition"] = tuple(self.camera.look_from)
        if "uFrameCount" in self.program_shading:
#       if "uFrameCount" in self.program_shading:
            self.program_shading["uFrameCount"] = self.frame_count
#           self.program_shading["uFrameCount"] = self.frame_count
        self.cache_frame_counter += 1
#       self.cache_frame_counter += 1
        if "uCacheFrameCounter" in self.program_shading:
#       if "uCacheFrameCounter" in self.program_shading:
            self.program_shading["uCacheFrameCounter"] = self.cache_frame_counter
#           self.program_shading["uCacheFrameCounter"] = self.cache_frame_counter
        if "uCacheBlendFactor" in self.program_shading:
#       if "uCacheBlendFactor" in self.program_shading:
            # The hash radiance cache and the Neural Radiance Cache are two answers to the same
#           # The hash radiance cache and the Neural Radiance Cache are two answers to the same
            # question, and with NRC on the hash cache cannot win: hybrid_shading_cs.slang breaks out
#           # question, and with NRC on the hash cache cannot win: hybrid_shading_cs.slang breaks out
            # of the path at the NRC vertex before writeCache is reached, so nothing is ever written
#           # of the path at the NRC vertex before writeCache is reached, so nothing is ever written
            # at a diffuse vertex. What survives is the worst of both -- readCache still runs on every
#           # at a diffuse vertex. What survives is the worst of both -- readCache still runs on every
            # secondary bounce, and can still fire on a stale entry left by a metallic or near-mirror
#           # secondary bounce, and can still fire on a stale entry left by a metallic or near-mirror
            # vertex, pre-empting the network's full outgoing radiance with a direct-only value.
#           # vertex, pre-empting the network's full outgoing radiance with a direct-only value.
            # Switching the blend off makes the supersession explicit instead of accidental.
#           # Switching the blend off makes the supersession explicit instead of accidental.
            self.program_shading["uCacheBlendFactor"] = 0.0 if self.is_nrc_enabled else 0.5
#           self.program_shading["uCacheBlendFactor"] = 0.0 if self.is_nrc_enabled else 0.5

        if "uUseHdri" in self.program_shading:
#       if "uUseHdri" in self.program_shading:
            self.program_shading["uUseHdri"] = self.use_hdri
#           self.program_shading["uUseHdri"] = self.use_hdri
        # Volumetrics: baked sky cloud layer and cloud noise volume (bound for the gather pass too)
#       # Volumetrics: baked sky cloud layer and cloud noise volume (bound for the gather pass too)
        self.upload_volumetric_uniforms(program=self.program_shading)
#       self.upload_volumetric_uniforms(program=self.program_shading)


        w, h = self.window_size
#       w, h = self.window_size

        # Manual Ray Construction Uniforms
#       # Manual Ray Construction Uniforms
        if "uPixel00Coordinates" in self.program_shading:
#       if "uPixel00Coordinates" in self.program_shading:
            cam_u, cam_v, cam_w = self.camera.get_basis_vectors()
#           cam_u, cam_v, cam_w = self.camera.get_basis_vectors()

            focal_length: float = rr.vector.length(self.camera.look_from - self.camera.look_at)
#           focal_length: float = rr.vector.length(self.camera.look_from - self.camera.look_at)
            # Use the camera's actual FOV: this was hardcoded to 60.0 and would silently desync
#           # Use the camera's actual FOV: this was hardcoded to 60.0 and would silently desync
            # the ray-traced primary rays from the rasterized G-Buffer if the FOV ever changed.
#           # the ray-traced primary rays from the rasterized G-Buffer if the FOV ever changed.
            tan_half_fovy: float = np.tan(np.deg2rad(self.camera.fov) / 2.0)
#           tan_half_fovy: float = np.tan(np.deg2rad(self.camera.fov) / 2.0)
            viewport_height: float = 2.0 * tan_half_fovy * focal_length
#           viewport_height: float = 2.0 * tan_half_fovy * focal_length
            viewport_width: float = viewport_height * self.aspect_ratio
#           viewport_width: float = viewport_height * self.aspect_ratio

            viewport_u: rr.Vector3 = cam_u * viewport_width
#           viewport_u: rr.Vector3 = cam_u * viewport_width
            viewport_v: rr.Vector3 = cam_v * viewport_height
#           viewport_v: rr.Vector3 = cam_v * viewport_height

            pixel_delta_u: rr.Vector3 = viewport_u / w
#           pixel_delta_u: rr.Vector3 = viewport_u / w
            pixel_delta_v: rr.Vector3 = viewport_v / h
#           pixel_delta_v: rr.Vector3 = viewport_v / h

            viewport_upper_left: rr.Vector3 = self.camera.look_from - (cam_w * focal_length) - (viewport_u / 2.0) - (viewport_v / 2.0)
#           viewport_upper_left: rr.Vector3 = self.camera.look_from - (cam_w * focal_length) - (viewport_u / 2.0) - (viewport_v / 2.0)
            pixel_00_coordinates: rr.Vector3 = viewport_upper_left + 0.5 * (pixel_delta_u + pixel_delta_v)
#           pixel_00_coordinates: rr.Vector3 = viewport_upper_left + 0.5 * (pixel_delta_u + pixel_delta_v)

            self.program_shading["uPixel00Coordinates"] = tuple(pixel_00_coordinates)
#           self.program_shading["uPixel00Coordinates"] = tuple(pixel_00_coordinates)
            self.program_shading["uPixelDeltaU"] = tuple(pixel_delta_u)
#           self.program_shading["uPixelDeltaU"] = tuple(pixel_delta_u)
            self.program_shading["uPixelDeltaV"] = tuple(pixel_delta_v)
#           self.program_shading["uPixelDeltaV"] = tuple(pixel_delta_v)
            self.program_shading["uJitter"] = (jitter_x, jitter_y)
#           self.program_shading["uJitter"] = (jitter_x, jitter_y)

        if "uNRCEnabled" in self.program_shading:
#       if "uNRCEnabled" in self.program_shading:
            self.program_shading["uNRCEnabled"] = self.is_nrc_enabled
#           self.program_shading["uNRCEnabled"] = self.is_nrc_enabled
        if "uReSTIREnabled" in self.program_shading:
#       if "uReSTIREnabled" in self.program_shading:
            self.program_shading["uReSTIREnabled"] = self.is_restir_enabled
#           self.program_shading["uReSTIREnabled"] = self.is_restir_enabled
        if "uVRTEnabled" in self.program_shading:
#       if "uVRTEnabled" in self.program_shading:
            self.program_shading["uVRTEnabled"] = self.is_vrt_enabled
#           self.program_shading["uVRTEnabled"] = self.is_vrt_enabled
        if "uVRTVisualize" in self.program_shading:
#       if "uVRTVisualize" in self.program_shading:
            self.program_shading["uVRTVisualize"] = self.is_vrt_visualize
#           self.program_shading["uVRTVisualize"] = self.is_vrt_visualize
        if "uNRCPositionOffset" in self.program_shading:
#       if "uNRCPositionOffset" in self.program_shading:
            self.program_shading["uNRCPositionOffset"] = self.nrc_position_offset
#           self.program_shading["uNRCPositionOffset"] = self.nrc_position_offset
            self.program_shading["uNRCPositionScale"] = self.nrc_position_scale
#           self.program_shading["uNRCPositionScale"] = self.nrc_position_scale
        if "uResolution" in self.program_shading:
#       if "uResolution" in self.program_shading:
            self.program_shading["uResolution"] = (w, h)
#           self.program_shading["uResolution"] = (w, h)
        if "uPreviousTransformViewProjection" in self.program_shading and self.last_view_proj_matrix is not None:
#       if "uPreviousTransformViewProjection" in self.program_shading and self.last_view_proj_matrix is not None:
            self.program_shading.write("uPreviousTransformViewProjection", self.last_view_proj_matrix.astype(dtype=np.float32).tobytes())
#           self.program_shading.write("uPreviousTransformViewProjection", self.last_view_proj_matrix.astype(dtype=np.float32).tobytes())
        if "uPreviousCameraPosition" in self.program_shading and self.last_camera_position is not None:
#       if "uPreviousCameraPosition" in self.program_shading and self.last_camera_position is not None:
            self.program_shading["uPreviousCameraPosition"] = self.last_camera_position
#           self.program_shading["uPreviousCameraPosition"] = self.last_camera_position
        if "uResetHistory" in self.program_shading:
#       if "uResetHistory" in self.program_shading:
            self.program_shading["uResetHistory"] = self.is_scene_reset
#           self.program_shading["uResetHistory"] = self.is_scene_reset

        shader_resources: dict[str, typing.Any] = self.get_shader_resources()
#       shader_resources: dict[str, typing.Any] = self.get_shader_resources()

        # Online Neural Radiance Cache (NRC) Training Passes (Gather + Backprop with Adam)
#       # Online Neural Radiance Cache (NRC) Training Passes (Gather + Backprop with Adam)
        if self.is_nrc_enabled:
#       if self.is_nrc_enabled:
            # 1. Gather pass: collect training records from the scene
#           # 1. Gather pass: collect training records from the scene
            # The gather pass reads texcoord V out of the tangent target's w channel, so it can resolve
#           # The gather pass reads texcoord V out of the tangent target's w channel, so it can resolve
            # the primary vertex's material from the same maps the shading pass uses on that pixel.
#           # the primary vertex's material from the same maps the shading pass uses on that pixel.
            if "uTrainingStep" in self.program_nrc_gather:
#           if "uTrainingStep" in self.program_nrc_gather:
                self.program_nrc_gather["uTrainingStep"] = self.nrc_training_step
#               self.program_nrc_gather["uTrainingStep"] = self.nrc_training_step
            if "uNRCPositionOffset" in self.program_nrc_gather:
#           if "uNRCPositionOffset" in self.program_nrc_gather:
                self.program_nrc_gather["uNRCPositionOffset"] = self.nrc_position_offset
#               self.program_nrc_gather["uNRCPositionOffset"] = self.nrc_position_offset
                self.program_nrc_gather["uNRCPositionScale"] = self.nrc_position_scale
#               self.program_nrc_gather["uNRCPositionScale"] = self.nrc_position_scale
            if "uResolution" in self.program_nrc_gather:
#           if "uResolution" in self.program_nrc_gather:
                self.program_nrc_gather["uResolution"] = (w, h)
#               self.program_nrc_gather["uResolution"] = (w, h)
            if "uCameraGlobalPosition" in self.program_nrc_gather:
#           if "uCameraGlobalPosition" in self.program_nrc_gather:
                self.program_nrc_gather["uCameraGlobalPosition"] = tuple(self.camera.look_from)
#               self.program_nrc_gather["uCameraGlobalPosition"] = tuple(self.camera.look_from)
            if "uUseHdri" in self.program_nrc_gather:
#           if "uUseHdri" in self.program_nrc_gather:
                self.program_nrc_gather["uUseHdri"] = self.use_hdri
#               self.program_nrc_gather["uUseHdri"] = self.use_hdri
            # Same sky cloud layer and cloud noise as the shading pass: the training target sees the cloudy
#           # Same sky cloud layer and cloud noise as the shading pass: the training target sees the cloudy
            # sky and the media shadows the shading pass renders.
#           # sky and the media shadows the shading pass renders.
            self.upload_volumetric_uniforms(program=self.program_nrc_gather)
#           self.upload_volumetric_uniforms(program=self.program_nrc_gather)
            # The scene texture array comes from the same shader_resources as the shading pass. Without it
#           # The scene texture array comes from the same shader_resources as the shading pass. Without it
            # the training target would be built from placeholder material scalars on every textured surface.
#           # the training target would be built from placeholder material scalars on every textured surface.
            if "uPointLightCount" in self.program_nrc_gather:
#           if "uPointLightCount" in self.program_nrc_gather:
                self.program_nrc_gather["uPointLightCount"] = len(point_lights)
#               self.program_nrc_gather["uPointLightCount"] = len(point_lights)
                cumulative_prob = 0.0
#               cumulative_prob = 0.0
                for i, light in enumerate(point_lights):
#               for i, light in enumerate(point_lights):
                    pos = light["position"]
#                   pos = light["position"]
                    color = light["color"]
#                   color = light["color"]
                    r = light["radius"]
#                   r = light["radius"]
                    power = get_light_power(light)
#                   power = get_light_power(light)
                    prob = power / total_power if total_power > 0.0 else 1.0 / len(point_lights)
#                   prob = power / total_power if total_power > 0.0 else 1.0 / len(point_lights)
                    cumulative_prob += prob
#                   cumulative_prob += prob
                    if f"uPointLights[{i}].position" in self.program_nrc_gather:
#                   if f"uPointLights[{i}].position" in self.program_nrc_gather:
                        self.program_nrc_gather[f"uPointLights[{i}].position"] = pos
#                       self.program_nrc_gather[f"uPointLights[{i}].position"] = pos
                        self.program_nrc_gather[f"uPointLights[{i}].color"] = color
#                       self.program_nrc_gather[f"uPointLights[{i}].color"] = color
                        self.program_nrc_gather[f"uPointLights[{i}].radius"] = r
#                       self.program_nrc_gather[f"uPointLights[{i}].radius"] = r
                        if f"uPointLights[{i}].cdf" in self.program_nrc_gather:
#                       if f"uPointLights[{i}].cdf" in self.program_nrc_gather:
                            self.program_nrc_gather[f"uPointLights[{i}].cdf"] = cumulative_prob
#                           self.program_nrc_gather[f"uPointLights[{i}].cdf"] = cumulative_prob
                            self.program_nrc_gather[f"uPointLights[{i}].pdf"] = prob
#                           self.program_nrc_gather[f"uPointLights[{i}].pdf"] = prob

            # 128 workgroups x 64 invocations = 8192 paths, and nrc_gather_cs.slang records at most one
#           # 128 workgroups x 64 invocations = 8192 paths, and nrc_gather_cs.slang records at most one
            # vertex per path, which is exactly the capacity of ssbo_nrc_training_records. Raising this
#           # vertex per path, which is exactly the capacity of ssbo_nrc_training_records. Raising this
            # count without resizing that buffer silently discards the surplus records.
#           # count without resizing that buffer silently discards the surplus records.
            self.program_nrc_gather.run(command_encoder=command_encoder, resources=shader_resources, group_x=128, group_y=1, group_z=1)
#           self.program_nrc_gather.run(command_encoder=command_encoder, resources=shader_resources, group_x=128, group_y=1, group_z=1)

            # 2. Train MLP pass: mini-batch SGD with Adam optimizer on GPU
#           # 2. Train MLP pass: mini-batch SGD with Adam optimizer on GPU
            if "uTrainingStep" in self.program_nrc_train:
#           if "uTrainingStep" in self.program_nrc_train:
                self.program_nrc_train["uTrainingStep"] = self.nrc_training_step
#               self.program_nrc_train["uTrainingStep"] = self.nrc_training_step
            if "uNRCPositionOffset" in self.program_nrc_train:
#           if "uNRCPositionOffset" in self.program_nrc_train:
                self.program_nrc_train["uNRCPositionOffset"] = self.nrc_position_offset
#               self.program_nrc_train["uNRCPositionOffset"] = self.nrc_position_offset
                self.program_nrc_train["uNRCPositionScale"] = self.nrc_position_scale
#               self.program_nrc_train["uNRCPositionScale"] = self.nrc_position_scale

            self.program_nrc_train.run(command_encoder=command_encoder, resources=shader_resources, group_x=1, group_y=1, group_z=1)
#           self.program_nrc_train.run(command_encoder=command_encoder, resources=shader_resources, group_x=1, group_y=1, group_z=1)
            self.nrc_training_step += 1
#           self.nrc_training_step += 1

        # The shading pass blends into the other half of the accumulation pair, reading this one as history.
#       # The shading pass blends into the other half of the accumulation pair, reading this one as history.
        texture_accum_target: wgpu.GPUTexture = self.texture_accum_pair[1] if self.texture_accum is self.texture_accum_pair[0] else self.texture_accum_pair[0]
#       texture_accum_target: wgpu.GPUTexture = self.texture_accum_pair[1] if self.texture_accum is self.texture_accum_pair[0] else self.texture_accum_pair[0]
        shading_resources: dict[str, typing.Any] = dict(shader_resources)
#       shading_resources: dict[str, typing.Any] = dict(shader_resources)
        shading_resources["textureAccumHistory"] = self.texture_accum
#       shading_resources["textureAccumHistory"] = self.texture_accum
        shading_resources["textureAccum"] = texture_accum_target
#       shading_resources["textureAccum"] = texture_accum_target

        gx, gy = (w + 15) // 16, (h + 15) // 16
#       gx, gy = (w + 15) // 16, (h + 15) // 16
        self.program_shading.run(command_encoder=command_encoder, resources=shading_resources, group_x=gx, group_y=gy, group_z=1)
#       self.program_shading.run(command_encoder=command_encoder, resources=shading_resources, group_x=gx, group_y=gy, group_z=1)
        self.texture_accum = texture_accum_target
#       self.texture_accum = texture_accum_target
        self.last_view_proj_matrix = current_view_proj
#       self.last_view_proj_matrix = current_view_proj
        self.last_camera_position = tuple(self.camera.look_from)
#       self.last_camera_position = tuple(self.camera.look_from)
        self.is_scene_reset = False
#       self.is_scene_reset = False

        # 3. Post Processing Pipeline
#       # 3. Post Processing Pipeline
        # The shading pass leaves its result in texture_accum (the temporal history buffer).
#       # The shading pass leaves its result in texture_accum (the temporal history buffer).
        # Feed it in as a read-only input and ping-pong between ping/output so the
#       # Feed it in as a read-only input and ping-pong between ping/output so the
        # accumulation buffer is never overwritten by a post-processing pass.
#       # accumulation buffer is never overwritten by a post-processing pass.
        current_read_texture: wgpu.GPUTexture = self.texture_accum
#       current_read_texture: wgpu.GPUTexture = self.texture_accum
        current_write_texture: wgpu.GPUTexture = self.texture_ping
#       current_write_texture: wgpu.GPUTexture = self.texture_ping

        for post_program in self.post_processing_pipeline:
#       for post_program in self.post_processing_pipeline:
            post_resources: dict[str, typing.Any] = {
#           post_resources: dict[str, typing.Any] = {
                "textureOutput": current_write_texture,                     # Output
#               "textureOutput": current_write_texture,                     # Output
                "textureInput": (current_read_texture, self.sampler_screen), # Input (FXAA samples it filtered, the others load texels)
#               "textureInput": (current_read_texture, self.sampler_screen), # Input (FXAA samples it filtered, the others load texels)
            }
#           }
            post_program.run(command_encoder=command_encoder, resources=post_resources, group_x=gx, group_y=gy, group_z=1)
#           post_program.run(command_encoder=command_encoder, resources=post_resources, group_x=gx, group_y=gy, group_z=1)

            # Swap textures
#           # Swap textures
            current_read_texture, current_write_texture = current_write_texture, current_read_texture
#           current_read_texture, current_write_texture = current_write_texture, current_read_texture

            # texture_accum is read-only outside the shading pass; once the first pass has
#           # texture_accum is read-only outside the shading pass; once the first pass has
            # consumed it, keep ping-ponging between ping and output only.
#           # consumed it, keep ping-ponging between ping and output only.
            if current_write_texture is self.texture_accum:
#           if current_write_texture is self.texture_accum:
                current_write_texture = self.texture_output
#               current_write_texture = self.texture_output

        # 4. Display Result to Screen
#       # 4. Display Result to Screen
        self.render_to_screen(command_encoder=command_encoder, texture=current_read_texture)
#       self.render_to_screen(command_encoder=command_encoder, texture=current_read_texture)
        pass
#       pass

    def get_key_name(self, event: dict[str, typing.Any]) -> str:
#   def get_key_name(self, event: dict[str, typing.Any]) -> str:
        key: str = event["key"]
#       key: str = event["key"]
        return self.key_names.get(key, key.upper())
#       return self.key_names.get(key, key.upper())

    def on_key_down(self, event: dict[str, typing.Any]) -> None:
#   def on_key_down(self, event: dict[str, typing.Any]) -> None:
        key: str = self.get_key_name(event=event)
#       key: str = self.get_key_name(event=event)
        if key in self.key_state:
#       if key in self.key_state:
            self.key_state[key] = True
#           self.key_state[key] = True
        self.on_key_event(key=key)
#       self.on_key_event(key=key)
        pass
#       pass

    def on_key_up(self, event: dict[str, typing.Any]) -> None:
#   def on_key_up(self, event: dict[str, typing.Any]) -> None:
        key: str = self.get_key_name(event=event)
#       key: str = self.get_key_name(event=event)
        if key in self.key_state:
#       if key in self.key_state:
            self.key_state[key] = False
#           self.key_state[key] = False
        pass
#       pass

    def on_key_event(self, key: str) -> None:
#   def on_key_event(self, key: str) -> None:
        # Key press actions (rendercanvas does not repeat key_down while a key is held).
#       # Key press actions (rendercanvas does not repeat key_down while a key is held).
        if key == "I":
#       if key == "I":
            self.denoise_and_show()
#           self.denoise_and_show()
        elif key == "0":
#       elif key == "0":
            self.render_mode = RenderMode.PATH_TRACE
#           self.render_mode = RenderMode.PATH_TRACE
            self.frame_count = 0
#           self.frame_count = 0
        elif key == "1":
#       elif key == "1":
            self.render_mode = RenderMode.ALBEDO
#           self.render_mode = RenderMode.ALBEDO
            self.frame_count = 0
#           self.frame_count = 0
        elif key == "2":
#       elif key == "2":
            self.render_mode = RenderMode.NORMAL
#           self.render_mode = RenderMode.NORMAL
            self.frame_count = 0
#           self.frame_count = 0
        elif key == "3":
#       elif key == "3":
            self.render_mode = RenderMode.POSITION
#           self.render_mode = RenderMode.POSITION
            self.frame_count = 0
#           self.frame_count = 0
        elif key == "4":
#       elif key == "4":
            self.render_mode = RenderMode.TANGENT
#           self.render_mode = RenderMode.TANGENT
            self.frame_count = 0
#           self.frame_count = 0
        elif key == "N":
#       elif key == "N":
            self.is_nrc_enabled = not self.is_nrc_enabled
#           self.is_nrc_enabled = not self.is_nrc_enabled
            if self.is_nrc_enabled:
#           if self.is_nrc_enabled:
                self.reset_neural_cache_weights()
#               self.reset_neural_cache_weights()
            self.frame_count = 0
#           self.frame_count = 0
        elif key == "R":
#       elif key == "R":
            self.is_restir_enabled = not self.is_restir_enabled
#           self.is_restir_enabled = not self.is_restir_enabled
            self.frame_count = 0
#           self.frame_count = 0
        elif key == "V":
#       elif key == "V":
            self.is_vrt_enabled = not self.is_vrt_enabled
#           self.is_vrt_enabled = not self.is_vrt_enabled
            self.frame_count = 0
#           self.frame_count = 0
        elif key == "H":
#       elif key == "H":
            self.is_vrt_visualize = not self.is_vrt_visualize
#           self.is_vrt_visualize = not self.is_vrt_visualize
            self.frame_count = 0
#           self.frame_count = 0
        elif key == "C":
#       elif key == "C":
            self.is_sky_clouds_enabled = not self.is_sky_clouds_enabled
#           self.is_sky_clouds_enabled = not self.is_sky_clouds_enabled
            self.bake_sky_clouds()
#           self.bake_sky_clouds()
            self.frame_count = 0
#           self.frame_count = 0
        elif key == "M":
#       elif key == "M":
            self.is_media_enabled = not self.is_media_enabled
#           self.is_media_enabled = not self.is_media_enabled
            self.frame_count = 0
#           self.frame_count = 0
        elif key == "TAB":
#       elif key == "TAB":
            self.set_scene_by_index(index=self.current_scene_index + 1)
#           self.set_scene_by_index(index=self.current_scene_index + 1)
            self.frame_count = 0
#           self.frame_count = 0
        elif key == "F11":
#       elif key == "F11":
            self.set_fullscreen(is_fullscreen=not self.is_fullscreen)
#           self.set_fullscreen(is_fullscreen=not self.is_fullscreen)
        elif key == "ESCAPE":
#       elif key == "ESCAPE":
            # Same path as the title-bar close button: emits "close" (on_close) and ends the event loop.
#           # Same path as the title-bar close button: emits "close" (on_close) and ends the event loop.
            self.canvas.close()
#           self.canvas.close()
            return
#           return
        if self.frame_count == 0:
#       if self.frame_count == 0:
            self.is_scene_reset = True
#           self.is_scene_reset = True
            self.last_view_proj_matrix = None
#           self.last_view_proj_matrix = None
            self.last_camera_position = None
#           self.last_camera_position = None
        pass
#       pass

    def set_fullscreen(self, is_fullscreen: bool) -> None:
#   def set_fullscreen(self, is_fullscreen: bool) -> None:
        # F11 / --fullscreen, as in moderngl_window's glfw backend: the window takes the primary monitor's video mode,
#       # F11 / --fullscreen, as in moderngl_window's glfw backend: the window takes the primary monitor's video mode,
        # and leaving fullscreen restores the previous position and size. Only the swapchain follows the new
#       # and leaving fullscreen restores the previous position and size. Only the swapchain follows the new
        # window size; the render targets stay at window_size and are stretched, like any other resize.
#       # window size; the render targets stay at window_size and are stretched, like any other resize.
        if is_fullscreen == self.is_fullscreen:
#       if is_fullscreen == self.is_fullscreen:
            return
#           return
        monitor: typing.Any = glfw.get_primary_monitor()
#       monitor: typing.Any = glfw.get_primary_monitor()
        mode: typing.Any = glfw.get_video_mode(monitor)
#       mode: typing.Any = glfw.get_video_mode(monitor)
        refresh_rate: int = mode.refresh_rate if self.window_options.vsync else glfw.DONT_CARE
#       refresh_rate: int = mode.refresh_rate if self.window_options.vsync else glfw.DONT_CARE
        if is_fullscreen:
#       if is_fullscreen:
            self.windowed_rect = (*glfw.get_window_pos(self.glfw_window), *glfw.get_window_size(self.glfw_window))
#           self.windowed_rect = (*glfw.get_window_pos(self.glfw_window), *glfw.get_window_size(self.glfw_window))
            glfw.set_window_monitor(self.glfw_window, monitor, 0, 0, mode.size.width, mode.size.height, refresh_rate)
#           glfw.set_window_monitor(self.glfw_window, monitor, 0, 0, mode.size.width, mode.size.height, refresh_rate)
        else:
#       else:
            # A None monitor means windowed mode in glfw; its type stubs only declare the monitor pointer.
#           # A None monitor means windowed mode in glfw; its type stubs only declare the monitor pointer.
            glfw.set_window_monitor(self.glfw_window, None, *self.windowed_rect, refresh_rate) # type: ignore[arg-type]
#           glfw.set_window_monitor(self.glfw_window, None, *self.windowed_rect, refresh_rate)
        self.is_fullscreen = is_fullscreen
#       self.is_fullscreen = is_fullscreen
        pass
#       pass

    def on_close(self, event: dict[str, typing.Any]) -> None:
#   def on_close(self, event: dict[str, typing.Any]) -> None:
        print(f"[CLOSE]")
#       print(f"[CLOSE]")
        self.oidn_device.release()
#       self.oidn_device.release()
        pass
#       pass
