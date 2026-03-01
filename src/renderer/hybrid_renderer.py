import moderngl as mgl
import moderngl as mgl
import moderngl_window as mglw
import moderngl_window as mglw
from moderngl_window.context.base import BaseKeys, BaseWindow, KeyModifiers
from moderngl_window.context.base import BaseKeys, BaseWindow, KeyModifiers
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
import os
import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
import cv2
import cv2
import pyoidn     # type: ignore[import-untyped]
import pyoidn
from src.renderer.shader_compiler import resolve_includes
from src.renderer.shader_compiler import resolve_includes
from src.scene.scene_builder import SceneBuilder, SceneBatch
from src.scene.scene_builder import SceneBuilder, SceneBatch
from src.scene.camera import Camera
from src.scene.camera import Camera
from src.core.common_types import vec2i32, vec3i32, vec4i32, vec2f32, vec3f32, vec4f32, Material, PointLight
from src.core.common_types import vec2i32, vec3i32, vec4i32, vec2f32, vec3f32, vec4f32, Material, PointLight

class HybridRenderer(mglw.WindowConfig): # type: ignore[name-defined, misc]
    # Main Renderer Class implementing a Hybrid Pipeline:
#   # Main Renderer Class implementing a Hybrid Pipeline:
    # 1. Rasterization Pass: Renders scene geometry to G-Buffer (Position, Normal, Albedo, etc.).
#   # 1. Rasterization Pass: Renders scene geometry to G-Buffer (Position, Normal, Albedo, etc.).
    # 2. Compute Pass (Ray Tracing): Uses G-Buffer + Ray Tracing (BVH) to calculate lighting/reflections.
#   # 2. Compute Pass (Ray Tracing): Uses G-Buffer + Ray Tracing (BVH) to calculate lighting/reflections.
    # 3. Denoise Pass: Cleans up the noisy ray-traced output (using A-Trous filter or OIDN).
#   # 3. Denoise Pass: Cleans up the noisy ray-traced output (using A-Trous filter or OIDN).
    # 4. Post Processing Pass: Applies effects like Chromatic Aberration and Vignette.
#   # 4. Post Processing Pass: Applies effects like Chromatic Aberration and Vignette.
    # 5. Composite Pass: Displays the final result to the screen.
#   # 5. Composite Pass: Displays the final result to the screen.
    gl_version: vec2i32 = (4, 3)
#   gl_version: vec2i32 = (4, 3)
    title: str = "Hybrid Rendering: Rasterization + Path Tracing"
#   title: str = "Hybrid Rendering: Rasterization + Path Tracing"
    window_size: vec2i32 = (800, 600)
#   window_size: vec2i32 = (800, 600)
    aspect_ratio: float = window_size[0] / window_size[1]
#   aspect_ratio: float = window_size[0] / window_size[1]
    resizable: bool = False
#   resizable: bool = False
    resource_dir: pl.Path = pl.Path(__file__).parent.resolve(strict=False)
#   resource_dir: pl.Path = pl.Path(__file__).parent.resolve(strict=False)

    def __init__(self, **kwargs: dict[str, typing.Any]) -> None:
#   def __init__(self, **kwargs: dict[str, typing.Any]) -> None:
        super().__init__(**kwargs)
#       super().__init__(**kwargs)

        self.frame_count: int = 0
#       self.frame_count: int = 0
        self.last_view_matrix: rr.Matrix44 = None
#       self.last_view_matrix: rr.Matrix44 = None

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
        self.texture_geometry_global_position: mgl.Texture = self.ctx.texture(size=self.window_size, components=4, dtype="f4")
#       self.texture_geometry_global_position: mgl.Texture = self.ctx.texture(size=self.window_size, components=4, dtype="f4")
        self.texture_geometry_global_normal: mgl.Texture = self.ctx.texture(size=self.window_size, components=4, dtype="f4")
#       self.texture_geometry_global_normal: mgl.Texture = self.ctx.texture(size=self.window_size, components=4, dtype="f4")
        self.texture_geometry_albedo: mgl.Texture = self.ctx.texture(size=self.window_size, components=4, dtype="f4")
#       self.texture_geometry_albedo: mgl.Texture = self.ctx.texture(size=self.window_size, components=4, dtype="f4")
        self.texture_geometry_global_tangent: mgl.Texture = self.ctx.texture(size=self.window_size, components=4, dtype="f4")
#       self.texture_geometry_global_tangent: mgl.Texture = self.ctx.texture(size=self.window_size, components=4, dtype="f4")
        self.texture_depth: mgl.Texture = self.ctx.depth_texture(size=self.window_size)
#       self.texture_depth: mgl.Texture = self.ctx.depth_texture(size=self.window_size)

        # Framebuffer to render geometry into these textures
#       # Framebuffer to render geometry into these textures
        self.gbuffer: mgl.Framebuffer = self.ctx.framebuffer(
#       self.gbuffer: mgl.Framebuffer = self.ctx.framebuffer(
            color_attachments=[
#           color_attachments=[
                self.texture_geometry_global_position,
#               self.texture_geometry_global_position,
                self.texture_geometry_global_normal,
#               self.texture_geometry_global_normal,
                self.texture_geometry_albedo,
#               self.texture_geometry_albedo,
                self.texture_geometry_global_tangent,
#               self.texture_geometry_global_tangent,
            ],
#           ],
            depth_attachment=self.texture_depth,
#           depth_attachment=self.texture_depth,
        )
#       )

        self.texture_output: mgl.Texture = self.ctx.texture(size=self.window_size, components=4, dtype="f4")
#       self.texture_output: mgl.Texture = self.ctx.texture(size=self.window_size, components=4, dtype="f4")
        self.texture_output.filter = (mgl.NEAREST, mgl.NEAREST)
#       self.texture_output.filter = (mgl.NEAREST, mgl.NEAREST)

        self.texture_accum: mgl.Texture = self.ctx.texture(size=self.window_size, components=4, dtype="f4")
#       self.texture_accum: mgl.Texture = self.ctx.texture(size=self.window_size, components=4, dtype="f4")
        self.texture_accum.filter = (mgl.NEAREST, mgl.NEAREST)
#       self.texture_accum.filter = (mgl.NEAREST, mgl.NEAREST)

        self.texture_ping: mgl.Texture = self.ctx.texture(size=self.window_size, components=4, dtype="f4")
#       self.texture_ping: mgl.Texture = self.ctx.texture(size=self.window_size, components=4, dtype="f4")
        self.texture_ping.filter = (mgl.NEAREST, mgl.NEAREST)
#       self.texture_ping.filter = (mgl.NEAREST, mgl.NEAREST)

        # OIDN Setup
#       # OIDN Setup
        self.oidn_device: pyoidn.Device = pyoidn.Device(device_type=pyoidn.OIDN_DEVICE_TYPE_CPU)
#       self.oidn_device: pyoidn.Device = pyoidn.Device(device_type=pyoidn.OIDN_DEVICE_TYPE_CPU)
        self.oidn_device.commit()
#       self.oidn_device.commit()

        # HDRI Texture Loading
#       # HDRI Texture Loading
        hdri_path: pl.Path = self.resource_dir / "../assets/wasteland_clouds_puresky_4k.exr"
#       hdri_path: pl.Path = self.resource_dir / "../assets/wasteland_clouds_puresky_4k.exr"
        self.use_hdri: bool = False # hdri_path.exists()
#       self.use_hdri: bool = False # hdri_path.exists()
        if self.use_hdri:
#       if self.use_hdri:
            loaded_data = cv2.imread(str(hdri_path), cv2.IMREAD_UNCHANGED)
#           loaded_data = cv2.imread(str(hdri_path), cv2.IMREAD_UNCHANGED)
            if loaded_data is None:
#           if loaded_data is None:
                raise RuntimeError(f"Failed to load HDRI: {hdri_path}")
#               raise RuntimeError(f"Failed to load HDRI: {hdri_path}")
            hdri_data: npt.NDArray[np.float32] = loaded_data.astype(dtype=np.float32)
#           hdri_data: npt.NDArray[np.float32] = loaded_data.astype(dtype=np.float32)
            hdri_data = typing.cast(npt.NDArray[np.float32], cv2.cvtColor(hdri_data, cv2.COLOR_BGR2RGB))
#           hdri_data = typing.cast(npt.NDArray[np.float32], cv2.cvtColor(hdri_data, cv2.COLOR_BGR2RGB))
            hdri_data = np.ascontiguousarray(np.flipud(hdri_data))
#           hdri_data = np.ascontiguousarray(np.flipud(hdri_data))
            h, w, c = hdri_data.shape
#           h, w, c = hdri_data.shape
            self.texture_hdri: mgl.Texture = self.ctx.texture((w, h), components=3, dtype="f4", data=hdri_data.tobytes())
#           self.texture_hdri: mgl.Texture = self.ctx.texture((w, h), components=3, dtype="f4", data=hdri_data.tobytes())
            self.texture_hdri.filter = (mgl.LINEAR, mgl.LINEAR)
#           self.texture_hdri.filter = (mgl.LINEAR, mgl.LINEAR)
            self.texture_hdri.repeat_x = True
#           self.texture_hdri.repeat_x = True
            self.texture_hdri.repeat_y = False
#           self.texture_hdri.repeat_y = False
        else:
#       else:
            # Create dummy 1x1 texture
#           # Create dummy 1x1 texture
            self.texture_hdri = self.ctx.texture((1, 1), components=3, dtype="f4", data=np.zeros(3, dtype=np.float32).tobytes())
#           self.texture_hdri = self.ctx.texture((1, 1), components=3, dtype="f4", data=np.zeros(3, dtype=np.float32).tobytes())

        # Texture Array for Scene Materials
#       # Texture Array for Scene Materials
        self.texture_array_size: int = 2048
#       self.texture_array_size: int = 2048
        self.texture_array_layers: int = 64
#       self.texture_array_layers: int = 64
        self.texture_array: mgl.TextureArray = self.ctx.texture_array(
#       self.texture_array: mgl.TextureArray = self.ctx.texture_array(
            size=(self.texture_array_size, self.texture_array_size, self.texture_array_layers), components=4, dtype="f4"
#           size=(self.texture_array_size, self.texture_array_size, self.texture_array_layers), components=4, dtype="f4"
        )
#       )
        self.texture_array.filter = (mgl.LINEAR_MIPMAP_LINEAR, mgl.LINEAR)
#       self.texture_array.filter = (mgl.LINEAR_MIPMAP_LINEAR, mgl.LINEAR)

        self.texture_cache: dict[str, float] = {}
#       self.texture_cache: dict[str, float] = {}
        self.next_texture_layer: int = 0
#       self.next_texture_layer: int = 0

        # -----------------------------
#       # -----------------------------
        # 2. Rasterization Shader (Geometry Pass)
#       # 2. Rasterization Shader (Geometry Pass)
        # -----------------------------
#       # -----------------------------
        hybrid_geometry_vs_path: pl.Path = self.resource_dir / "../shaders/hybrid_geometry_vs.glsl"
#       hybrid_geometry_vs_path: pl.Path = self.resource_dir / "../shaders/hybrid_geometry_vs.glsl"
        hybrid_geometry_vs_code: str = resolve_includes(hybrid_geometry_vs_path.read_text(encoding="utf-8"), self.resource_dir / "../shaders")
#       hybrid_geometry_vs_code: str = resolve_includes(hybrid_geometry_vs_path.read_text(encoding="utf-8"), self.resource_dir / "../shaders")
        hybrid_geometry_fs_path: pl.Path = self.resource_dir / "../shaders/hybrid_geometry_fs.glsl"
#       hybrid_geometry_fs_path: pl.Path = self.resource_dir / "../shaders/hybrid_geometry_fs.glsl"
        hybrid_geometry_fs_code: str = resolve_includes(hybrid_geometry_fs_path.read_text(encoding="utf-8"), self.resource_dir / "../shaders")
#       hybrid_geometry_fs_code: str = resolve_includes(hybrid_geometry_fs_path.read_text(encoding="utf-8"), self.resource_dir / "../shaders")
        self.program_geometry: mgl.Program = self.ctx.program(
#       self.program_geometry: mgl.Program = self.ctx.program(
              vertex_shader=hybrid_geometry_vs_code,
#             vertex_shader=hybrid_geometry_vs_code,
            fragment_shader=hybrid_geometry_fs_code,
#           fragment_shader=hybrid_geometry_fs_code,
        )
#       )

        # -----------------------------
#       # -----------------------------
        # 3. Compute Shader (Shading Pass)
#       # 3. Compute Shader (Shading Pass)
        # -----------------------------
#       # -----------------------------
        hybrid_shading_cs_path: pl.Path = self.resource_dir / "../shaders/hybrid_shading_cs.glsl"
#       hybrid_shading_cs_path: pl.Path = self.resource_dir / "../shaders/hybrid_shading_cs.glsl"
        hybrid_shading_cs_code: str = resolve_includes(hybrid_shading_cs_path.read_text(encoding="utf-8"), self.resource_dir / "../shaders")
#       hybrid_shading_cs_code: str = resolve_includes(hybrid_shading_cs_path.read_text(encoding="utf-8"), self.resource_dir / "../shaders")
        self.program_shading: mgl.ComputeShader = self.ctx.compute_shader(source=hybrid_shading_cs_code)
#       self.program_shading: mgl.ComputeShader = self.ctx.compute_shader(source=hybrid_shading_cs_code)

        # Denoise Shader
#       # Denoise Shader
        hybrid_denoise_cs_path: pl.Path = self.resource_dir / "../shaders/hybrid_denoise_cs.glsl"
#       hybrid_denoise_cs_path: pl.Path = self.resource_dir / "../shaders/hybrid_denoise_cs.glsl"
        hybrid_denoise_cs_code: str = resolve_includes(hybrid_denoise_cs_path.read_text(encoding="utf-8"), self.resource_dir / "../shaders")
#       hybrid_denoise_cs_code: str = resolve_includes(hybrid_denoise_cs_path.read_text(encoding="utf-8"), self.resource_dir / "../shaders")
        self.program_denoise: mgl.ComputeShader = self.ctx.compute_shader(source=hybrid_denoise_cs_code)
#       self.program_denoise: mgl.ComputeShader = self.ctx.compute_shader(source=hybrid_denoise_cs_code)

        # -----------------------------
#       # -----------------------------
        # 4. Rasterization Shader (Renderer Pass)
#       # 4. Rasterization Shader (Renderer Pass)
        # -----------------------------
#       # -----------------------------
        hybrid_renderer_vs_path: pl.Path = self.resource_dir / "../shaders/hybrid_renderer_vs.glsl"
#       hybrid_renderer_vs_path: pl.Path = self.resource_dir / "../shaders/hybrid_renderer_vs.glsl"
        hybrid_renderer_vs_code: str = resolve_includes(hybrid_renderer_vs_path.read_text(encoding="utf-8"), self.resource_dir / "../shaders")
#       hybrid_renderer_vs_code: str = resolve_includes(hybrid_renderer_vs_path.read_text(encoding="utf-8"), self.resource_dir / "../shaders")
        hybrid_renderer_fs_path: pl.Path = self.resource_dir / "../shaders/hybrid_renderer_fs.glsl"
#       hybrid_renderer_fs_path: pl.Path = self.resource_dir / "../shaders/hybrid_renderer_fs.glsl"
        hybrid_renderer_fs_code: str = resolve_includes(hybrid_renderer_fs_path.read_text(encoding="utf-8"), self.resource_dir / "../shaders")
#       hybrid_renderer_fs_code: str = resolve_includes(hybrid_renderer_fs_path.read_text(encoding="utf-8"), self.resource_dir / "../shaders")
        self.program_renderer: mgl.Program = self.ctx.program(
#       self.program_renderer: mgl.Program = self.ctx.program(
              vertex_shader=hybrid_renderer_vs_code,
#             vertex_shader=hybrid_renderer_vs_code,
            fragment_shader=hybrid_renderer_fs_code,
#           fragment_shader=hybrid_renderer_fs_code,
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
        self.vbo_screen: mgl.Buffer = self.ctx.buffer(data=screen_data.tobytes())
#       self.vbo_screen: mgl.Buffer = self.ctx.buffer(data=screen_data.tobytes())
        self.vao_screen: mgl.VertexArray = self.ctx.vertex_array(
#       self.vao_screen: mgl.VertexArray = self.ctx.vertex_array(
            self.program_renderer,
#           self.program_renderer,
            [
#           [
                (self.vbo_screen, "2f 2f", "inScreenVertexPosition", "inScreenVertexUV"),
#               (self.vbo_screen, "2f 2f", "inScreenVertexPosition", "inScreenVertexUV"),
            ],
#           ],
        )
#       )

        # -----------------------------
#       # -----------------------------
        # 5. Scene Geometry
#       # 5. Scene Geometry
        # -----------------------------
#       # -----------------------------

        self.materials: list[Material] = [
#       self.materials: list[Material] = [
            {"albedo": (1.0, 1.0, 1.0), "roughness": 1.0, "metallic": 0.0, "transmission": 0.0, "ior": 1.5, "texture_index_albedo": -1, "texture_index_roughness": -1, "texture_index_metallic": -1, "texture_index_normal": -1, "emissive": 9.0, "texture_index_emissive": -1},
#           {"albedo": (1.0, 1.0, 1.0), "roughness": 1.0, "metallic": 0.0, "transmission": 0.0, "ior": 1.5, "texture_index_albedo": -1, "texture_index_roughness": -1, "texture_index_metallic": -1, "texture_index_normal": -1, "emissive": 9.0, "texture_index_emissive": -1},
            {"albedo": (0.5, 1.0, 0.0), "roughness": 1.0, "metallic": 0.0, "transmission": 0.0, "ior": 1.5, "texture_index_albedo": -1, "texture_index_roughness": -1, "texture_index_metallic": -1, "texture_index_normal": -1, "emissive": 0.0, "texture_index_emissive": -1},
#           {"albedo": (0.5, 1.0, 0.0), "roughness": 1.0, "metallic": 0.0, "transmission": 0.0, "ior": 1.5, "texture_index_albedo": -1, "texture_index_roughness": -1, "texture_index_metallic": -1, "texture_index_normal": -1, "emissive": 0.0, "texture_index_emissive": -1},
            {"albedo": (0.0, 0.5, 1.0), "roughness": 1.0, "metallic": 0.0, "transmission": 0.0, "ior": 1.5, "texture_index_albedo": -1, "texture_index_roughness": -1, "texture_index_metallic": -1, "texture_index_normal": -1, "emissive": 0.0, "texture_index_emissive": -1},
#           {"albedo": (0.0, 0.5, 1.0), "roughness": 1.0, "metallic": 0.0, "transmission": 0.0, "ior": 1.5, "texture_index_albedo": -1, "texture_index_roughness": -1, "texture_index_metallic": -1, "texture_index_normal": -1, "emissive": 0.0, "texture_index_emissive": -1},
            {"albedo": (0.5, 0.5, 0.5), "roughness": 1.0, "metallic": 0.0, "transmission": 0.0, "ior": 1.5, "texture_index_albedo": -1, "texture_index_roughness": -1, "texture_index_metallic": -1, "texture_index_normal": -1, "emissive": 0.0, "texture_index_emissive": -1},
#           {"albedo": (0.5, 0.5, 0.5), "roughness": 1.0, "metallic": 0.0, "transmission": 0.0, "ior": 1.5, "texture_index_albedo": -1, "texture_index_roughness": -1, "texture_index_metallic": -1, "texture_index_normal": -1, "emissive": 0.0, "texture_index_emissive": -1},
        ]
#       ]

        self.scene_builder: SceneBuilder = SceneBuilder(self.ctx, self.program_geometry, materials=self.materials)
#       self.scene_builder: SceneBuilder = SceneBuilder(self.ctx, self.program_geometry, materials=self.materials)

        self.scene_builder.add_cube(position=(-5.0, 1.0, 5.0), rotation=(0.0, 0.0, 0.0), scale=(1.0, 1.0, 1.0), material_index=0)
#       self.scene_builder.add_cube(position=(-5.0, 1.0, 5.0), rotation=(0.0, 0.0, 0.0), scale=(1.0, 1.0, 1.0), material_index=0)
        self.scene_builder.add_cube(position=(5.0, 1.0, -5.0), rotation=(0.0, 0.0, 0.0), scale=(1.0, 1.0, 1.0), material_index=0)
#       self.scene_builder.add_cube(position=(5.0, 1.0, -5.0), rotation=(0.0, 0.0, 0.0), scale=(1.0, 1.0, 1.0), material_index=0)
        self.scene_builder.add_cube(position=(5.0, 1.0, 5.0), rotation=(0.0, 0.0, 0.0), scale=(1.0, 1.0, 1.0), material_index=0)
#       self.scene_builder.add_cube(position=(5.0, 1.0, 5.0), rotation=(0.0, 0.0, 0.0), scale=(1.0, 1.0, 1.0), material_index=0)
        self.scene_builder.add_cube(position=(-5.0, 1.0, -5.0), rotation=(0.0, 0.0, 0.0), scale=(1.0, 1.0, 1.0), material_index=0)
#       self.scene_builder.add_cube(position=(-5.0, 1.0, -5.0), rotation=(0.0, 0.0, 0.0), scale=(1.0, 1.0, 1.0), material_index=0)

        self.scene_builder.add_plane(position=(0.0, -0.5, 0.0), rotation=(0.0, 0.0, 0.0), scale=(20.0, 1.0, 20.0), material_index=3)
#       self.scene_builder.add_plane(position=(0.0, -0.5, 0.0), rotation=(0.0, 0.0, 0.0), scale=(20.0, 1.0, 20.0), material_index=3)

        # Load Vase
#       # Load Vase
        vase_albedo_idx = self.load_texture(self.resource_dir / "../assets/ChinaVase.jpg", is_srgb=True)
#       vase_albedo_idx = self.load_texture(self.resource_dir / "../assets/ChinaVase.jpg", is_srgb=True)
        # vase_roughness_idx = self.load_texture(self.resource_dir / "../assets/vase_base_roughness.jpg")
#       # vase_roughness_idx = self.load_texture(self.resource_dir / "../assets/vase_base_roughness.jpg")
        # vase_metallic_idx = self.load_texture(self.resource_dir / "../assets/vase_base_metallic.jpg")
#       # vase_metallic_idx = self.load_texture(self.resource_dir / "../assets/vase_base_metallic.jpg")
        # vase_normal_idx = self.load_texture(self.resource_dir / "../assets/vase_base_normal.png")
#       # vase_normal_idx = self.load_texture(self.resource_dir / "../assets/vase_base_normal.png")

        self.materials.append({
#       self.materials.append({
            "albedo": (1.0, 0.5, 1.0),
#           "albedo": (1.0, 0.5, 1.0),
            "roughness": 1.0,
#           "roughness": 1.0,
            "metallic": 0.0,
#           "metallic": 0.0,
            "transmission": 0.0,
#           "transmission": 0.0,
            "ior": 1.5,
#           "ior": 1.5,
            "texture_index_albedo": vase_albedo_idx,
#           "texture_index_albedo": vase_albedo_idx,
            "texture_index_roughness": -1.0,
#           "texture_index_roughness": -1.0,
            "texture_index_metallic": -1.0,
#           "texture_index_metallic": -1.0,
            "texture_index_normal": -1.0,
#           "texture_index_normal": -1.0,
            "emissive": 0.0,
#           "emissive": 0.0,
            "texture_index_emissive": -1.0,
#           "texture_index_emissive": -1.0,
        })
#       })
        vase_material_index = len(self.materials) - 1
#       vase_material_index = len(self.materials) - 1

        vase_path = str(self.resource_dir / "../assets/ChinaVase.obj")
#       vase_path = str(self.resource_dir / "../assets/ChinaVase.obj")
        self.scene_builder.load_model(path=vase_path, position=(0.0, 0.5, 0.0), rotation=(np.pi / 4.0, 0.0, 0.0), scale=(0.1, 0.1, 0.1), material_indices=vase_material_index)
#       self.scene_builder.load_model(path=vase_path, position=(0.0, 0.5, 0.0), rotation=(np.pi / 4.0, 0.0, 0.0), scale=(0.1, 0.1, 0.1), material_indices=vase_material_index)

        # Load Pistol
#       # Load Pistol
        pistol_base_path = self.resource_dir / "../assets/pistol"
#       pistol_base_path = self.resource_dir / "../assets/pistol"
        pistol_material_indices = []
#       pistol_material_indices = []

        # Material 0: glock
#       # Material 0: glock
        idx_albedo_pistol = self.load_texture(pistol_base_path / "pistol_albedo.png", is_srgb=True)
#       idx_albedo_pistol = self.load_texture(pistol_base_path / "pistol_albedo.png", is_srgb=True)
        idx_normal_pistol = self.load_texture(pistol_base_path / "pistol_normal.png")
#       idx_normal_pistol = self.load_texture(pistol_base_path / "pistol_normal.png")
        idx_met_sm_pistol = self.load_texture(pistol_base_path / "pistol_metallic_smoothness.png")
#       idx_met_sm_pistol = self.load_texture(pistol_base_path / "pistol_metallic_smoothness.png")

        # Material 0: lambert4
#       # Material 0: lambert4
        self.materials.append({
#       self.materials.append({
            "albedo": (1.0, 1.0, 1.0),
#           "albedo": (1.0, 1.0, 1.0),
            "roughness": 0.5,
#           "roughness": 0.5,
            "metallic": 0.0,
#           "metallic": 0.0,
            "transmission": 0.0,
#           "transmission": 0.0,
            "ior": 1.5,
#           "ior": 1.5,
            "texture_index_albedo": idx_albedo_pistol,
#           "texture_index_albedo": idx_albedo_pistol,
            "texture_index_roughness": -1.0,
#           "texture_index_roughness": -1.0,
            "texture_index_metallic": -1.0,
#           "texture_index_metallic": -1.0,
            "texture_index_normal": idx_normal_pistol,
#           "texture_index_normal": idx_normal_pistol,
            "emissive": 0.0,
#           "emissive": 0.0,
            "texture_index_emissive": -1.0,
#           "texture_index_emissive": -1.0,
        })
#       })
        pistol_material_indices.append(len(self.materials) - 1)
#       pistol_material_indices.append(len(self.materials) - 1)

        # Material 1: glock
#       # Material 1: glock
        self.materials.append({
#       self.materials.append({
            "albedo": (1.0, 1.0, 1.0),
#           "albedo": (1.0, 1.0, 1.0),
            "roughness": 1.0,
#           "roughness": 1.0,
            "metallic": 1.0,
#           "metallic": 1.0,
            "transmission": 0.0,
#           "transmission": 0.0,
            "ior": 1.5,
#           "ior": 1.5,
            "texture_index_albedo": idx_albedo_pistol,
#           "texture_index_albedo": idx_albedo_pistol,
            "texture_index_roughness": -1.0,
#           "texture_index_roughness": -1.0,
            "texture_index_metallic": idx_met_sm_pistol,
#           "texture_index_metallic": idx_met_sm_pistol,
            "texture_index_normal": idx_normal_pistol,
#           "texture_index_normal": idx_normal_pistol,
            "emissive": 0.0,
#           "emissive": 0.0,
            "texture_index_emissive": -1.0,
#           "texture_index_emissive": -1.0,
        })
#       })
        pistol_material_indices.append(len(self.materials) - 1)
#       pistol_material_indices.append(len(self.materials) - 1)

        pistol_path = str(self.resource_dir / "../assets/pistol/pistol.fbx")
#       pistol_path = str(self.resource_dir / "../assets/pistol/pistol.fbx")
        self.scene_builder.load_model(path=pistol_path, position=(5.0, 10.0, 0.0), rotation=(0.0, 0.0, 0.0), scale=(0.2, 0.2, 0.2), material_indices=pistol_material_indices)
#       self.scene_builder.load_model(path=pistol_path, position=(5.0, 10.0, 0.0), rotation=(0.0, 0.0, 0.0), scale=(0.2, 0.2, 0.2), material_indices=pistol_material_indices)

        # Load Rifle (SLR AR-15)
#       # Load Rifle (SLR AR-15)
        rifle_base_path = self.resource_dir / "../assets/slr-ar-15"
#       rifle_base_path = self.resource_dir / "../assets/slr-ar-15"
        rifle_material_indices = []
#       rifle_material_indices = []

        # Material 0: Upper Receiver
#       # Material 0: Upper Receiver
        idx_albedo_0 = self.load_texture(rifle_base_path / "low_UpperReciever_BaseColor.jpg", is_srgb=True)
#       idx_albedo_0 = self.load_texture(rifle_base_path / "low_UpperReciever_BaseColor.jpg", is_srgb=True)
        idx_normal_0 = self.load_texture(rifle_base_path / "low_UpperReciever_Normal.jpg")
#       idx_normal_0 = self.load_texture(rifle_base_path / "low_UpperReciever_Normal.jpg")
        idx_roughness_0 = self.load_texture(rifle_base_path / "UR_R.jpg")
#       idx_roughness_0 = self.load_texture(rifle_base_path / "UR_R.jpg")

        self.materials.append({
#       self.materials.append({
            "albedo": (1.0, 1.0, 1.0),
#           "albedo": (1.0, 1.0, 1.0),
            "roughness": 1.0,
#           "roughness": 1.0,
            "metallic": 1.0, # Assume metallic
#           "metallic": 1.0, # Assume metallic
            "transmission": 0.0,
#           "transmission": 0.0,
            "ior": 1.5,
#           "ior": 1.5,
            "texture_index_albedo": idx_albedo_0,
#           "texture_index_albedo": idx_albedo_0,
            "texture_index_roughness": idx_roughness_0,
#           "texture_index_roughness": idx_roughness_0,
            "texture_index_metallic": -1.0,
#           "texture_index_metallic": -1.0,
            "texture_index_normal": idx_normal_0,
#           "texture_index_normal": idx_normal_0,
            "emissive": 0.0,
#           "emissive": 0.0,
            "texture_index_emissive": -1.0,
#           "texture_index_emissive": -1.0,
        })
#       })
        rifle_material_indices.append(len(self.materials) - 1)
#       rifle_material_indices.append(len(self.materials) - 1)

        # Material 1: Lower Receiver
#       # Material 1: Lower Receiver
        idx_albedo_1 = self.load_texture(rifle_base_path / "low_LowerReciever_BaseColor.jpg", is_srgb=True)
#       idx_albedo_1 = self.load_texture(rifle_base_path / "low_LowerReciever_BaseColor.jpg", is_srgb=True)
        idx_normal_1 = self.load_texture(rifle_base_path / "low_LowerReciever_Normal.jpg")
#       idx_normal_1 = self.load_texture(rifle_base_path / "low_LowerReciever_Normal.jpg")
        idx_roughness_1 = self.load_texture(rifle_base_path / "LR_R.png")
#       idx_roughness_1 = self.load_texture(rifle_base_path / "LR_R.png")
        idx_metallic_1 = self.load_texture(rifle_base_path / "LR_M.png")
#       idx_metallic_1 = self.load_texture(rifle_base_path / "LR_M.png")

        self.materials.append({
#       self.materials.append({
            "albedo": (1.0, 1.0, 1.0),
#           "albedo": (1.0, 1.0, 1.0),
            "roughness": 1.0,
#           "roughness": 1.0,
            "metallic": 1.0,
#           "metallic": 1.0,
            "transmission": 0.0,
#           "transmission": 0.0,
            "ior": 1.5,
#           "ior": 1.5,
            "texture_index_albedo": idx_albedo_1,
#           "texture_index_albedo": idx_albedo_1,
            "texture_index_roughness": idx_roughness_1,
#           "texture_index_roughness": idx_roughness_1,
            "texture_index_metallic": idx_metallic_1,
#           "texture_index_metallic": idx_metallic_1,
            "texture_index_normal": idx_normal_1,
#           "texture_index_normal": idx_normal_1,
            "emissive": 0.0,
#           "emissive": 0.0,
            "texture_index_emissive": -1.0,
#           "texture_index_emissive": -1.0,
        })
#       })
        rifle_material_indices.append(len(self.materials) - 1)
#       rifle_material_indices.append(len(self.materials) - 1)

        # Material 2: Magazine
#       # Material 2: Magazine
        idx_albedo_2 = self.load_texture(rifle_base_path / "low_Magazine_BaseColor.jpg", is_srgb=True)
#       idx_albedo_2 = self.load_texture(rifle_base_path / "low_Magazine_BaseColor.jpg", is_srgb=True)
        idx_normal_2 = self.load_texture(rifle_base_path / "low_Magazine_Normal.jpg")
#       idx_normal_2 = self.load_texture(rifle_base_path / "low_Magazine_Normal.jpg")
        idx_roughness_2 = self.load_texture(rifle_base_path / "M_R.png")
#       idx_roughness_2 = self.load_texture(rifle_base_path / "M_R.png")
        idx_metallic_2 = self.load_texture(rifle_base_path / "M_M.png")
#       idx_metallic_2 = self.load_texture(rifle_base_path / "M_M.png")

        self.materials.append({
#       self.materials.append({
            "albedo": (1.0, 1.0, 1.0),
#           "albedo": (1.0, 1.0, 1.0),
            "roughness": 1.0,
#           "roughness": 1.0,
            "metallic": 1.0,
#           "metallic": 1.0,
            "transmission": 0.0,
#           "transmission": 0.0,
            "ior": 1.5,
#           "ior": 1.5,
            "texture_index_albedo": idx_albedo_2,
#           "texture_index_albedo": idx_albedo_2,
            "texture_index_roughness": idx_roughness_2,
#           "texture_index_roughness": idx_roughness_2,
            "texture_index_metallic": idx_metallic_2,
#           "texture_index_metallic": idx_metallic_2,
            "texture_index_normal": idx_normal_2,
#           "texture_index_normal": idx_normal_2,
            "emissive": 0.0,
#           "emissive": 0.0,
            "texture_index_emissive": -1.0,
#           "texture_index_emissive": -1.0,
        })
#       })
        rifle_material_indices.append(len(self.materials) - 1)
#       rifle_material_indices.append(len(self.materials) - 1)

        # Material 3: Trijicon MRO
#       # Material 3: Trijicon MRO
        idx_albedo_3 = self.load_texture(rifle_base_path / "low_TrijiconMRO_BaseColor.jpg", is_srgb=True)
#       idx_albedo_3 = self.load_texture(rifle_base_path / "low_TrijiconMRO_BaseColor.jpg", is_srgb=True)
        idx_normal_3 = self.load_texture(rifle_base_path / "low_TrijiconMRO_Normal.jpg")
#       idx_normal_3 = self.load_texture(rifle_base_path / "low_TrijiconMRO_Normal.jpg")
        idx_roughness_3 = self.load_texture(rifle_base_path / "T_R.png")
#       idx_roughness_3 = self.load_texture(rifle_base_path / "T_R.png")
        idx_metallic_3 = self.load_texture(rifle_base_path / "T_M.png")
#       idx_metallic_3 = self.load_texture(rifle_base_path / "T_M.png")

        self.materials.append({
#       self.materials.append({
            "albedo": (1.0, 1.0, 1.0),
#           "albedo": (1.0, 1.0, 1.0),
            "roughness": 1.0,
#           "roughness": 1.0,
            "metallic": 1.0,
#           "metallic": 1.0,
            "transmission": 0.0,
#           "transmission": 0.0,
            "ior": 1.5,
#           "ior": 1.5,
            "texture_index_albedo": idx_albedo_3,
#           "texture_index_albedo": idx_albedo_3,
            "texture_index_roughness": idx_roughness_3,
#           "texture_index_roughness": idx_roughness_3,
            "texture_index_metallic": idx_metallic_3,
#           "texture_index_metallic": idx_metallic_3,
            "texture_index_normal": idx_normal_3,
#           "texture_index_normal": idx_normal_3,
            "emissive": 0.0,
#           "emissive": 0.0,
            "texture_index_emissive": -1.0,
#           "texture_index_emissive": -1.0,
        })
#       })
        rifle_material_indices.append(len(self.materials) - 1)
#       rifle_material_indices.append(len(self.materials) - 1)

        rifle_path = str(self.resource_dir / "../assets/slr-ar-15/AR-15.fbx")
#       rifle_path = str(self.resource_dir / "../assets/slr-ar-15/AR-15.fbx")
        self.scene_builder.load_model(path=rifle_path, position=(-5.0, 5.0, 0.0), rotation=(0.0, 0.0, 0.0), scale=(0.1, 0.1, 0.1), material_indices=rifle_material_indices)
#       self.scene_builder.load_model(path=rifle_path, position=(-5.0, 5.0, 0.0), rotation=(0.0, 0.0, 0.0), scale=(0.1, 0.1, 0.1), material_indices=rifle_material_indices)

        # Load MP9
#       # Load MP9
        mp9_base_path = self.resource_dir / "../assets/mp9"
#       mp9_base_path = self.resource_dir / "../assets/mp9"
        mp9_material_indices = []
#       mp9_material_indices = []

        # Material 0: M_MP9
#       # Material 0: M_MP9
        idx_albedo_mp9 = self.load_texture(mp9_base_path / "M_MP9_Base_color.png", is_srgb=True)
#       idx_albedo_mp9 = self.load_texture(mp9_base_path / "M_MP9_Base_color.png", is_srgb=True)
        idx_normal_mp9 = self.load_texture(mp9_base_path / "M_MP9_Normal_OpenGL.png")
#       idx_normal_mp9 = self.load_texture(mp9_base_path / "M_MP9_Normal_OpenGL.png")
        idx_roughness_mp9 = self.load_texture(mp9_base_path / "M_MP9_Roughness.png")
#       idx_roughness_mp9 = self.load_texture(mp9_base_path / "M_MP9_Roughness.png")
        idx_metallic_mp9 = self.load_texture(mp9_base_path / "M_MP9_Metallic.png")
#       idx_metallic_mp9 = self.load_texture(mp9_base_path / "M_MP9_Metallic.png")
        idx_emissive_mp9 = self.load_texture(mp9_base_path / "M_MP9_Emissive.png")
#       idx_emissive_mp9 = self.load_texture(mp9_base_path / "M_MP9_Emissive.png")

        self.materials.append({
#       self.materials.append({
            "albedo": (1.0, 1.0, 1.0),
#           "albedo": (1.0, 1.0, 1.0),
            "roughness": 1.0,
#           "roughness": 1.0,
            "metallic": 1.0,
#           "metallic": 1.0,
            "transmission": 0.0,
#           "transmission": 0.0,
            "ior": 1.5,
#           "ior": 1.5,
            "texture_index_albedo": idx_albedo_mp9,
#           "texture_index_albedo": idx_albedo_mp9,
            "texture_index_roughness": idx_roughness_mp9,
#           "texture_index_roughness": idx_roughness_mp9,
            "texture_index_metallic": idx_metallic_mp9,
#           "texture_index_metallic": idx_metallic_mp9,
            "texture_index_normal": idx_normal_mp9,
#           "texture_index_normal": idx_normal_mp9,
            "emissive": 1.0,
#           "emissive": 1.0,
            "texture_index_emissive": idx_emissive_mp9,
#           "texture_index_emissive": idx_emissive_mp9,
        })
#       })
        mp9_material_indices.append(len(self.materials) - 1)
#       mp9_material_indices.append(len(self.materials) - 1)

        mp9_path = str(self.resource_dir / "../assets/mp9/MP9_Sketchfab.fbx")
#       mp9_path = str(self.resource_dir / "../assets/mp9/MP9_Sketchfab.fbx")
        self.scene_builder.load_model(path=mp9_path, position=(0.0, 5.0, 5.0), rotation=(np.pi * 0.25, 0.0, np.pi * -0.5), scale=(0.1, 0.1, 0.1), material_indices=mp9_material_indices)
#       self.scene_builder.load_model(path=mp9_path, position=(0.0, 5.0, 5.0), rotation=(np.pi * 0.25, 0.0, np.pi * -0.5), scale=(0.1, 0.1, 0.1), material_indices=mp9_material_indices)




        bvh_data, triangles_data, materials_data, uvs_data, normals_data, tangents_data = self.scene_builder.build()
#       bvh_data, triangles_data, materials_data, uvs_data, normals_data, tangents_data = self.scene_builder.build()
        self.scene_batches: list[SceneBatch] = self.scene_builder.scene_batches
#       self.scene_batches: list[SceneBatch] = self.scene_builder.scene_batches

        # Upload to SSBOs
#       # Upload to SSBOs
        # 1. Nodes (already bytes)
#       # 1. Nodes (already bytes)
        self.ssbo_bvh_nodes: mgl.Buffer = self.ctx.buffer(data=bvh_data)
#       self.ssbo_bvh_nodes: mgl.Buffer = self.ctx.buffer(data=bvh_data)

        # 2. Triangles (flattened world space)
#       # 2. Triangles (flattened world space)
        self.ssbo_triangles: mgl.Buffer = self.ctx.buffer(data=triangles_data)
#       self.ssbo_triangles: mgl.Buffer = self.ctx.buffer(data=triangles_data)

        # 3. Materials (Albedos)
#       # 3. Materials (Albedos)
        self.ssbo_materials: mgl.Buffer = self.ctx.buffer(data=materials_data)
#       self.ssbo_materials: mgl.Buffer = self.ctx.buffer(data=materials_data)

        # 4. UVs (vec2s per vertex per triangle: 6 floats per triangle)
#       # 4. UVs (vec2s per vertex per triangle: 6 floats per triangle)
        self.ssbo_uvs: mgl.Buffer = self.ctx.buffer(data=uvs_data)
#       self.ssbo_uvs: mgl.Buffer = self.ctx.buffer(data=uvs_data)

        # 5. Normals (vec3s per vertex per triangle: 9 floats per triangle)
#       # 5. Normals (vec3s per vertex per triangle: 9 floats per triangle)
        self.ssbo_normals: mgl.Buffer = self.ctx.buffer(data=normals_data)
#       self.ssbo_normals: mgl.Buffer = self.ctx.buffer(data=normals_data)

        # 6. Tangents (vec3s per vertex per triangle: 9 floats per triangle)
#       # 6. Tangents (vec3s per vertex per triangle: 9 floats per triangle)
        self.ssbo_tangents: mgl.Buffer = self.ctx.buffer(data=tangents_data)
#       self.ssbo_tangents: mgl.Buffer = self.ctx.buffer(data=tangents_data)

        self.ctx.enable(flags=mgl.DEPTH_TEST | mgl.CULL_FACE)
#       self.ctx.enable(flags=mgl.DEPTH_TEST | mgl.CULL_FACE)

        # -----------------------------
#       # -----------------------------
        # 6. Post Processing System
#       # 6. Post Processing System
        # -----------------------------
#       # -----------------------------
        post_chromatic_aberration_cs_path: pl.Path = self.resource_dir / "../shaders/post_chromatic_aberration_cs.glsl"
#       post_chromatic_aberration_cs_path: pl.Path = self.resource_dir / "../shaders/post_chromatic_aberration_cs.glsl"
        post_chromatic_aberration_cs_code: str = resolve_includes(post_chromatic_aberration_cs_path.read_text(encoding="utf-8"), self.resource_dir / "../shaders")
#       post_chromatic_aberration_cs_code: str = resolve_includes(post_chromatic_aberration_cs_path.read_text(encoding="utf-8"), self.resource_dir / "../shaders")
        self.program_post_chromatic_aberration: mgl.ComputeShader = self.ctx.compute_shader(source=post_chromatic_aberration_cs_code)
#       self.program_post_chromatic_aberration: mgl.ComputeShader = self.ctx.compute_shader(source=post_chromatic_aberration_cs_code)

        post_vignette_cs_path: pl.Path = self.resource_dir / "../shaders/post_vignette_cs.glsl"
#       post_vignette_cs_path: pl.Path = self.resource_dir / "../shaders/post_vignette_cs.glsl"
        post_vignette_cs_code: str = resolve_includes(post_vignette_cs_path.read_text(encoding="utf-8"), self.resource_dir / "../shaders")
#       post_vignette_cs_code: str = resolve_includes(post_vignette_cs_path.read_text(encoding="utf-8"), self.resource_dir / "../shaders")
        self.program_post_vignette: mgl.ComputeShader = self.ctx.compute_shader(source=post_vignette_cs_code)
#       self.program_post_vignette: mgl.ComputeShader = self.ctx.compute_shader(source=post_vignette_cs_code)

        # Pipeline of active post-processing shaders
#       # Pipeline of active post-processing shaders
        self.post_processing_pipeline: list[mgl.ComputeShader] = [
#       self.post_processing_pipeline: list[mgl.ComputeShader] = [
            self.program_post_chromatic_aberration,
#           self.program_post_chromatic_aberration,
            self.program_post_vignette,
#           self.program_post_vignette,
        ]
#       ]

        # Camera
#       # Camera
        self.camera: Camera = Camera(
#       self.camera: Camera = Camera(
            position=(-28.284, 0.0, -28.284),
#           position=(-28.284, 0.0, -28.284),
            look_at=(0.0, 0.0, 0.0),
#           look_at=(0.0, 0.0, 0.0),
            up=(0.0, 1.0, 0.0),
#           up=(0.0, 1.0, 0.0),
            aspect_ratio=self.aspect_ratio
#           aspect_ratio=self.aspect_ratio
        )
#       )

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
        pass
#       pass

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
        path_str = str(path.resolve())
#       path_str = str(path.resolve())
        if path_str in self.texture_cache:
#       if path_str in self.texture_cache:
            return self.texture_cache[path_str]
#           return self.texture_cache[path_str]

        if self.next_texture_layer >= self.texture_array_layers:
#       if self.next_texture_layer >= self.texture_array_layers:
            print(f"Warning: Texture array full. Cannot load {path}")
#           print(f"Warning: Texture array full. Cannot load {path}")
            return -1.0
#           return -1.0

        layer_index = self.next_texture_layer
#       layer_index = self.next_texture_layer
        self.load_texture_to_array(path, layer_index, is_srgb=is_srgb)
#       self.load_texture_to_array(path, layer_index, is_srgb=is_srgb)
        self.texture_cache[path_str] = float(layer_index)
#       self.texture_cache[path_str] = float(layer_index)
        self.next_texture_layer += 1
#       self.next_texture_layer += 1
        return float(layer_index)
#       return float(layer_index)

    def load_texture_to_array(self, path: pl.Path, layer_index: int, is_srgb: bool = False) -> None:
#   def load_texture_to_array(self, path: pl.Path, layer_index: int, is_srgb: bool = False) -> None:
        # Loads a texture from disk and uploads it to a specific layer in the 2D Texture Array.
#       # Loads a texture from disk and uploads it to a specific layer in the 2D Texture Array.
        # Texture Arrays allow the shader to access many different textures using a single sampler + index.
#       # Texture Arrays allow the shader to access many different textures using a single sampler + index.
        if not path.exists():
#       if not path.exists():
            print(f"Warning: Texture not found: {path}")
#           print(f"Warning: Texture not found: {path}")
            return
#           return

        loaded_data = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
#       loaded_data = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if loaded_data is None:
#       if loaded_data is None:
            print(f"Warning: Failed to load texture: {path}")
#           print(f"Warning: Failed to load texture: {path}")
            return
#           return

        # Ensure RGBA
        # Ensure RGBA
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

        # Resize to 2048x2048
        # Resize to 2048x2048
        if loaded_data.shape[0] != self.texture_array_size or loaded_data.shape[1] != self.texture_array_size:
#       if loaded_data.shape[0] != self.texture_array_size or loaded_data.shape[1] != self.texture_array_size:
            loaded_data = cv2.resize(loaded_data, (self.texture_array_size, self.texture_array_size), interpolation=cv2.INTER_LINEAR)
#           loaded_data = cv2.resize(loaded_data, (self.texture_array_size, self.texture_array_size), interpolation=cv2.INTER_LINEAR)

        # Flip for OpenGL!
        # Flip for OpenGL!
        loaded_data = np.flipud(loaded_data)
#       loaded_data = np.flipud(loaded_data)

        # Write to texture array
        # Write to texture array
        data_float: npt.NDArray[np.float32]
#       data_float: npt.NDArray[np.float32]

        if loaded_data.dtype == np.uint8:
#       if loaded_data.dtype == np.uint8:
            data_float = (loaded_data.astype(dtype=np.float32) / 255.0)
#           data_float = (loaded_data.astype(dtype=np.float32) / 255.0)
        elif loaded_data.dtype == np.uint16:
#       elif loaded_data.dtype == np.uint16:
            data_float = (loaded_data.astype(dtype=np.float32) / 65535.0)
#           data_float = (loaded_data.astype(dtype=np.float32) / 65535.0)
        else:
#       else:
            data_float = loaded_data.astype(dtype=np.float32)
#           data_float = loaded_data.astype(dtype=np.float32)

        # Apply sRGB -> Linear conversion if requested
#       # Apply sRGB -> Linear conversion if requested
        if is_srgb:
#       if is_srgb:
            # Approximate Gamma 2.2
#           # Approximate Gamma 2.2
            data_float = np.power(data_float, 2.2)
#           data_float = np.power(data_float, 2.2)

        data_bytes = data_float.tobytes()
#       data_bytes = data_float.tobytes()

        # Viewport: (x, y, layer, width, height, depth) -> for 2D array, it's (x, y, layer, width, height, 1)
        # Viewport: (x, y, layer, width, height, depth) -> for 2D array, it's (x, y, layer, width, height, 1)
        # For ModernGL TextureArray.write: viewport=(x, y, z, width, height, depth)
        # For ModernGL TextureArray.write: viewport=(x, y, z, width, height, depth)

        # Check ModernGL docs logic. Often it's just (x, y, layer, width, height, 1).
        # Check ModernGL docs logic. Often it's just (x, y, layer, width, height, 1).
        self.texture_array.write(data_bytes, viewport=(0, 0, layer_index, self.texture_array_size, self.texture_array_size, 1))
#       self.texture_array.write(data_bytes, viewport=(0, 0, layer_index, self.texture_array_size, self.texture_array_size, 1))

        # Generate mipmaps if needed (requires creating whole array though! or per layer!)
        # Generate mipmaps if needed (requires creating whole array though! or per layer!)
        self.texture_array.build_mipmaps()
#       self.texture_array.build_mipmaps()


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
        color_data: bytes = self.texture_accum.read()
#       color_data: bytes = self.texture_accum.read()
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
        albedo_data: bytes = self.texture_geometry_albedo.read()
#       albedo_data: bytes = self.texture_geometry_albedo.read()
        albedo_arr: npt.NDArray[np.float32] = np.frombuffer(albedo_data, dtype=np.float32).reshape((h, w, 4))[:, :, :3]
#       albedo_arr: npt.NDArray[np.float32] = np.frombuffer(albedo_data, dtype=np.float32).reshape((h, w, 4))[:, :, :3]
        albedo_arr = np.ascontiguousarray(albedo_arr)
#       albedo_arr = np.ascontiguousarray(albedo_arr)

        # Normal
#       # Normal
        normal_data: bytes = self.texture_geometry_global_normal.read()
#       normal_data: bytes = self.texture_geometry_global_normal.read()
        normal_arr: npt.NDArray[np.float32] = np.frombuffer(normal_data, dtype=np.float32).reshape((h, w, 4))[:, :, :3]
#       normal_arr: npt.NDArray[np.float32] = np.frombuffer(normal_data, dtype=np.float32).reshape((h, w, 4))[:, :, :3]
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

        # Apply Khronos PBR Neutral Tonemap
#       # Apply Khronos PBR Neutral Tonemap
        start_compression = 0.8 - 0.04
#       start_compression = 0.8 - 0.04
        desaturation = 0.15
#       desaturation = 0.15

        x = np.min(output_arr, axis=-1, keepdims=True)
#       x = np.min(output_arr, axis=-1, keepdims=True)
        offset = np.where(x < 0.08, x - 6.25 * x * x, 0.04)
#       offset = np.where(x < 0.08, x - 6.25 * x * x, 0.04)
        output_arr = output_arr - offset
#       output_arr = output_arr - offset

        peak = np.max(output_arr, axis=-1, keepdims=True)
#       peak = np.max(output_arr, axis=-1, keepdims=True)

        d = 1.0 - start_compression
#       d = 1.0 - start_compression
        new_peak = 1.0 - d * d / (np.maximum(peak, 1e-6) + d - start_compression)
#       new_peak = 1.0 - d * d / (np.maximum(peak, 1e-6) + d - start_compression)

        # Mask for peaks that need compression
#       # Mask for peaks that need compression
        mask = peak >= start_compression
#       mask = peak >= start_compression

        output_arr = np.where(mask, output_arr * (new_peak / np.maximum(peak, 1e-6)), output_arr)
#       output_arr = np.where(mask, output_arr * (new_peak / np.maximum(peak, 1e-6)), output_arr)

        current_peak = np.where(mask, new_peak, peak)
#       current_peak = np.where(mask, new_peak, peak)
        g = 1.0 - 1.0 / (desaturation * (peak - current_peak) + 1.0)
#       g = 1.0 - 1.0 / (desaturation * (peak - current_peak) + 1.0)
        output_arr = output_arr * (1.0 - g) + current_peak * g
#       output_arr = output_arr * (1.0 - g) + current_peak * g

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
        display_img = cv2.cvtColor(output_arr, cv2.COLOR_RGB2BGR)
#       display_img = cv2.cvtColor(output_arr, cv2.COLOR_RGB2BGR)
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
        self.frame_count += 1
#       self.frame_count += 1

        # Poll keys (Backup for key_event)
#       # Poll keys (Backup for key_event)
        keys: BaseKeys = self.wnd.keys
#       keys: BaseKeys = self.wnd.keys
        try:
#       try:
            self.key_state["W"] = self.wnd.is_key_pressed(keys.W)
#           self.key_state["W"] = self.wnd.is_key_pressed(keys.W)
            self.key_state["S"] = self.wnd.is_key_pressed(keys.S)
#           self.key_state["S"] = self.wnd.is_key_pressed(keys.S)
            self.key_state["A"] = self.wnd.is_key_pressed(keys.A)
#           self.key_state["A"] = self.wnd.is_key_pressed(keys.A)
            self.key_state["D"] = self.wnd.is_key_pressed(keys.D)
#           self.key_state["D"] = self.wnd.is_key_pressed(keys.D)
            self.key_state["Q"] = self.wnd.is_key_pressed(keys.Q)
#           self.key_state["Q"] = self.wnd.is_key_pressed(keys.Q)
            self.key_state["E"] = self.wnd.is_key_pressed(keys.E)
#           self.key_state["E"] = self.wnd.is_key_pressed(keys.E)
            self.key_state["UP"] = self.wnd.is_key_pressed(keys.UP)
#           self.key_state["UP"] = self.wnd.is_key_pressed(keys.UP)
            self.key_state["DOWN"] = self.wnd.is_key_pressed(keys.DOWN)
#           self.key_state["DOWN"] = self.wnd.is_key_pressed(keys.DOWN)
            self.key_state["LEFT"] = self.wnd.is_key_pressed(keys.LEFT)
#           self.key_state["LEFT"] = self.wnd.is_key_pressed(keys.LEFT)
            self.key_state["RIGHT"] = self.wnd.is_key_pressed(keys.RIGHT)
#           self.key_state["RIGHT"] = self.wnd.is_key_pressed(keys.RIGHT)
        except AttributeError:
#       except AttributeError:
            pass # method might not exist on all backends
#           pass # method might not exist on all backends

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
        jitter_x: float = (self.get_halton_jitter(index=self.frame_count, base=2) - 0.5)
#       jitter_x: float = (self.get_halton_jitter(index=self.frame_count, base=2) - 0.5)
        jitter_y: float = (self.get_halton_jitter(index=self.frame_count, base=3) - 0.5)
#       jitter_y: float = (self.get_halton_jitter(index=self.frame_count, base=3) - 0.5)

        transform_view: rr.Matrix44 = self.camera.get_view_matrix()
#       transform_view: rr.Matrix44 = self.camera.get_view_matrix()
        transform_projection: rr.Matrix44 = self.camera.get_projection_matrix(jitter=(jitter_x, jitter_y), window_size=self.window_size)
#       transform_projection: rr.Matrix44 = self.camera.get_projection_matrix(jitter=(jitter_x, jitter_y), window_size=self.window_size)

        # 1. Rasterize to G-Buffer
#       # 1. Rasterize to G-Buffer
        self.gbuffer.use()
#       self.gbuffer.use()
        self.gbuffer.clear(0.0, 0.0, 0.0, 0.0) # Zero out, .w = 0 means background
#       self.gbuffer.clear(0.0, 0.0, 0.0, 0.0) # Zero out, .w = 0 means background

        typing.cast(mgl.Uniform, self.program_geometry["uTransformView"]).write(transform_view.astype(dtype=np.float32).tobytes())
#       typing.cast(mgl.Uniform, self.program_geometry["uTransformView"]).write(transform_view.astype(dtype=np.float32).tobytes())
        typing.cast(mgl.Uniform, self.program_geometry["uTransformProjection"]).write(transform_projection.astype(dtype=np.float32).tobytes())
#       typing.cast(mgl.Uniform, self.program_geometry["uTransformProjection"]).write(transform_projection.astype(dtype=np.float32).tobytes())

        current_triangle_offset: int = 0
#       current_triangle_offset: int = 0
        for scene_batch in self.scene_batches:
#       for scene_batch in self.scene_batches:
            if "uBaseTriangleIndexOffset" in self.program_geometry:
#           if "uBaseTriangleIndexOffset" in self.program_geometry:
                self.program_geometry["uBaseTriangleIndexOffset"] = current_triangle_offset
#               self.program_geometry["uBaseTriangleIndexOffset"] = current_triangle_offset
            if "uTriangleCountPerInstance" in self.program_geometry:
#           if "uTriangleCountPerInstance" in self.program_geometry:
                self.program_geometry["uTriangleCountPerInstance"] = scene_batch.triangle_count_per_instance
#               self.program_geometry["uTriangleCountPerInstance"] = scene_batch.triangle_count_per_instance
            scene_batch.vao.render(instances=scene_batch.number_of_instances)
#           scene_batch.vao.render(instances=scene_batch.number_of_instances)
            current_triangle_offset += scene_batch.number_of_instances * scene_batch.triangle_count_per_instance
#           current_triangle_offset += scene_batch.number_of_instances * scene_batch.triangle_count_per_instance

        # 2. Compute Shade
#       # 2. Compute Shade
        self.texture_output.bind_to_image(0, read=False, write=True)
#       self.texture_output.bind_to_image(0, read=False, write=True)
        self.texture_geometry_global_position.bind_to_image(1, read=True, write=False)
#       self.texture_geometry_global_position.bind_to_image(1, read=True, write=False)
        self.texture_geometry_global_normal.bind_to_image(2, read=True, write=False)
#       self.texture_geometry_global_normal.bind_to_image(2, read=True, write=False)
        self.texture_geometry_albedo.bind_to_image(3, read=True, write=False)
#       self.texture_geometry_albedo.bind_to_image(3, read=True, write=False)
        self.texture_geometry_global_tangent.bind_to_image(4, read=True, write=False)
#       self.texture_geometry_global_tangent.bind_to_image(4, read=True, write=False)

        # Bind BVH buffers
#       # Bind BVH buffers
        self.ssbo_bvh_nodes.bind_to_storage_buffer(binding=6)
#       self.ssbo_bvh_nodes.bind_to_storage_buffer(binding=6)
        self.ssbo_triangles.bind_to_storage_buffer(binding=7)
#       self.ssbo_triangles.bind_to_storage_buffer(binding=7)
        self.ssbo_materials.bind_to_storage_buffer(binding=8)
#       self.ssbo_materials.bind_to_storage_buffer(binding=8)
        self.ssbo_uvs.bind_to_storage_buffer(binding=9)
#       self.ssbo_uvs.bind_to_storage_buffer(binding=9)
        self.ssbo_normals.bind_to_storage_buffer(binding=10)
#       self.ssbo_normals.bind_to_storage_buffer(binding=10)
        self.ssbo_tangents.bind_to_storage_buffer(binding=11)
#       self.ssbo_tangents.bind_to_storage_buffer(binding=11)

        self.texture_accum.bind_to_image(5, read=True, write=True)
#       self.texture_accum.bind_to_image(5, read=True, write=True)

        # Update Uniforms
#       # Update Uniforms
        if "uTime" in self.program_shading:
#       if "uTime" in self.program_shading:
            self.program_shading["uTime"] = time
#           self.program_shading["uTime"] = time

        # Point Lights Setup
#       # Point Lights Setup
        radius: float = 10.0
#       radius: float = 10.0
        x: float = np.cos(time) * radius
#       x: float = np.cos(time) * radius
        z: float = np.sin(time) * radius
#       z: float = np.sin(time) * radius

        point_lights: list[PointLight] = [
#       point_lights: list[PointLight] = [
            {"position": (x, 10.0, z), "color": (100.0, 100.0, 100.0), "radius": 0.5},
#           {"position": (x, 10.0, z), "color": (100.0, 100.0, 100.0), "radius": 0.5},
            {"position": (-x, 5.0, -z), "color": (100.0, 100.0, 100.0), "radius": 0.5},
#           {"position": (-x, 5.0, -z), "color": (100.0, 100.0, 100.0), "radius": 0.5},
            {"position": (0.0, 5.0, 0.0), "color": (100.0, 100.0, 100.0), "radius": 1.0},
#           {"position": (0.0, 5.0, 0.0), "color": (100.0, 100.0, 100.0), "radius": 1.0},
        ]
#       ]

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

        # HDRI Texture
#       # HDRI Texture
        self.texture_hdri.use(location=8)
#       self.texture_hdri.use(location=8)
        if "uHdriTexture" in self.program_shading:
#       if "uHdriTexture" in self.program_shading:
            self.program_shading["uHdriTexture"] = 8
#           self.program_shading["uHdriTexture"] = 8
        # Scene Texture Array
#       # Scene Texture Array
        self.texture_array.use(location=9)
#       self.texture_array.use(location=9)
        if "uSceneTextureArray" in self.program_shading:
#       if "uSceneTextureArray" in self.program_shading:
            self.program_shading["uSceneTextureArray"] = 9
#           self.program_shading["uSceneTextureArray"] = 9

        if "uUseHdri" in self.program_shading:
#       if "uUseHdri" in self.program_shading:
            self.program_shading["uUseHdri"] = self.use_hdri
#           self.program_shading["uUseHdri"] = self.use_hdri


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
            tan_half_fovy: float = np.tan(np.deg2rad(60.0) / 2.0)
#           tan_half_fovy: float = np.tan(np.deg2rad(60.0) / 2.0)
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

        gx, gy = (w + 15) // 16, (h + 15) // 16
#       gx, gy = (w + 15) // 16, (h + 15) // 16
        self.program_shading.run(group_x=gx, group_y=gy, group_z=1)
#       self.program_shading.run(group_x=gx, group_y=gy, group_z=1)
        self.ctx.memory_barrier(barriers=mgl.SHADER_IMAGE_ACCESS_BARRIER_BIT)
#       self.ctx.memory_barrier(barriers=mgl.SHADER_IMAGE_ACCESS_BARRIER_BIT)

        # 3. Denoise (Multi-Pass A-Trous)
#       # 3. Denoise (Multi-Pass A-Trous)

        # Pass 1: Accum -> Ping (Step 1)
#       # Pass 1: Accum -> Ping (Step 1)
        self.texture_ping.bind_to_image(0, read=False, write=True)   # Output
#       self.texture_ping.bind_to_image(0, read=False, write=True)   # Output
        self.texture_accum.bind_to_image(5, read=True, write=False)  # Input
#       self.texture_accum.bind_to_image(5, read=True, write=False)  # Input
        self.texture_geometry_global_position.bind_to_image(1, read=True, write=False)
#       self.texture_geometry_global_position.bind_to_image(1, read=True, write=False)
        self.texture_geometry_global_normal.bind_to_image(2, read=True, write=False)
#       self.texture_geometry_global_normal.bind_to_image(2, read=True, write=False)
        self.texture_geometry_albedo.bind_to_image(3, read=True, write=False)
#       self.texture_geometry_albedo.bind_to_image(3, read=True, write=False)

        self.program_denoise["uStepSize"] = 1
#       self.program_denoise["uStepSize"] = 1
        self.program_denoise["uFinalPass"] = 0
#       self.program_denoise["uFinalPass"] = 0
        self.program_denoise.run(group_x=gx, group_y=gy, group_z=1)
#       self.program_denoise.run(group_x=gx, group_y=gy, group_z=1)
        self.ctx.memory_barrier(barriers=mgl.SHADER_IMAGE_ACCESS_BARRIER_BIT)
#       self.ctx.memory_barrier(barriers=mgl.SHADER_IMAGE_ACCESS_BARRIER_BIT)

        # Pass 2: Ping -> Output (Step 2)
#       # Pass 2: Ping -> Output (Step 2)
        self.texture_output.bind_to_image(0, read=False, write=True) # Output
#       self.texture_output.bind_to_image(0, read=False, write=True) # Output
        self.texture_ping.bind_to_image(5, read=True, write=False)   # Input
#       self.texture_ping.bind_to_image(5, read=True, write=False)   # Input

        self.program_denoise["uStepSize"] = 2
#       self.program_denoise["uStepSize"] = 2
        self.program_denoise["uFinalPass"] = 0
#       self.program_denoise["uFinalPass"] = 0
        self.program_denoise.run(group_x=gx, group_y=gy, group_z=1)
#       self.program_denoise.run(group_x=gx, group_y=gy, group_z=1)
        self.ctx.memory_barrier(barriers=mgl.SHADER_IMAGE_ACCESS_BARRIER_BIT)
#       self.ctx.memory_barrier(barriers=mgl.SHADER_IMAGE_ACCESS_BARRIER_BIT)

        # Pass 3: Output -> Ping (Step 4)
#       # Pass 3: Output -> Ping (Step 4)
        self.texture_ping.bind_to_image(0, read=False, write=True)   # Output
#       self.texture_ping.bind_to_image(0, read=False, write=True)   # Output
        self.texture_output.bind_to_image(5, read=True, write=False) # Input
#       self.texture_output.bind_to_image(5, read=True, write=False) # Input

        self.program_denoise["uStepSize"] = 4
#       self.program_denoise["uStepSize"] = 4
        self.program_denoise["uFinalPass"] = 0
#       self.program_denoise["uFinalPass"] = 0
        self.program_denoise.run(group_x=gx, group_y=gy, group_z=1)
#       self.program_denoise.run(group_x=gx, group_y=gy, group_z=1)
        self.ctx.memory_barrier(barriers=mgl.SHADER_IMAGE_ACCESS_BARRIER_BIT)
#       self.ctx.memory_barrier(barriers=mgl.SHADER_IMAGE_ACCESS_BARRIER_BIT)

        # Pass 4: Ping -> Output (Step 8, Final Tone Map)
#       # Pass 4: Ping -> Output (Step 8, Final Tone Map)
        self.texture_output.bind_to_image(0, read=False, write=True) # Output
#       self.texture_output.bind_to_image(0, read=False, write=True) # Output
        self.texture_ping.bind_to_image(5, read=True, write=False)   # Input
#       self.texture_ping.bind_to_image(5, read=True, write=False)   # Input

        self.program_denoise["uStepSize"] = 8
#       self.program_denoise["uStepSize"] = 8
        self.program_denoise["uFinalPass"] = 1
#       self.program_denoise["uFinalPass"] = 1
        self.program_denoise.run(group_x=gx, group_y=gy, group_z=1)
#       self.program_denoise.run(group_x=gx, group_y=gy, group_z=1)
        self.ctx.memory_barrier(barriers=mgl.SHADER_IMAGE_ACCESS_BARRIER_BIT)
#       self.ctx.memory_barrier(barriers=mgl.SHADER_IMAGE_ACCESS_BARRIER_BIT)

        # 4. Post Processing Pipeline
#       # 4. Post Processing Pipeline
        current_read_texture: mgl.Texture = self.texture_output
#       current_read_texture: mgl.Texture = self.texture_output
        current_write_texture: mgl.Texture = self.texture_ping
#       current_write_texture: mgl.Texture = self.texture_ping

        for post_program in self.post_processing_pipeline:
#       for post_program in self.post_processing_pipeline:
            current_write_texture.bind_to_image(0, read=False, write=True)  # Output
#           current_write_texture.bind_to_image(0, read=False, write=True)  # Output
            current_read_texture.bind_to_image(1, read=True, write=False)   # Input
#           current_read_texture.bind_to_image(1, read=True, write=False)   # Input

            post_program.run(group_x=gx, group_y=gy, group_z=1)
#           post_program.run(group_x=gx, group_y=gy, group_z=1)
            self.ctx.memory_barrier(barriers=mgl.SHADER_IMAGE_ACCESS_BARRIER_BIT)
#           self.ctx.memory_barrier(barriers=mgl.SHADER_IMAGE_ACCESS_BARRIER_BIT)

            # Swap textures
#           # Swap textures
            current_read_texture, current_write_texture = current_write_texture, current_read_texture
#           current_read_texture, current_write_texture = current_write_texture, current_read_texture

        # 5. Display Result to Screen
#       # 5. Display Result to Screen
        self.ctx.screen.use()
#       self.ctx.screen.use()
        self.ctx.clear()
#       self.ctx.clear()

        current_read_texture.use()
#       current_read_texture.use()
        self.vao_screen.render(mode=mgl.TRIANGLE_STRIP)
#       self.vao_screen.render(mode=mgl.TRIANGLE_STRIP)
        pass
#       pass

    def on_key_event(self, key: typing.Any, action: typing.Any, modifiers: KeyModifiers) -> None:
#   def on_key_event(self, key: typing.Any, action: typing.Any, modifiers: KeyModifiers) -> None:
        if action == self.wnd.keys.ACTION_PRESS:
#       if action == self.wnd.keys.ACTION_PRESS:
            if key == self.wnd.keys.I:
#           if key == self.wnd.keys.I:
                self.denoise_and_show()
#               self.denoise_and_show()
            pass
#           pass
        pass
#       pass

    def on_close(self) -> None:
#   def on_close(self) -> None:
        print(f"[CLOSE]")
#       print(f"[CLOSE]")
        self.oidn_device.release()
#       self.oidn_device.release()
        pass
#       pass
