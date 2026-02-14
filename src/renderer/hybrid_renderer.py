import moderngl as mgl
import moderngl as mgl
import moderngl_window as mglw
import moderngl_window as mglw
import numpy as np
import numpy as np
import numpy.typing as npt
import numpy.typing as npt
import pyrr as rr
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
from src.renderer.shader_compiler import resolve_includes
from src.renderer.shader_compiler import resolve_includes
from src.scene.scene_builder import SceneBuilder, SceneBatch
from src.scene.scene_builder import SceneBuilder, SceneBatch
from src.scene.camera import Camera
from src.scene.camera import Camera
from src.core.common_types import vec2f32, vec3f32, vec4f32
from src.core.common_types import vec2f32, vec3f32, vec4f32

class HybridRenderer(mglw.WindowConfig):
    gl_version: tuple[int, int] = (4, 3)
#   gl_version: tuple[int, int] = (4, 3)
    title: str = "Hybrid Rendering: Rasterization + Path Tracing"
#   title: str = "Hybrid Rendering: Rasterization + Path Tracing"
    window_size: tuple[int, int] = (800, 600)
#   window_size: tuple[int, int] = (800, 600)
    aspect_ratio: float = window_size[0] / window_size[1]
#   aspect_ratio: float = window_size[0] / window_size[1]
    resizable: bool = False
#   resizable: bool = False
    resource_dir: pl.Path = pl.Path(__file__).parent.resolve(strict=False)
#   resource_dir: pl.Path = pl.Path(__file__).parent.resolve(strict=False)

    def __init__(self, **kwargs) -> None:
#   def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
#       super().__init__(**kwargs)

        self.frame_count: int = 0
#       self.frame_count: int = 0

        # -----------------------------
#       # -----------------------------
        # 1. G-Buffer Setup
#       # 1. G-Buffer Setup
        # -----------------------------
#       # -----------------------------
        self.texture_geometry_global_position: mgl.Texture = self.ctx.texture(size=self.window_size, components=4, dtype="f4")
#       self.texture_geometry_global_position: mgl.Texture = self.ctx.texture(size=self.window_size, components=4, dtype="f4")
        self.texture_geometry_global_normal: mgl.Texture = self.ctx.texture(size=self.window_size, components=4, dtype="f4")
#       self.texture_geometry_global_normal: mgl.Texture = self.ctx.texture(size=self.window_size, components=4, dtype="f4")
        self.texture_geometry_albedo: mgl.Texture = self.ctx.texture(size=self.window_size, components=4, dtype="f4")
#       self.texture_geometry_albedo: mgl.Texture = self.ctx.texture(size=self.window_size, components=4, dtype="f4")
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

        # HDRI Texture Loading
#       # HDRI Texture Loading
        hdri_path: pl.Path = self.resource_dir / "../assets/rostock_laage_airport_4k.exr"
#       hdri_path: pl.Path = self.resource_dir / "../assets/rostock_laage_airport_4k.exr"
        self.use_hdri: bool = hdri_path.exists()
#       self.use_hdri: bool = hdri_path.exists()
        if self.use_hdri:
#       if self.use_hdri:
            hdri_data: npt.NDArray[np.float32] = cv2.imread(str(hdri_path), cv2.IMREAD_UNCHANGED).astype("f4")
#           hdri_data: npt.NDArray[np.float32] = cv2.imread(str(hdri_path), cv2.IMREAD_UNCHANGED).astype("f4")
            hdri_data = cv2.cvtColor(hdri_data, cv2.COLOR_BGR2RGB)
#           hdri_data = cv2.cvtColor(hdri_data, cv2.COLOR_BGR2RGB)
            hdri_data = np.ascontiguousarray(np.flipud(hdri_data))
#           hdri_data = np.ascontiguousarray(np.flipud(hdri_data))
            h, w, c = hdri_data.shape
#           h, w, c = hdri_data.shape
            self.texture_hdri: mgl.Texture = self.ctx.texture((w, h), components=3, dtype="f4", data=hdri_data.tobytes())
#           self.texture_hdri: mgl.Texture = self.ctx.texture((w, h), components=3, dtype="f4", data=hdri_data.tobytes())
            self.texture_hdri.filter = (mgl.LINEAR_MIPMAP_LINEAR, mgl.LINEAR)
#           self.texture_hdri.filter = (mgl.LINEAR_MIPMAP_LINEAR, mgl.LINEAR)
            self.texture_hdri.build_mipmaps()
#           self.texture_hdri.build_mipmaps()
            self.texture_hdri.repeat_x = True
#           self.texture_hdri.repeat_x = True
            self.texture_hdri.repeat_y = False
#           self.texture_hdri.repeat_y = False
        else:
#       else:
            # Create dummy 1x1 texture
#           # Create dummy 1x1 texture
            self.texture_hdri: mgl.Texture = self.ctx.texture((1, 1), components=3, dtype="f4", data=np.zeros(3, dtype="f4").tobytes())
#           self.texture_hdri: mgl.Texture = self.ctx.texture((1, 1), components=3, dtype="f4", data=np.zeros(3, dtype="f4").tobytes())

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
        ], dtype="f4")
#       ], dtype="f4")
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

        self.scene_builder = SceneBuilder(self.ctx, self.program_geometry)
#       self.scene_builder = SceneBuilder(self.ctx, self.program_geometry)

        self.scene_builder.add_cube(position=(-1.5, 0.0, 1.2), rotation=(0.0, 0.0, 0.0), scale=(1.0, 1.0, 1.0), color=(1.0, 0.2, 0.1)) # Warm Red
#       self.scene_builder.add_cube(position=(-1.5, 0.0, 1.2), rotation=(0.0, 0.0, 0.0), scale=(1.0, 1.0, 1.0), color=(1.0, 0.2, 0.1)) # Warm Red
        self.scene_builder.add_cube(position=(0.0, 0.0, -1.2), rotation=(0.0, 0.0, 0.0), scale=(1.0, 1.0, 1.0), color=(0.4, 0.8, 0.1)) # Warm Green
#       self.scene_builder.add_cube(position=(0.0, 0.0, -1.2), rotation=(0.0, 0.0, 0.0), scale=(1.0, 1.0, 1.0), color=(0.4, 0.8, 0.1)) # Warm Green
        self.scene_builder.add_cube(position=(1.5, 0.0, 1.2), rotation=(0.0, 0.0, 0.0), scale=(1.0, 1.0, 1.0), color=(0.1, 0.3, 0.9)) # Warm Blue
#       self.scene_builder.add_cube(position=(1.5, 0.0, 1.2), rotation=(0.0, 0.0, 0.0), scale=(1.0, 1.0, 1.0), color=(0.1, 0.3, 0.9)) # Warm Blue

        self.scene_builder.add_plane(position=(0.0, -0.5, 0.0), rotation=(0.0, 0.0, 0.0), scale=(20.0, 1.0, 20.0), color=(0.3, 0.3, 0.3)) # Gray Plane
#       self.scene_builder.add_plane(position=(0.0, -0.5, 0.0), rotation=(0.0, 0.0, 0.0), scale=(20.0, 1.0, 20.0), color=(0.3, 0.3, 0.3)) # Gray Plane

        bvh_data, triangles_data, materials_data = self.scene_builder.build()
#       bvh_data, triangles_data, materials_data = self.scene_builder.build()
        self.scene_batches = self.scene_builder.scene_batches
#       self.scene_batches = self.scene_builder.scene_batches

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

        self.ctx.enable(mgl.DEPTH_TEST | mgl.CULL_FACE)
#       self.ctx.enable(mgl.DEPTH_TEST | mgl.CULL_FACE)

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

    def key_event(self, key: int, action: int, modifiers: int) -> None:
#   def key_event(self, key: int, action: int, modifiers: int) -> None:
        keys: typing.Any = self.wnd.keys
#       keys: typing.Any = self.wnd.keys
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

    def on_render(self, time: float, frame_time: float) -> None:
#   def on_render(self, time: float, frame_time: float) -> None:
        self.frame_count += 1
#       self.frame_count += 1

        # Poll keys (Backup for key_event)
#       # Poll keys (Backup for key_event)
        keys: typing.Any = self.wnd.keys
#       keys: typing.Any = self.wnd.keys
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

        self.camera.update(frame_time, self.key_state)
#       self.camera.update(frame_time, self.key_state)

        # TAA Jitter
#       # TAA Jitter
        # Halton sequence for x (base 2) and y (base 3)
#       # Halton sequence for x (base 2) and y (base 3)
        # Scale to [-0.5, 0.5] pixels
#       # Scale to [-0.5, 0.5] pixels
        jitter_x = (self.get_halton_jitter(index=self.frame_count, base=2) - 0.5)
#       jitter_x = (self.get_halton_jitter(index=self.frame_count, base=2) - 0.5)
        jitter_y = (self.get_halton_jitter(index=self.frame_count, base=3) - 0.5)
#       jitter_y = (self.get_halton_jitter(index=self.frame_count, base=3) - 0.5)

        transform_view = self.camera.get_view_matrix()
#       transform_view = self.camera.get_view_matrix()
        transform_projection = self.camera.get_projection_matrix(jitter=(jitter_x, jitter_y), window_size=self.window_size)
#       transform_projection = self.camera.get_projection_matrix(jitter=(jitter_x, jitter_y), window_size=self.window_size)

        # 1. Rasterize to G-Buffer
#       # 1. Rasterize to G-Buffer
        self.gbuffer.use()
#       self.gbuffer.use()
        self.gbuffer.clear(0.0, 0.0, 0.0, 0.0) # Zero out, .w = 0 means background
#       self.gbuffer.clear(0.0, 0.0, 0.0, 0.0) # Zero out, .w = 0 means background

        self.program_geometry["uTransformView"].write(transform_view.astype("f4").tobytes())
#       self.program_geometry["uTransformView"].write(transform_view.astype("f4").tobytes())
        self.program_geometry["uTransformProjection"].write(transform_projection.astype("f4").tobytes())
#       self.program_geometry["uTransformProjection"].write(transform_projection.astype("f4").tobytes())

        current_tri_offset = 0
#       current_tri_offset = 0
        for scene_batch in self.scene_batches:
#       for scene_batch in self.scene_batches:
            if "uBaseTriangleIndexOffset" in self.program_geometry:
#           if "uBaseTriangleIndexOffset" in self.program_geometry:
                self.program_geometry["uBaseTriangleIndexOffset"] = current_tri_offset
#               self.program_geometry["uBaseTriangleIndexOffset"] = current_tri_offset
            if "uTriangleCountPerInstance" in self.program_geometry:
#           if "uTriangleCountPerInstance" in self.program_geometry:
                self.program_geometry["uTriangleCountPerInstance"] = scene_batch.triangle_count_per_instance
#               self.program_geometry["uTriangleCountPerInstance"] = scene_batch.triangle_count_per_instance
            scene_batch.vao.render(instances=scene_batch.number_of_instances)
#           scene_batch.vao.render(instances=scene_batch.number_of_instances)
            current_tri_offset += scene_batch.number_of_instances * scene_batch.triangle_count_per_instance
#           current_tri_offset += scene_batch.number_of_instances * scene_batch.triangle_count_per_instance

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

        # Bind BVH buffers
#       # Bind BVH buffers
        self.ssbo_bvh_nodes.bind_to_storage_buffer(binding=4)
#       self.ssbo_bvh_nodes.bind_to_storage_buffer(binding=4)
        self.ssbo_triangles.bind_to_storage_buffer(binding=5)
#       self.ssbo_triangles.bind_to_storage_buffer(binding=5)
        self.ssbo_materials.bind_to_storage_buffer(binding=7)
#       self.ssbo_materials.bind_to_storage_buffer(binding=7)

        self.texture_accum.bind_to_image(6, read=True, write=True)
#       self.texture_accum.bind_to_image(6, read=True, write=True)

        # Update Uniforms
#       # Update Uniforms
        if "uTime" in self.program_shading:
#       if "uTime" in self.program_shading:
            self.program_shading["uTime"] = time
#           self.program_shading["uTime"] = time
        if "uPointLight001GlobalPosition" in self.program_shading:
#       if "uPointLight001GlobalPosition" in self.program_shading:
            radius = 6.0
#           radius = 6.0
            x = np.cos(time) * radius
#           x = np.cos(time) * radius
            z = np.sin(time) * radius
#           z = np.sin(time) * radius
            self.program_shading["uPointLight001GlobalPosition"] = (x, 5.0, z)
#           self.program_shading["uPointLight001GlobalPosition"] = (x, 5.0, z)
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

            focal_length = rr.vector.length(self.camera.look_from - self.camera.look_at)
#           focal_length = rr.vector.length(self.camera.look_from - self.camera.look_at)
            tan_half_fovy = np.tan(np.deg2rad(60.0) / 2.0)
#           tan_half_fovy = np.tan(np.deg2rad(60.0) / 2.0)
            viewport_height = 2.0 * tan_half_fovy * focal_length
#           viewport_height = 2.0 * tan_half_fovy * focal_length
            viewport_width = viewport_height * self.aspect_ratio
#           viewport_width = viewport_height * self.aspect_ratio

            viewport_u = cam_u * viewport_width
#           viewport_u = cam_u * viewport_width
            viewport_v = cam_v * viewport_height
#           viewport_v = cam_v * viewport_height

            pixel_delta_u = viewport_u / w
#           pixel_delta_u = viewport_u / w
            pixel_delta_v = viewport_v / h
#           pixel_delta_v = viewport_v / h

            viewport_upper_left = self.camera.look_from - (cam_w * focal_length) - (viewport_u / 2.0) - (viewport_v / 2.0)
#           viewport_upper_left = self.camera.look_from - (cam_w * focal_length) - (viewport_u / 2.0) - (viewport_v / 2.0)
            pixel00_loc = viewport_upper_left + 0.5 * (pixel_delta_u + pixel_delta_v)
#           pixel00_loc = viewport_upper_left + 0.5 * (pixel_delta_u + pixel_delta_v)

            self.program_shading["uPixel00Coordinates"] = tuple(pixel00_loc)
#           self.program_shading["uPixel00Coordinates"] = tuple(pixel00_loc)
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

        # 4. Display Result to Screen
#       # 4. Display Result to Screen
        self.ctx.screen.use()
#       self.ctx.screen.use()
        self.ctx.clear()
#       self.ctx.clear()

        self.texture_output.use()
#       self.texture_output.use()
        self.vao_screen.render(mode=mgl.TRIANGLE_STRIP)
#       self.vao_screen.render(mode=mgl.TRIANGLE_STRIP)
        pass
#       pass
