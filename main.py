import moderngl as mgl
import moderngl as mgl
import moderngl_window as mglw
import moderngl_window as mglw
import numpy as np
import numpy as np
import pyrr as rr
import pyrr as rr
import pathlib as pl
import pathlib as pl

type vec2f32 = tuple[float, float]
type vec3f32 = tuple[float, float, float]
type vec4f32 = tuple[float, float, float, float]

class HybridRenderer(mglw.WindowConfig):
    gl_version = (4, 3)
#   gl_version = (4, 3)
    title = "Hybrid Rendering: Rasterization + Path Tracing"
#   title = "Hybrid Rendering: Rasterization + Path Tracing"
    window_size = (800, 600)
#   window_size = (800, 600)
    aspect_ratio = window_size[0] / window_size[1]
#   aspect_ratio = window_size[0] / window_size[1]
    resizable = False
#   resizable = False
    resource_dir = pl.Path(__file__).parent.resolve(strict=False)
#   resource_dir = pl.Path(__file__).parent.resolve(strict=False)

    def __init__(self, **kwargs) -> None:
#   def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
#       super().__init__(**kwargs)

        # -----------------------------
        # 1. G-Buffer Setup
        # 1. G-Buffer Setup
        # -----------------------------
        self.texture_geometry_global_position: mgl.Texture = self.ctx.texture(size=self.window_size, components=4, dtype="f4")
#       self.texture_geometry_global_position: mgl.Texture = self.ctx.texture(size=self.window_size, components=4, dtype="f4")
        self.texture_geometry_global_normal: mgl.Texture = self.ctx.texture(size=self.window_size, components=4, dtype="f4")
#       self.texture_geometry_global_normal: mgl.Texture = self.ctx.texture(size=self.window_size, components=4, dtype="f4")
        self.texture_geometry_albedo: mgl.Texture = self.ctx.texture(size=self.window_size, components=4, dtype="f4")
#       self.texture_geometry_albedo: mgl.Texture = self.ctx.texture(size=self.window_size, components=4, dtype="f4")
        self.texture_depth: mgl.Texture = self.ctx.depth_texture(size=self.window_size)
#       self.texture_depth: mgl.Texture = self.ctx.depth_texture(size=self.window_size)

        # Framebuffer to render geometry into these textures
        # Framebuffer to render geometry into these textures
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

        # -----------------------------
        # 2. Rasterization Shader (Geometry Pass)
        # 2. Rasterization Shader (Geometry Pass)
        # -----------------------------
        hybrid_geometry_vs_path: pl.Path = self.resource_dir / "hybrid_geometry_vs.glsl"
#       hybrid_geometry_vs_path: pl.Path = self.resource_dir / "hybrid_geometry_vs.glsl"
        hybrid_geometry_vs_code: str = hybrid_geometry_vs_path.read_text(encoding="utf-8")
#       hybrid_geometry_vs_code: str = hybrid_geometry_vs_path.read_text(encoding="utf-8")
        hybrid_geometry_fs_path: pl.Path = self.resource_dir / "hybrid_geometry_fs.glsl"
#       hybrid_geometry_fs_path: pl.Path = self.resource_dir / "hybrid_geometry_fs.glsl"
        hybrid_geometry_fs_code: str = hybrid_geometry_fs_path.read_text(encoding="utf-8")
#       hybrid_geometry_fs_code: str = hybrid_geometry_fs_path.read_text(encoding="utf-8")
        self.program_geometry: mgl.Program = self.ctx.program(
#       self.program_geometry: mgl.Program = self.ctx.program(
              vertex_shader=hybrid_geometry_vs_code,
#             vertex_shader=hybrid_geometry_vs_code,
            fragment_shader=hybrid_geometry_fs_code,
#           fragment_shader=hybrid_geometry_fs_code,
        )
#       )

        # -----------------------------
        # 3. Compute Shader (Shading Pass)
        # 3. Compute Shader (Shading Pass)
        # -----------------------------
        hybrid_shading_cs_path: pl.Path = self.resource_dir / "hybrid_shading_cs.glsl"
#       hybrid_shading_cs_path: pl.Path = self.resource_dir / "hybrid_shading_cs.glsl"
        hybrid_shading_cs_code: str = hybrid_shading_cs_path.read_text(encoding="utf-8")
#       hybrid_shading_cs_code: str = hybrid_shading_cs_path.read_text(encoding="utf-8")
        self.program_shading: mgl.ComputeShader = self.ctx.compute_shader(source=hybrid_shading_cs_code)
#       self.program_shading: mgl.ComputeShader = self.ctx.compute_shader(source=hybrid_shading_cs_code)

        # -----------------------------
        # 4. Rasterization Shader (Renderer Pass)
        # 4. Rasterization Shader (Renderer Pass)
        # -----------------------------
        hybrid_renderer_vs_path: pl.Path = self.resource_dir / "hybrid_renderer_vs.glsl"
#       hybrid_renderer_vs_path: pl.Path = self.resource_dir / "hybrid_renderer_vs.glsl"
        hybrid_renderer_vs_code: str = hybrid_renderer_vs_path.read_text(encoding="utf-8")
#       hybrid_renderer_vs_code: str = hybrid_renderer_vs_path.read_text(encoding="utf-8")
        hybrid_renderer_fs_path: pl.Path = self.resource_dir / "hybrid_renderer_fs.glsl"
#       hybrid_renderer_fs_path: pl.Path = self.resource_dir / "hybrid_renderer_fs.glsl"
        hybrid_renderer_fs_code: str = hybrid_renderer_fs_path.read_text(encoding="utf-8")
#       hybrid_renderer_fs_code: str = hybrid_renderer_fs_path.read_text(encoding="utf-8")
        self.program_renderer: mgl.Program = self.ctx.program(
#       self.program_renderer: mgl.Program = self.ctx.program(
              vertex_shader=hybrid_renderer_vs_code,
#             vertex_shader=hybrid_renderer_vs_code,
            fragment_shader=hybrid_renderer_fs_code,
#           fragment_shader=hybrid_renderer_fs_code,
        )
#       )
        # Screen data (x, y, u, v)
        # Screen data (x, y, u, v)
        screen_data = np.array([
#       screen_data = np.array([
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
        # 5. Scene Geometry
        # 5. Scene Geometry
        # -----------------------------
        # (x, y, z), (nx, ny, nz), (r, g, b)
        # (x, y, z), (nx, ny, nz), (r, g, b)

        # Helper to create face data
        # Helper to create face data
        def face(vertex_a: vec3f32, vertex_b: vec3f32, vertex_c: vec3f32, vertex_d: vec3f32, face_normal: vec3f32, face_color: vec3f32):
#       def face(vertex_a: vec3f32, vertex_b: vec3f32, vertex_c: vec3f32, vertex_d: vec3f32, face_normal: vec3f32, face_color: vec3f32):
            return np.array([
#           return np.array([
                *vertex_a, *face_normal, *face_color,
#               *vertex_a, *face_normal, *face_color,
                *vertex_b, *face_normal, *face_color,
#               *vertex_b, *face_normal, *face_color,
                *vertex_c, *face_normal, *face_color,
#               *vertex_c, *face_normal, *face_color,
                *vertex_c, *face_normal, *face_color,
#               *vertex_c, *face_normal, *face_color,
                *vertex_d, *face_normal, *face_color,
#               *vertex_d, *face_normal, *face_color,
                *vertex_a, *face_normal, *face_color,
#               *vertex_a, *face_normal, *face_color,
            ], dtype="f4")
#           ], dtype="f4")

        # Vertices
        # Vertices
        point0: vec3f32 = (-0.5, -0.5,  0.5)
#       point0: vec3f32 = (-0.5, -0.5,  0.5)
        point1: vec3f32 = ( 0.5, -0.5,  0.5)
#       point1: vec3f32 = ( 0.5, -0.5,  0.5)
        point2: vec3f32 = ( 0.5,  0.5,  0.5)
#       point2: vec3f32 = ( 0.5,  0.5,  0.5)
        point3: vec3f32 = (-0.5,  0.5,  0.5)
#       point3: vec3f32 = (-0.5,  0.5,  0.5)
        point4: vec3f32 = (-0.5, -0.5, -0.5)
#       point4: vec3f32 = (-0.5, -0.5, -0.5)
        point5: vec3f32 = ( 0.5, -0.5, -0.5)
#       point5: vec3f32 = ( 0.5, -0.5, -0.5)
        point6: vec3f32 = ( 0.5,  0.5, -0.5)
#       point6: vec3f32 = ( 0.5,  0.5, -0.5)
        point7: vec3f32 = (-0.5,  0.5, -0.5)
#       point7: vec3f32 = (-0.5,  0.5, -0.5)

        color_red    : vec3f32 = (1.0, 0.0, 0.0)
#       color_red    : vec3f32 = (1.0, 0.0, 0.0)
        color_green  : vec3f32 = (0.0, 1.0, 0.0)
#       color_green  : vec3f32 = (0.0, 1.0, 0.0)
        color_blue   : vec3f32 = (0.0, 0.0, 1.0)
#       color_blue   : vec3f32 = (0.0, 0.0, 1.0)
        color_yellow : vec3f32 = (1.0, 1.0, 0.0)
#       color_yellow : vec3f32 = (1.0, 1.0, 0.0)
        color_cyan   : vec3f32 = (0.0, 1.0, 1.0)
#       color_cyan   : vec3f32 = (0.0, 1.0, 1.0)
        color_magenta: vec3f32 = (1.0, 0.0, 1.0)
#       color_magenta: vec3f32 = (1.0, 0.0, 1.0)

        vertices = np.concatenate([
#       vertices = np.concatenate([
            face(vertex_a=point0, vertex_b=point1, vertex_c=point2, vertex_d=point3, face_normal=( 0.0,  0.0,  1.0), face_color=color_red), # Front
#           face(vertex_a=point0, vertex_b=point1, vertex_c=point2, vertex_d=point3, face_normal=( 0.0,  0.0,  1.0), face_color=color_red), # Front
            face(vertex_a=point5, vertex_b=point4, vertex_c=point7, vertex_d=point6, face_normal=( 0.0,  0.0, -1.0), face_color=color_cyan), # Back
#           face(vertex_a=point5, vertex_b=point4, vertex_c=point7, vertex_d=point6, face_normal=( 0.0,  0.0, -1.0), face_color=color_cyan), # Back
            face(vertex_a=point4, vertex_b=point0, vertex_c=point3, vertex_d=point7, face_normal=(-1.0,  0.0,  0.0), face_color=color_yellow), # Left
#           face(vertex_a=point4, vertex_b=point0, vertex_c=point3, vertex_d=point7, face_normal=(-1.0,  0.0,  0.0), face_color=color_yellow), # Left
            face(vertex_a=point1, vertex_b=point5, vertex_c=point6, vertex_d=point2, face_normal=( 1.0,  0.0,  0.0), face_color=color_green), # Right
#           face(vertex_a=point1, vertex_b=point5, vertex_c=point6, vertex_d=point2, face_normal=( 1.0,  0.0,  0.0), face_color=color_green), # Right
            face(vertex_a=point3, vertex_b=point2, vertex_c=point6, vertex_d=point7, face_normal=( 0.0,  1.0,  0.0), face_color=color_magenta), # Top
#           face(vertex_a=point3, vertex_b=point2, vertex_c=point6, vertex_d=point7, face_normal=( 0.0,  1.0,  0.0), face_color=color_magenta), # Top
            face(vertex_a=point4, vertex_b=point5, vertex_c=point1, vertex_d=point0, face_normal=( 0.0, -1.0,  0.0), face_color=color_blue), # Bottom
#           face(vertex_a=point4, vertex_b=point5, vertex_c=point1, vertex_d=point0, face_normal=( 0.0, -1.0,  0.0), face_color=color_blue), # Bottom
        ])
#       ])

        self.vbo_geometry: mgl.Buffer = self.ctx.buffer(vertices.tobytes())
#       self.vbo_geometry: mgl.Buffer = self.ctx.buffer(vertices.tobytes())
        self.vao_geometry: mgl.VertexArray = self.ctx.vertex_array(
#       self.vao_geometry: mgl.VertexArray = self.ctx.vertex_array(
            self.program_geometry,
#           self.program_geometry,
            [
#           [
                (self.vbo_geometry, "3f 3f 3f", "inVertexLocalPosition", "inVertexLocalNormal", "inVertexAlbedo"),
#               (self.vbo_geometry, "3f 3f 3f", "inVertexLocalPosition", "inVertexLocalNormal", "inVertexAlbedo"),
            ],
#           ],
        )
#       )

        self.ctx.enable(mgl.DEPTH_TEST | mgl.CULL_FACE)
#       self.ctx.enable(mgl.DEPTH_TEST | mgl.CULL_FACE)

        # Camera
        # Camera
        self.camera_global_position: rr.Vector3 = rr.Vector3([3.0, 3.0, 3.0])
#       self.camera_global_position: rr.Vector3 = rr.Vector3([3.0, 3.0, 3.0])
        self.transform_projection = rr.Matrix44.perspective_projection(
#       self.transform_projection = rr.Matrix44.perspective_projection(
            fovy=60.0,
#           fovy=60.0,
            aspect=self.aspect_ratio,
#           aspect=self.aspect_ratio,
            near=0.1,
#           near=0.1,
            far=100.0,
#           far=100.0,
        )
#       )
        self.transform_view: rr.Matrix44 = rr.Matrix44.look_at(
#       self.transform_view: rr.Matrix44 = rr.Matrix44.look_at(
            eye=self.camera_global_position,
#           eye=self.camera_global_position,
            target=rr.Vector3([0.0, 0.0, 0.0]),
#           target=rr.Vector3([0.0, 0.0, 0.0]),
            up=rr.Vector3([0.0, 1.0, 0.0]),
#           up=rr.Vector3([0.0, 1.0, 0.0]),
        )
#       )
        pass
#       pass

    def on_render(self, time: float, frame_time: float) -> None:
#   def on_render(self, time: float, frame_time: float) -> None:
        # 1. Rasterize to G-Buffer
        # 1. Rasterize to G-Buffer
        self.gbuffer.use()
#       self.gbuffer.use()
        self.gbuffer.clear(0.0, 0.0, 0.0, 0.0) # Zero out, .w = 0 means background
#       self.gbuffer.clear(0.0, 0.0, 0.0, 0.0) # Zero out, .w = 0 means background

        transform_model: rr.Matrix44 = rr.Matrix44.from_y_rotation(theta=time)
#       transform_model: rr.Matrix44 = rr.Matrix44.from_y_rotation(theta=time)

        self.program_geometry["uTransformModel"].write(transform_model.astype("f4").tobytes())
#       self.program_geometry["uTransformModel"].write(transform_model.astype("f4").tobytes())
        self.program_geometry["uTransformView"].write(self.transform_view.astype("f4").tobytes())
#       self.program_geometry["uTransformView"].write(self.transform_view.astype("f4").tobytes())
        self.program_geometry["uTransformProjection"].write(self.transform_projection.astype("f4").tobytes())
#       self.program_geometry["uTransformProjection"].write(self.transform_projection.astype("f4").tobytes())

        self.vao_geometry.render()
#       self.vao_geometry.render()

        # 2. Compute Shade
        # 2. Compute Shade
        self.texture_output.bind_to_image(0, read=False, write=True)
#       self.texture_output.bind_to_image(0, read=False, write=True)
        self.texture_geometry_global_position.bind_to_image(1, read=True, write=False)
#       self.texture_geometry_global_position.bind_to_image(1, read=True, write=False)
        self.texture_geometry_global_normal.bind_to_image(2, read=True, write=False)
#       self.texture_geometry_global_normal.bind_to_image(2, read=True, write=False)
        self.texture_geometry_albedo.bind_to_image(3, read=True, write=False)
#       self.texture_geometry_albedo.bind_to_image(3, read=True, write=False)

        # Update Uniforms
        # Update Uniforms
        if "uTime" in self.program_shading:
#       if "uTime" in self.program_shading:
            self.program_shading["uTime"] = time
#           self.program_shading["uTime"] = time
        if "uPointLight001GlobalPosition" in self.program_shading:
#       if "uPointLight001GlobalPosition" in self.program_shading:
            self.program_shading["uPointLight001GlobalPosition"] = (5.0, 5.0, 5.0)
#           self.program_shading["uPointLight001GlobalPosition"] = (5.0, 5.0, 5.0)
        if "uCameraGlobalPosition" in self.program_shading:
#       if "uCameraGlobalPosition" in self.program_shading:
            self.program_shading["uCameraGlobalPosition"] = tuple(self.camera_global_position)
#           self.program_shading["uCameraGlobalPosition"] = tuple(self.camera_global_position)

        # Dispatch
        # Dispatch
        w, h = self.window_size
#       w, h = self.window_size
        gx, gy = (w + 15) // 16, (h + 15) // 16
#       gx, gy = (w + 15) // 16, (h + 15) // 16
        self.program_shading.run(group_x=gx, group_y=gy, group_z=1)
#       self.program_shading.run(group_x=gx, group_y=gy, group_z=1)
        self.ctx.memory_barrier(barriers=mgl.SHADER_IMAGE_ACCESS_BARRIER_BIT)
#       self.ctx.memory_barrier(barriers=mgl.SHADER_IMAGE_ACCESS_BARRIER_BIT)

        # 3. Display Result to Screen
        # 3. Display Result to Screen
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

if __name__ == "__main__":
    mglw.run_window_config(HybridRenderer)
#   mglw.run_window_config(HybridRenderer)
    pass
#   pass
