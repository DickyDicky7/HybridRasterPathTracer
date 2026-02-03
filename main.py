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

        class SceneBatch:
#       class SceneBatch:
            def __init__(self, vao: mgl.VertexArray, number_of_instances: int) -> None:
#           def __init__(self, vao: mgl.VertexArray, number_of_instances: int) -> None:
                self.vao: mgl.VertexArray = vao
#               self.vao: mgl.VertexArray = vao
                self.number_of_instances: int = number_of_instances
#               self.number_of_instances: int = number_of_instances
                pass
#               pass

        self.scene_batches: list[SceneBatch] = []
#       self.scene_batches: list[SceneBatch] = []

        # Helper to create face data
        # Helper to create face data
        def face(vertex_a: vec3f32, vertex_b: vec3f32, vertex_c: vec3f32, vertex_d: vec3f32, face_normal: vec3f32):
#       def face(vertex_a: vec3f32, vertex_b: vec3f32, vertex_c: vec3f32, vertex_d: vec3f32, face_normal: vec3f32):
            return np.array([
#           return np.array([
                *vertex_a, *face_normal,
#               *vertex_a, *face_normal,
                *vertex_b, *face_normal,
#               *vertex_b, *face_normal,
                *vertex_c, *face_normal,
#               *vertex_c, *face_normal,
                *vertex_c, *face_normal,
#               *vertex_c, *face_normal,
                *vertex_d, *face_normal,
#               *vertex_d, *face_normal,
                *vertex_a, *face_normal,
#               *vertex_a, *face_normal,
            ], dtype="f4")
#           ], dtype="f4")

        cube_instance_data: list = []
#       cube_instance_data: list = []
        plane_instance_data: list = []
#       plane_instance_data: list = []

        def add_cube(position: vec3f32, rotation: vec3f32, scale: vec3f32, color: vec3f32) -> None:
#       def add_cube(position: vec3f32, rotation: vec3f32, scale: vec3f32, color: vec3f32) -> None:
            matrix_translation = rr.Matrix44.from_translation(position)
#           matrix_translation = rr.Matrix44.from_translation(position)
            matrix_rotation = rr.Matrix44.from_eulers(rotation)
#           matrix_rotation = rr.Matrix44.from_eulers(rotation)
            matrix_scale = rr.Matrix44.from_scale(scale)
#           matrix_scale = rr.Matrix44.from_scale(scale)
            matrix = (matrix_translation * matrix_rotation * matrix_scale).astype("f4")
#           matrix = (matrix_translation * matrix_rotation * matrix_scale).astype("f4")
            data = np.concatenate([matrix.flatten(), color]).astype("f4")
#           data = np.concatenate([matrix.flatten(), color]).astype("f4")
            cube_instance_data.append(data)
#           cube_instance_data.append(data)

        def add_plane(position: vec3f32, rotation: vec3f32, scale: vec3f32, color: vec3f32) -> None:
#       def add_plane(position: vec3f32, rotation: vec3f32, scale: vec3f32, color: vec3f32) -> None:
            matrix_translation = rr.Matrix44.from_translation(position)
#           matrix_translation = rr.Matrix44.from_translation(position)
            matrix_rotation = rr.Matrix44.from_eulers(rotation)
#           matrix_rotation = rr.Matrix44.from_eulers(rotation)
            matrix_scale = rr.Matrix44.from_scale(scale)
#           matrix_scale = rr.Matrix44.from_scale(scale)
            matrix = (matrix_translation * matrix_rotation * matrix_scale).astype("f4")
#           matrix = (matrix_translation * matrix_rotation * matrix_scale).astype("f4")
            data = np.concatenate([matrix.flatten(), color]).astype("f4")
#           data = np.concatenate([matrix.flatten(), color]).astype("f4")
            plane_instance_data.append(data)
#           plane_instance_data.append(data)

        add_cube(position=(-1.5, 0.0, 1.2), rotation=(0.0, 0.0, 0.0), scale=(1.0, 1.0, 1.0), color=(1.0, 0.2, 0.1)) # Warm Red
#       add_cube(position=(-1.5, 0.0, 1.2), rotation=(0.0, 0.0, 0.0), scale=(1.0, 1.0, 1.0), color=(1.0, 0.2, 0.1)) # Warm Red
        add_cube(position=(0.0, 0.0, -1.2), rotation=(0.0, 0.0, 0.0), scale=(1.0, 1.0, 1.0), color=(0.4, 0.8, 0.1)) # Warm Green
#       add_cube(position=(0.0, 0.0, -1.2), rotation=(0.0, 0.0, 0.0), scale=(1.0, 1.0, 1.0), color=(0.4, 0.8, 0.1)) # Warm Green
        add_cube(position=(1.5, 0.0, 1.2), rotation=(0.0, 0.0, 0.0), scale=(1.0, 1.0, 1.0), color=(0.1, 0.3, 0.9)) # Warm Blue
#       add_cube(position=(1.5, 0.0, 1.2), rotation=(0.0, 0.0, 0.0), scale=(1.0, 1.0, 1.0), color=(0.1, 0.3, 0.9)) # Warm Blue

        add_plane(position=(0.0, -0.5, 0.0), rotation=(0.0, 0.0, 0.0), scale=(20.0, 1.0, 20.0), color=(0.5, 0.5, 0.5)) # Gray Plane
#       add_plane(position=(0.0, -0.5, 0.0), rotation=(0.0, 0.0, 0.0), scale=(20.0, 1.0, 20.0), color=(0.5, 0.5, 0.5)) # Gray Plane

        # Create Cube Batch
        # Create Cube Batch
        if cube_instance_data:
#       if cube_instance_data:
            # 1. Geometry
            # 1. Geometry
            point0: vec3f32 = (-0.5, -0.5, 0.5)
#           point0: vec3f32 = (-0.5, -0.5, 0.5)
            point1: vec3f32 = (0.5, -0.5, 0.5)
#           point1: vec3f32 = (0.5, -0.5, 0.5)
            point2: vec3f32 = (0.5, 0.5, 0.5)
#           point2: vec3f32 = (0.5, 0.5, 0.5)
            point3: vec3f32 = (-0.5, 0.5, 0.5)
#           point3: vec3f32 = (-0.5, 0.5, 0.5)
            point4: vec3f32 = (-0.5, -0.5, -0.5)
#           point4: vec3f32 = (-0.5, -0.5, -0.5)
            point5: vec3f32 = (0.5, -0.5, -0.5)
#           point5: vec3f32 = (0.5, -0.5, -0.5)
            point6: vec3f32 = (0.5, 0.5, -0.5)
#           point6: vec3f32 = (0.5, 0.5, -0.5)
            point7: vec3f32 = (-0.5, 0.5, -0.5)
#           point7: vec3f32 = (-0.5, 0.5, -0.5)

            geometries: list = []
#           geometries: list = []
            geometries.append(face(vertex_a=point0, vertex_b=point1, vertex_c=point2, vertex_d=point3, face_normal=(0.0, 0.0, 1.0)))
#           geometries.append(face(vertex_a=point0, vertex_b=point1, vertex_c=point2, vertex_d=point3, face_normal=(0.0, 0.0, 1.0)))
            geometries.append(face(vertex_a=point5, vertex_b=point4, vertex_c=point7, vertex_d=point6, face_normal=(0.0, 0.0, -1.0)))
#           geometries.append(face(vertex_a=point5, vertex_b=point4, vertex_c=point7, vertex_d=point6, face_normal=(0.0, 0.0, -1.0)))
            geometries.append(face(vertex_a=point4, vertex_b=point0, vertex_c=point3, vertex_d=point7, face_normal=(-1.0, 0.0, 0.0)))
#           geometries.append(face(vertex_a=point4, vertex_b=point0, vertex_c=point3, vertex_d=point7, face_normal=(-1.0, 0.0, 0.0)))
            geometries.append(face(vertex_a=point1, vertex_b=point5, vertex_c=point6, vertex_d=point2, face_normal=(1.0, 0.0, 0.0)))
#           geometries.append(face(vertex_a=point1, vertex_b=point5, vertex_c=point6, vertex_d=point2, face_normal=(1.0, 0.0, 0.0)))
            geometries.append(face(vertex_a=point3, vertex_b=point2, vertex_c=point6, vertex_d=point7, face_normal=(0.0, 1.0, 0.0)))
#           geometries.append(face(vertex_a=point3, vertex_b=point2, vertex_c=point6, vertex_d=point7, face_normal=(0.0, 1.0, 0.0)))
            geometries.append(face(vertex_a=point4, vertex_b=point5, vertex_c=point1, vertex_d=point0, face_normal=(0.0, -1.0, 0.0)))
#           geometries.append(face(vertex_a=point4, vertex_b=point5, vertex_c=point1, vertex_d=point0, face_normal=(0.0, -1.0, 0.0)))

            vbo_mesh: mgl.Buffer = self.ctx.buffer(np.concatenate(geometries).tobytes())
#           vbo_mesh: mgl.Buffer = self.ctx.buffer(np.concatenate(geometries).tobytes())

            # 2. Instances
            # 2. Instances
            instance_bytes: bytes = np.concatenate(cube_instance_data).tobytes()
#           instance_bytes: bytes = np.concatenate(cube_instance_data).tobytes()
            vbo_instances: mgl.Buffer = self.ctx.buffer(instance_bytes)
#           vbo_instances: mgl.Buffer = self.ctx.buffer(instance_bytes)

            vao_cube: mgl.VertexArray = self.ctx.vertex_array(
#           vao_cube: mgl.VertexArray = self.ctx.vertex_array(
                self.program_geometry,
#               self.program_geometry,
                [
#               [
                    (vbo_mesh, "3f 3f", "inVertexLocalPosition", "inVertexLocalNormal"),
#                   (vbo_mesh, "3f 3f", "inVertexLocalPosition", "inVertexLocalNormal"),
                    (vbo_instances, "16f 3f/i", "inInstanceTransformModel", "inInstanceAlbedo"),
#                   (vbo_instances, "16f 3f/i", "inInstanceTransformModel", "inInstanceAlbedo"),
                ],
#               ],
            )
#           )
            self.scene_batches.append(SceneBatch(vao=vao_cube, number_of_instances=len(cube_instance_data)))
#           self.scene_batches.append(SceneBatch(vao=vao_cube, number_of_instances=len(cube_instance_data)))

        # Create Plane Batch
        # Create Plane Batch
        if plane_instance_data:
#       if plane_instance_data:
            # 1. Geometry
            # 1. Geometry
            point0: vec3f32 = (-0.5, 0.0, 0.5)
#           point0: vec3f32 = (-0.5, 0.0, 0.5)
            point1: vec3f32 = (0.5, 0.0, 0.5)
#           point1: vec3f32 = (0.5, 0.0, 0.5)
            point2: vec3f32 = (0.5, 0.0, -0.5)
#           point2: vec3f32 = (0.5, 0.0, -0.5)
            point3: vec3f32 = (-0.5, 0.0, -0.5)
#           point3: vec3f32 = (-0.5, 0.0, -0.5)

            geometries: list = []
#           geometries: list = []
            geometries.append(face(vertex_a=point0, vertex_b=point1, vertex_c=point2, vertex_d=point3, face_normal=(0.0, 1.0, 0.0)))
#           geometries.append(face(vertex_a=point0, vertex_b=point1, vertex_c=point2, vertex_d=point3, face_normal=(0.0, 1.0, 0.0)))

            vbo_mesh: mgl.Buffer = self.ctx.buffer(np.concatenate(geometries).tobytes())
#           vbo_mesh: mgl.Buffer = self.ctx.buffer(np.concatenate(geometries).tobytes())

            # 2. Instances
            # 2. Instances
            instance_bytes: bytes = np.concatenate(plane_instance_data).tobytes()
#           instance_bytes: bytes = np.concatenate(plane_instance_data).tobytes()
            vbo_instances: mgl.Buffer = self.ctx.buffer(instance_bytes)
#           vbo_instances: mgl.Buffer = self.ctx.buffer(instance_bytes)

            vao_plane: mgl.VertexArray = self.ctx.vertex_array(
#           vao_plane: mgl.VertexArray = self.ctx.vertex_array(
                self.program_geometry,
#               self.program_geometry,
                [
#               [
                    (vbo_mesh, "3f 3f", "inVertexLocalPosition", "inVertexLocalNormal"),
#                   (vbo_mesh, "3f 3f", "inVertexLocalPosition", "inVertexLocalNormal"),
                    (vbo_instances, "16f 3f/i", "inInstanceTransformModel", "inInstanceAlbedo"),
#                   (vbo_instances, "16f 3f/i", "inInstanceTransformModel", "inInstanceAlbedo"),
                ],
#               ],
            )
#           )
            self.scene_batches.append(SceneBatch(vao=vao_plane, number_of_instances=len(plane_instance_data)))
#           self.scene_batches.append(SceneBatch(vao=vao_plane, number_of_instances=len(plane_instance_data)))

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

        self.program_geometry["uTransformView"].write(self.transform_view.astype("f4").tobytes())
#       self.program_geometry["uTransformView"].write(self.transform_view.astype("f4").tobytes())
        self.program_geometry["uTransformProjection"].write(self.transform_projection.astype("f4").tobytes())
#       self.program_geometry["uTransformProjection"].write(self.transform_projection.astype("f4").tobytes())

        for scene_batch in self.scene_batches:
#       for scene_batch in self.scene_batches:
            scene_batch.vao.render(instances=scene_batch.number_of_instances)
#           scene_batch.vao.render(instances=scene_batch.number_of_instances)

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
