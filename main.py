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
import re
import re
import typing
import typing
import bvh
import bvh

type vec2f32 = tuple[float, float]
type vec3f32 = tuple[float, float, float]
type vec4f32 = tuple[float, float, float, float]

def resolve_includes(source: str, base_path: pl.Path) -> str:
    """
    Recursively resolves #include "filename" directives in GLSL source code.
    Recursively resolves #include "filename" directives in GLSL source code.
    """
    # Match: #include "filename" (handling optional whitespace)
    # Match: #include "filename" (handling optional whitespace)
    pattern: re.Pattern[str] = re.compile(pattern=r'^\s*#include\s+"([^"]+)"', flags=re.MULTILINE)
#   pattern: re.Pattern[str] = re.compile(pattern=r'^\s*#include\s+"([^"]+)"', flags=re.MULTILINE)

    def replace(match: re.Match[str]) -> str:
#   def replace(match: re.Match[str]) -> str:
        filename: str | typing.Any = match.group(1)
#       filename: str | typing.Any = match.group(1)
        included_path: pl.Path | typing.Any = base_path / filename
#       included_path: pl.Path | typing.Any = base_path / filename

        if not included_path.exists():
#       if not included_path.exists():
            # You might want to raise an error or just warn
            # You might want to raise an error or just warn
            print(f"Warning: Included file not found: {included_path}")
#           print(f"Warning: Included file not found: {included_path}")
            return f"// ERROR: Include not found {filename}\n"
#           return f"// ERROR: Include not found {filename}\n"

        included_content: str | typing.Any = included_path.read_text(encoding="utf-8")
#       included_content: str | typing.Any = included_path.read_text(encoding="utf-8")
        # Recursively resolve includes within the included file
        # Recursively resolve includes within the included file
        return resolve_includes(source=included_content, base_path=base_path)
#       return resolve_includes(source=included_content, base_path=base_path)

    return pattern.sub(replace, source)
#   return pattern.sub(replace, source)

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

        self.texture_accum: mgl.Texture = self.ctx.texture(size=self.window_size, components=4, dtype="f4")
#       self.texture_accum: mgl.Texture = self.ctx.texture(size=self.window_size, components=4, dtype="f4")
        self.texture_accum.filter = (mgl.NEAREST, mgl.NEAREST)
#       self.texture_accum.filter = (mgl.NEAREST, mgl.NEAREST)

        self.texture_ping: mgl.Texture = self.ctx.texture(size=self.window_size, components=4, dtype="f4")
#       self.texture_ping: mgl.Texture = self.ctx.texture(size=self.window_size, components=4, dtype="f4")
        self.texture_ping.filter = (mgl.NEAREST, mgl.NEAREST)
#       self.texture_ping.filter = (mgl.NEAREST, mgl.NEAREST)

        # -----------------------------
        # 2. Rasterization Shader (Geometry Pass)
        # 2. Rasterization Shader (Geometry Pass)
        # -----------------------------
        hybrid_geometry_vs_path: pl.Path = self.resource_dir / "hybrid_geometry_vs.glsl"
#       hybrid_geometry_vs_path: pl.Path = self.resource_dir / "hybrid_geometry_vs.glsl"
        hybrid_geometry_vs_code: str = resolve_includes(hybrid_geometry_vs_path.read_text(encoding="utf-8"), self.resource_dir)
#       hybrid_geometry_vs_code: str = resolve_includes(hybrid_geometry_vs_path.read_text(encoding="utf-8"), self.resource_dir)
        hybrid_geometry_fs_path: pl.Path = self.resource_dir / "hybrid_geometry_fs.glsl"
#       hybrid_geometry_fs_path: pl.Path = self.resource_dir / "hybrid_geometry_fs.glsl"
        hybrid_geometry_fs_code: str = resolve_includes(hybrid_geometry_fs_path.read_text(encoding="utf-8"), self.resource_dir)
#       hybrid_geometry_fs_code: str = resolve_includes(hybrid_geometry_fs_path.read_text(encoding="utf-8"), self.resource_dir)
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
        hybrid_shading_cs_code: str = resolve_includes(hybrid_shading_cs_path.read_text(encoding="utf-8"), self.resource_dir)
#       hybrid_shading_cs_code: str = resolve_includes(hybrid_shading_cs_path.read_text(encoding="utf-8"), self.resource_dir)
        self.program_shading: mgl.ComputeShader = self.ctx.compute_shader(source=hybrid_shading_cs_code)
#       self.program_shading: mgl.ComputeShader = self.ctx.compute_shader(source=hybrid_shading_cs_code)

        # Denoise Shader
        # Denoise Shader
        hybrid_denoise_cs_path: pl.Path = self.resource_dir / "hybrid_denoise_cs.glsl"
#       hybrid_denoise_cs_path: pl.Path = self.resource_dir / "hybrid_denoise_cs.glsl"
        hybrid_denoise_cs_code: str = resolve_includes(hybrid_denoise_cs_path.read_text(encoding="utf-8"), self.resource_dir)
#       hybrid_denoise_cs_code: str = resolve_includes(hybrid_denoise_cs_path.read_text(encoding="utf-8"), self.resource_dir)
        self.program_denoise: mgl.ComputeShader = self.ctx.compute_shader(source=hybrid_denoise_cs_code)
#       self.program_denoise: mgl.ComputeShader = self.ctx.compute_shader(source=hybrid_denoise_cs_code)

        # -----------------------------
        # 4. Rasterization Shader (Renderer Pass)
        # 4. Rasterization Shader (Renderer Pass)
        # -----------------------------
        hybrid_renderer_vs_path: pl.Path = self.resource_dir / "hybrid_renderer_vs.glsl"
#       hybrid_renderer_vs_path: pl.Path = self.resource_dir / "hybrid_renderer_vs.glsl"
        hybrid_renderer_vs_code: str = resolve_includes(hybrid_renderer_vs_path.read_text(encoding="utf-8"), self.resource_dir)
#       hybrid_renderer_vs_code: str = resolve_includes(hybrid_renderer_vs_path.read_text(encoding="utf-8"), self.resource_dir)
        hybrid_renderer_fs_path: pl.Path = self.resource_dir / "hybrid_renderer_fs.glsl"
#       hybrid_renderer_fs_path: pl.Path = self.resource_dir / "hybrid_renderer_fs.glsl"
        hybrid_renderer_fs_code: str = resolve_includes(hybrid_renderer_fs_path.read_text(encoding="utf-8"), self.resource_dir)
#       hybrid_renderer_fs_code: str = resolve_includes(hybrid_renderer_fs_path.read_text(encoding="utf-8"), self.resource_dir)
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
        def face(vertex_a: vec3f32, vertex_b: vec3f32, vertex_c: vec3f32, vertex_d: vec3f32, face_normal: vec3f32) -> npt.NDArray[np.float32]:
#       def face(vertex_a: vec3f32, vertex_b: vec3f32, vertex_c: vec3f32, vertex_d: vec3f32, face_normal: vec3f32) -> npt.NDArray[np.float32]:
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

        cube_instance_data: list[npt.NDArray[np.float32]] = []
#       cube_instance_data: list[npt.NDArray[np.float32]] = []
        plane_instance_data: list[npt.NDArray[np.float32]] = []
#       plane_instance_data: list[npt.NDArray[np.float32]] = []

        def add_cube(position: vec3f32, rotation: vec3f32, scale: vec3f32, color: vec3f32) -> None:
#       def add_cube(position: vec3f32, rotation: vec3f32, scale: vec3f32, color: vec3f32) -> None:
            matrix_translation: rr.Matrix44 = rr.Matrix44.from_translation(position)
#           matrix_translation: rr.Matrix44 = rr.Matrix44.from_translation(position)
            matrix_rotation: rr.Matrix44 = rr.Matrix44.from_eulers(rotation)
#           matrix_rotation: rr.Matrix44 = rr.Matrix44.from_eulers(rotation)
            matrix_scale: rr.Matrix44 = rr.Matrix44.from_scale(scale)
#           matrix_scale: rr.Matrix44 = rr.Matrix44.from_scale(scale)
            matrix: npt.NDArray[np.float32] = (matrix_translation * matrix_rotation * matrix_scale).astype("f4")
#           matrix: npt.NDArray[np.float32] = (matrix_translation * matrix_rotation * matrix_scale).astype("f4")
            data: npt.NDArray[np.float32] = np.concatenate([matrix.flatten(), color]).astype("f4")
#           data: npt.NDArray[np.float32] = np.concatenate([matrix.flatten(), color]).astype("f4")
            cube_instance_data.append(data)
#           cube_instance_data.append(data)

        def add_plane(position: vec3f32, rotation: vec3f32, scale: vec3f32, color: vec3f32) -> None:
#       def add_plane(position: vec3f32, rotation: vec3f32, scale: vec3f32, color: vec3f32) -> None:
            matrix_translation: rr.Matrix44 = rr.Matrix44.from_translation(position)
#           matrix_translation: rr.Matrix44 = rr.Matrix44.from_translation(position)
            matrix_rotation: rr.Matrix44 = rr.Matrix44.from_eulers(rotation)
#           matrix_rotation: rr.Matrix44 = rr.Matrix44.from_eulers(rotation)
            matrix_scale: rr.Matrix44 = rr.Matrix44.from_scale(scale)
#           matrix_scale: rr.Matrix44 = rr.Matrix44.from_scale(scale)
            matrix: npt.NDArray[np.float32] = (matrix_translation * matrix_rotation * matrix_scale).astype("f4")
#           matrix: npt.NDArray[np.float32] = (matrix_translation * matrix_rotation * matrix_scale).astype("f4")
            data: npt.NDArray[np.float32] = np.concatenate([matrix.flatten(), color]).astype("f4")
#           data: npt.NDArray[np.float32] = np.concatenate([matrix.flatten(), color]).astype("f4")
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

        # -----------------------------
        # 6. BVH Construction
        # 6. BVH Construction
        # -----------------------------
        # We need to gather all world-space triangles to build the BVH.
        # We need to gather all world-space triangles to build the BVH.
        scene_triangles: list[npt.NDArray[np.float32]] = []
#       scene_triangles: list[npt.NDArray[np.float32]] = []

        # Helper to transform triangles
        # Helper to transform triangles
        def append_transformed_triangles(
#       def append_transformed_triangles(
            instance_data_list: list[npt.NDArray[np.float32]],
#           instance_data_list: list[npt.NDArray[np.float32]],
            base_triangles: list[npt.NDArray[np.float32]]
#           base_triangles: list[npt.NDArray[np.float32]]
        ) -> None:
#       ) -> None:
            for instance_data in instance_data_list:
#           for instance_data in instance_data_list:
                # Extract 4x4 model matrix (first 16 floats)
                # Extract 4x4 model matrix (first 16 floats)
                model_matrix_flat: npt.NDArray[np.float32] = instance_data[:16]
#               model_matrix_flat: npt.NDArray[np.float32] = instance_data[:16]
                model_matrix: npt.NDArray[np.float32] = model_matrix_flat.reshape((4, 4), order='F')
#               model_matrix: npt.NDArray[np.float32] = model_matrix_flat.reshape((4, 4), order='F')

                # Transform each triangle
                # Transform each triangle
                for tri_verts in base_triangles:
#               for tri_verts in base_triangles:
                    # tri_verts shape: (3, 3) -> (v0, v1, v2)
                    # tri_verts shape: (3, 3) -> (v0, v1, v2)
                    # Convert to homogeneous coordinates (3, 4)
                    # Convert to homogeneous coordinates (3, 4)
                    ones: npt.NDArray[np.float32] = np.ones((3, 1), dtype="f4")
#                   ones: npt.NDArray[np.float32] = np.ones((3, 1), dtype="f4")
                    verts_h: npt.NDArray[np.float32] = np.hstack([tri_verts, ones])
#                   verts_h: npt.NDArray[np.float32] = np.hstack([tri_verts, ones])

                    # Apply transformation: v_world = M * v_local
                    # Apply transformation: v_world = M * v_local
                    # Since our vectors are rows in numpy (N, 4), we do v @ M.T
                    # Since our vectors are rows in numpy (N, 4), we do v @ M.T
                    # Or using pyrr logic, matrix multiplication.
                    # Or using pyrr logic, matrix multiplication.
                    # Note: moderngl/glsl matrices are column-major, but when flattened and read into numpy
                    # Note: moderngl/glsl matrices are column-major, but when flattened and read into numpy
                    # as (4,4), the layout depends on how it was constructed.
                    # as (4,4), the layout depends on how it was constructed.
                    # Pyrr matrices are row-major in memory if used as numpy arrays? No, usually column-major logic.
                    # Pyrr matrices are row-major in memory if used as numpy arrays? No, usually column-major logic.
                    # Let's rely on matrix multiplication `matrix @ vector` where vector is column.
                    # Let's rely on matrix multiplication `matrix @ vector` where vector is column.

                    # Transpose to (4, 3) to multiply: M @ V_T
                    # Transpose to (4, 3) to multiply: M @ V_T
                    transformed_verts_h: npt.NDArray[np.float32] = model_matrix @ verts_h.T
#                   transformed_verts_h: npt.NDArray[np.float32] = model_matrix @ verts_h.T

                    # Transpose back to (3, 4)
                    # Transpose back to (3, 4)
                    transformed_verts_h = transformed_verts_h.T
#                   transformed_verts_h = transformed_verts_h.T

                    # Extract xyz
                    # Extract xyz
                    transformed_tri: npt.NDArray[np.float32] = transformed_verts_h[:, :3]
#                   transformed_tri: npt.NDArray[np.float32] = transformed_verts_h[:, :3]
                    scene_triangles.append(transformed_tri.copy())
#                   scene_triangles.append(transformed_tri.copy())


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

            # Base Cube Triangles (Local Space)
            # Base Cube Triangles (Local Space)
            cube_base_triangles: list[npt.NDArray[np.float32]] = []
#           cube_base_triangles: list[npt.NDArray[np.float32]] = []

            # Helper to just return the triangle vertices (3, 3)
            # Helper to just return the triangle vertices (3, 3)
            def get_tri_verts(v0, v1, v2) -> npt.NDArray[np.float32]:
#           def get_tri_verts(v0, v1, v2) -> npt.NDArray[np.float32]:
                return np.array([v0, v1, v2], dtype="f4")
#               return np.array([v0, v1, v2], dtype="f4")

            # Front
            # Front
            cube_base_triangles.append(get_tri_verts(point0, point1, point2))
#           cube_base_triangles.append(get_tri_verts(point0, point1, point2))
            cube_base_triangles.append(get_tri_verts(point0, point2, point3))
#           cube_base_triangles.append(get_tri_verts(point0, point2, point3))
            # Back
            # Back
            cube_base_triangles.append(get_tri_verts(point5, point4, point7))
#           cube_base_triangles.append(get_tri_verts(point5, point4, point7))
            cube_base_triangles.append(get_tri_verts(point5, point7, point6))
#           cube_base_triangles.append(get_tri_verts(point5, point7, point6))
            # Left
            # Left
            cube_base_triangles.append(get_tri_verts(point4, point0, point3))
#           cube_base_triangles.append(get_tri_verts(point4, point0, point3))
            cube_base_triangles.append(get_tri_verts(point4, point3, point7))
#           cube_base_triangles.append(get_tri_verts(point4, point3, point7))
            # Right
            # Right
            cube_base_triangles.append(get_tri_verts(point1, point5, point6))
#           cube_base_triangles.append(get_tri_verts(point1, point5, point6))
            cube_base_triangles.append(get_tri_verts(point1, point6, point2))
#           cube_base_triangles.append(get_tri_verts(point1, point6, point2))
            # Top
            # Top
            cube_base_triangles.append(get_tri_verts(point3, point2, point6))
#           cube_base_triangles.append(get_tri_verts(point3, point2, point6))
            cube_base_triangles.append(get_tri_verts(point3, point6, point7))
#           cube_base_triangles.append(get_tri_verts(point3, point6, point7))
            # Bottom
            # Bottom
            cube_base_triangles.append(get_tri_verts(point4, point5, point1))
#           cube_base_triangles.append(get_tri_verts(point4, point5, point1))
            cube_base_triangles.append(get_tri_verts(point4, point1, point0))
#           cube_base_triangles.append(get_tri_verts(point4, point1, point0))

            append_transformed_triangles(cube_instance_data, cube_base_triangles)
#           append_transformed_triangles(cube_instance_data, cube_base_triangles)

            geometries: list[npt.NDArray[np.float32]] = []
#           geometries: list[npt.NDArray[np.float32]] = []
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

            # Base Plane Triangles
            # Base Plane Triangles
            plane_base_triangles: list[npt.NDArray[np.float32]] = []
#           plane_base_triangles: list[npt.NDArray[np.float32]] = []

            # Helper duplicate
            # Helper duplicate
            def get_tri_verts(v0, v1, v2) -> npt.NDArray[np.float32]:
#           def get_tri_verts(v0, v1, v2) -> npt.NDArray[np.float32]:
                return np.array([v0, v1, v2], dtype="f4")
#               return np.array([v0, v1, v2], dtype="f4")

            # Triangle 1
            # Triangle 1
            plane_base_triangles.append(get_tri_verts(point0, point1, point2))
#           plane_base_triangles.append(get_tri_verts(point0, point1, point2))
            # Triangle 2
            # Triangle 2
            plane_base_triangles.append(get_tri_verts(point0, point2, point3))
#           plane_base_triangles.append(get_tri_verts(point0, point2, point3))

            append_transformed_triangles(plane_instance_data, plane_base_triangles)
#           append_transformed_triangles(plane_instance_data, plane_base_triangles)

            geometries: list[npt.NDArray[np.float32]] = []
#           geometries: list[npt.NDArray[np.float32]] = []
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

        # Build BVH
        # Build BVH
        world_triangles: npt.NDArray[np.float32] = np.array(scene_triangles, dtype="f4")
#       world_triangles: npt.NDArray[np.float32] = np.array(scene_triangles, dtype="f4")
        self.bvh: bvh.LBVH = bvh.LBVH(world_triangles)
#       self.bvh: bvh.LBVH = bvh.LBVH(world_triangles)
        bvh_data: bytes = self.bvh.simple_build()
#       bvh_data: bytes = self.bvh.simple_build()

        # Upload to SSBOs
        # Upload to SSBOs
        # 1. Nodes (already bytes)
        # 1. Nodes (already bytes)
        self.ssbo_bvh_nodes: mgl.Buffer = self.ctx.buffer(data=bvh_data)
#       self.ssbo_bvh_nodes: mgl.Buffer = self.ctx.buffer(data=bvh_data)

        # 2. Triangles (flattened world space)
        # 2. Triangles (flattened world space)
        # We need to flatten (N, 3, 3) -> (N * 9) floats.
        # We need to flatten (N, 3, 3) -> (N * 9) floats.
        # Ensure it is float32
        # Ensure it is float32
        self.ssbo_triangles: mgl.Buffer = self.ctx.buffer(data=world_triangles.flatten().tobytes())
#       self.ssbo_triangles: mgl.Buffer = self.ctx.buffer(data=world_triangles.flatten().tobytes())

        self.ctx.enable(mgl.DEPTH_TEST | mgl.CULL_FACE)
#       self.ctx.enable(mgl.DEPTH_TEST | mgl.CULL_FACE)

        # Camera
        # Camera
        self.camera_global_position: rr.Vector3 = rr.Vector3([3.0, 3.0, 3.0])
#       self.camera_global_position: rr.Vector3 = rr.Vector3([3.0, 3.0, 3.0])
        self.transform_projection: rr.Matrix44 = rr.Matrix44.perspective_projection(
#       self.transform_projection: rr.Matrix44 = rr.Matrix44.perspective_projection(
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
        self.frame_count += 1
#       self.frame_count += 1

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

        # Bind BVH buffers
        # Bind BVH buffers
        self.ssbo_bvh_nodes.bind_to_storage_buffer(binding=4)
#       self.ssbo_bvh_nodes.bind_to_storage_buffer(binding=4)
        self.ssbo_triangles.bind_to_storage_buffer(binding=5)
#       self.ssbo_triangles.bind_to_storage_buffer(binding=5)

        self.texture_accum.bind_to_image(6, read=True, write=True)
#       self.texture_accum.bind_to_image(6, read=True, write=True)

        # Update Uniforms
        # Update Uniforms
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
            self.program_shading["uCameraGlobalPosition"] = tuple(self.camera_global_position)
#           self.program_shading["uCameraGlobalPosition"] = tuple(self.camera_global_position)
        if "uFrameCount" in self.program_shading:
#       if "uFrameCount" in self.program_shading:
            self.program_shading["uFrameCount"] = self.frame_count
#           self.program_shading["uFrameCount"] = self.frame_count

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

        # 3. Denoise (Multi-Pass A-Trous)
        # 3. Denoise (Multi-Pass A-Trous)

        # Pass 1: Accum -> Ping (Step 1)
        # Pass 1: Accum -> Ping (Step 1)
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
        # Pass 2: Ping -> Output (Step 2)
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
        # Pass 3: Output -> Ping (Step 4)
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
        # Pass 4: Ping -> Output (Step 8, Final Tone Map)
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
        # 4. Display Result to Screen
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