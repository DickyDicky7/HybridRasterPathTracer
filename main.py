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
import os
import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
import cv2
import cv2

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

        # HDRI Texture Loading
        # HDRI Texture Loading
        hdri_path: pl.Path = self.resource_dir / "rostock_laage_airport_4k.exr"
#       hdri_path: pl.Path = self.resource_dir / "rostock_laage_airport_4k.exr"
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
            self.texture_hdri.filter = (mgl.LINEAR, mgl.LINEAR)
#           self.texture_hdri.filter = (mgl.LINEAR, mgl.LINEAR)
            self.texture_hdri.repeat_x = True
#           self.texture_hdri.repeat_x = True
            self.texture_hdri.repeat_y = False
#           self.texture_hdri.repeat_y = False
        else:
#       else:
            # Create dummy 1x1 texture
            # Create dummy 1x1 texture
            self.texture_hdri: mgl.Texture = self.ctx.texture((1, 1), components=3, dtype="f4", data=np.zeros(3, dtype="f4").tobytes())
#           self.texture_hdri: mgl.Texture = self.ctx.texture((1, 1), components=3, dtype="f4", data=np.zeros(3, dtype="f4").tobytes())

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
            def __init__(self, vao: mgl.VertexArray, number_of_instances: int, triangle_count_per_instance: int) -> None:
#           def __init__(self, vao: mgl.VertexArray, number_of_instances: int, triangle_count_per_instance: int) -> None:
                self.vao: mgl.VertexArray = vao
#               self.vao: mgl.VertexArray = vao
                self.number_of_instances: int = number_of_instances
#               self.number_of_instances: int = number_of_instances
                self.triangle_count_per_instance: int = triangle_count_per_instance
#               self.triangle_count_per_instance: int = triangle_count_per_instance
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

        add_plane(position=(0.0, -0.5, 0.0), rotation=(0.0, 0.0, 0.0), scale=(20.0, 1.0, 20.0), color=(0.3, 0.3, 0.3)) # Gray Plane
#       add_plane(position=(0.0, -0.5, 0.0), rotation=(0.0, 0.0, 0.0), scale=(20.0, 1.0, 20.0), color=(0.3, 0.3, 0.3)) # Gray Plane

        # -----------------------------
        # 6. BVH Construction
        # 6. BVH Construction
        # -----------------------------
        # We need to gather all world-space triangles to build the BVH.
        # We need to gather all world-space triangles to build the BVH.
        scene_triangles: list[npt.NDArray[np.float32]] = []
#       scene_triangles: list[npt.NDArray[np.float32]] = []
        scene_materials: list[npt.NDArray[np.float32]] = []
#       scene_materials: list[npt.NDArray[np.float32]] = []

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

                # Extract Color
                # Extract Color
                color: npt.NDArray[np.float32] = instance_data[16:19]
#               color: npt.NDArray[np.float32] = instance_data[16:19]

                # Material Parameters (Default Principled)
                # Material Parameters (Default Principled)
                # struct Material {
                # struct Material {
                #     vec4 albedo; // .w unused
                #     vec4 albedo; // .w unused
                #     float roughness;
                #     float roughness;
                #     float metallic;
                #     float metallic;
                #     float transmission;
                #     float transmission;
                #     float ior;
                #     float ior;
                # };
                # };
                # Layout: [r, g, b, padding, roughness, metallic, transmission, ior]
#               # Layout: [r, g, b, padding, roughness, metallic, transmission, ior]
                material_data = np.array([
#               material_data = np.array([
                    color[0], color[1], color[2], 0.0, # Albedo + Padding
#                   color[0], color[1], color[2], 0.0, # Albedo + Padding
                    0.0,                               # Roughness
#                   0.0,                               # Roughness
                    1.0,                               # Metallic
#                   1.0,                               # Metallic
                    0.0,                               # Transmission
#                   0.0,                               # Transmission
                    1.45                               # IOR
#                   1.45                               # IOR
                ], dtype="f4")
#               ], dtype="f4")

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
                    scene_materials.append(material_data)
#                   scene_materials.append(material_data)


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
            self.scene_batches.append(SceneBatch(vao=vao_cube, number_of_instances=len(cube_instance_data), triangle_count_per_instance=12))
#           self.scene_batches.append(SceneBatch(vao=vao_cube, number_of_instances=len(cube_instance_data), triangle_count_per_instance=12))

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
            self.scene_batches.append(SceneBatch(vao=vao_plane, number_of_instances=len(plane_instance_data), triangle_count_per_instance=2))
#           self.scene_batches.append(SceneBatch(vao=vao_plane, number_of_instances=len(plane_instance_data), triangle_count_per_instance=2))

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

        # 3. Materials (Albedos)
        # 3. Materials (Albedos)
        world_materials: npt.NDArray[np.float32] = np.array(scene_materials, dtype="f4")
#       world_materials: npt.NDArray[np.float32] = np.array(scene_materials, dtype="f4")
        # Flatten? It is list of (8,) arrays, so (N, 8). Flatten to N*8 floats.
        # Flatten? It is list of (8,) arrays, so (N, 8). Flatten to N*8 floats.
        self.ssbo_materials: mgl.Buffer = self.ctx.buffer(data=world_materials.flatten().tobytes())
#       self.ssbo_materials: mgl.Buffer = self.ctx.buffer(data=world_materials.flatten().tobytes())

        self.ctx.enable(mgl.DEPTH_TEST | mgl.CULL_FACE)
#       self.ctx.enable(mgl.DEPTH_TEST | mgl.CULL_FACE)

        # Camera
        # Camera
        self.camera_look_from: rr.Vector3 = rr.Vector3([-28.284, 0.0, -28.284])
#       self.camera_look_from: rr.Vector3 = rr.Vector3([-28.284, 0.0, -28.284])
        self.camera_look_at: rr.Vector3 = rr.Vector3([0.0, 0.0, 0.0])
#       self.camera_look_at: rr.Vector3 = rr.Vector3([0.0, 0.0, 0.0])
        self.camera_view_up: rr.Vector3 = rr.Vector3([0.0, 1.0, 0.0])
#       self.camera_view_up: rr.Vector3 = rr.Vector3([0.0, 1.0, 0.0])

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
        self.movement_speed: float = 10.0
#       self.movement_speed: float = 10.0
        self.camera_yaw: float = 0.0
#       self.camera_yaw: float = 0.0
        self.camera_pitch: float = 0.0
#       self.camera_pitch: float = 0.0

        # Initialize yaw/pitch from look_at - look_from
        # Initialize yaw/pitch from look_at - look_from
        direction: rr.Vector3 = rr.vector.normalize(self.camera_look_at - self.camera_look_from)
#       direction: rr.Vector3 = rr.vector.normalize(self.camera_look_at - self.camera_look_from)
        self.camera_yaw = np.arctan2(direction[2], direction[0])
#       self.camera_yaw = np.arctan2(direction[2], direction[0])
        self.camera_pitch = np.arcsin(direction[1])
#       self.camera_pitch = np.arcsin(direction[1])

        # Base Projection (No Jitter)
        # Base Projection (No Jitter)
        self.base_projection: rr.Matrix44 = rr.Matrix44.perspective_projection(
#       self.base_projection: rr.Matrix44 = rr.Matrix44.perspective_projection(
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
        self.transform_projection = self.base_projection # Initial
#       self.transform_projection = self.base_projection # Initial

        self.transform_view: rr.Matrix44 = rr.Matrix44.look_at(
#       self.transform_view: rr.Matrix44 = rr.Matrix44.look_at(
            eye=self.camera_look_from,
#           eye=self.camera_look_from,
            target=self.camera_look_at,
#           target=self.camera_look_at,
            up=self.camera_view_up,
#           up=self.camera_view_up,
        )
#       )
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
        # Poll keys (Backup for key_event)
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

        # Rotation from Keys
        # Rotation from Keys
        rotation_speed: float = 2.0 * frame_time
#       rotation_speed: float = 2.0 * frame_time
        if self.key_state["LEFT"]: self.camera_yaw -= rotation_speed
#       if self.key_state["LEFT"]: self.camera_yaw -= rotation_speed
        if self.key_state["RIGHT"]: self.camera_yaw += rotation_speed
#       if self.key_state["RIGHT"]: self.camera_yaw += rotation_speed
        if self.key_state["UP"]: self.camera_pitch += rotation_speed
#       if self.key_state["UP"]: self.camera_pitch += rotation_speed
        if self.key_state["DOWN"]: self.camera_pitch -= rotation_speed
#       if self.key_state["DOWN"]: self.camera_pitch -= rotation_speed
        self.camera_pitch = max(-np.pi/2 + 0.1, min(np.pi/2 - 0.1, self.camera_pitch))
#       self.camera_pitch = max(-np.pi/2 + 0.1, min(np.pi/2 - 0.1, self.camera_pitch))

        # Direction from Yaw/Pitch
        # Direction from Yaw/Pitch
        direction: rr.Vector3 = rr.Vector3([
#       direction: rr.Vector3 = rr.Vector3([
            np.cos(self.camera_yaw) * np.cos(self.camera_pitch),
#           np.cos(self.camera_yaw) * np.cos(self.camera_pitch),
            np.sin(self.camera_pitch),
#           np.sin(self.camera_pitch),
            np.sin(self.camera_yaw) * np.cos(self.camera_pitch)
#           np.sin(self.camera_yaw) * np.cos(self.camera_pitch)
        ])
#       ])

        forward: rr.Vector3 = rr.vector.normalize(direction)
#       forward: rr.Vector3 = rr.vector.normalize(direction)
        right: rr.Vector3 = rr.vector.normalize(rr.vector3.cross(forward, self.camera_view_up))
#       right: rr.Vector3 = rr.vector.normalize(rr.vector3.cross(forward, self.camera_view_up))
        # up: rr.Vector3 = rr.vector.normalize(rr.vector3.cross(right, forward))
#       # up: rr.Vector3 = rr.vector.normalize(rr.vector3.cross(right, forward))

        # Movement
        # Movement
        velocity: float = self.movement_speed * frame_time
#       velocity: float = self.movement_speed * frame_time
        if self.key_state["W"]: self.camera_look_from += forward * velocity
#       if self.key_state["W"]: self.camera_look_from += forward * velocity
        if self.key_state["S"]: self.camera_look_from -= forward * velocity
#       if self.key_state["S"]: self.camera_look_from -= forward * velocity
        if self.key_state["A"]: self.camera_look_from -= right * velocity
#       if self.key_state["A"]: self.camera_look_from -= right * velocity
        if self.key_state["D"]: self.camera_look_from += right * velocity
#       if self.key_state["D"]: self.camera_look_from += right * velocity
        if self.key_state["Q"]: self.camera_look_from += rr.Vector3([0.0, 1.0, 0.0]) * velocity
#       if self.key_state["Q"]: self.camera_look_from += rr.Vector3([0.0, 1.0, 0.0]) * velocity
        if self.key_state["E"]: self.camera_look_from -= rr.Vector3([0.0, 1.0, 0.0]) * velocity
#       if self.key_state["E"]: self.camera_look_from -= rr.Vector3([0.0, 1.0, 0.0]) * velocity

        # Re-calculate LookAt for Matrix
        # Re-calculate LookAt for Matrix
        self.camera_look_at = self.camera_look_from + forward
#       self.camera_look_at = self.camera_look_from + forward

        self.transform_view = rr.Matrix44.look_at(
#       self.transform_view = rr.Matrix44.look_at(
            eye=self.camera_look_from,
#           eye=self.camera_look_from,
            target=self.camera_look_at,
#           target=self.camera_look_at,
            up=self.camera_view_up
#           up=self.camera_view_up
        )
#       )

        # TAA Jitter
        # TAA Jitter
        # Halton sequence for x (base 2) and y (base 3)
        # Halton sequence for x (base 2) and y (base 3)
        # Scale to [-0.5, 0.5] pixels
        # Scale to [-0.5, 0.5] pixels
        jitter_x = (self.get_halton_jitter(index=self.frame_count, base=2) - 0.5)
#       jitter_x = (self.get_halton_jitter(index=self.frame_count, base=2) - 0.5)
        jitter_y = (self.get_halton_jitter(index=self.frame_count, base=3) - 0.5)
#       jitter_y = (self.get_halton_jitter(index=self.frame_count, base=3) - 0.5)

        # Convert pixel offset to clip space offset
        # Convert pixel offset to clip space offset
        # Clip space is [-1, 1], size is 2.0. Pixel size is 2.0 / resolution.
        # Clip space is [-1, 1], size is 2.0. Pixel size is 2.0 / resolution.
        w, h = self.window_size
#       w, h = self.window_size
        jitter_clip_x: float = (jitter_x * 2.0) / w
#       jitter_clip_x: float = (jitter_x * 2.0) / w
        jitter_clip_y: float = (jitter_y * 2.0) / h
#       jitter_clip_y: float = (jitter_y * 2.0) / h

        # Apply jitter to projection matrix
        # Apply jitter to projection matrix
        # Projection matrix layout (column-major logic in memory?):
        # Projection matrix layout (column-major logic in memory?):
        # [0][0]  [1][0]  [2][0]  [3][0]
        # [0][0]  [1][0]  [2][0]  [3][0]
        # ...     ...     [2][2]  [3][2]
        # ...     ...     [2][2]  [3][2]
        # ...     ...     ...     ...
        # ...     ...     ...     ...
        # Pyrr Matrix44 is basically a numpy array. Access might be row-major in numpy wrapper.
        # Pyrr Matrix44 is basically a numpy array. Access might be row-major in numpy wrapper.
        # Standard Perspective Matrix:
        # Standard Perspective Matrix:
        # X 0 A 0
        # X 0 A 0
        # 0 Y B 0
        # 0 Y B 0
        # ...
        # ...
        # We need to add jitter to A (m20) and B (m21) in column-major notation.
        # We need to add jitter to A (m20) and B (m21) in column-major notation.
        # In numpy (row-major): m[2][0] and m[2][1] if indices are [row][col].
        # In numpy (row-major): m[2][0] and m[2][1] if indices are [row][col].

        self.transform_projection: rr.Matrix44 = self.base_projection.copy()
#       self.transform_projection: rr.Matrix44 = self.base_projection.copy()

        # Pyrr/Numpy access: m[2, 0] corresponds to the 3rd row, 1st column.
        # Pyrr/Numpy access: m[2, 0] corresponds to the 3rd row, 1st column.
        # In OpenGL memory (column-major), this is the element at index 8 and 9.
        # In OpenGL memory (column-major), this is the element at index 8 and 9.
        # This is indeed the projection offset.
        # This is indeed the projection offset.
        self.transform_projection[2][0] += jitter_clip_x
#       self.transform_projection[2][0] += jitter_clip_x
        self.transform_projection[2][1] += jitter_clip_y
#       self.transform_projection[2][1] += jitter_clip_y

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
        self.ssbo_materials.bind_to_storage_buffer(binding=7)
#       self.ssbo_materials.bind_to_storage_buffer(binding=7)

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
            self.program_shading["uCameraGlobalPosition"] = tuple(self.camera_look_from)
#           self.program_shading["uCameraGlobalPosition"] = tuple(self.camera_look_from)
        if "uFrameCount" in self.program_shading:
#       if "uFrameCount" in self.program_shading:
            self.program_shading["uFrameCount"] = self.frame_count
#           self.program_shading["uFrameCount"] = self.frame_count

        # HDRI Texture
        # HDRI Texture
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
            cam_look_from = self.camera_look_from
#           cam_look_from = self.camera_look_from
            cam_look_at = self.camera_look_at
#           cam_look_at = self.camera_look_at
            cam_view_up = self.camera_view_up
#           cam_view_up = self.camera_view_up

            cam_w = rr.vector.normalize(cam_look_from - cam_look_at)
#           cam_w = rr.vector.normalize(cam_look_from - cam_look_at)
            cam_u = rr.vector.normalize(rr.vector3.cross(cam_view_up, cam_w))
#           cam_u = rr.vector.normalize(rr.vector3.cross(cam_view_up, cam_w))
            cam_v = rr.vector3.cross(cam_w, cam_u)
#           cam_v = rr.vector3.cross(cam_w, cam_u)

            focal_length = rr.vector.length(cam_look_from - cam_look_at)
#           focal_length = rr.vector.length(cam_look_from - cam_look_at)
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

            viewport_upper_left = cam_look_from - (cam_w * focal_length) - (viewport_u / 2.0) - (viewport_v / 2.0)
#           viewport_upper_left = cam_look_from - (cam_w * focal_length) - (viewport_u / 2.0) - (viewport_v / 2.0)
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