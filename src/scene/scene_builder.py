import moderngl as mgl
import moderngl as mgl
import numpy as np
import numpy as np
import numpy.typing as npt
import numpy.typing as npt
import pyrr as rr
import pyrr as rr
from src.scene import bvh
from src.scene import bvh
from src.core.common_types import vec3f32
from src.core.common_types import vec3f32

class SceneBatch:
    def __init__(self, vao: mgl.VertexArray, number_of_instances: int, triangle_count_per_instance: int) -> None:
#   def __init__(self, vao: mgl.VertexArray, number_of_instances: int, triangle_count_per_instance: int) -> None:
        self.vao: mgl.VertexArray = vao
#       self.vao: mgl.VertexArray = vao
        self.number_of_instances: int = number_of_instances
#       self.number_of_instances: int = number_of_instances
        self.triangle_count_per_instance: int = triangle_count_per_instance
#       self.triangle_count_per_instance: int = triangle_count_per_instance
        pass
#       pass

class SceneBuilder:
    def __init__(self, ctx: mgl.Context, program_geometry: mgl.Program) -> None:
#   def __init__(self, ctx: mgl.Context, program_geometry: mgl.Program) -> None:
        self.ctx = ctx
#       self.ctx = ctx
        self.program_geometry = program_geometry
#       self.program_geometry = program_geometry
        self.cube_instance_data: list[npt.NDArray[np.float32]] = []
#       self.cube_instance_data: list[npt.NDArray[np.float32]] = []
        self.plane_instance_data: list[npt.NDArray[np.float32]] = []
#       self.plane_instance_data: list[npt.NDArray[np.float32]] = []
        self.scene_triangles: list[npt.NDArray[np.float32]] = []
#       self.scene_triangles: list[npt.NDArray[np.float32]] = []
        self.scene_materials: list[npt.NDArray[np.float32]] = []
#       self.scene_materials: list[npt.NDArray[np.float32]] = []
        self.scene_batches: list[SceneBatch] = []
#       self.scene_batches: list[SceneBatch] = []

    def face(self, vertex_a: vec3f32, vertex_b: vec3f32, vertex_c: vec3f32, vertex_d: vec3f32, face_normal: vec3f32) -> npt.NDArray[np.float32]:
#   def face(self, vertex_a: vec3f32, vertex_b: vec3f32, vertex_c: vec3f32, vertex_d: vec3f32, face_normal: vec3f32) -> npt.NDArray[np.float32]:
        return np.array([
#       return np.array([
            *vertex_a, *face_normal,
#           *vertex_a, *face_normal,
            *vertex_b, *face_normal,
#           *vertex_b, *face_normal,
            *vertex_c, *face_normal,
#           *vertex_c, *face_normal,
            *vertex_c, *face_normal,
#           *vertex_c, *face_normal,
            *vertex_d, *face_normal,
#           *vertex_d, *face_normal,
            *vertex_a, *face_normal,
#           *vertex_a, *face_normal,
        ], dtype="f4")
#       ], dtype="f4")

    def add_cube(self, position: vec3f32, rotation: vec3f32, scale: vec3f32, color: vec3f32) -> None:
#   def add_cube(self, position: vec3f32, rotation: vec3f32, scale: vec3f32, color: vec3f32) -> None:
        matrix_translation: rr.Matrix44 = rr.Matrix44.from_translation(position)
#       matrix_translation: rr.Matrix44 = rr.Matrix44.from_translation(position)
        matrix_rotation: rr.Matrix44 = rr.Matrix44.from_eulers(rotation)
#       matrix_rotation: rr.Matrix44 = rr.Matrix44.from_eulers(rotation)
        matrix_scale: rr.Matrix44 = rr.Matrix44.from_scale(scale)
#       matrix_scale: rr.Matrix44 = rr.Matrix44.from_scale(scale)
        matrix: npt.NDArray[np.float32] = (matrix_translation * matrix_rotation * matrix_scale).astype("f4")
#       matrix: npt.NDArray[np.float32] = (matrix_translation * matrix_rotation * matrix_scale).astype("f4")
        data: npt.NDArray[np.float32] = np.concatenate([matrix.flatten(), color]).astype("f4")
#       data: npt.NDArray[np.float32] = np.concatenate([matrix.flatten(), color]).astype("f4")
        self.cube_instance_data.append(data)
#       self.cube_instance_data.append(data)

    def add_plane(self, position: vec3f32, rotation: vec3f32, scale: vec3f32, color: vec3f32) -> None:
#   def add_plane(self, position: vec3f32, rotation: vec3f32, scale: vec3f32, color: vec3f32) -> None:
        matrix_translation: rr.Matrix44 = rr.Matrix44.from_translation(position)
#       matrix_translation: rr.Matrix44 = rr.Matrix44.from_translation(position)
        matrix_rotation: rr.Matrix44 = rr.Matrix44.from_eulers(rotation)
#       matrix_rotation: rr.Matrix44 = rr.Matrix44.from_eulers(rotation)
        matrix_scale: rr.Matrix44 = rr.Matrix44.from_scale(scale)
#       matrix_scale: rr.Matrix44 = rr.Matrix44.from_scale(scale)
        matrix: npt.NDArray[np.float32] = (matrix_translation * matrix_rotation * matrix_scale).astype("f4")
#       matrix: npt.NDArray[np.float32] = (matrix_translation * matrix_rotation * matrix_scale).astype("f4")
        data: npt.NDArray[np.float32] = np.concatenate([matrix.flatten(), color]).astype("f4")
#       data: npt.NDArray[np.float32] = np.concatenate([matrix.flatten(), color]).astype("f4")
        self.plane_instance_data.append(data)
#       self.plane_instance_data.append(data)

    def append_transformed_triangles(
#   def append_transformed_triangles(
        self,
#       self,
        instance_data_list: list[npt.NDArray[np.float32]],
#       instance_data_list: list[npt.NDArray[np.float32]],
        base_triangles: list[npt.NDArray[np.float32]]
#       base_triangles: list[npt.NDArray[np.float32]]
    ) -> None:
#   ) -> None:
        for instance_data in instance_data_list:
#       for instance_data in instance_data_list:
            model_matrix_flat: npt.NDArray[np.float32] = instance_data[:16]
#           model_matrix_flat: npt.NDArray[np.float32] = instance_data[:16]
            model_matrix: npt.NDArray[np.float32] = model_matrix_flat.reshape((4, 4), order='F')
#           model_matrix: npt.NDArray[np.float32] = model_matrix_flat.reshape((4, 4), order='F')

            color: npt.NDArray[np.float32] = instance_data[16:19]
#           color: npt.NDArray[np.float32] = instance_data[16:19]

            material_data = np.array([
#           material_data = np.array([
                color[0], color[1], color[2], 0.0,
#               color[0], color[1], color[2], 0.0,
                1.0, 0.0, 0.0, 1.5,
#               1.0, 0.0, 0.0, 1.5,
            ], dtype="f4")
#           ], dtype="f4")

            for tri_verts in base_triangles:
#           for tri_verts in base_triangles:
                ones: npt.NDArray[np.float32] = np.ones((3, 1), dtype="f4")
#               ones: npt.NDArray[np.float32] = np.ones((3, 1), dtype="f4")
                verts_h: npt.NDArray[np.float32] = np.hstack([tri_verts, ones])
#               verts_h: npt.NDArray[np.float32] = np.hstack([tri_verts, ones])
                transformed_verts_h: npt.NDArray[np.float32] = model_matrix @ verts_h.T
#               transformed_verts_h: npt.NDArray[np.float32] = model_matrix @ verts_h.T
                transformed_verts_h = transformed_verts_h.T
#               transformed_verts_h = transformed_verts_h.T
                transformed_tri: npt.NDArray[np.float32] = transformed_verts_h[:, :3]
#               transformed_tri: npt.NDArray[np.float32] = transformed_verts_h[:, :3]
                self.scene_triangles.append(transformed_tri.copy())
#               self.scene_triangles.append(transformed_tri.copy())
                self.scene_materials.append(material_data)
#               self.scene_materials.append(material_data)

    def build(self) -> tuple[bytes, bytes, bytes]:
#   def build(self) -> tuple[bytes, bytes, bytes]:
        # Create Cube Batch
#       # Create Cube Batch
        if self.cube_instance_data:
#       if self.cube_instance_data:
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

            def get_tri_verts(v0, v1, v2) -> npt.NDArray[np.float32]:
#           def get_tri_verts(v0, v1, v2) -> npt.NDArray[np.float32]:
                return np.array([v0, v1, v2], dtype="f4")
#               return np.array([v0, v1, v2], dtype="f4")

            cube_base_triangles: list[npt.NDArray[np.float32]] = []
#           cube_base_triangles: list[npt.NDArray[np.float32]] = []
            cube_base_triangles.append(get_tri_verts(point0, point1, point2))
#           cube_base_triangles.append(get_tri_verts(point0, point1, point2))
            cube_base_triangles.append(get_tri_verts(point0, point2, point3))
#           cube_base_triangles.append(get_tri_verts(point0, point2, point3))
            cube_base_triangles.append(get_tri_verts(point5, point4, point7))
#           cube_base_triangles.append(get_tri_verts(point5, point4, point7))
            cube_base_triangles.append(get_tri_verts(point5, point7, point6))
#           cube_base_triangles.append(get_tri_verts(point5, point7, point6))
            cube_base_triangles.append(get_tri_verts(point4, point0, point3))
#           cube_base_triangles.append(get_tri_verts(point4, point0, point3))
            cube_base_triangles.append(get_tri_verts(point4, point3, point7))
#           cube_base_triangles.append(get_tri_verts(point4, point3, point7))
            cube_base_triangles.append(get_tri_verts(point1, point5, point6))
#           cube_base_triangles.append(get_tri_verts(point1, point5, point6))
            cube_base_triangles.append(get_tri_verts(point1, point6, point2))
#           cube_base_triangles.append(get_tri_verts(point1, point6, point2))
            cube_base_triangles.append(get_tri_verts(point3, point2, point6))
#           cube_base_triangles.append(get_tri_verts(point3, point2, point6))
            cube_base_triangles.append(get_tri_verts(point3, point6, point7))
#           cube_base_triangles.append(get_tri_verts(point3, point6, point7))
            cube_base_triangles.append(get_tri_verts(point4, point5, point1))
#           cube_base_triangles.append(get_tri_verts(point4, point5, point1))
            cube_base_triangles.append(get_tri_verts(point4, point1, point0))
#           cube_base_triangles.append(get_tri_verts(point4, point1, point0))

            self.append_transformed_triangles(self.cube_instance_data, cube_base_triangles)
#           self.append_transformed_triangles(self.cube_instance_data, cube_base_triangles)

            geometries: list[npt.NDArray[np.float32]] = []
#           geometries: list[npt.NDArray[np.float32]] = []
            geometries.append(self.face(vertex_a=point0, vertex_b=point1, vertex_c=point2, vertex_d=point3, face_normal=(0.0, 0.0, 1.0)))
#           geometries.append(self.face(vertex_a=point0, vertex_b=point1, vertex_c=point2, vertex_d=point3, face_normal=(0.0, 0.0, 1.0)))
            geometries.append(self.face(vertex_a=point5, vertex_b=point4, vertex_c=point7, vertex_d=point6, face_normal=(0.0, 0.0, -1.0)))
#           geometries.append(self.face(vertex_a=point5, vertex_b=point4, vertex_c=point7, vertex_d=point6, face_normal=(0.0, 0.0, -1.0)))
            geometries.append(self.face(vertex_a=point4, vertex_b=point0, vertex_c=point3, vertex_d=point7, face_normal=(-1.0, 0.0, 0.0)))
#           geometries.append(self.face(vertex_a=point4, vertex_b=point0, vertex_c=point3, vertex_d=point7, face_normal=(-1.0, 0.0, 0.0)))
            geometries.append(self.face(vertex_a=point1, vertex_b=point5, vertex_c=point6, vertex_d=point2, face_normal=(1.0, 0.0, 0.0)))
#           geometries.append(self.face(vertex_a=point1, vertex_b=point5, vertex_c=point6, vertex_d=point2, face_normal=(1.0, 0.0, 0.0)))
            geometries.append(self.face(vertex_a=point3, vertex_b=point2, vertex_c=point6, vertex_d=point7, face_normal=(0.0, 1.0, 0.0)))
#           geometries.append(self.face(vertex_a=point3, vertex_b=point2, vertex_c=point6, vertex_d=point7, face_normal=(0.0, 1.0, 0.0)))
            geometries.append(self.face(vertex_a=point4, vertex_b=point5, vertex_c=point1, vertex_d=point0, face_normal=(0.0, -1.0, 0.0)))
#           geometries.append(self.face(vertex_a=point4, vertex_b=point5, vertex_c=point1, vertex_d=point0, face_normal=(0.0, -1.0, 0.0)))

            vbo_mesh: mgl.Buffer = self.ctx.buffer(np.concatenate(geometries).tobytes())
#           vbo_mesh: mgl.Buffer = self.ctx.buffer(np.concatenate(geometries).tobytes())
            instance_bytes: bytes = np.concatenate(self.cube_instance_data).tobytes()
#           instance_bytes: bytes = np.concatenate(self.cube_instance_data).tobytes()
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
            self.scene_batches.append(SceneBatch(vao=vao_cube, number_of_instances=len(self.cube_instance_data), triangle_count_per_instance=12))
#           self.scene_batches.append(SceneBatch(vao=vao_cube, number_of_instances=len(self.cube_instance_data), triangle_count_per_instance=12))

        # Create Plane Batch
#       # Create Plane Batch
        if self.plane_instance_data:
#       if self.plane_instance_data:
            point0: vec3f32 = (-0.5, 0.0, 0.5)
#           point0: vec3f32 = (-0.5, 0.0, 0.5)
            point1: vec3f32 = (0.5, 0.0, 0.5)
#           point1: vec3f32 = (0.5, 0.0, 0.5)
            point2: vec3f32 = (0.5, 0.0, -0.5)
#           point2: vec3f32 = (0.5, 0.0, -0.5)
            point3: vec3f32 = (-0.5, 0.0, -0.5)
#           point3: vec3f32 = (-0.5, 0.0, -0.5)

            def get_tri_verts(v0, v1, v2) -> npt.NDArray[np.float32]:
#           def get_tri_verts(v0, v1, v2) -> npt.NDArray[np.float32]:
                return np.array([v0, v1, v2], dtype="f4")
#               return np.array([v0, v1, v2], dtype="f4")

            plane_base_triangles: list[npt.NDArray[np.float32]] = []
#           plane_base_triangles: list[npt.NDArray[np.float32]] = []
            plane_base_triangles.append(get_tri_verts(point0, point1, point2))
#           plane_base_triangles.append(get_tri_verts(point0, point1, point2))
            plane_base_triangles.append(get_tri_verts(point0, point2, point3))
#           plane_base_triangles.append(get_tri_verts(point0, point2, point3))

            self.append_transformed_triangles(self.plane_instance_data, plane_base_triangles)
#           self.append_transformed_triangles(self.plane_instance_data, plane_base_triangles)

            geometries: list[npt.NDArray[np.float32]] = []
#           geometries: list[npt.NDArray[np.float32]] = []
            geometries.append(self.face(vertex_a=point0, vertex_b=point1, vertex_c=point2, vertex_d=point3, face_normal=(0.0, 1.0, 0.0)))
#           geometries.append(self.face(vertex_a=point0, vertex_b=point1, vertex_c=point2, vertex_d=point3, face_normal=(0.0, 1.0, 0.0)))

            vbo_mesh: mgl.Buffer = self.ctx.buffer(np.concatenate(geometries).tobytes())
#           vbo_mesh: mgl.Buffer = self.ctx.buffer(np.concatenate(geometries).tobytes())
            instance_bytes: bytes = np.concatenate(self.plane_instance_data).tobytes()
#           instance_bytes: bytes = np.concatenate(self.plane_instance_data).tobytes()
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
            self.scene_batches.append(SceneBatch(vao=vao_plane, number_of_instances=len(self.plane_instance_data), triangle_count_per_instance=2))
#           self.scene_batches.append(SceneBatch(vao=vao_plane, number_of_instances=len(self.plane_instance_data), triangle_count_per_instance=2))

        # Build BVH
#       # Build BVH
        world_triangles: npt.NDArray[np.float32] = np.array(self.scene_triangles, dtype="f4")
#       world_triangles: npt.NDArray[np.float32] = np.array(self.scene_triangles, dtype="f4")
        lbvh: bvh.LBVH = bvh.LBVH(world_triangles)
#       lbvh: bvh.LBVH = bvh.LBVH(world_triangles)
        bvh_data: bytes = lbvh.simple_build()
#       bvh_data: bytes = lbvh.simple_build()

        world_materials: npt.NDArray[np.float32] = np.array(self.scene_materials, dtype="f4")
#       world_materials: npt.NDArray[np.float32] = np.array(self.scene_materials, dtype="f4")

        return bvh_data, world_triangles.flatten().tobytes(), world_materials.flatten().tobytes()
#       return bvh_data, world_triangles.flatten().tobytes(), world_materials.flatten().tobytes()
