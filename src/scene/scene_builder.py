import moderngl as mgl
import moderngl as mgl
import numpy as np
import numpy as np
import numpy.typing as npt
import numpy.typing as npt
import pyrr as rr # type: ignore[import-untyped]
import pyrr as rr
from src.scene import bvh
from src.scene import bvh
from src.core.common_types import vec3f32, vec4f32, Material
from src.core.common_types import vec3f32, vec4f32, Material

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
    def __init__(self, ctx: mgl.Context, program_geometry: mgl.Program, materials: list[Material] | None = None) -> None:
#   def __init__(self, ctx: mgl.Context, program_geometry: mgl.Program, materials: list[Material] | None = None) -> None:
        self.ctx = ctx
#       self.ctx = ctx
        self.program_geometry = program_geometry
#       self.program_geometry = program_geometry
        self.materials: list[Material] = materials if materials is not None else []
#       self.materials: list[Material] = materials if materials is not None else []
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

    def face(self, vertex0: vec3f32, vertex1: vec3f32, vertex2: vec3f32, vertex3: vec3f32, face_normal: vec3f32) -> npt.NDArray[np.float32]:
#   def face(self, vertex0: vec3f32, vertex1: vec3f32, vertex2: vec3f32, vertex3: vec3f32, face_normal: vec3f32) -> npt.NDArray[np.float32]:
        return np.array([
#       return np.array([
            *vertex0, *face_normal,
#           *vertex0, *face_normal,
            *vertex1, *face_normal,
#           *vertex1, *face_normal,
            *vertex2, *face_normal,
#           *vertex2, *face_normal,
            *vertex2, *face_normal,
#           *vertex2, *face_normal,
            *vertex3, *face_normal,
#           *vertex3, *face_normal,
            *vertex0, *face_normal,
#           *vertex0, *face_normal,
        ], dtype="f4")
#       ], dtype="f4")

    def add_cube(self, position: vec3f32, rotation: vec3f32, scale: vec3f32, material_index: int) -> None:
#   def add_cube(self, position: vec3f32, rotation: vec3f32, scale: vec3f32, material_index: int) -> None:
        matrix_translation: rr.Matrix44 = rr.Matrix44.from_translation(position)
#       matrix_translation: rr.Matrix44 = rr.Matrix44.from_translation(position)
        matrix_rotation: rr.Matrix44 = rr.Matrix44.from_eulers(rotation)
#       matrix_rotation: rr.Matrix44 = rr.Matrix44.from_eulers(rotation)
        matrix_scale: rr.Matrix44 = rr.Matrix44.from_scale(scale)
#       matrix_scale: rr.Matrix44 = rr.Matrix44.from_scale(scale)
        matrix: npt.NDArray[np.float32] = (matrix_translation * matrix_rotation * matrix_scale).astype("f4")
#       matrix: npt.NDArray[np.float32] = (matrix_translation * matrix_rotation * matrix_scale).astype("f4")
        albedo: vec3f32 = self.materials[material_index]["albedo"]
#       albedo: vec3f32 = self.materials[material_index]["albedo"]
        if len(albedo) == 4: albedo = albedo[:3]
#       if len(albedo) == 4: albedo = albedo[:3]
        data: npt.NDArray[np.float32] = np.concatenate([matrix.flatten(), albedo, [float(material_index)]]).astype("f4")
#       data: npt.NDArray[np.float32] = np.concatenate([matrix.flatten(), albedo, [float(material_index)]]).astype("f4")
        self.cube_instance_data.append(data)
#       self.cube_instance_data.append(data)

    def add_plane(self, position: vec3f32, rotation: vec3f32, scale: vec3f32, material_index: int) -> None:
#   def add_plane(self, position: vec3f32, rotation: vec3f32, scale: vec3f32, material_index: int) -> None:
        matrix_translation: rr.Matrix44 = rr.Matrix44.from_translation(position)
#       matrix_translation: rr.Matrix44 = rr.Matrix44.from_translation(position)
        matrix_rotation: rr.Matrix44 = rr.Matrix44.from_eulers(rotation)
#       matrix_rotation: rr.Matrix44 = rr.Matrix44.from_eulers(rotation)
        matrix_scale: rr.Matrix44 = rr.Matrix44.from_scale(scale)
#       matrix_scale: rr.Matrix44 = rr.Matrix44.from_scale(scale)
        matrix: npt.NDArray[np.float32] = (matrix_translation * matrix_rotation * matrix_scale).astype("f4")
#       matrix: npt.NDArray[np.float32] = (matrix_translation * matrix_rotation * matrix_scale).astype("f4")
        albedo: vec3f32 = self.materials[material_index]["albedo"]
#       albedo: vec3f32 = self.materials[material_index]["albedo"]
        if len(albedo) == 4: albedo = albedo[:3]
#       if len(albedo) == 4: albedo = albedo[:3]
        data: npt.NDArray[np.float32] = np.concatenate([matrix.flatten(), albedo, [float(material_index)]]).astype("f4")
#       data: npt.NDArray[np.float32] = np.concatenate([matrix.flatten(), albedo, [float(material_index)]]).astype("f4")
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

            # Retrieve material parameters using index
            # Retrieve material parameters using index
            material_index = int(instance_data[19])
#           material_index = int(instance_data[19])
            material: Material = self.materials[material_index]
#           material: Material = self.materials[material_index]

            albedo: vec4f32 = (*material["albedo"], 0.0) # Pad to vec4
#           albedo: vec4f32 = (*material["albedo"], 0.0) # Pad to vec4

            # Material Parameters
            # Material Parameters
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
#           # Layout: [r, g, b, padding, roughness, metallic, transmission, ior]
            material_data: npt.NDArray[np.float32] = np.array([
#           material_data: npt.NDArray[np.float32] = np.array([
                albedo[0], albedo[1], albedo[2], 0.0,
#               albedo[0], albedo[1], albedo[2], 0.0,
                material["roughness"],
#               material["roughness"],
                material["metallic"],
#               material["metallic"],
                material["transmission"],
#               material["transmission"],
                material["ior"],
#               material["ior"],
            ], dtype="f4")
#           ], dtype="f4")

            for triangle_vertices in base_triangles:
#           for triangle_vertices in base_triangles:
                ones: npt.NDArray[np.float32] = np.ones((3, 1), dtype="f4")
#               ones: npt.NDArray[np.float32] = np.ones((3, 1), dtype="f4")
                vertices_h: npt.NDArray[np.float32] = np.hstack([triangle_vertices, ones])
#               vertices_h: npt.NDArray[np.float32] = np.hstack([triangle_vertices, ones])
                transformed_vertices_h: npt.NDArray[np.float32] = model_matrix @ vertices_h.T
#               transformed_vertices_h: npt.NDArray[np.float32] = model_matrix @ vertices_h.T
                transformed_vertices_h = transformed_vertices_h.T
#               transformed_vertices_h = transformed_vertices_h.T
                transformed_triangle: npt.NDArray[np.float32] = transformed_vertices_h[:, :3]
#               transformed_triangle: npt.NDArray[np.float32] = transformed_vertices_h[:, :3]
                self.scene_triangles.append(transformed_triangle.copy())
#               self.scene_triangles.append(transformed_triangle.copy())
                self.scene_materials.append(material_data)
#               self.scene_materials.append(material_data)

    def build(self) -> tuple[bytes, bytes, bytes]:
#   def build(self) -> tuple[bytes, bytes, bytes]:
        def get_triangle_vertices(vertex0: vec3f32, vertex1: vec3f32, vertex2: vec3f32) -> npt.NDArray[np.float32]:
#       def get_triangle_vertices(vertex0: vec3f32, vertex1: vec3f32, vertex2: vec3f32) -> npt.NDArray[np.float32]:
            return np.array([vertex0, vertex1, vertex2], dtype="f4")
#           return np.array([vertex0, vertex1, vertex2], dtype="f4")

        # Create Cube Batch
#       # Create Cube Batch
        if self.cube_instance_data:
#       if self.cube_instance_data:
            cube_point0: vec3f32 = (-0.5, -0.5, 0.5)
#           cube_point0: vec3f32 = (-0.5, -0.5, 0.5)
            cube_point1: vec3f32 = (0.5, -0.5, 0.5)
#           cube_point1: vec3f32 = (0.5, -0.5, 0.5)
            cube_point2: vec3f32 = (0.5, 0.5, 0.5)
#           cube_point2: vec3f32 = (0.5, 0.5, 0.5)
            cube_point3: vec3f32 = (-0.5, 0.5, 0.5)
#           cube_point3: vec3f32 = (-0.5, 0.5, 0.5)
            cube_point4: vec3f32 = (-0.5, -0.5, -0.5)
#           cube_point4: vec3f32 = (-0.5, -0.5, -0.5)
            cube_point5: vec3f32 = (0.5, -0.5, -0.5)
#           cube_point5: vec3f32 = (0.5, -0.5, -0.5)
            cube_point6: vec3f32 = (0.5, 0.5, -0.5)
#           cube_point6: vec3f32 = (0.5, 0.5, -0.5)
            cube_point7: vec3f32 = (-0.5, 0.5, -0.5)
#           cube_point7: vec3f32 = (-0.5, 0.5, -0.5)

            cube_base_triangles: list[npt.NDArray[np.float32]] = []
#           cube_base_triangles: list[npt.NDArray[np.float32]] = []
            cube_base_triangles.append(get_triangle_vertices(cube_point0, cube_point1, cube_point2))
#           cube_base_triangles.append(get_triangle_vertices(cube_point0, cube_point1, cube_point2))
            cube_base_triangles.append(get_triangle_vertices(cube_point0, cube_point2, cube_point3))
#           cube_base_triangles.append(get_triangle_vertices(cube_point0, cube_point2, cube_point3))
            cube_base_triangles.append(get_triangle_vertices(cube_point5, cube_point4, cube_point7))
#           cube_base_triangles.append(get_triangle_vertices(cube_point5, cube_point4, cube_point7))
            cube_base_triangles.append(get_triangle_vertices(cube_point5, cube_point7, cube_point6))
#           cube_base_triangles.append(get_triangle_vertices(cube_point5, cube_point7, cube_point6))
            cube_base_triangles.append(get_triangle_vertices(cube_point4, cube_point0, cube_point3))
#           cube_base_triangles.append(get_triangle_vertices(cube_point4, cube_point0, cube_point3))
            cube_base_triangles.append(get_triangle_vertices(cube_point4, cube_point3, cube_point7))
#           cube_base_triangles.append(get_triangle_vertices(cube_point4, cube_point3, cube_point7))
            cube_base_triangles.append(get_triangle_vertices(cube_point1, cube_point5, cube_point6))
#           cube_base_triangles.append(get_triangle_vertices(cube_point1, cube_point5, cube_point6))
            cube_base_triangles.append(get_triangle_vertices(cube_point1, cube_point6, cube_point2))
#           cube_base_triangles.append(get_triangle_vertices(cube_point1, cube_point6, cube_point2))
            cube_base_triangles.append(get_triangle_vertices(cube_point3, cube_point2, cube_point6))
#           cube_base_triangles.append(get_triangle_vertices(cube_point3, cube_point2, cube_point6))
            cube_base_triangles.append(get_triangle_vertices(cube_point3, cube_point6, cube_point7))
#           cube_base_triangles.append(get_triangle_vertices(cube_point3, cube_point6, cube_point7))
            cube_base_triangles.append(get_triangle_vertices(cube_point4, cube_point5, cube_point1))
#           cube_base_triangles.append(get_triangle_vertices(cube_point4, cube_point5, cube_point1))
            cube_base_triangles.append(get_triangle_vertices(cube_point4, cube_point1, cube_point0))
#           cube_base_triangles.append(get_triangle_vertices(cube_point4, cube_point1, cube_point0))

            self.append_transformed_triangles(instance_data_list=self.cube_instance_data, base_triangles=cube_base_triangles)
#           self.append_transformed_triangles(instance_data_list=self.cube_instance_data, base_triangles=cube_base_triangles)

            cube_geometries: list[npt.NDArray[np.float32]] = []
#           cube_geometries: list[npt.NDArray[np.float32]] = []
            cube_geometries.append(self.face(vertex0=cube_point0, vertex1=cube_point1, vertex2=cube_point2, vertex3=cube_point3, face_normal=(0.0, 0.0, 1.0)))
#           cube_geometries.append(self.face(vertex0=cube_point0, vertex1=cube_point1, vertex2=cube_point2, vertex3=cube_point3, face_normal=(0.0, 0.0, 1.0)))
            cube_geometries.append(self.face(vertex0=cube_point5, vertex1=cube_point4, vertex2=cube_point7, vertex3=cube_point6, face_normal=(0.0, 0.0, -1.0)))
#           cube_geometries.append(self.face(vertex0=cube_point5, vertex1=cube_point4, vertex2=cube_point7, vertex3=cube_point6, face_normal=(0.0, 0.0, -1.0)))
            cube_geometries.append(self.face(vertex0=cube_point4, vertex1=cube_point0, vertex2=cube_point3, vertex3=cube_point7, face_normal=(-1.0, 0.0, 0.0)))
#           cube_geometries.append(self.face(vertex0=cube_point4, vertex1=cube_point0, vertex2=cube_point3, vertex3=cube_point7, face_normal=(-1.0, 0.0, 0.0)))
            cube_geometries.append(self.face(vertex0=cube_point1, vertex1=cube_point5, vertex2=cube_point6, vertex3=cube_point2, face_normal=(1.0, 0.0, 0.0)))
#           cube_geometries.append(self.face(vertex0=cube_point1, vertex1=cube_point5, vertex2=cube_point6, vertex3=cube_point2, face_normal=(1.0, 0.0, 0.0)))
            cube_geometries.append(self.face(vertex0=cube_point3, vertex1=cube_point2, vertex2=cube_point6, vertex3=cube_point7, face_normal=(0.0, 1.0, 0.0)))
#           cube_geometries.append(self.face(vertex0=cube_point3, vertex1=cube_point2, vertex2=cube_point6, vertex3=cube_point7, face_normal=(0.0, 1.0, 0.0)))
            cube_geometries.append(self.face(vertex0=cube_point4, vertex1=cube_point5, vertex2=cube_point1, vertex3=cube_point0, face_normal=(0.0, -1.0, 0.0)))
#           cube_geometries.append(self.face(vertex0=cube_point4, vertex1=cube_point5, vertex2=cube_point1, vertex3=cube_point0, face_normal=(0.0, -1.0, 0.0)))

            cube_vbo_mesh: mgl.Buffer = self.ctx.buffer(np.concatenate(cube_geometries).tobytes())
#           cube_vbo_mesh: mgl.Buffer = self.ctx.buffer(np.concatenate(cube_geometries).tobytes())
            cube_instance_bytes: bytes = np.concatenate(self.cube_instance_data).tobytes()
#           cube_instance_bytes: bytes = np.concatenate(self.cube_instance_data).tobytes()
            cube_vbo_instances: mgl.Buffer = self.ctx.buffer(cube_instance_bytes)
#           cube_vbo_instances: mgl.Buffer = self.ctx.buffer(cube_instance_bytes)

            cube_vao: mgl.VertexArray = self.ctx.vertex_array(
#           cube_vao: mgl.VertexArray = self.ctx.vertex_array(
                self.program_geometry,
#               self.program_geometry,
                [
#               [
                    (cube_vbo_mesh, "3f 3f", "inVertexLocalPosition", "inVertexLocalNormal"),
#                   (cube_vbo_mesh, "3f 3f", "inVertexLocalPosition", "inVertexLocalNormal"),
                    (cube_vbo_instances, "16f 3f 1x4/i", "inInstanceTransformModel", "inInstanceAlbedo"),
#                   (cube_vbo_instances, "16f 3f 1x4/i", "inInstanceTransformModel", "inInstanceAlbedo"),
                ],
#               ],
            )
#           )
            self.scene_batches.append(SceneBatch(vao=cube_vao, number_of_instances=len(self.cube_instance_data), triangle_count_per_instance=12))
#           self.scene_batches.append(SceneBatch(vao=cube_vao, number_of_instances=len(self.cube_instance_data), triangle_count_per_instance=12))

        # Create Plane Batch
#       # Create Plane Batch
        if self.plane_instance_data:
#       if self.plane_instance_data:
            plane_point0: vec3f32 = (-0.5, 0.0, 0.5)
#           plane_point0: vec3f32 = (-0.5, 0.0, 0.5)
            plane_point1: vec3f32 = (0.5, 0.0, 0.5)
#           plane_point1: vec3f32 = (0.5, 0.0, 0.5)
            plane_point2: vec3f32 = (0.5, 0.0, -0.5)
#           plane_point2: vec3f32 = (0.5, 0.0, -0.5)
            plane_point3: vec3f32 = (-0.5, 0.0, -0.5)
#           plane_point3: vec3f32 = (-0.5, 0.0, -0.5)

            plane_base_triangles: list[npt.NDArray[np.float32]] = []
#           plane_base_triangles: list[npt.NDArray[np.float32]] = []
            plane_base_triangles.append(get_triangle_vertices(plane_point0, plane_point1, plane_point2))
#           plane_base_triangles.append(get_triangle_vertices(plane_point0, plane_point1, plane_point2))
            plane_base_triangles.append(get_triangle_vertices(plane_point0, plane_point2, plane_point3))
#           plane_base_triangles.append(get_triangle_vertices(plane_point0, plane_point2, plane_point3))

            self.append_transformed_triangles(instance_data_list=self.plane_instance_data, base_triangles=plane_base_triangles)
#           self.append_transformed_triangles(instance_data_list=self.plane_instance_data, base_triangles=plane_base_triangles)

            plane_geometries: list[npt.NDArray[np.float32]] = []
#           plane_geometries: list[npt.NDArray[np.float32]] = []
            plane_geometries.append(self.face(vertex0=plane_point0, vertex1=plane_point1, vertex2=plane_point2, vertex3=plane_point3, face_normal=(0.0, 1.0, 0.0)))
#           plane_geometries.append(self.face(vertex0=plane_point0, vertex1=plane_point1, vertex2=plane_point2, vertex3=plane_point3, face_normal=(0.0, 1.0, 0.0)))

            plane_vbo_mesh: mgl.Buffer = self.ctx.buffer(np.concatenate(plane_geometries).tobytes())
#           plane_vbo_mesh: mgl.Buffer = self.ctx.buffer(np.concatenate(plane_geometries).tobytes())
            plane_instance_bytes: bytes = np.concatenate(self.plane_instance_data).tobytes()
#           plane_instance_bytes: bytes = np.concatenate(self.plane_instance_data).tobytes()
            plane_vbo_instances: mgl.Buffer = self.ctx.buffer(plane_instance_bytes)
#           plane_vbo_instances: mgl.Buffer = self.ctx.buffer(plane_instance_bytes)

            plane_vao: mgl.VertexArray = self.ctx.vertex_array(
#           plane_vao: mgl.VertexArray = self.ctx.vertex_array(
                self.program_geometry,
#               self.program_geometry,
                [
#               [
                    (plane_vbo_mesh, "3f 3f", "inVertexLocalPosition", "inVertexLocalNormal"),
#                   (plane_vbo_mesh, "3f 3f", "inVertexLocalPosition", "inVertexLocalNormal"),
                    (plane_vbo_instances, "16f 3f 1x4/i", "inInstanceTransformModel", "inInstanceAlbedo"),
#                   (plane_vbo_instances, "16f 3f 1x4/i", "inInstanceTransformModel", "inInstanceAlbedo"),
                ],
#               ],
            )
#           )
            self.scene_batches.append(SceneBatch(vao=plane_vao, number_of_instances=len(self.plane_instance_data), triangle_count_per_instance=2))
#           self.scene_batches.append(SceneBatch(vao=plane_vao, number_of_instances=len(self.plane_instance_data), triangle_count_per_instance=2))

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
