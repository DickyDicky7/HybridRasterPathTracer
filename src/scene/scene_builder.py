import moderngl as mgl
import moderngl as mgl
import numpy as np
import numpy as np
import numpy.typing as npt
import numpy.typing as npt
import pyrr as rr # type: ignore[import-untyped]
import pyrr as rr
import pyassimp   # type: ignore[import-untyped]
import pyassimp
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
        self.scene_uvs: list[npt.NDArray[np.float32]] = []
#       self.scene_uvs: list[npt.NDArray[np.float32]] = []
        self.scene_materials: list[npt.NDArray[np.float32]] = []
#       self.scene_materials: list[npt.NDArray[np.float32]] = []
        self.scene_normals: list[npt.NDArray[np.float32]] = []
#       self.scene_normals: list[npt.NDArray[np.float32]] = []
        self.scene_tangents: list[npt.NDArray[np.float32]] = []
#       self.scene_tangents: list[npt.NDArray[np.float32]] = []
        self.scene_batches: list[SceneBatch] = []
#       self.scene_batches: list[SceneBatch] = []

    def calculate_triangle_tangent(self, v0: vec3f32, v1: vec3f32, v2: vec3f32, uv0: vec2f32, uv1: vec2f32, uv2: vec2f32) -> vec3f32:
#   def calculate_triangle_tangent(self, v0: vec3f32, v1: vec3f32, v2: vec3f32, uv0: vec2f32, uv1: vec2f32, uv2: vec2f32) -> vec3f32:
        edge1 = np.array(v1) - np.array(v0)
#       edge1 = np.array(v1) - np.array(v0)
        edge2 = np.array(v2) - np.array(v0)
#       edge2 = np.array(v2) - np.array(v0)
        deltaUV1 = np.array(uv1) - np.array(uv0)
#       deltaUV1 = np.array(uv1) - np.array(uv0)
        deltaUV2 = np.array(uv2) - np.array(uv0)
#       deltaUV2 = np.array(uv2) - np.array(uv0)

        f = 1.0 / (deltaUV1[0] * deltaUV2[1] - deltaUV2[0] * deltaUV1[1])
#       f = 1.0 / (deltaUV1[0] * deltaUV2[1] - deltaUV2[0] * deltaUV1[1])

        tangent = np.array([
#       tangent = np.array([
            f * (deltaUV2[1] * edge1[0] - deltaUV1[1] * edge2[0]),
#           f * (deltaUV2[1] * edge1[0] - deltaUV1[1] * edge2[0]),
            f * (deltaUV2[1] * edge1[1] - deltaUV1[1] * edge2[1]),
#           f * (deltaUV2[1] * edge1[1] - deltaUV1[1] * edge2[1]),
            f * (deltaUV2[1] * edge1[2] - deltaUV1[1] * edge2[2])
#           f * (deltaUV2[1] * edge1[2] - deltaUV1[1] * edge2[2])
        ], dtype="f4")
#       ], dtype="f4")

        norm = np.linalg.norm(tangent)
#       norm = np.linalg.norm(tangent)
        if norm > 1e-6:
#       if norm > 1e-6:
            tangent = tangent / norm
#           tangent = tangent / norm
        return (tangent[0], tangent[1], tangent[2])
#       return (tangent[0], tangent[1], tangent[2])

    def face(self, vertex0: vec3f32, vertex1: vec3f32, vertex2: vec3f32, vertex3: vec3f32, face_normal: vec3f32, face_tangent: vec3f32, uv0: vec2f32 = (0.0, 0.0), uv1: vec2f32 = (1.0, 0.0), uv2: vec2f32 = (1.0, 1.0), uv3: vec2f32 = (0.0, 1.0)) -> npt.NDArray[np.float32]:
#   def face(self, vertex0: vec3f32, vertex1: vec3f32, vertex2: vec3f32, vertex3: vec3f32, face_normal: vec3f32, face_tangent: vec3f32, uv0: vec2f32 = (0.0, 0.0), uv1: vec2f32 = (1.0, 0.0), uv2: vec2f32 = (1.0, 1.0), uv3: vec2f32 = (0.0, 1.0)) -> npt.NDArray[np.float32]:
        return np.array([
#       return np.array([
            *vertex0, *face_normal, *face_tangent, *uv0,
#           *vertex0, *face_normal, *face_tangent, *uv0,
            *vertex1, *face_normal, *face_tangent, *uv1,
#           *vertex1, *face_normal, *face_tangent, *uv1,
            *vertex2, *face_normal, *face_tangent, *uv2,
#           *vertex2, *face_normal, *face_tangent, *uv2,
            *vertex2, *face_normal, *face_tangent, *uv2,
#           *vertex2, *face_normal, *face_tangent, *uv2,
            *vertex3, *face_normal, *face_tangent, *uv3,
#           *vertex3, *face_normal, *face_tangent, *uv3,
            *vertex0, *face_normal, *face_tangent, *uv0,
#           *vertex0, *face_normal, *face_tangent, *uv0,
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

    def load_obj(self, path: str, position: vec3f32, rotation: vec3f32, scale: vec3f32, material_index: int) -> None:
#   def load_obj(self, path: str, position: vec3f32, rotation: vec3f32, scale: vec3f32, material_index: int) -> None:
        # Prepare transformation matrix
#       # Prepare transformation matrix
        matrix_translation: rr.Matrix44 = rr.Matrix44.from_translation(position)
#       matrix_translation: rr.Matrix44 = rr.Matrix44.from_translation(position)
        matrix_rotation: rr.Matrix44 = rr.Matrix44.from_eulers(rotation)
#       matrix_rotation: rr.Matrix44 = rr.Matrix44.from_eulers(rotation)
        matrix_scale: rr.Matrix44 = rr.Matrix44.from_scale(scale)
#       matrix_scale: rr.Matrix44 = rr.Matrix44.from_scale(scale)
        model_matrix: npt.NDArray[np.float32] = (matrix_translation * matrix_rotation * matrix_scale).astype("f4")
#       model_matrix: npt.NDArray[np.float32] = (matrix_translation * matrix_rotation * matrix_scale).astype("f4")

        # Calculate Normal Matrix (Transpose of Inverse of upper 3x3)
#       # Calculate Normal Matrix (Transpose of Inverse of upper 3x3)
        m33 = model_matrix[:3, :3]
#       m33 = model_matrix[:3, :3]
        try:
#       try:
            m33_inv = np.linalg.inv(m33)
#           m33_inv = np.linalg.inv(m33)
            normal_matrix = m33_inv.T
#           normal_matrix = m33_inv.T
        except np.linalg.LinAlgError:
#       except np.linalg.LinAlgError:
            normal_matrix = np.eye(3, dtype="f4")
#           normal_matrix = np.eye(3, dtype="f4")

        # Prepare material data
#       # Prepare material data
        material: Material = self.materials[material_index]
#       material: Material = self.materials[material_index]
        albedo: vec4f32 = (*material["albedo"], 0.0)
#       albedo: vec4f32 = (*material["albedo"], 0.0)
        material_data: npt.NDArray[np.float32] = np.array([
#       material_data: npt.NDArray[np.float32] = np.array([
            albedo[0], albedo[1], albedo[2], 0.0,
#           albedo[0], albedo[1], albedo[2], 0.0,
            material["roughness"],
#           material["roughness"],
            material["metallic"],
#           material["metallic"],
            material["transmission"],
#           material["transmission"],
            material["ior"],
#           material["ior"],
            material["texture_index_albedo"],
#           material["texture_index_albedo"],
            material["texture_index_roughness"],
#           material["texture_index_roughness"],
            material["texture_index_metallic"],
#           material["texture_index_metallic"],
            material["texture_index_normal"],
#           material["texture_index_normal"],
        ], dtype="f4")
#       ], dtype="f4")

        try:
#       try:
            with pyassimp.load(filename=path, processing=
#           with pyassimp.load(filename=path, processing=
                pyassimp.postprocess.aiProcess_Triangulate |
#               pyassimp.postprocess.aiProcess_Triangulate |
                pyassimp.postprocess.aiProcess_CalcTangentSpace |
#               pyassimp.postprocess.aiProcess_CalcTangentSpace |
                pyassimp.postprocess.aiProcess_GenSmoothNormals |
#               pyassimp.postprocess.aiProcess_GenSmoothNormals |
                pyassimp.postprocess.aiProcess_JoinIdenticalVertices |
#               pyassimp.postprocess.aiProcess_JoinIdenticalVertices |
                pyassimp.postprocess.aiProcess_FixInfacingNormals |
#               pyassimp.postprocess.aiProcess_FixInfacingNormals |
                pyassimp.postprocess.aiProcess_OptimizeMeshes |
#               pyassimp.postprocess.aiProcess_OptimizeMeshes |
                pyassimp.postprocess.aiProcess_OptimizeGraph |
#               pyassimp.postprocess.aiProcess_OptimizeGraph |
                pyassimp.postprocess.aiProcess_OptimizeAnimations |
#               pyassimp.postprocess.aiProcess_OptimizeAnimations |
                pyassimp.postprocess.aiProcess_ImproveCacheLocality |
#               pyassimp.postprocess.aiProcess_ImproveCacheLocality |
                pyassimp.postprocess.aiProcess_PreTransformVertices
#               pyassimp.postprocess.aiProcess_PreTransformVertices
            ) as scene:
#           ) as scene:

                # Iterate meshes
#               # Iterate meshes
                for mesh in scene.meshes:
#               for mesh in scene.meshes:
                    # Flatten indices
#                   # Flatten indices
                    # mesh.faces is typically a list of indices per face (mostly 3 if triangulated)
#                   # mesh.faces is typically a list of indices per face (mostly 3 if triangulated)
                    indices = mesh.faces.flatten() if hasattr(mesh.faces, 'flatten') else np.array(mesh.faces).flatten()
#                   indices = mesh.faces.flatten() if hasattr(mesh.faces, 'flatten') else np.array(mesh.faces).flatten()

                    # Vertices
#                   # Vertices
                    vertices = mesh.vertices[indices]
#                   vertices = mesh.vertices[indices]

                    # Normals
#                   # Normals
                    normals = mesh.normals[indices]
#                   normals = mesh.normals[indices]

                    # Tangents
#                   # Tangents
                    if hasattr(mesh, 'tangents') and len(mesh.tangents) > 0:
#                   if hasattr(mesh, 'tangents') and len(mesh.tangents) > 0:
                        tangents = mesh.tangents[indices]
#                       tangents = mesh.tangents[indices]
                    else:
#                   else:
                        # Fallback: Zero tangents? Or compute them?
#                       # Fallback: Zero tangents? Or compute them?
                        # For now, let's just use (1,0,0) or (0,0,1)
#                       # For now, let's just use (1,0,0) or (0,0,1)
                        tangents = np.zeros_like(vertices)
#                       tangents = np.zeros_like(vertices)
                        tangents[:, 0] = 1.0
#                       tangents[:, 0] = 1.0

                    # UVs
#                   # UVs
                    # mesh.texturecoords is a list of arrays (layers). We use layer 0.
#                   # mesh.texturecoords is a list of arrays (layers). We use layer 0.
                    # Shape is (N_layers, N_vertices, 3). We want (N_vertices, 2)
#                   # Shape is (N_layers, N_vertices, 3). We want (N_vertices, 2)
                    if hasattr(mesh, 'texturecoords') and len(mesh.texturecoords) > 0:
#                   if hasattr(mesh, 'texturecoords') and len(mesh.texturecoords) > 0:
                        uvs_raw = mesh.texturecoords[0][indices] # (N, 3)
#                       uvs_raw = mesh.texturecoords[0][indices] # (N, 3)
                        uvs = uvs_raw[:, :2] # (N, 2)
#                       uvs = uvs_raw[:, :2] # (N, 2)
                    else:
#                   else:
                        uvs = np.zeros((len(vertices), 2), dtype="f4")
#                       uvs = np.zeros((len(vertices), 2), dtype="f4")

                    # Transform and Append
#                   # Transform and Append

                    # Transform Vertices
#                   # Transform Vertices
                    ones = np.ones((len(vertices), 1), dtype="f4")
#                   ones = np.ones((len(vertices), 1), dtype="f4")
                    vertices_h = np.hstack([vertices, ones])
#                   vertices_h = np.hstack([vertices, ones])
                    transformed_vertices_h = vertices_h @ model_matrix.T # (N, 4)
#                   transformed_vertices_h = vertices_h @ model_matrix.T # (N, 4)
                    transformed_vertices = transformed_vertices_h[:, :3] # (N, 3)
#                   transformed_vertices = transformed_vertices_h[:, :3] # (N, 3)

                    # Transform Normals
#                   # Transform Normals
                    transformed_normals = normals @ normal_matrix.T # (N, 3)
#                   transformed_normals = normals @ normal_matrix.T # (N, 3)
                    # Normalize
#                   # Normalize
                    norms = np.linalg.norm(transformed_normals, axis=1, keepdims=True)
#                   norms = np.linalg.norm(transformed_normals, axis=1, keepdims=True)
                    # Avoid division by zero
#                   # Avoid division by zero
                    valid_norms = norms > 1e-6
#                   valid_norms = norms > 1e-6
                    transformed_normals[valid_norms.flatten()] /= norms[valid_norms.flatten()]
#                   transformed_normals[valid_norms.flatten()] /= norms[valid_norms.flatten()]

                    # Transform Tangents
#                   # Transform Tangents
                    # Tangents rotate with the model (ignoring non-uniform scale for now or handling it similarly)
#                   # Tangents rotate with the model (ignoring non-uniform scale for now or handling it similarly)
                    # Tangents are vectors, so use upper 3x3 of model matrix (linear part)
#                   # Tangents are vectors, so use upper 3x3 of model matrix (linear part)
                    # Actually, standard practice is to use normal matrix for tangents too? Or just linear part?
#                   # Actually, standard practice is to use normal matrix for tangents too? Or just linear part?
                    # Let's use linear part (rotation + scale).
#                   # Let's use linear part (rotation + scale).
                    transformed_tangents = tangents @ model_matrix[:3, :3].T # (N, 3)
#                   transformed_tangents = tangents @ model_matrix[:3, :3].T # (N, 3)
                    # Normalize
#                   # Normalize
                    tan_norms = np.linalg.norm(transformed_tangents, axis=1, keepdims=True)
#                   tan_norms = np.linalg.norm(transformed_tangents, axis=1, keepdims=True)
                    valid_tan_norms = tan_norms > 1e-6
#                   valid_tan_norms = tan_norms > 1e-6
                    transformed_tangents[valid_tan_norms.flatten()] /= tan_norms[valid_tan_norms.flatten()]
#                   transformed_tangents[valid_tan_norms.flatten()] /= tan_norms[valid_tan_norms.flatten()]

                    # Append (triangle by triangle)
#                   # Append (triangle by triangle)
                    # Reshape to (N_triangles, 3, 3/2)
#                   # Reshape to (N_triangles, 3, 3/2)
                    n_triangles = len(vertices) // 3
#                   n_triangles = len(vertices) // 3

                    tri_vertices = transformed_vertices.reshape((n_triangles, 3, 3))
#                   tri_vertices = transformed_vertices.reshape((n_triangles, 3, 3))
                    tri_normals = transformed_normals.reshape((n_triangles, 3, 3))
#                   tri_normals = transformed_normals.reshape((n_triangles, 3, 3))
                    tri_tangents = transformed_tangents.reshape((n_triangles, 3, 3))
#                   tri_tangents = transformed_tangents.reshape((n_triangles, 3, 3))
                    tri_uvs = uvs.reshape((n_triangles, 3, 2))
#                   tri_uvs = uvs.reshape((n_triangles, 3, 2))

                    for i in range(n_triangles):
#                   for i in range(n_triangles):
                        self.scene_triangles.append(tri_vertices[i].astype("f4"))
#                       self.scene_triangles.append(tri_vertices[i].astype("f4"))
                        self.scene_normals.append(tri_normals[i].astype("f4"))
#                       self.scene_normals.append(tri_normals[i].astype("f4"))
                        self.scene_tangents.append(tri_tangents[i].astype("f4"))
#                       self.scene_tangents.append(tri_tangents[i].astype("f4"))
                        self.scene_uvs.append(tri_uvs[i].astype("f4"))
#                       self.scene_uvs.append(tri_uvs[i].astype("f4"))
                        self.scene_materials.append(material_data)
#                       self.scene_materials.append(material_data)

                    # TODO: Add VBO creation for rasterization pass if needed?
#                   # TODO: Add VBO creation for rasterization pass if needed?
                    # Yes, we need to add to scene_batches!
#                   # Yes, we need to add to scene_batches!
                    # But SceneBatch assumes instancing.
#                   # But SceneBatch assumes instancing.
                    # Here we have a single instance (or merged geometry).
#                   # Here we have a single instance (or merged geometry).
                    # We can create a "Single Instance" batch.
#                   # We can create a "Single Instance" batch.
                    # Local positions = loaded vertices (untransformed).
#                   # Local positions = loaded vertices (untransformed).
                    # Instance transform = Identity (since we bake transform in vertex shader? No, we transform on CPU for SSBOs).
#                   # Instance transform = Identity (since we bake transform in vertex shader? No, we transform on CPU for SSBOs).
                    # Wait. For Rasterization (VBOs), we typically upload Local Geometry + Instance Transforms.
#                   # Wait. For Rasterization (VBOs), we typically upload Local Geometry + Instance Transforms.
                    # But here I am baking the transform for the SSBO (Ray Tracing).
#                   # But here I am baking the transform for the SSBO (Ray Tracing).
                    # For consistency, I should use the same transform for rasterization.
#                   # For consistency, I should use the same transform for rasterization.
                    # Or I can just upload the Transformed Geometry as Local Geometry and use Identity transform for the instance.
#                   # Or I can just upload the Transformed Geometry as Local Geometry and use Identity transform for the instance.
                    # That's simpler for now.
#                   # That's simpler for now.

                    # Create VBOs for this mesh
#                   # Create VBOs for this mesh
                    # Interleave data: pos(3), norm(3), tan(3), uv(2)
#                   # Interleave data: pos(3), norm(3), tan(3), uv(2)
                    # Vertices are already flattened (N_vertices, 3)
#                   # Vertices are already flattened (N_vertices, 3)
                    # Use transformed vertices! So vertex shader uses Identity model matrix.
#                   # Use transformed vertices! So vertex shader uses Identity model matrix.

                    mesh_data = np.hstack([transformed_vertices, transformed_normals, transformed_tangents, uvs]).astype("f4")
#                   mesh_data = np.hstack([transformed_vertices, transformed_normals, transformed_tangents, uvs]).astype("f4")
                    vbo_mesh = self.ctx.buffer(mesh_data.tobytes())
#                   vbo_mesh = self.ctx.buffer(mesh_data.tobytes())

                    # Instance Data: Identity Matrix + Material Index
#                   # Instance Data: Identity Matrix + Material Index
                    identity = np.eye(4, dtype="f4").flatten()
#                   identity = np.eye(4, dtype="f4").flatten()
                    inst_data = np.concatenate([identity, albedo[:3], [float(material_index)]]).astype("f4")
#                   inst_data = np.concatenate([identity, albedo[:3], [float(material_index)]]).astype("f4")
                    vbo_instance = self.ctx.buffer(inst_data.tobytes())
#                   vbo_instance = self.ctx.buffer(inst_data.tobytes())

                    vao = self.ctx.vertex_array(
#                   vao = self.ctx.vertex_array(
                        self.program_geometry,
#                       self.program_geometry,
                        [
#                       [
                            (vbo_mesh, "3f 3f 3f 2f", "inVertexLocalPosition", "inVertexLocalNormal", "inVertexLocalTangent", "inVertexLocalUV"),
#                           (vbo_mesh, "3f 3f 3f 2f", "inVertexLocalPosition", "inVertexLocalNormal", "inVertexLocalTangent", "inVertexLocalUV"),
                            (vbo_instance, "16f 3f 1x4/i", "inInstanceTransformModel", "inInstanceAlbedo"),
#                           (vbo_instance, "16f 3f 1x4/i", "inInstanceTransformModel", "inInstanceAlbedo"),
                        ],
#                       ],
                    )
#                   )
                    self.scene_batches.append(SceneBatch(vao=vao, number_of_instances=1, triangle_count_per_instance=n_triangles))
#                   self.scene_batches.append(SceneBatch(vao=vao, number_of_instances=1, triangle_count_per_instance=n_triangles))

        except Exception as e:
#       except Exception as e:
            print(f"Failed to load model {path}: {e}")
#           print(f"Failed to load model {path}: {e}")
            return
#           return

    def append_transformed_triangles(
#   def append_transformed_triangles(
        self,
#       self,
        instance_data_list: list[npt.NDArray[np.float32]],
#       instance_data_list: list[npt.NDArray[np.float32]],
        base_triangles: list[npt.NDArray[np.float32]],
#       base_triangles: list[npt.NDArray[np.float32]],
        base_normals: list[npt.NDArray[np.float32]],
#       base_normals: list[npt.NDArray[np.float32]],
        base_uvs: list[npt.NDArray[np.float32]],
#       base_uvs: list[npt.NDArray[np.float32]],
        base_tangents: list[npt.NDArray[np.float32]]
#       base_tangents: list[npt.NDArray[np.float32]]
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
                float(material["texture_index_albedo"]),
#               float(material["texture_index_albedo"]),
                float(material["texture_index_roughness"]),
#               float(material["texture_index_roughness"]),
                float(material["texture_index_metallic"]),
#               float(material["texture_index_metallic"]),
                float(material["texture_index_normal"]),
#               float(material["texture_index_normal"]),
            ], dtype="f4")
#           ], dtype="f4")

            for i, triangle_vertices in enumerate(base_triangles):
#           for i, triangle_vertices in enumerate(base_triangles):
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
                # Append UVs (no transformation needed usually for simple mapping)
#               # Append UVs (no transformation needed usually for simple mapping)
                self.scene_uvs.append(base_uvs[i])
#               self.scene_uvs.append(base_uvs[i])
                # Calculate Normal Matrix (Transpose of Inverse) to handle non-uniform scaling
#               # Calculate Normal Matrix (Transpose of Inverse) to handle non-uniform scaling
                m33 = model_matrix[:3, :3]
#               m33 = model_matrix[:3, :3]
                try:
#               try:
                    m33_inv = np.linalg.inv(m33)
#                   m33_inv = np.linalg.inv(m33)
                    normal_matrix = m33_inv.T
#                   normal_matrix = m33_inv.T
                except np.linalg.LinAlgError:
#               except np.linalg.LinAlgError:
                    normal_matrix = np.eye(3, dtype="f4")
#                   normal_matrix = np.eye(3, dtype="f4")
                triangle_normals = base_normals[i]
#               triangle_normals = base_normals[i]
                transformed_normals = []
#               transformed_normals = []
                for n in triangle_normals:
#               for n in triangle_normals:
                    Tn = normal_matrix @ n
#                   Tn = normal_matrix @ n
                    # Normalize
#                   # Normalize
                    norm = np.linalg.norm(Tn)
#                   norm = np.linalg.norm(Tn)
                    if norm > 1e-6:
#                   if norm > 1e-6:
                        Tn = Tn / norm
#                       Tn = Tn / norm
                    transformed_normals.append(Tn)
#                   transformed_normals.append(Tn)
                self.scene_normals.append(np.array(transformed_normals, dtype="f4"))
#               self.scene_normals.append(np.array(transformed_normals, dtype="f4"))

                # Tangents (Transform with model rotation)
#               # Tangents (Transform with model rotation)
                triangle_tangents = base_tangents[i]
#               triangle_tangents = base_tangents[i]
                transformed_tangents = []
#               transformed_tangents = []
                m33_rot = model_matrix[:3, :3] # Linear part
#               m33_rot = model_matrix[:3, :3] # Linear part
                for t in triangle_tangents:
#               for t in triangle_tangents:
                    Tt = m33_rot @ t
#                   Tt = m33_rot @ t
                    norm = np.linalg.norm(Tt)
#                   norm = np.linalg.norm(Tt)
                    if norm > 1e-6:
#                   if norm > 1e-6:
                        Tt = Tt / norm
#                       Tt = Tt / norm
                    transformed_tangents.append(Tt)
#                   transformed_tangents.append(Tt)
                self.scene_tangents.append(np.array(transformed_tangents, dtype="f4"))
#               self.scene_tangents.append(np.array(transformed_tangents, dtype="f4"))

    def build(self) -> tuple[bytes, bytes, bytes, bytes, bytes]:
#   def build(self) -> tuple[bytes, bytes, bytes, bytes, bytes]:
        def get_triangle_vertices(vertex0: vec3f32, vertex1: vec3f32, vertex2: vec3f32) -> npt.NDArray[np.float32]:
#       def get_triangle_vertices(vertex0: vec3f32, vertex1: vec3f32, vertex2: vec3f32) -> npt.NDArray[np.float32]:
            return np.array([vertex0, vertex1, vertex2], dtype="f4")
#           return np.array([vertex0, vertex1, vertex2], dtype="f4")

        def get_triangle_normals(normal0: vec3f32, normal1: vec3f32, normal2: vec3f32) -> npt.NDArray[np.float32]:
#       def get_triangle_normals(normal0: vec3f32, normal1: vec3f32, normal2: vec3f32) -> npt.NDArray[np.float32]:
            return np.array([normal0, normal1, normal2], dtype="f4")
#           return np.array([normal0, normal1, normal2], dtype="f4")

        def get_triangle_uvs(uv0: vec2f32, uv1: vec2f32, uv2: vec2f32) -> npt.NDArray[np.float32]:
#       def get_triangle_uvs(uv0: vec2f32, uv1: vec2f32, uv2: vec2f32) -> npt.NDArray[np.float32]:
            return np.array([uv0, uv1, uv2], dtype="f4")
#           return np.array([uv0, uv1, uv2], dtype="f4")

        def get_triangle_tangents(t0: vec3f32, t1: vec3f32, t2: vec3f32) -> npt.NDArray[np.float32]:
#       def get_triangle_tangents(t0: vec3f32, t1: vec3f32, t2: vec3f32) -> npt.NDArray[np.float32]:
            return np.array([t0, t1, t2], dtype="f4")
#           return np.array([t0, t1, t2], dtype="f4")

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
            # Front (z+)
#           # Front (z+)
            cube_base_triangles.append(get_triangle_vertices(cube_point0, cube_point1, cube_point2))
#           cube_base_triangles.append(get_triangle_vertices(cube_point0, cube_point1, cube_point2))
            cube_base_triangles.append(get_triangle_vertices(cube_point0, cube_point2, cube_point3))
#           cube_base_triangles.append(get_triangle_vertices(cube_point0, cube_point2, cube_point3))
            # Back (z-)
#           # Back (z-)
            cube_base_triangles.append(get_triangle_vertices(cube_point5, cube_point4, cube_point7))
#           cube_base_triangles.append(get_triangle_vertices(cube_point5, cube_point4, cube_point7))
            cube_base_triangles.append(get_triangle_vertices(cube_point5, cube_point7, cube_point6))
#           cube_base_triangles.append(get_triangle_vertices(cube_point5, cube_point7, cube_point6))
            # Left (x-)
#           # Left (x-)
            cube_base_triangles.append(get_triangle_vertices(cube_point4, cube_point0, cube_point3))
#           cube_base_triangles.append(get_triangle_vertices(cube_point4, cube_point0, cube_point3))
            cube_base_triangles.append(get_triangle_vertices(cube_point4, cube_point3, cube_point7))
#           cube_base_triangles.append(get_triangle_vertices(cube_point4, cube_point3, cube_point7))
            # Right (x+)
#           # Right (x+)
            cube_base_triangles.append(get_triangle_vertices(cube_point1, cube_point5, cube_point6))
#           cube_base_triangles.append(get_triangle_vertices(cube_point1, cube_point5, cube_point6))
            cube_base_triangles.append(get_triangle_vertices(cube_point1, cube_point6, cube_point2))
#           cube_base_triangles.append(get_triangle_vertices(cube_point1, cube_point6, cube_point2))
            # Top (y+)
#           # Top (y+)
            cube_base_triangles.append(get_triangle_vertices(cube_point3, cube_point2, cube_point6))
#           cube_base_triangles.append(get_triangle_vertices(cube_point3, cube_point2, cube_point6))
            cube_base_triangles.append(get_triangle_vertices(cube_point3, cube_point6, cube_point7))
#           cube_base_triangles.append(get_triangle_vertices(cube_point3, cube_point6, cube_point7))
            # Bottom (y-)
#           # Bottom (y-)
            cube_base_triangles.append(get_triangle_vertices(cube_point4, cube_point5, cube_point1))
#           cube_base_triangles.append(get_triangle_vertices(cube_point4, cube_point5, cube_point1))
            cube_base_triangles.append(get_triangle_vertices(cube_point4, cube_point1, cube_point0))
#           cube_base_triangles.append(get_triangle_vertices(cube_point4, cube_point1, cube_point0))

            # Cube UVs
#           # Cube UVs
            cube_uv00: vec2f32 = (0.0, 0.0)
#           cube_uv00: vec2f32 = (0.0, 0.0)
            cube_uv10: vec2f32 = (1.0, 0.0)
#           cube_uv10: vec2f32 = (1.0, 0.0)
            cube_uv11: vec2f32 = (1.0, 1.0)
#           cube_uv11: vec2f32 = (1.0, 1.0)
            cube_uv01: vec2f32 = (0.0, 1.0)
#           cube_uv01: vec2f32 = (0.0, 1.0)

            # Map faces to 0-1 range
#           # Map faces to 0-1 range
            cube_base_uvs: list[npt.NDArray[np.float32]] = []
#           cube_base_uvs: list[npt.NDArray[np.float32]] = []
            # Front (z+)
#           # Front (z+)
            cube_base_uvs.append(get_triangle_uvs(cube_uv00, cube_uv10, cube_uv11))
#           cube_base_uvs.append(get_triangle_uvs(cube_uv00, cube_uv10, cube_uv11))
            cube_base_uvs.append(get_triangle_uvs(cube_uv00, cube_uv11, cube_uv01))
#           cube_base_uvs.append(get_triangle_uvs(cube_uv00, cube_uv11, cube_uv01))
            # Back (z-)
#           # Back (z-)
            cube_base_uvs.append(get_triangle_uvs(cube_uv00, cube_uv10, cube_uv11)) # Point5, 4, 7
#           cube_base_uvs.append(get_triangle_uvs(cube_uv00, cube_uv10, cube_uv11)) # Point5, 4, 7
            cube_base_uvs.append(get_triangle_uvs(cube_uv00, cube_uv11, cube_uv01)) # Point5, 7, 6
#           cube_base_uvs.append(get_triangle_uvs(cube_uv00, cube_uv11, cube_uv01)) # Point5, 7, 6
            # Left (x-)
#           # Left (x-)
            cube_base_uvs.append(get_triangle_uvs(cube_uv00, cube_uv10, cube_uv11)) # Point4, 0, 3
#           cube_base_uvs.append(get_triangle_uvs(cube_uv00, cube_uv10, cube_uv11)) # Point4, 0, 3
            cube_base_uvs.append(get_triangle_uvs(cube_uv00, cube_uv11, cube_uv01)) # Point4, 3, 7
#           cube_base_uvs.append(get_triangle_uvs(cube_uv00, cube_uv11, cube_uv01)) # Point4, 3, 7
            # Right (x+)
#           # Right (x+)
            cube_base_uvs.append(get_triangle_uvs(cube_uv00, cube_uv10, cube_uv11)) # Point1, 5, 6
#           cube_base_uvs.append(get_triangle_uvs(cube_uv00, cube_uv10, cube_uv11)) # Point1, 5, 6
            cube_base_uvs.append(get_triangle_uvs(cube_uv00, cube_uv11, cube_uv01)) # Point1, 6, 2
#           cube_base_uvs.append(get_triangle_uvs(cube_uv00, cube_uv11, cube_uv01)) # Point1, 6, 2
            # Top (y+)
#           # Top (y+)
            cube_base_uvs.append(get_triangle_uvs(cube_uv00, cube_uv10, cube_uv11)) # Point3, 2, 6
#           cube_base_uvs.append(get_triangle_uvs(cube_uv00, cube_uv10, cube_uv11)) # Point3, 2, 6
            cube_base_uvs.append(get_triangle_uvs(cube_uv00, cube_uv11, cube_uv01)) # Point3, 6, 7
#           cube_base_uvs.append(get_triangle_uvs(cube_uv00, cube_uv11, cube_uv01)) # Point3, 6, 7
            # Bottom (y-)
#           # Bottom (y-)
            cube_base_uvs.append(get_triangle_uvs(cube_uv00, cube_uv10, cube_uv11)) # Point4, 5, 1
#           cube_base_uvs.append(get_triangle_uvs(cube_uv00, cube_uv10, cube_uv11)) # Point4, 5, 1
            cube_base_uvs.append(get_triangle_uvs(cube_uv00, cube_uv11, cube_uv01)) # Point4, 1, 0
#           cube_base_uvs.append(get_triangle_uvs(cube_uv00, cube_uv11, cube_uv01)) # Point4, 1, 0

            # Cube Normals (Face Normals for now)
#           # Cube Normals (Face Normals for now)
            cube_n_front: vec3f32 = (0.0, 0.0, 1.0)
#           cube_n_front: vec3f32 = (0.0, 0.0, 1.0)
            cube_n_back: vec3f32 = (0.0, 0.0, -1.0)
#           cube_n_back: vec3f32 = (0.0, 0.0, -1.0)
            cube_n_left: vec3f32 = (-1.0, 0.0, 0.0)
#           cube_n_left: vec3f32 = (-1.0, 0.0, 0.0)
            cube_n_right: vec3f32 = (1.0, 0.0, 0.0)
#           cube_n_right: vec3f32 = (1.0, 0.0, 0.0)
            cube_n_top: vec3f32 = (0.0, 1.0, 0.0)
#           cube_n_top: vec3f32 = (0.0, 1.0, 0.0)
            cube_n_bottom: vec3f32 = (0.0, -1.0, 0.0)
#           cube_n_bottom: vec3f32 = (0.0, -1.0, 0.0)

            cube_base_normals: list[npt.NDArray[np.float32]] = []
#           cube_base_normals: list[npt.NDArray[np.float32]] = []
            # Front (z+)
#           # Front (z+)
            cube_base_normals.append(get_triangle_normals(cube_n_front, cube_n_front, cube_n_front))
#           cube_base_normals.append(get_triangle_normals(cube_n_front, cube_n_front, cube_n_front))
            cube_base_normals.append(get_triangle_normals(cube_n_front, cube_n_front, cube_n_front))
#           cube_base_normals.append(get_triangle_normals(cube_n_front, cube_n_front, cube_n_front))
            # Back (z-)
#           # Back (z-)
            cube_base_normals.append(get_triangle_normals(cube_n_back, cube_n_back, cube_n_back))
#           cube_base_normals.append(get_triangle_normals(cube_n_back, cube_n_back, cube_n_back))
            cube_base_normals.append(get_triangle_normals(cube_n_back, cube_n_back, cube_n_back))
#           cube_base_normals.append(get_triangle_normals(cube_n_back, cube_n_back, cube_n_back))
            # Left (x-)
#           # Left (x-)
            cube_base_normals.append(get_triangle_normals(cube_n_left, cube_n_left, cube_n_left))
#           cube_base_normals.append(get_triangle_normals(cube_n_left, cube_n_left, cube_n_left))
            cube_base_normals.append(get_triangle_normals(cube_n_left, cube_n_left, cube_n_left))
#           cube_base_normals.append(get_triangle_normals(cube_n_left, cube_n_left, cube_n_left))
            # Right (x+)
#           # Right (x+)
            cube_base_normals.append(get_triangle_normals(cube_n_right, cube_n_right, cube_n_right))
#           cube_base_normals.append(get_triangle_normals(cube_n_right, cube_n_right, cube_n_right))
            cube_base_normals.append(get_triangle_normals(cube_n_right, cube_n_right, cube_n_right))
#           cube_base_normals.append(get_triangle_normals(cube_n_right, cube_n_right, cube_n_right))
            # Top (y+)
#           # Top (y+)
            cube_base_normals.append(get_triangle_normals(cube_n_top, cube_n_top, cube_n_top))
#           cube_base_normals.append(get_triangle_normals(cube_n_top, cube_n_top, cube_n_top))
            cube_base_normals.append(get_triangle_normals(cube_n_top, cube_n_top, cube_n_top))
#           cube_base_normals.append(get_triangle_normals(cube_n_top, cube_n_top, cube_n_top))
            # Bottom (y-)
#           # Bottom (y-)
            cube_base_normals.append(get_triangle_normals(cube_n_bottom, cube_n_bottom, cube_n_bottom))
#           cube_base_normals.append(get_triangle_normals(cube_n_bottom, cube_n_bottom, cube_n_bottom))
            cube_base_normals.append(get_triangle_normals(cube_n_bottom, cube_n_bottom, cube_n_bottom))
#           cube_base_normals.append(get_triangle_normals(cube_n_bottom, cube_n_bottom, cube_n_bottom))

            cube_base_tangents: list[npt.NDArray[np.float32]] = []
#           cube_base_tangents: list[npt.NDArray[np.float32]] = []
            for i in range(len(cube_base_triangles)):
#           for i in range(len(cube_base_triangles)):
                tri = cube_base_triangles[i]
#               tri = cube_base_triangles[i]
                uvs = cube_base_uvs[i]
#               uvs = cube_base_uvs[i]
                tan = self.calculate_triangle_tangent(
#               tan = self.calculate_triangle_tangent(
                    tri[0], tri[1], tri[2],
#                   tri[0], tri[1], tri[2],
                    uvs[0], uvs[1], uvs[2]
#                   uvs[0], uvs[1], uvs[2]
                )
#               )
                cube_base_tangents.append(get_triangle_tangents(tan, tan, tan))
#               cube_base_tangents.append(get_triangle_tangents(tan, tan, tan))

            self.append_transformed_triangles(instance_data_list=self.cube_instance_data, base_triangles=cube_base_triangles, base_normals=cube_base_normals, base_uvs=cube_base_uvs, base_tangents=cube_base_tangents)
#           self.append_transformed_triangles(instance_data_list=self.cube_instance_data, base_triangles=cube_base_triangles, base_normals=cube_base_normals, base_uvs=cube_base_uvs, base_tangents=cube_base_tangents)

            cube_geometries: list[npt.NDArray[np.float32]] = []
#           cube_geometries: list[npt.NDArray[np.float32]] = []
            cube_geometries.append(self.face(vertex0=cube_point0, vertex1=cube_point1, vertex2=cube_point2, vertex3=cube_point3, face_normal=(0.0, 0.0, 1.0), face_tangent=(1.0, 0.0, 0.0)))
#           cube_geometries.append(self.face(vertex0=cube_point0, vertex1=cube_point1, vertex2=cube_point2, vertex3=cube_point3, face_normal=(0.0, 0.0, 1.0), face_tangent=(1.0, 0.0, 0.0)))
            cube_geometries.append(self.face(vertex0=cube_point5, vertex1=cube_point4, vertex2=cube_point7, vertex3=cube_point6, face_normal=(0.0, 0.0, -1.0), face_tangent=(-1.0, 0.0, 0.0)))
#           cube_geometries.append(self.face(vertex0=cube_point5, vertex1=cube_point4, vertex2=cube_point7, vertex3=cube_point6, face_normal=(0.0, 0.0, -1.0), face_tangent=(-1.0, 0.0, 0.0)))
            cube_geometries.append(self.face(vertex0=cube_point4, vertex1=cube_point0, vertex2=cube_point3, vertex3=cube_point7, face_normal=(-1.0, 0.0, 0.0), face_tangent=(0.0, 0.0, 1.0)))
#           cube_geometries.append(self.face(vertex0=cube_point4, vertex1=cube_point0, vertex2=cube_point3, vertex3=cube_point7, face_normal=(-1.0, 0.0, 0.0), face_tangent=(0.0, 0.0, 1.0)))
            cube_geometries.append(self.face(vertex0=cube_point1, vertex1=cube_point5, vertex2=cube_point6, vertex3=cube_point2, face_normal=(1.0, 0.0, 0.0), face_tangent=(0.0, 0.0, -1.0)))
#           cube_geometries.append(self.face(vertex0=cube_point1, vertex1=cube_point5, vertex2=cube_point6, vertex3=cube_point2, face_normal=(1.0, 0.0, 0.0), face_tangent=(0.0, 0.0, -1.0)))
            cube_geometries.append(self.face(vertex0=cube_point3, vertex1=cube_point2, vertex2=cube_point6, vertex3=cube_point7, face_normal=(0.0, 1.0, 0.0), face_tangent=(1.0, 0.0, 0.0)))
#           cube_geometries.append(self.face(vertex0=cube_point3, vertex1=cube_point2, vertex2=cube_point6, vertex3=cube_point7, face_normal=(0.0, 1.0, 0.0), face_tangent=(1.0, 0.0, 0.0)))
            cube_geometries.append(self.face(vertex0=cube_point4, vertex1=cube_point5, vertex2=cube_point1, vertex3=cube_point0, face_normal=(0.0, -1.0, 0.0), face_tangent=(1.0, 0.0, 0.0)))
#           cube_geometries.append(self.face(vertex0=cube_point4, vertex1=cube_point5, vertex2=cube_point1, vertex3=cube_point0, face_normal=(0.0, -1.0, 0.0), face_tangent=(1.0, 0.0, 0.0)))

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
                    (cube_vbo_mesh, "3f 3f 3f 2f", "inVertexLocalPosition", "inVertexLocalNormal", "inVertexLocalTangent", "inVertexLocalUV"),
#                   (cube_vbo_mesh, "3f 3f 3f 2f", "inVertexLocalPosition", "inVertexLocalNormal", "inVertexLocalTangent", "inVertexLocalUV"),
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

            # Plane UVs (tiled 20x20 since plane is large!)
#           # Plane UVs (tiled 20x20 since plane is large!)
            # Or just 1x1. Let's do 10x10 tiling
#           # Or just 1x1. Let's do 10x10 tiling
            tiling = 10.0
#           tiling = 10.0
            plane_uv0: vec2f32 = (0.0, 0.0)
#           plane_uv0: vec2f32 = (0.0, 0.0)
            plane_uv1: vec2f32 = (tiling, 0.0)
#           plane_uv1: vec2f32 = (tiling, 0.0)
            plane_uv2: vec2f32 = (tiling, tiling)
#           plane_uv2: vec2f32 = (tiling, tiling)
            plane_uv3: vec2f32 = (0.0, tiling)
#           plane_uv3: vec2f32 = (0.0, tiling)

            plane_base_uvs: list[npt.NDArray[np.float32]] = []
#           plane_base_uvs: list[npt.NDArray[np.float32]] = []
            plane_base_uvs.append(get_triangle_uvs(plane_uv0, plane_uv1, plane_uv2))
#           plane_base_uvs.append(get_triangle_uvs(plane_uv0, plane_uv1, plane_uv2))
            plane_base_uvs.append(get_triangle_uvs(plane_uv0, plane_uv2, plane_uv3))
#           plane_base_uvs.append(get_triangle_uvs(plane_uv0, plane_uv2, plane_uv3))

            plane_n: vec3f32 = (0.0, 1.0, 0.0)
#           plane_n: vec3f32 = (0.0, 1.0, 0.0)
            plane_base_normals: list[npt.NDArray[np.float32]] = []
#           plane_base_normals: list[npt.NDArray[np.float32]] = []
            plane_base_normals.append(get_triangle_normals(plane_n, plane_n, plane_n))
#           plane_base_normals.append(get_triangle_normals(plane_n, plane_n, plane_n))
            plane_base_normals.append(get_triangle_normals(plane_n, plane_n, plane_n))
#           plane_base_normals.append(get_triangle_normals(plane_n, plane_n, plane_n))

            plane_base_tangents: list[npt.NDArray[np.float32]] = []
#           plane_base_tangents: list[npt.NDArray[np.float32]] = []
            for i in range(len(plane_base_triangles)):
#           for i in range(len(plane_base_triangles)):
                tri = plane_base_triangles[i]
#               tri = plane_base_triangles[i]
                uvs = plane_base_uvs[i]
#               uvs = plane_base_uvs[i]
                tan = self.calculate_triangle_tangent(
#               tan = self.calculate_triangle_tangent(
                    tri[0], tri[1], tri[2],
#                   tri[0], tri[1], tri[2],
                    uvs[0], uvs[1], uvs[2]
#                   uvs[0], uvs[1], uvs[2]
                )
#               )
                plane_base_tangents.append(get_triangle_tangents(tan, tan, tan))
#               plane_base_tangents.append(get_triangle_tangents(tan, tan, tan))

            self.append_transformed_triangles(instance_data_list=self.plane_instance_data, base_triangles=plane_base_triangles, base_normals=plane_base_normals, base_uvs=plane_base_uvs, base_tangents=plane_base_tangents)
#           self.append_transformed_triangles(instance_data_list=self.plane_instance_data, base_triangles=plane_base_triangles, base_normals=plane_base_normals, base_uvs=plane_base_uvs, base_tangents=plane_base_tangents)

            plane_geometries: list[npt.NDArray[np.float32]] = []
#           plane_geometries: list[npt.NDArray[np.float32]] = []
            plane_geometries.append(self.face(vertex0=plane_point0, vertex1=plane_point1, vertex2=plane_point2, vertex3=plane_point3, face_normal=(0.0, 1.0, 0.0), face_tangent=(1.0, 0.0, 0.0), uv0=plane_uv0, uv1=plane_uv1, uv2=plane_uv2, uv3=plane_uv3))
#           plane_geometries.append(self.face(vertex0=plane_point0, vertex1=plane_point1, vertex2=plane_point2, vertex3=plane_point3, face_normal=(0.0, 1.0, 0.0), face_tangent=(1.0, 0.0, 0.0), uv0=plane_uv0, uv1=plane_uv1, uv2=plane_uv2, uv3=plane_uv3))

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
                    (plane_vbo_mesh, "3f 3f 3f 2f", "inVertexLocalPosition", "inVertexLocalNormal", "inVertexLocalTangent", "inVertexLocalUV"),
#                   (plane_vbo_mesh, "3f 3f 3f 2f", "inVertexLocalPosition", "inVertexLocalNormal", "inVertexLocalTangent", "inVertexLocalUV"),
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

        world_uvs: npt.NDArray[np.float32] = np.array(self.scene_uvs, dtype="f4")
#       world_uvs: npt.NDArray[np.float32] = np.array(self.scene_uvs, dtype="f4")

        world_normals: npt.NDArray[np.float32] = np.array(self.scene_normals, dtype="f4")
#       world_normals: npt.NDArray[np.float32] = np.array(self.scene_normals, dtype="f4")

        world_tangents: npt.NDArray[np.float32] = np.array(self.scene_tangents, dtype="f4")
#       world_tangents: npt.NDArray[np.float32] = np.array(self.scene_tangents, dtype="f4")

        return bvh_data, world_triangles.flatten().tobytes(), world_materials.flatten().tobytes(), world_uvs.flatten().tobytes(), world_normals.flatten().tobytes(), world_tangents.flatten().tobytes()
#       return bvh_data, world_triangles.flatten().tobytes(), world_materials.flatten().tobytes(), world_uvs.flatten().tobytes(), world_normals.flatten().tobytes(), world_tangents.flatten().tobytes()
