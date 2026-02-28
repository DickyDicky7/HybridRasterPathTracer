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
from src.core.common_types import vec2f32, vec3f32, vec4f32, Material
from src.core.common_types import vec2f32, vec3f32, vec4f32, Material

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
    # The SceneBuilder is the central coordinator for preparing geometry for the Hybrid Renderer.
#   # The SceneBuilder is the central coordinator for preparing geometry for the Hybrid Renderer.
    # It has two main outputs:
#   # It has two main outputs:
    # 1. Rasterization Resources: VAOs/VBOs for the primary G-Buffer pass (OpenGL state).
#   # 1. Rasterization Resources: VAOs/VBOs for the primary G-Buffer pass (OpenGL state).
    # 2. Ray Tracing Resources: Compact packed float arrays (triangles, normals, BVH nodes) for Compute Shaders.
#   # 2. Ray Tracing Resources: Compact packed float arrays (triangles, normals, BVH nodes) for Compute Shaders.
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
        # We need to find a vector 'T' (Tangent) such that T aligns with the U-axis of the texture map.
#       # We need to find a vector 'T' (Tangent) such that T aligns with the U-axis of the texture map.
        # This allows us to construct a TBN matrix (Tangent-Bitangent-Normal) per pixel for Normal Mapping.
#       # This allows us to construct a TBN matrix (Tangent-Bitangent-Normal) per pixel for Normal Mapping.

        # Calculate the edges of the triangle in position space (E1, E2)
#       # Calculate the edges of the triangle in position space (E1, E2)
        edge1 = np.array(v1) - np.array(v0)
#       edge1 = np.array(v1) - np.array(v0)
        edge2 = np.array(v2) - np.array(v0)
#       edge2 = np.array(v2) - np.array(v0)

        # Calculate the edges of the triangle in texture space (deltaUV1, deltaUV2)
#       # Calculate the edges of the triangle in texture space (deltaUV1, deltaUV2)
        deltaUV1 = np.array(uv1) - np.array(uv0)
#       deltaUV1 = np.array(uv1) - np.array(uv0)
        deltaUV2 = np.array(uv2) - np.array(uv0)
#       deltaUV2 = np.array(uv2) - np.array(uv0)

        # Calculate the fractional part of the equation (inverse determinant of the UV matrix)
#       # Calculate the fractional part of the equation (inverse determinant of the UV matrix)
        # This factor accounts for the area distortion between texture space and object space.
#       # This factor accounts for the area distortion between texture space and object space.
        # f = 1.0 / (du1 * dv2 - du2 * dv1)
#       # f = 1.0 / (du1 * dv2 - du2 * dv1)
        det = deltaUV1[0] * deltaUV2[1] - deltaUV2[0] * deltaUV1[1]
#       det = deltaUV1[0] * deltaUV2[1] - deltaUV2[0] * deltaUV1[1]
        f = 1.0 / det if abs(det) > 1e-6 else 0.0
#       f = 1.0 / det if abs(det) > 1e-6 else 0.0

        # Solve for the tangent vector (T)
#       # Solve for the tangent vector (T)
        # T = f * (dv2 * E1 - dv1 * E2)
#       # T = f * (dv2 * E1 - dv1 * E2)
        tangent = np.array([
#       tangent = np.array([
            f * (deltaUV2[1] * edge1[0] - deltaUV1[1] * edge2[0]),
#           f * (deltaUV2[1] * edge1[0] - deltaUV1[1] * edge2[0]),
            f * (deltaUV2[1] * edge1[1] - deltaUV1[1] * edge2[1]),
#           f * (deltaUV2[1] * edge1[1] - deltaUV1[1] * edge2[1]),
            f * (deltaUV2[1] * edge1[2] - deltaUV1[1] * edge2[2])
#           f * (deltaUV2[1] * edge1[2] - deltaUV1[1] * edge2[2])
        ], dtype=np.float32)
#       ], dtype=np.float32)

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
        ], dtype=np.float32)
#       ], dtype=np.float32)

    def add_cube(self, position: vec3f32, rotation: vec3f32, scale: vec3f32, material_index: int) -> None:
#   def add_cube(self, position: vec3f32, rotation: vec3f32, scale: vec3f32, material_index: int) -> None:
        matrix_translation: rr.Matrix44 = rr.Matrix44.from_translation(position)
#       matrix_translation: rr.Matrix44 = rr.Matrix44.from_translation(position)
        matrix_rotation: rr.Matrix44 = rr.Matrix44.from_eulers(rotation)
#       matrix_rotation: rr.Matrix44 = rr.Matrix44.from_eulers(rotation)
        matrix_scale: rr.Matrix44 = rr.Matrix44.from_scale(scale)
#       matrix_scale: rr.Matrix44 = rr.Matrix44.from_scale(scale)
        matrix: npt.NDArray[np.float32] = (matrix_translation * matrix_rotation * matrix_scale).astype(dtype=np.float32)
#       matrix: npt.NDArray[np.float32] = (matrix_translation * matrix_rotation * matrix_scale).astype(dtype=np.float32)
        albedo: vec3f32 = self.materials[material_index]["albedo"]
#       albedo: vec3f32 = self.materials[material_index]["albedo"]
        if len(albedo) == 4: albedo = albedo[:3]
#       if len(albedo) == 4: albedo = albedo[:3]
        data: npt.NDArray[np.float32] = np.concatenate([matrix.flatten(), albedo, [float(material_index)]]).astype(dtype=np.float32)
#       data: npt.NDArray[np.float32] = np.concatenate([matrix.flatten(), albedo, [float(material_index)]]).astype(dtype=np.float32)
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
        matrix: npt.NDArray[np.float32] = (matrix_translation * matrix_rotation * matrix_scale).astype(dtype=np.float32)
#       matrix: npt.NDArray[np.float32] = (matrix_translation * matrix_rotation * matrix_scale).astype(dtype=np.float32)
        albedo: vec3f32 = self.materials[material_index]["albedo"]
#       albedo: vec3f32 = self.materials[material_index]["albedo"]
        if len(albedo) == 4: albedo = albedo[:3]
#       if len(albedo) == 4: albedo = albedo[:3]
        data: npt.NDArray[np.float32] = np.concatenate([matrix.flatten(), albedo, [float(material_index)]]).astype(dtype=np.float32)
#       data: npt.NDArray[np.float32] = np.concatenate([matrix.flatten(), albedo, [float(material_index)]]).astype(dtype=np.float32)
        self.plane_instance_data.append(data)
#       self.plane_instance_data.append(data)

    def load_model(self, path: str, position: vec3f32, rotation: vec3f32, scale: vec3f32, material_indices: list[int] | int) -> None:
#   def load_model(self, path: str, position: vec3f32, rotation: vec3f32, scale: vec3f32, material_indices: list[int] | int) -> None:
        # Construct the Model Matrix (T * R * S) to transform vertices from local to world space
#       # Construct the Model Matrix (T * R * S) to transform vertices from local to world space
        matrix_translation: rr.Matrix44 = rr.Matrix44.from_translation(position)
#       matrix_translation: rr.Matrix44 = rr.Matrix44.from_translation(position)
        matrix_rotation: rr.Matrix44 = rr.Matrix44.from_eulers(rotation)
#       matrix_rotation: rr.Matrix44 = rr.Matrix44.from_eulers(rotation)
        matrix_scale: rr.Matrix44 = rr.Matrix44.from_scale(scale)
#       matrix_scale: rr.Matrix44 = rr.Matrix44.from_scale(scale)
        model_matrix: npt.NDArray[np.float32] = (matrix_translation * matrix_rotation * matrix_scale).astype(dtype=np.float32)
#       model_matrix: npt.NDArray[np.float32] = (matrix_translation * matrix_rotation * matrix_scale).astype(dtype=np.float32)

        # Calculate Normal Matrix to correctly transform normals (handles non-uniform scaling)
#       # Calculate Normal Matrix to correctly transform normals (handles non-uniform scaling)
        # If we scale a sphere by (1, 0.5, 1), normals pointing "up" would flatten if multiplied by the Model Matrix directly.
#       # If we scale a sphere by (1, 0.5, 1), normals pointing "up" would flatten if multiplied by the Model Matrix directly.
        # The Normal Matrix is the Transpose of the Inverse of the upper 3x3 model matrix
#       # The Normal Matrix is the Transpose of the Inverse of the upper 3x3 model matrix
        # This restores their perpendicularity to the surface ("Covariant" vs "Contravariant" vector transformation).
#       # This restores their perpendicularity to the surface ("Covariant" vs "Contravariant" vector transformation).
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
            normal_matrix = np.eye(3, dtype=np.float32)
#           normal_matrix = np.eye(3, dtype=np.float32)

        try:
#       try:
            # Load the model using Assimp with extensive post-processing for optimization and standardization
#           # Load the model using Assimp with extensive post-processing for optimization and standardization
            # - aiProcess_Triangulate: Ensure all geometry is triangles (no quads/polys).
#           # - aiProcess_Triangulate: Ensure all geometry is triangles (no quads/polys).
            # - aiProcess_CalcTangentSpace: specific method to generate tangents for normal mapping.
#           # - aiProcess_CalcTangentSpace: specific method to generate tangents for normal mapping.
            # - aiProcess_GenSmoothNormals: generate vertex normals if missing.
#           # - aiProcess_GenSmoothNormals: generate vertex normals if missing.
            # - aiProcess_PreTransformVertices: bake the scene graph hierarchy into the vertices (simplifies rendering).
#           # - aiProcess_PreTransformVertices: bake the scene graph hierarchy into the vertices (simplifies rendering).
            #   This creates a static World Space mesh, which is essential for our simplified LBVH builder (no TLAS/BLAS yet).
#           #   This creates a static World Space mesh, which is essential for our simplified LBVH builder (no TLAS/BLAS yet).
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

                # Iterate over all meshes in the loaded scene
#               # Iterate over all meshes in the loaded scene
                for mesh in scene.meshes:
#               for mesh in scene.meshes:
                    # Determine the correct material index for this mesh, handling potential mismatches with the provided indices
#                   # Determine the correct material index for this mesh, handling potential mismatches with the provided indices
                    current_material_index: int = 0
#                   current_material_index: int = 0
                    if isinstance(material_indices, int):
#                   if isinstance(material_indices, int):
                        current_material_index = material_indices
#                       current_material_index = material_indices
                    elif isinstance(material_indices, list):
#                   elif isinstance(material_indices, list):
                        # Ensure we don't access an out-of-bounds index if the mesh's material index is unexpected
#                       # Ensure we don't access an out-of-bounds index if the mesh's material index is unexpected
                        if mesh.materialindex < len(material_indices):
#                       if mesh.materialindex < len(material_indices):
                            current_material_index = material_indices[mesh.materialindex]
#                           current_material_index = material_indices[mesh.materialindex]
                        else:
#                       else:
                            print(f"Warning: Mesh material index {mesh.materialindex} out of bounds for provided list.")
#                           print(f"Warning: Mesh material index {mesh.materialindex} out of bounds for provided list.")
                            current_material_index = material_indices[0] # Fallback
#                           current_material_index = material_indices[0] # Fallback

                    # Prepare material data for GPU upload
#                   # Prepare material data for GPU upload
                    # This involves packing material properties (albedo, roughness, metallic, etc.) into a float array
#                   # This involves packing material properties (albedo, roughness, metallic, etc.) into a float array
                    material: Material = self.materials[current_material_index]
#                   material: Material = self.materials[current_material_index]
                    albedo: vec4f32 = (*material["albedo"], 0.0)
#                   albedo: vec4f32 = (*material["albedo"], 0.0)
                    material_data: npt.NDArray[np.float32] = np.array([
#                   material_data: npt.NDArray[np.float32] = np.array([
                        albedo[0], albedo[1], albedo[2], 0.0,
#                       albedo[0], albedo[1], albedo[2], 0.0,
                        material["roughness"],
#                       material["roughness"],
                        material["metallic"],
#                       material["metallic"],
                        material["transmission"],
#                       material["transmission"],
                        material["ior"],
#                       material["ior"],
                        material["texture_index_albedo"],
#                       material["texture_index_albedo"],
                        material["texture_index_roughness"],
#                       material["texture_index_roughness"],
                        material["texture_index_metallic"],
#                       material["texture_index_metallic"],
                        material["texture_index_normal"],
#                       material["texture_index_normal"],
                        material["emissive"],
#                       material["emissive"],
                        material["texture_index_emissive"],
#                       material["texture_index_emissive"],
                        0.0,
#                       0.0,
                        0.0,
#                       0.0,
                    ], dtype=np.float32)
#                   ], dtype=np.float32)

                    # Flatten the mesh indices for processing
#                   # Flatten the mesh indices for processing
                    # Assimp might provide indices per face (e.g., list of 3 ints), so we flatten them into a single 1D array
#                   # Assimp might provide indices per face (e.g., list of 3 ints), so we flatten them into a single 1D array
                    indices = mesh.faces.flatten() if hasattr(mesh.faces, 'flatten') else np.array(mesh.faces).flatten()
#                   indices = mesh.faces.flatten() if hasattr(mesh.faces, 'flatten') else np.array(mesh.faces).flatten()

                    # Extract Vertices using the flattened indices
#                   # Extract Vertices using the flattened indices
                    vertices = mesh.vertices[indices]
#                   vertices = mesh.vertices[indices]

                    # Extract Normals
#                   # Extract Normals
                    normals = mesh.normals[indices]
#                   normals = mesh.normals[indices]

                    # Extract Tangents (needed for normal mapping)
#                   # Extract Tangents (needed for normal mapping)
                    if hasattr(mesh, 'tangents') and len(mesh.tangents) > 0:
#                   if hasattr(mesh, 'tangents') and len(mesh.tangents) > 0:
                        tangents = mesh.tangents[indices]
#                       tangents = mesh.tangents[indices]
                    else:
#                   else:
                        # Fallback: No tangents found in the mesh source.
#                       # Fallback: No tangents found in the mesh source.
                        # Assign a default tangent vector (1, 0, 0) to ensure the shader receives valid data.
#                       # Assign a default tangent vector (1, 0, 0) to ensure the shader receives valid data.
                        # In a production pipeline, MikkTSpace generation should be preferred here.
#                       # In a production pipeline, MikkTSpace generation should be preferred here.
                        tangents = np.zeros_like(vertices)
#                       tangents = np.zeros_like(vertices)
                        tangents[:, 0] = 1.0
#                       tangents[:, 0] = 1.0

                    # Extract Texture Coordinates (UVs)
#                   # Extract Texture Coordinates (UVs)
                    # Assimp supports multiple UV channels (layers). We typically use channel 0 for albedo/normal maps.
#                   # Assimp supports multiple UV channels (layers). We typically use channel 0 for albedo/normal maps.
                    # The raw data is (N, 3), but we only need (u, v), so we slice the first two components.
#                   # The raw data is (N, 3), but we only need (u, v), so we slice the first two components.
                    if hasattr(mesh, 'texturecoords') and len(mesh.texturecoords) > 0:
#                   if hasattr(mesh, 'texturecoords') and len(mesh.texturecoords) > 0:
                        uvs_raw = mesh.texturecoords[0][indices] # (N, 3)
#                       uvs_raw = mesh.texturecoords[0][indices] # (N, 3)
                        uvs = uvs_raw[:, :2] # (N, 2)
#                       uvs = uvs_raw[:, :2] # (N, 2)
                    else:
#                   else:
                        uvs = np.zeros((len(vertices), 2), dtype=np.float32)
#                       uvs = np.zeros((len(vertices), 2), dtype=np.float32)

                    # Transform Geometry to World Space and Append to Scene Arrays
#                   # Transform Geometry to World Space and Append to Scene Arrays

                    # Transform Vertices by the Model Matrix
#                   # Transform Vertices by the Model Matrix
                    ones = np.ones((len(vertices), 1), dtype=np.float32)
#                   ones = np.ones((len(vertices), 1), dtype=np.float32)
                    vertices_h = np.hstack([vertices, ones])
#                   vertices_h = np.hstack([vertices, ones])
                    transformed_vertices_h = vertices_h @ model_matrix # (N, 4)
#                   transformed_vertices_h = vertices_h @ model_matrix # (N, 4)
                    transformed_vertices = transformed_vertices_h[:, :3] # (N, 3)
#                   transformed_vertices = transformed_vertices_h[:, :3] # (N, 3)

                    # Transform Normals by the Normal Matrix (inverse transpose of upper 3x3)
#                   # Transform Normals by the Normal Matrix (inverse transpose of upper 3x3)
                    transformed_normals = normals @ normal_matrix # (N, 3)
#                   transformed_normals = normals @ normal_matrix # (N, 3)
                    # Re-normalize normals after transformation to correct for potential scaling
#                   # Re-normalize normals after transformation to correct for potential scaling
                    norms = np.linalg.norm(transformed_normals, axis=1, keepdims=True)
#                   norms = np.linalg.norm(transformed_normals, axis=1, keepdims=True)
                    # Avoid division by zero for degenerate normals
#                   # Avoid division by zero for degenerate normals
                    valid_norms = norms > 1e-6
#                   valid_norms = norms > 1e-6
                    transformed_normals[valid_norms.flatten()] /= norms[valid_norms.flatten()]
#                   transformed_normals[valid_norms.flatten()] /= norms[valid_norms.flatten()]

                    # Transform Tangents
#                   # Transform Tangents
                    # Tangents generally rotate with the model. We use the upper 3x3 of the model matrix (linear part).
#                   # Tangents generally rotate with the model. We use the upper 3x3 of the model matrix (linear part).
                    # Note: For non-uniform scaling, using the Normal Matrix might be more correct for orthogonality,
#                   # Note: For non-uniform scaling, using the Normal Matrix might be more correct for orthogonality,
                    # but using the linear part is a standard approximation if the mesh is reasonably behaved.
#                   # but using the linear part is a standard approximation if the mesh is reasonably behaved.
                    transformed_tangents = tangents @ model_matrix[:3, :3] # (N, 3)
#                   transformed_tangents = tangents @ model_matrix[:3, :3] # (N, 3)
                    # Re-normalize tangents
#                   # Re-normalize tangents
                    tan_norms = np.linalg.norm(transformed_tangents, axis=1, keepdims=True)
#                   tan_norms = np.linalg.norm(transformed_tangents, axis=1, keepdims=True)
                    valid_tan_norms = tan_norms > 1e-6
#                   valid_tan_norms = tan_norms > 1e-6
                    transformed_tangents[valid_tan_norms.flatten()] /= tan_norms[valid_tan_norms.flatten()]
#                   transformed_tangents[valid_tan_norms.flatten()] /= tan_norms[valid_tan_norms.flatten()]

                    # Reshape geometry into triangle primitives (N_triangles, 3 vertices, components) for easy appending
#                   # Reshape geometry into triangle primitives (N_triangles, 3 vertices, components) for easy appending
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
                        self.scene_triangles.append(tri_vertices[i].astype(dtype=np.float32))
#                       self.scene_triangles.append(tri_vertices[i].astype(dtype=np.float32))
                        self.scene_normals.append(tri_normals[i].astype(dtype=np.float32))
#                       self.scene_normals.append(tri_normals[i].astype(dtype=np.float32))
                        self.scene_tangents.append(tri_tangents[i].astype(dtype=np.float32))
#                       self.scene_tangents.append(tri_tangents[i].astype(dtype=np.float32))
                        self.scene_uvs.append(tri_uvs[i].astype(dtype=np.float32))
#                       self.scene_uvs.append(tri_uvs[i].astype(dtype=np.float32))
                        self.scene_materials.append(material_data)
#                       self.scene_materials.append(material_data)

                    # Create VBOs for the rasterization pass to support hybrid rendering.
#                   # Create VBOs for the rasterization pass to support hybrid rendering.
                    # 1. Rasterization: Uses these VBOs for fast primary visibility (G-Buffer).
#                   # 1. Rasterization: Uses these VBOs for fast primary visibility (G-Buffer).
                    # 2. Ray Tracing: Uses the SSBOs (built later from self.scene_*) for reflections/GI.
#                   # 2. Ray Tracing: Uses the SSBOs (built later from self.scene_*) for reflections/GI.

                    # Since SceneBatch is designed for instancing, we treat this merged model as a single-instance batch.
#                   # Since SceneBatch is designed for instancing, we treat this merged model as a single-instance batch.
                    # While rasterization typically uses local geometry + instance transforms, we have already baked
#                   # While rasterization typically uses local geometry + instance transforms, we have already baked
                    # the transforms into the vertices for the Ray Tracing SSBOs (to simplify BVH construction).
#                   # the transforms into the vertices for the Ray Tracing SSBOs (to simplify BVH construction).
                    # To maintain consistency and avoid duplicating geometry buffers, we upload the
#                   # To maintain consistency and avoid duplicating geometry buffers, we upload the
                    # Pre-Transformed Geometry as the "local" geometry for the VBO.
#                   # Pre-Transformed Geometry as the "local" geometry for the VBO.
                    # Consequently, the Instance Transform passed to the shader will be the Identity Matrix.
#                   # Consequently, the Instance Transform passed to the shader will be the Identity Matrix.

                    # Prepare VBO data: Interleaved [Position(3), Normal(3), Tangent(3), UV(2)]
#                   # Prepare VBO data: Interleaved [Position(3), Normal(3), Tangent(3), UV(2)]
                    # Note: Vertices are already flattened (N, 3) and transformed to World Space.
#                   # Note: Vertices are already flattened (N, 3) and transformed to World Space.

                    mesh_data = np.hstack([transformed_vertices, transformed_normals, transformed_tangents, uvs]).astype(dtype=np.float32)
#                   mesh_data = np.hstack([transformed_vertices, transformed_normals, transformed_tangents, uvs]).astype(dtype=np.float32)
                    vbo_mesh = self.ctx.buffer(mesh_data.tobytes())
#                   vbo_mesh = self.ctx.buffer(mesh_data.tobytes())

                    # Instance Data: Identity Matrix + Material Index
#                   # Instance Data: Identity Matrix + Material Index
                    identity = np.eye(4, dtype=np.float32).flatten()
#                   identity = np.eye(4, dtype=np.float32).flatten()
                    inst_data = np.concatenate([identity, albedo[:3], [float(current_material_index)]]).astype(dtype=np.float32)
#                   inst_data = np.concatenate([identity, albedo[:3], [float(current_material_index)]]).astype(dtype=np.float32)
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
        # Iterate over all instances of this geometry to bake them into world space for the BVH.
#       # Iterate over all instances of this geometry to bake them into world space for the BVH.
        # Unlike Rasterization which uses a small VBO + Instance Transform Buffer,
#       # Unlike Rasterization which uses a small VBO + Instance Transform Buffer,
        # Ray Tracing (currently) needs all geometry explicitly in World Space for the LBVH construction.
#       # Ray Tracing (currently) needs all geometry explicitly in World Space for the LBVH construction.
        for instance_data in instance_data_list:
#       for instance_data in instance_data_list:
            # Reconstruct the 4x4 Model Matrix from the flattened instance data (stored in column-major order)
#           # Reconstruct the 4x4 Model Matrix from the flattened instance data (stored in column-major order)
            model_matrix_flat: npt.NDArray[np.float32] = instance_data[:16]
#           model_matrix_flat: npt.NDArray[np.float32] = instance_data[:16]
            model_matrix: npt.NDArray[np.float32] = model_matrix_flat.reshape((4, 4), order='F')
#           model_matrix: npt.NDArray[np.float32] = model_matrix_flat.reshape((4, 4), order='F')

            # Retrieve material parameters using the material index stored in the instance data (float at index 19)
#           # Retrieve material parameters using the material index stored in the instance data (float at index 19)
            material_index = int(instance_data[19])
#           material_index = int(instance_data[19])
            material: Material = self.materials[material_index]
#           material: Material = self.materials[material_index]

            albedo: vec4f32 = (*material["albedo"], 0.0) # Pad to vec4
#           albedo: vec4f32 = (*material["albedo"], 0.0) # Pad to vec4

            # Prepare Material Data for the SSBO
#           # Prepare Material Data for the SSBO
            # The layout must match the 'Material' struct definition in the shader (std430 alignment)
#           # The layout must match the 'Material' struct definition in the shader (std430 alignment)
            # struct Material {
#           # struct Material {
            #     vec4 albedo; // .w is unused/padding
#           #     vec4 albedo; // .w is unused/padding
            #     float roughness;
#           #     float roughness;
            #     float metallic;
#           #     float metallic;
            #     float transmission;
#           #     float transmission;
            #     float ior;
#           #     float ior;
            #     float texture_index_albedo;
#           #     float texture_index_albedo;
            #     float texture_index_roughness;
#           #     float texture_index_roughness;
            #     float texture_index_metallic;
#           #     float texture_index_metallic;
            #     float texture_index_normal;
#           #     float texture_index_normal;
            #     float emissive;
#           #     float emissive;
            #     float texture_index_emissive;
#           #     float texture_index_emissive;
            #     float padding001;
#           #     float padding001;
            #     float padding002;
#           #     float padding002;
            # };
#           # };
            # Data Layout: [r, g, b, pad, roughness, metallic, transmission, ior, texture_index_albedo, texture_index_roughness, texture_index_metallic, texture_index_normal, emissive, texture_index_emissive, pad, pad]
#           # Data Layout: [r, g, b, pad, roughness, metallic, transmission, ior, texture_index_albedo, texture_index_roughness, texture_index_metallic, texture_index_normal, emissive, texture_index_emissive, pad, pad]
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
                material["texture_index_albedo"],
#               material["texture_index_albedo"],
                material["texture_index_roughness"],
#               material["texture_index_roughness"],
                material["texture_index_metallic"],
#               material["texture_index_metallic"],
                material["texture_index_normal"],
#               material["texture_index_normal"],
                material["emissive"],
#               material["emissive"],
                material["texture_index_emissive"],
#               material["texture_index_emissive"],
                0.0,
#               0.0,
                0.0,
#               0.0,
            ], dtype=np.float32)
#           ], dtype=np.float32)

            for i, triangle_vertices in enumerate(base_triangles):
#           for i, triangle_vertices in enumerate(base_triangles):
                ones: npt.NDArray[np.float32] = np.ones((3, 1), dtype=np.float32)
#               ones: npt.NDArray[np.float32] = np.ones((3, 1), dtype=np.float32)
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
                # Normals are directions, not positions. Using the Model Matrix directly on normals would cause distortion
#               # Normals are directions, not positions. Using the Model Matrix directly on normals would cause distortion
                # if the scale is non-uniform (e.g. stretching a sphere). The Inverse-Transpose cancels this out.
#               # if the scale is non-uniform (e.g. stretching a sphere). The Inverse-Transpose cancels this out.
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
                    normal_matrix = np.eye(3, dtype=np.float32)
#                   normal_matrix = np.eye(3, dtype=np.float32)
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
                self.scene_normals.append(np.array(transformed_normals, dtype=np.float32))
#               self.scene_normals.append(np.array(transformed_normals, dtype=np.float32))

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
                self.scene_tangents.append(np.array(transformed_tangents, dtype=np.float32))
#               self.scene_tangents.append(np.array(transformed_tangents, dtype=np.float32))

    def build(self) -> tuple[bytes, bytes, bytes, bytes, bytes, bytes]:
#   def build(self) -> tuple[bytes, bytes, bytes, bytes, bytes, bytes]:
        def get_triangle_vertices(vertex0: vec3f32, vertex1: vec3f32, vertex2: vec3f32) -> npt.NDArray[np.float32]:
#       def get_triangle_vertices(vertex0: vec3f32, vertex1: vec3f32, vertex2: vec3f32) -> npt.NDArray[np.float32]:
            return np.array([vertex0, vertex1, vertex2], dtype=np.float32)
#           return np.array([vertex0, vertex1, vertex2], dtype=np.float32)

        def get_triangle_normals(normal0: vec3f32, normal1: vec3f32, normal2: vec3f32) -> npt.NDArray[np.float32]:
#       def get_triangle_normals(normal0: vec3f32, normal1: vec3f32, normal2: vec3f32) -> npt.NDArray[np.float32]:
            return np.array([normal0, normal1, normal2], dtype=np.float32)
#           return np.array([normal0, normal1, normal2], dtype=np.float32)

        def get_triangle_uvs(uv0: vec2f32, uv1: vec2f32, uv2: vec2f32) -> npt.NDArray[np.float32]:
#       def get_triangle_uvs(uv0: vec2f32, uv1: vec2f32, uv2: vec2f32) -> npt.NDArray[np.float32]:
            return np.array([uv0, uv1, uv2], dtype=np.float32)
#           return np.array([uv0, uv1, uv2], dtype=np.float32)

        def get_triangle_tangents(t0: vec3f32, t1: vec3f32, t2: vec3f32) -> npt.NDArray[np.float32]:
#       def get_triangle_tangents(t0: vec3f32, t1: vec3f32, t2: vec3f32) -> npt.NDArray[np.float32]:
            return np.array([t0, t1, t2], dtype=np.float32)
#           return np.array([t0, t1, t2], dtype=np.float32)

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

        # Consolidate all scene geometry into a single array of triangles in world space
#       # Consolidate all scene geometry into a single array of triangles in world space
        world_triangles: npt.NDArray[np.float32] = np.array(self.scene_triangles, dtype=np.float32)
#       world_triangles: npt.NDArray[np.float32] = np.array(self.scene_triangles, dtype=np.float32)

        # Build the Linear Bounding Volume Hierarchy (LBVH) for the scene
#       # Build the Linear Bounding Volume Hierarchy (LBVH) for the scene
        # This acceleration structure is essential for efficient ray-triangle intersection tests
#       # This acceleration structure is essential for efficient ray-triangle intersection tests
        lbvh: bvh.LBVH = bvh.LBVH(world_triangles)
#       lbvh: bvh.LBVH = bvh.LBVH(world_triangles)
        bvh_data: bytes = lbvh.simple_build()
#       bvh_data: bytes = lbvh.simple_build()

        # Flatten and pack all scene data (materials, UVs, normals, tangents) into byte arrays for GPU upload
#       # Flatten and pack all scene data (materials, UVs, normals, tangents) into byte arrays for GPU upload
        # These flattened arrays become the content of the readonly SSBOs (Shader Storage Buffer Objects).
#       # These flattened arrays become the content of the readonly SSBOs (Shader Storage Buffer Objects).
        # The Compute Shader will index into these using 'gl_InstanceID' or BVH leaf references.
#       # The Compute Shader will index into these using 'gl_InstanceID' or BVH leaf references.
        world_materials: npt.NDArray[np.float32] = np.array(self.scene_materials, dtype=np.float32)
#       world_materials: npt.NDArray[np.float32] = np.array(self.scene_materials, dtype=np.float32)

        world_uvs: npt.NDArray[np.float32] = np.array(self.scene_uvs, dtype=np.float32)
#       world_uvs: npt.NDArray[np.float32] = np.array(self.scene_uvs, dtype=np.float32)

        world_normals: npt.NDArray[np.float32] = np.array(self.scene_normals, dtype=np.float32)
#       world_normals: npt.NDArray[np.float32] = np.array(self.scene_normals, dtype=np.float32)

        world_tangents: npt.NDArray[np.float32] = np.array(self.scene_tangents, dtype=np.float32)
#       world_tangents: npt.NDArray[np.float32] = np.array(self.scene_tangents, dtype=np.float32)

        return bvh_data, world_triangles.flatten().tobytes(), world_materials.flatten().tobytes(), world_uvs.flatten().tobytes(), world_normals.flatten().tobytes(), world_tangents.flatten().tobytes()
#       return bvh_data, world_triangles.flatten().tobytes(), world_materials.flatten().tobytes(), world_uvs.flatten().tobytes(), world_normals.flatten().tobytes(), world_tangents.flatten().tobytes()
