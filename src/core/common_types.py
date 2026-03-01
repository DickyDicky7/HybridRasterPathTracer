import typing
import typing

type vec2i32 = tuple[
    int,
    int,
]
"""
type vec2i32 = tuple[
    int,
    int,
]
"""

type vec3i32 = tuple[
    int,
    int,
    int,
]
"""
type vec3i32 = tuple[
    int,
    int,
    int,
]
"""

type vec4i32 = tuple[
    int,
    int,
    int,
    int,
]
"""
type vec4i32 = tuple[
    int,
    int,
    int,
    int,
]
"""

type vec2f32 = tuple[
    float,
    float,
]
"""
type vec2f32 = tuple[
    float,
    float,
]
"""

type vec3f32 = tuple[
    float,
    float,
    float,
]
"""
type vec3f32 = tuple[
    float,
    float,
    float,
]
"""

type vec4f32 = tuple[
    float,
    float,
    float,
    float,
]
"""
type vec4f32 = tuple[
    float,
    float,
    float,
    float,
]
"""

class Material(typing.TypedDict):
    # Defines the CPU-side structure for a Material.
#   # Defines the CPU-side structure for a Material.
    # IMPORTANT: This must strictly match the packing layout used when uploading to the SSBO
#   # IMPORTANT: This must strictly match the packing layout used when uploading to the SSBO
    # (see SceneBuilder.append_transformed_triangles) and the struct definition in the Shader.
#   # (see SceneBuilder.append_transformed_triangles) and the struct definition in the Shader.
    albedo: vec3f32
#   albedo: vec3f32
    roughness: float
#   roughness: float
    metallic: float
#   metallic: float
    transmission: float
#   transmission: float
    ior: float
#   ior: float
    texture_index_albedo: float
#   texture_index_albedo: float
    texture_index_roughness: float
#   texture_index_roughness: float
    texture_index_metallic: float
#   texture_index_metallic: float
    texture_index_normal: float
#   texture_index_normal: float
    emissive: float
#   emissive: float
    texture_index_emissive: float
#   texture_index_emissive: float

class PointLight(typing.TypedDict):
    # Defines the CPU-side structure for a Point Light.
#   # Defines the CPU-side structure for a Point Light.
    position: vec3f32
#   position: vec3f32
    color: vec3f32
#   color: vec3f32
    radius: float
#   radius: float
