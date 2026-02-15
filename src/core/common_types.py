import typing
import typing

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
