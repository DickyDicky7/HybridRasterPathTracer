import abc
import abc
import pathlib as pl
import pathlib as pl
import typing
import typing
from src.core.common_types import vec3f32, PointLight
from src.core.common_types import vec3f32, PointLight
from src.scene.scene_builder import SceneBuilder
from src.scene.scene_builder import SceneBuilder

class BaseScene(abc.ABC):
    # Abstract Base Class representing a 3D Scene for the Hybrid Renderer.
#   # Abstract Base Class representing a 3D Scene for the Hybrid Renderer.
    # Defines the interface for scene geometry, materials, lighting, and camera configuration.
#   # Defines the interface for scene geometry, materials, lighting, and camera configuration.
    def __init__(self, resource_dir: pl.Path) -> None:
#   def __init__(self, resource_dir: pl.Path) -> None:
        self.resource_dir: pl.Path = resource_dir
#       self.resource_dir: pl.Path = resource_dir
        self.name: str = "Base Scene"
#       self.name: str = "Base Scene"
        self.camera_position: vec3f32 = (0.0, 0.0, 0.0)
#       self.camera_position: vec3f32 = (0.0, 0.0, 0.0)
        self.camera_look_at: vec3f32 = (0.0, 0.0, -1.0)
#       self.camera_look_at: vec3f32 = (0.0, 0.0, -1.0)
        self.camera_up: vec3f32 = (0.0, 1.0, 0.0)
#       self.camera_up: vec3f32 = (0.0, 1.0, 0.0)
        self.camera_fov: float = 60.0
#       self.camera_fov: float = 60.0
        self.camera_speed: float = 10.0
#       self.camera_speed: float = 10.0
        self.use_hdri: bool = False
#       self.use_hdri: bool = False
        self.hdri_path: pl.Path | None = None
#       self.hdri_path: pl.Path | None = None
        pass
#       pass

    @abc.abstractmethod
#   @abc.abstractmethod
    def build(self, scene_builder: SceneBuilder, renderer: typing.Any) -> None:
#   def build(self, scene_builder: SceneBuilder, renderer: typing.Any) -> None:
        # Populate geometry, materials, and textures using the SceneBuilder and Renderer.
#       # Populate geometry, materials, and textures using the SceneBuilder and Renderer.
        pass
#       pass

    @abc.abstractmethod
#   @abc.abstractmethod
    def get_point_lights(self, time: float) -> list[PointLight]:
#   def get_point_lights(self, time: float) -> list[PointLight]:
        # Return point light configurations for the given timestamp.
#       # Return point light configurations for the given timestamp.
        return []
#       return []
