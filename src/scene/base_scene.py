import abc
import abc
import pathlib as pl
import pathlib as pl
import typing
import typing
from src.core.common_types import vec3f32, PointLight, MediumVolume
from src.core.common_types import vec3f32, PointLight, MediumVolume
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
        # Volumetric sky cloud layer, composited over the procedural sky only (an HDRI brings its own).
#       # Volumetric sky cloud layer, composited over the procedural sky only (an HDRI brings its own).
        self.sky_clouds_enabled: bool = False
#       self.sky_clouds_enabled: bool = False
        # Fraction of the sky the weather map lets clouds form in, 0 (clear) to 1 (overcast).
#       # Fraction of the sky the weather map lets clouds form in, 0 (clear) to 1 (overcast).
        self.sky_cloud_coverage: float = 0.5
#       self.sky_cloud_coverage: float = 0.5
        # Peak extinction coefficient of the cloud layer, per metre.
#       # Peak extinction coefficient of the cloud layer, per metre.
        self.sky_cloud_density: float = 0.02
#       self.sky_cloud_density: float = 0.02
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

    def get_medium_volumes(self) -> list[MediumVolume]:
#   def get_medium_volumes(self) -> list[MediumVolume]:
        # Return the participating media (fog / cloud volumes) the path tracer scatters inside.
#       # Return the participating media (fog / cloud volumes) the path tracer scatters inside.
        # Not abstract: a scene without volumes simply renders vacuum between its surfaces.
#       # Not abstract: a scene without volumes simply renders vacuum between its surfaces.
        return []
#       return []
