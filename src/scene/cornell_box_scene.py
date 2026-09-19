import pathlib as pl
import pathlib as pl
import typing
import typing
from src.core.common_types import vec3f32, Material, PointLight
from src.core.common_types import vec3f32, Material, PointLight
from src.scene.scene_builder import SceneBuilder
from src.scene.scene_builder import SceneBuilder
from src.scene.base_scene import BaseScene
from src.scene.base_scene import BaseScene

class CornellBoxScene(BaseScene):
    # Classic Cornell Box scene with diffuse white, red, and green walls, ceiling light, and two rotated interior blocks.
#   # Classic Cornell Box scene with diffuse white, red, and green walls, ceiling light, and two rotated interior blocks.
    def __init__(self, resource_dir: pl.Path) -> None:
#   def __init__(self, resource_dir: pl.Path) -> None:
        super().__init__(resource_dir=resource_dir)
#       super().__init__(resource_dir=resource_dir)
        self.name: str = "Cornell Box"
#       self.name: str = "Cornell Box"
        # Camera placed in front of the open box looking inside
#       # Camera placed in front of the open box looking inside
        self.camera_position: vec3f32 = (0.0, 5.0, 14.5)
#       self.camera_position: vec3f32 = (0.0, 5.0, 14.5)
        self.camera_look_at: vec3f32 = (0.0, 5.0, 0.0)
#       self.camera_look_at: vec3f32 = (0.0, 5.0, 0.0)
        self.camera_up: vec3f32 = (0.0, 1.0, 0.0)
#       self.camera_up: vec3f32 = (0.0, 1.0, 0.0)
        self.camera_fov: float = 45.0
#       self.camera_fov: float = 45.0
        self.camera_speed: float = 5.0
#       self.camera_speed: float = 5.0
        self.use_hdri: bool = False
#       self.use_hdri: bool = False
        self.hdri_path: pl.Path | None = None
#       self.hdri_path: pl.Path | None = None
        pass
#       pass

    def build(self, scene_builder: SceneBuilder, renderer: typing.Any) -> None:
#   def build(self, scene_builder: SceneBuilder, renderer: typing.Any) -> None:
        cb_base: pl.Path = self.resource_dir / "../assets/cornelbox"
#       cb_base: pl.Path = self.resource_dir / "../assets/cornelbox"

        # 1. White Plastic Material (Floor, Ceiling, Back Wall, Tall Box, Short Box)
#       # 1. White Plastic Material (Floor, Ceiling, Back Wall, Tall Box, Short Box)
        white_folder: pl.Path = cb_base / "Plastic013B_2K-JPG"
#       white_folder: pl.Path = cb_base / "Plastic013B_2K-JPG"
        idx_albedo_white: float = renderer.load_texture(white_folder / "Plastic013B_2K-JPG_Color.jpg", is_srgb=True) if (white_folder / "Plastic013B_2K-JPG_Color.jpg").exists() else -1.0
#       idx_albedo_white: float = renderer.load_texture(white_folder / "Plastic013B_2K-JPG_Color.jpg", is_srgb=True) if (white_folder / "Plastic013B_2K-JPG_Color.jpg").exists() else -1.0
        idx_normal_white: float = renderer.load_texture(white_folder / "Plastic013B_2K-JPG_NormalGL.jpg") if (white_folder / "Plastic013B_2K-JPG_NormalGL.jpg").exists() else -1.0
#       idx_normal_white: float = renderer.load_texture(white_folder / "Plastic013B_2K-JPG_NormalGL.jpg") if (white_folder / "Plastic013B_2K-JPG_NormalGL.jpg").exists() else -1.0
        idx_rough_white, _, _ = renderer.load_packed_orm(roughness_path=white_folder / "Plastic013B_2K-JPG_Roughness.jpg") if (white_folder / "Plastic013B_2K-JPG_Roughness.jpg").exists() else (-1.0, -1.0, -1.0)
#       idx_rough_white, _, _ = renderer.load_packed_orm(roughness_path=white_folder / "Plastic013B_2K-JPG_Roughness.jpg") if (white_folder / "Plastic013B_2K-JPG_Roughness.jpg").exists() else (-1.0, -1.0, -1.0)

        mat_white: Material = {
#       mat_white: Material = {
            "albedo": (1.0, 1.0, 1.0) if idx_albedo_white >= 0.0 else (0.73, 0.73, 0.73),
#           "albedo": (1.0, 1.0, 1.0) if idx_albedo_white >= 0.0 else (0.73, 0.73, 0.73),
            "roughness": 1.0,
#           "roughness": 1.0,
            "metallic": 0.0,
#           "metallic": 0.0,
            "transmission": 0.0,
#           "transmission": 0.0,
            "ior": 1.5,
#           "ior": 1.5,
            "texture_index_albedo": idx_albedo_white,
#           "texture_index_albedo": idx_albedo_white,
            "texture_index_roughness": idx_rough_white,
#           "texture_index_roughness": idx_rough_white,
            "texture_index_metallic": -1.0,
#           "texture_index_metallic": -1.0,
            "texture_index_normal": idx_normal_white,
#           "texture_index_normal": idx_normal_white,
            "emissive": 0.0,
#           "emissive": 0.0,
            "texture_index_emissive": -1.0,
#           "texture_index_emissive": -1.0,
            "texture_index_transmission": -1.0,
#           "texture_index_transmission": -1.0,
            "uv_scale": (1.0, 1.0),
#           "uv_scale": (1.0, 1.0),
        }
#       }

        # 2. Red Plastic Material (Left Wall)
#       # 2. Red Plastic Material (Left Wall)
        red_folder: pl.Path = cb_base / "Plastic014B_2K-JPG"
#       red_folder: pl.Path = cb_base / "Plastic014B_2K-JPG"
        idx_albedo_red: float = renderer.load_texture(red_folder / "Plastic014B_2K-JPG_Color.jpg", is_srgb=True) if (red_folder / "Plastic014B_2K-JPG_Color.jpg").exists() else -1.0
#       idx_albedo_red: float = renderer.load_texture(red_folder / "Plastic014B_2K-JPG_Color.jpg", is_srgb=True) if (red_folder / "Plastic014B_2K-JPG_Color.jpg").exists() else -1.0
        idx_normal_red: float = renderer.load_texture(red_folder / "Plastic014B_2K-JPG_NormalGL.jpg") if (red_folder / "Plastic014B_2K-JPG_NormalGL.jpg").exists() else -1.0
#       idx_normal_red: float = renderer.load_texture(red_folder / "Plastic014B_2K-JPG_NormalGL.jpg") if (red_folder / "Plastic014B_2K-JPG_NormalGL.jpg").exists() else -1.0
        idx_rough_red, _, _ = renderer.load_packed_orm(roughness_path=red_folder / "Plastic014B_2K-JPG_Roughness.jpg") if (red_folder / "Plastic014B_2K-JPG_Roughness.jpg").exists() else (-1.0, -1.0, -1.0)
#       idx_rough_red, _, _ = renderer.load_packed_orm(roughness_path=red_folder / "Plastic014B_2K-JPG_Roughness.jpg") if (red_folder / "Plastic014B_2K-JPG_Roughness.jpg").exists() else (-1.0, -1.0, -1.0)

        mat_red: Material = {
#       mat_red: Material = {
            "albedo": (1.0, 1.0, 1.0) if idx_albedo_red >= 0.0 else (0.65, 0.05, 0.05),
#           "albedo": (1.0, 1.0, 1.0) if idx_albedo_red >= 0.0 else (0.65, 0.05, 0.05),
            "roughness": 1.0,
#           "roughness": 1.0,
            "metallic": 0.0,
#           "metallic": 0.0,
            "transmission": 0.0,
#           "transmission": 0.0,
            "ior": 1.5,
#           "ior": 1.5,
            "texture_index_albedo": idx_albedo_red,
#           "texture_index_albedo": idx_albedo_red,
            "texture_index_roughness": idx_rough_red,
#           "texture_index_roughness": idx_rough_red,
            "texture_index_metallic": -1.0,
#           "texture_index_metallic": -1.0,
            "texture_index_normal": idx_normal_red,
#           "texture_index_normal": idx_normal_red,
            "emissive": 0.0,
#           "emissive": 0.0,
            "texture_index_emissive": -1.0,
#           "texture_index_emissive": -1.0,
            "texture_index_transmission": -1.0,
#           "texture_index_transmission": -1.0,
            "uv_scale": (1.0, 1.0),
#           "uv_scale": (1.0, 1.0),
        }
#       }

        # 3. Green Plastic Material (Right Wall)
#       # 3. Green Plastic Material (Right Wall)
        green_folder: pl.Path = cb_base / "Plastic009_2K-JPG"
#       green_folder: pl.Path = cb_base / "Plastic009_2K-JPG"
        idx_albedo_green: float = renderer.load_texture(green_folder / "Plastic009_2K-JPG_Color.jpg", is_srgb=True) if (green_folder / "Plastic009_2K-JPG_Color.jpg").exists() else -1.0
#       idx_albedo_green: float = renderer.load_texture(green_folder / "Plastic009_2K-JPG_Color.jpg", is_srgb=True) if (green_folder / "Plastic009_2K-JPG_Color.jpg").exists() else -1.0
        idx_normal_green: float = renderer.load_texture(green_folder / "Plastic009_2K-JPG_NormalGL.jpg") if (green_folder / "Plastic009_2K-JPG_NormalGL.jpg").exists() else -1.0
#       idx_normal_green: float = renderer.load_texture(green_folder / "Plastic009_2K-JPG_NormalGL.jpg") if (green_folder / "Plastic009_2K-JPG_NormalGL.jpg").exists() else -1.0
        idx_rough_green, _, _ = renderer.load_packed_orm(roughness_path=green_folder / "Plastic009_2K-JPG_Roughness.jpg") if (green_folder / "Plastic009_2K-JPG_Roughness.jpg").exists() else (-1.0, -1.0, -1.0)
#       idx_rough_green, _, _ = renderer.load_packed_orm(roughness_path=green_folder / "Plastic009_2K-JPG_Roughness.jpg") if (green_folder / "Plastic009_2K-JPG_Roughness.jpg").exists() else (-1.0, -1.0, -1.0)

        mat_green: Material = {
#       mat_green: Material = {
            "albedo": (1.0, 1.0, 1.0) if idx_albedo_green >= 0.0 else (0.12, 0.45, 0.15),
#           "albedo": (1.0, 1.0, 1.0) if idx_albedo_green >= 0.0 else (0.12, 0.45, 0.15),
            "roughness": 1.0,
#           "roughness": 1.0,
            "metallic": 0.0,
#           "metallic": 0.0,
            "transmission": 0.0,
#           "transmission": 0.0,
            "ior": 1.5,
#           "ior": 1.5,
            "texture_index_albedo": idx_albedo_green,
#           "texture_index_albedo": idx_albedo_green,
            "texture_index_roughness": idx_rough_green,
#           "texture_index_roughness": idx_rough_green,
            "texture_index_metallic": -1.0,
#           "texture_index_metallic": -1.0,
            "texture_index_normal": idx_normal_green,
#           "texture_index_normal": idx_normal_green,
            "emissive": 0.0,
#           "emissive": 0.0,
            "texture_index_emissive": -1.0,
#           "texture_index_emissive": -1.0,
            "texture_index_transmission": -1.0,
#           "texture_index_transmission": -1.0,
            "uv_scale": (1.0, 1.0),
#           "uv_scale": (1.0, 1.0),
        }
#       }

        # 4. Ceiling Light Panel Material (Emissive White)
#       # 4. Ceiling Light Panel Material (Emissive White)
        mat_light: Material = {
#       mat_light: Material = {
            "albedo": (1.0, 1.0, 1.0),
#           "albedo": (1.0, 1.0, 1.0),
            "roughness": 1.0,
#           "roughness": 1.0,
            "metallic": 0.0,
#           "metallic": 0.0,
            "transmission": 0.0,
#           "transmission": 0.0,
            "ior": 1.5,
#           "ior": 1.5,
            "texture_index_albedo": -1.0,
#           "texture_index_albedo": -1.0,
            "texture_index_roughness": -1.0,
#           "texture_index_roughness": -1.0,
            "texture_index_metallic": -1.0,
#           "texture_index_metallic": -1.0,
            "texture_index_normal": -1.0,
#           "texture_index_normal": -1.0,
            "emissive": 15.0,
#           "emissive": 15.0,
            "texture_index_emissive": -1.0,
#           "texture_index_emissive": -1.0,
            "texture_index_transmission": -1.0,
#           "texture_index_transmission": -1.0,
            "uv_scale": (1.0, 1.0),
#           "uv_scale": (1.0, 1.0),
        }
#       }

        scene_builder.materials.append(mat_white)
#       scene_builder.materials.append(mat_white)
        idx_mat_white: int = len(scene_builder.materials) - 1
#       idx_mat_white: int = len(scene_builder.materials) - 1

        scene_builder.materials.append(mat_red)
#       scene_builder.materials.append(mat_red)
        idx_mat_red: int = len(scene_builder.materials) - 1
#       idx_mat_red: int = len(scene_builder.materials) - 1

        scene_builder.materials.append(mat_green)
#       scene_builder.materials.append(mat_green)
        idx_mat_green: int = len(scene_builder.materials) - 1
#       idx_mat_green: int = len(scene_builder.materials) - 1

        scene_builder.materials.append(mat_light)
#       scene_builder.materials.append(mat_light)
        idx_mat_light: int = len(scene_builder.materials) - 1
#       idx_mat_light: int = len(scene_builder.materials) - 1

        # Geometry Construction
#       # Geometry Construction
        # Floor (White)
#       # Floor (White)
        scene_builder.add_cube(position=(0.0, -0.1, 0.0), rotation=(0.0, 0.0, 0.0), scale=(10.2, 0.2, 10.2), material_index=idx_mat_white)
#       scene_builder.add_cube(position=(0.0, -0.1, 0.0), rotation=(0.0, 0.0, 0.0), scale=(10.2, 0.2, 10.2), material_index=idx_mat_white)

        # Ceiling (White)
#       # Ceiling (White)
        scene_builder.add_cube(position=(0.0, 10.1, 0.0), rotation=(0.0, 0.0, 0.0), scale=(10.2, 0.2, 10.2), material_index=idx_mat_white)
#       scene_builder.add_cube(position=(0.0, 10.1, 0.0), rotation=(0.0, 0.0, 0.0), scale=(10.2, 0.2, 10.2), material_index=idx_mat_white)

        # Back Wall (White)
#       # Back Wall (White)
        scene_builder.add_cube(position=(0.0, 5.0, -5.1), rotation=(0.0, 0.0, 0.0), scale=(10.0, 10.0, 0.2), material_index=idx_mat_white)
#       scene_builder.add_cube(position=(0.0, 5.0, -5.1), rotation=(0.0, 0.0, 0.0), scale=(10.0, 10.0, 0.2), material_index=idx_mat_white)

        # Left Wall (Red)
#       # Left Wall (Red)
        scene_builder.add_cube(position=(-5.1, 5.0, 0.0), rotation=(0.0, 0.0, 0.0), scale=(0.2, 10.0, 10.0), material_index=idx_mat_red)
#       scene_builder.add_cube(position=(-5.1, 5.0, 0.0), rotation=(0.0, 0.0, 0.0), scale=(0.2, 10.0, 10.0), material_index=idx_mat_red)

        # Right Wall (Green)
#       # Right Wall (Green)
        scene_builder.add_cube(position=(5.1, 5.0, 0.0), rotation=(0.0, 0.0, 0.0), scale=(0.2, 10.0, 10.0), material_index=idx_mat_green)
#       scene_builder.add_cube(position=(5.1, 5.0, 0.0), rotation=(0.0, 0.0, 0.0), scale=(0.2, 10.0, 10.0), material_index=idx_mat_green)

        # Ceiling Light Panel (Emissive White)
#       # Ceiling Light Panel (Emissive White)
        scene_builder.add_cube(position=(0.0, 9.98, 0.0), rotation=(0.0, 0.0, 0.0), scale=(3.0, 0.04, 3.0), material_index=idx_mat_light)
#       scene_builder.add_cube(position=(0.0, 9.98, 0.0), rotation=(0.0, 0.0, 0.0), scale=(3.0, 0.04, 3.0), material_index=idx_mat_light)

        # Tall Box (White) - back right
#       # Tall Box (White) - back right
        scene_builder.add_cube(position=(1.8, 3.0, -1.5), rotation=(0.0, -0.305, 0.0), scale=(3.0, 6.0, 3.0), material_index=idx_mat_white)
#       scene_builder.add_cube(position=(1.8, 3.0, -1.5), rotation=(0.0, -0.305, 0.0), scale=(3.0, 6.0, 3.0), material_index=idx_mat_white)

        # Short Box (White) - front left
#       # Short Box (White) - front left
        scene_builder.add_cube(position=(-1.8, 1.5, 1.5), rotation=(0.0, 0.305, 0.0), scale=(3.0, 3.0, 3.0), material_index=idx_mat_white)
#       scene_builder.add_cube(position=(-1.8, 1.5, 1.5), rotation=(0.0, 0.305, 0.0), scale=(3.0, 3.0, 3.0), material_index=idx_mat_white)
        pass
#       pass

    def get_point_lights(self, time: float) -> list[PointLight]:
#   def get_point_lights(self, time: float) -> list[PointLight]:
        # Classic Cornell Box overhead area light modeled with a point light just below ceiling
#       # Classic Cornell Box overhead area light modeled with a point light just below ceiling
        return [
#       return [
            {"position": (0.0, 9.4, 0.0), "color": (100.0, 98.0, 95.0), "radius": 0.8},
#           {"position": (0.0, 9.4, 0.0), "color": (100.0, 98.0, 95.0), "radius": 0.8},
        ]
#       ]
