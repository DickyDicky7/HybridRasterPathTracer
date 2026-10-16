import pathlib as pl
import pathlib as pl
import typing
import typing
from src.core.common_types import vec3f32, Material, PointLight, MediumVolume
from src.core.common_types import vec3f32, Material, PointLight, MediumVolume
from src.scene.scene_builder import SceneBuilder
from src.scene.scene_builder import SceneBuilder
from src.scene.base_scene import BaseScene
from src.scene.base_scene import BaseScene

class OfficeScene(BaseScene):
    # Mersus Office scene featuring open-plan workstations, meeting room, and illuminated ceiling panels.
#   # Mersus Office scene featuring open-plan workstations, meeting room, and illuminated ceiling panels.
    def __init__(self, resource_dir: pl.Path) -> None:
#   def __init__(self, resource_dir: pl.Path) -> None:
        super().__init__(resource_dir=resource_dir)
#       super().__init__(resource_dir=resource_dir)
        self.name: str = "Mersus Office"
#       self.name: str = "Mersus Office"
        # Camera positioned in the central walkway looking across the open-plan office desks and windows
#       # Camera positioned in the central walkway looking across the open-plan office desks and windows
        self.camera_position: vec3f32 = (2.5, 1.5, 3.5)
#       self.camera_position: vec3f32 = (2.5, 1.5, 3.5)
        self.camera_look_at: vec3f32 = (-1.0, 1.1, -2.0)
#       self.camera_look_at: vec3f32 = (-1.0, 1.1, -2.0)
        self.camera_up: vec3f32 = (0.0, 1.0, 0.0)
#       self.camera_up: vec3f32 = (0.0, 1.0, 0.0)
        self.camera_fov: float = 60.0
#       self.camera_fov: float = 60.0
        self.camera_speed: float = 5.0
#       self.camera_speed: float = 5.0
        self.use_hdri: bool = False
#       self.use_hdri: bool = False
        self.hdri_path: pl.Path | None = None
#       self.hdri_path: pl.Path | None = None
        self.sky_clouds_enabled: bool = False
#       self.sky_clouds_enabled: bool = False
        pass
#       pass

    def build(self, scene_builder: SceneBuilder, renderer: typing.Any) -> None:
#   def build(self, scene_builder: SceneBuilder, renderer: typing.Any) -> None:
        office_base: pl.Path = self.resource_dir / "../assets/mersus-office"
#       office_base: pl.Path = self.resource_dir / "../assets/mersus-office"
        tex_dir: pl.Path = office_base / "textures"
#       tex_dir: pl.Path = office_base / "textures"
        material_indices: list[int] = []
#       material_indices: list[int] = []

        # 0. Walls Material
#       # 0. Walls Material
        idx_albedo_walls: float = renderer.load_texture(tex_dir / "wall_low_Walls_BaseColor.png", is_srgb=True)
#       idx_albedo_walls: float = renderer.load_texture(tex_dir / "wall_low_Walls_BaseColor.png", is_srgb=True)
        idx_normal_walls: float = renderer.load_texture(tex_dir / "wall_low_Walls_Normal.png")
#       idx_normal_walls: float = renderer.load_texture(tex_dir / "wall_low_Walls_Normal.png")
        idx_rough_walls, _, _ = renderer.load_packed_orm(roughness_path=tex_dir / "wall_low_Walls_Roughness.png")
#       idx_rough_walls, _, _ = renderer.load_packed_orm(roughness_path=tex_dir / "wall_low_Walls_Roughness.png")
        mat_walls: Material = {
#       mat_walls: Material = {
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
            "texture_index_albedo": idx_albedo_walls,
#           "texture_index_albedo": idx_albedo_walls,
            "texture_index_roughness": idx_rough_walls,
#           "texture_index_roughness": idx_rough_walls,
            "texture_index_metallic": -1.0,
#           "texture_index_metallic": -1.0,
            "texture_index_normal": idx_normal_walls,
#           "texture_index_normal": idx_normal_walls,
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
        scene_builder.materials.append(mat_walls)
#       scene_builder.materials.append(mat_walls)
        material_indices.append(len(scene_builder.materials) - 1)
#       material_indices.append(len(scene_builder.materials) - 1)

        # 1. Wood Floor Material
#       # 1. Wood Floor Material
        idx_albedo_floor: float = renderer.load_texture(tex_dir / "plane_divided_wood_floor_BaseColor.jpeg", is_srgb=True)
#       idx_albedo_floor: float = renderer.load_texture(tex_dir / "plane_divided_wood_floor_BaseColor.jpeg", is_srgb=True)
        idx_normal_floor: float = renderer.load_texture(tex_dir / "plane_divided_wood_floor_Normal.jpeg")
#       idx_normal_floor: float = renderer.load_texture(tex_dir / "plane_divided_wood_floor_Normal.jpeg")
        idx_rough_floor, _, _ = renderer.load_packed_orm(roughness_path=tex_dir / "plane_divided_wood_floor_Roughness.jpeg")
#       idx_rough_floor, _, _ = renderer.load_packed_orm(roughness_path=tex_dir / "plane_divided_wood_floor_Roughness.jpeg")
        mat_floor: Material = {
#       mat_floor: Material = {
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
            "texture_index_albedo": idx_albedo_floor,
#           "texture_index_albedo": idx_albedo_floor,
            "texture_index_roughness": idx_rough_floor,
#           "texture_index_roughness": idx_rough_floor,
            "texture_index_metallic": -1.0,
#           "texture_index_metallic": -1.0,
            "texture_index_normal": idx_normal_floor,
#           "texture_index_normal": idx_normal_floor,
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
        scene_builder.materials.append(mat_floor)
#       scene_builder.materials.append(mat_floor)
        material_indices.append(len(scene_builder.materials) - 1)
#       material_indices.append(len(scene_builder.materials) - 1)

        # 2. Carpet Material
#       # 2. Carpet Material
        idx_albedo_carpet: float = renderer.load_texture(tex_dir / "Building_01_Base_Color.tga.png", is_srgb=True)
#       idx_albedo_carpet: float = renderer.load_texture(tex_dir / "Building_01_Base_Color.tga.png", is_srgb=True)
        idx_normal_carpet: float = renderer.load_texture(tex_dir / "Building_01_Normal.tga.png")
#       idx_normal_carpet: float = renderer.load_texture(tex_dir / "Building_01_Normal.tga.png")
        idx_rough_carpet, _, _ = renderer.load_packed_orm(roughness_path=tex_dir / "Building_01_Roughness.tga.png")
#       idx_rough_carpet, _, _ = renderer.load_packed_orm(roughness_path=tex_dir / "Building_01_Roughness.tga.png")
        mat_carpet: Material = {
#       mat_carpet: Material = {
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
            "texture_index_albedo": idx_albedo_carpet,
#           "texture_index_albedo": idx_albedo_carpet,
            "texture_index_roughness": idx_rough_carpet,
#           "texture_index_roughness": idx_rough_carpet,
            "texture_index_metallic": -1.0,
#           "texture_index_metallic": -1.0,
            "texture_index_normal": idx_normal_carpet,
#           "texture_index_normal": idx_normal_carpet,
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
        scene_builder.materials.append(mat_carpet)
#       scene_builder.materials.append(mat_carpet)
        material_indices.append(len(scene_builder.materials) - 1)
#       material_indices.append(len(scene_builder.materials) - 1)

        # 3. Plain Ceiling Tiles Material
#       # 3. Plain Ceiling Tiles Material
        idx_albedo_ceil_plain: float = renderer.load_texture(tex_dir / "CeilingTile_Basis_CeilingTiles_Plain_BaseC.png", is_srgb=True)
#       idx_albedo_ceil_plain: float = renderer.load_texture(tex_dir / "CeilingTile_Basis_CeilingTiles_Plain_BaseC.png", is_srgb=True)
        idx_normal_ceil_plain: float = renderer.load_texture(tex_dir / "CeilingTile_Basis_CeilingTiles_Plain_Norma.png")
#       idx_normal_ceil_plain: float = renderer.load_texture(tex_dir / "CeilingTile_Basis_CeilingTiles_Plain_Norma.png")
        idx_rough_ceil_plain, idx_met_ceil_plain, _ = renderer.load_packed_orm(
#       idx_rough_ceil_plain, idx_met_ceil_plain, _ = renderer.load_packed_orm(
            roughness_path=tex_dir / "CeilingTile_Basis_CeilingTiles_Plain_Rough.png",
#           roughness_path=tex_dir / "CeilingTile_Basis_CeilingTiles_Plain_Rough.png",
            metallic_path=tex_dir / "CeilingTile_Basis_CeilingTiles_Plain_Metal.png",
#           metallic_path=tex_dir / "CeilingTile_Basis_CeilingTiles_Plain_Metal.png",
        )
#       )
        mat_ceil_plain: Material = {
#       mat_ceil_plain: Material = {
            "albedo": (1.0, 1.0, 1.0),
#           "albedo": (1.0, 1.0, 1.0),
            "roughness": 1.0,
#           "roughness": 1.0,
            "metallic": 1.0,
#           "metallic": 1.0,
            "transmission": 0.0,
#           "transmission": 0.0,
            "ior": 1.5,
#           "ior": 1.5,
            "texture_index_albedo": idx_albedo_ceil_plain,
#           "texture_index_albedo": idx_albedo_ceil_plain,
            "texture_index_roughness": idx_rough_ceil_plain,
#           "texture_index_roughness": idx_rough_ceil_plain,
            "texture_index_metallic": idx_met_ceil_plain,
#           "texture_index_metallic": idx_met_ceil_plain,
            "texture_index_normal": idx_normal_ceil_plain,
#           "texture_index_normal": idx_normal_ceil_plain,
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
        scene_builder.materials.append(mat_ceil_plain)
#       scene_builder.materials.append(mat_ceil_plain)
        material_indices.append(len(scene_builder.materials) - 1)
#       material_indices.append(len(scene_builder.materials) - 1)

        # 4. Ceiling Tiles with Emissive Light Fixtures Material
#       # 4. Ceiling Tiles with Emissive Light Fixtures Material
        idx_albedo_ceil_light: float = renderer.load_texture(tex_dir / "CeilingTile_Basis_CeilingTiles_BaseColor.png", is_srgb=True)
#       idx_albedo_ceil_light: float = renderer.load_texture(tex_dir / "CeilingTile_Basis_CeilingTiles_BaseColor.png", is_srgb=True)
        idx_normal_ceil_light: float = renderer.load_texture(tex_dir / "CeilingTile_Basis_CeilingTiles_Normal.png")
#       idx_normal_ceil_light: float = renderer.load_texture(tex_dir / "CeilingTile_Basis_CeilingTiles_Normal.png")
        idx_rough_ceil_light, idx_met_ceil_light, _ = renderer.load_packed_orm(
#       idx_rough_ceil_light, idx_met_ceil_light, _ = renderer.load_packed_orm(
            roughness_path=tex_dir / "CeilingTile_Basis_CeilingTiles_Roughness.png",
#           roughness_path=tex_dir / "CeilingTile_Basis_CeilingTiles_Roughness.png",
            metallic_path=tex_dir / "CeilingTile_Basis_CeilingTiles_Metallic.png",
#           metallic_path=tex_dir / "CeilingTile_Basis_CeilingTiles_Metallic.png",
        )
#       )
        idx_emissive_ceil_light: float = renderer.load_texture(tex_dir / "CeilingTile_Basis_CeilingTiles_Emissive.png", is_srgb=False)
#       idx_emissive_ceil_light: float = renderer.load_texture(tex_dir / "CeilingTile_Basis_CeilingTiles_Emissive.png", is_srgb=False)
        mat_ceil_light: Material = {
#       mat_ceil_light: Material = {
            "albedo": (1.0, 1.0, 1.0),
#           "albedo": (1.0, 1.0, 1.0),
            "roughness": 1.0,
#           "roughness": 1.0,
            "metallic": 1.0,
#           "metallic": 1.0,
            "transmission": 0.0,
#           "transmission": 0.0,
            "ior": 1.5,
#           "ior": 1.5,
            "texture_index_albedo": idx_albedo_ceil_light,
#           "texture_index_albedo": idx_albedo_ceil_light,
            "texture_index_roughness": idx_rough_ceil_light,
#           "texture_index_roughness": idx_rough_ceil_light,
            "texture_index_metallic": idx_met_ceil_light,
#           "texture_index_metallic": idx_met_ceil_light,
            "texture_index_normal": idx_normal_ceil_light,
#           "texture_index_normal": idx_normal_ceil_light,
            "emissive": 1.5,
#           "emissive": 1.5,
            "texture_index_emissive": idx_emissive_ceil_light,
#           "texture_index_emissive": idx_emissive_ceil_light,
            "texture_index_transmission": -1.0,
#           "texture_index_transmission": -1.0,
            "uv_scale": (1.0, 1.0),
#           "uv_scale": (1.0, 1.0),
        }
#       }
        scene_builder.materials.append(mat_ceil_light)
#       scene_builder.materials.append(mat_ceil_light)
        material_indices.append(len(scene_builder.materials) - 1)
#       material_indices.append(len(scene_builder.materials) - 1)

        # 5. Curtain Material
#       # 5. Curtain Material
        idx_albedo_curtain: float = renderer.load_texture(tex_dir / "curtain_low_curtain_BaseColor.png", is_srgb=True)
#       idx_albedo_curtain: float = renderer.load_texture(tex_dir / "curtain_low_curtain_BaseColor.png", is_srgb=True)
        idx_normal_curtain: float = renderer.load_texture(tex_dir / "curtain_low_curtain_Normal.png")
#       idx_normal_curtain: float = renderer.load_texture(tex_dir / "curtain_low_curtain_Normal.png")
        mat_curtain: Material = {
#       mat_curtain: Material = {
            "albedo": (1.0, 1.0, 1.0),
#           "albedo": (1.0, 1.0, 1.0),
            "roughness": 0.85,
#           "roughness": 0.85,
            "metallic": 0.0,
#           "metallic": 0.0,
            "transmission": 0.0,
#           "transmission": 0.0,
            "ior": 1.5,
#           "ior": 1.5,
            "texture_index_albedo": idx_albedo_curtain,
#           "texture_index_albedo": idx_albedo_curtain,
            "texture_index_roughness": -1.0,
#           "texture_index_roughness": -1.0,
            "texture_index_metallic": -1.0,
#           "texture_index_metallic": -1.0,
            "texture_index_normal": idx_normal_curtain,
#           "texture_index_normal": idx_normal_curtain,
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
        scene_builder.materials.append(mat_curtain)
#       scene_builder.materials.append(mat_curtain)
        material_indices.append(len(scene_builder.materials) - 1)
#       material_indices.append(len(scene_builder.materials) - 1)

        # 6. Small Window Material (Metal window mullions / frame)
#       # 6. Small Window Material (Metal window mullions / frame)
        mat_small_window: Material = {
#       mat_small_window: Material = {
            "albedo": (0.12, 0.12, 0.12),
#           "albedo": (0.12, 0.12, 0.12),
            "roughness": 0.3,
#           "roughness": 0.3,
            "metallic": 0.8,
#           "metallic": 0.8,
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
        scene_builder.materials.append(mat_small_window)
#       scene_builder.materials.append(mat_small_window)
        material_indices.append(len(scene_builder.materials) - 1)
#       material_indices.append(len(scene_builder.materials) - 1)

        # 7. Glass Material (Architectural window glazing)
#       # 7. Glass Material (Architectural window glazing)
        mat_glass: Material = {
#       mat_glass: Material = {
            "albedo": (0.95, 0.95, 0.95),
#           "albedo": (0.95, 0.95, 0.95),
            "roughness": 0.02,
#           "roughness": 0.02,
            "metallic": 0.0,
#           "metallic": 0.0,
            "transmission": 1.0,
#           "transmission": 1.0,
            "ior": 1.52,
#           "ior": 1.52,
            "texture_index_albedo": -1.0,
#           "texture_index_albedo": -1.0,
            "texture_index_roughness": -1.0,
#           "texture_index_roughness": -1.0,
            "texture_index_metallic": -1.0,
#           "texture_index_metallic": -1.0,
            "texture_index_normal": -1.0,
#           "texture_index_normal": -1.0,
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
        scene_builder.materials.append(mat_glass)
#       scene_builder.materials.append(mat_glass)
        material_indices.append(len(scene_builder.materials) - 1)
#       material_indices.append(len(scene_builder.materials) - 1)

        # 8. Metal Black Material
#       # 8. Metal Black Material
        mat_metal_black: Material = {
#       mat_metal_black: Material = {
            "albedo": (0.06, 0.06, 0.06),
#           "albedo": (0.06, 0.06, 0.06),
            "roughness": 0.25,
#           "roughness": 0.25,
            "metallic": 0.9,
#           "metallic": 0.9,
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
        scene_builder.materials.append(mat_metal_black)
#       scene_builder.materials.append(mat_metal_black)
        material_indices.append(len(scene_builder.materials) - 1)
#       material_indices.append(len(scene_builder.materials) - 1)

        # 9. Solid Door Material
#       # 9. Solid Door Material
        idx_albedo_door: float = renderer.load_texture(tex_dir / "Door_Solid_low_Door_Solid_BaseColor.png", is_srgb=True)
#       idx_albedo_door: float = renderer.load_texture(tex_dir / "Door_Solid_low_Door_Solid_BaseColor.png", is_srgb=True)
        idx_normal_door: float = renderer.load_texture(tex_dir / "Door_Solid_low_Door_Solid_Normal.png")
#       idx_normal_door: float = renderer.load_texture(tex_dir / "Door_Solid_low_Door_Solid_Normal.png")
        idx_rough_door, idx_met_door, _ = renderer.load_packed_orm(
#       idx_rough_door, idx_met_door, _ = renderer.load_packed_orm(
            roughness_path=tex_dir / "Door_Solid_low_Door_Solid_Roughness.png",
#           roughness_path=tex_dir / "Door_Solid_low_Door_Solid_Roughness.png",
            metallic_path=tex_dir / "Door_Solid_low_Door_Solid_Metallic.png",
#           metallic_path=tex_dir / "Door_Solid_low_Door_Solid_Metallic.png",
        )
#       )
        mat_door: Material = {
#       mat_door: Material = {
            "albedo": (1.0, 1.0, 1.0),
#           "albedo": (1.0, 1.0, 1.0),
            "roughness": 1.0,
#           "roughness": 1.0,
            "metallic": 1.0,
#           "metallic": 1.0,
            "transmission": 0.0,
#           "transmission": 0.0,
            "ior": 1.5,
#           "ior": 1.5,
            "texture_index_albedo": idx_albedo_door,
#           "texture_index_albedo": idx_albedo_door,
            "texture_index_roughness": idx_rough_door,
#           "texture_index_roughness": idx_rough_door,
            "texture_index_metallic": idx_met_door,
#           "texture_index_metallic": idx_met_door,
            "texture_index_normal": idx_normal_door,
#           "texture_index_normal": idx_normal_door,
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
        scene_builder.materials.append(mat_door)
#       scene_builder.materials.append(mat_door)
        material_indices.append(len(scene_builder.materials) - 1)
#       material_indices.append(len(scene_builder.materials) - 1)

        # 10. Shelving Material
#       # 10. Shelving Material
        idx_albedo_shelving: float = renderer.load_texture(tex_dir / "shelving_low_shelving_BaseColor.png", is_srgb=True)
#       idx_albedo_shelving: float = renderer.load_texture(tex_dir / "shelving_low_shelving_BaseColor.png", is_srgb=True)
        idx_normal_shelving: float = renderer.load_texture(tex_dir / "shelving_low_shelving_Normal.png")
#       idx_normal_shelving: float = renderer.load_texture(tex_dir / "shelving_low_shelving_Normal.png")
        idx_rough_shelving, _, _ = renderer.load_packed_orm(roughness_path=tex_dir / "shelving_low_shelving_Roughness.png")
#       idx_rough_shelving, _, _ = renderer.load_packed_orm(roughness_path=tex_dir / "shelving_low_shelving_Roughness.png")
        mat_shelving: Material = {
#       mat_shelving: Material = {
            "albedo": (1.0, 1.0, 1.0),
#           "albedo": (1.0, 1.0, 1.0),
            "roughness": 1.0,
#           "roughness": 1.0,
            "metallic": 0.1,
#           "metallic": 0.1,
            "transmission": 0.0,
#           "transmission": 0.0,
            "ior": 1.5,
#           "ior": 1.5,
            "texture_index_albedo": idx_albedo_shelving,
#           "texture_index_albedo": idx_albedo_shelving,
            "texture_index_roughness": idx_rough_shelving,
#           "texture_index_roughness": idx_rough_shelving,
            "texture_index_metallic": -1.0,
#           "texture_index_metallic": -1.0,
            "texture_index_normal": idx_normal_shelving,
#           "texture_index_normal": idx_normal_shelving,
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
        scene_builder.materials.append(mat_shelving)
#       scene_builder.materials.append(mat_shelving)
        material_indices.append(len(scene_builder.materials) - 1)
#       material_indices.append(len(scene_builder.materials) - 1)

        # 11. Default Material (Neutral office trim)
#       # 11. Default Material (Neutral office trim)
        mat_default: Material = {
#       mat_default: Material = {
            "albedo": (0.8, 0.8, 0.8),
#           "albedo": (0.8, 0.8, 0.8),
            "roughness": 0.5,
#           "roughness": 0.5,
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
        scene_builder.materials.append(mat_default)
#       scene_builder.materials.append(mat_default)
        material_indices.append(len(scene_builder.materials) - 1)
#       material_indices.append(len(scene_builder.materials) - 1)

        # 12. Meeting Table Material
#       # 12. Meeting Table Material
        idx_albedo_meet_table: float = renderer.load_texture(tex_dir / "MeetingTable_low_meetingTable_BaseColor.png", is_srgb=True)
#       idx_albedo_meet_table: float = renderer.load_texture(tex_dir / "MeetingTable_low_meetingTable_BaseColor.png", is_srgb=True)
        idx_normal_meet_table: float = renderer.load_texture(tex_dir / "MeetingTable_low_meetingTable_Normal.png")
#       idx_normal_meet_table: float = renderer.load_texture(tex_dir / "MeetingTable_low_meetingTable_Normal.png")
        idx_rough_meet_table, idx_met_meet_table, _ = renderer.load_packed_orm(
#       idx_rough_meet_table, idx_met_meet_table, _ = renderer.load_packed_orm(
            roughness_path=tex_dir / "MeetingTable_low_meetingTable_Roughness.png",
#           roughness_path=tex_dir / "MeetingTable_low_meetingTable_Roughness.png",
            metallic_path=tex_dir / "MeetingTable_low_meetingTable_Metallic.png",
#           metallic_path=tex_dir / "MeetingTable_low_meetingTable_Metallic.png",
        )
#       )
        mat_meet_table: Material = {
#       mat_meet_table: Material = {
            "albedo": (1.0, 1.0, 1.0),
#           "albedo": (1.0, 1.0, 1.0),
            "roughness": 1.0,
#           "roughness": 1.0,
            "metallic": 1.0,
#           "metallic": 1.0,
            "transmission": 0.0,
#           "transmission": 0.0,
            "ior": 1.5,
#           "ior": 1.5,
            "texture_index_albedo": idx_albedo_meet_table,
#           "texture_index_albedo": idx_albedo_meet_table,
            "texture_index_roughness": idx_rough_meet_table,
#           "texture_index_roughness": idx_rough_meet_table,
            "texture_index_metallic": idx_met_meet_table,
#           "texture_index_metallic": idx_met_meet_table,
            "texture_index_normal": idx_normal_meet_table,
#           "texture_index_normal": idx_normal_meet_table,
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
        scene_builder.materials.append(mat_meet_table)
#       scene_builder.materials.append(mat_meet_table)
        material_indices.append(len(scene_builder.materials) - 1)
#       material_indices.append(len(scene_builder.materials) - 1)

        # 13. Brown Chair Material
#       # 13. Brown Chair Material
        idx_albedo_brown_chair: float = renderer.load_texture(tex_dir / "BrownChairLP_BrownChair_BaseColor.png", is_srgb=True)
#       idx_albedo_brown_chair: float = renderer.load_texture(tex_dir / "BrownChairLP_BrownChair_BaseColor.png", is_srgb=True)
        idx_normal_brown_chair: float = renderer.load_texture(tex_dir / "BrownChairLP_BrownChair_Normal.png")
#       idx_normal_brown_chair: float = renderer.load_texture(tex_dir / "BrownChairLP_BrownChair_Normal.png")
        idx_rough_brown_chair, _, _ = renderer.load_packed_orm(roughness_path=tex_dir / "BrownChairLP_BrownChair_Roughness.png")
#       idx_rough_brown_chair, _, _ = renderer.load_packed_orm(roughness_path=tex_dir / "BrownChairLP_BrownChair_Roughness.png")
        mat_brown_chair: Material = {
#       mat_brown_chair: Material = {
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
            "texture_index_albedo": idx_albedo_brown_chair,
#           "texture_index_albedo": idx_albedo_brown_chair,
            "texture_index_roughness": idx_rough_brown_chair,
#           "texture_index_roughness": idx_rough_brown_chair,
            "texture_index_metallic": -1.0,
#           "texture_index_metallic": -1.0,
            "texture_index_normal": idx_normal_brown_chair,
#           "texture_index_normal": idx_normal_brown_chair,
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
        scene_builder.materials.append(mat_brown_chair)
#       scene_builder.materials.append(mat_brown_chair)
        material_indices.append(len(scene_builder.materials) - 1)
#       material_indices.append(len(scene_builder.materials) - 1)

        # 14. Big White Table Material
#       # 14. Big White Table Material
        idx_albedo_white_table: float = renderer.load_texture(tex_dir / "BiglWhieTable_low_BigWhiteTable_BaseColor.png", is_srgb=True)
#       idx_albedo_white_table: float = renderer.load_texture(tex_dir / "BiglWhieTable_low_BigWhiteTable_BaseColor.png", is_srgb=True)
        idx_normal_white_table: float = renderer.load_texture(tex_dir / "BiglWhieTable_low_BigWhiteTable_Normal.png")
#       idx_normal_white_table: float = renderer.load_texture(tex_dir / "BiglWhieTable_low_BigWhiteTable_Normal.png")
        idx_rough_white_table, _, _ = renderer.load_packed_orm(roughness_path=tex_dir / "BiglWhieTable_low_BigWhiteTable_Roughness.png")
#       idx_rough_white_table, _, _ = renderer.load_packed_orm(roughness_path=tex_dir / "BiglWhieTable_low_BigWhiteTable_Roughness.png")
        mat_white_table: Material = {
#       mat_white_table: Material = {
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
            "texture_index_albedo": idx_albedo_white_table,
#           "texture_index_albedo": idx_albedo_white_table,
            "texture_index_roughness": idx_rough_white_table,
#           "texture_index_roughness": idx_rough_white_table,
            "texture_index_metallic": -1.0,
#           "texture_index_metallic": -1.0,
            "texture_index_normal": idx_normal_white_table,
#           "texture_index_normal": idx_normal_white_table,
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
        scene_builder.materials.append(mat_white_table)
#       scene_builder.materials.append(mat_white_table)
        material_indices.append(len(scene_builder.materials) - 1)
#       material_indices.append(len(scene_builder.materials) - 1)

        # 15. Wooden Desk Material
#       # 15. Wooden Desk Material
        idx_albedo_wood_desk: float = renderer.load_texture(tex_dir / "Table2_LP_table_BaseColor.png", is_srgb=True)
#       idx_albedo_wood_desk: float = renderer.load_texture(tex_dir / "Table2_LP_table_BaseColor.png", is_srgb=True)
        mat_wood_desk: Material = {
#       mat_wood_desk: Material = {
            "albedo": (1.0, 1.0, 1.0),
#           "albedo": (1.0, 1.0, 1.0),
            "roughness": 0.5,
#           "roughness": 0.5,
            "metallic": 0.0,
#           "metallic": 0.0,
            "transmission": 0.0,
#           "transmission": 0.0,
            "ior": 1.5,
#           "ior": 1.5,
            "texture_index_albedo": idx_albedo_wood_desk,
#           "texture_index_albedo": idx_albedo_wood_desk,
            "texture_index_roughness": -1.0,
#           "texture_index_roughness": -1.0,
            "texture_index_metallic": -1.0,
#           "texture_index_metallic": -1.0,
            "texture_index_normal": -1.0,
#           "texture_index_normal": -1.0,
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
        scene_builder.materials.append(mat_wood_desk)
#       scene_builder.materials.append(mat_wood_desk)
        material_indices.append(len(scene_builder.materials) - 1)
#       material_indices.append(len(scene_builder.materials) - 1)

        # 16. Small White Table Material
#       # 16. Small White Table Material
        idx_albedo_small_table: float = renderer.load_texture(tex_dir / "cOMICTable_LP_ComicTable_BaseColor.png", is_srgb=True)
#       idx_albedo_small_table: float = renderer.load_texture(tex_dir / "cOMICTable_LP_ComicTable_BaseColor.png", is_srgb=True)
        idx_normal_small_table: float = renderer.load_texture(tex_dir / "cOMICTable_LP_ComicTable_Normal.png")
#       idx_normal_small_table: float = renderer.load_texture(tex_dir / "cOMICTable_LP_ComicTable_Normal.png")
        idx_rough_small_table, _, _ = renderer.load_packed_orm(roughness_path=tex_dir / "cOMICTable_LP_ComicTable_Roughness.png")
#       idx_rough_small_table, _, _ = renderer.load_packed_orm(roughness_path=tex_dir / "cOMICTable_LP_ComicTable_Roughness.png")
        mat_small_table: Material = {
#       mat_small_table: Material = {
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
            "texture_index_albedo": idx_albedo_small_table,
#           "texture_index_albedo": idx_albedo_small_table,
            "texture_index_roughness": idx_rough_small_table,
#           "texture_index_roughness": idx_rough_small_table,
            "texture_index_metallic": -1.0,
#           "texture_index_metallic": -1.0,
            "texture_index_normal": idx_normal_small_table,
#           "texture_index_normal": idx_normal_small_table,
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
        scene_builder.materials.append(mat_small_table)
#       scene_builder.materials.append(mat_small_table)
        material_indices.append(len(scene_builder.materials) - 1)
#       material_indices.append(len(scene_builder.materials) - 1)

        # 17. Desktop Material (Workstation Computers)
#       # 17. Desktop Material (Workstation Computers)
        idx_albedo_desktop: float = renderer.load_texture(tex_dir / "Computer_low_Desktop_BaseColor.png", is_srgb=True)
#       idx_albedo_desktop: float = renderer.load_texture(tex_dir / "Computer_low_Desktop_BaseColor.png", is_srgb=True)
        idx_normal_desktop: float = renderer.load_texture(tex_dir / "Computer_low_Desktop_Normal.png")
#       idx_normal_desktop: float = renderer.load_texture(tex_dir / "Computer_low_Desktop_Normal.png")
        idx_rough_desktop, _, _ = renderer.load_packed_orm(roughness_path=tex_dir / "Computer_low_Desktop_Roughness.png")
#       idx_rough_desktop, _, _ = renderer.load_packed_orm(roughness_path=tex_dir / "Computer_low_Desktop_Roughness.png")
        mat_desktop: Material = {
#       mat_desktop: Material = {
            "albedo": (1.0, 1.0, 1.0),
#           "albedo": (1.0, 1.0, 1.0),
            "roughness": 1.0,
#           "roughness": 1.0,
            "metallic": 0.2,
#           "metallic": 0.2,
            "transmission": 0.0,
#           "transmission": 0.0,
            "ior": 1.5,
#           "ior": 1.5,
            "texture_index_albedo": idx_albedo_desktop,
#           "texture_index_albedo": idx_albedo_desktop,
            "texture_index_roughness": idx_rough_desktop,
#           "texture_index_roughness": idx_rough_desktop,
            "texture_index_metallic": -1.0,
#           "texture_index_metallic": -1.0,
            "texture_index_normal": idx_normal_desktop,
#           "texture_index_normal": idx_normal_desktop,
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
        scene_builder.materials.append(mat_desktop)
#       scene_builder.materials.append(mat_desktop)
        material_indices.append(len(scene_builder.materials) - 1)
#       material_indices.append(len(scene_builder.materials) - 1)

        # 18. Swivel Chair Material
#       # 18. Swivel Chair Material
        idx_albedo_swivel: float = renderer.load_texture(tex_dir / "SwivlChair_low_SwivlChair_BaseColor.png", is_srgb=True)
#       idx_albedo_swivel: float = renderer.load_texture(tex_dir / "SwivlChair_low_SwivlChair_BaseColor.png", is_srgb=True)
        idx_normal_swivel: float = renderer.load_texture(tex_dir / "SwivlChair_low_SwivlChair_Normal.png")
#       idx_normal_swivel: float = renderer.load_texture(tex_dir / "SwivlChair_low_SwivlChair_Normal.png")
        idx_rough_swivel, idx_met_swivel, _ = renderer.load_packed_orm(
#       idx_rough_swivel, idx_met_swivel, _ = renderer.load_packed_orm(
            roughness_path=tex_dir / "SwivlChair_low_SwivlChair_Roughness.png",
#           roughness_path=tex_dir / "SwivlChair_low_SwivlChair_Roughness.png",
            metallic_path=tex_dir / "SwivlChair_low_SwivlChair_Metallic.png",
#           metallic_path=tex_dir / "SwivlChair_low_SwivlChair_Metallic.png",
        )
#       )
        mat_swivel: Material = {
#       mat_swivel: Material = {
            "albedo": (1.0, 1.0, 1.0),
#           "albedo": (1.0, 1.0, 1.0),
            "roughness": 1.0,
#           "roughness": 1.0,
            "metallic": 1.0,
#           "metallic": 1.0,
            "transmission": 0.0,
#           "transmission": 0.0,
            "ior": 1.5,
#           "ior": 1.5,
            "texture_index_albedo": idx_albedo_swivel,
#           "texture_index_albedo": idx_albedo_swivel,
            "texture_index_roughness": idx_rough_swivel,
#           "texture_index_roughness": idx_rough_swivel,
            "texture_index_metallic": idx_met_swivel,
#           "texture_index_metallic": idx_met_swivel,
            "texture_index_normal": idx_normal_swivel,
#           "texture_index_normal": idx_normal_swivel,
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
        scene_builder.materials.append(mat_swivel)
#       scene_builder.materials.append(mat_swivel)
        material_indices.append(len(scene_builder.materials) - 1)
#       material_indices.append(len(scene_builder.materials) - 1)

        # 19. Swivel Chair Black Material (Black variation)
#       # 19. Swivel Chair Black Material (Black variation)
        mat_swivel_black: Material = {
#       mat_swivel_black: Material = {
            "albedo": (0.25, 0.25, 0.25),
#           "albedo": (0.25, 0.25, 0.25),
            "roughness": 1.0,
#           "roughness": 1.0,
            "metallic": 1.0,
#           "metallic": 1.0,
            "transmission": 0.0,
#           "transmission": 0.0,
            "ior": 1.5,
#           "ior": 1.5,
            "texture_index_albedo": idx_albedo_swivel,
#           "texture_index_albedo": idx_albedo_swivel,
            "texture_index_roughness": idx_rough_swivel,
#           "texture_index_roughness": idx_rough_swivel,
            "texture_index_metallic": idx_met_swivel,
#           "texture_index_metallic": idx_met_swivel,
            "texture_index_normal": idx_normal_swivel,
#           "texture_index_normal": idx_normal_swivel,
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
        scene_builder.materials.append(mat_swivel_black)
#       scene_builder.materials.append(mat_swivel_black)
        material_indices.append(len(scene_builder.materials) - 1)
#       material_indices.append(len(scene_builder.materials) - 1)

        # Load Office Model FBX
#       # Load Office Model FBX
        office_fbx_path: str = str(office_base / "source/OfficeforSketchFab.fbx")
#       office_fbx_path: str = str(office_base / "source/OfficeforSketchFab.fbx")
        scene_builder.load_model(
#       scene_builder.load_model(
            path=office_fbx_path,
#           path=office_fbx_path,
            position=(0.0, 0.0, 0.0),
#           position=(0.0, 0.0, 0.0),
            rotation=(0.0, 0.0, 0.0),
#           rotation=(0.0, 0.0, 0.0),
            scale=(1.0, 1.0, 1.0),
#           scale=(1.0, 1.0, 1.0),
            material_indices=material_indices,
#           material_indices=material_indices,
        )
#       )
        pass
#       pass

    def get_point_lights(self, time: float) -> list[PointLight]:
#   def get_point_lights(self, time: float) -> list[PointLight]:
        # Strategic ceiling point lights placed directly below ceiling fixtures across functional zones:
#       # Strategic ceiling point lights placed directly below ceiling fixtures across functional zones:
        # Conference table, collaboration table, aisles, workstations, and entrance.
#       # Conference table, collaboration table, aisles, workstations, and entrance.
        return [
#       return [
            {"position": (0.35, 2.6, 7.8), "color": (14.0, 13.5, 12.5), "radius": 0.5},
#           {"position": (0.35, 2.6, 7.8), "color": (14.0, 13.5, 12.5), "radius": 0.5},
            {"position": (-2.0, 2.6, 6.0), "color": (8.0, 7.8, 7.2), "radius": 0.5},
#           {"position": (-2.0, 2.6, 6.0), "color": (8.0, 7.8, 7.2), "radius": 0.5},
            {"position": (-1.3, 2.6, 1.0), "color": (14.0, 13.5, 12.5), "radius": 0.5},
#           {"position": (-1.3, 2.6, 1.0), "color": (14.0, 13.5, 12.5), "radius": 0.5},
            {"position": (1.8, 2.6, 0.5), "color": (11.0, 10.8, 10.2), "radius": 0.5},
#           {"position": (1.8, 2.6, 0.5), "color": (11.0, 10.8, 10.2), "radius": 0.5},
            {"position": (-4.0, 2.6, 0.0), "color": (9.0, 9.0, 9.5), "radius": 0.5},
#           {"position": (-4.0, 2.6, 0.0), "color": (9.0, 9.0, 9.5), "radius": 0.5},
            {"position": (-1.5, 2.6, -5.5), "color": (13.0, 12.7, 12.0), "radius": 0.5},
#           {"position": (-1.5, 2.6, -5.5), "color": (13.0, 12.7, 12.0), "radius": 0.5},
            {"position": (1.5, 2.6, -5.5), "color": (13.0, 12.7, 12.0), "radius": 0.5},
#           {"position": (1.5, 2.6, -5.5), "color": (13.0, 12.7, 12.0), "radius": 0.5},
            {"position": (5.5, 2.6, -7.0), "color": (10.0, 9.5, 9.0), "radius": 0.5},
#           {"position": (5.5, 2.6, -7.0), "color": (10.0, 9.5, 9.0), "radius": 0.5},
        ]
#       ]

    def get_medium_volumes(self) -> list[MediumVolume]:
#   def get_medium_volumes(self) -> list[MediumVolume]:
        return []
#       return []
