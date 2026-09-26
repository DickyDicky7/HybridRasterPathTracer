import numpy as np
import numpy as np
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

class ShowcaseScene(BaseScene):
    # Showcase scene featuring weapons, vase, ground plane, and orbiting point lights.
#   # Showcase scene featuring weapons, vase, ground plane, and orbiting point lights.
    def __init__(self, resource_dir: pl.Path) -> None:
#   def __init__(self, resource_dir: pl.Path) -> None:
        super().__init__(resource_dir=resource_dir)
#       super().__init__(resource_dir=resource_dir)
        self.name: str = "Showcase Scene"
#       self.name: str = "Showcase Scene"
        self.camera_position: vec3f32 = (-28.284, 0.0, -28.284)
#       self.camera_position: vec3f32 = (-28.284, 0.0, -28.284)
        self.camera_look_at: vec3f32 = (0.0, 0.0, 0.0)
#       self.camera_look_at: vec3f32 = (0.0, 0.0, 0.0)
        self.camera_up: vec3f32 = (0.0, 1.0, 0.0)
#       self.camera_up: vec3f32 = (0.0, 1.0, 0.0)
        self.camera_fov: float = 60.0
#       self.camera_fov: float = 60.0
        # Scattered fair-weather cumulus over the procedural sky.
#       # Scattered fair-weather cumulus over the procedural sky.
        self.sky_clouds_enabled: bool = True
#       self.sky_clouds_enabled: bool = True
        self.sky_cloud_coverage: float = 0.5
#       self.sky_cloud_coverage: float = 0.5
        self.sky_cloud_density: float = 0.02
#       self.sky_cloud_density: float = 0.02
        self.camera_speed: float = 10.0
#       self.camera_speed: float = 10.0
        self.hdri_path: pl.Path | None = self.resource_dir / "../assets/wasteland_clouds_puresky_4k.exr"
#       self.hdri_path: pl.Path | None = self.resource_dir / "../assets/wasteland_clouds_puresky_4k.exr"
        self.use_hdri: bool = False
#       self.use_hdri: bool = False
        pass
#       pass

    def build(self, scene_builder: SceneBuilder, renderer: typing.Any) -> None:
#   def build(self, scene_builder: SceneBuilder, renderer: typing.Any) -> None:
        # Base cubes and ground plane materials
#       # Base cubes and ground plane materials
        scene_builder.materials.extend([
#       scene_builder.materials.extend([
            {"albedo": (1.0, 1.0, 1.0), "roughness": 1.0, "metallic": 0.0, "transmission": 0.0, "ior": 1.5, "texture_index_albedo": -1.0, "texture_index_roughness": -1.0, "texture_index_metallic": -1.0, "texture_index_normal": -1.0, "emissive": 9.0, "texture_index_emissive": -1.0, "texture_index_transmission": -1.0, "uv_scale": (1.0, 1.0)},
#           {"albedo": (1.0, 1.0, 1.0), "roughness": 1.0, "metallic": 0.0, "transmission": 0.0, "ior": 1.5, "texture_index_albedo": -1.0, "texture_index_roughness": -1.0, "texture_index_metallic": -1.0, "texture_index_normal": -1.0, "emissive": 9.0, "texture_index_emissive": -1.0, "texture_index_transmission": -1.0, "uv_scale": (1.0, 1.0)},
            {"albedo": (0.5, 1.0, 0.0), "roughness": 1.0, "metallic": 0.0, "transmission": 0.0, "ior": 1.5, "texture_index_albedo": -1.0, "texture_index_roughness": -1.0, "texture_index_metallic": -1.0, "texture_index_normal": -1.0, "emissive": 0.0, "texture_index_emissive": -1.0, "texture_index_transmission": -1.0, "uv_scale": (1.0, 1.0)},
#           {"albedo": (0.5, 1.0, 0.0), "roughness": 1.0, "metallic": 0.0, "transmission": 0.0, "ior": 1.5, "texture_index_albedo": -1.0, "texture_index_roughness": -1.0, "texture_index_metallic": -1.0, "texture_index_normal": -1.0, "emissive": 0.0, "texture_index_emissive": -1.0, "texture_index_transmission": -1.0, "uv_scale": (1.0, 1.0)},
            {"albedo": (0.0, 0.5, 1.0), "roughness": 1.0, "metallic": 0.0, "transmission": 0.0, "ior": 1.5, "texture_index_albedo": -1.0, "texture_index_roughness": -1.0, "texture_index_metallic": -1.0, "texture_index_normal": -1.0, "emissive": 0.0, "texture_index_emissive": -1.0, "texture_index_transmission": -1.0, "uv_scale": (1.0, 1.0)},
#           {"albedo": (0.0, 0.5, 1.0), "roughness": 1.0, "metallic": 0.0, "transmission": 0.0, "ior": 1.5, "texture_index_albedo": -1.0, "texture_index_roughness": -1.0, "texture_index_metallic": -1.0, "texture_index_normal": -1.0, "emissive": 0.0, "texture_index_emissive": -1.0, "texture_index_transmission": -1.0, "uv_scale": (1.0, 1.0)},
            {"albedo": (0.5, 0.5, 0.5), "roughness": 1.0, "metallic": 0.0, "transmission": 0.0, "ior": 1.5, "texture_index_albedo": -1.0, "texture_index_roughness": -1.0, "texture_index_metallic": -1.0, "texture_index_normal": -1.0, "emissive": 0.0, "texture_index_emissive": -1.0, "texture_index_transmission": -1.0, "uv_scale": (10.0, 10.0)},
#           {"albedo": (0.5, 0.5, 0.5), "roughness": 1.0, "metallic": 0.0, "transmission": 0.0, "ior": 1.5, "texture_index_albedo": -1.0, "texture_index_roughness": -1.0, "texture_index_metallic": -1.0, "texture_index_normal": -1.0, "emissive": 0.0, "texture_index_emissive": -1.0, "texture_index_transmission": -1.0, "uv_scale": (10.0, 10.0)},
        ])
#       ])

        scene_builder.add_cube(position=(-5.0, 1.0, 5.0), rotation=(0.0, 0.0, 0.0), scale=(1.0, 1.0, 1.0), material_index=0)
#       scene_builder.add_cube(position=(-5.0, 1.0, 5.0), rotation=(0.0, 0.0, 0.0), scale=(1.0, 1.0, 1.0), material_index=0)
        scene_builder.add_cube(position=(5.0, 1.0, -5.0), rotation=(0.0, 0.0, 0.0), scale=(1.0, 1.0, 1.0), material_index=0)
#       scene_builder.add_cube(position=(5.0, 1.0, -5.0), rotation=(0.0, 0.0, 0.0), scale=(1.0, 1.0, 1.0), material_index=0)
        scene_builder.add_cube(position=(5.0, 1.0, 5.0), rotation=(0.0, 0.0, 0.0), scale=(1.0, 1.0, 1.0), material_index=0)
#       scene_builder.add_cube(position=(5.0, 1.0, 5.0), rotation=(0.0, 0.0, 0.0), scale=(1.0, 1.0, 1.0), material_index=0)
        scene_builder.add_cube(position=(-5.0, 1.0, -5.0), rotation=(0.0, 0.0, 0.0), scale=(1.0, 1.0, 1.0), material_index=0)
#       scene_builder.add_cube(position=(-5.0, 1.0, -5.0), rotation=(0.0, 0.0, 0.0), scale=(1.0, 1.0, 1.0), material_index=0)

        scene_builder.add_plane(position=(0.0, -0.5, 0.0), rotation=(0.0, 0.0, 0.0), scale=(20.0, 1.0, 20.0), material_index=3)
#       scene_builder.add_plane(position=(0.0, -0.5, 0.0), rotation=(0.0, 0.0, 0.0), scale=(20.0, 1.0, 20.0), material_index=3)

        # Load Vase
#       # Load Vase
        vase_albedo_idx: float = renderer.load_texture(self.resource_dir / "../assets/ChinaVase.jpg", is_srgb=True)
#       vase_albedo_idx: float = renderer.load_texture(self.resource_dir / "../assets/ChinaVase.jpg", is_srgb=True)
        scene_builder.materials.append({
#       scene_builder.materials.append({
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
            "texture_index_albedo": vase_albedo_idx,
#           "texture_index_albedo": vase_albedo_idx,
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
        })
#       })
        vase_material_index: int = len(scene_builder.materials) - 1
#       vase_material_index: int = len(scene_builder.materials) - 1

        vase_path: str = str(self.resource_dir / "../assets/ChinaVase.obj")
#       vase_path: str = str(self.resource_dir / "../assets/ChinaVase.obj")
        scene_builder.load_model(path=vase_path, position=(0.0, 0.5, 0.0), rotation=(np.pi / 4.0, 0.0, 0.0), scale=(0.1, 0.1, 0.1), material_indices=vase_material_index)
#       scene_builder.load_model(path=vase_path, position=(0.0, 0.5, 0.0), rotation=(np.pi / 4.0, 0.0, 0.0), scale=(0.1, 0.1, 0.1), material_indices=vase_material_index)

        # Load Pistol
#       # Load Pistol
        pistol_base_path: pl.Path = self.resource_dir / "../assets/pistol"
#       pistol_base_path: pl.Path = self.resource_dir / "../assets/pistol"
        pistol_material_indices: list[int] = []
#       pistol_material_indices: list[int] = []

        idx_albedo_pistol: float = renderer.load_texture(pistol_base_path / "pistol_albedo.png", is_srgb=True)
#       idx_albedo_pistol: float = renderer.load_texture(pistol_base_path / "pistol_albedo.png", is_srgb=True)
        idx_normal_pistol: float = renderer.load_texture(pistol_base_path / "pistol_normal.png")
#       idx_normal_pistol: float = renderer.load_texture(pistol_base_path / "pistol_normal.png")
        _, idx_met_sm_pistol, _ = renderer.load_packed_orm(metallic_path=pistol_base_path / "pistol_metallic_smoothness.png")
#       _, idx_met_sm_pistol, _ = renderer.load_packed_orm(metallic_path=pistol_base_path / "pistol_metallic_smoothness.png")

        # Material 0: lambert4
#       # Material 0: lambert4
        scene_builder.materials.append({
#       scene_builder.materials.append({
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
            "texture_index_albedo": idx_albedo_pistol,
#           "texture_index_albedo": idx_albedo_pistol,
            "texture_index_roughness": -1.0,
#           "texture_index_roughness": -1.0,
            "texture_index_metallic": -1.0,
#           "texture_index_metallic": -1.0,
            "texture_index_normal": idx_normal_pistol,
#           "texture_index_normal": idx_normal_pistol,
            "emissive": 0.0,
#           "emissive": 0.0,
            "texture_index_emissive": -1.0,
#           "texture_index_emissive": -1.0,
            "texture_index_transmission": -1.0,
#           "texture_index_transmission": -1.0,
            "uv_scale": (1.0, 1.0),
#           "uv_scale": (1.0, 1.0),
        })
#       })
        pistol_material_indices.append(len(scene_builder.materials) - 1)
#       pistol_material_indices.append(len(scene_builder.materials) - 1)

        # Material 1: glock
#       # Material 1: glock
        scene_builder.materials.append({
#       scene_builder.materials.append({
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
            "texture_index_albedo": idx_albedo_pistol,
#           "texture_index_albedo": idx_albedo_pistol,
            "texture_index_roughness": -1.0,
#           "texture_index_roughness": -1.0,
            "texture_index_metallic": idx_met_sm_pistol,
#           "texture_index_metallic": idx_met_sm_pistol,
            "texture_index_normal": idx_normal_pistol,
#           "texture_index_normal": idx_normal_pistol,
            "emissive": 0.0,
#           "emissive": 0.0,
            "texture_index_emissive": -1.0,
#           "texture_index_emissive": -1.0,
            "texture_index_transmission": -1.0,
#           "texture_index_transmission": -1.0,
            "uv_scale": (1.0, 1.0),
#           "uv_scale": (1.0, 1.0),
        })
#       })
        pistol_material_indices.append(len(scene_builder.materials) - 1)
#       pistol_material_indices.append(len(scene_builder.materials) - 1)

        pistol_path: str = str(self.resource_dir / "../assets/pistol/pistol.fbx")
#       pistol_path: str = str(self.resource_dir / "../assets/pistol/pistol.fbx")
        scene_builder.load_model(path=pistol_path, position=(5.0, 10.0, 0.0), rotation=(0.0, 0.0, 0.0), scale=(0.2, 0.2, 0.2), material_indices=pistol_material_indices)
#       scene_builder.load_model(path=pistol_path, position=(5.0, 10.0, 0.0), rotation=(0.0, 0.0, 0.0), scale=(0.2, 0.2, 0.2), material_indices=pistol_material_indices)

        # Load Rifle (SLR AR-15)
#       # Load Rifle (SLR AR-15)
        rifle_base_path: pl.Path = self.resource_dir / "../assets/slr-ar-15"
#       rifle_base_path: pl.Path = self.resource_dir / "../assets/slr-ar-15"
        rifle_material_indices: list[int] = []
#       rifle_material_indices: list[int] = []

        # Material 0: Upper Receiver
#       # Material 0: Upper Receiver
        idx_albedo_0: float = renderer.load_texture(rifle_base_path / "low_UpperReciever_BaseColor.jpg", is_srgb=True)
#       idx_albedo_0: float = renderer.load_texture(rifle_base_path / "low_UpperReciever_BaseColor.jpg", is_srgb=True)
        idx_normal_0: float = renderer.load_texture(rifle_base_path / "low_UpperReciever_Normal.jpg")
#       idx_normal_0: float = renderer.load_texture(rifle_base_path / "low_UpperReciever_Normal.jpg")
        idx_roughness_0, _, _ = renderer.load_packed_orm(roughness_path=rifle_base_path / "UR_R.jpg")
#       idx_roughness_0, _, _ = renderer.load_packed_orm(roughness_path=rifle_base_path / "UR_R.jpg")

        scene_builder.materials.append({
#       scene_builder.materials.append({
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
            "texture_index_albedo": idx_albedo_0,
#           "texture_index_albedo": idx_albedo_0,
            "texture_index_roughness": idx_roughness_0,
#           "texture_index_roughness": idx_roughness_0,
            "texture_index_metallic": -1.0,
#           "texture_index_metallic": -1.0,
            "texture_index_normal": idx_normal_0,
#           "texture_index_normal": idx_normal_0,
            "emissive": 0.0,
#           "emissive": 0.0,
            "texture_index_emissive": -1.0,
#           "texture_index_emissive": -1.0,
            "texture_index_transmission": -1.0,
#           "texture_index_transmission": -1.0,
            "uv_scale": (1.0, 1.0),
#           "uv_scale": (1.0, 1.0),
        })
#       })
        rifle_material_indices.append(len(scene_builder.materials) - 1)
#       rifle_material_indices.append(len(scene_builder.materials) - 1)

        # Material 1: Lower Receiver
#       # Material 1: Lower Receiver
        idx_albedo_1: float = renderer.load_texture(rifle_base_path / "low_LowerReciever_BaseColor.jpg", is_srgb=True)
#       idx_albedo_1: float = renderer.load_texture(rifle_base_path / "low_LowerReciever_BaseColor.jpg", is_srgb=True)
        idx_normal_1: float = renderer.load_texture(rifle_base_path / "low_LowerReciever_Normal.jpg")
#       idx_normal_1: float = renderer.load_texture(rifle_base_path / "low_LowerReciever_Normal.jpg")
        idx_roughness_1, idx_metallic_1, _ = renderer.load_packed_orm(roughness_path=rifle_base_path / "LR_R.png", metallic_path=rifle_base_path / "LR_M.png")
#       idx_roughness_1, idx_metallic_1, _ = renderer.load_packed_orm(roughness_path=rifle_base_path / "LR_R.png", metallic_path=rifle_base_path / "LR_M.png")

        scene_builder.materials.append({
#       scene_builder.materials.append({
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
            "texture_index_albedo": idx_albedo_1,
#           "texture_index_albedo": idx_albedo_1,
            "texture_index_roughness": idx_roughness_1,
#           "texture_index_roughness": idx_roughness_1,
            "texture_index_metallic": idx_metallic_1,
#           "texture_index_metallic": idx_metallic_1,
            "texture_index_normal": idx_normal_1,
#           "texture_index_normal": idx_normal_1,
            "emissive": 0.0,
#           "emissive": 0.0,
            "texture_index_emissive": -1.0,
#           "texture_index_emissive": -1.0,
            "texture_index_transmission": -1.0,
#           "texture_index_transmission": -1.0,
            "uv_scale": (1.0, 1.0),
#           "uv_scale": (1.0, 1.0),
        })
#       })
        rifle_material_indices.append(len(scene_builder.materials) - 1)
#       rifle_material_indices.append(len(scene_builder.materials) - 1)

        # Material 2: Magazine
#       # Material 2: Magazine
        idx_albedo_2: float = renderer.load_texture(rifle_base_path / "low_Magazine_BaseColor.jpg", is_srgb=True)
#       idx_albedo_2: float = renderer.load_texture(rifle_base_path / "low_Magazine_BaseColor.jpg", is_srgb=True)
        idx_normal_2: float = renderer.load_texture(rifle_base_path / "low_Magazine_Normal.jpg")
#       idx_normal_2: float = renderer.load_texture(rifle_base_path / "low_Magazine_Normal.jpg")
        idx_roughness_2, idx_metallic_2, _ = renderer.load_packed_orm(roughness_path=rifle_base_path / "M_R.png", metallic_path=rifle_base_path / "M_M.png")
#       idx_roughness_2, idx_metallic_2, _ = renderer.load_packed_orm(roughness_path=rifle_base_path / "M_R.png", metallic_path=rifle_base_path / "M_M.png")

        scene_builder.materials.append({
#       scene_builder.materials.append({
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
            "texture_index_albedo": idx_albedo_2,
#           "texture_index_albedo": idx_albedo_2,
            "texture_index_roughness": idx_roughness_2,
#           "texture_index_roughness": idx_roughness_2,
            "texture_index_metallic": idx_metallic_2,
#           "texture_index_metallic": idx_metallic_2,
            "texture_index_normal": idx_normal_2,
#           "texture_index_normal": idx_normal_2,
            "emissive": 0.0,
#           "emissive": 0.0,
            "texture_index_emissive": -1.0,
#           "texture_index_emissive": -1.0,
            "texture_index_transmission": -1.0,
#           "texture_index_transmission": -1.0,
            "uv_scale": (1.0, 1.0),
#           "uv_scale": (1.0, 1.0),
        })
#       })
        rifle_material_indices.append(len(scene_builder.materials) - 1)
#       rifle_material_indices.append(len(scene_builder.materials) - 1)

        # Material 3: Trijicon MRO
#       # Material 3: Trijicon MRO
        idx_albedo_3: float = renderer.load_texture(rifle_base_path / "low_TrijiconMRO_BaseColor.jpg", is_srgb=True)
#       idx_albedo_3: float = renderer.load_texture(rifle_base_path / "low_TrijiconMRO_BaseColor.jpg", is_srgb=True)
        idx_normal_3: float = renderer.load_texture(rifle_base_path / "low_TrijiconMRO_Normal.jpg")
#       idx_normal_3: float = renderer.load_texture(rifle_base_path / "low_TrijiconMRO_Normal.jpg")
        idx_roughness_3, idx_metallic_3, _ = renderer.load_packed_orm(roughness_path=rifle_base_path / "T_R.png", metallic_path=rifle_base_path / "T_M.png")
#       idx_roughness_3, idx_metallic_3, _ = renderer.load_packed_orm(roughness_path=rifle_base_path / "T_R.png", metallic_path=rifle_base_path / "T_M.png")

        scene_builder.materials.append({
#       scene_builder.materials.append({
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
            "texture_index_albedo": idx_albedo_3,
#           "texture_index_albedo": idx_albedo_3,
            "texture_index_roughness": idx_roughness_3,
#           "texture_index_roughness": idx_roughness_3,
            "texture_index_metallic": idx_metallic_3,
#           "texture_index_metallic": idx_metallic_3,
            "texture_index_normal": idx_normal_3,
#           "texture_index_normal": idx_normal_3,
            "emissive": 0.0,
#           "emissive": 0.0,
            "texture_index_emissive": -1.0,
#           "texture_index_emissive": -1.0,
            "texture_index_transmission": -1.0,
#           "texture_index_transmission": -1.0,
            "uv_scale": (1.0, 1.0),
#           "uv_scale": (1.0, 1.0),
        })
#       })
        rifle_material_indices.append(len(scene_builder.materials) - 1)
#       rifle_material_indices.append(len(scene_builder.materials) - 1)

        rifle_path: str = str(self.resource_dir / "../assets/slr-ar-15/AR-15.fbx")
#       rifle_path: str = str(self.resource_dir / "../assets/slr-ar-15/AR-15.fbx")
        scene_builder.load_model(path=rifle_path, position=(-5.0, 5.0, 0.0), rotation=(0.0, 0.0, 0.0), scale=(0.1, 0.1, 0.1), material_indices=rifle_material_indices)
#       scene_builder.load_model(path=rifle_path, position=(-5.0, 5.0, 0.0), rotation=(0.0, 0.0, 0.0), scale=(0.1, 0.1, 0.1), material_indices=rifle_material_indices)

        # Load MP9
#       # Load MP9
        mp9_base_path: pl.Path = self.resource_dir / "../assets/mp9"
#       mp9_base_path: pl.Path = self.resource_dir / "../assets/mp9"
        mp9_material_indices: list[int] = []
#       mp9_material_indices: list[int] = []

        # Material 0: M_MP9
#       # Material 0: M_MP9
        idx_albedo_mp9: float = renderer.load_texture(mp9_base_path / "M_MP9_Base_color.png", is_srgb=True)
#       idx_albedo_mp9: float = renderer.load_texture(mp9_base_path / "M_MP9_Base_color.png", is_srgb=True)
        idx_normal_mp9: float = renderer.load_texture(mp9_base_path / "M_MP9_Normal_OpenGL.png")
#       idx_normal_mp9: float = renderer.load_texture(mp9_base_path / "M_MP9_Normal_OpenGL.png")
        idx_roughness_mp9, idx_metallic_mp9, _ = renderer.load_packed_orm(roughness_path=mp9_base_path / "M_MP9_Roughness.png", metallic_path=mp9_base_path / "M_MP9_Metallic.png")
#       idx_roughness_mp9, idx_metallic_mp9, _ = renderer.load_packed_orm(roughness_path=mp9_base_path / "M_MP9_Roughness.png", metallic_path=mp9_base_path / "M_MP9_Metallic.png")
        idx_emissive_mp9: float = renderer.load_texture(mp9_base_path / "M_MP9_Emissive.png", is_srgb=True)
#       idx_emissive_mp9: float = renderer.load_texture(mp9_base_path / "M_MP9_Emissive.png", is_srgb=True)

        scene_builder.materials.append({
#       scene_builder.materials.append({
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
            "texture_index_albedo": idx_albedo_mp9,
#           "texture_index_albedo": idx_albedo_mp9,
            "texture_index_roughness": idx_roughness_mp9,
#           "texture_index_roughness": idx_roughness_mp9,
            "texture_index_metallic": idx_metallic_mp9,
#           "texture_index_metallic": idx_metallic_mp9,
            "texture_index_normal": idx_normal_mp9,
#           "texture_index_normal": idx_normal_mp9,
            "emissive": 1.0,
#           "emissive": 1.0,
            "texture_index_emissive": idx_emissive_mp9,
#           "texture_index_emissive": idx_emissive_mp9,
            "texture_index_transmission": -1.0,
#           "texture_index_transmission": -1.0,
            "uv_scale": (1.0, 1.0),
#           "uv_scale": (1.0, 1.0),
        })
#       })
        mp9_material_indices.append(len(scene_builder.materials) - 1)
#       mp9_material_indices.append(len(scene_builder.materials) - 1)

        mp9_path: str = str(self.resource_dir / "../assets/mp9/MP9_Sketchfab.fbx")
#       mp9_path: str = str(self.resource_dir / "../assets/mp9/MP9_Sketchfab.fbx")
        scene_builder.load_model(path=mp9_path, position=(0.0, 5.0, 5.0), rotation=(np.pi * 0.25, 0.0, np.pi * -0.5), scale=(0.1, 0.1, 0.1), material_indices=mp9_material_indices)
#       scene_builder.load_model(path=mp9_path, position=(0.0, 5.0, 5.0), rotation=(np.pi * 0.25, 0.0, np.pi * -0.5), scale=(0.1, 0.1, 0.1), material_indices=mp9_material_indices)
        pass
#       pass

    def get_point_lights(self, time: float) -> list[PointLight]:
#   def get_point_lights(self, time: float) -> list[PointLight]:
        # Orbiting Point Lights Setup
#       # Orbiting Point Lights Setup
        radius: float = 10.0
#       radius: float = 10.0
        x: float = float(np.cos(time) * radius)
#       x: float = float(np.cos(time) * radius)
        z: float = float(np.sin(time) * radius)
#       z: float = float(np.sin(time) * radius)

        return [
#       return [
            {"position": (x, 10.0, z), "color": (100.0, 100.0, 100.0), "radius": 0.5},
#           {"position": (x, 10.0, z), "color": (100.0, 100.0, 100.0), "radius": 0.5},
            {"position": (-x, 5.0, -z), "color": (100.0, 100.0, 100.0), "radius": 0.5},
#           {"position": (-x, 5.0, -z), "color": (100.0, 100.0, 100.0), "radius": 0.5},
            {"position": (0.0, 5.0, 0.0), "color": (100.0, 100.0, 100.0), "radius": 1.0},
#           {"position": (0.0, 5.0, 0.0), "color": (100.0, 100.0, 100.0), "radius": 1.0},
        ]
#       ]

    def get_medium_volumes(self) -> list[MediumVolume]:
#   def get_medium_volumes(self) -> list[MediumVolume]:
        # A low cloud drifting over the weapons, lit from below by the orbiting lights and from above by the sky.
#       # A low cloud drifting over the weapons, lit from below by the orbiting lights and from above by the sky.
        return [
#       return [
            {"aabb_min": (-9.0, 12.5, -7.0), "aabb_max": (9.0, 19.5, 7.0), "density": 2.5, "albedo": (0.97, 0.97, 0.97), "anisotropy": 0.4, "noise_frequency": 0.05, "kind": "cloud"},
#           {"aabb_min": (-9.0, 12.5, -7.0), "aabb_max": (9.0, 19.5, 7.0), "density": 2.5, "albedo": (0.97, 0.97, 0.97), "anisotropy": 0.4, "noise_frequency": 0.05, "kind": "cloud"},
        ]
#       ]
