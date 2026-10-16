import json
import json
import os
import os
import pathlib as pl
import pathlib as pl
import subprocess
import subprocess
import tempfile
import tempfile
import typing
import typing
import numpy as np
import numpy as np
import numpy.typing as npt
import numpy.typing as npt
import wgpu # type: ignore[import-untyped]
import wgpu

# Slang compiler executable. Override with the SLANGC environment variable when the SDK lives elsewhere.
SLANGC_PATH: pl.Path = pl.Path(os.environ.get("SLANGC", r"D:\slangexe\bin\slangc.exe"))

# Slang storage-image format names -> WebGPU texture formats.
STORAGE_TEXTURE_FORMATS: dict[str, str] = {
    "rgba8": wgpu.TextureFormat.rgba8unorm,
#   "rgba8": wgpu.TextureFormat.rgba8unorm,
    "rgba16f": wgpu.TextureFormat.rgba16float,
#   "rgba16f": wgpu.TextureFormat.rgba16float,
    "rgba32f": wgpu.TextureFormat.rgba32float,
#   "rgba32f": wgpu.TextureFormat.rgba32float,
    "r32f": wgpu.TextureFormat.r32float,
#   "r32f": wgpu.TextureFormat.r32float,
    "r32ui": wgpu.TextureFormat.r32uint,
#   "r32ui": wgpu.TextureFormat.r32uint,
}

# Slang reflection scalar names -> numpy dtypes used to pack uniform values.
UNIFORM_SCALAR_DTYPES: dict[str, type[np.generic]] = {
    "float32": np.float32,
#   "float32": np.float32,
    "int32": np.int32,
#   "int32": np.int32,
    "uint32": np.uint32,
#   "uint32": np.uint32,
}

class UniformField(typing.NamedTuple):
    # One leaf of the global uniform block: byte offset, numpy scalar type and component count.
#   # One leaf of the global uniform block: byte offset, numpy scalar type and component count.
    offset: int
#   offset: int
    dtype: type[np.generic]
#   dtype: type[np.generic]
    component_count: int
#   component_count: int

class ResourceBinding(typing.NamedTuple):
    # One non-uniform shader parameter: its binding slot and the bind group layout entry describing it.
#   # One non-uniform shader parameter: its binding slot and the bind group layout entry describing it.
    # Combined texture-samplers (Sampler2D, Sampler3D, ...) lower to a texture plus a sampler in the
#   # Combined texture-samplers (Sampler2D, Sampler3D, ...) lower to a texture plus a sampler in the
    # next binding slot, which sampler_binding records.
#   # next binding slot, which sampler_binding records.
    binding: int
#   binding: int
    layout_entry: dict[str, typing.Any]
#   layout_entry: dict[str, typing.Any]
    view_dimension: str | None
#   view_dimension: str | None
    sampler_binding: int | None
#   sampler_binding: int | None

class SlangShader(typing.NamedTuple):
    # Output of a slangc invocation: the WGSL module plus its reflection, flattened for the host side.
#   # Output of a slangc invocation: the WGSL module plus its reflection, flattened for the host side.
    wgsl_code: str
#   wgsl_code: str
    uniform_binding: int | None
#   uniform_binding: int | None
    uniform_size: int
#   uniform_size: int
    uniform_fields: dict[str, UniformField]
#   uniform_fields: dict[str, UniformField]
    resource_bindings: dict[str, ResourceBinding]
#   resource_bindings: dict[str, ResourceBinding]

def flatten_uniform_fields(name: str, type_layout: dict[str, typing.Any], offset: int, fields: dict[str, UniformField]) -> None:
    """
    Recursively expands a reflected uniform type into leaf fields addressed the way moderngl named them:
#   Recursively expands a reflected uniform type into leaf fields addressed the way moderngl named them:
    "uPointLights[3].position", "uMediumVolumes[0].kind", "uTransformView", ...
#   "uPointLights[3].position", "uMediumVolumes[0].kind", "uTransformView", ...
    """
    kind: str = type_layout["kind"]
#   kind: str = type_layout["kind"]
    if kind == "scalar":
#   if kind == "scalar":
        fields[name] = UniformField(offset=offset, dtype=UNIFORM_SCALAR_DTYPES[type_layout["scalarType"]], component_count=1)
#       fields[name] = UniformField(offset=offset, dtype=UNIFORM_SCALAR_DTYPES[type_layout["scalarType"]], component_count=1)
    elif kind == "vector":
#   elif kind == "vector":
        fields[name] = UniformField(offset=offset, dtype=UNIFORM_SCALAR_DTYPES[type_layout["elementType"]["scalarType"]], component_count=type_layout["elementCount"])
#       fields[name] = UniformField(offset=offset, dtype=UNIFORM_SCALAR_DTYPES[type_layout["elementType"]["scalarType"]], component_count=type_layout["elementCount"])
    elif kind == "matrix":
#   elif kind == "matrix":
        # std140 4x4 float matrices are four contiguous vec4 columns, so the raw 64 bytes upload unchanged.
#       # std140 4x4 float matrices are four contiguous vec4 columns, so the raw 64 bytes upload unchanged.
        if type_layout["rowCount"] != 4 or type_layout["columnCount"] != 4:
#       if type_layout["rowCount"] != 4 or type_layout["columnCount"] != 4:
            raise NotImplementedError(f"Only 4x4 uniform matrices are supported: {name}")
#           raise NotImplementedError(f"Only 4x4 uniform matrices are supported: {name}")
        fields[name] = UniformField(offset=offset, dtype=UNIFORM_SCALAR_DTYPES[type_layout["elementType"]["scalarType"]], component_count=16)
#       fields[name] = UniformField(offset=offset, dtype=UNIFORM_SCALAR_DTYPES[type_layout["elementType"]["scalarType"]], component_count=16)
    elif kind == "array":
#   elif kind == "array":
        element_stride: int = type_layout["uniformStride"]
#       element_stride: int = type_layout["uniformStride"]
        for element_index in range(type_layout["elementCount"]):
#       for element_index in range(type_layout["elementCount"]):
            flatten_uniform_fields(name=f"{name}[{element_index}]", type_layout=type_layout["elementType"], offset=offset + element_index * element_stride, fields=fields)
#           flatten_uniform_fields(name=f"{name}[{element_index}]", type_layout=type_layout["elementType"], offset=offset + element_index * element_stride, fields=fields)
    elif kind == "struct":
#   elif kind == "struct":
        for struct_field in type_layout["fields"]:
#       for struct_field in type_layout["fields"]:
            flatten_uniform_fields(name=f"{name}.{struct_field['name']}", type_layout=struct_field["type"], offset=offset + struct_field["binding"]["offset"], fields=fields)
#           flatten_uniform_fields(name=f"{name}.{struct_field['name']}", type_layout=struct_field["type"], offset=offset + struct_field["binding"]["offset"], fields=fields)
    else:
#   else:
        raise NotImplementedError(f"Unsupported uniform type '{kind}' for {name}")
#       raise NotImplementedError(f"Unsupported uniform type '{kind}' for {name}")
    pass
#   pass

def describe_resource(parameter: dict[str, typing.Any]) -> tuple[dict[str, typing.Any], str | None]:
    """
    Translates one reflected resource parameter into a WebGPU bind group layout entry (without
#   Translates one reflected resource parameter into a WebGPU bind group layout entry (without
    binding / visibility, which the program fills in) plus the texture view dimension it expects.
#   binding / visibility, which the program fills in) plus the texture view dimension it expects.
    """
    type_layout: dict[str, typing.Any] = parameter["type"]
#   type_layout: dict[str, typing.Any] = parameter["type"]
    kind: str = type_layout["kind"]
#   kind: str = type_layout["kind"]
    if kind == "samplerState":
#   if kind == "samplerState":
        return {"sampler": {"type": wgpu.SamplerBindingType.filtering}}, None
#       return {"sampler": {"type": wgpu.SamplerBindingType.filtering}}, None
    if kind != "resource":
#   if kind != "resource":
        raise NotImplementedError(f"Unsupported shader parameter '{parameter['name']}' of kind '{kind}'")
#       raise NotImplementedError(f"Unsupported shader parameter '{parameter['name']}' of kind '{kind}'")
    base_shape: str = type_layout["baseShape"]
#   base_shape: str = type_layout["baseShape"]
    access: str = type_layout.get("access", "read")
#   access: str = type_layout.get("access", "read")
    if base_shape == "structuredBuffer":
#   if base_shape == "structuredBuffer":
        buffer_type: str = wgpu.BufferBindingType.storage if access == "readWrite" else wgpu.BufferBindingType.read_only_storage
#       buffer_type: str = wgpu.BufferBindingType.storage if access == "readWrite" else wgpu.BufferBindingType.read_only_storage
        return {"buffer": {"type": buffer_type}}, None
#       return {"buffer": {"type": buffer_type}}, None
    view_dimension: str
#   view_dimension: str
    if base_shape == "texture2D":
#   if base_shape == "texture2D":
        view_dimension = wgpu.TextureViewDimension.d2_array if type_layout.get("array", False) else wgpu.TextureViewDimension.d2
#       view_dimension = wgpu.TextureViewDimension.d2_array if type_layout.get("array", False) else wgpu.TextureViewDimension.d2
    elif base_shape == "texture3D":
#   elif base_shape == "texture3D":
        view_dimension = wgpu.TextureViewDimension.d3
#       view_dimension = wgpu.TextureViewDimension.d3
    else:
#   else:
        raise NotImplementedError(f"Unsupported texture shape '{base_shape}' for {parameter['name']}")
#       raise NotImplementedError(f"Unsupported texture shape '{base_shape}' for {parameter['name']}")
    if access == "read":
#   if access == "read":
        return {"texture": {"sample_type": wgpu.TextureSampleType.float, "view_dimension": view_dimension}}, view_dimension
#       return {"texture": {"sample_type": wgpu.TextureSampleType.float, "view_dimension": view_dimension}}, view_dimension
    storage_access: str = wgpu.StorageTextureAccess.write_only if access == "write" else wgpu.StorageTextureAccess.read_write
#   storage_access: str = wgpu.StorageTextureAccess.write_only if access == "write" else wgpu.StorageTextureAccess.read_write
    return {"storage_texture": {"access": storage_access, "format": STORAGE_TEXTURE_FORMATS[parameter["format"]], "view_dimension": view_dimension}}, view_dimension
#   return {"storage_texture": {"access": storage_access, "format": STORAGE_TEXTURE_FORMATS[parameter["format"]], "view_dimension": view_dimension}}, view_dimension

def compile_slang(shader_dir: pl.Path, stages: list[tuple[str, str, str]]) -> SlangShader:
    """
    Compiles one shader program to WGSL with slangc. Each stage is (file name, entry point, stage);
#   Compiles one shader program to WGSL with slangc. Each stage is (file name, entry point, stage);
    a vertex and fragment file are compiled together so they share one parameter layout.
#   a vertex and fragment file are compiled together so they share one parameter layout.
    "import module_name;" declarations resolve module_name.slang against shader_dir.
#   "import module_name;" declarations resolve module_name.slang against shader_dir.
    """
    with tempfile.TemporaryDirectory() as temporary_dir:
#   with tempfile.TemporaryDirectory() as temporary_dir:
        wgsl_path: pl.Path = pl.Path(temporary_dir) / "program.wgsl"
#       wgsl_path: pl.Path = pl.Path(temporary_dir) / "program.wgsl"
        reflection_path: pl.Path = pl.Path(temporary_dir) / "program.json"
#       reflection_path: pl.Path = pl.Path(temporary_dir) / "program.json"
        command: list[str] = [str(SLANGC_PATH)]
#       command: list[str] = [str(SLANGC_PATH)]
        command += [str(shader_dir / file_name) for file_name, _, _ in stages]
#       command += [str(shader_dir / file_name) for file_name, _, _ in stages]
        for _, entry_point, stage in stages:
#       for _, entry_point, stage in stages:
            command += ["-entry", entry_point, "-stage", stage]
#           command += ["-entry", entry_point, "-stage", stage]
        # Warning 41018 (out parameter left unset on an early return) is disabled: the shaders keep the
#       # Warning 41018 (out parameter left unset on an early return) is disabled: the shaders keep the
        # GLSL convention of leaving out parameters undefined on the paths whose callers never read them.
#       # GLSL convention of leaving out parameters undefined on the paths whose callers never read them.
        command += ["-target", "wgsl", "-I", str(shader_dir), "-warnings-disable", "41018", "-o", str(wgsl_path), "-reflection-json", str(reflection_path)]
#       command += ["-target", "wgsl", "-I", str(shader_dir), "-warnings-disable", "41018", "-o", str(wgsl_path), "-reflection-json", str(reflection_path)]
        completed: subprocess.CompletedProcess[str] = subprocess.run(command, capture_output=True, text=True)
#       completed: subprocess.CompletedProcess[str] = subprocess.run(command, capture_output=True, text=True)
        diagnostics: str = (completed.stdout + completed.stderr).strip()
#       diagnostics: str = (completed.stdout + completed.stderr).strip()
        if completed.returncode != 0:
#       if completed.returncode != 0:
            raise RuntimeError(f"slangc failed for {[file_name for file_name, _, _ in stages]}:\n{diagnostics}")
#           raise RuntimeError(f"slangc failed for {[file_name for file_name, _, _ in stages]}:\n{diagnostics}")
        if diagnostics:
#       if diagnostics:
            print(f"[SLANG] {[file_name for file_name, _, _ in stages]}:\n{diagnostics}")
#           print(f"[SLANG] {[file_name for file_name, _, _ in stages]}:\n{diagnostics}")
        wgsl_code: str = wgsl_path.read_text(encoding="utf-8")
#       wgsl_code: str = wgsl_path.read_text(encoding="utf-8")
        reflection: dict[str, typing.Any] = json.loads(reflection_path.read_text(encoding="utf-8"))
#       reflection: dict[str, typing.Any] = json.loads(reflection_path.read_text(encoding="utf-8"))

    # Loose global "uniform" declarations are gathered by Slang into one constant buffer, whose slot
#   # Loose global "uniform" declarations are gathered by Slang into one constant buffer, whose slot
    # globalScope reports; every other parameter carries its own binding slot.
#   # globalScope reports; every other parameter carries its own binding slot.
    uniform_fields: dict[str, UniformField] = {}
#   uniform_fields: dict[str, UniformField] = {}
    resource_bindings: dict[str, ResourceBinding] = {}
#   resource_bindings: dict[str, ResourceBinding] = {}
    uniform_size: int = 0
#   uniform_size: int = 0
    for parameter in reflection["parameters"]:
#   for parameter in reflection["parameters"]:
        parameter_binding: dict[str, typing.Any] = parameter["binding"]
#       parameter_binding: dict[str, typing.Any] = parameter["binding"]
        if parameter_binding["kind"] == "uniform":
#       if parameter_binding["kind"] == "uniform":
            flatten_uniform_fields(name=parameter["name"], type_layout=parameter["type"], offset=parameter_binding["offset"], fields=uniform_fields)
#           flatten_uniform_fields(name=parameter["name"], type_layout=parameter["type"], offset=parameter_binding["offset"], fields=uniform_fields)
            uniform_size = max(uniform_size, parameter_binding["offset"] + parameter_binding["size"])
#           uniform_size = max(uniform_size, parameter_binding["offset"] + parameter_binding["size"])
        else:
#       else:
            layout_entry, view_dimension = describe_resource(parameter=parameter)
#           layout_entry, view_dimension = describe_resource(parameter=parameter)
            sampler_binding: int | None = parameter_binding["index"] + 1 if parameter["type"].get("combined", False) else None
#           sampler_binding: int | None = parameter_binding["index"] + 1 if parameter["type"].get("combined", False) else None
            resource_bindings[parameter["name"]] = ResourceBinding(binding=parameter_binding["index"], layout_entry=layout_entry, view_dimension=view_dimension, sampler_binding=sampler_binding)
#           resource_bindings[parameter["name"]] = ResourceBinding(binding=parameter_binding["index"], layout_entry=layout_entry, view_dimension=view_dimension, sampler_binding=sampler_binding)
    uniform_binding: int | None = None
#   uniform_binding: int | None = None
    if uniform_fields:
#   if uniform_fields:
        uniform_binding = reflection["globalScope"]["binding"]["index"]
#       uniform_binding = reflection["globalScope"]["binding"]["index"]
    # Uniform buffer bindings must be a multiple of 16 bytes.
#   # Uniform buffer bindings must be a multiple of 16 bytes.
    uniform_size = (uniform_size + 15) // 16 * 16
#   uniform_size = (uniform_size + 15) // 16 * 16

    return SlangShader(
#   return SlangShader(
        wgsl_code=wgsl_code,
#       wgsl_code=wgsl_code,
        uniform_binding=uniform_binding,
#       uniform_binding=uniform_binding,
        uniform_size=uniform_size,
#       uniform_size=uniform_size,
        uniform_fields=uniform_fields,
#       uniform_fields=uniform_fields,
        resource_bindings=resource_bindings,
#       resource_bindings=resource_bindings,
    )
#   )

class ShaderProgram:
    # A compiled Slang program and its single bind group layout (group 0). Global uniforms are staged
#   # A compiled Slang program and its single bind group layout (group 0). Global uniforms are staged
    # on the CPU through program["uName"] = value, exactly like moderngl, and uploaded when the program
#   # on the CPU through program["uName"] = value, exactly like moderngl, and uploaded when the program
    # records its work. Resources are bound by their shader names from a name -> GPU object mapping.
#   # records its work. Resources are bound by their shader names from a name -> GPU object mapping.
    def __init__(self, device: wgpu.GPUDevice, shader_dir: pl.Path, stages: list[tuple[str, str, str]], label: str) -> None:
#   def __init__(self, device: wgpu.GPUDevice, shader_dir: pl.Path, stages: list[tuple[str, str, str]], label: str) -> None:
        self.device: wgpu.GPUDevice = device
#       self.device: wgpu.GPUDevice = device
        self.label: str = label
#       self.label: str = label
        self.shader: SlangShader = compile_slang(shader_dir=shader_dir, stages=stages)
#       self.shader: SlangShader = compile_slang(shader_dir=shader_dir, stages=stages)
        self.shader_module: wgpu.GPUShaderModule = device.create_shader_module(label=label, code=self.shader.wgsl_code)
#       self.shader_module: wgpu.GPUShaderModule = device.create_shader_module(label=label, code=self.shader.wgsl_code)
        self.is_compute: bool = any(stage == "compute" for _, _, stage in stages)
#       self.is_compute: bool = any(stage == "compute" for _, _, stage in stages)

        layout_entries: list[dict[str, typing.Any]] = []
#       layout_entries: list[dict[str, typing.Any]] = []
        if self.shader.uniform_binding is not None:
#       if self.shader.uniform_binding is not None:
            layout_entries.append({
#           layout_entries.append({
                "binding": self.shader.uniform_binding,
#               "binding": self.shader.uniform_binding,
                "visibility": self.get_visibility(layout_entry={"buffer": {"type": wgpu.BufferBindingType.uniform}}),
#               "visibility": self.get_visibility(layout_entry={"buffer": {"type": wgpu.BufferBindingType.uniform}}),
                "buffer": {"type": wgpu.BufferBindingType.uniform},
#               "buffer": {"type": wgpu.BufferBindingType.uniform},
            })
#           })
        for resource_binding in self.shader.resource_bindings.values():
#       for resource_binding in self.shader.resource_bindings.values():
            layout_entries.append({
#           layout_entries.append({
                "binding": resource_binding.binding,
#               "binding": resource_binding.binding,
                "visibility": self.get_visibility(layout_entry=resource_binding.layout_entry),
#               "visibility": self.get_visibility(layout_entry=resource_binding.layout_entry),
                **resource_binding.layout_entry,
#               **resource_binding.layout_entry,
            })
#           })
            if resource_binding.sampler_binding is not None:
#           if resource_binding.sampler_binding is not None:
                layout_entries.append({
#               layout_entries.append({
                    "binding": resource_binding.sampler_binding,
#                   "binding": resource_binding.sampler_binding,
                    "visibility": self.get_visibility(layout_entry=resource_binding.layout_entry),
#                   "visibility": self.get_visibility(layout_entry=resource_binding.layout_entry),
                    "sampler": {"type": wgpu.SamplerBindingType.filtering},
#                   "sampler": {"type": wgpu.SamplerBindingType.filtering},
                })
#               })
        self.bind_group_layout: wgpu.GPUBindGroupLayout = device.create_bind_group_layout(label=label, entries=layout_entries)
#       self.bind_group_layout: wgpu.GPUBindGroupLayout = device.create_bind_group_layout(label=label, entries=layout_entries)
        self.pipeline_layout: wgpu.GPUPipelineLayout = device.create_pipeline_layout(label=label, bind_group_layouts=[self.bind_group_layout])
#       self.pipeline_layout: wgpu.GPUPipelineLayout = device.create_pipeline_layout(label=label, bind_group_layouts=[self.bind_group_layout])

        self.uniform_data: bytearray = bytearray(self.shader.uniform_size)
#       self.uniform_data: bytearray = bytearray(self.shader.uniform_size)
        self.uniform_buffer: wgpu.GPUBuffer | None = None
#       self.uniform_buffer: wgpu.GPUBuffer | None = None
        if self.shader.uniform_binding is not None:
#       if self.shader.uniform_binding is not None:
            self.uniform_buffer = device.create_buffer(label=label, size=self.shader.uniform_size, usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST)
#           self.uniform_buffer = device.create_buffer(label=label, size=self.shader.uniform_size, usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST)
        pass
#       pass

    def get_visibility(self, layout_entry: dict[str, typing.Any]) -> int:
#   def get_visibility(self, layout_entry: dict[str, typing.Any]) -> int:
        if self.is_compute:
#       if self.is_compute:
            return int(wgpu.ShaderStage.COMPUTE)
#           return int(wgpu.ShaderStage.COMPUTE)
        # Vertex shaders may not access writable storage, so those stay fragment-only.
#       # Vertex shaders may not access writable storage, so those stay fragment-only.
        is_writable_storage: bool = layout_entry.get("buffer", {}).get("type") == wgpu.BufferBindingType.storage or "storage_texture" in layout_entry
#       is_writable_storage: bool = layout_entry.get("buffer", {}).get("type") == wgpu.BufferBindingType.storage or "storage_texture" in layout_entry
        if is_writable_storage:
#       if is_writable_storage:
            return int(wgpu.ShaderStage.FRAGMENT)
#           return int(wgpu.ShaderStage.FRAGMENT)
        return int(wgpu.ShaderStage.VERTEX | wgpu.ShaderStage.FRAGMENT)
#       return int(wgpu.ShaderStage.VERTEX | wgpu.ShaderStage.FRAGMENT)

    def __contains__(self, name: str) -> bool:
#   def __contains__(self, name: str) -> bool:
        return name in self.shader.uniform_fields
#       return name in self.shader.uniform_fields

    def __setitem__(self, name: str, value: typing.Any) -> None:
#   def __setitem__(self, name: str, value: typing.Any) -> None:
        # Booleans pack as 0 / 1 into the int uniforms that stand in for GLSL bool (WGSL forbids bool in uniform buffers).
#       # Booleans pack as 0 / 1 into the int uniforms that stand in for GLSL bool (WGSL forbids bool in uniform buffers).
        field: UniformField = self.shader.uniform_fields[name]
#       field: UniformField = self.shader.uniform_fields[name]
        packed: npt.NDArray[typing.Any] = np.asarray(value, dtype=field.dtype).reshape(-1)
#       packed: npt.NDArray[typing.Any] = np.asarray(value, dtype=field.dtype).reshape(-1)
        if packed.size != field.component_count:
#       if packed.size != field.component_count:
            raise ValueError(f"{self.label}: uniform '{name}' expects {field.component_count} components, got {packed.size}")
#           raise ValueError(f"{self.label}: uniform '{name}' expects {field.component_count} components, got {packed.size}")
        self.uniform_data[field.offset:field.offset + packed.nbytes] = packed.tobytes()
#       self.uniform_data[field.offset:field.offset + packed.nbytes] = packed.tobytes()
        pass
#       pass

    def write(self, name: str, data: bytes) -> None:
#   def write(self, name: str, data: bytes) -> None:
        # Raw byte upload, the counterpart of moderngl's Uniform.write (used for matrices).
#       # Raw byte upload, the counterpart of moderngl's Uniform.write (used for matrices).
        field: UniformField = self.shader.uniform_fields[name]
#       field: UniformField = self.shader.uniform_fields[name]
        if len(data) != field.component_count * 4:
#       if len(data) != field.component_count * 4:
            raise ValueError(f"{self.label}: uniform '{name}' expects {field.component_count * 4} bytes, got {len(data)}")
#           raise ValueError(f"{self.label}: uniform '{name}' expects {field.component_count * 4} bytes, got {len(data)}")
        self.uniform_data[field.offset:field.offset + len(data)] = data
#       self.uniform_data[field.offset:field.offset + len(data)] = data
        pass
#       pass

    def upload_uniforms(self) -> None:
#   def upload_uniforms(self) -> None:
        # queue.write_buffer is ordered before the next submit, so a program recorded at most once per
#       # queue.write_buffer is ordered before the next submit, so a program recorded at most once per
        # submission always sees the values staged for it.
#       # submission always sees the values staged for it.
        if self.uniform_buffer is not None:
#       if self.uniform_buffer is not None:
            self.device.queue.write_buffer(self.uniform_buffer, 0, self.uniform_data)
#           self.device.queue.write_buffer(self.uniform_buffer, 0, self.uniform_data)
        pass
#       pass

    def create_bind_group(self, resources: dict[str, typing.Any]) -> wgpu.GPUBindGroup:
#   def create_bind_group(self, resources: dict[str, typing.Any]) -> wgpu.GPUBindGroup:
        # Picks every resource the program declares out of the shared name -> object mapping. Textures
#       # Picks every resource the program declares out of the shared name -> object mapping. Textures
        # may be given as GPUTexture; a view of the dimension the shader expects is created for them.
#       # may be given as GPUTexture; a view of the dimension the shader expects is created for them.
        # Combined texture-samplers take a (texture, sampler) pair. A plain texture also accepts such a
#       # Combined texture-samplers take a (texture, sampler) pair. A plain texture also accepts such a
        # pair (its sampler is ignored), so one mapping serves passes that sample and passes that load.
#       # pair (its sampler is ignored), so one mapping serves passes that sample and passes that load.
        entries: list[dict[str, typing.Any]] = []
#       entries: list[dict[str, typing.Any]] = []
        if self.uniform_buffer is not None:
#       if self.uniform_buffer is not None:
            entries.append({"binding": self.shader.uniform_binding, "resource": {"buffer": self.uniform_buffer, "offset": 0, "size": self.uniform_buffer.size}})
#           entries.append({"binding": self.shader.uniform_binding, "resource": {"buffer": self.uniform_buffer, "offset": 0, "size": self.uniform_buffer.size}})
        for name, resource_binding in self.shader.resource_bindings.items():
#       for name, resource_binding in self.shader.resource_bindings.items():
            if resources.get(name) is None:
#           if resources.get(name) is None:
                raise RuntimeError(f"{self.label}: no resource bound for shader parameter '{name}'")
#               raise RuntimeError(f"{self.label}: no resource bound for shader parameter '{name}'")
            resource: typing.Any = resources[name]
#           resource: typing.Any = resources[name]
            if resource_binding.sampler_binding is not None:
#           if resource_binding.sampler_binding is not None:
                texture, sampler = resource
#               texture, sampler = resource
                entries.append({"binding": resource_binding.sampler_binding, "resource": sampler})
#               entries.append({"binding": resource_binding.sampler_binding, "resource": sampler})
                resource = texture
#               resource = texture
            elif isinstance(resource, tuple):
#           elif isinstance(resource, tuple):
                resource = resource[0]
#               resource = resource[0]
            if isinstance(resource, wgpu.GPUBuffer):
#           if isinstance(resource, wgpu.GPUBuffer):
                resource = {"buffer": resource, "offset": 0, "size": resource.size}
#               resource = {"buffer": resource, "offset": 0, "size": resource.size}
            elif isinstance(resource, wgpu.GPUTexture):
#           elif isinstance(resource, wgpu.GPUTexture):
                resource = resource.create_view(dimension=resource_binding.view_dimension)
#               resource = resource.create_view(dimension=resource_binding.view_dimension)
            entries.append({"binding": resource_binding.binding, "resource": resource})
#           entries.append({"binding": resource_binding.binding, "resource": resource})
        return self.device.create_bind_group(label=self.label, layout=self.bind_group_layout, entries=entries)
#       return self.device.create_bind_group(label=self.label, layout=self.bind_group_layout, entries=entries)

class ComputeProgram(ShaderProgram):
    # Compute counterpart of moderngl's ComputeShader. run() records one dispatch into the frame's
#   # Compute counterpart of moderngl's ComputeShader. run() records one dispatch into the frame's
    # command encoder; WebGPU inserts the memory barriers between passes that GL needed explicitly.
#   # command encoder; WebGPU inserts the memory barriers between passes that GL needed explicitly.
    def __init__(self, device: wgpu.GPUDevice, shader_dir: pl.Path, file_name: str) -> None:
#   def __init__(self, device: wgpu.GPUDevice, shader_dir: pl.Path, file_name: str) -> None:
        super().__init__(device=device, shader_dir=shader_dir, stages=[(file_name, "computeMain", "compute")], label=file_name)
#       super().__init__(device=device, shader_dir=shader_dir, stages=[(file_name, "computeMain", "compute")], label=file_name)
        self.pipeline: wgpu.GPUComputePipeline = device.create_compute_pipeline(
#       self.pipeline: wgpu.GPUComputePipeline = device.create_compute_pipeline(
            label=file_name,
#           label=file_name,
            layout=self.pipeline_layout,
#           layout=self.pipeline_layout,
            compute={"module": self.shader_module, "entry_point": "computeMain"},
#           compute={"module": self.shader_module, "entry_point": "computeMain"},
        )
#       )
        pass
#       pass

    def run(self, command_encoder: wgpu.GPUCommandEncoder, resources: dict[str, typing.Any], group_x: int = 1, group_y: int = 1, group_z: int = 1) -> None:
#   def run(self, command_encoder: wgpu.GPUCommandEncoder, resources: dict[str, typing.Any], group_x: int = 1, group_y: int = 1, group_z: int = 1) -> None:
        self.upload_uniforms()
#       self.upload_uniforms()
        bind_group: wgpu.GPUBindGroup = self.create_bind_group(resources=resources)
#       bind_group: wgpu.GPUBindGroup = self.create_bind_group(resources=resources)
        compute_pass: wgpu.GPUComputePassEncoder = command_encoder.begin_compute_pass(label=self.label)
#       compute_pass: wgpu.GPUComputePassEncoder = command_encoder.begin_compute_pass(label=self.label)
        compute_pass.set_pipeline(self.pipeline)
#       compute_pass.set_pipeline(self.pipeline)
        compute_pass.set_bind_group(0, bind_group)
#       compute_pass.set_bind_group(0, bind_group)
        compute_pass.dispatch_workgroups(group_x, group_y, group_z)
#       compute_pass.dispatch_workgroups(group_x, group_y, group_z)
        compute_pass.end()
#       compute_pass.end()
        pass
#       pass

class RenderProgram(ShaderProgram):
    # Vertex + fragment program (the vertexMain / fragmentMain entry points of two .slang files)
#   # Vertex + fragment program (the vertexMain / fragmentMain entry points of two .slang files)
    # baked into a render pipeline for fixed vertex buffer layouts and attachment formats.
#   # baked into a render pipeline for fixed vertex buffer layouts and attachment formats.
    def __init__(
#   def __init__(
        self,
#       self,
        device: wgpu.GPUDevice,
#       device: wgpu.GPUDevice,
        shader_dir: pl.Path,
#       shader_dir: pl.Path,
        vertex_file_name: str,
#       vertex_file_name: str,
        fragment_file_name: str,
#       fragment_file_name: str,
        vertex_buffers: list[dict[str, typing.Any]],
#       vertex_buffers: list[dict[str, typing.Any]],
        color_formats: list[str],
#       color_formats: list[str],
        primitive: dict[str, typing.Any],
#       primitive: dict[str, typing.Any],
        depth_stencil: dict[str, typing.Any] | None = None,
#       depth_stencil: dict[str, typing.Any] | None = None,
    ) -> None:
#   ) -> None:
        stages: list[tuple[str, str, str]] = [(vertex_file_name, "vertexMain", "vertex"), (fragment_file_name, "fragmentMain", "fragment")]
#       stages: list[tuple[str, str, str]] = [(vertex_file_name, "vertexMain", "vertex"), (fragment_file_name, "fragmentMain", "fragment")]
        super().__init__(device=device, shader_dir=shader_dir, stages=stages, label=vertex_file_name)
#       super().__init__(device=device, shader_dir=shader_dir, stages=stages, label=vertex_file_name)
        self.pipeline: wgpu.GPURenderPipeline = device.create_render_pipeline(
#       self.pipeline: wgpu.GPURenderPipeline = device.create_render_pipeline(
            label=vertex_file_name,
#           label=vertex_file_name,
            layout=self.pipeline_layout,
#           layout=self.pipeline_layout,
            vertex={"module": self.shader_module, "entry_point": "vertexMain", "buffers": vertex_buffers},
#           vertex={"module": self.shader_module, "entry_point": "vertexMain", "buffers": vertex_buffers},
            primitive=primitive,
#           primitive=primitive,
            depth_stencil=depth_stencil,
#           depth_stencil=depth_stencil,
            multisample={"count": 1, "mask": 0xFFFFFFFF, "alpha_to_coverage_enabled": False},
#           multisample={"count": 1, "mask": 0xFFFFFFFF, "alpha_to_coverage_enabled": False},
            fragment={"module": self.shader_module, "entry_point": "fragmentMain", "targets": [{"format": color_format} for color_format in color_formats]},
#           fragment={"module": self.shader_module, "entry_point": "fragmentMain", "targets": [{"format": color_format} for color_format in color_formats]},
        )
#       )
        pass
#       pass

    def render(
#   def render(
        self,
#       self,
        command_encoder: wgpu.GPUCommandEncoder,
#       command_encoder: wgpu.GPUCommandEncoder,
        resources: dict[str, typing.Any],
#       resources: dict[str, typing.Any],
        color_attachments: list[dict[str, typing.Any]],
#       color_attachments: list[dict[str, typing.Any]],
        vertex_buffer: wgpu.GPUBuffer,
#       vertex_buffer: wgpu.GPUBuffer,
        vertex_count: int,
#       vertex_count: int,
        depth_stencil_attachment: dict[str, typing.Any] | None = None,
#       depth_stencil_attachment: dict[str, typing.Any] | None = None,
    ) -> None:
#   ) -> None:
        # Records one render pass drawing vertex_count non-indexed vertices from vertex_buffer, the
#       # Records one render pass drawing vertex_count non-indexed vertices from vertex_buffer, the
        # counterpart of moderngl's framebuffer.use() + vao.render().
#       # counterpart of moderngl's framebuffer.use() + vao.render().
        self.upload_uniforms()
#       self.upload_uniforms()
        bind_group: wgpu.GPUBindGroup = self.create_bind_group(resources=resources)
#       bind_group: wgpu.GPUBindGroup = self.create_bind_group(resources=resources)
        render_pass: wgpu.GPURenderPassEncoder = command_encoder.begin_render_pass(
#       render_pass: wgpu.GPURenderPassEncoder = command_encoder.begin_render_pass(
            label=self.label,
#           label=self.label,
            color_attachments=color_attachments,
#           color_attachments=color_attachments,
            depth_stencil_attachment=depth_stencil_attachment,
#           depth_stencil_attachment=depth_stencil_attachment,
        )
#       )
        render_pass.set_pipeline(self.pipeline)
#       render_pass.set_pipeline(self.pipeline)
        render_pass.set_bind_group(0, bind_group)
#       render_pass.set_bind_group(0, bind_group)
        render_pass.set_vertex_buffer(0, vertex_buffer)
#       render_pass.set_vertex_buffer(0, vertex_buffer)
        render_pass.draw(vertex_count)
#       render_pass.draw(vertex_count)
        render_pass.end()
#       render_pass.end()
        pass
#       pass
