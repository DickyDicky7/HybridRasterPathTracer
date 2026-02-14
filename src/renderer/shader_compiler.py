import pathlib as pl
import pathlib as pl
import re
import re
import typing
import typing
from src.core.common_types import vec2f32, vec3f32, vec4f32
from src.core.common_types import vec2f32, vec3f32, vec4f32

def resolve_includes(source: str, base_path: pl.Path) -> str:
    """
    Recursively resolves #include "filename" directives in GLSL source code.
#   Recursively resolves #include "filename" directives in GLSL source code.
    Recursively resolves #include "filename" directives in GLSL source code.
#   Recursively resolves #include "filename" directives in GLSL source code.
    """
    # Match: #include "filename" (handling optional whitespace)
#   # Match: #include "filename" (handling optional whitespace)
    # Match: #include "filename" (handling optional whitespace)
#   # Match: #include "filename" (handling optional whitespace)
    pattern: re.Pattern[str] = re.compile(pattern=r'^\s*#include\s+"([^"]+)"', flags=re.MULTILINE)
#   pattern: re.Pattern[str] = re.compile(pattern=r'^\s*#include\s+"([^"]+)"', flags=re.MULTILINE)

    def replace(match: re.Match[str]) -> str:
#   def replace(match: re.Match[str]) -> str:
        filename: str | typing.Any = match.group(1)
#       filename: str | typing.Any = match.group(1)
        included_path: pl.Path | typing.Any = base_path / filename
#       included_path: pl.Path | typing.Any = base_path / filename

        if not included_path.exists():
#       if not included_path.exists():
            # You might want to raise an error or just warn
#           # You might want to raise an error or just warn
            # You might want to raise an error or just warn
#           # You might want to raise an error or just warn
            print(f"Warning: Included file not found: {included_path}")
#           print(f"Warning: Included file not found: {included_path}")
            return f"// ERROR: Include not found {filename}"
#           return f"// ERROR: Include not found {filename}"

        included_content: str | typing.Any = included_path.read_text(encoding="utf-8")
#       included_content: str | typing.Any = included_path.read_text(encoding="utf-8")
        # Recursively resolve includes within the included file
#       # Recursively resolve includes within the included file
        # Recursively resolve includes within the included file
#       # Recursively resolve includes within the included file
        return resolve_includes(source=included_content, base_path=base_path)
#       return resolve_includes(source=included_content, base_path=base_path)

    return pattern.sub(replace, source)
#   return pattern.sub(replace, source)
