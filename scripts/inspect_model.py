import sys
import sys
import os
import os
import pyassimp
import pyassimp

def inspect(path: str) -> None:
    if not os.path.exists(path):
#   if not os.path.exists(path):
        print(f"Error: File {path} not found.")
#       print(f"Error: File {path} not found.")
        return
#       return

    print(f"Inspecting {path}...")
#   print(f"Inspecting {path}...")

    try:
#   try:
        # Load the scene with minimal processing just to inspect structure
#       # Load the scene with minimal processing just to inspect structure
        with pyassimp.load(path) as scene:
#       with pyassimp.load(path) as scene:

            # Print Materials
#           # Print Materials
            print(f"Materials ({len(scene.materials)}):")
#           print(f"Materials ({len(scene.materials)}):")
            for i, mat in enumerate(scene.materials):
#           for i, mat in enumerate(scene.materials):
                name = "Unknown"
#               name = "Unknown"
                try:
#               try:
                    # Attempt to find the name property
#                   # Attempt to find the name property
                    if hasattr(mat, "properties"):
#                   if hasattr(mat, "properties"):
                        name = mat.properties.get("name", "Unknown")
#                       name = mat.properties.get("name", "Unknown")
                        if name == "Unknown":
#                       if name == "Unknown":
                            for key in mat.properties:
#                           for key in mat.properties:
                                if "name" in key.lower():
#                               if "name" in key.lower():
                                    name = mat.properties[key]
#                                   name = mat.properties[key]
                                    break
#                                   break
                    elif hasattr(mat, "name"):
#                   elif hasattr(mat, "name"):
                         name = mat.name
#                        name = mat.name
                except Exception:
#               except Exception:
                    pass
#                   pass
                print(f"  [{i}] Name: {name}")
#               print(f"  [{i}] Name: {name}")

            # Print Meshes
#           # Print Meshes
            print(f"Meshes ({len(scene.meshes)}):")
#           print(f"Meshes ({len(scene.meshes)}):")
            for i, mesh in enumerate(scene.meshes):
#           for i, mesh in enumerate(scene.meshes):
                material_idx = mesh.materialindex
#               material_idx = mesh.materialindex
                vertex_count = len(mesh.vertices)
#               vertex_count = len(mesh.vertices)
                print(f"  [{i}] Material Index: {material_idx} | Vertices: {vertex_count}")
#               print(f"  [{i}] Material Index: {material_idx} | Vertices: {vertex_count}")

    except Exception as e:
#   except Exception as e:
        print(f"Failed to load model: {e}")
#       print(f"Failed to load model: {e}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
#   if len(sys.argv) < 2:
        print("Usage: python inspect_model.py <path_to_model>")
#       print("Usage: python inspect_model.py <path_to_model>")
    else:
#   else:
        inspect(sys.argv[1])
#       inspect(sys.argv[1])
