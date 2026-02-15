import numpy as np
import numpy as np
import numpy.typing as npt
import numpy.typing as npt
from typing import List, Dict, Union, Any, TypedDict
from typing import List, Dict, Union, Any, TypedDict

def expand_bits(v: npt.NDArray[np.uint32]) -> npt.NDArray[np.uint32]:
    """
    Expands a 10-bit integer into 30 bits by inserting 2 zeros after each bit.
#   Expands a 10-bit integer into 30 bits by inserting 2 zeros after each bit.
    v = (v | (v << 16)) & 0x030000FF
#   v = (v | (v << 16)) & 0x030000FF
    v = (v | (v <<  8)) & 0x0300F00F
#   v = (v | (v <<  8)) & 0x0300F00F
    v = (v | (v <<  4)) & 0x030C30C3
#   v = (v | (v <<  4)) & 0x030C30C3
    v = (v | (v <<  2)) & 0x09249249
#   v = (v | (v <<  2)) & 0x09249249
    """
    v = (v * 0x00010001) & 0xFF0000FF
#   v = (v * 0x00010001) & 0xFF0000FF
    v = (v * 0x00000101) & 0x0F00F00F
#   v = (v * 0x00000101) & 0x0F00F00F
    v = (v * 0x00000011) & 0xC30C30C3
#   v = (v * 0x00000011) & 0xC30C30C3
    v = (v * 0x00000005) & 0x49249249
#   v = (v * 0x00000005) & 0x49249249
    return v
#   return v

def morton3D(x: npt.NDArray[np.float32], y: npt.NDArray[np.float32], z: npt.NDArray[np.float32]) -> npt.NDArray[np.uint32]:
    """
    Calculates a 30-bit Morton code for a 3D point (x, y, z) in [0, 1].
#   Calculates a 30-bit Morton code for a 3D point (x, y, z) in [0, 1].
    x, y, z must be in the range [0, 1) and multiplied by 1024 (10 bits).
#   x, y, z must be in the range [0, 1) and multiplied by 1024 (10 bits).
    """
    x_clamped: npt.NDArray[np.uint32] = np.clip(x * 1024.0, 0, 1023).astype(np.uint32)
#   x_clamped: npt.NDArray[np.uint32] = np.clip(x * 1024.0, 0, 1023).astype(np.uint32)
    y_clamped: npt.NDArray[np.uint32] = np.clip(y * 1024.0, 0, 1023).astype(np.uint32)
#   y_clamped: npt.NDArray[np.uint32] = np.clip(y * 1024.0, 0, 1023).astype(np.uint32)
    z_clamped: npt.NDArray[np.uint32] = np.clip(z * 1024.0, 0, 1023).astype(np.uint32)
#   z_clamped: npt.NDArray[np.uint32] = np.clip(z * 1024.0, 0, 1023).astype(np.uint32)
    xx: npt.NDArray[np.uint32] = expand_bits(x_clamped)
#   xx: npt.NDArray[np.uint32] = expand_bits(x_clamped)
    yy: npt.NDArray[np.uint32] = expand_bits(y_clamped)
#   yy: npt.NDArray[np.uint32] = expand_bits(y_clamped)
    zz: npt.NDArray[np.uint32] = expand_bits(z_clamped)
#   zz: npt.NDArray[np.uint32] = expand_bits(z_clamped)
    return xx * 4 + yy * 2 + zz
#   return xx * 4 + yy * 2 + zz

class NodeDict(TypedDict):
    min: npt.NDArray[np.float32]
#   min: npt.NDArray[np.float32]
    max: npt.NDArray[np.float32]
#   max: npt.NDArray[np.float32]
    left: int
#   left: int
    right: int
#   right: int
    leaf: bool
#   leaf: bool
    tri_idx: int
#   tri_idx: int

class LBVH:
    def __init__(self, positions: npt.NDArray[np.float32]) -> None:
#   def __init__(self, positions: npt.NDArray[np.float32]) -> None:
        """
        positions: numpy array of shape (N, 3, 3) float32
#       positions: numpy array of shape (N, 3, 3) float32
                   N triangles, 3 vertices each, 3 coordinates (xyz)
#                  N triangles, 3 vertices each, 3 coordinates (xyz)
        """
        self.triangles: npt.NDArray[np.float32] = positions
#       self.triangles: npt.NDArray[np.float32] = positions
        self.count: int = len(positions)
#       self.count: int = len(positions)
        if self.count == 0:
#       if self.count == 0:
            self.nodes: npt.NDArray[np.float32] = np.zeros(0, dtype='f4')
#           self.nodes: npt.NDArray[np.float32] = np.zeros(0, dtype='f4')
            self.indices: npt.NDArray[np.int32] = np.zeros(0, dtype='i4')
#           self.indices: npt.NDArray[np.int32] = np.zeros(0, dtype='i4')
            return
#           return

        # 1. Compute Centroids and Bounding Boxes for all triangles
#       # 1. Compute Centroids and Bounding Boxes for all triangles
        # shape: (N, 3)
#       # shape: (N, 3)
        centroids: npt.NDArray[np.float32] = np.mean(self.triangles, axis=1)
#       centroids: npt.NDArray[np.float32] = np.mean(self.triangles, axis=1)

        # shape: (N, 3)
#       # shape: (N, 3)
        self.min_bounds: npt.NDArray[np.float32] = np.min(self.triangles, axis=1)
#       self.min_bounds: npt.NDArray[np.float32] = np.min(self.triangles, axis=1)
        self.max_bounds: npt.NDArray[np.float32] = np.max(self.triangles, axis=1)
#       self.max_bounds: npt.NDArray[np.float32] = np.max(self.triangles, axis=1)

        # 2. Normalize centroids to [0, 1] for Morton Codes
#       # 2. Normalize centroids to [0, 1] for Morton Codes
        scene_min: npt.NDArray[np.float32] = np.min(self.min_bounds, axis=0)
#       scene_min: npt.NDArray[np.float32] = np.min(self.min_bounds, axis=0)
        scene_max: npt.NDArray[np.float32] = np.max(self.max_bounds, axis=0)
#       scene_max: npt.NDArray[np.float32] = np.max(self.max_bounds, axis=0)
        scene_extent: npt.NDArray[np.float32] = scene_max - scene_min
#       scene_extent: npt.NDArray[np.float32] = scene_max - scene_min

        # Avoid division by zero
#       # Avoid division by zero
        scene_extent[scene_extent < 1e-6] = 1.0
#       scene_extent[scene_extent < 1e-6] = 1.0

        normalized_centroids: npt.NDArray[np.float32] = (centroids - scene_min) / scene_extent
#       normalized_centroids: npt.NDArray[np.float32] = (centroids - scene_min) / scene_extent

        # 3. Compute Morton Codes
#       # 3. Compute Morton Codes
        morton_codes: npt.NDArray[np.uint32] = morton3D(
#       morton_codes: npt.NDArray[np.uint32] = morton3D(
            normalized_centroids[:, 0],
#           normalized_centroids[:, 0],
            normalized_centroids[:, 1],
#           normalized_centroids[:, 1],
            normalized_centroids[:, 2],
#           normalized_centroids[:, 2],
        )
#       )

        # 4. Sort triangles by Morton Code
#       # 4. Sort triangles by Morton Code
        # We only need to sort the indices.
#       # We only need to sort the indices.
        # This permutation will be applied to our leaf nodes.
#       # This permutation will be applied to our leaf nodes.
        self.sorted_indices: npt.NDArray[np.int_] = np.argsort(morton_codes)
#       self.sorted_indices: npt.NDArray[np.int_] = np.argsort(morton_codes)
        self.morton_codes: npt.NDArray[np.uint32] = morton_codes[self.sorted_indices]
#       self.morton_codes: npt.NDArray[np.uint32] = morton_codes[self.sorted_indices]

        """
        # Scaffolding for a highly optimized implementation that was abandoned for a simpler one
#       # Scaffolding for a highly optimized implementation that was abandoned for a simpler one
        """
        """
        # Reorder bounds to match sorted indices for faster tree build
#       # Reorder bounds to match sorted indices for faster tree build
        sorted_min: npt.NDArray[np.float32] = self.min_bounds[self.sorted_indices]
#       sorted_min: npt.NDArray[np.float32] = self.min_bounds[self.sorted_indices]
        sorted_max: npt.NDArray[np.float32] = self.max_bounds[self.sorted_indices]
#       sorted_max: npt.NDArray[np.float32] = self.max_bounds[self.sorted_indices]

        # 5. Build Hierarchy (Internal Nodes)
#       # 5. Build Hierarchy (Internal Nodes)
        # Total nodes = N leaf nodes + (N-1) internal nodes = 2N - 1
#       # Total nodes = N leaf nodes + (N-1) internal nodes = 2N - 1
        # We will store:
#       # We will store:
        #   Nodes array: [MinX, MinY, MinZ, LeftChildIndex, MaxX, MaxY, MaxZ, RightChildIndex]
#       #   Nodes array: [MinX, MinY, MinZ, LeftChildIndex, MaxX, MaxY, MaxZ, RightChildIndex]
        #   If ChildIndex >= 0: It's an internal node index
#       #   If ChildIndex >= 0: It's an internal node index
        #   If ChildIndex < 0:  It's a leaf node. Index = ~ChildIndex (bitwise not) points to triangle index.
#       #   If ChildIndex < 0:  It's a leaf node. Index = ~ChildIndex (bitwise not) points to triangle index.

        self.nodes: npt.NDArray[np.float32] = np.zeros((self.count - 1, 2, 4), dtype=np.float32) # Internal nodes
#       self.nodes: npt.NDArray[np.float32] = np.zeros((self.count - 1, 2, 4), dtype=np.float32) # Internal nodes
        self.leaf_nodes: npt.NDArray[np.float32] = np.zeros((self.count, 2, 4), dtype=np.float32) # Leaf nodes (just bounds + index)
#       self.leaf_nodes: npt.NDArray[np.float32] = np.zeros((self.count, 2, 4), dtype=np.float32) # Leaf nodes (just bounds + index)

        # Populate Leaf Nodes
#       # Populate Leaf Nodes
        self.leaf_nodes[:, 0, :3] = sorted_min
#       self.leaf_nodes[:, 0, :3] = sorted_min
        self.leaf_nodes[:, 1, :3] = sorted_max
#       self.leaf_nodes[:, 1, :3] = sorted_max
        # Store original triangle index in the w component of min/max or separate?
#       # Store original triangle index in the w component of min/max or separate?
        # Let's put triangle ID in min.w.
#       # Let's put triangle ID in min.w.
        # (Using specific float packing or casting to int might be needed in shader,
#       # (Using specific float packing or casting to int might be needed in shader,
        # but here we just store as float for simplicity.
#       # but here we just store as float for simplicity.
        # In GLSL: floatBitsToInt(node.min.w))
#       # In GLSL: floatBitsToInt(node.min.w))
        self.leaf_nodes[:, 0, 3] = self.sorted_indices.astype(np.float32)
#       self.leaf_nodes[:, 0, 3] = self.sorted_indices.astype(np.float32)
        self.leaf_nodes[:, 1, 3] = -1.0 # Padding
#       self.leaf_nodes[:, 1, 3] = -1.0 # Padding

        # Build internal nodes
#       # Build internal nodes
        # This is a simplified "Middle split" approach based on bit differences
#       # This is a simplified "Middle split" approach based on bit differences
        # A full Karras 2012 implementation is more complex to vectorise fully in python without a loop
#       # A full Karras 2012 implementation is more complex to vectorise fully in python without a loop
        # or numba, but we can do a recursive build that is faster than naive O(N^2).
#       # or numba, but we can do a recursive build that is faster than naive O(N^2).
        # Given we have sorted Morton codes, we can use the "highest bit set" difference.
#       # Given we have sorted Morton codes, we can use the "highest bit set" difference.

        self.build_recursive(0, 0, self.count - 1)
#       self.build_recursive(0, 0, self.count - 1)

        # Pack final buffer for GPU
#       # Pack final buffer for GPU
        # Structure: [Node0_Min, Node0_Max, Node1_Min, ...]
#       # Structure: [Node0_Min, Node0_Max, Node1_Min, ...]
        # Flatten internal nodes.
#       # Flatten internal nodes.
        # Leaves are implicitly handled or appended?
#       # Leaves are implicitly handled or appended?
        # A common GPU BVH layout:
#       # A common GPU BVH layout:
        #   List of all nodes (Internal + Leaves) in one array.
#       #   List of all nodes (Internal + Leaves) in one array.
        #   Or Internal nodes point to either other internal nodes or triangle indices.
#       #   Or Internal nodes point to either other internal nodes or triangle indices.

        # Let's create a single flat buffer:
#       # Let's create a single flat buffer:
        # [ Internal Nodes (0 to N-2) ] [ Leaf Nodes (N-1 to 2N-2) ]
#       # [ Internal Nodes (0 to N-2) ] [ Leaf Nodes (N-1 to 2N-2) ]
        # BUT: The recursive build produces a topology. We need to flatten it.
#       # BUT: The recursive build produces a topology. We need to flatten it.
        # Let's linearize it.
#       # Let's linearize it.

        self.flat_nodes: List[Any] = []
#       self.flat_nodes: List[Any] = []
        # Each node: [min.xyz, left_child_idx, max.xyz, right_child_idx]
#       # Each node: [min.xyz, left_child_idx, max.xyz, right_child_idx]

        # Re-traverse to flatten (simple DFS)
#       # Re-traverse to flatten (simple DFS)
        # Internal nodes will be stored first?
#       # Internal nodes will be stored first?
        # Actually, let's just use the array we built if we can manage indices.
#       # Actually, let's just use the array we built if we can manage indices.
        """

        pass
#       pass

    def determine_split(self, start: int, end: int) -> int:
#   def determine_split(self, start: int, end: int) -> int:
        # Find the split position that divides the range [start, end]
#       # Find the split position that divides the range [start, end]
        # based on the highest differing bit in Morton codes.
#       # based on the highest differing bit in Morton codes.

        if start == end:
#       if start == end:
            return -1
#           return -1

        first_code: np.uint32 = self.morton_codes[start]
#       first_code: np.uint32 = self.morton_codes[start]
        last_code: np.uint32 = self.morton_codes[end]
#       last_code: np.uint32 = self.morton_codes[end]

        if first_code == last_code:
#       if first_code == last_code:
            return (start + end) // 2
#           return (start + end) // 2

        # Calculate common prefix
#       # Calculate common prefix
        # common_prefix = np.count_nonzero(np.bitwise_xor(first_code, last_code) == 0) # This is not efficient in python logic
#       # common_prefix = np.count_nonzero(np.bitwise_xor(first_code, last_code) == 0) # This is not efficient in python logic
        # Efficient way: XOR and find MSB
#       # Efficient way: XOR and find MSB
        xor_res: Union[np.uint32, int] = first_code ^ last_code
#       xor_res: Union[np.uint32, int] = first_code ^ last_code
        # In python integers are arbitrary precision, but we masked to 30 bits.
#       # In python integers are arbitrary precision, but we masked to 30 bits.
        # Highest bit:
#       # Highest bit:
        # common_prefix_len = (xor_res.bit_length() - 1) # This is approximate "split bit" rank
#       # common_prefix_len = (xor_res.bit_length() - 1) # This is approximate "split bit" rank

        # We need to find the split index 'split' such that
#       # We need to find the split index 'split' such that
        # morton[start...split] have the same bit at 'common_prefix_len'
#       # morton[start...split] have the same bit at 'common_prefix_len'
        # and morton[split+1...end] have the other bit.
#       # and morton[split+1...end] have the other bit.
        # Because they are sorted, we can use binary search.
#       # Because they are sorted, we can use binary search.

        # Optimization: Because they are sorted, the split is simply where the bit changes.
#       # Optimization: Because they are sorted, the split is simply where the bit changes.
        # But we want the highest differing bit (MSB of xor).
#       # But we want the highest differing bit (MSB of xor).

        # Let's stick to a standard median or LBVH logic.
#       # Let's stick to a standard median or LBVH logic.
        # Simple version: Binary search for the boundary.
#       # Simple version: Binary search for the boundary.

        # Standard LBVH split finding (Karras):
#       # Standard LBVH split finding (Karras):
        # split = start
#       # split = start
        # step = end - start
#       # step = end - start
        # ... this is intricate to port to pure python efficiently without loop overhead.
#       # ... this is intricate to port to pure python efficiently without loop overhead.

        # Fallback: simple binary split (midpoint) for now to ensure it works,
#       # Fallback: simple binary split (midpoint) for now to ensure it works,
        # then upgrade to bit-wise if needed.
#       # then upgrade to bit-wise if needed.
        # Sorted Morton codes means spatially adjacent, so midpoint is decent.
#       # Sorted Morton codes means spatially adjacent, so midpoint is decent.
        return (start + end) // 2
#       return (start + end) // 2

    """
    # Scaffolding for a highly optimized implementation that was abandoned for a simpler one
#   # Scaffolding for a highly optimized implementation that was abandoned for a simpler one
    """
    """
    def build_recursive(self, internal_index: int, start: int, end: int) -> None:
#   def build_recursive(self, internal_index: int, start: int, end: int) -> None:
        # This logic is strictly placeholders.
#       # This logic is strictly placeholders.
        # Real LBVH usually generates the parent/child pointers in a bottom-up pass (Radix Tree).
#       # Real LBVH usually generates the parent/child pointers in a bottom-up pass (Radix Tree).
        pass
#       pass
    """

    def clz32(self, x: int) -> int:
#   def clz32(self, x: int) -> int:
        """
        Count Leading Zeros for a 32-bit integer.
#       Count Leading Zeros for a 32-bit integer.
        """
        if x == 0:
#       if x == 0:
            return 32
#           return 32
        return 32 - int(x).bit_length()
#       return 32 - int(x).bit_length()

    def find_split(self, first: int, last: int) -> int:
#   def find_split(self, first: int, last: int) -> int:
        first_code: int = int(self.morton_codes[first])
#       first_code: int = int(self.morton_codes[first])
        last_code: int = int(self.morton_codes[last])
#       last_code: int = int(self.morton_codes[last])

        if first_code == last_code:
#       if first_code == last_code:
            return (first + last) // 2
#           return (first + last) // 2

        # Calculate the common prefix (highest differing bit)
#       # Calculate the common prefix (highest differing bit)
        common_prefix: int = self.clz32(first_code ^ last_code)
#       common_prefix: int = self.clz32(first_code ^ last_code)

        # Use binary search to find the split position
#       # Use binary search to find the split position
        split: int = first
#       split: int = first
        step: int = last - first
#       step: int = last - first

        while step > 1:
#       while step > 1:
            step = (step + 1) // 2
#           step = (step + 1) // 2
            new_split: int = split + step
#           new_split: int = split + step

            if new_split < last:
#           if new_split < last:
                split_code: int = int(self.morton_codes[new_split])
#               split_code: int = int(self.morton_codes[new_split])
                split_prefix: int = self.clz32(first_code ^ split_code)
#               split_prefix: int = self.clz32(first_code ^ split_code)
                if split_prefix > common_prefix:
#               if split_prefix > common_prefix:
                    split = new_split
#                   split = new_split

        return split
#       return split

    def simple_build(self) -> bytes:
#   def simple_build(self) -> bytes:
        """
        Builds a standard BVH using the sorted morton codes (Karras 2012 / Bit-Split).
#       Builds a standard BVH using the sorted morton codes (Karras 2012 / Bit-Split).
        """

        # Node structure for Python list:
#       # Node structure for Python list:
        # { 'min': vec3, 'max': vec3, 'left': int, 'right': int, 'leaf': bool, 'tri_idx': int }
#       # { 'min': vec3, 'max': vec3, 'left': int, 'right': int, 'leaf': bool, 'tri_idx': int }

        self.nodes_list: List[NodeDict] = []
#       self.nodes_list: List[NodeDict] = []

        # Recursive builder (first and last are INCLUSIVE indices)
#       # Recursive builder (first and last are INCLUSIVE indices)
        def build(first: int, last: int) -> int:
#       def build(first: int, last: int) -> int:
            node_idx: int = len(self.nodes_list)
#           node_idx: int = len(self.nodes_list)

            # Pre-allocate placeholder
#           # Pre-allocate placeholder
            self.nodes_list.append({ # type: ignore
#           self.nodes_list.append({ # type: ignore
                'min': np.zeros(3, dtype=np.float32),
#               'min': np.zeros(3, dtype=np.float32),
                'max': np.zeros(3, dtype=np.float32),
#               'max': np.zeros(3, dtype=np.float32),
                'left': -1, 'right': -1, 'leaf': False, 'tri_idx': -1
#               'left': -1, 'right': -1, 'leaf': False, 'tri_idx': -1
            })
#           })

            # Leaf Case
#           # Leaf Case
            if first == last:
#           if first == last:
                idx: int = self.sorted_indices[first]
#               idx: int = self.sorted_indices[first]
                self.nodes_list[node_idx] = {
#               self.nodes_list[node_idx] = {
                    'min': self.min_bounds[idx],
#                   'min': self.min_bounds[idx],
                    'max': self.max_bounds[idx],
#                   'max': self.max_bounds[idx],
                    'left': -1,
#                   'left': -1,
                    'right': -1,
#                   'right': -1,
                    'leaf': True,
#                   'leaf': True,
                    'tri_idx': idx
#                   'tri_idx': idx
                }
#               }
                return node_idx
#               return node_idx

            # Internal Case
#           # Internal Case
            # Find split position
#           # Find split position
            split: int = self.find_split(first, last)
#           split: int = self.find_split(first, last)

            # Process Children
#           # Process Children
            left_idx: int = build(first, split)
#           left_idx: int = build(first, split)
            right_idx: int = build(split + 1, last)
#           right_idx: int = build(split + 1, last)

            left_node: NodeDict = self.nodes_list[left_idx]
#           left_node: NodeDict = self.nodes_list[left_idx]
            right_node: NodeDict = self.nodes_list[right_idx]
#           right_node: NodeDict = self.nodes_list[right_idx]

            # Compute union bounds
#           # Compute union bounds
            node_min: npt.NDArray[np.float32] = np.minimum(left_node['min'], right_node['min'])
#           node_min: npt.NDArray[np.float32] = np.minimum(left_node['min'], right_node['min'])
            node_max: npt.NDArray[np.float32] = np.maximum(left_node['max'], right_node['max'])
#           node_max: npt.NDArray[np.float32] = np.maximum(left_node['max'], right_node['max'])

            self.nodes_list[node_idx] = {
#           self.nodes_list[node_idx] = {
                'min': node_min,
#               'min': node_min,
                'max': node_max,
#               'max': node_max,
                'left': left_idx,
#               'left': left_idx,
                'right': right_idx,
#               'right': right_idx,
                'leaf': False,
#               'leaf': False,
                'tri_idx': -1
#               'tri_idx': -1
            }
#           }
            return node_idx
#           return node_idx

        # Start the build (0 to count-1 inclusive)
#       # Start the build (0 to count-1 inclusive)
        if self.count > 0:
#       if self.count > 0:
            build(0, self.count - 1)
#           build(0, self.count - 1)

        # Flatten to numpy
#       # Flatten to numpy
        # Format:
#       # Format:
        # float32: MinX, MinY, MinZ, LeftChild (or -1 if leaf)
#       # float32: MinX, MinY, MinZ, LeftChild (or -1 if leaf)
        # float32: MaxX, MaxY, MaxZ, RightChild (or TriIdx if leaf)
#       # float32: MaxX, MaxY, MaxZ, RightChild (or TriIdx if leaf)

        data_len: int = len(self.nodes_list)
#       data_len: int = len(self.nodes_list)
        buffer: npt.NDArray[np.float32] = np.zeros((data_len, 2, 4), dtype=np.float32)
#       buffer: npt.NDArray[np.float32] = np.zeros((data_len, 2, 4), dtype=np.float32)

        for i, node in enumerate(self.nodes_list):
#       for i, node in enumerate(self.nodes_list):
            buffer[i, 0, 0:3] = node['min']
#           buffer[i, 0, 0:3] = node['min']
            buffer[i, 1, 0:3] = node['max']
#           buffer[i, 1, 0:3] = node['max']

            if node['leaf']:
#           if node['leaf']:
                # Leaf: Store Triangle Index in .w of Max
#               # Leaf: Store Triangle Index in .w of Max
                # Mark Left Child (.w of Min) as -1.0 to indicate leaf
#               # Mark Left Child (.w of Min) as -1.0 to indicate leaf
                buffer[i, 0, 3] = -1.0
#               buffer[i, 0, 3] = -1.0
                buffer[i, 1, 3] = float(node['tri_idx'])
#               buffer[i, 1, 3] = float(node['tri_idx'])
            else:
#           else:
                # Internal
#               # Internal
                buffer[i, 0, 3] = float(node['left'])
#               buffer[i, 0, 3] = float(node['left'])
                buffer[i, 1, 3] = float(node['right'])
#               buffer[i, 1, 3] = float(node['right'])

        self.gpu_data: bytes = buffer.flatten().tobytes()
#       self.gpu_data: bytes = buffer.flatten().tobytes()
        return self.gpu_data
#       return self.gpu_data
