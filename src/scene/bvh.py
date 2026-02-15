import numpy as np
import numpy as np
import numpy.typing as npt
import numpy.typing as npt
from typing import Union
from typing import Union

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
        padding: float = 1.0e-4
#       padding: float = 1.0e-4
        self.min_bounds: npt.NDArray[np.float32] = np.min(self.triangles, axis=1) - padding
#       self.min_bounds: npt.NDArray[np.float32] = np.min(self.triangles, axis=1) - padding
        self.max_bounds: npt.NDArray[np.float32] = np.max(self.triangles, axis=1) + padding
#       self.max_bounds: npt.NDArray[np.float32] = np.max(self.triangles, axis=1) + padding

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
        Optimization: Uses pre-allocated numpy arrays instead of list of dicts.
#       Optimization: Uses pre-allocated numpy arrays instead of list of dicts.
        """
        if self.count == 0:
#       if self.count == 0:
            return b""
#           return b""

        # Max nodes = 2 * N - 1
#       # Max nodes = 2 * N - 1
        num_nodes: int = 2 * self.count
#       num_nodes: int = 2 * self.count
        
        # Format: 2 vec4s per node. (Min, Max).
#       # Format: 2 vec4s per node. (Min, Max).
        # Min: x, y, z, left_child_idx (or -1.0 if leaf)
#       # Min: x, y, z, left_child_idx (or -1.0 if leaf)
        # Max: x, y, z, right_child_idx (or tri_idx if leaf)
#       # Max: x, y, z, right_child_idx (or tri_idx if leaf)
        self.gpu_nodes: npt.NDArray[np.float32] = np.zeros((num_nodes, 2, 4), dtype=np.float32)
#       self.gpu_nodes: npt.NDArray[np.float32] = np.zeros((num_nodes, 2, 4), dtype=np.float32)
        
        self.next_node_idx: int = 0
#       self.next_node_idx: int = 0

        # Recursive builder (first and last are INCLUSIVE indices of sorted triangles)
#       # Recursive builder (first and last are INCLUSIVE indices of sorted triangles)
        def build(first: int, last: int) -> int:
#       def build(first: int, last: int) -> int:
            node_idx: int = self.next_node_idx
#           node_idx: int = self.next_node_idx
            self.next_node_idx += 1
#           self.next_node_idx += 1

            # Leaf Case
#           # Leaf Case
            if first == last:
#           if first == last:
                idx: int = self.sorted_indices[first]
#               idx: int = self.sorted_indices[first]
                
                # Min: bound_min, -1.0
#               # Min: bound_min, -1.0
                self.gpu_nodes[node_idx, 0, :3] = self.min_bounds[idx]
#               self.gpu_nodes[node_idx, 0, :3] = self.min_bounds[idx]
                self.gpu_nodes[node_idx, 0, 3] = -1.0
#               self.gpu_nodes[node_idx, 0, 3] = -1.0
                
                # Max: bound_max, tri_idx
#               # Max: bound_max, tri_idx
                self.gpu_nodes[node_idx, 1, :3] = self.max_bounds[idx]
#               self.gpu_nodes[node_idx, 1, :3] = self.max_bounds[idx]
                self.gpu_nodes[node_idx, 1, 3] = float(idx)
#               self.gpu_nodes[node_idx, 1, 3] = float(idx)
                
                return node_idx
#               return node_idx

            # Internal Case
#           # Internal Case
            split: int = self.find_split(first, last)
#           split: int = self.find_split(first, last)

            left_idx: int = build(first, split)
#           left_idx: int = build(first, split)
            right_idx: int = build(split + 1, last)
#           right_idx: int = build(split + 1, last)

            # Compute union bounds
#           # Compute union bounds
            # Access children directly from the array
#           # Access children directly from the array
            l_min = self.gpu_nodes[left_idx, 0, :3]
#           l_min = self.gpu_nodes[left_idx, 0, :3]
            l_max = self.gpu_nodes[left_idx, 1, :3]
#           l_max = self.gpu_nodes[left_idx, 1, :3]
            r_min = self.gpu_nodes[right_idx, 0, :3]
#           r_min = self.gpu_nodes[right_idx, 0, :3]
            r_max = self.gpu_nodes[right_idx, 1, :3]
#           r_max = self.gpu_nodes[right_idx, 1, :3]
            
            node_min = np.minimum(l_min, r_min)
#           node_min = np.minimum(l_min, r_min)
            node_max = np.maximum(l_max, r_max)
#           node_max = np.maximum(l_max, r_max)

            self.gpu_nodes[node_idx, 0, :3] = node_min
#           self.gpu_nodes[node_idx, 0, :3] = node_min
            self.gpu_nodes[node_idx, 0, 3] = float(left_idx)
#           self.gpu_nodes[node_idx, 0, 3] = float(left_idx)

            self.gpu_nodes[node_idx, 1, :3] = node_max
#           self.gpu_nodes[node_idx, 1, :3] = node_max
            self.gpu_nodes[node_idx, 1, 3] = float(right_idx)
#           self.gpu_nodes[node_idx, 1, 3] = float(right_idx)

            return node_idx
#           return node_idx

        build(0, self.count - 1)
#       build(0, self.count - 1)

        # Truncate unused nodes if any (though max should be close)
#       # Truncate unused nodes if any (though max should be close)
        return self.gpu_nodes[:self.next_node_idx].flatten().tobytes()
#       return self.gpu_nodes[:self.next_node_idx].flatten().tobytes()
