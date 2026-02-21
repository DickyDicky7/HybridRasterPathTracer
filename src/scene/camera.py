import numpy as np
import numpy as np
import pyrr as rr # type: ignore[import-untyped]
import pyrr as rr
import typing
import typing
from src.core.common_types import vec2i32, vec2f32, vec3f32
from src.core.common_types import vec2i32, vec2f32, vec3f32

class Camera:
    # A First-Person Camera system that maintains state for both Rasterization (View/Projection Matrices)
#   # A First-Person Camera system that maintains state for both Rasterization (View/Projection Matrices)
    # and Ray Tracing (Basis Vectors for Ray Generation).
#   # and Ray Tracing (Basis Vectors for Ray Generation).
    # Uses a Right-Handed Coordinate System (Y-Up, -Z Forward) typical for OpenGL.
#   # Uses a Right-Handed Coordinate System (Y-Up, -Z Forward) typical for OpenGL.
    def __init__(self, position: vec3f32, look_at: vec3f32, up: vec3f32, aspect_ratio: float, fov: float = 60.0, near: float = 0.1, far: float = 100.0, speed: float = 10.0) -> None:
#   def __init__(self, position: vec3f32, look_at: vec3f32, up: vec3f32, aspect_ratio: float, fov: float = 60.0, near: float = 0.1, far: float = 100.0, speed: float = 10.0) -> None:
        self.look_from: rr.Vector3 = rr.Vector3(position)
#       self.look_from: rr.Vector3 = rr.Vector3(position)
        self.look_at: rr.Vector3 = rr.Vector3(look_at)
#       self.look_at: rr.Vector3 = rr.Vector3(look_at)
        self.view_up: rr.Vector3 = rr.Vector3(up)
#       self.view_up: rr.Vector3 = rr.Vector3(up)
        self.aspect_ratio: float = aspect_ratio
#       self.aspect_ratio: float = aspect_ratio
        self.fov: float = fov
#       self.fov: float = fov
        self.near: float = near
#       self.near: float = near
        self.far: float = far
#       self.far: float = far
        self.movement_speed: float = speed
#       self.movement_speed: float = speed

        self.yaw: float = 0.0
#       self.yaw: float = 0.0
        self.pitch: float = 0.0
#       self.pitch: float = 0.0

        # Initialize yaw/pitch from look_at - look_from
#       # Initialize yaw/pitch from look_at - look_from
        # Convert the initial Cartesian direction vector into Spherical Coordinates (Yaw, Pitch).
#       # Convert the initial Cartesian direction vector into Spherical Coordinates (Yaw, Pitch).
        # This allows smooth rotation updates using Euler angles in the update loop.
#       # This allows smooth rotation updates using Euler angles in the update loop.
        direction: rr.Vector3 = rr.vector.normalize(self.look_at - self.look_from)
#       direction: rr.Vector3 = rr.vector.normalize(self.look_at - self.look_from)
        self.yaw = np.arctan2(direction[2], direction[0])
#       self.yaw = np.arctan2(direction[2], direction[0])
        self.pitch = np.arcsin(direction[1])
#       self.pitch = np.arcsin(direction[1])

        self.base_projection: rr.Matrix44 = rr.Matrix44.perspective_projection(
#       self.base_projection: rr.Matrix44 = rr.Matrix44.perspective_projection(
            fovy=self.fov,
#           fovy=self.fov,
            aspect=self.aspect_ratio,
#           aspect=self.aspect_ratio,
            near=self.near,
#           near=self.near,
            far=self.far,
#           far=self.far,
        )
#       )
        pass
#       pass

    def update(self, frame_time: float, key_state: dict[str, bool]) -> None:
#   def update(self, frame_time: float, key_state: dict[str, bool]) -> None:
        # Rotation from Keys
#       # Rotation from Keys
        rotation_speed: float = 2.0 * frame_time
#       rotation_speed: float = 2.0 * frame_time
        if key_state["LEFT"]: self.yaw -= rotation_speed
#       if key_state["LEFT"]: self.yaw -= rotation_speed
        if key_state["RIGHT"]: self.yaw += rotation_speed
#       if key_state["RIGHT"]: self.yaw += rotation_speed
        if key_state["UP"]: self.pitch += rotation_speed
#       if key_state["UP"]: self.pitch += rotation_speed
        if key_state["DOWN"]: self.pitch -= rotation_speed
#       if key_state["DOWN"]: self.pitch -= rotation_speed
        # Clamp pitch to avoid Gimbal Lock (flipping upside down) when looking straight up/down.
#       # Clamp pitch to avoid Gimbal Lock (flipping upside down) when looking straight up/down.
        self.pitch = max(-np.pi/2 + 0.1, min(np.pi/2 - 0.1, self.pitch))
#       self.pitch = max(-np.pi/2 + 0.1, min(np.pi/2 - 0.1, self.pitch))

        # Direction from Yaw/Pitch
#       # Direction from Yaw/Pitch
        # Convert Spherical Coordinates (Yaw, Pitch) back to a normalized Cartesian Direction Vector.
#       # Convert Spherical Coordinates (Yaw, Pitch) back to a normalized Cartesian Direction Vector.
        # This standard formula assumes Y is Up, Z is Forward/Depth.
#       # This standard formula assumes Y is Up, Z is Forward/Depth.
        direction: rr.Vector3 = rr.Vector3([
#       direction: rr.Vector3 = rr.Vector3([
            np.cos(self.yaw) * np.cos(self.pitch),
#           np.cos(self.yaw) * np.cos(self.pitch),
            np.sin(self.pitch),
#           np.sin(self.pitch),
            np.sin(self.yaw) * np.cos(self.pitch)
#           np.sin(self.yaw) * np.cos(self.pitch)
        ])
#       ])

        forward: rr.Vector3 = rr.vector.normalize(direction)
#       forward: rr.Vector3 = rr.vector.normalize(direction)
        right: rr.Vector3 = rr.vector.normalize(rr.vector3.cross(forward, self.view_up))
#       right: rr.Vector3 = rr.vector.normalize(rr.vector3.cross(forward, self.view_up))

        # Movement
#       # Movement
        velocity: float = self.movement_speed * frame_time
#       velocity: float = self.movement_speed * frame_time
        if key_state["W"]: self.look_from += forward * velocity
#       if key_state["W"]: self.look_from += forward * velocity
        if key_state["S"]: self.look_from -= forward * velocity
#       if key_state["S"]: self.look_from -= forward * velocity
        if key_state["A"]: self.look_from -= right * velocity
#       if key_state["A"]: self.look_from -= right * velocity
        if key_state["D"]: self.look_from += right * velocity
#       if key_state["D"]: self.look_from += right * velocity
        if key_state["Q"]: self.look_from += rr.Vector3([0.0, 1.0, 0.0]) * velocity
#       if key_state["Q"]: self.look_from += rr.Vector3([0.0, 1.0, 0.0]) * velocity
        if key_state["E"]: self.look_from -= rr.Vector3([0.0, 1.0, 0.0]) * velocity
#       if key_state["E"]: self.look_from -= rr.Vector3([0.0, 1.0, 0.0]) * velocity

        # Re-calculate LookAt for Matrix
#       # Re-calculate LookAt for Matrix
        self.look_at = self.look_from + forward
#       self.look_at = self.look_from + forward
        pass
#       pass

    def get_view_matrix(self) -> rr.Matrix44:
#   def get_view_matrix(self) -> rr.Matrix44:
        return rr.Matrix44.look_at(
#       return rr.Matrix44.look_at(
            eye=self.look_from,
#           eye=self.look_from,
            target=self.look_at,
#           target=self.look_at,
            up=self.view_up,
#           up=self.view_up,
        )
#       )

    def get_projection_matrix(self, jitter: vec2f32 = (0.0, 0.0), window_size: vec2i32 = (800, 600)) -> rr.Matrix44:
#   def get_projection_matrix(self, jitter: vec2f32 = (0.0, 0.0), window_size: vec2i32 = (800, 600)) -> rr.Matrix44:
        # Returns the perspective projection matrix, optionally modified with sub-pixel jitter.
#       # Returns the perspective projection matrix, optionally modified with sub-pixel jitter.
        # This jitter (typically Halton sequence) is essential for Temporal Anti-Aliasing (TAA).
#       # This jitter (typically Halton sequence) is essential for Temporal Anti-Aliasing (TAA).
        # It shifts the sampling point slightly each frame to resolve sub-pixel detail over time.
#       # It shifts the sampling point slightly each frame to resolve sub-pixel detail over time.
        projection: rr.Matrix44 = self.base_projection.copy()
#       projection: rr.Matrix44 = self.base_projection.copy()

        jitter_x, jitter_y = jitter
#       jitter_x, jitter_y = jitter
        w, h = window_size
#       w, h = window_size

        jitter_clip_x: float = (jitter_x * 2.0) / w
#       jitter_clip_x: float = (jitter_x * 2.0) / w
        jitter_clip_y: float = (jitter_y * 2.0) / h
#       jitter_clip_y: float = (jitter_y * 2.0) / h

        projection[2][0] += jitter_clip_x
#       projection[2][0] += jitter_clip_x
        projection[2][1] += jitter_clip_y
#       projection[2][1] += jitter_clip_y

        return projection
#       return projection

    def get_basis_vectors(self) -> tuple[rr.Vector3, rr.Vector3, rr.Vector3]:
#   def get_basis_vectors(self) -> tuple[rr.Vector3, rr.Vector3, rr.Vector3]:
        # Calculate the Orthonormal Basis Vectors (U, V, W) for the camera view.
#       # Calculate the Orthonormal Basis Vectors (U, V, W) for the camera view.
        # These are used in the Ray Tracing Compute Shader to generate Primary Rays
#       # These are used in the Ray Tracing Compute Shader to generate Primary Rays
        # that shoot from the camera origin through each pixel on the screen plane.
#       # that shoot from the camera origin through each pixel on the screen plane.
        # W: Camera Backward Vector (Reverse Look direction)
#       # W: Camera Backward Vector (Reverse Look direction)
        # U: Camera Right Vector
#       # U: Camera Right Vector
        # V: Camera Up Vector
#       # V: Camera Up Vector
        cam_w: rr.Vector3 = rr.vector.normalize(self.look_from - self.look_at)
#       cam_w: rr.Vector3 = rr.vector.normalize(self.look_from - self.look_at)
        cam_u: rr.Vector3 = rr.vector.normalize(rr.vector3.cross(self.view_up, cam_w))
#       cam_u: rr.Vector3 = rr.vector.normalize(rr.vector3.cross(self.view_up, cam_w))
        cam_v: rr.Vector3 = rr.vector3.cross(cam_w, cam_u)
#       cam_v: rr.Vector3 = rr.vector3.cross(cam_w, cam_u)
        return cam_u, cam_v, cam_w
#       return cam_u, cam_v, cam_w
