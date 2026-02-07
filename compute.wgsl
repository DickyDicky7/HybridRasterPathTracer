    enable f16;
//  enable f16;

    const BACKGROUND_TYPE_SKY_BOX_BLUE: u32 = 0u;
//  const BACKGROUND_TYPE_SKY_BOX_BLUE: u32 = 0u;
    const BACKGROUND_TYPE_SKY_BOX_DARK: u32 = 1u;
//  const BACKGROUND_TYPE_SKY_BOX_DARK: u32 = 1u;
    const BACKGROUND_TYPE_SKY_BOX_HDRI: u32 = 2u;
//  const BACKGROUND_TYPE_SKY_BOX_HDRI: u32 = 2u;

    const MATERIAL_TYPE_DIFFUSE   : u32 = 0u;
//  const MATERIAL_TYPE_DIFFUSE   : u32 = 0u;
    const MATERIAL_TYPE_METAL     : u32 = 1u;
//  const MATERIAL_TYPE_METAL     : u32 = 1u;
    const MATERIAL_TYPE_GLOSS     : u32 = 2u;
//  const MATERIAL_TYPE_GLOSS     : u32 = 2u;
    const MATERIAL_TYPE_DIELECTRIC: u32 = 3u;
//  const MATERIAL_TYPE_DIELECTRIC: u32 = 3u;
    const MATERIAL_TYPE_LIGHT     : u32 = 4u;
//  const MATERIAL_TYPE_LIGHT     : u32 = 4u;
    const MATERIAL_TYPE_PRINCIPLED: u32 = 5u;
//  const MATERIAL_TYPE_PRINCIPLED: u32 = 5u;

    const TEXTURE_TYPE_COLOR: u32 = 0u;
//  const TEXTURE_TYPE_COLOR: u32 = 0u;
    const TEXTURE_TYPE_IMAGE: u32 = 1u;
//  const TEXTURE_TYPE_IMAGE: u32 = 1u;
    const TEXTURE_TYPE_CHECKER_STYLE_A: u32 = 2u;
//  const TEXTURE_TYPE_CHECKER_STYLE_A: u32 = 2u;
    const TEXTURE_TYPE_CHECKER_STYLE_B: u32 = 3u;
//  const TEXTURE_TYPE_CHECKER_STYLE_B: u32 = 3u;

/* --- FOG --- */
    const FOG_HEIGHT_DENSITY: f32 = 0.01;
//  const FOG_HEIGHT_DENSITY: f32 = 0.01;
    const FOG_HEIGHT_FALLOFF: f32 = 0.08;
//  const FOG_HEIGHT_FALLOFF: f32 = 0.08;
    const FOG_HEIGHT_COLOR: vec3<f32> = vec3<f32>(0.5, 0.6, 0.7);
//  const FOG_HEIGHT_COLOR: vec3<f32> = vec3<f32>(0.5, 0.6, 0.7);
/* --- FOG --- */

    struct Ray
//  struct Ray
{
    origin: vec3<f32>,
//  origin: vec3<f32>,
    direction: vec3<f32>,
//  direction: vec3<f32>,
    directionInverse: vec3<f32>,
//  directionInverse: vec3<f32>,
}

    struct Sphere
//  struct Sphere
{
    center: vec3<f32>,
//  center: vec3<f32>,
    radius: f32,
//  radius: f32,
    materialIndex: u32,
//  materialIndex: u32,
}

    struct RayHitResult
//  struct RayHitResult
{
    at: vec3<f32>,
//  at: vec3<f32>,
    hittedSideNormal: vec3<f32>,
//  hittedSideNormal: vec3<f32>,
    uvSurfaceCoordinate: vec2<f32>,
//  uvSurfaceCoordinate: vec2<f32>,
    minDistance: f32,
//  minDistance: f32,
    isHitted: bool,
//  isHitted: bool,
    isFrontFaceHitted: bool,
//  isFrontFaceHitted: bool,
    materialIndex: u32,
//  materialIndex: u32,
    triangleIndex: u32,
//  triangleIndex: u32,
}

    struct MaterialLightScatteringResult
//  struct MaterialLightScatteringResult
{
    scatteredRay: Ray,
//  scatteredRay: Ray,
    attenuation: vec3<f32>,
//  attenuation: vec3<f32>,
    emission: vec3<f32>,
//  emission: vec3<f32>,
    isScattered: bool,
//  isScattered: bool,
}

    struct Material
//  struct Material
{
    layer1Roughness: f32,    // 0.0=smooth     1.0=rough
//  layer1Roughness: f32,    // 0.0=smooth     1.0=rough
    layer1Metallic: f32,     // 0.0=dielectric 1.0=metal
//  layer1Metallic: f32,     // 0.0=dielectric 1.0=metal
    layer1Transmission: f32, // 0.0=opaque     1.0=glass
//  layer1Transmission: f32, // 0.0=opaque     1.0=glass
    layer0IOR: f32,
//  layer0IOR: f32,
    layer1IOR: f32,
//  layer1IOR: f32,
    textureIndex: u32,
//  textureIndex: u32,
    materialType: u32,
//  materialType: u32,
}

    struct Interval
//  struct Interval
{
    min: f32,
//  min: f32,
    max: f32,
//  max: f32,
}

    fn _intervalContain (interval: Interval, value: f32) -> bool { return interval.min <= value && value <= interval.max; }
//  fn _intervalContain (interval: Interval, value: f32) -> bool { return interval.min <= value && value <= interval.max; }
    fn _intervalSurround(interval: Interval, value: f32) -> bool { return interval.min <  value && value <  interval.max; }
//  fn _intervalSurround(interval: Interval, value: f32) -> bool { return interval.min <  value && value <  interval.max; }

    struct Texture
//  struct Texture
{
    albedo: vec3<f32>,
//  albedo: vec3<f32>,
    albedoImageIndex: u32,
//  albedoImageIndex: u32,
    albedoTextureType: u32,
//  albedoTextureType: u32,
    albedoScale: f32,
//  albedoScale: f32,
    albedoOTileTextureIndex: u32,
//  albedoOTileTextureIndex: u32,
    albedoETileTextureIndex: u32,
//  albedoETileTextureIndex: u32,
}

    struct Triangle
//  struct Triangle
{
    vertex0: vec3<f32>,
//  vertex0: vec3<f32>,
    vertex0FrontFaceNormalEncoded: u32,
//  vertex0FrontFaceNormalEncoded: u32,
    vertex1: vec3<f32>,
//  vertex1: vec3<f32>,
    vertex1FrontFaceNormalEncoded: u32,
//  vertex1FrontFaceNormalEncoded: u32,
    vertex2: vec3<f32>,
//  vertex2: vec3<f32>,
    vertex2FrontFaceNormalEncoded: u32,
//  vertex2FrontFaceNormalEncoded: u32,
    vertex0UVEncoded: u32,
//  vertex0UVEncoded: u32,
    vertex1UVEncoded: u32,
//  vertex1UVEncoded: u32,
    vertex2UVEncoded: u32,
//  vertex2UVEncoded: u32,
    materialIndex: u32,
//  materialIndex: u32,
}

//     struct Cube
// //  struct Cube
// {
//     center: vec3<f32>,
// //  center: vec3<f32>,
//     halfSize: vec3<f32>,
// //  halfSize: vec3<f32>,
//     materialIndex: u32,
// //  materialIndex: u32,
// }

    struct AABB3D
//  struct AABB3D
{
    cornerLimitAxisX: Interval,
//  cornerLimitAxisX: Interval,
    cornerLimitAxisY: Interval,
//  cornerLimitAxisY: Interval,
    cornerLimitAxisZ: Interval,
//  cornerLimitAxisZ: Interval,
}

    struct BVHNode
//  struct BVHNode
{
    aabb3d: AABB3D,
//  aabb3d: AABB3D,
    data1: i32, // Leaf: ~triangleIndex, Internal: childIndexL,
//  data1: i32, // Leaf: ~triangleIndex, Internal: childIndexL,
    data2: i32, // Leaf: unused        , Internal: childIndexR,
//  data2: i32, // Leaf: unused        , Internal: childIndexR,
}

    const      PI: f32 = 3.1415926535897930; // 1*π
//  const      PI: f32 = 3.1415926535897930; // 1*π
    const TAU    : f32 = 6.2831853071795860; // 2*π
//  const TAU    : f32 = 6.2831853071795860; // 2*π
    const HALF_PI: f32 = 1.5707963267948966; // π/2
//  const HALF_PI: f32 = 1.5707963267948966; // π/2

    fn _lengthSquared(value: vec3<f32>) -> f32
//  fn _lengthSquared(value: vec3<f32>) -> f32
{
    return dot(value, value);
//  return dot(value, value);
}

    fn _makeRay(origin: vec3<f32>, direction: vec3<f32>) -> Ray
//  fn _makeRay(origin: vec3<f32>, direction: vec3<f32>) -> Ray
{
    return Ray(origin, direction, 1.0 / direction);
//  return Ray(origin, direction, 1.0 / direction);
}

    // Helper functions to get min/max component of a vec3
//  // Helper functions to get min/max component of a vec3
    fn _maxVec3F32(v: vec3<f32>) -> f32 { return max(v.x, max(v.y, v.z)); }
//  fn _maxVec3F32(v: vec3<f32>) -> f32 { return max(v.x, max(v.y, v.z)); }
    fn _minVec3F32(v: vec3<f32>) -> f32 { return min(v.x, min(v.y, v.z)); }
//  fn _minVec3F32(v: vec3<f32>) -> f32 { return min(v.x, min(v.y, v.z)); }

    fn _rayHitAABB3D(ray: Ray, rayTravelDistanceLimit: Interval, aabb3d: AABB3D) -> bool
//  fn _rayHitAABB3D(ray: Ray, rayTravelDistanceLimit: Interval, aabb3d: AABB3D) -> bool
{
    // Represent the AABB as two vectors for min and max corners.
//  // Represent the AABB as two vectors for min and max corners.
    let aabb3dMin: vec3<f32> = vec3<f32>(aabb3d.cornerLimitAxisX.min, aabb3d.cornerLimitAxisY.min, aabb3d.cornerLimitAxisZ.min);
//  let aabb3dMin: vec3<f32> = vec3<f32>(aabb3d.cornerLimitAxisX.min, aabb3d.cornerLimitAxisY.min, aabb3d.cornerLimitAxisZ.min);
    let aabb3dMax: vec3<f32> = vec3<f32>(aabb3d.cornerLimitAxisX.max, aabb3d.cornerLimitAxisY.max, aabb3d.cornerLimitAxisZ.max);
//  let aabb3dMax: vec3<f32> = vec3<f32>(aabb3d.cornerLimitAxisX.max, aabb3d.cornerLimitAxisY.max, aabb3d.cornerLimitAxisZ.max);

    // Perform all calculations in parallel on a vector.
//  // Perform all calculations in parallel on a vector.
    let distance0: vec3<f32> = (aabb3dMin - ray.origin) * ray.directionInverse;
//  let distance0: vec3<f32> = (aabb3dMin - ray.origin) * ray.directionInverse;
    let distance1: vec3<f32> = (aabb3dMax - ray.origin) * ray.directionInverse;
//  let distance1: vec3<f32> = (aabb3dMax - ray.origin) * ray.directionInverse;

    // Find the min and max distances for each axis slab.
//  // Find the min and max distances for each axis slab.
    let distanceMinSlabs: vec3<f32> = min(distance0, distance1);
//  let distanceMinSlabs: vec3<f32> = min(distance0, distance1);
    let distanceMaxSlabs: vec3<f32> = max(distance0, distance1);
//  let distanceMaxSlabs: vec3<f32> = max(distance0, distance1);

    // Find the overall nearest and farthest intersection points.
//  // Find the overall nearest and farthest intersection points.
    // The intersection starts at the FARTHEST of the near planes...
//  // The intersection starts at the FARTHEST of the near planes...
    let overallDistanceMin: f32 = _maxVec3F32(distanceMinSlabs);
//  let overallDistanceMin: f32 = _maxVec3F32(distanceMinSlabs);
    // ...and ends at the NEAREST of the far planes.
//  // ...and ends at the NEAREST of the far planes.
    let overallDistanceMax: f32 = _minVec3F32(distanceMaxSlabs);
//  let overallDistanceMax: f32 = _minVec3F32(distanceMaxSlabs);

    // Combine with the user's initial ray distance limits.
//  // Combine with the user's initial ray distance limits.
    let finalDistanceMin: f32 = max(rayTravelDistanceLimit.min, overallDistanceMin);
//  let finalDistanceMin: f32 = max(rayTravelDistanceLimit.min, overallDistanceMin);
    let finalDistanceMax: f32 = min(rayTravelDistanceLimit.max, overallDistanceMax);
//  let finalDistanceMax: f32 = min(rayTravelDistanceLimit.max, overallDistanceMax);

    // The final check is the same: the interval must be valid.
//  // The final check is the same: the interval must be valid.
    return finalDistanceMax > finalDistanceMin;
//  return finalDistanceMax > finalDistanceMin;
}

    fn decodeOctahedralNormal(packedNormal: u32) -> vec3<f32>
//  fn decodeOctahedralNormal(packedNormal: u32) -> vec3<f32>
    {
        var encodedXY: vec2<f32> = vec2<f32>(f32((packedNormal >> 16u) & 0xFFFFu) / 65535.0 * 2.0 - 1.0, f32(packedNormal & 0xFFFFu) / 65535.0 * 2.0 - 1.0);
//      var encodedXY: vec2<f32> = vec2<f32>(f32((packedNormal >> 16u) & 0xFFFFu) / 65535.0 * 2.0 - 1.0, f32(packedNormal & 0xFFFFu) / 65535.0 * 2.0 - 1.0);
        var normal: vec3<f32> = vec3<f32>(encodedXY.x, encodedXY.y, 1.0 - abs(encodedXY.x) - abs(encodedXY.y));
//      var normal: vec3<f32> = vec3<f32>(encodedXY.x, encodedXY.y, 1.0 - abs(encodedXY.x) - abs(encodedXY.y));
        if (normal.z < 0.0)
//      if (normal.z < 0.0)
        {
            let tempX: f32 = normal.x;
//          let tempX: f32 = normal.x;
            let signX: f32 = select(-1.0, 1.0, tempX >= 0.0);
//          let signX: f32 = select(-1.0, 1.0, tempX >= 0.0);
            let signY: f32 = select(-1.0, 1.0, normal.y >= 0.0);
//          let signY: f32 = select(-1.0, 1.0, normal.y >= 0.0);
            normal.x = (1.0 - abs(normal.y)) * signX;
//          normal.x = (1.0 - abs(normal.y)) * signX;
            normal.y = (1.0 - abs(tempX)) * signY;
//          normal.y = (1.0 - abs(tempX)) * signY;
        }
        return normalize(normal);
//      return normalize(normal);
    }

    fn encodeOctahedralNormal(normal: vec3<f32>) -> u32
//  fn encodeOctahedralNormal(normal: vec3<f32>) -> u32
    {
        var n: vec3<f32> = normalize(normal);
//      var n: vec3<f32> = normalize(normal);
        n = n / (abs(n.x) + abs(n.y) + abs(n.z));
//      n = n / (abs(n.x) + abs(n.y) + abs(n.z));
        if (n.z < 0.0)
//      if (n.z < 0.0)
        {
            let prevX: f32 = n.x;
//          let prevX: f32 = n.x;
            let signX: f32 = select(-1.0, 1.0, prevX >= 0.0);
//          let signX: f32 = select(-1.0, 1.0, prevX >= 0.0);
            let signY: f32 = select(-1.0, 1.0, n.y >= 0.0);
//          let signY: f32 = select(-1.0, 1.0, n.y >= 0.0);
            n.x = (1.0 - abs(n.y)) * signX;
//          n.x = (1.0 - abs(n.y)) * signX;
            n.y = (1.0 - abs(prevX)) * signY;
//          n.y = (1.0 - abs(prevX)) * signY;
        }
        let quantizedX: u32 = u32(clamp(n.x * 0.5 + 0.5, 0.0, 1.0) * 65535.0 + 0.5);
//      let quantizedX: u32 = u32(clamp(n.x * 0.5 + 0.5, 0.0, 1.0) * 65535.0 + 0.5);
        let quantizedY: u32 = u32(clamp(n.y * 0.5 + 0.5, 0.0, 1.0) * 65535.0 + 0.5);
//      let quantizedY: u32 = u32(clamp(n.y * 0.5 + 0.5, 0.0, 1.0) * 65535.0 + 0.5);
        return (quantizedX << 16u) | quantizedY;
//      return (quantizedX << 16u) | quantizedY;
    }

    fn decodeQuantizedUV(packedUV: u32) -> vec2<f32>
//  fn decodeQuantizedUV(packedUV: u32) -> vec2<f32>
    {
        let u: f32 = f32((packedUV >> 16u) & 0xFFFFu) / 65535.0;
//      let u: f32 = f32((packedUV >> 16u) & 0xFFFFu) / 65535.0;
        let v: f32 = f32(packedUV & 0xFFFFu) / 65535.0;
//      let v: f32 = f32(packedUV & 0xFFFFu) / 65535.0;
        return vec2<f32>(u, v);
//      return vec2<f32>(u, v);
    }

    fn encodeQuantizedUV(uv: vec2<f32>) -> u32
//  fn encodeQuantizedUV(uv: vec2<f32>) -> u32
    {
        let quantizedU: u32 = u32(clamp(uv.x, 0.0, 1.0) * 65535.0 + 0.5);
//      let quantizedU: u32 = u32(clamp(uv.x, 0.0, 1.0) * 65535.0 + 0.5);
        let quantizedV: u32 = u32(clamp(uv.y, 0.0, 1.0) * 65535.0 + 0.5);
//      let quantizedV: u32 = u32(clamp(uv.y, 0.0, 1.0) * 65535.0 + 0.5);
        return (quantizedU << 16u) | quantizedV;
//      return (quantizedU << 16u) | quantizedV;
    }

    fn _rayHitTriangle_AnyHit(ray: Ray, triangleIndex: u32, rayTravelDistanceLimit: Interval) -> bool
//  fn _rayHitTriangle_AnyHit(ray: Ray, triangleIndex: u32, rayTravelDistanceLimit: Interval) -> bool
{
    let triangle: Triangle = triangles[triangleIndex];
//  let triangle: Triangle = triangles[triangleIndex];

    const EPSILON: f32 = 1.0e-4;
//  const EPSILON: f32 = 1.0e-4;

    let triangleEdge1: vec3<f32> = triangle.vertex1 - triangle.vertex0;
//  let triangleEdge1: vec3<f32> = triangle.vertex1 - triangle.vertex0;
    let triangleEdge2: vec3<f32> = triangle.vertex2 - triangle.vertex0;
//  let triangleEdge2: vec3<f32> = triangle.vertex2 - triangle.vertex0;

    let rayDirectionCrossTriangleEdge2: vec3<f32> = cross(ray.direction, triangleEdge2);
//  let rayDirectionCrossTriangleEdge2: vec3<f32> = cross(ray.direction, triangleEdge2);
    let determinant: f32 = dot(triangleEdge1, rayDirectionCrossTriangleEdge2);
//  let determinant: f32 = dot(triangleEdge1, rayDirectionCrossTriangleEdge2);

    if (abs(determinant) < EPSILON)
//  if (abs(determinant) < EPSILON)
    {
        return false;
//      return false;
    }

    let inverseDeterminant: f32 = 1.0 / determinant;
//  let inverseDeterminant: f32 = 1.0 / determinant;
    let vectorFromTriangleVertex0ToRayOrigin: vec3<f32> = ray.origin - triangle.vertex0;
//  let vectorFromTriangleVertex0ToRayOrigin: vec3<f32> = ray.origin - triangle.vertex0;

    let w1Barycentric: f32 = inverseDeterminant * dot(vectorFromTriangleVertex0ToRayOrigin, rayDirectionCrossTriangleEdge2);
//  let w1Barycentric: f32 = inverseDeterminant * dot(vectorFromTriangleVertex0ToRayOrigin, rayDirectionCrossTriangleEdge2);
    if (w1Barycentric < 0.0 || w1Barycentric > 1.0)
//  if (w1Barycentric < 0.0 || w1Barycentric > 1.0)
    {
        return false;
//      return false;
    }

    let rayOriginCrossTriangleEdge1: vec3<f32> = cross(vectorFromTriangleVertex0ToRayOrigin, triangleEdge1);
//  let rayOriginCrossTriangleEdge1: vec3<f32> = cross(vectorFromTriangleVertex0ToRayOrigin, triangleEdge1);
    let w2Barycentric: f32 = inverseDeterminant * dot(ray.direction, rayOriginCrossTriangleEdge1);
//  let w2Barycentric: f32 = inverseDeterminant * dot(ray.direction, rayOriginCrossTriangleEdge1);
    if (w2Barycentric < 0.0 || w1Barycentric + w2Barycentric > 1.0)
//  if (w2Barycentric < 0.0 || w1Barycentric + w2Barycentric > 1.0)
    {
        return false;
//      return false;
    }

    let distanceFromRayOriginToIntersectionPoint: f32 = inverseDeterminant * dot(triangleEdge2, rayOriginCrossTriangleEdge1);
//  let distanceFromRayOriginToIntersectionPoint: f32 = inverseDeterminant * dot(triangleEdge2, rayOriginCrossTriangleEdge1);

    if (!_intervalSurround(rayTravelDistanceLimit, distanceFromRayOriginToIntersectionPoint))
//  if (!_intervalSurround(rayTravelDistanceLimit, distanceFromRayOriginToIntersectionPoint))
    {
        return false;
//      return false;
    }

    // Hit found within interval. Since we don't care about the *closest* hit for occlusion, just return true.
//  // Hit found within interval. Since we don't care about the *closest* hit for occlusion, just return true.
    return true;
//  return true;
}

    fn _rayHitTriangle(ray: Ray, triangleIndex: u32, rayTravelDistanceLimit: Interval) -> RayHitResult
//  fn _rayHitTriangle(ray: Ray, triangleIndex: u32, rayTravelDistanceLimit: Interval) -> RayHitResult
{
    let triangle: Triangle = triangles[triangleIndex];
//  let triangle: Triangle = triangles[triangleIndex];

    var rayHitResult: RayHitResult;
//  var rayHitResult: RayHitResult;

    rayHitResult.materialIndex = triangle.materialIndex;
//  rayHitResult.materialIndex = triangle.materialIndex;

    const EPSILON: f32 = 1.0e-4;
//  const EPSILON: f32 = 1.0e-4;

    let triangleEdge1: vec3<f32> = triangle.vertex1 - triangle.vertex0;
//  let triangleEdge1: vec3<f32> = triangle.vertex1 - triangle.vertex0;
    let triangleEdge2: vec3<f32> = triangle.vertex2 - triangle.vertex0;
//  let triangleEdge2: vec3<f32> = triangle.vertex2 - triangle.vertex0;

    let rayDirectionCrossTriangleEdge2: vec3<f32> = cross(ray.direction, triangleEdge2);
//  let rayDirectionCrossTriangleEdge2: vec3<f32> = cross(ray.direction, triangleEdge2);
    let determinant: f32 = dot(triangleEdge1, rayDirectionCrossTriangleEdge2);
//  let determinant: f32 = dot(triangleEdge1, rayDirectionCrossTriangleEdge2);

    if (abs(determinant) < EPSILON)
//  if (abs(determinant) < EPSILON)
    {
        rayHitResult.isHitted = false;
//      rayHitResult.isHitted = false;
        return rayHitResult;
//      return rayHitResult;
    }

    let inverseDeterminant: f32 = 1.0 / determinant;
//  let inverseDeterminant: f32 = 1.0 / determinant;
    let vectorFromTriangleVertex0ToRayOrigin: vec3<f32> = ray.origin - triangle.vertex0;
//  let vectorFromTriangleVertex0ToRayOrigin: vec3<f32> = ray.origin - triangle.vertex0;

    let w1Barycentric: f32 = inverseDeterminant * dot(vectorFromTriangleVertex0ToRayOrigin, rayDirectionCrossTriangleEdge2);
//  let w1Barycentric: f32 = inverseDeterminant * dot(vectorFromTriangleVertex0ToRayOrigin, rayDirectionCrossTriangleEdge2);
    if (w1Barycentric < 0.0 || w1Barycentric > 1.0)
//  if (w1Barycentric < 0.0 || w1Barycentric > 1.0)
    {
        rayHitResult.isHitted = false;
//      rayHitResult.isHitted = false;
        return rayHitResult;
//      return rayHitResult;
    }

    let rayOriginCrossTriangleEdge1: vec3<f32> = cross(vectorFromTriangleVertex0ToRayOrigin, triangleEdge1);
//  let rayOriginCrossTriangleEdge1: vec3<f32> = cross(vectorFromTriangleVertex0ToRayOrigin, triangleEdge1);
    let w2Barycentric: f32 = inverseDeterminant * dot(ray.direction, rayOriginCrossTriangleEdge1);
//  let w2Barycentric: f32 = inverseDeterminant * dot(ray.direction, rayOriginCrossTriangleEdge1);
    if (w2Barycentric < 0.0 || w1Barycentric + w2Barycentric > 1.0)
//  if (w2Barycentric < 0.0 || w1Barycentric + w2Barycentric > 1.0)
    {
        rayHitResult.isHitted = false;
//      rayHitResult.isHitted = false;
        return rayHitResult;
//      return rayHitResult;
    }

    let distanceFromRayOriginToIntersectionPoint: f32 = inverseDeterminant * dot(triangleEdge2, rayOriginCrossTriangleEdge1);
//  let distanceFromRayOriginToIntersectionPoint: f32 = inverseDeterminant * dot(triangleEdge2, rayOriginCrossTriangleEdge1);

    if (!_intervalSurround(rayTravelDistanceLimit, distanceFromRayOriginToIntersectionPoint))
//  if (!_intervalSurround(rayTravelDistanceLimit, distanceFromRayOriginToIntersectionPoint))
    {
        rayHitResult.isHitted = false;
//      rayHitResult.isHitted = false;
        return rayHitResult;
//      return rayHitResult;
    }

    /*
        DEFERRED ATTRIBUTE OPTIMIZATION:
//      DEFERRED ATTRIBUTE OPTIMIZATION:
        We skip decoding expensive triangle-vertex-front-face-normals/triangle-vertex-UVs here. We just store the barycentrics
//      We skip decoding expensive triangle-vertex-front-face-normals/triangle-vertex-UVs here. We just store the barycentrics
        temporarily in the 'uvSurfaceCoordinate' field.
//      temporarily in the 'uvSurfaceCoordinate' field.
        'materialIndex' and 'triangleIndex' are stored to allow decoding later.
//      'materialIndex' and 'triangleIndex' are stored to allow decoding later.
    */

    if (distanceFromRayOriginToIntersectionPoint > EPSILON)
//  if (distanceFromRayOriginToIntersectionPoint > EPSILON)
    {
        rayHitResult.isHitted = true;
//      rayHitResult.isHitted = true;

        rayHitResult.minDistance = distanceFromRayOriginToIntersectionPoint;
//      rayHitResult.minDistance = distanceFromRayOriginToIntersectionPoint;

        rayHitResult.triangleIndex = triangleIndex;
//      rayHitResult.triangleIndex = triangleIndex;

        // Store Barycentrics (w1 & w2) in UV field temporarily
//      // Store Barycentrics (w1 & w2) in UV field temporarily
        rayHitResult.uvSurfaceCoordinate = vec2<f32>(w1Barycentric, w2Barycentric);
//      rayHitResult.uvSurfaceCoordinate = vec2<f32>(w1Barycentric, w2Barycentric);

        return rayHitResult;
//      return rayHitResult;
    }
    else
    {
        rayHitResult.isHitted = false;
//      rayHitResult.isHitted = false;
        return rayHitResult;
//      return rayHitResult;
    }
}

    fn _rayHitSphere(ray: Ray, sphereIndex: u32, rayTravelDistanceLimit: Interval) -> RayHitResult
//  fn _rayHitSphere(ray: Ray, sphereIndex: u32, rayTravelDistanceLimit: Interval) -> RayHitResult
{
    let sphere: Sphere = spheres[sphereIndex];
//  let sphere: Sphere = spheres[sphereIndex];
    let fromSphereCenterToRayOrigin: vec3<f32> = sphere.center - ray.origin;
//  let fromSphereCenterToRayOrigin: vec3<f32> = sphere.center - ray.origin;
    let a: f32 = _lengthSquared(ray.direction);
//  let a: f32 = _lengthSquared(ray.direction);
    let h: f32 = dot(ray.direction, fromSphereCenterToRayOrigin);
//  let h: f32 = dot(ray.direction, fromSphereCenterToRayOrigin);
    let c: f32 = _lengthSquared(fromSphereCenterToRayOrigin) - sphere.radius * sphere.radius;
//  let c: f32 = _lengthSquared(fromSphereCenterToRayOrigin) - sphere.radius * sphere.radius;
    let discriminant: f32 = h * h - a * c;
//  let discriminant: f32 = h * h - a * c;
    var rayHitResult: RayHitResult;
//  var rayHitResult: RayHitResult;
    rayHitResult.materialIndex = sphere.materialIndex;
//  rayHitResult.materialIndex = sphere.materialIndex;
    if (discriminant < 0.0)
//  if (discriminant < 0.0)
    {
        rayHitResult.isHitted = false;
//      rayHitResult.isHitted = false;
    }
    else
//  else
    {
        let sqrtDiscriminant: f32 = sqrt(discriminant);
//      let sqrtDiscriminant: f32 = sqrt(discriminant);

        var t: f32 = (h - sqrtDiscriminant) / a;
//      var t: f32 = (h - sqrtDiscriminant) / a;

        if (!_intervalSurround(rayTravelDistanceLimit, t))
//      if (!_intervalSurround(rayTravelDistanceLimit, t))
        {
            t = (h + sqrtDiscriminant) / a;
//          t = (h + sqrtDiscriminant) / a;

            if (!_intervalSurround(rayTravelDistanceLimit, t))
//          if (!_intervalSurround(rayTravelDistanceLimit, t))
            {
                rayHitResult.isHitted = false;
//              rayHitResult.isHitted = false;
                return rayHitResult;
//              return rayHitResult;
            }
        }

        rayHitResult.isHitted = true;
//      rayHitResult.isHitted = true;

        rayHitResult.minDistance = t;
//      rayHitResult.minDistance = t;

        rayHitResult.at = _rayMarch(ray, rayHitResult.minDistance);
//      rayHitResult.at = _rayMarch(ray, rayHitResult.minDistance);

        let outwardNormal: vec3<f32> = (rayHitResult.at - sphere.center) / sphere.radius;
//      let outwardNormal: vec3<f32> = (rayHitResult.at - sphere.center) / sphere.radius;

        rayHitResult.isFrontFaceHitted = dot(ray.direction, outwardNormal) < 0.0;
//      rayHitResult.isFrontFaceHitted = dot(ray.direction, outwardNormal) < 0.0;
        rayHitResult.hittedSideNormal = select(-outwardNormal, outwardNormal, rayHitResult.isFrontFaceHitted);
//      rayHitResult.hittedSideNormal = select(-outwardNormal, outwardNormal, rayHitResult.isFrontFaceHitted);

        let theta: f32 = acos (-outwardNormal.y); // latitude
//      let theta: f32 = acos (-outwardNormal.y); // latitude
        let phi  : f32 = atan2(-outwardNormal.z, outwardNormal.x) + PI; // longitude
//      let phi  : f32 = atan2(-outwardNormal.z, outwardNormal.x) + PI; // longitude

        rayHitResult.uvSurfaceCoordinate = vec2<f32>(phi / (2.0 * PI), theta / PI);
//      rayHitResult.uvSurfaceCoordinate = vec2<f32>(phi / (2.0 * PI), theta / PI);
    }

    return rayHitResult;
//  return rayHitResult;
}

    fn _rayHitSpheresThenTriangles(ray: Ray, rayTravelDistanceLimit: Interval) -> RayHitResult
//  fn _rayHitSpheresThenTriangles(ray: Ray, rayTravelDistanceLimit: Interval) -> RayHitResult
{
    let numberOfSpheres: u32 = arrayLength(&spheres);
//  let numberOfSpheres: u32 = arrayLength(&spheres);
    let numberOfTriangles: u32 = arrayLength(&triangles);
//  let numberOfTriangles: u32 = arrayLength(&triangles);
    var finalRayHitResult: RayHitResult;
//  var finalRayHitResult: RayHitResult;
    var closestRayDistanceSoFar: f32 = rayTravelDistanceLimit.max;
//  var closestRayDistanceSoFar: f32 = rayTravelDistanceLimit.max;
    for (var sphereIndex: u32 = 0u; sphereIndex < numberOfSpheres; sphereIndex++)
//  for (var sphereIndex: u32 = 0u; sphereIndex < numberOfSpheres; sphereIndex++)
    {
        let temporaryRayHitResult: RayHitResult = _rayHitSphere(ray, sphereIndex, Interval(rayTravelDistanceLimit.min, closestRayDistanceSoFar));
//      let temporaryRayHitResult: RayHitResult = _rayHitSphere(ray, sphereIndex, Interval(rayTravelDistanceLimit.min, closestRayDistanceSoFar));
        if (temporaryRayHitResult.isHitted)
//      if (temporaryRayHitResult.isHitted)
        {
            finalRayHitResult = temporaryRayHitResult;
//          finalRayHitResult = temporaryRayHitResult;
            closestRayDistanceSoFar = temporaryRayHitResult.minDistance;
//          closestRayDistanceSoFar = temporaryRayHitResult.minDistance;
        }
    }
    for (var triangleIndex: u32 = 0u; triangleIndex < numberOfTriangles; triangleIndex++)
//  for (var triangleIndex: u32 = 0u; triangleIndex < numberOfTriangles; triangleIndex++)
    {
        let temporaryRayHitResult: RayHitResult = _rayHitTriangle(ray, triangleIndex, Interval(rayTravelDistanceLimit.min, closestRayDistanceSoFar));
//      let temporaryRayHitResult: RayHitResult = _rayHitTriangle(ray, triangleIndex, Interval(rayTravelDistanceLimit.min, closestRayDistanceSoFar));
        if (temporaryRayHitResult.isHitted)
//      if (temporaryRayHitResult.isHitted)
        {
            finalRayHitResult = temporaryRayHitResult;
//          finalRayHitResult = temporaryRayHitResult;
            closestRayDistanceSoFar = temporaryRayHitResult.minDistance;
//          closestRayDistanceSoFar = temporaryRayHitResult.minDistance;
        }
    }

    return finalRayHitResult;
//  return finalRayHitResult;
}

    fn _rayScatter(incomingRay: Ray, recentRayHitResult: RayHitResult, rng: ptr<function, RNG>) -> MaterialLightScatteringResult
//  fn _rayScatter(incomingRay: Ray, recentRayHitResult: RayHitResult, rng: ptr<function, RNG>) -> MaterialLightScatteringResult
{
    var materialLightScatteringResult: MaterialLightScatteringResult;
//  var materialLightScatteringResult: MaterialLightScatteringResult;
    let material: Material = materials[recentRayHitResult.materialIndex];
//  let material: Material = materials[recentRayHitResult.materialIndex];
    switch (material.materialType)
//  switch (material.materialType)
    {
        case MATERIAL_TYPE_DIFFUSE:
//      case MATERIAL_TYPE_DIFFUSE:
        {
            materialLightScatteringResult.scatteredRay.origin = recentRayHitResult.at;
//          materialLightScatteringResult.scatteredRay.origin = recentRayHitResult.at;
            materialLightScatteringResult.scatteredRay.direction = normalize(recentRayHitResult.hittedSideNormal + _generateRandomUnitVector(rng));
//          materialLightScatteringResult.scatteredRay.direction = normalize(recentRayHitResult.hittedSideNormal + _generateRandomUnitVector(rng));
            materialLightScatteringResult.scatteredRay.directionInverse = 1.0 / materialLightScatteringResult.scatteredRay.direction;
//          materialLightScatteringResult.scatteredRay.directionInverse = 1.0 / materialLightScatteringResult.scatteredRay.direction;
            materialLightScatteringResult.attenuation = _textureSampleAlbedo(material.textureIndex, recentRayHitResult.uvSurfaceCoordinate, recentRayHitResult.at);
//          materialLightScatteringResult.attenuation = _textureSampleAlbedo(material.textureIndex, recentRayHitResult.uvSurfaceCoordinate, recentRayHitResult.at);
            materialLightScatteringResult.emission = vec3<f32>(0.0, 0.0, 0.0);
//          materialLightScatteringResult.emission = vec3<f32>(0.0, 0.0, 0.0);
            materialLightScatteringResult.isScattered = true;
//          materialLightScatteringResult.isScattered = true;
        }
        case MATERIAL_TYPE_METAL:
//      case MATERIAL_TYPE_METAL:
        {
            materialLightScatteringResult.scatteredRay.origin = recentRayHitResult.at;
//          materialLightScatteringResult.scatteredRay.origin = recentRayHitResult.at;
            materialLightScatteringResult.scatteredRay.direction = normalize(reflect(incomingRay.direction, recentRayHitResult.hittedSideNormal) + material.layer1Roughness * _generateRandomUnitVector(rng));
//          materialLightScatteringResult.scatteredRay.direction = normalize(reflect(incomingRay.direction, recentRayHitResult.hittedSideNormal) + material.layer1Roughness * _generateRandomUnitVector(rng));
            materialLightScatteringResult.scatteredRay.directionInverse = 1.0 / materialLightScatteringResult.scatteredRay.direction;
//          materialLightScatteringResult.scatteredRay.directionInverse = 1.0 / materialLightScatteringResult.scatteredRay.direction;
            materialLightScatteringResult.attenuation = _textureSampleAlbedo(material.textureIndex, recentRayHitResult.uvSurfaceCoordinate, recentRayHitResult.at);
//          materialLightScatteringResult.attenuation = _textureSampleAlbedo(material.textureIndex, recentRayHitResult.uvSurfaceCoordinate, recentRayHitResult.at);
            materialLightScatteringResult.emission = vec3<f32>(0.0, 0.0, 0.0);
//          materialLightScatteringResult.emission = vec3<f32>(0.0, 0.0, 0.0);
            materialLightScatteringResult.isScattered = true;
//          materialLightScatteringResult.isScattered = true;
        }
        case MATERIAL_TYPE_GLOSS:
//      case MATERIAL_TYPE_GLOSS:
        {
            var etaRatioOfIncidenceOverTransmission: f32 = material.layer1IOR / material.layer0IOR;
//          var etaRatioOfIncidenceOverTransmission: f32 = material.layer1IOR / material.layer0IOR;

            let cosThetaIncident: f32 = min(dot(-incomingRay.direction, recentRayHitResult.hittedSideNormal), 1.0);
//          let cosThetaIncident: f32 = min(dot(-incomingRay.direction, recentRayHitResult.hittedSideNormal), 1.0);

            if (recentRayHitResult.isFrontFaceHitted)
//          if (recentRayHitResult.isFrontFaceHitted)
            {
                etaRatioOfIncidenceOverTransmission = 1.0 / etaRatioOfIncidenceOverTransmission;
//              etaRatioOfIncidenceOverTransmission = 1.0 / etaRatioOfIncidenceOverTransmission;
            }

            let reflectanceProbability: f32 = _reflectance(cosThetaIncident, etaRatioOfIncidenceOverTransmission);
//          let reflectanceProbability: f32 = _reflectance(cosThetaIncident, etaRatioOfIncidenceOverTransmission);

            var scatteredDirection: vec3<f32>; var attenuationColor: vec3<f32>;
//          var scatteredDirection: vec3<f32>; var attenuationColor: vec3<f32>;

            materialLightScatteringResult.isScattered = true;
//          materialLightScatteringResult.isScattered = true;

            if (_pcg32Next(rng) < reflectanceProbability)
//          if (_pcg32Next(rng) < reflectanceProbability)
            {
                scatteredDirection = normalize(reflect(incomingRay.direction, recentRayHitResult.hittedSideNormal) + material.layer1Roughness * _generateRandomUnitVector(rng));
//              scatteredDirection = normalize(reflect(incomingRay.direction, recentRayHitResult.hittedSideNormal) + material.layer1Roughness * _generateRandomUnitVector(rng));
                if (dot(scatteredDirection, recentRayHitResult.hittedSideNormal) <= 0.0)
//              if (dot(scatteredDirection, recentRayHitResult.hittedSideNormal) <= 0.0)
                {
                    /*
                    materialLightScatteringResult.isScattered = false;
//                  materialLightScatteringResult.isScattered = false;
                    */
                    // The rough reflection scattered below the surface. Fall back to a diffuse reflection to conserve energy.
//                  // The rough reflection scattered below the surface. Fall back to a diffuse reflection to conserve energy.
                    scatteredDirection = normalize(recentRayHitResult.hittedSideNormal + _generateRandomUnitVector(rng));
//                  scatteredDirection = normalize(recentRayHitResult.hittedSideNormal + _generateRandomUnitVector(rng));
                }
                attenuationColor = vec3<f32>(1.0, 1.0, 1.0);
//              attenuationColor = vec3<f32>(1.0, 1.0, 1.0);
            }
            else
//          else
            {
                scatteredDirection = normalize(recentRayHitResult.hittedSideNormal + _generateRandomUnitVector(rng));
//              scatteredDirection = normalize(recentRayHitResult.hittedSideNormal + _generateRandomUnitVector(rng));
                attenuationColor = _textureSampleAlbedo(material.textureIndex, recentRayHitResult.uvSurfaceCoordinate, recentRayHitResult.at);
//              attenuationColor = _textureSampleAlbedo(material.textureIndex, recentRayHitResult.uvSurfaceCoordinate, recentRayHitResult.at);
            }

            if (materialLightScatteringResult.isScattered)
//          if (materialLightScatteringResult.isScattered)
            {
                materialLightScatteringResult.scatteredRay.origin = recentRayHitResult.at;
//              materialLightScatteringResult.scatteredRay.origin = recentRayHitResult.at;
                materialLightScatteringResult.scatteredRay.direction = scatteredDirection;
//              materialLightScatteringResult.scatteredRay.direction = scatteredDirection;
                materialLightScatteringResult.scatteredRay.directionInverse = 1.0 / materialLightScatteringResult.scatteredRay.direction;
//              materialLightScatteringResult.scatteredRay.directionInverse = 1.0 / materialLightScatteringResult.scatteredRay.direction;
                materialLightScatteringResult.attenuation = attenuationColor;
//              materialLightScatteringResult.attenuation = attenuationColor;
            }
            else
//          else
            {
                materialLightScatteringResult.attenuation = vec3<f32>(0.0, 0.0, 0.0);
//              materialLightScatteringResult.attenuation = vec3<f32>(0.0, 0.0, 0.0);
            }
            materialLightScatteringResult.emission = vec3<f32>(0.0, 0.0, 0.0);
//          materialLightScatteringResult.emission = vec3<f32>(0.0, 0.0, 0.0);
        }
        case MATERIAL_TYPE_DIELECTRIC:
//      case MATERIAL_TYPE_DIELECTRIC:
        {
            let diffuseRayDirection: vec3<f32> = normalize(recentRayHitResult.hittedSideNormal + _generateRandomUnitVector(rng));
//          let diffuseRayDirection: vec3<f32> = normalize(recentRayHitResult.hittedSideNormal + _generateRandomUnitVector(rng));

            let etaRatioOfIncidenceOverTransmission: f32 = select(material.layer1IOR / material.layer0IOR, material.layer0IOR / material.layer1IOR, recentRayHitResult.isFrontFaceHitted);
//          let etaRatioOfIncidenceOverTransmission: f32 = select(material.layer1IOR / material.layer0IOR, material.layer0IOR / material.layer1IOR, recentRayHitResult.isFrontFaceHitted);

            let cosTheta: f32 = min(dot(-incomingRay.direction, recentRayHitResult.hittedSideNormal), 1.0);
//          let cosTheta: f32 = min(dot(-incomingRay.direction, recentRayHitResult.hittedSideNormal), 1.0);
            let sinTheta: f32 = sqrt(1.0 - cosTheta * cosTheta);
//          let sinTheta: f32 = sqrt(1.0 - cosTheta * cosTheta);
            let notAbleToRefract: bool = sinTheta * etaRatioOfIncidenceOverTransmission > 1.0 || _reflectance(cosTheta, etaRatioOfIncidenceOverTransmission) > _pcg32Next(rng);
//          let notAbleToRefract: bool = sinTheta * etaRatioOfIncidenceOverTransmission > 1.0 || _reflectance(cosTheta, etaRatioOfIncidenceOverTransmission) > _pcg32Next(rng);
            var scatteredRayDirection: vec3<f32>;
//          var scatteredRayDirection: vec3<f32>;

            let reflectDirection: vec3<f32> = mix(reflect(incomingRay.direction, recentRayHitResult.hittedSideNormal), diffuseRayDirection, material.layer1Roughness);
//          let reflectDirection: vec3<f32> = mix(reflect(incomingRay.direction, recentRayHitResult.hittedSideNormal), diffuseRayDirection, material.layer1Roughness);
            let refractDirection: vec3<f32> = mix(refract(incomingRay.direction, recentRayHitResult.hittedSideNormal, etaRatioOfIncidenceOverTransmission), -diffuseRayDirection, material.layer1Roughness);
//          let refractDirection: vec3<f32> = mix(refract(incomingRay.direction, recentRayHitResult.hittedSideNormal, etaRatioOfIncidenceOverTransmission), -diffuseRayDirection, material.layer1Roughness);
            scatteredRayDirection = select(refractDirection, reflectDirection, notAbleToRefract);
//          scatteredRayDirection = select(refractDirection, reflectDirection, notAbleToRefract);

            materialLightScatteringResult.scatteredRay.origin = recentRayHitResult.at;
//          materialLightScatteringResult.scatteredRay.origin = recentRayHitResult.at;
            materialLightScatteringResult.scatteredRay.direction = normalize(scatteredRayDirection);
//          materialLightScatteringResult.scatteredRay.direction = normalize(scatteredRayDirection);
            materialLightScatteringResult.scatteredRay.directionInverse = 1.0 / materialLightScatteringResult.scatteredRay.direction;
//          materialLightScatteringResult.scatteredRay.directionInverse = 1.0 / materialLightScatteringResult.scatteredRay.direction;
            materialLightScatteringResult.attenuation = _textureSampleAlbedo(material.textureIndex, recentRayHitResult.uvSurfaceCoordinate, recentRayHitResult.at);
//          materialLightScatteringResult.attenuation = _textureSampleAlbedo(material.textureIndex, recentRayHitResult.uvSurfaceCoordinate, recentRayHitResult.at);
            materialLightScatteringResult.emission = vec3<f32>(0.0, 0.0, 0.0);
//          materialLightScatteringResult.emission = vec3<f32>(0.0, 0.0, 0.0);
            materialLightScatteringResult.isScattered = true;
//          materialLightScatteringResult.isScattered = true;

/*
            // Absorption
            // Absorption
            if (!recentRayHitResult.isFrontFaceHitted)
//          if (!recentRayHitResult.isFrontFaceHitted)
            {
                materialLightScatteringResult.attenuation *= vec3<f32>
//              materialLightScatteringResult.attenuation *= vec3<f32>
                (
                    exp(-recentRayHitResult.minDistance * 1.0),
//                  exp(-recentRayHitResult.minDistance * 1.0),
                    exp(-recentRayHitResult.minDistance * 1.0),
//                  exp(-recentRayHitResult.minDistance * 1.0),
                    exp(-recentRayHitResult.minDistance * 1.0),
//                  exp(-recentRayHitResult.minDistance * 1.0),
                );
            }
*/
        }
        case MATERIAL_TYPE_LIGHT:
//      case MATERIAL_TYPE_LIGHT:
        {
            materialLightScatteringResult.scatteredRay.origin = vec3<f32>(0.0, 0.0, 0.0);
//          materialLightScatteringResult.scatteredRay.origin = vec3<f32>(0.0, 0.0, 0.0);
            materialLightScatteringResult.scatteredRay.direction = vec3<f32>(0.0, 0.0, 0.0);
//          materialLightScatteringResult.scatteredRay.direction = vec3<f32>(0.0, 0.0, 0.0);
            materialLightScatteringResult.scatteredRay.directionInverse = vec3<f32>(0.0, 0.0, 0.0);
//          materialLightScatteringResult.scatteredRay.directionInverse = vec3<f32>(0.0, 0.0, 0.0);
            materialLightScatteringResult.attenuation = vec3<f32>(0.0, 0.0, 0.0);
//          materialLightScatteringResult.attenuation = vec3<f32>(0.0, 0.0, 0.0);
            materialLightScatteringResult.emission = _textureSampleAlbedo(material.textureIndex, recentRayHitResult.uvSurfaceCoordinate, recentRayHitResult.at);
//          materialLightScatteringResult.emission = _textureSampleAlbedo(material.textureIndex, recentRayHitResult.uvSurfaceCoordinate, recentRayHitResult.at);
            materialLightScatteringResult.isScattered = false;
//          materialLightScatteringResult.isScattered = false;
        }
        case MATERIAL_TYPE_PRINCIPLED:
//      case MATERIAL_TYPE_PRINCIPLED:
        {
            let albedo: vec3<f32> = _textureSampleAlbedo(material.textureIndex, recentRayHitResult.uvSurfaceCoordinate, recentRayHitResult.at);
//          let albedo: vec3<f32> = _textureSampleAlbedo(material.textureIndex, recentRayHitResult.uvSurfaceCoordinate, recentRayHitResult.at);

            materialLightScatteringResult.emission = vec3<f32>(0.0, 0.0, 0.0);
//          materialLightScatteringResult.emission = vec3<f32>(0.0, 0.0, 0.0);

            // Principled Logic
//          // Principled Logic

            // F0 calculation: 0.04 for dielectrics, albedo for metals
//          // F0 calculation: 0.04 for dielectrics, albedo for metals
            let f0: vec3<f32> = mix(vec3<f32>(0.04), albedo, material.layer1Metallic);
//          let f0: vec3<f32> = mix(vec3<f32>(0.04), albedo, material.layer1Metallic);

            // Schlick's fresnel approximation at incident angle
//          // Schlick's fresnel approximation at incident angle
            let cosTheta: f32 = min(dot(-incomingRay.direction, recentRayHitResult.hittedSideNormal), 1.0);
//          let cosTheta: f32 = min(dot(-incomingRay.direction, recentRayHitResult.hittedSideNormal), 1.0);
            let fresnel: vec3<f32> = _schlickFresnel(cosTheta, f0);
//          let fresnel: vec3<f32> = _schlickFresnel(cosTheta, f0);
            // Use average fresnel for importance sampling probability
//          // Use average fresnel for importance sampling probability
            let fresnelProbability: f32 = (fresnel.r + fresnel.g + fresnel.b) / 3.0;
//          let fresnelProbability: f32 = (fresnel.r + fresnel.g + fresnel.b) / 3.0;

            let randomChoice: f32 = _pcg32Next(rng);
//          let randomChoice: f32 = _pcg32Next(rng);

            // --- PATH A: METALLIC REFLECTION ---
//          // --- PATH A: METALLIC REFLECTION ---
            // Metals always reflect. They have no diffuse and no transmission.
//          // Metals always reflect. They have no diffuse and no transmission.
            // We also use fresnelProbability to decide between Specular Reflection and Diffuse/Transmission for Dielectrics.
//          // We also use fresnelProbability to decide between Specular Reflection and Diffuse/Transmission for Dielectrics.
            // [ Probability to Reflect = Metallic + (1.0 - Metallic) * Fresnel ]
//          // [ Probability to Reflect = Metallic + (1.0 - Metallic) * Fresnel ]

            let specularProbability: f32 = mix(fresnelProbability, 1.0, material.layer1Metallic);
//          let specularProbability: f32 = mix(fresnelProbability, 1.0, material.layer1Metallic);

            if (randomChoice < specularProbability)
//          if (randomChoice < specularProbability)
            {
                // SPECULAR REFLECTION (METAL OR DIELECTRIC COAT)
//              // SPECULAR REFLECTION (METAL OR DIELECTRIC COAT)
                let microfacetNormal: vec3<f32> = _sampleGGX(recentRayHitResult.hittedSideNormal, material.layer1Roughness, rng);
//              let microfacetNormal: vec3<f32> = _sampleGGX(recentRayHitResult.hittedSideNormal, material.layer1Roughness, rng);
                let specularReflectedDirection: vec3<f32> = _reflectPrincipled(incomingRay.direction, microfacetNormal);
//              let specularReflectedDirection: vec3<f32> = _reflectPrincipled(incomingRay.direction, microfacetNormal);

                if (dot(specularReflectedDirection, recentRayHitResult.hittedSideNormal) > 0.0)
//              if (dot(specularReflectedDirection, recentRayHitResult.hittedSideNormal) > 0.0)
                {
                    materialLightScatteringResult.scatteredRay.origin = recentRayHitResult.at;
//                  materialLightScatteringResult.scatteredRay.origin = recentRayHitResult.at;
                    materialLightScatteringResult.scatteredRay.direction = specularReflectedDirection;
//                  materialLightScatteringResult.scatteredRay.direction = specularReflectedDirection;
                    materialLightScatteringResult.scatteredRay.directionInverse = 1.0 / materialLightScatteringResult.scatteredRay.direction;
//                  materialLightScatteringResult.scatteredRay.directionInverse = 1.0 / materialLightScatteringResult.scatteredRay.direction;
                    materialLightScatteringResult.attenuation = mix(vec3<f32>(1.0), albedo, material.layer1Metallic);
//                  materialLightScatteringResult.attenuation = mix(vec3<f32>(1.0), albedo, material.layer1Metallic);
                    materialLightScatteringResult.isScattered = true;
//                  materialLightScatteringResult.isScattered = true;
                }
                else
//              else
                {
                    // Current/Recent ray is/was absorbed (next ray is scattering into surface)
//                  // Current/Recent ray is/was absorbed (next ray is scattering into surface)
                    materialLightScatteringResult.isScattered = false;
//                  materialLightScatteringResult.isScattered = false;
                }
            }
            else
//          else
            {
                // --- PATH B: DIELECTRIC (DIFFUSE OR TRANSMISSION) ---
//              // --- PATH B: DIELECTRIC (DIFFUSE OR TRANSMISSION) ---

                // Re-normalize random variable for the next choice
//              // Re-normalize random variable for the next choice
                let randomNextChoice: f32 = (randomChoice - specularProbability) / (1.0 - specularProbability);
//              let randomNextChoice: f32 = (randomChoice - specularProbability) / (1.0 - specularProbability);

                if (material.layer1Transmission > 0.0 && randomNextChoice < material.layer1Transmission)
//              if (material.layer1Transmission > 0.0 && randomNextChoice < material.layer1Transmission)
                {
                    // TRANSMISSION (REFRACTION)
//                  // TRANSMISSION (REFRACTION)

                    var etaRatioOfIncidenceOverTransmission: f32 = select(material.layer1IOR / material.layer0IOR, material.layer0IOR / material.layer1IOR, recentRayHitResult.isFrontFaceHitted);
//                  var etaRatioOfIncidenceOverTransmission: f32 = select(material.layer1IOR / material.layer0IOR, material.layer0IOR / material.layer1IOR, recentRayHitResult.isFrontFaceHitted);

                    let microfacetNormal: vec3<f32> = _sampleGGX(recentRayHitResult.hittedSideNormal, material.layer1Roughness, rng);
//                  let microfacetNormal: vec3<f32> = _sampleGGX(recentRayHitResult.hittedSideNormal, material.layer1Roughness, rng);

                    let cosThetaIncidence: f32 = min(dot(-incomingRay.direction, microfacetNormal), 1.0);
//                  let cosThetaIncidence: f32 = min(dot(-incomingRay.direction, microfacetNormal), 1.0);
                    let sinThetaTransmission: f32 = (1.0 - cosThetaIncidence * cosThetaIncidence) * (etaRatioOfIncidenceOverTransmission * etaRatioOfIncidenceOverTransmission);
//                  let sinThetaTransmission: f32 = (1.0 - cosThetaIncidence * cosThetaIncidence) * (etaRatioOfIncidenceOverTransmission * etaRatioOfIncidenceOverTransmission);

                    let refractedDirection: vec3<f32> = _refractPrincipled(incomingRay.direction, microfacetNormal, etaRatioOfIncidenceOverTransmission, cosThetaIncidence, sinThetaTransmission);
//                  let refractedDirection: vec3<f32> = _refractPrincipled(incomingRay.direction, microfacetNormal, etaRatioOfIncidenceOverTransmission, cosThetaIncidence, sinThetaTransmission);
                    let reflectedDirection: vec3<f32> = _reflectPrincipled(incomingRay.direction, microfacetNormal);
//                  let reflectedDirection: vec3<f32> = _reflectPrincipled(incomingRay.direction, microfacetNormal);

                    // When [ sinThetaTransmission <= 1.0 ] then Refraction happened else Total Internal Reflection happened
//                  // When [ sinThetaTransmission <= 1.0 ] then Refraction happened else Total Internal Reflection happened
                    materialLightScatteringResult.scatteredRay.origin = recentRayHitResult.at;
//                  materialLightScatteringResult.scatteredRay.origin = recentRayHitResult.at;
                    materialLightScatteringResult.scatteredRay.direction = select(reflectedDirection, refractedDirection, sinThetaTransmission <= 1.0);
//                  materialLightScatteringResult.scatteredRay.direction = select(reflectedDirection, refractedDirection, sinThetaTransmission <= 1.0);
                    materialLightScatteringResult.scatteredRay.directionInverse = 1.0 / materialLightScatteringResult.scatteredRay.direction;
//                  materialLightScatteringResult.scatteredRay.directionInverse = 1.0 / materialLightScatteringResult.scatteredRay.direction;
                    materialLightScatteringResult.attenuation = select(vec3<f32>(1.0), albedo, sinThetaTransmission <= 1.0);
//                  materialLightScatteringResult.attenuation = select(vec3<f32>(1.0), albedo, sinThetaTransmission <= 1.0);
                    materialLightScatteringResult.isScattered = true;
//                  materialLightScatteringResult.isScattered = true;
                }
                else
//              else
                {
                    // DIFFUSE (LAMBERTIAN)
//                  // DIFFUSE (LAMBERTIAN)

                    let diffuseDirection: vec3<f32> = normalize(recentRayHitResult.hittedSideNormal + _generateRandomUnitVector(rng));
//                  let diffuseDirection: vec3<f32> = normalize(recentRayHitResult.hittedSideNormal + _generateRandomUnitVector(rng));

                    materialLightScatteringResult.scatteredRay.origin = recentRayHitResult.at;
//                  materialLightScatteringResult.scatteredRay.origin = recentRayHitResult.at;
                    materialLightScatteringResult.scatteredRay.direction = diffuseDirection;
//                  materialLightScatteringResult.scatteredRay.direction = diffuseDirection;
                    materialLightScatteringResult.scatteredRay.directionInverse = 1.0 / materialLightScatteringResult.scatteredRay.direction;
//                  materialLightScatteringResult.scatteredRay.directionInverse = 1.0 / materialLightScatteringResult.scatteredRay.direction;
                    materialLightScatteringResult.attenuation = albedo;
//                  materialLightScatteringResult.attenuation = albedo;
                    materialLightScatteringResult.isScattered = true;
//                  materialLightScatteringResult.isScattered = true;
                }
            }
        }
        default:
//      default:
        {

        }
    }
    return materialLightScatteringResult;
//  return materialLightScatteringResult;
}

    fn _rayMarch(ray: Ray, distance: f32) -> vec3<f32>
//  fn _rayMarch(ray: Ray, distance: f32) -> vec3<f32>
{
    return ray.origin + ray.direction * distance;
//  return ray.origin + ray.direction * distance;
}

    fn _rayTrace(ray: Ray) -> vec4<f32>
//  fn _rayTrace(ray: Ray) -> vec4<f32>
{
    let unitDirection: vec3<f32> = normalize(ray.direction);
//  let unitDirection: vec3<f32> = normalize(ray.direction);
    let ratio: f32 = 0.5 * (unitDirection.y + 1.0);
//  let ratio: f32 = 0.5 * (unitDirection.y + 1.0);
    return (1.0 - ratio) * vec4<f32>(1.0, 0.5, 0.0, 1.0) + ratio * vec4<f32>(0.0, 0.5, 1.0, 1.0);
//  return (1.0 - ratio) * vec4<f32>(1.0, 0.5, 0.0, 1.0) + ratio * vec4<f32>(0.0, 0.5, 1.0, 1.0);
}

    fn _rayTraceMain(initialRay: Ray, maxDepth: u32, backgroundType: u32, rng: ptr<function, RNG>) -> TracingResult
//  fn _rayTraceMain(initialRay: Ray, maxDepth: u32, backgroundType: u32, rng: ptr<function, RNG>) -> TracingResult
{
        var tracingResult: TracingResult;
//      var tracingResult: TracingResult;
        var accumulatedColor: vec3<f32> = vec3<f32>(0.00, 0.00, 0.00); // @radiance@
//      var accumulatedColor: vec3<f32> = vec3<f32>(0.00, 0.00, 0.00); // @radiance@
        var attenuation     : vec3<f32> = vec3<f32>(1.00, 1.00, 1.00); // throughput
//      var attenuation     : vec3<f32> = vec3<f32>(1.00, 1.00, 1.00); // throughput
        var currentRay: Ray = initialRay;
//      var currentRay: Ray = initialRay;
        var pixelNormal: vec3<f32>;
//      var pixelNormal: vec3<f32>;
/* --- FOG --- */
//         var firstMinDistance: f32 = 1.0e+4;
// //      var firstMinDistance: f32 = 1.0e+4;
/* --- FOG --- */


        for (var depth: u32 = 0u; depth < maxDepth; depth++)
//      for (var depth: u32 = 0u; depth < maxDepth; depth++)
        {
            let rayHitResultClosestHit: RayHitResult = _rayHitLBVH_ClosestHit(currentRay, Interval(1.0e-4, 1.0e+4));
//          let rayHitResultClosestHit: RayHitResult = _rayHitLBVH_ClosestHit(currentRay, Interval(1.0e-4, 1.0e+4));
            let rayHitResult: RayHitResult = _finalizeFromRayHitResultClosestHit(currentRay, rayHitResultClosestHit);
//          let rayHitResult: RayHitResult = _finalizeFromRayHitResultClosestHit(currentRay, rayHitResultClosestHit);


            pixelNormal = select(pixelNormal, select(vec3<f32>(0.0, 1.0, 0.0), rayHitResult.hittedSideNormal, rayHitResult.isHitted), depth == 0u);
//          pixelNormal = select(pixelNormal, select(vec3<f32>(0.0, 1.0, 0.0), rayHitResult.hittedSideNormal, rayHitResult.isHitted), depth == 0u);
/* --- FOG --- */
//             firstMinDistance = select(firstMinDistance, select(firstMinDistance, rayHitResult.minDistance, rayHitResult.isHitted), depth == 0u);
// //          firstMinDistance = select(firstMinDistance, select(firstMinDistance, rayHitResult.minDistance, rayHitResult.isHitted), depth == 0u);
/* --- FOG --- */


            if (!rayHitResult.isHitted)
//          if (!rayHitResult.isHitted)
            {
                var backgroundColor: vec3<f32>;
//              var backgroundColor: vec3<f32>;
                switch (backgroundType)
//              switch (backgroundType)
                {
                    case BACKGROUND_TYPE_SKY_BOX_BLUE:
//                  case BACKGROUND_TYPE_SKY_BOX_BLUE:
                    {
                        let normalizedRayDirection: vec3<f32> = normalize(currentRay.direction);
//                      let normalizedRayDirection: vec3<f32> = normalize(currentRay.direction);
                        let ratio: f32 = 0.5 * (normalizedRayDirection.y + 1.0);
//                      let ratio: f32 = 0.5 * (normalizedRayDirection.y + 1.0);
                        backgroundColor = mix(vec3<f32>(1.00, 1.00, 1.00), vec3<f32>(0.50, 0.70, 1.00), ratio);
//                      backgroundColor = mix(vec3<f32>(1.00, 1.00, 1.00), vec3<f32>(0.50, 0.70, 1.00), ratio);
                    }


                    case BACKGROUND_TYPE_SKY_BOX_DARK:
//                  case BACKGROUND_TYPE_SKY_BOX_DARK:
                    {
                        backgroundColor = vec3<f32>(0.00, 0.00, 0.00);
//                      backgroundColor = vec3<f32>(0.00, 0.00, 0.00);
                    }


                    case BACKGROUND_TYPE_SKY_BOX_HDRI:
//                  case BACKGROUND_TYPE_SKY_BOX_HDRI:
                    {
                        let theta: f32 = acos (-currentRay.direction.y); // latitude
//                      let theta: f32 = acos (-currentRay.direction.y); // latitude
                        let phi  : f32 = atan2(-currentRay.direction.z, currentRay.direction.x) + PI; // longitude
//                      let phi  : f32 = atan2(-currentRay.direction.z, currentRay.direction.x) + PI; // longitude

                        var u: f32 = phi   / (2.0 * PI);
//                      var u: f32 = phi   / (2.0 * PI);
                        var v: f32 = theta /        PI ;
//                      var v: f32 = theta /        PI ;

                        u = clamp(u, 0.0, 1.0);
//                      u = clamp(u, 0.0, 1.0);
                        v = clamp(v, 0.0, 1.0);
//                      v = clamp(v, 0.0, 1.0);


                        // Tonemapping Solution
                        // Tonemapping Solution
                        /*
                        backgroundColor = _tonemapACES(textureSampleLevel(hdriTexture, hdriSampler, vec2<f32>(u, v), 0.0).rgb);
//                      backgroundColor = _tonemapACES(textureSampleLevel(hdriTexture, hdriSampler, vec2<f32>(u, v), 0.0).rgb);
                        */


                        // Radiance Clamping Solution
                        // Radiance Clamping Solution

                        let rawBackgroundColor: vec3<f32> = textureSampleLevel(hdriTexture, hdriSampler, vec2<f32>(u, v), 0.0).rgb;
//                      let rawBackgroundColor: vec3<f32> = textureSampleLevel(hdriTexture, hdriSampler, vec2<f32>(u, v), 0.0).rgb;
                        let radianceClampValue: f32 = 10.0; // Tune this value to control fireflies
//                      let radianceClampValue: f32 = 10.0; // Tune this value to control fireflies
                        backgroundColor = min(rawBackgroundColor, vec3<f32>(radianceClampValue));
//                      backgroundColor = min(rawBackgroundColor, vec3<f32>(radianceClampValue));



                        // No Solution
                        // No Solution
                        /*
                        backgroundColor = textureSampleLevel(hdriTexture, hdriSampler, vec2<f32>(u, v), 0.0).rgb;
//                      backgroundColor = textureSampleLevel(hdriTexture, hdriSampler, vec2<f32>(u, v), 0.0).rgb;
                        */
                    }


                    default:
//                  default:
                    {
                        backgroundColor = vec3<f32>(1.00, 1.00, 1.00);
//                      backgroundColor = vec3<f32>(1.00, 1.00, 1.00);
                    }
                }

                accumulatedColor += attenuation * backgroundColor;
//              accumulatedColor += attenuation * backgroundColor;
                break;
//              break;
            }


            let materialLightScatteringResult: MaterialLightScatteringResult = _rayScatter(currentRay, rayHitResult, rng);
//          let materialLightScatteringResult: MaterialLightScatteringResult = _rayScatter(currentRay, rayHitResult, rng);

            accumulatedColor += attenuation * materialLightScatteringResult.emission;
//          accumulatedColor += attenuation * materialLightScatteringResult.emission;


            // --- Multiple Importance Sampling: Next Event Estimation ---
            // --- Multiple Importance Sampling: Next Event Estimation ---
            // Explicitly sample the sun for surfaces to add its direct light contribution.
            // Explicitly sample the sun for surfaces to add its direct light contribution.

            // Hard Shadow (Point Light) Solution
            // Hard Shadow (Point Light) Solution
            /*
            let sunDirection: vec3<f32> = vec3<f32>(0.0, 0.707, 0.5);
//          let sunDirection: vec3<f32> = vec3<f32>(0.0, 0.707, 0.5);
            let sunColor: vec3<f32> = vec3<f32>(2.0, 2.0, 2.0);
//          let sunColor: vec3<f32> = vec3<f32>(2.0, 2.0, 2.0);

            let shadowRay: Ray = _makeRay(rayHitResult.at, sunDirection);
//          let shadowRay: Ray = _makeRay(rayHitResult.at, sunDirection);

            // Hard Shadow (AnyHit Optimization)
//          // Hard Shadow (AnyHit Optimization)
            if (!_rayHitLBVH_AnyHit(shadowRay, Interval(1.0e-4, 1.0e+4)))
//          if (!_rayHitLBVH_AnyHit(shadowRay, Interval(1.0e-4, 1.0e+4)))
            {
                // The path to the sun is clear. Add direct light.
                // The path to the sun is clear. Add direct light.
                let cosTheta: f32 = max(0.0, dot(rayHitResult.hittedSideNormal, sunDirection));
//              let cosTheta: f32 = max(0.0, dot(rayHitResult.hittedSideNormal, sunDirection));

                // Attenuation in scatter result is the albedo/color.
                // Attenuation in scatter result is the albedo/color.
                let albedo: vec3<f32> = materialLightScatteringResult.attenuation;
//              let albedo: vec3<f32> = materialLightScatteringResult.attenuation;

                let directLight: vec3<f32> = albedo * sunColor * cosTheta / PI;
//              let directLight: vec3<f32> = albedo * sunColor * cosTheta / PI;

                accumulatedColor += attenuation * directLight;
//              accumulatedColor += attenuation * directLight;
            }
            */


            // Soft Shadow (Area Light) Solution
            // Soft Shadow (Area Light) Solution

            let sunDirectionCenter: vec3<f32> = vec3<f32>(0.0, 0.707, 0.707);
//          let sunDirectionCenter: vec3<f32> = vec3<f32>(0.0, 0.707, 0.707);
            let sunColor: vec3<f32> = vec3<f32>(2.0, 2.0, 2.0);
//          let sunColor: vec3<f32> = vec3<f32>(2.0, 2.0, 2.0);
            let sunRadius: f32 = 0.025; // Increase for softer shadows. Decrease for harder shadows.
//          let sunRadius: f32 = 0.025; // Increase for softer shadows. Decrease for harder shadows.

            // Jitter the sun direction to simulate an area light and get soft shadows.
            // Jitter the sun direction to simulate an area light and get soft shadows.
            let jitteredSunDirection: vec3<f32> = normalize(sunDirectionCenter + _generateRandomUnitVector(rng) * sunRadius);
//          let jitteredSunDirection: vec3<f32> = normalize(sunDirectionCenter + _generateRandomUnitVector(rng) * sunRadius);

            let shadowRay: Ray = _makeRay(rayHitResult.at, jitteredSunDirection);
//          let shadowRay: Ray = _makeRay(rayHitResult.at, jitteredSunDirection);

            // Soft Shadow (AnyHit Optimization)
//          // Soft Shadow (AnyHit Optimization)
            if (!_rayHitLBVH_AnyHit(shadowRay, Interval(1.0e-4, 1.0e+4)))
//          if (!_rayHitLBVH_AnyHit(shadowRay, Interval(1.0e-4, 1.0e+4)))
            {
                // The path to the sun is clear. Add direct light.
                // The path to the sun is clear. Add direct light.
                let cosTheta: f32 = max(0.0, dot(rayHitResult.hittedSideNormal, jitteredSunDirection));
//              let cosTheta: f32 = max(0.0, dot(rayHitResult.hittedSideNormal, jitteredSunDirection));

                // Attenuation in scatter result is the albedo/color.
                // Attenuation in scatter result is the albedo/color.
                let albedo: vec3<f32> = materialLightScatteringResult.attenuation;
//              let albedo: vec3<f32> = materialLightScatteringResult.attenuation;

                let directLight: vec3<f32> = albedo * sunColor * cosTheta / PI;
//              let directLight: vec3<f32> = albedo * sunColor * cosTheta / PI;

                accumulatedColor += attenuation * directLight;
//              accumulatedColor += attenuation * directLight;
            }




            if (!materialLightScatteringResult.isScattered)
//          if (!materialLightScatteringResult.isScattered)
            {
                break;
//              break;
            }

            attenuation *= materialLightScatteringResult.attenuation;
//          attenuation *= materialLightScatteringResult.attenuation;


            // Russian Roulette Solution
            // Russian Roulette Solution

            let minBouncesForRR: u32 = 4u;
//          let minBouncesForRR: u32 = 4u;
            if (depth >= minBouncesForRR)
//          if (depth >= minBouncesForRR)
            {
                let p: f32 = max(attenuation.r, max(attenuation.g, attenuation.b));
//              let p: f32 = max(attenuation.r, max(attenuation.g, attenuation.b));
                if (_pcg32Next(rng) > p)
//              if (_pcg32Next(rng) > p)
                {
                    break; // Terminate the ray
//                  break; // Terminate the ray
                }
                attenuation *= 1.0 / p; // Boost the survivor
//              attenuation *= 1.0 / p; // Boost the survivor
            }



            currentRay   = materialLightScatteringResult.scatteredRay;
//          currentRay   = materialLightScatteringResult.scatteredRay;
        }

/* --- FOG --- */
//         accumulatedColor = _applyFogHeight(accumulatedColor, firstMinDistance, initialRay, FOG_HEIGHT_COLOR, FOG_HEIGHT_DENSITY, FOG_HEIGHT_FALLOFF);
// //      accumulatedColor = _applyFogHeight(accumulatedColor, firstMinDistance, initialRay, FOG_HEIGHT_COLOR, FOG_HEIGHT_DENSITY, FOG_HEIGHT_FALLOFF);
/* --- FOG --- */

        tracingResult.pixelOutput = vec4<f32>(accumulatedColor, 1.0);
//      tracingResult.pixelOutput = vec4<f32>(accumulatedColor, 1.0);
        tracingResult.pixelNormal = vec4<f32>((pixelNormal + 1.0) * 0.5, 1.0);
//      tracingResult.pixelNormal = vec4<f32>((pixelNormal + 1.0) * 0.5, 1.0);

        return tracingResult;
//      return tracingResult;
}

    struct GeneralData
//  struct GeneralData
{
    canvasSize: vec2<u32>,
//  canvasSize: vec2<u32>,
    stratifiedSamplesPerPixel: f32,
//  stratifiedSamplesPerPixel: f32,
    inverseStratifiedSamplesPerPixel: f32,
//  inverseStratifiedSamplesPerPixel: f32,
    cameraCenter: vec3<f32>,
//  cameraCenter: vec3<f32>,
    pixelSamplesScale: f32,
//  pixelSamplesScale: f32,
    fromPixelToPixelDeltaU: vec3<f32>,
//  fromPixelToPixelDeltaU: vec3<f32>,
    stratifiedSampleX: f32,
//  stratifiedSampleX: f32,
    fromPixelToPixelDeltaV: vec3<f32>,
//  fromPixelToPixelDeltaV: vec3<f32>,
    stratifiedSampleY: f32,
//  stratifiedSampleY: f32,
    pixel00Coordinates: vec3<f32>,
//  pixel00Coordinates: vec3<f32>,
    backgroundType: u32,
//  backgroundType: u32,
    numberOfImages: u32,
//  numberOfImages: u32,
    timeInSeconds: f32,
//  timeInSeconds: f32,
    accumulatedSampleCount: u32,
//  accumulatedSampleCount: u32,
    defocusDiskRadiusU: vec3<f32>,
//  defocusDiskRadiusU: vec3<f32>,
    defocusDiskRadiusV: vec3<f32>,
//  defocusDiskRadiusV: vec3<f32>,
}

    struct TracingResult
//  struct TracingResult
{
    pixelOutput: vec4<f32>,
//  pixelOutput: vec4<f32>,
    pixelNormal: vec4<f32>,
//  pixelNormal: vec4<f32>,
}

    @group(0) @binding(0) var<uniform> generalData: GeneralData;
//  @group(0) @binding(0) var<uniform> generalData: GeneralData;
    @group(0) @binding(1) var<storage, read_write> outputStorage: array<vec4<f32>>;
//  @group(0) @binding(1) var<storage, read_write> outputStorage: array<vec4<f32>>;
    @group(0) @binding(2) var<storage, read_write> accumulatedOutputStorage: array<vec4<f32>>;
//  @group(0) @binding(2) var<storage, read_write> accumulatedOutputStorage: array<vec4<f32>>;
    @group(0) @binding(3) var<storage, read> spheres: array<Sphere>;
//  @group(0) @binding(3) var<storage, read> spheres: array<Sphere>;
    @group(0) @binding(4) var<storage, read> materials: array<Material>;
//  @group(0) @binding(4) var<storage, read> materials: array<Material>;
    @group(0) @binding(5) var<storage, read> textures: array<Texture>;
//  @group(0) @binding(5) var<storage, read> textures: array<Texture>;
    @group(0) @binding(6) var columnAtlasSampler: sampler;
//  @group(0) @binding(6) var columnAtlasSampler: sampler;
    @group(0) @binding(7) var columnAtlasTexture: texture_2d<f32>;
//  @group(0) @binding(7) var columnAtlasTexture: texture_2d<f32>;
    @group(0) @binding(8) var hdriSampler: sampler;
//  @group(0) @binding(8) var hdriSampler: sampler;
    @group(0) @binding(9) var hdriTexture: texture_2d<f32>;
//  @group(0) @binding(9) var hdriTexture: texture_2d<f32>;
    @group(0) @binding(10) var<storage, read> triangles: array<Triangle>;
//  @group(0) @binding(10) var<storage, read> triangles: array<Triangle>;
//     const cubes: array<Cube, 1> = array<Cube, 1>(
// //  const cubes: array<Cube, 1> = array<Cube, 1>(
//         Cube(vec3<f32>(0.0, -10.0, 0.0), vec3<f32>(5.0, 5.0, 5.0)),
// //      Cube(vec3<f32>(0.0, -10.0, 0.0), vec3<f32>(5.0, 5.0, 5.0)),
//     );
// //  );
    @group(0) @binding(11) var<storage, read> bvhNodes: array<BVHNode>;
//  @group(0) @binding(11) var<storage, read> bvhNodes: array<BVHNode>;

    @compute @workgroup_size(32, 32) fn main(@builtin(global_invocation_id) gid: vec3<u32>)
//  @compute @workgroup_size(32, 32) fn main(@builtin(global_invocation_id) gid: vec3<u32>)
{

    if (gid.x >= generalData.canvasSize.x || gid.y >= generalData.canvasSize.y)
//  if (gid.x >= generalData.canvasSize.x || gid.y >= generalData.canvasSize.y)
    {
        return;
//      return;
    }

    let frameIndex: f32 = generalData.stratifiedSampleY * generalData.stratifiedSamplesPerPixel + generalData.stratifiedSampleX;
//  let frameIndex: f32 = generalData.stratifiedSampleY * generalData.stratifiedSamplesPerPixel + generalData.stratifiedSampleX;
    let frameIndexForSeed: u32 = u32(frameIndex * 10000.0);
//  let frameIndexForSeed: u32 = u32(frameIndex * 10000.0);
    var rng: RNG = _rngInit(gid.x, gid.y, generalData.canvasSize.x, frameIndexForSeed);
//  var rng: RNG = _rngInit(gid.x, gid.y, generalData.canvasSize.x, frameIndexForSeed);
    _ = _pcg32Next(&rng);
//  _ = _pcg32Next(&rng);

    var uv: vec2<i32> = vec2<i32>(gid.xy);
//  var uv: vec2<i32> = vec2<i32>(gid.xy);
    var jitter: vec2<i32> = _hash2ToVec2I32(_hash2(generalData.timeInSeconds), 1, 10) * 2 - 1;
//  var jitter: vec2<i32> = _hash2ToVec2I32(_hash2(generalData.timeInSeconds), 1, 10) * 2 - 1;
    jitter = jitter / vec2<i32>(generalData.canvasSize.xy);
//  jitter = jitter / vec2<i32>(generalData.canvasSize.xy);
    var uvJitter: vec2<i32> = uv + jitter;
//  var uvJitter: vec2<i32> = uv + jitter;
    uvJitter = clamp(uvJitter, vec2<i32>(0), vec2<i32>(generalData.canvasSize.xy));
//  uvJitter = clamp(uvJitter, vec2<i32>(0), vec2<i32>(generalData.canvasSize.xy));

    let vogelDiskSampleIndex: f32 = frameIndex;
//  let vogelDiskSampleIndex: f32 = frameIndex;
    let vogelDiskSampleCount: f32 = generalData.stratifiedSamplesPerPixel * generalData.stratifiedSamplesPerPixel;
//  let vogelDiskSampleCount: f32 = generalData.stratifiedSamplesPerPixel * generalData.stratifiedSamplesPerPixel;

    let ray: Ray = _generatePrimaryRay(
//  let ray: Ray = _generatePrimaryRay(
        generalData.stratifiedSampleX,
//      generalData.stratifiedSampleX,
        generalData.stratifiedSampleY,
//      generalData.stratifiedSampleY,
        generalData.inverseStratifiedSamplesPerPixel,
//      generalData.inverseStratifiedSamplesPerPixel,
        generalData.pixel00Coordinates,
//      generalData.pixel00Coordinates,
        generalData.fromPixelToPixelDeltaU,
//      generalData.fromPixelToPixelDeltaU,
        generalData.fromPixelToPixelDeltaV,
//      generalData.fromPixelToPixelDeltaV,
        f32(uvJitter.x),
//      f32(uvJitter.x),
        f32(uvJitter.y),
//      f32(uvJitter.y),
        generalData.cameraCenter,
//      generalData.cameraCenter,
        generalData.defocusDiskRadiusU,
//      generalData.defocusDiskRadiusU,
        generalData.defocusDiskRadiusV,
//      generalData.defocusDiskRadiusV,
        vogelDiskSampleIndex,
//      vogelDiskSampleIndex,
        vogelDiskSampleCount,
//      vogelDiskSampleCount,
        &rng,
//      &rng,
    );
//  );

    let tracingResult: TracingResult = _rayTraceMain(ray, 4, generalData.backgroundType, &rng);
//  let tracingResult: TracingResult = _rayTraceMain(ray, 4, generalData.backgroundType, &rng);

    var imageJit: vec4<f32> = tracingResult.pixelOutput;
//  var imageJit: vec4<f32> = tracingResult.pixelOutput;

    let pixelIndex: u32 = gid.y * generalData.canvasSize.x + gid.x;
//  let pixelIndex: u32 = gid.y * generalData.canvasSize.x + gid.x;

    var imageAcc: vec4<f32> = accumulatedOutputStorage[pixelIndex];
//  var imageAcc: vec4<f32> = accumulatedOutputStorage[pixelIndex];

    //  Exponential Moving Average (EMA) -> Cumulative Moving Average (CMA)
//  //  Exponential Moving Average (EMA) -> Cumulative Moving Average (CMA)
    //  If it's the first frame (count = 0 ), mix factor is 1.0 (replace old data).
//  //  If it's the first frame (count = 0 ), mix factor is 1.0 (replace old data).
    //  If it's the 100th frame (count = 99), mix factor is 1/100.
//  //  If it's the 100th frame (count = 99), mix factor is 1/100.
    /*
    let weight: f32 = 0.01;
//  let weight: f32 = 0.01;
    */
    /*
    let weight: f32 = 1.0 / (f32(generalData.accumulatedSampleCount) + 1.0);
//  let weight: f32 = 1.0 / (f32(generalData.accumulatedSampleCount) + 1.0);
    */
    let weight: f32 = select(1.0 / (f32(generalData.accumulatedSampleCount) + 1.0), 1.0, generalData.accumulatedSampleCount == 0);
//  let weight: f32 = select(1.0 / (f32(generalData.accumulatedSampleCount) + 1.0), 1.0, generalData.accumulatedSampleCount == 0);
    imageAcc = mix(imageAcc, imageJit, weight);
//  imageAcc = mix(imageAcc, imageJit, weight);

    outputStorage[pixelIndex] = imageAcc;
//  outputStorage[pixelIndex] = imageAcc;

    accumulatedOutputStorage[pixelIndex] = imageAcc;
//  accumulatedOutputStorage[pixelIndex] = imageAcc;
}

    fn _generatePrimaryRay(
//  fn _generatePrimaryRay(
        stratifiedSampleX: f32,
//      stratifiedSampleX: f32,
        stratifiedSampleY: f32,
//      stratifiedSampleY: f32,
        inverseStratifiedSamplesPerPixel: f32,
//      inverseStratifiedSamplesPerPixel: f32,
        pixel00Coordinates: vec3<f32>,
//      pixel00Coordinates: vec3<f32>,
        fromPixelToPixelDeltaU: vec3<f32>,
//      fromPixelToPixelDeltaU: vec3<f32>,
        fromPixelToPixelDeltaV: vec3<f32>,
//      fromPixelToPixelDeltaV: vec3<f32>,
        pixelX: f32,
//      pixelX: f32,
        pixelY: f32,
//      pixelY: f32,
        cameraCenter: vec3<f32>,
//      cameraCenter: vec3<f32>,
        defocusDiskRadiusU: vec3<f32>,
//      defocusDiskRadiusU: vec3<f32>,
        defocusDiskRadiusV: vec3<f32>,
//      defocusDiskRadiusV: vec3<f32>,
        vogelDiskSampleIndex: f32,
//      vogelDiskSampleIndex: f32,
        vogelDiskSampleCount: f32,
//      vogelDiskSampleCount: f32,
        rng: ptr<function, RNG>,
//      rng: ptr<function, RNG>,
    ) -> Ray
//  ) -> Ray
    {
        let sampleOffset: vec2<f32> = _getVogelDiskSample(vogelDiskSampleIndex, vogelDiskSampleCount, 0.0 /* 1.6180 */) * 1.5;
//      let sampleOffset: vec2<f32> = _getVogelDiskSample(vogelDiskSampleIndex, vogelDiskSampleCount, 0.0 /* 1.6180 */) * 1.5;
//         let sampleOffset: vec3<f32> = vec3<f32>(((stratifiedSampleX + _pcg32Next(rng)) * inverseStratifiedSamplesPerPixel) - 0.5, ((stratifiedSampleY + _pcg32Next(rng)) * inverseStratifiedSamplesPerPixel) - 0.5, 0.0);
// //      let sampleOffset: vec3<f32> = vec3<f32>(((stratifiedSampleX + _pcg32Next(rng)) * inverseStratifiedSamplesPerPixel) - 0.5, ((stratifiedSampleY + _pcg32Next(rng)) * inverseStratifiedSamplesPerPixel) - 0.5, 0.0);
        let pixelSampleCenter: vec3<f32> = pixel00Coordinates + fromPixelToPixelDeltaU * (pixelX + sampleOffset.x) + fromPixelToPixelDeltaV * (pixelY + sampleOffset.y);
//      let pixelSampleCenter: vec3<f32> = pixel00Coordinates + fromPixelToPixelDeltaU * (pixelX + sampleOffset.x) + fromPixelToPixelDeltaV * (pixelY + sampleOffset.y);
        let rayOrigin: vec3<f32> = _getDefocusDiskSample(cameraCenter, defocusDiskRadiusU, defocusDiskRadiusV, rng);
//      let rayOrigin: vec3<f32> = _getDefocusDiskSample(cameraCenter, defocusDiskRadiusU, defocusDiskRadiusV, rng);
//         let rayOrigin: vec3<f32> = cameraCenter;
// //      let rayOrigin: vec3<f32> = cameraCenter;
        let rayDirection: vec3<f32> = pixelSampleCenter - rayOrigin;
//      let rayDirection: vec3<f32> = pixelSampleCenter - rayOrigin;
        let ray: Ray = _makeRay(rayOrigin, normalize(rayDirection));
//      let ray: Ray = _makeRay(rayOrigin, normalize(rayDirection));
        return ray;
//      return ray;
    }

    struct RNG
//  struct RNG
{
    state: u32,
//  state: u32,
}

    fn _pcg32Next(rng: ptr<function, RNG>) -> f32
//  fn _pcg32Next(rng: ptr<function, RNG>) -> f32
{
    // Advance LCG state (PCG parameters)
    // Advance LCG state (PCG parameters)
    let old: u32 = (*rng).state;
//  let old: u32 = (*rng).state;
    (*rng).state = old * 747796405u + 2891336453u;
//  (*rng).state = old * 747796405u + 2891336453u;

    // Output permutation (PCG XSH RR)
    // Output permutation (PCG XSH RR)
    var x: u32 = ((old >> ((old >> 28u) + 4u)) ^ old) * 277803737u;
//  var x: u32 = ((old >> ((old >> 28u) + 4u)) ^ old) * 277803737u;
    x = (x >> 22u) ^ x;
//  x = (x >> 22u) ^ x;

    // Convert to float in [0,1)
    // Convert to float in [0,1)
    return f32(x) / 4294967296.0; // divide by 2^32
//  return f32(x) / 4294967296.0; // divide by 2^32
}

    fn _pcg32NextRangeU32(rng: ptr<function, RNG>, min: u32, max: u32) -> u32
//  fn _pcg32NextRangeU32(rng: ptr<function, RNG>, min: u32, max: u32) -> u32
{
    return min + u32(_pcg32Next(rng) * f32(max - min));
//  return min + u32(_pcg32Next(rng) * f32(max - min));
}

    fn _pcg32NextRangeF32(rng: ptr<function, RNG>, min: f32, max: f32) -> f32
//  fn _pcg32NextRangeF32(rng: ptr<function, RNG>, min: f32, max: f32) -> f32
{
    return min +    (_pcg32Next(rng) *    (max - min));
//  return min +    (_pcg32Next(rng) *    (max - min));
}

// A helper "constructor" to build a highly randomized RNG state from specific inputs.
// A helper "constructor" to build a highly randomized RNG state from specific inputs.
    fn _rngInit(pixelX: u32, pixelY: u32, canvasWidth: u32, frameIndex: u32) -> RNG
//  fn _rngInit(pixelX: u32, pixelY: u32, canvasWidth: u32, frameIndex: u32) -> RNG
{
    // 1. Create a unique linear index for this pixel.
    // 1. Create a unique linear index for this pixel.
    let pixelId: u32 = pixelY * canvasWidth + pixelX;
//  let pixelId: u32 = pixelY * canvasWidth + pixelX;

    // 2. We need to mix the pixelId and the frameIndex together securely.
    // 2. We need to mix the pixelId and the frameIndex together securely.
    // A simple addition (pixelId + frameIndex) creates diagonal artifacts.
    // A simple addition (pixelId + frameIndex) creates diagonal artifacts.
    // We use a hashing step (borrowing PCG's own math) to scramble the bits.
    // We use a hashing step (borrowing PCG's own math) to scramble the bits.

    // Initialize state with pixel ID and scramble
    // Initialize state with pixel ID and scramble
    var state: u32 = pixelId;
//  var state: u32 = pixelId;
    state = state * 747796405u + 2891336453u;
//  state = state * 747796405u + 2891336453u;
    state = (state >> 22u) ^ state;
//  state = (state >> 22u) ^ state;

    // Mix in the frame index (using XOR to avoid carrying patterns)
    // Mix in the frame index (using XOR to avoid carrying patterns)
    state = state ^ frameIndex;
//  state = state ^ frameIndex;

    // Scramble again to ensure the frame bits are thoroughly mixed
    // Scramble again to ensure the frame bits are thoroughly mixed
    state = state * 747796405u + 2891336453u;
//  state = state * 747796405u + 2891336453u;
    state = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
//  state = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    state = (state >> 22u) ^ state;
//  state = (state >> 22u) ^ state;

    return RNG(state);
//  return RNG(state);
}

    fn _schlickFresnel(cosine: f32, f0: vec3<f32>) -> vec3<f32>
//  fn _schlickFresnel(cosine: f32, f0: vec3<f32>) -> vec3<f32>
    {
        let oneMinusCos: f32 = 1.0 - cosine;
//      let oneMinusCos: f32 = 1.0 - cosine;
        let oneMinusCos2: f32 = oneMinusCos * oneMinusCos;
//      let oneMinusCos2: f32 = oneMinusCos * oneMinusCos;
        let oneMinusCos4: f32 = oneMinusCos2 * oneMinusCos2;
//      let oneMinusCos4: f32 = oneMinusCos2 * oneMinusCos2;
        let oneMinusCos5: f32 = oneMinusCos4 * oneMinusCos;
//      let oneMinusCos5: f32 = oneMinusCos4 * oneMinusCos;
        return f0 + (1.0 - f0) * oneMinusCos5;
//      return f0 + (1.0 - f0) * oneMinusCos5;
    }

    // Generate a microfacet normal (H) based on roughness (GGX/Trowbridge-Reitz)
//  // Generate a microfacet normal (H) based on roughness (GGX/Trowbridge-Reitz)
    fn _sampleGGX(normal: vec3<f32>, roughness: f32, rng: ptr<function, RNG>) -> vec3<f32>
//  fn _sampleGGX(normal: vec3<f32>, roughness: f32, rng: ptr<function, RNG>) -> vec3<f32>
    {
        let random1: f32 = _pcg32Next(rng);
//      let random1: f32 = _pcg32Next(rng);
        let random2: f32 = _pcg32Next(rng);
//      let random2: f32 = _pcg32Next(rng);

        let term: f32 = roughness * roughness;
//      let term: f32 = roughness * roughness;
        let phi: f32 = 2.0 * PI * random1;
//      let phi: f32 = 2.0 * PI * random1;
        let cosTheta: f32 = sqrt((1.0 - random2) / (1.0 + (term * term - 1.0) * random2));
//      let cosTheta: f32 = sqrt((1.0 - random2) / (1.0 + (term * term - 1.0) * random2));
        let sinTheta: f32 = sqrt(1.0 - cosTheta * cosTheta);
//      let sinTheta: f32 = sqrt(1.0 - cosTheta * cosTheta);

        let x: f32 = sinTheta * cos(phi);
//      let x: f32 = sinTheta * cos(phi);
        let y: f32 = sinTheta * sin(phi);
//      let y: f32 = sinTheta * sin(phi);
        let z: f32 = cosTheta;
//      let z: f32 = cosTheta;

        // Tangent space basis construction (orthonormal basis)
//      // Tangent space basis construction (orthonormal basis)
        let up: vec3<f32> = select(vec3<f32>(1.0, 0.0, 0.0), vec3<f32>(0.0, 0.0, 1.0), abs(normal.z) < 0.999);
//      let up: vec3<f32> = select(vec3<f32>(1.0, 0.0, 0.0), vec3<f32>(0.0, 0.0, 1.0), abs(normal.z) < 0.999);
        let tangent: vec3<f32> = normalize(cross(up, normal));
//      let tangent: vec3<f32> = normalize(cross(up, normal));
        let bitangent: vec3<f32> = cross(normal, tangent);
//      let bitangent: vec3<f32> = cross(normal, tangent);

        // Transform H to world space
//      // Transform H to world space
        return normalize(tangent * x + bitangent * y + normal * z);
//      return normalize(tangent * x + bitangent * y + normal * z);
    }

    fn _reflectPrincipled(incomingVector: vec3<f32>, normal: vec3<f32>) -> vec3<f32> { return incomingVector - 2.0 * dot(incomingVector, normal) * normal; }
//  fn _reflectPrincipled(incomingVector: vec3<f32>, normal: vec3<f32>) -> vec3<f32> { return incomingVector - 2.0 * dot(incomingVector, normal) * normal; }

    fn _refractPrincipled(incomingVector: vec3<f32>, normal: vec3<f32>, etaRatioOfIncidenceOverTransmission: f32, cosThetaIncidence: f32, sinThetaTransmission: f32) -> vec3<f32>
//  fn _refractPrincipled(incomingVector: vec3<f32>, normal: vec3<f32>, etaRatioOfIncidenceOverTransmission: f32, cosThetaIncidence: f32, sinThetaTransmission: f32) -> vec3<f32>
    {
        let cosThetaTransmission: f32 = sqrt(1.0 - sinThetaTransmission);
//      let cosThetaTransmission: f32 = sqrt(1.0 - sinThetaTransmission);
        let refractedDirection: vec3<f32> = normalize(etaRatioOfIncidenceOverTransmission * incomingVector + (etaRatioOfIncidenceOverTransmission * cosThetaIncidence - cosThetaTransmission) * normal);
//      let refractedDirection: vec3<f32> = normalize(etaRatioOfIncidenceOverTransmission * incomingVector + (etaRatioOfIncidenceOverTransmission * cosThetaIncidence - cosThetaTransmission) * normal);
        return refractedDirection;
//      return refractedDirection;
    }

    /*
    fn _reflect(incomingVector: vec3<f32>, normal: vec3<f32>) -> vec3<f32> { return incomingVector - 2.0 * dot(incomingVector, normal) * normal; }
//  fn _reflect(incomingVector: vec3<f32>, normal: vec3<f32>) -> vec3<f32> { return incomingVector - 2.0 * dot(incomingVector, normal) * normal; }

    fn _refract(incomingVector: vec3<f32>, normal: vec3<f32>, etaRatioOfIncidenceOverTransmission: f32) -> vec3<f32>
//  fn _refract(incomingVector: vec3<f32>, normal: vec3<f32>, etaRatioOfIncidenceOverTransmission: f32) -> vec3<f32>
    {
        let cosTheta: f32 = min(dot(-incomingVector, normal), 1.0);
//      let cosTheta: f32 = min(dot(-incomingVector, normal), 1.0);
        let refractedRayPerpendicular: vec3<f32> = etaRatioOfIncidenceOverTransmission * (incomingVector + cosTheta * normal);
//      let refractedRayPerpendicular: vec3<f32> = etaRatioOfIncidenceOverTransmission * (incomingVector + cosTheta * normal);
        let refractedRayParallel: vec3<f32> = -sqrt(abs(1.0 - _lengthSquared(refractedRayPerpendicular))) * normal;
//      let refractedRayParallel: vec3<f32> = -sqrt(abs(1.0 - _lengthSquared(refractedRayPerpendicular))) * normal;
        return refractedRayPerpendicular + refractedRayParallel;
//      return refractedRayPerpendicular + refractedRayParallel;
    }
    */

    fn _reflectance(cosine: f32, etaRatioOfIncidenceOverTransmission: f32) -> f32
//  fn _reflectance(cosine: f32, etaRatioOfIncidenceOverTransmission: f32) -> f32
    {
        // Use Schlick's approximation for reflectance.
        // Use Schlick's approximation for reflectance.
        var r0: f32 = (1.0 - etaRatioOfIncidenceOverTransmission) / (1.0 + etaRatioOfIncidenceOverTransmission);
//      var r0: f32 = (1.0 - etaRatioOfIncidenceOverTransmission) / (1.0 + etaRatioOfIncidenceOverTransmission);
        r0 = r0 * r0;
//      r0 = r0 * r0;
        let temp: f32 = 1.0 - cosine;
//      let temp: f32 = 1.0 - cosine;
        let temp2: f32 = temp * temp;
//      let temp2: f32 = temp * temp;
        let temp4: f32 = temp2 * temp2;
//      let temp4: f32 = temp2 * temp2;
        let temp5: f32 = temp4 * temp;
//      let temp5: f32 = temp4 * temp;
        return r0 + (1.0 - r0) * temp5;
//      return r0 + (1.0 - r0) * temp5;
    }

    fn _generateRandomUnitVector(rng: ptr<function, RNG>) -> vec3<f32>
//  fn _generateRandomUnitVector(rng: ptr<function, RNG>) -> vec3<f32>
    {
//  Archimedes' Method
//  Archimedes' Method

        let theta: f32 = _pcg32NextRangeF32(rng,  0.0,  2.0 * PI); // Longitude          (uniform)
//      let theta: f32 = _pcg32NextRangeF32(rng,  0.0,  2.0 * PI); // Longitude          (uniform)
        let z    : f32 = _pcg32NextRangeF32(rng, -1.0,  1.0     ); // Height on cylinder (uniform)
//      let z    : f32 = _pcg32NextRangeF32(rng, -1.0,  1.0     ); // Height on cylinder (uniform)

        let r: f32 = sqrt(1.0 - z * z); // Radius of the circle at this height
//      let r: f32 = sqrt(1.0 - z * z); // Radius of the circle at this height

        let x: f32 = r * cos(theta);
//      let x: f32 = r * cos(theta);
        let y: f32 = r * sin(theta);
//      let y: f32 = r * sin(theta);

        return vec3<f32>(x, y, z);
//      return vec3<f32>(x, y, z);

    }

    fn _vec4LinearToGamma(value: vec4<f32>) -> vec4<f32> { return sqrt(value); };
//  fn _vec4LinearToGamma(value: vec4<f32>) -> vec4<f32> { return sqrt(value); };
    fn _vec4GammaToLinear(value: vec4<f32>) -> vec4<f32> { return value * value; };
//  fn _vec4GammaToLinear(value: vec4<f32>) -> vec4<f32> { return value * value; };
    fn _vec3LinearToGamma(value: vec3<f32>) -> vec3<f32> { return sqrt(value); };
//  fn _vec3LinearToGamma(value: vec3<f32>) -> vec3<f32> { return sqrt(value); };
    fn _vec3GammaToLinear(value: vec3<f32>) -> vec3<f32> { return value * value; };
//  fn _vec3GammaToLinear(value: vec3<f32>) -> vec3<f32> { return value * value; };
    fn _vec2LinearToGamma(value: vec2<f32>) -> vec2<f32> { return sqrt(value); };
//  fn _vec2LinearToGamma(value: vec2<f32>) -> vec2<f32> { return sqrt(value); };
    fn _vec2GammaToLinear(value: vec2<f32>) -> vec2<f32> { return value * value; };
//  fn _vec2GammaToLinear(value: vec2<f32>) -> vec2<f32> { return value * value; };
    fn _f32LinearToGamma(value: f32) -> f32 { return sqrt(value); };
//  fn _f32LinearToGamma(value: f32) -> f32 { return sqrt(value); };
    fn _f32GammaToLinear(value: f32) -> f32 { return value * value; };
//  fn _f32GammaToLinear(value: f32) -> f32 { return value * value; };

    fn _textureSampleAlbedo(textureIndex: u32, uvSurfaceCoordinate: vec2<f32>, surfaceCoordinate: vec3<f32>) -> vec3<f32>
//  fn _textureSampleAlbedo(textureIndex: u32, uvSurfaceCoordinate: vec2<f32>, surfaceCoordinate: vec3<f32>) -> vec3<f32>
{
    var textureSampleValue: vec3<f32>;
//  var textureSampleValue: vec3<f32>;
    let texture: Texture = textures[textureIndex];
//  let texture: Texture = textures[textureIndex];
    switch (texture.albedoTextureType)
//  switch (texture.albedoTextureType)
    {
        case TEXTURE_TYPE_COLOR:
//      case TEXTURE_TYPE_COLOR:
        {
            textureSampleValue = texture.albedo;
//          textureSampleValue = texture.albedo;
        }
        case TEXTURE_TYPE_IMAGE:
//      case TEXTURE_TYPE_IMAGE:
        {
            var uvTextureCoordinate: vec2<f32> = clamp(uvSurfaceCoordinate, vec2<f32>(0.0), vec2<f32>(1.0));
//          var uvTextureCoordinate: vec2<f32> = clamp(uvSurfaceCoordinate, vec2<f32>(0.0), vec2<f32>(1.0));
            uvTextureCoordinate.y = 1.0 - uvTextureCoordinate.y;
//          uvTextureCoordinate.y = 1.0 - uvTextureCoordinate.y;
            textureSampleValue = _sampleColumnAtlas(uvTextureCoordinate, texture.albedoImageIndex);
//          textureSampleValue = _sampleColumnAtlas(uvTextureCoordinate, texture.albedoImageIndex);
        }
        case TEXTURE_TYPE_CHECKER_STYLE_A:
//      case TEXTURE_TYPE_CHECKER_STYLE_A:
        {
            let oTileTexture: Texture = textures[texture.albedoOTileTextureIndex];
//          let oTileTexture: Texture = textures[texture.albedoOTileTextureIndex];
            let eTileTexture: Texture = textures[texture.albedoETileTextureIndex];
//          let eTileTexture: Texture = textures[texture.albedoETileTextureIndex];
            let textureCoordinate: vec3<i32> = vec3<i32>(floor(texture.albedoScale * surfaceCoordinate));
//          let textureCoordinate: vec3<i32> = vec3<i32>(floor(texture.albedoScale * surfaceCoordinate));
            textureSampleValue = select(oTileTexture.albedo, eTileTexture.albedo, ((textureCoordinate.x + textureCoordinate.y + textureCoordinate.z) & 1) == 0);
//          textureSampleValue = select(oTileTexture.albedo, eTileTexture.albedo, ((textureCoordinate.x + textureCoordinate.y + textureCoordinate.z) & 1) == 0);
        }
        case TEXTURE_TYPE_CHECKER_STYLE_B:
//      case TEXTURE_TYPE_CHECKER_STYLE_B:
        {
            let oTileTexture: Texture = textures[texture.albedoOTileTextureIndex];
//          let oTileTexture: Texture = textures[texture.albedoOTileTextureIndex];
            let eTileTexture: Texture = textures[texture.albedoETileTextureIndex];
//          let eTileTexture: Texture = textures[texture.albedoETileTextureIndex];
            let uvTextureCoordinate: vec2<i32> = vec2<i32>(floor(texture.albedoScale * uvSurfaceCoordinate));
//          let uvTextureCoordinate: vec2<i32> = vec2<i32>(floor(texture.albedoScale * uvSurfaceCoordinate));
            textureSampleValue = select(oTileTexture.albedo, eTileTexture.albedo, ((uvTextureCoordinate.x + uvTextureCoordinate.y) & 1) == 0);
//          textureSampleValue = select(oTileTexture.albedo, eTileTexture.albedo, ((uvTextureCoordinate.x + uvTextureCoordinate.y) & 1) == 0);
        }
        case default:
//      case default:
        {
            textureSampleValue = vec3<f32>(0.0, 0.0, 0.0);
//          textureSampleValue = vec3<f32>(0.0, 0.0, 0.0);
        }
    }
    return textureSampleValue;
//  return textureSampleValue;
}

    fn _sampleColumnAtlas(uvTextureCoordinate: vec2<f32>, imageIndex: u32) -> vec3<f32>
//  fn _sampleColumnAtlas(uvTextureCoordinate: vec2<f32>, imageIndex: u32) -> vec3<f32>
{
    let columnAtlasUVTextureCoordinate: vec2<f32> = uvTextureCoordinate * vec2<f32>(1.0, 1.0 / f32(generalData.numberOfImages)) + vec2<f32>(0.0, f32(imageIndex) * (1.0 / f32(generalData.numberOfImages)));
//  let columnAtlasUVTextureCoordinate: vec2<f32> = uvTextureCoordinate * vec2<f32>(1.0, 1.0 / f32(generalData.numberOfImages)) + vec2<f32>(0.0, f32(imageIndex) * (1.0 / f32(generalData.numberOfImages)));
    return textureSampleLevel(columnAtlasTexture, columnAtlasSampler, columnAtlasUVTextureCoordinate, 0.0).rgb;
//  return textureSampleLevel(columnAtlasTexture, columnAtlasSampler, columnAtlasUVTextureCoordinate, 0.0).rgb;
}

    /*
    fn _f32Saturate(value: f32) -> f32 { return clamp(value, 0.0, 1.0); }
//  fn _f32Saturate(value: f32) -> f32 { return clamp(value, 0.0, 1.0); }
    fn _vec2Saturate(value: vec2<f32>) -> vec2<f32> { return clamp(value, vec2<f32>(0.0), vec2<f32>(1.0)); }
//  fn _vec2Saturate(value: vec2<f32>) -> vec2<f32> { return clamp(value, vec2<f32>(0.0), vec2<f32>(1.0)); }
    fn _vec3Saturate(value: vec3<f32>) -> vec3<f32> { return clamp(value, vec3<f32>(0.0), vec3<f32>(1.0)); }
//  fn _vec3Saturate(value: vec3<f32>) -> vec3<f32> { return clamp(value, vec3<f32>(0.0), vec3<f32>(1.0)); }
    fn _vec4Saturate(value: vec4<f32>) -> vec4<f32> { return clamp(value, vec4<f32>(0.0), vec4<f32>(1.0)); }
//  fn _vec4Saturate(value: vec4<f32>) -> vec4<f32> { return clamp(value, vec4<f32>(0.0), vec4<f32>(1.0)); }
    */

    fn _tonemapACES(value: vec3<f32>) -> vec3<f32>
//  fn _tonemapACES(value: vec3<f32>) -> vec3<f32>
    {
        return saturate((value * (2.51 * value + vec3<f32>(0.03, 0.03, 0.03))) / (value * (2.43 * value + vec3<f32>(0.59, 0.59, 0.59)) + vec3<f32>(0.14, 0.14, 0.14)));
//      return saturate((value * (2.51 * value + vec3<f32>(0.03, 0.03, 0.03))) / (value * (2.43 * value + vec3<f32>(0.59, 0.59, 0.59)) + vec3<f32>(0.14, 0.14, 0.14)));
    }
    fn _tonemapFilmic(value: vec3<f32>) -> vec3<f32>
//  fn _tonemapFilmic(value: vec3<f32>) -> vec3<f32>
    {
        let m: vec3<f32> = vec3<f32>(max(0.0, value.x - 0.004), max(0.0, value.y - 0.004), max(0.0, value.z - 0.004));
//      let m: vec3<f32> = vec3<f32>(max(0.0, value.x - 0.004), max(0.0, value.y - 0.004), max(0.0, value.z - 0.004));
        return (m * (6.2 * m + vec3<f32>(0.5, 0.5, 0.5))) / (m * (6.2 * m + vec3<f32>(1.7, 1.7, 1.7)) + vec3<f32>(0.06, 0.06, 0.06));
//      return (m * (6.2 * m + vec3<f32>(0.5, 0.5, 0.5))) / (m * (6.2 * m + vec3<f32>(1.7, 1.7, 1.7)) + vec3<f32>(0.06, 0.06, 0.06));
    }
    fn _tonemapReinhard(value: vec3<f32>) -> vec3<f32>
//  fn _tonemapReinhard(value: vec3<f32>) -> vec3<f32>
    {
        return value / (1.0 + dot(value, vec3<f32>(0.21250175, 0.71537574, 0.07212251)));
//      return value / (1.0 + dot(value, vec3<f32>(0.21250175, 0.71537574, 0.07212251)));
    }
    fn _tonemapReinhardJodie(value: vec3<f32>) -> vec3<f32>
//  fn _tonemapReinhardJodie(value: vec3<f32>) -> vec3<f32>
    {
        let l: f32 = dot(value, vec3<f32>(0.21250175, 0.71537574, 0.07212251)); let tc: vec3<f32> = value / (value + vec3<f32>(1.0, 1.0, 1.0)); return mix(value / (l + 1.0), tc, tc);
//      let l: f32 = dot(value, vec3<f32>(0.21250175, 0.71537574, 0.07212251)); let tc: vec3<f32> = value / (value + vec3<f32>(1.0, 1.0, 1.0)); return mix(value / (l + 1.0), tc, tc);
    }
    fn _tonemapUncharted2(value: vec3<f32>) -> vec3<f32>
//  fn _tonemapUncharted2(value: vec3<f32>) -> vec3<f32>
    {
        const A: f32 = 0.15;
//      const A: f32 = 0.15;
        const B: f32 = 0.50;
//      const B: f32 = 0.50;
        const C: f32 = 0.10;
//      const C: f32 = 0.10;
        const D: f32 = 0.20;
//      const D: f32 = 0.20;
        const E: f32 = 0.02;
//      const E: f32 = 0.02;
        const F: f32 = 0.30;
//      const F: f32 = 0.30;
        const VCB: vec3<f32> = vec3<f32>(C * B, C * B, C * B);
//      const VCB: vec3<f32> = vec3<f32>(C * B, C * B, C * B);
        const VB : vec3<f32> = vec3<f32>(    B,     B,     B);
//      const VB : vec3<f32> = vec3<f32>(    B,     B,     B);
        const VDE: vec3<f32> = vec3<f32>(D * E, D * E, D * E);
//      const VDE: vec3<f32> = vec3<f32>(D * E, D * E, D * E);
        const VDF: vec3<f32> = vec3<f32>(D * F, D * F, D * F);
//      const VDF: vec3<f32> = vec3<f32>(D * F, D * F, D * F);
        const VEF: vec3<f32> = vec3<f32>(E / F, E / F, E / F);
//      const VEF: vec3<f32> = vec3<f32>(E / F, E / F, E / F);
        return ((value * (A * value + VCB) + VDE) / (value * (A * value + VB) + VDF)) - VEF;
//      return ((value * (A * value + VCB) + VDE) / (value * (A * value + VB) + VDF)) - VEF;
    }
    fn _tonemapUncharted1(value: vec3<f32>) -> vec3<f32>
//  fn _tonemapUncharted1(value: vec3<f32>) -> vec3<f32>
    {
        const W: f32 = 11.2;
//      const W: f32 = 11.2;
        const exposureBias: f32 = 2.0;
//      const exposureBias: f32 = 2.0;
        let curr: vec3<f32> = _tonemapUncharted2(exposureBias * value);
//      let curr: vec3<f32> = _tonemapUncharted2(exposureBias * value);
        let whiteScale: vec3<f32> = 1.0 / _tonemapUncharted2(vec3<f32>(W, W, W));
//      let whiteScale: vec3<f32> = 1.0 / _tonemapUncharted2(vec3<f32>(W, W, W));
        return curr * whiteScale;
//      return curr * whiteScale;
    }
    fn _tonemapUnreal(value: vec3<f32>) -> vec3<f32>
//  fn _tonemapUnreal(value: vec3<f32>) -> vec3<f32>
    {
        return value / (value + vec3<f32>(0.155, 0.155, 0.155)) * 1.019;
//      return value / (value + vec3<f32>(0.155, 0.155, 0.155)) * 1.019;
    }
    fn _tonemapLinear(value: vec3<f32>) -> vec3<f32>
//  fn _tonemapLinear(value: vec3<f32>) -> vec3<f32>
    {
        return value;
//      return value;
    }
    fn _tonemapDrago(value: vec3<f32>) -> vec3<f32>
//  fn _tonemapDrago(value: vec3<f32>) -> vec3<f32>
    {
        let lum: f32 = dot(value, vec3<f32>(0.2126, 0.7152, 0.0722));
//      let lum: f32 = dot(value, vec3<f32>(0.2126, 0.7152, 0.0722));
        let logLum: f32 = log2(1.0 + lum);
//      let logLum: f32 = log2(1.0 + lum);
        let bias: f32 = log2(0.85);
//      let bias: f32 = log2(0.85);
        let exponent: f32 = logLum / bias;
//      let exponent: f32 = logLum / bias;
        let scale: f32 = pow(0.96, exponent);
//      let scale: f32 = pow(0.96, exponent);
        return value * scale;
//      return value * scale;
    }
    fn _tonemapExponential(value: vec3<f32>) -> vec3<f32>
//  fn _tonemapExponential(value: vec3<f32>) -> vec3<f32>
    {
        const exposure: f32 = 1.0;
//      const exposure: f32 = 1.0;
        return 1.0 - exp(-value * exposure);
//      return 1.0 - exp(-value * exposure);
    }
    fn _tonemapLogarithmic(value: vec3<f32>) -> vec3<f32>
//  fn _tonemapLogarithmic(value: vec3<f32>) -> vec3<f32>
    {
        return value / (value + 1.0);
//      return value / (value + 1.0);
    }
    fn _tonemapExtendedReinhard(value: vec3<f32>) -> vec3<f32>
//  fn _tonemapExtendedReinhard(value: vec3<f32>) -> vec3<f32>
    {
        const whitePoint: f32 = 4.0; // This can be adjusted, or calculated from the scene's max brightness.
//      const whitePoint: f32 = 4.0; // This can be adjusted, or calculated from the scene's max brightness.
        let numerator: vec3<f32> = value * (1.0 + (value / (whitePoint * whitePoint)));
//      let numerator: vec3<f32> = value * (1.0 + (value / (whitePoint * whitePoint)));
        return numerator / (1.0 + value);
//      return numerator / (1.0 + value);
    }
    fn _tonemapHableFilmic(value: vec3<f32>) -> vec3<f32>
//  fn _tonemapHableFilmic(value: vec3<f32>) -> vec3<f32>
    {
        const A: f32 = 0.22; // Shoulder Strength
//      const A: f32 = 0.22; // Shoulder Strength
        const B: f32 = 0.30; // Linear   Strength
//      const B: f32 = 0.30; // Linear   Strength
        const C: f32 = 0.10; // Linear   Angle
//      const C: f32 = 0.10; // Linear   Angle
        const D: f32 = 0.20; // Toe      Strength
//      const D: f32 = 0.20; // Toe      Strength
        const E: f32 = 0.01; // Toe        Numerator
//      const E: f32 = 0.01; // Toe        Numerator
        const F: f32 = 0.30; // Toe      Denominator
//      const F: f32 = 0.30; // Toe      Denominator
        const exposure: f32 = 2.0;
//      const exposure: f32 = 2.0;

        let curr: vec3<f32> = ((value * (A * value + C * B) + D * E) / (value * (A * value + B) + D * F)) - E / F;
//      let curr: vec3<f32> = ((value * (A * value + C * B) + D * E) / (value * (A * value + B) + D * F)) - E / F;
        const whitePoint: vec3<f32> = ((vec3<f32>(11.2, 11.2, 11.2) * (A * vec3<f32>(11.2, 11.2, 11.2) + C * B) + D * E) / (vec3<f32>(11.2, 11.2, 11.2) * (A * vec3<f32>(11.2, 11.2, 11.2) + B) + D * F)) - E / F;
//      const whitePoint: vec3<f32> = ((vec3<f32>(11.2, 11.2, 11.2) * (A * vec3<f32>(11.2, 11.2, 11.2) + C * B) + D * E) / (vec3<f32>(11.2, 11.2, 11.2) * (A * vec3<f32>(11.2, 11.2, 11.2) + B) + D * F)) - E / F;

        return curr * (1.0 / whitePoint);
//      return curr * (1.0 / whitePoint);
    }
    fn _tonemapACESFitted(value: vec3<f32>) -> vec3<f32>
//  fn _tonemapACESFitted(value: vec3<f32>) -> vec3<f32>
    {
        // Narkowicz 2015, "ACES Filmic Tone Mapping Curve"
        // Narkowicz 2015, "ACES Filmic Tone Mapping Curve"
        const a: f32 = 2.51;
//      const a: f32 = 2.51;
        const b: f32 = 0.03;
//      const b: f32 = 0.03;
        const c: f32 = 2.43;
//      const c: f32 = 2.43;
        const d: f32 = 0.59;
//      const d: f32 = 0.59;
        const e: f32 = 0.14;
//      const e: f32 = 0.14;

        let clamped: vec3<f32> = saturate((value * (a * value + b)) / (value * (c * value + d) + e));
//      let clamped: vec3<f32> = saturate((value * (a * value + b)) / (value * (c * value + d) + e));
        return clamped;
//      return clamped;
    }
    fn _tonemapLottes(value: vec3<f32>) -> vec3<f32>
//  fn _tonemapLottes(value: vec3<f32>) -> vec3<f32>
    {
        // A simpler, more robust curve inspired by Lottes' GDC talks
        // A simpler, more robust curve inspired by Lottes' GDC talks
        const contrast: f32 = 1.60;
//      const contrast: f32 = 1.60;
        const shoulder: f32 = 0.98;
//      const shoulder: f32 = 0.98;

        // Apply contrast
        // Apply contrast
        let contrasted: vec3<f32> = vec3<f32>
//      let contrasted: vec3<f32> = vec3<f32>
        (
            pow(value.x, contrast),
            pow(value.y, contrast),
            pow(value.z, contrast),
        );

        // Apply shoulder curve to compress highlights
        // Apply shoulder curve to compress highlights
        let tonemapped: vec3<f32> = vec3<f32>
//      let tonemapped: vec3<f32> = vec3<f32>
        (
            (1.0 - shoulder) * contrasted.x + shoulder * contrasted.x / (1.0 + contrasted.x),
            (1.0 - shoulder) * contrasted.y + shoulder * contrasted.y / (1.0 + contrasted.y),
            (1.0 - shoulder) * contrasted.z + shoulder * contrasted.z / (1.0 + contrasted.z),
        );

        return tonemapped;
//      return tonemapped;
    }
    fn _tonemapHejlFilmic(value: vec3<f32>) -> vec3<f32>
//  fn _tonemapHejlFilmic(value: vec3<f32>) -> vec3<f32>
    {
        let temp: vec3<f32> = vec3<f32>
//      let temp: vec3<f32> = vec3<f32>
        (
            max(0.0, value.x - 0.004),
            max(0.0, value.y - 0.004),
            max(0.0, value.z - 0.004),
        );

        let   numerator: vec3<f32> = (temp * (6.2 * temp + 0.5));
//      let   numerator: vec3<f32> = (temp * (6.2 * temp + 0.5));
        let denominator: vec3<f32> = (temp * (6.2 * temp + 1.7)) + 0.06;
//      let denominator: vec3<f32> = (temp * (6.2 * temp + 1.7)) + 0.06;

        return vec3<f32>
//      return vec3<f32>
        (
            numerator.x / denominator.x,
            numerator.y / denominator.y,
            numerator.z / denominator.z,
        );
    }
    fn _tonemapNeutral(value: vec3<f32>) -> vec3<f32>
//  fn _tonemapNeutral(value: vec3<f32>) -> vec3<f32>
    {
        const A: f32 = 0.15;
//      const A: f32 = 0.15;
        const B: f32 = 0.50;
//      const B: f32 = 0.50;
        const C: f32 = 0.10;
//      const C: f32 = 0.10;
        const D: f32 = 0.20;
//      const D: f32 = 0.20;
        const E: f32 = 0.02;
//      const E: f32 = 0.02;
        const F: f32 = 0.30;
//      const F: f32 = 0.30;
        const W: f32 = 11.2;
//      const W: f32 = 11.2;
        const exposure: f32 = 2.0;
//      const exposure: f32 = 2.0;

        let exposedColor: vec3<f32> = exposure * value;
//      let exposedColor: vec3<f32> = exposure * value;

        let VCB: vec3<f32> = vec3<f32>(C * B, C * B, C * B);
//      let VCB: vec3<f32> = vec3<f32>(C * B, C * B, C * B);
        let VB : vec3<f32> = vec3<f32>(    B,     B,     B);
//      let VB : vec3<f32> = vec3<f32>(    B,     B,     B);
        let VDE: vec3<f32> = vec3<f32>(D * E, D * E, D * E);
//      let VDE: vec3<f32> = vec3<f32>(D * E, D * E, D * E);
        let VDF: vec3<f32> = vec3<f32>(D * F, D * F, D * F);
//      let VDF: vec3<f32> = vec3<f32>(D * F, D * F, D * F);
        let VEF: vec3<f32> = vec3<f32>(E / F, E / F, E / F);
//      let VEF: vec3<f32> = vec3<f32>(E / F, E / F, E / F);

        let curr: vec3<f32> = ((exposedColor * (A * exposedColor + VCB) + VDE) / (exposedColor * (A * exposedColor + VB) + VDF)) - VEF;
//      let curr: vec3<f32> = ((exposedColor * (A * exposedColor + VCB) + VDE) / (exposedColor * (A * exposedColor + VB) + VDF)) - VEF;

        let whitePointVec: vec3<f32> = vec3<f32>(W, W, W);
//      let whitePointVec: vec3<f32> = vec3<f32>(W, W, W);
        let whitePoint: vec3<f32> = ((whitePointVec * (A * whitePointVec + VCB) + VDE) / (whitePointVec * (A * whitePointVec + VB) + VDF)) - VEF;
//      let whitePoint: vec3<f32> = ((whitePointVec * (A * whitePointVec + VCB) + VDE) / (whitePointVec * (A * whitePointVec + VB) + VDF)) - VEF;

        return curr / whitePoint;
//      return curr / whitePoint;
    }
    fn _tonemapPerceptual(value: vec3<f32>) -> vec3<f32>
//  fn _tonemapPerceptual(value: vec3<f32>) -> vec3<f32>
    {
        let luminance: f32 = dot(value, vec3<f32>(0.2126, 0.7152, 0.0722));
//      let luminance: f32 = dot(value, vec3<f32>(0.2126, 0.7152, 0.0722));
        let tonemappedLuminance: f32 = luminance / (1.0 + luminance);
//      let tonemappedLuminance: f32 = luminance / (1.0 + luminance);
        let scale: f32 = select(0.0, tonemappedLuminance / luminance, luminance > 0.0);
//      let scale: f32 = select(0.0, tonemappedLuminance / luminance, luminance > 0.0);
        return value * scale;
//      return value * scale;
    }

    struct SDFHitResult
//  struct SDFHitResult
{
    distance: f32,
//  distance: f32,
    geometryIndex: u32,
//  geometryIndex: u32,
    isSphere: bool,
//  isSphere: bool,
    isTriangle: bool,
//  isTriangle: bool,
//     isCube: bool,
// //  isCube: bool,
}

    fn _dot2Vec2(v: vec2<f32>) -> f32 { return dot(v, v); }
//  fn _dot2Vec2(v: vec2<f32>) -> f32 { return dot(v, v); }
    fn _dot2Vec3(v: vec3<f32>) -> f32 { return dot(v, v); }
//  fn _dot2Vec3(v: vec3<f32>) -> f32 { return dot(v, v); }
    fn _sdfOpRound(distance: f32, radius: f32) -> f32 { return distance - radius; }
//  fn _sdfOpRound(distance: f32, radius: f32) -> f32 { return distance - radius; }
    fn _sdfOpOnion(distance: f32, thickness: f32) -> f32 { return abs(distance) - thickness; }
//  fn _sdfOpOnion(distance: f32, thickness: f32) -> f32 { return abs(distance) - thickness; }
    fn _sdfSphere(samplePoint: vec3<f32>, sphereIndex: u32) -> f32
//  fn _sdfSphere(samplePoint: vec3<f32>, sphereIndex: u32) -> f32
    {
        let sphere: Sphere = spheres[sphereIndex];
//      let sphere: Sphere = spheres[sphereIndex];
        return length(samplePoint - sphere.center) - sphere.radius;
//      return length(samplePoint - sphere.center) - sphere.radius;
    }
    fn _sdfSpheres(samplePoint: vec3<f32>) -> SDFHitResult
//  fn _sdfSpheres(samplePoint: vec3<f32>) -> SDFHitResult
    {
        var sdfHitResult: SDFHitResult;
//      var sdfHitResult: SDFHitResult;
        sdfHitResult.distance = 99999999;
//      sdfHitResult.distance = 99999999;
        sdfHitResult.geometryIndex = 0;
//      sdfHitResult.geometryIndex = 0;
        sdfHitResult.isSphere = true;
//      sdfHitResult.isSphere = true;
        sdfHitResult.isTriangle = false;
//      sdfHitResult.isTriangle = false;
//         sdfHitResult.isCube = false;
// //      sdfHitResult.isCube = false;
        let numberOfSpheres: u32 = arrayLength(&spheres);
//      let numberOfSpheres: u32 = arrayLength(&spheres);
        for (var sphereIndex: u32 = 0; sphereIndex < numberOfSpheres; sphereIndex++)
//      for (var sphereIndex: u32 = 0; sphereIndex < numberOfSpheres; sphereIndex++)
        {
            let temporaryDistance: f32 = _sdfSphere(samplePoint, sphereIndex);
//          let temporaryDistance: f32 = _sdfSphere(samplePoint, sphereIndex);
            let condition: bool = temporaryDistance <= sdfHitResult.distance;
//          let condition: bool = temporaryDistance <= sdfHitResult.distance;
            sdfHitResult.distance = select(sdfHitResult.distance, temporaryDistance, condition);
//          sdfHitResult.distance = select(sdfHitResult.distance, temporaryDistance, condition);
            sdfHitResult.geometryIndex = select(sdfHitResult.geometryIndex, sphereIndex, condition);
//          sdfHitResult.geometryIndex = select(sdfHitResult.geometryIndex, sphereIndex, condition);
        }
        return sdfHitResult;
//      return sdfHitResult;
    }
    fn _sdfTriangle(samplePoint: vec3<f32>, triangleIndex: u32) -> f32
//  fn _sdfTriangle(samplePoint: vec3<f32>, triangleIndex: u32) -> f32
    {
        let triangle: Triangle = triangles[triangleIndex];
//      let triangle: Triangle = triangles[triangleIndex];
        let edge01: vec3<f32> = triangle.vertex1 - triangle.vertex0; let vertex0ToSamplePoint: vec3<f32> = samplePoint - triangle.vertex0;
//      let edge01: vec3<f32> = triangle.vertex1 - triangle.vertex0; let vertex0ToSamplePoint: vec3<f32> = samplePoint - triangle.vertex0;
        let edge12: vec3<f32> = triangle.vertex2 - triangle.vertex1; let vertex1ToSamplePoint: vec3<f32> = samplePoint - triangle.vertex1;
//      let edge12: vec3<f32> = triangle.vertex2 - triangle.vertex1; let vertex1ToSamplePoint: vec3<f32> = samplePoint - triangle.vertex1;
        let edge20: vec3<f32> = triangle.vertex0 - triangle.vertex2; let vertex2ToSamplePoint: vec3<f32> = samplePoint - triangle.vertex2;
//      let edge20: vec3<f32> = triangle.vertex0 - triangle.vertex2; let vertex2ToSamplePoint: vec3<f32> = samplePoint - triangle.vertex2;
        let normal: vec3<f32> = cross(edge01, edge20);
//      let normal: vec3<f32> = cross(edge01, edge20);
        let isOutside: bool =
//      let isOutside: bool =
            sign(dot(cross(edge01, normal), vertex0ToSamplePoint)) +
//          sign(dot(cross(edge01, normal), vertex0ToSamplePoint)) +
            sign(dot(cross(edge12, normal), vertex1ToSamplePoint)) +
//          sign(dot(cross(edge12, normal), vertex1ToSamplePoint)) +
            sign(dot(cross(edge20, normal), vertex2ToSamplePoint)) < 2.0;
//          sign(dot(cross(edge20, normal), vertex2ToSamplePoint)) < 2.0;
        var distanceSquared: f32;
//      var distanceSquared: f32;
        if (isOutside)
//      if (isOutside)
        {
            distanceSquared = min(min(
//          distanceSquared = min(min(
                _dot2Vec3(edge01 * clamp(dot(edge01, vertex0ToSamplePoint) / _dot2Vec3(edge01), 0.0, 1.0) - vertex0ToSamplePoint) ,
//              _dot2Vec3(edge01 * clamp(dot(edge01, vertex0ToSamplePoint) / _dot2Vec3(edge01), 0.0, 1.0) - vertex0ToSamplePoint) ,
                _dot2Vec3(edge12 * clamp(dot(edge12, vertex1ToSamplePoint) / _dot2Vec3(edge12), 0.0, 1.0) - vertex1ToSamplePoint)),
//              _dot2Vec3(edge12 * clamp(dot(edge12, vertex1ToSamplePoint) / _dot2Vec3(edge12), 0.0, 1.0) - vertex1ToSamplePoint)),
                _dot2Vec3(edge20 * clamp(dot(edge20, vertex2ToSamplePoint) / _dot2Vec3(edge20), 0.0, 1.0) - vertex2ToSamplePoint));
//              _dot2Vec3(edge20 * clamp(dot(edge20, vertex2ToSamplePoint) / _dot2Vec3(edge20), 0.0, 1.0) - vertex2ToSamplePoint));
        }
        else
//      else
        {
            distanceSquared = dot(normal, vertex0ToSamplePoint) * dot(normal, vertex0ToSamplePoint) / _dot2Vec3(normal);
//          distanceSquared = dot(normal, vertex0ToSamplePoint) * dot(normal, vertex0ToSamplePoint) / _dot2Vec3(normal);
        }
        return sqrt(distanceSquared);
//      return sqrt(distanceSquared);
    }
    fn _sdfTriangles(samplePoint: vec3<f32>) -> SDFHitResult
//  fn _sdfTriangles(samplePoint: vec3<f32>) -> SDFHitResult
    {
        var sdfHitResult: SDFHitResult;
//      var sdfHitResult: SDFHitResult;
        sdfHitResult.distance = 99999999;
//      sdfHitResult.distance = 99999999;
        sdfHitResult.geometryIndex = 0;
//      sdfHitResult.geometryIndex = 0;
        sdfHitResult.isSphere = false;
//      sdfHitResult.isSphere = false;
        sdfHitResult.isTriangle = true;
//      sdfHitResult.isTriangle = true;
//         sdfHitResult.isCube = false;
// //      sdfHitResult.isCube = false;
        let numberOfTriangles: u32 = arrayLength(&triangles);
//      let numberOfTriangles: u32 = arrayLength(&triangles);
        for (var triangleIndex: u32 = 0; triangleIndex < numberOfTriangles; triangleIndex++)
//      for (var triangleIndex: u32 = 0; triangleIndex < numberOfTriangles; triangleIndex++)
        {
            let temporaryDistance: f32 = _sdfTriangle(samplePoint, triangleIndex);
//          let temporaryDistance: f32 = _sdfTriangle(samplePoint, triangleIndex);
            let condition: bool = temporaryDistance <= sdfHitResult.distance;
//          let condition: bool = temporaryDistance <= sdfHitResult.distance;
            sdfHitResult.distance = select(sdfHitResult.distance, temporaryDistance, condition);
//          sdfHitResult.distance = select(sdfHitResult.distance, temporaryDistance, condition);
            sdfHitResult.geometryIndex = select(sdfHitResult.geometryIndex, triangleIndex, condition);
//          sdfHitResult.geometryIndex = select(sdfHitResult.geometryIndex, triangleIndex, condition);
        }
        return sdfHitResult;
//      return sdfHitResult;
    }
//     fn _sdfCube(samplePoint: vec3<f32>, cubeIndex: u32) -> f32
// //  fn _sdfCube(samplePoint: vec3<f32>, cubeIndex: u32) -> f32
//     {
//         let cube: Cube = cubes[cubeIndex];
// //      let cube: Cube = cubes[cubeIndex];
//         let offsetFromCube: vec3<f32> = abs(samplePoint - cube.center) - cube.halfSize;
// //      let offsetFromCube: vec3<f32> = abs(samplePoint - cube.center) - cube.halfSize;
//         let outsideDistance: f32 = length(max(offsetFromCube, vec3<f32>(0.0, 0.0, 0.0)));
// //      let outsideDistance: f32 = length(max(offsetFromCube, vec3<f32>(0.0, 0.0, 0.0)));
//         let insideDistance: f32 = min(max(offsetFromCube.x, max(offsetFromCube.y, offsetFromCube.z)), 0.0);
// //      let insideDistance: f32 = min(max(offsetFromCube.x, max(offsetFromCube.y, offsetFromCube.z)), 0.0);
//         return outsideDistance + insideDistance;
// //      return outsideDistance + insideDistance;
//     }
//     fn _sdfCubes(samplePoint: vec3<f32>) -> SDFHitResult
// //  fn _sdfCubes(samplePoint: vec3<f32>) -> SDFHitResult
//     {
//         var sdfHitResult: SDFHitResult;
// //      var sdfHitResult: SDFHitResult;
//         sdfHitResult.distance = 99999999;
// //      sdfHitResult.distance = 99999999;
//         sdfHitResult.geometryIndex = 0;
// //      sdfHitResult.geometryIndex = 0;
//         sdfHitResult.isSphere = false;
// //      sdfHitResult.isSphere = false;
//         sdfHitResult.isTriangle = false;
// //      sdfHitResult.isTriangle = false;
//         sdfHitResult.isCube = true;
// //      sdfHitResult.isCube = true;
//         let numberOfCubes: u32 = 1; // arrayLength(&cubes);
// //      let numberOfCubes: u32 = 1; // arrayLength(&cubes);
//         for (var cubeIndex: u32 = 0; cubeIndex < numberOfCubes; cubeIndex++)
// //      for (var cubeIndex: u32 = 0; cubeIndex < numberOfCubes; cubeIndex++)
//         {
//             let temporaryDistance: f32 = _sdfCube(samplePoint, cubeIndex);
// //          let temporaryDistance: f32 = _sdfCube(samplePoint, cubeIndex);
//             let condition: bool = temporaryDistance <= sdfHitResult.distance;
// //          let condition: bool = temporaryDistance <= sdfHitResult.distance;
//             sdfHitResult.distance = select(sdfHitResult.distance, temporaryDistance, condition);
// //          sdfHitResult.distance = select(sdfHitResult.distance, temporaryDistance, condition);
//             sdfHitResult.geometryIndex = select(sdfHitResult.geometryIndex, cubeIndex, condition);
// //          sdfHitResult.geometryIndex = select(sdfHitResult.geometryIndex, cubeIndex, condition);
//         }
//         return sdfHitResult;
// //      return sdfHitResult;
//     }
    fn _selectSDFHitResult(falseSDFHitResult: SDFHitResult, trueSDFHitResult: SDFHitResult, condition: bool) -> SDFHitResult
//  fn _selectSDFHitResult(falseSDFHitResult: SDFHitResult, trueSDFHitResult: SDFHitResult, condition: bool) -> SDFHitResult
    {
        var sdfHitResult: SDFHitResult;
//      var sdfHitResult: SDFHitResult;
        sdfHitResult.distance = select(falseSDFHitResult.distance, trueSDFHitResult.distance, condition);
//      sdfHitResult.distance = select(falseSDFHitResult.distance, trueSDFHitResult.distance, condition);
        sdfHitResult.geometryIndex = select(falseSDFHitResult.geometryIndex, trueSDFHitResult.geometryIndex, condition);
//      sdfHitResult.geometryIndex = select(falseSDFHitResult.geometryIndex, trueSDFHitResult.geometryIndex, condition);
        sdfHitResult.isSphere = select(falseSDFHitResult.isSphere, trueSDFHitResult.isSphere, condition);
//      sdfHitResult.isSphere = select(falseSDFHitResult.isSphere, trueSDFHitResult.isSphere, condition);
        sdfHitResult.isTriangle = select(falseSDFHitResult.isTriangle, trueSDFHitResult.isTriangle, condition);
//      sdfHitResult.isTriangle = select(falseSDFHitResult.isTriangle, trueSDFHitResult.isTriangle, condition);
//         sdfHitResult.isCube = select(falseSDFHitResult.isCube, trueSDFHitResult.isCube, condition);
// //      sdfHitResult.isCube = select(falseSDFHitResult.isCube, trueSDFHitResult.isCube, condition);
        return sdfHitResult;
//      return sdfHitResult;
    }
    fn _sdfMain(samplePoint: vec3<f32>) -> SDFHitResult
//  fn _sdfMain(samplePoint: vec3<f32>) -> SDFHitResult
    {
        var sdfHitResultFinal: SDFHitResult;
//      var sdfHitResultFinal: SDFHitResult;
        let sdfHitResultSpheres: SDFHitResult = _sdfSpheres(samplePoint);
//      let sdfHitResultSpheres: SDFHitResult = _sdfSpheres(samplePoint);
        let sdfHitResultTriangles: SDFHitResult = _sdfTriangles(samplePoint);
//      let sdfHitResultTriangles: SDFHitResult = _sdfTriangles(samplePoint);
//         let sdfHitResultCubes: SDFHitResult = _sdfCubes(samplePoint);
// //      let sdfHitResultCubes: SDFHitResult = _sdfCubes(samplePoint);
        sdfHitResultFinal = _selectSDFHitResult(sdfHitResultTriangles, sdfHitResultSpheres, sdfHitResultSpheres.distance <= sdfHitResultTriangles.distance);
//      sdfHitResultFinal = _selectSDFHitResult(sdfHitResultTriangles, sdfHitResultSpheres, sdfHitResultSpheres.distance <= sdfHitResultTriangles.distance);
//         sdfHitResultFinal = _selectSDFHitResult(sdfHitResultFinal, sdfHitResultCubes, sdfHitResultCubes.distance <= sdfHitResultFinal.distance);
// //      sdfHitResultFinal = _selectSDFHitResult(sdfHitResultFinal, sdfHitResultCubes, sdfHitResultCubes.distance <= sdfHitResultFinal.distance);
        return sdfHitResultFinal;
//      return sdfHitResultFinal;
    }

    /*
    const IMPLICIT_NORMAL_EPSILON: f32 = 1.0e-4;
//  const IMPLICIT_NORMAL_EPSILON: f32 = 1.0e-4;

    fn _calculateImplicitNormal(samplePoint: vec3<f32>) -> vec3<f32>
//  fn _calculateImplicitNormal(samplePoint: vec3<f32>) -> vec3<f32>
    {
        return normalize(
//      return normalize(
            vec3<f32>(
//          vec3<f32>(
                _sdfMain(vec3<f32>(samplePoint.x + IMPLICIT_NORMAL_EPSILON, samplePoint.y, samplePoint.z)).distance
//              _sdfMain(vec3<f32>(samplePoint.x + IMPLICIT_NORMAL_EPSILON, samplePoint.y, samplePoint.z)).distance
            -   _sdfMain(vec3<f32>(samplePoint.x - IMPLICIT_NORMAL_EPSILON, samplePoint.y, samplePoint.z)).distance
//          -   _sdfMain(vec3<f32>(samplePoint.x - IMPLICIT_NORMAL_EPSILON, samplePoint.y, samplePoint.z)).distance
            ,
//          ,
                _sdfMain(vec3<f32>(samplePoint.x, samplePoint.y + IMPLICIT_NORMAL_EPSILON, samplePoint.z)).distance
//              _sdfMain(vec3<f32>(samplePoint.x, samplePoint.y + IMPLICIT_NORMAL_EPSILON, samplePoint.z)).distance
            -   _sdfMain(vec3<f32>(samplePoint.x, samplePoint.y - IMPLICIT_NORMAL_EPSILON, samplePoint.z)).distance
//          -   _sdfMain(vec3<f32>(samplePoint.x, samplePoint.y - IMPLICIT_NORMAL_EPSILON, samplePoint.z)).distance
            ,
//          ,
                _sdfMain(vec3<f32>(samplePoint.x, samplePoint.y, samplePoint.z + IMPLICIT_NORMAL_EPSILON)).distance
//              _sdfMain(vec3<f32>(samplePoint.x, samplePoint.y, samplePoint.z + IMPLICIT_NORMAL_EPSILON)).distance
            -   _sdfMain(vec3<f32>(samplePoint.x, samplePoint.y, samplePoint.z - IMPLICIT_NORMAL_EPSILON)).distance
//          -   _sdfMain(vec3<f32>(samplePoint.x, samplePoint.y, samplePoint.z - IMPLICIT_NORMAL_EPSILON)).distance
            )
//          )
        );
//      );
    }

    /*
    Gradient-based local parameterization
//  Gradient-based local parameterization
    n = normalize(grad(sdf(p)))
//  n = normalize(grad(sdf(p)))
    a = (abs(n.x) < 0.5) ? (1,0,0) : (0,1,0)
//  a = (abs(n.x) < 0.5) ? (1,0,0) : (0,1,0)
    t = normalize(cross(a, n))
//  t = normalize(cross(a, n))
    b = cross(n, t)
//  b = cross(n, t)
    u = dot(p, t)
//  u = dot(p, t)
    v = dot(p, b)
//  v = dot(p, b)
    */
    fn _calculateImplicitUVSurfaceCoordinate(samplePoint: vec3<f32>, normal: vec3<f32>, scale: vec2<f32>) -> vec2<f32>
//  fn _calculateImplicitUVSurfaceCoordinate(samplePoint: vec3<f32>, normal: vec3<f32>, scale: vec2<f32>) -> vec2<f32>
    {
        let anchor: vec3<f32> = select(vec3<f32>(0.0, 1.0, 0.0), vec3<f32>(1.0, 0.0, 0.0), abs(normal.x) < 0.5);
//      let anchor: vec3<f32> = select(vec3<f32>(0.0, 1.0, 0.0), vec3<f32>(1.0, 0.0, 0.0), abs(normal.x) < 0.5);
        let   tangent: vec3<f32> = normalize(cross(anchor, normal));
//      let   tangent: vec3<f32> = normalize(cross(anchor, normal));
        let bitangent: vec3<f32> = cross(normal, tangent);
//      let bitangent: vec3<f32> = cross(normal, tangent);
        let u: f32 = dot(samplePoint,   tangent);
//      let u: f32 = dot(samplePoint,   tangent);
        let v: f32 = dot(samplePoint, bitangent);
//      let v: f32 = dot(samplePoint, bitangent);
        return vec2<f32>(u, v) * scale;
//      return vec2<f32>(u, v) * scale;
    }
    */

    const SDF_MAX_MARCHING_STEPS: u32 = 1000;
//  const SDF_MAX_MARCHING_STEPS: u32 = 1000;
    const SDF_MAX_MARCHING_DISTANCE: f32 = 1000.0;
//  const SDF_MAX_MARCHING_DISTANCE: f32 = 1000.0;
    const SDF_HIT_EPSILON: f32 = 1.0e-5;
//  const SDF_HIT_EPSILON: f32 = 1.0e-5;

    fn _rayHitSDF(ray: Ray, rayTravelDistanceLimit: Interval) -> RayHitResult
//  fn _rayHitSDF(ray: Ray, rayTravelDistanceLimit: Interval) -> RayHitResult
    {
        const sceneAABB3D = AABB3D(
//      const sceneAABB3D = AABB3D(
            Interval(-50, 50),
//          Interval(-50, 50),
            Interval(-50, 50),
//          Interval(-50, 50),
            Interval(-50, 50),
//          Interval(-50, 50),
        );
//      );
        if (!_rayHitAABB3D(ray, rayTravelDistanceLimit, sceneAABB3D))
//      if (!_rayHitAABB3D(ray, rayTravelDistanceLimit, sceneAABB3D))
        {
            var rayHitResult: RayHitResult;
//          var rayHitResult: RayHitResult;
            rayHitResult.isHitted = false;
//          rayHitResult.isHitted = false;
            return rayHitResult;
//          return rayHitResult;
        }
        var currentDistance: f32 = rayTravelDistanceLimit.min;
//      var currentDistance: f32 = rayTravelDistanceLimit.min;
        var sdfHitResult: SDFHitResult;
//      var sdfHitResult: SDFHitResult;
        var rayHitResult: RayHitResult;
//      var rayHitResult: RayHitResult;
        rayHitResult.isHitted = false;
//      rayHitResult.isHitted = false;
        for (var sdfMarchingStep: u32 = 0; sdfMarchingStep < SDF_MAX_MARCHING_STEPS; sdfMarchingStep += 1)
//      for (var sdfMarchingStep: u32 = 0; sdfMarchingStep < SDF_MAX_MARCHING_STEPS; sdfMarchingStep += 1)
            {
                let currentPoint: vec3<f32> = _rayMarch(ray, currentDistance);
//              let currentPoint: vec3<f32> = _rayMarch(ray, currentDistance);
                sdfHitResult = _sdfMain(currentPoint);
//              sdfHitResult = _sdfMain(currentPoint);
                if (sdfHitResult.distance < SDF_HIT_EPSILON)
//              if (sdfHitResult.distance < SDF_HIT_EPSILON)
                {
                    rayHitResult.isHitted = true;
//                  rayHitResult.isHitted = true;
                    rayHitResult.at = currentPoint;
//                  rayHitResult.at = currentPoint;
                    rayHitResult.minDistance = currentDistance;
//                  rayHitResult.minDistance = currentDistance;
                    var materialIndex: u32;
//                  var materialIndex: u32;
                    var outwardNormal: vec3<f32>;
//                  var outwardNormal: vec3<f32>;
                    var uvSurfaceCoordinate: vec2<f32>;
//                  var uvSurfaceCoordinate: vec2<f32>;
                    if (sdfHitResult.isSphere)
//                  if (sdfHitResult.isSphere)
                    {
                        let sphere: Sphere = spheres[sdfHitResult.geometryIndex];
//                      let sphere: Sphere = spheres[sdfHitResult.geometryIndex];
                        materialIndex = sphere.materialIndex;
//                      materialIndex = sphere.materialIndex;
                        outwardNormal = normalize(rayHitResult.at - sphere.center);
//                      outwardNormal = normalize(rayHitResult.at - sphere.center);
                        let theta: f32 = acos (-outwardNormal.y); // latitude
//                      let theta: f32 = acos (-outwardNormal.y); // latitude
                        let phi  : f32 = atan2(-outwardNormal.z, outwardNormal.x) + PI; // longitude
//                      let phi  : f32 = atan2(-outwardNormal.z, outwardNormal.x) + PI; // longitude
                        uvSurfaceCoordinate = vec2<f32>(phi / (2.0 * PI), theta / PI);
//                      uvSurfaceCoordinate = vec2<f32>(phi / (2.0 * PI), theta / PI);
                    }
                    else
//                  else
                    if (sdfHitResult.isTriangle)
//                  if (sdfHitResult.isTriangle)
                    {
                        let triangle: Triangle = triangles[sdfHitResult.geometryIndex];
//                      let triangle: Triangle = triangles[sdfHitResult.geometryIndex];
                        materialIndex = triangle.materialIndex;
//                      materialIndex = triangle.materialIndex;

                        let triangleEdge1: vec3<f32> = triangle.vertex1 - triangle.vertex0;
//                      let triangleEdge1: vec3<f32> = triangle.vertex1 - triangle.vertex0;
                        let triangleEdge2: vec3<f32> = triangle.vertex2 - triangle.vertex0;
//                      let triangleEdge2: vec3<f32> = triangle.vertex2 - triangle.vertex0;

                        let rayDirectionCrossTriangleEdge2: vec3<f32> = cross(ray.direction, triangleEdge2);
//                      let rayDirectionCrossTriangleEdge2: vec3<f32> = cross(ray.direction, triangleEdge2);
                        let determinant: f32 = dot(triangleEdge1, rayDirectionCrossTriangleEdge2);
//                      let determinant: f32 = dot(triangleEdge1, rayDirectionCrossTriangleEdge2);

                        let inverseDeterminant: f32 = 1.0 / determinant;
//                      let inverseDeterminant: f32 = 1.0 / determinant;
                        let vectorFromTriangleVertex0ToRayOrigin: vec3<f32> = ray.origin - triangle.vertex0;
//                      let vectorFromTriangleVertex0ToRayOrigin: vec3<f32> = ray.origin - triangle.vertex0;

                        let w1Barycentric: f32 = inverseDeterminant * dot(vectorFromTriangleVertex0ToRayOrigin, rayDirectionCrossTriangleEdge2);
//                      let w1Barycentric: f32 = inverseDeterminant * dot(vectorFromTriangleVertex0ToRayOrigin, rayDirectionCrossTriangleEdge2);

                        let rayOriginCrossTriangleEdge1: vec3<f32> = cross(vectorFromTriangleVertex0ToRayOrigin, triangleEdge1);
//                      let rayOriginCrossTriangleEdge1: vec3<f32> = cross(vectorFromTriangleVertex0ToRayOrigin, triangleEdge1);
                        let w2Barycentric: f32 = inverseDeterminant * dot(ray.direction, rayOriginCrossTriangleEdge1);
//                      let w2Barycentric: f32 = inverseDeterminant * dot(ray.direction, rayOriginCrossTriangleEdge1);

                        let triangleVertex0FrontFaceNormal: vec3<f32> = decodeOctahedralNormal(triangle.vertex0FrontFaceNormalEncoded);
//                      let triangleVertex0FrontFaceNormal: vec3<f32> = decodeOctahedralNormal(triangle.vertex0FrontFaceNormalEncoded);
                        let triangleVertex1FrontFaceNormal: vec3<f32> = decodeOctahedralNormal(triangle.vertex1FrontFaceNormalEncoded);
//                      let triangleVertex1FrontFaceNormal: vec3<f32> = decodeOctahedralNormal(triangle.vertex1FrontFaceNormalEncoded);
                        let triangleVertex2FrontFaceNormal: vec3<f32> = decodeOctahedralNormal(triangle.vertex2FrontFaceNormalEncoded);
//                      let triangleVertex2FrontFaceNormal: vec3<f32> = decodeOctahedralNormal(triangle.vertex2FrontFaceNormalEncoded);
                        let triangleVertex0UV: vec2<f32> = decodeQuantizedUV(triangle.vertex0UVEncoded); // unpack2x16unorm(triangle.vertex0UVEncoded);
//                      let triangleVertex0UV: vec2<f32> = decodeQuantizedUV(triangle.vertex0UVEncoded); // unpack2x16unorm(triangle.vertex0UVEncoded);
                        let triangleVertex1UV: vec2<f32> = decodeQuantizedUV(triangle.vertex1UVEncoded); // unpack2x16unorm(triangle.vertex1UVEncoded);
//                      let triangleVertex1UV: vec2<f32> = decodeQuantizedUV(triangle.vertex1UVEncoded); // unpack2x16unorm(triangle.vertex1UVEncoded);
                        let triangleVertex2UV: vec2<f32> = decodeQuantizedUV(triangle.vertex2UVEncoded); // unpack2x16unorm(triangle.vertex2UVEncoded);
//                      let triangleVertex2UV: vec2<f32> = decodeQuantizedUV(triangle.vertex2UVEncoded); // unpack2x16unorm(triangle.vertex2UVEncoded);

                        let w0Barycentric: f32 = 1.0 - w1Barycentric - w2Barycentric;
//                      let w0Barycentric: f32 = 1.0 - w1Barycentric - w2Barycentric;
                        let interpolatedFrontFaceNormal: vec3<f32> = normalize(w0Barycentric * triangleVertex0FrontFaceNormal + w1Barycentric * triangleVertex1FrontFaceNormal + w2Barycentric * triangleVertex2FrontFaceNormal);
//                      let interpolatedFrontFaceNormal: vec3<f32> = normalize(w0Barycentric * triangleVertex0FrontFaceNormal + w1Barycentric * triangleVertex1FrontFaceNormal + w2Barycentric * triangleVertex2FrontFaceNormal);
                        outwardNormal = interpolatedFrontFaceNormal; // normalize(cross(triangle.vertex1 - triangle.vertex0, triangle.vertex2 - triangle.vertex0));
//                      outwardNormal = interpolatedFrontFaceNormal; // normalize(cross(triangle.vertex1 - triangle.vertex0, triangle.vertex2 - triangle.vertex0));

                        uvSurfaceCoordinate = w0Barycentric * triangleVertex0UV + w1Barycentric * triangleVertex1UV + w2Barycentric * triangleVertex2UV;
//                      uvSurfaceCoordinate = w0Barycentric * triangleVertex0UV + w1Barycentric * triangleVertex1UV + w2Barycentric * triangleVertex2UV;
                    }
//                     else
// //                  else
//                     if (sdfHitResult.isCube)
// //                  if (sdfHitResult.isCube)
//                     {
//                         let cube: Cube = cubes[sdfHitResult.geometryIndex];
// //                      let cube: Cube = cubes[sdfHitResult.geometryIndex];
//                         materialIndex = cube.materialIndex;
// //                      materialIndex = cube.materialIndex;
//                     }
                    rayHitResult.materialIndex = materialIndex;
//                  rayHitResult.materialIndex = materialIndex;
                    rayHitResult.isFrontFaceHitted = true;
//                  rayHitResult.isFrontFaceHitted = true;
                    rayHitResult.hittedSideNormal = outwardNormal;
//                  rayHitResult.hittedSideNormal = outwardNormal;
                    rayHitResult.uvSurfaceCoordinate = uvSurfaceCoordinate;
//                  rayHitResult.uvSurfaceCoordinate = uvSurfaceCoordinate;
                    /*
                    let finalSDFHitResult: SDFHitResult = _sdfMain(rayHitResult.at);
//                  let finalSDFHitResult: SDFHitResult = _sdfMain(rayHitResult.at);
                    if (finalSDFHitResult.distance < 0.0)
//                  if (finalSDFHitResult.distance < 0.0)
                    {
                        rayHitResult.at -= outwardNormal * finalSDFHitResult.distance;
//                      rayHitResult.at -= outwardNormal * finalSDFHitResult.distance;
                    }
                    */
                    break;
//                  break;
                }
                currentDistance += sdfHitResult.distance;
//              currentDistance += sdfHitResult.distance;
                if (currentDistance > rayTravelDistanceLimit.max || currentDistance > SDF_MAX_MARCHING_DISTANCE)
//              if (currentDistance > rayTravelDistanceLimit.max || currentDistance > SDF_MAX_MARCHING_DISTANCE)
                {
                    break;
//                  break;
                }
            }
            return rayHitResult;
//          return rayHitResult;
    }

    fn _rayHitLBVH_AnyHit(ray: Ray, rayTravelDistanceLimit: Interval) -> bool
//  fn _rayHitLBVH_AnyHit(ray: Ray, rayTravelDistanceLimit: Interval) -> bool
    {

        // 1. Initialize stack.
//      // 1. Initialize stack.
        // Size 32 is sufficient for 2^32 triangles (4 billion).
//      // Size 32 is sufficient for 2^32 triangles (4 billion).
        var stack: array<i32, 32>;
//      var stack: array<i32, 32>;
        var stackPointer: i32 = 0;
//      var stackPointer: i32 = 0;

        // Check Root AABB Intersection outside the loop (Optimized)
//      // Check Root AABB Intersection outside the loop (Optimized)
        let rootNode: BVHNode = bvhNodes[0];
//      let rootNode: BVHNode = bvhNodes[0];
        if (!_rayHitAABB3D(ray, rayTravelDistanceLimit, rootNode.aabb3d))
//      if (!_rayHitAABB3D(ray, rayTravelDistanceLimit, rootNode.aabb3d))
        {
            return false;
//          return false;
        }

        // Push root node (index 0) into stack.
//      // Push root node (index 0) into stack.
        stack[0] = 0;
//      stack[0] = 0;
        stackPointer = 1;
//      stackPointer = 1;

        loop
//      loop
        {
            // Check if stack is empty.
//          // Check if stack is empty.
            if (stackPointer == 0)
//          if (stackPointer == 0)
            {
                break;
//              break;
            }

            // Pop stack.
//          // Pop stack.
            stackPointer = stackPointer - 1;
//          stackPointer = stackPointer - 1;
            let nodeIndex: i32 = stack[stackPointer];
//          let nodeIndex: i32 = stack[stackPointer];
            let node: BVHNode = bvhNodes[nodeIndex];
//          let node: BVHNode = bvhNodes[nodeIndex];

            // 3. Leaf node case.
//          // 3. Leaf node case.
            if (node.data1 < 0)
//          if (node.data1 < 0)
            {
                let triangleIndex: u32 = u32(~node.data1);
//              let triangleIndex: u32 = u32(~node.data1);
                if (_rayHitTriangle_AnyHit(ray, triangleIndex, rayTravelDistanceLimit))
//              if (_rayHitTriangle_AnyHit(ray, triangleIndex, rayTravelDistanceLimit))
                {
                    // For shadows (AnyHit), we return true on the FIRST hit.
//                  // For shadows (AnyHit), we return true on the FIRST hit.
                    return true;
//                  return true;
                }
                continue;
//              continue;
            }

            // 4. Internal node case.
//          // 4. Internal node case.

            let childIndexL: i32 = node.data1;
//          let childIndexL: i32 = node.data1;
            let childIndexR: i32 = node.data2;
//          let childIndexR: i32 = node.data2;

            let childL: BVHNode = bvhNodes[childIndexL];
//          let childL: BVHNode = bvhNodes[childIndexL];
            let childR: BVHNode = bvhNodes[childIndexR];
//          let childR: BVHNode = bvhNodes[childIndexR];

            // Calculate distances to children
//          // Calculate distances to children
            let distanceL: f32 = _calculateDistanceToAABB3D(ray, rayTravelDistanceLimit, childL.aabb3d);
//          let distanceL: f32 = _calculateDistanceToAABB3D(ray, rayTravelDistanceLimit, childL.aabb3d);
            let distanceR: f32 = _calculateDistanceToAABB3D(ray, rayTravelDistanceLimit, childR.aabb3d);
//          let distanceR: f32 = _calculateDistanceToAABB3D(ray, rayTravelDistanceLimit, childR.aabb3d);

            // OPTIMIZATION: If both are Infinity (missed), continue.
//          // OPTIMIZATION: If both are Infinity (missed), continue.
            if (distanceL == 9999999.0 && distanceR == 9999999.0)
//          if (distanceL == 9999999.0 && distanceR == 9999999.0)
            {
                continue;
//              continue;
            }

            // PUSH ORDER: Push the FAR one first, so we process the NEAR one next.
//          // PUSH ORDER: Push the FAR one first, so we process the NEAR one next.
            if (distanceL < distanceR)
//          if (distanceL < distanceR)
            {
                // L is closer. Push R first (if hit).
//              // L is closer. Push R first (if hit).
                if (distanceR < 9999999.0)
//              if (distanceR < 9999999.0)
                {
                    stack[stackPointer] = childIndexR;
//                  stack[stackPointer] = childIndexR;
                    stackPointer = stackPointer + 1;
//                  stackPointer = stackPointer + 1;
                }
                // Push L second.
//              // Push L second.
                if (distanceL < 9999999.0)
//              if (distanceL < 9999999.0)
                {
                    stack[stackPointer] = childIndexL;
//                  stack[stackPointer] = childIndexL;
                    stackPointer = stackPointer + 1;
//                  stackPointer = stackPointer + 1;
                }
            }
            else
//          else
            {
                // R is closer. Push L first (if hit).
//              // R is closer. Push L first (if hit).
                if (distanceL < 9999999.0)
//              if (distanceL < 9999999.0)
                {
                    stack[stackPointer] = childIndexL;
//                  stack[stackPointer] = childIndexL;
                    stackPointer = stackPointer + 1;
//                  stackPointer = stackPointer + 1;
                }
                // Push R second.
//              // Push R second.
                if (distanceR < 9999999.0)
//              if (distanceR < 9999999.0)
                {
                    stack[stackPointer] = childIndexR;
//                  stack[stackPointer] = childIndexR;
                    stackPointer = stackPointer + 1;
//                  stackPointer = stackPointer + 1;
                }
            }
        }

        return false;
//      return false;
    }

    // stack[stackPointer] = value; stackPointer++; -> push
//  // stack[stackPointer] = value; stackPointer++; -> push
    // stackPointer--; value = stack[stackPointer]; -> pop
//  // stackPointer--; value = stack[stackPointer]; -> pop
    fn _rayHitLBVH_ClosestHit(ray: Ray, rayTravelDistanceLimit: Interval) -> RayHitResult
//  fn _rayHitLBVH_ClosestHit(ray: Ray, rayTravelDistanceLimit: Interval) -> RayHitResult
    {

        var finalRayHitResult: RayHitResult;
//      var finalRayHitResult: RayHitResult;
        finalRayHitResult.isHitted = false;
//      finalRayHitResult.isHitted = false;
        var maxTravelableDistanceSoFarForRay: f32 = rayTravelDistanceLimit.max;
//      var maxTravelableDistanceSoFarForRay: f32 = rayTravelDistanceLimit.max;

        // 1. Initialize stack.
//      // 1. Initialize stack.
        // Size 32 is sufficient for 2^32 triangles (4 billion), which is plenty.
//      // Size 32 is sufficient for 2^32 triangles (4 billion), which is plenty.
        var stack: array<i32, 32>;
//      var stack: array<i32, 32>;
        var stackPointer: i32 = 0;
//      var stackPointer: i32 = 0;

        // Push root node (index 0) into stack.
//      // Push root node (index 0) into stack.
        stack[0] = 0;
//      stack[0] = 0;
        stackPointer = 1;
//      stackPointer = 1;

        loop
//      loop
        {
            // Check if stack is empty.
//          // Check if stack is empty.
            if (stackPointer == 0)
//          if (stackPointer == 0)
            {
                break;
//              break;
            }

            // Pop stack.
//          // Pop stack.
            stackPointer = stackPointer - 1;
//          stackPointer = stackPointer - 1;
            let nodeIndex: i32 = stack[stackPointer];
//          let nodeIndex: i32 = stack[stackPointer];
            let node: BVHNode = bvhNodes[nodeIndex];
//          let node: BVHNode = bvhNodes[nodeIndex];

            let updatedRayTravelDistanceLimit: Interval = Interval(rayTravelDistanceLimit.min, maxTravelableDistanceSoFarForRay);
//          let updatedRayTravelDistanceLimit: Interval = Interval(rayTravelDistanceLimit.min, maxTravelableDistanceSoFarForRay);

            // 2. Check AABB intersection.
//          // 2. Check AABB intersection.
            if (!_rayHitAABB3D(ray, updatedRayTravelDistanceLimit, node.aabb3d))
//          if (!_rayHitAABB3D(ray, updatedRayTravelDistanceLimit, node.aabb3d))
            {
                continue;
//              continue;
            }

            // 3. Leaf node case.
//          // 3. Leaf node case.
            if (node.data1 < 0)
//          if (node.data1 < 0)
            {
                let triangleIndex: u32 = u32(~node.data1);
//              let triangleIndex: u32 = u32(~node.data1);
                let temporaryRayHitResult: RayHitResult = _rayHitTriangle(ray, triangleIndex, updatedRayTravelDistanceLimit);
//              let temporaryRayHitResult: RayHitResult = _rayHitTriangle(ray, triangleIndex, updatedRayTravelDistanceLimit);
                if (temporaryRayHitResult.isHitted && temporaryRayHitResult.minDistance < maxTravelableDistanceSoFarForRay)
//              if (temporaryRayHitResult.isHitted && temporaryRayHitResult.minDistance < maxTravelableDistanceSoFarForRay)
                {
                    finalRayHitResult = temporaryRayHitResult;
//                  finalRayHitResult = temporaryRayHitResult;
                    maxTravelableDistanceSoFarForRay = temporaryRayHitResult.minDistance;
//                  maxTravelableDistanceSoFarForRay = temporaryRayHitResult.minDistance;
                }
                continue;
//              continue;
            }

            // 4. Internal node case (THE OPTIMIZATION).
//          // 4. Internal node case (THE OPTIMIZATION).

            let childIndexL: i32 = node.data1;
//          let childIndexL: i32 = node.data1;
            let childIndexR: i32 = node.data2;
//          let childIndexR: i32 = node.data2;

            // We need to look at the children.
//          // We need to look at the children.
            // To do front-to-back traversal, we calculate the distance to both children.
//          // To do front-to-back traversal, we calculate the distance to both children.
            // NOTE: This adds some overhead, but usually pays off by culling huge branches later.
//          // NOTE: This adds some overhead, but usually pays off by culling huge branches later.

            let childL: BVHNode = bvhNodes[childIndexL];
//          let childL: BVHNode = bvhNodes[childIndexL];
            let childR: BVHNode = bvhNodes[childIndexR];
//          let childR: BVHNode = bvhNodes[childIndexR];

            // Quick distance check (returns infinity if miss).
//          // Quick distance check (returns infinity if miss).
            let distanceL: f32 = _calculateDistanceToAABB3D(ray, updatedRayTravelDistanceLimit, childL.aabb3d);
//          let distanceL: f32 = _calculateDistanceToAABB3D(ray, updatedRayTravelDistanceLimit, childL.aabb3d);
            let distanceR: f32 = _calculateDistanceToAABB3D(ray, updatedRayTravelDistanceLimit, childR.aabb3d);
//          let distanceR: f32 = _calculateDistanceToAABB3D(ray, updatedRayTravelDistanceLimit, childR.aabb3d);

            // OPTIMIZATION: If both are Infinity (missed), continue.
//          // OPTIMIZATION: If both are Infinity (missed), continue.
            if (distanceL == 9999999.0 && distanceR == 9999999.0)
//          if (distanceL == 9999999.0 && distanceR == 9999999.0)
            {
                continue;
//              continue;
            }

            // PUSH ORDER: Push the FAR one first, so we process the NEAR one next.
//          // PUSH ORDER: Push the FAR one first, so we process the NEAR one next.
            if (distanceL < distanceR)
//          if (distanceL < distanceR)
            {
                // L is closer.
//              // L is closer.
                // Push R first (if it is hitted).
//              // Push R first (if it is hitted).
                if (distanceR < maxTravelableDistanceSoFarForRay)
//              if (distanceR < maxTravelableDistanceSoFarForRay)
                {
                    stack[stackPointer] = childIndexR;
//                  stack[stackPointer] = childIndexR;
                    stackPointer = stackPointer + 1;
//                  stackPointer = stackPointer + 1;
                }
                // Push L second (so it pops first).
//              // Push L second (so it pops first).
                if (distanceL < maxTravelableDistanceSoFarForRay)
//              if (distanceL < maxTravelableDistanceSoFarForRay)
                {
                    stack[stackPointer] = childIndexL;
//                  stack[stackPointer] = childIndexL;
                    stackPointer = stackPointer + 1;
//                  stackPointer = stackPointer + 1;
                }
            }
            else
            {
                // R is closer (or equal).
//              // R is closer (or equal).
                // Push L first (if it is hitted).
//              // Push L first (if it is hitted).
                if (distanceL < maxTravelableDistanceSoFarForRay)
//              if (distanceL < maxTravelableDistanceSoFarForRay)
                {
                    stack[stackPointer] = childIndexL;
//                  stack[stackPointer] = childIndexL;
                    stackPointer = stackPointer + 1;
//                  stackPointer = stackPointer + 1;
                }
                // Push R second (so it pops first).
//              // Push R second (so it pops first).
                if (distanceR < maxTravelableDistanceSoFarForRay)
//              if (distanceR < maxTravelableDistanceSoFarForRay)
                {
                    stack[stackPointer] = childIndexR;
//                  stack[stackPointer] = childIndexR;
                    stackPointer = stackPointer + 1;
//                  stackPointer = stackPointer + 1;
                }
            }
        }

        return finalRayHitResult;
//      return finalRayHitResult;

    }

    // HELPER: Simply returns the distance to the bounding box, or huge number if miss.
//  // HELPER: Simply returns the distance to the bounding box, or huge number if miss.
    fn _calculateDistanceToAABB3D(ray: Ray, rayTravelDistanceLimit: Interval, aabb3d: AABB3D) -> f32
//  fn _calculateDistanceToAABB3D(ray: Ray, rayTravelDistanceLimit: Interval, aabb3d: AABB3D) -> f32
    {

        let aabb3dMinCornersLimit: vec3<f32> = vec3<f32>(aabb3d.cornerLimitAxisX.min, aabb3d.cornerLimitAxisY.min, aabb3d.cornerLimitAxisZ.min);
//      let aabb3dMinCornersLimit: vec3<f32> = vec3<f32>(aabb3d.cornerLimitAxisX.min, aabb3d.cornerLimitAxisY.min, aabb3d.cornerLimitAxisZ.min);
        let aabb3dMaxCornersLimit: vec3<f32> = vec3<f32>(aabb3d.cornerLimitAxisX.max, aabb3d.cornerLimitAxisY.max, aabb3d.cornerLimitAxisZ.max);
//      let aabb3dMaxCornersLimit: vec3<f32> = vec3<f32>(aabb3d.cornerLimitAxisX.max, aabb3d.cornerLimitAxisY.max, aabb3d.cornerLimitAxisZ.max);

        let distancesFromRayOriginToMinCornersLimit: vec3<f32> = (aabb3dMinCornersLimit - ray.origin) * ray.directionInverse;
//      let distancesFromRayOriginToMinCornersLimit: vec3<f32> = (aabb3dMinCornersLimit - ray.origin) * ray.directionInverse;
        let distancesFromRayOriginToMaxCornersLimit: vec3<f32> = (aabb3dMaxCornersLimit - ray.origin) * ray.directionInverse;
//      let distancesFromRayOriginToMaxCornersLimit: vec3<f32> = (aabb3dMaxCornersLimit - ray.origin) * ray.directionInverse;

        let distancesMin: vec3<f32> = max(min(distancesFromRayOriginToMinCornersLimit, distancesFromRayOriginToMaxCornersLimit), vec3<f32>(rayTravelDistanceLimit.min));
//      let distancesMin: vec3<f32> = max(min(distancesFromRayOriginToMinCornersLimit, distancesFromRayOriginToMaxCornersLimit), vec3<f32>(rayTravelDistanceLimit.min));
        let distancesMax: vec3<f32> = min(max(distancesFromRayOriginToMinCornersLimit, distancesFromRayOriginToMaxCornersLimit), vec3<f32>(rayTravelDistanceLimit.max));
//      let distancesMax: vec3<f32> = min(max(distancesFromRayOriginToMinCornersLimit, distancesFromRayOriginToMaxCornersLimit), vec3<f32>(rayTravelDistanceLimit.max));

        let distanceToBoxMin: f32 = max(max(distancesMin.x, distancesMin.y), distancesMin.z);
//      let distanceToBoxMin: f32 = max(max(distancesMin.x, distancesMin.y), distancesMin.z);
        let distanceToBoxMax: f32 = min(min(distancesMax.x, distancesMax.y), distancesMax.z);
//      let distanceToBoxMax: f32 = min(min(distancesMax.x, distancesMax.y), distancesMax.z);

        return select(9999999.0 /* Miss */, distanceToBoxMin, distanceToBoxMax >= distanceToBoxMin && distanceToBoxMin < rayTravelDistanceLimit.max);
//      return select(9999999.0 /* Miss */, distanceToBoxMin, distanceToBoxMax >= distanceToBoxMin && distanceToBoxMin < rayTravelDistanceLimit.max);

    }

    fn _hash2(seed: f32) -> vec2<f32>
//  fn _hash2(seed: f32) -> vec2<f32>
    {
        return fract(sin(vec2(seed + 0.1, seed + 0.2)) * vec2(43758.5453123, 22578.1459123));
//      return fract(sin(vec2(seed + 0.1, seed + 0.2)) * vec2(43758.5453123, 22578.1459123));
    }

    fn _hash2ToVec2I32(hash2Value: vec2<f32>, step: i32, base: i32) -> vec2<i32>
//  fn _hash2ToVec2I32(hash2Value: vec2<f32>, step: i32, base: i32) -> vec2<i32>
    {
        return vec2<i32>(
//      return vec2<i32>(
            i32(hash2Value.x * f32(base * step)),
            i32(hash2Value.y * f32(base * step))
        );
//      );
    }

    //  Screen Space Volumetric Lighting as Post Pross, by Ridge/winlandiano
//  //  Screen Space Volumetric Lighting as Post Pross, by Ridge/winlandiano
    //  Originally presented in GPU Gem3 Chap. 13, by Kenny Mitchell.
//  //  Originally presented in GPU Gem3 Chap. 13, by Kenny Mitchell.
    //  License Creative Commons Attribution-NonCommercial-ShareAlike 3.0 Unported License
//  //  License Creative Commons Attribution-NonCommercial-ShareAlike 3.0 Unported License

    const SSVL_NUM_SAMPLES: u32 = 100;
//  const SSVL_NUM_SAMPLES: u32 = 100;
    const SSVL_DENSITY: f32 = 2.0;
//  const SSVL_DENSITY: f32 = 2.0;
    const SSVL_WEIGHT: f32 = 0.1;
//  const SSVL_WEIGHT: f32 = 0.1;
    const SSVL_DECAY: f32 = 0.92;
//  const SSVL_DECAY: f32 = 0.92;
    const SSVL_EXPOSURE: f32 = 0.4;
//  const SSVL_EXPOSURE: f32 = 0.4;

    fn _screenSpaceVolumetricLighting(tex: texture_2d<f32>, sam: sampler, fragmentShaderInputUV: vec2<f32>, lightPosition: vec2<f32>) -> vec3<f32>
//  fn _screenSpaceVolumetricLighting(tex: texture_2d<f32>, sam: sampler, fragmentShaderInputUV: vec2<f32>, lightPosition: vec2<f32>) -> vec3<f32>
    {
        var uv: vec2<f32> = fragmentShaderInputUV;
//      var uv: vec2<f32> = fragmentShaderInputUV;
        var color: vec3<f32> = textureSample(tex, sam, uv).rgb;
//      var color: vec3<f32> = textureSample(tex, sam, uv).rgb;
        var sampleStep: vec2<f32> = uv - lightPosition;
//      var sampleStep: vec2<f32> = uv - lightPosition;
        sampleStep *= SSVL_DENSITY / f32(SSVL_NUM_SAMPLES);
//      sampleStep *= SSVL_DENSITY / f32(SSVL_NUM_SAMPLES);
        var illuminationDecay: f32 = 1.0;
//      var illuminationDecay: f32 = 1.0;
        for (var i: u32 = 0; i < SSVL_NUM_SAMPLES; i++)
//      for (var i: u32 = 0; i < SSVL_NUM_SAMPLES; i++)
        {
            uv -= sampleStep;
//          uv -= sampleStep;
            let sampleColor: vec3<f32> = textureSample(tex, sam, uv).rgb;
//          let sampleColor: vec3<f32> = textureSample(tex, sam, uv).rgb;
            var brightness: f32 = (sampleColor.r + sampleColor.g + sampleColor.b) / 3.0;
//          var brightness: f32 = (sampleColor.r + sampleColor.g + sampleColor.b) / 3.0;
            brightness *= illuminationDecay * SSVL_WEIGHT;
//          brightness *= illuminationDecay * SSVL_WEIGHT;
            color += vec3<f32>(brightness);
//          color += vec3<f32>(brightness);
            illuminationDecay *= SSVL_DECAY;
//          illuminationDecay *= SSVL_DECAY;
        }
        return color * SSVL_EXPOSURE;
//      return color * SSVL_EXPOSURE;
    }

    fn _getVogelDiskSample(sampleIndex: f32, sampleCount: f32, phi: f32) -> vec2<f32>
//  fn _getVogelDiskSample(sampleIndex: f32, sampleCount: f32, phi: f32) -> vec2<f32>
    {
        const goldenAngle: f32 = radians(180.0) * (3.0 - sqrt(5.0));
//      const goldenAngle: f32 = radians(180.0) * (3.0 - sqrt(5.0));

        let r: f32 = sqrt((sampleIndex + 0.5) / sampleCount); // Assuming index and count are positive
//      let r: f32 = sqrt((sampleIndex + 0.5) / sampleCount); // Assuming index and count are positive
        let theta: f32 = sampleIndex * goldenAngle + phi;
//      let theta: f32 = sampleIndex * goldenAngle + phi;

        let   sine: f32 = sin(theta);
//      let   sine: f32 = sin(theta);
        let cosine: f32 = cos(theta);
//      let cosine: f32 = cos(theta);

        return vec2(cosine, sine) * r;
//      return vec2(cosine, sine) * r;
    }

    fn _generateRandomPointInsideNormalizedDisk(rng: ptr<function, RNG>) -> vec3<f32>
//  fn _generateRandomPointInsideNormalizedDisk(rng: ptr<function, RNG>) -> vec3<f32>
    {
        //  polar sampling
//      //  polar sampling
        let r: f32 = sqrt(_pcg32Next(rng)); // radius (sqrt ensures uniform distribution)
//      let r: f32 = sqrt(_pcg32Next(rng)); // radius (sqrt ensures uniform distribution)
        let theta: f32 = _pcg32NextRangeF32(rng, 0.0, 2.0 * PI); // angle in radians
//      let theta: f32 = _pcg32NextRangeF32(rng, 0.0, 2.0 * PI); // angle in radians
        return vec3<f32>(r * cos(theta), r * sin(theta), 0.0);
//      return vec3<f32>(r * cos(theta), r * sin(theta), 0.0);
    }

    fn _getDefocusDiskSample(diskCenter: vec3<f32>, defocusDiskRadiusU: vec3<f32>, defocusDiskRadiusV: vec3<f32>, rng: ptr<function, RNG>) -> vec3<f32>
//  fn _getDefocusDiskSample(diskCenter: vec3<f32>, defocusDiskRadiusU: vec3<f32>, defocusDiskRadiusV: vec3<f32>, rng: ptr<function, RNG>) -> vec3<f32>
    {
        let randomPointInsideNormalizedDisk: vec3<f32> = _generateRandomPointInsideNormalizedDisk(rng);
//      let randomPointInsideNormalizedDisk: vec3<f32> = _generateRandomPointInsideNormalizedDisk(rng);
        return diskCenter + randomPointInsideNormalizedDisk.x * defocusDiskRadiusU
//      return diskCenter + randomPointInsideNormalizedDisk.x * defocusDiskRadiusU
                          + randomPointInsideNormalizedDisk.y * defocusDiskRadiusV;
//                        + randomPointInsideNormalizedDisk.y * defocusDiskRadiusV;
    }

    fn _finalizeFromRayHitResultClosestHit(currentRay: Ray, rayHitResultClosestHit: RayHitResult) -> RayHitResult
//  fn _finalizeFromRayHitResultClosestHit(currentRay: Ray, rayHitResultClosestHit: RayHitResult) -> RayHitResult
    {
        if (!rayHitResultClosestHit.isHitted)
//      if (!rayHitResultClosestHit.isHitted)
        {
            return rayHitResultClosestHit;
//          return rayHitResultClosestHit;
        }

        let triangle: Triangle = triangles[rayHitResultClosestHit.triangleIndex];
//      let triangle: Triangle = triangles[rayHitResultClosestHit.triangleIndex];

        let triangleVertex0FrontFaceNormal: vec3<f32> = decodeOctahedralNormal(triangle.vertex0FrontFaceNormalEncoded);
//      let triangleVertex0FrontFaceNormal: vec3<f32> = decodeOctahedralNormal(triangle.vertex0FrontFaceNormalEncoded);
        let triangleVertex1FrontFaceNormal: vec3<f32> = decodeOctahedralNormal(triangle.vertex1FrontFaceNormalEncoded);
//      let triangleVertex1FrontFaceNormal: vec3<f32> = decodeOctahedralNormal(triangle.vertex1FrontFaceNormalEncoded);
        let triangleVertex2FrontFaceNormal: vec3<f32> = decodeOctahedralNormal(triangle.vertex2FrontFaceNormalEncoded);
//      let triangleVertex2FrontFaceNormal: vec3<f32> = decodeOctahedralNormal(triangle.vertex2FrontFaceNormalEncoded);

        let triangleVertex0UV: vec2<f32> = decodeQuantizedUV(triangle.vertex0UVEncoded); // unpack2x16unorm(triangle.vertex0UVEncoded);
//      let triangleVertex0UV: vec2<f32> = decodeQuantizedUV(triangle.vertex0UVEncoded); // unpack2x16unorm(triangle.vertex0UVEncoded);
        let triangleVertex1UV: vec2<f32> = decodeQuantizedUV(triangle.vertex1UVEncoded); // unpack2x16unorm(triangle.vertex1UVEncoded);
//      let triangleVertex1UV: vec2<f32> = decodeQuantizedUV(triangle.vertex1UVEncoded); // unpack2x16unorm(triangle.vertex1UVEncoded);
        let triangleVertex2UV: vec2<f32> = decodeQuantizedUV(triangle.vertex2UVEncoded); // unpack2x16unorm(triangle.vertex2UVEncoded);
//      let triangleVertex2UV: vec2<f32> = decodeQuantizedUV(triangle.vertex2UVEncoded); // unpack2x16unorm(triangle.vertex2UVEncoded);

        var rayHitResultFinal: RayHitResult = rayHitResultClosestHit;
//      var rayHitResultFinal: RayHitResult = rayHitResultClosestHit;

        rayHitResultFinal.at = _rayMarch(currentRay, rayHitResultFinal.minDistance);
//      rayHitResultFinal.at = _rayMarch(currentRay, rayHitResultFinal.minDistance);

        // Retrieve Barycentrics from temporary storage
//      // Retrieve Barycentrics from temporary storage
        let w1Barycentric: f32 = rayHitResultFinal.uvSurfaceCoordinate.x;
//      let w1Barycentric: f32 = rayHitResultFinal.uvSurfaceCoordinate.x;
        let w2Barycentric: f32 = rayHitResultFinal.uvSurfaceCoordinate.y;
//      let w2Barycentric: f32 = rayHitResultFinal.uvSurfaceCoordinate.y;
        let w0Barycentric: f32 = 1.0 - w1Barycentric - w2Barycentric;
//      let w0Barycentric: f32 = 1.0 - w1Barycentric - w2Barycentric;

        let interpolatedFrontFaceNormal: vec3<f32> = normalize(w0Barycentric * triangleVertex0FrontFaceNormal + w1Barycentric * triangleVertex1FrontFaceNormal + w2Barycentric * triangleVertex2FrontFaceNormal);
//      let interpolatedFrontFaceNormal: vec3<f32> = normalize(w0Barycentric * triangleVertex0FrontFaceNormal + w1Barycentric * triangleVertex1FrontFaceNormal + w2Barycentric * triangleVertex2FrontFaceNormal);

        rayHitResultFinal.isFrontFaceHitted = dot(currentRay.direction, interpolatedFrontFaceNormal) < 0.0;
//      rayHitResultFinal.isFrontFaceHitted = dot(currentRay.direction, interpolatedFrontFaceNormal) < 0.0;
        rayHitResultFinal.hittedSideNormal = select(-interpolatedFrontFaceNormal, interpolatedFrontFaceNormal, rayHitResultFinal.isFrontFaceHitted);
//      rayHitResultFinal.hittedSideNormal = select(-interpolatedFrontFaceNormal, interpolatedFrontFaceNormal, rayHitResultFinal.isFrontFaceHitted);

        rayHitResultFinal.uvSurfaceCoordinate = w0Barycentric * triangleVertex0UV + w1Barycentric * triangleVertex1UV + w2Barycentric * triangleVertex2UV;
//      rayHitResultFinal.uvSurfaceCoordinate = w0Barycentric * triangleVertex0UV + w1Barycentric * triangleVertex1UV + w2Barycentric * triangleVertex2UV;

        return rayHitResultFinal;
//      return rayHitResultFinal;
    }

/* --- FOG --- */
    fn _applyFogHeight(accumulatedColor: vec3<f32>, minDistance: f32, initialRay: Ray, fogColor: vec3<f32>, fogDensity: f32, fogFalloff: f32) -> vec3<f32> {
//  fn _applyFogHeight(accumulatedColor: vec3<f32>, minDistance: f32, initialRay: Ray, fogColor: vec3<f32>, fogDensity: f32, fogFalloff: f32) -> vec3<f32> {
        var fogAmount: f32 = select(
//      var fogAmount: f32 = select(
            (fogDensity / fogFalloff) * (exp(-initialRay.origin.y * fogFalloff)) * ((1.0 - exp(-minDistance * initialRay.direction.y * fogFalloff)) / initialRay.direction.y)
//          (fogDensity / fogFalloff) * (exp(-initialRay.origin.y * fogFalloff)) * ((1.0 - exp(-minDistance * initialRay.direction.y * fogFalloff)) / initialRay.direction.y)
            ,fogDensity * exp(-initialRay.origin.y * fogFalloff) * minDistance
//          ,fogDensity * exp(-initialRay.origin.y * fogFalloff) * minDistance
            ,abs(initialRay.direction.y) < 1.0e-4
//          ,abs(initialRay.direction.y) < 1.0e-4
        );
//      );
        fogAmount = clamp(fogAmount, 0.0, 1.0);
//      fogAmount = clamp(fogAmount, 0.0, 1.0);
        return mix(accumulatedColor, fogColor, fogAmount);
//      return mix(accumulatedColor, fogColor, fogAmount);
    }
//  }
/* --- FOG --- */
