<script lang="ts">
    import { onMount } from "svelte";
//  import { onMount } from "svelte";
    import { onDestroy } from "svelte";
//  import { onDestroy } from "svelte";
    import shaderStringCompute from "../assets/compute.wgsl?raw";
//  import shaderStringCompute from "../assets/compute.wgsl?raw";
    import shaderStringGather from "../assets/gather.wgsl?raw";
//  import shaderStringGather from "../assets/gather.wgsl?raw";
    import shaderStringBlur2X from "../assets/blur2X.wgsl?raw";
//  import shaderStringBlur2X from "../assets/blur2X.wgsl?raw";
    import shaderStringBlur4X from "../assets/blur4X.wgsl?raw";
//  import shaderStringBlur4X from "../assets/blur4X.wgsl?raw";
    import shaderStringBlur8X from "../assets/blur8X.wgsl?raw";
//  import shaderStringBlur8X from "../assets/blur8X.wgsl?raw";
    import shaderStringRender from "../assets/render.wgsl?raw";
//  import shaderStringRender from "../assets/render.wgsl?raw";
    import * as m from "mathjs";
//  import * as m from "mathjs";




    import { parseEXRWithWorker } from "./exr/parse-exr-worker-wrapper";
//  import { parseEXRWithWorker } from "./exr/parse-exr-worker-wrapper";
    import * as t from "@tweenjs/tween.js";
//  import * as t from "@tweenjs/tween.js";
    import type { EasingFunction } from "svelte/transition";
//  import type { EasingFunction } from "svelte/transition";
    import { compareArrays, isEqual } from "./Helper.svelte";
//  import { compareArrays, isEqual } from "./Helper.svelte";
    import { encodeOctahedralNormal, decodeOctahedralNormal, encodeQuantizedUV, decodeQuantizedUV } from "./Helper.svelte";
//  import { encodeOctahedralNormal, decodeOctahedralNormal, encodeQuantizedUV, decodeQuantizedUV } from "./Helper.svelte";
    import * as assimp from "./assimp/assimp-worker-wrapper";
//  import * as assimp from "./assimp/assimp-worker-wrapper";
    import { createColumnAtlas, type ColumnAtlas } from "./atlas-images-merger-column/atlas-images-merger-column-worker-wrapper";
//  import { createColumnAtlas, type ColumnAtlas } from "./atlas-images-merger-column/atlas-images-merger-column-worker-wrapper";
    import * as utility from "./utility";
//  import * as utility from "./utility";



    import {
//  import {
        type AABB3D,
//      type AABB3D,
        type BackgroundType,
//      type BackgroundType,
        type BVHNode,
//      type BVHNode,
        type CPUTimeMeasurement,
//      type CPUTimeMeasurement,
        type Material,
//      type Material,
        type MaterialType,
//      type MaterialType,
        type RefractiveIndex,
//      type RefractiveIndex,
        type Sphere,
//      type Sphere,
        type Texture,
//      type Texture,
        type TextureType,
//      type TextureType,
        type Triangle,
//      type Triangle,
        type Vec2,
//      type Vec2,
        type Vec3,
//      type Vec3,
        type Vec4,
//      type Vec4,
    } from "./type";
//  } from "./type";
    import * as type from "./type";
//  import * as type from "./type";



    let _timeInSeconds: number;
//  let _timeInSeconds: number;
    let _accumulatedSampleCount: number = 0;
//  let _accumulatedSampleCount: number = 0;
    let _adapter: GPUAdapter;
//  let _adapter: GPUAdapter;
    let _device: GPUDevice;
//  let _device: GPUDevice;
    let _canvas: HTMLCanvasElement;
//  let _canvas: HTMLCanvasElement;
    let _canvasContext: GPUCanvasContext;
//  let _canvasContext: GPUCanvasContext;
    let _presentationFormat: GPUTextureFormat;
//  let _presentationFormat: GPUTextureFormat;
    let _computeShaderModule: GPUShaderModule;
//  let _computeShaderModule: GPUShaderModule;
    let _gatherShaderModule: GPUShaderModule;
//  let _gatherShaderModule: GPUShaderModule;
    let _blur2XShaderModule: GPUShaderModule;
//  let _blur2XShaderModule: GPUShaderModule;
    let _blur4XShaderModule: GPUShaderModule;
//  let _blur4XShaderModule: GPUShaderModule;
    let _blur8XShaderModule: GPUShaderModule;
//  let _blur8XShaderModule: GPUShaderModule;
    let _renderShaderModule: GPUShaderModule;
//  let _renderShaderModule: GPUShaderModule;
    let _computeBindGroupLayout0: GPUBindGroupLayout | null;
//  let _computeBindGroupLayout0: GPUBindGroupLayout | null;
    let _gatherBindGroupLayout0: GPUBindGroupLayout | null;
//  let _gatherBindGroupLayout0: GPUBindGroupLayout | null;
    let _blur2XBindGroupLayout0: GPUBindGroupLayout | null;
//  let _blur2XBindGroupLayout0: GPUBindGroupLayout | null;
    let _blur4XBindGroupLayout0: GPUBindGroupLayout | null;
//  let _blur4XBindGroupLayout0: GPUBindGroupLayout | null;
    let _blur8XBindGroupLayout0: GPUBindGroupLayout | null;
//  let _blur8XBindGroupLayout0: GPUBindGroupLayout | null;
    let _renderBindGroupLayout0: GPUBindGroupLayout | null;
//  let _renderBindGroupLayout0: GPUBindGroupLayout | null;
    let _computePipelineLayout: GPUPipelineLayout | "auto";
//  let _computePipelineLayout: GPUPipelineLayout | "auto";
    let _gatherPipelineLayout: GPUPipelineLayout | "auto";
//  let _gatherPipelineLayout: GPUPipelineLayout | "auto";
    let _blur2XPipelineLayout: GPUPipelineLayout | "auto";
//  let _blur2XPipelineLayout: GPUPipelineLayout | "auto";
    let _blur4XPipelineLayout: GPUPipelineLayout | "auto";
//  let _blur4XPipelineLayout: GPUPipelineLayout | "auto";
    let _blur8XPipelineLayout: GPUPipelineLayout | "auto";
//  let _blur8XPipelineLayout: GPUPipelineLayout | "auto";
    let _renderPipelineLayout: GPUPipelineLayout | "auto";
//  let _renderPipelineLayout: GPUPipelineLayout | "auto";
    let _computePipeline: GPUComputePipeline;
//  let _computePipeline: GPUComputePipeline;
    let _gatherPipeline: GPUComputePipeline;
//  let _gatherPipeline: GPUComputePipeline;
    let _blur2XPipeline: GPURenderPipeline;
//  let _blur2XPipeline: GPURenderPipeline;
    let _blur4XPipeline: GPURenderPipeline;
//  let _blur4XPipeline: GPURenderPipeline;
    let _blur8XPipeline: GPURenderPipeline;
//  let _blur8XPipeline: GPURenderPipeline;
    let _renderPipeline: GPURenderPipeline;
//  let _renderPipeline: GPURenderPipeline;
    let _commandEncoderDescriptor: GPUCommandEncoderDescriptor;
//  let _commandEncoderDescriptor: GPUCommandEncoderDescriptor;
    let _computePassDescriptor: GPUComputePassDescriptor;
//  let _computePassDescriptor: GPUComputePassDescriptor;
    let _gatherPassDescriptor: GPUComputePassDescriptor;
//  let _gatherPassDescriptor: GPUComputePassDescriptor;
    let _blur2XPassDescriptor: GPURenderPassDescriptor;
//  let _blur2XPassDescriptor: GPURenderPassDescriptor;
    let _blur4XPassDescriptor: GPURenderPassDescriptor;
//  let _blur4XPassDescriptor: GPURenderPassDescriptor;
    let _blur8XPassDescriptor: GPURenderPassDescriptor;
//  let _blur8XPassDescriptor: GPURenderPassDescriptor;
    let _renderPassDescriptor: GPURenderPassDescriptor;
//  let _renderPassDescriptor: GPURenderPassDescriptor;
    let _commandEncoder: GPUCommandEncoder;
//  let _commandEncoder: GPUCommandEncoder;
    let _computePass: GPUComputePassEncoder;
//  let _computePass: GPUComputePassEncoder;
    let _gatherPass: GPUComputePassEncoder;
//  let _gatherPass: GPUComputePassEncoder;
    let _blur2XPass: GPURenderPassEncoder;
//  let _blur2XPass: GPURenderPassEncoder;
    let _blur4XPass: GPURenderPassEncoder;
//  let _blur4XPass: GPURenderPassEncoder;
    let _blur8XPass: GPURenderPassEncoder;
//  let _blur8XPass: GPURenderPassEncoder;
    let _renderPass: GPURenderPassEncoder;
//  let _renderPass: GPURenderPassEncoder;
    let _commandBuffer: GPUCommandBuffer;
//  let _commandBuffer: GPUCommandBuffer;
    let _resizeObserver: ResizeObserver;
//  let _resizeObserver: ResizeObserver;
    let _generalDataUniformValuesDataView: DataView;
//  let _generalDataUniformValuesDataView: DataView;
    let _generalDataUniformValues: ArrayBuffer;
//  let _generalDataUniformValues: ArrayBuffer;
    let _generalDataUniformBuffer: GPUBuffer;
//  let _generalDataUniformBuffer: GPUBuffer;
    let _spheresStorageValuesDataView: DataView;
//  let _spheresStorageValuesDataView: DataView;
    let _spheresStorageValues: ArrayBuffer;
//  let _spheresStorageValues: ArrayBuffer;
    let _spheresStorageBuffer: GPUBuffer;
//  let _spheresStorageBuffer: GPUBuffer;
    let _spheres: Sphere[] = [];
//  let _spheres: Sphere[] = [];
    let _materialsStorageValuesDataView: DataView;
//  let _materialsStorageValuesDataView: DataView;
    let _materialsStorageValues: ArrayBuffer;
//  let _materialsStorageValues: ArrayBuffer;
    let _materialsStorageBuffer: GPUBuffer;
//  let _materialsStorageBuffer: GPUBuffer;
    let _materials: Material[] = [];
//  let _materials: Material[] = [];
    let _texturesStorageValuesDataView: DataView;
//  let _texturesStorageValuesDataView: DataView;
    let _texturesStorageValues: ArrayBuffer;
//  let _texturesStorageValues: ArrayBuffer;
    let _texturesStorageBuffer: GPUBuffer;
//  let _texturesStorageBuffer: GPUBuffer;
    let _textures: Texture[] = [];
//  let _textures: Texture[] = [];
    let _columnAtlasSampler: GPUSampler;
//  let _columnAtlasSampler: GPUSampler;
    let _columnAtlasTexture: GPUTexture;
//  let _columnAtlasTexture: GPUTexture;
    let _hdriSampler: GPUSampler;
//  let _hdriSampler: GPUSampler;
    let _hdriTexture: GPUTexture;
//  let _hdriTexture: GPUTexture;
    let _trianglesStorageValuesDataView: DataView;
//  let _trianglesStorageValuesDataView: DataView;
    let _trianglesStorageValues: ArrayBuffer;
//  let _trianglesStorageValues: ArrayBuffer;
    let _trianglesStorageBuffer: GPUBuffer;
//  let _trianglesStorageBuffer: GPUBuffer;
    let _triangles: Triangle[] = [];
//  let _triangles: Triangle[] = [];
    let _bvhNodesStorageValuesDataView: DataView;
//  let _bvhNodesStorageValuesDataView: DataView;
    let _bvhNodesStorageValues: ArrayBuffer;
//  let _bvhNodesStorageValues: ArrayBuffer;
    let _bvhNodesStorageBuffer: GPUBuffer;
//  let _bvhNodesStorageBuffer: GPUBuffer;
    let _bvhNodes: BVHNode[] = [];
//  let _bvhNodes: BVHNode[] = [];
    let _computeBindGroup0: GPUBindGroup;
//  let _computeBindGroup0: GPUBindGroup;
    let _gatherBindGroup0: GPUBindGroup;
//  let _gatherBindGroup0: GPUBindGroup;
    let _blur2XBindGroup0: GPUBindGroup;
//  let _blur2XBindGroup0: GPUBindGroup;
    let _blur4XBindGroup0: GPUBindGroup;
//  let _blur4XBindGroup0: GPUBindGroup;
    let _blur8XBindGroup0: GPUBindGroup;
//  let _blur8XBindGroup0: GPUBindGroup;
    let _renderBindGroup0: GPUBindGroup;
//  let _renderBindGroup0: GPUBindGroup;
    let _outputStorage: GPUBuffer;
//  let _outputStorage: GPUBuffer;
    let _accumulatedOutputStorage: GPUBuffer;
//  let _accumulatedOutputStorage: GPUBuffer;
    let _outputTexture: GPUTexture;
//  let _outputTexture: GPUTexture;
    let _outputSampler: GPUSampler;
//  let _outputSampler: GPUSampler;
    let _blurTexturePing: GPUTexture;
//  let _blurTexturePing: GPUTexture;
    let _blurTexturePong: GPUTexture;
//  let _blurTexturePong: GPUTexture;
    let _blurSampler: GPUSampler;
//  let _blurSampler: GPUSampler;
    let _samplesPerPixel: number = $state(10000.0);
//  let _samplesPerPixel: number = $state(10000.0);
    let _pixelSamplesScale: number = $derived(1.0 / _samplesPerPixel);
//  let _pixelSamplesScale: number = $derived(1.0 / _samplesPerPixel);
    let _stratifiedSamplesPerPixel: number = $derived(Math.sqrt(_samplesPerPixel));
//  let _stratifiedSamplesPerPixel: number = $derived(Math.sqrt(_samplesPerPixel));
    let _inverseStratifiedSamplesPerPixel: number = $derived(1.0 / _stratifiedSamplesPerPixel);
//  let _inverseStratifiedSamplesPerPixel: number = $derived(1.0 / _stratifiedSamplesPerPixel);
    let _stratifiedSampleX: number;
//  let _stratifiedSampleX: number;
    let _stratifiedSampleY: number;
//  let _stratifiedSampleY: number;
    let _lookFrom: Vec3 = $state([-28.284, 0.0, -28.284]);
//  let _lookFrom: Vec3 = $state([-28.284, 0.0, -28.284]);
    let _lookAt: Vec3 = $state([0.0, 0.0, 0.0]);
//  let _lookAt: Vec3 = $state([0.0, 0.0, 0.0]);
    let _viewUp: Vec3 = $state([0.0, 1.0, 0.0]);
//  let _viewUp: Vec3 = $state([0.0, 1.0, 0.0]);
    let _lookAtSubtractLookFrom: Vec3 = $derived(m.subtract(_lookAt, _lookFrom));
//  let _lookAtSubtractLookFrom: Vec3 = $derived(m.subtract(_lookAt, _lookFrom));
    let _lengthLookAtSubtractLookFrom: number = $derived(m.norm(_lookAtSubtractLookFrom) as number);
//  let _lengthLookAtSubtractLookFrom: number = $derived(m.norm(_lookAtSubtractLookFrom) as number);
    let _lookFromSubtractLookAt: Vec3 = $derived(m.subtract(_lookFrom, _lookAt));
//  let _lookFromSubtractLookAt: Vec3 = $derived(m.subtract(_lookFrom, _lookAt));
    let _vFOV: number = $state(m.pi / 2.5);
//  let _vFOV: number = $state(m.pi / 2.5);
    let _hFOV: number = $state(m.pi / 2.5);
//  let _hFOV: number = $state(m.pi / 2.5);
    let _h: number = $derived(m.tan(_vFOV / 2.0));
//  let _h: number = $derived(m.tan(_vFOV / 2.0));
    let _w: number = $derived(m.tan(_hFOV / 2.0));
//  let _w: number = $derived(m.tan(_hFOV / 2.0));
    let _focalLength: number = $derived(_lengthLookAtSubtractLookFrom);
//  let _focalLength: number = $derived(_lengthLookAtSubtractLookFrom);
    let _cameraW: Vec3 = $derived(m.divide(_lookFromSubtractLookAt, m.norm(_lookFromSubtractLookAt)) as Vec3);
//  let _cameraW: Vec3 = $derived(m.divide(_lookFromSubtractLookAt, m.norm(_lookFromSubtractLookAt)) as Vec3);
    let _viewUpCrossCameraW: Vec3 = $derived(m.cross(_viewUp, _cameraW) as Vec3);
//  let _viewUpCrossCameraW: Vec3 = $derived(m.cross(_viewUp, _cameraW) as Vec3);
    let _cameraU: Vec3 = $derived(m.divide(_viewUpCrossCameraW, m.norm(_viewUpCrossCameraW)) as Vec3);
//  let _cameraU: Vec3 = $derived(m.divide(_viewUpCrossCameraW, m.norm(_viewUpCrossCameraW)) as Vec3);
    let _cameraV: Vec3 = $derived(m.cross(_cameraW, _cameraU) as Vec3);
//  let _cameraV: Vec3 = $derived(m.cross(_cameraW, _cameraU) as Vec3);
    let _defocusAngle: number = $state(0.0 * m.pi);
//  let _defocusAngle: number = $state(0.0 * m.pi);
    let _focusDistance: number = $derived(_lengthLookAtSubtractLookFrom);
//  let _focusDistance: number = $derived(_lengthLookAtSubtractLookFrom);
    let _defocusRadius: number = $derived(_focusDistance * Math.tan(_defocusAngle / 2.0));
//  let _defocusRadius: number = $derived(_focusDistance * Math.tan(_defocusAngle / 2.0));
    let _defocusDiskRadiusU: Vec3 = $derived(m.multiply(_cameraU, _defocusRadius)) as Vec3;
//  let _defocusDiskRadiusU: Vec3 = $derived(m.multiply(_cameraU, _defocusRadius)) as Vec3;
    let _defocusDiskRadiusV: Vec3 = $derived(m.multiply(_cameraV, _defocusRadius)) as Vec3;
//  let _defocusDiskRadiusV: Vec3 = $derived(m.multiply(_cameraV, _defocusRadius)) as Vec3;
    let _viewportH: number = $derived(2.0 * _h * _focalLength);
//  let _viewportH: number = $derived(2.0 * _h * _focalLength);
    let _viewportW: number = $state(1.0);
//  let _viewportW: number = $state(1.0);
    let _cameraCenter: Vec3 = $derived(_lookFrom);
//  let _cameraCenter: Vec3 = $derived(_lookFrom);
    let _viewportU: Vec3 = $derived(m.multiply( _viewportW, _cameraU) as Vec3);
//  let _viewportU: Vec3 = $derived(m.multiply( _viewportW, _cameraU) as Vec3);
    let _viewportV: Vec3 = $derived(m.multiply(-_viewportH, _cameraV) as Vec3);
//  let _viewportV: Vec3 = $derived(m.multiply(-_viewportH, _cameraV) as Vec3);
    let _fromPixelToPixelDeltaU: Vec3 = $state([1.0, 1.0, 1.0]);
//  let _fromPixelToPixelDeltaU: Vec3 = $state([1.0, 1.0, 1.0]);
    let _fromPixelToPixelDeltaV: Vec3 = $state([1.0, 1.0, 1.0]);
//  let _fromPixelToPixelDeltaV: Vec3 = $state([1.0, 1.0, 1.0]);
    let _viewportTL: Vec3 = $derived(m.chain(_cameraCenter).subtract(m.multiply(_cameraW, _focalLength)).subtract(m.divide(_viewportU, 2)).subtract(m.divide(_viewportV, 2)).done() as Vec3);
//  let _viewportTL: Vec3 = $derived(m.chain(_cameraCenter).subtract(m.multiply(_cameraW, _focalLength)).subtract(m.divide(_viewportU, 2)).subtract(m.divide(_viewportV, 2)).done() as Vec3);
    let _pixel00Coordinates: Vec3 = $derived(m.chain(_viewportTL).add(m.multiply(0.5, m.add(_fromPixelToPixelDeltaU, _fromPixelToPixelDeltaV))).done() as Vec3);
//  let _pixel00Coordinates: Vec3 = $derived(m.chain(_viewportTL).add(m.multiply(0.5, m.add(_fromPixelToPixelDeltaU, _fromPixelToPixelDeltaV))).done() as Vec3);
    let _backgroundType: BackgroundType = $state(type.BackgroundType.SKY_BOX_HDRI);
//  let _backgroundType: BackgroundType = $state(type.BackgroundType.SKY_BOX_HDRI);
    let _numberOfImages: number;
//  let _numberOfImages: number;
    let _frameHandleRenderLoop: number;
//  let _frameHandleRenderLoop: number;
    let _tweenCameraLookFrom: t.Tween<Vec3> = null!;
//  let _tweenCameraLookFrom: t.Tween<Vec3> = null!;
    let _tweenCameraLookAt: t.Tween<Vec3> = null!;
//  let _tweenCameraLookAt: t.Tween<Vec3> = null!;
    let _tweenCameraViewUp: t.Tween<Vec3> = null!;
//  let _tweenCameraViewUp: t.Tween<Vec3> = null!;
    let _tweenCameraLookFromIsBusy: boolean = false;
//  let _tweenCameraLookFromIsBusy: boolean = false;
    let _tweenCameraLookAtIsBusy: boolean = false;
//  let _tweenCameraLookAtIsBusy: boolean = false;
    let _tweenCameraViewUpIsBusy: boolean = false;
//  let _tweenCameraViewUpIsBusy: boolean = false;
    let _cpuTimeMeasurementRenderLoop: CPUTimeMeasurement = $state({
//  let _cpuTimeMeasurementRenderLoop: CPUTimeMeasurement = $state({
        thenInSeconds: 0,
//      thenInSeconds: 0,
        nowInSeconds: 0,
//      nowInSeconds: 0,
        deltaTimeInSeconds: 0,
//      deltaTimeInSeconds: 0,
        startInMilliseconds: 0,
//      startInMilliseconds: 0,
        ceaseInMilliseconds: 0,
//      ceaseInMilliseconds: 0,
        fps: "",
//      fps: "",
        eachCycleExecutionInMilliseconds: "",
//      eachCycleExecutionInMilliseconds: "",
    });
//  });
    let _cpuTimeMeasurementGeneralLoop: CPUTimeMeasurement = $state({
//  let _cpuTimeMeasurementGeneralLoop: CPUTimeMeasurement = $state({
        thenInSeconds: 0,
//      thenInSeconds: 0,
        nowInSeconds: 0,
//      nowInSeconds: 0,
        deltaTimeInSeconds: 0,
//      deltaTimeInSeconds: 0,
        startInMilliseconds: 0,
//      startInMilliseconds: 0,
        ceaseInMilliseconds: 0,
//      ceaseInMilliseconds: 0,
        fps: "",
//      fps: "",
        eachCycleExecutionInMilliseconds: "",
//      eachCycleExecutionInMilliseconds: "",
    });
//  });



//  $inspect(_viewportU, _viewportTL, _pixel00Coordinates, );
//  $inspect(_viewportU, _viewportTL, _pixel00Coordinates, );



    function startCPUTimeMeasurement(cpuTimeMeasurement: CPUTimeMeasurement, cpuTimeInMilliseconds: number): void {
//  function startCPUTimeMeasurement(cpuTimeMeasurement: CPUTimeMeasurement, cpuTimeInMilliseconds: number): void {
        cpuTimeMeasurement.nowInSeconds = cpuTimeInMilliseconds * 0.001; // convert to seconds
//      cpuTimeMeasurement.nowInSeconds = cpuTimeInMilliseconds * 0.001; // convert to seconds
        cpuTimeMeasurement.deltaTimeInSeconds = cpuTimeMeasurement.nowInSeconds - cpuTimeMeasurement.thenInSeconds;
//      cpuTimeMeasurement.deltaTimeInSeconds = cpuTimeMeasurement.nowInSeconds - cpuTimeMeasurement.thenInSeconds;
        cpuTimeMeasurement.thenInSeconds = cpuTimeMeasurement.nowInSeconds;
//      cpuTimeMeasurement.thenInSeconds = cpuTimeMeasurement.nowInSeconds;
        cpuTimeMeasurement.startInMilliseconds = performance.now();
//      cpuTimeMeasurement.startInMilliseconds = performance.now();
    };
//  };
    function ceaseCPUTimeMeasurement(cpuTimeMeasurement: CPUTimeMeasurement): void {
//  function ceaseCPUTimeMeasurement(cpuTimeMeasurement: CPUTimeMeasurement): void {
        cpuTimeMeasurement.ceaseInMilliseconds = performance.now();
//      cpuTimeMeasurement.ceaseInMilliseconds = performance.now();
        cpuTimeMeasurement.fps = `${(1 / cpuTimeMeasurement.deltaTimeInSeconds).toFixed(1)}fps`;
//      cpuTimeMeasurement.fps = `${(1 / cpuTimeMeasurement.deltaTimeInSeconds).toFixed(1)}fps`;
        cpuTimeMeasurement.eachCycleExecutionInMilliseconds = `${(cpuTimeMeasurement.ceaseInMilliseconds - cpuTimeMeasurement.startInMilliseconds).toFixed(1)}ms`;
//      cpuTimeMeasurement.eachCycleExecutionInMilliseconds = `${(cpuTimeMeasurement.ceaseInMilliseconds - cpuTimeMeasurement.startInMilliseconds).toFixed(1)}ms`;
    };
//  };



    function calculateAABB3D(triangle: Triangle): AABB3D {
//  function calculateAABB3D(triangle: Triangle): AABB3D {
        const padding: number = 1.0e-4;
//      const padding: number = 1.0e-4;
        const aabb3d: AABB3D = {
//      const aabb3d: AABB3D = {
            minCornersLimit: [ 0.0, 0.0, 0.0, ],
//          minCornersLimit: [ 0.0, 0.0, 0.0, ],
            maxCornersLimit: [ 0.0, 0.0, 0.0, ],
//          maxCornersLimit: [ 0.0, 0.0, 0.0, ],
        };
//      };
        aabb3d.minCornersLimit[0] = Math.min(triangle.vertex0[0], Math.min(triangle.vertex1[0], triangle.vertex2[0])) - padding;
//      aabb3d.minCornersLimit[0] = Math.min(triangle.vertex0[0], Math.min(triangle.vertex1[0], triangle.vertex2[0])) - padding;
        aabb3d.minCornersLimit[1] = Math.min(triangle.vertex0[1], Math.min(triangle.vertex1[1], triangle.vertex2[1])) - padding;
//      aabb3d.minCornersLimit[1] = Math.min(triangle.vertex0[1], Math.min(triangle.vertex1[1], triangle.vertex2[1])) - padding;
        aabb3d.minCornersLimit[2] = Math.min(triangle.vertex0[2], Math.min(triangle.vertex1[2], triangle.vertex2[2])) - padding;
//      aabb3d.minCornersLimit[2] = Math.min(triangle.vertex0[2], Math.min(triangle.vertex1[2], triangle.vertex2[2])) - padding;
        aabb3d.maxCornersLimit[0] = Math.max(triangle.vertex0[0], Math.max(triangle.vertex1[0], triangle.vertex2[0])) + padding;
//      aabb3d.maxCornersLimit[0] = Math.max(triangle.vertex0[0], Math.max(triangle.vertex1[0], triangle.vertex2[0])) + padding;
        aabb3d.maxCornersLimit[1] = Math.max(triangle.vertex0[1], Math.max(triangle.vertex1[1], triangle.vertex2[1])) + padding;
//      aabb3d.maxCornersLimit[1] = Math.max(triangle.vertex0[1], Math.max(triangle.vertex1[1], triangle.vertex2[1])) + padding;
        aabb3d.maxCornersLimit[2] = Math.max(triangle.vertex0[2], Math.max(triangle.vertex1[2], triangle.vertex2[2])) + padding;
//      aabb3d.maxCornersLimit[2] = Math.max(triangle.vertex0[2], Math.max(triangle.vertex1[2], triangle.vertex2[2])) + padding;
        return aabb3d;
//      return aabb3d;
    };
//  };
    // Helper: expands a 10-bit integer into 30 bits by inserting 2 zeros after each bit.
    // Helper: expands a 10-bit integer into 30 bits by inserting 2 zeros after each bit.
    // Used for Morton code generation.
    // Used for Morton code generation.
    function expandBits(v: number): number {
//  function expandBits(v: number): number {
        v = (v * 0x00010001) & 0xFF0000FF;
//      v = (v * 0x00010001) & 0xFF0000FF;
        v = (v * 0x00000101) & 0x0F00F00F;
//      v = (v * 0x00000101) & 0x0F00F00F;
        v = (v * 0x00000011) & 0xC30C30C3;
//      v = (v * 0x00000011) & 0xC30C30C3;
        v = (v * 0x00000005) & 0x49249249;
//      v = (v * 0x00000005) & 0x49249249;
        return v;
//      return v;
    };
//  };
    // Calculates a 30-bit Morton code for a 3D point within a unit box [0,1].
    // Calculates a 30-bit Morton code for a 3D point within a unit box [0,1].
    function morton3D(x: number, y: number, z: number): number {
//  function morton3D(x: number, y: number, z: number): number {
        x = Math.min(Math.max(x * 1024.0, 0.0), 1023.0);
//      x = Math.min(Math.max(x * 1024.0, 0.0), 1023.0);
        y = Math.min(Math.max(y * 1024.0, 0.0), 1023.0);
//      y = Math.min(Math.max(y * 1024.0, 0.0), 1023.0);
        z = Math.min(Math.max(z * 1024.0, 0.0), 1023.0);
//      z = Math.min(Math.max(z * 1024.0, 0.0), 1023.0);
        return expandBits(Math.floor(x)) | (expandBits(Math.floor(y)) << 1) | (expandBits(Math.floor(z)) << 2);
//      return expandBits(Math.floor(x)) | (expandBits(Math.floor(y)) << 1) | (expandBits(Math.floor(z)) << 2);
    };
//  };
    // Unions two AABB3Ds.
    // Unions two AABB3Ds.
    function unionAABB(a: AABB3D, b: AABB3D): AABB3D {
//  function unionAABB(a: AABB3D, b: AABB3D): AABB3D {
        return {
//      return {
            minCornersLimit: [
//          minCornersLimit: [
                Math.min(a.minCornersLimit[0], b.minCornersLimit[0]),
//              Math.min(a.minCornersLimit[0], b.minCornersLimit[0]),
                Math.min(a.minCornersLimit[1], b.minCornersLimit[1]),
//              Math.min(a.minCornersLimit[1], b.minCornersLimit[1]),
                Math.min(a.minCornersLimit[2], b.minCornersLimit[2]),
//              Math.min(a.minCornersLimit[2], b.minCornersLimit[2]),
            ],
//          ],
            maxCornersLimit: [
//          maxCornersLimit: [
                Math.max(a.maxCornersLimit[0], b.maxCornersLimit[0]),
//              Math.max(a.maxCornersLimit[0], b.maxCornersLimit[0]),
                Math.max(a.maxCornersLimit[1], b.maxCornersLimit[1]),
//              Math.max(a.maxCornersLimit[1], b.maxCornersLimit[1]),
                Math.max(a.maxCornersLimit[2], b.maxCornersLimit[2]),
//              Math.max(a.maxCornersLimit[2], b.maxCornersLimit[2]),
            ],
//          ],
        };
//      };
    };
//  };
    // Internal structure to hold sorting data
    // Internal structure to hold sorting data
    type PrimitiveInfo = {
        index: number; // Original index in _triangles
//      index: number; // Original index in _triangles
        code: number; // Morton code
//      code: number; // Morton code
        centroid: Vec3; // Centroid for verification
//      centroid: Vec3; // Centroid for verification
        aabb: AABB3D; // Cached AABB
//      aabb: AABB3D; // Cached AABB
    };
//  };
    function buildLBVH(triangles: Triangle[]): BVHNode[] {
//  function buildLBVH(triangles: Triangle[]): BVHNode[] {
        if (triangles.length === 0) {
//      if (triangles.length === 0) {
            return [];
//          return [];
        }
//      }
        // 1. Calculate Global Bounds and Triangle Centroids
        // 1. Calculate Global Bounds and Triangle Centroids
        let globalMin: Vec3 = [ +Infinity, +Infinity, +Infinity ];
//      let globalMin: Vec3 = [ +Infinity, +Infinity, +Infinity ];
        let globalMax: Vec3 = [ -Infinity, -Infinity, -Infinity ];
//      let globalMax: Vec3 = [ -Infinity, -Infinity, -Infinity ];
        const primitives: PrimitiveInfo[] = triangles.map(
//      const primitives: PrimitiveInfo[] = triangles.map(
            (triangle: Triangle, triangleIndex: number): PrimitiveInfo => {
//          (triangle: Triangle, triangleIndex: number): PrimitiveInfo => {
                // Centroid = (min + max) * 0.5
                // Centroid = (min + max) * 0.5
                const cx: number = (triangle.aabb3d.minCornersLimit[0] + triangle.aabb3d.maxCornersLimit[0]) * 0.5;
//              const cx: number = (triangle.aabb3d.minCornersLimit[0] + triangle.aabb3d.maxCornersLimit[0]) * 0.5;
                const cy: number = (triangle.aabb3d.minCornersLimit[1] + triangle.aabb3d.maxCornersLimit[1]) * 0.5;
//              const cy: number = (triangle.aabb3d.minCornersLimit[1] + triangle.aabb3d.maxCornersLimit[1]) * 0.5;
                const cz: number = (triangle.aabb3d.minCornersLimit[2] + triangle.aabb3d.maxCornersLimit[2]) * 0.5;
//              const cz: number = (triangle.aabb3d.minCornersLimit[2] + triangle.aabb3d.maxCornersLimit[2]) * 0.5;
                // Update global bounds
                // Update global bounds
                globalMin[0] = Math.min(globalMin[0], cx);
//              globalMin[0] = Math.min(globalMin[0], cx);
                globalMin[1] = Math.min(globalMin[1], cy);
//              globalMin[1] = Math.min(globalMin[1], cy);
                globalMin[2] = Math.min(globalMin[2], cz);
//              globalMin[2] = Math.min(globalMin[2], cz);
                globalMax[0] = Math.max(globalMax[0], cx);
//              globalMax[0] = Math.max(globalMax[0], cx);
                globalMax[1] = Math.max(globalMax[1], cy);
//              globalMax[1] = Math.max(globalMax[1], cy);
                globalMax[2] = Math.max(globalMax[2], cz);
//              globalMax[2] = Math.max(globalMax[2], cz);
                return {
//              return {
                    index: triangleIndex,
//                  index: triangleIndex,
                    code: 0,
//                  code: 0,
                    centroid: [ cx, cy, cz ],
//                  centroid: [ cx, cy, cz ],
                    aabb: triangle.aabb3d,
//                  aabb: triangle.aabb3d,
                };
//              };
            }
//          }
        );
//      );
        // Prevent divide by zero if scene is flat or single point
        // Prevent divide by zero if scene is flat or single point
        const extent: Vec3 = [
//      const extent: Vec3 = [
            Math.max(globalMax[0] - globalMin[0], 1e-6),
//          Math.max(globalMax[0] - globalMin[0], 1e-6),
            Math.max(globalMax[1] - globalMin[1], 1e-6),
//          Math.max(globalMax[1] - globalMin[1], 1e-6),
            Math.max(globalMax[2] - globalMin[2], 1e-6),
//          Math.max(globalMax[2] - globalMin[2], 1e-6),
        ];
//      ];
        // 2. Calculate Morton Codes
        // 2. Calculate Morton Codes
        for (let i = 0; i < primitives.length; i++) {
//      for (let i = 0; i < primitives.length; i++) {
            const p: PrimitiveInfo = primitives[i];
//          const p: PrimitiveInfo = primitives[i];
            const nx: number = (p.centroid[0] - globalMin[0]) / extent[0];
//          const nx: number = (p.centroid[0] - globalMin[0]) / extent[0];
            const ny: number = (p.centroid[1] - globalMin[1]) / extent[1];
//          const ny: number = (p.centroid[1] - globalMin[1]) / extent[1];
            const nz: number = (p.centroid[2] - globalMin[2]) / extent[2];
//          const nz: number = (p.centroid[2] - globalMin[2]) / extent[2];
            p.code = morton3D(nx, ny, nz);
//          p.code = morton3D(nx, ny, nz);
        }
//      }
        // 3. Sort Primitives by Morton Code
        // 3. Sort Primitives by Morton Code
        primitives.sort((a: PrimitiveInfo, b: PrimitiveInfo): number => a.code - b.code);
//      primitives.sort((a: PrimitiveInfo, b: PrimitiveInfo): number => a.code - b.code);
        // 4. Build Hierarchy
        // 4. Build Hierarchy
        const bvhNodes: BVHNode[] = [];
//      const bvhNodes: BVHNode[] = [];
        // Recursive builder function
        // Recursive builder function
        // Returns the index of the node created in bvhNodes array
        // Returns the index of the node created in bvhNodes array
        function buildRecursive(first: number, last: number): number {
//      function buildRecursive(first: number, last: number): number {
            // Create a new node placeholder
            // Create a new node placeholder
            const nodeIndex: number = bvhNodes.length;
//          const nodeIndex: number = bvhNodes.length;
            bvhNodes.push({ // Temporary init, will be overwritten
//          bvhNodes.push({ // Temporary init, will be overwritten
                aabb3d: {
//              aabb3d: {
                    minCornersLimit: [ 0, 0, 0 ],
//                  minCornersLimit: [ 0, 0, 0 ],
                    maxCornersLimit: [ 0, 0, 0 ],
//                  maxCornersLimit: [ 0, 0, 0 ],
                },
//              },
                triangleIndex: -1,
//              triangleIndex: -1,
                childIndexL: -1,
//              childIndexL: -1,
                childIndexR: -1,
//              childIndexR: -1,
            });
//          });
            // CASE: Leaf Node
            // CASE: Leaf Node
            if (first === last) {
//          if (first === last) {
                const p: PrimitiveInfo = primitives[first];
//              const p: PrimitiveInfo = primitives[first];
                bvhNodes[nodeIndex].aabb3d = p.aabb;
//              bvhNodes[nodeIndex].aabb3d = p.aabb;
                bvhNodes[nodeIndex].triangleIndex = p.index;
//              bvhNodes[nodeIndex].triangleIndex = p.index;
                return nodeIndex;
//              return nodeIndex;
            }
//          }
            // CASE: Internal Node
            // CASE: Internal Node
            // Determine split position
            // Determine split position
            const splitIndex: number = findSplit(primitives, first, last);
//          const splitIndex: number = findSplit(primitives, first, last);
            // Process Children
            // Process Children
            const childL: number = buildRecursive(first, splitIndex);
//          const childL: number = buildRecursive(first, splitIndex);
            const childR: number = buildRecursive(splitIndex + 1, last);
//          const childR: number = buildRecursive(splitIndex + 1, last);
            // Link children
            // Link children
            bvhNodes[nodeIndex].childIndexL = childL;
//          bvhNodes[nodeIndex].childIndexL = childL;
            bvhNodes[nodeIndex].childIndexR = childR;
//          bvhNodes[nodeIndex].childIndexR = childR;
            // Refit AABB (Union of children)
            // Refit AABB (Union of children)
            bvhNodes[nodeIndex].aabb3d = unionAABB(
//          bvhNodes[nodeIndex].aabb3d = unionAABB(
                bvhNodes[childL].aabb3d,
//              bvhNodes[childL].aabb3d,
                bvhNodes[childR].aabb3d,
//              bvhNodes[childR].aabb3d,
            );
//          );
            return nodeIndex;
//          return nodeIndex;
        };
//      };
        // Start the build
        // Start the build
        buildRecursive(0, primitives.length - 1);
//      buildRecursive(0, primitives.length - 1);
        return bvhNodes;
//      return bvhNodes;
    };
//  };
    // Finds the split index for the range [first, last] using Morton codes.
    // Finds the split index for the range [first, last] using Morton codes.
    // It finds the highest bit that differs between the first and last keys,
    // It finds the highest bit that differs between the first and last keys,
    // and finds the position where that bit flips.
    // and finds the position where that bit flips.
    function findSplit(primitives: PrimitiveInfo[], first: number, last: number): number {
//  function findSplit(primitives: PrimitiveInfo[], first: number, last: number): number {
        const firstCode: number = primitives[first].code;
//      const firstCode: number = primitives[first].code;
        const lastCode: number = primitives[last].code;
//      const lastCode: number = primitives[last].code;
        // If all primitives in this range have the same Morton code,
        // If all primitives in this range have the same Morton code,
        // we cannot split by bit. Split exactly in the middle.
        // we cannot split by bit. Split exactly in the middle.
        if (firstCode === lastCode) {
//      if (firstCode === lastCode) {
            return (first + last) >> 1;
//          return (first + last) >> 1;
        }
//      }
        // Find the highest differing bit between the first and last code
        // Find the highest differing bit between the first and last code
        // XOR results in 1s where bits differ.
        // XOR results in 1s where bits differ.
        // clz32 (Count Leading Zeros) gives us the most significant differing bit.
        // clz32 (Count Leading Zeros) gives us the most significant differing bit.
        const commonPrefix: number = Math.clz32(firstCode ^ lastCode);
//      const commonPrefix: number = Math.clz32(firstCode ^ lastCode);
        // Use binary search to find the index where the bit at 'commonPrefix' flips
        // Use binary search to find the index where the bit at 'commonPrefix' flips
        let split: number = first; // Initialize split
//      let split: number = first; // Initialize split
        let step: number = last - first;
//      let step: number = last - first;
        // Standard binary search to find the boundary
        // Standard binary search to find the boundary
        // We look for the largest index 'i' such that primitives[i] shares the
        // We look for the largest index 'i' such that primitives[i] shares the
        // same bit at 'commonPrefix' as primitives[first].
        // same bit at 'commonPrefix' as primitives[first].
        do {
//      do {
            step = (step + 1) >> 1;
//          step = (step + 1) >> 1;
            const newSplit: number = split + step;
//          const newSplit: number = split + step;
            if (newSplit < last) {
//          if (newSplit < last) {
                const splitCode: number = primitives[newSplit].code;
//              const splitCode: number = primitives[newSplit].code;
                // Compare the most significant differing bit
                // Compare the most significant differing bit
                const splitPrefix: number = Math.clz32(firstCode ^ splitCode);
//              const splitPrefix: number = Math.clz32(firstCode ^ splitCode);
                // If the prefix is larger (more leading zeros), it means the differing bit
                // If the prefix is larger (more leading zeros), it means the differing bit
                // hasn't been encountered yet, so this code is on the "Left" side.
                // hasn't been encountered yet, so this code is on the "Left" side.
                if (splitPrefix > commonPrefix) {
//              if (splitPrefix > commonPrefix) {
                    split = newSplit;
//                  split = newSplit;
                }
//              }
            }
//          }
        } while (step > 1);
//      } while (step > 1);
        return split;
//      return split;
    };
//  };


    const prepareSpheres = (): void => {
//  const prepareSpheres = (): void => {
        if (!_spheresStorageValues || _spheres.length !== _spheresStorageValues.byteLength / 32) {
//      if (!_spheresStorageValues || _spheres.length !== _spheresStorageValues.byteLength / 32) {
            _spheresStorageValues = new ArrayBuffer(_spheres.length * 32);
//          _spheresStorageValues = new ArrayBuffer(_spheres.length * 32);
            _spheresStorageValuesDataView = new DataView(_spheresStorageValues);
//          _spheresStorageValuesDataView = new DataView(_spheresStorageValues);
            if (_spheresStorageBuffer) {
//          if (_spheresStorageBuffer) {
                _spheresStorageBuffer.destroy();
//              _spheresStorageBuffer.destroy();
            }
//          }
            _spheresStorageBuffer = _device.createBuffer({
//          _spheresStorageBuffer = _device.createBuffer({
                label: "GPU_STORAGE_BUFFER_SPHERES",
//              label: "GPU_STORAGE_BUFFER_SPHERES",
                size: _spheresStorageValues.byteLength,
//              size: _spheresStorageValues.byteLength,
                usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
//              usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
            });
//          });
        }
//      }
        _spheres.forEach((
//      _spheres.forEach((
            {
                center,
//              center,
                radius,
//              radius,
                materialIndex,
//              materialIndex,
            }
            : Sphere, sphereIndex: number): void => {
//          : Sphere, sphereIndex: number): void => {
                const base: number = sphereIndex * 32;
//              const base: number = sphereIndex * 32;
                _spheresStorageValuesDataView.setFloat32(base + 0, center[0], true);
//              _spheresStorageValuesDataView.setFloat32(base + 0, center[0], true);
                _spheresStorageValuesDataView.setFloat32(base + 4, center[1], true);
//              _spheresStorageValuesDataView.setFloat32(base + 4, center[1], true);
                _spheresStorageValuesDataView.setFloat32(base + 8, center[2], true);
//              _spheresStorageValuesDataView.setFloat32(base + 8, center[2], true);
                _spheresStorageValuesDataView.setFloat32(base + 12, radius, true);
//              _spheresStorageValuesDataView.setFloat32(base + 12, radius, true);
                _spheresStorageValuesDataView.setUint32(base + 16, materialIndex, true);
//              _spheresStorageValuesDataView.setUint32(base + 16, materialIndex, true);
        });
//      });
        _device.queue.writeBuffer(_spheresStorageBuffer, 0, _spheresStorageValues as GPUAllowSharedBufferSource);
//      _device.queue.writeBuffer(_spheresStorageBuffer, 0, _spheresStorageValues as GPUAllowSharedBufferSource);
    };
//  };
    const prepareMaterials = (): void => {
//  const prepareMaterials = (): void => {
        if (!_materialsStorageValues || _materials.length !== _materialsStorageValues.byteLength / 28) {
//      if (!_materialsStorageValues || _materials.length !== _materialsStorageValues.byteLength / 28) {
            _materialsStorageValues = new ArrayBuffer(_materials.length * 28);
//          _materialsStorageValues = new ArrayBuffer(_materials.length * 28);
            _materialsStorageValuesDataView = new DataView(_materialsStorageValues);
//          _materialsStorageValuesDataView = new DataView(_materialsStorageValues);
            if (_materialsStorageBuffer) {
//          if (_materialsStorageBuffer) {
                _materialsStorageBuffer.destroy();
//              _materialsStorageBuffer.destroy();
            }
//          }
            _materialsStorageBuffer = _device.createBuffer({
//          _materialsStorageBuffer = _device.createBuffer({
                label: "GPU_STORAGE_BUFFER_MATERIALS",
//              label: "GPU_STORAGE_BUFFER_MATERIALS",
                size: _materialsStorageValues.byteLength,
//              size: _materialsStorageValues.byteLength,
                usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
//              usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
            });
//          });
        }
//      }
        _materials.forEach((
//      _materials.forEach((
            {
                layer1Roughness,
//              layer1Roughness,
                layer1Metallic,
//              layer1Metallic,
                layer1Transmission,
//              layer1Transmission,
                layer0IOR,
//              layer0IOR,
                layer1IOR,
//              layer1IOR,
                textureIndex,
//              textureIndex,
                materialType,
//              materialType,
            }
            : Material, materialIndex: number): void => {
//          : Material, materialIndex: number): void => {
                const base: number = materialIndex * 28;
//              const base: number = materialIndex * 28;
                _materialsStorageValuesDataView.setFloat32(base + 0, layer1Roughness, true);
//              _materialsStorageValuesDataView.setFloat32(base + 0, layer1Roughness, true);
                _materialsStorageValuesDataView.setFloat32(base + 4, layer1Metallic, true);
//              _materialsStorageValuesDataView.setFloat32(base + 4, layer1Metallic, true);
                _materialsStorageValuesDataView.setFloat32(base + 8, layer1Transmission, true);
//              _materialsStorageValuesDataView.setFloat32(base + 8, layer1Transmission, true);
                _materialsStorageValuesDataView.setFloat32(base + 12, layer0IOR, true);
//              _materialsStorageValuesDataView.setFloat32(base + 12, layer0IOR, true);
                _materialsStorageValuesDataView.setFloat32(base + 16, layer1IOR, true);
//              _materialsStorageValuesDataView.setFloat32(base + 16, layer1IOR, true);
                _materialsStorageValuesDataView.setUint32(base + 20, textureIndex, true);
//              _materialsStorageValuesDataView.setUint32(base + 20, textureIndex, true);
                _materialsStorageValuesDataView.setUint32(base + 24, materialType, true);
//              _materialsStorageValuesDataView.setUint32(base + 24, materialType, true);
        });
//      });
        _device.queue.writeBuffer(_materialsStorageBuffer, 0, _materialsStorageValues as GPUAllowSharedBufferSource);
//      _device.queue.writeBuffer(_materialsStorageBuffer, 0, _materialsStorageValues as GPUAllowSharedBufferSource);
    };
//  };
    const prepareTextures = (): void => {
//  const prepareTextures = (): void => {
        if (!_texturesStorageValues || _textures.length !== _texturesStorageValues.byteLength / 32) {
//      if (!_texturesStorageValues || _textures.length !== _texturesStorageValues.byteLength / 32) {
            _texturesStorageValues = new ArrayBuffer(_textures.length * 32);
//          _texturesStorageValues = new ArrayBuffer(_textures.length * 32);
            _texturesStorageValuesDataView = new DataView(_texturesStorageValues);
//          _texturesStorageValuesDataView = new DataView(_texturesStorageValues);
            if (_texturesStorageBuffer) {
//          if (_texturesStorageBuffer) {
                _texturesStorageBuffer.destroy();
//              _texturesStorageBuffer.destroy();
            }
//          }
            _texturesStorageBuffer = _device.createBuffer({
//          _texturesStorageBuffer = _device.createBuffer({
                label: "GPU_STORAGE_BUFFER_TEXTURES",
//              label: "GPU_STORAGE_BUFFER_TEXTURES",
                size: _texturesStorageValues.byteLength,
//              size: _texturesStorageValues.byteLength,
                usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
//              usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
            });
//          });
        }
//      }
        _textures.forEach((
//      _textures.forEach((
            {
                albedo,
//              albedo,
                albedoImageIndex,
//              albedoImageIndex,
                albedoTextureType,
//              albedoTextureType,
                albedoScale,
//              albedoScale,
                albedoOTileTextureIndex,
//              albedoOTileTextureIndex,
                albedoETileTextureIndex,
//              albedoETileTextureIndex,
            }
            : Texture, textureIndex: number): void => {
//          : Texture, textureIndex: number): void => {
                const base: number = textureIndex * 32;
//              const base: number = textureIndex * 32;
                _texturesStorageValuesDataView.setFloat32(base + 0, albedo[0], true);
//              _texturesStorageValuesDataView.setFloat32(base + 0, albedo[0], true);
                _texturesStorageValuesDataView.setFloat32(base + 4, albedo[1], true);
//              _texturesStorageValuesDataView.setFloat32(base + 4, albedo[1], true);
                _texturesStorageValuesDataView.setFloat32(base + 8, albedo[2], true);
//              _texturesStorageValuesDataView.setFloat32(base + 8, albedo[2], true);
                _texturesStorageValuesDataView.setUint32(base + 12, albedoImageIndex, true);
//              _texturesStorageValuesDataView.setUint32(base + 12, albedoImageIndex, true);
                _texturesStorageValuesDataView.setUint32(base + 16, albedoTextureType, true);
//              _texturesStorageValuesDataView.setUint32(base + 16, albedoTextureType, true);
                _texturesStorageValuesDataView.setFloat32(base + 20, albedoScale, true);
//              _texturesStorageValuesDataView.setFloat32(base + 20, albedoScale, true);
                _texturesStorageValuesDataView.setUint32(base + 24, albedoOTileTextureIndex, true);
//              _texturesStorageValuesDataView.setUint32(base + 24, albedoOTileTextureIndex, true);
                _texturesStorageValuesDataView.setUint32(base + 28, albedoETileTextureIndex, true);
//              _texturesStorageValuesDataView.setUint32(base + 28, albedoETileTextureIndex, true);
        });
//      });
        _device.queue.writeBuffer(_texturesStorageBuffer, 0, _texturesStorageValues as GPUAllowSharedBufferSource);
//      _device.queue.writeBuffer(_texturesStorageBuffer, 0, _texturesStorageValues as GPUAllowSharedBufferSource);
    };
//  };
    const prepareTriangles = (): void => {
//  const prepareTriangles = (): void => {
        console.log("triangles count:", _triangles.length);
//      console.log("triangles count:", _triangles.length);
        if (!_trianglesStorageValues || _triangles.length !== _trianglesStorageValues.byteLength / 64) {
//      if (!_trianglesStorageValues || _triangles.length !== _trianglesStorageValues.byteLength / 64) {
            _trianglesStorageValues = new ArrayBuffer(_triangles.length * 64);
//          _trianglesStorageValues = new ArrayBuffer(_triangles.length * 64);
            _trianglesStorageValuesDataView = new DataView(_trianglesStorageValues);
//          _trianglesStorageValuesDataView = new DataView(_trianglesStorageValues);
            if (_trianglesStorageBuffer) {
//          if (_trianglesStorageBuffer) {
                _trianglesStorageBuffer.destroy();
//              _trianglesStorageBuffer.destroy();
            }
//          }
            _trianglesStorageBuffer = _device.createBuffer({
//          _trianglesStorageBuffer = _device.createBuffer({
                label: "GPU_STORAGE_BUFFER_TRIANGLES",
//              label: "GPU_STORAGE_BUFFER_TRIANGLES",
                size: _trianglesStorageValues.byteLength,
//              size: _trianglesStorageValues.byteLength,
                usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
//              usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
            });
//          });
        }
//      }
        _triangles.forEach((
//      _triangles.forEach((
            triangle: Triangle, triangleIndex: number): void => {
//          triangle: Triangle, triangleIndex: number): void => {
                triangle.aabb3d = calculateAABB3D(triangle);
//              triangle.aabb3d = calculateAABB3D(triangle);
                if (!triangle.perVertexFrontFaceNormalAvailable) {
//              if (!triangle.perVertexFrontFaceNormalAvailable) {
                    const vertex1SubtractVertex0: Vec3 = m.subtract(triangle.vertex1, triangle.vertex0);
//                  const vertex1SubtractVertex0: Vec3 = m.subtract(triangle.vertex1, triangle.vertex0);
                    const vertex2SubtractVertex0: Vec3 = m.subtract(triangle.vertex2, triangle.vertex0);
//                  const vertex2SubtractVertex0: Vec3 = m.subtract(triangle.vertex2, triangle.vertex0);
                    const crossVector: Vec3 = m.cross(vertex1SubtractVertex0, vertex2SubtractVertex0) as Vec3;
//                  const crossVector: Vec3 = m.cross(vertex1SubtractVertex0, vertex2SubtractVertex0) as Vec3;
                    const normal: Vec3 = m.divide(crossVector, m.norm(crossVector)) as Vec3;
//                  const normal: Vec3 = m.divide(crossVector, m.norm(crossVector)) as Vec3;
                    triangle.vertex0FrontFaceNormal = [...normal];
//                  triangle.vertex0FrontFaceNormal = [...normal];
                    triangle.vertex1FrontFaceNormal = [...normal];
//                  triangle.vertex1FrontFaceNormal = [...normal];
                    triangle.vertex2FrontFaceNormal = [...normal];
//                  triangle.vertex2FrontFaceNormal = [...normal];
                    triangle.perVertexFrontFaceNormalAvailable = true;
//                  triangle.perVertexFrontFaceNormalAvailable = true;
                }
//              }
                const base: number = triangleIndex * 64;
//              const base: number = triangleIndex * 64;
                _trianglesStorageValuesDataView.setFloat32(base + 0, triangle.vertex0[0], true);
//              _trianglesStorageValuesDataView.setFloat32(base + 0, triangle.vertex0[0], true);
                _trianglesStorageValuesDataView.setFloat32(base + 4, triangle.vertex0[1], true);
//              _trianglesStorageValuesDataView.setFloat32(base + 4, triangle.vertex0[1], true);
                _trianglesStorageValuesDataView.setFloat32(base + 8, triangle.vertex0[2], true);
//              _trianglesStorageValuesDataView.setFloat32(base + 8, triangle.vertex0[2], true);
                _trianglesStorageValuesDataView.setUint32(base + 12, encodeOctahedralNormal(triangle.vertex0FrontFaceNormal), true);
//              _trianglesStorageValuesDataView.setUint32(base + 12, encodeOctahedralNormal(triangle.vertex0FrontFaceNormal), true);
                _trianglesStorageValuesDataView.setFloat32(base + 16, triangle.vertex1[0], true);
//              _trianglesStorageValuesDataView.setFloat32(base + 16, triangle.vertex1[0], true);
                _trianglesStorageValuesDataView.setFloat32(base + 20, triangle.vertex1[1], true);
//              _trianglesStorageValuesDataView.setFloat32(base + 20, triangle.vertex1[1], true);
                _trianglesStorageValuesDataView.setFloat32(base + 24, triangle.vertex1[2], true);
//              _trianglesStorageValuesDataView.setFloat32(base + 24, triangle.vertex1[2], true);
                _trianglesStorageValuesDataView.setUint32(base + 28, encodeOctahedralNormal(triangle.vertex1FrontFaceNormal), true);
//              _trianglesStorageValuesDataView.setUint32(base + 28, encodeOctahedralNormal(triangle.vertex1FrontFaceNormal), true);
                _trianglesStorageValuesDataView.setFloat32(base + 32, triangle.vertex2[0], true);
//              _trianglesStorageValuesDataView.setFloat32(base + 32, triangle.vertex2[0], true);
                _trianglesStorageValuesDataView.setFloat32(base + 36, triangle.vertex2[1], true);
//              _trianglesStorageValuesDataView.setFloat32(base + 36, triangle.vertex2[1], true);
                _trianglesStorageValuesDataView.setFloat32(base + 40, triangle.vertex2[2], true);
//              _trianglesStorageValuesDataView.setFloat32(base + 40, triangle.vertex2[2], true);
                _trianglesStorageValuesDataView.setUint32(base + 44, encodeOctahedralNormal(triangle.vertex2FrontFaceNormal), true);
//              _trianglesStorageValuesDataView.setUint32(base + 44, encodeOctahedralNormal(triangle.vertex2FrontFaceNormal), true);
                _trianglesStorageValuesDataView.setUint32(base + 48, encodeQuantizedUV(triangle.vertex0UV), true);
//              _trianglesStorageValuesDataView.setUint32(base + 48, encodeQuantizedUV(triangle.vertex0UV), true);
                _trianglesStorageValuesDataView.setUint32(base + 52, encodeQuantizedUV(triangle.vertex1UV), true);
//              _trianglesStorageValuesDataView.setUint32(base + 52, encodeQuantizedUV(triangle.vertex1UV), true);
                _trianglesStorageValuesDataView.setUint32(base + 56, encodeQuantizedUV(triangle.vertex2UV), true);
//              _trianglesStorageValuesDataView.setUint32(base + 56, encodeQuantizedUV(triangle.vertex2UV), true);
                _trianglesStorageValuesDataView.setUint32(base + 60, triangle.materialIndex, true);
//              _trianglesStorageValuesDataView.setUint32(base + 60, triangle.materialIndex, true);
        });
//      });
        _device.queue.writeBuffer(_trianglesStorageBuffer, 0, _trianglesStorageValues as GPUAllowSharedBufferSource);
//      _device.queue.writeBuffer(_trianglesStorageBuffer, 0, _trianglesStorageValues as GPUAllowSharedBufferSource);
        _bvhNodes = buildLBVH(_triangles);
//      _bvhNodes = buildLBVH(_triangles);
        if (!_bvhNodesStorageValues || _bvhNodes.length !== _bvhNodesStorageValues.byteLength / 32) {
//      if (!_bvhNodesStorageValues || _bvhNodes.length !== _bvhNodesStorageValues.byteLength / 32) {
            _bvhNodesStorageValues = new ArrayBuffer(_bvhNodes.length * 32);
//          _bvhNodesStorageValues = new ArrayBuffer(_bvhNodes.length * 32);
            _bvhNodesStorageValuesDataView = new DataView(_bvhNodesStorageValues);
//          _bvhNodesStorageValuesDataView = new DataView(_bvhNodesStorageValues);
            if (_bvhNodesStorageBuffer) {
//          if (_bvhNodesStorageBuffer) {
                _bvhNodesStorageBuffer.destroy();
//              _bvhNodesStorageBuffer.destroy();
            }
//          }
            _bvhNodesStorageBuffer = _device.createBuffer({
//          _bvhNodesStorageBuffer = _device.createBuffer({
                label: "GPU_STORAGE_BUFFER_BVH_NODES",
//              label: "GPU_STORAGE_BUFFER_BVH_NODES",
                size: _bvhNodesStorageValues.byteLength,
//              size: _bvhNodesStorageValues.byteLength,
                usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
//              usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
            });
//          });
        }
//      }
        _bvhNodes.forEach(
//      _bvhNodes.forEach(
            (bvhNode: BVHNode, bvhNodeIndex: number): void => {
//          (bvhNode: BVHNode, bvhNodeIndex: number): void => {
                const base: number = bvhNodeIndex * 32;
//              const base: number = bvhNodeIndex * 32;
                _bvhNodesStorageValuesDataView.setFloat32(base + 0 , bvhNode.aabb3d.minCornersLimit[0], true);
//              _bvhNodesStorageValuesDataView.setFloat32(base + 0 , bvhNode.aabb3d.minCornersLimit[0], true);
                _bvhNodesStorageValuesDataView.setFloat32(base + 4 , bvhNode.aabb3d.maxCornersLimit[0], true);
//              _bvhNodesStorageValuesDataView.setFloat32(base + 4 , bvhNode.aabb3d.maxCornersLimit[0], true);
                _bvhNodesStorageValuesDataView.setFloat32(base + 8 , bvhNode.aabb3d.minCornersLimit[1], true);
//              _bvhNodesStorageValuesDataView.setFloat32(base + 8 , bvhNode.aabb3d.minCornersLimit[1], true);
                _bvhNodesStorageValuesDataView.setFloat32(base + 12, bvhNode.aabb3d.maxCornersLimit[1], true);
//              _bvhNodesStorageValuesDataView.setFloat32(base + 12, bvhNode.aabb3d.maxCornersLimit[1], true);
                _bvhNodesStorageValuesDataView.setFloat32(base + 16, bvhNode.aabb3d.minCornersLimit[2], true);
//              _bvhNodesStorageValuesDataView.setFloat32(base + 16, bvhNode.aabb3d.minCornersLimit[2], true);
                _bvhNodesStorageValuesDataView.setFloat32(base + 20, bvhNode.aabb3d.maxCornersLimit[2], true);
//              _bvhNodesStorageValuesDataView.setFloat32(base + 20, bvhNode.aabb3d.maxCornersLimit[2], true);
                if (bvhNode.triangleIndex >= 0) {
//              if (bvhNode.triangleIndex >= 0) {
                    // LEAF: Store ~triangleIndex in data1 (makes it negative).
//                  // LEAF: Store ~triangleIndex in data1 (makes it negative).
                    _bvhNodesStorageValuesDataView.setInt32(base + 24, ~bvhNode.triangleIndex, true);
//                  _bvhNodesStorageValuesDataView.setInt32(base + 24, ~bvhNode.triangleIndex, true);
                    _bvhNodesStorageValuesDataView.setInt32(base + 28, 0, true); // Padding/Unused
//                  _bvhNodesStorageValuesDataView.setInt32(base + 28, 0, true); // Padding/Unused
                } else {
//              } else {
                    // INTERNAL: Store child indices.
//                  // INTERNAL: Store child indices.
                    _bvhNodesStorageValuesDataView.setInt32(base + 24, bvhNode.childIndexL, true);
//                  _bvhNodesStorageValuesDataView.setInt32(base + 24, bvhNode.childIndexL, true);
                    _bvhNodesStorageValuesDataView.setInt32(base + 28, bvhNode.childIndexR, true);
//                  _bvhNodesStorageValuesDataView.setInt32(base + 28, bvhNode.childIndexR, true);
                }
//              }
            }
//          }
        );
//      );
        _device.queue.writeBuffer(_bvhNodesStorageBuffer, 0, _bvhNodesStorageValues as GPUAllowSharedBufferSource);
//      _device.queue.writeBuffer(_bvhNodesStorageBuffer, 0, _bvhNodesStorageValues as GPUAllowSharedBufferSource);
    };
//  };



    const initOnce = async (): Promise<void> => {
//  const initOnce = async (): Promise<void> => {
        _adapter = (await navigator.gpu.requestAdapter())!;
//      _adapter = (await navigator.gpu.requestAdapter())!;
        _device = await _adapter.requestDevice({
//      _device = await _adapter.requestDevice({
            requiredFeatures: [
//          requiredFeatures: [
                "shader-f16",
//              "shader-f16",
                "float32-filterable",
//              "float32-filterable",
            ] as Iterable<GPUFeatureName>,
//          ] as Iterable<GPUFeatureName>,
            requiredLimits: {
//          requiredLimits: {
                maxStorageBuffersPerShaderStage: 10,
//              maxStorageBuffersPerShaderStage: 10,
                maxComputeInvocationsPerWorkgroup: 1024,
//              maxComputeInvocationsPerWorkgroup: 1024,
            },
//          },
        });
//      });
        _canvasContext = _canvas.getContext("webgpu")!;
//      _canvasContext = _canvas.getContext("webgpu")!;
        _presentationFormat = "rgba16float"; // navigator.gpu.getPreferredCanvasFormat();
//      _presentationFormat = "rgba16float"; // navigator.gpu.getPreferredCanvasFormat();
        _canvasContext.configure({ alphaMode: "opaque", colorSpace: "display-p3", device: _device, format: _presentationFormat, toneMapping: { mode: "extended", }, usage: GPUTextureUsage.RENDER_ATTACHMENT, viewFormats: [ "rgba16float", ], });
//      _canvasContext.configure({ alphaMode: "opaque", colorSpace: "display-p3", device: _device, format: _presentationFormat, toneMapping: { mode: "extended", }, usage: GPUTextureUsage.RENDER_ATTACHMENT, viewFormats: [ "rgba16float", ], });
        _computeShaderModule = _device.createShaderModule({
//      _computeShaderModule = _device.createShaderModule({
            label: "GPU_SHADER_MODULE_COMPUTE",
//          label: "GPU_SHADER_MODULE_COMPUTE",
            code: shaderStringCompute,
//          code: shaderStringCompute,
        });
//      });
        _gatherShaderModule = _device.createShaderModule({
//      _gatherShaderModule = _device.createShaderModule({
            label: "GPU_SHADER_MODULE_GATHER",
//          label: "GPU_SHADER_MODULE_GATHER",
            code: shaderStringGather,
//          code: shaderStringGather,
        });
//      });
        _blur2XShaderModule = _device.createShaderModule({
//      _blur2XShaderModule = _device.createShaderModule({
            label: "GPU_SHADER_MODULE_BLUR_2X",
//          label: "GPU_SHADER_MODULE_BLUR_2X",
            code: shaderStringBlur2X,
//          code: shaderStringBlur2X,
        });
//      });
        _blur4XShaderModule = _device.createShaderModule({
//      _blur4XShaderModule = _device.createShaderModule({
            label: "GPU_SHADER_MODULE_BLUR_4X",
//          label: "GPU_SHADER_MODULE_BLUR_4X",
            code: shaderStringBlur4X,
//          code: shaderStringBlur4X,
        });
//      });
        _blur8XShaderModule = _device.createShaderModule({
//      _blur8XShaderModule = _device.createShaderModule({
            label: "GPU_SHADER_MODULE_BLUR_8X",
//          label: "GPU_SHADER_MODULE_BLUR_8X",
            code: shaderStringBlur8X,
//          code: shaderStringBlur8X,
        });
//      });
        _renderShaderModule = _device.createShaderModule({
//      _renderShaderModule = _device.createShaderModule({
            label: "GPU_SHADER_MODULE_RENDER",
//          label: "GPU_SHADER_MODULE_RENDER",
            code: shaderStringRender,
//          code: shaderStringRender,
        });
//      });
        _computeBindGroupLayout0 = _device.createBindGroupLayout({
//      _computeBindGroupLayout0 = _device.createBindGroupLayout({
            label: "GPU_BIND_GROUP_LAYOUT_0_COMPUTE",
//          label: "GPU_BIND_GROUP_LAYOUT_0_COMPUTE",
            entries: [
//          entries: [
                { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform", }, },
//              { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform", }, },
                { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage", }, },
//              { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage", }, },
                { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage", }, },
//              { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage", }, },
                { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage", }, },
//              { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage", }, },
                { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage", }, },
//              { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage", }, },
                { binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage", }, },
//              { binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage", }, },
                { binding: 6, visibility: GPUShaderStage.COMPUTE, sampler: {}, },
//              { binding: 6, visibility: GPUShaderStage.COMPUTE, sampler: {}, },
                { binding: 7, visibility: GPUShaderStage.COMPUTE, texture: {}, },
//              { binding: 7, visibility: GPUShaderStage.COMPUTE, texture: {}, },
                { binding: 8, visibility: GPUShaderStage.COMPUTE, sampler: {}, },
//              { binding: 8, visibility: GPUShaderStage.COMPUTE, sampler: {}, },
                { binding: 9, visibility: GPUShaderStage.COMPUTE, texture: {}, },
//              { binding: 9, visibility: GPUShaderStage.COMPUTE, texture: {}, },
                { binding: 10, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage", }, },
//              { binding: 10, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage", }, },
                { binding: 11, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage", }, },
//              { binding: 11, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage", }, },
            ],
//          ],
        });
//      });
        _gatherBindGroupLayout0 = null;
//      _gatherBindGroupLayout0 = null;
        _blur2XBindGroupLayout0 = null;
//      _blur2XBindGroupLayout0 = null;
        _blur4XBindGroupLayout0 = null;
//      _blur4XBindGroupLayout0 = null;
        _blur8XBindGroupLayout0 = null;
//      _blur8XBindGroupLayout0 = null;
        _renderBindGroupLayout0 = null;
//      _renderBindGroupLayout0 = null;
        _computePipelineLayout = _device.createPipelineLayout({
//      _computePipelineLayout = _device.createPipelineLayout({
            label: "GPU_PIPELINE_LAYOUT_COMPUTE",
//          label: "GPU_PIPELINE_LAYOUT_COMPUTE",
            bindGroupLayouts: [
//          bindGroupLayouts: [
                _computeBindGroupLayout0,
//              _computeBindGroupLayout0,
            ],
//          ],
        });
//      });
        _gatherPipelineLayout = "auto";
//      _gatherPipelineLayout = "auto";
        _blur2XPipelineLayout = "auto";
//      _blur2XPipelineLayout = "auto";
        _blur4XPipelineLayout = "auto";
//      _blur4XPipelineLayout = "auto";
        _blur8XPipelineLayout = "auto";
//      _blur8XPipelineLayout = "auto";
        _renderPipelineLayout = "auto";
//      _renderPipelineLayout = "auto";
        _computePipeline = await _device.createComputePipelineAsync({
//      _computePipeline = await _device.createComputePipelineAsync({
            label: "GPU_PIPELINE_COMPUTE",
//          label: "GPU_PIPELINE_COMPUTE",
            layout: _computePipelineLayout,
//          layout: _computePipelineLayout,
            compute: {
//          compute: {
                module: _computeShaderModule,
//              module: _computeShaderModule,
                entryPoint: "main",
//              entryPoint: "main",
            },
//          },
        });
//      });
        _gatherPipeline = await _device.createComputePipelineAsync({
//      _gatherPipeline = await _device.createComputePipelineAsync({
            label: "GPU_PIPELINE_GATHER",
//          label: "GPU_PIPELINE_GATHER",
            layout: _gatherPipelineLayout,
//          layout: _gatherPipelineLayout,
            compute: {
//          compute: {
                module: _gatherShaderModule,
//              module: _gatherShaderModule,
                entryPoint: "main",
//              entryPoint: "main",
            },
//          },
        });
//      });
        _blur2XPipeline = await _device.createRenderPipelineAsync({
//      _blur2XPipeline = await _device.createRenderPipelineAsync({
            label: "GPU_PIPELINE_BLUR_2X",
//          label: "GPU_PIPELINE_BLUR_2X",
            layout: _blur2XPipelineLayout,
//          layout: _blur2XPipelineLayout,
            vertex: {
//          vertex: {
                module: _blur2XShaderModule,
//              module: _blur2XShaderModule,
                entryPoint: "vertexShader",
//              entryPoint: "vertexShader",
            },
//          },
            fragment: {
//          fragment: {
                module: _blur2XShaderModule,
//              module: _blur2XShaderModule,
                entryPoint: "fragmentShader",
//              entryPoint: "fragmentShader",
                targets: [{ format: "rgba32float" }],
//              targets: [{ format: "rgba32float" }],
            },
//          },
        });
//      });
        _blur4XPipeline = await _device.createRenderPipelineAsync({
//      _blur4XPipeline = await _device.createRenderPipelineAsync({
            label: "GPU_PIPELINE_BLUR_4X",
//          label: "GPU_PIPELINE_BLUR_4X",
            layout: _blur4XPipelineLayout,
//          layout: _blur4XPipelineLayout,
            vertex: {
//          vertex: {
                module: _blur4XShaderModule,
//              module: _blur4XShaderModule,
                entryPoint: "vertexShader",
//              entryPoint: "vertexShader",
            },
//          },
            fragment: {
//          fragment: {
                module: _blur4XShaderModule,
//              module: _blur4XShaderModule,
                entryPoint: "fragmentShader",
//              entryPoint: "fragmentShader",
                targets: [{ format: "rgba32float" }],
//              targets: [{ format: "rgba32float" }],
            },
//          },
        });
//      });
        _blur8XPipeline = await _device.createRenderPipelineAsync({
//      _blur8XPipeline = await _device.createRenderPipelineAsync({
            label: "GPU_PIPELINE_BLUR_8X",
//          label: "GPU_PIPELINE_BLUR_8X",
            layout: _blur8XPipelineLayout,
//          layout: _blur8XPipelineLayout,
            vertex: {
//          vertex: {
                module: _blur8XShaderModule,
//              module: _blur8XShaderModule,
                entryPoint: "vertexShader",
//              entryPoint: "vertexShader",
            },
//          },
            fragment: {
//          fragment: {
                module: _blur8XShaderModule,
//              module: _blur8XShaderModule,
                entryPoint: "fragmentShader",
//              entryPoint: "fragmentShader",
                targets: [{ format: "rgba32float" }],
//              targets: [{ format: "rgba32float" }],
            },
//          },
        });
//      });
        _renderPipeline = await _device.createRenderPipelineAsync({
//      _renderPipeline = await _device.createRenderPipelineAsync({
            label: "GPU_PIPELINE_RENDER",
//          label: "GPU_PIPELINE_RENDER",
            layout: _renderPipelineLayout,
//          layout: _renderPipelineLayout,
            vertex: {
//          vertex: {
                module: _renderShaderModule,
//              module: _renderShaderModule,
                entryPoint: "vertexShader",
//              entryPoint: "vertexShader",
            },
//          },
            fragment: {
//          fragment: {
                module: _renderShaderModule,
//              module: _renderShaderModule,
                entryPoint: "fragmentShader",
//              entryPoint: "fragmentShader",
                targets: [{ format: _presentationFormat }],
//              targets: [{ format: _presentationFormat }],
            },
//          },
        });
//      });
        _commandEncoderDescriptor = { label: "GPU_COMMAND_ENCODER_DESCRIPTOR", };
//      _commandEncoderDescriptor = { label: "GPU_COMMAND_ENCODER_DESCRIPTOR", };
        _computePassDescriptor = { label: "GPU_PASS_DESCRIPTOR_COMPUTE", };
//      _computePassDescriptor = { label: "GPU_PASS_DESCRIPTOR_COMPUTE", };
        _gatherPassDescriptor = { label: "GPU_PASS_DESCRIPTOR_GATHER", };
//      _gatherPassDescriptor = { label: "GPU_PASS_DESCRIPTOR_GATHER", };
        _blur2XPassDescriptor = {
//      _blur2XPassDescriptor = {
            label: "GPU_PASS_DESCRIPTOR_BLUR_2X",
//          label: "GPU_PASS_DESCRIPTOR_BLUR_2X",
            colorAttachments: [
//          colorAttachments: [
                {
                    view: undefined as any,
//                  view: undefined as any,
                    clearValue: [0.3, 0.3, 0.3, 1],
//                  clearValue: [0.3, 0.3, 0.3, 1],
                    loadOp: "clear",
//                  loadOp: "clear",
                    storeOp: "store",
//                  storeOp: "store",
                }
            ] as GPURenderPassColorAttachment[],
//          ] as GPURenderPassColorAttachment[],
        };
//      };
        _blur4XPassDescriptor = {
//      _blur4XPassDescriptor = {
            label: "GPU_PASS_DESCRIPTOR_BLUR_4X",
//          label: "GPU_PASS_DESCRIPTOR_BLUR_4X",
            colorAttachments: [
//          colorAttachments: [
                {
                    view: undefined as any,
//                  view: undefined as any,
                    clearValue: [0.3, 0.3, 0.3, 1],
//                  clearValue: [0.3, 0.3, 0.3, 1],
                    loadOp: "clear",
//                  loadOp: "clear",
                    storeOp: "store",
//                  storeOp: "store",
                }
            ] as GPURenderPassColorAttachment[],
//          ] as GPURenderPassColorAttachment[],
        };
//      };
        _blur8XPassDescriptor = {
//      _blur8XPassDescriptor = {
            label: "GPU_PASS_DESCRIPTOR_BLUR_8X",
//          label: "GPU_PASS_DESCRIPTOR_BLUR_8X",
            colorAttachments: [
//          colorAttachments: [
                {
                    view: undefined as any,
//                  view: undefined as any,
                    clearValue: [0.3, 0.3, 0.3, 1],
//                  clearValue: [0.3, 0.3, 0.3, 1],
                    loadOp: "clear",
//                  loadOp: "clear",
                    storeOp: "store",
//                  storeOp: "store",
                }
            ] as GPURenderPassColorAttachment[],
//          ] as GPURenderPassColorAttachment[],
        };
//      };
        _renderPassDescriptor = {
//      _renderPassDescriptor = {
            label: "GPU_PASS_DESCRIPTOR_RENDER",
//          label: "GPU_PASS_DESCRIPTOR_RENDER",
            colorAttachments: [
//          colorAttachments: [
                {
                    view: undefined as any,
//                  view: undefined as any,
                    clearValue: [0.3, 0.3, 0.3, 1],
//                  clearValue: [0.3, 0.3, 0.3, 1],
                    loadOp: "clear",
//                  loadOp: "clear",
                    storeOp: "store",
//                  storeOp: "store",
                }
            ] as GPURenderPassColorAttachment[],
//          ] as GPURenderPassColorAttachment[],
        };
//      };
        _generalDataUniformValues = new ArrayBuffer(32 * 4);
//      _generalDataUniformValues = new ArrayBuffer(32 * 4);
        _generalDataUniformBuffer = _device.createBuffer({
//      _generalDataUniformBuffer = _device.createBuffer({
            label: "GPU_UNIFORM_BUFFER_GENERAL_DATA",
//          label: "GPU_UNIFORM_BUFFER_GENERAL_DATA",
            size: _generalDataUniformValues.byteLength,
//          size: _generalDataUniformValues.byteLength,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
//          usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });
//      });
        _generalDataUniformValuesDataView = new DataView(_generalDataUniformValues);
//      _generalDataUniformValuesDataView = new DataView(_generalDataUniformValues);
        _columnAtlasSampler = _device.createSampler({
//      _columnAtlasSampler = _device.createSampler({
            magFilter: "linear",
//          magFilter: "linear",
            minFilter: "linear",
//          minFilter: "linear",
        });
//      });
        _hdriSampler = _device.createSampler({
//      _hdriSampler = _device.createSampler({
            magFilter: "linear",
//          magFilter: "linear",
            minFilter: "linear",
//          minFilter: "linear",
        });
//      });
        _outputSampler = _device.createSampler({
//      _outputSampler = _device.createSampler({
            magFilter: "linear",
//          magFilter: "linear",
            minFilter: "linear",
//          minFilter: "linear",
        });
//      });
        _blurSampler = _device.createSampler({
//      _blurSampler = _device.createSampler({
            magFilter: "linear",
//          magFilter: "linear",
            minFilter: "linear",
//          minFilter: "linear",
        });
//      });
/*
        // Corrected bytesPerPixel for rgba8unorm
        // Corrected bytesPerPixel for rgba8unorm
        const bytesPerPixel: number = 4; // 4 channels * 1 byte(s) per channel (8-bit unsigned-int)
//      const bytesPerPixel: number = 4; // 4 channels * 1 byte(s) per channel (8-bit unsigned-int)
*/
/*
        // Corrected bytesPerPixel for rgba16float
        // Corrected bytesPerPixel for rgba16float
        const bytesPerPixel: number = 8; // 4 channels * 2 byte(s) per channel (16-bit float)
//      const bytesPerPixel: number = 8; // 4 channels * 2 byte(s) per channel (16-bit float)
*/
/*
        // Corrected bytesPerPixel for rgba32float
        // Corrected bytesPerPixel for rgba32float
        const bytesPerPixel: number = 16; // 4 channels * 4 byte(s) per channel (32-bit float)
//      const bytesPerPixel: number = 16; // 4 channels * 4 byte(s) per channel (32-bit float)
*/
        const publicURLs: string[] = [
//      const publicURLs: string[] = [
            "/test/WoodToy3_albedo.jpg",
//          "/test/WoodToy3_albedo.jpg",
        ];
//      ];
        const cellImageWidth: number = 8192;
//      const cellImageWidth: number = 8192;
        const cellImageHeight: number = 8192;
//      const cellImageHeight: number = 8192;
        _numberOfImages = publicURLs.length;
//      _numberOfImages = publicURLs.length;
        const columnAtlas: ColumnAtlas = await createColumnAtlas({
//      const columnAtlas: ColumnAtlas = await createColumnAtlas({
            publicURLs: publicURLs,
//          publicURLs: publicURLs,
            cellImageWidth: cellImageWidth,
//          cellImageWidth: cellImageWidth,
            cellImageHeight: cellImageHeight,
//          cellImageHeight: cellImageHeight,
        });
//      });
        _columnAtlasTexture = _device.createTexture({
//      _columnAtlasTexture = _device.createTexture({
            label: "GPU_TEXTURE_COLUMN_ATLAS",
//          label: "GPU_TEXTURE_COLUMN_ATLAS",
            size: [ columnAtlas.mergedImageWidth, columnAtlas.mergedImageHeight, ],
//          size: [ columnAtlas.mergedImageWidth, columnAtlas.mergedImageHeight, ],
            format: "rgba8unorm",
//          format: "rgba8unorm",
            usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
//          usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
        });
//      });
        const columnAtlasTextureBytesPerPixel: number = 4;
//      const columnAtlasTextureBytesPerPixel: number = 4;
        const columnAtlasTextureUnalignedBytesPerRow: number = columnAtlas.mergedImageWidth * columnAtlasTextureBytesPerPixel;
//      const columnAtlasTextureUnalignedBytesPerRow: number = columnAtlas.mergedImageWidth * columnAtlasTextureBytesPerPixel;
        const columnAtlasTextureAlignedBytesPerRow: number = Math.ceil(columnAtlasTextureUnalignedBytesPerRow / 256) * 256;
//      const columnAtlasTextureAlignedBytesPerRow: number = Math.ceil(columnAtlasTextureUnalignedBytesPerRow / 256) * 256;
        _device.queue.writeTexture(
//      _device.queue.writeTexture(
            {
//          {
                aspect: "all",
//              aspect: "all",
                mipLevel: 0,
//              mipLevel: 0,
                origin: undefined,
//              origin: undefined,
                texture: _columnAtlasTexture,
//              texture: _columnAtlasTexture,
            },
//          },
            columnAtlas.mergedImageBuffer as GPUAllowSharedBufferSource,
//          columnAtlas.mergedImageBuffer as GPUAllowSharedBufferSource,
            {
//          {
                bytesPerRow: columnAtlasTextureAlignedBytesPerRow,
//              bytesPerRow: columnAtlasTextureAlignedBytesPerRow,
                offset: undefined,
//              offset: undefined,
                rowsPerImage: columnAtlas.mergedImageHeight,
//              rowsPerImage: columnAtlas.mergedImageHeight,
            },
//          },
            {
//          {
                depthOrArrayLayers: undefined,
//              depthOrArrayLayers: undefined,
                height: columnAtlas.mergedImageHeight,
//              height: columnAtlas.mergedImageHeight,
                width: columnAtlas.mergedImageWidth,
//              width: columnAtlas.mergedImageWidth,
                depth: undefined,
//              depth: undefined,
            },
//          },
        );
//      );
        const skyboxImage: ArrayBuffer = await (await fetch("/skyboxes/kloppenheim_05_puresky_4k.exr")).arrayBuffer();
//      const skyboxImage: ArrayBuffer = await (await fetch("/skyboxes/kloppenheim_05_puresky_4k.exr")).arrayBuffer();
        const skyboxImageParsed = await parseEXRWithWorker(skyboxImage, 1015);
//      const skyboxImageParsed = await parseEXRWithWorker(skyboxImage, 1015);
        _hdriTexture = _device.createTexture({
//      _hdriTexture = _device.createTexture({
            label: "GPU_TEXTURE_HDRI",
//          label: "GPU_TEXTURE_HDRI",
            size: [ skyboxImageParsed.width, skyboxImageParsed.height, ],
//          size: [ skyboxImageParsed.width, skyboxImageParsed.height, ],
            format: "rgba32float",
//          format: "rgba32float",
            usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
//          usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
        });
//      });
        const hdriTextureBytesPerPixel: number = 16;
//      const hdriTextureBytesPerPixel: number = 16;
        const hdriTextureUnalignedBytesPerRow: number = skyboxImageParsed.width * hdriTextureBytesPerPixel;
//      const hdriTextureUnalignedBytesPerRow: number = skyboxImageParsed.width * hdriTextureBytesPerPixel;
        const hdriTextureAlignedBytesPerRow: number = Math.ceil(hdriTextureUnalignedBytesPerRow / 256) * 256;
//      const hdriTextureAlignedBytesPerRow: number = Math.ceil(hdriTextureUnalignedBytesPerRow / 256) * 256;
        _device.queue.writeTexture(
//      _device.queue.writeTexture(
            {
//          {
                aspect: "all",
//              aspect: "all",
                mipLevel: 0,
//              mipLevel: 0,
                origin: undefined,
//              origin: undefined,
                texture: _hdriTexture,
//              texture: _hdriTexture,
            },
//          },
            skyboxImageParsed.data as GPUAllowSharedBufferSource,
//          skyboxImageParsed.data as GPUAllowSharedBufferSource,
            {
//          {
                bytesPerRow: hdriTextureAlignedBytesPerRow,
//              bytesPerRow: hdriTextureAlignedBytesPerRow,
                offset: undefined,
//              offset: undefined,
                rowsPerImage: skyboxImageParsed.height,
//              rowsPerImage: skyboxImageParsed.height,
            },
//          },
            {
//          {
                depthOrArrayLayers: undefined,
//              depthOrArrayLayers: undefined,
                height: skyboxImageParsed.height,
//              height: skyboxImageParsed.height,
                width: skyboxImageParsed.width,
//              width: skyboxImageParsed.width,
                depth: undefined,
//              depth: undefined,
            },
//          },
        );
//      );
        _spheres.push(
//      _spheres.push(
//             {
// //          {
//                 center: [ +0.0, -1020.0, +0.0 ],
// //              center: [ +0.0, -1020.0, +0.0 ],
//                 radius: 1000.0,
// //              radius: 1000.0,
//                 materialIndex: 0,
// //              materialIndex: 0,
//             },
// //          },
            {
//          {
                center: [ -20.0, -12.0, +20.0 ],
//              center: [ -20.0, -12.0, +20.0 ],
                radius: 8.0,
//              radius: 8.0,
                materialIndex: 1,
//              materialIndex: 1,
            },
//          },
            {
//          {
                center: [ +20.0, -12.0, -20.0 ],
//              center: [ +20.0, -12.0, -20.0 ],
                radius: 8.0,
//              radius: 8.0,
                materialIndex: 2,
//              materialIndex: 2,
            },
//          },
            {
//          {
                center: [ -20.0, -12.0, -20.0 ],
//              center: [ -20.0, -12.0, -20.0 ],
                radius: 8.0,
//              radius: 8.0,
                materialIndex: 3,
//              materialIndex: 3,
            },
//          },
        );
//      );
        _materials.push(
//      _materials.push(
            // MARBLE GLOSS (GLOSSY DIELECTRIC)
//          // MARBLE GLOSS (GLOSSY DIELECTRIC)
            {
//          {
                layer1Roughness: 0.1,
//              layer1Roughness: 0.1,
                layer1Metallic: 0.01,
//              layer1Metallic: 0.01,
                layer1Transmission: 0.0,
//              layer1Transmission: 0.0,
                layer0IOR: type.RefractiveIndex.AIR,
//              layer0IOR: type.RefractiveIndex.AIR,
                layer1IOR: type.RefractiveIndex.MARBLE,
//              layer1IOR: type.RefractiveIndex.MARBLE,
                materialType: type.MaterialType.GLOSS,
//              materialType: type.MaterialType.GLOSS,
                textureIndex: 0,
//              textureIndex: 0,
            },
//          },
            // MARBLE GLOSS (GLOSSY DIELECTRIC)
//          // MARBLE GLOSS (GLOSSY DIELECTRIC)
            {
//          {
                layer1Roughness: 0.1,
//              layer1Roughness: 0.1,
                layer1Metallic: 0.01,
//              layer1Metallic: 0.01,
                layer1Transmission: 0.0,
//              layer1Transmission: 0.0,
                layer0IOR: type.RefractiveIndex.AIR,
//              layer0IOR: type.RefractiveIndex.AIR,
                layer1IOR: type.RefractiveIndex.MARBLE,
//              layer1IOR: type.RefractiveIndex.MARBLE,
                materialType: type.MaterialType.GLOSS,
//              materialType: type.MaterialType.GLOSS,
                textureIndex: 1,
//              textureIndex: 1,
            },
//          },
            // DIFFUSE
//          // DIFFUSE
            {
//          {
                layer1Roughness: 1.0,
//              layer1Roughness: 1.0,
                layer1Metallic: 0.0,
//              layer1Metallic: 0.0,
                layer1Transmission: 0.0,
//              layer1Transmission: 0.0,
                layer0IOR: type.RefractiveIndex.NOTHING,
//              layer0IOR: type.RefractiveIndex.NOTHING,
                layer1IOR: type.RefractiveIndex.NOTHING,
//              layer1IOR: type.RefractiveIndex.NOTHING,
                materialType: type.MaterialType.DIFFUSE,
//              materialType: type.MaterialType.DIFFUSE,
                textureIndex: 2,
//              textureIndex: 2,
            },
//          },
            // DIFFUSE
//          // DIFFUSE
            {
//          {
                layer1Roughness: 1.0,
//              layer1Roughness: 1.0,
                layer1Metallic: 0.0,
//              layer1Metallic: 0.0,
                layer1Transmission: 0.0,
//              layer1Transmission: 0.0,
                layer0IOR: type.RefractiveIndex.NOTHING,
//              layer0IOR: type.RefractiveIndex.NOTHING,
                layer1IOR: type.RefractiveIndex.NOTHING,
//              layer1IOR: type.RefractiveIndex.NOTHING,
                materialType: type.MaterialType.DIFFUSE,
//              materialType: type.MaterialType.DIFFUSE,
                textureIndex: 3,
//              textureIndex: 3,
            },
//          },
            // MARBLE GLOSS (GLOSSY DIELECTRIC)
//          // MARBLE GLOSS (GLOSSY DIELECTRIC)
            {
//          {
                layer1Roughness: 0.1,
//              layer1Roughness: 0.1,
                layer1Metallic: 0.01,
//              layer1Metallic: 0.01,
                layer1Transmission: 0.0,
//              layer1Transmission: 0.0,
                layer0IOR: type.RefractiveIndex.AIR,
//              layer0IOR: type.RefractiveIndex.AIR,
                layer1IOR: type.RefractiveIndex.MARBLE,
//              layer1IOR: type.RefractiveIndex.MARBLE,
                materialType: type.MaterialType.GLOSS,
//              materialType: type.MaterialType.GLOSS,
                textureIndex: 4,
//              textureIndex: 4,
            },
//          },
            // LIGHT
//          // LIGHT
            {
//          {
                layer1Roughness: 0.0,
//              layer1Roughness: 0.0,
                layer1Metallic: 0.0,
//              layer1Metallic: 0.0,
                layer1Transmission: 0.0,
//              layer1Transmission: 0.0,
                layer0IOR: type.RefractiveIndex.NOTHING,
//              layer0IOR: type.RefractiveIndex.NOTHING,
                layer1IOR: type.RefractiveIndex.NOTHING,
//              layer1IOR: type.RefractiveIndex.NOTHING,
                materialType: type.MaterialType.LIGHT,
//              materialType: type.MaterialType.LIGHT,
                textureIndex: 5,
//              textureIndex: 5,
            },
//          },
            // MARBLE GLOSS (GLOSSY DIELECTRIC)
//          // MARBLE GLOSS (GLOSSY DIELECTRIC)
            {
//          {
                layer1Roughness: 0.1,
//              layer1Roughness: 0.1,
                layer1Metallic: 0.01,
//              layer1Metallic: 0.01,
                layer1Transmission: 0.0,
//              layer1Transmission: 0.0,
                layer0IOR: type.RefractiveIndex.AIR,
//              layer0IOR: type.RefractiveIndex.AIR,
                layer1IOR: type.RefractiveIndex.MARBLE,
//              layer1IOR: type.RefractiveIndex.MARBLE,
                materialType: type.MaterialType.GLOSS,
//              materialType: type.MaterialType.GLOSS,
                textureIndex: 6,
//              textureIndex: 6,
            },
//          },
            // LIGHT A
//          // LIGHT A
            {
//          {
                layer1Roughness: 0.0,
//              layer1Roughness: 0.0,
                layer1Metallic: 0.0,
//              layer1Metallic: 0.0,
                layer1Transmission: 0.0,
//              layer1Transmission: 0.0,
                layer0IOR: type.RefractiveIndex.NOTHING,
//              layer0IOR: type.RefractiveIndex.NOTHING,
                layer1IOR: type.RefractiveIndex.NOTHING,
//              layer1IOR: type.RefractiveIndex.NOTHING,
                materialType: type.MaterialType.LIGHT,
//              materialType: type.MaterialType.LIGHT,
                textureIndex: 11,
//              textureIndex: 11,
            },
//          },
            // LIGHT B
//          // LIGHT B
            {
//          {
                layer1Roughness: 0.0,
//              layer1Roughness: 0.0,
                layer1Metallic: 0.0,
//              layer1Metallic: 0.0,
                layer1Transmission: 0.0,
//              layer1Transmission: 0.0,
                layer0IOR: type.RefractiveIndex.NOTHING,
//              layer0IOR: type.RefractiveIndex.NOTHING,
                layer1IOR: type.RefractiveIndex.NOTHING,
//              layer1IOR: type.RefractiveIndex.NOTHING,
                materialType: type.MaterialType.LIGHT,
//              materialType: type.MaterialType.LIGHT,
                textureIndex: 12,
//              textureIndex: 12,
            },
//          },
            // LIGHT C
//          // LIGHT C
            {
//          {
                layer1Roughness: 0.0,
//              layer1Roughness: 0.0,
                layer1Metallic: 0.0,
//              layer1Metallic: 0.0,
                layer1Transmission: 0.0,
//              layer1Transmission: 0.0,
                layer0IOR: type.RefractiveIndex.NOTHING,
//              layer0IOR: type.RefractiveIndex.NOTHING,
                layer1IOR: type.RefractiveIndex.NOTHING,
//              layer1IOR: type.RefractiveIndex.NOTHING,
                materialType: type.MaterialType.LIGHT,
//              materialType: type.MaterialType.LIGHT,
                textureIndex: 13,
//              textureIndex: 13,
            },
//          },
            // LIGHT D
//          // LIGHT D
            {
//          {
                layer1Roughness: 0.0,
//              layer1Roughness: 0.0,
                layer1Metallic: 0.0,
//              layer1Metallic: 0.0,
                layer1Transmission: 0.0,
//              layer1Transmission: 0.0,
                layer0IOR: type.RefractiveIndex.NOTHING,
//              layer0IOR: type.RefractiveIndex.NOTHING,
                layer1IOR: type.RefractiveIndex.NOTHING,
//              layer1IOR: type.RefractiveIndex.NOTHING,
                materialType: type.MaterialType.LIGHT,
//              materialType: type.MaterialType.LIGHT,
                textureIndex: 14,
//              textureIndex: 14,
            },
//          },
        );
//      );
        _textures.push(
//      _textures.push(
            {
//          {
                albedo: [ 0.96, 0.96, 0.94 ],
//              albedo: [ 0.96, 0.96, 0.94 ],
                albedoImageIndex: 0,
//              albedoImageIndex: 0,
                albedoTextureType: type.TextureType.COLOR,
//              albedoTextureType: type.TextureType.COLOR,
                albedoScale: 1.0,
//              albedoScale: 1.0,
                albedoOTileTextureIndex: 0,
//              albedoOTileTextureIndex: 0,
                albedoETileTextureIndex: 0,
//              albedoETileTextureIndex: 0,
            },
//          },
            {
//          {
                albedo: [ 0.96, 0.96, 0.94 ],
//              albedo: [ 0.96, 0.96, 0.94 ],
                albedoImageIndex: 1,
//              albedoImageIndex: 1,
                albedoTextureType: type.TextureType.COLOR,
//              albedoTextureType: type.TextureType.COLOR,
                albedoScale: 1.0,
//              albedoScale: 1.0,
                albedoOTileTextureIndex: 0,
//              albedoOTileTextureIndex: 0,
                albedoETileTextureIndex: 0,
//              albedoETileTextureIndex: 0,
            },
//          },
            {
//          {
                albedo: [ 0.78, 0.12, 0.15 ],
//              albedo: [ 0.78, 0.12, 0.15 ],
                albedoImageIndex: 2,
//              albedoImageIndex: 2,
                albedoTextureType: type.TextureType.COLOR,
//              albedoTextureType: type.TextureType.COLOR,
                albedoScale: 1.0,
//              albedoScale: 1.0,
                albedoOTileTextureIndex: 0,
//              albedoOTileTextureIndex: 0,
                albedoETileTextureIndex: 0,
//              albedoETileTextureIndex: 0,
            },
//          },
            {
//          {
                albedo: [ 0.05, 0.55, 0.50 ],
//              albedo: [ 0.05, 0.55, 0.50 ],
                albedoImageIndex: 3,
//              albedoImageIndex: 3,
                albedoTextureType: type.TextureType.COLOR,
//              albedoTextureType: type.TextureType.COLOR,
                albedoScale: 1.0,
//              albedoScale: 1.0,
                albedoOTileTextureIndex: 0,
//              albedoOTileTextureIndex: 0,
                albedoETileTextureIndex: 0,
//              albedoETileTextureIndex: 0,
            },
//          },
            {
//          {
                albedo: [ 0.96, 0.96, 0.94 ],
//              albedo: [ 0.96, 0.96, 0.94 ],
                albedoImageIndex: 4,
//              albedoImageIndex: 4,
                albedoTextureType: type.TextureType.COLOR,
//              albedoTextureType: type.TextureType.COLOR,
                albedoScale: 1.0,
//              albedoScale: 1.0,
                albedoOTileTextureIndex: 0,
//              albedoOTileTextureIndex: 0,
                albedoETileTextureIndex: 0,
//              albedoETileTextureIndex: 0,
            },
//          },
            // LIGHT
            // LIGHT
            {
//          {
                albedo: [ 5.00, 5.00, 5.00 ],
//              albedo: [ 5.00, 5.00, 5.00 ],
                albedoImageIndex: 5,
//              albedoImageIndex: 5,
                albedoTextureType: type.TextureType.COLOR,
//              albedoTextureType: type.TextureType.COLOR,
                albedoScale: 1.0,
//              albedoScale: 1.0,
                albedoOTileTextureIndex: 0,
//              albedoOTileTextureIndex: 0,
                albedoETileTextureIndex: 0,
//              albedoETileTextureIndex: 0,
            },
//          },
            {
//          {
                albedo: [ 1.00, 1.00, 1.00 ],
//              albedo: [ 1.00, 1.00, 1.00 ],
                albedoImageIndex: 0,
//              albedoImageIndex: 0,
                albedoTextureType: type.TextureType.IMAGE,
//              albedoTextureType: type.TextureType.IMAGE,
                albedoScale: 1.0,
//              albedoScale: 1.0,
                albedoOTileTextureIndex: 0,
//              albedoOTileTextureIndex: 0,
                albedoETileTextureIndex: 0,
//              albedoETileTextureIndex: 0,
            },
//          },
            {
//          {
                albedo: [ 0.10, 0.45, 1.00 ],
//              albedo: [ 0.10, 0.45, 1.00 ],
                albedoImageIndex: 7,
//              albedoImageIndex: 7,
                albedoTextureType: type.TextureType.COLOR,
//              albedoTextureType: type.TextureType.COLOR,
                albedoScale: 1.0,
//              albedoScale: 1.0,
                albedoOTileTextureIndex: 0,
//              albedoOTileTextureIndex: 0,
                albedoETileTextureIndex: 0,
//              albedoETileTextureIndex: 0,
            },
//          },
            {
//          {
                albedo: [ 0.80, 0.83, 0.88 ],
//              albedo: [ 0.80, 0.83, 0.88 ],
                albedoImageIndex: 8,
//              albedoImageIndex: 8,
                albedoTextureType: type.TextureType.COLOR,
//              albedoTextureType: type.TextureType.COLOR,
                albedoScale: 1.0,
//              albedoScale: 1.0,
                albedoOTileTextureIndex: 0,
//              albedoOTileTextureIndex: 0,
                albedoETileTextureIndex: 0,
//              albedoETileTextureIndex: 0,
            },
//          },
            {
//          {
                albedo: [ 1.00, 0.10, 0.20 ],
//              albedo: [ 1.00, 0.10, 0.20 ],
                albedoImageIndex: 9,
//              albedoImageIndex: 9,
                albedoTextureType: type.TextureType.COLOR,
//              albedoTextureType: type.TextureType.COLOR,
                albedoScale: 1.0,
//              albedoScale: 1.0,
                albedoOTileTextureIndex: 0,
//              albedoOTileTextureIndex: 0,
                albedoETileTextureIndex: 0,
//              albedoETileTextureIndex: 0,
            },
//          },
            {
//          {
                albedo: [ 0.80, 0.83, 0.88 ],
//              albedo: [ 0.80, 0.83, 0.88 ],
                albedoImageIndex: 10,
//              albedoImageIndex: 10,
                albedoTextureType: type.TextureType.COLOR,
//              albedoTextureType: type.TextureType.COLOR,
                albedoScale: 1.0,
//              albedoScale: 1.0,
                albedoOTileTextureIndex: 0,
//              albedoOTileTextureIndex: 0,
                albedoETileTextureIndex: 0,
//              albedoETileTextureIndex: 0,
            },
//          },
            // LIGHT A
            // LIGHT A
            {
//          {
                albedo: [ 5.00, 5.00, 5.00 ],
//              albedo: [ 5.00, 5.00, 5.00 ],
                albedoImageIndex: 11,
//              albedoImageIndex: 11,
                albedoTextureType: type.TextureType.COLOR,
//              albedoTextureType: type.TextureType.COLOR,
                albedoScale: 1.0,
//              albedoScale: 1.0,
                albedoOTileTextureIndex: 0,
//              albedoOTileTextureIndex: 0,
                albedoETileTextureIndex: 0,
//              albedoETileTextureIndex: 0,
            },
//          },
            // LIGHT B
            // LIGHT B
            {
//          {
                albedo: [ 5.00, 5.00, 5.00 ],
//              albedo: [ 5.00, 5.00, 5.00 ],
                albedoImageIndex: 12,
//              albedoImageIndex: 12,
                albedoTextureType: type.TextureType.COLOR,
//              albedoTextureType: type.TextureType.COLOR,
                albedoScale: 1.0,
//              albedoScale: 1.0,
                albedoOTileTextureIndex: 0,
//              albedoOTileTextureIndex: 0,
                albedoETileTextureIndex: 0,
//              albedoETileTextureIndex: 0,
            },
//          },
            // LIGHT C
            // LIGHT C
            {
//          {
                albedo: [ 5.00, 5.00, 5.00 ],
//              albedo: [ 5.00, 5.00, 5.00 ],
                albedoImageIndex: 13,
//              albedoImageIndex: 13,
                albedoTextureType: type.TextureType.COLOR,
//              albedoTextureType: type.TextureType.COLOR,
                albedoScale: 1.0,
//              albedoScale: 1.0,
                albedoOTileTextureIndex: 0,
//              albedoOTileTextureIndex: 0,
                albedoETileTextureIndex: 0,
//              albedoETileTextureIndex: 0,
            },
//          },
            // LIGHT D
            // LIGHT D
            {
//          {
                albedo: [ 5.00, 5.00, 5.00 ],
//              albedo: [ 5.00, 5.00, 5.00 ],
                albedoImageIndex: 14,
//              albedoImageIndex: 14,
                albedoTextureType: type.TextureType.COLOR,
//              albedoTextureType: type.TextureType.COLOR,
                albedoScale: 1.0,
//              albedoScale: 1.0,
                albedoOTileTextureIndex: 0,
//              albedoOTileTextureIndex: 0,
                albedoETileTextureIndex: 0,
//              albedoETileTextureIndex: 0,
            },
//          },
        );
//      );
        _triangles.push(
//      _triangles.push(
            // BOT
//          // BOT
            {
//          {
                aabb3d: undefined!,
//              aabb3d: undefined!,
                vertex0: [ -20.00, -20.00, -20.00 ],
//              vertex0: [ -20.00, -20.00, -20.00 ],
                vertex1: [ -20.00, -20.00, +20.00 ],
//              vertex1: [ -20.00, -20.00, +20.00 ],
                vertex2: [ +20.00, -20.00, +20.00 ],
//              vertex2: [ +20.00, -20.00, +20.00 ],
                vertex0FrontFaceNormal: [ 0.0, 0.0, 0.0 ],
//              vertex0FrontFaceNormal: [ 0.0, 0.0, 0.0 ],
                vertex1FrontFaceNormal: [ 0.0, 0.0, 0.0 ],
//              vertex1FrontFaceNormal: [ 0.0, 0.0, 0.0 ],
                vertex2FrontFaceNormal: [ 0.0, 0.0, 0.0 ],
//              vertex2FrontFaceNormal: [ 0.0, 0.0, 0.0 ],
                vertex0UV: [ 0.0, 0.0 ],
//              vertex0UV: [ 0.0, 0.0 ],
                vertex1UV: [ 0.0, 1.0 ],
//              vertex1UV: [ 0.0, 1.0 ],
                vertex2UV: [ 1.0, 1.0 ],
//              vertex2UV: [ 1.0, 1.0 ],
                materialIndex: 0,
//              materialIndex: 0,
                perVertexFrontFaceNormalAvailable: false,
//              perVertexFrontFaceNormalAvailable: false,
            },
//          },
            {
//          {
                aabb3d: undefined!,
//              aabb3d: undefined!,
                vertex0: [ +20.00, -20.00, +20.00 ],
//              vertex0: [ +20.00, -20.00, +20.00 ],
                vertex1: [ +20.00, -20.00, -20.00 ],
//              vertex1: [ +20.00, -20.00, -20.00 ],
                vertex2: [ -20.00, -20.00, -20.00 ],
//              vertex2: [ -20.00, -20.00, -20.00 ],
                vertex0FrontFaceNormal: [ 0.0, 0.0, 0.0 ],
//              vertex0FrontFaceNormal: [ 0.0, 0.0, 0.0 ],
                vertex1FrontFaceNormal: [ 0.0, 0.0, 0.0 ],
//              vertex1FrontFaceNormal: [ 0.0, 0.0, 0.0 ],
                vertex2FrontFaceNormal: [ 0.0, 0.0, 0.0 ],
//              vertex2FrontFaceNormal: [ 0.0, 0.0, 0.0 ],
                vertex0UV: [ 1.0, 1.0 ],
//              vertex0UV: [ 1.0, 1.0 ],
                vertex1UV: [ 1.0, 0.0 ],
//              vertex1UV: [ 1.0, 0.0 ],
                vertex2UV: [ 0.0, 0.0 ],
//              vertex2UV: [ 0.0, 0.0 ],
                materialIndex: 0,
//              materialIndex: 0,
                perVertexFrontFaceNormalAvailable: false,
//              perVertexFrontFaceNormalAvailable: false,
            },
//          },
            // TOP
//          // TOP
            {
//          {
                aabb3d: undefined!,
//              aabb3d: undefined!,
                vertex0: [ -20.00, +20.00, +20.00 ],
//              vertex0: [ -20.00, +20.00, +20.00 ],
                vertex1: [ -20.00, +20.00, -20.00 ],
//              vertex1: [ -20.00, +20.00, -20.00 ],
                vertex2: [ +20.00, +20.00, -20.00 ],
//              vertex2: [ +20.00, +20.00, -20.00 ],
                vertex0FrontFaceNormal: [ 0.0, 0.0, 0.0 ],
//              vertex0FrontFaceNormal: [ 0.0, 0.0, 0.0 ],
                vertex1FrontFaceNormal: [ 0.0, 0.0, 0.0 ],
//              vertex1FrontFaceNormal: [ 0.0, 0.0, 0.0 ],
                vertex2FrontFaceNormal: [ 0.0, 0.0, 0.0 ],
//              vertex2FrontFaceNormal: [ 0.0, 0.0, 0.0 ],
                vertex0UV: [ 0.0, 0.0 ],
//              vertex0UV: [ 0.0, 0.0 ],
                vertex1UV: [ 0.0, 1.0 ],
//              vertex1UV: [ 0.0, 1.0 ],
                vertex2UV: [ 1.0, 1.0 ],
//              vertex2UV: [ 1.0, 1.0 ],
                materialIndex: 1,
//              materialIndex: 1,
                perVertexFrontFaceNormalAvailable: false,
//              perVertexFrontFaceNormalAvailable: false,
            },
//          },
            {
//          {
                aabb3d: undefined!,
//              aabb3d: undefined!,
                vertex0: [ +20.00, +20.00, -20.00 ],
//              vertex0: [ +20.00, +20.00, -20.00 ],
                vertex1: [ +20.00, +20.00, +20.00 ],
//              vertex1: [ +20.00, +20.00, +20.00 ],
                vertex2: [ -20.00, +20.00, +20.00 ],
//              vertex2: [ -20.00, +20.00, +20.00 ],
                vertex0FrontFaceNormal: [ 0.0, 0.0, 0.0 ],
//              vertex0FrontFaceNormal: [ 0.0, 0.0, 0.0 ],
                vertex1FrontFaceNormal: [ 0.0, 0.0, 0.0 ],
//              vertex1FrontFaceNormal: [ 0.0, 0.0, 0.0 ],
                vertex2FrontFaceNormal: [ 0.0, 0.0, 0.0 ],
//              vertex2FrontFaceNormal: [ 0.0, 0.0, 0.0 ],
                vertex0UV: [ 1.0, 1.0 ],
//              vertex0UV: [ 1.0, 1.0 ],
                vertex1UV: [ 1.0, 0.0 ],
//              vertex1UV: [ 1.0, 0.0 ],
                vertex2UV: [ 0.0, 0.0 ],
//              vertex2UV: [ 0.0, 0.0 ],
                materialIndex: 1,
//              materialIndex: 1,
                perVertexFrontFaceNormalAvailable: false,
//              perVertexFrontFaceNormalAvailable: false,
            },
//          },
            // LEFT
//          // LEFT
            {
//          {
                aabb3d: undefined!,
//              aabb3d: undefined!,
                vertex0: [ -20.00, +20.00, +20.00 ],
//              vertex0: [ -20.00, +20.00, +20.00 ],
                vertex1: [ -20.00, -20.00, +20.00 ],
//              vertex1: [ -20.00, -20.00, +20.00 ],
                vertex2: [ -20.00, -20.00, -20.00 ],
//              vertex2: [ -20.00, -20.00, -20.00 ],
                vertex0FrontFaceNormal: [ 0.0, 0.0, 0.0 ],
//              vertex0FrontFaceNormal: [ 0.0, 0.0, 0.0 ],
                vertex1FrontFaceNormal: [ 0.0, 0.0, 0.0 ],
//              vertex1FrontFaceNormal: [ 0.0, 0.0, 0.0 ],
                vertex2FrontFaceNormal: [ 0.0, 0.0, 0.0 ],
//              vertex2FrontFaceNormal: [ 0.0, 0.0, 0.0 ],
                vertex0UV: [ 0.0, 0.0 ],
//              vertex0UV: [ 0.0, 0.0 ],
                vertex1UV: [ 0.0, 1.0 ],
//              vertex1UV: [ 0.0, 1.0 ],
                vertex2UV: [ 1.0, 1.0 ],
//              vertex2UV: [ 1.0, 1.0 ],
                materialIndex: 2,
//              materialIndex: 2,
                perVertexFrontFaceNormalAvailable: false,
//              perVertexFrontFaceNormalAvailable: false,
            },
//          },
            {
//          {
                aabb3d: undefined!,
//              aabb3d: undefined!,
                vertex0: [ -20.00, -20.00, -20.00 ],
//              vertex0: [ -20.00, -20.00, -20.00 ],
                vertex1: [ -20.00, +20.00, -20.00 ],
//              vertex1: [ -20.00, +20.00, -20.00 ],
                vertex2: [ -20.00, +20.00, +20.00 ],
//              vertex2: [ -20.00, +20.00, +20.00 ],
                vertex0FrontFaceNormal: [ 0.0, 0.0, 0.0 ],
//              vertex0FrontFaceNormal: [ 0.0, 0.0, 0.0 ],
                vertex1FrontFaceNormal: [ 0.0, 0.0, 0.0 ],
//              vertex1FrontFaceNormal: [ 0.0, 0.0, 0.0 ],
                vertex2FrontFaceNormal: [ 0.0, 0.0, 0.0 ],
//              vertex2FrontFaceNormal: [ 0.0, 0.0, 0.0 ],
                vertex0UV: [ 1.0, 1.0 ],
//              vertex0UV: [ 1.0, 1.0 ],
                vertex1UV: [ 1.0, 0.0 ],
//              vertex1UV: [ 1.0, 0.0 ],
                vertex2UV: [ 0.0, 0.0 ],
//              vertex2UV: [ 0.0, 0.0 ],
                materialIndex: 2,
//              materialIndex: 2,
                perVertexFrontFaceNormalAvailable: false,
//              perVertexFrontFaceNormalAvailable: false,
            },
//          },
            // RIGHT
//          // RIGHT
            {
//          {
                aabb3d: undefined!,
//              aabb3d: undefined!,
                vertex0: [ +20.00, +20.00, -20.00 ],
//              vertex0: [ +20.00, +20.00, -20.00 ],
                vertex1: [ +20.00, -20.00, -20.00 ],
//              vertex1: [ +20.00, -20.00, -20.00 ],
                vertex2: [ +20.00, -20.00, +20.00 ],
//              vertex2: [ +20.00, -20.00, +20.00 ],
                vertex0FrontFaceNormal: [ 0.0, 0.0, 0.0 ],
//              vertex0FrontFaceNormal: [ 0.0, 0.0, 0.0 ],
                vertex1FrontFaceNormal: [ 0.0, 0.0, 0.0 ],
//              vertex1FrontFaceNormal: [ 0.0, 0.0, 0.0 ],
                vertex2FrontFaceNormal: [ 0.0, 0.0, 0.0 ],
//              vertex2FrontFaceNormal: [ 0.0, 0.0, 0.0 ],
                vertex0UV: [ 0.0, 0.0 ],
//              vertex0UV: [ 0.0, 0.0 ],
                vertex1UV: [ 0.0, 1.0 ],
//              vertex1UV: [ 0.0, 1.0 ],
                vertex2UV: [ 1.0, 1.0 ],
//              vertex2UV: [ 1.0, 1.0 ],
                materialIndex: 3,
//              materialIndex: 3,
                perVertexFrontFaceNormalAvailable: false,
//              perVertexFrontFaceNormalAvailable: false,
            },
//          },
            {
//          {
                aabb3d: undefined!,
//              aabb3d: undefined!,
                vertex0: [ +20.00, -20.00, +20.00 ],
//              vertex0: [ +20.00, -20.00, +20.00 ],
                vertex1: [ +20.00, +20.00, +20.00 ],
//              vertex1: [ +20.00, +20.00, +20.00 ],
                vertex2: [ +20.00, +20.00, -20.00 ],
//              vertex2: [ +20.00, +20.00, -20.00 ],
                vertex0FrontFaceNormal: [ 0.0, 0.0, 0.0 ],
//              vertex0FrontFaceNormal: [ 0.0, 0.0, 0.0 ],
                vertex1FrontFaceNormal: [ 0.0, 0.0, 0.0 ],
//              vertex1FrontFaceNormal: [ 0.0, 0.0, 0.0 ],
                vertex2FrontFaceNormal: [ 0.0, 0.0, 0.0 ],
//              vertex2FrontFaceNormal: [ 0.0, 0.0, 0.0 ],
                vertex0UV: [ 1.0, 1.0 ],
//              vertex0UV: [ 1.0, 1.0 ],
                vertex1UV: [ 1.0, 0.0 ],
//              vertex1UV: [ 1.0, 0.0 ],
                vertex2UV: [ 0.0, 0.0 ],
//              vertex2UV: [ 0.0, 0.0 ],
                materialIndex: 3,
//              materialIndex: 3,
                perVertexFrontFaceNormalAvailable: false,
//              perVertexFrontFaceNormalAvailable: false,
            },
//          },
            // BACK
//          // BACK
            {
//          {
                aabb3d: undefined!,
//              aabb3d: undefined!,
                vertex0: [ -20.00, +20.00, -20.00 ],
//              vertex0: [ -20.00, +20.00, -20.00 ],
                vertex1: [ -20.00, -20.00, -20.00 ],
//              vertex1: [ -20.00, -20.00, -20.00 ],
                vertex2: [ +20.00, -20.00, -20.00 ],
//              vertex2: [ +20.00, -20.00, -20.00 ],
                vertex0FrontFaceNormal: [ 0.0, 0.0, 0.0 ],
//              vertex0FrontFaceNormal: [ 0.0, 0.0, 0.0 ],
                vertex1FrontFaceNormal: [ 0.0, 0.0, 0.0 ],
//              vertex1FrontFaceNormal: [ 0.0, 0.0, 0.0 ],
                vertex2FrontFaceNormal: [ 0.0, 0.0, 0.0 ],
//              vertex2FrontFaceNormal: [ 0.0, 0.0, 0.0 ],
                vertex0UV: [ 0.0, 0.0 ],
//              vertex0UV: [ 0.0, 0.0 ],
                vertex1UV: [ 0.0, 1.0 ],
//              vertex1UV: [ 0.0, 1.0 ],
                vertex2UV: [ 1.0, 1.0 ],
//              vertex2UV: [ 1.0, 1.0 ],
                materialIndex: 4,
//              materialIndex: 4,
                perVertexFrontFaceNormalAvailable: false,
//              perVertexFrontFaceNormalAvailable: false,
            },
//          },
            {
//          {
                aabb3d: undefined!,
//              aabb3d: undefined!,
                vertex0: [ +20.00, -20.00, -20.00 ],
//              vertex0: [ +20.00, -20.00, -20.00 ],
                vertex1: [ +20.00, +20.00, -20.00 ],
//              vertex1: [ +20.00, +20.00, -20.00 ],
                vertex2: [ -20.00, +20.00, -20.00 ],
//              vertex2: [ -20.00, +20.00, -20.00 ],
                vertex0FrontFaceNormal: [ 0.0, 0.0, 0.0 ],
//              vertex0FrontFaceNormal: [ 0.0, 0.0, 0.0 ],
                vertex1FrontFaceNormal: [ 0.0, 0.0, 0.0 ],
//              vertex1FrontFaceNormal: [ 0.0, 0.0, 0.0 ],
                vertex2FrontFaceNormal: [ 0.0, 0.0, 0.0 ],
//              vertex2FrontFaceNormal: [ 0.0, 0.0, 0.0 ],
                vertex0UV: [ 1.0, 1.0 ],
//              vertex0UV: [ 1.0, 1.0 ],
                vertex1UV: [ 1.0, 0.0 ],
//              vertex1UV: [ 1.0, 0.0 ],
                vertex2UV: [ 0.0, 0.0 ],
//              vertex2UV: [ 0.0, 0.0 ],
                materialIndex: 4,
//              materialIndex: 4,
                perVertexFrontFaceNormalAvailable: false,
//              perVertexFrontFaceNormalAvailable: false,
            },
//          },
            // LIGHT A (Corner -X, -Z, Size 10x10, Offset 3)
//          // LIGHT A (Corner -X, -Z, Size 10x10, Offset 3)
            {
//          {
                aabb3d: undefined!,
//              aabb3d: undefined!,
                vertex0: [ - 7.00, +19.99, - 7.00 ],
//              vertex0: [ - 7.00, +19.99, - 7.00 ],
                vertex1: [ - 7.00, +19.99, -17.00 ],
//              vertex1: [ - 7.00, +19.99, -17.00 ],
                vertex2: [ -17.00, +19.99, -17.00 ],
//              vertex2: [ -17.00, +19.99, -17.00 ],
                vertex0FrontFaceNormal: [ 0.0, 0.0, 0.0 ],
//              vertex0FrontFaceNormal: [ 0.0, 0.0, 0.0 ],
                vertex1FrontFaceNormal: [ 0.0, 0.0, 0.0 ],
//              vertex1FrontFaceNormal: [ 0.0, 0.0, 0.0 ],
                vertex2FrontFaceNormal: [ 0.0, 0.0, 0.0 ],
//              vertex2FrontFaceNormal: [ 0.0, 0.0, 0.0 ],
                vertex0UV: [ 0.0, 0.0 ],
//              vertex0UV: [ 0.0, 0.0 ],
                vertex1UV: [ 0.0, 1.0 ],
//              vertex1UV: [ 0.0, 1.0 ],
                vertex2UV: [ 1.0, 1.0 ],
//              vertex2UV: [ 1.0, 1.0 ],
                materialIndex: 7,
//              materialIndex: 7,
                perVertexFrontFaceNormalAvailable: false,
//              perVertexFrontFaceNormalAvailable: false,
            },
//          },
            {
//          {
                aabb3d: undefined!,
//              aabb3d: undefined!,
                vertex0: [ -17.00, +19.99, -17.00 ],
//              vertex0: [ -17.00, +19.99, -17.00 ],
                vertex1: [ -17.00, +19.99, - 7.00 ],
//              vertex1: [ -17.00, +19.99, - 7.00 ],
                vertex2: [ - 7.00, +19.99, - 7.00 ],
//              vertex2: [ - 7.00, +19.99, - 7.00 ],
                vertex0FrontFaceNormal: [ 0.0, 0.0, 0.0 ],
//              vertex0FrontFaceNormal: [ 0.0, 0.0, 0.0 ],
                vertex1FrontFaceNormal: [ 0.0, 0.0, 0.0 ],
//              vertex1FrontFaceNormal: [ 0.0, 0.0, 0.0 ],
                vertex2FrontFaceNormal: [ 0.0, 0.0, 0.0 ],
//              vertex2FrontFaceNormal: [ 0.0, 0.0, 0.0 ],
                vertex0UV: [ 1.0, 1.0 ],
//              vertex0UV: [ 1.0, 1.0 ],
                vertex1UV: [ 1.0, 0.0 ],
//              vertex1UV: [ 1.0, 0.0 ],
                vertex2UV: [ 0.0, 0.0 ],
//              vertex2UV: [ 0.0, 0.0 ],
                materialIndex: 7,
//              materialIndex: 7,
                perVertexFrontFaceNormalAvailable: false,
//              perVertexFrontFaceNormalAvailable: false,
            },
//          },
            // LIGHT B (Corner +X, +Z, Size 10x10, Offset 3)
//          // LIGHT B (Corner +X, +Z, Size 10x10, Offset 3)
            {
//          {
                aabb3d: undefined!,
//              aabb3d: undefined!,
                vertex0: [ + 7.00, +19.99, +17.00 ],
//              vertex0: [ + 7.00, +19.99, +17.00 ],
                vertex1: [ + 7.00, +19.99, + 7.00 ],
//              vertex1: [ + 7.00, +19.99, + 7.00 ],
                vertex2: [ +17.00, +19.99, + 7.00 ],
//              vertex2: [ +17.00, +19.99, + 7.00 ],
                vertex0FrontFaceNormal: [ 0.0, 0.0, 0.0 ],
//              vertex0FrontFaceNormal: [ 0.0, 0.0, 0.0 ],
                vertex1FrontFaceNormal: [ 0.0, 0.0, 0.0 ],
//              vertex1FrontFaceNormal: [ 0.0, 0.0, 0.0 ],
                vertex2FrontFaceNormal: [ 0.0, 0.0, 0.0 ],
//              vertex2FrontFaceNormal: [ 0.0, 0.0, 0.0 ],
                vertex0UV: [ 0.0, 0.0 ],
//              vertex0UV: [ 0.0, 0.0 ],
                vertex1UV: [ 0.0, 1.0 ],
//              vertex1UV: [ 0.0, 1.0 ],
                vertex2UV: [ 1.0, 1.0 ],
//              vertex2UV: [ 1.0, 1.0 ],
                materialIndex: 8,
//              materialIndex: 8,
                perVertexFrontFaceNormalAvailable: false,
//              perVertexFrontFaceNormalAvailable: false,
            },
//          },
            {
//          {
                aabb3d: undefined!,
//              aabb3d: undefined!,
                vertex0: [ +17.00, +19.99, + 7.00 ],
//              vertex0: [ +17.00, +19.99, + 7.00 ],
                vertex1: [ +17.00, +19.99, +17.00 ],
//              vertex1: [ +17.00, +19.99, +17.00 ],
                vertex2: [ + 7.00, +19.99, +17.00 ],
//              vertex2: [ + 7.00, +19.99, +17.00 ],
                vertex0FrontFaceNormal: [ 0.0, 0.0, 0.0 ],
//              vertex0FrontFaceNormal: [ 0.0, 0.0, 0.0 ],
                vertex1FrontFaceNormal: [ 0.0, 0.0, 0.0 ],
//              vertex1FrontFaceNormal: [ 0.0, 0.0, 0.0 ],
                vertex2FrontFaceNormal: [ 0.0, 0.0, 0.0 ],
//              vertex2FrontFaceNormal: [ 0.0, 0.0, 0.0 ],
                vertex0UV: [ 1.0, 1.0 ],
//              vertex0UV: [ 1.0, 1.0 ],
                vertex1UV: [ 1.0, 0.0 ],
//              vertex1UV: [ 1.0, 0.0 ],
                vertex2UV: [ 0.0, 0.0 ],
//              vertex2UV: [ 0.0, 0.0 ],
                materialIndex: 8,
//              materialIndex: 8,
                perVertexFrontFaceNormalAvailable: false,
//              perVertexFrontFaceNormalAvailable: false,
            },
//          },
            // LIGHT C (Corner -X, +Z, Size 10x10, Offset 3)
//          // LIGHT C (Corner -X, +Z, Size 10x10, Offset 3)
            {
//          {
                aabb3d: undefined!,
//              aabb3d: undefined!,
                vertex0: [ - 7.00, +19.99, +17.00 ],
//              vertex0: [ - 7.00, +19.99, +17.00 ],
                vertex1: [ - 7.00, +19.99, + 7.00 ],
//              vertex1: [ - 7.00, +19.99, + 7.00 ],
                vertex2: [ -17.00, +19.99, + 7.00 ],
//              vertex2: [ -17.00, +19.99, + 7.00 ],
                vertex0FrontFaceNormal: [ 0.0, 0.0, 0.0 ],
//              vertex0FrontFaceNormal: [ 0.0, 0.0, 0.0 ],
                vertex1FrontFaceNormal: [ 0.0, 0.0, 0.0 ],
//              vertex1FrontFaceNormal: [ 0.0, 0.0, 0.0 ],
                vertex2FrontFaceNormal: [ 0.0, 0.0, 0.0 ],
//              vertex2FrontFaceNormal: [ 0.0, 0.0, 0.0 ],
                vertex0UV: [ 0.0, 0.0 ],
//              vertex0UV: [ 0.0, 0.0 ],
                vertex1UV: [ 0.0, 1.0 ],
//              vertex1UV: [ 0.0, 1.0 ],
                vertex2UV: [ 1.0, 1.0 ],
//              vertex2UV: [ 1.0, 1.0 ],
                materialIndex: 9,
//              materialIndex: 9,
                perVertexFrontFaceNormalAvailable: false,
//              perVertexFrontFaceNormalAvailable: false,
            },
//          },
            {
//          {
                aabb3d: undefined!,
//              aabb3d: undefined!,
                vertex0: [ -17.00, +19.99, + 7.00 ],
//              vertex0: [ -17.00, +19.99, + 7.00 ],
                vertex1: [ -17.00, +19.99, +17.00 ],
//              vertex1: [ -17.00, +19.99, +17.00 ],
                vertex2: [ - 7.00, +19.99, +17.00 ],
//              vertex2: [ - 7.00, +19.99, +17.00 ],
                vertex0FrontFaceNormal: [ 0.0, 0.0, 0.0 ],
//              vertex0FrontFaceNormal: [ 0.0, 0.0, 0.0 ],
                vertex1FrontFaceNormal: [ 0.0, 0.0, 0.0 ],
//              vertex1FrontFaceNormal: [ 0.0, 0.0, 0.0 ],
                vertex2FrontFaceNormal: [ 0.0, 0.0, 0.0 ],
//              vertex2FrontFaceNormal: [ 0.0, 0.0, 0.0 ],
                vertex0UV: [ 1.0, 1.0 ],
//              vertex0UV: [ 1.0, 1.0 ],
                vertex1UV: [ 1.0, 0.0 ],
//              vertex1UV: [ 1.0, 0.0 ],
                vertex2UV: [ 0.0, 0.0 ],
//              vertex2UV: [ 0.0, 0.0 ],
                materialIndex: 9,
//              materialIndex: 9,
                perVertexFrontFaceNormalAvailable: false,
//              perVertexFrontFaceNormalAvailable: false,
            },
//          },
            // LIGHT D (Corner +X, -Z, Size 10x10, Offset 3)
//          // LIGHT D (Corner +X, -Z, Size 10x10, Offset 3)
            {
//          {
                aabb3d: undefined!,
//              aabb3d: undefined!,
                vertex0: [ + 7.00, +19.99, - 7.00 ],
//              vertex0: [ + 7.00, +19.99, - 7.00 ],
                vertex1: [ + 7.00, +19.99, -17.00 ],
//              vertex1: [ + 7.00, +19.99, -17.00 ],
                vertex2: [ +17.00, +19.99, -17.00 ],
//              vertex2: [ +17.00, +19.99, -17.00 ],
                vertex0FrontFaceNormal: [ 0.0, 0.0, 0.0 ],
//              vertex0FrontFaceNormal: [ 0.0, 0.0, 0.0 ],
                vertex1FrontFaceNormal: [ 0.0, 0.0, 0.0 ],
//              vertex1FrontFaceNormal: [ 0.0, 0.0, 0.0 ],
                vertex2FrontFaceNormal: [ 0.0, 0.0, 0.0 ],
//              vertex2FrontFaceNormal: [ 0.0, 0.0, 0.0 ],
                vertex0UV: [ 0.0, 0.0 ],
//              vertex0UV: [ 0.0, 0.0 ],
                vertex1UV: [ 0.0, 1.0 ],
//              vertex1UV: [ 0.0, 1.0 ],
                vertex2UV: [ 1.0, 1.0 ],
//              vertex2UV: [ 1.0, 1.0 ],
                materialIndex: 10,
//              materialIndex: 10,
                perVertexFrontFaceNormalAvailable: false,
//              perVertexFrontFaceNormalAvailable: false,
            },
//          },
            {
//          {
                aabb3d: undefined!,
//              aabb3d: undefined!,
                vertex0: [ +17.00, +19.99, -17.00 ],
//              vertex0: [ +17.00, +19.99, -17.00 ],
                vertex1: [ +17.00, +19.99, - 7.00 ],
//              vertex1: [ +17.00, +19.99, - 7.00 ],
                vertex2: [ + 7.00, +19.99, - 7.00 ],
//              vertex2: [ + 7.00, +19.99, - 7.00 ],
                vertex0FrontFaceNormal: [ 0.0, 0.0, 0.0 ],
//              vertex0FrontFaceNormal: [ 0.0, 0.0, 0.0 ],
                vertex1FrontFaceNormal: [ 0.0, 0.0, 0.0 ],
//              vertex1FrontFaceNormal: [ 0.0, 0.0, 0.0 ],
                vertex2FrontFaceNormal: [ 0.0, 0.0, 0.0 ],
//              vertex2FrontFaceNormal: [ 0.0, 0.0, 0.0 ],
                vertex0UV: [ 1.0, 1.0 ],
//              vertex0UV: [ 1.0, 1.0 ],
                vertex1UV: [ 1.0, 0.0 ],
//              vertex1UV: [ 1.0, 0.0 ],
                vertex2UV: [ 0.0, 0.0 ],
//              vertex2UV: [ 0.0, 0.0 ],
                materialIndex: 10,
//              materialIndex: 10,
                perVertexFrontFaceNormalAvailable: false,
//              perVertexFrontFaceNormalAvailable: false,
            },
//          },
        );
//      );
        _frameHandleRenderLoop = null!;
//      _frameHandleRenderLoop = null!;
    };
//  };



    const renderLoop = (cpuTimeInMilliseconds: number): void => {
//  const renderLoop = (cpuTimeInMilliseconds: number): void => {
        _timeInSeconds = cpuTimeInMilliseconds / 1000.0;
//      _timeInSeconds = cpuTimeInMilliseconds / 1000.0;
        startCPUTimeMeasurement(_cpuTimeMeasurementRenderLoop, cpuTimeInMilliseconds);
//      startCPUTimeMeasurement(_cpuTimeMeasurementRenderLoop, cpuTimeInMilliseconds);
        prepare();
//      prepare();
        render();
//      render();
        ceaseCPUTimeMeasurement(_cpuTimeMeasurementRenderLoop);
//      ceaseCPUTimeMeasurement(_cpuTimeMeasurementRenderLoop);
        _stratifiedSampleX++;
//      _stratifiedSampleX++;
        if (_stratifiedSampleX === _stratifiedSamplesPerPixel) {
//      if (_stratifiedSampleX === _stratifiedSamplesPerPixel) {
            _stratifiedSampleX = 0;
//          _stratifiedSampleX = 0;
            _stratifiedSampleY++;
//          _stratifiedSampleY++;
        }
//      }
        if (_stratifiedSampleY === _stratifiedSamplesPerPixel) {
//      if (_stratifiedSampleY === _stratifiedSamplesPerPixel) {
            _stratifiedSampleX = 0;
//          _stratifiedSampleX = 0;
            _stratifiedSampleY = 0;
//          _stratifiedSampleY = 0;
        }
//      }
        _accumulatedSampleCount++;
//      _accumulatedSampleCount++;
        _frameHandleRenderLoop = requestAnimationFrame(renderLoop);
//      _frameHandleRenderLoop = requestAnimationFrame(renderLoop);
    };
//  };
    const startRenderLoop = (): void => {
//  const startRenderLoop = (): void => {
        _stratifiedSampleX = 0;
//      _stratifiedSampleX = 0;
        _stratifiedSampleY = 0;
//      _stratifiedSampleY = 0;
        if (!_frameHandleRenderLoop) {
//      if (!_frameHandleRenderLoop) {
            _frameHandleRenderLoop = requestAnimationFrame(renderLoop);
//          _frameHandleRenderLoop = requestAnimationFrame(renderLoop);
        }
//      }
    };
//  };
    const stopRenderLoop = (): void => {
//  const stopRenderLoop = (): void => {
        if (_frameHandleRenderLoop) {
//      if (_frameHandleRenderLoop) {
            cancelAnimationFrame(_frameHandleRenderLoop);
//          cancelAnimationFrame(_frameHandleRenderLoop);
            _frameHandleRenderLoop = null!;
//          _frameHandleRenderLoop = null!;
        }
//      }
    };
//  };



    async function loadModels(): Promise<void> {
//  async function loadModels(): Promise<void> {
        const models = await assimp.customLoadFileObj(
//      const models = await assimp.customLoadFileObj(
            [
//          [
                {
//              {
                    name: "WoodToy_3.obj",
//                  name: "WoodToy_3.obj",
                    publicURL: "/test/WoodToy_3.obj",
//                  publicURL: "/test/WoodToy_3.obj",
                },
//              },
            ]
//          ]
        );
//      );
        const model = models[0];
//      const model = models[0];
        console.log(model);
//      console.log(model);
        if (model) {
//      if (model) {
            const floorY: number = -20.0;
//          const floorY: number = -20.0;
            let minModelY: number = Infinity;
//          let minModelY: number = Infinity;
            const scaleX: number = 2.1;
//          const scaleX: number = 2.1;
            const scaleY: number = 2.1;
//          const scaleY: number = 2.1;
            const scaleZ: number = 2.1;
//          const scaleZ: number = 2.1;
            for (let mesh of model["meshes"]) {
//          for (let mesh of model["meshes"]) {
                const l: number = (mesh["vertices"] as number[]).length;
//              const l: number = (mesh["vertices"] as number[]).length;
                for (let i: number = 0; i < l; i += 3) {
//              for (let i: number = 0; i < l; i += 3) {
                    mesh["vertices"][i + 0] *= scaleX;
//                  mesh["vertices"][i + 0] *= scaleX;
                    mesh["vertices"][i + 1] *= scaleY;
//                  mesh["vertices"][i + 1] *= scaleY;
                    mesh["vertices"][i + 2] *= scaleZ;
//                  mesh["vertices"][i + 2] *= scaleZ;
                }
//              }
            }
//          }
            for (let mesh of model["meshes"]) {
//          for (let mesh of model["meshes"]) {
                const l: number = (mesh["vertices"] as number[]).length;
//              const l: number = (mesh["vertices"] as number[]).length;
                for (let i: number = 0; i < l; i += 3) {
//              for (let i: number = 0; i < l; i += 3) {
                    let y: number = mesh["vertices"][i + 1];
//                  let y: number = mesh["vertices"][i + 1];
                    minModelY = Math.min(minModelY, y);
//                  minModelY = Math.min(minModelY, y);
                }
//              }
            }
//          }
            const offsetY: number = minModelY - floorY;
//          const offsetY: number = minModelY - floorY;
            for (let mesh of model["meshes"]) {
//          for (let mesh of model["meshes"]) {
                const meshVertices: number[] = mesh["vertices"];
//              const meshVertices: number[] = mesh["vertices"];
                const meshNormals: number[] = mesh["normals"];
//              const meshNormals: number[] = mesh["normals"];
                const meshTextureCoords: number[] = mesh["texturecoords"][0];
//              const meshTextureCoords: number[] = mesh["texturecoords"][0];
                for (let face of mesh["faces"]) {
//              for (let face of mesh["faces"]) {

                    const vertex0Index: number = face[0];
//                  const vertex0Index: number = face[0];
                    const vertex1Index: number = face[1];
//                  const vertex1Index: number = face[1];
                    const vertex2Index: number = face[2];
//                  const vertex2Index: number = face[2];

                    const vertex0IndexBase3: number = vertex0Index * 3; const vertex0ComponentX: number = vertex0IndexBase3 + 0; const vertex0ComponentY: number = vertex0IndexBase3 + 1; const vertex0ComponentZ: number = vertex0IndexBase3 + 2;
//                  const vertex0IndexBase3: number = vertex0Index * 3; const vertex0ComponentX: number = vertex0IndexBase3 + 0; const vertex0ComponentY: number = vertex0IndexBase3 + 1; const vertex0ComponentZ: number = vertex0IndexBase3 + 2;
                    const vertex1IndexBase3: number = vertex1Index * 3; const vertex1ComponentX: number = vertex1IndexBase3 + 0; const vertex1ComponentY: number = vertex1IndexBase3 + 1; const vertex1ComponentZ: number = vertex1IndexBase3 + 2;
//                  const vertex1IndexBase3: number = vertex1Index * 3; const vertex1ComponentX: number = vertex1IndexBase3 + 0; const vertex1ComponentY: number = vertex1IndexBase3 + 1; const vertex1ComponentZ: number = vertex1IndexBase3 + 2;
                    const vertex2IndexBase3: number = vertex2Index * 3; const vertex2ComponentX: number = vertex2IndexBase3 + 0; const vertex2ComponentY: number = vertex2IndexBase3 + 1; const vertex2ComponentZ: number = vertex2IndexBase3 + 2;
//                  const vertex2IndexBase3: number = vertex2Index * 3; const vertex2ComponentX: number = vertex2IndexBase3 + 0; const vertex2ComponentY: number = vertex2IndexBase3 + 1; const vertex2ComponentZ: number = vertex2IndexBase3 + 2;

                    const vertex0IndexBase2: number = vertex0Index * 2; const vertex0ComponentU: number = vertex0IndexBase2 + 0; const vertex0ComponentV: number = vertex0IndexBase2 + 1;
//                  const vertex0IndexBase2: number = vertex0Index * 2; const vertex0ComponentU: number = vertex0IndexBase2 + 0; const vertex0ComponentV: number = vertex0IndexBase2 + 1;
                    const vertex1IndexBase2: number = vertex1Index * 2; const vertex1ComponentU: number = vertex1IndexBase2 + 0; const vertex1ComponentV: number = vertex1IndexBase2 + 1;
//                  const vertex1IndexBase2: number = vertex1Index * 2; const vertex1ComponentU: number = vertex1IndexBase2 + 0; const vertex1ComponentV: number = vertex1IndexBase2 + 1;
                    const vertex2IndexBase2: number = vertex2Index * 2; const vertex2ComponentU: number = vertex2IndexBase2 + 0; const vertex2ComponentV: number = vertex2IndexBase2 + 1;
//                  const vertex2IndexBase2: number = vertex2Index * 2; const vertex2ComponentU: number = vertex2IndexBase2 + 0; const vertex2ComponentV: number = vertex2IndexBase2 + 1;

                    const vertex0Base3ComponentIndices: Vec3 = [ vertex0ComponentX, vertex0ComponentY, vertex0ComponentZ, ];
//                  const vertex0Base3ComponentIndices: Vec3 = [ vertex0ComponentX, vertex0ComponentY, vertex0ComponentZ, ];
                    const vertex1Base3ComponentIndices: Vec3 = [ vertex1ComponentX, vertex1ComponentY, vertex1ComponentZ, ];
//                  const vertex1Base3ComponentIndices: Vec3 = [ vertex1ComponentX, vertex1ComponentY, vertex1ComponentZ, ];
                    const vertex2Base3ComponentIndices: Vec3 = [ vertex2ComponentX, vertex2ComponentY, vertex2ComponentZ, ];
//                  const vertex2Base3ComponentIndices: Vec3 = [ vertex2ComponentX, vertex2ComponentY, vertex2ComponentZ, ];

                    const vertex0Base2ComponentIndices: Vec2 = [ vertex0ComponentU, vertex0ComponentV, ];
//                  const vertex0Base2ComponentIndices: Vec2 = [ vertex0ComponentU, vertex0ComponentV, ];
                    const vertex1Base2ComponentIndices: Vec2 = [ vertex1ComponentU, vertex1ComponentV, ];
//                  const vertex1Base2ComponentIndices: Vec2 = [ vertex1ComponentU, vertex1ComponentV, ];
                    const vertex2Base2ComponentIndices: Vec2 = [ vertex2ComponentU, vertex2ComponentV, ];
//                  const vertex2Base2ComponentIndices: Vec2 = [ vertex2ComponentU, vertex2ComponentV, ];

                    const vpo0: Vec3 = vertex0Base3ComponentIndices.map((componentIndex: number): number => meshVertices[componentIndex]) as Vec3;
//                  const vpo0: Vec3 = vertex0Base3ComponentIndices.map((componentIndex: number): number => meshVertices[componentIndex]) as Vec3;
                    const vpo1: Vec3 = vertex1Base3ComponentIndices.map((componentIndex: number): number => meshVertices[componentIndex]) as Vec3;
//                  const vpo1: Vec3 = vertex1Base3ComponentIndices.map((componentIndex: number): number => meshVertices[componentIndex]) as Vec3;
                    const vpo2: Vec3 = vertex2Base3ComponentIndices.map((componentIndex: number): number => meshVertices[componentIndex]) as Vec3;
//                  const vpo2: Vec3 = vertex2Base3ComponentIndices.map((componentIndex: number): number => meshVertices[componentIndex]) as Vec3;
                    const vno0: Vec3 = vertex0Base3ComponentIndices.map((componentIndex: number): number => meshNormals[componentIndex]) as Vec3;
//                  const vno0: Vec3 = vertex0Base3ComponentIndices.map((componentIndex: number): number => meshNormals[componentIndex]) as Vec3;
                    const vno1: Vec3 = vertex1Base3ComponentIndices.map((componentIndex: number): number => meshNormals[componentIndex]) as Vec3;
//                  const vno1: Vec3 = vertex1Base3ComponentIndices.map((componentIndex: number): number => meshNormals[componentIndex]) as Vec3;
                    const vno2: Vec3 = vertex2Base3ComponentIndices.map((componentIndex: number): number => meshNormals[componentIndex]) as Vec3;
//                  const vno2: Vec3 = vertex2Base3ComponentIndices.map((componentIndex: number): number => meshNormals[componentIndex]) as Vec3;
                    const vuv0: Vec2 = vertex0Base2ComponentIndices.map((componentIndex: number): number => meshTextureCoords[componentIndex]) as Vec2;
//                  const vuv0: Vec2 = vertex0Base2ComponentIndices.map((componentIndex: number): number => meshTextureCoords[componentIndex]) as Vec2;
                    const vuv1: Vec2 = vertex1Base2ComponentIndices.map((componentIndex: number): number => meshTextureCoords[componentIndex]) as Vec2;
//                  const vuv1: Vec2 = vertex1Base2ComponentIndices.map((componentIndex: number): number => meshTextureCoords[componentIndex]) as Vec2;
                    const vuv2: Vec2 = vertex2Base2ComponentIndices.map((componentIndex: number): number => meshTextureCoords[componentIndex]) as Vec2;
//                  const vuv2: Vec2 = vertex2Base2ComponentIndices.map((componentIndex: number): number => meshTextureCoords[componentIndex]) as Vec2;

                    vpo0[1] = vpo0[1] - offsetY;
//                  vpo0[1] = vpo0[1] - offsetY;
                    vpo1[1] = vpo1[1] - offsetY;
//                  vpo1[1] = vpo1[1] - offsetY;
                    vpo2[1] = vpo2[1] - offsetY;
//                  vpo2[1] = vpo2[1] - offsetY;

                    _triangles.push(
//                  _triangles.push(
                        {
//                      {
                            aabb3d: undefined!,
//                          aabb3d: undefined!,
                            vertex0: vpo0,
//                          vertex0: vpo0,
                            vertex1: vpo1,
//                          vertex1: vpo1,
                            vertex2: vpo2,
//                          vertex2: vpo2,
                            vertex0FrontFaceNormal: vno0,
//                          vertex0FrontFaceNormal: vno0,
                            vertex1FrontFaceNormal: vno1,
//                          vertex1FrontFaceNormal: vno1,
                            vertex2FrontFaceNormal: vno2,
//                          vertex2FrontFaceNormal: vno2,
                            vertex0UV: vuv0,
//                          vertex0UV: vuv0,
                            vertex1UV: vuv1,
//                          vertex1UV: vuv1,
                            vertex2UV: vuv2,
//                          vertex2UV: vuv2,
                            materialIndex: 6,
//                          materialIndex: 6,
                            perVertexFrontFaceNormalAvailable: true,
//                          perVertexFrontFaceNormalAvailable: true,
                        },
//                      },
                    );
//                  );
                    utility.rotateAroundPivotAndAxisTriangle(_triangles[_triangles.length - 1], [0.0, 0.0, 0.0], [0.0, 1.0, 0.0], m.pi / 4.0);
//                  utility.rotateAroundPivotAndAxisTriangle(_triangles[_triangles.length - 1], [0.0, 0.0, 0.0], [0.0, 1.0, 0.0], m.pi / 4.0);
                }
//              }
            }
//          }
        }
//      }
    };
//  };



    onMount(async (): Promise<void> => {
//  onMount(async (): Promise<void> => {
        await loadModels();
//      await loadModels();


        await initOnce();
//      await initOnce();


        prepareSpheres();
//      prepareSpheres();
        prepareMaterials();
//      prepareMaterials();
        prepareTextures();
//      prepareTextures();
        prepareTriangles();
//      prepareTriangles();


        let firstTimeDispatch: (() => void) | undefined = (): void => { startRenderLoop(); };
//      let firstTimeDispatch: (() => void) | undefined = (): void => { startRenderLoop(); };


        _resizeObserver = new ResizeObserver(
//      _resizeObserver = new ResizeObserver(
            (entries: ResizeObserverEntry[]) => {
//          (entries: ResizeObserverEntry[]) => {
                for (const entry of entries) {
//              for (const entry of entries) {
                    const entryAsCanvas: HTMLCanvasElement = entry.target as HTMLCanvasElement;
//                  const entryAsCanvas: HTMLCanvasElement = entry.target as HTMLCanvasElement;
                    if (!entryAsCanvas) {
//                  if (!entryAsCanvas) {
                        return;
//                      return;
                    }
//                  }
                    const width : number = entry.contentBoxSize[0].inlineSize;
//                  const width : number = entry.contentBoxSize[0].inlineSize;
                    const height: number = entry.contentBoxSize[0]. blockSize;
//                  const height: number = entry.contentBoxSize[0]. blockSize;
                    entryAsCanvas.width  = Math.max(
//                  entryAsCanvas.width  = Math.max(
                        1,
//                      1,
                        Math.min(width, _device.limits.maxTextureDimension2D),
//                      Math.min(width, _device.limits.maxTextureDimension2D),
                    );
//                  );
                    entryAsCanvas.height = Math.max(
//                  entryAsCanvas.height = Math.max(
                        1,
//                      1,
                        Math.min(height, _device.limits.maxTextureDimension2D),
//                      Math.min(height, _device.limits.maxTextureDimension2D),
                    );
//                  );
                    _viewportW = _viewportH * _canvas.width / _canvas.height;
//                  _viewportW = _viewportH * _canvas.width / _canvas.height;
                    _fromPixelToPixelDeltaU = m.divide(_viewportU, _canvas.width ) as Vec3;
//                  _fromPixelToPixelDeltaU = m.divide(_viewportU, _canvas.width ) as Vec3;
                    _fromPixelToPixelDeltaV = m.divide(_viewportV, _canvas.height) as Vec3;
//                  _fromPixelToPixelDeltaV = m.divide(_viewportV, _canvas.height) as Vec3;
                    if (_outputStorage) {
//                  if (_outputStorage) {
                        _outputStorage.destroy();
//                      _outputStorage.destroy();
                    }
//                  }
                    _outputStorage = _device.createBuffer({
//                  _outputStorage = _device.createBuffer({
                        label: "GPU_STORAGE_OUTPUT",
//                      label: "GPU_STORAGE_OUTPUT",
                        size: _canvas.width * _canvas.height * 16, // image width * image height * 16 bytes (a.k.a vec4<f32>)
//                      size: _canvas.width * _canvas.height * 16, // image width * image height * 16 bytes (a.k.a vec4<f32>)
                        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
//                      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
                    });
//                  });
                    if (_accumulatedOutputStorage) {
//                  if (_accumulatedOutputStorage) {
                        _accumulatedOutputStorage.destroy();
//                      _accumulatedOutputStorage.destroy();
                    }
//                  }
                    _accumulatedOutputStorage = _device.createBuffer({
//                  _accumulatedOutputStorage = _device.createBuffer({
                        label: "GPU_STORAGE_ACCUMULATED_OUTPUT",
//                      label: "GPU_STORAGE_ACCUMULATED_OUTPUT",
                        size: _canvas.width * _canvas.height * 16, // image width * image height * 16 bytes (a.k.a vec4<f32>)
//                      size: _canvas.width * _canvas.height * 16, // image width * image height * 16 bytes (a.k.a vec4<f32>)
                        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
//                      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
                    });
//                  });
                    if (_outputTexture) {
//                  if (_outputTexture) {
                        _outputTexture.destroy();
//                      _outputTexture.destroy();
                    }
//                  }
                    _outputTexture = _device.createTexture({
//                  _outputTexture = _device.createTexture({
                        label: "GPU_TEXTURE_OUTPUT",
//                      label: "GPU_TEXTURE_OUTPUT",
                        size: [ _canvas.width, _canvas.height, ],
//                      size: [ _canvas.width, _canvas.height, ],
                        format: "rgba32float",
//                      format: "rgba32float",
                        usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_SRC | GPUTextureUsage.STORAGE_BINDING,
//                      usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_SRC | GPUTextureUsage.STORAGE_BINDING,
                    });
//                  });
                    if (_blurTexturePing) {
//                  if (_blurTexturePing) {
                        _blurTexturePing.destroy();
//                      _blurTexturePing.destroy();
                    }
//                  }
                    _blurTexturePing = _device.createTexture({
//                  _blurTexturePing = _device.createTexture({
                        label: "GPU_TEXTURE_BLUR_PING",
//                      label: "GPU_TEXTURE_BLUR_PING",
                        size: [ _canvas.width, _canvas.height, ],
//                      size: [ _canvas.width, _canvas.height, ],
                        format: "rgba32float",
//                      format: "rgba32float",
                        usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST | GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.RENDER_ATTACHMENT,
//                      usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST | GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.RENDER_ATTACHMENT,
                    });
//                  });
                    if (_blurTexturePong) {
//                  if (_blurTexturePong) {
                        _blurTexturePong.destroy();
//                      _blurTexturePong.destroy();
                    }
//                  }
                    _blurTexturePong = _device.createTexture({
//                  _blurTexturePong = _device.createTexture({
                        label: "GPU_TEXTURE_BLUR_PONG",
//                      label: "GPU_TEXTURE_BLUR_PONG",
                        size: [ _canvas.width, _canvas.height, ],
//                      size: [ _canvas.width, _canvas.height, ],
                        format: "rgba32float",
//                      format: "rgba32float",
                        usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST | GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.RENDER_ATTACHMENT,
//                      usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST | GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.RENDER_ATTACHMENT,
                    });
//                  });
                    _computeBindGroup0 = _device.createBindGroup({
//                  _computeBindGroup0 = _device.createBindGroup({
                        label: "GPU_BIND_GROUP_0_COMPUTE",
//                      label: "GPU_BIND_GROUP_0_COMPUTE",
                        layout: _computePipeline.getBindGroupLayout(0),
//                      layout: _computePipeline.getBindGroupLayout(0),
                        entries: [
//                      entries: [
                            {
//                          {
                                binding: 0,
//                              binding: 0,
                                resource: _generalDataUniformBuffer,
//                              resource: _generalDataUniformBuffer,
                            },
//                          },
                            {
//                          {
                                binding: 1,
//                              binding: 1,
                                resource: _outputStorage,
//                              resource: _outputStorage,
                            },
//                          },
                            {
//                          {
                                binding: 2,
//                              binding: 2,
                                resource: _accumulatedOutputStorage,
//                              resource: _accumulatedOutputStorage,
                            },
//                          },
                            {
//                          {
                                binding: 3,
//                              binding: 3,
                                resource: _spheresStorageBuffer,
//                              resource: _spheresStorageBuffer,
                            },
//                          },
                            {
//                          {
                                binding: 4,
//                              binding: 4,
                                resource: _materialsStorageBuffer,
//                              resource: _materialsStorageBuffer,
                            },
//                          },
                            {
//                          {
                                binding: 5,
//                              binding: 5,
                                resource: _texturesStorageBuffer,
//                              resource: _texturesStorageBuffer,
                            },
//                          },
                            {
//                          {
                                binding: 6,
//                              binding: 6,
                                resource: _columnAtlasSampler,
//                              resource: _columnAtlasSampler,
                            },
//                          },
                            {
//                          {
                                binding: 7,
//                              binding: 7,
                                resource: _columnAtlasTexture,
//                              resource: _columnAtlasTexture,
                            },
//                          },
                            {
//                          {
                                binding: 8,
//                              binding: 8,
                                resource: _hdriSampler,
//                              resource: _hdriSampler,
                            },
//                          },
                            {
//                          {
                                binding: 9,
//                              binding: 9,
                                resource: _hdriTexture,
//                              resource: _hdriTexture,
                            },
//                          },
                            {
//                          {
                                binding: 10,
//                              binding: 10,
                                resource: _trianglesStorageBuffer,
//                              resource: _trianglesStorageBuffer,
                            },
//                          },
                            {
//                          {
                                binding: 11,
//                              binding: 11,
                                resource: _bvhNodesStorageBuffer,
//                              resource: _bvhNodesStorageBuffer,
                            },
//                          },
                        ] as Iterable<GPUBindGroupEntry>,
//                      ] as Iterable<GPUBindGroupEntry>,
                    });
//                  });
                    _gatherBindGroup0 = _device.createBindGroup({
//                  _gatherBindGroup0 = _device.createBindGroup({
                        label: "GPU_BIND_GROUP_0_GATHER",
//                      label: "GPU_BIND_GROUP_0_GATHER",
                        layout: _gatherPipeline.getBindGroupLayout(0),
//                      layout: _gatherPipeline.getBindGroupLayout(0),
                        entries: [
//                      entries: [
                            {
//                          {
                                binding: 0,
//                              binding: 0,
                                resource: _generalDataUniformBuffer,
//                              resource: _generalDataUniformBuffer,
                            },
//                          },
                            {
//                          {
                                binding: 1,
//                              binding: 1,
                                resource: _outputStorage,
//                              resource: _outputStorage,
                            },
//                          },
                            {
//                          {
                                binding: 2,
//                              binding: 2,
                                resource: _outputTexture,
//                              resource: _outputTexture,
                            },
//                          },
                            {
//                          {
                                binding: 3,
//                              binding: 3,
                                resource: _blurTexturePing,
//                              resource: _blurTexturePing,
                            },
//                          },
                        ] as Iterable<GPUBindGroupEntry>,
//                      ] as Iterable<GPUBindGroupEntry>,
                    });
//                  });
                    _blur2XBindGroup0 = _device.createBindGroup({
//                  _blur2XBindGroup0 = _device.createBindGroup({
                        label: "GPU_BIND_GROUP_0_BLUR_2X",
//                      label: "GPU_BIND_GROUP_0_BLUR_2X",
                        layout: _blur2XPipeline.getBindGroupLayout(0),
//                      layout: _blur2XPipeline.getBindGroupLayout(0),
                        entries: [
//                      entries: [
                            {
//                          {
                                binding: 0,
//                              binding: 0,
                                resource: _blurSampler,
//                              resource: _blurSampler,
                            },
//                          },
                            {
//                          {
                                binding: 1,
//                              binding: 1,
                                resource: _blurTexturePing,
//                              resource: _blurTexturePing,
                            },
//                          },
                            {
//                          {
                                binding: 2,
//                              binding: 2,
                                resource: _generalDataUniformBuffer,
//                              resource: _generalDataUniformBuffer,
                            },
//                          },
                        ] as Iterable<GPUBindGroupEntry>,
//                      ] as Iterable<GPUBindGroupEntry>,
                    });
//                  });
                    _blur4XBindGroup0 = _device.createBindGroup({
//                  _blur4XBindGroup0 = _device.createBindGroup({
                        label: "GPU_BIND_GROUP_0_BLUR_4X",
//                      label: "GPU_BIND_GROUP_0_BLUR_4X",
                        layout: _blur4XPipeline.getBindGroupLayout(0),
//                      layout: _blur4XPipeline.getBindGroupLayout(0),
                        entries: [
//                      entries: [
                            {
//                          {
                                binding: 0,
//                              binding: 0,
                                resource: _blurSampler,
//                              resource: _blurSampler,
                            },
//                          },
                            {
//                          {
                                binding: 1,
//                              binding: 1,
                                resource: _blurTexturePong,
//                              resource: _blurTexturePong,
                            },
//                          },
                            {
//                          {
                                binding: 2,
//                              binding: 2,
                                resource: _generalDataUniformBuffer,
//                              resource: _generalDataUniformBuffer,
                            },
//                          },
                        ] as Iterable<GPUBindGroupEntry>,
//                      ] as Iterable<GPUBindGroupEntry>,
                    });
//                  });
                    _blur8XBindGroup0 = _device.createBindGroup({
//                  _blur8XBindGroup0 = _device.createBindGroup({
                        label: "GPU_BIND_GROUP_0_BLUR_8X",
//                      label: "GPU_BIND_GROUP_0_BLUR_8X",
                        layout: _blur8XPipeline.getBindGroupLayout(0),
//                      layout: _blur8XPipeline.getBindGroupLayout(0),
                        entries: [
//                      entries: [
                            {
//                          {
                                binding: 0,
//                              binding: 0,
                                resource: _blurSampler,
//                              resource: _blurSampler,
                            },
//                          },
                            {
//                          {
                                binding: 1,
//                              binding: 1,
                                resource: _blurTexturePing,
//                              resource: _blurTexturePing,
                            },
//                          },
                            {
//                          {
                                binding: 2,
//                              binding: 2,
                                resource: _generalDataUniformBuffer,
//                              resource: _generalDataUniformBuffer,
                            },
//                          },
                        ] as Iterable<GPUBindGroupEntry>,
//                      ] as Iterable<GPUBindGroupEntry>,
                    });
//                  });
                    _renderBindGroup0 = _device.createBindGroup({
//                  _renderBindGroup0 = _device.createBindGroup({
                        label: "GPU_BIND_GROUP_0_RENDER",
//                      label: "GPU_BIND_GROUP_0_RENDER",
                        layout: _renderPipeline.getBindGroupLayout(0),
//                      layout: _renderPipeline.getBindGroupLayout(0),
                        entries: [
//                      entries: [
                            {
//                          {
                                binding: 0,
//                              binding: 0,
                                resource: _outputSampler,
//                              resource: _outputSampler,
                            },
//                          },
                            {
//                          {
                                binding: 1,
//                              binding: 1,
                                resource: _outputTexture,
//                              resource: _outputTexture,
                            },
//                          },
                            {
//                          {
                                binding: 2,
//                              binding: 2,
                                resource: _generalDataUniformBuffer,
//                              resource: _generalDataUniformBuffer,
                            },
//                          },
                            {
//                          {
                                binding: 3,
//                              binding: 3,
                                resource: _blurSampler,
//                              resource: _blurSampler,
                            },
//                          },
                            {
//                          {
                                binding: 4,
//                              binding: 4,
                                resource: _blurTexturePong,
//                              resource: _blurTexturePong,
                            },
//                          },
                        ] as Iterable<GPUBindGroupEntry>,
//                      ] as Iterable<GPUBindGroupEntry>,
                    });
//                  });
                }
//              }
                if (firstTimeDispatch) {
//              if (firstTimeDispatch) {
                    firstTimeDispatch();
//                  firstTimeDispatch();
                    firstTimeDispatch = undefined;
//                  firstTimeDispatch = undefined;
                }
//              }
            },
//          },
        );
//      );


        _resizeObserver.observe(_canvas);
//      _resizeObserver.observe(_canvas);
    });
//  });



    onDestroy(async (): Promise<void> => {
//  onDestroy(async (): Promise<void> => {
        stopRenderLoop();
//      stopRenderLoop();
        if (_blurTexturePong) {
//      if (_blurTexturePong) {
            _blurTexturePong.destroy();
//          _blurTexturePong.destroy();
        }
//      }
        if (_blurTexturePing) {
//      if (_blurTexturePing) {
            _blurTexturePing.destroy();
//          _blurTexturePing.destroy();
        }
//      }
        if (_outputTexture) {
//      if (_outputTexture) {
            _outputTexture.destroy();
//          _outputTexture.destroy();
        }
//      }
        if (_accumulatedOutputStorage) {
//      if (_accumulatedOutputStorage) {
            _accumulatedOutputStorage.destroy();
//          _accumulatedOutputStorage.destroy();
        }
//      }
        if (_outputStorage) {
//      if (_outputStorage) {
            _outputStorage.destroy();
//          _outputStorage.destroy();
        }
//      }
        if (_hdriTexture) {
//      if (_hdriTexture) {
            _hdriTexture.destroy();
//          _hdriTexture.destroy();
        }
//      }
        if (_columnAtlasTexture) {
//      if (_columnAtlasTexture) {
            _columnAtlasTexture.destroy();
//          _columnAtlasTexture.destroy();
        }
//      }
        if (_texturesStorageBuffer) {
//      if (_texturesStorageBuffer) {
            _texturesStorageBuffer.destroy();
//          _texturesStorageBuffer.destroy();
        }
//      }
        if (_materialsStorageBuffer) {
//      if (_materialsStorageBuffer) {
            _materialsStorageBuffer.destroy();
//          _materialsStorageBuffer.destroy();
        }
//      }
        if (_spheresStorageBuffer) {
//      if (_spheresStorageBuffer) {
            _spheresStorageBuffer.destroy();
//          _spheresStorageBuffer.destroy();
        }
//      }
        if (_generalDataUniformBuffer) {
//      if (_generalDataUniformBuffer) {
            _generalDataUniformBuffer.destroy();
//          _generalDataUniformBuffer.destroy();
        }
//      }
        if (_device) {
//      if (_device) {
            _device.destroy();
//          _device.destroy();
        }
//      }
        _resizeObserver.unobserve(_canvas);
//      _resizeObserver.unobserve(_canvas);
        _canvas = null!;
//      _canvas = null!;
    });
//  });



    const render = (): void => {
//  const render = (): void => {
        (_blur2XPassDescriptor.colorAttachments as GPURenderPassColorAttachment[])[0].view = _blurTexturePong;
//      (_blur2XPassDescriptor.colorAttachments as GPURenderPassColorAttachment[])[0].view = _blurTexturePong;
        (_blur4XPassDescriptor.colorAttachments as GPURenderPassColorAttachment[])[0].view = _blurTexturePing;
//      (_blur4XPassDescriptor.colorAttachments as GPURenderPassColorAttachment[])[0].view = _blurTexturePing;
        (_blur8XPassDescriptor.colorAttachments as GPURenderPassColorAttachment[])[0].view = _blurTexturePong;
//      (_blur8XPassDescriptor.colorAttachments as GPURenderPassColorAttachment[])[0].view = _blurTexturePong;
        (_renderPassDescriptor.colorAttachments as GPURenderPassColorAttachment[])[0].view = _canvasContext.getCurrentTexture();
//      (_renderPassDescriptor.colorAttachments as GPURenderPassColorAttachment[])[0].view = _canvasContext.getCurrentTexture();
        _commandEncoder = _device.createCommandEncoder(_commandEncoderDescriptor);
//      _commandEncoder = _device.createCommandEncoder(_commandEncoderDescriptor);


        _computePass = _commandEncoder.beginComputePass(_computePassDescriptor);
//      _computePass = _commandEncoder.beginComputePass(_computePassDescriptor);
        _computePass.setPipeline(_computePipeline);
//      _computePass.setPipeline(_computePipeline);
        _computePass.setBindGroup(0, _computeBindGroup0);
//      _computePass.setBindGroup(0, _computeBindGroup0);
        _computePass.dispatchWorkgroups(Math.ceil(_canvas.width / 32), Math.ceil(_canvas.height / 32),);
//      _computePass.dispatchWorkgroups(Math.ceil(_canvas.width / 32), Math.ceil(_canvas.height / 32),);
        _computePass.end();
//      _computePass.end();


        _gatherPass = _commandEncoder.beginComputePass(_gatherPassDescriptor);
//      _gatherPass = _commandEncoder.beginComputePass(_gatherPassDescriptor);
        _gatherPass.setPipeline(_gatherPipeline);
//      _gatherPass.setPipeline(_gatherPipeline);
        _gatherPass.setBindGroup(0, _gatherBindGroup0);
//      _gatherPass.setBindGroup(0, _gatherBindGroup0);
        _gatherPass.dispatchWorkgroups(Math.ceil(_canvas.width / 32), Math.ceil(_canvas.height / 32),);
//      _gatherPass.dispatchWorkgroups(Math.ceil(_canvas.width / 32), Math.ceil(_canvas.height / 32),);
        _gatherPass.end();
//      _gatherPass.end();


        _blur2XPass = _commandEncoder.beginRenderPass(_blur2XPassDescriptor);
//      _blur2XPass = _commandEncoder.beginRenderPass(_blur2XPassDescriptor);
        _blur2XPass.setPipeline(_blur2XPipeline);
//      _blur2XPass.setPipeline(_blur2XPipeline);
        _blur2XPass.setBindGroup(0, _blur2XBindGroup0);
//      _blur2XPass.setBindGroup(0, _blur2XBindGroup0);
        _blur2XPass.draw(6);
//      _blur2XPass.draw(6);
        _blur2XPass.end();
//      _blur2XPass.end();


        _blur4XPass = _commandEncoder.beginRenderPass(_blur4XPassDescriptor);
//      _blur4XPass = _commandEncoder.beginRenderPass(_blur4XPassDescriptor);
        _blur4XPass.setPipeline(_blur4XPipeline);
//      _blur4XPass.setPipeline(_blur4XPipeline);
        _blur4XPass.setBindGroup(0, _blur4XBindGroup0);
//      _blur4XPass.setBindGroup(0, _blur4XBindGroup0);
        _blur4XPass.draw(6);
//      _blur4XPass.draw(6);
        _blur4XPass.end();
//      _blur4XPass.end();


        _blur8XPass = _commandEncoder.beginRenderPass(_blur8XPassDescriptor);
//      _blur8XPass = _commandEncoder.beginRenderPass(_blur8XPassDescriptor);
        _blur8XPass.setPipeline(_blur8XPipeline);
//      _blur8XPass.setPipeline(_blur8XPipeline);
        _blur8XPass.setBindGroup(0, _blur8XBindGroup0);
//      _blur8XPass.setBindGroup(0, _blur8XBindGroup0);
        _blur8XPass.draw(6);
//      _blur8XPass.draw(6);
        _blur8XPass.end();
//      _blur8XPass.end();


        _renderPass = _commandEncoder.beginRenderPass(_renderPassDescriptor);
//      _renderPass = _commandEncoder.beginRenderPass(_renderPassDescriptor);
        _renderPass.setPipeline(_renderPipeline);
//      _renderPass.setPipeline(_renderPipeline);
        _renderPass.setBindGroup(0, _renderBindGroup0);
//      _renderPass.setBindGroup(0, _renderBindGroup0);
        _renderPass.draw(6);
//      _renderPass.draw(6);
        _renderPass.end();
//      _renderPass.end();


        _commandBuffer = _commandEncoder.finish();
//      _commandBuffer = _commandEncoder.finish();
        _device.queue.submit([_commandBuffer]);
//      _device.queue.submit([_commandBuffer]);
    };
//  };



    const prepare = (): void => {
//  const prepare = (): void => {

        /*
        console.log("_stratifiedSampleX:", _stratifiedSampleX, "_stratifiedSampleY:", _stratifiedSampleY);
//      console.log("_stratifiedSampleX:", _stratifiedSampleX, "_stratifiedSampleY:", _stratifiedSampleY);
        */

        _generalDataUniformValuesDataView.setUint32(0, _canvas.width, true);
//      _generalDataUniformValuesDataView.setUint32(0, _canvas.width, true);
        _generalDataUniformValuesDataView.setUint32(4, _canvas.height, true);
//      _generalDataUniformValuesDataView.setUint32(4, _canvas.height, true);
        _generalDataUniformValuesDataView.setFloat32(8, _stratifiedSamplesPerPixel, true);
//      _generalDataUniformValuesDataView.setFloat32(8, _stratifiedSamplesPerPixel, true);
        _generalDataUniformValuesDataView.setFloat32(12, _inverseStratifiedSamplesPerPixel, true);
//      _generalDataUniformValuesDataView.setFloat32(12, _inverseStratifiedSamplesPerPixel, true);
        _generalDataUniformValuesDataView.setFloat32(16, _cameraCenter[0], true);
//      _generalDataUniformValuesDataView.setFloat32(16, _cameraCenter[0], true);
        _generalDataUniformValuesDataView.setFloat32(20, _cameraCenter[1], true);
//      _generalDataUniformValuesDataView.setFloat32(20, _cameraCenter[1], true);
        _generalDataUniformValuesDataView.setFloat32(24, _cameraCenter[2], true);
//      _generalDataUniformValuesDataView.setFloat32(24, _cameraCenter[2], true);
        _generalDataUniformValuesDataView.setFloat32(28, _pixelSamplesScale, true);
//      _generalDataUniformValuesDataView.setFloat32(28, _pixelSamplesScale, true);
        _generalDataUniformValuesDataView.setFloat32(32, _fromPixelToPixelDeltaU[0], true);
//      _generalDataUniformValuesDataView.setFloat32(32, _fromPixelToPixelDeltaU[0], true);
        _generalDataUniformValuesDataView.setFloat32(36, _fromPixelToPixelDeltaU[1], true);
//      _generalDataUniformValuesDataView.setFloat32(36, _fromPixelToPixelDeltaU[1], true);
        _generalDataUniformValuesDataView.setFloat32(40, _fromPixelToPixelDeltaU[2], true);
//      _generalDataUniformValuesDataView.setFloat32(40, _fromPixelToPixelDeltaU[2], true);
        _generalDataUniformValuesDataView.setFloat32(44, _stratifiedSampleX, true);
//      _generalDataUniformValuesDataView.setFloat32(44, _stratifiedSampleX, true);
        _generalDataUniformValuesDataView.setFloat32(48, _fromPixelToPixelDeltaV[0], true);
//      _generalDataUniformValuesDataView.setFloat32(48, _fromPixelToPixelDeltaV[0], true);
        _generalDataUniformValuesDataView.setFloat32(52, _fromPixelToPixelDeltaV[1], true);
//      _generalDataUniformValuesDataView.setFloat32(52, _fromPixelToPixelDeltaV[1], true);
        _generalDataUniformValuesDataView.setFloat32(56, _fromPixelToPixelDeltaV[2], true);
//      _generalDataUniformValuesDataView.setFloat32(56, _fromPixelToPixelDeltaV[2], true);
        _generalDataUniformValuesDataView.setFloat32(60, _stratifiedSampleY, true);
//      _generalDataUniformValuesDataView.setFloat32(60, _stratifiedSampleY, true);
        _generalDataUniformValuesDataView.setFloat32(64, _pixel00Coordinates[0], true);
//      _generalDataUniformValuesDataView.setFloat32(64, _pixel00Coordinates[0], true);
        _generalDataUniformValuesDataView.setFloat32(68, _pixel00Coordinates[1], true);
//      _generalDataUniformValuesDataView.setFloat32(68, _pixel00Coordinates[1], true);
        _generalDataUniformValuesDataView.setFloat32(72, _pixel00Coordinates[2], true);
//      _generalDataUniformValuesDataView.setFloat32(72, _pixel00Coordinates[2], true);
        _generalDataUniformValuesDataView.setUint32(76, _backgroundType, true);
//      _generalDataUniformValuesDataView.setUint32(76, _backgroundType, true);
        _generalDataUniformValuesDataView.setUint32(80, _numberOfImages, true);
//      _generalDataUniformValuesDataView.setUint32(80, _numberOfImages, true);
        _generalDataUniformValuesDataView.setFloat32(84, _timeInSeconds, true);
//      _generalDataUniformValuesDataView.setFloat32(84, _timeInSeconds, true);
        _generalDataUniformValuesDataView.setUint32(88, _accumulatedSampleCount, true);
//      _generalDataUniformValuesDataView.setUint32(88, _accumulatedSampleCount, true);
        _generalDataUniformValuesDataView.setUint32(92, 0, true);
//      _generalDataUniformValuesDataView.setUint32(92, 0, true);
        _generalDataUniformValuesDataView.setFloat32( 96, _defocusDiskRadiusU[0], true);
//      _generalDataUniformValuesDataView.setFloat32( 96, _defocusDiskRadiusU[0], true);
        _generalDataUniformValuesDataView.setFloat32(100, _defocusDiskRadiusU[1], true);
//      _generalDataUniformValuesDataView.setFloat32(100, _defocusDiskRadiusU[1], true);
        _generalDataUniformValuesDataView.setFloat32(104, _defocusDiskRadiusU[2], true);
//      _generalDataUniformValuesDataView.setFloat32(104, _defocusDiskRadiusU[2], true);
        _generalDataUniformValuesDataView.setUint32(108, 0, true);
//      _generalDataUniformValuesDataView.setUint32(108, 0, true);
        _generalDataUniformValuesDataView.setFloat32(112, _defocusDiskRadiusV[0], true);
//      _generalDataUniformValuesDataView.setFloat32(112, _defocusDiskRadiusV[0], true);
        _generalDataUniformValuesDataView.setFloat32(116, _defocusDiskRadiusV[1], true);
//      _generalDataUniformValuesDataView.setFloat32(116, _defocusDiskRadiusV[1], true);
        _generalDataUniformValuesDataView.setFloat32(120, _defocusDiskRadiusV[2], true);
//      _generalDataUniformValuesDataView.setFloat32(120, _defocusDiskRadiusV[2], true);
        _generalDataUniformValuesDataView.setUint32(124, 0, true);
//      _generalDataUniformValuesDataView.setUint32(124, 0, true);

        _device.queue.writeBuffer(_generalDataUniformBuffer, 0, _generalDataUniformValues as GPUAllowSharedBufferSource);
//      _device.queue.writeBuffer(_generalDataUniformBuffer, 0, _generalDataUniformValues as GPUAllowSharedBufferSource);

    };
//  };



    function moveCamera(newLookFrom: Vec3, newLookAt: Vec3, newViewUp: Vec3): void {
//  function moveCamera(newLookFrom: Vec3, newLookAt: Vec3, newViewUp: Vec3): void {
        _accumulatedSampleCount = 0;
//      _accumulatedSampleCount = 0;
        _lookFrom = newLookFrom;
//      _lookFrom = newLookFrom;
        _lookAt = newLookAt;
//      _lookAt = newLookAt;
        _viewUp = newViewUp;
//      _viewUp = newViewUp;
        _viewportW = _viewportH * _canvas.width / _canvas.height;
//      _viewportW = _viewportH * _canvas.width / _canvas.height;
        _fromPixelToPixelDeltaU = m.divide(_viewportU, _canvas.width ) as Vec3;
//      _fromPixelToPixelDeltaU = m.divide(_viewportU, _canvas.width ) as Vec3;
        _fromPixelToPixelDeltaV = m.divide(_viewportV, _canvas.height) as Vec3;
//      _fromPixelToPixelDeltaV = m.divide(_viewportV, _canvas.height) as Vec3;
    };
//  };



    const OnKeydown = async (keyboardEvent: KeyboardEvent): Promise<void> => {
//  const OnKeydown = async (keyboardEvent: KeyboardEvent): Promise<void> => {
        if (keyboardEvent.key === "0") {
//      if (keyboardEvent.key === "0") {
            tweenCamera([0.0, 0.0, +46.0], [0.0, 0.0, 0.0], [0.0, 1.0, 0.0], 1000, 1000, 1000, t.Easing.Quartic.InOut, t.Easing.Quartic.InOut, t.Easing.Quartic.InOut);
//          tweenCamera([0.0, 0.0, +46.0], [0.0, 0.0, 0.0], [0.0, 1.0, 0.0], 1000, 1000, 1000, t.Easing.Quartic.InOut, t.Easing.Quartic.InOut, t.Easing.Quartic.InOut);
        } else if (keyboardEvent.key === "1") {
//      } else if (keyboardEvent.key === "1") {
            tweenCamera([0.0, 0.0, +40.0], [0.0, 0.0, 0.0], [0.0, 1.0, 0.0], 1000, 1000, 1000, t.Easing.Quartic.InOut, t.Easing.Quartic.InOut, t.Easing.Quartic.InOut);
//          tweenCamera([0.0, 0.0, +40.0], [0.0, 0.0, 0.0], [0.0, 1.0, 0.0], 1000, 1000, 1000, t.Easing.Quartic.InOut, t.Easing.Quartic.InOut, t.Easing.Quartic.InOut);
        } else if (keyboardEvent.key === "2") {
//      } else if (keyboardEvent.key === "2") {
            tweenCamera([+28.284, 0.0, +28.284], [0.0, 0.0, 0.0], [0.0, 1.0, 0.0], 1000, 1000, 1000, t.Easing.Quartic.InOut, t.Easing.Quartic.InOut, t.Easing.Quartic.InOut);
//          tweenCamera([+28.284, 0.0, +28.284], [0.0, 0.0, 0.0], [0.0, 1.0, 0.0], 1000, 1000, 1000, t.Easing.Quartic.InOut, t.Easing.Quartic.InOut, t.Easing.Quartic.InOut);
        } else if (keyboardEvent.key === "3") {
//      } else if (keyboardEvent.key === "3") {
            tweenCamera([+40.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 1.0, 0.0], 1000, 1000, 1000, t.Easing.Quartic.InOut, t.Easing.Quartic.InOut, t.Easing.Quartic.InOut);
//          tweenCamera([+40.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 1.0, 0.0], 1000, 1000, 1000, t.Easing.Quartic.InOut, t.Easing.Quartic.InOut, t.Easing.Quartic.InOut);
        } else if (keyboardEvent.key === "4") {
//      } else if (keyboardEvent.key === "4") {
            tweenCamera([+28.284, 0.0, -28.284], [0.0, 0.0, 0.0], [0.0, 1.0, 0.0], 1000, 1000, 1000, t.Easing.Quartic.InOut, t.Easing.Quartic.InOut, t.Easing.Quartic.InOut);
//          tweenCamera([+28.284, 0.0, -28.284], [0.0, 0.0, 0.0], [0.0, 1.0, 0.0], 1000, 1000, 1000, t.Easing.Quartic.InOut, t.Easing.Quartic.InOut, t.Easing.Quartic.InOut);
        } else if (keyboardEvent.key === "5") {
//      } else if (keyboardEvent.key === "5") {
            tweenCamera([0.0, 0.0, -40.0], [0.0, 0.0, 0.0], [0.0, 1.0, 0.0], 1000, 1000, 1000, t.Easing.Quartic.InOut, t.Easing.Quartic.InOut, t.Easing.Quartic.InOut);
//          tweenCamera([0.0, 0.0, -40.0], [0.0, 0.0, 0.0], [0.0, 1.0, 0.0], 1000, 1000, 1000, t.Easing.Quartic.InOut, t.Easing.Quartic.InOut, t.Easing.Quartic.InOut);
        } else if (keyboardEvent.key === "6") {
//      } else if (keyboardEvent.key === "6") {
            tweenCamera([-28.284, 0.0, -28.284], [0.0, 0.0, 0.0], [0.0, 1.0, 0.0], 1000, 1000, 1000, t.Easing.Quartic.InOut, t.Easing.Quartic.InOut, t.Easing.Quartic.InOut);
//          tweenCamera([-28.284, 0.0, -28.284], [0.0, 0.0, 0.0], [0.0, 1.0, 0.0], 1000, 1000, 1000, t.Easing.Quartic.InOut, t.Easing.Quartic.InOut, t.Easing.Quartic.InOut);
        } else if (keyboardEvent.key === "7") {
//      } else if (keyboardEvent.key === "7") {
            tweenCamera([-40.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 1.0, 0.0], 1000, 1000, 1000, t.Easing.Quartic.InOut, t.Easing.Quartic.InOut, t.Easing.Quartic.InOut);
//          tweenCamera([-40.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 1.0, 0.0], 1000, 1000, 1000, t.Easing.Quartic.InOut, t.Easing.Quartic.InOut, t.Easing.Quartic.InOut);
        } else if (keyboardEvent.key === "8") {
//      } else if (keyboardEvent.key === "8") {
            tweenCamera([-28.284, 0.0, +28.284], [0.0, 0.0, 0.0], [0.0, 1.0, 0.0], 1000, 1000, 1000, t.Easing.Quartic.InOut, t.Easing.Quartic.InOut, t.Easing.Quartic.InOut);
//          tweenCamera([-28.284, 0.0, +28.284], [0.0, 0.0, 0.0], [0.0, 1.0, 0.0], 1000, 1000, 1000, t.Easing.Quartic.InOut, t.Easing.Quartic.InOut, t.Easing.Quartic.InOut);
        } else if (keyboardEvent.key === "9") {
//      } else if (keyboardEvent.key === "9") {
            tweenCamera([0.0, +40.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 1.0], 1000, 1000, 1000, t.Easing.Quartic.InOut, t.Easing.Quartic.InOut, t.Easing.Sinusoidal.Out);
//          tweenCamera([0.0, +40.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 1.0], 1000, 1000, 1000, t.Easing.Quartic.InOut, t.Easing.Quartic.InOut, t.Easing.Sinusoidal.Out);
        }
//      }

        const speed: number = 10.0;
//      const speed: number = 10.0;
        const n = (v: Vec3): Vec3 => m.divide(v, m.norm(v)) as Vec3;
//      const n = (v: Vec3): Vec3 => m.divide(v, m.norm(v)) as Vec3;
        const directionForward: Vec3 = n(m.subtract(_lookAt, _lookFrom));
//      const directionForward: Vec3 = n(m.subtract(_lookAt, _lookFrom));
        const directionBackward: Vec3 = m.rotate(directionForward, m.pi, [ 0.0, 1.0, 0.0 ]);
//      const directionBackward: Vec3 = m.rotate(directionForward, m.pi, [ 0.0, 1.0, 0.0 ]);
        const directionLeft: Vec3 = m.rotate(directionForward, m.pi / 2.0, [ 0.0, 1.0, 0.0 ]);
//      const directionLeft: Vec3 = m.rotate(directionForward, m.pi / 2.0, [ 0.0, 1.0, 0.0 ]);
        const directionRight: Vec3 = m.rotate(directionForward, -m.pi / 2.0, [ 0.0, 1.0, 0.0 ]);
//      const directionRight: Vec3 = m.rotate(directionForward, -m.pi / 2.0, [ 0.0, 1.0, 0.0 ]);
        const f: Vec3 = m.multiply(directionForward, speed) as Vec3;
//      const f: Vec3 = m.multiply(directionForward, speed) as Vec3;
        const b: Vec3 = m.multiply(directionBackward, speed) as Vec3;
//      const b: Vec3 = m.multiply(directionBackward, speed) as Vec3;
        const l: Vec3 = m.multiply(directionLeft, speed) as Vec3;
//      const l: Vec3 = m.multiply(directionLeft, speed) as Vec3;
        const r: Vec3 = m.multiply(directionRight, speed) as Vec3;
//      const r: Vec3 = m.multiply(directionRight, speed) as Vec3;

        if (keyboardEvent.key === "w") {
//      if (keyboardEvent.key === "w") {
            tweenCamera(m.add(_lookFrom, f), m.add(_lookAt, f), _viewUp, 1000, 1000, 1000, t.Easing.Quartic.InOut, t.Easing.Quartic.InOut, t.Easing.Quartic.InOut);
//          tweenCamera(m.add(_lookFrom, f), m.add(_lookAt, f), _viewUp, 1000, 1000, 1000, t.Easing.Quartic.InOut, t.Easing.Quartic.InOut, t.Easing.Quartic.InOut);
        } else if (keyboardEvent.key === "a") {
//      } else if (keyboardEvent.key === "a") {
            tweenCamera(m.add(_lookFrom, l), m.add(_lookAt, l), _viewUp, 1000, 1000, 1000, t.Easing.Quartic.InOut, t.Easing.Quartic.InOut, t.Easing.Quartic.InOut);
//          tweenCamera(m.add(_lookFrom, l), m.add(_lookAt, l), _viewUp, 1000, 1000, 1000, t.Easing.Quartic.InOut, t.Easing.Quartic.InOut, t.Easing.Quartic.InOut);
        } else if (keyboardEvent.key === "s") {
//      } else if (keyboardEvent.key === "s") {
            tweenCamera(m.add(_lookFrom, b), m.add(_lookAt, b), _viewUp, 1000, 1000, 1000, t.Easing.Quartic.InOut, t.Easing.Quartic.InOut, t.Easing.Quartic.InOut);
//          tweenCamera(m.add(_lookFrom, b), m.add(_lookAt, b), _viewUp, 1000, 1000, 1000, t.Easing.Quartic.InOut, t.Easing.Quartic.InOut, t.Easing.Quartic.InOut);
        } else if (keyboardEvent.key === "d") {
//      } else if (keyboardEvent.key === "d") {
            tweenCamera(m.add(_lookFrom, r), m.add(_lookAt, r), _viewUp, 1000, 1000, 1000, t.Easing.Quartic.InOut, t.Easing.Quartic.InOut, t.Easing.Quartic.InOut);
//          tweenCamera(m.add(_lookFrom, r), m.add(_lookAt, r), _viewUp, 1000, 1000, 1000, t.Easing.Quartic.InOut, t.Easing.Quartic.InOut, t.Easing.Quartic.InOut);
        } else if (keyboardEvent.key === "q") {
//      } else if (keyboardEvent.key === "q") {
            tweenCamera(m.add(_lookFrom, [ 0, +10, 0 ]), m.add(_lookAt, [ 0, +10, 0 ]), _viewUp, 1000, 1000, 1000, t.Easing.Quartic.InOut, t.Easing.Quartic.InOut, t.Easing.Quartic.InOut);
//          tweenCamera(m.add(_lookFrom, [ 0, +10, 0 ]), m.add(_lookAt, [ 0, +10, 0 ]), _viewUp, 1000, 1000, 1000, t.Easing.Quartic.InOut, t.Easing.Quartic.InOut, t.Easing.Quartic.InOut);
        } else if (keyboardEvent.key === "e") {
//      } else if (keyboardEvent.key === "e") {
            tweenCamera(m.add(_lookFrom, [ 0, -10, 0 ]), m.add(_lookAt, [ 0, -10, 0 ]), _viewUp, 1000, 1000, 1000, t.Easing.Quartic.InOut, t.Easing.Quartic.InOut, t.Easing.Quartic.InOut);
//          tweenCamera(m.add(_lookFrom, [ 0, -10, 0 ]), m.add(_lookAt, [ 0, -10, 0 ]), _viewUp, 1000, 1000, 1000, t.Easing.Quartic.InOut, t.Easing.Quartic.InOut, t.Easing.Quartic.InOut);
        }
//      }
    };
//  };



    function generalLoop(cpuTimeInMilliseconds: number): void {
//  function generalLoop(cpuTimeInMilliseconds: number): void {
        startCPUTimeMeasurement(_cpuTimeMeasurementGeneralLoop, cpuTimeInMilliseconds);
//      startCPUTimeMeasurement(_cpuTimeMeasurementGeneralLoop, cpuTimeInMilliseconds);
        if (_tweenCameraLookFrom) {
//      if (_tweenCameraLookFrom) {
            _tweenCameraLookFrom.update(cpuTimeInMilliseconds);
//          _tweenCameraLookFrom.update(cpuTimeInMilliseconds);
        }
//      }
        if (_tweenCameraLookAt) {
//      if (_tweenCameraLookAt) {
            _tweenCameraLookAt.update(cpuTimeInMilliseconds);
//          _tweenCameraLookAt.update(cpuTimeInMilliseconds);
        }
//      }
        if (_tweenCameraViewUp) {
//      if (_tweenCameraViewUp) {
            _tweenCameraViewUp.update(cpuTimeInMilliseconds);
//          _tweenCameraViewUp.update(cpuTimeInMilliseconds);
        }
//      }
        ceaseCPUTimeMeasurement(_cpuTimeMeasurementGeneralLoop);
//      ceaseCPUTimeMeasurement(_cpuTimeMeasurementGeneralLoop);
        requestAnimationFrame(generalLoop);
//      requestAnimationFrame(generalLoop);
    }
//  }
    requestAnimationFrame(generalLoop);
//  requestAnimationFrame(generalLoop);



    function tweenCamera(newLookFrom: Vec3, newLookAt: Vec3, newViewUp: Vec3, durationTweenLookFrom: number, durationTweenLookAt: number, durationTweenViewUp: number, easingFunctionTweenLookFrom: EasingFunction, easingFunctionTweenLookAt: EasingFunction, easingFunctionTweenViewUp: EasingFunction): void {
//  function tweenCamera(newLookFrom: Vec3, newLookAt: Vec3, newViewUp: Vec3, durationTweenLookFrom: number, durationTweenLookAt: number, durationTweenViewUp: number, easingFunctionTweenLookFrom: EasingFunction, easingFunctionTweenLookAt: EasingFunction, easingFunctionTweenViewUp: EasingFunction): void {
        if (!compareArrays(newLookFrom, _lookFrom, isEqual) && (!_tweenCameraLookFrom || !_tweenCameraLookFromIsBusy)) {
//      if (!compareArrays(newLookFrom, _lookFrom, isEqual) && (!_tweenCameraLookFrom || !_tweenCameraLookFromIsBusy)) {
            const lookFrom: Vec3 = _lookFrom;
//          const lookFrom: Vec3 = _lookFrom;
            _tweenCameraLookFrom = new t.Tween(lookFrom)
//          _tweenCameraLookFrom = new t.Tween(lookFrom)
                .to(newLookFrom, durationTweenLookFrom)
//              .to(newLookFrom, durationTweenLookFrom)
                .easing(easingFunctionTweenLookFrom)
//              .easing(easingFunctionTweenLookFrom)
                .onStart((object: Vec3): void => {
//              .onStart((object: Vec3): void => {
                    _tweenCameraLookFromIsBusy = true;
//                  _tweenCameraLookFromIsBusy = true;
                })
//              })
                .onUpdate((object: Vec3, elapsed: number): void => {
//              .onUpdate((object: Vec3, elapsed: number): void => {
                    moveCamera(object, _lookAt, _viewUp);
//                  moveCamera(object, _lookAt, _viewUp);
                })
//              })
                .onStop((object: Vec3): void => {
//              .onStop((object: Vec3): void => {
                    _tweenCameraLookFromIsBusy = false;
//                  _tweenCameraLookFromIsBusy = false;
                })
//              })
                .onComplete((object: Vec3): void => {
//              .onComplete((object: Vec3): void => {
                    _tweenCameraLookFromIsBusy = false;
//                  _tweenCameraLookFromIsBusy = false;
                })
//              })
                .start();
//              .start();
        }
//      }
        if (!compareArrays(newLookAt, _lookAt, isEqual) && (!_tweenCameraLookAt || !_tweenCameraLookAtIsBusy)) {
//      if (!compareArrays(newLookAt, _lookAt, isEqual) && (!_tweenCameraLookAt || !_tweenCameraLookAtIsBusy)) {
            const lookAt: Vec3 = _lookAt;
//          const lookAt: Vec3 = _lookAt;
            _tweenCameraLookAt = new t.Tween(lookAt)
//          _tweenCameraLookAt = new t.Tween(lookAt)
                .to(newLookAt, durationTweenLookAt)
//              .to(newLookAt, durationTweenLookAt)
                .easing(easingFunctionTweenLookAt)
//              .easing(easingFunctionTweenLookAt)
                .onStart((object: Vec3): void => {
//              .onStart((object: Vec3): void => {
                    _tweenCameraLookAtIsBusy = true;
//                  _tweenCameraLookAtIsBusy = true;
                })
//              })
                .onUpdate((object: Vec3, elapsed: number): void => {
//              .onUpdate((object: Vec3, elapsed: number): void => {
                    moveCamera(_lookFrom, object, _viewUp);
//                  moveCamera(_lookFrom, object, _viewUp);
                })
//              })
                .onStop((object: Vec3): void => {
//              .onStop((object: Vec3): void => {
                    _tweenCameraLookAtIsBusy = false;
//                  _tweenCameraLookAtIsBusy = false;
                })
//              })
                .onComplete((object: Vec3): void => {
//              .onComplete((object: Vec3): void => {
                    _tweenCameraLookAtIsBusy = false;
//                  _tweenCameraLookAtIsBusy = false;
                })
//              })
                .start();
//              .start();
        }
//      }
        if (!compareArrays(newViewUp, _viewUp, isEqual) && (!_tweenCameraViewUp || !_tweenCameraViewUpIsBusy)) {
//      if (!compareArrays(newViewUp, _viewUp, isEqual) && (!_tweenCameraViewUp || !_tweenCameraViewUpIsBusy)) {
            const viewUp: Vec3 = _viewUp;
//          const viewUp: Vec3 = _viewUp;
            _tweenCameraViewUp = new t.Tween(viewUp)
//          _tweenCameraViewUp = new t.Tween(viewUp)
                .to(newViewUp, durationTweenViewUp)
//              .to(newViewUp, durationTweenViewUp)
                .easing(easingFunctionTweenViewUp)
//              .easing(easingFunctionTweenViewUp)
                .onStart((object: Vec3): void => {
//              .onStart((object: Vec3): void => {
                    _tweenCameraViewUpIsBusy = true;
//                  _tweenCameraViewUpIsBusy = true;
                })
//              })
                .onUpdate((object: Vec3, elapsed: number): void => {
//              .onUpdate((object: Vec3, elapsed: number): void => {
                    moveCamera(_lookFrom, _lookAt, object);
//                  moveCamera(_lookFrom, _lookAt, object);
                })
//              })
                .onStop((object: Vec3): void => {
//              .onStop((object: Vec3): void => {
                    _tweenCameraViewUpIsBusy = false;
//                  _tweenCameraViewUpIsBusy = false;
                })
//              })
                .onComplete((object: Vec3): void => {
//              .onComplete((object: Vec3): void => {
                    _tweenCameraViewUpIsBusy = false;
//                  _tweenCameraViewUpIsBusy = false;
                })
//              })
                .start();
//              .start();
        }
//      }
    }
//  }



</script>

<!--<svelte:window on:keydown={OnKeydown} />-->
    <svelte:window on:keydown={OnKeydown} />
<!--<svelte:window on:keydown={OnKeydown} />-->

<!--<canvas class="large-elevate" bind:this={_canvas} width="960px" height="540px" style:width="960px" style:height="540px" style:display="block"></canvas>-->
    <canvas class="large-elevate" bind:this={_canvas} width="960px" height="540px" style:width="960px" style:height="540px" style:display="block"></canvas>
<!--<canvas class="large-elevate" bind:this={_canvas} width="960px" height="540px" style:width="960px" style:height="540px" style:display="block"></canvas>-->

<!--<span>{_cpuTimeMeasurementRenderLoop.fps}</span>-->
<!--<span>{_cpuTimeMeasurementRenderLoop.fps}</span>-->
<!--<span>{_cpuTimeMeasurementRenderLoop.fps}</span>-->

<!--<svelte:options runes={true}></svelte:options>-->
    <svelte:options runes={true}></svelte:options>
<!--<svelte:options runes={true}></svelte:options>-->
