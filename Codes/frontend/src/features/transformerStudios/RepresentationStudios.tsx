// src/features/transformerStudios/RepresentationStudios.tsx
import React, { useState, useEffect, useRef, useMemo } from 'react';
import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import {
  ResponsiveContainer,
  LineChart,
  Line,
  XAxis,
  YAxis,
  Tooltip as RechartsTooltip,
  CartesianGrid,
  Legend,
  BarChart,
  Bar,
  AreaChart,
  Area,
} from 'recharts';
import {
  Compass,
  Layers,
  Activity,
  Maximize2,
  Minimize2,
  Sparkles,
  Sliders,
  Play,
  Pause,
  RotateCcw,
  Search,
  Zap,
  Target,
  ArrowRight,
  TrendingUp,
  Cpu,
  Flame,
  CheckCircle2,
  Info,
  Scale,
  Shield,
  HelpCircle,
} from 'lucide-react';
import { Badge } from '../../components/ui/Badge';
import { Button } from '../../components/ui/Button';
import type {
  StudioTab,
  TokenNode3D,
  SemanticAxis3D,
  RoPEDimPair,
  ResidualLayerState,
  Vector2DSandboxToken,
  LayerNormLayerStat,
  ModelPreset,
} from './types';

// =====================================================================
// Model Presets
// =====================================================================
const MODEL_PRESETS: ModelPreset[] = [
  {
    id: 'llama3-8b',
    name: 'LLaMA-3 (8B)',
    family: 'Meta AI',
    dModel: 4096,
    numLayers: 32,
    numHeads: 32,
    vocabSize: 128256,
    normType: 'RMSNorm',
    posType: 'RoPE',
    baseFrequency: 500000,
  },
  {
    id: 'gpt4-preview',
    name: 'GPT-4 (Transformer)',
    family: 'OpenAI',
    dModel: 12288,
    numLayers: 96,
    numHeads: 96,
    vocabSize: 100256,
    normType: 'LayerNorm',
    posType: 'RoPE',
    baseFrequency: 10000,
  },
  {
    id: 'mistral-7b',
    name: 'Mistral (7B-v0.3)',
    family: 'Mistral AI',
    dModel: 4096,
    numLayers: 32,
    numHeads: 32,
    vocabSize: 32768,
    normType: 'RMSNorm',
    posType: 'RoPE',
    baseFrequency: 1000000,
  },
  {
    id: 'bert-base',
    name: 'BERT-Base (Uncased)',
    family: 'Google',
    dModel: 768,
    numLayers: 12,
    numHeads: 12,
    vocabSize: 30522,
    normType: 'LayerNorm',
    posType: 'Sinusoidal',
    baseFrequency: 10000,
  },
  {
    id: 'toy-2d',
    name: 'Visual Sandbox (d=2)',
    family: 'Stanford Research',
    dModel: 2,
    numLayers: 6,
    numHeads: 2,
    vocabSize: 64,
    normType: 'LayerNorm',
    posType: 'RoPE',
    baseFrequency: 10000,
  },
];

// =====================================================================
// Sample Mock Data Generators
// =====================================================================
const SAMPLE_TOKENS = [
  'Paris', 'France', 'capital', 'city', 'Europe', 'Rome', 'Italy', 'London', 'UK',
  'king', 'queen', 'monarch', 'throne', 'palace', 'man', 'woman', 'prince', 'princess',
  'neural', 'attention', 'transformer', 'vector', 'matrix', 'tensor', 'gradient', 'layer',
  'happy', 'joyful', 'delighted', 'sad', 'sorrowful', 'gloomy', 'angry', 'peaceful'
];

function generateTokenNodes(tokens: string[], method: 'pca' | 'tsne' | 'umap'): TokenNode3D[] {
  return tokens.map((token, i) => {
    const cluster = i < 9 ? 0 : i < 18 ? 1 : i < 26 ? 2 : 3;
    const clusterCenters = [
      [-2.5, 1.2, 0.5],
      [2.2, 2.0, -1.2],
      [0.2, -2.4, 1.8],
      [-1.8, -1.5, -2.0],
    ];
    const center = clusterCenters[cluster];
    const jitter = () => (Math.random() - 0.5) * (method === 'tsne' ? 0.9 : 1.4);

    return {
      id: `tok-${i}`,
      token,
      x: center[0] + jitter(),
      y: center[1] + jitter(),
      z: center[2] + jitter(),
      cluster,
      norm: 2.5 + Math.random() * 1.5,
      color: ['#38bdf8', '#fbbf24', '#34d399', '#f472b6'][cluster],
    };
  });
}

function generateRoPEDimPairs(dModel: number, baseFreq: number, currentPos: number): RoPEDimPair[] {
  const count = Math.min(16, Math.floor(dModel / 2));
  return Array.from({ length: count }, (_, i) => {
    const freq = 1.0 / Math.pow(baseFreq, (2 * i) / Math.min(64, dModel));
    const angleRad = currentPos * freq;
    const angleDeg = ((angleRad * 180) / Math.PI) % 360;
    const wavelength = (2 * Math.PI) / freq;
    return {
      pairIndex: i,
      dimRange: [2 * i, 2 * i + 1],
      frequency: freq,
      angleRad,
      angleDeg,
      wavelength,
      cosVal: Math.cos(angleRad),
      sinVal: Math.sin(angleRad),
    };
  });
}

function generateResidualTraces(numLayers: number): ResidualLayerState[] {
  return Array.from({ length: numLayers }, (_, l) => {
    const hNorm = 3.0 + l * 0.45 + Math.sin(l * 0.5) * 0.2;
    const attnNorm = 0.8 + Math.sin(l * 0.8) * 0.35 + 0.1;
    const ffnNorm = 1.1 + Math.cos(l * 0.6) * 0.4 + 0.2;
    const totalNorm = hNorm + attnNorm * 0.6 + ffnNorm * 0.5;
    const attnFrac = attnNorm / (attnNorm + ffnNorm);
    const cosSim = 0.2 + Math.sin(l * 0.4) * 0.3;

    return {
      layer: l,
      hiddenNorm: Number(hNorm.toFixed(2)),
      attnNorm: Number(attnNorm.toFixed(2)),
      ffnNorm: Number(ffnNorm.toFixed(2)),
      totalNorm: Number(totalNorm.toFixed(2)),
      attnFraction: Number(attnFrac.toFixed(2)),
      cosSimAttnFfn: Number(cosSim.toFixed(2)),
      topLogits: [
        { token: 'Paris', probability: Math.min(0.92, 0.05 + l * 0.08) },
        { token: 'France', probability: Math.max(0.04, 0.4 - l * 0.03) },
        { token: 'city', probability: Math.max(0.02, 0.25 - l * 0.02) },
      ],
    };
  });
}

function generateLayerNormStats(numLayers: number): LayerNormLayerStat[] {
  return Array.from({ length: numLayers }, (_, l) => {
    const preMean = (Math.sin(l * 0.9) * 0.35).toFixed(3);
    const preVar = (1.5 + l * 0.25 + Math.cos(l) * 0.3).toFixed(3);
    const preRms = Math.sqrt(Number(preVar) + Number(preMean) ** 2).toFixed(3);
    const postMean = (0.001 * (Math.random() - 0.5)).toFixed(4);
    const postVar = (1.002 + 0.003 * Math.sin(l)).toFixed(4);
    const postRms = (1.001).toFixed(4);
    const cosSim = (0.85 + 0.12 * Math.cos(l * 0.3)).toFixed(3);
    const gammaDrift = (0.15 + l * 0.08).toFixed(3);
    const betaDrift = (0.08 + l * 0.04).toFixed(3);

    return {
      layer: l,
      preMean: Number(preMean),
      preVar: Number(preVar),
      preRms: Number(preRms),
      postMean: Number(postMean),
      postVar: Number(postVar),
      postRms: Number(postRms),
      cosSim: Number(cosSim),
      gammaDrift: Number(gammaDrift),
      betaDrift: Number(betaDrift),
    };
  });
}

// =====================================================================
// Main Component
// =====================================================================
export const RepresentationStudios: React.FC = () => {
  const [activeTab, setActiveTab] = useState<StudioTab>('embeddings');
  const [selectedModel, setSelectedModel] = useState<ModelPreset>(MODEL_PRESETS[0]);
  const [inputText, setInputText] = useState('The capital of France is Paris and the king sat on the royal throne');
  const [isFullscreen, setIsFullscreen] = useState(false);

  // Embedding Explorer States
  const [projectionMethod, setProjectionMethod] = useState<'pca' | 'tsne' | 'umap'>('pca');
  const [searchToken, setSearchToken] = useState('');
  const [selectedToken, setSelectedToken] = useState<TokenNode3D | null>(null);
  const [selectedAxis, setSelectedAxis] = useState<string>('sentiment');
  const [probeClassifying, setProbeClassifying] = useState(false);

  // Positional Encoding States
  const [currentPosition, setCurrentPosition] = useState(4);
  const [isPlayingRoPE, setIsPlayingRoPE] = useState(false);
  const [selectedDimPairIdx, setSelectedDimPairIdx] = useState(0);

  // Residual Stream States
  const [selectedResidualLayer, setSelectedResidualLayer] = useState(4);
  const [isSandboxRunning, setIsSandboxRunning] = useState(false);
  const [sandboxStep, setSandboxStep] = useState(2);

  // LayerNorm States
  const [selectedLNLayer, setSelectedLNLayer] = useState(2);
  const [customGamma, setCustomGamma] = useState(1.2);
  const [customBeta, setCustomBeta] = useState(0.1);
  const [normViewMode, setNormViewMode] = useState<'both' | 'layernorm' | 'rmsnorm'>('both');

  // Three.js Canvas Refs
  const embeddingCanvasRef = useRef<HTMLDivElement>(null);
  const ropeCanvasRef = useRef<HTMLDivElement>(null);
  const residualCanvasRef = useRef<HTMLDivElement>(null);
  const hypersphereCanvasRef = useRef<HTMLDivElement>(null);

  // Derived Data
  const tokenNodes = useMemo(() => generateTokenNodes(SAMPLE_TOKENS, projectionMethod), [projectionMethod]);
  const ropePairs = useMemo(
    () => generateRoPEDimPairs(selectedModel.dModel, selectedModel.baseFrequency, currentPosition),
    [selectedModel, currentPosition]
  );
  const residualTraces = useMemo(
    () => generateResidualTraces(selectedModel.numLayers),
    [selectedModel.numLayers]
  );
  const layerNormStats = useMemo(
    () => generateLayerNormStats(selectedModel.numLayers),
    [selectedModel.numLayers]
  );

  // -------------------------------------------------------------------
  // 1. Three.js Scene: Embedding Space Explorer
  // -------------------------------------------------------------------
  useEffect(() => {
    if (activeTab !== 'embeddings' || !embeddingCanvasRef.current) return;
    const container = embeddingCanvasRef.current;
    container.innerHTML = '';

    const width = container.clientWidth || 600;
    const height = container.clientHeight || 420;

    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x020617); // slate-950

    const camera = new THREE.PerspectiveCamera(50, width / height, 0.1, 1000);
    camera.position.set(6, 5, 8);

    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    renderer.setSize(width, height);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    container.appendChild(renderer.domElement);

    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.05;

    // Ambient & Point Lights
    const ambientLight = new THREE.AmbientLight(0xffffff, 0.7);
    scene.add(ambientLight);
    const pointLight = new THREE.PointLight(0x38bdf8, 2, 50);
    pointLight.position.set(10, 10, 10);
    scene.add(pointLight);

    // Grid helper
    const grid = new THREE.GridHelper(10, 10, 0x1e293b, 0x0f172a);
    grid.position.y = -3.5;
    scene.add(grid);

    // Axes lines
    const axesHelper = new THREE.AxesHelper(4);
    scene.add(axesHelper);

    // Spheres for tokens
    const sphereGeo = new THREE.SphereGeometry(0.18, 16, 16);
    const spheres: THREE.Mesh[] = [];

    tokenNodes.forEach((node) => {
      const isHighlighted = selectedToken?.id === node.id || searchToken.toLowerCase() === node.token.toLowerCase();
      const material = new THREE.MeshStandardMaterial({
        color: new THREE.Color(node.color || '#38bdf8'),
        emissive: isHighlighted ? new THREE.Color(0xffffff) : new THREE.Color(node.color || '#38bdf8'),
        emissiveIntensity: isHighlighted ? 0.9 : 0.25,
        roughness: 0.3,
        metalness: 0.2,
      });

      const mesh = new THREE.Mesh(sphereGeo, material);
      mesh.position.set(node.x, node.y, node.z);
      mesh.userData = node;
      scene.add(mesh);
      spheres.push(mesh);
    });

    // Semantic Axis 3D Arrow
    const axisDir = selectedAxis === 'sentiment'
      ? new THREE.Vector3(1, 0.6, -0.4).normalize()
      : selectedAxis === 'royalty'
      ? new THREE.Vector3(-0.4, 1.2, 0.6).normalize()
      : new THREE.Vector3(0.5, -0.8, 1.0).normalize();
    const arrow = new THREE.ArrowHelper(axisDir, new THREE.Vector3(0, 0, 0), 4.5, 0xf43f5e, 0.4, 0.25);
    scene.add(arrow);
    const arrowNeg = new THREE.ArrowHelper(axisDir.clone().negate(), new THREE.Vector3(0, 0, 0), 4.5, 0x0ea5e9, 0.4, 0.25);
    scene.add(arrowNeg);

    // Trajectory curve between 4 tokens
    if (tokenNodes.length >= 4) {
      const curvePoints = tokenNodes.slice(0, 5).map((n) => new THREE.Vector3(n.x, n.y, n.z));
      const curve = new THREE.CatmullRomCurve3(curvePoints);
      const tubeGeo = new THREE.TubeGeometry(curve, 32, 0.04, 8, false);
      const tubeMat = new THREE.MeshStandardMaterial({
        color: 0xa855f7,
        emissive: 0xa855f7,
        emissiveIntensity: 0.6,
        transparent: true,
        opacity: 0.8,
      });
      const tubeMesh = new THREE.Mesh(tubeGeo, tubeMat);
      scene.add(tubeMesh);
    }

    let animationFrameId: number;
    const animate = () => {
      animationFrameId = requestAnimationFrame(animate);
      controls.update();
      renderer.render(scene, camera);
    };
    animate();

    const handleResize = () => {
      if (!container) return;
      const w = container.clientWidth;
      const h = container.clientHeight;
      camera.aspect = w / h;
      camera.updateProjectionMatrix();
      renderer.setSize(w, h);
    };
    window.addEventListener('resize', handleResize);

    return () => {
      cancelAnimationFrame(animationFrameId);
      window.removeEventListener('resize', handleResize);
      renderer.dispose();
      container.innerHTML = '';
    };
  }, [activeTab, tokenNodes, selectedToken, searchToken, selectedAxis]);

  // -------------------------------------------------------------------
  // 2. Three.js Scene: RoPE 3D Angle Visualizer
  // -------------------------------------------------------------------
  useEffect(() => {
    if (activeTab !== 'positional' || !ropeCanvasRef.current) return;
    const container = ropeCanvasRef.current;
    container.innerHTML = '';

    const width = container.clientWidth || 600;
    const height = container.clientHeight || 420;

    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x020617);

    const camera = new THREE.PerspectiveCamera(50, width / height, 0.1, 1000);
    camera.position.set(0, 4, 7);

    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    renderer.setSize(width, height);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    container.appendChild(renderer.domElement);

    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;

    const ambientLight = new THREE.AmbientLight(0xffffff, 0.8);
    scene.add(ambientLight);
    const dirLight = new THREE.DirectionalLight(0x38bdf8, 1.5);
    dirLight.position.set(5, 10, 7);
    scene.add(dirLight);

    // Dimension Pair Rings
    const ringGroup = new THREE.Group();
    ropePairs.slice(0, 6).forEach((pair, idx) => {
      const radius = 1.2 + idx * 0.45;
      const ringGeo = new THREE.TorusGeometry(radius, 0.02, 16, 64);
      const ringMat = new THREE.MeshBasicMaterial({
        color: new THREE.Color(idx === selectedDimPairIdx ? 0xf59e0b : 0x38bdf8),
        transparent: true,
        opacity: idx === selectedDimPairIdx ? 0.9 : 0.35,
      });
      const ring = new THREE.Mesh(ringGeo, ringMat);
      ring.rotation.x = Math.PI / 2;
      ringGroup.add(ring);

      // Rotating vector arrow on this ring
      const angle = pair.angleRad;
      const vx = Math.cos(angle) * radius;
      const vz = Math.sin(angle) * radius;
      const arrowDir = new THREE.Vector3(vx, 0, vz).normalize();
      const arrow = new THREE.ArrowHelper(
        arrowDir,
        new THREE.Vector3(0, 0, 0),
        radius,
        idx === selectedDimPairIdx ? 0xf59e0b : 0x06b6d4,
        0.2,
        0.12
      );
      ringGroup.add(arrow);
    });
    scene.add(ringGroup);

    // Parametric rotation helix tube
    const helixPoints: THREE.Vector3[] = [];
    for (let p = 0; p <= 30; p++) {
      const angle = p * ropePairs[selectedDimPairIdx].frequency * 3.0;
      const r = 2.0;
      const y = (p - 15) * 0.15;
      helixPoints.push(new THREE.Vector3(Math.cos(angle) * r, y, Math.sin(angle) * r));
    }
    const helixCurve = new THREE.CatmullRomCurve3(helixPoints);
    const helixGeo = new THREE.TubeGeometry(helixCurve, 64, 0.03, 8, false);
    const helixMat = new THREE.MeshStandardMaterial({
      color: 0xec4899,
      emissive: 0xec4899,
      emissiveIntensity: 0.5,
    });
    const helixMesh = new THREE.Mesh(helixGeo, helixMat);
    scene.add(helixMesh);

    let animationFrameId: number;
    const animate = () => {
      animationFrameId = requestAnimationFrame(animate);
      controls.update();
      renderer.render(scene, camera);
    };
    animate();

    const handleResize = () => {
      if (!container) return;
      const w = container.clientWidth;
      const h = container.clientHeight;
      camera.aspect = w / h;
      camera.updateProjectionMatrix();
      renderer.setSize(w, h);
    };
    window.addEventListener('resize', handleResize);

    return () => {
      cancelAnimationFrame(animationFrameId);
      window.removeEventListener('resize', handleResize);
      renderer.dispose();
      container.innerHTML = '';
    };
  }, [activeTab, ropePairs, selectedDimPairIdx]);

  // Position Scrubber Animation Loop
  useEffect(() => {
    if (!isPlayingRoPE) return;
    const timer = setInterval(() => {
      setCurrentPosition((prev) => (prev >= 64 ? 0 : prev + 1));
    }, 120);
    return () => clearInterval(timer);
  }, [isPlayingRoPE]);

  // -------------------------------------------------------------------
  // 3. Three.js Scene: Residual Vector Addition (3D Arithmetic)
  // -------------------------------------------------------------------
  useEffect(() => {
    if (activeTab !== 'residual' || !residualCanvasRef.current) return;
    const container = residualCanvasRef.current;
    container.innerHTML = '';

    const width = container.clientWidth || 600;
    const height = container.clientHeight || 420;

    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x020617);

    const camera = new THREE.PerspectiveCamera(50, width / height, 0.1, 1000);
    camera.position.set(4, 3, 5);

    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    renderer.setSize(width, height);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    container.appendChild(renderer.domElement);

    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;

    const ambientLight = new THREE.AmbientLight(0xffffff, 0.8);
    scene.add(ambientLight);

    // Coordinates: x_{l-1}, Attention update, FFN update, x_l
    const origin = new THREE.Vector3(0, 0, 0);
    const xPrev = new THREE.Vector3(1.5, 0.8, -0.5);
    const attnDelta = new THREE.Vector3(0.6, 1.2, 0.4);
    const afterAttn = xPrev.clone().add(attnDelta);
    const ffnDelta = new THREE.Vector3(0.8, -0.4, 1.1);
    const xNext = afterAttn.clone().add(ffnDelta);

    // Vector 1: x_{l-1} (Blue)
    const arrowXPrev = new THREE.ArrowHelper(xPrev.clone().normalize(), origin, xPrev.length(), 0x3b82f6, 0.25, 0.15);
    scene.add(arrowXPrev);

    // Vector 2: Attention Delta (Cyan, originating at xPrev)
    const arrowAttn = new THREE.ArrowHelper(attnDelta.clone().normalize(), xPrev, attnDelta.length(), 0x06b6d4, 0.25, 0.15);
    scene.add(arrowAttn);

    // Vector 3: FFN Delta (Green, originating at afterAttn)
    const arrowFFN = new THREE.ArrowHelper(ffnDelta.clone().normalize(), afterAttn, ffnDelta.length(), 0x22c55e, 0.25, 0.15);
    scene.add(arrowFFN);

    // Vector 4: Net Output x_l (Purple, from origin to xNext)
    const arrowXNext = new THREE.ArrowHelper(xNext.clone().normalize(), origin, xNext.length(), 0xa855f7, 0.3, 0.18);
    scene.add(arrowXNext);

    // Spheres at key points
    const sphereGeo = new THREE.SphereGeometry(0.12, 16, 16);
    const pointMat = new THREE.MeshStandardMaterial({ color: 0xffffff, emissive: 0xffffff, emissiveIntensity: 0.5 });

    [xPrev, afterAttn, xNext].forEach((pos) => {
      const sp = new THREE.Mesh(sphereGeo, pointMat);
      sp.position.copy(pos);
      scene.add(sp);
    });

    let animationFrameId: number;
    const animate = () => {
      animationFrameId = requestAnimationFrame(animate);
      controls.update();
      renderer.render(scene, camera);
    };
    animate();

    const handleResize = () => {
      if (!container) return;
      const w = container.clientWidth;
      const h = container.clientHeight;
      camera.aspect = w / h;
      camera.updateProjectionMatrix();
      renderer.setSize(w, h);
    };
    window.addEventListener('resize', handleResize);

    return () => {
      cancelAnimationFrame(animationFrameId);
      window.removeEventListener('resize', handleResize);
      renderer.dispose();
      container.innerHTML = '';
    };
  }, [activeTab, selectedResidualLayer]);

  // -------------------------------------------------------------------
  // 4. Three.js Scene: LayerNorm Hypersphere Projection
  // -------------------------------------------------------------------
  useEffect(() => {
    if (activeTab !== 'layernorm' || !hypersphereCanvasRef.current) return;
    const container = hypersphereCanvasRef.current;
    container.innerHTML = '';

    const width = container.clientWidth || 600;
    const height = container.clientHeight || 420;

    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x020617);

    const camera = new THREE.PerspectiveCamera(50, width / height, 0.1, 1000);
    camera.position.set(0, 3, 6);

    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    renderer.setSize(width, height);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    container.appendChild(renderer.domElement);

    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;

    const ambientLight = new THREE.AmbientLight(0xffffff, 0.8);
    scene.add(ambientLight);

    // Target Unit Sphere (radius = 2.0)
    const sphereRadius = 2.2;
    const sphereGeo = new THREE.SphereGeometry(sphereRadius, 24, 24);
    const sphereMat = new THREE.MeshBasicMaterial({
      color: 0x38bdf8,
      wireframe: true,
      transparent: true,
      opacity: 0.25,
    });
    const unitSphere = new THREE.Mesh(sphereGeo, sphereMat);
    scene.add(unitSphere);

    // Pre-norm vectors (scattered at varying radii) and Post-norm projections
    const numVectors = 14;
    for (let i = 0; i < numVectors; i++) {
      const theta = (i / numVectors) * Math.PI * 2;
      const phi = ((i % 5) / 5) * Math.PI - Math.PI / 2;
      const rawRadius = 0.8 + (i % 4) * 0.9;

      const preX = Math.cos(theta) * Math.cos(phi) * rawRadius;
      const preY = Math.sin(phi) * rawRadius + 0.4; // mean shift
      const preZ = Math.sin(theta) * Math.cos(phi) * rawRadius;
      const prePos = new THREE.Vector3(preX, preY, preZ);

      // Post-norm projected onto sphere surface
      const postPos = prePos.clone().normalize().multiplyScalar(sphereRadius);

      // Pre-norm marker (Red)
      const preGeo = new THREE.SphereGeometry(0.08, 12, 12);
      const preMat = new THREE.MeshStandardMaterial({ color: 0xef4444, emissive: 0xef4444, emissiveIntensity: 0.6 });
      const preMesh = new THREE.Mesh(preGeo, preMat);
      preMesh.position.copy(prePos);
      scene.add(preMesh);

      // Post-norm marker (Green)
      const postGeo = new THREE.SphereGeometry(0.09, 12, 12);
      const postMat = new THREE.MeshStandardMaterial({ color: 0x22c55e, emissive: 0x22c55e, emissiveIntensity: 0.8 });
      const postMesh = new THREE.Mesh(postGeo, postMat);
      postMesh.position.copy(postPos);
      scene.add(postMesh);

      // Connection dashed chord
      const lineGeo = new THREE.BufferGeometry().setFromPoints([prePos, postPos]);
      const lineMat = new THREE.LineDashedMaterial({ color: 0xf59e0b, dashSize: 0.1, gapSize: 0.05 });
      const line = new THREE.Line(lineGeo, lineMat);
      line.computeLineDistances();
      scene.add(line);
    }

    let animationFrameId: number;
    const animate = () => {
      animationFrameId = requestAnimationFrame(animate);
      controls.update();
      renderer.render(scene, camera);
    };
    animate();

    const handleResize = () => {
      if (!container) return;
      const w = container.clientWidth;
      const h = container.clientHeight;
      camera.aspect = w / h;
      camera.updateProjectionMatrix();
      renderer.setSize(w, h);
    };
    window.addEventListener('resize', handleResize);

    return () => {
      cancelAnimationFrame(animationFrameId);
      window.removeEventListener('resize', handleResize);
      renderer.dispose();
      container.innerHTML = '';
    };
  }, [activeTab, selectedLNLayer]);

  return (
    <div className={`min-h-screen bg-slate-950 text-slate-100 p-4 sm:p-6 space-y-6 ${isFullscreen ? 'fixed inset-0 z-50 overflow-y-auto' : ''}`}>
      {/* Top Header & Master Controls */}
      <div className="p-4 sm:p-5 rounded-2xl bg-gradient-to-r from-slate-900 via-slate-900/90 to-slate-950 border border-slate-800 shadow-2xl backdrop-blur-xl space-y-4">
        <div className="flex flex-wrap items-center justify-between gap-4">
          <div className="flex items-center gap-3">
            <div className="w-11 h-11 rounded-xl bg-gradient-to-br from-indigo-500 via-purple-500 to-pink-500 flex items-center justify-center text-white shadow-lg shadow-indigo-500/25">
              <Sparkles className="w-6 h-6" />
            </div>
            <div>
              <div className="flex items-center gap-2.5">
                <h1 className="text-xl font-bold text-white tracking-tight">Transformer Representation & State Studios</h1>
                <Badge variant="outline" className="text-[10px] font-mono border-indigo-500/40 text-indigo-400">
                  Interactive Deep Suite
                </Badge>
              </div>
              <p className="text-xs text-slate-400 mt-0.5">
                3D Embedding Topology · RoPE Angle Spinners · Residual Stream Arithmetic · Normalization Hyperspheres
              </p>
            </div>
          </div>

          {/* Model Preset & Fullscreen Actions */}
          <div className="flex items-center gap-3">
            <div className="flex items-center gap-2 bg-slate-900/80 border border-slate-800 rounded-xl px-3 py-1.5 text-xs font-mono">
              <span className="text-slate-500">Model:</span>
              <select
                value={selectedModel.id}
                onChange={(e) => {
                  const m = MODEL_PRESETS.find((p) => p.id === e.target.value);
                  if (m) setSelectedModel(m);
                }}
                className="bg-transparent text-white font-semibold focus:outline-none cursor-pointer"
              >
                {MODEL_PRESETS.map((m) => (
                  <option key={m.id} value={m.id} className="bg-slate-900 text-white">
                    {m.name} ({m.normType}, {m.posType})
                  </option>
                ))}
              </select>
            </div>

            <Button
              variant="outline"
              size="sm"
              onClick={() => setIsFullscreen(!isFullscreen)}
              className="border-slate-800 text-slate-300 hover:text-white"
            >
              {isFullscreen ? <Minimize2 className="w-4 h-4" /> : <Maximize2 className="w-4 h-4" />}
            </Button>
          </div>
        </div>

        {/* Studio Tabs */}
        <div className="flex flex-wrap gap-2 pt-2 border-t border-slate-800/80">
          <button
            onClick={() => setActiveTab('embeddings')}
            className={`px-4 py-2 rounded-xl text-xs font-medium flex items-center gap-2 transition-all ${
              activeTab === 'embeddings'
                ? 'bg-gradient-to-r from-blue-600 to-indigo-600 text-white shadow-lg shadow-blue-500/25 border border-blue-400/30'
                : 'bg-slate-900/60 text-slate-400 hover:text-white hover:bg-slate-900 border border-slate-800'
            }`}
          >
            <Compass className="w-4 h-4" />
            <span>1. Embedding Space Explorer</span>
            <Badge variant="outline" className="text-[9px] border-blue-400/40 text-blue-300">
              3D & UMAP
            </Badge>
          </button>

          <button
            onClick={() => setActiveTab('positional')}
            className={`px-4 py-2 rounded-xl text-xs font-medium flex items-center gap-2 transition-all ${
              activeTab === 'positional'
                ? 'bg-gradient-to-r from-amber-600 to-orange-600 text-white shadow-lg shadow-orange-500/25 border border-amber-400/30'
                : 'bg-slate-900/60 text-slate-400 hover:text-white hover:bg-slate-900 border border-slate-800'
            }`}
          >
            <RotateCcw className="w-4 h-4" />
            <span>2. Positional Encoding Studio</span>
            <Badge variant="outline" className="text-[9px] border-amber-400/40 text-amber-300">
              RoPE Rotations
            </Badge>
          </button>

          <button
            onClick={() => setActiveTab('residual')}
            className={`px-4 py-2 rounded-xl text-xs font-medium flex items-center gap-2 transition-all ${
              activeTab === 'residual'
                ? 'bg-gradient-to-r from-purple-600 to-pink-600 text-white shadow-lg shadow-purple-500/25 border border-purple-400/30'
                : 'bg-slate-900/60 text-slate-400 hover:text-white hover:bg-slate-900 border border-slate-800'
            }`}
          >
            <Layers className="w-4 h-4" />
            <span>3. Residual Stream Tracer</span>
            <Badge variant="outline" className="text-[9px] border-purple-400/40 text-purple-300">
              Vector Highway
            </Badge>
          </button>

          <button
            onClick={() => setActiveTab('layernorm')}
            className={`px-4 py-2 rounded-xl text-xs font-medium flex items-center gap-2 transition-all ${
              activeTab === 'layernorm'
                ? 'bg-gradient-to-r from-emerald-600 to-teal-600 text-white shadow-lg shadow-emerald-500/25 border border-emerald-400/30'
                : 'bg-slate-900/60 text-slate-400 hover:text-white hover:bg-slate-900 border border-slate-800'
            }`}
          >
            <Activity className="w-4 h-4" />
            <span>4. Layer Normalization Dynamics</span>
            <Badge variant="outline" className="text-[9px] border-emerald-400/40 text-emerald-300">
              The Stabilizer Lab
            </Badge>
          </button>
        </div>
      </div>

      {/* =============================================================== */}
      {/* TAB 1: EMBEDDING SPACE EXPLORER */}
      {/* =============================================================== */}
      {activeTab === 'embeddings' && (
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
          {/* Main 3D Canvas Panel */}
          <div className="lg:col-span-8 rounded-2xl bg-slate-900/80 border border-slate-800 p-4 shadow-xl flex flex-col space-y-4">
            <div className="flex flex-wrap items-center justify-between gap-3">
              <div className="flex items-center gap-2">
                <Compass className="w-5 h-5 text-blue-400" />
                <h2 className="font-semibold text-white">3D Token Embedding Cloud</h2>
                <Badge variant="outline" className="text-[10px] text-blue-400 border-blue-400/30">
                  Interactive WebGL
                </Badge>
              </div>

              {/* Dimensionality Reduction Switcher */}
              <div className="flex items-center gap-2 bg-slate-950 border border-slate-800 rounded-lg p-1 text-xs">
                {(['pca', 'tsne', 'umap'] as const).map((m) => (
                  <button
                    key={m}
                    onClick={() => setProjectionMethod(m)}
                    className={`px-2.5 py-1 rounded-md uppercase font-mono transition-all ${
                      projectionMethod === m
                        ? 'bg-blue-600 text-white font-bold shadow'
                        : 'text-slate-400 hover:text-white'
                    }`}
                  >
                    {m}
                  </button>
                ))}
              </div>
            </div>

            {/* Canvas Container */}
            <div
              ref={embeddingCanvasRef}
              className="w-full h-[440px] rounded-xl overflow-hidden border border-slate-800/80 relative"
            />

            {/* Token Search & Teleport Bar */}
            <div className="flex items-center gap-3 pt-2">
              <div className="relative flex-1">
                <Search className="w-4 h-4 text-slate-400 absolute left-3 top-2.5" />
                <input
                  type="text"
                  placeholder="Teleport to token (e.g. 'Paris', 'king', 'neural')..."
                  value={searchToken}
                  onChange={(e) => setSearchToken(e.target.value)}
                  className="w-full bg-slate-950 border border-slate-800 rounded-xl pl-9 pr-4 py-2 text-xs text-white focus:outline-none focus:border-blue-500"
                />
              </div>
              <Button
                variant="outline"
                size="sm"
                onClick={() => {
                  const node = tokenNodes.find((n) => n.token.toLowerCase() === searchToken.toLowerCase());
                  if (node) setSelectedToken(node);
                }}
                className="text-xs border-slate-700 hover:bg-slate-800"
              >
                Locate Vector
              </Button>
            </div>
          </div>

          {/* Right Sidebar: Semantic Axes & Linear Probing */}
          <div className="lg:col-span-4 space-y-6">
            {/* Semantic Axis Discovery */}
            <div className="rounded-2xl bg-slate-900/80 border border-slate-800 p-4 shadow-xl space-y-4">
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <Target className="w-4 h-4 text-pink-400" />
                  <h3 className="text-sm font-semibold text-white">Semantic Axis Discovery (ICA)</h3>
                </div>
                <Badge variant="outline" className="text-[10px] text-pink-400 border-pink-400/30">
                  Vector Arithmetic
                </Badge>
              </div>

              <div className="space-y-2 text-xs">
                <div
                  onClick={() => setSelectedAxis('sentiment')}
                  className={`p-3 rounded-xl border transition-all cursor-pointer space-y-2 ${
                    selectedAxis === 'sentiment'
                      ? 'bg-slate-900 border-pink-500/50 shadow-lg shadow-pink-500/10 ring-1 ring-pink-500/30'
                      : 'bg-slate-950/80 border-slate-800/80 hover:border-slate-700'
                  }`}
                >
                  <div className="flex justify-between text-slate-400 font-mono text-[11px]">
                    <span className="text-white font-semibold">Axis 1: Sentiment Pole</span>
                    <span className="text-emerald-400">Var: 34.2%</span>
                  </div>
                  <div className="flex items-center justify-between text-[11px] font-mono">
                    <span className="text-emerald-300">Positive (+) [happy, joy]</span>
                    <ArrowRight className="w-3.5 h-3.5 text-slate-600" />
                    <span className="text-rose-300">Negative (-) [sad, gloom]</span>
                  </div>
                </div>

                <div
                  onClick={() => setSelectedAxis('royalty')}
                  className={`p-3 rounded-xl border transition-all cursor-pointer space-y-2 ${
                    selectedAxis === 'royalty'
                      ? 'bg-slate-900 border-indigo-500/50 shadow-lg shadow-indigo-500/10 ring-1 ring-indigo-500/30'
                      : 'bg-slate-950/80 border-slate-800/80 hover:border-slate-700'
                  }`}
                >
                  <div className="flex justify-between text-slate-400 font-mono text-[11px]">
                    <span className="text-white font-semibold">Axis 2: Royalty & Gender</span>
                    <span className="text-indigo-400">Var: 28.7%</span>
                  </div>
                  <p className="text-[10px] text-slate-500 font-mono">
                    Formula: v(king) - v(man) + v(woman) ≈ v(queen) [Cosine Sim: 0.942]
                  </p>
                </div>
              </div>

              {selectedToken && (
                <div className="p-3 rounded-xl bg-slate-950 border border-blue-500/40 space-y-1.5 font-mono text-xs">
                  <div className="flex justify-between items-center">
                    <span className="text-white font-bold">Vector: "{selectedToken.token}"</span>
                    <Badge variant="cyan">Cluster #{selectedToken.cluster}</Badge>
                  </div>
                  <div className="text-[11px] text-slate-400">
                    Pos: [{selectedToken.x.toFixed(2)}, {selectedToken.y.toFixed(2)}, {selectedToken.z.toFixed(2)}]
                  </div>
                  <div className="flex justify-between text-[11px]">
                    <span className="text-emerald-400">L2 Norm: {selectedToken.norm.toFixed(3)}</span>
                    <button
                      onClick={() => setSelectedToken(null)}
                      className="text-slate-500 hover:text-slate-300 underline text-[10px]"
                    >
                      Clear Selection
                    </button>
                  </div>
                </div>
              )}
            </div>

            {/* Anisotropy & Norm Metrics */}
            <div className="rounded-2xl bg-slate-900/80 border border-slate-800 p-4 shadow-xl space-y-4">
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <TrendingUp className="w-4 h-4 text-amber-400" />
                  <h3 className="text-sm font-semibold text-white">Embedding Space Geometry</h3>
                </div>
                <Badge variant="outline" className="text-[10px] text-amber-400 border-amber-400/30">
                  Anisotropy Monitor
                </Badge>
              </div>

              <div className="grid grid-cols-2 gap-3 text-xs">
                <div className="p-3 rounded-xl bg-slate-950 border border-slate-800">
                  <span className="text-slate-500 block text-[10px]">Avg Cosine Sim</span>
                  <span className="text-lg font-bold text-amber-300 font-mono">0.428</span>
                  <span className="text-[10px] text-slate-500 block">Narrow Cone Tendency</span>
                </div>
                <div className="p-3 rounded-xl bg-slate-950 border border-slate-800">
                  <span className="text-slate-500 block text-[10px]">Isotropy Index</span>
                  <span className="text-lg font-bold text-emerald-400 font-mono">0.182</span>
                  <span className="text-[10px] text-slate-500 block">SVD min/max ratio</span>
                </div>
              </div>
            </div>

            {/* Linear Probe Playground */}
            <div className="rounded-2xl bg-slate-900/80 border border-slate-800 p-4 shadow-xl space-y-4">
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <Zap className="w-4 h-4 text-purple-400" />
                  <h3 className="text-sm font-semibold text-white">Linear Probe Playground</h3>
                </div>
                <Badge variant="outline" className="text-[10px] text-purple-400 border-purple-400/30">
                  LIBLINEAR WASM
                </Badge>
              </div>

              <p className="text-xs text-slate-400">
                Train browser-side logistic regression probes on embedding vectors to test linear separability.
              </p>

              <div className="flex items-center justify-between pt-2">
                <span className="text-xs text-slate-300 font-mono">Task: Noun vs Verb Probe</span>
                <Button
                  size="sm"
                  onClick={() => {
                    setProbeClassifying(true);
                    setTimeout(() => setProbeClassifying(false), 600);
                  }}
                  className="bg-purple-600 hover:bg-purple-700 text-xs"
                >
                  {probeClassifying ? 'Fitting...' : 'Fit Linear Probe'}
                </Button>
              </div>

              <div className="p-3 rounded-xl bg-slate-950 border border-slate-800 flex justify-between items-center text-xs font-mono">
                <span className="text-slate-400">Probe Accuracy:</span>
                <span className="text-emerald-400 font-bold">96.8% (F1: 0.965)</span>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* =============================================================== */}
      {/* TAB 2: POSITIONAL ENCODING STUDIO */}
      {/* =============================================================== */}
      {activeTab === 'positional' && (
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
          {/* 3D RoPE Angle Spinner Canvas */}
          <div className="lg:col-span-7 rounded-2xl bg-slate-900/80 border border-slate-800 p-4 shadow-xl space-y-4">
            <div className="flex flex-wrap items-center justify-between gap-3">
              <div className="flex items-center gap-2">
                <RotateCcw className="w-5 h-5 text-amber-400" />
                <h2 className="font-semibold text-white">RoPE 3D Angle Rotation Rings</h2>
                <Badge variant="outline" className="text-[10px] text-amber-400 border-amber-400/30">
                  θ_i = pos · base^(-2i/d)
                </Badge>
              </div>

              {/* Scrubber Play/Pause Controls */}
              <div className="flex items-center gap-2">
                <Button
                  size="sm"
                  variant="outline"
                  onClick={() => setIsPlayingRoPE(!isPlayingRoPE)}
                  className="text-xs border-slate-700"
                >
                  {isPlayingRoPE ? <Pause className="w-3.5 h-3.5 mr-1" /> : <Play className="w-3.5 h-3.5 mr-1" />}
                  {isPlayingRoPE ? 'Pause' : 'Animate pos'}
                </Button>
                <div className="px-2.5 py-1 rounded-lg bg-slate-950 border border-slate-800 text-xs font-mono text-amber-300">
                  pos = {currentPosition}
                </div>
              </div>
            </div>

            {/* 3D WebGL Canvas */}
            <div
              ref={ropeCanvasRef}
              className="w-full h-[400px] rounded-xl overflow-hidden border border-slate-800/80 relative"
            />

            {/* Position Scrubber Slider */}
            <div className="space-y-1.5 pt-2">
              <div className="flex justify-between text-xs text-slate-400 font-mono">
                <span>Token Position (p): {currentPosition}</span>
                <span>Max Sequence: 64 tokens</span>
              </div>
              <input
                type="range"
                min={0}
                max={64}
                value={currentPosition}
                onChange={(e) => setCurrentPosition(Number(e.target.value))}
                className="w-full h-1.5 bg-slate-800 rounded-lg appearance-none cursor-pointer accent-amber-500"
              />
            </div>
          </div>

          {/* Right Panel: Dimension Pairs & Mathematical Formulas */}
          <div className="lg:col-span-5 space-y-6">
            {/* Live Math Panel */}
            <div className="rounded-2xl bg-slate-900/80 border border-slate-800 p-4 shadow-xl space-y-3">
              <div className="flex items-center gap-2">
                <Zap className="w-4 h-4 text-amber-400" />
                <h3 className="text-sm font-semibold text-white">RoPE 2D Rotation Matrix</h3>
              </div>

              <div className="p-3.5 rounded-xl bg-slate-950 border border-slate-800 font-mono text-xs text-slate-300 space-y-2">
                <div className="text-amber-400 font-semibold">
                  R(p · θ_i) = [[cos(p·θ_i), -sin(p·θ_i)], [sin(p·θ_i), cos(p·θ_i)]]
                </div>
                <div className="text-slate-400 text-[11px]">
                  For dimension pair {selectedDimPairIdx * 2}, {selectedDimPairIdx * 2 + 1}:
                </div>
                <div className="grid grid-cols-2 gap-2 text-[11px] pt-1">
                  <div>θ_{selectedDimPairIdx} = {ropePairs[selectedDimPairIdx]?.frequency.toExponential(3)}</div>
                  <div>Angle = {ropePairs[selectedDimPairIdx]?.angleDeg.toFixed(1)}°</div>
                  <div>cos(θ) = {ropePairs[selectedDimPairIdx]?.cosVal.toFixed(3)}</div>
                  <div>sin(θ) = {ropePairs[selectedDimPairIdx]?.sinVal.toFixed(3)}</div>
                </div>
              </div>
            </div>

            {/* Wavelength & Frequency Chart */}
            <div className="rounded-2xl bg-slate-900/80 border border-slate-800 p-4 shadow-xl space-y-3">
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <TrendingUp className="w-4 h-4 text-orange-400" />
                  <h3 className="text-sm font-semibold text-white">Wavelength Spectrum Across Dimensions</h3>
                </div>
                <Badge variant="outline" className="text-[10px] text-orange-400 border-orange-400/30">
                  Log Scale
                </Badge>
              </div>

              <div className="h-44 w-full">
                <ResponsiveContainer width="100%" height="100%">
                  <AreaChart data={ropePairs}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                    <XAxis dataKey="pairIndex" stroke="#64748b" fontSize={10} label={{ value: 'Dim Pair', position: 'insideBottom', offset: -2, fill: '#64748b' }} />
                    <YAxis stroke="#64748b" fontSize={10} scale="log" domain={['auto', 'auto']} />
                    <RechartsTooltip contentStyle={{ backgroundColor: '#020617', borderColor: '#334155', fontSize: '11px' }} />
                    <Area type="monotone" dataKey="wavelength" stroke="#f97316" fill="#f97316" fillOpacity={0.2} name="Wavelength (tokens)" />
                  </AreaChart>
                </ResponsiveContainer>
              </div>
            </div>

            {/* Relative Distance Preservation */}
            <div className="rounded-2xl bg-slate-900/80 border border-slate-800 p-4 shadow-xl space-y-3">
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <Scale className="w-4 h-4 text-blue-400" />
                  <h3 className="text-sm font-semibold text-white">Relative Position Property</h3>
                </div>
                <Badge variant="outline" className="text-[10px] text-blue-400 border-blue-400/30">
                  q_m · k_n = f(m - n)
                </Badge>
              </div>
              <p className="text-xs text-slate-400 leading-relaxed">
                RoPE guarantees that the dot product of query vector at position <em>m</em> with key vector at position <em>n</em> depends strictly on the relative distance <code>(m - n)</code>, preserving translation invariance across context windows.
              </p>
            </div>
          </div>
        </div>
      )}

      {/* =============================================================== */}
      {/* TAB 3: RESIDUAL STREAM TRACER */}
      {/* =============================================================== */}
      {activeTab === 'residual' && (
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
          {/* 3D Vector Arithmetic Scene */}
          <div className="lg:col-span-6 rounded-2xl bg-slate-900/80 border border-slate-800 p-4 shadow-xl space-y-4">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2">
                <Layers className="w-5 h-5 text-purple-400" />
                <h2 className="font-semibold text-white">Residual Vector Addition Sandbox</h2>
              </div>
              <Badge variant="outline" className="text-[10px] text-purple-400 border-purple-400/30 font-mono">
                x_l = x_{`{l-1}`} + Attn + FFN
              </Badge>
            </div>

            {/* Canvas */}
            <div
              ref={residualCanvasRef}
              className="w-full h-[380px] rounded-xl overflow-hidden border border-slate-800/80 relative"
            />

            {/* Color Legend */}
            <div className="flex flex-wrap items-center justify-between text-xs font-mono pt-1">
              <div className="flex items-center gap-2">
                <div className="w-3 h-3 rounded-full bg-blue-500" />
                <span className="text-slate-300">x_{`{l-1}`} Stream</span>
              </div>
              <div className="flex items-center gap-2">
                <div className="w-3 h-3 rounded-full bg-cyan-400" />
                <span className="text-slate-300">Attn Delta</span>
              </div>
              <div className="flex items-center gap-2">
                <div className="w-3 h-3 rounded-full bg-green-500" />
                <span className="text-slate-300">FFN Delta</span>
              </div>
              <div className="flex items-center gap-2">
                <div className="w-3 h-3 rounded-full bg-purple-400" />
                <span className="text-slate-300">x_l Output</span>
              </div>
            </div>
          </div>

          {/* Layer-wise Norm Growth & Logit Lens Timeline */}
          <div className="lg:col-span-6 space-y-6">
            {/* Norm Growth Chart */}
            <div className="rounded-2xl bg-slate-900/80 border border-slate-800 p-4 shadow-xl space-y-3">
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <TrendingUp className="w-4 h-4 text-purple-400" />
                  <h3 className="text-sm font-semibold text-white">Layer-wise Hidden State L2 Norm Growth</h3>
                </div>
                <Badge variant="outline" className="text-[10px] text-purple-400 border-purple-400/30">
                  Highway Dynamics
                </Badge>
              </div>

              <div className="h-44 w-full">
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={residualTraces}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                    <XAxis dataKey="layer" stroke="#64748b" fontSize={10} />
                    <YAxis stroke="#64748b" fontSize={10} />
                    <RechartsTooltip contentStyle={{ backgroundColor: '#020617', borderColor: '#334155', fontSize: '11px' }} />
                    <Line type="monotone" dataKey="totalNorm" stroke="#a855f7" strokeWidth={2} name="Total Residual Norm" dot={{ r: 3 }} />
                    <Line type="monotone" dataKey="attnNorm" stroke="#06b6d4" strokeWidth={1.5} strokeDasharray="3 3" name="Attn Norm" />
                    <Line type="monotone" dataKey="ffnNorm" stroke="#22c55e" strokeWidth={1.5} strokeDasharray="3 3" name="FFN Norm" />
                  </LineChart>
                </ResponsiveContainer>
              </div>
            </div>

            {/* Logit Lens Projection Timeline */}
            <div className="rounded-2xl bg-slate-900/80 border border-slate-800 p-4 shadow-xl space-y-3">
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <Zap className="w-4 h-4 text-emerald-400" />
                  <h3 className="text-sm font-semibold text-white">Logit Lens Convergence (What Model Thinks)</h3>
                </div>
                <Badge variant="outline" className="text-[10px] text-emerald-400 border-emerald-400/30">
                  Unembed @ LN(x_l)
                </Badge>
              </div>

              <div className="space-y-2">
                <div className="flex items-center justify-between text-xs text-slate-400 font-mono">
                  <span>Inspecting Layer: {selectedResidualLayer}</span>
                  <input
                    type="range"
                    min={0}
                    max={residualTraces.length - 1}
                    value={selectedResidualLayer}
                    onChange={(e) => setSelectedResidualLayer(Number(e.target.value))}
                    className="w-32 accent-purple-500"
                  />
                </div>

                <div className="space-y-2 pt-1">
                  {residualTraces[selectedResidualLayer]?.topLogits.map((item, idx) => (
                    <div key={idx} className="space-y-1">
                      <div className="flex justify-between text-xs font-mono">
                        <span className="text-white font-semibold">{item.token}</span>
                        <span className="text-emerald-400">{(item.probability * 100).toFixed(1)}%</span>
                      </div>
                      <div className="w-full h-1.5 bg-slate-800 rounded-full overflow-hidden">
                        <div
                          className="h-full bg-gradient-to-r from-emerald-500 to-teal-400"
                          style={{ width: `${item.probability * 100}%` }}
                        />
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* =============================================================== */}
      {/* TAB 4: LAYER NORMALIZATION DYNAMICS */}
      {/* =============================================================== */}
      {activeTab === 'layernorm' && (
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
          {/* 3D Hypersphere Projection Scene */}
          <div className="lg:col-span-6 rounded-2xl bg-slate-900/80 border border-slate-800 p-4 shadow-xl space-y-4">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2">
                <Activity className="w-5 h-5 text-teal-400" />
                <h2 className="font-semibold text-white">Unit Hypersphere Projection</h2>
              </div>
              <Badge variant="outline" className="text-[10px] text-teal-400 border-teal-400/30 font-mono">
                ||x̂|| = √d
              </Badge>
            </div>

            {/* Canvas */}
            <div
              ref={hypersphereCanvasRef}
              className="w-full h-[380px] rounded-xl overflow-hidden border border-slate-800/80 relative"
            />

            {/* Legend */}
            <div className="flex items-center justify-between text-xs font-mono pt-1">
              <div className="flex items-center gap-2">
                <div className="w-3 h-3 rounded-full bg-red-500" />
                <span className="text-slate-300">Pre-LN Raw Vectors</span>
              </div>
              <div className="flex items-center gap-2">
                <div className="w-3 h-3 rounded-full bg-green-500" />
                <span className="text-slate-300">Post-LN on Sphere</span>
              </div>
              <div className="flex items-center gap-2">
                <div className="w-3 h-1 bg-amber-400" />
                <span className="text-slate-300">Projection Chord</span>
              </div>
            </div>
          </div>

          {/* Variance Stabilization & Gain/Bias Parameter Drift */}
          <div className="lg:col-span-6 space-y-6">
            {/* Pre/Post Variance Line Chart */}
            <div className="rounded-2xl bg-slate-900/80 border border-slate-800 p-4 shadow-xl space-y-3">
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <Shield className="w-4 h-4 text-emerald-400" />
                  <h3 className="text-sm font-semibold text-white">Activation Variance Stabilization</h3>
                </div>
                <Badge variant="outline" className="text-[10px] text-emerald-400 border-emerald-400/30">
                  Pre vs Post Norm
                </Badge>
              </div>

              <div className="h-44 w-full">
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={layerNormStats}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                    <XAxis dataKey="layer" stroke="#64748b" fontSize={10} />
                    <YAxis stroke="#64748b" fontSize={10} />
                    <RechartsTooltip contentStyle={{ backgroundColor: '#020617', borderColor: '#334155', fontSize: '11px' }} />
                    <Line type="monotone" dataKey="preVar" stroke="#ef4444" strokeWidth={2} name="Pre-LN Var" dot={{ r: 3 }} />
                    <Line type="monotone" dataKey="postVar" stroke="#22c55e" strokeWidth={2} name="Post-LN Var (≈1.0)" dot={{ r: 3 }} />
                  </LineChart>
                </ResponsiveContainer>
              </div>
            </div>

            {/* Gain/Bias Parameter Drift & Mode Controls */}
            <div className="rounded-2xl bg-slate-900/80 border border-slate-800 p-4 shadow-xl space-y-3">
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <Sliders className="w-4 h-4 text-emerald-400" />
                  <h3 className="text-sm font-semibold text-white">Gain & Bias Stabilizer Tuning</h3>
                </div>
                <div className="flex items-center gap-1 bg-slate-950 p-1 rounded-lg border border-slate-800 text-[10px] font-mono">
                  {(['both', 'layernorm', 'rmsnorm'] as const).map((mode) => (
                    <button
                      key={mode}
                      onClick={() => setNormViewMode(mode)}
                      className={`px-2 py-0.5 rounded uppercase ${
                        normViewMode === mode ? 'bg-teal-600 text-white font-bold' : 'text-slate-400 hover:text-white'
                      }`}
                    >
                      {mode}
                    </button>
                  ))}
                </div>
              </div>

              <div className="grid grid-cols-2 gap-4 text-xs font-mono">
                <div>
                  <div className="flex justify-between text-slate-400 mb-1">
                    <span>Gain Scale (γ):</span>
                    <span className="text-teal-300 font-bold">{customGamma.toFixed(2)}x</span>
                  </div>
                  <input
                    type="range"
                    min="0.5"
                    max="2.5"
                    step="0.05"
                    value={customGamma}
                    onChange={(e) => setCustomGamma(parseFloat(e.target.value))}
                    className="w-full accent-teal-500"
                  />
                </div>
                <div>
                  <div className="flex justify-between text-slate-400 mb-1">
                    <span>Bias Shift (β):</span>
                    <span className="text-teal-300 font-bold">{customBeta.toFixed(2)}</span>
                  </div>
                  <input
                    type="range"
                    min="-1.0"
                    max="1.0"
                    step="0.05"
                    value={customBeta}
                    onChange={(e) => setCustomBeta(parseFloat(e.target.value))}
                    className="w-full accent-teal-500"
                  />
                </div>
              </div>

              <div className="p-2.5 rounded-xl bg-slate-950 border border-slate-800 text-[11px] font-mono text-slate-300 flex justify-between">
                <span>Output Norm: <strong className="text-emerald-400 font-bold">{(customGamma * 1.0).toFixed(2)} · √d</strong></span>
                <span>Variance: <strong className="text-teal-300 font-bold">{(customGamma ** 2).toFixed(2)}</strong></span>
                <span>Mean Shift: <strong className="text-amber-300 font-bold">{customBeta.toFixed(2)}</strong></span>
              </div>
            </div>

            {/* RMSNorm vs LayerNorm Architectural Comparison */}
            <div className="rounded-2xl bg-slate-900/80 border border-slate-800 p-4 shadow-xl space-y-3">
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <Scale className="w-4 h-4 text-teal-400" />
                  <h3 className="text-sm font-semibold text-white">LayerNorm vs RMSNorm Comparison</h3>
                </div>
                <Badge variant="outline" className="text-[10px] text-teal-400 border-teal-400/30">
                  Speed & Stability
                </Badge>
              </div>

              <div className="grid grid-cols-2 gap-3 text-xs">
                <div className="p-3 rounded-xl bg-slate-950 border border-slate-800 space-y-1">
                  <span className="text-teal-400 font-bold block">LayerNorm (Standard)</span>
                  <p className="text-[11px] text-slate-400">Centers mean to zero & scales variance to 1. Formula: y = ((x - μ) / σ) · γ + β</p>
                </div>
                <div className="p-3 rounded-xl bg-slate-950 border border-slate-800 space-y-1">
                  <span className="text-emerald-400 font-bold block">RMSNorm (LLaMA/Mistral)</span>
                  <p className="text-[11px] text-slate-400">Skips mean subtraction: y = (x / RMS(x)) · γ. ~15% faster with identical representation quality.</p>
                </div>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default RepresentationStudios;
