import { useEffect, useRef, useState } from "react";
import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import {
  CSS2DRenderer,
  CSS2DObject,
} from "three/examples/jsm/renderers/CSS2DRenderer.js";
import type { ModelArchitecture } from "../types";

interface Props {
  modelData: ModelArchitecture | null;
  onClose?: () => void;
}

export function Model3DViewer({ modelData, onClose }: Props) {
  const containerRef = useRef<HTMLDivElement>(null);
  const [status, setStatus] = useState<"loading" | "ready" | "error">(
    "loading",
  );
  const sceneRef = useRef<THREE.Scene | null>(null);
  const cameraRef = useRef<THREE.PerspectiveCamera | null>(null);
  const rendererRef = useRef<THREE.WebGLRenderer | null>(null);
  const labelRendererRef = useRef<CSS2DRenderer | null>(null);
  const controlsRef = useRef<OrbitControls | null>(null);

  useEffect(() => {
    if (!containerRef.current) return;
    const init = () => {
      try {
        const scene = new THREE.Scene();
        scene.background = new THREE.Color(0x03050b);
        sceneRef.current = scene;

        const camera = new THREE.PerspectiveCamera(
          45,
          containerRef.current!.clientWidth /
            containerRef.current!.clientHeight,
          0.1,
          1000,
        );
        camera.position.set(5, 5, 10);
        camera.lookAt(0, 0, 0);
        cameraRef.current = camera;

        const renderer = new THREE.WebGLRenderer({ antialias: true });
        renderer.setSize(
          containerRef.current!.clientWidth,
          containerRef.current!.clientHeight,
        );
        renderer.setPixelRatio(window.devicePixelRatio);
        containerRef.current!.appendChild(renderer.domElement);
        rendererRef.current = renderer;

        const labelRenderer = new CSS2DRenderer();
        labelRenderer.setSize(
          containerRef.current!.clientWidth,
          containerRef.current!.clientHeight,
        );
        labelRenderer.domElement.style.position = "absolute";
        labelRenderer.domElement.style.top = "0px";
        labelRenderer.domElement.style.left = "0px";
        labelRenderer.domElement.style.pointerEvents = "none";
        containerRef.current!.appendChild(labelRenderer.domElement);
        labelRendererRef.current = labelRenderer;

        const controls = new OrbitControls(camera, renderer.domElement);
        controls.enableDamping = true;
        controls.dampingFactor = 0.05;
        controls.autoRotate = true;
        controls.autoRotateSpeed = 1.0;
        controls.enableZoom = true;
        controls.enablePan = true;
        controlsRef.current = controls;

        // Lighting
        const ambientLight = new THREE.AmbientLight(0x404060);
        scene.add(ambientLight);
        const directionalLight = new THREE.DirectionalLight(0xffffff, 1);
        directionalLight.position.set(1, 2, 1);
        scene.add(directionalLight);
        const backLight = new THREE.DirectionalLight(0x88aaff, 0.5);
        backLight.position.set(-1, 1, -1);
        scene.add(backLight);

        const gridHelper = new THREE.GridHelper(20, 20, 0x3b82f6, 0x1e293b);
        gridHelper.position.y = -1.5;
        scene.add(gridHelper);

        setStatus("ready");

        let animationFrameId: number;
        const animate = () => {
          animationFrameId = requestAnimationFrame(animate);
          controls.update();
          if (
            rendererRef.current &&
            sceneRef.current &&
            cameraRef.current &&
            labelRendererRef.current
          ) {
            rendererRef.current.render(sceneRef.current, cameraRef.current);
            labelRendererRef.current.render(
              sceneRef.current,
              cameraRef.current,
            );
          }
        };
        animate();

        const handleResize = () => {
          if (
            !containerRef.current ||
            !cameraRef.current ||
            !rendererRef.current ||
            !labelRendererRef.current
          )
            return;
          const width = containerRef.current.clientWidth;
          const height = containerRef.current.clientHeight;
          cameraRef.current.aspect = width / height;
          cameraRef.current.updateProjectionMatrix();
          rendererRef.current.setSize(width, height);
          labelRendererRef.current.setSize(width, height);
        };
        window.addEventListener("resize", handleResize);
        return () => {
          cancelAnimationFrame(animationFrameId);
          window.removeEventListener("resize", handleResize);
          if (controlsRef.current) controlsRef.current.dispose();
          if (rendererRef.current) {
            rendererRef.current.dispose();
          }
          if (containerRef.current) {
            containerRef.current.innerHTML = "";
          }
        };
      } catch (err) {
        console.error(err);
        setStatus("error");
      }
    };
    init();
  }, []);

  useEffect(() => {
    if (status !== "ready" || !sceneRef.current) return;
    if (!modelData?.layers) {
      setStatus("error");
      return;
    }

    // Remove previous model layers
    const toRemove: THREE.Object3D[] = [];
    sceneRef.current.traverse((obj) => {
      if (obj.userData?.isModelLayer) toRemove.push(obj);
    });
    toRemove.forEach((obj) => sceneRef.current!.remove(obj));

    const layerSpacing = 1.2;
    const totalWidth = (modelData.layers.length - 1) * layerSpacing;
    const startX = -totalWidth / 2;

    modelData.layers.forEach((layer, idx) => {
      const x = startX + idx * layerSpacing;
      const geometry = new THREE.BoxGeometry(0.8, 0.8, 0.8);
      const material = new THREE.MeshStandardMaterial({
        color: layer.color ? parseInt(layer.color, 16) : 0x3b82f6,
        emissive: 0x0a0f1a,
        metalness: 0.7,
        roughness: 0.3,
      });
      const cube = new THREE.Mesh(geometry, material);
      cube.position.set(x, 0, 0);
      cube.userData = { isModelLayer: true, layerData: layer };
      sceneRef.current!.add(cube);

      // Label
      const div = document.createElement("div");
      div.textContent = `${layer.name}\n${layer.params.toLocaleString()} params`;
      div.style.color = "#00ff9d";
      div.style.fontSize = "12px";
      div.style.fontFamily = "monospace";
      div.style.backgroundColor = "rgba(0,0,0,0.7)";
      div.style.padding = "2px 6px";
      div.style.borderRadius = "4px";
      div.style.borderLeft = `3px solid ${layer.color ? "#" + layer.color : "#3b82f6"}`;
      div.style.pointerEvents = "none";
      const label = new CSS2DObject(div);
      label.position.set(x, -0.6, 0);
      label.userData = { isModelLayer: true };
      sceneRef.current!.add(label);
    });

    setStatus("ready");
  }, [status, modelData]);

  return (
    <div className="flex flex-col h-full">
      <div className="p-3 border-b border-white/10 flex justify-between items-center">
        <div className="flex items-center gap-2">
          <span className="text-sm text-neon-mint">🧊 3D Model Viewer</span>
          <span
            className={`text-xs ${status === "ready" ? "text-green-400" : "text-yellow-400"}`}
          >
            {status === "ready"
              ? "● Active"
              : status === "loading"
                ? "○ Loading..."
                : "⚠ Error"}
          </span>
        </div>
        {onClose && (
          <button onClick={onClose} className="text-slate-400 hover:text-white">
            ✕
          </button>
        )}
      </div>
      <div ref={containerRef} className="flex-1 relative" />
      <div className="p-2 border-t border-white/10 text-xs text-slate-500 text-center">
        Drag to rotate • Right-click to pan • Scroll to zoom
      </div>
    </div>
  );
}
