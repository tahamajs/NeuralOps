# Tutorial 04: Custom PBR Materials & Shaders

Learn how to create custom shaders and glowing materials for specific tensor components.

---

## 1. Material Cache in `src/core/materials.js`

The engine automatically manages PBR materials via `getOrCreateCustomMaterial(hexColor)`:

```javascript
import * as THREE from 'three';

export function createHolographicMaterial(hexColor) {
    const c = new THREE.Color(hexColor);
    return new THREE.MeshStandardMaterial({
        color: c,
        roughness: 0.15,
        metalness: 0.85,
        transparent: true,
        opacity: 0.88,
        emissive: c.clone().multiplyScalar(0.4),
        emissiveIntensity: 0.5
    });
}
```

---

## 2. Using Custom Shaders via `THREE.ShaderMaterial`

If you want custom animated vertex displacement (e.g. quantum probability clouds or activation ripples), instantiate a `ShaderMaterial`:

```javascript
const rippleShaderMaterial = new THREE.ShaderMaterial({
    uniforms: {
        time: { value: 0 },
        baseColor: { value: new THREE.Color(0x53ecd9) }
    },
    vertexShader: `
        uniform float time;
        varying vec3 vNormal;
        void main() {
            vNormal = normal;
            vec3 pos = position;
            pos += normal * sin(pos.y * 10.0 + time * 4.0) * 0.05;
            gl_Position = projectionMatrix * modelViewMatrix * vec4(pos, 1.0);
        }
    `,
    fragmentShader: `
        uniform vec3 baseColor;
        varying vec3 vNormal;
        void main() {
            float intensity = pow(0.7 - dot(vNormal, vec3(0.0, 0.0, 1.0)), 2.0);
            gl_FragColor = vec4(baseColor + vec3(intensity), 0.9);
        }
    `,
    transparent: true
});
```
Update `rippleShaderMaterial.uniforms.time.value = timestamp * 0.001;` inside the animation loop in `src/index.js`.
