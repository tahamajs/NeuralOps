# Semantic Zoom & Level-of-Detail (LOD) Engine

## 1. Three Visual Tiers

The visualizer implements a continuous Level-of-Detail (LOD) manager that automatically balances visual clarity with 60 FPS WebGL rendering performance based on camera distance:

```
Camera Distance from Model Center:
Distance > 40 units  ───▶  LOD 0: Macro Bounding Blocks
Distance 25 - 40     ───▶  Transition Zone (Smooth Fade)
Distance < 25 units  ───▶  LOD 1: Granular Matrices & Badges
```

| Level | Camera Distance | Visible Subsystems | Educational Purpose |
|---|---|---|---|
| **Macro (LOD 0)** | $> 40$ units | Consolidated layer boundary blocks, layer indices | Architecture-wide structure, depth comparison |
| **Transition** | $25 - 40$ units | Macro blocks fade; internal tensors appear | Connects layer macro volume to internal sublayers |
| **Granular (LOD 1)** | $< 25$ units | Matrix boxes, head partition lines, floating shape badges | Exact dimensional transformations, rank bottlenecks |

---

## 2. Mathematical Switching Logic

In `src/index.js`, the distance is computed every frame:

```javascript
const camDist = camera.position.distanceTo(controls.target);

if (camDist > VISUAL_CONFIG.ZOOM.macroThreshold) {
    // LOD 0: Only Macro blocks
    macroGroup.visible = true;
    granularGroup.visible = false;
    dimensionBadgesGroup.visible = false;
} else if (camDist > VISUAL_CONFIG.ZOOM.transitionStart) {
    // Transition: Both Macro and Granular visible
    macroGroup.visible = true;
    granularGroup.visible = true;
    dimensionBadgesGroup.visible = false;
} else {
    // LOD 1: Detailed Granular + Floating Dimension Badges
    macroGroup.visible = false;
    granularGroup.visible = true;
    dimensionBadgesGroup.visible = true;
}
```

---

## 3. Dimension Badges Staggering & Overlap Prevention

Floating dimension badges (`[768 × 768]`, `r=64`, `12H`, `2H GQA`) are rendered on 2D billboard sprites positioned on the front face of each 3D box ($Z = \text{node.z} + \frac{\text{node.depth}}{2} + 0.05$).
For adjacent narrow projections (such as Key and Value projections in GQA), vertical offsets are automatically staggered ($\pm 0.06$) to ensure text never overlaps.
