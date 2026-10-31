# Tutorial 03: Adding a New Foundation Model

Learn how to add a 5th foundation model (e.g., LLaMA-3-8B) to the visualizer.

---

## Step 1: Add Specification to `src/constants.js`

Open `Docs/visualizer/src/constants.js` and append to `MODEL_SPECS`:

```javascript
export const MODEL_SPECS = {
    // ... existing models ...
    llama3: {
        name: 'LLaMA-3-8B',
        family: 'GQA Decoder',
        num_layers: 32,
        hidden_size: 4096,
        intermediate_size: 14336,
        base_layer_params: 250000000,
        base_total: 8030000000,
        descr: '32 Layers · 4096d · GQA (32Q:8KV) · SwiGLU MLP',
        vocab_size: 128256,
    }
};

export const VOCAB_SIZES = {
    // ...
    llama3: 128256,
};
```

---

## Step 2: Add Trajectory Rules to `src/physics/token_trajectory.js`

Add routing behavior in `computeToken3DTrajectory`:

```javascript
if (modelKey === 'llama3') {
    // 32Q heads : 8KV heads GQA logic
    if (t.subStream === 'q') {
        const qOffset = -0.85 + (t.headIdx / 32.0) * 0.50;
        return { x: baseOffsetX + qOffset, z: -0.1, sublayer: 'q_proj' };
    }
    // ...
}
```

---

## Step 3: Add Navigation Tab in `index.html`

In `index.html` inside `<nav class="mechanism-nav">`, add:

```html
<button class="mechanism-button" data-model="llama3" data-module="svd" role="tab" aria-selected="false">
    <b>LLaMA-3</b>
    <small>Dense vs SVD · 8.0B</small>
</button>
```
