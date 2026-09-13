const fs = require('fs');
const jsdom = require("jsdom");
const { JSDOM } = jsdom;

const html = fs.readFileSync('Docs/visualizer/index.html', 'utf8');
const script = fs.readFileSync('Docs/visualizer/app.js', 'utf8');

const dom = new JSDOM(html, { runScripts: "outside-only" });
// Mock THREE and katex
dom.window.THREE = { 
    Vector3: class {}, Color: class {}, MathUtils: { lerp: ()=>0 },
    Mesh: class {}, WebGLRenderer: class { setSize(){} setPixelRatio(){} render(){} },
    Scene: class {}, PerspectiveCamera: class { position={set:()=>{}, clone:()=>{project:()=>{}}}; lookAt(){} },
    PointLight: class {}, AmbientLight: class {}, Object3D: class {}, TextureLoader: class{load(){}},
    BufferGeometry: class { setFromPoints(){} }, Line: class {}, Material: class{ clone(){} }, BoxGeometry: class{}, CylinderGeometry: class{}, LineBasicMaterial: class{}, MeshStandardMaterial: class{}
};
dom.window.katex = { render: () => {} };
dom.window.requestAnimationFrame = () => {};

try {
    dom.window.eval(script.replace(/import \* as THREE from 'three';/, ''));
    console.log("JSDOM Eval Success!");
} catch (e) {
    console.error("JSDOM Error:", e);
}
