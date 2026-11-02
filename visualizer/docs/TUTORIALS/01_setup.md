# Tutorial 01: Setup & Local Development

## 1. Prerequisites
The visualizer is built on pure standard web technologies with no compilation steps or node_modules dependencies needed to run in the browser.

* Any modern web browser with WebGL enabled (Chrome, Firefox, Safari, Edge)
* Python 3.8+ (for local HTTP serving)

---

## 2. Launching Locally

From the root of the project, run:
```bash
python3 Docs/visualizer/serve.py
```
This starts an HTTP server at `http://localhost:8080` with proper CORS headers and automatically opens your browser to `http://localhost:8080/Docs/visualizer/index.html`.

---

## 3. Running Unit Tests

Run the Node.js built-in automated test runner:
```bash
node --test Docs/visualizer/tests/*.test.js
```
All 19 tests validating constants, model specs, token physics trajectories, and JSON schemas will execute.
