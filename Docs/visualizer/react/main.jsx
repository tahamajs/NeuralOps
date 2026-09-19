/**
 * react/main.jsx
 * ==============
 * Entry point for mounting the declarative React routing layer.
 */

import React from 'react';
import { createRoot } from 'react-dom/client';
import App from './App.jsx';

export function mountReactVisualizer(targetElement) {
  if (!targetElement) {
    console.error("Cannot mount React visualizer: Target container element is null");
    return null;
  }
  const root = createRoot(targetElement);
  root.render(<App />);
  return root;
}

// Auto-mount if root element present in DOM
if (typeof document !== 'undefined') {
  const rootEl = document.getElementById('react-root');
  if (rootEl) {
    mountReactVisualizer(rootEl);
  }
}

export default App;
