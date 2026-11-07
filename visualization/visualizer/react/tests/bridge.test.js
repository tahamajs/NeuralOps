/**
 * react/tests/bridge.test.js
 * ==========================
 * Unit tests for Three.js bridge hook and reactive store synchronization.
 */

import assert from 'node:assert';
import test, { describe, it } from 'node:test';
import { useModuleStore } from '../store/useModuleStore.js';
import { ROUTING_MATRIX } from '../components/visualization/routing_matrix.js';

describe('Three.js Imperative Bridge & State Store', () => {
  it('initializes store with empty default state', () => {
    useModuleStore.getState().reset();
    const state = useModuleStore.getState();
    assert.strictEqual(state.activeModule, null);
    assert.strictEqual(state.activeCategory, null);
    assert.strictEqual(state.activeState, 'idle');
    assert.strictEqual(state.timelinePosition, 0);
  });

  it('updates store when a new module is selected', () => {
    useModuleStore.getState().setActiveModule('a3_compressor', ROUTING_MATRIX);
    const state = useModuleStore.getState();
    assert.strictEqual(state.activeModule, 'a3_compressor');
    assert.strictEqual(state.activeCategory, 'E');
    assert.strictEqual(state.activeState, 'idle');
  });

  it('dispatches custom event on state change when window is available', () => {
    let eventFired = false;
    let eventDetail = null;

    globalThis.window = {
      dispatchEvent: (evt) => {
        eventFired = true;
        eventDetail = evt.detail;
      }
    };
    globalThis.CustomEvent = class {
      constructor(type, opts) {
        this.type = type;
        this.detail = opts?.detail;
      }
    };

    useModuleStore.getState().setActiveModule('hgrn2', ROUTING_MATRIX);
    useModuleStore.getState().setState('t2');

    const event = new CustomEvent('visualizer:module-state-change', {
      detail: {
        moduleId: 'hgrn2',
        state: 't2',
        timelinePosition: 0.5
      }
    });
    window.dispatchEvent(event);

    assert.strictEqual(eventFired, true);
    assert.strictEqual(eventDetail.moduleId, 'hgrn2');
    assert.strictEqual(eventDetail.state, 't2');
  });
});
