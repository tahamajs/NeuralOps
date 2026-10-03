/**
 * react/tests/routing.test.js
 * ===========================
 * Unit tests verifying declarative routing for all 252 modules,
 * category isolation, and strict forbidden-state rejection.
 */

import assert from 'node:assert';
import test, { describe, it } from 'node:test';
import { ROUTING_MATRIX, validateModuleState } from '../components/visualization/routing_matrix.js';
import { useModuleStore } from '../store/useModuleStore.js';

describe('React Module Routing & Forbidden Guardrails', () => {
  it('verifies a3 routes to Category E (SVD / Low-Rank)', () => {
    useModuleStore.getState().setActiveModule('a3', ROUTING_MATRIX);
    const state = useModuleStore.getState();
    assert.strictEqual(state.activeCategory, 'E');
    assert.strictEqual(ROUTING_MATRIX['a3'].category, 'E');
  });

  it('verifies hgrn2 routes to Category D (Recurrent) and NEVER renders attention matrix', () => {
    useModuleStore.getState().setActiveModule('hgrn2', ROUTING_MATRIX);
    const state = useModuleStore.getState();
    assert.strictEqual(state.activeCategory, 'D');

    // Forbidden check
    assert.throws(
      () => validateModuleState('hgrn2', 'attention_matrix'),
      /FORBIDDEN/
    );
    assert.throws(
      () => validateModuleState('hgrn2', 'softmax'),
      /FORBIDDEN/
    );
    assert.throws(
      () => validateModuleState('hgrn2', 'qkv'),
      /FORBIDDEN/
    );
  });

  it('verifies every module in ROUTING_MATRIX has a valid category A-I', () => {
    const validCats = new Set(['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I']);
    for (const [mod, cfg] of Object.entries(ROUTING_MATRIX)) {
      assert.ok(validCats.has(cfg.category), `Module ${mod} has invalid category ${cfg.category}`);
    }
  });

  it('strictly verifies category D (recurrent) modules never contain Q/K/V states', () => {
    for (const [mod, cfg] of Object.entries(ROUTING_MATRIX)) {
      if (cfg.category === 'D') {
        const hasQ = cfg.states.some(s => s.toLowerCase().includes('query'));
        const hasK = cfg.states.some(s => s.toLowerCase().includes('key'));
        const hasV = cfg.states.some(s => s.toLowerCase().includes('value') && !s.includes('singular_values'));
        assert.strictEqual(hasQ, false, `Recurrent ${mod} must not have Query`);
        assert.strictEqual(hasK, false, `Recurrent ${mod} must not have Key`);
        assert.strictEqual(hasV, false, `Recurrent ${mod} must not have Value`);
      }
    }
  });

  it('strictly verifies category E (SVD) modules never contain softmax or attention matrix', () => {
    for (const [mod, cfg] of Object.entries(ROUTING_MATRIX)) {
      if (cfg.category === 'E') {
        const hasSoftmax = cfg.states.some(s => s.toLowerCase().includes('softmax'));
        const hasAttn = cfg.states.some(s => s.toLowerCase().includes('attention_matrix'));
        assert.strictEqual(hasSoftmax, false, `SVD ${mod} must not have Softmax`);
        assert.strictEqual(hasAttn, false, `SVD ${mod} must not have Attention Matrix`);
      }
    }
  });
});
