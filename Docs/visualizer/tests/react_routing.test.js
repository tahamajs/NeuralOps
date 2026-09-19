/**
 * tests/react_routing.test.js
 * ===========================
 * Comprehensive verification suite for the Declarative React Routing Layer:
 * - Routing matrix verification for all 252 catalog modules
 * - Canonical Category mapping (A through I)
 * - Strict forbidden state rejection (Category D, Category E, Category B)
 * - Substate resolution for specialized attention mechanisms
 * - State machine reachability and progression monotonicity
 * - Zustand-compatible reactive state store lifecycle
 */

import assert from 'node:assert';
import test, { describe, it, beforeEach } from 'node:test';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

import { useModuleStore } from '../react/store/useModuleStore.js';
import {
  ROUTING_MATRIX,
  validateModuleState,
  getModuleRouting,
} from '../react/components/visualization/routing_matrix.js';
import {
  SVD_STATES,
  RECURRENT_STATES,
  DENSE_STATES,
  SPARSE_STATES,
  WINDOW_STATES,
  MOE_STATES,
  PRUNING_STATES,
  QUANT_STATES,
  HYBRID_STATES,
  SUBSTATE_MAP,
} from '../react/types/module.types.js';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

describe('Declarative React Layer: Module Routing & State Invariants', () => {
  const taxonomyPath = path.resolve(__dirname, '../src/data/module_taxonomy.json');
  const taxonomy = JSON.parse(fs.readFileSync(taxonomyPath, 'utf-8'));
  const catalogModuleIds = Object.keys(taxonomy);

  beforeEach(() => {
    useModuleStore.getState().reset();
  });

  it('verifies all 252 catalog modules are present in ROUTING_MATRIX with valid categories A-I', () => {
    assert.strictEqual(catalogModuleIds.length, 252, `Expected 252 catalog modules, got ${catalogModuleIds.length}`);
    const validCategories = new Set(['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I']);

    catalogModuleIds.forEach(id => {
      const config = ROUTING_MATRIX[id];
      assert.ok(config, `Module ${id} missing from ROUTING_MATRIX`);
      assert.ok(validCategories.has(config.category), `Module ${id} has invalid category: ${config.category}`);
      assert.ok(Array.isArray(config.states) && config.states.length > 0, `Module ${id} must have valid states`);
      assert.ok(config.defaultState, `Module ${id} must specify defaultState`);
      assert.ok(config.states.includes(config.defaultState), `Module ${id} defaultState must be in states array`);
      assert.ok(Array.isArray(config.forbidden), `Module ${id} must define forbidden array`);
    });
  });

  it('routes canonical sample modules to their exact respective architectural categories', () => {
    // Category A (Dense)
    assert.strictEqual(ROUTING_MATRIX['gqa'].category, 'A');
    assert.strictEqual(ROUTING_MATRIX['gqa_attention'].category, 'A');

    // Category B (Sparse / Approximate)
    assert.strictEqual(ROUTING_MATRIX['performer_attention'].category, 'B');
    assert.strictEqual(ROUTING_MATRIX['linformer_attention'].category, 'B');
    assert.strictEqual(ROUTING_MATRIX['nystrom_attention'].category, 'B');

    // Category C (Sliding Window)
    assert.strictEqual(ROUTING_MATRIX['adaptive_span_attention'].category, 'C');

    // Category D (Recurrent / Linear)
    assert.strictEqual(ROUTING_MATRIX['hgrn2_outer'].category, 'D');
    assert.strictEqual(ROUTING_MATRIX['mamba'].category, 'D');

    // Category E (SVD / Low-Rank)
    assert.strictEqual(ROUTING_MATRIX['a3_compressor'].category, 'E');
    assert.strictEqual(ROUTING_MATRIX['asvd_attention'].category, 'E');

    // Category F (MoE)
    assert.strictEqual(ROUTING_MATRIX['bam_umoe'].category, 'F');

    // Category G (Pruning)
    assert.strictEqual(ROUTING_MATRIX['sparsegpt'].category, 'G');

    // Category H (Quantization)
    assert.strictEqual(ROUTING_MATRIX['aegis_spectron_fp8'].category, 'H');

    // Category I (Hybrid)
    assert.strictEqual(ROUTING_MATRIX['granite_hybrid'].category, 'I');
  });

  it('supports shorthand canonical aliases (a3, hgrn2, performer, linformer, nystrom)', () => {
    assert.ok(ROUTING_MATRIX['a3'], 'Alias a3 must exist');
    assert.strictEqual(ROUTING_MATRIX['a3'].category, 'E');

    assert.ok(ROUTING_MATRIX['hgrn2'], 'Alias hgrn2 must exist');
    assert.strictEqual(ROUTING_MATRIX['hgrn2'].category, 'D');

    assert.ok(ROUTING_MATRIX['performer'], 'Alias performer must exist');
    assert.strictEqual(ROUTING_MATRIX['performer'].category, 'B');

    assert.ok(ROUTING_MATRIX['linformer'], 'Alias linformer must exist');
    assert.strictEqual(ROUTING_MATRIX['linformer'].category, 'B');

    assert.ok(ROUTING_MATRIX['nystrom'], 'Alias nystrom must exist');
    assert.strictEqual(ROUTING_MATRIX['nystrom'].category, 'B');
  });

  it('strictly enforces Category D invariants: NEVER contains Q/K/V or S×S attention matrix', () => {
    const recurrentModules = Object.keys(ROUTING_MATRIX).filter(id => ROUTING_MATRIX[id].category === 'D');
    assert.ok(recurrentModules.length >= 20, 'Must have at least 20 recurrent modules');

    const forbiddenTerms = ['qkv', 'softmax', 'attention_matrix', 'square_attention', 'dense_score_matrix'];

    recurrentModules.forEach(modId => {
      const cfg = ROUTING_MATRIX[modId];

      // 1. None of the module states may include forbidden patterns
      cfg.states.forEach(state => {
        forbiddenTerms.forEach(term => {
          assert.strictEqual(
            state.toLowerCase().includes(term),
            false,
            `Recurrent module ${modId} state "${state}" contains forbidden term "${term}"`
          );
        });
      });

      // 2. validateModuleState must throw if any forbidden state is attempted
      forbiddenTerms.forEach(badState => {
        assert.throws(
          () => validateModuleState(modId, badState),
          /FORBIDDEN/,
          `Recurrent module ${modId} must throw on forbidden state "${badState}"`
        );
      });
    });
  });

  it('strictly enforces Category E invariants: NEVER contains attention or softmax', () => {
    const svdModules = Object.keys(ROUTING_MATRIX).filter(id => ROUTING_MATRIX[id].category === 'E');
    assert.ok(svdModules.length >= 100, 'Must have at least 100 SVD modules');

    const forbiddenTerms = ['qkv', 'softmax', 'attention_matrix', 'causal_mask'];

    svdModules.forEach(modId => {
      const cfg = ROUTING_MATRIX[modId];

      // SVD states must be SVD-specific
      assert.ok(cfg.states.includes('decompose'), `SVD module ${modId} must include "decompose" state`);
      assert.ok(cfg.states.includes('singular_values'), `SVD module ${modId} must include "singular_values" state`);
      assert.ok(cfg.states.includes('truncate'), `SVD module ${modId} must include "truncate" state`);
      assert.ok(cfg.states.includes('error_heatmap'), `SVD module ${modId} must include "error_heatmap" state`);

      // Forbidden checks
      forbiddenTerms.forEach(badState => {
        assert.throws(
          () => validateModuleState(modId, badState),
          /FORBIDDEN/,
          `SVD module ${modId} must throw on forbidden state "${badState}"`
        );
      });
    });
  });

  it('strictly enforces Category B invariants: NEVER contains dense quadratic S×S matrix', () => {
    const sparseModules = Object.keys(ROUTING_MATRIX).filter(id => ROUTING_MATRIX[id].category === 'B');
    assert.ok(sparseModules.length >= 40, 'Must have at least 40 sparse modules');

    const forbiddenTerms = ['dense_attention_matrix', 'full_softmax_matrix', 'quadratic_square'];

    sparseModules.forEach(modId => {
      forbiddenTerms.forEach(badState => {
        assert.throws(
          () => validateModuleState(modId, badState),
          /FORBIDDEN/,
          `Sparse module ${modId} must throw on forbidden state "${badState}"`
        );
      });
    });
  });

  it('verifies substate resolution for specialized modules (Fisher SVD, GQA, Mamba, Performer)', () => {
    // Fisher SVD (a3)
    const a3Cfg = ROUTING_MATRIX['a3'];
    assert.ok(a3Cfg.substates.includes('fisher_weights_visible'), 'a3 must feature fisher_weights_visible substate');
    assert.ok(a3Cfg.substates.includes('ln_recalibration_visible'), 'a3 must feature ln_recalibration_visible substate');

    // GQA
    const gqaCfg = ROUTING_MATRIX['gqa'];
    assert.ok(gqaCfg.substates.includes('kv_sharing_visible'), 'gqa must feature kv_sharing_visible substate');

    // Mamba / SSM
    const mambaCfg = ROUTING_MATRIX['mamba'];
    assert.ok(mambaCfg.substates.includes('selective_scan_visible'), 'mamba must feature selective_scan_visible');
    assert.ok(mambaCfg.substates.includes('state_space_visible'), 'mamba must feature state_space_visible');

    // Performer
    const performerCfg = ROUTING_MATRIX['performer_attention'];
    assert.ok(performerCfg.substates.includes('kernel_features_visible'), 'performer must feature kernel_features_visible');
  });

  it('proves state transitions follow discrete timeline and are valid in reactive store', () => {
    const store = useModuleStore;

    // Load a3
    store.getState().setActiveModule('a3', ROUTING_MATRIX);
    assert.strictEqual(store.getState().activeModule, 'a3');
    assert.strictEqual(store.getState().activeCategory, 'E');
    assert.strictEqual(store.getState().activeState, 'idle');

    // Step through SVD states
    const svdFlow = ['show_W', 'decompose', 'singular_values', 'truncate', 'reconstruct', 'error_heatmap'];
    svdFlow.forEach(st => {
      validateModuleState('a3', st);
      store.getState().setState(st);
      assert.strictEqual(store.getState().activeState, st);
    });

    // Substate toggle
    store.getState().setSubstate('fisher_weights_visible');
    assert.strictEqual(store.getState().activeSubstate, 'fisher_weights_visible');

    // Timeline position update
    store.getState().setTimeline(0.75);
    assert.strictEqual(store.getState().timelinePosition, 0.75);

    // Switch to hgrn2 recurrent
    store.getState().setActiveModule('hgrn2', ROUTING_MATRIX);
    assert.strictEqual(store.getState().activeModule, 'hgrn2');
    assert.strictEqual(store.getState().activeCategory, 'D');
    assert.strictEqual(store.getState().activeState, 'idle');
    assert.strictEqual(store.getState().activeSubstate, null);

    // Recurrent timeline steps
    ['t0', 't1', 't2', 't3', 't4', 'output'].forEach(st => {
      validateModuleState('hgrn2', st);
      store.getState().setState(st);
      assert.strictEqual(store.getState().activeState, st);
    });
  });
});
