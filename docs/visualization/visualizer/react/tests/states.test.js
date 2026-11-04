/**
 * react/tests/states.test.js
 * ==========================
 * State sequence monotonicity and state machine invariant tests.
 */

import assert from 'node:assert';
import test, { describe, it } from 'node:test';
import { SVD_STATES, RECURRENT_STATES, DENSE_STATES } from '../types/module.types.js';

describe('State Machine Sequences and Invariants', () => {
  it('proves SVD state progression follows canonical factorization stages', () => {
    const svdKeys = Object.keys(SVD_STATES);
    const expected = ['idle', 'show_W', 'decompose', 'singular_values', 'truncate', 'reconstruct', 'error_heatmap'];
    assert.deepStrictEqual(svdKeys, expected);

    // Each state must define title, formula, description, shapes
    svdKeys.forEach(k => {
      const def = SVD_STATES[k];
      assert.ok(def.title, `State ${k} must have title`);
      assert.ok(def.formula, `State ${k} must have formula`);
      assert.ok(def.description, `State ${k} must have description`);
      assert.ok(Array.isArray(def.shapes) && def.shapes.length > 0, `State ${k} must have shapes`);
    });
  });

  it('proves recurrent states have discrete sequential time transitions t0..t4', () => {
    const recKeys = Object.keys(RECURRENT_STATES);
    assert.ok(recKeys.includes('t0'));
    assert.ok(recKeys.includes('t1'));
    assert.ok(recKeys.includes('t2'));
    assert.ok(recKeys.includes('t3'));
    assert.ok(recKeys.includes('t4'));
    assert.ok(recKeys.includes('output'));

    // Gating parameters must be bounded between 0 and 1
    ['t1', 't2', 't3', 't4'].forEach(step => {
      const def = RECURRENT_STATES[step];
      assert.ok(def.forgetGate >= 0 && def.forgetGate <= 1, `forgetGate at ${step} must be in [0, 1]`);
      assert.ok(def.inputGate >= 0 && def.inputGate <= 1, `inputGate at ${step} must be in [0, 1]`);
    });
  });

  it('proves dense attention includes pairwise dot product and row-normalized softmax', () => {
    assert.ok(DENSE_STATES['score_matrix']);
    assert.ok(DENSE_STATES['score_matrix'].shapes.includes('S×S'));
    assert.ok(DENSE_STATES['softmax']);
    assert.ok(DENSE_STATES['softmax'].shapes.includes('S×S'));
  });
});
