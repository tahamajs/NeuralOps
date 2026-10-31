/**
 * tests/model_race.test.js
 * ========================
 * Unit tests for Feature 1.4: Side-by-Side Model Race.
 */

import assert from 'node:assert';
import test, { describe, it } from 'node:test';
import { MODEL_RACE_PROFILES, initRaceState, stepRaceSimulation } from '../src/interactive/model_race.js';

describe('Feature 1.4: Side-by-Side Model Race', () => {
    it('verifies all model race profiles have valid physical parameters', () => {
        const profileKeys = Object.keys(MODEL_RACE_PROFILES);
        assert.ok(profileKeys.length >= 6);

        profileKeys.forEach(k => {
            const p = MODEL_RACE_PROFILES[k];
            assert.ok(p.id === k);
            assert.ok(typeof p.name === 'string');
            assert.ok(typeof p.paradigm === 'string');
            assert.ok(p.layers >= 6);
            assert.ok(p.hiddenDim >= 512);
            assert.ok(p.paramCountM > 0);
            assert.ok(p.msPerToken > 0);
            assert.ok(p.gflopsPerToken > 0);
            assert.ok(p.accuracyScore > 80.0 && p.accuracyScore <= 100.0);
        });
    });

    it('initializes clean race state between two models', () => {
        const state = initRaceState('roberta_baseline', 'roberta_svd', 16);
        assert.strictEqual(state.status, 'idle');
        assert.strictEqual(state.totalTokens, 16);
        assert.strictEqual(state.trackA.tokensDone, 0);
        assert.strictEqual(state.trackB.tokensDone, 0);
        assert.strictEqual(state.verdict, null);
    });

    it('simulates race progression and determines winner with correct speedup', () => {
        // RoBERTa baseline (2.8 ms/tok) vs RoBERTa SVD (1.3 ms/tok)
        const state = initRaceState('roberta_baseline', 'roberta_svd', 8);
        state.status = 'racing';

        // Step simulation forward until both finish
        let steps = 0;
        while (state.status === 'racing' && steps < 1000) {
            stepRaceSimulation(state, 5.0); // 5ms per step
            steps++;
        }

        assert.strictEqual(state.status, 'completed');
        assert.strictEqual(state.trackA.tokensDone, 8);
        assert.strictEqual(state.trackB.tokensDone, 8);

        // SVD model (Track B) must finish earlier than baseline (Track A)
        assert.ok(state.trackB.finishTimeMs < state.trackA.finishTimeMs);
        assert.strictEqual(state.verdict.winner, 'B');
        assert.ok(state.verdict.speedup > 1.9 && state.verdict.speedup < 2.4);
        assert.ok(state.verdict.bannerMessage.includes('finished'));
        assert.ok(state.verdict.bannerMessage.includes('quality retention'));
    });
});
