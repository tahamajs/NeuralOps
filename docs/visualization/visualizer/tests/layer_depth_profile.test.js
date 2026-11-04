/**
 * tests/layer_depth_profile.test.js
 * =================================
 * Comprehensive test suite for Layer-by-Layer Depth Specialization & Telemetry:
 * - Mathematical validation of normalized relative depth tau in [0.0, 1.0]
 * - Proves monotonic decrease of attention entropy across network depth
 * - Proves peak SVD compressibility (singular decay ratio) in middle transformer layers
 * - Proves monotonic increase of residual highway trunk accumulation
 * - Verifies linguistic role and pattern summary progression
 * - Verifies mock DOM telemetry card rendering across DistilBERT, RoBERTa, GPT-2, and Qwen
 */

import assert from 'node:assert';
import test, { describe, it } from 'node:test';
import { 
    getLayerDepthProfile, 
    renderLayerDepthProfile, 
    LAYER_PHASES 
} from '../src/educational/layer_depth_profile.js';

describe('Layer-by-Layer Depth Specialization Engine', () => {
    const models = ['distilbert', 'roberta', 'gpt2', 'qwen'];

    it('verifies 4 depth phases are rigorously defined with valid colors and thresholds', () => {
        assert.strictEqual(LAYER_PHASES.length, 4);
        LAYER_PHASES.forEach((phase, idx) => {
            assert.ok(phase.name.length > 10, `Phase ${idx} must have non-empty name`);
            assert.ok(phase.shortTag.length > 0, `Phase ${idx} must have non-empty shortTag`);
            assert.ok(phase.colorHex.startsWith('#'), `Phase ${idx} colorHex must be valid hex`);
            assert.strictEqual(phase.colorRgb.length, 3, `Phase ${idx} colorRgb must be 3 elements`);
            assert.ok(phase.tauMax > 0 && phase.tauMax <= 1.0, `Phase ${idx} tauMax must be in (0, 1]`);
        });
        assert.strictEqual(LAYER_PHASES[LAYER_PHASES.length - 1].tauMax, 1.0);
    });

    it('computes exact normalized relative depth tau and stage mapping across models', () => {
        const distilbertTotal = 6;
        const profile0 = getLayerDepthProfile(0, 'distilbert', distilbertTotal);
        const profileLast = getLayerDepthProfile(distilbertTotal - 1, 'distilbert', distilbertTotal);

        assert.strictEqual(profile0.normalizedDepth, 0.0);
        assert.strictEqual(profile0.phaseIndex, 0);
        assert.strictEqual(profile0.shortTag, 'Lexical / Local Syntax');

        assert.strictEqual(profileLast.normalizedDepth, 1.0);
        assert.strictEqual(profileLast.phaseIndex, 3);
        assert.strictEqual(profileLast.shortTag, 'Semantic Output');
    });

    it('proves monotonic decrease of attention entropy across network depth', () => {
        // As tokens ascend through transformer layers, attention sharpens (entropy drops)
        const totalLayers = 12;
        let prevEntropy = Infinity;

        for (let l = 0; l < totalLayers; l++) {
            const profile = getLayerDepthProfile(l, 'roberta', totalLayers);
            assert.ok(profile.attentionEntropy <= prevEntropy, `Layer ${l} entropy (${profile.attentionEntropy}) must be <= layer ${l-1} entropy (${prevEntropy})`);
            assert.ok(profile.attentionEntropy >= 1.0, `Layer ${l} entropy must be >= 1.0 nat`);
            assert.ok(profile.attentionEntropy <= 3.2, `Layer ${l} entropy must be <= 3.2 nats`);
            prevEntropy = profile.attentionEntropy;
        }

        // Check contrast between layer 0 and layer 11
        const p0 = getLayerDepthProfile(0, 'roberta', 12);
        const p11 = getLayerDepthProfile(11, 'roberta', 12);
        assert.ok(p0.attentionEntropy - p11.attentionEntropy >= 1.8, 'Expected significant entropy drop (>1.8 nats) between early and late layers');
    });

    it('proves middle layers exhibit maximum singular value decay ratio (optimal SVD compression zone)', () => {
        const totalLayers = 6;
        const p0 = getLayerDepthProfile(0, 'distilbert', totalLayers);
        const pMid = getLayerDepthProfile(3, 'distilbert', totalLayers);
        const pLast = getLayerDepthProfile(5, 'distilbert', totalLayers);

        // Middle layers have fastest singular decay and lowest recommended rank
        assert.ok(pMid.singularDecayRatio > p0.singularDecayRatio, `Mid layer decay (${pMid.singularDecayRatio}) must exceed L0 decay (${p0.singularDecayRatio})`);
        assert.ok(pMid.singularDecayRatio > pLast.singularDecayRatio, `Mid layer decay (${pMid.singularDecayRatio}) must exceed L5 decay (${pLast.singularDecayRatio})`);
        assert.strictEqual(pMid.optimalSvdRank, 24, 'Middle layer should tolerate aggressive rank r=24 compression');
        assert.strictEqual(p0.optimalSvdRank, 64, 'Input boundary layer requires higher rank r=64');
    });

    it('proves monotonic accumulation of residual highway trunk dominance', () => {
        const totalLayers = 6;
        let prevResidual = 0;

        for (let l = 0; l < totalLayers; l++) {
            const profile = getLayerDepthProfile(l, 'distilbert', totalLayers);
            assert.ok(profile.residualPercent >= prevResidual, `Residual % at L${l} (${profile.residualPercent}%) must be >= L${l-1} (${prevResidual}%)`);
            assert.strictEqual(profile.residualPercent + profile.layerDeltaPercent, 100);
            prevResidual = profile.residualPercent;
        }

        const p0 = getLayerDepthProfile(0, 'distilbert', 6);
        const p5 = getLayerDepthProfile(5, 'distilbert', 6);
        assert.ok(p0.residualPercent < 65, 'Early layers should have significant sublayer update delta');
        assert.ok(p5.residualPercent > 85, 'Deep layers should be heavily dominated by accumulated residual trunk');
    });

    it('provides distinct linguistic representations across all 4 depth phases', () => {
        const p0 = getLayerDepthProfile(0, 'distilbert', 6);
        const p1 = getLayerDepthProfile(1, 'distilbert', 6);
        const p3 = getLayerDepthProfile(3, 'distilbert', 6);
        const p5 = getLayerDepthProfile(5, 'distilbert', 6);

        assert.ok(p0.linguisticRole.includes('local n-gram'));
        assert.ok(p1.linguisticRole.includes('syntactic trees') || p1.linguisticRole.includes('local'));
        assert.ok(p3.linguisticRole.includes('polysemous') || p3.linguisticRole.includes('syntactic') || p3.linguisticRole.includes('Induction'));
        assert.ok(p5.linguisticRole.includes('global semantic') || p5.linguisticRole.includes('vocabulary'));

        // All mathematical LaTeX equations must be present
        models.forEach(m => {
            const prof = getLayerDepthProfile(2, m);
            assert.ok(prof.mathFormulaResidual.includes('\\mathbf{x}'));
            assert.ok(prof.mathFormulaEntropy.includes('\\mathcal{H}'));
        });
    });

    it('renders layer depth telemetry card into mock DOM container without throwing', () => {
        // Minimal Mock DOM Container
        const mockContainer = {
            id: 'layer-depth-profile-card',
            innerHTML: '',
            style: {},
            querySelectorAll: () => []
        };

        // Inject container into global document mock
        const prevDoc = global.document;
        global.document = {
            getElementById: (id) => id === 'layer-depth-profile-card' ? mockContainer : null,
            querySelector: () => ({ appendChild: () => {} }),
            createElement: () => mockContainer
        };

        try {
            renderLayerDepthProfile(2, 'distilbert', 'layer-depth-profile-card');
            assert.ok(mockContainer.innerHTML.includes('LAYER 2 OF 6'));
            assert.ok(mockContainer.innerHTML.includes('ATTN ENTROPY'));
            assert.ok(mockContainer.innerHTML.includes('SPECTRUM DECAY'));
            assert.ok(mockContainer.innerHTML.includes('RESIDUAL RATIO'));
            assert.ok(mockContainer.innerHTML.includes('FFN SPARSITY'));
            assert.ok(mockContainer.innerHTML.includes('L0'));
            assert.ok(mockContainer.innerHTML.includes('L5'));
        } finally {
            global.document = prevDoc;
        }
    });
});
