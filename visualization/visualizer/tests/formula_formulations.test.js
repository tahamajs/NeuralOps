/**
 * tests/formula_formulations.test.js
 * ==================================
 * Verification suite for Complete Mathematical Formulations across all modules & components:
 * - 100% coverage of modules in modules-catalog.json (no empty or generic placeholder formulas)
 * - 100% coverage of components in modules_components.json (all sublayer blocks have math_latex)
 * - Sublayer-level operational tensor formulations in getNodeMathematicalFormulation
 * - KaTeX rendering pipeline and clipboard copy functionality
 */

import assert from 'node:assert';
import test, { describe, it } from 'node:test';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

describe('Mathematical Formulations Registry Audit', () => {
    const catalogPath = path.resolve(__dirname, '../modules-catalog.json');
    const catalogData = JSON.parse(fs.readFileSync(catalogPath, 'utf-8'));
    const modules = catalogData.modules || [];

    it('verifies all catalog modules contain non-empty mathematical formulations', () => {
        assert.ok(modules.length >= 240, `Expected at least 240 modules, found ${modules.length}`);

        const missingMath = [];
        const genericMath = [];

        modules.forEach(m => {
            const math = (m.math || '').trim();
            if (!math) {
                missingMath.push(m.id);
            } else if (math === 'W_{comp} = \\mathcal{C}(W_{dense})' || math === 'W_{comp} = \\mathcal{F}(W_{dense})') {
                genericMath.push(m.id);
            }
        });

        assert.strictEqual(missingMath.length, 0, `Modules missing math: ${missingMath.join(', ')}`);
        assert.strictEqual(genericMath.length, 0, `Modules with generic placeholder: ${genericMath.join(', ')}`);
    });

    it('verifies 100% of components in modules_components.json have math_latex', () => {
        const compsPath = path.resolve(__dirname, '../modules_components.json');
        const compsData = JSON.parse(fs.readFileSync(compsPath, 'utf-8'));
        const compModules = compsData.modules || [];

        let totalComponents = 0;
        const missingComponentMath = [];

        compModules.forEach(m => {
            (m.components || []).forEach(c => {
                totalComponents++;
                const cmath = (c.math_latex || '').trim();
                if (!cmath) {
                    missingComponentMath.push(`${m.id}:${c.id}`);
                }
            });
        });

        assert.ok(totalComponents >= 700, `Expected at least 700 components, found ${totalComponents}`);
        assert.strictEqual(missingComponentMath.length, 0, `Components missing math_latex: ${missingComponentMath.length}`);
    });
});

describe('Sublayer Node Mathematical Formulation Engine', () => {
    it('generates exact KaTeX formulations for dense and low-rank projections', async () => {
        const { getNodeMathematicalFormulation } = await import('../src/educational/formula_renderer.js');

        // Dense Query
        const denseQ = getNodeMathematicalFormulation({ id: 'transformer.layer.0.attention.q_lin', sub_type: 'dense_linear' }, true);
        assert.ok(denseQ !== null);
        assert.ok(denseQ.formula.includes('W_Q'));
        assert.ok(denseQ.title.includes('Query'));

        // SVD Factor V
        const svdV = getNodeMathematicalFormulation({ id: 'transformer.layer.0.attention.q_lin_v', sub_type: 'v_factor' }, false);
        assert.ok(svdV !== null);
        assert.ok(svdV.formula.includes('V_{Q, r}^\\top'));
        assert.ok(svdV.formula.includes('r=64'));

        // SVD Factor U
        const svdU = getNodeMathematicalFormulation({ id: 'transformer.layer.0.attention.q_lin_u', sub_type: 'u_factor' }, false);
        assert.ok(svdU !== null);
        assert.ok(svdU.formula.includes('U_{Q, r}^\\top'));

        // FFN GELU
        const ffnNode = getNodeMathematicalFormulation({ id: 'transformer.layer.0.ffn.intermediate' }, false);
        assert.ok(ffnNode !== null);
        assert.ok(ffnNode.formula.includes('GELU'));

        // LayerNorm
        const normNode = getNodeMathematicalFormulation({ id: 'transformer.layer.0.attention_norm', type: 'norm' }, false);
        assert.ok(normNode !== null);
        assert.ok(normNode.formula.includes('\\text{LN}(x)'));
        assert.ok(normNode.formula.includes('\\sqrt{\\sigma^2 + \\epsilon}'));

        // MoE Router
        const routerNode = getNodeMathematicalFormulation({ id: 'transformer.layer.0.moe.router', sub_type: 'router' }, false);
        assert.ok(routerNode !== null);
        assert.ok(routerNode.formula.includes('TopK'));

        // Residual Highway
        const resNode = getNodeMathematicalFormulation({ id: 'residual_highway_0' }, false);
        assert.ok(resNode !== null);
        assert.ok(resNode.formula.includes('\\nabla_X \\mathcal{L}'));
    });

    it('renders multi-tier mathematical cards into inspector DOM target', async () => {
        const elementsById = {
            'formula-latex-body': {
                innerHTML: '',
                appendChild: function(c) { this.children.push(c); return c; },
                querySelectorAll: () => [],
                children: []
            }
        };

        const originalDoc = global.document;
        global.document = {
            getElementById: (id) => elementsById[id] || null,
            createElement: (tag) => ({
                tagName: tag.toUpperCase(),
                style: {},
                children: [],
                innerHTML: '',
                className: '',
                appendChild: function(c) { this.children.push(c); return c; },
                querySelectorAll: () => [],
                setAttribute: () => {},
                getAttribute: () => ''
            })
        };

        const { renderLatexFormula } = await import('../src/educational/formula_renderer.js');

        const sampleMod = {
            id: 'SOAFusedAttention',
            name: 'SOA Fused Attention',
            paradigm: 'svd',
            math_latex: 'W_Q, W_K, W_V \\in \\mathbb{R}^{D \\times D}'
        };

        const sampleNode = {
            id: 'transformer.layer.0.attention.q_lin_v',
            sub_type: 'v_factor',
            label: 'Factor V'
        };

        renderLatexFormula(sampleMod, null, sampleNode, false, 'distilbert');

        assert.ok(elementsById['formula-latex-body'].children.length > 0);
        const container = elementsById['formula-latex-body'].children[0];
        assert.ok(container.children.length >= 3, 'Should render Node card, Paradigm card, and Scaling card');

        global.document = originalDoc;
    });
});
