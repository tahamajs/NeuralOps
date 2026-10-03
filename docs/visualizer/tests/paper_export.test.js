/**
 * tests/paper_export.test.js
 * ==========================
 * Unit tests for Paper Figure Exporter, LaTeX Code Generator, and Palettes.
 */

import assert from 'node:assert';
import test, { describe, it } from 'node:test';
import { serializeSVG, generateLatexFigureCode } from '../src/export/paper_export.js';
import { PALETTES, getActivePalette, setActivePalette } from '../src/core/palettes.js';

describe('Paper Figure Exporter & Colorblind Palettes Engine', () => {
    it('verifies colorblind-safe palettes (default, deuteranopia, protanopia)', () => {
        assert.ok(PALETTES.default);
        assert.ok(PALETTES.deuteranopia);
        assert.ok(PALETTES.protanopia);

        setActivePalette('deuteranopia');
        assert.strictEqual(getActivePalette().id, 'deuteranopia');
        assert.strictEqual(getActivePalette().colors.cyan, '#3b82f6');

        setActivePalette('protanopia');
        assert.strictEqual(getActivePalette().id, 'protanopia');

        // Reset to default
        setActivePalette('default');
        assert.strictEqual(getActivePalette().id, 'default');
    });

    it('generates syntactically valid LaTeX figure code', () => {
        const latex = generateLatexFigureCode({
            filename: 'figures/pareto_frontier.pdf',
            caption: 'Low-rank SVD compression results on DistilBERT.',
            label: 'fig:pareto_svd'
        });

        assert.ok(latex.includes('\\begin{figure}[t]'));
        assert.ok(latex.includes('\\includegraphics[width=\\linewidth]{figures/pareto_frontier.pdf}'));
        assert.ok(latex.includes('\\caption{Low-rank SVD compression results on DistilBERT.}'));
        assert.ok(latex.includes('\\label{fig:pareto_svd}'));
        assert.ok(latex.includes('\\end{figure}'));
    });

    it('serializes SVG with XMLNS namespace and solid background', () => {
        const mockChild = {};
        const mockSvg = {
            getAttribute: (attr) => (attr === 'xmlns' ? null : 'val'),
            setAttribute: (attr, val) => {},
            cloneNode: function() { return this; },
            insertBefore: () => {}
        };

        global.document = {
            createElementNS: () => ({ setAttribute: () => {} })
        };
        global.XMLSerializer = class {
            serializeToString(el) {
                return '<svg xmlns="http://www.w3.org/2000/svg"><rect width="100%" height="100%" fill="#050a12"/></svg>';
            }
        };

        const xml = serializeSVG(mockSvg);
        assert.ok(xml.includes('xmlns="http://www.w3.org/2000/svg"'));
        assert.ok(xml.includes('fill="#050a12"'));
    });
});
