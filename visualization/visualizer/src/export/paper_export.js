/**
 * export/paper_export.js
 * ======================
 * Publication-Grade Academic Figure Exporter & LaTeX Caption Generator.
 * Exports high-resolution vector SVG graphics and publication-ready LaTeX figure code
 * for inclusion in ICML, NeurIPS, ICLR, and ACL conference papers.
 */

/**
 * Serializes an SVG element into a standalone vector file.
 * @param {SVGElement} svgElement - SVG DOM element
 * @param {Object} [options] - Options { filename, scale }
 * @returns {string} Serialized XML string
 */
export function serializeSVG(svgElement, options = {}) {
    if (!svgElement) throw new Error('SVG element is required for export');

    const clone = svgElement.cloneNode(true);
    if (!clone.getAttribute('xmlns')) {
        clone.setAttribute('xmlns', 'http://www.w3.org/2000/svg');
    }
    if (!clone.getAttribute('xmlns:xlink')) {
        clone.setAttribute('xmlns:xlink', 'http://www.w3.org/1999/xlink');
    }

    // Ensure background is solid for paper publication
    const rect = document.createElementNS('http://www.w3.org/2000/svg', 'rect');
    rect.setAttribute('width', '100%');
    rect.setAttribute('height', '100%');
    rect.setAttribute('fill', options.background || '#050a12');
    clone.insertBefore(rect, clone.firstChild);

    const serializer = new XMLSerializer();
    return serializer.serializeToString(clone);
}

/**
 * Triggers a client-side download of the SVG element.
 */
export function exportAsSVG(svgElement, options = {}) {
    const filename = options.filename || 'figure_neuralops_attention.svg';
    const xml = serializeSVG(svgElement, options);
    const blob = new Blob([xml], { type: 'image/svg+xml;charset=utf-8' });
    const url = URL.createObjectURL(blob);

    if (typeof document !== 'undefined') {
        const a = document.createElement('a');
        a.href = url;
        a.download = filename;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        setTimeout(() => URL.revokeObjectURL(url), 1000);
    }
    return url;
}

/**
 * Generates publication-ready LaTeX figure environment code.
 * @param {Object} options - { filename, caption, label, width }
 * @returns {string} LaTeX code snippet
 */
export function generateLatexFigureCode(options = {}) {
    const filename = options.filename || 'figures/attention_compression_pareto.pdf';
    const caption = options.caption || 'Empirical Pareto frontier and low-rank singular value spectrum across transformer depth, showing graceful accuracy preservation under 6× rank compression.';
    const label = options.label || 'fig:svd_attention_pareto';
    const width = options.width || '\\linewidth';

    return `\\begin{figure}[t]
  \\centering
  \\includegraphics[width=${width}]{${filename}}
  \\caption{${caption}}
  \\label{${label}}
\\end{figure}`;
}
