/**
 * ui/tree_explorer.js
 * ===================
 * Hierarchical Layer Explorer Tree & Real-time Filter Engine.
 */

/**
 * Builds the hierarchical DOM tree of layers and sublayers.
 * @param {string} modelKey - Active model
 * @param {string} moduleKey - Active module
 * @param {object} modelData - Model spatial dataset
 * @param {Function} onSelectNodeId - Callback when node is clicked
 */
export function buildHierarchicalTree(modelKey, moduleKey, modelData, onSelectNodeId) {
    const container = document.getElementById('layer-tree-container');
    const countEl = document.getElementById('insp-layer-count');
    if (!container || !modelData) return;

    container.innerHTML = '';

    const moduleInfo = (modelData.modules && modelData.modules[moduleKey]) || modelData;
    const layersMap = new Map();

    const processStream = (nodes, streamType) => {
        if (!nodes) return;
        nodes.forEach(node => {
            const lIdx = node.parent_layer_index ?? 999;
            const lName = node.parent_layer_name || (lIdx === 999 ? 'Global Projections' : `Layer ${lIdx}`);

            if (!layersMap.has(lIdx)) {
                layersMap.set(lIdx, {
                    name: lName,
                    index: lIdx,
                    baseline: [],
                    compressed: []
                });
            }
            layersMap.get(lIdx)[streamType].push(node);
        });
    };

    processStream(modelData.baseline_nodes, 'baseline');
    processStream(moduleInfo.compressed_nodes, 'compressed');

    const totalNodes = (modelData.baseline_nodes ? modelData.baseline_nodes.length : 0) +
                       (moduleInfo.compressed_nodes ? moduleInfo.compressed_nodes.length : 0);
    if (countEl) countEl.innerText = `${totalNodes} modules`;

    const rootEl = document.createElement('div');
    rootEl.className = 'tree-model-root';

    const rootHeader = document.createElement('div');
    rootHeader.className = 'tree-model-header';
    rootHeader.innerHTML = `<span>⚡</span> <span>${modelKey.toUpperCase()} · ${moduleInfo.name || moduleKey.toUpperCase()}</span>`;
    rootEl.appendChild(rootHeader);

    const sortedLayers = Array.from(layersMap.values()).sort((a, b) => a.index - b.index);

    sortedLayers.forEach((layerGroup, idx) => {
        const layerNode = document.createElement('div');
        layerNode.className = `tree-layer-node ${idx === 0 ? 'open' : ''}`;
        layerNode.dataset.layerIndex = layerGroup.index;

        const layerHeader = document.createElement('div');
        layerHeader.className = 'tree-layer-header';
        layerHeader.innerHTML = `
            <span class="tree-chevron">▶</span>
            <span class="tree-layer-title">${layerGroup.name}</span>
            <small style="color: var(--text-faint); margin-left: auto;">(${layerGroup.baseline.length + layerGroup.compressed.length})</small>
        `;

        layerHeader.addEventListener('click', (e) => {
            e.stopPropagation();
            layerNode.classList.toggle('open');
            if (window.navigateToLayer) {
                window.navigateToLayer(layerGroup.index);
            }
        });

        const layerContent = document.createElement('div');
        layerContent.className = 'tree-layer-content';

        if (layerGroup.baseline.length > 0) {
            const bGroup = document.createElement('div');
            bGroup.className = 'tree-stream-group';
            bGroup.innerHTML = `<div class="tree-stream-label">Dense Baseline</div>`;
            layerGroup.baseline.sort((a, b) => a.sequence_index - b.sequence_index).forEach(n => {
                bGroup.appendChild(createLeafNode(n, false, onSelectNodeId));
            });
            layerContent.appendChild(bGroup);
        }

        if (layerGroup.compressed.length > 0) {
            const cGroup = document.createElement('div');
            cGroup.className = 'tree-stream-group';
            cGroup.innerHTML = `<div class="tree-stream-label">${moduleInfo.name || 'Compressed'}</div>`;
            layerGroup.compressed.sort((a, b) => a.sequence_index - b.sequence_index).forEach(n => {
                cGroup.appendChild(createLeafNode(n, true, onSelectNodeId));
            });
            layerContent.appendChild(cGroup);
        }

        layerNode.appendChild(layerHeader);
        layerNode.appendChild(layerContent);
        rootEl.appendChild(layerNode);
    });

    container.appendChild(rootEl);
    setupTreeFilter();
}

function createLeafNode(node, isCompressed, onSelectNodeId) {
    const leaf = document.createElement('div');
    leaf.className = `tree-leaf ${isCompressed ? 'type-svd' : 'type-dense'}`;
    leaf.dataset.id = node.id;
    leaf.dataset.label = (node.label || node.id).toLowerCase();

    let cleanLabel = node.label || node.id;
    cleanLabel = cleanLabel.replace(/^(transformer|encoder|model)?\.?layer(s)?\.\d+\./, '');
    cleanLabel = cleanLabel.replace(/^h\.\d+\./, '');

    let badgeText = 'Dense';
    if (isCompressed) {
        if (node.sub_type === 'v_factor') badgeText = 'V-Rank';
        else if (node.sub_type === 'u_factor') badgeText = 'U-Exp';
        else if (node.sub_type === 'router') badgeText = 'Router';
        else if (node.sub_type === 'expert') badgeText = 'Expert';
        else if (node.sub_type === 'latent_kv') badgeText = 'Latent';
        else if (node.sub_type === 'sliced_linear') badgeText = 'Sliced';
        else badgeText = 'Comp';
    }

    leaf.innerHTML = `
        <span class="leaf-name" title="${node.label}">${cleanLabel}</span>
        <span class="leaf-badge">${badgeText}</span>
    `;

    leaf.addEventListener('click', (e) => {
        e.stopPropagation();
        if (onSelectNodeId) onSelectNodeId(node.id);
    });

    return leaf;
}

function setupTreeFilter() {
    const searchInput = document.getElementById('tree-search-input');
    const clearBtn = document.getElementById('btn-tree-clear');
    if (!searchInput) return;

    const filterTree = () => {
        const query = searchInput.value.trim().toLowerCase();
        const leaves = document.querySelectorAll('.tree-leaf');
        const layerNodes = document.querySelectorAll('.tree-layer-node');

        if (!query) {
            leaves.forEach(l => l.style.display = '');
            layerNodes.forEach(ln => ln.style.display = '');
            return;
        }

        layerNodes.forEach(ln => {
            let hasVisibleLeaf = false;
            const lnLeaves = ln.querySelectorAll('.tree-leaf');
            lnLeaves.forEach(leaf => {
                const label = leaf.dataset.label || '';
                const id = leaf.dataset.id || '';
                const match = label.includes(query) || id.includes(query);
                leaf.style.display = match ? '' : 'none';
                if (match) hasVisibleLeaf = true;
            });

            ln.style.display = hasVisibleLeaf ? '' : 'none';
            if (hasVisibleLeaf) ln.classList.add('open');
        });
    };

    searchInput.addEventListener('input', filterTree);
    if (clearBtn) {
        clearBtn.addEventListener('click', () => {
            searchInput.value = '';
            filterTree();
            searchInput.focus();
        });
    }
}
