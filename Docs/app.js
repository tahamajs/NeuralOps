document.addEventListener("DOMContentLoaded", async () => {
    // -------------------------------------------------------------
    // 1. Initialize Mermaid Diagrams Engine
    // -------------------------------------------------------------
    if (window.mermaid) {
        mermaid.initialize({ 
            startOnLoad: false, 
            theme: 'dark',
            securityLevel: 'loose',
            themeVariables: {
                darkMode: true,
                background: '#141721',
                primaryColor: '#3b82f6',
                primaryTextColor: '#f8fafc',
                primaryBorderColor: '#60a5fa',
                lineColor: '#94a3b8',
                secondaryColor: '#1a1e2b',
                tertiaryColor: '#0e1117'
            }
        });
    }

    // -------------------------------------------------------------
    // 2. DOM Elements
    // -------------------------------------------------------------
    const navContent = document.getElementById("nav-content");
    const searchInput = document.getElementById("search-input");
    const searchClearBtn = document.getElementById("search-clear-btn");
    const searchKbdHint = document.getElementById("search-shortcut-hint");
    const markdownContainer = document.getElementById("markdown-container");
    const mobileMenuBtn = document.getElementById("mobile-menu-btn");
    const sidebar = document.getElementById("sidebar");
    const contentViewport = document.getElementById("content-viewport");
    const progressBar = document.getElementById("reading-progress-bar");
    const backToTopBtn = document.getElementById("back-to-top-btn");
    const breadcrumbs = document.getElementById("breadcrumbs");
    const tocNav = document.getElementById("toc-nav");
    const homeBtn = document.getElementById("home-btn");
    const compendiumQuickBtn = document.getElementById("compendium-quick-btn");
    const filterChips = document.querySelectorAll(".filter-chip");

    let catalogData = {};
    let allItems = [];
    let activeFilter = "all";
    let activeItem = null;

    // Detect Mac vs Windows for Keyboard Shortcut
    if (searchKbdHint) {
        const isMac = navigator.platform.toUpperCase().indexOf('MAC') >= 0;
        searchKbdHint.textContent = isMac ? '⌘K' : 'Ctrl+K';
    }

    // Global Keyboard Shortcut: ⌘K or Ctrl+K to search
    window.addEventListener('keydown', (e) => {
        if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === 'k') {
            e.preventDefault();
            searchInput.focus();
            searchInput.select();
        }
    });

    // -------------------------------------------------------------
    // 3. Mobile Sidebar Toggle & Scroll Interactions
    // -------------------------------------------------------------
    if (mobileMenuBtn && sidebar) {
        mobileMenuBtn.addEventListener("click", () => {
            sidebar.classList.toggle("open");
        });
    }

    // Close mobile sidebar on click outside
    document.addEventListener("click", (e) => {
        if (window.innerWidth <= 768 && sidebar && sidebar.classList.contains("open")) {
            if (!sidebar.contains(e.target) && !mobileMenuBtn.contains(e.target)) {
                sidebar.classList.remove("open");
            }
        }
    });

    // Scroll Progress & Back-to-Top Button
    if (contentViewport) {
        contentViewport.addEventListener("scroll", () => {
            const scrollTop = contentViewport.scrollTop;
            const scrollHeight = contentViewport.scrollHeight - contentViewport.clientHeight;
            const progress = scrollHeight > 0 ? (scrollTop / scrollHeight) * 100 : 0;
            
            if (progressBar) progressBar.style.width = `${progress}%`;

            if (backToTopBtn) {
                if (scrollTop > 300) {
                    backToTopBtn.classList.add("visible");
                } else {
                    backToTopBtn.classList.remove("visible");
                }
            }

            // Scroll-spy for TOC
            updateTocActiveLink();
        });
    }

    if (backToTopBtn) {
        backToTopBtn.addEventListener("click", () => {
            if (contentViewport) {
                contentViewport.scrollTo({ top: 0, behavior: 'smooth' });
            }
        });
    }

    if (homeBtn) {
        homeBtn.addEventListener("click", () => {
            renderHubWelcomeScreen();
        });
    }

    if (compendiumQuickBtn) {
        compendiumQuickBtn.addEventListener("click", () => {
            loadCompendiumDirectly();
        });
    }

    // -------------------------------------------------------------
    // 4. Marked.js Markdown Configuration
    // -------------------------------------------------------------
    if (window.marked) {
        const renderer = new marked.Renderer();
        const originalCodeRenderer = renderer.code.bind(renderer);
        
        renderer.code = function(code, language, isEscaped) {
            if (language === 'mermaid') {
                return `<div class="mermaid">${code}</div>`;
            }
            return originalCodeRenderer(code, language, isEscaped);
        };
        
        marked.setOptions({ 
            renderer: renderer,
            gfm: true,
            breaks: false
        });
    }

    // KaTeX Math Macros & Utilities
    const KATEX_MACROS = {
        "\\bm": "\\boldsymbol",
        "\\mathbf": "\\boldsymbol",
        "\\llbracket": "[\\![",
        "\\rrbracket": "]\\!]",
        "\\diag": "\\operatorname{diag}",
        "\\rank": "\\operatorname{rank}",
        "\\Tr": "\\operatorname{Tr}",
        "\\softmax": "\\operatorname{Softmax}",
        "\\argmin": "\\operatorname*{arg\\,min}",
        "\\argmax": "\\operatorname*{arg\\,max}",
        "\\E": "\\mathbb{E}",
        "\\Var": "\\operatorname{Var}"
    };

    function escapeHtml(str) {
        return str
            .replace(/&/g, "&amp;")
            .replace(/</g, "&lt;")
            .replace(/>/g, "&gt;")
            .replace(/"/g, "&quot;")
            .replace(/'/g, "&#039;");
    }

    function safeRenderKaTeX(tex, isDisplay) {
        if (typeof katex === "undefined") {
            return isDisplay ? `<pre class="katex-fallback">${escapeHtml(tex)}</pre>` : `<code>${escapeHtml(tex)}</code>`;
        }
        try {
            return katex.renderToString(tex, {
                displayMode: isDisplay,
                throwOnError: false,
                trust: true,
                strict: false,
                macros: KATEX_MACROS
            });
        } catch (err) {
            console.warn("KaTeX render error:", tex, err);
            return `<span class="katex-error" title="${escapeHtml(err.message)}">${escapeHtml(tex)}</span>`;
        }
    }

    // Robust LaTeX Math & Markdown Parsing Engine
    function renderMarkdownWithKaTeX(markdownText) {
        if (!markdownText) return "";

        // Sanitize internal markdown links
        let processed = markdownText.replace(/\]\(([^)]+)\.md\)/g, "](javascript:void(0))");

        // 1. Protect Fenced Code Blocks & Inline Code
        const codeBlocks = [];
        processed = processed.replace(/(```[\s\S]*?```)/g, (match) => {
            const id = codeBlocks.length;
            codeBlocks.push(match);
            return `%%%CODEBLOCK_${id}%%%`;
        });

        const inlineCodes = [];
        processed = processed.replace(/(`[^`\n]+?`)/g, (match) => {
            const id = inlineCodes.length;
            inlineCodes.push(match);
            return `%%%INLINECODE_${id}%%%`;
        });

        // 2. Extract Display Math Blocks
        const mathBlocks = [];
        const displayMathRegex = /\$\$([\s\S]*?)\$\$|\\\[([\s\S]*?)\\\]|(\\begin\{(?:equation\*?|align\*?|aligned|gather\*?|pmatrix|bmatrix|matrix|cases)\}[\s\S]*?\\end\{(?:equation\*?|align\*?|aligned|gather\*?|pmatrix|bmatrix|matrix|cases)\})/g;

        processed = processed.replace(displayMathRegex, (match, p1, p2, p3) => {
            let tex = (p1 !== undefined ? p1 : (p2 !== undefined ? p2 : p3)).trim();
            const id = mathBlocks.length;
            mathBlocks.push(tex);
            return `\n\n%%%MATHBLOCK_${id}%%%\n\n`;
        });

        // 3. Extract Inline Math
        const mathInlines = [];
        const inlineMathRegex = /\\\(([\s\S]*?)\\\)|\$(?!\s)([^\$\n\r]+?)(?<!\s)\$/g;

        processed = processed.replace(inlineMathRegex, (match, p1, p2) => {
            let tex = (p1 !== undefined ? p1 : p2).trim();
            const id = mathInlines.length;
            mathInlines.push(tex);
            return `%%%MATHINLINE_${id}%%%`;
        });

        // 4. Restore Code Blocks before passing to Marked
        for (let i = 0; i < codeBlocks.length; i++) {
            processed = processed.replace(`%%%CODEBLOCK_${i}%%%`, codeBlocks[i]);
        }
        for (let i = 0; i < inlineCodes.length; i++) {
            processed = processed.replace(`%%%INLINECODE_${i}%%%`, inlineCodes[i]);
        }

        // 5. Parse Markdown with marked.js
        let html = marked.parse(processed);

        // 6. Replace Display Math Placeholders
        for (let i = 0; i < mathBlocks.length; i++) {
            const tex = mathBlocks[i];
            const rendered = `<div class="katex-display-container">${safeRenderKaTeX(tex, true)}</div>`;
            const pWrapped = new RegExp(`<p>\\s*%%%MATHBLOCK_${i}%%%\\s*<\\/p>`, "g");
            if (pWrapped.test(html)) {
                html = html.replace(pWrapped, rendered);
            } else {
                html = html.replace(new RegExp(`%%%MATHBLOCK_${i}%%%`, "g"), rendered);
            }
        }

        // 7. Replace Inline Math Placeholders
        for (let i = 0; i < mathInlines.length; i++) {
            const tex = mathInlines[i];
            const rendered = safeRenderKaTeX(tex, false);
            html = html.replace(new RegExp(`%%%MATHINLINE_${i}%%%`, "g"), rendered);
        }

        return html;
    }

    // -------------------------------------------------------------
    // 5. Fetch Catalog & Build Navigation
    // -------------------------------------------------------------
    try {
        let response = await fetch("catalog.json");
        if (!response.ok) response = await fetch("Docs/catalog.json");
        if (!response.ok) response = await fetch("./catalog.json");
        
        catalogData = await response.json();
        buildSidebar(catalogData);
        renderHubWelcomeScreen();
    } catch (error) {
        console.error("Failed to load catalog:", error);
        markdownContainer.innerHTML = `<div style="color:#f43f5e; text-align:center; padding: 50px;">
            <h3>Failed to load architecture catalog</h3>
            <p>Please make sure <code>catalog.json</code> exists.</p>
        </div>`;
    }

    function getParadigmTag(item) {
        const name = (item.id + " " + item.title).toLowerCase();
        if (name.includes("svd") || name.includes("spectral") || name.includes("asvd") || name.includes("dobi") || name.includes("cur")) return { text: "SVD", cls: "tag-svd", cat: "svd" };
        if (name.includes("mamba") || name.includes("ssm") || name.includes("delta") || name.includes("hgrn") || name.includes("griffin")) return { text: "SSM", cls: "tag-ssm", cat: "ssm" };
        if (name.includes("moe") || name.includes("moba") || name.includes("samba") || name.includes("route") || name.includes("cluster")) return { text: "MoE", cls: "tag-moe", cat: "moe" };
        if (name.includes("quant") || name.includes("prun") || name.includes("sparse") || name.includes("slice") || name.includes("astro")) return { text: "Quant", cls: "tag-quant", cat: "quant" };
        if (name.includes("hybrid") || name.includes("aegis") || name.includes("spectron") || name.includes("flash")) return { text: "Hybrid", cls: "tag-hybrid", cat: "hybrid" };
        return { text: "Core", cls: "tag-default", cat: "core" };
    }

    function buildSidebar(data) {
        navContent.innerHTML = "";
        allItems = [];

        const sections = [
            { key: "framework", title: "Framework Overview" },
            { key: "attentions", title: "Attention Architectures" },
            { key: "compressions", title: "Compression Engines" }
        ];

        sections.forEach(sec => {
            if (data[sec.key] && data[sec.key].length > 0) {
                const sectionDiv = document.createElement("div");
                sectionDiv.className = "nav-section";

                const titleDiv = document.createElement("div");
                titleDiv.className = "nav-section-title";
                titleDiv.innerHTML = `<span>${sec.title}</span><span class="nav-section-count">${data[sec.key].length}</span>`;
                sectionDiv.appendChild(titleDiv);

                const ul = document.createElement("ul");
                ul.className = "nav-list";

                data[sec.key].forEach(item => {
                    item.section = sec.title;
                    item.sectionKey = sec.key;
                    item.tag = getParadigmTag(item);
                    allItems.push(item);

                    const li = document.createElement("li");
                    li.className = "nav-item";
                    li.dataset.id = item.id;
                    li.dataset.section = sec.key;
                    li.dataset.cat = item.tag.cat;

                    li.innerHTML = `
                        <span class="nav-item-title">${item.title}</span>
                        <span class="nav-badge-tag ${item.tag.cls}">${item.tag.text}</span>
                    `;

                    li.addEventListener("click", () => {
                        selectNavItem(li, item);
                    });

                    ul.appendChild(li);
                });

                sectionDiv.appendChild(ul);
                navContent.appendChild(sectionDiv);
            }
        });
    }

    function selectNavItem(element, item) {
        document.querySelectorAll(".nav-item").forEach(el => el.classList.remove("active"));
        if (element) element.classList.add("active");
        
        if (window.innerWidth <= 768 && sidebar) {
            sidebar.classList.remove("open");
        }
        
        loadMarkdown(item);
    }

    // -------------------------------------------------------------
    // 6. Search & Filter System
    // -------------------------------------------------------------
    if (searchInput) {
        searchInput.addEventListener("input", (e) => {
            const query = e.target.value.toLowerCase().trim();
            if (searchClearBtn) {
                searchClearBtn.classList.toggle("hidden", query.length === 0);
            }
            filterNavItems();
        });
    }

    if (searchClearBtn) {
        searchClearBtn.addEventListener("click", () => {
            searchInput.value = "";
            searchClearBtn.classList.add("hidden");
            filterNavItems();
            searchInput.focus();
        });
    }

    // Paradigm Filter Chips
    filterChips.forEach(chip => {
        chip.addEventListener("click", () => {
            filterChips.forEach(c => c.classList.remove("active"));
            chip.classList.add("active");
            activeFilter = chip.dataset.filter;
            filterNavItems();
        });
    });

    function filterNavItems() {
        const query = searchInput ? searchInput.value.toLowerCase().trim() : "";
        const navItems = document.querySelectorAll(".nav-item");
        const navSections = document.querySelectorAll(".nav-section");

        navItems.forEach(item => {
            const title = item.querySelector(".nav-item-title").textContent.toLowerCase();
            const secKey = item.dataset.section;
            const cat = item.dataset.cat;

            // Search query match
            const matchesQuery = !query || title.includes(query);

            // Category filter match
            let matchesFilter = true;
            if (activeFilter === "attentions") matchesFilter = (secKey === "attentions");
            else if (activeFilter === "compressions") matchesFilter = (secKey === "compressions");
            else if (activeFilter === "svd") matchesFilter = (cat === "svd");
            else if (activeFilter === "ssm") matchesFilter = (cat === "ssm");
            else if (activeFilter === "moe") matchesFilter = (cat === "moe");

            if (matchesQuery && matchesFilter) {
                item.style.display = "flex";
            } else {
                item.style.display = "none";
            }
        });

        // Hide empty sections
        navSections.forEach(sec => {
            const visibleChildren = sec.querySelectorAll(".nav-item:not([style*='display: none'])");
            sec.style.display = visibleChildren.length > 0 ? "block" : "none";
        });
    }

    // -------------------------------------------------------------
    // 7. Welcome Hub Screen
    // -------------------------------------------------------------
    function renderHubWelcomeScreen() {
        activeItem = null;
        document.querySelectorAll(".nav-item").forEach(el => el.classList.remove("active"));
        
        if (breadcrumbs) {
            breadcrumbs.innerHTML = `
                <span class="breadcrumb-item home-crumb">Hub</span>
                <span class="breadcrumb-separator">/</span>
                <span class="breadcrumb-item current-section">Overview</span>
            `;
        }

        if (tocNav) tocNav.innerHTML = `<span style="font-size:0.75rem; color:var(--text-faint);">Select a module to view contents</span>`;

        const totalAttn = catalogData.attentions ? catalogData.attentions.length : 108;
        const totalComp = catalogData.compressions ? catalogData.compressions.length : 58;

        markdownContainer.innerHTML = `
            <div class="hub-hero">
                <div class="hub-badge">
                    <svg width="12" height="12" viewBox="0 0 24 24" fill="currentColor"><circle cx="12" cy="12" r="10"/></svg>
                    <span>Publication-Grade Research Repository</span>
                </div>
                <h1>Attention & Neural Compression Hub</h1>
                <p>Explore 180+ publication-grade mathematical derivations, analytical FLOPs complexity proofs, tensor shape execution traces, and production Triton kernel blueprints.</p>
                
                <div class="metrics-grid">
                    <div class="metric-card">
                        <div class="metric-value accent-blue">${totalAttn}+</div>
                        <div class="metric-label">Attention Architectures</div>
                        <div class="metric-sub">Dense, SVD, SSM, MoE & Fused</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value accent-emerald">${totalComp}+</div>
                        <div class="metric-label">Compression Engines</div>
                        <div class="metric-sub">SVD, SliceGPT, SparseGPT, KV</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value accent-amber">5</div>
                        <div class="metric-label">Next-Gen Hybrids</div>
                        <div class="metric-sub">Cross-paradigm innovations</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value accent-indigo">1,600+</div>
                        <div class="metric-label">KaTeX TeX Proofs</div>
                        <div class="metric-sub">Triton acceleration blueprints</div>
                    </div>
                </div>

                <div class="hub-section-title">
                    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polygon points="12 2 15.09 8.26 22 9.27 17 14.14 18.18 21.02 12 17.77 5.82 21.02 7 14.14 2 9.27 8.91 8.26 12 2"></polygon></svg>
                    <span>Featured Architectures & Breakthrough Hybrids</span>
                </div>

                <div class="hub-cards-grid">
                    <div class="hub-card" data-id="mamba_svd_flash">
                        <div class="hub-card-header">
                            <span class="hub-card-tag tag-hybrid">Next-Gen Hybrid</span>
                            <span style="color:var(--accent-sky); font-size:0.8rem;">O(N) Scan</span>
                        </div>
                        <h3>MambaSVD-Flash</h3>
                        <p>Tri-hybrid unifying chunked Selective State-Space associative scans, low-rank SVD projections, and Hopper FP8 WGMMA tensor cores.</p>
                        <div class="hub-card-meta">
                            <span>FP8 / Hopper</span>
                            <span>Triton Ready</span>
                        </div>
                    </div>

                    <div class="hub-card" data-id="svd_linear_attention">
                        <div class="hub-card-header">
                            <span class="hub-card-tag tag-svd">Spectral SVD</span>
                            <span style="color:var(--accent-sky); font-size:0.8rem;">Rank Factorized</span>
                        </div>
                        <h3>SVD Linear Attention</h3>
                        <p>Zero-shot parameter compression for Multi-Head Attention via truncated Singular Value Decomposition and semi-orthogonal weights.</p>
                        <div class="hub-card-meta">
                            <span>Memory -40%</span>
                            <span>Exact Truncation</span>
                        </div>
                    </div>

                    <div class="hub-card" data-id="rotational_slicegpt_int4">
                        <div class="hub-card-header">
                            <span class="hub-card-tag tag-quant">Compression</span>
                            <span style="color:var(--accent-emerald); font-size:0.8rem;">4-Bit Quant</span>
                        </div>
                        <h3>Rotational-SliceGPT-INT4</h3>
                        <p>Orthogonal rotational invariance coupled with second-order Hessian slicing and Astro INT4 symmetric codebook quantization.</p>
                        <div class="hub-card-meta">
                            <span>4x VRAM Savings</span>
                            <span>Hessian Calibration</span>
                        </div>
                    </div>

                    <div class="hub-card" data-id="moe_deltakv_paged">
                        <div class="hub-card-header">
                            <span class="hub-card-tag tag-moe">Sparse MoE</span>
                            <span style="color:var(--accent-amber); font-size:0.8rem;">Paged Cache</span>
                        </div>
                        <h3>MoE-DeltaKV-Paged</h3>
                        <p>Mixture of Block Attention (MoBA) sparse routing with paged temporal difference delta KV-cache compression.</p>
                        <div class="hub-card-meta">
                            <span>128k Context</span>
                            <span>Paged Memory</span>
                        </div>
                    </div>

                    <div class="hub-card" data-id="flash_attention3_fp8">
                        <div class="hub-card-header">
                            <span class="hub-card-tag tag-hybrid">Hardware Fused</span>
                            <span style="color:var(--accent-sky); font-size:0.8rem;">Hopper TMA</span>
                        </div>
                        <h3>FlashAttention-3 FP8</h3>
                        <p>Asynchronous Tensor Memory Accelerator (TMA) and WGMMA warped GEMM execution pipeline for maximum throughput.</p>
                        <div class="hub-card-meta">
                            <span>800+ TFLOPs</span>
                            <span>Hardware Fused</span>
                        </div>
                    </div>

                    <div class="hub-card" data-id="NEXT_GEN_INNOVATIONS_AND_SYNTHESIS_MASTER_COMPENDIUM">
                        <div class="hub-card-header">
                            <span class="hub-card-tag tag-hybrid">Master Synthesis</span>
                            <span style="color:var(--accent-sky); font-size:0.8rem;">Compendium</span>
                        </div>
                        <h3>Innovations Compendium</h3>
                        <p>Comprehensive architectural cross-synthesis and unified Triton roadmap across all 108 attention modules and 58 compression engines.</p>
                        <div class="hub-card-meta">
                            <span>Master Blueprint</span>
                            <span>Layerwise Matrix</span>
                        </div>
                    </div>
                </div>
            </div>
        `;

        // Add 1-click listeners to featured cards
        document.querySelectorAll(".hub-card").forEach(card => {
            card.addEventListener("click", () => {
                const targetId = card.dataset.id;
                if (targetId === "NEXT_GEN_INNOVATIONS_AND_SYNTHESIS_MASTER_COMPENDIUM") {
                    loadCompendiumDirectly();
                } else {
                    const found = allItems.find(it => it.id === `${targetId}.md` || it.id === targetId || it.id.includes(targetId));
                    if (found) {
                        const targetLi = document.querySelector(`.nav-item[data-id="${found.id}"]`);
                        selectNavItem(targetLi, found);
                    }
                }
            });
        });

        if (contentViewport) contentViewport.scrollTo({ top: 0, behavior: 'smooth' });
    }

    async function loadCompendiumDirectly() {
        const item = {
            id: "NEXT_GEN_INNOVATIONS_AND_SYNTHESIS_MASTER_COMPENDIUM.md",
            title: "Next-Gen Innovations & Synthesis Compendium",
            section: "Framework Overview",
            path: "Codes/framework/NEXT_GEN_INNOVATIONS_AND_SYNTHESIS_MASTER_COMPENDIUM.md"
        };
        await loadMarkdown(item);
    }

    // -------------------------------------------------------------
    // 8. Markdown Document Loader & Renderer
    // -------------------------------------------------------------
    async function loadMarkdown(item) {
        activeItem = item;

        // Update Breadcrumbs
        if (breadcrumbs) {
            breadcrumbs.innerHTML = `
                <span class="breadcrumb-item home-crumb" id="crumb-home">Hub</span>
                <span class="breadcrumb-separator">/</span>
                <span class="breadcrumb-item">${item.section || 'Documentation'}</span>
                <span class="breadcrumb-separator">/</span>
                <span class="breadcrumb-item current-section">${item.title}</span>
            `;
            const crumbHome = document.getElementById("crumb-home");
            if (crumbHome) crumbHome.addEventListener("click", renderHubWelcomeScreen);
        }

        markdownContainer.innerHTML = `
            <div style="text-align:center; padding: 80px 20px; color: var(--text-muted);">
                <div class="skeleton-line" style="width: 50%; margin: 0 auto 16px auto; height: 28px;"></div>
                <div class="skeleton-line" style="width: 80%; margin: 0 auto 10px auto;"></div>
                <div class="skeleton-line" style="width: 65%; margin: 0 auto 30px auto;"></div>
                <div style="font-size: 0.9rem; font-family: var(--font-mono);">Loading ${item.title}...</div>
            </div>
        `;

        const candidatePaths = [
            item.path,
            item.path.replace(/^\.\.\//, ""),
            item.path.replace(/^\.\.\//, "./"),
            `./${item.path}`,
            `Codes/framework/${item.id}`,
            `Codes/framework/attentions/${item.id}`,
            `Codes/framework/compression_engine/${item.id}`,
            `../Codes/framework/attentions/${item.id}`,
            `../Codes/framework/compression_engine/${item.id}`
        ];

        let markdownText = null;
        for (const candidate of candidatePaths) {
            try {
                const response = await fetch(candidate);
                if (response.ok) {
                    markdownText = await response.text();
                    break;
                }
            } catch (err) {
                // Continue to next candidate path
            }
        }

        if (markdownText !== null) {
            try {
                // Render Markdown with KaTeX math protection
                const htmlContent = renderMarkdownWithKaTeX(markdownText);
                markdownContainer.innerHTML = htmlContent;

                // Build On-This-Page Table of Contents (TOC)
                buildTableOfContents();

                // Render Mermaid Diagrams
                if (window.mermaid) {
                    try {
                        const mermaidElements = document.querySelectorAll('.mermaid');
                        if (mermaidElements.length > 0) {
                            mermaid.init(undefined, mermaidElements);
                        }
                    } catch (mErr) {
                        console.warn("Mermaid rendering warning:", mErr);
                    }
                }

                // Render Prism Syntax Highlighting
                if (window.Prism) {
                    Prism.highlightAllUnder(markdownContainer);
                }

                // Add Copy Buttons to Code Blocks
                document.querySelectorAll('.markdown-body pre').forEach(pre => {
                    if (!pre.querySelector('.copy-code-btn')) {
                        const btn = document.createElement('button');
                        btn.className = 'copy-code-btn';
                        btn.textContent = 'Copy';
                        btn.addEventListener('click', async () => {
                            const code = pre.querySelector('code')?.innerText || pre.innerText;
                            try {
                                await navigator.clipboard.writeText(code);
                                btn.textContent = 'Copied!';
                                btn.style.background = 'var(--accent-emerald)';
                                btn.style.borderColor = 'var(--accent-emerald)';
                                setTimeout(() => { 
                                    btn.textContent = 'Copy';
                                    btn.style.background = '';
                                    btn.style.borderColor = '';
                                }, 2000);
                            } catch (e) {
                                btn.textContent = 'Failed';
                            }
                        });
                        pre.appendChild(btn);
                    }
                });

                // Scroll viewport to top
                if (contentViewport) contentViewport.scrollTo({ top: 0, behavior: 'smooth' });

            } catch (renderError) {
                console.error("Rendering error:", renderError);
                markdownContainer.innerHTML = `
                    <div style="text-align:center; padding: 60px 20px; color: var(--accent-rose);">
                        <h3>Rendering Compilation Error</h3>
                        <p>${renderError.message}</p>
                    </div>
                `;
            }
        } else {
            markdownContainer.innerHTML = `
                <div style="text-align:center; padding: 60px 20px; color: var(--accent-rose);">
                    <h3>Document Not Found</h3>
                    <p>Could not load the documentation file for <strong>${item.title}</strong>.</p>
                </div>
            `;
        }
    }

    // -------------------------------------------------------------
    // 9. Table of Contents & Scroll-Spy
    // -------------------------------------------------------------
    function buildTableOfContents() {
        if (!tocNav) return;
        tocNav.innerHTML = "";

        const headings = markdownContainer.querySelectorAll("h2, h3");
        if (headings.length === 0) {
            tocNav.innerHTML = `<span style="font-size:0.75rem; color:var(--text-faint);">No section headers</span>`;
            return;
        }

        headings.forEach((heading, idx) => {
            const id = heading.id || `section-${idx}`;
            heading.id = id;

            const link = document.createElement("a");
            link.className = "toc-link";
            link.href = `#${id}`;
            link.textContent = heading.textContent.replace(/^#+\s*/, "");
            link.title = link.textContent;
            
            if (heading.tagName.toLowerCase() === "h3") {
                link.style.paddingLeft = "12px";
                link.style.fontSize = "0.75rem";
            }

            link.addEventListener("click", (e) => {
                e.preventDefault();
                heading.scrollIntoView({ behavior: "smooth", block: "start" });
            });

            tocNav.appendChild(link);
        });

        updateTocActiveLink();
    }

    function updateTocActiveLink() {
        if (!tocNav || !contentViewport) return;
        const headings = markdownContainer.querySelectorAll("h2, h3");
        const tocLinks = tocNav.querySelectorAll(".toc-link");

        let currentActive = null;
        const viewportTop = contentViewport.scrollTop;

        headings.forEach(heading => {
            const top = heading.offsetTop - 100;
            if (viewportTop >= top) {
                currentActive = heading.id;
            }
        });

        tocLinks.forEach(link => {
            if (currentActive && link.getAttribute("href") === `#${currentActive}`) {
                link.classList.add("active");
            } else {
                link.classList.remove("active");
            }
        });
    }
});
