document.addEventListener("DOMContentLoaded", async () => {
    // Initialize Mermaid
    if (window.mermaid) {
        mermaid.initialize({ 
            startOnLoad: false, 
            theme: 'dark',
            securityLevel: 'loose',
            themeVariables: {
                darkMode: true,
                background: '#16181d',
                primaryColor: '#3b82f6',
                primaryTextColor: '#f8fafc',
                primaryBorderColor: '#60a5fa',
                lineColor: '#94a3b8',
                secondaryColor: '#1e293b',
                tertiaryColor: '#0f172a'
            }
        });
    }

    const navContent = document.getElementById("nav-content");
    const searchInput = document.getElementById("search-input");
    const markdownContainer = document.getElementById("markdown-container");
    const currentDocTitle = document.getElementById("current-doc-title");
    const mobileMenuBtn = document.getElementById("mobile-menu-btn");
    const sidebar = document.getElementById("sidebar");

    let catalogData = {};
    let allItems = [];

    // Mobile Menu Toggle
    if (mobileMenuBtn && sidebar) {
        mobileMenuBtn.addEventListener("click", () => {
            sidebar.classList.toggle("open");
        });
    }

    // Configure Marked Renderer
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

    // Robust LaTeX Math Macros
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
            console.warn("KaTeX compilation warning:", tex, err);
            return `<span class="katex-error" title="${escapeHtml(err.message)}">${escapeHtml(tex)}</span>`;
        }
    }

    // Comprehensive Markdown + LaTeX Parsing Pipeline
    function renderMarkdownWithKaTeX(markdownText) {
        if (!markdownText) return "";

        // 1. Sanitize local markdown cross-references
        let processed = markdownText.replace(/\]\(([^)]+)\.md\)/g, "](javascript:void(0))");

        // 2. Protect Code Blocks (fenced ```...``` and inline `...`)
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

        // 3. Extract Display Math Blocks ($$...$$, \[...\], \begin{...}...\end{...})
        const mathBlocks = [];
        const displayMathRegex = /\$\$([\s\S]*?)\$\$|\\\[([\s\S]*?)\\\]|(\\begin\{(?:equation\*?|align\*?|aligned|gather\*?|pmatrix|bmatrix|matrix|cases)\}[\s\S]*?\\end\{(?:equation\*?|align\*?|aligned|gather\*?|pmatrix|bmatrix|matrix|cases)\})/g;

        processed = processed.replace(displayMathRegex, (match, p1, p2, p3) => {
            let tex = (p1 !== undefined ? p1 : (p2 !== undefined ? p2 : p3)).trim();
            const id = mathBlocks.length;
            mathBlocks.push(tex);
            return `\n\n%%%MATHBLOCK_${id}%%%\n\n`;
        });

        // 4. Extract Inline Math (\(...\) and $...$)
        const mathInlines = [];
        const inlineMathRegex = /\\\(([\s\S]*?)\\\)|\$(?!\s)([^\$\n\r]+?)(?<!\s)\$/g;

        processed = processed.replace(inlineMathRegex, (match, p1, p2) => {
            let tex = (p1 !== undefined ? p1 : p2).trim();
            const id = mathInlines.length;
            mathInlines.push(tex);
            return `%%%MATHINLINE_${id}%%%`;
        });

        // 5. Restore Code Blocks before passing to Marked
        for (let i = 0; i < codeBlocks.length; i++) {
            processed = processed.replace(`%%%CODEBLOCK_${i}%%%`, codeBlocks[i]);
        }
        for (let i = 0; i < inlineCodes.length; i++) {
            processed = processed.replace(`%%%INLINECODE_${i}%%%`, inlineCodes[i]);
        }

        // 6. Convert Markdown to HTML
        let html = marked.parse(processed);

        // 7. Inject Rendered Display Math
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

        // 8. Inject Rendered Inline Math
        for (let i = 0; i < mathInlines.length; i++) {
            const tex = mathInlines[i];
            const rendered = safeRenderKaTeX(tex, false);
            html = html.replace(new RegExp(`%%%MATHINLINE_${i}%%%`, "g"), rendered);
        }

        return html;
    }

    // Fetch catalog with multiple path fallbacks
    try {
        let response = await fetch("catalog.json");
        if (!response.ok) {
            response = await fetch("Docs/catalog.json");
        }
        if (!response.ok) {
            response = await fetch("./catalog.json");
        }
        catalogData = await response.json();
        buildSidebar(catalogData);
    } catch (error) {
        console.error("Failed to load catalog:", error);
        markdownContainer.innerHTML = `<div style="color:#ef4444; text-align:center; padding: 40px;">
            <h3>Failed to load module catalog</h3>
            <p>Please make sure <code>catalog.json</code> exists.</p>
        </div>`;
    }

    function buildSidebar(data) {
        navContent.innerHTML = "";
        allItems = [];

        const sections = [
            { key: "framework", title: "Framework Overview" },
            { key: "attentions", title: "Attention Modules" },
            { key: "compressions", title: "Compression Engines" }
        ];

        sections.forEach(sec => {
            if (data[sec.key] && data[sec.key].length > 0) {
                const sectionDiv = document.createElement("div");
                sectionDiv.className = "nav-section";

                const titleDiv = document.createElement("div");
                titleDiv.className = "nav-section-title";
                titleDiv.textContent = `${sec.title} (${data[sec.key].length})`;
                sectionDiv.appendChild(titleDiv);

                const ul = document.createElement("ul");
                ul.className = "nav-list";

                data[sec.key].forEach(item => {
                    item.section = sec.title;
                    allItems.push(item);

                    const li = document.createElement("li");
                    li.className = "nav-item";
                    li.textContent = item.title;
                    li.title = item.title;
                    li.dataset.path = item.path;
                    li.dataset.id = item.id;

                    li.addEventListener("click", () => {
                        document.querySelectorAll(".nav-item").forEach(el => el.classList.remove("active"));
                        li.classList.add("active");
                        if (window.innerWidth <= 768 && sidebar) sidebar.classList.remove("open");
                        loadMarkdown(item);
                    });

                    ul.appendChild(li);
                });

                sectionDiv.appendChild(ul);
                navContent.appendChild(sectionDiv);
            }
        });
    }

    // Search Filtering
    if (searchInput) {
        searchInput.addEventListener("input", (e) => {
            const query = e.target.value.toLowerCase();
            const navItems = document.querySelectorAll(".nav-item");
            
            navItems.forEach(item => {
                const text = item.textContent.toLowerCase();
                if (text.includes(query)) {
                    item.style.display = "block";
                } else {
                    item.style.display = "none";
                }
            });
        });
    }

    async function loadMarkdown(item) {
        if (currentDocTitle) currentDocTitle.textContent = item.title;
        markdownContainer.innerHTML = `<div style="text-align:center; padding: 50px; color: #94a3b8;">
            <div style="font-size: 1.2rem; margin-bottom: 10px;">Loading ${item.title}...</div>
        </div>`;
        
        // Candidate paths to support local serving, docs subfolder, and gh-pages branch
        const candidatePaths = [
            item.path,
            item.path.replace(/^\.\.\//, ""),
            item.path.replace(/^\.\.\//, "./"),
            `./${item.path}`,
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
                // Continue to next candidate
            }
        }
        
        if (markdownText !== null) {
            try {
                // Render Markdown with KaTeX math protection
                const htmlContent = renderMarkdownWithKaTeX(markdownText);
                markdownContainer.innerHTML = htmlContent;

                // Render Mermaid Diagrams
                if (window.mermaid) {
                    try {
                        const mermaidElements = document.querySelectorAll('.mermaid');
                        if (mermaidElements.length > 0) {
                            mermaid.init(undefined, mermaidElements);
                        }
                    } catch (mErr) {
                        console.warn("Mermaid init error:", mErr);
                    }
                }

                // Render Code Highlighting
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
                                setTimeout(() => { btn.textContent = 'Copy'; }, 2000);
                            } catch (e) {
                                btn.textContent = 'Failed';
                            }
                        });
                        pre.style.position = 'relative';
                        pre.appendChild(btn);
                    }
                });
                
                // Scroll to top of content
                const contentWrapper = document.querySelector('.content-wrapper');
                if (contentWrapper) contentWrapper.scrollTop = 0;

            } catch (renderError) {
                console.error("Rendering error:", renderError);
                markdownContainer.innerHTML = `<div style="text-align:center; padding: 50px; color: #f87171;">
                    <h3>Rendering Error</h3>
                    <p>${renderError.message}</p>
                </div>`;
            }
        } else {
            console.error("Failed to load markdown from candidate paths for", item);
            markdownContainer.innerHTML = `<div style="text-align:center; padding: 50px; color: #f87171;">
                <h3>Error Loading Document</h3>
                <p>Could not fetch the documentation for <strong>${item.title}</strong>.</p>
                <p style="font-size:0.85rem; color:#94a3b8; margin-top:10px;">Tried candidate paths: <code>${candidatePaths.slice(0, 3).join(', ')}</code></p>
            </div>`;
        }
    }
});
