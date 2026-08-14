document.addEventListener("DOMContentLoaded", async () => {
    // Initialize Mermaid
    if (window.mermaid) {
        mermaid.initialize({ startOnLoad: false, theme: 'dark' });
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

    // Marked options setup for mermaid
    if (window.marked) {
        const renderer = new marked.Renderer();
        const originalCodeRenderer = renderer.code.bind(renderer);
        renderer.code = function(code, language, isEscaped) {
            if (language === 'mermaid') {
                return `<div class="mermaid">${code}</div>`;
            }
            return originalCodeRenderer(code, language, isEscaped);
        };
        marked.setOptions({ renderer });
    }

    async function loadMarkdown(item) {
        if (currentDocTitle) currentDocTitle.textContent = item.title;
        markdownContainer.innerHTML = `<div style="text-align:center; padding: 50px; color: #94a3b8;">
            <div style="font-size: 1.2rem; margin-bottom: 10px;">Loading ${item.title}...</div>
        </div>`;
        
        // Candidate paths to support both local serving, docs subfolder, and root GitHub Pages
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
                // Fix relative links in markdown if they refer to other md files
                markdownText = markdownText.replace(/\]\(([^)]+)\.md\)/g, "](javascript:void(0))");

                const htmlContent = marked.parse(markdownText);
                markdownContainer.innerHTML = htmlContent;

                // Render Math (KaTeX)
                if (window.renderMathInElement) {
                    renderMathInElement(markdownContainer, {
                        delimiters: [
                            {left: '$$', right: '$$', display: true},
                            {left: '$', right: '$', display: false},
                            {left: '\\(', right: '\\)', display: false},
                            {left: '\\[', right: '\\]', display: true}
                        ],
                        throwOnError: false
                    });
                }

                // Render Mermaid
                if (window.mermaid) {
                    const mermaidElements = document.querySelectorAll('.mermaid');
                    if (mermaidElements.length > 0) {
                        mermaid.init(undefined, mermaidElements);
                    }
                }

                // Render Code Highlighting
                if (window.Prism) {
                    Prism.highlightAllUnder(markdownContainer);
                }
                
                // Scroll to top
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
            console.error("Failed to load markdown from all candidate paths for", item);
            markdownContainer.innerHTML = `<div style="text-align:center; padding: 50px; color: #f87171;">
                <h3>Error Loading Document</h3>
                <p>Could not fetch the documentation for <strong>${item.title}</strong>.</p>
                <p style="font-size:0.85rem; color:#94a3b8; margin-top:10px;">Tried candidate paths: <code>${candidatePaths.slice(0, 3).join(', ')}</code></p>
            </div>`;
        }
    }
});
