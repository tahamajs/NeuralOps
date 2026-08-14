document.addEventListener("DOMContentLoaded", async () => {
    // Initialize Mermaid
    mermaid.initialize({ startOnLoad: false, theme: 'default' });

    const navContent = document.getElementById("nav-content");
    const searchInput = document.getElementById("search-input");
    const markdownContainer = document.getElementById("markdown-container");
    const currentDocTitle = document.getElementById("current-doc-title");
    const mobileMenuBtn = document.getElementById("mobile-menu-btn");
    const sidebar = document.getElementById("sidebar");

    let catalogData = {};
    let allItems = [];

    // Mobile Menu Toggle
    mobileMenuBtn.addEventListener("click", () => {
        sidebar.classList.toggle("open");
    });

    // Fetch catalog
    try {
        const response = await fetch("catalog.json");
        catalogData = await response.json();
        buildSidebar(catalogData);
    } catch (error) {
        console.error("Failed to load catalog:", error);
        markdownContainer.innerHTML = `<div style="color:red; text-align:center;">Failed to load module catalog.</div>`;
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
                titleDiv.textContent = sec.title;
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
                        if (window.innerWidth <= 768) sidebar.classList.remove("open");
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

    // Marked options setup for mermaid
    const renderer = new marked.Renderer();
    const originalCodeRenderer = renderer.code.bind(renderer);
    renderer.code = function(code, language, isEscaped) {
        if (language === 'mermaid') {
            return `<div class="mermaid">${code}</div>`;
        }
        return originalCodeRenderer(code, language, isEscaped);
    };
    marked.setOptions({ renderer });

    async function loadMarkdown(item) {
        currentDocTitle.textContent = item.title;
        markdownContainer.innerHTML = `<div style="text-align:center; padding: 50px; color: #94a3b8;">Loading document...</div>`;
        
        try {
            const response = await fetch(item.path);
            if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);
            
            let markdownText = await response.text();
            
            // Fix relative links in markdown if they refer to other md files
            markdownText = markdownText.replace(/\]\(([^)]+)\.md\)/g, "](javascript:void(0))");

            const htmlContent = marked.parse(markdownText);
            markdownContainer.innerHTML = htmlContent;

            // Render Math
            renderMathInElement(markdownContainer, {
                delimiters: [
                    {left: '$$', right: '$$', display: true},
                    {left: '$', right: '$', display: false},
                    {left: '\\(', right: '\\)', display: false},
                    {left: '\\[', right: '\\]', display: true}
                ],
                throwOnError : false
            });

            // Render Mermaid
            mermaid.init(undefined, document.querySelectorAll('.mermaid'));

            // Render Code Highlighting
            Prism.highlightAllUnder(markdownContainer);
            
            // Scroll to top
            document.querySelector('.content-wrapper').scrollTop = 0;

        } catch (error) {
            console.error("Failed to load markdown:", error);
            markdownContainer.innerHTML = `<div style="text-align:center; padding: 50px; color: #f87171;">
                <h3>Error Loading Document</h3>
                <p>Could not fetch the documentation for ${item.title}.</p>
            </div>`;
        }
    }
});
