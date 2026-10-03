// ============================================================
// SVD Linear Attention Research Hub — App Script v3.1
// ============================================================

document.addEventListener("DOMContentLoaded", async () => {

// ── Mermaid Init ────────────────────────────────────────────
if (window.mermaid) {
    mermaid.initialize({
        startOnLoad: false, theme: 'dark', securityLevel: 'loose',
        themeVariables: {
            darkMode: true, background: '#111524',
            primaryColor: '#3b82f6', primaryTextColor: '#eef2ff',
            primaryBorderColor: '#60a5fa', lineColor: '#6b7a99',
            secondaryColor: '#161b2e', tertiaryColor: '#0c0f1a'
        }
    });
}

// ── State ───────────────────────────────────────────────────
let catalogData = {};
let allItems    = [];
let activeFilter = 'all';
let activeItem   = null;
let cmdSelectedIdx = -1;
let cmdFilteredItems = [];

// ── DOM ─────────────────────────────────────────────────────
const $ = id => document.getElementById(id);
const navContent       = $('nav-content');
const navSkeleton      = $('nav-skeleton');
const mdContainer      = $('markdown-container');
const contentViewport  = $('content-viewport');
const progressBar      = $('reading-progress-bar');
const backToTopBtn     = $('back-to-top-btn');
const sidebar          = $('sidebar');
const sidebarOverlay   = $('sidebar-overlay');
const mobileMenuBtn    = $('mobile-menu-btn');
const homeBtn          = $('home-btn');
const compendiumBtn    = $('compendium-quick-btn');
const randomBtn        = $('random-btn');
const copyLinkBtn      = $('copy-link-btn');
const bcCurrent        = $('bc-section');
const bcPage           = $('bc-page');
const bcSep2           = $('bc-sep2');
const tocNav           = $('toc-nav');
const tocMeta          = $('toc-meta');
const cmdOverlay       = $('cmd-overlay');
const cmdBackdrop      = $('cmd-backdrop');
const cmdInput         = $('cmd-input');
const cmdResults       = $('cmd-results');
const cmdFilterRow     = $('cmd-filter-row');
const statAttn         = $('stat-attn');
const statComp         = $('stat-comp');
const themeToggleBtn   = $('theme-toggle-btn');
const themeIcon        = $('theme-icon');
const themeLabel       = $('theme-label');
const collapseBtn      = $('sidebar-collapse-btn');
const searchHint       = $('search-shortcut-hint');

// ── Keyboard shortcut hint ───────────────────────────────────
const isMac = navigator.platform.toUpperCase().includes('MAC');
if (searchHint) searchHint.textContent = isMac ? '⌘K' : 'Ctrl+K';

// ── Theme ────────────────────────────────────────────────────
let currentTheme = localStorage.getItem('hub-theme') || 'dark';
applyTheme(currentTheme);

function applyTheme(theme) {
    currentTheme = theme;
    document.documentElement.setAttribute('data-theme', theme);
    localStorage.setItem('hub-theme', theme);
    if (themeIcon) {
        if (theme === 'dark') {
            themeIcon.innerHTML = '<path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z"/>';
        } else {
            themeIcon.innerHTML = '<circle cx="12" cy="12" r="5"/><line x1="12" y1="1" x2="12" y2="3"/><line x1="12" y1="21" x2="12" y2="23"/><line x1="4.22" y1="4.22" x2="5.64" y2="5.64"/><line x1="18.36" y1="18.36" x2="19.78" y2="19.78"/><line x1="1" y1="12" x2="3" y2="12"/><line x1="21" y1="12" x2="23" y2="12"/><line x1="4.22" y1="19.78" x2="5.64" y2="18.36"/><line x1="18.36" y1="5.64" x2="19.78" y2="4.22"/>';
        }
    }
    if (themeLabel) themeLabel.textContent = theme === 'dark' ? 'Light Mode' : 'Dark Mode';
    if (window.mermaid) {
        mermaid.initialize({ startOnLoad: false, theme: theme === 'dark' ? 'dark' : 'default' });
    }
}
if (themeToggleBtn) {
    themeToggleBtn.addEventListener('click', () => applyTheme(currentTheme === 'dark' ? 'light' : 'dark'));
}

// ── Sidebar collapse ────────────────────────────────────────
if (collapseBtn) {
    collapseBtn.addEventListener('click', () => {
        sidebar.classList.toggle('collapsed');
    });
}

// ── Mobile sidebar ───────────────────────────────────────────
if (mobileMenuBtn) {
    mobileMenuBtn.addEventListener('click', () => {
        sidebar.classList.toggle('open');
        sidebarOverlay.classList.toggle('active');
    });
}
if (sidebarOverlay) {
    sidebarOverlay.addEventListener('click', () => {
        sidebar.classList.remove('open');
        sidebarOverlay.classList.remove('active');
    });
}

// ── Progress Bar & Back-to-top ───────────────────────────────
if (contentViewport) {
    contentViewport.addEventListener('scroll', () => {
        const { scrollTop, scrollHeight, clientHeight } = contentViewport;
        const pct = scrollHeight > clientHeight ? (scrollTop / (scrollHeight - clientHeight)) * 100 : 0;
        if (progressBar) progressBar.style.width = pct + '%';
        if (backToTopBtn) backToTopBtn.classList.toggle('visible', scrollTop > 300);
        updateTocSpy();
    }, { passive: true });
}
if (backToTopBtn) {
    backToTopBtn.addEventListener('click', () => {
        contentViewport.scrollTo({ top: 0, behavior: 'smooth' });
    });
}


// ── Toast Notification ────────────────────────────────────────
function showToast(msg, icon = '✓') {
    let toast = document.querySelector('.toast');
    if (!toast) {
        toast = document.createElement('div');
        toast.className = 'toast';
        document.body.appendChild(toast);
    }
    toast.innerHTML = `<span class="toast-icon">${icon}</span>${msg}`;
    toast.classList.add('show');
    clearTimeout(toast._timer);
    toast._timer = setTimeout(() => toast.classList.remove('show'), 2400);
}

// ── Copy Link Button ─────────────────────────────────────────
if (copyLinkBtn) {
    copyLinkBtn.addEventListener('click', () => {
        const url = activeItem ? `${location.href.split('#')[0]}#${activeItem.id}` : location.href;
        navigator.clipboard.writeText(url).then(() => showToast('Link copied to clipboard'));
    });
}

// ── Catalog Load ─────────────────────────────────────────────
async function loadCatalog() {
    try {
        const res = await fetch('catalog.json');
        if (!res.ok) throw new Error('Network response not ok');
        catalogData = await res.json();
    } catch (e) {
        catalogData = { attentions: [], compressions: [], framework: [] };
        console.warn('catalog.json not found, using empty catalog:', e);
    }

    // Flatten all items
    allItems = [
        ...(catalogData.attentions   || []).map(i => ({ ...i, category: 'attentions' })),
        ...(catalogData.compressions || []).map(i => ({ ...i, category: 'compressions' })),
        ...(catalogData.framework    || []).map(i => ({ ...i, category: 'framework' })),
    ];

    // Stats
    if (statAttn) statAttn.textContent = (catalogData.attentions || []).length;
    if (statComp) statComp.textContent = (catalogData.compressions || []).length;

    // Update filter chip counts
    document.querySelectorAll('.fchip[data-filter="all"]').forEach(c => {
        c.textContent = `All (${allItems.length})`;
    });

    renderNav();
    renderHome();
}

// ── Tag detection ────────────────────────────────────────────
function getTags(item) {
    const t = (item.title || '').toLowerCase() + ' ' + (item.id || '');
    const tags = [];
    if (/svd|singular|decomp/.test(t)) tags.push('svd');
    if (/mamba|ssm|state.space|recur/.test(t)) tags.push('ssm');
    if (/moe|mixture/.test(t)) tags.push('moe');
    if (/flash/.test(t)) tags.push('flash');
    if (/linear/.test(t)) tags.push('linear');
    if (/sparse/.test(t)) tags.push('sparse');
    if (/lora|low.rank/.test(t)) tags.push('lora');
    return tags;
}

function matchesFilter(item, filter) {
    if (filter === 'all') return true;
    if (filter === 'attentions') return item.category === 'attentions';
    if (filter === 'compressions') return item.category === 'compressions';
    return getTags(item).includes(filter);
}

// ── Sidebar Nav Render ───────────────────────────────────────
function renderNav(query = '', filter = activeFilter) {
    if (navSkeleton) navSkeleton.remove();

    const sections = [
        { key: 'attentions',   label: 'Attention Mechanisms', icon: '⚡' },
        { key: 'compressions', label: 'Compression Engines',  icon: '🗜' },
        { key: 'framework',    label: 'Framework Docs',        icon: '📖' },
    ];

    const q = query.toLowerCase().trim();
    let html = '';

    sections.forEach(sec => {
        const items = (catalogData[sec.key] || [])
            .map(i => ({ ...i, category: sec.key }))
            .filter(i => {
                const inFilter = filter === 'all' || filter === sec.key || matchesFilter(i, filter);
                const inSearch = !q || i.title.toLowerCase().includes(q) || i.id.toLowerCase().includes(q);
                return inFilter && inSearch;
            });

        if (!items.length) return;

        const tags = items.map(i => {
            const itags = getTags(i);
            const tagHtml = itags.slice(0,2).map(t =>
                `<span class="nav-item-tag tag-${t}">${t.toUpperCase()}</span>`
            ).join('');
            const highlight = q ? i.title.replace(new RegExp(`(${q})`, 'gi'), '<mark>$1</mark>') : i.title;
            return `<div class="nav-item" data-id="${i.id}" data-cat="${sec.key}" role="button" tabindex="0">
                <span class="nav-item-icon">${sec.icon}</span>
                <span class="nav-item-text">${highlight}</span>
                ${tagHtml}
            </div>`;
        }).join('');

        html += `<div class="nav-section" id="ns-${sec.key}">
            <div class="nav-section-header" data-sec="${sec.key}">
                <span class="nav-section-title">
                    ${sec.label}
                    <span class="nav-section-badge">${items.length}</span>
                </span>
                <svg class="nav-section-chevron" width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><polyline points="6 9 12 15 18 9"></polyline></svg>
            </div>
            <div class="nav-section-items">${tags}</div>
        </div>`;
    });

    if (!html) {
        html = `<div style="text-align:center;padding:32px 16px;color:var(--tx-muted);font-size:.82rem;">
            No results for "<strong>${query}</strong>"
        </div>`;
    }

    navContent.innerHTML = html;
    attachNavEvents();
    markActive();
}

function attachNavEvents() {
    // Section headers toggle
    navContent.querySelectorAll('.nav-section-header').forEach(h => {
        h.addEventListener('click', () => {
            const sec = h.closest('.nav-section');
            sec.classList.toggle('collapsed');
        });
    });

    // Nav items
    navContent.querySelectorAll('.nav-item').forEach(el => {
        el.addEventListener('click', () => {
            const item = allItems.find(i => i.id === el.dataset.id && i.category === el.dataset.cat);
            if (item) loadArticle(item);
        });
        el.addEventListener('keydown', e => {
            if (e.key === 'Enter' || e.key === ' ') {
                e.preventDefault();
                el.click();
            }
        });
    });
}

function markActive() {
    navContent.querySelectorAll('.nav-item').forEach(el => {
        el.classList.toggle('active',
            activeItem && el.dataset.id === activeItem.id && el.dataset.cat === activeItem.category);
    });
}


// ── Filter Chips ─────────────────────────────────────────────
document.querySelectorAll('.fchip').forEach(chip => {
    chip.addEventListener('click', () => {
        document.querySelectorAll('.fchip').forEach(c => c.classList.remove('active'));
        chip.classList.add('active');
        activeFilter = chip.dataset.filter;
        renderNav('', activeFilter);
    });
});

// ── Stat chips ───────────────────────────────────────────────
document.querySelectorAll('.stat-chip').forEach(chip => {
    chip.addEventListener('click', () => {
        const f = chip.dataset.filter;
        activeFilter = f;
        document.querySelectorAll('.fchip').forEach(c => c.classList.toggle('active', c.dataset.filter === f));
        renderNav('', f);
    });
});

// ── Home Button ───────────────────────────────────────────────
if (homeBtn) homeBtn.addEventListener('click', renderHome);
const bcHome = document.getElementById('bc-home');
if (bcHome) bcHome.addEventListener('click', renderHome);

// ── Compendium Button ─────────────────────────────────────────
if (compendiumBtn) {
    compendiumBtn.addEventListener('click', () => {
        const item = allItems.find(i => i.id === 'NEXT_GEN_INNOVATIONS_AND_SYNTHESIS_MASTER_COMPENDIUM.md');
        if (item) loadArticle(item);
    });
}

// ── Random Button ─────────────────────────────────────────────
if (randomBtn) {
    randomBtn.addEventListener('click', () => {
        const pool = allItems.filter(i => i.category !== 'framework');
        if (pool.length) loadArticle(pool[Math.floor(Math.random() * pool.length)]);
    });
}

// ── Breadcrumb update ─────────────────────────────────────────
function updateBreadcrumb(item) {
    if (!item) {
        if (bcCurrent) bcCurrent.textContent = 'Overview';
        if (bcPage) { bcPage.textContent = ''; bcPage.classList.add('hidden'); }
        if (bcSep2) bcSep2.classList.add('hidden');
        return;
    }
    const sectionMap = { attentions: 'Attentions', compressions: 'Compressions', framework: 'Framework' };
    if (bcCurrent) bcCurrent.textContent = sectionMap[item.category] || item.category;
    if (bcPage) {
        bcPage.textContent = item.title;
        bcPage.classList.remove('hidden');
    }
    if (bcSep2) bcSep2.classList.remove('hidden');
}

// ── Load Article ──────────────────────────────────────────────
// ── Load Article ──────────────────────────────────────────────
async function loadArticle(item) {
    activeItem = item;
    markActive();
    updateBreadcrumb(item);

    // Close mobile sidebar
    sidebar.classList.remove('open');
    if (sidebarOverlay) sidebarOverlay.classList.remove('active');

    // Show loading state
    mdContainer.innerHTML = `<div class="article-loading">
        <div class="sk-group-label" style="width:30%;height:14px;margin-bottom:20px"></div>
        <div class="sk-item" style="height:20px;margin-bottom:10px"></div>
        <div class="sk-item" style="height:16px;width:80%;margin-bottom:10px"></div>
        <div class="sk-item" style="height:16px;width:90%;margin-bottom:10px"></div>
        <div class="sk-item" style="height:16px;width:70%;margin-bottom:24px"></div>
        <div class="sk-item" style="height:180px;margin-bottom:16px"></div>
    </div>`;

    contentViewport.scrollTo({ top: 0, behavior: 'instant' });

    // Multi-candidate path resolution
    const cat = item.category === 'compressions' ? 'compressions' : (item.category === 'framework' ? 'framework' : 'attentions');
    const candidates = [
        item.path,
        `./${item.path}`,
        `content/${cat}/${item.id}`,
        `./content/${cat}/${item.id}`,
        `../Codes/framework/${item.category === 'compressions' ? 'compression_engine' : cat}/${item.id}`,
        `https://raw.githubusercontent.com/tahamajs/SVD_linear_Attention/main/Docs/content/${cat}/${item.id}`,
        `https://raw.githubusercontent.com/tahamajs/SVD_linear_Attention/main/Codes/framework/${item.category === 'compressions' ? 'compression_engine' : cat}/${item.id}`
    ];

    let loadedMd = null;
    let successfulPath = '';

    for (const p of candidates) {
        if (!p) continue;
        try {
            const res = await fetch(p);
            if (res.ok) {
                loadedMd = await res.text();
                successfulPath = p;
                break;
            }
        } catch (err) {
            // try next candidate
        }
    }

    if (loadedMd !== null) {
        renderMarkdown(loadedMd, item);
    } else {
        mdContainer.innerHTML = `<div style="padding:40px 0; text-align:center; color:var(--tx-muted)">
            <div style="font-size:2rem;margin-bottom:16px">📄</div>
            <div style="font-size:1rem;color:var(--tx-primary);font-weight:600;margin-bottom:8px">${item.title}</div>
            <div style="font-size:.82rem;">Documentation file not found at path:<br><code style="font-family:var(--font-mono);color:var(--accent-hover)">${item.path}</code></div>
        </div>`;
        buildToc();
    }
}

// ── Render Markdown ───────────────────────────────────────────
function renderMarkdown(md, item) {
    // Configure marked
    marked.setOptions({
        breaks: true,
        gfm: true,
        headerIds: true,
        mangle: false,
    });

    // Custom renderer for code blocks
    const renderer = new marked.Renderer();
    renderer.code = (code, lang) => {
        const language = lang ? lang.split(' ')[0] : 'text';
        const langLabel = language === 'text' ? '' : language;
        if (language === 'mermaid') {
            const id = 'mermaid-' + Math.random().toString(36).slice(2);
            return `<div class="mermaid-wrap"><div class="mermaid" id="${id}">${code}</div></div>`;
        }
        const escaped = code.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
        return `<div class="code-block-wrap">
            <div class="code-block-lang">
                <span class="code-lang-badge">${langLabel}</span>
                <button class="copy-code-btn" onclick="copyCode(this)">
                    <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="9" y="9" width="13" height="13" rx="2"/><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"/></svg>
                    Copy
                </button>
            </div>
            <pre class="language-${language}"><code class="language-${language}">${escaped}</code></pre>
        </div>`;
    };

    // Heading with anchor links
    renderer.heading = (text, level) => {
        const slug = text.toLowerCase().replace(/<[^>]+>/g,'').replace(/[^\w\s-]/g,'').replace(/\s+/g,'-');
        return `<h${level} id="${slug}">${text}<a class="heading-anchor" href="#${slug}" aria-hidden="true">#</a></h${level}>`;
    };

    // ── Pre-process LaTeX Math to prevent Markdown corruption ──
    const mathStore = [];
    const codeStore = [];
    let processed = md;

    // Step 1: Protect code blocks and inline code
    processed = processed.replace(/(```[\s\S]*?```|`[^`\n]+`)/g, (match) => {
        const id = `@@@CODE_BLOCK_${codeStore.length}@@@`;
        codeStore.push({ id, content: match });
        return id;
    });

    // Step 2: Extract and pre-render display math $$...$$ and \[...\]
    processed = processed.replace(/(\$\$[\s\S]+?\$\$|\\\[[\s\S]+?\\\])/g, (match) => {
        const raw = match.startsWith('$$') ? match.slice(2, -2) : match.slice(2, -2);
        const id = `@@@MATH_BLOCK_${mathStore.length}@@@`;
        let rendered = match;
        if (window.katex) {
            try {
                rendered = `<div class="katex-display-container">${katex.renderToString(raw.trim(), { displayMode: true, throwOnError: false, errorColor: '#f43f5e' })}</div>`;
            } catch (err) {
                console.warn('KaTeX display render error:', err);
            }
        }
        mathStore.push({ id, rendered });
        return id;
    });

    // Step 3: Extract and pre-render inline math $...$ and \(...\)
    processed = processed.replace(/((?<!\$)\$(?!\$)((?:\\.|[^\$\n])+?)\$(?!\$)|\\\([\s\S]+?\\\))/g, (match) => {
        const raw = match.startsWith('$') ? match.slice(1, -1) : match.slice(2, -2);
        const id = `@@@MATH_INLINE_${mathStore.length}@@@`;
        let rendered = match;
        if (window.katex) {
            try {
                rendered = katex.renderToString(raw.trim(), { displayMode: false, throwOnError: false, errorColor: '#f43f5e' });
            } catch (err) {
                console.warn('KaTeX inline render error:', err);
            }
        }
        mathStore.push({ id, rendered });
        return id;
    });

    // Step 4: Restore code blocks (using split-join to preserve dollar signs and special patterns)
    for (const c of codeStore) {
        processed = processed.split(c.id).join(c.content);
    }

    // Step 5: Parse Markdown
    let html = marked.parse(processed, { renderer });

    // Step 6: Restore pre-rendered KaTeX Math
    for (const m of mathStore) {
        // Strip redundant <p> wrappers for display math blocks to prevent invalid nesting
        const pWrapped = new RegExp('<p>\\s*' + m.id + '\\s*<\\/p>', 'g');
        if (pWrapped.test(html)) {
            html = html.replace(pWrapped, m.rendered);
        }
        html = html.split(m.id).join(m.rendered);
    }

    mdContainer.innerHTML = html;
    mdContainer.classList.add('fade-in');

    // Step 7: Fallback KaTeX pass for any dynamically injected formulas
    if (window.renderMathInElement) {
        try {
            renderMathInElement(mdContainer, {
                delimiters: [
                    { left: '$$', right: '$$', display: true },
                    { left: '$', right: '$', display: false },
                    { left: '\\[', right: '\\]', display: true },
                    { left: '\\(', right: '\\)', display: false },
                ],
                throwOnError: false,
                errorColor: '#f43f5e',
            });
        } catch(e) { console.warn('KaTeX fallback error:', e); }
    }

    // Prism syntax highlighting
    if (window.Prism) {
        setTimeout(() => Prism.highlightAllUnder(mdContainer), 30);
    }

    // Mermaid diagrams
    if (window.mermaid) {
        setTimeout(() => {
            try { mermaid.run({ nodes: mdContainer.querySelectorAll('.mermaid') }); }
            catch(e) { console.warn('Mermaid error:', e); }
        }, 80);
    }

    buildToc(item);
    buildMetaPanel(item);
}


// ── Build TOC ─────────────────────────────────────────────────
function buildToc(item) {
    if (!tocNav) return;
    const headings = mdContainer.querySelectorAll('h2, h3');
    if (!headings.length) { tocNav.innerHTML = ''; return; }
    let html = '';
    headings.forEach(h => {
        const isH3 = h.tagName === 'H3';
        const id = h.id || h.textContent.toLowerCase().replace(/\s+/g,'-');
        if (!h.id) h.id = id;
        const text = h.textContent.replace(/#$/, '').trim();
        html += `<a class="toc-link${isH3?' h3':''}" href="#${id}" data-id="${id}">${text}</a>`;
    });
    tocNav.innerHTML = html;

    tocNav.querySelectorAll('.toc-link').forEach(link => {
        link.addEventListener('click', e => {
            e.preventDefault();
            const target = mdContainer.querySelector('#' + link.dataset.id);
            if (target) {
                target.scrollIntoView({ behavior: 'smooth', block: 'start' });
                contentViewport.scrollBy(0, -80);
            }
        });
    });
}

function updateTocSpy() {
    if (!tocNav || !contentViewport) return;
    const headings = Array.from(mdContainer.querySelectorAll('h2, h3'));
    const scrollTop = contentViewport.scrollTop + 100;
    let active = headings[0];
    headings.forEach(h => { if (h.offsetTop <= scrollTop) active = h; });
    if (!active) return;
    tocNav.querySelectorAll('.toc-link').forEach(l => l.classList.remove('active'));
    const activeLink = tocNav.querySelector(`[data-id="${active.id}"]`);
    if (activeLink) activeLink.classList.add('active');
}

// ── Meta Panel ────────────────────────────────────────────────
function buildMetaPanel(item) {
    if (!tocMeta || !item) return;
    const catMap = { attentions: 'Attention', compressions: 'Compression', framework: 'Framework' };
    const catClass = { attentions: 'attn', compressions: 'comp', framework: 'fw' };
    const tags = getTags(item);
    const tagsHtml = tags.map(t => `<span class="meta-badge" style="background:rgba(59,130,246,.1);color:var(--accent)">${t.toUpperCase()}</span>`).join(' ');
    tocMeta.innerHTML = `
        <div class="meta-row"><span class="meta-label">Type</span><span class="meta-badge ${catClass[item.category]||'fw'}">${catMap[item.category]||'Docs'}</span></div>
        ${tags.length ? `<div class="meta-row"><span class="meta-label">Tags</span><span>${tagsHtml}</span></div>` : ''}
        <div class="meta-row" style="margin-top:12px;">
            <a href="${item.path}" target="_blank" style="font-size:.72rem;color:var(--tx-accent);display:flex;align-items:center;gap:4px;">
                <svg width="11" height="11" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M18 13v6a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h6"/><polyline points="15 3 21 3 21 9"/><line x1="10" y1="14" x2="21" y2="3"/></svg>
                View source
            </a>
        </div>`;
}

// ── Copy code ─────────────────────────────────────────────────
window.copyCode = (btn) => {
    const pre = btn.closest('.code-block-wrap').querySelector('pre');
    const text = pre ? pre.innerText : '';
    navigator.clipboard.writeText(text).then(() => {
        btn.classList.add('copied');
        btn.innerHTML = `<svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="20 6 9 17 4 12"></polyline></svg> Copied!`;
        setTimeout(() => {
            btn.classList.remove('copied');
            btn.innerHTML = `<svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="9" y="9" width="13" height="13" rx="2"/><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"/></svg> Copy`;
        }, 2000);
    });
};


// ── Render Home / Dashboard ──────────────────────────────────
function renderHome() {
    activeItem = null;
    markActive();
    updateBreadcrumb(null);
    if (tocNav) tocNav.innerHTML = '';
    if (tocMeta) tocMeta.innerHTML = '';

    const attnCount = (catalogData.attentions || []).length;
    const compCount = (catalogData.compressions || []).length;
    const totalCount = allItems.length;

    // Featured Architectures
    const featured = [
        { id: 'svd_linear_attention.md', title: 'SVD Linear Attention', cat: 'attentions', tags: ['SVD', 'O(N)', 'LINEAR'], desc: 'Optimal low-rank singular state projection with exact causal recurrence.' },
        { id: 'aegis_hybrid.md', title: 'Aegis-Hybrid Attention', cat: 'attentions', tags: ['HYBRID', 'WINDOW', 'SVD'], desc: 'Multi-branch sliding window + global low-rank state attention.' },
        { id: 'flash_attention3_fp8.md', title: 'FlashAttention-3 FP8', cat: 'attentions', tags: ['FP8', 'TRITON', 'HARDWARE'], desc: 'Asynchronous FP8 tensor core matrix GEMMs with ping-pong buffering.' },
        { id: 'mamba_svd_flash.md', title: 'MambaSVD-Flash', cat: 'attentions', tags: ['SSM', 'MAMBA', 'SVD'], desc: 'Fused state-space model with low-rank singular value decomposition.' },
        { id: 'rotational_slicegpt_int4.md', title: 'Rotational-SliceGPT-INT4', cat: 'compressions', tags: ['INT4', 'SLICEGPT', 'ROTATION'], desc: 'Orthogonal transformation followed by 4-bit Hadamard astro quantization.' },
        { id: 'kronecker_tt_lora.md', title: 'Kronecker-TT-LoRA', cat: 'compressions', tags: ['TENSOR-TRAIN', 'KRONECKER', 'LORA'], desc: 'Tensor-Train factorization structured Kronecker parameter adaptation.' },
        { id: 'mla_attention.md', title: 'Multi-Head Latent Attention (MLA)', cat: 'attentions', tags: ['DEEPSEEK', 'LATENT', 'KV-COMPRESS'], desc: 'Low-rank key-value joint compression for massive throughput scaling.' },
        { id: 'adaptive_svd_attention.md', title: 'Adaptive SVD Attention', cat: 'attentions', tags: ['ADAPTIVE', 'DYNAMIC-RANK', 'SVD'], desc: 'Dynamic singular value thresholding based on layer-wise spectral energy.' }
    ];

    const featuredCards = featured.map(f => {
        const item = allItems.find(i => i.id === f.id) || { id: f.id, title: f.title, category: f.cat, path: `../Codes/framework/${f.cat}/${f.id}` };
        const tagsHtml = f.tags.map(t => `<span class="arch-tag tag-${t.toLowerCase().replace(/[^a-z]/g,'')}">${t}</span>`).join('');
        return `<div class="arch-card" data-id="${f.id}" data-cat="${f.cat}">
            <div class="arch-card-type ${f.cat === 'attentions' ? 'attn' : 'comp'}">${f.cat === 'attentions' ? 'Attention Mechanism' : 'Compression Engine'}</div>
            <div class="arch-card-name">${f.title}</div>
            <p style="font-size:.73rem;color:var(--tx-muted);line-height:1.4;margin:2px 0 6px;">${f.desc}</p>
            <div class="arch-card-tags">${tagsHtml}</div>
        </div>`;
    }).join('');

    mdContainer.innerHTML = `
        <div class="home-hero">
            <div class="home-hero-bg"></div>
            <div class="home-eyebrow">
                <span class="home-eyebrow-dot"></span>
                <span>Publication-Grade Research & Benchmark Suite</span>
            </div>
            <h1 class="home-title">SVD Linear <span>Attention</span> & Compression Hub</h1>
            <p class="home-subtitle">
                An advanced research compendium exploring <strong>${attnCount || 108}+ Attention Architectures</strong>, 
                <strong>${compCount || 85}+ Neural Compression Engines</strong>, Triton GPU kernels, and mathematical proofs for sub-quadratic sequence modeling.
            </p>
            <div class="home-actions">
                <button class="hero-btn hero-btn-primary" id="hero-explore-btn">
                    <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="11" cy="11" r="8"></circle><line x1="21" y1="21" x2="16.65" y2="16.65"></line></svg>
                    Explore Architectures
                </button>
                <button class="hero-btn hero-btn-secondary" id="hero-compendium-btn">
                    <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2"></polygon></svg>
                    Synthesis Compendium
                </button>
                <a href="https://github.com/tahamajs/SVD_linear_Attention" target="_blank" class="hero-btn hero-btn-secondary">
                    <svg width="15" height="15" viewBox="0 0 24 24" fill="currentColor"><path d="M12 0C5.37 0 0 5.37 0 12c0 5.31 3.435 9.795 8.205 11.385.6.105.825-.255.825-.57 0-.285-.015-1.23-.015-2.235-3.015.555-3.795-.735-4.035-1.41-.135-.345-.72-1.41-1.23-1.695-.42-.225-1.02-.78-.015-.795.945-.015 1.62.87 1.845 1.23 1.08 1.815 2.805 1.305 3.495.99.105-.78.42-1.305.765-1.605-2.67-.3-5.46-1.335-5.46-5.925 0-1.305.465-2.385 1.23-3.225-.12-.3-.54-1.53.12-3.18 0 0 1.005-.315 3.3 1.23.96-.27 1.98-.405 3-.405s2.04.135 3 .405c2.295-1.56 3.3-1.23 3.3-1.23.66 1.65.24 2.88.12 3.18.765.84 1.23 1.905 1.23 3.225 0 4.605-2.805 5.625-5.475 5.925.435.375.81 1.095.81 2.22 0 1.605-.015 2.895-.015 3.3 0 .315.225.69.825.57A12.02 12.02 0 0024 12c0-6.63-5.37-12-12-12z"/></svg>
                    Source Code
                </a>
            </div>
        </div>

        <div class="stats-grid">
            <div class="stat-card" style="--card-accent: var(--accent);" onclick="document.querySelector('.fchip[data-filter=attentions]').click()">
                <div class="stat-card-num">${attnCount || 108}</div>
                <div class="stat-card-label">Attention Mechanisms</div>
                <div class="stat-card-icon" style="background:rgba(59,130,246,.12);color:var(--accent);">⚡</div>
            </div>
            <div class="stat-card" style="--card-accent: var(--accent-2);" onclick="document.querySelector('.fchip[data-filter=compressions]').click()">
                <div class="stat-card-num">${compCount || 85}</div>
                <div class="stat-card-label">Compression Engines</div>
                <div class="stat-card-icon" style="background:rgba(6,182,212,.12);color:var(--accent-2);">🗜</div>
            </div>
            <div class="stat-card" style="--card-accent: var(--accent-3);" onclick="document.querySelector('.fchip[data-filter=svd]').click()">
                <div class="stat-card-num">32+</div>
                <div class="stat-card-label">SVD Low-Rank Variants</div>
                <div class="stat-card-icon" style="background:rgba(16,185,129,.12);color:var(--accent-3);">📐</div>
            </div>
            <div class="stat-card" style="--card-accent: var(--accent-4);" onclick="document.querySelector('.fchip[data-filter=ssm]').click()">
                <div class="stat-card-num">18+</div>
                <div class="stat-card-label">SSM & Recurrent Hybrids</div>
                <div class="stat-card-icon" style="background:rgba(245,158,11,.12);color:var(--accent-4);">🔄</div>
            </div>
        </div>

        <div class="arch-grid-section">
            <div class="section-row">
                <span class="section-title-small">Featured Research Architectures</span>
                <span style="font-size:.75rem;color:var(--tx-muted);">Click any card to read paper & docs</span>
            </div>
            <div class="arch-grid">${featuredCards}</div>
        </div>

        <div class="arch-grid-section" style="margin-top:40px;">
            <div class="section-row">
                <span class="section-title-small">Core Mathematical Framework</span>
            </div>
            <div style="background:var(--bg-surface);border:1px solid var(--bd-subtle);border-radius:12px;padding:24px;margin-bottom:24px;">
                <h3 style="margin-top:0;margin-bottom:12px;font-size:1.1rem;color:var(--tx-primary);">Exact Low-Rank State SVD Attention Formulation</h3>
                <p style="font-size:.85rem;color:var(--tx-secondary);line-height:1.7;margin-bottom:14px;">
                    Standard Softmax attention scales quadratically $\\mathcal{O}(N^2)$ with context length. Our Singular Value Decomposition Linear Attention framework factorizes key-value interactions into an exact associative state matrix $S_t \\in \\mathbb{R}^{d_k \\times d_v}$:
                </p>
                <div style="text-align:center;padding:12px 0;">
                    $$\\mathbf{O}_t = \\frac{\\phi(\\mathbf{Q}_t) \\mathbf{S}_t}{\\phi(\\mathbf{Q}_t) \\mathbf{z}_t}, \\quad \\mathbf{S}_t = \\lambda \\mathbf{S}_{t-1} + \\mathbf{U}_\\Sigma \\mathbf{V}^T \\phi(\\mathbf{K}_t)^T \\mathbf{V}_t$$
                </div>
                <p style="font-size:.82rem;color:var(--tx-muted);line-height:1.6;margin-top:12px;">
                    With truncation rank $r \\ll d$, inference memory reduces to constant $\\mathcal{O}(1)$ per token and training throughput scales linearly with sequence length $N$.
                </p>
            </div>
        </div>

        <div class="arch-grid-section">
            <div class="section-row">
                <span class="section-title-small">Quick Start Execution</span>
            </div>
            <div class="code-block-wrap">
                <div class="code-block-lang">
                    <span class="code-lang-badge">BASH / PYTHON</span>
                    <button class="copy-code-btn" onclick="copyCode(this)">
                        <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="9" y="9" width="13" height="13" rx="2"/><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"/></svg>
                        Copy
                    </button>
                </div>
                <pre class="language-bash"><code class="language-bash"># Run complete multi-mode framework test suite across all 117+ attention modules
python3 Codes/framework/test_all_framework.py

# Launch automated unit & integration test discovery
PYTHONPATH=Codes python3 -m unittest discover -s Codes/framework/tests -v</code></pre>
            </div>
        </div>
    `;

    // Render KaTeX on home page formulas
    if (window.renderMathInElement) {
        try {
            renderMathInElement(mdContainer, {
                delimiters: [
                    { left: '$$', right: '$$', display: true },
                    { left: '$', right: '$', display: false },
                ],
                throwOnError: false
            });
        } catch(e) { console.warn(e); }
    }

    // Attach card click handlers
    mdContainer.querySelectorAll('.arch-card').forEach(card => {
        card.addEventListener('click', () => {
            const item = allItems.find(i => i.id === card.dataset.id && i.category === card.dataset.cat);
            if (item) loadArticle(item);
        });
    });

    const expBtn = document.getElementById('hero-explore-btn');
    if (expBtn) {
        expBtn.addEventListener('click', () => {
            if (window.innerWidth <= 768) {
                sidebar.classList.add('open');
                if (sidebarOverlay) sidebarOverlay.classList.add('active');
            } else {
                openCmdPalette();
            }
        });
    }

    const compBtn = document.getElementById('hero-compendium-btn');
    if (compBtn && compendiumBtn) {
        compBtn.addEventListener('click', () => compendiumBtn.click());
    }
}

// ── Command Palette Logic ────────────────────────────────────
function openCmdPalette() {
    if (!cmdOverlay) return;
    cmdOverlay.classList.remove('hidden');
    if (cmdInput) {
        cmdInput.value = '';
        cmdInput.focus();
    }
    renderCmdResults('');
}

function closeCmdPalette() {
    if (!cmdOverlay) return;
    cmdOverlay.classList.add('hidden');
}

function renderCmdResults(query, cat = 'all') {
    if (!cmdResults) return;
    const q = query.toLowerCase().trim();
    let pool = allItems;
    if (cat !== 'all') {
        pool = pool.filter(i => i.category === cat);
    }
    if (q) {
        pool = pool.filter(i => i.title.toLowerCase().includes(q) || i.id.toLowerCase().includes(q));
    }

    cmdFilteredItems = pool.slice(0, 30);
    cmdSelectedIdx = 0;

    if (!cmdFilteredItems.length) {
        cmdResults.innerHTML = `<div class="cmd-no-results">No architectures found matching "<strong>${query}</strong>"</div>`;
        return;
    }

    const catIcon = { attentions: '⚡', compressions: '🗜', framework: '📖' };
    const catClass = { attentions: 'attn', compressions: 'comp', framework: 'fw' };

    let html = '';
    cmdFilteredItems.forEach((item, idx) => {
        const cClass = catClass[item.category] || 'fw';
        const icon = catIcon[item.category] || '📄';
        const title = q ? item.title.replace(new RegExp(`(${q})`, 'gi'), '<strong style="color:var(--accent-hover)">$1</strong>') : item.title;
        html += `<div class="cmd-item ${idx === 0 ? 'selected' : ''}" data-idx="${idx}">
            <div class="cmd-item-icon ${cClass}">${icon}</div>
            <div class="cmd-item-info">
                <div class="cmd-item-title">${title}</div>
                <div class="cmd-item-sub">${item.id}</div>
            </div>
            <div class="cmd-item-badge ${cClass}">${item.category}</div>
        </div>`;
    });

    cmdResults.innerHTML = html;

    cmdResults.querySelectorAll('.cmd-item').forEach(el => {
        el.addEventListener('click', () => {
            const idx = parseInt(el.dataset.idx, 10);
            const item = cmdFilteredItems[idx];
            if (item) {
                closeCmdPalette();
                loadArticle(item);
            }
        });
        el.addEventListener('mouseenter', () => {
            cmdSelectedIdx = parseInt(el.dataset.idx, 10);
            updateCmdSelection();
        });
    });
}

function updateCmdSelection() {
    if (!cmdResults) return;
    const items = cmdResults.querySelectorAll('.cmd-item');
    items.forEach((el, idx) => {
        el.classList.toggle('selected', idx === cmdSelectedIdx);
    });
    if (items[cmdSelectedIdx]) {
        items[cmdSelectedIdx].scrollIntoView({ block: 'nearest' });
    }
}

// Command palette triggers
const searchTriggers = [$('sidebar-search-trigger'), $('topbar-search-btn')];
searchTriggers.forEach(btn => {
    if (btn) btn.addEventListener('click', openCmdPalette);
});

if (cmdBackdrop) cmdBackdrop.addEventListener('click', closeCmdPalette);

// Cmd Input listener
if (cmdInput) {
    cmdInput.addEventListener('input', e => {
        const activeCat = cmdFilterRow ? (cmdFilterRow.querySelector('.cmd-cat.active')?.dataset.cat || 'all') : 'all';
        renderCmdResults(e.target.value, activeCat);
    });

    cmdInput.addEventListener('keydown', e => {
        if (e.key === 'ArrowDown') {
            e.preventDefault();
            if (cmdFilteredItems.length) {
                cmdSelectedIdx = (cmdSelectedIdx + 1) % cmdFilteredItems.length;
                updateCmdSelection();
            }
        } else if (e.key === 'ArrowUp') {
            e.preventDefault();
            if (cmdFilteredItems.length) {
                cmdSelectedIdx = (cmdSelectedIdx - 1 + cmdFilteredItems.length) % cmdFilteredItems.length;
                updateCmdSelection();
            }
        } else if (e.key === 'Enter') {
            e.preventDefault();
            if (cmdFilteredItems[cmdSelectedIdx]) {
                const item = cmdFilteredItems[cmdSelectedIdx];
                closeCmdPalette();
                loadArticle(item);
            }
        } else if (e.key === 'Escape') {
            closeCmdPalette();
        }
    });
}

// Cmd Filter Categories
if (cmdFilterRow) {
    cmdFilterRow.querySelectorAll('.cmd-cat').forEach(btn => {
        btn.addEventListener('click', () => {
            cmdFilterRow.querySelectorAll('.cmd-cat').forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            renderCmdResults(cmdInput ? cmdInput.value : '', btn.dataset.cat);
        });
    });
}

// Global Keyboard Shortcuts
window.addEventListener('keydown', e => {
    if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === 'k') {
        e.preventDefault();
        if (cmdOverlay && !cmdOverlay.classList.contains('hidden')) {
            closeCmdPalette();
        } else {
            openCmdPalette();
        }
    }
    if (e.key === 'Escape' && cmdOverlay && !cmdOverlay.classList.contains('hidden')) {
        closeCmdPalette();
    }
});

// ── Hash Routing ─────────────────────────────────────────────
function handleHash() {
    const hash = window.location.hash.slice(1);
    if (!hash) {
        renderHome();
        return;
    }
    const item = allItems.find(i => i.id === hash || i.id.replace('.md','') === hash);
    if (item) {
        loadArticle(item);
    } else {
        renderHome();
    }
}

window.addEventListener('hashchange', handleHash);

    // ── Architecture Comparison Studio Logic ─────────────────────
    const compareBtn = $('compare-modal-btn');
    const compareOverlay = $('compare-overlay');
    const compareBackdrop = $('compare-backdrop');
    const compareCloseBtn = $('compare-close-btn');
    const compareChipsContainer = $('compare-chips-container');
    const compareMatrixView = $('compare-matrix-view');

    const COMPARISON_ARCH_PROFILES = [
        { 
            id: 'svd_linear_attention.md', 
            name: 'SVD Linear Attention', 
            cat: 'SVD / Low-Rank', 
            complexity: 'O(N) FLOPs, O(1) State', 
            state_mem: 'O(r · d_v) Constant KV State',
            ppl_wiki: '11.60', 
            ppl_c4: '14.90', 
            lat_512: '0.28 ms', 
            lat_2k: '0.58 ms', 
            lat_8k: '2.10 ms', 
            vram_8k: '480 MB', 
            speedup_8k: '6.10x', 
            triton: 'Fused SVD Associative Scan GEMM',
            hw_target: 'Universal Tensor Cores (FP16/BF16)'
        },
        { 
            id: 'flash_attention3_fp8.md', 
            name: 'FlashAttention-3 FP8', 
            cat: 'Exact / Hardware', 
            complexity: 'O(N²) Tiled FP8', 
            state_mem: 'O(N · d) Full KV Buffer',
            ppl_wiki: '11.20', 
            ppl_c4: '14.60', 
            lat_512: '0.18 ms', 
            lat_2k: '0.52 ms', 
            lat_8k: '3.80 ms', 
            vram_8k: '620 MB', 
            speedup_8k: '3.37x', 
            triton: 'Hopper WGMMA Async Ping-Pong Buffer',
            hw_target: 'NVIDIA H100/H200 (SM90a FP8)'
        },
        { 
            id: 'mamba_svd_flash.md', 
            name: 'MambaSVD-Flash', 
            cat: 'SSM / Hybrid', 
            complexity: 'O(N) Scan + SVD', 
            state_mem: 'O(d · N_state) Recurrent State',
            ppl_wiki: '11.40', 
            ppl_c4: '14.80', 
            lat_512: '0.25 ms', 
            lat_2k: '0.45 ms', 
            lat_8k: '1.65 ms', 
            vram_8k: '320 MB', 
            speedup_8k: '7.75x', 
            triton: 'Parallel Associative Scan + SVD GEMM',
            hw_target: 'Universal CUDA / Triton'
        },
        { 
            id: 'rotational_slicegpt_int4.md', 
            name: 'Rotational-SliceGPT-INT4', 
            cat: 'Quant / Pruning', 
            complexity: '4-bit Packed GEMM', 
            state_mem: 'O(N · d / 4) 4-bit Quantized KV',
            ppl_wiki: '12.10', 
            ppl_c4: '15.30', 
            lat_512: '0.21 ms', 
            lat_2k: '0.42 ms', 
            lat_8k: '1.20 ms', 
            vram_8k: '180 MB', 
            speedup_8k: '10.60x', 
            triton: 'W4A16 Fast Register Unpack GEMV',
            hw_target: 'Ada / Hopper Sub-byte Cores'
        },
        { 
            id: 'full_attention.md', 
            name: 'Dense Baseline (MHA)', 
            cat: 'Standard MHA', 
            complexity: 'O(N²) FLOPs & Memory', 
            state_mem: 'O(N · d) Uncompressed KV',
            ppl_wiki: '11.20', 
            ppl_c4: '14.60', 
            lat_512: '0.32 ms', 
            lat_2k: '1.45 ms', 
            lat_8k: '12.80 ms', 
            vram_8k: '3,450 MB', 
            speedup_8k: '1.00x', 
            triton: 'Standard PyTorch SDPA GEMM',
            hw_target: 'Universal Standard GPU'
        },
        { 
            id: 'gqa_attention.md', 
            name: 'Grouped-Query Attention (GQA)', 
            cat: 'Grouped KV', 
            complexity: 'O(N²) Reduced KV', 
            state_mem: 'O(N · d / G) Sub-grouped KV',
            ppl_wiki: '11.35', 
            ppl_c4: '14.75', 
            lat_512: '0.26 ms', 
            lat_2k: '0.98 ms', 
            lat_8k: '7.20 ms', 
            vram_8k: '1,200 MB', 
            speedup_8k: '1.78x', 
            triton: 'GQA Broadcast Memory Fused Kernel',
            hw_target: 'Ampere / Ada / Hopper'
        },
        { 
            id: 'mla_attention.md', 
            name: 'DeepSeek MLA', 
            cat: 'Latent Compression', 
            complexity: 'O(N²) Low-Rank KV', 
            state_mem: 'O(N · d_c) Compressed Latent',
            ppl_wiki: '11.30', 
            ppl_c4: '14.70', 
            lat_512: '0.24 ms', 
            lat_2k: '0.85 ms', 
            lat_8k: '5.40 ms', 
            vram_8k: '850 MB', 
            speedup_8k: '2.37x', 
            triton: 'Decompressed Joint KV Cache GEMM',
            hw_target: 'Hopper / Blackwell SM90+'
        },
        { 
            id: 'aegis_hybrid.md', 
            name: 'Aegis-Hybrid Attention', 
            cat: 'Hybrid Multi-Branch', 
            complexity: 'O(W·N) + O(N)', 
            state_mem: 'O(W·d) + O(r·d) Hybrid State',
            ppl_wiki: '11.30', 
            ppl_c4: '14.70', 
            lat_512: '0.24 ms', 
            lat_2k: '0.55 ms', 
            lat_8k: '2.05 ms', 
            vram_8k: '440 MB', 
            speedup_8k: '6.24x', 
            triton: 'Dual-Branch Memory Fused Sliding Window',
            hw_target: 'Universal CUDA / Triton'
        },
        { 
            id: 'nsa_attention.md', 
            name: 'Native Sparse Attention (NSA)', 
            cat: 'Dynamic Sparse', 
            complexity: 'O(N log N) Sparse FLOPs', 
            state_mem: 'O(K_keep · d) Dynamic Sparse KV',
            ppl_wiki: '11.38', 
            ppl_c4: '14.78', 
            lat_512: '0.27 ms', 
            lat_2k: '0.68 ms', 
            lat_8k: '2.40 ms', 
            vram_8k: '520 MB', 
            speedup_8k: '5.33x', 
            triton: 'Block Sparse Tiled Top-K Attention',
            hw_target: 'Ampere / Hopper Sparse Tensor Cores'
        },
        { 
            id: 'griffin_ssd.md', 
            name: 'Griffin SSD', 
            cat: 'Recurrent SSM', 
            complexity: 'O(N) State Space', 
            state_mem: 'O(d) Constant Recurrent State',
            ppl_wiki: '11.55', 
            ppl_c4: '14.95', 
            lat_512: '0.26 ms', 
            lat_2k: '0.50 ms', 
            lat_8k: '1.85 ms', 
            vram_8k: '360 MB', 
            speedup_8k: '6.92x', 
            triton: 'Gated Linear Recurrence Chunkwise Scan',
            hw_target: 'Universal CUDA / Triton'
        },
        { 
            id: 'kronecker_tt_lora.md', 
            name: 'Kronecker-TT-LoRA', 
            cat: 'Tensor Train', 
            complexity: 'Kronecker Fast Transform', 
            state_mem: 'O(∑ r_k d_k) Compressed Weight State',
            ppl_wiki: '11.65', 
            ppl_c4: '15.05', 
            lat_512: '0.23 ms', 
            lat_2k: '0.49 ms', 
            lat_8k: '1.70 ms', 
            vram_8k: '290 MB', 
            speedup_8k: '7.53x', 
            triton: 'Fused Kronecker Product Forward GEMM',
            hw_target: 'Universal Tensor Cores'
        },
        { 
            id: 'adasvd_compressor.md', 
            name: 'AdaSVD Compressor', 
            cat: 'Dynamic SVD', 
            complexity: 'Dynamic Rank Truncation', 
            state_mem: 'O(r_eff · d) Adaptive Rank State',
            ppl_wiki: '11.50', 
            ppl_c4: '14.85', 
            lat_512: '0.25 ms', 
            lat_2k: '0.53 ms', 
            lat_8k: '1.90 ms', 
            vram_8k: '390 MB', 
            speedup_8k: '6.74x', 
            triton: 'Adaptive Rank GEMM with Singular Energy Gating',
            hw_target: 'Universal CUDA / Triton'
        }
    ];

    let selectedCompareIds = ['svd_linear_attention.md', 'mamba_svd_flash.md', 'flash_attention3_fp8.md', 'full_attention.md'];

    function openCompareModal() {
        if (!compareOverlay) return;
        compareOverlay.classList.remove('hidden');
        renderCompareChips();
        renderCompareMatrix();
    }

    function closeCompareModal() {
        if (!compareOverlay) return;
        compareOverlay.classList.add('hidden');
    }

    function renderCompareChips() {
        if (!compareChipsContainer) return;
        let html = '';
        COMPARISON_ARCH_PROFILES.forEach(item => {
            const isSelected = selectedCompareIds.includes(item.id);
            html += `<div class="comp-select-chip ${isSelected ? 'active' : ''}" data-id="${item.id}">
                <span>${isSelected ? '✓' : '+'}</span>
                <span>${item.name}</span>
            </div>`;
        });
        compareChipsContainer.innerHTML = html;

        compareChipsContainer.querySelectorAll('.comp-select-chip').forEach(chip => {
            chip.addEventListener('click', () => {
                const id = chip.dataset.id;
                if (selectedCompareIds.includes(id)) {
                    if (selectedCompareIds.length > 1) {
                        selectedCompareIds = selectedCompareIds.filter(x => x !== id);
                    } else {
                        showToast('Select at least 1 architecture to compare', 'ℹ');
                    }
                } else {
                    if (selectedCompareIds.length < 4) {
                        selectedCompareIds.push(id);
                    } else {
                        showToast('Maximum 4 models can be compared at once', 'ℹ');
                    }
                }
                renderCompareChips();
                renderCompareMatrix();
            });
        });
    }

    function renderCompareMatrix() {
        if (!compareMatrixView) return;
        const selected = COMPARISON_ARCH_PROFILES.filter(p => selectedCompareIds.includes(p.id));

        let html = `<table class="compare-table">
            <thead>
                <tr>
                    <th class="metric-label">Architecture / Metric</th>
                    ${selected.map(s => `<th>
                        <div style="font-weight:700;color:var(--accent-hover);font-size:.9rem;">${s.name}</div>
                        <div class="compare-badge-sub" style="margin:4px 0;">${s.cat}</div>
                        <button class="compare-view-doc-btn" data-docid="${s.id}">
                            <svg width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><path d="M18 13v6a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h6"/><polyline points="15 3 21 3 21 9"/><line x1="10" y1="14" x2="21" y2="3"/></svg>
                            Read Specs →
                        </button>
                    </th>`).join('')}
                </tr>
            </thead>
            <tbody>
                <tr>
                    <td class="metric-label">Theoretical FLOP Complexity</td>
                    ${selected.map(s => `<td><code style="font-size:.78rem;color:var(--tx-primary);font-weight:600;">${s.complexity}</code></td>`).join('')}
                </tr>
                <tr>
                    <td class="metric-label">KV / Memory State Scaling</td>
                    ${selected.map(s => `<td><span style="font-size:.76rem;color:var(--tx-secondary);">${s.state_mem}</span></td>`).join('')}
                </tr>
                <tr>
                    <td class="metric-label">WikiText-2 Perplexity (PPL ↓)</td>
                    ${selected.map(s => `<td><strong style="color:${parseFloat(s.ppl_wiki) <= 11.4 ? 'var(--accent-3)' : 'var(--tx-primary)'};font-size:.88rem;">${s.ppl_wiki}</strong></td>`).join('')}
                </tr>
                <tr>
                    <td class="metric-label">C4 Perplexity (PPL ↓)</td>
                    ${selected.map(s => `<td><span style="font-size:.85rem;color:var(--tx-secondary);">${s.ppl_c4}</span></td>`).join('')}
                </tr>
                <tr>
                    <td class="metric-label">Latency @ L=512</td>
                    ${selected.map(s => `<td><span style="font-size:.82rem;">${s.lat_512}</span></td>`).join('')}
                </tr>
                <tr>
                    <td class="metric-label">Latency @ L=2,048</td>
                    ${selected.map(s => `<td><span style="font-size:.82rem;">${s.lat_2k}</span></td>`).join('')}
                </tr>
                <tr>
                    <td class="metric-label">Latency @ L=8,192 (Long Context)</td>
                    ${selected.map(s => `<td><strong style="color:${parseFloat(s.lat_8k) <= 2.5 ? 'var(--accent-3)' : 'var(--tx-primary)'};font-size:.88rem;">${s.lat_8k}</strong></td>`).join('')}
                </tr>
                <tr>
                    <td class="metric-label">Peak VRAM @ L=8,192</td>
                    ${selected.map(s => `<td><span style="font-size:.82rem;font-weight:600;color:var(--tx-secondary);">${s.vram_8k}</span></td>`).join('')}
                </tr>
                <tr>
                    <td class="metric-label">Speedup vs Dense MHA @ 8k</td>
                    ${selected.map(s => `<td><span style="background:rgba(16,185,129,.14);color:var(--accent-3);padding:3px 8px;border-radius:6px;font-weight:700;font-size:.82rem;">${s.speedup_8k}</span></td>`).join('')}
                </tr>
                <tr>
                    <td class="metric-label">Triton Hardware Kernel</td>
                    ${selected.map(s => `<td style="font-size:.76rem;color:var(--tx-secondary);line-height:1.4;">${s.triton}</td>`).join('')}
                </tr>
                <tr>
                    <td class="metric-label">Hardware Acceleration Target</td>
                    ${selected.map(s => `<td style="font-size:.74rem;color:var(--tx-muted);">${s.hw_target}</td>`).join('')}
                </tr>
            </tbody>
        </table>`;

        compareMatrixView.innerHTML = html;

        // Attach Read Specs click handlers
        compareMatrixView.querySelectorAll('.compare-view-doc-btn').forEach(btn => {
            btn.addEventListener('click', () => {
                const docId = btn.dataset.docid;
                const item = allItems.find(i => i.id === docId);
                if (item) {
                    closeCompareModal();
                    loadArticle(item);
                }
            });
        });
    }

    if (compareBtn) compareBtn.addEventListener('click', openCompareModal);
    if (compareBackdrop) compareBackdrop.addEventListener('click', closeCompareModal);
    if (compareCloseBtn) compareCloseBtn.addEventListener('click', closeCompareModal);

    // Close on Escape
    window.addEventListener('keydown', e => {
        if (e.key === 'Escape' && compareOverlay && !compareOverlay.classList.contains('hidden')) {
            closeCompareModal();
        }
    });

    // ── Startup ──────────────────────────────────────────────────
    await loadCatalog();
    handleHash();

});

