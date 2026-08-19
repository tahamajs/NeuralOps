import { useState, useEffect, useMemo } from "react";
import { Link, useSearchParams } from "react-router-dom";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import remarkMath from "remark-math";
import rehypeKatex from "rehype-katex";
import rehypeHighlight from "rehype-highlight";
import {
  ArrowLeft,
  BookOpen,
  Cpu,
  Zap,
  Layers,
  Search,
  FileText,
  Sparkles,
  ExternalLink,
  ChevronRight,
  RefreshCw,
  FolderOpen,
  Filter,
  CheckCircle2,
  Sliders
} from "lucide-react";

interface CatalogItem {
  id?: string;
  name?: string;
  title?: string;
  file?: string;
  path?: string;
  category?: string;
  words?: number;
}

interface CatalogData {
  attentions?: Record<string, any>;
  compressions?: Record<string, any>;
  attentions_blog?: Record<string, any>;
  compressions_blog?: Record<string, any>;
  blog_series?: Record<string, any>;
  framework?: Record<string, any>;
  [key: string]: any;
}

export default function DocsPage() {
  const [searchParams, setSearchParams] = useSearchParams();
  const [catalog, setCatalog] = useState<CatalogData | null>(null);
  const [loading, setLoading] = useState(true);
  const [activeTab, setActiveTab] = useState<string>("attentions_blog");
  const [selectedDocPath, setSelectedDocPath] = useState<string>("");
  const [docContent, setDocContent] = useState<string>("");
  const [contentLoading, setContentLoading] = useState<boolean>(false);
  const [searchQuery, setSearchQuery] = useState<string>("");
  const [filterCategory, setFilterCategory] = useState<string>("all");

  // Load catalog.json on mount
  useEffect(() => {
    async function fetchCatalog() {
      try {
        setLoading(true);
        const res = await fetch("/Docs/catalog.json");
        if (res.ok) {
          const data = await res.json();
          setCatalog(data);
          
          // Check URL param or default
          const docParam = searchParams.get("doc");
          if (docParam) {
            setSelectedDocPath(docParam);
          } else if (data.attentions_blog && Object.keys(data.attentions_blog).length > 0) {
            const firstKey = Object.keys(data.attentions_blog)[0];
            const firstItem = data.attentions_blog[firstKey];
            const path = typeof firstItem === "string" ? firstItem : firstItem.path || firstItem.file || `content/attentions_blog/${firstKey}.md`;
            setSelectedDocPath(path);
          }
        }
      } catch (err) {
        console.error("Failed to load catalog.json:", err);
      } finally {
        setLoading(false);
      }
    }
    fetchCatalog();
  }, []);

  // Fetch document content when selectedDocPath changes
  useEffect(() => {
    if (!selectedDocPath) return;
    async function fetchDoc() {
      try {
        setContentLoading(true);
        // Normalize path
        let cleanPath = selectedDocPath;
        if (cleanPath.startsWith("Docs/")) {
          cleanPath = cleanPath.replace("Docs/", "");
        }
        if (!cleanPath.startsWith("/")) {
          cleanPath = `/${cleanPath}`;
        }
        const res = await fetch(`/Docs${cleanPath}`);
        if (res.ok) {
          const text = await res.text();
          setDocContent(text);
        } else {
          setDocContent(`# 404 - Document Not Found\n\nCould not load \`/Docs${cleanPath}\`.`);
        }
      } catch (err) {
        setDocContent(`# Error\n\nFailed to fetch document content.`);
      } finally {
        setContentLoading(false);
      }
    }
    fetchDoc();
  }, [selectedDocPath]);

  // Extract items list for current active tab
  const items = useMemo(() => {
    if (!catalog) return [];
    const tabData = catalog[activeTab];
    if (!tabData) return [];

    if (Array.isArray(tabData)) {
      return tabData.map((item, idx) => ({
        id: item.id || item.name || `item-${idx}`,
        title: item.title || item.name || `Document ${idx + 1}`,
        path: item.path || item.file || "",
        category: item.category || "General"
      }));
    } else if (typeof tabData === "object") {
      return Object.entries(tabData).map(([key, val]) => {
        if (typeof val === "string") {
          const cleanTitle = key.replace(/_/g, " ").replace(".md", "").toUpperCase();
          return {
            id: key,
            title: cleanTitle,
            path: val.startsWith("Docs/") ? val.replace("Docs/", "") : val,
            category: "General"
          };
        }
        const v = val as any;
        return {
          id: key,
          title: v?.title || v?.name || key.replace(/_/g, " "),
          path: v?.path || v?.file || (activeTab === "attentions_blog" ? `content/attentions_blog/${key}.md` : `content/compressions_blog/${key}.md`),
          category: v?.category || "General"
        };
      });
    }
    return [];
  }, [catalog, activeTab]);

  // Filter items by search query
  const filteredItems = useMemo(() => {
    return items.filter((item) => {
      const matchSearch =
        item.title.toLowerCase().includes(searchQuery.toLowerCase()) ||
        item.path.toLowerCase().includes(searchQuery.toLowerCase());
      const matchCategory = filterCategory === "all" || item.category === filterCategory;
      return matchSearch && matchCategory;
    });
  }, [items, searchQuery, filterCategory]);

  const tabs = [
    { id: "attentions_blog", label: "Attention Blogs (106)", icon: Zap, color: "text-amber-400" },
    { id: "compressions_blog", label: "Compression Blogs (84)", icon: Cpu, color: "text-emerald-400" },
    { id: "blog_series", label: "Series Guides (6)", icon: Layers, color: "text-purple-400" },
    { id: "attentions", label: "Attention Specs (105)", icon: BookOpen, color: "text-cyan-400" },
    { id: "compressions", label: "Compression Specs (80)", icon: Sliders, color: "text-rose-400" },
    { id: "framework", label: "Framework Reports (3)", icon: FileText, color: "text-blue-400" }
  ];

  return (
    <div className="min-h-screen bg-slate-950 text-slate-300 font-sans flex flex-col relative overflow-hidden">
      {/* Background Cyber Grid */}
      <div className="absolute inset-0 bg-[linear-gradient(to_right,#0f172a_1px,transparent_1px),linear-gradient(to_bottom,#0f172a_1px,transparent_1px)] bg-[size:4rem_4rem] [mask-image:radial-gradient(ellipse_60%_50%_at_50%_0%,#000_70%,transparent_100%)] opacity-30 pointer-events-none" />

      {/* Header */}
      <header className="relative z-10 border-b border-slate-900 bg-slate-950/80 backdrop-blur-md px-6 py-3.5 flex items-center justify-between">
        <div className="flex items-center gap-3">
          <Link
            to="/"
            className="p-2 hover:bg-white/5 rounded-lg text-slate-400 hover:text-white transition-colors"
          >
            <ArrowLeft size={16} />
          </Link>
          <div className="flex items-center gap-2">
            <div className="w-8 h-8 bg-gradient-to-br from-cyan-500 to-purple-600 rounded-lg flex items-center justify-center shadow-lg shadow-cyan-500/20">
              <Sparkles size={18} className="text-white" />
            </div>
            <div>
              <span className="font-bold text-white tracking-wider text-sm flex items-center gap-2">
                NEURALOPS MASTERCLASS HUB
                <span className="px-2 py-0.5 rounded-full text-[10px] bg-cyan-500/10 text-cyan-400 border border-cyan-500/20 font-mono">
                  384 DOCS
                </span>
              </span>
            </div>
          </div>
        </div>

        {/* Global Tab Selection */}
        <div className="hidden lg:flex items-center gap-1 bg-slate-900/80 p-1 rounded-xl border border-slate-800">
          {tabs.map((tab) => {
            const Icon = tab.icon;
            const isActive = activeTab === tab.id;
            return (
              <button
                key={tab.id}
                onClick={() => {
                  setActiveTab(tab.id);
                  setSearchQuery("");
                }}
                className={`flex items-center gap-2 px-3 py-1.5 rounded-lg text-xs font-semibold transition-all ${
                  isActive
                    ? "bg-cyan-500/20 text-cyan-300 border border-cyan-500/40 shadow-sm"
                    : "text-slate-400 hover:text-slate-200 hover:bg-white/5 border border-transparent"
                }`}
              >
                <Icon size={14} className={tab.color} />
                {tab.label}
              </button>
            );
          })}
        </div>

        <Link
          to="/login"
          className="px-4 py-1.5 bg-cyan-500/10 hover:bg-cyan-500/20 border border-cyan-500/30 rounded-lg text-xs font-semibold tracking-wider text-cyan-300 transition-all"
        >
          Console
        </Link>
      </header>

      {/* Main Layout */}
      <div className="flex-1 flex relative z-10 w-full px-6 py-6 gap-6 overflow-hidden">
        {/* Left Sidebar: Document List & Search */}
        <aside className="w-80 shrink-0 flex flex-col bg-slate-900/50 backdrop-blur-sm border border-slate-800/80 rounded-2xl p-4 gap-4 overflow-hidden">
          {/* Search Box */}
          <div className="relative">
            <Search size={15} className="absolute left-3 top-1/2 -translate-y-1/2 text-slate-500" />
            <input
              type="text"
              placeholder={`Search ${items.length} documents...`}
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              className="w-full pl-9 pr-4 py-2 bg-slate-950/60 border border-slate-800 rounded-xl text-xs text-white placeholder-slate-500 focus:outline-none focus:border-cyan-500/50 transition-all"
            />
          </div>

          {/* Mobile Tab Selector */}
          <div className="lg:hidden flex overflow-x-auto gap-1 pb-1">
            {tabs.map((tab) => (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id)}
                className={`px-2.5 py-1 text-[11px] rounded-lg whitespace-nowrap font-medium ${
                  activeTab === tab.id
                    ? "bg-cyan-500/20 text-cyan-300"
                    : "text-slate-400 bg-slate-950/40"
                }`}
              >
                {tab.label}
              </button>
            ))}
          </div>

          {/* Document List */}
          <div className="flex-1 overflow-y-auto pr-1 space-y-1 custom-scrollbar">
            {loading ? (
              <div className="flex flex-col items-center justify-center h-48 gap-2 text-slate-500 text-xs">
                <RefreshCw size={18} className="animate-spin text-cyan-400" />
                <span>Loading catalog...</span>
              </div>
            ) : filteredItems.length === 0 ? (
              <div className="text-center py-12 text-xs text-slate-500">
                No matching documents found.
              </div>
            ) : (
              filteredItems.map((item) => {
                const isSelected = selectedDocPath === item.path || selectedDocPath.endsWith(item.path);
                return (
                  <button
                    key={item.id}
                    onClick={() => {
                      setSelectedDocPath(item.path);
                      setSearchParams({ doc: item.path });
                    }}
                    className={`w-full text-left p-2.5 rounded-xl text-xs transition-all flex items-center justify-between group ${
                      isSelected
                        ? "bg-gradient-to-r from-cyan-500/20 to-purple-500/10 border border-cyan-500/30 text-white font-medium"
                        : "hover:bg-slate-800/40 text-slate-400 hover:text-slate-200 border border-transparent"
                    }`}
                  >
                    <div className="truncate pr-2">
                      <span className="block truncate font-mono text-[11px] text-slate-200">
                        {item.title}
                      </span>
                      <span className="text-[10px] text-slate-500 truncate block">
                        {item.path.split("/").pop()}
                      </span>
                    </div>
                    <ChevronRight
                      size={14}
                      className={`shrink-0 transition-transform ${
                        isSelected ? "text-cyan-400 translate-x-0.5" : "text-slate-600 group-hover:text-slate-400"
                      }`}
                    />
                  </button>
                );
              })
            )}
          </div>

          {/* Footer Metadata */}
          <div className="pt-2 border-t border-slate-800/60 flex items-center justify-between text-[11px] text-slate-500">
            <span>{filteredItems.length} of {items.length} files</span>
            <span className="font-mono text-cyan-400/80">100% PASS</span>
          </div>
        </aside>

        {/* Right Main Pane: Interactive Markdown Viewer */}
        <main className="flex-1 bg-slate-900/40 backdrop-blur-sm border border-slate-800/80 rounded-2xl p-8 overflow-y-auto custom-scrollbar flex flex-col">
          {contentLoading ? (
            <div className="flex-1 flex flex-col items-center justify-center gap-3 text-slate-400">
              <RefreshCw size={24} className="animate-spin text-cyan-400" />
              <span className="text-xs font-mono">Loading Masterclass Deep Dive...</span>
            </div>
          ) : !docContent ? (
            <div className="flex-1 flex flex-col items-center justify-center text-slate-500 gap-2">
              <FolderOpen size={32} className="text-slate-600" />
              <p className="text-xs">Select a masterclass blog or reference document from the sidebar.</p>
            </div>
          ) : (
            <article className="prose prose-invert prose-cyan max-w-none prose-headings:font-bold prose-headings:tracking-tight prose-a:text-cyan-400 prose-pre:bg-slate-950 prose-pre:border prose-pre:border-slate-800 prose-pre:rounded-xl">
              <ReactMarkdown
                remarkPlugins={[remarkGfm, remarkMath]}
                rehypePlugins={[rehypeKatex, rehypeHighlight]}
              >
                {docContent}
              </ReactMarkdown>
            </article>
          )}
        </main>
      </div>
    </div>
  );
}
