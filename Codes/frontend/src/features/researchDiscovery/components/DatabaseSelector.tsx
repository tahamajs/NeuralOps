import { Badge } from "../../../components/ui/Badge";

const DB_NAMES: Record<string, string> = {
  arxiv: "arXiv",
  pubmed: "PubMed",
  semantic_scholar: "Semantic Scholar",
  crossref: "Crossref",
  openalex: "OpenAlex",
  dblp: "DBLP",
  doaj: "DOAJ",
  europe_pmc: "Europe PMC",
  hal: "HAL",
  medrxiv: "medRxiv",
  unpaywall: "Unpaywall",
};

export function DatabaseSelector({ selected, onChange, databases }: any) {
  return (
    <div className="flex flex-wrap gap-2">
      {databases.map((db) => (
        <button
          key={db}
          onClick={() =>
            onChange(
              selected.includes(db)
                ? selected.filter((s: string) => s !== db)
                : [...selected, db],
            )
          }
          className={`px-2 py-1 rounded text-xs font-medium transition-colors ${selected.includes(db) ? "bg-neon-blue text-white" : "bg-white/10 text-slate-300 hover:bg-white/20"}`}
        >
          {DB_NAMES[db] || db}
        </button>
      ))}
    </div>
  );
}
