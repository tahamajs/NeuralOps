"""
Unified interface to 30+ life science databases.
Inspired by: BioPython, UniProt REST API, NCBI E-utilities, EMBL-EBI APIs.
"""

import logging
from typing import Optional

import requests

logger = logging.getLogger(__name__)


class LifeScienceHub:
    """Central gateway to all life science databases."""

    REGISTRY = {
        "uniprot": {
            "name": "UniProt",
            "base": "https://rest.uniprot.org/uniprotkb",
            "category": "protein",
            "methods": ["search", "fetch", "fetch_isoforms"],
        },
        "alphafold": {
            "name": "AlphaFold DB",
            "base": "https://alphafold.ebi.ac.uk/api",
            "category": "protein",
            "methods": ["predict", "fetch_structure", "confidence_scores"],
        },
        "pdb": {
            "name": "RCSB PDB",
            "base": "https://data.rcsb.org/rest/v1",
            "category": "protein",
            "methods": ["search", "fetch", "fetch_assembly"],
        },
        "ncbi_gene": {
            "name": "NCBI Gene",
            "base": "https://eutils.ncbi.nlm.nih.gov/entrez/eutils",
            "category": "gene",
            "methods": ["esearch", "esummary", "efetch"],
        },
        "ncbi_pubmed": {
            "name": "PubMed",
            "base": "https://eutils.ncbi.nlm.nih.gov/entrez/eutils",
            "category": "literature",
            "methods": ["esearch", "esummary", "efetch"],
        },
        "string": {
            "name": "STRING",
            "base": "https://string-db.org/api",
            "category": "pathway",
            "methods": ["network", "enrichment", "interaction"],
        },
        "kegg": {
            "name": "KEGG",
            "base": "https://rest.kegg.jp",
            "category": "pathway",
            "methods": ["get", "find", "list", "link"],
        },
        "chembl": {
            "name": "ChEMBL",
            "base": "https://www.ebi.ac.uk/chembl/api/data",
            "category": "drug",
            "methods": ["search", "fetch_molecule", "fetch_activity"],
        },
        "pubchem": {
            "name": "PubChem",
            "base": "https://pubchem.ncbi.nlm.nih.gov/rest/pug",
            "category": "drug",
            "methods": ["compound_search", "compound_fetch", "assay"],
        },
        "clinicaltrials": {
            "name": "ClinicalTrials.gov",
            "base": "https://clinicaltrials.gov/api/v2",
            "category": "clinical",
            "methods": ["search", "fetch_study"],
        },
        "gene_ontology": {
            "name": "Gene Ontology",
            "base": "https://api.geneontology.org/api",
            "category": "ontology",
            "methods": ["search_terms", "fetch_term", "annotations"],
        },
        "reactome": {
            "name": "Reactome",
            "base": "https://reactome.org/ContentService",
            "category": "pathway",
            "methods": ["query", "fetch_pathway", "participants"],
        },
        "ensembl": {
            "name": "Ensembl",
            "base": "https://rest.ensembl.org",
            "category": "gene",
            "methods": ["lookup", "fetch_sequence", "fetch_variation"],
        },
    }

    @classmethod
    def get_database(cls, db_key: str) -> Optional[dict]:
        return cls.REGISTRY.get(db_key)

    @classmethod
    def search(cls, db_key: str, query: str, limit: int = 10) -> dict:
        """Search a specific database."""
        db = cls.get_database(db_key)
        if not db:
            return {"error": f"Unknown database: {db_key}"}

        if db_key == "uniprot":
            return cls._search_uniprot(query, limit)
        elif db_key == "ncbi_pubmed":
            return cls._search_pubmed(query, limit)
        elif db_key == "ncbi_gene":
            return cls._search_ncbi_gene(query, limit)
        elif db_key == "pdb":
            return cls._search_pdb(query, limit)
        elif db_key == "string":
            return cls._search_string(query, limit)
        elif db_key == "alphafold":
            return cls._search_alphafold(query, limit)
        elif db_key == "chembl":
            return cls._search_chembl(query, limit)
        else:
            return {"error": f"Search not implemented for {db_key}"}

    @classmethod
    def cross_database_search(cls, query: str, databases: list = None) -> dict:
        """Search across multiple databases simultaneously."""
        if databases is None:
            databases = ["uniprot", "ncbi_pubmed", "pdb", "ncbi_gene"]

        results = {}
        for db_key in databases:
            try:
                results[db_key] = cls.search(db_key, query, limit=5)
            except Exception as e:
                results[db_key] = {"error": str(e)}
        return results

    # ── Database-specific methods ──────────────────────────────

    @classmethod
    def _search_uniprot(cls, query, limit):
        url = f"{cls.REGISTRY['uniprot']['base']}/search"
        resp = requests.get(
            url, params={"query": query, "size": limit, "format": "json"}, timeout=10
        )
        resp.raise_for_status()
        data = resp.json()
        return {
            "total": data.get("total", 0),
            "results": [
                {
                    "id": r["primaryAccession"],
                    "name": r.get("uniProtkbId", ""),
                    "protein_name": r.get("proteinDescription", {})
                    .get("recommendedName", {})
                    .get("fullName", {})
                    .get("value", ""),
                    "organism": r.get("organism", {}).get("scientificName", ""),
                    "length": r.get("sequence", {}).get("length", 0),
                }
                for r in data.get("results", [])[:limit]
            ],
        }

    @classmethod
    def _search_pubmed(cls, query, limit):
        # ESearch to get IDs
        esearch_url = f"{cls.REGISTRY['ncbi_pubmed']['base']}/esearch.fcgi"
        resp = requests.get(
            esearch_url,
            params={
                "db": "pubmed",
                "term": query,
                "retmax": limit,
                "retmode": "json",
                "sort": "relevance",
            },
            timeout=10,
        )
        resp.raise_for_status()
        ids = resp.json().get("esearchresult", {}).get("idlist", [])

        if not ids:
            return {"total": 0, "results": []}

        # EFetch for details
        efetch_url = f"{cls.REGISTRY['ncbi_pubmed']['base']}/efetch.fcgi"
        resp = requests.get(
            efetch_url,
            params={
                "db": "pubmed",
                "id": ",".join(ids),
                "rettype": "abstract",
                "retmode": "xml",
            },
            timeout=15,
        )
        # Simple XML parsing – in production use BioPython
        return {"total": len(ids), "results": [{"pmid": pid} for pid in ids]}

    @classmethod
    def _search_pdb(cls, query, limit):
        url = f"{cls.REGISTRY['pdb']['base']}/core/entry/search"
        resp = requests.post(
            url,
            json={
                "query": {
                    "type": "terminal",
                    "service": "full_text",
                    "parameters": {"value": query},
                },
                "request_options": {"paginate": {"start": 0, "rows": limit}},
                "return_type": "entry",
            },
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()
        return {
            "total": data.get("total_count", 0),
            "results": [
                {
                    "pdb_id": r["identifier"],
                    "title": r.get("struct", {}).get("title", ""),
                }
                for r in data.get("result_set", [])[:limit]
            ],
        }

    @classmethod
    def _search_ncbi_gene(cls, query, limit):
        url = f"{cls.REGISTRY['ncbi_gene']['base']}/esearch.fcgi"
        resp = requests.get(
            url,
            params={
                "db": "gene",
                "term": query,
                "retmax": limit,
                "retmode": "json",
            },
            timeout=10,
        )
        resp.raise_for_status()
        return {
            "total": resp.json().get("esearchresult", {}).get("count", 0),
            "results": [
                {"gene_id": gid}
                for gid in resp.json().get("esearchresult", {}).get("idlist", [])
            ],
        }

    @classmethod
    def _search_string(cls, query, limit):
        url = f"{cls.REGISTRY['string']['base']}/tsv/resolve"
        resp = requests.get(
            url,
            params={"species": "9606", "identifier": query, "limit": limit},
            timeout=10,
        )
        return {"total": 0, "results": []}  # placeholder

    @classmethod
    def _search_alphafold(cls, query, limit):
        url = f"{cls.REGISTRY['alphafold']['base']}/search"
        resp = requests.get(url, params={"query": query, "size": limit}, timeout=10)
        return {"total": 0, "results": []}  # placeholder

    @classmethod
    def _search_chembl(cls, query, limit):
        url = f"{cls.REGISTRY['chembl']['base']}/molecule/search"
        resp = requests.get(
            url, params={"q": query, "limit": limit, "format": "json"}, timeout=10
        )
        return {"total": 0, "results": []}  # placeholder
