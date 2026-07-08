# management/commands/seed_databases.py
from django.core.management.base import BaseCommand
from research_discovery.models import LifeScienceDatabase

DATABASES = [
    # Protein
    (
        "UniProt",
        "uniprot",
        "protein",
        "https://rest.uniprot.org/uniprotkb",
        "https://rest.uniprot.org/uniprotkb/search",
    ),
    (
        "AlphaFold DB",
        "alphafold",
        "protein",
        "https://alphafold.ebi.ac.uk/api",
        "https://alphafold.ebi.ac.uk/api/search",
    ),
    (
        "RCSB PDB",
        "pdb",
        "protein",
        "https://data.rcsb.org/rest/v1",
        "https://data.rcsb.org/rest/v1/core/entry/search",
    ),
    # Gene
    (
        "NCBI Gene",
        "ncbi_gene",
        "gene",
        "https://eutils.ncbi.nlm.nih.gov/entrez/eutils",
        "",
    ),
    ("Ensembl", "ensembl", "gene", "https://rest.ensembl.org", ""),
    # Pathway
    ("STRING", "string", "pathway", "https://string-db.org/api", ""),
    ("KEGG", "kegg", "pathway", "https://rest.kegg.jp", ""),
    ("Reactome", "reactome", "pathway", "https://reactome.org/ContentService", ""),
    # Drug
    ("ChEMBL", "chembl", "drug", "https://www.ebi.ac.uk/chembl/api/data", ""),
    ("PubChem", "pubchem", "drug", "https://pubchem.ncbi.nlm.nih.gov/rest/pug", ""),
    # Literature
    (
        "PubMed",
        "ncbi_pubmed",
        "literature",
        "https://eutils.ncbi.nlm.nih.gov/entrez/eutils",
        "",
    ),
    # Clinical
    (
        "ClinicalTrials.gov",
        "clinicaltrials",
        "clinical",
        "https://clinicaltrials.gov/api/v2",
        "",
    ),
    # Ontology
    (
        "Gene Ontology",
        "gene_ontology",
        "ontology",
        "https://api.geneontology.org/api",
        "",
    ),
    # Additional databases
    ("DrugBank", "drugbank", "drug", "https://go.drugbank.com", ""),
    ("PDBbind", "pdbbind", "drug", "https://www.pdbbind.org.cn", ""),
    ("BindingDB", "bindingdb", "drug", "https://www.bindingdb.org", ""),
    ("BioGRID", "biogrid", "pathway", "https://webservice.thebiogrid.org", ""),
    ("IntAct", "intact", "pathway", "https://www.ebi.ac.uk/intact", ""),
    ("DIP", "dip", "pathway", "https://dip.doe-mbi.ucla.edu", ""),
    ("CORUM", "corum", "pathway", "https://mips.helmholtz-muenchen.de/corum", ""),
    ("DisGeNET", "disgenet", "disease", "https://www.disgenet.org", ""),
    ("OMIM", "omim", "disease", "https://api.omim.org", ""),
    (
        "ClinVar",
        "clinvar",
        "disease",
        "https://eutils.ncbi.nlm.nih.gov/entrez/eutils",
        "",
    ),
    (
        "OpenTargets",
        "opentargets",
        "disease",
        "https://api.platform.opentargets.org",
        "",
    ),
    ("PharmGKB", "pharmgkb", "drug", "https://api.pharmgkb.org", ""),
    ("SIDER", "sider", "drug", "https://sideeffects.embl.de", ""),
    ("STITCH", "stitch", "drug", "https://stitch.embl.de", ""),
    ("InterPro", "interpro", "protein", "https://www.ebi.ac.uk/interpro", ""),
    ("Pfam", "pfam", "protein", "https://pfam.xfam.org", ""),
    ("PROSITE", "prosite", "protein", "https://prosite.expasy.org", ""),
    ("PRIDE", "pride", "protein", "https://www.ebi.ac.uk/pride", ""),
    ("PeptideAtlas", "peptideatlas", "protein", "https://www.peptideatlas.org", ""),
    ("MetaboLights", "metabolights", "drug", "https://www.ebi.ac.uk/metabolights", ""),
]


class Command(BaseCommand):
    help = "Seed 30+ life science databases"

    def handle(self, *args, **options):
        count = 0
        for name, abbr, cat, base_url, api_url in DATABASES:
            _, created = LifeScienceDatabase.objects.get_or_create(
                abbreviation=abbr,
                defaults={
                    "name": name,
                    "category": cat,
                    "base_url": base_url,
                    "api_endpoint": api_url,
                },
            )
            if created:
                count += 1
        self.stdout.write(
            self.style.SUCCESS(f"Seeded {count} new life science databases")
        )
