import re
import os

with open('paper/neuralops_paper_final/appendices/app_06_visual_catalog.tex', 'r', encoding='utf-8') as f:
    text = f.read()

# Only keep Attention section up to Compression section if it exists
if r'\section{Compression, Quantization & Low-Rank Engines' in text:
    text = text.split(r'\section{Compression, Quantization & Low-Rank Engines')[0]
if r'\section{Compression, Quantization \& Low-Rank Engines' in text:
    text = text.split(r'\section{Compression, Quantization \& Low-Rank Engines')[0]

os.makedirs('paper/neuralops_paper_final/appendices/modules_catalog', exist_ok=True)

# Split by \subsection{
parts = re.split(r'(?=\\subsection\{)', text)
header = parts[0]
module_parts = parts[1:]

print(f'Total module parts detected: {len(module_parts)}')

includes = []
for p in module_parts:
    # Match the module name in \texttt{...}
    m = re.search(r'\\subsection\{[^\(]+\(\\texttt\{([^\}]+)\}\)\}', p)
    if not m:
        continue
    raw_name = m.group(1)
    mod_name = raw_name.replace(r'\_', '_').replace('\\_', '_').strip()
    
    fpath = f'paper/neuralops_paper_final/appendices/modules_catalog/{mod_name}.tex'
    with open(fpath, 'w', encoding='utf-8') as mf:
        mf.write(p.strip() + '\n')
    
    includes.append(f'\\input{{appendices/modules_catalog/{mod_name}}}')

print(f'Successfully wrote {len(includes)} standalone module files!')

index_content = r'''% AUTO-GENERATED MASTER CATALOG OF INDIVIDUAL MODULE SPECIFICATIONS
\clearpage
\section{Exhaustive Standalone Architectural Catalog (Attention \& SSM Mechanisms)}
\label{app:visual_catalog}

This appendix presents an exhaustive, publication-grade academic characterization for each individual attention and state-space mechanism within the unified evaluation framework.
Each entry is formulated as a complete standalone modular analysis comprising:
(1) Module Overview \& Track Classification;
(2) Exact Mathematical Forward Formulations;
(3) Theoretical Computational \& Memory Complexity;
(4) Key Hyperparameter Configurations;
(5) Empirical Hardware Performance Telemetry on NVIDIA RTX 4070 (Latency, Throughput, Speedup, Params);
(6) Architectural Strengths \& Potential Limitations;
(7) Implementation \& Systems Notes; and
(8) Bespoke 3-Panel TikZ Architectural Diagrams.

\clearpage
''' + '\n'.join(includes) + '\n'

with open('paper/neuralops_paper_final/appendices/app_06_visual_catalog.tex', 'w', encoding='utf-8') as f:
    f.write(index_content)

print('Updated app_06_visual_catalog.tex to cleanly include all multi-files!')
