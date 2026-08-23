import os
import re

dest_dir = "paper/neuralops_paper"
os.makedirs(dest_dir, exist_ok=True)
os.makedirs(os.path.join(dest_dir, "sections"), exist_ok=True)
os.makedirs(os.path.join(dest_dir, "appendices"), exist_ok=True)

with open("paper/main.tex", "r") as f:
    lines = f.readlines()

# 1. Preamble (lines 1 to 78, before \title)
# Line indices are 0-based
preamble_lines = lines[0:78]
# Clean up preamble: remove \documentclass (since main.tex has it)
preamble_text = "".join(preamble_lines)
preamble_text = re.sub(r"\\documentclass\[.*?\]\{.*?\}\n*", "", preamble_text).strip()
with open(os.path.join(dest_dir, "preamble.tex"), "w") as f:
    f.write(preamble_text + "\n")

# 2. Title (lines 79 to 88)
title_lines = lines[78:88]
with open(os.path.join(dest_dir, "title.tex"), "w") as f:
    f.write("".join(title_lines).strip() + "\n")

# 3. Abstract (lines 92 to 94)
abstract_lines = lines[92:94]
with open(os.path.join(dest_dir, "sections", "00_abstract.tex"), "w") as f:
    f.write("".join(abstract_lines).strip() + "\n")

# 4. Main Sections
sections_map = [
    ("01_introduction.tex", 94, 136),
    ("02_background.tex", 136, 145),
    ("03_related_work.tex", 145, 155),
    ("04_methodology.tex", 155, 346),
    ("05_systems_roofline.tex", 346, 360),
    ("06_experimental_evaluation.tex", 360, 446),
    ("07_broader_impacts.tex", 446, 455),
    ("08_conclusion.tex", 455, 462),
]

for filename, start, end in sections_map:
    sec_content = "".join(lines[start:end]).strip()
    with open(os.path.join(dest_dir, "sections", filename), "w") as f:
        f.write(sec_content + "\n")

# 5. Appendices
# Line 531: \clearpage \appendix
# Line 533-534: appendix header
overview_content = "".join(lines[530:534]).strip()
with open(os.path.join(dest_dir, "appendices", "app_00_overview.tex"), "w") as f:
    f.write(overview_content + "\n")

appendices_map = [
    ("app_01_attentions.tex", 534, 27473),
    ("app_02_compressions.tex", 27473, 38575),
    ("app_03_drift_detectors.tex", 38575, 38678),
    ("app_04_visualizations.tex", 38678, 38690),
    ("app_05_proofs.tex", 38690, 38780),
]

for filename, start, end in appendices_map:
    app_content = "".join(lines[start:end]).strip()
    with open(os.path.join(dest_dir, "appendices", filename), "w") as f:
        f.write(app_content + "\n")

# 6. BibTeX References (references.bib)
bibtex_text = """@book{absil2008optimization,
  author    = {Absil, P.-A. and Mahony, R. and Sepulchre, R.},
  title     = {Optimization Algorithms on Matrix Manifolds},
  publisher = {Princeton University Press},
  year      = {2008}
}

@article{ashkboos2024slicegpt,
  author    = {Ashkboos, Saleh and Croci, I. M. and do Amaro, M. L. and Pfenninger, B. M. and Rotenberg, A. B. and Hoefler, Torsten},
  title     = {SliceGPT: Compress Large Language Models by Deleting Rows and Columns},
  journal   = {arXiv preprint arXiv:2401.15024},
  year      = {2024}
}

@article{choromanski2020rethinking,
  author    = {Choromanski, Krzysztof and Likhosherstov, Valerii and Dohan, David and Song, Xingyou and Gane, Andreea and Sarlos, Tamas and Hawkins, Peter and Davis, Jared and Mohiuddin, Afroz and Kaiser, {\L}ukasz and others},
  title     = {Rethinking Attention with Performers},
  journal   = {arXiv preprint arXiv:2009.14794},
  year      = {2020}
}

@article{edelman1998geometry,
  author    = {Edelman, Alan and Arias, T. A. and Smith, S. T.},
  title     = {The Geometry of Algorithms with Orthogonality Constraints},
  journal   = {SIAM Journal on Matrix Analysis and Applications},
  volume    = {20},
  number    = {2},
  pages     = {303--353},
  year      = {1998}
}

@article{gu2023mamba,
  author    = {Gu, Albert and Dao, Tri},
  title     = {Mamba: Linear-Time Sequence Modeling with Selective State Spaces},
  journal   = {arXiv preprint arXiv:2312.00752},
  year      = {2023}
}

@article{hu2021lora,
  author    = {Hu, Edward J. and Shen, Yelong and Wallis, Phillip and Allen-Zhu, Zeyuan and Li, Yuanzhi and Wang, Shean and Wang, Lu and Chen, Weizhu},
  title     = {LoRA: Low-Rank Adaptation of Large Language Models},
  journal   = {arXiv preprint arXiv:2106.09685},
  year      = {2021}
}

@inproceedings{katharopoulos2020transformers,
  author    = {Katharopoulos, Angelos and Vyas, Apoorv and Pappas, Nikolaos and Fleuret, Fran{\c{c}}ois},
  title     = {Transformers are {RNNs}: Fast Autoregressive Transformers with Linear Attention},
  booktitle = {International Conference on Machine Learning (ICML)},
  pages     = {5156--5165},
  year      = {2020}
}

@article{peng2023rwkv,
  author    = {Peng, Bo and Alcaide, Eric and Anthony, Quentin and Albalak, Alon and Arcadinho, Samuel and Cao, Huanqi and Cheng, Xin and Chung, Michael and Grella, Matteo and Kranth, K. G. and others},
  title     = {{RWKV}: Reinventing {RNNs} for the Transformer Era},
  journal   = {arXiv preprint arXiv:2305.13048},
  year      = {2023}
}

@inproceedings{qin2022cosformer,
  author    = {Qin, Zhen and Sun, Weixuan and Deng, Hui and Li, Dongxu and Wei, Yunshen and Lv, Baohong and Yan, Jun and Kong, Lingpeng and Zhong, Yiran},
  title     = {cosFormer: Rethinking Softmax in Attention with Linear Scale},
  booktitle = {International Conference on Learning Representations (ICLR)},
  year      = {2022}
}

@inproceedings{schlag2021linear,
  author    = {Schlag, Imanol and Irie, Kazuki and Schmidhuber, J{\"u}rgen},
  title     = {Linear Transformers Are Secretly Fast Weight Programmers},
  booktitle = {International Conference on Machine Learning (ICML)},
  pages     = {9355--9366},
  year      = {2021}
}

@article{sun2023retentive,
  author    = {Sun, Yutao and Dong, Li and Huang, Shaohan and Ma, Shuming and Xia, Yuqing and Xue, Jilong and Wang, Jianyong and Wei, Furu},
  title     = {Retentive Network: A Successor to Transformer for Large Language Models},
  journal   = {arXiv preprint arXiv:2307.08621},
  year      = {2023}
}

@inproceedings{vaswani2017attention,
  author    = {Vaswani, Ashish and Shazeer, Noam and Parmar, Niki and Uszkoreit, Jakob and Jones, Llion and Gomez, Aidan N. and Kaiser, {\L}ukasz and Polosukhin, Illia},
  title     = {Attention Is All You Need},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  pages     = {5998--6008},
  year      = {2017}
}

@article{yuan2023asvd,
  author    = {Yuan, Zhihang and Niu, Lin and Liu, Jiawei and Liu, Wenzhe and Wang, Xing and Shang, Yuzhang and Sun, Guangyu and Wu, Qiang and Wu, Jiaxiang and Zheng, Bingzhe},
  title     = {{ASVD}: Activation-Aware Singular Value Decomposition for Compressing Large Language Models},
  journal   = {arXiv preprint arXiv:2310.13510},
  year      = {2023}
}
"""

with open(os.path.join(dest_dir, "references.bib"), "w") as f:
    f.write(bibtex_text.strip() + "\n")

# 7. Root main.tex
main_latex = r"""\documentclass[10pt,twocolumn,letterpaper]{article}

% =========================================================================
% PREAMBLE: Packages, Fonts, Macros, and Theorems
% =========================================================================
\input{preamble}

\begin{document}

% =========================================================================
% TITLE & AUTHOR METADATA
% =========================================================================
\input{title}
\maketitle

% =========================================================================
% ABSTRACT
% =========================================================================
\input{sections/00_abstract}

% =========================================================================
% MAIN PAPER SECTIONS
% =========================================================================
\input{sections/01_introduction}
\input{sections/02_background}
\input{sections/03_related_work}
\input{sections/04_methodology}
\input{sections/05_systems_roofline}
\input{sections/06_experimental_evaluation}
\input{sections/07_broader_impacts}
\input{sections/08_conclusion}

% =========================================================================
% BIBLIOGRAPHY
% =========================================================================
\bibliographystyle{plain}
\bibliography{references}

% =========================================================================
% APPENDICES
% =========================================================================
\input{appendices/app_00_overview}
\input{appendices/app_01_attentions}
\input{appendices/app_02_compressions}
\input{appendices/app_03_drift_detectors}
\input{appendices/app_04_visualizations}
\input{appendices/app_05_proofs}

\end{document}
"""

with open(os.path.join(dest_dir, "main.tex"), "w") as f:
    f.write(main_latex.strip() + "\n")

# 8. Makefile
makefile = """# =========================================================================
# PROFESSIONAL LATEX MAKEFILE FOR NEURALOPS TECHNICAL REPORT
# =========================================================================

DOC = main
LATEXMK = latexmk
LATEXMK_FLAGS = -pdf -interaction=nonstopmode -synctex=1 -file-line-error

.PHONY: all clean fullclean watch

all: $(DOC).pdf

$(DOC).pdf: $(DOC).tex preamble.tex title.tex references.bib sections/*.tex appendices/*.tex
\t$(LATEXMK) $(LATEXMK_FLAGS) $(DOC).tex

watch:
\t$(LATEXMK) $(LATEXMK_FLAGS) -pvc $(DOC).tex

clean:
\t$(LATEXMK) -c
\trm -f *.bbl *.blg *.aux *.log *.out *.toc *.lof *.lot *.synctex.gz *.fls *.fdb_latexmk
\trm -rf sections/*.aux sections/*.log appendices/*.aux appendices/*.log

fullclean: clean
\trm -f $(DOC).pdf
"""

with open(os.path.join(dest_dir, "Makefile"), "w") as f:
    f.write(makefile)

# 9. .gitignore
gitignore = """# LaTeX auxiliary build files
*.aux
*.bbl
*.blg
*.fdb_latexmk
*.fls
*.log
*.out
*.synctex.gz
*.toc
*.lof
*.lot
sections/*.aux
appendices/*.aux
"""

with open(os.path.join(dest_dir, ".gitignore"), "w") as f:
    f.write(gitignore)

print("Modularization complete! All files created in", dest_dir)
