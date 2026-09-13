import re

with open('thesis_final/master_thesis.tex', 'r', encoding='utf-8') as f:
    content = f.read()

old_eq = r"""\begin{equation}
\text{Quality Gate Status} = \begin{cases}
\text{PASS} & \text{Accuracy} \ge 56.00\% \quad \text{and} \quad \text{No CUDA Error} \\
\text{FAIL} & \text{Accuracy} < 56.00\% \quad \text{or} \quad \text{OOM Crash}
\end{cases}
\end{equation}"""

new_eq = r"""\begin{equation}
\text{\lr{Quality Gate Status}} = \begin{cases}
\text{\lr{PASS}} & \text{\lr{Accuracy}} \ge 56.00\% \quad \text{\lr{and}} \quad \text{\lr{No CUDA Error}} \\
\text{\lr{FAIL}} & \text{\lr{Accuracy}} < 56.00\% \quad \text{\lr{or}} \quad \text{\lr{OOM Crash}}
\end{cases}
\end{equation}"""

content = content.replace(old_eq, new_eq)

with open('thesis_final/master_thesis.tex', 'w', encoding='utf-8') as f:
    f.write(content)

print("Equation replaced.")
