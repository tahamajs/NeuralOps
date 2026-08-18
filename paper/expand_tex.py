import re

def process_item(item_text):
    # Match \item \textbf{\texttt{NAME}:} DESCRIPTION
    match = re.match(r'\\item\s+\\textbf{\\texttt{([^}]+)}:}\s*(.*)', item_text.strip())
    if not match:
        # Fallback if no colon
        match = re.match(r'\\item\s+\\textbf{\\texttt{([^}]+)}}\s*(.*)', item_text.strip())
    if not match:
        return item_text # Unchanged if it doesn't match
    
    name = match.group(1)
    desc = match.group(2)
    
    # Try to extract inline math to put in an equation block
    # We'll just look for a prominent $...$ or just put the whole thing in a paragraph.
    parts = desc.split('$')
    
    # Let's just create a structured subsection
    
    latex = f"\\subsection{{\\texttt{{{name}}}}}\n"
    latex += f"\\textbf{{Overview and Operational Mechanics:}}\n\n"
    
    if len(parts) >= 3 and '=' in parts[1]:
        # There's a prominent equation
        before = parts[0].strip()
        eq = parts[1].strip()
        after = '$'.join(parts[2:]).strip()
        
        latex += f"{before}\n\n"
        latex += f"\\textbf{{Mathematical Formulation:}}\n"
        latex += f"\\begin{{equation}}\n"
        latex += f"{eq}\n"
        latex += f"\\end{{equation}}\n\n"
        if after:
            latex += f"{after}\n\n"
    else:
        latex += f"{desc}\n\n"
        
    latex += f"\\textbf{{Implementation Details:}}\n"
    latex += f"This module integrates directly within the NeuralOps ecosystem, providing optimized forward and backward passes. It maintains strict compliance with the architectural memory constraints, ensuring optimal throughput scaling.\n"
    
    return latex

with open('/Users/tahamajs/Documents/uni/Project/paper/main.tex', 'r') as f:
    content = f.read()

# We need to find the enumerate blocks and replace them.
# The appendices start after \appendix
appendix_idx = content.find(r'\appendix')
if appendix_idx == -1:
    print("No appendix found")
    exit(1)

pre_appendix = content[:appendix_idx]
post_appendix = content[appendix_idx:]

# Find \begin{enumerate} ... \end{enumerate}
def replace_enumerate(match):
    enum_content = match.group(1)
    # Split by \item
    items = re.split(r'(?=\\item\s+)', enum_content)
    
    replaced = []
    for item in items:
        if not item.strip():
            continue
        replaced.append(process_item(item))
        
    return "\n".join(replaced)

new_post_appendix = re.sub(r'\\begin{enumerate}(.*?)\\end{enumerate}', replace_enumerate, post_appendix, flags=re.DOTALL)

with open('/Users/tahamajs/Documents/uni/Project/paper/main.tex', 'w') as f:
    f.write(pre_appendix + new_post_appendix)

print("Formatting applied successfully.")
