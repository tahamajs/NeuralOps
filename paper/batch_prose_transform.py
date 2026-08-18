import glob
import re
import os

def convert_bullets_to_prose(text):
    lines = text.split('\n')
    new_lines = []
    
    in_code = False
    in_table = False
    
    current_bullet_group = []
    
    def flush_bullets():
        nonlocal current_bullet_group
        if not current_bullet_group:
            return
        
        # Check if the group is Mechanism/Consequence/Mitigation or Key-Value pairs
        combined_prose = []
        
        # Check if it's Mechanism / Consequence / Mitigation
        mech = None
        cons = None
        mit = None
        other_items = []
        
        for item in current_bullet_group:
            m_match = re.match(r'^\s*[-*]\s+\*\*Mechanism\*\*:\s*(.*)', item, re.DOTALL)
            c_match = re.match(r'^\s*[-*]\s+\*\*Consequence\*\*:\s*(.*)', item, re.DOTALL)
            t_match = re.match(r'^\s*[-*]\s+\*\*Mitigation\*\*:\s*(.*)', item, re.DOTALL)
            
            if m_match:
                mech = m_match.group(1).strip()
            elif c_match:
                cons = c_match.group(1).strip()
            elif t_match:
                mit = t_match.group(1).strip()
            else:
                other_items.append(item)
                
        if mech or cons or mit:
            para = []
            if mech:
                para.append(f"At the architectural level, the underlying operational mechanism reveals that {mech[0].lower() + mech[1:] if mech and mech[0].isupper() and not mech.startswith('$') and not mech.startswith('\\') else mech}")
            if cons:
                para.append(f"Consequently, this dynamic induces structural implications wherein {cons[0].lower() + cons[1:] if cons and cons[0].isupper() and not cons.startswith('$') and not cons.startswith('\\') else cons}")
            if mit:
                para.append(f"To systematically mitigate this constraint, the framework enforces an optimization protocol such that {mit[0].lower() + mit[1:] if mit and mit[0].isupper() and not mit.startswith('$') and not mit.startswith('\\') else mit}")
            combined_prose.append(" ".join(para))
            
        if other_items:
            # Handle general bullet points: convert each to a continuous sentence with formal transitions
            transitions = [
                "Specifically,", "Furthermore,", "Concurrently,", "In this setting,",
                "Additionally,", "Moreover,", "From an analytical standpoint,", "In terms of operational dynamics,"
            ]
            para = []
            for i, it in enumerate(other_items):
                # Clean bullet marker
                it_clean = re.sub(r'^\s*[-*]\s+', '', it).strip()
                # Check for bold title e.g. **Title**: content
                bold_match = re.match(r'^\*\*(.*?)\*\*:\s*(.*)', it_clean, re.DOTALL)
                if bold_match:
                    title = bold_match.group(1).strip()
                    desc = bold_match.group(2).strip()
                    t_word = transitions[i % len(transitions)]
                    para.append(f"{t_word} regarding \\textbf{{{title}}}, {desc[0].lower() + desc[1:] if desc and desc[0].isupper() and not desc.startswith('$') and not desc.startswith('\\') else desc}")
                else:
                    t_word = transitions[i % len(transitions)] if i > 0 else ""
                    para.append(f"{t_word} {it_clean}".strip())
            combined_prose.append(" ".join(para))
            
        for p in combined_prose:
            new_lines.append(p)
            new_lines.append("")
            
        current_bullet_group = []

    for line in lines:
        stripped = line.strip()
        
        if stripped.startswith('```'):
            flush_bullets()
            in_code = not in_code
            new_lines.append(line)
            continue
            
        if in_code:
            new_lines.append(line)
            continue
            
        if stripped.startswith('|') and stripped.endswith('|'):
            flush_bullets()
            new_lines.append(line)
            continue
            
        # Check if line is a bullet item
        if re.match(r'^\s*[-*]\s+', line):
            current_bullet_group.append(stripped)
            continue
        elif current_bullet_group and line.startswith('   '):
            # continuation of bullet
            current_bullet_group[-1] += " " + stripped
            continue
        else:
            flush_bullets()
            new_lines.append(line)
            
    flush_bullets()
    return "\n".join(new_lines)

# Run across all 185 files
files = sorted(glob.glob('/Users/tahamajs/Documents/uni/Project/Docs/content/attentions/*.md') + glob.glob('/Users/tahamajs/Documents/uni/Project/Docs/content/compressions/*.md'))
print(f"Transforming bullet points into continuous academic prose for {len(files)} files...")

transformed_count = 0
for f in files:
    with open(f, 'r', encoding='utf-8') as fp:
        content = fp.read()
    
    if re.search(r'^\s*[-*]\s+', content, re.MULTILINE):
        new_content = convert_bullets_to_prose(content)
        with open(f, 'w', encoding='utf-8') as fp:
            fp.write(new_content)
        transformed_count += 1

print(f"Successfully converted all bullet points to academic prose across {transformed_count} files!")
