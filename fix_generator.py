with open('generate_all_12_sections.py') as f:
    c = f.read()

# Fix line 8: \\lr{{{mod_name}}} -> \\lr{{{mod_name_escaped}}}
c = c.replace(r"\lr{{{mod_name}}}", r"\lr{{{mod_name_escaped}}}")

# Fix labels: fig:arch_" + mod_name -> fig:arch_" + mod_name.replace('_', '-')
c = c.replace('fig:arch_" + mod_name + r"}"', 'fig:arch_" + mod_name.replace(\'_\', \'-\') + r"}"')
c = c.replace('fig:spectral_" + mod_name + r"}"', 'fig:spectral_" + mod_name.replace(\'_\', \'-\') + r"}"')
c = c.replace('fig:bar_" + mod_name + r"}"', 'fig:bar_" + mod_name.replace(\'_\', \'-\') + r"}"')

with open('generate_all_12_sections.py', 'w') as f:
    f.write(c)

print("Updated generate_all_12_sections.py")
