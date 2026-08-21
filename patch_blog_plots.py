import re

with open("Docs/app.js", "r") as f:
    content = f.read()

mapping_code = """    let plotMatches = [];
    if (plotHistories) {
        if (item.category === 'blog_series') {
            const blogMap = {
                '01': ['svd_linear', 'aegis', 'svd', 'asvd', 'nystrom'],
                '02': ['mamba', 'gated_delta_net', 'linformer', 'performer', 'flash3'],
                '03': ['mla', 'nsa', 'adaptive', 'sparse'],
                '04': ['lora', 'lowrank', 'svd_compressed'],
                '05': ['quant', 'prun', 'fp8', 'int4']
            };
            const prefix = item.id.substring(0, 2);
            const targets = blogMap[prefix] || [];
            plotMatches = Object.keys(plotHistories).filter(k => {
                return targets.some(t => k.startsWith(t));
            });
        } else {
            plotMatches = Object.keys(plotHistories).filter(k => {
                for (let t of taskNames) {
                    if (k.includes('_' + t + '_')) {
                        let attn = k.split('_' + t + '_')[0];
                        let idBase = item.id.replace('.md', '').replace('_attention', '');
                        let attnBase = attn.replace('_attention', '');
                        if (idBase === attnBase || item.id.startsWith(attn + '.md') || item.id.startsWith(attn + '_')) {
                            return true;
                        }
                    }
                }
                return false;
            });
        }
    }"""

pattern = re.compile(r'    let plotMatches = \[\];\n    if \(plotHistories\) \{.*?    \}\n', re.DOTALL)
new_content = pattern.sub(mapping_code + "\n", content)

with open("Docs/app.js", "w") as f:
    f.write(new_content)
