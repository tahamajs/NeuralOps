import re

with open("Docs/app.js", "r") as f:
    content = f.read()

replacement = """            let attnName = k;
            for(let t of taskNames) {
                if(k.includes('_' + t + '_')) {
                    attnName = k.split('_' + t + '_')[0];
                    break;
                }
            }
            plotHtml += `<div class="glass p-4 rounded-lg border border-[var(--border-color)]">
                <div class="flex justify-between items-center mb-2">
                    <h4 class="text-sm font-bold text-[var(--tx-muted)] uppercase tracking-wide">
                        <span class="text-white">${attnName}</span> | GLUE: ${taskName}
                    </h4>
                </div>"""

pattern = re.compile(r'            plotHtml \+= `<div class="glass p-4 rounded-lg border border-\[var\(--border-color\)]">\n                <div class="flex justify-between items-center mb-2">\n                    <h4 class="text-sm font-bold text-\[var\(--tx-muted\)] uppercase tracking-wide">GLUE Task: \$\{taskName\}</h4>\n                </div>', re.DOTALL)
new_content = pattern.sub(replacement, content)

with open("Docs/app.js", "w") as f:
    f.write(new_content)
