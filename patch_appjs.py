from pathlib import Path

p = Path("Codes/framework/web/app.js")
content = p.read_text(encoding="utf-8")

old_text = 'noteElem.textContent = `Convergence: ${firstL} -> ${lastL} (${lossCurve.length} sampled points across 3 epochs)`;'
new_text = '''      let epCount = 3;
      if (data.module_progress && data.module_progress.length > 0) {
        const fullMod = data.module_progress.find(m => m.module === modName) || data.module_progress[0];
        if (fullMod && fullMod.epoch) {
           const match = fullMod.epoch.match(/\/ (\d+)/) || fullMod.epoch.match(/\/(\d+)/);
           if (match) epCount = match[1];
        }
      }
      noteElem.textContent = `Convergence: ${firstL} -> ${lastL} (${lossCurve.length} sampled points across ${epCount} epoch${epCount > 1 ? 's' : ''})`;'''

content = content.replace(old_text, new_text)
p.write_text(content, encoding="utf-8")
print("Patched app.js")
