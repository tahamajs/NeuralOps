import re

with open('Docs/visualizer/app.js', 'r') as f:
    code = f.read()

# Fix Speed
code = code.replace("simulationSpeed = parseFloat(e.target.value) * 0.015;", "simulationSpeed = parseFloat(e.target.value);")

# Fix duplicate Play
pattern = re.compile(r"const btnPlay = document\.getElementById\('btn-play-pause'\);\s*const selSpeed = document\.getElementById\('select-speed'\);\s*const scrubber = document\.getElementById\('timeline-scrubber'\);\s*const timeCode = document\.getElementById\('timeline-step'\);\s*if \(btnPlay\)\s*\{\s*btnPlay\.addEventListener\('click', \(\) => \{\s*if \(typeof isPlaying !== 'undefined'\) \{\s*isPlaying = !isPlaying;\s*btnPlay\.innerText = isPlaying \? 'Pause' : 'Play';\s*btnPlay\.classList\.toggle\('paused', !isPlaying\);\s*\}\s*\}\);\s*\}", re.DOTALL)

code = pattern.sub("const selSpeed = document.getElementById('select-speed');\n    const scrubber = document.getElementById('timeline-scrubber');\n    const timeCode = document.getElementById('timeline-step');", code)

with open('Docs/visualizer/app.js', 'w') as f:
    f.write(code)

print("Fixed controls")
