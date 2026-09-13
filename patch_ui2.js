const fs = require('fs');
let code = fs.readFileSync('Docs/visualizer/app.js', 'utf8');

// Replace the buggy Play/Pause logic from my previous patch
code = code.replace(/btnPlay\.addEventListener\('click', \(\) => {[\s\S]*?}\);/, `btnPlay.addEventListener('click', () => {
            if (typeof isPlaying !== 'undefined') {
                isPlaying = !isPlaying;
                btnPlay.innerText = isPlaying ? 'Pause' : 'Play';
                btnPlay.classList.toggle('paused', !isPlaying);
            }
        });`);

fs.writeFileSync('Docs/visualizer/app.js', code);
console.log("Patched play button logic");
