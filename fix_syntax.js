const fs = require('fs');
let code = fs.readFileSync('Docs/visualizer/app.js', 'utf8');

code = code.replace(/8const isAttn =/, '896 : 768);\n    const dFfn = proj.intermediate_size || (modelKey === "qwen" ? 4864 : 3072);\n    const vocabSize = VOCAB_SIZES[modelKey] || 30522;\n\n    const isAttn =');

fs.writeFileSync('Docs/visualizer/app.js', code);
console.log("Fixed syntax error");
