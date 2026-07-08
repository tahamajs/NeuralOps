const fs = require("fs");
const path = require("path");

const base = path.join(__dirname, "../src");

const features = [
  "analysis",
  "analytics",
  "admin",
  "agent",
  "cluster",
  "config",
  "drift",
  "fairness",
  "robustness",
  "hpo",
  "federated",
  "jobs",
  "modelVersioning",
  "organizations",
  "portfolios",
  "reviews",
  "sweeps",
  "search",
  "activity",
  "templates",
  "system",
];

features.forEach((f) => {
  const dir = path.join(base, "features", f, "pages");
  fs.mkdirSync(dir, { recursive: true });
  const pageName = f.charAt(0).toUpperCase() + f.slice(1) + "Page";
  const content = `import { Card } from '../../../components/ui/TerminalCard'\n\nexport default function ${pageName}() {\n  return <Card className="p-6"><h1 className="text-white">${pageName}</h1></Card>\n}\n`;
  fs.writeFileSync(path.join(dir, `${pageName}.tsx`), content);
});

console.log("All stub pages generated.");
