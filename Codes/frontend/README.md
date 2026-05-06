# ET Frontend

This is the Vite + React frontend for the Neural Ops Control Suite.

Quick run (development):

```bash
cd Codes/frontend
npm install
npm run dev
```

Environment:
- Copy `.env.development` to `.env` or set `VITE_API_URL` if your API is hosted elsewhere.

Build:

```bash
npm run build
npm run preview
```

If you see Tailwind classes not being applied, ensure `tailwind.config.cjs` and `postcss.config.cjs` are present (they are included here).
