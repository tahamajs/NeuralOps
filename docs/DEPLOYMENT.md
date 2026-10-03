# Research Hub deployment

The GitHub Pages site is the static `Docs/` application. Its generated catalog unifies guides, verification reports, benchmark results, interactive studios, architecture specifications, research deep dives, and framework references. The deployment workflow also builds the optional frontend studio into `Docs/app/` and copies the framework Markdown sources so every catalog entry remains readable after publishing.

## Local preview

From the repository root:

```bash
python generate_catalog.py
python -m http.server 8080 --directory Docs
```

Open <http://127.0.0.1:8080/>. The catalog is generated deterministically from `Docs/content/` and should be committed whenever documentation files are added or renamed.

## Experiment telemetry

The hub is static by default and reports telemetry as offline. To connect it to a compatible experiment server, set `window.FRAMEWORK_API_BASE` before loading `app.js`, for example:

```html
<script>window.FRAMEWORK_API_BASE = 'https://experiments.example.com';</script>
<script src="app.js"></script>
```

The server should expose `/api/stats` and `/results/all_histories.json`. The UI accepts both the legacy GPU array response and the named object form (`gpu_util`, `mem_used_mb`). Keep CORS enabled for the deployed Pages origin and do not embed private-network addresses in the published site.

## CI and Pages checks

The repository workflows regenerate and verify `Docs/catalog.json`, build the frontend studio, serve the resulting `_site/`, and probe the published catalog plus representative Markdown pages. Run the same catalog check locally with:

```bash
python generate_catalog.py
git diff --exit-code -- Docs/catalog.json
```
