# Framework web tools

The `web/` directory contains the framework dashboard and the documentation
browser assets.

## Runtime dashboard

`index.html` is a dependency-light dashboard for `/api/stats`. It displays
telemetry only when the endpoint responds and shows an offline state otherwise.
Hardware names, memory totals, experiment counts, and thermal values come from
runtime data; no GPU model or benchmark total is hard-coded.

## Documentation catalog

`catalog.json` indexes the attention, compression, and framework Markdown
pages. `app.js` is the interactive documentation-browser client and supports a
same-origin API by default. For a remote telemetry server, set:

```html
<script>window.FRAMEWORK_API_BASE = "http://server.example:8080";</script>
```

Serve from the framework directory so the browser can read both `web/` and the
source documentation directories:

```bash
cd Codes/framework
python -m http.server 8080 --directory .
```

Open `http://127.0.0.1:8080/web/`. Do not use `file://`, because browser
security prevents local `fetch()` requests for the catalog and Markdown files.

The catalog is intentionally explicit and reviewable. When a new Markdown
page is added, add it to `catalog.json` or regenerate the catalog before
shipping the UI.
