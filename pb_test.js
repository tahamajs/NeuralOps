const { chromium } = require('playwright');
(async () => {
  const browser = await chromium.launch({ headless: true });
  const page = await browser.newPage();
  const errors = [];
  page.on('pageerror', err => errors.push(err.message));
  page.on('console', msg => { if (msg.type()==='error') errors.push(msg.text()); });
  await page.goto('http://127.0.0.1:8000', { waitUntil: 'networkidle', timeout: 20000 });
  await page.waitForTimeout(4000);
  console.log('PAGE ERRORS:', errors.length > 0 ? errors.join('\n') : 'NONE');
  const elems = ['#fps-counter','#btn-auto-tour','#btn-legend-toggle','#btn-pin-module','#param-breakdown-table'];
  for (const s of elems) {
    const el = await page.$(s);
    console.log(s + ': ' + (el ? 'FOUND' : 'MISSING'));
  }
  await browser.close();
})();
