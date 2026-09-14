const { chromium } = require('playwright');
(async () => {
  const browser = await chromium.launch({ headless: true });
  const page = await browser.newPage();
  const logs = [], errors = [];
  page.on('console', msg => { if (msg.type() !== 'log') return; logs.push(msg.text()); });
  page.on('pageerror', err => errors.push(err.message));
  await page.goto('http://127.0.0.1:8000', { waitUntil: 'networkidle', timeout: 20000 });
  await page.waitForTimeout(5500);
  
  // Check toggleTour exists
  const hasTour = await page.evaluate(() => typeof window.toggleTour);
  console.log('toggleTour:', hasTour);
  
  // Try clicking tour button via JS (bypassing pointer event issues)
  await page.evaluate(() => { if (window.toggleTour) window.toggleTour(); });
  await page.waitForTimeout(500);
  
  // Check btn text changed
  const btnText = await page.$eval('#btn-auto-tour', el => el.textContent.trim()).catch(() => 'NOT FOUND');
  console.log('btn-auto-tour text after click:', btnText);
  
  // Check tour label visible
  const lblOpacity = await page.$eval('#tour-step-label', el => el.style.opacity).catch(() => 'N/A');
  console.log('tour-step-label opacity:', lblOpacity);
  
  console.log('ERRORS:', errors.length ? errors.join('\n') : 'NONE');
  await browser.close();
})();
