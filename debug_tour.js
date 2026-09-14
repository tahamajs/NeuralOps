const { chromium } = require('playwright');
(async () => {
  const browser = await chromium.launch({ headless: true });
  const page = await browser.newPage();
  const logs = [];
  page.on('console', msg => logs.push(`[${msg.type()}] ${msg.text()}`));
  page.on('pageerror', err => logs.push(`[ERROR] ${err.message}`));
  await page.goto('http://127.0.0.1:8000', { waitUntil: 'networkidle', timeout: 20000 });
  await page.waitForTimeout(5000);

  // Check if toggleTour exists
  const hasTour = await page.evaluate(() => typeof window.toggleTour);
  console.log('window.toggleTour type:', hasTour);

  // Check if smoothFlyCamera exists
  const hasFly = await page.evaluate(() => typeof window.__CAMERA__);
  console.log('window.__CAMERA__ type:', hasFly);

  // Try clicking the tour button
  const tourBtn = await page.$('#btn-auto-tour');
  if (tourBtn) {
    await tourBtn.click();
    await page.waitForTimeout(1000);
    const isActive = await page.evaluate(() => window._tourActive);
    console.log('_tourActive after click:', isActive);
  } else {
    console.log('btn-auto-tour NOT FOUND');
  }
  
  console.log('\nCONSOLE LOGS (first 30):');
  logs.slice(0, 30).forEach(l => console.log(l));
  await browser.close();
})();
