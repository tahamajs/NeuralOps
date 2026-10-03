import asyncio
from playwright.async_api import async_playwright
import os


async def main():
    errors = []
    brave_path = "/Applications/Brave Browser.app/Contents/MacOS/Brave Browser"
    async with async_playwright() as p:
        browser = await p.chromium.launch(executable_path=brave_path, headless=True)
        page = await browser.new_page(viewport={"width": 1440, "height": 900})

        page.on("console", lambda msg: print(f"BROWSER CONSOLE [{msg.type}]: {msg.text}"))
        page.on("pageerror", lambda err: errors.append(str(err)))

        print("Navigating to http://localhost:8000/index.html ...")
        await page.goto("http://localhost:8000/index.html", wait_until="networkidle")
        await asyncio.sleep(2)

        out_dir = os.path.dirname(os.path.abspath(__file__))

        models = ["distilbert", "roberta", "gpt2", "qwen"]
        for m in models:
            print(f"\n--- Testing Model: {m.upper()} ---")
            await page.click(f".model-matrix-card[data-model='{m}']")
            await asyncio.sleep(1.2)

            # Ensure inspector is scrolled to transformation card
            card = await page.query_selector("#transformation-card")
            if card:
                await card.scroll_into_view_if_needed()
                await asyncio.sleep(0.5)
                c_path = os.path.join(out_dir, f"tf_card_{m}.png")
                await card.screenshot(path=c_path)
                print(f"[OK] Transformation Card {m}: {c_path}")

            insp = await page.query_selector("#inspector")
            if insp:
                i_path = os.path.join(out_dir, f"insp_{m}.png")
                await insp.screenshot(path=i_path)
                print(f"[OK] Inspector {m}: {i_path}")

        await browser.close()

    if errors:
        print(f"[FAIL] Page errors encountered: {errors}")
    else:
        print("[SUCCESS] All renders verified without any errors!")


if __name__ == "__main__":
    asyncio.run(main())
