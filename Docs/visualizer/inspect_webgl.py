import asyncio
from playwright.async_api import async_playwright

async def main():
    brave_path = "/Applications/Brave Browser.app/Contents/MacOS/Brave Browser"
    async with async_playwright() as p:
        browser = await p.chromium.launch(
            executable_path=brave_path,
            headless=True,
            args=[
                "--enable-webgl",
                "--ignore-gpu-blocklist",
                "--use-gl=angle",
                "--use-angle=metal"
            ]
        )
        page = await browser.new_page(viewport={"width": 1440, "height": 900})

        page.on("console", lambda msg: print(f"CONSOLE [{msg.type}]: {msg.text}"))

        await page.goto("http://localhost:8000/index.html", wait_until="networkidle")
        await asyncio.sleep(2)

        info = await page.evaluate("""() => {
            const gl = window.__SCENE__;
            const cam = window.__CAMERA__;
            const ctrl = window.__CONTROLS__;
            const granular = window.__GRANULAR__;
            const canvas = document.getElementById('webgl-canvas');
            const webgl = canvas.getContext('webgl2') || canvas.getContext('webgl');

            return {
                sceneChildrenCount: gl ? gl.children.length : 0,
                granularChildrenCount: granular ? granular.children.length : 0,
                camPos: cam ? { x: cam.position.x, y: cam.position.y, z: cam.position.z } : null,
                target: ctrl ? { x: ctrl.target.x, y: ctrl.target.y, z: ctrl.target.z } : null,
                camNear: cam ? cam.near : null,
                camFar: cam ? cam.far : null,
                hasWebGL: !!webgl,
                webglError: webgl ? webgl.getError() : 'no webgl',
                canvasRect: canvas ? canvas.getBoundingClientRect() : null
            };
        }""")

        print("DIAGNOSTIC INFO:", info)
        await page.screenshot(path="Docs/visualizer/screenshot_diag.png")
        await browser.close()

if __name__ == "__main__":
    asyncio.run(main())
