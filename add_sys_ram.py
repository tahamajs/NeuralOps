import re
path = r"C:\Users\Hadi Schumacher\SVD_Project\Codes\framework\live_monitor.py"
with open(path, "r", encoding="utf-8") as f:
    content = f.read()

# Add psutil to get sys ram
if "import psutil" not in content:
    content = content.replace("import json", "import json\nimport psutil")

# Update get_gpu_stats to return sys ram
content = content.replace(
    'return {"gpu_util": gpu_util, "mem_used_mb": mem_used, "mem_total_mb": mem_total, "temp_c": temp, "power_w": power}',
    'return {"gpu_util": gpu_util, "mem_used_mb": mem_used, "mem_total_mb": mem_total, "temp_c": temp, "power_w": power, "sys_ram_used_gb": round(psutil.virtual_memory().used / 1024**3, 1), "sys_ram_total_gb": round(psutil.virtual_memory().total / 1024**3, 1)}'
)

# Update HTML to show sys ram
html_snippet = """
                            <p class="text-xs text-gray-400 mt-2">VRAM Usage</p>
                            <p class="text-xl font-bold"><span id="vram-used" class="text-white">--</span> / <span id="vram-total" class="text-gray-300">--</span> GB</p>
                            <p class="text-xs text-gray-400 mt-2">System RAM</p>
                            <p class="text-xl font-bold"><span id="ram-used" class="text-white">--</span> / <span id="ram-total" class="text-gray-300">--</span> GB</p>
"""
content = re.sub(r'<p class="text-xs text-gray-400 mt-2">VRAM Usage</p>\s*<p class="text-xl font-bold"><span id="vram-used" class="text-white">--</span> / <span id="vram-total" class="text-gray-300">--</span> GB</p>', html_snippet, content)

# Update JS to fill sys ram
js_snippet = """
                    document.getElementById('gpu-power').innerText = data.gpu.power_w + " W";
                    document.getElementById('gpu-temp').innerText = data.gpu.temp_c + " °C";
                    if(data.gpu.sys_ram_used_gb !== undefined) {
                        document.getElementById('ram-used').innerText = data.gpu.sys_ram_used_gb;
                        document.getElementById('ram-total').innerText = data.gpu.sys_ram_total_gb;
                    }
"""
content = re.sub(r'document.getElementById\(\'gpu-power\'\).innerText = data.gpu.power_w \+ " W";\s*document.getElementById\(\'gpu-temp\'\).innerText = data.gpu.temp_c \+ " .C";', js_snippet, content)


with open(path, "w", encoding="utf-8") as f:
    f.write(content)
