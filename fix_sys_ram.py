path = r"C:\Users\Hadi Schumacher\SVD_Project\Codes\framework\live_monitor.py"
with open(path, "r", encoding="utf-8") as f:
    content = f.read()

# Fix get_gpu_stats to include psutil sys ram
old = '''                  "temp_c": float(parts[3]) if len(parts) > 3 else 0,
                  "power_w": float(parts[4]) if len(parts) > 4 else 0,
              }
      except Exception:
          pass
      return {"gpu_util": 0, "mem_used_mb": 0, "mem_total_mb": 8188, "temp_c": 0, "power_w": 0}'''

new = '''                  "temp_c": float(parts[3]) if len(parts) > 3 else 0,
                  "power_w": float(parts[4]) if len(parts) > 4 else 0,
                  "sys_ram_used_gb": round(psutil.virtual_memory().used / 1024**3, 1),
                  "sys_ram_total_gb": round(psutil.virtual_memory().total / 1024**3, 1),
              }
      except Exception:
          pass
      return {"gpu_util": 0, "mem_used_mb": 0, "mem_total_mb": 8188, "temp_c": 0, "power_w": 0, "sys_ram_used_gb": round(psutil.virtual_memory().used / 1024**3, 1), "sys_ram_total_gb": round(psutil.virtual_memory().total / 1024**3, 1)}'''

content = content.replace(old, new)
with open(path, "w", encoding="utf-8") as f:
    f.write(content)
print("Done patching sys_ram into get_gpu_stats")
