from pathlib import Path
p = Path("Codes/framework/live_monitor.py")
content = p.read_text(encoding="utf-8")

old_text = '''        "active_run": {
            "run_id": "run_active",
            "name": "Phase 2 Quality Sweep (Fast)",
            "is_active": True,
            "module_count": 6,
            "completed": progress.get("completed", 0),
        },'''

new_text = '''        "active_run": {
            "run_id": "run_active",
            "name": "Phase 2 Quality Screening (1-Seed 1-Epoch)" if ((PROJECT_DIR / "results" / "run_quality_ep1").exists() or (PROJECT_DIR / "run_quality_ep1.log").exists()) else ("Phase 2 Quality Screening (1 Seed, 20 Modules)" if ((PROJECT_DIR / "results" / "run_quality_screening_fast").exists() or (PROJECT_DIR / "run_quality_screening_fast.log").exists()) else ("Phase 2 Quality Screening (20 Modules)" if ((PROJECT_DIR / "results" / "run_quality_screening").exists() or (PROJECT_DIR / "run_quality_screening.log").exists()) else "Phase 2 Quality Sweep (Fast)")),
            "is_active": True,
            "module_count": 20 if ((PROJECT_DIR / "results" / "run_quality_screening").exists() or (PROJECT_DIR / "run_quality_screening.log").exists() or (PROJECT_DIR / "results" / "run_quality_screening_fast").exists() or (PROJECT_DIR / "run_quality_screening_fast.log").exists() or (PROJECT_DIR / "results" / "run_quality_ep1").exists() or (PROJECT_DIR / "run_quality_ep1.log").exists()) else 6,
            "completed": progress.get("completed", 0),
        },'''

content = content.replace(old_text, new_text)
p.write_text(content, encoding="utf-8")
print("Patched _build_stats_response!")
