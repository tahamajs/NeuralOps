from pathlib import Path
p = Path('D:/SVD_Project/Codes/framework/live_monitor.py').resolve().parent.parent.parent
print('PROJECT_DIR:', p)
print('ep1 log:', (p / 'run_quality_ep1.log').exists())
print('fast log:', (p / 'run_quality_screening_fast.log').exists())
