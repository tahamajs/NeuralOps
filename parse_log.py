with open(r"C:\Users\Hadi Schumacher\SVD_Project\trajectory_master.log", "r", encoding="utf-8", errors="ignore") as f:
    for line in f:
        if "SUCCESS:" in line or "NOTICE:" in line:
            print(line.strip())
