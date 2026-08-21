with open("Codes/framework/live_monitor.py", "r") as f:
    content = f.read()

content = content.replace('WORKDIR.parent / "Docs"', 'WORKDIR.parent.parent / "Docs"')

with open("Codes/framework/live_monitor.py", "w") as f:
    f.write(content)
