import glob

note = "> [!NOTE]\n> **Live Empirical Trajectories:** As part of this comprehensive review, we are actively executing a 120-run GPU benchmark grid. You will find the **real-time training loss charts, hyperparameters, and GLUE metrics** dynamically injected at the absolute bottom of this article. These metrics synchronize directly from the GPU orchestrator every 60 seconds.\n\n"

for f in glob.glob("Docs/blog/*.md"):
    if "README.md" in f:
        continue
    with open(f, 'r') as file:
        content = file.read()
    
    if "> **Live Empirical Trajectories" not in content:
        # Find the first heading or the top of the file
        new_content = note + content
        with open(f, 'w') as file:
            file.write(new_content)
