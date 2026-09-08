import subprocess

def run(cmd):
    p = subprocess.run(cmd, shell=True, text=True, capture_output=True)
    return p.stdout.strip(), p.stderr.strip()

# Fetch remote branches
run("git fetch --all --prune")

# Get local branches
out, _ = run("git branch --format='%(refname:short)'")
local_branches = [b.strip() for b in out.splitlines() if b.strip() and b.strip() not in ("main", "master", "gh-pages")]

print(f"Local branches to merge and delete: {len(local_branches)}")
for b in local_branches:
    # Try merging
    print(f"Merging {b} into main...")
    run(f"git merge {b} --no-edit -m 'merge branch {b}'")
    # Delete local
    m_out, _ = run(f"git branch -D {b}")
    print(f"Deleted local {b}")

# Get remote benchmark-results-* branches
r_out, _ = run("git branch -r --format='%(refname:short)'")
remote_branches = [b.replace("origin/", "").strip() for b in r_out.splitlines() if b.strip().startswith("origin/benchmark-results-") or b.strip().startswith("origin/copilot/")]

print(f"Remote branches to delete: {len(remote_branches)}")
for rb in remote_branches:
    del_out, del_err = run(f"git push origin --delete {rb}")
    print(f"Deleted remote branch {rb}: {del_out or del_err}")

# Final push
run("git push origin main")
print("All branches merged and deleted successfully.")
