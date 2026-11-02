# Skill: Sharif HPC Cluster

Full guidance in [SKILL.md](SKILL.md).

Quick start:
1. `ssh hpc` (requires `~/.ssh/config` alias per SKILL.md §3.1)
2. Verify credit: `grep 'credit Quota' /etc/motd`
3. Write `run.sh` from `scripts/run.sh`, adjust queue per `qstat -Q`
4. `qsub run.sh`
5. `qstat -f <JOB_ID>` until `C`
6. Read `neuralops_gpu_verify.o<JOB_ID>`

See AGENTS.md §3.4 for the mandatory rules.
