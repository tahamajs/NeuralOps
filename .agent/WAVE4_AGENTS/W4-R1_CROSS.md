# Cross-Agent Findings Logged by W4-R1 (Infrastructure & Release)

Per the instruction in `W4-R1.md` and `AGENTS.md` (Prime Directive / Isolation):
> "ANYTHING outside this list is OFF LIMITS. If you find a bug outside your Owned Paths, you MUST NOT fix it. Instead, append to: `.agent/WAVE4_AGENTS/W4-R1_CROSS.md`"

The following issues outside W4-R1 owned paths were identified during the pre-release verification audit (`scripts/wave4_verify.sh` Phase 5: Security Greps):

---

## 1. Subsystem: Surgery (`neuralops/surgery/model_surgery.py`)
- **Target File:** `neuralops/surgery/model_surgery.py:127`
- **Owner Agent:** W4-S1 (Surgery Remediation)
- **Violation:** PyTorch 2.6+ Security Policy / CVE Risk (`weights_only=False`)
- **Details:**
  ```python
  model = torch.load(model_name_or_path, map_location="cpu", weights_only=False)
  ```
- **Remediation Required:** Change to `weights_only=True` (or implement safe checkpoint loading / state_dict deserialization) to prevent arbitrary code execution on untrusted models.

---

## 2. Subsystem: Subsystem Integration (`neuralops/__init__.py`)
- **Target File:** `neuralops/__init__.py:512, 520`
- **Owner Agent:** W4-I1 (Integration / Package Entrypoints)
- **Violation:** AGENTS.md Rule 2.1 (`torch.cuda.amp` deprecated API)
- **Details:**
  ```python
  return torch.cuda.amp.autocast(**kwargs)
  GradScaler = getattr(torch.cuda.amp, "GradScaler", None)
  ```
- **Remediation Required:** Replace with modern unified `torch.amp.autocast("cuda", **kwargs)` and `torch.amp.GradScaler("cuda", ...)`.

---

## 3. Subsystem: Trainer (`neuralops/trainer/`)
- **Target Files:**
  - `neuralops/trainer/utils.py:182`
  - `neuralops/trainer/smart_gpu_trainer.py:351, 432`
  - `neuralops/trainer/custom_loop_trainer.py:92`
- **Owner Agent:** W4-T1 (Trainer Remediation)
- **Violation:** AGENTS.md Rule 2.1 (`torch.cuda.amp` deprecated API)
- **Details:**
  References `torch.cuda.amp.GradScaler`.
- **Remediation Required:** Replace with `torch.amp.GradScaler("cuda", enabled=...)`.
