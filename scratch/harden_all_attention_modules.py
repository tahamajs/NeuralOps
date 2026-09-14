import os
import re
import sys

ATTENTIONS_DIR = "/Users/tahamajs/Documents/uni/Project/Codes/framework/attentions"

def harden_file(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()

    orig_content = content
    modified = False

    # 1. Fix 3D head_mask reshape bug:
    # Pattern: elif head_mask.dim() == 3:\n    head_mask = head_mask.unsqueeze(1)
    p_headmask = re.compile(r'(elif\s+head_mask\.dim\(\)\s*==\s*3\s*:\s*\n\s*head_mask\s*=\s*head_mask\.unsqueeze\()1(\))')
    if p_headmask.search(content):
        content = p_headmask.sub(r'\g<1>-1\g<2>', content)
        modified = True

    # 2. Fix cumulative is_nested_encoder in copy_weights:
    # Pattern: self.is_nested_encoder = bool(is_nested or self.is_nested_encoder)
    if "self.is_nested_encoder = bool(is_nested or self.is_nested_encoder)" in content:
        content = content.replace(
            "self.is_nested_encoder = bool(is_nested or self.is_nested_encoder)",
            "self.is_nested_encoder = bool(is_nested)"
        )
        modified = True

    # 3. Fix output_scale before out_proj:
    # Pattern:
    # context = context * self.output_scale
    # if self.is_nested_encoder:
    #     output = context
    # else:
    #     output = self.out_proj(context)
    pattern_scale = re.compile(
        r'(\s+)context\s*=\s*context\s*\*\s*self\.output_scale\s*\n'
        r'(\s+)if\s+self\.is_nested_encoder\s*:\s*\n'
        r'(\s+)output\s*=\s*context\s*\n'
        r'(\s+)else\s*:\s*\n'
        r'(\s+)output\s*=\s*self\.out_proj\(context\)'
    )
    if pattern_scale.search(content):
        content = pattern_scale.sub(
            r'\1if self.is_nested_encoder:\n'
            r'\3output = context * self.output_scale\n'
            r'\4else:\n'
            r'\5output = self.out_proj(context) * self.output_scale',
            content
        )
        modified = True

    # Pattern:
    # context = context * self.output_scale
    # output = self.out_proj(context)
    pattern_scale2 = re.compile(
        r'(\s+)context\s*=\s*context\s*\*\s*self\.output_scale\s*\n'
        r'(\s+)output\s*=\s*self\.out_proj\(context\)'
    )
    if pattern_scale2.search(content):
        content = pattern_scale2.sub(
            r'\2output = self.out_proj(context) * self.output_scale',
            content
        )
        modified = True

    # 4. Fix is_decoder in ("gpt2", "qwen", "qwen2", "llama")
    # Replace with any(str(mt).lower().startswith(p) for p in ("gpt", "qwen", "llama", "mistral", "gemma", "phi", "opt", "bloom", "falcon", "olmo", "mpt", "stablelm"))
    if 'in ("gpt2", "qwen", "qwen2", "llama")' in content:
        content = content.replace(
            'in ("gpt2", "qwen", "qwen2", "llama")',
            'in ("gpt2", "qwen", "qwen2", "llama") or any(str(getattr(cfg if "cfg" in locals() else (config if "config" in locals() else None), "model_type", "")).lower().startswith(p) for p in ("gpt", "qwen", "llama", "mistral", "gemma", "phi", "opt", "bloom", "falcon", "olmo", "mpt", "stablelm"))'
        )
        modified = True

    # 5. Fix LoRALinear / FactorisedLinear device & dtype mutation in forward:
    # If self.weight.device != x.device or self.weight.dtype != x.dtype: self.to(device=x.device, dtype=x.dtype)
    p_dev_dtype = re.compile(
        r'if\s+(self\.(?:weight|down\.weight)\.device\s*!=\s*x\.device)\s+or\s+self\.(?:weight|down\.weight)\.dtype\s*!=\s*x\.dtype\s*:\s*\n\s*self\.to\(device=x\.device,\s*dtype=x\.dtype\)'
    )
    if p_dev_dtype.search(content):
        content = p_dev_dtype.sub(r'if \1:\n            self.to(device=x.device)', content)
        modified = True

    if modified and content != orig_content:
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(content)
        return True
    return False

def main():
    print(f"Scanning attention modules in {ATTENTIONS_DIR}...")
    files = [os.path.join(ATTENTIONS_DIR, f) for f in os.listdir(ATTENTIONS_DIR) if f.endswith(".py")]
    modified_files = []
    for filepath in files:
        try:
            if harden_file(filepath):
                modified_files.append(os.path.basename(filepath))
        except Exception as e:
            print(f"Error processing {filepath}: {e}")

    print(f"\nSuccessfully hardened {len(modified_files)} attention modules:")
    for f in modified_files:
        print(f"  - {f}")

if __name__ == "__main__":
    main()
