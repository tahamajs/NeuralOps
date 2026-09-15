import re

file_path = "/Users/tahamajs/Documents/uni/Project/Codes/framework/attentions/simple_ssm.py"
with open(file_path, "r", encoding="utf-8") as f:
    code = f.read()

# Make sure conv1d uses groups=1 or inner_dim doesn't cause bad allocation
# Line 154: groups=self.inner_dim
# Let's inspect line 452: x_ssm = self.conv1d(x_ssm.transpose(1, 2))[..., :L].transpose(1, 2)
# If x_ssm transpose is not contiguous, conv1d might fail with bad allocation
old_conv = "x_ssm = self.conv1d(x_ssm.transpose(1, 2))[..., :L].transpose(1, 2)"
new_conv = "x_ssm = self.conv1d(x_ssm.transpose(1, 2).contiguous())[..., :L].transpose(1, 2).contiguous()"

if old_conv in code:
    code = code.replace(old_conv, new_conv)
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(code)
    print("Successfully patched simple_ssm.py conv1d contiguous call")
else:
    print("Could not find exact old_conv pattern")

