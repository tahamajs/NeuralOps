import os

def insert_after(filepath, target, insertion):
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()
    if target in content and insertion not in content:
        content = content.replace(target, target + "\n" + insertion)
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(content)

base = "D:/SVD_Project/Codes/framework/compression_engine/"

# Fix hybrid_svd_layer
insert_after(
    base + "hybrid_svd_layer.py",
    '@COMPRESSOR_REGISTRY.register("hybrid_svd")',
    '@COMPRESSOR_REGISTRY.register("hybrid_svd_layer")'
)

# Fix kronecker_compressor
insert_after(
    base + "kronecker_compressor.py",
    '@COMPRESSOR_REGISTRY.register("kronecker")',
    '@COMPRESSOR_REGISTRY.register("kronecker_linear")'
)

# Fix hashed_net_compressor (already had it but just in case)
insert_after(
    base + "hashed_net_compressor.py",
    '@COMPRESSOR_REGISTRY.register("hashed_net")',
    '@COMPRESSOR_REGISTRY.register("hashed_linear")'
)

print("Registry names patched.")
