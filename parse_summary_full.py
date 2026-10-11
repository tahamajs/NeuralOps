with open('neuralops/compression_engine/pruning/mamba_shedder.py') as f:
    lines = f.readlines()
for i in range(655, 670):
    print(f"{i+1}: {repr(lines[i])}")
