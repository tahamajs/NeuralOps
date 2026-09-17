# Results Template

## 6. Experimental Setup (before results)

### 6.1 Models
Table with: layers, hidden, heads, params, attention type

### 6.2 Dataset
Table with: split sizes, class distribution, majority baseline

### 6.3 Training Protocol
Table with: optimizer, lr, steps, batch size, precision

### 6.4 Evaluation Metrics
Definitions of every metric

### 6.5 Hardware
GPU, CUDA, PyTorch, Transformers versions

## 7. Results

### 7.1 Main Results
Table 7.1: all backbones × all ranks
- Bold best accuracy per row
- Underline best Pareto point

### 7.2 Parameter Analysis
Table 7.2: params before/after per backbone

### 7.3 Efficiency Profile
Table 7.3: latency, VRAM, throughput

### 7.4 Ablations
- Table 7.4: LoRA rank
- Table 7.5: recovery steps
- Table 7.6: calibration size

### 7.7 Scaling with Sequence Length
Table 7.7: latency vs. seq len

### 7.9 Training Curves
Figure 7.1

## Rules

- Every table must have a caption above
- Every figure must have a caption below
- Every claim in text must reference a table/figure
- Use ± for std, report n for each row
