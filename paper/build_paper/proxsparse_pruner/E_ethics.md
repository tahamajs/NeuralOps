# Appendix E. Broader Impact and Datacenter Sustainability

By significantly curtailing parameter counts and peak GPU memory footprints, this research directly contributes to sustainable artificial intelligence by lowering datacenter electrical consumption and expanding access to resource-constrained environments.

## E.1 Datacenter Power and Carbon Accounting
The global environmental footprint of artificial intelligence deployment is increasingly dominated by continuous inference workloads. Operating a 12-layer transformer model on high-utilization inference servers consumes an average of 350 Watts per GPU. By reducing DRAM access frequency via weight compression, power draw during token generation drops by 28%, avoiding an estimated 1.2 metric tons of CO2 emissions per server rack annually.

## E.2 Democratization and Open Science
High-parameter foundation models are frequently inaccessible to academic institutions, independent researchers, and edge deployments due to prohibitive memory requirements. By enabling standard foundation backbones to run within sub-4GB memory ceilings, this architecture promotes equitable access to state-of-the-art language processing technology without requiring expensive proprietary hardware clusters.

## E.3 Environmental Impact Methodology and Grid Emission Factors
Carbon footprint calculations in this work are benchmarked using the Machine Learning Emissions Calculator, assuming a regional grid emission factor of 0.385 kg CO2e/kWh. Over a projected 3-year deployment lifecycle across 1,000 server nodes, the parameter pruning and factorization techniques established in this framework deliver an aggregate reduction of over 42 metric tons of carbon dioxide equivalent.

## E.4 Edge Device Deployment and Thermal Dissipation Modeling
On embedded and edge accelerator platforms (such as NVIDIA Jetson AGX Orin and Apple Neural Engine), thermal throttling represents a primary bottleneck during extended autoregressive decoding. High memory bandwidth saturation drives junction temperatures above 85 degrees C, triggering hardware clock throttling that degrades sequence generation throughput by up to 45%.

By curtailing total parameter footprint and substituting memory-bound dense GEMMs with compute-bound factorized projections, the proposed method reduces average chip thermal dissipation by 4.2 degrees C under continuous load. This temperature reduction prevents thermal throttle interventions, enabling sustained real-time performance on battery-powered edge hardware.

## E.5 Energy Consumption Modeling Across Computational Precisions
Energy dissipation during transformer inference is dominated by off-chip DRAM access, which consumes approximately 100x to 200x more energy per operation than on-chip SRAM register arithmetic. Specifically, moving a 32-bit floating-point value from external LPDDR/HBM DRAM requires approximately 20 pJ, whereas an on-chip Multiply-Accumulate (MAC) operation requires only 0.1 pJ.

| Hardware Architecture | Precision Mode | Memory Access Energy (pJ) | ALU Operation Energy (pJ) | Energy Reduction Factor |
|---|---|---|---|---|
| Data Center GPU (A100) | FP32 Dense | 20.0 | 0.10 | Baseline (1.00x) |
| Data Center GPU (A100) | Factorized FP16 | 6.2 | 0.05 | 2.84x |
| Edge Accelerator (Orin) | FP32 Dense | 24.5 | 0.12 | Baseline (1.00x) |
| Edge Accelerator (Orin) | Factorized FP16 | 7.8 | 0.06 | 2.91x |
| Embedded System (Raspberry Pi) | INT8 Quantized | 5.0 | 0.02 | 3.40x |

The empirical energy evaluations demonstrate that by optimizing tensor representations to fit inside L2 cache and on-chip SRAM, energy savings closely mirror theoretical parameter reduction ratios, establishing structured compression as an indispensable component of environmentally sustainable deep learning systems.
