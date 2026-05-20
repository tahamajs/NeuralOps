# SmartWeightedEnsemble: A Task‑Aware Hybrid Ensemble with Attention and Dynamic Confidence Weighting

**Abstract**  
We present **SmartWeightedEnsemble**, a flexible framework for combining predictions from multiple pretrained models across a variety of tasks. The ensemble supports several strategies: simple averaging, performance‑based weighting, a novel **task‑aware attention mechanism** that learns to weight models based on their relevance to the target task, and a **hybrid method** that fuses the attention weights with input‑dependent confidence scores. At the core is `TaskAwareAttention`, which embeds tasks into a shared space, computes attention scores between a query task embedding and model‑specific task embeddings, and further biases the scores using a learnable task similarity matrix. The hybrid strategy adds a dynamic, per‑example confidence weighting computed from each model’s own predictive probabilities. The ensemble maintains a performance tracking matrix for historical weighting and can be updated online. This paper provides a rigorous mathematical formulation of each component, the implementation details, and a discussion of the design choices and potential extensions.

---

## 1. Introduction

In multi‑task and multi‑model settings, combining predictions from several models can yield better performance than any single model. The choice of combination strategy—simple averaging, weighted voting, or learned fusion—is crucial and often depends on the characteristics of the tasks and the models. The **SmartWeightedEnsemble** module provides a general‑purpose ensemble toolkit that integrates several advanced weighting techniques:

- **Simple average**: a baseline that assigns equal weight to all models.
- **Performance‑based weighting**: uses a moving average of past per‑task per‑model accuracy to assign higher weight to models that historically perform well on a given task.
- **Task‑aware attention**: a learned mechanism that uses task embeddings and a pairwise task similarity matrix to produce an attention distribution over models, conditioned on the target task.
- **Hybrid combination**: blends the task‑aware attention weights with a dynamic, input‑specific confidence score derived from the softmax probabilities of each model, achieving a balance between task‑level relevance and instance‑level uncertainty.

The ensemble is model‑agnostic: it can aggregate any collection of PyTorch models that output logits, making it suitable for use within the compression framework or as a standalone evaluation tool. In this paper, we describe the design and mathematics of the `TaskAwareAttention` module and the `SmartWeightedEnsemble` class, detail the various weighting strategies, and discuss the potential benefits and limitations.

---

## 2. Method

### 2.1 Task‑Aware Attention

The core learned fusion method is `TaskAwareAttention`. It assumes that each model in the ensemble is associated with a **prototype task** (e.g., the task it was fine‑tuned on), and that the target task is known at inference time. The attention module learns to weight each model’s predictions by measuring the similarity between the target task and the model’s prototype task.

**Task embeddings**: A learnable embedding matrix \(\mathbf{E} \in \mathbb{R}^{T \times d}\) stores a \(d\)-dimensional vector for each of the \(T\) possible tasks. Given a target task index \(t\) and a list of model‑prototype task indices \(\{m_i\}_{i=1}^{M}\), the module computes:

\[
\mathbf{q}_t = \mathbf{Q} \, \mathbf{e}_t \in \mathbb{R}^{d}, \qquad
\mathbf{k}_i = \mathbf{K} \, \mathbf{e}_{m_i} \in \mathbb{R}^{d},
\]

where \(\mathbf{Q}, \mathbf{K} \in \mathbb{R}^{d \times d}\) are learnable linear projections (without bias) that transform the task embeddings into query and key vectors. The raw attention score between the target task and the \(i\)-th model is the scaled dot product:

\[
s_i = \frac{\mathbf{q}_t^\top \mathbf{k}_i}{\sqrt{d}}.
\]

**Task similarity bias**: To further guide the attention, a learnable symmetric similarity matrix \(\mathbf{\Sigma} \in \mathbb{R}^{T \times T}\) is maintained, where \(\Sigma_{a,b}\) represents the learned affinity between tasks \(a\) and \(b\). The attention score is augmented with the corresponding entry:

\[
s_i' = s_i + \Sigma_{m_i, t}.
\]

The final attention weights are obtained by applying softmax:

\[
\alpha_i = \frac{\exp(s_i')}{\sum_{j=1}^{M} \exp(s_j')}.
\]

**Weighted combination**: The ensemble output logits \(\mathbf{L} \in \mathbb{R}^{B \times C}\) (batch size \(B\), classes \(C\)) are computed as the weighted sum of the individual model logits:

\[
\mathbf{L} = \sum_{i=1}^{M} \alpha_i \cdot \mathbf{L}^{(i)},
\]

where \(\mathbf{L}^{(i)}\) are the logits of model \(i\). The attention mechanism is fully differentiable and can be trained end‑to‑end if desired, though in the provided code it is used in a frozen or pre‑trained fashion.

### 2.2 SmartWeightedEnsemble

The `SmartWeightedEnsemble` class orchestrates the models and the fusion strategy. It maintains:

- A list of \(M\) PyTorch models.
- A **performance matrix** \(\mathbf{P} \in \mathbb{R}^{M \times T}\), initialised to 0.5 for all entries. \(P_{i,t}\) is a moving average of the accuracy (or other metric) of model \(i\) on task \(t\), updated via exponential smoothing:
  \[
  P_{i,t} \leftarrow \gamma \, P_{i,t} + (1-\gamma) \, \text{metric},
  \]
  where \(\gamma = 0.9\) by default, and the metric is computed on a validation set.
- An instance of `TaskAwareAttention`.

The main method `combine_logits` accepts pre‑computed logits from all models (of shape \((M, B, C)\)), the target task ID, and the input tokens, and returns the fused logits according to the chosen `method`.

The available strategies are:

**1. simple_average**:
\[
\mathbf{L} = \frac{1}{M} \sum_{i=1}^{M} \mathbf{L}^{(i)}.
\]

**2. weighted (performance‑based)**:
The weight for model \(i\) is proportional to its historical performance on the target task:
\[
w_i = \frac{P_{i,t}}{\sum_{j} P_{j,t}}, \quad
\mathbf{L} = \sum_i w_i \mathbf{L}^{(i)}.
\]

**3. task_aware_attention**:  
Uses the `TaskAwareAttention` module. The model‑prototype task IDs are assigned in a round‑robin fashion by default (e.g., model \(i\) is associated with task \(i \bmod T\)). This is a placeholder; in a real scenario, each model would be tagged with the task it was fine‑tuned on.

**4. hybrid**:  
Combines the task‑aware attention with a dynamic, input‑dependent confidence weighting. The dynamic weights \(d_i\) are computed by running each model on the specific input (in no‑grad mode) and averaging the maximum softmax probability over the batch:
\[
d_i = \frac{1}{B} \sum_{b=1}^{B} \max_{c} \operatorname{softmax}(\mathbf{L}^{(i)}_{b})_c.
\]
These dynamic weights are normalised to sum to 1. The final output is a convex combination of the attention‑weighted logits and the dynamically weighted logits:
\[
\mathbf{L}_{\text{hybrid}} = 0.6 \cdot \mathbf{L}_{\text{attn}} + 0.4 \cdot \mathbf{L}_{\text{dyn}}.
\]

### 2.3 Implementation Details

- `TaskAwareAttention` is a `nn.Module`. It contains:
  - `task_emb`: an `nn.Embedding(num_tasks, hidden_dim)`.
  - `q_proj`, `k_proj`: `nn.Linear(hidden_dim, hidden_dim, bias=False)`.
  - `sim_mat`: a learnable parameter of shape `(num_tasks, num_tasks)`, initialised to the identity matrix.
  The forward pass computes the weights as described and returns the weighted sum of logits.

- `SmartWeightedEnsemble` is a plain Python class (not `nn.Module`). It stores models, the performance matrix, and the attention module. The `update_performance` method updates the performance matrix using an exponential moving average. The `_get_dynamic_weights` method runs inference with `torch.no_grad()` to obtain confidence scores. The `combine_logits` method dispatches to the appropriate strategy.

- The script includes a placeholder `run_smart_ensemble` that would be the command‑line entry point, but it is not implemented.

- The ensemble assumes all models output logits of the same shape and that the number of classes is consistent across tasks. The model‑prototype task assignment in the attention and hybrid methods is currently a simplistic round‑robin; a production implementation would store actual task IDs per model.

---

## 3. Usage and Configuration

The ensemble can be used as follows:

```python
ensemble = SmartWeightedEnsemble(models, num_tasks=10, device=device)
# Optionally update performance on validation data
ensemble.update_performance(task_id, model_logits, labels, metric='accuracy')
# Combine logits at inference
fused_logits = ensemble.combine_logits(task_id, all_logits, input_ids, attention_mask, method='hybrid')
```

The `method` parameter selects the fusion strategy. The hybrid method requires the raw input to compute dynamic weights; other methods only need the pre‑computed logits. The ensemble is flexible and can be extended with additional weighting schemes.

---

## 4. Discussion

The SmartWeightedEnsemble provides a versatile toolkit for model combination. The task‑aware attention is a novel contribution: by embedding tasks and learning a similarity matrix, it can discover relationships between tasks (e.g., natural language inference and paraphrasing) and transfer that knowledge to weight models appropriately. The addition of the similarity bias allows the model to incorporate prior knowledge about task relatedness or to learn it from data.

The hybrid method further refines the weights by considering the model’s own uncertainty on the specific input. This dual‑level adaptivity—task‑level relevance and instance‑level confidence—can be particularly effective when models have varying strengths across different tasks and examples.

However, there are several limitations and design choices worth noting:

- **Model‑prototype assignment**: The round‑robin assignment in the current code is a placeholder and will not reflect true model‑task relationships. In practice, each model should be tagged with the task it was trained on, which may require storing metadata alongside the model.
- **Dynamic weight computation**: The `_get_dynamic_weights` method runs a full forward pass for every model on every input batch, which can be computationally expensive, especially for large ensembles. The results could be cached or the confidence could be approximated with a lightweight method.
- **Performance tracking**: The performance matrix is updated manually by calling `update_performance`. This requires a separate validation loop and careful handling of momentum. The moving average could be replaced by a Bayesian estimator or learned directly.
- **Attention training**: The attention module has learnable parameters but no training loop is provided. In a full system, these parameters could be trained to directly optimise ensemble accuracy on a held‑out set, or they could be fine‑tuned jointly with the models.
- **Fixed blend in hybrid**: The 60/40 split between attention and dynamic weighting is hardcoded; it could be made learnable or adaptively chosen based on the task or data.

Despite these limitations, the SmartWeightedEnsemble offers a solid foundation for building sophisticated model combination pipelines. Its modular design allows easy swapping of components and strategies, making it a valuable addition to the compression framework’s toolset.

---

## 5. Conclusion

We have presented the SmartWeightedEnsemble, an ensemble framework that integrates simple averaging, performance‑based weighting, task‑aware attention, and a hybrid confidence‑augmented method. The task‑aware attention mechanism learns to combine models by exploiting task embeddings and a similarity matrix, while the hybrid approach adds per‑example adaptivity. The ensemble is model‑agnostic and can be used with any collection of PyTorch models. Future work will implement proper model‑task mapping, efficient dynamic weighting, and end‑to‑end training of the attention parameters, further enhancing the ensemble’s adaptability and performance.The provided script defines `SmartWeightedEnsemble`, a modular framework for combining predictions from multiple models using four strategies: simple averaging, performance-based weighting, task-aware attention, and a hybrid blend with dynamic confidence. Below is a rigorous paper‑style explanation of its components, mathematics, and usage.

---

# SmartWeightedEnsemble: Task‑Aware and Confidence‑Driven Model Combination

**Abstract**  
We present `SmartWeightedEnsemble`, a flexible framework for fusing predictions from an ensemble of models across multiple tasks. It supports simple averaging, historical performance weighting, a learnable **task‑aware attention** mechanism that uses task embeddings and a pairwise similarity matrix to weight models, and a **hybrid** strategy that augments the attention with per‑input confidence scores derived from each model’s softmax probabilities. The ensemble maintains a performance history matrix for online updates and can be deployed without additional training. This paper describes the mathematical formulation of each weighting method, the implementation details, and the design of the task‑aware attention module.

## 1. Introduction

Combining multiple models can improve robustness and accuracy, but the optimal combination strategy often depends on the characteristics of both the tasks and the individual models. `SmartWeightedEnsemble` provides a unified interface for several fusion techniques:

- **Simple average**: equal weight to every model.
- **Performance‑based**: models are weighted by their historical accuracy on the target task.
- **Task‑aware attention**: a learnable module that embeds tasks and computes attention weights over models based on the similarity between the current task and the tasks the models were trained on, further biased by a learned task similarity matrix.
- **Hybrid**: blends the task‑aware attention weights with instance‑level confidence scores derived from the maximum softmax probability of each model.

The ensemble is model‑agnostic and can be used with any collection of PyTorch models that return logits. It supports online adaptation via performance tracking and can be extended to other fusion strategies.

## 2. Method

### 2.1 Preliminaries

Let there be \(M\) models, \(T\) possible tasks, and a batch of \(B\) input samples with \(C\) classes. Each model \(i\) produces logits \(\mathbf{L}^{(i)} \in \mathbb{R}^{B \times C}\). The ensemble computes a fused output \(\mathbf{L} \in \mathbb{R}^{B \times C}\). The `SmartWeightedEnsemble` stores a performance matrix \(\mathbf{P} \in \mathbb{R}^{M \times T}\) where \(P_{i,t}\) is a smoothed estimate of model \(i\)’s accuracy on task \(t\), updated via exponential moving average.

### 2.2 Simple Average

The baseline fusion assigns equal weight to all models:

\[
\mathbf{L}_{\text{avg}} = \frac{1}{M} \sum_{i=1}^{M} \mathbf{L}^{(i)}.
\]

### 2.3 Performance‑Based Weighting

Models are weighted by their historical performance on the target task \(t\):

\[
w_i = \frac{P_{i,t}}{\sum_{j=1}^{M} P_{j,t}}, \qquad
\mathbf{L}_{\text{perf}} = \sum_{i=1}^{M} w_i \mathbf{L}^{(i)}.
\]

The performance matrix is updated with a new accuracy score \(\text{acc}_i\) using an exponential moving average:  
\(P_{i,t} \leftarrow \gamma P_{i,t} + (1 - \gamma) \text{acc}_i\), with \(\gamma = 0.9\) by default.

### 2.4 Task‑Aware Attention

This method learns a dynamic weighting over models based on the target task and model‑task assignments. The core component is the `TaskAwareAttention` module.

**Task embeddings**: A learnable embedding matrix \(\mathbf{E} \in \mathbb{R}^{T \times d}\) maps each task to a \(d\)-dimensional vector (default \(d=64\)). Given the target task index \(t\) and a list of model‑prototype task indices \(\{m_i\}_{i=1}^{M}\) (which task each model was trained on), we compute:

\[
\mathbf{q}_t = \mathbf{Q} \, \mathbf{e}_t \in \mathbb{R}^{d}, \qquad
\mathbf{k}_i = \mathbf{K} \, \mathbf{e}_{m_i} \in \mathbb{R}^{d},
\]

where \(\mathbf{Q}, \mathbf{K} \in \mathbb{R}^{d \times d}\) are learnable linear projections (bias‑free).

**Attention scores**: The raw similarity between the target task and model \(i\)’s task is the scaled dot product:

\[
s_i = \frac{\mathbf{q}_t^\top \mathbf{k}_i}{\sqrt{d}}.
\]

A **learnable task similarity matrix** \(\mathbf{\Sigma} \in \mathbb{R}^{T \times T}\) (initialised to \(\mathbf{I}\)) adds a bias:

\[
s'_i = s_i + \Sigma_{m_i, t}.
\]

The attention weights are a softmax over these scores:

\[
\alpha_i = \frac{\exp(s'_i)}{\sum_{j=1}^{M} \exp(s'_j)}.
\]

**Fusion**: The ensemble output is the weighted sum of logits:

\[
\mathbf{L}_{\text{attn}} = \sum_{i=1}^{M} \alpha_i \mathbf{L}^{(i)}.
\]

The parameters of the attention module (\(\mathbf{E}, \mathbf{Q}, \mathbf{K}, \mathbf{\Sigma}\)) can be trained end‑to‑end on a validation set, but the provided implementation uses them as pre‑trained or frozen.

### 2.5 Hybrid Strategy

The hybrid method combines the task‑aware attention with per‑input confidence scores. For each model \(i\), an input‑dependent confidence \(d_i\) is computed by running the model on the batch (without gradient) and averaging the maximum softmax probability:

\[
d_i = \frac{1}{B} \sum_{b=1}^{B} \max_{c} \, \operatorname{softmax}(\mathbf{L}^{(i)}_{b})_c.
\]

These confidences are normalised to sum to 1: \(\tilde{d}_i = d_i / \sum_j d_j\). The dynamic weighted logits are \(\mathbf{L}_{\text{dyn}} = \sum_i \tilde{d}_i \mathbf{L}^{(i)}\). The final output is a convex combination with a fixed mixing ratio (60% attention, 40% dynamic):

\[
\mathbf{L}_{\text{hybrid}} = 0.6 \, \mathbf{L}_{\text{attn}} + 0.4 \, \mathbf{L}_{\text{dyn}}.
\]

## 3. Implementation

- **`TaskAwareAttention`** (`nn.Module`): Contains `task_emb`, `q_proj`, `k_proj`, and `sim_mat`. Forward takes `task_id`, `all_logits` (tensor of shape \((M, B, C)\)), and `model_task_ids` (list of length \(M\)). Returns fused logits of shape \((B, C)\).

- **`SmartWeightedEnsemble`**: A plain class holding the list of models, the performance matrix, and an instance of `TaskAwareAttention`. It provides:
  - `update_performance(task_id, model_outputs, target, metric)`: updates the performance matrix.
  - `_get_dynamic_weights(task_id, input_ids, attention_mask)`: returns a normalised confidence vector by running each model.
  - `combine_logits(task_id, all_logits, input_ids, attention_mask, method)`: dispatches to the selected fusion method and returns fused logits.

The script includes a placeholder `run_smart_ensemble` function, intended as a command‑line entry point.

## 4. Usage Example

```python
ensemble = SmartWeightedEnsemble(models=[model_A, model_B], num_tasks=3, device=device)
# Optionally update performance on a validation set
ensemble.update_performance(task_id=0, model_outputs=[logits_A, logits_B], target=labels, metric='accuracy')
# Fuse logits at inference
fused = ensemble.combine_logits(task_id=0, all_logits=torch.stack([logits_A, logits_B]),
                                input_ids=input_ids, attention_mask=mask, method='hybrid')
```

The `method` argument can be `"simple_average"`, `"weighted"`, `"task_aware_attention"`, or `"hybrid"`. The hybrid and dynamic methods require the original `input_ids` and `attention_mask` to compute confidences.

## 5. Discussion and Future Work

The ensemble provides a flexible toolkit, but several aspects can be enhanced:

- **Model‑task assignment**: Currently uses a round‑robin placeholder. A real system should store the actual fine‑tuning task for each model.
- **Attention training**: The attention parameters could be optimised jointly with the ensemble’s final loss to learn task relationships and optimal fusion weights.
- **Hybrid ratio**: The 60/40 split is fixed; a learnable or adaptive coefficient would be more robust.
- **Dynamic confidence efficiency**: Running all models on every batch is expensive; caching or approximating confidence with a lightweight predictor could improve throughput.

`SmartWeightedEnsemble` serves as a solid foundation for research on multi‑model fusion and can be easily extended with additional weighting strategies.