# 🔥 Phase 3 – Production‑Scale Deployment Blueprint

**Timeline:** 4 weeks  
**Prerequisites:** Phase 2 fully functional (experiment tracking, sweeps, model registry, cluster dashboard)  
**Goal:** A hardened, multi‑tenant, auto‑scaling, fault‑tolerant platform ready for 10‑100 researchers across dozens of GPU nodes.

---

## 📐 High‑Level Production Architecture

```
┌──────────────────────────────────────────────────────────────────────────────────────┐
│                           GIT REPOSITORY (Single Source of Truth)                      │
│  ┌─────────────┐  ┌─────────────────┐  ┌──────────────────┐  ┌───────────────────┐   │
│  │ App Source   │  │ Helm Charts     │  │ K8s Manifests    │  │ Dockerfiles       │   │
│  │ (React+API)  │  │ (values/prod)   │  │ (Kueue,Ray,etc.) │  │ (multi‑stage)     │   │
│  └─────────────┘  └─────────────────┘  └──────────────────┘  └───────────────────┘   │
└──────────────────────────────────────┬───────────────────────────────────────────────┘
                                       │ Git Push
                                       ▼
┌──────────────────────────────────────────────────────────────────────────────────────┐
│                            CI/CD PIPELINE (GitHub Actions)                             │
│  ┌───────────────────┐  ┌──────────────────┐  ┌───────────────────────────────────┐  │
│  │ Lint & Test        │  │ Build & Push      │  │ ArgoCD Sync                       │  │
│  │ (pytest, eslint)   │  │ (Docker + Kaniko) │  │ (ApplicationSet auto‑deploy)       │  │
│  └───────────────────┘  └──────────────────┘  └───────────────────────────────────┘  │
└──────────────────────────────────────┬───────────────────────────────────────────────┘
                                       │
                                       ▼
┌──────────────────────────────────────────────────────────────────────────────────────┐
│                      KUBERNETES CLUSTER (1.31+ with GPU nodes)                         │
│                                                                                        │
│  ┌──────────────────────────────────────────────────────────────────────────────────┐ │
│  │                         CONTROL PLANE SERVICES                                     │ │
│  │                                                                                    │ │
│  │  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐  │ │
│  │  │ Kueue    │ │ KubeRay  │ │ ArgoCD   │ │ Cert-    │ │ Ingress  │ │ Karpenter│  │ │
│  │  │ (Queue)  │ │ Operator │ │ (GitOps) │ │ Manager  │ │ Nginx    │ │ (Scale)  │  │ │
│  │  └──────────┘ └──────────┘ └──────────┘ └──────────┘ └──────────┘ └──────────┘  │ │
│  └──────────────────────────────────────────────────────────────────────────────────┘ │
│                                                                                        │
│  ┌──────────────────────────────────────────────────────────────────────────────────┐ │
│  │                           APPLICATION NAMESPACES (per‑tenant)                       │ │
│  │                                                                                    │ │
│  │  ┌──────────────────────┐  ┌──────────────────────┐  ┌──────────────────────┐    │ │
│  │  │ tenant: team‑nlp      │  │ tenant: team‑vision   │  │ tenant: team‑speech   │    │ │
│  │  │  ├─ django‑api        │  │  ├─ django‑api        │  │  ├─ django‑api        │    │ │
│  │  │  ├─ celery‑worker     │  │  ├─ celery‑worker     │  │  ├─ celery‑worker     │    │ │
│  │  │  ├─ react‑frontend    │  │  ├─ react‑frontend    │  │  ├─ react‑frontend    │    │ │
│  │  │  ├─ kueue‑localqueue  │  │  ├─ kueue‑localqueue  │  │  ├─ kueue‑localqueue  │    │ │
│  │  │  └─ resourceQuota     │  │  └─ resourceQuota     │  │  └─ resourceQuota     │    │ │
│  │  └──────────────────────┘  └──────────────────────┘  └──────────────────────┘    │ │
│  └──────────────────────────────────────────────────────────────────────────────────┘ │
│                                                                                        │
│  ┌──────────────────────────────────────────────────────────────────────────────────┐ │
│  │                       GPU WORKER POOL (Auto‑scaled by Karpenter)                   │ │
│  │                                                                                    │ │
│  │  ┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐              │ │
│  │  │ GPU    │ │ GPU    │ │ GPU    │ │ GPU    │ │ GPU    │ │ GPU    │   …           │ │
│  │  │ Node   │ │ Node   │ │ Node   │ │ Node   │ │ Node   │ │ Node   │              │ │
│  │  │ A100   │ │ A100   │ │ H100   │ │ H100   │ │ A100   │ │ H100   │              │ │
│  │  │ (spot) │ │ (on‑dm)│ │ (spot) │ │ (on‑dm)│ │ (spot) │ │ (on‑dm)│              │ │
│  │  └────────┘ └────────┘ └────────┘ └────────┘ └────────┘ └────────┘              │ │
│  │                    ↕ RDMA / InfiniBand / RoCE v2                                    │ │
│  └──────────────────────────────────────────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────────────────────────────────────┘
                                       │
                                       ▼
┌──────────────────────────────────────────────────────────────────────────────────────┐
│                           STORAGE & OBSERVABILITY LAYER                                │
│                                                                                        │
│  ┌────────────┐ ┌────────────┐ ┌──────────────┐ ┌──────────┐ ┌──────────────────┐   │
│  │ PostgreSQL │ │ Redis      │ │ MinIO / S3   │ │ NFS/Lustre│ │ Prometheus Stack │   │
│  │ (HA)       │ │ (Sentinel) │ │ (artifacts)  │ │ (checkpts)│ │ + Grafana        │   │
│  └────────────┘ └────────────┘ └──────────────┘ └──────────┘ └──────────────────┘   │
└──────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 1. ⛵ Helm Charts – GitOps Deployment

### 1.1 Helm Chart Structure

```
charts/efficient-transformer/
├── Chart.yaml                      # name: efficient-transformer, version: 1.0.0
├── values.yaml                     # default values (dev)
├── values-prod.yaml                # production overrides
├── values-staging.yaml             # staging overrides
├── templates/
│   ├── _helpers.tpl                # common labels, names
│   ├── namespace.yaml              # tenant namespace
│   ├── resourcequota.yaml          # CPU/GPU/memory limits per tenant
│   ├── networkpolicy.yaml          # namespace isolation
│   ├── secret.yaml                 # DB creds, S3 keys, Slack webhook
│   ├── configmap.yaml              # Django settings, Celery config
│   ├── deployment-api.yaml         # Django API
│   ├── deployment-worker.yaml      # Celery worker
│   ├── deployment-frontend.yaml    # React frontend (Nginx)
│   ├── deployment-beat.yaml        # Celery beat
│   ├── service.yaml                # ClusterIP services
│   ├── ingress.yaml                # Ingress with TLS
│   ├── hpa.yaml                    # Horizontal Pod Autoscaler
│   ├── pvc.yaml                    # Persistent volumes (checkpoints, media)
│   ├── servicemonitor.yaml         # Prometheus scraping
│   ├── kueue-localqueue.yaml       # Kueue tenant queue
│   └── raycluster-job.yaml         # RayCluster template (optional)
└── README.md
```

### 1.2 Key Values (values.yaml)

```yaml
# ── Global ──
global:
  tenant: "team-nlp"
  environment: "production"
  domain: "ml.internal.company.com"

# ── Django API ──
api:
  replicas: 3
  image: "registry.company.com/efficient-transformer-api:latest"
  resources:
    requests: {cpu: "500m", memory: "1Gi"}
    limits:   {cpu: "2", memory: "4Gi"}
  env:
    DJANGO_DEBUG: "false"
    DJANGO_SECRET_KEY: "${SECRET_KEY}"
    DB_HOST: "postgres-ha.database.svc"
    REDIS_URL: "redis://redis-sentinel:26379/0"
    S3_ENDPOINT: "https://minio.storage.svc"
    CELERY_BROKER_URL: "redis://redis-sentinel:26379/1"
  autoscaling:
    enabled: true
    minReplicas: 2
    maxReplicas: 10
    targetCPUUtilization: 70

# ── Celery Worker ──
worker:
  replicas: 4
  queues: ["gpu_training", "gpu_eval", "cpu_default"]
  resources:
    requests: {cpu: "1", memory: "4Gi"}
    limits:   {cpu: "4", memory: "16Gi"}
  # GPU workers are separate – scheduled via Kueue + Ray
  gpuWorker:
    enabled: true
    replicas: 2
    nodeSelector:
      nvidia.com/gpu.product: "NVIDIA-A100-SXM4-40GB"
    tolerations:
      - key: "nvidia.com/gpu"
        operator: "Exists"
        effect: "NoSchedule"
    resources:
      limits: {nvidia.com/gpu: 1}

# ── React Frontend ──
frontend:
  replicas: 2
  image: "registry.company.com/efficient-transformer-frontend:latest"
  ingress:
    enabled: true
    host: "ml.internal.company.com"
    tls:
      enabled: true
      secretName: "ml-tls-cert"

# ── Kueue Tenant Queue ──
kueue:
  clusterQueue: "gpu-cluster-queue"
  localQueue: "team-nlp-queue"
  flavorGPU:
    - name: "a100-40gb"
      resources: [{name: "cpu", nominalQuota: 64}, {name: "nvidia.com/gpu", nominalQuota: 8}]
    - name: "h100-80gb"
      resources: [{name: "cpu", nominalQuota: 32}, {name: "nvidia.com/gpu", nominalQuota: 4}]
  priorityClasses:
    - name: "high-priority"
      value: 1000
    - name: "medium-priority"
      value: 100
    - name: "low-priority"
      value: 10

# ── Monitoring ──
monitoring:
  servicemonitor:
    enabled: true
    interval: "30s"
  alerts:
    gpuOOM:
      enabled: true
      threshold: 95   # alert when GPU memory > 95%
    jobFailed:
      enabled: true
    nodeDown:
      enabled: true
      severity: "critical"

# ── Storage ──
persistence:
  checkpoints:
    storageClass: "nfs-csi"
    size: 2Ti
    accessMode: ReadWriteMany
  media:
    storageClass: "nfs-csi"
    size: 500Gi
    accessMode: ReadWriteMany
```

### 1.3 ArgoCD ApplicationSet (Auto‑Deploy per Tenant)

```yaml
apiVersion: argoproj.io/v1alpha1
kind: ApplicationSet
metadata:
  name: efficient-transformer-tenants
spec:
  generators:
    - list:
        elements:
          - tenant: team-nlp
            domain: nlp.ml.company.com
            gpuQuota: 8
          - tenant: team-vision
            domain: vision.ml.company.com
            gpuQuota: 16
          - tenant: team-speech
            domain: speech.ml.company.com
            gpuQuota: 4
  template:
    metadata:
      name: "et-{{tenant}}"
    spec:
      project: default
      source:
        repoURL: "https://github.com/company/efficient-transformer"
        targetRevision: main
        path: charts/efficient-transformer
        helm:
          values: |
            global:
              tenant: "{{tenant}}"
              domain: "{{domain}}"
            kueue:
              localQueue: "{{tenant}}-queue"
              flavorGPU:
                - name: a100-40gb
                  resources:
                    - name: nvidia.com/gpu
                      nominalQuota: {{gpuQuota}}
      destination:
        server: "https://kubernetes.default.svc"
        namespace: "{{tenant}}"
      syncPolicy:
        automated:
          prune: true
          selfHeal: true
          allowEmpty: true
        syncOptions:
          - CreateNamespace=true
```

---

## 2. 🎛️ GPU Scheduling with Kueue

### 2.1 Kueue Architecture

Kueue introduces a **queueing layer** between the API server and the scheduler. Jobs are submitted to a `LocalQueue` (per tenant/team), which drains into a `ClusterQueue` that enforces quotas and fair‑sharing.

```
API Server → LocalQueue (tenant‑nlp) ──┐
                                        ├──→ ClusterQueue (gpu‑pool) → Admission → Pod Creation
API Server → LocalQueue (tenant‑vis) ──┘
```

### 2.2 Kueue Resources (Production Manifests)

**ClusterQueue – defines the global GPU pool:**

```yaml
apiVersion: kueue.x-k8s.io/v1beta1
kind: ClusterQueue
metadata:
  name: gpu-cluster-queue
spec:
  namespaceSelector: {}  # all namespaces
  resourceGroups:
    - coveredResources: ["cpu", "memory", "nvidia.com/gpu"]
      flavors:
        - name: a100-40gb
          resources:
            - name: "cpu"
              nominalQuota: 256
            - name: "memory"
              nominalQuota: 1024Gi
            - name: "nvidia.com/gpu"
              nominalQuota: 32
        - name: h100-80gb
          resources:
            - name: "cpu"
              nominalQuota: 128
            - name: "memory"
              nominalQuota: 512Gi
            - name: "nvidia.com/gpu"
              nominalQuota: 16
  fairSharing:
    weight: 1
  preemption:
    reclaimWithinCohort: "LowerPriority"
    withinClusterQueue: "LowerPriority"
  flavorFungibility:
    whenCanBorrow: "Borrow"
    whenCanPreempt: "TryNextFlavor"
```

**LocalQueue – per tenant:**

```yaml
apiVersion: kueue.x-k8s.io/v1beta1
kind: LocalQueue
metadata:
  name: team-nlp-queue
  namespace: team-nlp
spec:
  clusterQueue: gpu-cluster-queue
```

**WorkloadPriorityClass – determines preemption order:**

```yaml
apiVersion: kueue.x-k8s.io/v1beta1
kind: WorkloadPriorityClass
metadata:
  name: prod-priority
spec:
  priority: 1000
  description: "Production training jobs (preempt others)"
---
apiVersion: kueue.x-k8s.io/v1beta1
kind: WorkloadPriorityClass
metadata:
  name: research-priority
spec:
  priority: 500
  description: "Research experiments"
---
apiVersion: kueue.x-k8s.io/v1beta1
kind: WorkloadPriorityClass
metadata:
  name: background-priority
spec:
  priority: 10
  description: "Background sweeps / overnight jobs"
```

### 2.3 Submitting a Training Job via Kueue

```yaml
apiVersion: batch/v1
kind: Job
metadata:
  name: train-mrpc-aurora
  namespace: team-nlp
  labels:
    kueue.x-k8s.io/queue-name: team-nlp-queue         # ← targets local queue
    kueue.x-k8s.io/priority-class: research-priority   # ← preemption behaviour
spec:
  suspend: true   # Kueue will unsuspend when admitted
  template:
    spec:
      nodeSelector:
        nvidia.com/gpu.product: "NVIDIA-A100-SXM4-40GB"
      containers:
        - name: trainer
          image: registry.company.com/efficient-transformer-worker:latest
          command: ["celery", "-A", "django_api", "worker", "-Q", "gpu_training"]
          resources:
            limits:
              nvidia.com/gpu: 2   # request 2 GPUs
              cpu: 16
              memory: 64Gi
      restartPolicy: Never
```

When the user clicks "Launch" in the React UI, the Django backend creates this `Job` manifest, applies it to the cluster (via `kubernetes-client` Python library), and Kueue handles the rest.

### 2.4 NVIDIA MIG Partitioning (A100/H100)

For smaller experiments, the A100 can be partitioned into up to 7 MIG instances:

```yaml
# GPU node with MIG configured
nodeSelector:
  nvidia.com/gpu.product: "NVIDIA-A100-SXM4-40GB"
resources:
  limits:
    nvidia.com/mig-3g.20gb: 1   # one 3‑slice MIG instance (20 GB)
```

Kueue flavors can distinguish between full GPUs and MIG slices.

### 2.5 NVIDIA Run:ai Integration (Alternative to Kueue)

Run:ai is the commercial alternative. Its **KAI Scheduler** integrates directly with RayCluster for gang‑scheduling. It supports:

| Feature | Kueue | Run:ai |
|---------|-------|--------|
| GPU fractions | via MIG only | Dynamic GPU fractions (any %) |
| Gang scheduling | ✅ (via RayCluster) | ✅ (native) |
| Preemption | ✅ | ✅ |
| GUI | ❌ (CLI only) | ✅ (rich web UI) |
| SSO integration | ❌ | ✅ (LDAP, SAML) |
| Cost tracking | ❌ | ✅ (per‑project billing) |

---

## 3. 🚀 Ray Train – Multi‑GPU Distributed Training

### 3.1 When to Use Ray Train vs. Celery

| Scenario | Tool | Why |
|----------|------|-----|
| Single‑GPU fine‑tuning | Celery worker | Simple, lightweight |
| Multi‑GPU (2‑8) on same node | PyTorch DDP via Celery | NCCL works within node |
| Multi‑node (8+ GPUs) | Ray Train | Handles networking, fault recovery |
| Hyperparameter sweeps (parallel) | Ray Tune | Built‑in Optuna integration |
| Large model (FSDP / ZeRO) | Ray Train + FSDP | Sharding across dozens of GPUs |

### 3.2 RayCluster CRD Deployment

The KubeRay operator manages the lifecycle of Ray clusters via a `RayCluster` CRD.

**Minimal RayCluster manifest:**

```yaml
apiVersion: ray.io/v1
kind: RayCluster
metadata:
  name: raycluster-train
  namespace: team-nlp
spec:
  rayVersion: "2.9.0"
  headGroupSpec:
    rayStartParams:
      dashboard-host: "0.0.0.0"
    template:
      spec:
        containers:
          - name: ray-head
            image: "rayproject/ray:2.9.0-py310-gpu"
            ports:
              - containerPort: 6379   # GCS
              - containerPort: 8265   # Dashboard
            resources:
              limits: {cpu: 4, memory: 16Gi}
  workerGroupSpecs:
    - groupName: gpu-workers
      replicas: 4
      minReplicas: 2
      maxReplicas: 8
      rayStartParams:
        num-gpus: "1"
      template:
        spec:
          nodeSelector:
            nvidia.com/gpu.product: "NVIDIA-A100-SXM4-40GB"
          containers:
            - name: ray-worker
              image: "rayproject/ray:2.9.0-py310-gpu"
              resources:
                limits: {nvidia.com/gpu: 1, cpu: 8, memory: 64Gi}
                requests: {nvidia.com/gpu: 1, cpu: 4, memory: 32Gi}
```

### 3.3 Ray Train Integration in the Runner

Expand `api/runners.py` to detect the `distributed` flag:

```python
def run_training(experiment, stop_event=None):
    config = experiment.config
    if config.get("distributed") and config.get("num_gpus", 1) > 1:
        return _run_distributed_training(experiment, stop_event)
    return _run_single_gpu_training(experiment, stop_event)

def _run_distributed_training(experiment, stop_event):
    """Launch via Ray Train for multi‑GPU."""
    import ray
    from ray.train.torch import TorchTrainer
    from ray.train import ScalingConfig, RunConfig, CheckpointConfig

    ray.init(address="auto")   # connect to existing RayCluster

    def train_loop_per_worker(config):
        # Same logic as single‑GPU but wrapped for Ray
        model, tokenizer = load_model(...)
        # ... training loop with Ray‑managed checkpointing
        ray.train.report(metrics={"loss": loss, "accuracy": acc})

    trainer = TorchTrainer(
        train_loop_per_worker=train_loop_per_worker,
        scaling_config=ScalingConfig(
            num_workers=config.get("num_gpus", 4),
            use_gpu=True,
            resources_per_worker={"GPU": 1},
            trainer_resources={"CPU": 2}
        ),
        run_config=RunConfig(
            name=experiment.name,
            storage_path="s3://models/checkpoints/",
            checkpoint_config=CheckpointConfig(
                num_to_keep=5,
                checkpoint_score_attribute="eval_loss",
                checkpoint_score_order="min"
            ),
            failure_config=FailureConfig(
                max_failures=3,
                fail_fast=False
            )
        )
    )
    result = trainer.fit()
    return result
```

### 3.4 NCCL Optimization for Multi‑Node

Add to `values.yaml`:

```yaml
worker:
  env:
    NCCL_DEBUG: "INFO"
    NCCL_SOCKET_IFNAME: "eth0"
    NCCL_IB_DISABLE: "0"           # enable InfiniBand
    NCCL_IB_HCA: "mlx5_0,mlx5_1"   # IB interfaces
    NCCL_NET_GDR_LEVEL: "5"        # GPU Direct RDMA
    NCCL_MIN_NCHANNELS: "4"
    CUDA_VISIBLE_DEVICES: "0,1,2,3,4,5,6,7"
```

---

## 4. 🛡️ Fault Tolerance

### 4.1 Auto‑Checkpointing Strategy

| Level | Mechanism | Frequency |
|-------|-----------|-----------|
| **Application (Trainer)** | HuggingFace `save_steps` + `save_total_limit` | Every 500 steps, keep last 3 |
| **Framework (PyTorch)** | `torch.save` with `state_dict` | Every epoch |
| **Orchestrator (Ray)** | Ray Train checkpoint via `ray.train.report` | Every N batches |
| **Infrastructure (K8s)** | Pod termination with checkpoint flush | On SIGTERM |

### 4.2 Graceful Shutdown + Resume

**Django signal handler in the worker:**

```python
import signal
import sys

_shutdown_flag = False

def _handle_sigterm(signum, frame):
    global _shutdown_flag
    _shutdown_flag = True
    # Flush checkpoint to S3
    trainer.save_checkpoint()

signal.signal(signal.SIGTERM, _handle_sigterm)
```

**Kubernetes Pod termination grace period:**

```yaml
# In the worker deployment template:
spec:
  terminationGracePeriodSeconds: 120   # give 2 minutes to save checkpoint
  containers:
    - name: worker
      lifecycle:
        preStop:
          exec:
            command: ["/bin/sh", "-c", "celery control shutdown; sleep 90"]
```

### 4.3 Node Failure Recovery

**Detection:** Prometheus monitors node health via `kube-node-unreachable` alert.  
**Recovery:** Karpenter detects the lost node, provisions a replacement GPU node within 2‑3 minutes. Meanwhile, the Celery task retries on the new node.

**Job retry policy:**

```yaml
# In the Kubernetes Job spec:
spec:
  backoffLimit: 3
  podFailurePolicy:
    rules:
      - onExitCodes:
          operator: In
          values: [137]   # SIGKILL (OOM)
        action: FailJob
      - onPodConditions:
          - type: DisruptionTarget
        action: FailJob
```

### 4.4 Flash Checkpoint (DLRover Pattern)

For large models, use the **DLRover** approach: save checkpoints to **local NVMe first** (under 5 seconds), then asynchronously copy to S3/Lustre. This minimizes the blocking time during checkpoint serialization.

---

## 5. 🔐 Role‑Based Access Control (RBAC)

### 5.1 Three‑Tier Permission Model

| Role | Permissions | Typical User |
|------|------------|--------------|
| **Admin** | Full CRUD on all resources, manage tenants, view all experiments | ML platform engineer |
| **Researcher** | Create/read/stop own experiments, launch sweeps, register models | Data scientist |
| **Viewer** | Read‑only access to experiments in assigned projects | Manager / stakeholder |

### 5.2 Implementation

**Django side** (using `django-guardian` for object‑level permissions):

```python
# In the Experiment model:
class Experiment(models.Model):
    owner = models.ForeignKey(User, on_delete=models.CASCADE)
    project = models.ForeignKey(Project, on_delete=models.CASCADE)

    class Meta:
        permissions = [
            ("can_stop_experiment", "Can stop running experiments"),
            ("can_download_model", "Can download trained models"),
        ]
```

**DRF views with permission classes:**

```python
from rest_framework.permissions import DjangoObjectPermissions

class ExperimentViewSet(ModelViewSet):
    permission_classes = [DjangoObjectPermissions]
    queryset = Experiment.objects.all()

    def get_queryset(self):
        if self.request.user.is_superuser:
            return Experiment.objects.all()
        return Experiment.objects.filter(owner=self.request.user)
```

### 5.3 Kubernetes RBAC per Tenant

Using **Capsule** operator for multi‑tenancy:

```yaml
apiVersion: capsule.clastix.io/v1beta2
kind: Tenant
metadata:
  name: team-nlp
spec:
  owners:
    - name: nlp-admin
      kind: User
  namespaceOptions:
    additionalMetadata:
      labels:
        tenant: team-nlp
  resourceQuota:
    scope: Tenant
    hard:
      requests.cpu: "128"
      requests.memory: "512Gi"
      requests.nvidia.com/gpu: "16"
  additionalRoleBindings:
    - clusterRoleName: "tenant-admin"
      subjects:
        - name: nlp-admin
          kind: User
```

This ensures that each research team has its own isolated namespace with hard GPU quotas, and cannot access other teams' experiments or resources.

---

## 6. 🚨 Alerting

### 6.1 Alerting Pipeline

```
Application Events → Django Signals → Slack/Email
System Metrics → Prometheus → AlertManager → Slack/Email/Webhook
Infrastructure Events → K8s Events → Grafana Alerts → PagerDuty
```

### 6.2 Prometheus Alert Rules

```yaml
apiVersion: monitoring.coreos.com/v1
kind: PrometheusRule
metadata:
  name: ml-platform-alerts
spec:
  groups:
    - name: training
      rules:
        - alert: TrainingJobFailed
          expr: increase(experiment_failed_total[15m]) > 0
          for: 1m
          labels: {severity: warning}
          annotations:
            summary: "Training job failed in {{ $labels.tenant }}"
            description: "{{ $value }} jobs have failed in the last 15 minutes."
            slack_channel: "#ml-ops-alerts"

        - alert: GPUOOM
          expr: DCGM_FI_DEV_FB_USED / DCGM_FI_DEV_FB_TOTAL > 0.95
          for: 2m
          labels: {severity: critical}
          annotations:
            summary: "GPU {{ $labels.gpu }} on {{ $labels.node }} is out of memory"
            description: "Memory usage is {{ $value | humanizePercentage }}."

        - alert: GPUNodeDown
          expr: up{job="dcgm-exporter"} == 0
          for: 5m
          labels: {severity: critical}
          annotations:
            summary: "GPU node {{ $labels.node }} is unreachable"
            slack_channel: "#infra-alerts"

        - alert: GPUThermalThrottle
          expr: DCGM_FI_DEV_GPU_TEMP > 85
          for: 5m
          labels: {severity: warning}
          annotations:
            summary: "GPU {{ $labels.gpu }} temperature is {{ $value }}°C"

        - alert: CeleryQueueBacklog
          expr: celery_queue_length{queue="gpu_training"} > 10
          for: 10m
          labels: {severity: info}
          annotations:
            summary: "Training queue has {{ $value }} pending jobs"

        - alert: ClusterGPUCapacity
          expr: kube_resourcequota{resource="requests.nvidia.com/gpu", type="used"} /
                kube_resourcequota{resource="requests.nvidia.com/gpu", type="hard"} > 0.90
          for: 5m
          labels: {severity: warning}
          annotations:
            summary: "GPU capacity at {{ $value | humanizePercentage }} – consider adding nodes"
```

### 6.3 AlertManager Slack Configuration

```yaml
apiVersion: v1
kind: Secret
metadata:
  name: alertmanager-slack
type: Opaque
stringData:
  slack_api_url: "https://hooks.slack.com/services/T00000000/B00000000/xxxxxxxxxx"
---
apiVersion: monitoring.coreos.com/v1alpha1
kind: AlertmanagerConfig
metadata:
  name: ml-platform-alert-routing
spec:
  route:
    groupBy: ["severity"]
    groupWait: 30s
    groupInterval: 5m
    repeatInterval: 12h
    receiver: "slack-ml-ops"
    routes:
      - match:
          severity: critical
        receiver: "slack-infra-critical"
        repeatInterval: 1h
      - match:
          severity: warning
        receiver: "slack-ml-ops"
  receivers:
    - name: "slack-ml-ops"
      slackConfigs:
        - channel: "#ml-ops-alerts"
          apiURL:
            key: slack_api_url
          title: '[{{ .Status | toUpper }}{{ if eq .Status "firing" }}:{{ .Alerts.Firing | len }}{{ end }}] {{ .CommonAnnotations.summary }}'
          text: "{{ range .Alerts }}{{ .Annotations.description }}\n{{ end }}"
    - name: "slack-infra-critical"
      slackConfigs:
        - channel: "#infra-alerts"
          apiURL:
            key: slack_api_url
```

---

## 7. 📈 Inference Endpoint Auto‑Scaling

### 7.1 Architecture

```
                     ┌──────────────────────────┐
                     │   Ingress (Nginx/Traefik) │
                     └───────────┬──────────────┘
                                 │
                     ┌───────────▼──────────────┐
                     │   Inference Service       │
                     │   (FastAPI + vLLM)         │
                     │   Pods: 1 → N             │
                     └───────────┬──────────────┘
                                 │
              ┌──────────────────┼──────────────────┐
              │                  │                  │
     ┌────────▼────────┐ ┌──────▼──────┐ ┌────────▼────────┐
     │ HPA (CPU/Mem)   │ │ KEDA        │ │ Prometheus       │
     │ min: 1, max: 10 │ │ (GPU util,  │ │ Adapter          │
     │ targetCPU: 70%  │ │  queue len) │ │ (custom metrics) │
     └─────────────────┘ └─────────────┘ └─────────────────┘
```

### 7.2 KEDA ScaledObject (GPU‑Aware)

```yaml
apiVersion: keda.sh/v1alpha1
kind: ScaledObject
metadata:
  name: inference-scaler
  namespace: team-nlp
spec:
  scaleTargetRef:
    name: inference-deployment
  pollingInterval: 15
  cooldownPeriod: 300           # wait 5 min before scaling down
  minReplicaCount: 1
  maxReplicaCount: 10
  triggers:
    # Scale based on GPU utilization
    - type: prometheus
      metadata:
        serverAddress: http://prometheus-operated.monitoring.svc:9090
        metricName: gpu_utilization
        query: |
          avg(DCGM_FI_DEV_GPU_UTIL{pod=~"inference-.*"})
        threshold: "70"         # scale up when GPU > 70%
    # Scale based on request queue depth
    - type: prometheus
      metadata:
        serverAddress: http://prometheus-operated.monitoring.svc:9090
        metricName: inference_queue_depth
        query: |
          sum(inference_request_queue_size{pod=~"inference-.*"})
        threshold: "100"
    # Scale to zero when no traffic
    - type: cron
      metadata:
        timezone: America/New_York
        start: "0 20 * * *"      # 8 PM
        end: "0 6 * * *"         # 6 AM
        desiredReplicas: "0"
```

### 7.3 Karpenter Node Autoscaling

Karpenter watches for pods that cannot be scheduled due to GPU scarcity and provisions new nodes on‑demand.

```yaml
apiVersion: karpenter.sh/v1beta1
kind: NodePool
metadata:
  name: gpu-a100-pool
spec:
  template:
    spec:
      requirements:
        - key: "karpenter.sh/capacity-type"
          operator: In
          values: ["on-demand", "spot"]
        - key: "nvidia.com/gpu.product"
          operator: In
          values: ["NVIDIA-A100-SXM4-40GB", "NVIDIA-H100-SXM5-80GB"]
      nodeClassRef:
        name: gpu-node-class
      taints:
        - key: "nvidia.com/gpu"
          effect: "NoSchedule"
  limits:
    nvidia.com/gpu: 64          # hard cap to prevent runaway costs
  disruption:
    consolidationPolicy: WhenUnderutilized
    consolidateAfter: 5m
    budgets:
      - nodes: "20%"
        reason: "CostControl"
        schedule: "* * * * *"
        duration: 10m
```

Karpenter nodes can also be pre‑provisioned for known workload spikes (e.g., scheduled sweeps) using `NodePool` with `consolidationPolicy: WhenEmpty` and Cron triggers.

---

## 8. 🏗️ Production Storage Architecture

| Data | Storage | Reason |
|------|---------|--------|
| Experiment metadata, users, sweeps | PostgreSQL 16 (HA, 3 replicas) | ACID, indexed queries |
| Celery task queue + results | Redis 7 (Sentinel, 3 nodes) | Low‑latency pub/sub |
| Model artifacts (pytorch_model.bin, tokenizer) | MinIO (S3‑compatible, erasure coding) | Scalable, cost‑effective |
| Checkpoints (mid‑training) | NFS CSI / Lustre (RWX) | Fast read/write, accessible from all pods |
| Container images | Private registry (Harbor / ECR) | Versioned, scanned |
| Logs | Loki + Grafana | Centralized, searchable |
| Metrics | Prometheus + Thanos (long‑term) | 13‑month retention |

### 8.1 S3 Artifact Layout

```
s3://ml-artifacts/
├── tenants/
│   ├── team-nlp/
│   │   ├── runs/
│   │   │   └── <run_uuid>/
│   │   │       ├── config.json
│   │   │       ├── metrics.jsonl
│   │   │       ├── training_history.png
│   │   │       └── model/
│   │   │           ├── pytorch_model.bin
│   │   │           ├── config.json
│   │   │           └── tokenizer.json
│   │   └── sweeps/
│   │       └── <sweep_uuid>/
│   │           └── optuna_study.db
│   └── team-vision/
│       └── ...
└── global/
    └── checkpoints/
```

---

## 9. 📊 Phase 3 Implementation Schedule (4 Weeks)

### Week 1 – Helm & Kueue

1. Create complete Helm chart for Django API, Celery worker, React frontend.
2. Add `values.yaml`, `values-prod.yaml`, `values-staging.yaml`.
3. Set up Kueue `ClusterQueue`, `LocalQueue`, `WorkloadPriorityClass`.
4. Integrate Kueue Job submission into the Django API (replace threading).
5. Test: submit a training job via API → suspended → admitted → runs → completes.

### Week 2 – Ray Train & Checkpointing

6. Deploy KubeRay operator via Helm.
7. Add `RayCluster` manifest to Helm chart.
8. Implement distributed training runner in `runners.py` (Ray Train integration).
9. Set up NFS CSI driver and configure checkpoint persistence.
10. Test: 4‑GPU distributed training with auto‑resume after pod restart.

### Week 3 – RBAC, Alerting, Monitoring

11. Deploy Capsule for multi‑tenancy, configure tenant isolation.
12. Add `django-guardian` object‑level permissions to API.
13. Deploy Prometheus + Grafana stack (kube‑prom‑stack).
14. Deploy DCGM Exporter on all GPU nodes.
15. Create all PrometheusRule alerts (GPU OOM, node down, queue backlog).
16. Configure AlertManager with Slack webhook.

### Week 4 – Inference & CI/CD Polish

17. Deploy ArgoCD ApplicationSet for multi‑tenant auto‑deployment.
18. Set up KEDA ScaledObject for inference auto‑scaling.
19. Deploy Karpenter NodePool for GPU node autoscaling.
20. Configure GitHub Actions CI/CD pipeline (lint → test → build → ArgoCD sync).
21. End‑to‑end test: PR merge → CI passes → ArgoCD deploys → training job runs across GPUs → model saved to S3 → inference endpoint scales.
22. Write runbooks for common failures (GPU OOM, NCCL timeout, node drain).

---

## 🔮 What Phase 3 Delivers

At the end of Phase 3, you have a **hardened, production‑grade ML platform** that:

- Deploys entirely via **Helm + ArgoCD** with one `git push`.
- Manages GPU resources fairly across teams via **Kueue** priority queues.
- Handles multi‑GPU distributed training with **Ray Train**.
- Recovers from node failures automatically with **checkpointing + Karpenter**.
- Enforces **role‑based access** at both Django and Kubernetes levels.
- Alerts operators via **Slack/email** when GPUs overheat, jobs fail, or queues back up.
- Auto‑scales inference pods from 0→N based on **GPU utilization and queue depth**.

The platform can now serve dozens of research teams, each with isolated resources, across hundreds of GPUs, with full observability and self‑healing capabilities.