# Inference Stack Model (ISM) — L0–L6

A layered model for understanding, diagnosing and operating LLM inference systems.

**Quick rule:** who decides sits on top (L4a/L4b), who computes sits at L3, who makes it possible sits at L2/L1/L0.

---

## L6 — Client / Entry Point

### What it is
The origin point of the request. A human interface or application system that consumes inference.

### Responsibilities
- User input collection
- Client-side session management
- Output rendering
- Generation parameters (temperature, max_tokens)
- UX

### Where it runs
- Browser (React, Next.js)
- Mobile app
- Desktop app
- CLI
- Application backend

### Who owns it in a company
Product / Frontend Team

### Typical deployment
- CDN
- App store
- CPU web container
- Internal application server

### What it passes down
HTTP/gRPC request containing:
- prompt
- generation parameters
- metadata (user_id, session_id)

### Concrete examples
- Open WebUI
- ChatGPT web interface
- Internal Slack bot
- Internal Python SDK

---

## L5 — API Gateway / Edge

### What it is
The security boundary between the external world and the internal AI domain.

### Responsibilities
- Authentication (API key, OAuth, SSO)
- Authorization
- Rate limiting
- TLS termination
- Logging
- Routing to L4a

### Where it runs
- Reverse proxy
- CPU API gateway cluster
- Multi-region edge

### Who owns it in a company
Platform / Security Team

### Typical deployment
- Nginx / Envoy / Kong
- AWS API Gateway / Azure API Management
- WAF + Load balancer
- CPU nodes separate from GPUs

### What it passes down
Validated and enriched request:
- Internal format
- Authenticated metadata
- No public exposure

### Concrete examples
- Kong Gateway
- Cloudflare Edge
- Nginx + Lua
- AWS API Gateway

---

## L4a — Request Orchestration (Control Plane)

### What it is
The layer that governs internal traffic before it touches the GPU.

### Responsibilities
- Admission control
- Request prioritization
- Multi-tenant policy
- Quota enforcement
- Backpressure
- Load shedding
- Global queue management
- Routing to L4b instances

### Where it runs
- Internal CPU service
- Kubernetes microservice
- Dedicated control plane

### Who owns it in a company
ML Infrastructure / Platform AI Team

### Typical deployment
- Containerized microservice
- CPU nodes
- Autoscaling independent from GPUs

### What it passes down
Admitted requests with:
- Assigned priority
- Validated parameters
- Serving instance assignment

### Concrete examples
- Custom internal scheduler
- Ray Serve (control logic)
- Custom admission controller
- Multi-tenant policy engine

---

## L4b — GPU Workload Optimization (Execution Control)

### What it is
The layer that optimizes GPU usage for already-admitted requests.

### Responsibilities
- Continuous batching
- Dynamic batching
- KV cache allocation
- Prefix caching
- Decode scheduling
- GPU preemption
- Memory pressure control
- Token streaming

### Where it runs
- Serving process with GPU access
- GPU-bound Kubernetes pod
- HPC node

### Who owns it in a company
ML Infrastructure / Inference Engineering Team

### Typical deployment
- vLLM container
- TGI GPU pod
- Triton server GPU
- Persistent Slurm job

### What it passes down
Tensor batch:
- input_ids
- attention_mask
- past_key_values
- decode configuration

### Concrete examples
- vLLM
- Hugging Face TGI
- NVIDIA Triton (LLM serving)
- Ollama (aggregated environment)

---

## L3 — Inference Backend / Execution Engine

### What it is
The engine that runs the model forward pass.

### Responsibilities
- Tokenization
- Transformer forward pass
- Model weight management
- Multi-GPU parallelism
- CUDA calls
- Kernel-level optimizations

### Where it runs
- GPU container
- Python/C++ process
- Runtime with CUDA access

### Who owns it in a company
ML Engineering / MLOps

### Typical deployment
- GPU Docker container
- CUDA-based image
- Model weights loaded from object storage

### What it passes down
CUDA calls:
- cudaMalloc
- cudaLaunch
- cudaMemcpy
- NCCL primitives

### Concrete examples
- PyTorch + Transformers
- TensorRT-LLM
- llama.cpp
- JAX runtime

---

## L2 — Runtime & Packaging

### What it is
The execution layer that ensures a consistent and reproducible environment.

### Responsibilities
- Container runtime
- nvidia-container-runtime
- Library versioning
- Dependency isolation
- Image lifecycle

### Where it runs
- Host operating system
- Container runtime engine

### Who owns it in a company
Infrastructure / SRE

### Typical deployment
- containerd
- Docker
- CRI-O
- Kubernetes Nvidia device plugin

### What it passes down
Access to:
- /dev/nvidia* devices
- CUDA libraries
- syscalls to driver

### Concrete examples
- containerd + nvidia-container-runtime
- Docker GPU mode
- Kubernetes GPU node

---

## L1 — Driver & CUDA

### What it is
The translator between CUDA calls and GPU hardware.

### Responsibilities
- VRAM allocation
- Kernel scheduling
- SM dispatch
- NCCL communication
- Interrupt handling
- Hardware-level memory management

### Where it runs
- Kernel space (driver)
- User space (CUDA runtime)

### Who owns it in a company
Infrastructure / GPU Ops

### Typical deployment
- NVIDIA driver installed on host
- Version alignment with CUDA
- DCGM monitoring

### What it passes down
Hardware commands:
- GPU register writes
- Grid/block configuration
- DMA transfers

### Concrete examples
- NVIDIA driver 535+
- CUDA runtime
- NCCL
- cuBLAS

---

## L0 — Hardware

### What it is
The physical GPU chip and the infrastructure that hosts it.

### Responsibilities
- Mathematical operation execution
- Massive parallelism
- Physical VRAM management
- NVLink / PCIe interconnect

### Where it runs
- Server rack
- Datacenter
- GPU node

### Who owns it in a company
Datacenter Ops / Cloud Provider

### Typical deployment
- 4–8 GPU server
- Rack with 100–400 Gbps networking
- 3–5 kW per server power

### What it passes down
Nothing.
This is the final layer.

### Concrete examples
- NVIDIA A100
- H100
- RTX 4090
- DGX system

---

*Part of the Dielabs framework. See also: [Inference Mastery Model](inference-mastery-model.md), [Inference Diagnostic Framework](inference-diagnostic-framework.md)*
