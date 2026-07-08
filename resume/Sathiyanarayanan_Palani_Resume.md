# SATHIYANARAYANAN PALANI
**Senior Software Engineer (Transitioning to AI Safety) | Interpretability & Civic Technology**

[sathiyanarayanan396@gmail.com](mailto:sathiyanarayanan396@gmail.com) | +91-8072500974 |
[Github](https://github.com/Palani-SN) | [LinkedIn](https://www.linkedin.com/in/sathiyanarayanan-palani-19b621117/) | [Portfolio](https://palani-sn.github.io/) | [Kaggle](https://www.kaggle.com/code/sathiyanarayanan396)

---

## Professional Summary

Software Engineer with 8+ years building production systems and 2+ years developing LLM-powered applications for civic impact.

- Built FNMA, a full-stack fact-checking platform (React, Python, Flutter, LLM APIs) designed to combat misinformation—WEF's #1 global risk—through adaptive behavioral systems
- Published 6 PyPI packages with 86k+ downloads, demonstrating commitment to public goods and developer enablement
- Implemented Anthropic's Sparse Autoencoder methodology for LLM interpretability research across 4 projects
- 4.5 years as Founding Engineer building telemetry infrastructure, ETL pipelines, and scalable web applications
- Eager to help mission-driven organizations (education, healthcare, nonprofits) deploy AI safely and effectively

---

## Key Projects (LLM & Mission-Driven Work)

### FNMA - Civic Information Platform | May 2024 – Present

[OnePager : River Banks, Proprioceptors & An Intangible Network](https://palani-sn.github.io/PT/FNMA.html) | [Kaggle](https://www.kaggle.com/code/sathiyanarayanan396/fact-check)

- Architected full-stack misinformation detection platform addressing WEF's #1 short-term global risk
- Built React website + Flutter mobile app + Python backend with LLM API integration for fact-checking organizations
- Designed Adaptive Targeted Diffusion Control system—self-regulating behavioral engine reducing spam without manual moderation
- Implemented comprehensive data analytics with 7+ visualization types (heatmaps, trend analysis, geo-spatial mapping)
- Platform designed for reproducible deployment across regions to improve collective civic reasoning
- **Tech:** React, Python (FastAPI), ELK Stack, Claude/Gemini API, Flutter

---

### Abliteration & Constitutional Classifiers++ | Jun 2026 - Jul 2026

[Article : Refusal-Direction Abliteration & Constitutional-Classifiers++ Pipeline](https://palani-sn.github.io/LLM2/README.html) | [Repo](https://github.com/Palani-SN/abliteration-et-constitutional-classifier)

- Located and extracted refusal direction from Falcon3-1B-Instruct via runtime activation ablation (no weight modification)
- Implemented Anthropic's Constitutional Classifiers++ two-stage architecture: FastGate (activation probe) + ExchangeClassifier
- 36-coordinate Cohen's-d signature matches 38,912-coordinate baseline at identical 99.5% OOD accuracy (0.08% of information)
- Classifier restores 99/100 harmful prompt blocking at 60% lower latency than original model's refusals
- **Tech:** PyTorch, Transformers, Falcon3-1B, Ollama/gemma4, Plotly

---

### LLM-Emotions - Interpretability Research | Mar 2026 – May 2026

[Article : LLM Emotions — Sparse Autoencoder Analysis of Emotion Representations in Gemma 3 1B IT](https://palani-sn.github.io/LLM1/README.html) | [Repo](https://github.com/Palani-SN/LLM-Emotions)

- Implemented Anthropic's Sparse Autoencoder methodology on Gemma 3 1B IT to decompose emotional representations
- Trained SAE (2304 latent features, top-k=32 sparsity) achieving perfect feature disjointness (identity matrix similarity)
- Discovered valence and arousal axes via PCA, recovering Russell's circumplex model from raw neural activations
- Demonstrated causal steering: injecting learned vectors during inference shifted emotional tone without weight modification
- **Tech:** PyTorch, Transformers, scikit-learn, Matplotlib

---

### Mechanistic Interpretability Projects | Jan 2026 – Mar 2026

Articles:

1. [Article : Monosemanticity-MLP-Interpretability](https://palani-sn.github.io/ML/ReadMe.html) | [Repo](https://github.com/Palani-SN/monosemanticity-mlp-interpretability)
2. [Article : Activation-Steering-et-Ablation](https://palani-sn.github.io/ML2/ReadMe.html) | [Repo](https://github.com/Palani-SN/Activation-Steering-et-Ablation)
3. [Article : Feature-Ranking-et-Throttle-Testing](https://palani-sn.github.io/ML3/ReadMe.html) | [Repo](https://github.com/Palani-SN/Feature-Ranking-et-Throttle-Testing)

- Implemented Anthropic's "Towards Monosemanticity" paper on toy MLP for index-based arithmetic task
- Built complete pipeline: dataset generation → MLP training → activation harvesting → SAE training → feature probing
- Generated interactive visualizations (Sankey diagrams, heatmaps, bell curves) mapping input → neuron → feature → output circuits
- Achieved monosemantic features with 60–66 active features out of 2304 latent space (~3% sparsity)
- **Tech:** PyTorch, pandas, visualization libraries

---

### PyPI Open Source Packages | Jan 2022 – Apr 2022

[PyPI : Palani-SN](https://pypi.org/user/Palani-SN/)

- Published 6 production-grade Python packages: Py4Cli, LogExAn, ScEqAn (86k+ lifetime downloads)
- Built LogExAnWebApp (FastAPI, Flask) and ScEqAnWebApp (FastAPI, React) for log analysis and script equation analysis
- Demonstrates commitment to public goods and developer enablement

---

### Infrastructure as Code & Deployment | Sep 2025 & Oct 2022
[Repo : Code2Cloud-Terraform-Helm](https://github.com/Palani-SN/Code2Cloud-Terraform-Helm) | [Repo : ansible-LogExAnWebApp-deployment](https://github.com/Palani-SN/ansible-LogExAnWebApp-deployment)

- Built end-to-end IaC pipeline: Terraform → Kubernetes → Helm charts 
  for deploying web applications on GCP/GKE
- Automated deployment of LogExAnWebApp using Ansible playbooks for 
  multi-environment configuration management
- Demonstrates production deployment patterns applicable to LLM 
  application infrastructure
- **Tech:** Terraform, Helm, Kubernetes, Ansible, GCP, Docker

---

## Professional Experience

### Telemetry Software Data Engineer | Intel SRR, Bangalore
**Nov 2023 – Present**

- **ETL Pipeline:** Architected scalable data harvesting pipelines for bulk insertion to Elasticsearch, aggregating timeseries metrics from centralized TSDB cluster
- **Telemetry UI:** Built scalable web UI framework using MVC architecture and Dash, visualizing KPIs for distributed systems
- **EBDT (Event-Based Distributed Tracing):** Proposed and prototyped distributed tracing architecture inspired by OpenTelemetry for enhanced observability. Submitted to ETTC 2025.
- **QTOPP:** Built queued task processing system on process pools as headless CI/CD runner for scheduled statistical calculations
- **V&V POC:** Evaluated and prototyped infrastructure for AI-assisted verification and validation using pre-trained models

---

### Semiconductor S/W Product Development Engineer | UST Global, Bangalore
**May 2022 – Oct 2023**

- Built Python automation test framework with GitHub Actions CI/CD and custom runners for telemetry testing
- Implemented Serial-Over-LAN using IPMI with monitoring threads for auto-retry during target reboot
- Collected performance KPIs from thermal/power subsystems via Redfish (DMTF spec) using OData queries over REST APIs
- Developed scripts to upload register values to InfluxDB using low-latency line protocol for real-time visualization

---

### Senior Engineer | Ielektron Technologies, Chennai
**Oct 2018 – Dec 2021**

As Founding Engineer (4.5 years), led architecture and development:

#### Software Validation Engineer – MBRDI ADAS (as external) | Apr 2019 – Dec 2021

- Wrote Python test scripts for automated validation of ADAS measurements on HIL scenarios from DOORS requirements
- Automated HIL processes via SSH using paramiko and SCP
- Built custom framework for automated test case generation using pandas dataframes as pre-HIL execution task
- Performed runtime dynamic analysis of ADAS utility modules using ASTREE, unit/L2 testing with Google Test (C++)

#### Python Developer – SAFRAN Aerospace (as external) | Oct 2018 – Mar 2019

- Built multi-display broadcast system over local network using TCP/IP
- Created 3D scalar plot visualizations for sensor data using Plotly
- Developed event classification algorithm based on volumetric data normal distribution analysis

---

### Embedded Software Developer | Ielektron Technologies, Chennai
**Jul 2017 – Sep 2018**

- Improved embedded software unit and functional testing practices
- Built multi-branch CI/CD pipeline in Jenkins with GitHub app integration and Slack notifications
- Developed semi-automated hardware testing platform for Renesas microcontrollers using IronPython Renesas APIs

---

## Technical Skills

| Category | Skills |
|---|---|
| **Languages** | Python, C/C++, Shell/Batch scripting |
| **LLM & ML** | PyTorch, Transformers, OpenCV, TensorFlow, Keras, scikit-learn, Sparse Autoencoders, activation steering, interpretability research, Claude Code, Copilot CLI, google-genai, agent architectures, evaluation frameworks, dspy, LangChain, LangGraph, Moshi-ASR, Moshi-TTS, delayed-streams-modelling |
| **Web Development** | FastAPI, Flask, React, Next.js, shadcn/ui |
| **Data & Databases** | pandas, polars, numpy, PostgreSQL, MySQL, SQLAlchemy, InfluxDB, Elasticsearch |
| **Visualization** | Matplotlib, Plotly, Dash |
| **DevOps & Cloud** | Docker, Kubernetes, Terraform, Helm, Jenkins, GitHub Actions, GCP (Cloud Run, Artifact Registry), AWS (EKS basics) |
| **Protocols** | TCP/IP, UDP, HTTP, MQTT, WebRTC, WebSockets, Redfish/OData |
| **Testing** | pytest, Google Test, unit/integration testing |

---

## Education

**B.E. Electrical & Electronics Engineering** | Anna University (AIHT)
2013 – 2017 | First Class

Final Project: Automatic Attendance System using Face Recognition (OpenCV, Raspberry Pi)

---

## Participations & Recognition

- European Test & Telemetry Conference 2025 – Paper Submission (2025)
- Google Gemini Long-Context Kaggle Competition Participant (2024)
- Civicus Nelson Mandela & Graca Machel Innovation Awards Participant (2024)
- OpenCV AI Scavenger Hunt – Week 4 Winner (2022)
- Hacktoberfest 2022 – Contributor
- Ist Prize - Inter-Department Working Model Development Competition during UG (2015).
