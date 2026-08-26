# Foundation Models & LLM Engineering for Real-World AI Applications

## Overview

This project evaluates open-weight large language models for practical local deployment on a single GPU.

The goal is to understand the trade-offs between:

- response quality
- inference speed
- generation latency
- GPU memory usage
- model size and efficiency
- deployment practicality

The project is designed for both:

- **industry portfolio use** — demonstrating practical LLM engineering and benchmarking skills
- **academic use** — providing students with a reproducible example of how to evaluate modern language models scientifically

---

## Research Question

> Which open-weight language model provides the best balance of response quality, inference speed, latency, and GPU memory consumption for local deployment on a single NVIDIA RTX A6000?

---

## Models Evaluated

The current study compares three instruction-tuned models:

1. **Qwen2.5-7B-Instruct**
2. **Phi-4-mini-instruct**
3. **Mistral-7B-Instruct-v0.3**

All models were evaluated under comparable benchmark conditions.

---

## Hardware Environment

Experiments were conducted on:

- **GPU:** NVIDIA RTX A6000
- **GPU Memory:** approximately 44 GB available to PyTorch
- **Platform:** RunPod
- **Framework:** PyTorch
- **Model Library:** Hugging Face Transformers
- **Precision:** BF16

---

## Benchmark Architecture

The project uses a modular benchmark framework.

```text
src/
├── benchmark.py
└── benchmark_core/
    ├── __init__.py
    ├── config.py
    ├── hardware.py
    ├── metrics.py
    ├── model.py
    ├── reporting.py
    └── runner.py
```

### Module Responsibilities

- `config.py` — loads and validates YAML experiment configurations
- `hardware.py` — collects GPU and environment metadata
- `model.py` — loads tokenizers and language models
- `runner.py` — executes warm-up and measured inference runs
- `metrics.py` — calculates latency, throughput, percentiles, and summaries
- `reporting.py` — saves JSON and CSV experiment outputs
- `benchmark.py` — command-line entry point and experiment orchestrator

---

## Experiment Configuration

Experiments are controlled through YAML configuration files.

```text
configs/
├── baseline.yaml
├── models/
│   ├── mistral7b_v03_bf16.yaml
│   ├── phi4_mini_bf16.yaml
│   └── qwen25_7b_bf16.yaml
└── quality/
    ├── mistral7b_v03_quality.yaml
    ├── phi4_mini_quality.yaml
    └── qwen25_7b_quality.yaml
```

This configuration-driven design allows models, generation settings, prompts, and experiment parameters to be changed without modifying the benchmark source code.

---

## Performance Benchmark

The first experiment focused on inference performance.

The benchmark used:

- 3 prompts
- 1 warm-up run per prompt
- 3 measured runs per prompt
- deterministic generation (`do_sample: false`)
- BF16 precision
- maximum generation length of 256 tokens
- the same NVIDIA RTX A6000 GPU

The prompts represented three practical domains:

1. business applications
2. healthcare
3. education

### Performance Results

| Model | Mean Generation Time | Mean Throughput | Peak VRAM Allocated |
|---|---:|---:|---:|
| **Phi-4-mini** | **5.655 s** | **43.748 tokens/s** | **7.191 GB** |
| Qwen2.5-7B | 7.412 s | 34.538 tokens/s | 14.211 GB |
| Mistral-7B-v0.3 | 7.559 s | 33.867 tokens/s | 13.546 GB |

### Performance Interpretation

Phi-4-mini was the most computationally efficient model in this experiment.

Compared with Qwen2.5-7B, Phi-4-mini achieved approximately:

- **26.66% higher throughput**
- **23.71% lower generation latency**
- **49.40% lower peak VRAM usage**

Compared with Mistral-7B-v0.3, Phi-4-mini achieved approximately:

- **29.18% higher throughput**
- **25.20% lower generation latency**
- **46.91% lower peak VRAM usage**

This is important for real-world deployment because a model that requires less GPU memory and generates tokens faster can potentially serve more requests using the same hardware.

---

## Why We Created a Separate Quality Benchmark

During the initial performance experiment, we discovered an important methodological issue.

The performance benchmark limited responses to **256 generated tokens**.

When we examined the outputs, we found:

| Model | Responses Hitting 256 Tokens | Total Responses | Truncation Rate |
|---|---:|---:|---:|
| Qwen2.5-7B | 3 | 3 | 100% |
| Phi-4-mini | 2 | 3 | 66.7% |
| Mistral-7B-v0.3 | 3 | 3 | 100% |

Eight of the nine responses reached the token limit.

This meant that the original benchmark was useful for measuring inference performance, but it was not ideal for comparing response quality.

A response might appear incomplete simply because generation was stopped at 256 tokens rather than because the model was incapable of producing a complete answer.

Therefore, performance testing and quality testing were separated.

This is an important lesson in experimental design:

> **A benchmark configuration that is suitable for measuring speed may not be suitable for measuring answer quality.**

---

## Quality Evaluation

A second experiment was created specifically for response-quality evaluation.

The quality benchmark used:

- the same three models
- the same three prompts
- deterministic generation
- 1 measured run per prompt
- no warm-up runs
- maximum generation length of **768 tokens**

The larger generation limit allowed the models to complete their responses naturally.

### Generated Token Counts

| Model | Business Applications | Healthcare | Education |
|---|---:|---:|---:|
| Qwen2.5-7B | 520 | 633 | 392 |
| Phi-4-mini | 370 | 228 | 333 |
| Mistral-7B-v0.3 | 458 | 517 | 560 |

None of the nine quality responses reached the 768-token limit.

This gave us much better material for comparing answer quality.

---

## Quality Evaluation Rubric

Each response was evaluated using five dimensions.

### 1. Instruction Following

Did the model answer what the prompt actually asked?

### 2. Relevance

Was the response focused on the requested topic without unnecessary or unrelated material?

### 3. Technical Soundness

Were the technical explanations generally accurate and reasonable?

### 4. Completeness

Did the response cover the important parts of the question?

### 5. Clarity

Was the answer organized, readable, and easy to understand?

Each dimension was scored from:

```text
1 = Poor
2 = Weak
3 = Acceptable
4 = Good
5 = Excellent
```

The maximum total score for a response was therefore **25 points**.

---

## Overall Quality Results

After evaluating the responses across the three prompts, the average scores were:

| Model | Instruction Following | Relevance | Technical Soundness | Completeness | Clarity | Total Score | Overall Quality Score |
|---|---:|---:|---:|---:|---:|---:|---:|
| **Qwen2.5-7B** | 5.000 | 5.000 | 3.667 | 5.000 | 5.000 | 23.667 | **4.733** |
| **Phi-4-mini** | 5.000 | 5.000 | 3.667 | 4.000 | 4.667 | 22.333 | **4.467** |
| **Mistral-7B-v0.3** | 4.333 | 4.333 | 3.000 | 4.333 | 4.000 | 20.000 | **4.000** |

### Quality Ranking

Under this rubric and prompt set:

1. **Qwen2.5-7B — 4.733 / 5**
2. **Phi-4-mini — 4.467 / 5**
3. **Mistral-7B-v0.3 — 4.000 / 5**

Qwen2.5-7B produced the highest overall quality score.

Phi-4-mini remained relatively close while using substantially less GPU memory and generating tokens considerably faster.

---

## Quality vs. Efficiency

Looking only at quality would favor Qwen2.5-7B.

Looking only at inference efficiency would favor Phi-4-mini.

The combined comparison is:

| Model | Quality Score | Throughput | Latency | Peak VRAM |
|---|---:|---:|---:|---:|
| Qwen2.5-7B | **4.733** | 34.538 tokens/s | 7.412 s | 14.211 GB |
| **Phi-4-mini** | 4.467 | **43.748 tokens/s** | **5.655 s** | **7.191 GB** |
| Mistral-7B-v0.3 | 4.000 | 33.867 tokens/s | 7.559 s | 13.546 GB |

This demonstrates one of the central ideas in practical LLM engineering:

> **The best model is not necessarily the model with the highest quality score or the fastest inference speed. The best model depends on the requirements and constraints of the application.**

---

## Model-Level Interpretation

### Qwen2.5-7B

Qwen2.5-7B achieved the highest response-quality score in this experiment.

Strengths:

- strongest overall rubric score
- clear and well-structured responses
- strong instruction following
- high completeness
- strong relevance

Trade-offs:

- approximately twice the peak VRAM usage of Phi-4-mini
- lower throughput than Phi-4-mini
- higher generation latency than Phi-4-mini

Qwen2.5-7B may therefore be attractive when response quality is the primary concern and sufficient GPU resources are available.

---

### Phi-4-mini

Phi-4-mini achieved the strongest inference efficiency.

Strengths:

- highest throughput
- lowest generation latency
- lowest GPU memory usage
- good response quality despite its smaller resource footprint

Trade-offs:

- slightly lower quality score than Qwen2.5-7B
- some responses were less complete or detailed than Qwen's responses

Phi-4-mini demonstrated the strongest overall **quality-efficiency trade-off** in this experiment.

This makes it particularly interesting for applications where:

- GPU memory is limited
- low latency matters
- serving cost matters
- multiple users may need to share the same GPU
- strong but not necessarily maximum response quality is sufficient

---

### Mistral-7B-v0.3

Mistral-7B-v0.3 successfully completed the benchmark and produced detailed responses.

However, under the tested conditions it ranked behind the other two models on the main metrics.

Compared with Phi-4-mini, it had:

- lower throughput
- higher latency
- substantially higher GPU memory usage
- lower quality score

This does not mean Mistral-7B-v0.3 is a poor model.

It means that **under this particular hardware environment, prompt set, generation configuration, and evaluation rubric**, it did not provide the strongest trade-off among the three models tested.

---

## Jupyter Analysis

The main analysis notebook is:

```text
notebooks/01_multi_model_benchmark_analysis.ipynb
```

The notebook contains:

- benchmark result loading
- model comparison tables
- throughput analysis
- generation latency analysis
- GPU memory analysis
- truncation detection
- quality-response inspection
- manual rubric scoring
- aggregate quality scoring
- quality-versus-throughput visualization
- interpretation of model-selection trade-offs

The notebook provides a reproducible record of how the final conclusions were developed from the raw benchmark results.

---

## Example Benchmark Commands

### Performance Benchmark

```bash
python src/benchmark.py \
  --config configs/models/qwen25_7b_bf16.yaml
```

The same benchmark framework can be used with another model configuration:

```bash
python src/benchmark.py \
  --config configs/models/phi4_mini_bf16.yaml
```

or:

```bash
python src/benchmark.py \
  --config configs/models/mistral7b_v03_bf16.yaml
```

### Quality Benchmark

```bash
python src/benchmark.py \
  --config configs/quality/qwen25_7b_quality.yaml
```

Equivalent quality configurations exist for Phi-4-mini and Mistral-7B-v0.3.

---

## Project Structure

```text
foundation-models-llm-engineering/
│
├── configs/
│   ├── baseline.yaml
│   │
│   ├── models/
│   │   ├── mistral7b_v03_bf16.yaml
│   │   ├── phi4_mini_bf16.yaml
│   │   └── qwen25_7b_bf16.yaml
│   │
│   └── quality/
│       ├── mistral7b_v03_quality.yaml
│       ├── phi4_mini_quality.yaml
│       └── qwen25_7b_quality.yaml
│
├── notebooks/
│   └── 01_multi_model_benchmark_analysis.ipynb
│
├── results/
│   └── generated benchmark JSON and CSV outputs
│
├── src/
│   ├── benchmark.py
│   └── benchmark_core/
│       ├── __init__.py
│       ├── config.py
│       ├── hardware.py
│       ├── metrics.py
│       ├── model.py
│       ├── reporting.py
│       └── runner.py
│
├── .gitignore
└── README.md
```

---

## Reproducibility

The project separates:

- source code
- model configurations
- quality configurations
- generated results
- analysis notebooks

This makes experiments easier to reproduce and extend.

Instead of changing Python source code every time a model is tested, experiment settings are stored in YAML configuration files.

This is closer to how real machine-learning experimentation is managed in professional environments.

---

## Important Engineering Lessons

This project demonstrates several practical lessons that go beyond simply running an LLM.

### 1. Benchmark Models Under the Same Conditions

Comparisons are only meaningful when models are evaluated using comparable:

- hardware
- prompts
- precision
- generation settings
- measurement methods

### 2. Warm-Up Runs Matter

The first inference request can behave differently because libraries and GPU operations may still be initializing.

Warm-up runs help reduce this effect when measuring steady-state inference performance.

### 3. Throughput and Latency Measure Different Things

**Latency** tells us how long generation takes.

**Throughput** tells us how many tokens are generated per second.

Both metrics matter when evaluating deployment performance.

### 4. GPU Memory Is a Deployment Constraint

A model that produces good answers but consumes significantly more GPU memory may be more expensive or difficult to deploy.

Peak VRAM usage therefore matters alongside model quality.

### 5. Token Limits Can Distort Quality Evaluation

The original 256-token benchmark truncated eight of nine responses.

Without checking this, we could have incorrectly interpreted incomplete responses as poor model quality.

### 6. Performance and Quality Should Be Evaluated Separately

The 256-token benchmark was useful for performance comparison.

The 768-token benchmark was better suited for quality evaluation.

Separating the two produced a more defensible experiment.

### 7. There Is No Universally Best Model

Model selection is a trade-off among:

- quality
- speed
- memory
- hardware availability
- deployment cost
- application requirements

---

## Key Findings

The experiment produced three major findings.

### Finding 1 — Qwen2.5-7B Produced the Highest Evaluated Quality

Qwen2.5-7B achieved an overall quality score of approximately:

**4.733 / 5**

This was the highest score among the three evaluated models.

### Finding 2 — Phi-4-mini Was the Most Efficient

Phi-4-mini achieved:

- the highest throughput
- the lowest generation latency
- the lowest GPU memory consumption

Its peak allocated VRAM was approximately **7.19 GB**, compared with approximately **14.21 GB** for Qwen2.5-7B.

### Finding 3 — Phi-4-mini Offered the Strongest Quality-Efficiency Balance

Although Qwen achieved the highest quality score, Phi-4-mini remained relatively close in quality while providing substantial performance and memory advantages.

For resource-constrained or latency-sensitive applications, this may make Phi-4-mini the more practical deployment choice.

---

## Limitations

The results should be interpreted within the scope of this experiment.

Important limitations include:

- only three models were evaluated
- only three prompt categories were used
- the quality sample contained only nine responses
- quality scoring used a manually defined rubric
- experiments were conducted on one GPU architecture
- only BF16 inference was evaluated in this phase
- standardized benchmark datasets were not included
- statistical quality evaluation was limited by the small prompt set
- model load times can be affected by local caching and previous downloads
- the benchmark focused on single-model local inference rather than production-scale concurrent serving

Therefore, the results should not be interpreted as proving that one model is universally superior.

They describe performance **under the specific experimental conditions used in this project**.

---

## Next Steps

Potential extensions include:

- 8-bit quantization
- 4-bit quantization
- comparison of BF16 and quantized inference
- larger prompt suites
- standardized evaluation datasets
- automated quality evaluation
- additional open-weight models
- longer-context experiments
- batch-size experiments
- concurrent-user inference testing
- cost-efficiency analysis
- energy-efficiency analysis
- local API serving
- browser-based model playground
- benchmark dashboard
- deployment recommendations
- educational notebooks
- final research report

---

## Educational Value

This project is also intended as a teaching example.

Students can use it to understand the difference between simply **using an AI model** and **engineering an AI system**.

The project demonstrates how to:

- formulate a research question
- design controlled experiments
- configure models reproducibly
- use GPU hardware for LLM inference
- measure latency
- measure token throughput
- measure GPU memory consumption
- compare multiple models fairly
- inspect generated responses
- detect experimental problems such as truncation
- design a quality rubric
- separate quality testing from performance testing
- visualize model trade-offs
- interpret experimental results
- document limitations
- use Git and GitHub for reproducibility
- make evidence-based deployment decisions

---

## Simple Explanation for Students

Imagine that three students are asked to complete the same assignment.

One student produces the best answer but takes more resources.

Another produces an almost equally good answer much faster and with fewer resources.

The third produces a reasonable answer but is neither the fastest nor the highest scoring.

Choosing an LLM is similar.

We should not ask only:

> "Which model gives the best answer?"

We should also ask:

> "How fast is it?"

> "How much GPU memory does it require?"

> "How much does that performance cost?"

> "Is the small improvement in quality worth the additional computing resources?"

These are the types of questions AI engineers need to answer before deploying models in real applications.

---

## Current Conclusion

Under the tested NVIDIA RTX A6000 environment, prompt set, BF16 generation settings, and rubric-based quality evaluation:

> **Qwen2.5-7B achieved the highest response-quality score, while Phi-4-mini provided substantially better throughput, latency, and GPU-memory efficiency. Phi-4-mini therefore demonstrated the strongest overall quality-efficiency trade-off among the three evaluated models.**

The main lesson is that **model selection should be based on the requirements of the application rather than model size or a single benchmark score alone**.
