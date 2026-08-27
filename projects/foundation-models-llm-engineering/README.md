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


## Quantization Experiment

After comparing Qwen2.5-7B, Phi-4-mini, and Mistral-7B-v0.3, the next question was:

> Can a strong model be made significantly more memory-efficient without sacrificing practical inference performance?

To study this, Qwen2.5-7B-Instruct was evaluated in three numerical formats on the same NVIDIA RTX A6000:

- **BF16** — the original 16-bit baseline
- **INT8** — 8-bit quantization using bitsandbytes
- **NF4** — 4-bit NormalFloat quantization using bitsandbytes

The model, prompts, GPU, generation settings, and benchmark procedure were kept consistent so that precision/quantization was the primary experimental variable.

### Quantization Benchmark Configuration

The official benchmark used:

- 3 prompts
- 1 warm-up run per prompt
- 3 measured runs per prompt
- deterministic generation
- maximum generation length of 256 tokens
- NVIDIA RTX A6000
- PyTorch 2.8.0 with CUDA 12.8
- bitsandbytes 0.50.1

### Quantization Results

| Mode | Mean Generation Time | Mean Throughput | Peak VRAM Allocated |
|---|---:|---:|---:|
| BF16 | 7.3072 s | 35.0341 tokens/s | 14.2114 GB |
| INT8 | 25.7012 s | 9.9661 tokens/s | 8.2189 GB |
| **NF4** | **6.4946 s** | **39.4199 tokens/s** | **5.2041 GB** |

### NF4 Compared with BF16

NF4 produced the strongest deployment-efficiency result.

Compared with BF16, NF4 achieved approximately:

- **12.52% higher throughput**
- **11.12% lower generation latency**
- **63.38% lower peak GPU-memory usage**

In simple terms, the model required roughly one-third of the peak GPU memory while generating tokens slightly faster under the tested conditions.

This makes NF4 particularly interesting for:

- memory-constrained GPUs
- lower-cost inference systems
- local deployment
- multi-model environments
- systems where GPU capacity is more important than maximum numerical precision

### INT8 Result

INT8 reduced GPU-memory requirements, but it was substantially slower than BF16 in this experiment.

Its mean throughput was approximately:

**9.97 tokens/second**

compared with:

**35.03 tokens/second for BF16**

The INT8 path therefore demonstrated an important engineering lesson:

> **Lower numerical precision does not automatically mean faster inference.**

Runtime performance depends on factors such as:

- GPU architecture
- quantization kernels
- datatype conversions
- library implementation
- memory movement
- model architecture

During INT8 inference, bitsandbytes also reported internal conversion of BF16 inputs to FP16 for its 8-bit matrix operations.

The result should therefore be interpreted as a property of this specific hardware/software configuration rather than a universal statement about INT8 inference.

### Why Quantization Matters

Quantization is important because GPU memory is one of the main practical constraints when deploying large language models.

Reducing VRAM usage can allow organizations to:

- use less expensive hardware
- run larger models on existing GPUs
- host multiple models on the same machine
- increase deployment flexibility
- reduce infrastructure requirements

The experiment therefore connects model optimization directly to real-world deployment decisions.

### Important Limitation

This phase primarily measured:

- latency
- throughput
- GPU-memory consumption

It does **not yet establish that NF4 preserves the same response quality as BF16**.

A dedicated quantized-output quality evaluation is still required before concluding that NF4 is superior across both efficiency and response quality.

The correct conclusion from this phase is:

> **NF4 provided the strongest measured inference-efficiency result, while response-quality preservation remains a separate evaluation question.**

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

The project produced four major findings.

### Finding 1 — Qwen2.5-7B Produced the Highest Evaluated Quality

Qwen2.5-7B achieved an overall quality score of approximately:

**4.733 / 5**

This was the highest score among the three evaluated models.

### Finding 2 — Phi-4-mini Was the Most Efficient Model in the BF16 Comparison

Among the three models evaluated using BF16, Phi-4-mini achieved:

- the highest throughput
- the lowest generation latency
- the lowest GPU memory consumption

Its peak allocated VRAM was approximately **7.19 GB**, compared with approximately **14.21 GB** for Qwen2.5-7B.

This made Phi-4-mini the strongest overall quality-efficiency trade-off in the original multi-model comparison.

### Finding 3 — Quantization Dramatically Reduced Qwen2.5-7B Memory Requirements

Qwen2.5-7B was subsequently evaluated using BF16, INT8, and NF4.

The measured peak allocated GPU memory was:

| Mode | Peak VRAM |
|---|---:|
| BF16 | 14.2114 GB |
| INT8 | 8.2189 GB |
| NF4 | 5.2041 GB |

INT8 reduced peak allocated VRAM by approximately **42.17%** relative to BF16.

NF4 reduced peak allocated VRAM by approximately **63.38%**.

This demonstrates how quantization can substantially change the hardware requirements of the same model.

### Finding 4 — Lower Precision Did Not Automatically Mean Faster Inference

NF4 achieved approximately:

- **39.42 tokens/second**
- **6.49 seconds mean generation time**

compared with approximately:

- **35.03 tokens/second**
- **7.31 seconds mean generation time**

for BF16.

INT8, however, achieved only approximately:

- **9.97 tokens/second**
- **25.70 seconds mean generation time**

under the tested configuration.

This is an important engineering result:

> **Quantization can reduce memory consumption without guaranteeing higher inference speed.**

Actual performance depends on the interaction between the model, GPU architecture, numerical format, kernels, and software implementation.

---

## Limitations

The results should be interpreted within the scope of this experiment.

Important limitations include:

- only three models were included in the multi-model comparison
- only one model, Qwen2.5-7B, was evaluated in the quantization experiment
- only three prompt categories were used
- the original quality sample contained only nine responses
- quality scoring used a manually defined rubric
- experiments were conducted on one GPU architecture
- the quantization phase focused primarily on latency, throughput, and GPU memory
- BF16, INT8, and NF4 response quality has not yet been evaluated using the same dedicated quality benchmark
- standardized benchmark datasets were not included
- statistical quality evaluation was limited by the small prompt set
- model load times can be affected by caching, storage, and previous downloads
- the benchmark focused on single-model local inference rather than production-scale concurrent serving
- quantization performance may differ with other GPUs, CUDA versions, libraries, kernels, and model architectures

Therefore, the results should not be interpreted as proving that one model or numerical format is universally superior.

They describe performance **under the specific experimental conditions used in this project**.

---

## Next Steps

The most useful extensions are now:

- evaluate BF16, INT8, and NF4 response quality using the same rubric
- expand the prompt suite
- add standardized evaluation datasets
- automate more of the quality-evaluation workflow
- evaluate additional open-weight models
- test longer context lengths
- test different batch sizes
- test concurrent-user inference
- analyze cost per generated token
- investigate energy efficiency
- expose selected models through a local API
- build a simple browser-based model playground
- create a benchmark dashboard
- develop deployment recommendations for different application requirements
- expand the educational notebook
- prepare a final research-style report

The immediate experimental priority is the **quantized quality comparison**.

That would answer the remaining question:

> Does the large memory reduction achieved by NF4 come with a meaningful response-quality trade-off?

---

## Educational Value

This project is intended not only as a benchmark but also as a teaching example.

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
- compare numerical precision and quantization strategies
- distinguish memory savings from speed improvements
- perform smoke tests before expensive experiments
- troubleshoot GPU and CUDA environments
- recover experiments using Git and GitHub
- visualize model trade-offs
- interpret experimental results
- document limitations
- make evidence-based deployment decisions

---

## Simple Explanation for Students

Imagine that three students are asked to complete the same assignment.

One produces the highest-quality answer.

Another produces an almost equally good answer faster and using fewer resources.

The third produces a reasonable answer but requires a different balance of time and resources.

That is similar to our first experiment: **choosing between different language models**.

We then asked a second question.

Suppose we take one of those students and ask them to carry their books in three different ways:

- a normal large bag
- a smaller compressed bag
- a very compact bag

The contents are based on the same underlying model, but the amount of space required to carry them changes.

That is roughly the idea behind **quantization**.

In our experiment, NF4 allowed Qwen2.5-7B to use much less GPU memory while maintaining strong inference performance.

INT8 also reduced memory, but unexpectedly ran much slower.

This teaches an important lesson:

> **Smaller does not automatically mean faster.**

AI engineers therefore need to ask several questions:

> "How good are the answers?"

> "How fast does the model generate?"

> "How much GPU memory does it require?"

> "Can quantization reduce the hardware requirement?"

> "Does compression affect answer quality?"

> "How much does deployment cost?"

These are the kinds of questions that turn an AI demonstration into an engineering experiment.

---

## Current Conclusion

Under the tested NVIDIA RTX A6000 environment, this project produced two complementary conclusions.

### Model Selection

In the original BF16 multi-model comparison:

> **Qwen2.5-7B achieved the highest evaluated response-quality score, while Phi-4-mini provided substantially better throughput, latency, and GPU-memory efficiency. Phi-4-mini therefore demonstrated the strongest overall quality-efficiency trade-off among the three evaluated models.**

### Model Optimization

When Qwen2.5-7B was subsequently evaluated using BF16, INT8, and NF4:

> **NF4 provided the strongest measured inference-efficiency result, reducing peak allocated GPU memory by approximately 63% relative to BF16 while also producing slightly higher throughput and lower generation latency under the tested configuration.**

INT8 also reduced memory consumption, but its inference throughput was substantially lower in this environment.

The quantization experiment does **not yet establish that NF4 preserves the same response quality as BF16**. That remains the most important next evaluation.

The broader lesson from Project 1 is:

> **Model deployment should be treated as a multi-objective engineering problem involving quality, speed, memory, hardware requirements, and cost—not as a search for the largest model or the highest single benchmark score.**

