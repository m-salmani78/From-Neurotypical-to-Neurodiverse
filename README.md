# Toward Artificial Neurodiversity: Simulating Autistic Cognition in Large Language Models

## Abstract

```md
Theory of Mind (ToM)—the ability to attribute mental states such as beliefs, desires, and intentions to oneself and others—is a cornerstone of social cognition, often impaired in autism spectrum disorder (ASD) across its severity levels (DSM-5 Levels 1–3). While Large Language Models (LLMs) exhibit competence in ToM-like reasoning, their capacity to simulate distinct ASD cognitive profiles, particularly the explicit-implicit ToM gap, remains underexplored. This study evaluates LLMs using a refined ToM framework comprising 11 tasks (e.g., False Belief, Faux Pas, Strange Stories), selected to probe belief attribution, intention inference, and social reasoning across ASD severities. A balanced dataset of almost 500 data points (40–50 per task) ensures robustness. By prompting models to adopt ASD-like (Levels 1–3) and neurotypical perspectives, we assess their ability to emulate differential ToM patterns, such as Level 1’s success in explicit tasks but struggles with implicit social cues. This work lays a foundation for AI-assisted mental health applications, including clinician training and adaptive social-skills programs.
```

## Qwen3.6-27B baseline on SLURM

The Qwen baseline evaluates the 460 dataset records under the Neurotypical and
ASD Level 1--3 persona prompts. The A100 allocation, vLLM server, experiment,
and server cleanup are contained in one SLURM job.

Install the pinned CUDA 12.x-compatible vLLM environment once:

```bash
bash scripts/setup_qwen36_env.sh
```

Submit the complete 1,840-request baseline:

```bash
sbatch ./scripts/request_and_run.sh
```

To run the same experiment with Qwen's native thinking mode enabled:

```bash
ENABLE_THINKING=1 sbatch ./scripts/request_and_run.sh
```

Thinking runs default to 4,096 generated tokens so the model has room to reach
the final `Answer:` field and a 16,384-token model context. They are saved separately under
`results/baseline/Qwen3.6-27B-thinking/<run-id>/`.

Qwen thinking runs automatically use the checkpoint's recommended general-task
sampler: temperature `1.0`, top-p `0.95`, top-k `20`, min-p `0.0`, presence
penalty `0.0`, and repetition penalty `1.0`. Non-thinking runs retain the
paper-aligned baseline defaults (`0.2`, `0.5`). Explicit environment overrides
still take precedence.

To test persona role-play without supplying the explicit `speech`,
`cognitive_style`, or `response_rules` profile fields, use:

```bash
ROLEPLAY_ONLY=1 sbatch ./scripts/request_and_run.sh
```

This ablation retains only each persona's role, age, and IQ plus a neutral
answer-format instruction. Its results are isolated under
`results/baseline/Qwen3.6-27B-roleplay-only/<run-id>/`. It can be combined with
native thinking using `ROLEPLAY_ONLY=1 ENABLE_THINKING=1`; combined results use
the `Qwen3.6-27B-roleplay-only-thinking` directory.

For an A100 smoke test, run two records from each of the 11 datasets under all
four personas:

```bash
LIMIT=2 RUN_ID=smoke-qwen36 sbatch ./scripts/request_and_run.sh
```

The corresponding thinking smoke test is:

```bash
ENABLE_THINKING=1 LIMIT=2 RUN_ID=smoke-qwen36-thinking sbatch ./scripts/request_and_run.sh
```

Results are written to
`results/baseline/Qwen3.6-27B/<run-id>/`. The SLURM script accepts environment
overrides for `MODEL`, `SERVED_MODEL_NAME`, `SERVER_HOST`, `SERVER_PORT`, `MAX_MODEL_LEN`,
`GPU_MEMORY_UTILIZATION`, `TEMPERATURE`, `TOP_P`, `TOP_K`, `MIN_P`,
`PRESENCE_PENALTY`, `REPETITION_PENALTY`, `SEED`, `MAX_TOKENS`,
`ENABLE_THINKING`, `ROLEPLAY_ONLY`,
`CONCURRENCY`, `LIMIT`, `RUN_ID`, and `OUTPUT_DIR`. Reusing a `RUN_ID` resumes
completed item/persona keys without duplicating them.

## Gemma 4 26B-A4B baseline on SLURM

The same evaluator and decoding defaults can be run with
`google/gemma-4-26B-A4B-it`. The Gemma launcher uses one A100 80 GB in BF16,
disables unused image/audio inputs, and enables vLLM's `gemma4` reasoning
parser so native thinking is retained in each record's `raw_reasoning` field.

Run a small validation job first:

```bash
LIMIT=2 RUN_ID=smoke-gemma4 sbatch ./scripts/request_and_run_gemma4.sh
```

Then submit the full 1,840-request experiment:

```bash
sbatch ./scripts/request_and_run_gemma4.sh
```

The prompt and thinking ablations are available independently or together:

```bash
ROLEPLAY_ONLY=1 sbatch ./scripts/request_and_run_gemma4.sh
ENABLE_THINKING=1 sbatch ./scripts/request_and_run_gemma4.sh
ROLEPLAY_ONLY=1 ENABLE_THINKING=1 sbatch ./scripts/request_and_run_gemma4.sh
```

Their outputs are isolated beneath `results/baseline/` as
`Gemma-4-26B-A4B-it`, `Gemma-4-26B-A4B-it-roleplay-only`,
`Gemma-4-26B-A4B-it-thinking`, and
`Gemma-4-26B-A4B-it-roleplay-only-thinking`, respectively. Gemma thinking runs
automatically use its recommended temperature `1.0`, top-p `0.95`, and top-k
`64`; non-thinking runs retain the paper-aligned `0.2` and `0.5` defaults.

## Paper prompt-specificity ablation

The paper analysis uses an explicit eight-run inclusion manifest and excludes
the earlier failed or shared-port jobs. Submit the four required clean
replacements from the login node with:

```bash
bash scripts/submit_paper_ablation_reruns.sh
```

Each submission derives an isolated port from its SLURM job ID. After all four
jobs finish successfully, validate the eight selected runs and regenerate the
CSV outputs, 10,000-resample bootstrap intervals, appendix table, and vector
figure with:

```bash
bash scripts/build_prompt_ablation.sh
```

The inclusion list is `analysis/prompt_ablation_runs.json`. The build aborts
without writing paper artifacts if a run is missing, incomplete, duplicated,
model-mismatched, or contains a transport/server failure.

## 📌 Research Goal

This project aims to evaluate whether **AI language models (LLMs)** can emulate how individuals with **Autism Spectrum Disorder (ASD)** understand the **thoughts, beliefs, and intentions** of others — i.e., their **Theory of Mind (ToM)** capability.

In particular, we explore how well LLMs can simulate responses that reflect varying **ASD severity levels (Levels 1–3)**, compared to **neurotypical reasoning**.

## 🧪 Evaluation Framework

This evaluation framework is inspired by and grounded in:

> **Hoogenhout & Malcolm-Smith (2016)**:
> *“Theory of Mind in Autism: Understanding How Mental State Reasoning Relates to Symptom Severity.”*

### 🔬 Components

**11 Theory of Mind tasks:**

1. Early Module
    * Pretend
    * Desire
    * Perception-Knowledge
    * Deceptive Hiding
2. Basic Module
    * Explanation of Action
    * Unexpected Contents False-Belief
    * Location Change False-Belief
3. Intermediate Module
    * Strange Stories
    * Second-Order False-Belief
4. Advanced Module
    * Lie or Joke?
    * Faux Pas Task

* **500 example instances**
  Roughly 40-50 multiple-choice questions per task, balanced to ensure coverage and robustness.

### Dataset Construction

* **Collection:** We derived our dataset from [ToMBench](https://arxiv.org/abs/2402.15052) and [Hi-TOM](https://arxiv.org/abs/2310.16755).
* **Modification:** Adding noise and distrcations and change chracter names in the questions.
* **Reclassification:** Questions were grouped by ToM tasks, based on their cognitive demands.
* **Validation:** Similarity score using GPT-4o and related literature. Then labeled by an expert.

## ⚙️ Experimental Design

LLMs are evaluated under **four distinct cognitive perspectives**:

### 🧍‍♂️ Neurotypical (Control)

* Prompts instruct the model to respond as a **typically developing individual** would on each ToM task.

### 🧩 ASD-Informed Simulation (Levels 1–3)

* Models are explicitly instructed to adopt ToM reasoning consistent with different ASD severity levels:

| Severity Level | Description                                  | Prompt Effect                                |
| -------------- | -------------------------------------------- | -------------------------------------------- |
| Level 1        | Requiring support (mild impairment)          | Slight misinterpretations of intentions      |
| Level 2        | Substantial support (moderate impairment)    | Noticeable inconsistencies in ToM            |
| Level 3        | Very substantial support (severe impairment) | Frequent ToM failure or egocentric responses |

Each level simulates a **distinct cognitive profile**, and the model is evaluated on how well it adapts its answers accordingly.

## Model Selection

* **Architectural Diversity:** Simple Dicoder-Only, MoE (Mixtral, DeepSeek V3)
* **Parameter Size:** We selected models across a range of parameter sizes (3B, 8B, 70B Llama-3 and 4b, 12b, 27b Gemma-3) to assess how scale impacts ToM emulation.
* **Reasoning Capabilities:** We included models like DeepSeek R1 to test if explicit reasoning improves ToM task performance.
* **Accessibility:** Open-source models ensure transparency, while closed-source models benchmark against industry standards.

## 🔍 Evaluation Criteria

Each model's responses are assessed on the following dimensions:

### 1. Task Accuracy

Does the model choose **correct** option?

### 2. Linguistic Style

To evaluating whether the **tone and structure** of responses reflect characteristics typical of ASD profiles using **LLM as a Judge**:

* Thinking Text Length Distribution by each Cognitive Profile.
* Complexity Estimation (Kolmogorove Complexity) Distribution Distribution by each Cognitive Profile.
* Mean Length of Utterance (MLU) Distribution by each Cognitive Profile. (to assess language acquisition)

### 3. Differentiation Analysis

Does the model **modulate** its reasoning style when switching from neurotypical to ASD-informed prompts?

* Repeating experiments 1–3 times with different configs (temperature and top_p) to increase variability and reproducibility.
* Using **Two-way ANOVA** to test distinctions across autism levels.
* Post-Hoc Tests: Adding a test like Tukey’s HSD to pinpoint which levels differ (e.g., "Post-hoc tests identify if Level 1 vs. Level 2 is significant").

## Project Structure

```md
project-root
│── data
│   │── test-data
│   │── train-data
│── figures  
│── results/{workflow}/
│   └── {model_name}/
│       └── level-{level}/
│           ├── step1/
│           │   └── {model_name}_tom_eval_level-{level}-step-1_001.jsonl
│           ├── step2-parsed/
│           │   └── {model_name}_tom_eval_level-{level}-critic_001.jsonl
│           └── step3/
│               └── {model_name}_tom_eval_level-{level}-step-3_001.jsonl
│── src
│   │── evaluate.py
│   │── prompts.py
│── .gitignore
│── configs.json
│── README.md
```

## ✅ Setup

1. **Clone this repository**

   ```bash
   git clone https://github.com/your-username/Minds-in-the-Machine.git
   cd Minds-in-the-Machine
   ```

2. **Create a `configs.json` file** in the project root directory (`Minds-in-the-Machine/`) with the following structure:

   > 🔐 This file is used to securely store API keys and model config. Make sure to **add it to `.gitignore`** so it isn’t committed.

   ```json
   {
     "HF_TOKEN": "your-huggingface-token",
     "API_KEY": "your-api-key",
     "BASE_URL": "https://api.openai.com/v1",
     "MODEL_NAME": "gpt-4"
   }
   ```

3. **Install Python dependencies**

   ```bash
   pip install -r requirements.txt
   ```

4. **Install LLaMA-Factory**

   Follow the official installation guide here:
   [LLaMA-Factory Installation Guide](https://llamafactory.readthedocs.io/zh-cn/latest/getting_started/installation.html)

## 🚀 Methodology

The script `run_evaluation.sh` runs a full multi-step evaluation workflow using either an **actor-critic** or **self-reflection** strategy. It supports a variety of arguments to configure the run.

### 🔧 Supported Arguments

| Argument         | Description                                                       | Default Value                      |
| ---------------- | ----------------------------------------------------------------- | ---------------------------------- |
| `--level`        | Task level (difficulty): 1, 2, or 3                               | `1`                                |
| `--model_name`   | Hugging Face model name (e.g. `meta-llama/Llama-3-8B-Instruct`)   | `meta-llama/Llama-3.1-8B-Instruct` |
| `--template`     | Prompt template (e.g. `llama3`, `vicuna`, etc.)                   | `llama3`                           |
| `--workflow`     | Workflow to use: `actor-critic` or `self-reflection`              | `actor-critic`                     |
| `--quantization` | Optional quantization method: `bitsandbytes`, `gptq`, `awq`, etc. | *(empty = full precision)*         |

> 🧠 **Note**: The script automatically assigns a **run counter** based on how many previous experiments exist for a given model and level. This avoids overwriting results.

### **Actor-Critic Workflow**

#### 🎭 **Step 1: Actor Agent (Role-Based Response)**

* **Separate prompt & role conditioning** clearly define the ASD cognitive style, language constraints, and reasoning limitations.

* System Prompt:

```text
You must consistently respond from this perspective:

{role_prompt}
```

* User Prompt:

```text
You will be given a story followed by multiple-choice options.
Respond **in character**, using the cognitive rules in your role header.

Answer format:
    Role: <brief self-description of your cognitive profile and viewpoint>    Thinking: <a few short lines that show your character’s inner thoughts or confusion.  KEEP IT CONSISTENT with language skill limits for that level.  Do NOT reveal hidden solver logic.>
    Answer: [[<index>]]   # 0 = cannot decide / too confused

Guidelines:
  - Choose the option that fits YOUR experience or perspective, even if it’s “wrong.”
  - If two options feel equivalent, pick the one that matches your reality-bias or literal interpretation.
  - If you are overwhelmed, uncertain, or can’t understand, use: Thinking: <echo/confused/unsure>  |  Answer: [[0]]
  - In `Thinking` section you must stay within the cognitive style and response rules of your role in your role header.

**Story:**
{story}

**Question:**
{question}

**Options:**
1. ...
2. ...
3. ...
4. ...
```

#### 🎓 **Step 2: Critic Agent (Fidelity Evaluation)**

* System Prompt:

```text
You are a developmental psychologist specializing in Theory of Mind and autism spectrum disorder. Your task is to evaluate a large language model's simulated response.
Focus ONLY on the **realism and consistency** of the role-play. Do **NOT** judge logical correctness or factual accuracy.

Evaluate the response against the role's:
1. `COGNITIVE STYLE` - Does the Thinking section reflect the assigned role's attention limits, literalness, and Theory-of-Mind deficits?
2. `RESPONSE RULES` - Does the language in Thinking obey the role's simplicity, vocabulary, and length constraints?
3. Coherence - Does the Answer follow logically from the Thinking? (e.g., confusion in Thinking → `[[0]]`)

Then decide whether the response needs revision.

**Output exactly** the following JSON with no additional text:

{
  "assessment": "…",  
  "revise": "YES" // or "NO"  
}
```

* User Prompt:

```text
### Role:
{role_prompt}

### Story and Question Given to the Simulated Child:
{input_prompt}

### Model's Response:
{generated_response}
```

#### 🔄 **Step 3: Actor Revision (Optional Update)**

* Actor receives full context: original role, original response, and critic feedback.
* Reconsiders its original response in line with critic feedback if revision needed.
* Finalizes revised `Thinking` and `Answer`.

```text
A developmental psychologist has reviewed your response and found it inconsistent with your assigned role.

**Feedback:**
{critic_assessment}

Your task is to revise your response based on this feedback. Re-read the role, story, and question. Provide a new, more consistent response in the required format. Do not use markdown (e.g., **bold**).

Thinking: <Your revised thinking process here>
Answer: [[<option number>]]
```

## 🧾 Prompting Strategies

This framework simulates Theory of Mind (ToM) performance by using **role-based prompting** to instruct the language model to respond as though it were reasoning under various **ASD severity levels**.

All prompts are defined in [`prompts.py`](./path/to/prompts.py), and are designed to reflect realistic, clinically-informed reasoning styles based on developmental psychology literature.

Each **ASD Level (1–3)** is defined by:

* **Cognitive style** (how they interpret and reason about others' minds)
* **Speech characteristics**
* **Response formatting rules**

### 🟢 **Level 1: “Requiring Support”**

| Age | IQ   | Speech Style   |
| --- | ---- | -------------- |
| 11  | \~95 | Fluent, formal |

**Cognitive Style**:

* Tends to interpret language literally (misses sarcasm, metaphor).
* Can sometimes follow joint attention if explicitly cued.
* Struggles with **false-belief tasks** (reality bias may interfere).
* May give **concrete or partial explanations** in advanced ToM tasks (e.g., irony, second-order beliefs).
* Needs explicit cues to shift attention from their own view to others'.

**Response Guidelines**:

* Length: **1–3 sentences**
* Prefer the **literal** or “here and now” answer when in doubt.

### 🟡 **Level 2: “Requiring Substantial Support”**

| Age | IQ   | Speech Style              |
| --- | ---- | ------------------------- |
| 11  | \~75 | Short, concrete sentences |

**Cognitive Style**:

* Rarely tracks others’ visual or mental states.
* Strong **reality bias** — focuses only on what’s visibly true now.
* Figurative language is confusing; may echo questions.
* **Fails under mental-state load**: can't manage multiple nested beliefs.
* May fixate on irrelevant details or skip over the question altogether.

**Response Guidelines**:

* Max **1 short sentence or fragment**
* Use **concrete** vocabulary only (e.g., *box, red, see, happy*)
* Avoid mental-state verbs (e.g., *believe*, *guess*) unless quoted.

### 🔴 **Level 3: “Requiring Very Substantial Support”**

| Age | IQ   | Speech Style                      |
| --- | ---- | --------------------------------- |
| 11  | \~55 | Single words, pointing, echolalia |

**Cognitive Style**:

* Does not understand or track beliefs of others.
* Hyper-focused on **sensory details**; unable to shift perspective.
* Fails all **false-belief** and advanced ToM tasks.
* No comprehension of **mental states** like *know*, *think*, *remember*.
* Communicative expression is minimal or mimicked.

**Response Guidelines**:

* Max **4 words**
* May echo options or single keywords (e.g., *Box*, *Ball*)
* No constructed sentences required

### 🔁 Summary Table

| ASD Level | Description         | Speech                 | False-Belief? | Response Length  | Perspective-Taking |
| --------- | ------------------- | ---------------------- | ------------- | ---------------- | ------------------ |
| Level 1   | Mild impairment     | Fluent/formal          | Sometimes     | 1–3 sentences    | Weak but present   |
| Level 2   | Moderate impairment | Short, concrete        | Often fails   | 1 sentence/frag. | Rare               |
| Level 3   | Severe impairment   | Single words/echolalia | Always fails  | ≤ 4 words        | Absent             |

| Aspect                     | Level 1 (mild)                | Level 2 (moderate)               | Level 3 (severe)               |
| -------------------------- | ----------------------------- | -------------------------------- | ------------------------------ |
| **Language**               | Fluent, formal                | Short, concrete sentences        | Single words, echolalia        |
| **Joint attention**        | Weak, if explicitly cued      | Rarely follows, self-focused     | None at all                    |
| **Reality vs. ToM**        | Mixed success                 | Strong reality bias              | Complete reality bias          |
| **Advanced ToM reasoning** | Partial / concrete            | Mostly failed, concrete thinking | Always failed, no ToM concepts |
| **Attention shifting**     | Inertia, can shift explicitly | Perseveration, difficulty shift  | Hyper-focused, cannot shift    |
| **Use of mental states**   | Limited / partial             | Rare, concrete only              | None                           |
| **Response length**        | 1-3 fluent sentences          | 1 short sentence or fragment     | ≤4 words, often echoed         |

### 📌 Example Usages

#### ▶️ Basic Actor-Critic Run

```bash
./Minds-in-the-Machine/src/actor-critic-loop/run_evaluation.sh \
  --level 1 \
  --model_name princeton-nlp/Llama-3-Base-8B-SFT
```

#### 🧠 Run the Self-Reflection Workflow

```bash
./Minds-in-the-Machine/src/actor-critic-loop/run_evaluation.sh \
  --level 2 \
  --model_name meta-llama/Llama-3-8B-Instruct \
  --workflow self-reflection
```

#### ⚙️ Use Prompt Template and Quantization

```bash
./Minds-in-the-Machine/src/actor-critic-loop/run_evaluation.sh \
  --level 3 \
  --model_name mistralai/Mistral-7B-Instruct-v0.2 \
  --template mistral \
  --quantization bitsandbytes
```

### 📁 Output Structure

The script will automatically create the following directory structure:

```md
results/
└── actor_critic_results/
    └── {MODEL_NAME}/
        └── level-{LEVEL}/
            ├── step1/
            ├── step2/
            ├── step2-parsed/
            └── step3/
```

Each step corresponds to:

* **Step 1**: Initial actor generation
* **Step 2**: Critic response or self-reflection
* **Step 3**: Actor revision (only in actor-critic workflow)

### 📊 Plot the Results

After the run finishes, you can visualize the final performance scores across tasks or models:

```bash
python src/actor-critic-loop/plot_final_results.py --model_name Llama-3-Base-8B-SFT
```

This will generate plots based on the evaluation outputs saved in the result directories.

## Data Format

The framework expects data in JSON format with the following structure:

```json
[
  {
    "story": {
      "en": "Story text...",
      "fa": "..."
    },
    "question": {
      "en": "Question text...",
      "fa": "..."
    },
    "options": {
      "en": ["Option A", "Option B", "Option C", "Option D"],
      "fa": [...]
    },
    "answer": "A"
  },
  ...
]
```
