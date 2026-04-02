# 🚀 Space Mission Control RL Environment

An OpenAI Gymnasium-compatible Reinforcement Learning environment where an AI agent acts as a **Mission Control Operator**. The agent must process complex sensor data and make critical, high-stakes decisions to ensure mission success and crew safety.

## 🧠 Important Clarification

This project does NOT build or train an AI model.

It implements a Reinforcement Learning (RL) environment where external agents interact, make decisions, and receive rewards.

This environment defines:
- tasks
- evaluation logic
- reward system

Any AI agent can be plugged into this environment for training.

## ✅ OpenEnv Compliance

This project satisfies:

- Gym-style API (reset, step)
- Task generation (20–50 tasks)
- Automated grading system
- Reward logic
- Docker execution
- Inference script for evaluation
- Structured logging output

## 🚀 Inference Script (Evaluation Entry Point)

This project includes `inference.py`, which is the required execution script for automated evaluation.

It:
- initializes the environment
- runs a full episode
- simulates an agent
- prints structured logs

### Run:

```bash
python inference.py
```

## 📊 Structured Output Format

The inference script prints logs in the required format:

```text
[START]
[STEP] step=1 action=(...) reward=... cumulative=...
[STEP] step=2 action=(...) reward=... cumulative=...
...
[END]
```

This format is required for automated scoring.

## 🔧 Environment Variables

The system supports the following variables (for compliance):

- `API_BASE_URL`
- `MODEL_NAME`
- `HF_TOKEN`

These are optional and not required for basic execution.

## 📁 Data Requirement

This project does NOT require external datasets.

- Works with SQLite + JSON if provided
- Automatically falls back to synthetic data if not

This ensures portability and reproducibility.

## 🕹️ Environment Specifications

### 🎮 Action Space

The agent provides a multi-discrete action: `(Category, Priority, Decision)`

| Component | `0` | `1` | `2` | 
| ----- | ----- | ----- | ----- | 
| **Category** | System Failure | Navigation | Resource Management | 
| **Priority** | Low | Medium | High | 
| **Decision** | Continue | Adjust | Abort | 

### 🧪 Observation Space

The observation returns a dictionary containing task metadata. Detailed mission descriptions are provided via the `info` dictionary to facilitate NLP-based processing.

```json
{
  "difficulty": 4,
  "task_index": 12
}
```

### 🏆 Reward System

Rewards are dynamically scaled by task difficulty: `Reward = Base Reward × (Difficulty / 3)`

| Score | Meaning | Base Reward | 
| ----- | ----- | ----- | 
| **3** | Perfect Match | `+10` | 
| **2** | Partial Match | `+5` | 
| **1** | Weak Match | `+2` | 
| **0** | Incorrect Action | `-3` | 

## 🧱 Architecture & Workflow

### Environment Loop

1. **`reset()`**: Environment initializes with a new set of tasks.
2. **Task Generation**: A scenario is pulled or synthetically generated.
3. **Agent Action**: The RL agent takes an action.
4. **Evaluation**: The `Evaluator` checks the action's correctness.
5. **Reward Computation**: The reward is calculated and returned.
6. **Iteration**: Moves to the next task until the episode is `done`.

## 🚀 Quick Start

### Local Setup

```bash
# Clone the repository
git clone [https://github.com/samraddha/space-mission-rl.git](https://github.com/samraddha/space-mission-rl.git)
cd space-mission-rl

# Install dependencies
pip install -r requirements.txt

# Run the inference evaluation script
python inference.py

# Run the full test suite
python test_system.py
```

### 🐳 Docker

Build:

```bash
docker build -t space-mission-rl .
```

Run:

```bash
docker run --rm space-mission-rl
```

## 🔧 API Usage (For RL Training)

Integrating this environment into your RL pipeline is standard and straightforward:

```python
from task_builder import build_tasks
from env import SpaceMissionEnv

# Initialize 20 to 50 tasks for the episode
tasks = build_tasks([], min_tasks=20, max_tasks=50)
env = SpaceMissionEnv(tasks)

obs, info = env.reset()
done = False

while not done:
    # Your agent's logic goes here
    action = env.action_space.sample() 
    
    # Standard Gym step
    obs, reward, done, truncated, info = env.step(action)
```

## 📁 Project Structure

```text
space-mission-rl/
├── inference.py         # Required execution script for evaluation
├── main.py              # CLI Entry point & Demo agents
├── env.py               # Core Gymnasium Environment
├── data_loader.py       # SQLite & JSON ingestion logic
├── task_builder.py      # Scenario/Task generation engine
├── evaluator.py         # Action grading logic (0–3)
├── rewards.py           # Reward computation math
├── test_system.py       # Comprehensive System Tests
├── requirements.txt     # Python dependencies
└── Dockerfile           # Containerized execution
```

## 📊 Test Results

```text
Total Tests   : 160+
Failures      : 0
Errors        : 0
```

**✅ 100% PASS — Production Ready**

**Author:** Samraddha Shrivastava

**License:** [MIT License](https://opensource.org/licenses/MIT)
