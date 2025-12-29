# MAPs Agents

Agent implementations for the **Mini Amusement Park Simulator (MAPs)** environment.

This package provides various AI agent implementations (Reinforcement Learning, Large Language Models, and Vision-Language Models) that can interact with the MAPs theme park management simulation environment.

## Features

- **Abstract Agent Interface**: Common interface for all agent types
- **RL Agents**: Stable-Baselines3 PPO agents with hierarchical action policies
- **LLM Agents**: ReAct-style agents using OpenRouter API
- **VLM Agents**: ReAct-style agents augmented with images using OpenRouter API
- **Evaluation Framework**: Unified evaluation system with resource tracking

## Installation

### Dependencies

The package requires the [MAPs](https://github.com/Skyfall-Research/MAPs) environment. Install it first using instructions in its README.md

```bash
cd MAPs_Agents
pip install -e .
```

### Install with Specific Features

```bash
# Install with RL dependencies only
pip install maps-agents[rl]

# Install with LLM dependencies only
pip install maps-agents[llm]

# Install with all dependencies
pip install maps-agents[all]

# Install for development
pip install maps-agents[dev]
```

## Quick Start

### Using RL Agents

There are two versions of RL agents that vary in obversation and action spaces.

The simple RL agent trades expressiveness for simplicity allowing for faster learning. The observation space is reduced to high-level aggregated statistics only, omitting spatial and entity specific information. The action space also has fewer parameters, which are filled using sub-optimal, but efficient and passable heuristics.

The full RL agent keeps nearly full expressiveness using a lossless observation. The only simplification is discretizing the order quantity into multiples of 25 to reduce the overall parameter space.

Both RL agents use a hierarchical cascading masking in order to prevent the prediction of invalid actions. This first begins by only allowing valid action types (e.g., you can only move/remove/modify if an existing attraction exists). Then, each parameter is filled in order. After each parameter is selected, later parameters are masked based on the output (e.g., if action place is selected, and type ride is selected, then the subtypes get masked to carousel, ferris wheel, and roller coaster, and the x/y coordinates are masked to empty tiles adjacent to a path).

Agents can be tuned a Bayesian hyperparameter optimization using Optuna [Akiba et al., 2019] with a Tree-structured Parzen Estimator sampler. This tunes 8 PPO hyperparameters: learning rate (log-uniform in [1e-5, 1e-3]), rollout steps (categorical: {512, 1024, 2048, 4096, 8192}), batch size (categorical: {32, 64, 128, 256, 512, 1024}), number of epochs (uniform in [3, 20]), discount factor γ (log-uniform in [0.95, 0.9999]), GAE λ (uniform in [0.9, 0.99]), clip range ε (uniform in [0.1, 0.4]), and entropy coefficient (log-uniform in [1e-8, 0.1]). We enforce the constraint batch_size ≤ n_steps. Each trial trains for n_timesteps timesteps with intermediate evaluation every 2,500 steps. Unpromising trials are pruned using median pruning (5 startup trials, 10 warmup steps). The objective maximizes mean episode reward on held-out test layouts, evaluated deterministically over 5 episodes. We perform [N] trials to identify optimal hyperparameters, which are then used to train final agents for 2.5M timesteps.  

  - Optuna: Akiba, T., Sano, S., Yanase, T., Ohta, T., & Koyama, M. (2019). Optuna: A next-generation hyperparameter optimization framework. In KDD.                                       
  - PPO: Schulman, J., Wolski, F., Dhariwal, P., Radford, A., & Klimov, O. (2017). Proximal policy optimization algorithms. arXiv:1707.06347.                                              
  - Hyperparameter Importance: Engstrom, L., Ilyas, A., Santurkar, S., Tsipras, D., Janoos, F., Rudolph, L., & Madry, A. (2020). Implementation matters in deep policy gradients: A case study on PPO and TRPO. In ICLR.     

### Training RL Agents

```bash
# Train a simple mode agent
python -m maps_agents.rl.train --mode simple --timesteps 1000000

# Train a full mode agent
python -m maps_agents.rl.train --mode full --timesteps 2000000 

# Tune an agent
python -m maps_agents.rl.tune_hyperparameters \
       --training-layouts all \
       --mode full \
       --difficulty easy \
       --n-trials 100 \
       --n-jobs 4
```



### Using LLM Agents

```python
from maps_agents import ReactAgent

# Initialize ReAct agent
agent = ReactAgent(
    model="anthropic/claude-3.5-sonnet",
    api_key="your-openrouter-key",
    horizon=250,
    difficulty="easy",
    actions_list=["place", "move", "remove", "modify", "wait"],
    config_path="llm/config.yaml"
)

# Generate action using LLM reasoning
action_str = agent.act(game_response, run_id=0)
```

### Evaluating Agents

```python
from maps_agents import evaluate_agent

# Evaluate your agent
results = evaluate_agent(
    agent=agent,
    config_path="eval_config.json",
    num_episodes=10
)

print(f"Average reward: {results['average_reward']}")
print(f"Total cost: {results['total_cost']}")
```

## Package Structure

```
maps_agents/
├── __init__.py           # Main package exports
├── eval/                 # Evaluation framework
│   ├── agent_interface.py       # AbstractAgent base class
│   ├── evaluator.py            # Agent evaluation logic
│   ├── resource_interface.py   # Resource tracking (tokens, compute)
│   ├── state_interface.py      # Game state representations
│   └── utils.py                # Utilities and config
├── rl/                   # Reinforcement Learning agents
│   ├── sb3_agent.py            # SB3Agent for inference
│   ├── sb3_test.py             # Training script
│   └── policies/               # Custom SB3 policies
│       ├── hierarchical_multidiscrete_policy.py
│       └── simple_hierarchical_policy.py
├── llm/                  # Large Language Model agents
│   ├── react.py                # ReAct agent implementation
│   └── config.yaml             # LLM configuration
└── vlm/                  # Vision-Language Model agents (future)
```

## Agent Types

### RL Agents (`SB3Agent`)

- Uses Stable-Baselines3 PPO algorithm
- Two modes:
  - **Simple**: 5 actions, vector observations only (faster training)
  - **Full**: 11 actions, grid + vector observations (better performance)
- Hierarchical action space for efficient learning
- Pre-trained models can be loaded for inference

### LLM Agents (`ReactAgent`)

- ReAct (Reasoning + Acting) pattern
- Uses OpenRouter API for various LLM backends
- Maintains conversation history for context
- Tracks token usage and costs
- Configurable temperature and model selection

## Configuration

### Agent Configuration

Agents can be configured via:
- Constructor parameters
- YAML config files (for LLM agents)
- EvalConfig objects (for evaluation)

### Evaluation Configuration

```python
from maps_agents import EvalConfig

config = EvalConfig(
    run_idx=0,
    observation_type="gym",  # or "gym_simple", "pydantic", "raw"
    difficulty="easy",       # or "medium", "hard"
    horizon=250,            # max steps per episode
    logging_id="experiment_1"
)
```

## Observation Modes

| Mode | Description | Use Case |
|------|-------------|----------|
| `gym` | Full gym observation (grid + vectors) | RL training with spatial awareness |
| `gym_simple` | Simplified vectors only | Faster RL training |
| `pydantic` | Structured Pydantic models | LLM agents, type safety |
| `raw` | Raw dictionary format | Debugging, custom agents |

## Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Run tests (`pytest`)
5. Format code (`black .`)
6. Commit changes (`git commit -m 'Add amazing feature'`)
7. Push to branch (`git push origin feature/amazing-feature`)
8. Open a Pull Request

## Development

### Setting up Development Environment

```bash
# Clone the repository
git clone https://github.com/yourusername/MAPs_Agents.git
cd MAPs_Agents

# Install in editable mode with dev dependencies
pip install -e .[dev]

# Run tests
pytest

# Format code
black .

# Type checking
mypy maps_agents
```

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=maps_agents --cov-report=html

# Run specific test file
pytest tests/test_sb3_agent.py
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use this package in your research, please cite:

```bibtex
@software{maps_agents,
  title = {MAPs Agents: Agent Implementations for Mini Amusement Park Simulator},
  author = {MAPs Team},
  year = {2024},
  url = {https://github.com/yourusername/MAPs_Agents}
}
```

## Acknowledgments

- Built on [Stable-Baselines3](https://github.com/DLR-RM/stable-baselines3) for RL agents
- Uses [OpenRouter](https://openrouter.ai/) for LLM API access
- Integrates with the MAPs environment (`map-py` package)

## Contact

For questions, issues, or contributions, please:
- Open an issue on GitHub
- Contact the maintainers at [email]

## Roadmap

- [ ] Vision-Language Model (VLM) agents
- [ ] Multi-agent coordination
- [ ] Curriculum learning for RL agents
- [ ] More LLM backends (direct API support)
- [ ] Offline RL training
- [ ] Web UI for agent visualization
- [ ] Benchmark suite with leaderboard
