# EFEasVFE



## Prerequisites

This project requires:
- [Julia](https://julialang.org/) (v1.10+)
- [Python](https://www.python.org/) (v3.8+)
- [uv](https://github.com/astral-sh/uv) - Ultra-fast Python package installer and resolver


## Experiments

This project implements three main experiments:

### 1. Stochastic Maze

A grid-world environment with stochastic transitions and noisy observations. The agent must navigate to a goal state while dealing with:
- Stochastic transitions in certain states
- Noisy observations in specific locations
- Sink states that trap the agent
- Multiple possible paths to the goal

Run the experiment:
```bash
make stochastic_maze
```

Debug the stochastic maze environment:
```bash
make debug_stochastic_maze
```

### 2. MiniGrid DoorKey

A door-key environment where the agent must:
- Find and pick up a key
- Navigate to and unlock a door
- Reach the goal state

The environment features:
- Partial observability (7x7 field of view)
- Multiple objects (walls, doors, keys)
- State-dependent transitions
- Complex goal structure

To run this experiment:

```bash
# Start the Python FastAPI server and run the experiment
make minigrid
```

Debug the minigrid environment:
```bash
make debug_minigrid
```

### 3. T-Maze

A T-shaped maze environment where the agent must:
- Navigate through a T-shaped corridor
- Make decisions at the junction based on reward observations
- Reach the correct goal state

Run the experiment:
```bash
julia scripts/tmaze_experiments.jl
```

Parameters can be customized:
```bash
julia scripts/tmaze_experiments.jl --time-horizon 6 --n-episodes 50 --n-iterations 20 --record-episode --save-results
```

### Running Experiments with Make

For convenience, a Makefile is provided to run all experiments:

```bash
# Run all experiments
make all

# Run specific experiments
make minigrid
make debug_minigrid
make stochastic_maze
make debug_stochastic_maze

# Start the Minigrid API server
make start_api

# Clean generated files
make clean

# Show help
make help
```

The Makefile automatically detects the number of available CPU cores and configures Julia to use an appropriate number of threads.

### Running Experiments with Shell Script

You can also use the run_experiments.sh script directly:

```bash
# Start the API server
./run_experiments.sh start_api

# Run specific experiments
./run_experiments.sh minigrid [threads] [parameters]
./run_experiments.sh debug_minigrid [threads] [parameters]
./run_experiments.sh stochastic_maze [threads] [parameters]
./run_experiments.sh debug_stochastic_maze [threads] [parameters]

# Stop the API server
./run_experiments.sh stop_api
```

### Experiment Parameters

Each experiment supports various command-line parameters:

#### MiniGrid Parameters
```
--grid-size         Grid size for the environment (default: 3)
--time-horizon      Maximum steps per episode (default: 15)
--n-episodes        Number of episodes to run (default: 10)
--n-iterations      Iterations per step (default: 70)
--wait-time         Time to wait between steps (default: 0.0)
--number-type       Number type to use (default: Float32)
--visualize         Enable visualization
--save-results      Save experiment results
--verbosity         Logging level (debug, info, warn)
--seed              Random seed
--experiment-name   Custom name for the experiment
--save-video        Save video of the last episode
--sparse-tensor     Use sparse tensor representation
--parallel          Enable parallel execution
```

#### Stochastic Maze Parameters
```
--time-horizon      Maximum steps per episode (default: 10)
--n-episodes        Number of episodes to run (default: 100)
--n-iterations      Iterations per step (default: 40)
--wait-time         Time to wait between steps (default: 0.0)
--seed              Random seed
--record-episode    Record episode frames
--experiment-name   Custom name for the experiment
--tikz              Use PGFPlotsX backend for visualizations
--debug             Enable debug mode
--save-results      Save experiment results
--show-legend       Show legend in visualizations
```

#### T-Maze Parameters
```
--time-horizon      Maximum steps per episode (default: 6)
--n-episodes        Number of episodes to run (default: 50)
--n-iterations      Iterations per step (default: 20)
--wait-time         Time to wait between steps (default: 0.0)
--seed              Random seed
--record-episode    Record episode frames
--experiment-name   Custom name for the experiment
--tikz              Use PGFPlotsX backend for visualizations
--debug             Enable debug mode
--save-results      Save experiment results
```

### Visualization

All environments include comprehensive visualization tools:

- **Belief visualization**: Visualize agent beliefs about states, actions, and free energy
- **Animation**: Create animations of belief evolution over time
- **TikZ export**: Save plots in both PNG and TikZ formats for publication-quality figures

Example visualization options:
```bash
# For Minigrid
julia scripts/debug_minigrid.jl --save-animation

# For Stochastic Maze
julia scripts/debug_stochastic_maze.jl --save-frame
```

## Python Setup

This project supports both Julia and Python implementations. The Python infrastructure is organized as follows:

### Directory Structure
- `src/environments/`: Environment implementations including the minigrid FastAPI server
- `pyproject.toml`: Python dependencies configuration

### Getting Started with Python

1. Create a Python environment using uv:
```bash
uv venv .venv
source .venv/bin/activate  # On Unix/macOS
# or
.\.venv\Scripts\activate  # On Windows
```

2. Install dependencies:
```bash
uv pip install -e .
```

3. Run the minigrid environment server:
```bash
cd src/environments
uv run -m uvicorn minigrid_environment:app --reload
```

### Python Dependencies

The project depends on the following Python packages:
- fastapi
- gymnasium
- minigrid
- pydantic
- uvicorn

## Project Structure

The project is organized as follows:

```
.
├── scripts/
│   ├── minigrid.jl              # MiniGrid experiment
│   ├── debug_minigrid.jl        # Debug version of MiniGrid experiment
│   ├── stochastic_maze.jl       # Stochastic Maze experiment
│   ├── debug_stochastic_maze.jl # Debug version of Stochastic Maze experiment
│   └── tmaze_experiments.jl     # T-Maze experiment
├── src/
│   ├── environments/            # Environment implementations
│   │   └── minigrid_environment.py  # FastAPI server for MiniGrid
│   └── models/                  # Agent and model implementations
├── Makefile                     # Targets for running experiments
├── run_experiments.sh           # Helper script for running experiments
└── pyproject.toml               # Python dependencies
```

## Dependencies

### Julia
- DrWatson
- RxInfer
- ReactiveMP
- HTTP
- JSON
- ProgressMeter
- Plots
- PGFPlotsX
- StableRNGs
- Distributions
- ArgParse
- And various other packages as specified in the Project.toml file

### Python
- fastapi
- gymnasium
- minigrid
- pydantic
- uvicorn
