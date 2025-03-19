# EFEasVFE

This code base is using the [Julia Language](https://julialang.org/) and
[DrWatson](https://juliadynamics.github.io/DrWatson.jl/stable/)
to make a reproducible scientific project named
> EFEasVFE

It is authored by wouterwln.

To (locally) reproduce this project, do the following:

0. Download this code base. Notice that raw data are typically not included in the
   git-history and may need to be downloaded independently.
1. Open a Julia console and do:
   ```
   julia> using Pkg
   julia> Pkg.add("DrWatson") # install globally, for using `quickactivate`
   julia> Pkg.activate("path/to/this/project")
   julia> Pkg.instantiate()
   ```

This will install all necessary packages for you to be able to run the scripts and
everything should work out of the box, including correctly finding local paths.

You may notice that most scripts start with the commands:
```julia
using DrWatson
@quickactivate "EFEasVFE"
```
which auto-activate the project and enable local path handling from DrWatson.

## Experiments

This project implements two main experiments:

### 1. Stochastic Maze

A grid-world environment with stochastic transitions and noisy observations. The agent must navigate to a goal state while dealing with:
- Stochastic transitions in certain states
- Noisy observations in specific locations
- Sink states that trap the agent
- Multiple possible paths to the goal

Run the experiment:
```julia
julia scripts/stochastic_maze.jl
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

To run this experiment, you need to:

1. Start the Python FastAPI server:
```bash
cd src/environments
uvicorn minigrid_environment:app --reload
```

2. In a separate terminal, run the Julia experiment:
```julia
julia scripts/minigrid.jl
```

## Python Setup

This project supports both Julia and Python implementations. The Python infrastructure is organized as follows:

### Directory Structure
- `src/python/`: Python module containing shared code
  - `environments/`: Environment implementations
  - `models/`: Agent and model implementations
- `scripts/python/`: Python experiment scripts
- `requirements.txt`: Python dependencies

### Getting Started with Python

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Unix/macOS
# or
.\venv\Scripts\activate  # On Windows
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run experiments:
```bash
python scripts/python/stochastic_maze.py
```

### Development

The Python codebase follows these conventions:
- Use Python 3.8+
- Format code with `black`
- Use type hints where possible
- Follow the same experiment structure as Julia implementations

## Project Structure

The project is organized as follows:

```
.
├── src/
│   ├── environments/     # Environment implementations
│   │   ├── minigrid.jl  # MiniGrid environment in Julia
│   │   └── stochastic_maze.jl
│   └── models/          # Agent and model implementations
│       └── minigrid/    # MiniGrid-specific models
├── scripts/
│   ├── minigrid.jl      # MiniGrid experiments
│   └── stochastic_maze.jl
└── src/python/          # Python implementations
    └── environments/
        └── minigrid_environment.py  # FastAPI server for MiniGrid
```

## Dependencies

### Julia
- DrWatson
- ReactiveMP
- HTTP
- JSON
- ProgressMeter
- RxEnvironmentsZoo.GLMakie (for visualization)

### Python
- gymnasium
- minigrid
- fastapi
- pydantic
- uvicorn (for FastAPI server)
