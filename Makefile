.PHONY: experiments minigrid debug_minigrid stochastic_maze debug_stochastic_maze clean start_api

# Detect number of CPU cores and use (cores - 2) for Julia threads, minimum 1
NPROC := $(shell nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)
JULIA_THREADS := $(shell expr $(NPROC) - 2 || echo 1)
# Ensure minimum of 1 thread
ifeq ($(shell expr $(JULIA_THREADS) \< 1), 1)
    JULIA_THREADS := 1
endif

# Define experiment parameters
MINIGRID_PARAMS := --save-results --parallel --save-video --n-iterations 50 --n-episodes 10 --time-horizon 25 --grid-size 4
DEBUG_MINIGRID_PARAMS := --grid-size 4 --time-horizon 25 --save-frame --iterations 40 --save-animation
STOCHASTIC_MAZE_PARAMS := -r --save-results
DEBUG_STOCHASTIC_MAZE_PARAMS := --save-frame --iterations 50

# Default target
all: experiments

# Target to run all experiments
experiments: start_api minigrid debug_minigrid stochastic_maze debug_stochastic_maze

# Start Minigrid API server
start_api:
	@echo "Starting Minigrid API server..."
	@./run_experiments.sh start_api

# Run Minigrid experiment
minigrid:
	@echo "Running Minigrid experiments..."
	@./run_experiments.sh minigrid $(JULIA_THREADS) "$(MINIGRID_PARAMS)"

# Run Debug Minigrid experiment
debug_minigrid:
	@echo "Running Debug Minigrid experiments..."
	@./run_experiments.sh debug_minigrid $(JULIA_THREADS) "$(DEBUG_MINIGRID_PARAMS)"

# Run Stochastic Maze experiment
stochastic_maze:
	@echo "Running Stochastic Maze experiments..."
	@./run_experiments.sh stochastic_maze $(JULIA_THREADS) "$(STOCHASTIC_MAZE_PARAMS)"

# Run Debug Stochastic Maze experiment
debug_stochastic_maze:
	@echo "Running Debug Stochastic Maze experiments..."
	@./run_experiments.sh debug_stochastic_maze $(JULIA_THREADS) "$(DEBUG_STOCHASTIC_MAZE_PARAMS)"

# Clean generated files
clean:
	@echo "Cleaning generated files..."
	@rm -rf data/results/*
	@rm -rf data/debug/*
	@find . -name "*.log" -type f -delete

# Help target
help:
	@echo "Available targets:"
	@echo "  all            - Run all experiments (default)"
	@echo "  experiments    - Run all experiments (alias for all)"
	@echo "  minigrid       - Run Minigrid experiments"
	@echo "  debug_minigrid - Run Debug Minigrid experiments"
	@echo "  stochastic_maze - Run Stochastic Maze experiments"
	@echo "  debug_stochastic_maze - Run Debug Stochastic Maze experiments"
	@echo "  start_api      - Start the Minigrid API server"
	@echo "  clean          - Clean generated files"
	@echo "  help           - Show this help message" 