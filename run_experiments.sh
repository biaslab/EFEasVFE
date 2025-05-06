#!/bin/bash

# Exit on error
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
API_PROCESS_FILE="/tmp/minigrid_api_pid"

# Function to run Julia scripts
run_julia_script() {
    local script=$1
    local threads=$2
    local params=$3
    
    echo "Running $script with $threads threads and parameters: $params"
    cd "$SCRIPT_DIR"
    JULIA_NUM_THREADS=$threads julia --project=. "scripts/$script.jl" $params
}

# Function to start the API server
start_api_server() {
    echo "Starting Minigrid API server..."
    
    # Check if API is already running
    if [ -f "$API_PROCESS_FILE" ]; then
        PID=$(cat "$API_PROCESS_FILE")
        if ps -p "$PID" > /dev/null; then
            echo "API server already running with PID $PID"
            return 0
        else
            echo "Stale PID file found, will start a new API server"
            rm -f "$API_PROCESS_FILE"
        fi
    fi
    
    # Start API server
    cd "$SCRIPT_DIR/src/environments"
    nohup uvicorn minigrid_environment:app --reload > /tmp/minigrid_api.log 2>&1 &
    PID=$!
    echo $PID > "$API_PROCESS_FILE"
    echo "API server started with PID $PID"
    
    # Wait for API to be ready
    echo "Waiting for API server to start..."
    sleep 5
    cd "$SCRIPT_DIR"
}

# Function to stop the API server
stop_api_server() {
    if [ -f "$API_PROCESS_FILE" ]; then
        PID=$(cat "$API_PROCESS_FILE")
        if ps -p "$PID" > /dev/null; then
            echo "Stopping API server with PID $PID"
            kill "$PID"
            rm -f "$API_PROCESS_FILE"
        else
            echo "API server not running (stale PID file)"
            rm -f "$API_PROCESS_FILE"
        fi
    else
        echo "No API server running"
    fi
}

# Main execution logic
case "$1" in
    start_api)
        start_api_server
        ;;
    stop_api)
        stop_api_server
        ;;
    minigrid)
        # Ensure API is running
        start_api_server
        run_julia_script "minigrid" "$2" "$3"
        ;;
    debug_minigrid)
        # Ensure API is running
        start_api_server
        run_julia_script "debug_minigrid" "$2" "$3"
        ;;
    stochastic_maze)
        run_julia_script "stochastic_maze" "$2" "$3"
        ;;
    debug_stochastic_maze)
        run_julia_script "debug_stochastic_maze" "$2" "$3"
        ;;
    all)
        # Run all experiments
        start_api_server
        run_julia_script "minigrid" "$2" "${3:-"--save-results --parallel --save-video --n-iterations 40 --n-episodes 200 --time-horizon 25 --grid-size 4"}"
        run_julia_script "debug_minigrid" "$2" "${4:-"--grid-size 4 --time-horizon 25 --save-frame --iterations 40 --save-animation"}"
        run_julia_script "stochastic_maze" "$2" "${5:-"-r --save-results"}"
        run_julia_script "debug_stochastic_maze" "$2" "${6:-"--save-frame --iterations 50"}"
        ;;
    *)
        echo "Usage: $0 {start_api|stop_api|minigrid|debug_minigrid|stochastic_maze|debug_stochastic_maze|all} [threads] [parameters]"
        exit 1
        ;;
esac

exit 0 