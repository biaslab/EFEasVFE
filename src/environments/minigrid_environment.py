import gymnasium as gym
from gymnasium.envs.registration import register
import minigrid

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Any

# Initialize FastAPI app
app = FastAPI()


# Create environment
env = gym.make("MiniGrid-DoorKey-5x5-v0", render_mode="rgb_array")

# Initialize environment state
observation, info = env.reset()

class Action(BaseModel):
    action: int

class GridSize(BaseModel):
    grid_size: int
    render_mode: str = "human"  # Default to human rendering
    seed: int = 42

def create_response_dict(observation, info, reward=0.0, terminated=False, truncated=False, frame=[]):
    """Convert observation to serializable dictionary format.
    
    Args:
        observation: Raw observation from environment
        info: Additional info from environment
        reward: Optional reward value from step
        terminated: Optional termination flag from step
        truncated: Optional truncation flag from step
        frame: Optional frame from step
    Returns:
        dict: Serializable observation dictionary containing image, direction and optional step info
    """
    response = {
        "observation": {
            "image": observation["image"].tolist(),
            "direction": int(observation["direction"])
        },
        "info": info,
        "reward": reward,
        "terminated": terminated,
        "truncated": truncated,
        "frame": frame
    }       
    return response

@app.post("/reset")
async def reset_environment():
    global observation, info
    observation, info = env.reset()
    response = create_response_dict(observation, info)
    return response

@app.post("/step")
async def step_environment(action: Action):
    """Take a step in the environment with the given action"""
    try:
        observation, reward, terminated, truncated, info = env.step(action.action)
        if env.render_mode == "rgb_array":
            frame = env.render().tolist()
        else:
            frame = []
        # Convert observation dictionary to serializable format
        response = create_response_dict(observation, info, reward, terminated, truncated, frame)
        return response
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/action_space")
async def get_action_space():
    """Return information about the action space"""
    return {
        "n": env.action_space.n,
        "actions": {
            0: "left",
            1: "right",
            2: "forward",
            3: "pickup",
            4: "drop",
            5: "toggle",
            6: "done"
        }
    }

@app.post("/reinitialize")
async def reinitialize_environment(grid_size: GridSize):
    """Reinitialize the environment with a new grid size and optional rendering mode"""
    new_grid_size = grid_size.grid_size
    render_mode = grid_size.render_mode if grid_size.render_mode in ["human", "rgb_array"] else None
    global env, observation, info
    try:
        # Create environment
        register(
            id=f"MiniGrid-DoorKey-{new_grid_size}x{new_grid_size}-v0",
            entry_point="minigrid.envs:DoorKeyEnv",
            kwargs={"size": new_grid_size},
        )
        env = gym.make(f"MiniGrid-DoorKey-{new_grid_size}x{new_grid_size}-v0", render_mode=render_mode)
        observation, info = env.reset(seed=grid_size.seed)
        response = create_response_dict(observation, info)
        return response
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
