import gymnasium as gym
import minigrid

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Any

# Initialize FastAPI app
app = FastAPI()

# Create environment
env = gym.make("MiniGrid-DoorKey-6x6-v0", render_mode="human")

# Initialize environment state
observation, info = env.reset()

class Action(BaseModel):
    action: int

@app.get("/reset")
async def reset_environment():
    """Reset the environment and return initial observation"""
    global observation, info
    observation, info = env.reset()
    observation = {
        "image": observation["image"].tolist(),
        "direction": int(observation["direction"])
    }
    return {
        "observation": observation,
        "info": info
    }

@app.post("/step")
async def step_environment(action: Action):
    """Take a step in the environment with the given action"""
    try:
        observation, reward, terminated, truncated, info = env.step(action.action)
        # Convert observation dictionary to serializable format
        observation = {
            "image": observation["image"].tolist(),
            "direction": int(observation["direction"])
        }
        return {
            "observation": observation,
            "reward": reward,
            "terminated": terminated,
            "truncated": truncated,
            "info": info
        }
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