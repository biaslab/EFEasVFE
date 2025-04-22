import gymnasium as gym
from gymnasium.envs.registration import register
import minigrid
import uuid
from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel
from typing import Dict, Any, Optional
import time

# Initialize FastAPI app
app = FastAPI()

# Store environments with unique session IDs
environments = {}

class Action(BaseModel):
    action: int
    session_id: str

class GridSize(BaseModel):
    grid_size: int
    render_mode: str = "human"  # Default to human rendering
    seed: int = 42
    session_id: Optional[str] = None  # Optional session ID, will be generated if not provided

class EnvSession(BaseModel):
    session_id: str

def get_environment(session_id: str):
    """Helper to retrieve an environment by session ID"""
    if session_id not in environments:
        raise HTTPException(status_code=404, detail=f"Environment session {session_id} not found")
    return environments[session_id]

def create_response_dict(observation, info, reward=0.0, terminated=False, truncated=False, frame=[], session_id=None):
    """Convert observation to serializable dictionary format.
    
    Args:
        observation: Raw observation from environment
        info: Additional info from environment
        reward: Optional reward value from step
        terminated: Optional termination flag from step
        truncated: Optional truncation flag from step
        frame: Optional frame from step
        session_id: Optional session ID
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
        "frame": frame,
        "session_id": session_id
    }       
    return response

@app.post("/create")
async def create_environment(config: GridSize):
    """Create a new environment instance with a unique session ID"""
    # Generate a session ID if not provided
    session_id = config.session_id or str(uuid.uuid4())
    
    try:
        # Register environment if it doesn't exist
        env_id = f"MiniGrid-DoorKey-{config.grid_size}x{config.grid_size}-v0"
        try:
            register(
                id=env_id,
                entry_point="minigrid.envs:DoorKeyEnv",
                kwargs={"size": config.grid_size},
            )
        except Exception:
            # Environment may already be registered
            pass
            
        # Create environment
        env = gym.make(env_id, render_mode=config.render_mode)
        observation, info = env.reset(seed=config.seed)
        
        # Store environment
        environments[session_id] = {
            "env": env,
            "observation": observation,
            "info": info,
            "last_access": time.time()
        }
        
        response = create_response_dict(observation, info, session_id=session_id)
        return response
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/reset")
async def reset_environment(session: EnvSession):
    """Reset the environment for a specific session"""
    env_data = get_environment(session.session_id)
    env = env_data["env"]
    
    observation, info = env.reset()
    env_data["observation"] = observation
    env_data["info"] = info
    env_data["last_access"] = time.time()
    
    response = create_response_dict(observation, info, session_id=session.session_id)
    return response

@app.post("/step")
async def step_environment(action: Action):
    """Take a step in the environment with the given action"""
    env_data = get_environment(action.session_id)
    env = env_data["env"]
    
    try:
        observation, reward, terminated, truncated, info = env.step(action.action)
        env_data["observation"] = observation
        env_data["info"] = info
        env_data["last_access"] = time.time()
        
        if env.render_mode == "rgb_array":
            frame = env.render().tolist()
        else:
            frame = []
            
        # Convert observation dictionary to serializable format
        response = create_response_dict(
            observation, info, reward, terminated, truncated, frame, 
            session_id=action.session_id
        )
        return response
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/action_space")
async def get_action_space(session: EnvSession):
    """Return information about the action space"""
    env_data = get_environment(session.session_id)
    env = env_data["env"]
    env_data["last_access"] = time.time()
    
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
        },
        "session_id": session.session_id
    }

@app.post("/reinitialize")
async def reinitialize_environment(grid_size: GridSize):
    """Reinitialize the environment with a new grid size and optional rendering mode"""
    # This is now an alias for create_environment for backward compatibility
    return await create_environment(grid_size)

@app.post("/close")
async def close_environment(session: EnvSession):
    """Close a specific environment session"""
    if session.session_id in environments:
        env_data = environments[session.session_id]
        env = env_data["env"]
        env.close()
        del environments[session.session_id]
        return {"success": True, "message": f"Environment {session.session_id} closed"}
    return {"success": False, "message": "Environment not found"}

# Cleanup task - periodically remove unused environments (could be set up with a background task)
@app.get("/cleanup")
async def cleanup_environments(timeout_minutes: int = 30):
    """Remove environments that haven't been accessed in a while"""
    now = time.time()
    timeout = timeout_minutes * 60
    closed_count = 0
    
    for session_id in list(environments.keys()):
        if now - environments[session_id]["last_access"] > timeout:
            environments[session_id]["env"].close()
            del environments[session_id]
            closed_count += 1
            
    return {"success": True, "closed_count": closed_count}
