using ColorSchemes

scheme = colorschemes[:Paired_9]

"""
    MAZE_THEME

A consistent color theme for maze environments.

# Fields
- `agent`: Color for the agent
- `cue`: Color for cue indicators
- `reward_positive`: Color for positive rewards
- `reward_negative`: Color for negative rewards
- `stochastic`: Color for stochastic elements (bridges, etc.)
- `obstacle`: Color for obstacles/walls
- `corridor`: Color for corridors/walkable areas
- `sink`: Color for sink states
- `noisy`: Color for states with noisy observations
- `wall`: Color for walls
- `background`: Background color
"""
const MAZE_THEME = (
    agent=scheme[2],
    cue=scheme[7],
    reward_positive=scheme[4],
    reward_negative=scheme[6],
    stochastic=scheme[8],
    corridor=:white,
    sink=scheme[5],
    noisy=:lightgray,
    wall=:black,
    background=:white
)