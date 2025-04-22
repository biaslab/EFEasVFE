using HTTP
using JSON
using UUIDs

export reset_environment, step_environment, get_action_space, EnvironmentError, create_environment, close_environment

const API_URL = "http://localhost:8000"

"""
    EnvironmentError <: Exception

Custom exception type for environment communication errors.
"""
struct EnvironmentError <: Exception
    message::String
    status::Int
    response::String
end

Base.showerror(io::IO, e::EnvironmentError) = print(io, "EnvironmentError: $(e.message) (Status: $(e.status))\nResponse: $(e.response)")

"""
    check_response(response::HTTP.Response)

Check if the HTTP response is successful and return the parsed JSON body.
Throws an EnvironmentError if the request failed.
"""
function check_response(response::HTTP.Response)
    if response.status != 200
        throw(EnvironmentError(
            "Environment request failed",
            response.status,
            String(response.body)
        ))
    end
    return JSON.parse(String(response.body))
end

"""
    create_environment(grid_size::Int; render_mode::String="rgb_array", seed::UInt32=42)

Create a new environment instance with a unique session ID.

# Arguments
- `grid_size::Int`: The grid size
- `render_mode::String`: The rendering mode ("human" or "rgb_array")
- `seed::UInt32`: Random seed for environment initialization

# Returns
- Dictionary containing the initial environment state and session_id

# Throws
- `EnvironmentError` if the creation request fails
"""
function create_environment(grid_size::Int; render_mode::String="rgb_array", seed::UInt32=42)
    response = HTTP.request(
        "POST",
        "$API_URL/create",
        ["Content-Type" => "application/json"],
        JSON.json(Dict(
            "grid_size" => grid_size,
            "render_mode" => render_mode,
            "seed" => seed
        ))
    )
    return check_response(response)
end

"""
    reset_environment(session_id::String)

Reset the environment for a specific session.

# Arguments
- `session_id::String`: The session ID

# Returns
- Dictionary containing the reset environment state

# Throws
- `EnvironmentError` if the reset request fails
"""
function reset_environment(session_id::String)
    response = HTTP.request(
        "POST",
        "$API_URL/reset",
        ["Content-Type" => "application/json"],
        JSON.json(Dict("session_id" => session_id));
        retry=true
    )
    return check_response(response)
end

"""
    step_environment(action::Int, session_id::String)

Execute an action in the environment.

# Arguments
- `action::Int`: The action to execute (0-6)
- `session_id::String`: The session ID

# Returns
- Dictionary containing the new environment state and reward

# Throws
- `EnvironmentError` if the step request fails
"""
function step_environment(action::Int, session_id::String)
    for attempt in 1:3  # 3 retries
        try
            response = HTTP.request(
                "POST",
                "$API_URL/step",
                ["Content-Type" => "application/json"],
                JSON.json(Dict(
                    "action" => action,
                    "session_id" => session_id
                ));
                retry=false,  # we'll handle retries ourselves
                connect_timeout=5,  # 5 second connection timeout
            )
            return check_response(response)
        catch e
            if attempt < 3
                sleep(0.1 * 2^(attempt - 1))  # exponential backoff: 0.1s, 0.2s, 0.4s
                continue
            end
            rethrow(e)
        end
    end
end

"""
    reinitialize_environment(grid_size::Int; render_mode::String="rgb_array", seed::UInt32=42)

Reinitialize the environment with a new grid size. (Alias for create_environment)

# Arguments
- `grid_size::Int`: The new grid size
- `render_mode::String`: The rendering mode ("human" or "rgb_array")
- `seed::UInt32`: Random seed for environment initialization

# Returns
- Dictionary containing the new environment state and session_id

# Throws
- `EnvironmentError` if the reinitialize request fails
"""
function reinitialize_environment(grid_size::Int; render_mode::String="rgb_array", seed::UInt32=42)
    # This is now an alias for create_environment for backward compatibility
    return create_environment(grid_size; render_mode=render_mode, seed=seed)
end

"""
    get_action_space(session_id::String)

Get information about the available actions in the environment.

# Arguments
- `session_id::String`: The session ID

# Returns
- Dictionary containing action space information

# Throws
- `EnvironmentError` if the request fails
"""
function get_action_space(session_id::String)
    response = HTTP.request(
        "POST",
        "$API_URL/action_space",
        ["Content-Type" => "application/json"],
        JSON.json(Dict("session_id" => session_id))
    )
    return check_response(response)
end

"""
    get_observation_space()

Get information about the observation space of the environment.

# Returns
- Dictionary containing observation space information

# Throws
- `EnvironmentError` if the request fails
"""
function get_observation_space()
    response = HTTP.request("GET", "$API_URL/observation_space")
    return check_response(response)
end

"""
    close_environment(session_id::String)

Close a specific environment session and clean up resources.

# Arguments
- `session_id::String`: The session ID

# Returns
- Dictionary containing success status

# Throws
- `EnvironmentError` if the close request fails
"""
function close_environment(session_id::String)
    response = HTTP.request(
        "POST",
        "$API_URL/close",
        ["Content-Type" => "application/json"],
        JSON.json(Dict("session_id" => session_id))
    )
    return check_response(response)
end

"""
    render_environment()

Render the current state of the environment.

# Returns
- Dictionary containing rendering information

# Throws
- `EnvironmentError` if the render request fails
"""
function render_environment()
    response = HTTP.request("GET", "$API_URL/render")
    return check_response(response)
end

# Maintain backward compatibility with non-session functions
reset_environment() = error("reset_environment without session_id is no longer supported. Use create_environment instead.")
step_environment(action::Int) = error("step_environment without session_id is no longer supported. Create an environment first.")
get_action_space() = error("get_action_space without session_id is no longer supported. Create an environment first.")