using HTTP
using JSON

export reset_environment, step_environment, get_action_space, EnvironmentError

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
    reset_environment()

Reset the environment to its initial state.

# Returns
- Dictionary containing the initial environment state

# Throws
- `EnvironmentError` if the reset request fails
"""
function reset_environment()
    response = HTTP.request("GET", "$API_URL/reset", retry=true)
    return check_response(response)
end

"""
    step_environment(action::Int)

Execute an action in the environment.

# Arguments
- `action::Int`: The action to execute (0-6)

# Returns
- Dictionary containing the new environment state and reward

# Throws
- `EnvironmentError` if the step request fails
"""
function step_environment(action::Int)
    for attempt in 1:3  # 3 retries
        try
            response = HTTP.request(
                "POST",
                "$API_URL/step",
                ["Content-Type" => "application/json"],
                JSON.json(Dict("action" => action));
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
    reinitialize_environment(grid_size::Int; render_mode::Union{String,Nothing}="human")

Reinitialize the environment with a new grid size.

# Arguments
- `grid_size::Int`: The new grid size
- `render_mode::Union{String,Nothing}`: The rendering mode ("human" or nothing for no rendering)

# Returns
- Dictionary containing the new environment state

# Throws
- `EnvironmentError` if the reinitialize request fails
"""
function reinitialize_environment(grid_size::Int; render_mode::String="human")
    response = HTTP.request(
        "POST",
        "$API_URL/reinitialize",
        ["Content-Type" => "application/json"],
        JSON.json(Dict(
            "grid_size" => grid_size,
            "render_mode" => render_mode
        ))
    )
    return check_response(response)
end

"""
    get_action_space()

Get information about the available actions in the environment.

# Returns
- Dictionary containing action space information

# Throws
- `EnvironmentError` if the request fails
"""
function get_action_space()
    response = HTTP.request("GET", "$API_URL/action_space")
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
    close_environment()

Close the environment and clean up resources.

# Throws
- `EnvironmentError` if the close request fails
"""
function close_environment()
    response = HTTP.request("POST", "$API_URL/close")
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