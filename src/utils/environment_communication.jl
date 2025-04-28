using HTTP
using JSON
using UUIDs

export create_environment, step_environment, close_environment, EnvironmentError

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
                status_exception=false  # Don't throw HTTP exceptions, handle them ourselves
            )

            # Check if response is successful (status code 200)
            if response.status != 200
                throw(EnvironmentError(
                    "Environment step request failed",
                    response.status,
                    String(response.body)
                ))
            end

            return JSON.parse(String(response.body))
        catch e
            if e isa EnvironmentError
                rethrow(e)
            elseif e isa HTTP.Exceptions.StatusError
                # Convert HTTP errors to EnvironmentError
                throw(EnvironmentError(
                    "Environment step request failed",
                    e.status,
                    String(e.response.body)
                ))
            elseif attempt < 3
                sleep(0.1 * 2^(attempt - 1))  # exponential backoff: 0.1s, 0.2s, 0.4s
                continue
            else
                # For other types of errors, wrap them in EnvironmentError
                throw(EnvironmentError(
                    "Environment communication failed",
                    -1,
                    string(e)
                ))
            end
        end
    end
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