import requests

# Base URL for the API
BASE_URL = "http://localhost:8080"

# Test health check endpoint
response = requests.get(f"{BASE_URL}/health")
print("Health Check Response:", response.json())

# Test creating a new agent
agent_config = {
    "agent_name": "test_agent",
    "model_name": "gpt-3.5-turbo",
    "description": "A test agent",
    "system_prompt": "You are a helpful assistant",
    "temperature": 0.5,
    "max_loops": 1,
    "dynamic_temperature_enabled": True,
    "user_name": "test_user",
    "retry_attempts": 1,
    "context_length": 200000,
    "output_type": "string",
    "streaming_on": False,
    "tags": ["test"],
    "stopping_token": "<DONE>",
    "auto_generate_prompt": False,
}

response = requests.post(f"{BASE_URL}/v1/agent", json=agent_config)
print("\nCreate Agent Response:", response.json())
agent_id = response.json()["agent_id"]

# Test getting rate limit status
response = requests.get(f"{BASE_URL}/v1/rate-limit-status")
print("\nRate Limit Status:", response.json())

# Test listing all agents
response = requests.get(f"{BASE_URL}/v1/agents")
print("\nList Agents Response:", response.json())

# Test updating an agent
update_data = {
    "description": "Updated test agent",
    "system_prompt": "Updated system prompt",
    "temperature": 0.7,
    "max_loops": 2,
    "tags": ["test", "updated"],
}

response = requests.patch(f"{BASE_URL}/v1/agent/{agent_id}", json=update_data)
print("\nUpdate Agent Response:", response.json())

# Test getting agent metrics
response = requests.get(f"{BASE_URL}/v1/agent/{agent_id}/metrics")
print("\nAgent Metrics Response:", response.json())

# Test creating a completion
completion_request = {
    "prompt": "Hello, how are you?",
    "agent_id": agent_id,
    "max_tokens": 100,
    "temperature_override": 0.8,
    "stream": False,
}

response = requests.post(f"{BASE_URL}/v1/agent/completions", json=completion_request)
print("\nCompletion Response:", response.json())

# Test cloning an agent
response = requests.post(
    f"{BASE_URL}/v1/agent/{agent_id}/clone", params={"new_name": "cloned_agent"}
)
print("\nClone Agent Response:", response.json())

# Test getting agent status
response = requests.get(f"{BASE_URL}/v1/agent/{agent_id}/status")
print("\nAgent Status Response:", response.json())

# Test batch completion status
batch_requests = [
    {
        "prompt": "What is 2+2?",
        "agent_id": agent_id,
        "max_tokens": 50,
        "temperature_override": 0.5,
        "stream": False,
    },
    {
        "prompt": "Who are you?",
        "agent_id": agent_id,
        "max_tokens": 50,
        "temperature_override": 0.5,
        "stream": False,
    },
]

response = requests.get(
    f"{BASE_URL}/v1/agent/batch/completions/status", json=batch_requests
)
print("\nBatch Completion Status Response:", response.json())

# Test deleting an agent
response = requests.delete(f"{BASE_URL}/v1/agent/{agent_id}")
print("\nDelete Agent Response:", response.json())

# Test listing agents with filters
response = requests.get(
    f"{BASE_URL}/v1/agents", params={"tags": ["test"], "status": "idle"}
)
print("\nFiltered Agents List Response:", response.json())
