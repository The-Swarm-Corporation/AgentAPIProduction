# Swarms Agent API

[![Join our Discord](https://img.shields.io/badge/Discord-Join%20our%20server-5865F2?style=for-the-badge&logo=discord&logoColor=white)](https://discord.gg/agora-999382051935506503) [![Subscribe on YouTube](https://img.shields.io/badge/YouTube-Subscribe-red?style=for-the-badge&logo=youtube&logoColor=white)](https://www.youtube.com/@kyegomez3242) [![Connect on LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/kye-g-38759a207/) [![Follow on X.com](https://img.shields.io/badge/X.com-Follow-1DA1F2?style=for-the-badge&logo=x&logoColor=white)](https://x.com/kyegomezb)

A production-grade REST API for deploying and managing AI agents in the cloud. This API provides a scalable and secure way to create, manage, and interact with AI agents.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Docker](https://img.shields.io/badge/docker-supported-brightgreen.svg)](https://www.docker.com/)

## Features

- 🚀 Create and manage multiple AI agents
- 🔄 Process completions with rate limiting and error handling
- 📊 Track agent metrics and performance
- 🔒 Built-in security features
- 🎯 Horizontal scaling support
- 🛠️ Easy deployment with Docker
- 💫 Graceful shutdown handling
- 🔍 Comprehensive logging
- ⚡ High-performance FastAPI backend

## Quick Start

### Prerequisites

- Python 3.11+
- Docker (optional)
- OpenAI API key (or other LLM provider key)

### Local Development Setup

1. Clone the repository:
```bash
git clone https://github.com/The-Swarm-Corporation/AgentAPIProduction.git
cd agent-api
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: .\venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Create a `.env` file:
```env
OPENAI_API_KEY=your_api_key_here
PORT=8080
WORKSPACE_DIR=agent_workspace
```

5. Run the server:
```bash
python -m uvicorn api.api:app --host 0.0.0.0 --port 8080 --reload
```

### Docker Deployment

1. Build the Docker image:
```bash
docker build -t agent-api .
```

2. Run the container:
```bash
docker run -d \
  -p 8080:8080 \
  -e OPENAI_API_KEY=your_api_key_here \
  -e PORT=8080 \
  --name agent-api \
  agent-api
```

## API Documentation

### Base URL
```
http://localhost:8080/v1
```

### Endpoints

#### Create Agent
```http
POST /v1/agent
```
```json
{
  "agent_name": "string",
  "model_name": "string",
  "description": "string",
  "system_prompt": "string",
  "temperature": 0.1,
  "max_loops": 1
}
```

#### List Agents
```http
GET /v1/agents
```

#### Process Completion
```http
POST /v1/agent/completions
```
```json
{
  "prompt": "string",
  "agent_id": "uuid",
  "max_tokens": null,
  "temperature_override": 0.5
}
```

#### Get Agent Metrics
```http
GET /v1/agent/{agent_id}/metrics
```

For full API documentation, visit `/v1/docs` or `/v1/redoc` after starting the server.

## Rate Limiting

The API implements rate limiting with the following defaults:
- 2 requests per second
- 30 requests per minute
- 1000 requests per hour
- 10000 requests per day

## Deployment Considerations

### Production Environment

For production deployment, consider:

1. Using a reverse proxy (e.g., Nginx)
2. Implementing SSL/TLS
3. Setting up monitoring (e.g., Prometheus + Grafana)
4. Using environment-specific configuration
5. Implementing proper backup strategies

Example `docker-compose.yml` for production:

```yaml
version: '3.8'

services:
  agent-api:
    build: .
    ports:
      - "8080:8080"
    environment:
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - PORT=8080
    volumes:
      - agent_data:/app/agent_workspace
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8080/health"]
      interval: 30s
      timeout: 3s
      retries: 3
    restart: unless-stopped

volumes:
  agent_data:
```

### Cloud Deployment

The API can be deployed to various cloud providers:

#### AWS Elastic Beanstalk
```bash
eb init -p docker agent-api
eb create production
```

#### Google Cloud Run
```bash
gcloud builds submit --tag gcr.io/PROJECT_ID/agent-api
gcloud run deploy agent-api --image gcr.io/PROJECT_ID/agent-api
```

#### Azure Container Instances
```bash
az container create \
  --resource-group myResourceGroup \
  --name agent-api \
  --image your-registry.azurecr.io/agent-api \
  --dns-name-label agent-api \
  --ports 8080
```

## Error Handling

The API implements comprehensive error handling:
- Rate limit exceeded: 429
- Not found: 404
- Internal server error: 500

Error responses follow this format:
```json
{
  "detail": {
    "error": "string",
    "message": "string"
  }
}
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Support

For support and questions:
- Open an issue on GitHub
- Join our Discord community
- Email support@yourdomain.com

## Acknowledgments

- FastAPI for the excellent web framework
- Swarms for the agent implementation
- The open-source community

---

Made with ❤️ by [swarms.ai](https://www.swarms.ai)
