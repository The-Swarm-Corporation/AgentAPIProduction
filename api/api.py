import asyncio
import os
import signal
import time
import traceback
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, AsyncGenerator, Dict, List, Optional
from uuid import UUID, uuid4

import uvicorn
from dotenv import load_dotenv
from fastapi import (
    BackgroundTasks,
    FastAPI,
    HTTPException,
    Query,
    Request,
    status,
)
from fastapi.concurrency import asynccontextmanager
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger
from pydantic import BaseModel, Field
from swarms.structs.agent import Agent

load_dotenv()


@dataclass
class RateLimitWindow:
    requests: int
    start_time: float


class RateLimiter:
    def __init__(self):
        # Store rate limit windows per IP
        self.ip_windows: Dict[str, Dict[str, RateLimitWindow]] = defaultdict(dict)

        # Configure different rate limits
        self.limits = {
            "per_second": 2,  # 2 requests per second
            "per_minute": 30,  # 30 requests per minute
            "per_hour": 1000,  # 1000 requests per hour
            "per_day": 10000,  # 10000 requests per day
        }

        # Window durations in seconds
        self.windows = {
            "per_second": 1,
            "per_minute": 60,
            "per_hour": 3600,
            "per_day": 86400,
        }

    def _clean_old_windows(self, ip: str):
        """Remove expired windows for an IP"""
        current_time = time.time()
        for window_type in list(self.ip_windows[ip].keys()):
            window = self.ip_windows[ip][window_type]
            if current_time - window.start_time >= self.windows[window_type]:
                del self.ip_windows[ip][window_type]

    def _get_window(self, ip: str, window_type: str) -> RateLimitWindow:
        """Get or create a rate limit window"""
        current_time = time.time()
        window = self.ip_windows[ip].get(window_type)

        # If window doesn't exist or is expired, create new one
        if not window or (
            current_time - window.start_time >= self.windows[window_type]
        ):
            window = RateLimitWindow(requests=0, start_time=current_time)
            self.ip_windows[ip][window_type] = window

        return window

    async def check_rate_limit(self, request: Request) -> None:
        """Check if request exceeds rate limits"""
        # Skip rate limiting for health check endpoint
        if request.url.path == "/health":
            return

        ip = request.client.host
        self._clean_old_windows(ip)

        # Check each window type
        for window_type, limit in self.limits.items():
            window = self._get_window(ip, window_type)

            if window.requests >= limit:
                retry_after = self.windows[window_type] - (
                    time.time() - window.start_time
                )
                raise HTTPException(
                    status_code=429,
                    detail={
                        "error": "Rate limit exceeded",
                        "window": window_type,
                        "limit": limit,
                        "retry_after": int(retry_after),
                    },
                    headers={
                        "Retry-After": str(int(retry_after)),
                        "X-RateLimit-Limit": str(limit),
                        "X-RateLimit-Remaining": "0",
                        "X-RateLimit-Reset": str(
                            int(window.start_time + self.windows[window_type])
                        ),
                    },
                )

            window.requests += 1

    def get_rate_limit_status(self, ip: str) -> Dict[str, Dict[str, int]]:
        """Get current rate limit status for an IP"""
        self._clean_old_windows(ip)
        status = {}

        for window_type, limit in self.limits.items():
            window = self._get_window(ip, window_type)
            remaining = max(0, limit - window.requests)
            reset_time = int(window.start_time + self.windows[window_type])

            status[window_type] = {
                "limit": limit,
                "remaining": remaining,
                "reset": reset_time,
            }

        return status


class UvicornServer(uvicorn.Server):
    """Customized uvicorn server with graceful shutdown support"""

    async def setup(self, sockets=None):
        """Setup the server"""
        await super().setup(sockets)

    async def shutdown(self, sockets=None):
        """Gracefully shutdown the server"""
        logger.info("Shutting down server...")
        await super().shutdown(sockets)


class AgentStatus(str, Enum):
    """Enum for agent status."""

    IDLE = "idle"
    PROCESSING = "processing"
    ERROR = "error"
    MAINTENANCE = "maintenance"


class AgentConfig(BaseModel):
    """Configuration model for creating a new agent."""

    agent_name: str = Field(..., description="Name of the agent")
    model_name: str = Field(
        ..., description="Name of the llm you want to use provided by litellm"
    )
    description: str = Field(
        default="", description="Description of the agent's purpose"
    )
    system_prompt: str = Field(..., description="System prompt for the agent")
    temperature: float = Field(
        default=0.1, ge=0.0, le=2.0, description="Temperature for the model"
    )
    max_loops: int = Field(default=1, ge=1, description="Maximum number of loops")
    dynamic_temperature_enabled: bool = Field(
        default=True, description="Enable dynamic temperature"
    )
    user_name: str = Field(default="default_user", description="Username for the agent")
    retry_attempts: int = Field(default=1, ge=1, description="Number of retry attempts")
    context_length: int = Field(default=200000, ge=1000, description="Context length")
    output_type: str = Field(
        default="string", description="Output type (string or json)"
    )
    streaming_on: bool = Field(default=False, description="Enable streaming")
    tags: List[str] = Field(
        default_factory=list, description="Tags for categorizing the agent"
    )
    stopping_token: str = Field(
        default="<DONE>", description="Stopping token for the agent"
    )
    auto_generate_prompt: bool = Field(
        default=False, description="Auto-generate prompt based on agent details"
    )


class AgentUpdate(BaseModel):
    """Model for updating agent configuration."""

    description: Optional[str] = None
    system_prompt: Optional[str] = None
    temperature: Optional[float] = 0.5
    max_loops: Optional[int] = 1
    tags: Optional[List[str]] = None
    status: Optional[AgentStatus] = None


class AgentSummary(BaseModel):
    """Summary model for agent listing."""

    agent_id: UUID
    agent_name: str
    description: str
    system_prompt: str
    created_at: datetime
    last_used: datetime
    total_completions: int
    tags: List[str]
    status: AgentStatus


class AgentMetrics(BaseModel):
    """Model for agent performance metrics."""

    total_completions: int
    average_response_time: float
    error_rate: float
    last_24h_completions: int
    total_tokens_used: int
    uptime_percentage: float
    success_rate: float
    peak_tokens_per_minute: int


class CompletionRequest(BaseModel):
    """Model for completion requests."""

    prompt: str = Field(..., description="The prompt to process")
    agent_id: UUID = Field(..., description="ID of the agent to use")
    max_tokens: Optional[int] = Field(None, description="Maximum tokens to generate")
    temperature_override: Optional[float] = 0.5
    stream: bool = Field(default=False, description="Enable streaming response")


class CompletionResponse(BaseModel):
    """Model for completion responses."""

    agent_id: UUID
    response: str
    metadata: Dict[str, Any]
    timestamp: datetime
    processing_time: float
    token_usage: Dict[str, int]


class AgentStore:
    """Store for managing agents."""

    def __init__(self):
        self.agents: Dict[UUID, Agent] = {}
        self.agent_metadata: Dict[UUID, Dict[str, Any]] = {}
        self.executor = ThreadPoolExecutor(max_workers=4)
        self._ensure_directories()

    def _ensure_directories(self):
        """Ensure required directories exist."""
        Path("logs").mkdir(exist_ok=True)
        Path("states").mkdir(exist_ok=True)

    async def create_agent(self, config: AgentConfig) -> UUID:
        """Create a new agent with the given configuration."""
        try:
            agent = Agent(
                agent_name=config.agent_name,
                system_prompt=config.system_prompt,
                model_name=config.model_name,
                max_loops=config.max_loops,
                dynamic_temperature_enabled=True,
                user_name=config.user_name,
                retry_attempts=config.retry_attempts,
                context_length=config.context_length,
                return_step_meta=False,
                output_type="str",
                streaming_on=config.streaming_on,
                stopping_token=config.stopping_token,
                auto_generate_prompt=config.auto_generate_prompt,
            )

            agent_id = uuid4()
            self.agents[agent_id] = agent
            self.agent_metadata[agent_id] = {
                "description": config.description,
                "created_at": datetime.utcnow(),
                "last_used": datetime.utcnow(),
                "total_completions": 0,
                "tags": config.tags,
                "total_tokens": 0,
                "error_count": 0,
                "response_times": [],
                "status": AgentStatus.IDLE,
                "start_time": datetime.utcnow(),
                "downtime": timedelta(),
                "successful_completions": 0,
            }

            return agent_id

        except Exception as e:
            logger.error(f"Error creating agent: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to create agent: {str(e)}",
            )

    async def get_agent(self, agent_id: UUID) -> Agent:
        """Retrieve an agent by ID."""
        agent = self.agents.get(agent_id)
        if not agent:
            logger.error(f"Agent not found: {agent_id}")
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Agent {agent_id} not found",
            )
        return agent

    async def update_agent(self, agent_id: UUID, update: AgentUpdate) -> None:
        """Update agent configuration."""
        agent = await self.get_agent(agent_id)
        metadata = self.agent_metadata[agent_id]

        if update.system_prompt:
            agent.system_prompt = update.system_prompt
        if update.max_loops is not None:
            agent.max_loops = update.max_loops
        if update.tags is not None:
            metadata["tags"] = update.tags
        if update.description is not None:
            metadata["description"] = update.description
        if update.status is not None:
            metadata["status"] = update.status
            if update.status == AgentStatus.MAINTENANCE:
                metadata["downtime"] += datetime.utcnow() - metadata["last_used"]

        logger.info(f"Updated agent {agent_id}")

    async def list_agents(
        self,
        tags: Optional[List[str]] = None,
        status: Optional[AgentStatus] = None,
    ) -> List[AgentSummary]:
        """List all agents, optionally filtered by tags and status."""
        summaries = []
        for agent_id, agent in self.agents.items():
            metadata = self.agent_metadata[agent_id]

            if tags and not any(tag in metadata["tags"] for tag in tags):
                continue
            if status and metadata["status"] != status:
                continue

            summaries.append(
                AgentSummary(
                    agent_id=agent_id,
                    agent_name=agent.agent_name,
                    system_prompt=agent.system_prompt,
                    description=metadata["description"],
                    created_at=metadata["created_at"],
                    last_used=metadata["last_used"],
                    total_completions=metadata["total_completions"],
                    tags=metadata["tags"],
                    status=metadata["status"],
                )
            )
        return summaries

    async def get_agent_metrics(self, agent_id: UUID) -> AgentMetrics:
        """Get performance metrics for an agent."""
        metadata = self.agent_metadata[agent_id]
        response_times = metadata["response_times"]

        total_time = datetime.utcnow() - metadata["start_time"]
        uptime = total_time - metadata["downtime"]
        uptime_percentage = (uptime.total_seconds() / total_time.total_seconds()) * 100

        success_rate = (
            metadata["successful_completions"] / metadata["total_completions"] * 100
            if metadata["total_completions"] > 0
            else 0
        )

        return AgentMetrics(
            total_completions=metadata["total_completions"],
            average_response_time=(
                sum(response_times) / len(response_times) if response_times else 0
            ),
            error_rate=(
                metadata["error_count"] / metadata["total_completions"]
                if metadata["total_completions"] > 0
                else 0
            ),
            last_24h_completions=sum(
                1 for t in response_times if (datetime.utcnow() - t).days < 1
            ),
            total_tokens_used=metadata["total_tokens"],
            uptime_percentage=uptime_percentage,
            success_rate=success_rate,
            peak_tokens_per_minute=max(metadata.get("tokens_per_minute", [0])),
        )

    async def clone_agent(self, agent_id: UUID, new_name: str) -> UUID:
        """Clone an existing agent with a new name."""
        original_agent = await self.get_agent(agent_id)
        original_metadata = self.agent_metadata[agent_id]

        config = AgentConfig(
            agent_name=new_name,
            description=f"Clone of {original_agent.agent_name}",
            system_prompt=original_agent.system_prompt,
            model_name=original_agent.model_name,
            temperature=0.5,
            max_loops=original_agent.max_loops,
            tags=original_metadata["tags"],
        )

        return await self.create_agent(config)

    async def delete_agent(self, agent_id: UUID) -> None:
        """Delete an agent."""
        if agent_id not in self.agents:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Agent {agent_id} not found",
            )

        agent = self.agents[agent_id]
        if agent.autosave and os.path.exists(agent.saved_state_path):
            os.remove(agent.saved_state_path)

        del self.agents[agent_id]
        del self.agent_metadata[agent_id]
        logger.info(f"Deleted agent {agent_id}")

    async def process_completion(
        self,
        agent: Agent,
        prompt: str,
        agent_id: UUID,
        max_tokens: Optional[int] = None,
        temperature_override: Optional[float] = 0.5,
    ) -> CompletionResponse:
        """Process a completion request using the specified agent."""
        start_time = datetime.utcnow()
        metadata = self.agent_metadata[agent_id]

        try:
            metadata["status"] = AgentStatus.PROCESSING
            metadata["last_used"] = start_time

            response = agent.run(prompt)

            processing_time = (datetime.utcnow() - start_time).total_seconds()
            metadata["response_times"].append(processing_time)
            metadata["total_completions"] += 1
            metadata["successful_completions"] += 1

            prompt_tokens = len(prompt.split()) * 1.3
            completion_tokens = len(response.split()) * 1.3
            total_tokens = int(prompt_tokens + completion_tokens)
            metadata["total_tokens"] += total_tokens

            current_minute = datetime.utcnow().replace(second=0, microsecond=0)
            if "tokens_per_minute" not in metadata:
                metadata["tokens_per_minute"] = {}
            metadata["tokens_per_minute"][current_minute] = (
                metadata["tokens_per_minute"].get(current_minute, 0) + total_tokens
            )

            return CompletionResponse(
                agent_id=agent_id,
                response=response,
                metadata={
                    "agent_name": agent.agent_name,
                },
                timestamp=datetime.utcnow(),
                processing_time=processing_time,
                token_usage={
                    "prompt_tokens": int(prompt_tokens),
                    "completion_tokens": int(completion_tokens),
                    "total_tokens": total_tokens,
                },
            )

        except Exception as e:
            metadata["error_count"] += 1
            metadata["status"] = AgentStatus.ERROR
            logger.error(
                f"Error in completion processing: {str(e)}\n{traceback.format_exc()}"
            )
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Error processing completion: {str(e)}",
            )
        finally:
            metadata["status"] = AgentStatus.IDLE


class StoreManager:
    _instance = None

    @classmethod
    def get_instance(cls) -> AgentStore:
        if cls._instance is None:
            cls._instance = AgentStore()
        return cls._instance


def get_store() -> AgentStore:
    """Dependency to get the AgentStore instance."""
    return StoreManager.get_instance()


class SwarmsAPI:
    def __init__(self):
        self.app = FastAPI(
            title="Swarms Agent API",
            description="Free API for Swarms agent interaction",
            version="1.0.0",
            docs_url="/v1/docs",
            redoc_url="/v1/redoc",
        )
        self.store = StoreManager.get_instance()

        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        self.rate_limiter = RateLimiter()

        self._setup_routes()

    def _setup_routes(self):
        """Set up API routes."""

        def _setup_middleware(self):
            @self.app.middleware("http")
            async def rate_limit_middleware(request: Request, call_next):
                # Apply rate limiting
                await self.rate_limiter.check_rate_limit(request)

                # Add rate limit headers to response
                response = await call_next(request)

                # Add rate limit status headers
                status = self.rate_limiter.get_rate_limit_status(request.client.host)
                for window_type, info in status.items():
                    response.headers[f"X-RateLimit-{window_type}-Limit"] = str(
                        info["limit"]
                    )
                    response.headers[f"X-RateLimit-{window_type}-Remaining"] = str(
                        info["remaining"]
                    )
                    response.headers[f"X-RateLimit-{window_type}-Reset"] = str(
                        info["reset"]
                    )

                return response

        @self.app.get("/v1/rate-limit-status")
        async def get_rate_limit_status(request: Request):
            """Get current rate limit status"""
            return self.rate_limiter.get_rate_limit_status(request.client.host)

        @self.app.get("/v1/agents", response_model=List[AgentSummary])
        async def list_agents(
            tags: Optional[List[str]] = Query(None),
            status: Optional[AgentStatus] = None,
        ):
            """List all agents, optionally filtered by tags and status."""
            return await self.store.list_agents(tags, status)

        @self.app.post("/v1/agent", response_model=Dict[str, UUID])
        async def create_agent(config: AgentConfig):
            """Create a new agent with the specified configuration."""
            agent_id = await self.store.create_agent(config)
            return {"agent_id": agent_id}

        @self.app.patch("/v1/agent/{agent_id}", response_model=Dict[str, str])
        async def update_agent(
            agent_id: UUID,
            update: AgentUpdate,
        ):
            """Update an existing agent's configuration."""
            await self.store.update_agent(agent_id, update)
            return {"status": "updated"}

        @self.app.get("/v1/agent/{agent_id}/metrics", response_model=AgentMetrics)
        async def get_agent_metrics(agent_id: UUID):
            """Get performance metrics for a specific agent."""
            return await self.store.get_agent_metrics(agent_id)

        @self.app.post("/v1/agent/{agent_id}/clone", response_model=Dict[str, UUID])
        async def clone_agent(agent_id: UUID, new_name: str):
            """Clone an existing agent with a new name."""
            new_id = await self.store.clone_agent(agent_id, new_name)
            return {"agent_id": new_id}

        @self.app.delete("/v1/agent/{agent_id}")
        async def delete_agent(agent_id: UUID):
            """Delete an agent."""
            await self.store.delete_agent(agent_id)
            return {"status": "deleted"}

        @self.app.post("/v1/agent/completions", response_model=CompletionResponse)
        async def create_completion(
            request: CompletionRequest,
            background_tasks: BackgroundTasks,
        ):
            """Process a completion request with the specified agent."""
            try:
                agent = await self.store.get_agent(request.agent_id)

                response = await self.store.process_completion(
                    agent,
                    request.prompt,
                    request.agent_id,
                    request.max_tokens,
                    request.temperature_override,
                )

                background_tasks.add_task(self._cleanup_old_metrics, request.agent_id)

                return response

            except Exception as e:
                logger.error(f"Error processing completion: {str(e)}")
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=f"Error processing completion: {str(e)}",
                )

        @self.app.get("/v1/agent/batch/completions/status")
        async def get_batch_completion_status(requests: List[CompletionRequest]):
            """Get status of batch completion requests."""
            responses = []
            for req in requests:
                try:
                    agent = await self.store.get_agent(req.agent_id)
                    response = await self.store.process_completion(
                        agent,
                        req.prompt,
                        req.agent_id,
                        req.max_tokens,
                        req.temperature_override,
                    )
                    responses.append(response)
                except Exception as e:
                    logger.error(f"Error processing batch completion: {str(e)}")
                    responses.append(
                        {
                            "error": f"Error processing completion: {str(e)}",
                            "agent_id": req.agent_id,
                        }
                    )
            return responses

        @self.app.get("/v1/agent/{agent_id}/status")
        async def get_agent_status(agent_id: UUID):
            """Get the current status of an agent."""
            metadata = self.store.agent_metadata.get(agent_id)
            if not metadata:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Agent {agent_id} not found",
                )

            return {
                "agent_id": agent_id,
                "status": metadata["status"],
                "last_used": metadata["last_used"],
                "total_completions": metadata["total_completions"],
                "error_count": metadata["error_count"],
            }

        @self.app.get("/health")
        async def health_check():
            """Health check endpoint."""
            return {"status": "healthy"}

    async def _cleanup_old_metrics(self, agent_id: UUID):
        """Clean up old metrics data to prevent memory bloat."""
        metadata = self.store.agent_metadata.get(agent_id)
        if metadata:
            # Keep only last 24 hours of response times
            cutoff = datetime.utcnow() - timedelta(days=1)
            metadata["response_times"] = [
                t
                for t in metadata["response_times"]
                if isinstance(t, (int, float)) and t > cutoff.timestamp()
            ]

            # Clean up old tokens per minute data
            if "tokens_per_minute" in metadata:
                metadata["tokens_per_minute"] = {
                    k: v for k, v in metadata["tokens_per_minute"].items() if k > cutoff
                }


class APIServer:
    def __init__(self, app: FastAPI, host: str = "0.0.0.0", port: int = 8080):
        self.app = app
        self.host = host
        self.port = port
        self.config = uvicorn.Config(
            app=app,
            host=host,
            port=port,
            log_level="info",
            access_log=True,
        )
        self.server = UvicornServer(config=self.config)

        # Setup signal handlers
        signal.signal(signal.SIGTERM, self._handle_signal)
        signal.signal(signal.SIGINT, self._handle_signal)

    def _handle_signal(self, signum, frame):
        """Handle shutdown signals"""
        logger.info(f"Received signal {signum}")
        asyncio.create_task(self.shutdown())

    async def startup(self) -> None:
        """Start the server"""
        try:
            logger.info(f"Starting API server on http://{self.host}:{self.port}")
            print(f"Starting API server on http://{self.host}:{self.port}")
            await self.server.serve()
        except Exception as e:
            logger.error(f"Failed to start server: {str(e)}")
            raise

    async def shutdown(self) -> None:
        """Shutdown the server"""
        try:
            logger.info("Initiating graceful shutdown...")
            await self.server.shutdown()
        except Exception as e:
            logger.error(f"Error during shutdown: {str(e)}")
            raise


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator:
    """Lifespan context manager for the FastAPI app"""
    # Startup
    logger.info("Starting up API server...")
    yield
    # Shutdown
    logger.info("Shutting down API server...")


def create_app() -> FastAPI:
    """Create and configure the FastAPI application"""
    logger.info("Creating FastAPI application")
    api = SwarmsAPI()
    app = api.app

    # Add lifespan handling
    app.router.lifespan_context = lifespan

    logger.info("FastAPI application created successfully")
    return app


def run_server():
    """Run the API server"""
    try:
        # Create the FastAPI app
        app = create_app()

        # Create and run the server
        server = APIServer(app)
        asyncio.run(server.startup())
    except Exception as e:
        logger.error(f"Failed to start API: {str(e)}")
        print(f"Error starting server: {str(e)}")


if __name__ == "__main__":
    run_server()
