"""ACP agent server — exposes Hermes Agent via the Agent Client Protocol."""

from __future__ import annotations

import asyncio
import json
import logging
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Deque, Optional

import acp
from acp.schema import (
    AgentCapabilities,
    AuthenticateResponse,
    AvailableCommand,
    AvailableCommandsUpdate,
    ClientCapabilities,
    EmbeddedResourceContentBlock,
    ForkSessionResponse,
    ImageContentBlock,
    AudioContentBlock,
    Implementation,
    InitializeResponse,
    ListSessionsResponse,
    LoadSessionResponse,
    McpServerHttp,
    McpServerSse,
    McpServerStdio,
    ModelInfo,
    NewSessionResponse,
    PromptResponse,
    ResumeSessionResponse,
    SessionConfigOptionSelect,
    SessionConfigSelectOption,
    SessionModelState,
    SetSessionConfigOptionResponse,
    SetSessionModelResponse,
    SetSessionModeResponse,
    ResourceContentBlock,
    SessionCapabilities,
    SessionForkCapabilities,
    SessionListCapabilities,
    SessionResumeCapabilities,
    SessionInfo,
    TextContentBlock,
    UnstructuredCommandInput,
    Usage,
)

# AuthMethodAgent was renamed from AuthMethod in agent-client-protocol 0.9.0
try:
    from acp.schema import AuthMethodAgent
except ImportError:
    from acp.schema import AuthMethod as AuthMethodAgent  # type: ignore[attr-defined]

from acp_adapter.auth import detect_provider, has_provider
from acp_adapter.events import (
    make_message_cb,
    make_step_cb,
    make_thinking_cb,
    make_tool_progress_cb,
)
from acp_adapter.permissions import make_approval_callback
from acp_adapter.session import SessionManager, SessionState

logger = logging.getLogger(__name__)

try:
    from hermes_cli import __version__ as HERMES_VERSION
except Exception:
    HERMES_VERSION = "0.0.0"

# Thread pool for running AIAgent (synchronous) in parallel.
_executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="acp-agent")


def _extract_text(
    prompt: list[
        TextContentBlock
        | ImageContentBlock
        | AudioContentBlock
        | ResourceContentBlock
        | EmbeddedResourceContentBlock
    ],
) -> str:
    """Extract plain text from ACP content blocks."""
    parts: list[str] = []
    for block in prompt:
        if isinstance(block, TextContentBlock):
            parts.append(block.text)
        elif hasattr(block, "text"):
            parts.append(str(block.text))
        # Non-text blocks are ignored for now.
    return "\n".join(parts)


class HermesACPAgent(acp.Agent):
    """ACP Agent implementation wrapping Hermes AIAgent."""

    _SLASH_COMMANDS = {
        "help": "Show available commands",
        "model": "Show or change current model",
        "tools": "List available tools",
        "context": "Show conversation context info",
        "reset": "Clear conversation history",
        "compact": "Compress conversation context",
        "version": "Show Hermes version",
    }

    _ADVERTISED_COMMANDS = (
        {
            "name": "help",
            "description": "List available commands",
        },
        {
            "name": "model",
            "description": "Show current model and provider, or switch models",
            "input_hint": "model name to switch to",
        },
        {
            "name": "tools",
            "description": "List available tools with descriptions",
        },
        {
            "name": "context",
            "description": "Show conversation message counts by role",
        },
        {
            "name": "reset",
            "description": "Clear conversation history",
        },
        {
            "name": "compact",
            "description": "Compress conversation context",
        },
        {
            "name": "version",
            "description": "Show Hermes version",
        },
    )

    def __init__(self, session_manager: SessionManager | None = None):
        super().__init__()
        self.session_manager = session_manager or SessionManager()
        self._conn: Optional[acp.Client] = None

    # ---- Connection lifecycle -----------------------------------------------

    def on_connect(self, conn: acp.Client) -> None:
        """Store the client connection for sending session updates."""
        self._conn = conn
        logger.info("ACP client connected")

    async def _register_session_mcp_servers(
        self,
        state: SessionState,
        mcp_servers: list[McpServerStdio | McpServerHttp | McpServerSse] | None,
    ) -> None:
        """Register ACP-provided MCP servers and refresh the agent tool surface."""
        if not mcp_servers:
            return

        try:
            from tools.mcp_tool import register_mcp_servers

            config_map: dict[str, dict] = {}
            for server in mcp_servers:
                name = server.name
                if isinstance(server, McpServerStdio):
                    config = {
                        "command": server.command,
                        "args": list(server.args),
                        "env": {item.name: item.value for item in server.env},
                    }
                else:
                    config = {
                        "url": server.url,
                        "headers": {item.name: item.value for item in server.headers},
                    }
                config_map[name] = config

            await asyncio.to_thread(register_mcp_servers, config_map)
        except Exception:
            logger.warning(
                "Session %s: failed to register ACP MCP servers",
                state.session_id,
                exc_info=True,
            )
            return

        try:
            from model_tools import get_tool_definitions

            enabled_toolsets = getattr(state.agent, "enabled_toolsets", None) or ["hermes-acp"]
            disabled_toolsets = getattr(state.agent, "disabled_toolsets", None)
            state.agent.tools = get_tool_definitions(
                enabled_toolsets=enabled_toolsets,
                disabled_toolsets=disabled_toolsets,
                quiet_mode=True,
            )
            state.agent.valid_tool_names = {
                tool["function"]["name"] for tool in state.agent.tools or []
            }
            invalidate = getattr(state.agent, "_invalidate_system_prompt", None)
            if callable(invalidate):
                invalidate()
            logger.info(
                "Session %s: refreshed tool surface after ACP MCP registration (%d tools)",
                state.session_id,
                len(state.agent.tools or []),
            )
        except Exception:
            logger.warning(
                "Session %s: failed to refresh tool surface after ACP MCP registration",
                state.session_id,
                exc_info=True,
            )

    # ---- ACP lifecycle ------------------------------------------------------

    async def initialize(
        self,
        protocol_version: int | None = None,
        client_capabilities: ClientCapabilities | None = None,
        client_info: Implementation | None = None,
        **kwargs: Any,
    ) -> InitializeResponse:
        resolved_protocol_version = (
            protocol_version if isinstance(protocol_version, int) else acp.PROTOCOL_VERSION
        )
        provider = detect_provider()
        auth_methods = None
        if provider:
            auth_methods = [
                AuthMethodAgent(
                    id=provider,
                    name=f"{provider} runtime credentials",
                    description=f"Authenticate Hermes using the currently configured {provider} runtime credentials.",
                )
            ]

        client_name = client_info.name if client_info else "unknown"
        logger.info(
            "Initialize from %s (protocol v%s)",
            client_name,
            resolved_protocol_version,
        )

        return InitializeResponse(
            protocol_version=acp.PROTOCOL_VERSION,
            agent_info=Implementation(name="hermes-agent", version=HERMES_VERSION),
            agent_capabilities=AgentCapabilities(
                load_session=True,
                session_capabilities=SessionCapabilities(
                    fork=SessionForkCapabilities(),
                    list=SessionListCapabilities(),
                    resume=SessionResumeCapabilities(),
                ),
            ),
            auth_methods=auth_methods,
        )

    async def authenticate(self, method_id: str, **kwargs: Any) -> AuthenticateResponse | None:
        if has_provider():
            return AuthenticateResponse()
        return None

    # ---- Model config options for ACP clients (e.g., Multicoder) ------------

    def _get_available_providers_with_models(self) -> dict[str, list[str]]:
        """Return provider -> model IDs using the shared CLI model-switch logic."""
        try:
            from hermes_cli.model_switch import list_authenticated_providers
            from hermes_cli.models import curated_models_for_provider, normalize_provider
        except Exception:
            return {}

        current_provider = ""
        try:
            current_provider = getattr(getattr(self, "session_manager", None), "current_provider", "") or ""
        except Exception:
            current_provider = ""

        result: dict[str, list[str]] = {}
        try:
            providers = list_authenticated_providers(current_provider=current_provider, max_models=200)
        except Exception:
            providers = []

        discovered_provider_ids: set[str] = set()
        for info in providers:
            provider_id = normalize_provider(str(info.get("slug") or "").strip())
            if not provider_id:
                continue
            discovered_provider_ids.add(provider_id)
            try:
                models = [mid for mid, _ in curated_models_for_provider(provider_id)]
            except Exception:
                models = list(info.get("models") or [])
            if models:
                result[provider_id] = models

        # Fallback: merge providers found only in auth.json/credential_pool so ACP
        # still sees API-key-backed providers even when list_authenticated_providers()
        # lacks enough environment/context to surface them in test or editor sessions.
        try:
            from hermes_constants import get_hermes_home
            from hermes_cli.models import normalize_provider

            auth_path = get_hermes_home() / "auth.json"
            if auth_path.exists():
                auth = json.loads(auth_path.read_text())
                provider_ids = set((auth.get("providers") or {}).keys()) | set((auth.get("credential_pool") or {}).keys())
                for provider_id in provider_ids:
                    canonical = normalize_provider(provider_id)
                    if canonical in discovered_provider_ids or canonical in result:
                        continue
                    try:
                        models = [mid for mid, _ in curated_models_for_provider(canonical)]
                    except Exception:
                        models = []
                    if models:
                        result[canonical] = models
        except Exception:
            pass

        return result

    def _encode_model_choice(self, provider_id: str, model_id: str) -> str:
        """Encode ACP model selections with provider identity preserved."""
        return f"{provider_id}:{model_id}"

    def _current_model_choice_id(self, state: SessionState | None, current_model: str = "") -> str:
        """Return the ACP model choice id for the current session state."""
        if not isinstance(current_model, str) or not current_model:
            current_model = getattr(getattr(state, "agent", None), "model", "") or ""
        provider_id = getattr(getattr(state, "agent", None), "provider", None) or ""
        if provider_id and current_model:
            return self._encode_model_choice(provider_id, current_model)
        return current_model or ""

    def _get_session_model_state(
        self,
        current_model: str = "",
        state: SessionState | None = None,
    ) -> SessionModelState | None:
        """Build ACP SessionModelState for native client model dropdowns."""
        try:
            from hermes_cli.providers import get_label
        except Exception:
            return None

        provider_models_map = self._get_available_providers_with_models()
        if not provider_models_map:
            return None

        available_models: list[ModelInfo] = []
        for provider_id, models in provider_models_map.items():
            provider_label = get_label(provider_id)
            for model_id in models:
                if not isinstance(model_id, str) or not model_id:
                    continue
                display_id = model_id.split("/")[-1] if "/" in model_id else model_id
                available_models.append(
                    ModelInfo(
                        model_id=self._encode_model_choice(provider_id, model_id),
                        name=f"{display_id} ({provider_label})",
                        description=f"Via {provider_label}",
                    )
                )

        if not available_models:
            return None

        current_choice = self._current_model_choice_id(state, current_model)
        if not current_choice:
            current_choice = available_models[0].model_id

        return SessionModelState(
            current_model_id=current_choice,
            available_models=available_models,
        )

    def _get_model_config_options(
        self,
        current_model: str = "",
        state: SessionState | None = None,
    ) -> list[SessionConfigOptionSelect]:
        """Build ACP config options for model selection from configured providers."""
        try:
            from hermes_cli.config import load_config
            from hermes_cli.providers import get_label
        except Exception:
            return []

        provider_models_map = self._get_available_providers_with_models()
        if not provider_models_map:
            return []

        try:
            config = load_config()
            default_model_config = config.get("model", {}) if isinstance(config, dict) else {}
        except Exception:
            default_model_config = {}

        model_options: list[SessionConfigSelectOption] = []
        for provider_id, models in provider_models_map.items():
            provider_label = get_label(provider_id)
            for model_id in models:
                if not isinstance(model_id, str) or not model_id:
                    continue
                display_id = model_id.split("/")[-1] if "/" in model_id else model_id
                model_options.append(
                    SessionConfigSelectOption(
                        value=self._encode_model_choice(provider_id, model_id),
                        name=f"{display_id} ({provider_label})",
                        description=f"Via {provider_label}"[:50],
                    )
                )

        current_choice = self._current_model_choice_id(state, current_model)
        if not current_choice:
            default_model = default_model_config.get("default", "") if isinstance(default_model_config, dict) else ""
            default_provider = default_model_config.get("provider", "") if isinstance(default_model_config, dict) else ""
            if default_model and default_provider:
                current_choice = self._encode_model_choice(default_provider, default_model)
        if not current_choice and model_options:
            current_choice = model_options[0].value

        return [
            SessionConfigOptionSelect(
                id="model",
                name="Model",
                description=f"AI model ({len(provider_models_map)} provider(s))",
                category="model",
                type="select",
                current_value=current_choice,
                options=model_options,
            )
        ]

    # ---- Session management -------------------------------------------------

    async def new_session(
        self,
        cwd: str,
        mcp_servers: list | None = None,
        **kwargs: Any,
    ) -> NewSessionResponse:
        state = self.session_manager.create_session(cwd=cwd)
        await self._register_session_mcp_servers(state, mcp_servers)
        logger.info("New session %s (cwd=%s)", state.session_id, cwd)
        self._schedule_available_commands_update(state.session_id)
        
        # Get current model from agent and expose model selector to ACP clients
        current_model = getattr(state.agent, "model", "")
        config_options = self._get_model_config_options(current_model, state=state)
        models_state = self._get_session_model_state(current_model, state=state)

        return NewSessionResponse(
            session_id=state.session_id,
            config_options=config_options,
            models=models_state,
        )

    async def load_session(
        self,
        cwd: str,
        session_id: str,
        mcp_servers: list | None = None,
        **kwargs: Any,
    ) -> LoadSessionResponse | None:
        state = self.session_manager.update_cwd(session_id, cwd)
        if state is None:
            logger.warning("load_session: session %s not found", session_id)
            return None
        await self._register_session_mcp_servers(state, mcp_servers)
        logger.info("Loaded session %s", session_id)
        self._schedule_available_commands_update(session_id)
        
        # Expose model selector to ACP clients when loading session
        current_model = getattr(state.agent, "model", "")
        config_options = self._get_model_config_options(current_model, state=state)
        models_state = self._get_session_model_state(current_model, state=state)

        return LoadSessionResponse(config_options=config_options, models=models_state)

    async def resume_session(
        self,
        cwd: str,
        session_id: str,
        mcp_servers: list | None = None,
        **kwargs: Any,
    ) -> ResumeSessionResponse:
        state = self.session_manager.update_cwd(session_id, cwd)
        if state is None:
            logger.warning("resume_session: session %s not found, creating new", session_id)
            state = self.session_manager.create_session(cwd=cwd)
        await self._register_session_mcp_servers(state, mcp_servers)
        logger.info("Resumed session %s", state.session_id)
        self._schedule_available_commands_update(state.session_id)
        
        # Expose model selector to ACP clients when resuming session
        current_model = getattr(state.agent, "model", "")
        config_options = self._get_model_config_options(current_model, state=state)
        models_state = self._get_session_model_state(current_model, state=state)

        return ResumeSessionResponse(config_options=config_options, models=models_state)

    async def cancel(self, session_id: str, **kwargs: Any) -> None:
        state = self.session_manager.get_session(session_id)
        if state and state.cancel_event:
            state.cancel_event.set()
            try:
                if getattr(state, "agent", None) and hasattr(state.agent, "interrupt"):
                    state.agent.interrupt()
            except Exception:
                logger.debug("Failed to interrupt ACP session %s", session_id, exc_info=True)
            logger.info("Cancelled session %s", session_id)

    async def fork_session(
        self,
        cwd: str,
        session_id: str,
        mcp_servers: list | None = None,
        **kwargs: Any,
    ) -> ForkSessionResponse:
        state = self.session_manager.fork_session(session_id, cwd=cwd)
        new_id = state.session_id if state else ""
        if state is not None:
            await self._register_session_mcp_servers(state, mcp_servers)
        logger.info("Forked session %s -> %s", session_id, new_id)
        if new_id:
            self._schedule_available_commands_update(new_id)
        return ForkSessionResponse(session_id=new_id)

    async def list_sessions(
        self,
        cursor: str | None = None,
        cwd: str | None = None,
        **kwargs: Any,
    ) -> ListSessionsResponse:
        infos = self.session_manager.list_sessions()
        sessions = [
            SessionInfo(session_id=s["session_id"], cwd=s["cwd"])
            for s in infos
        ]
        return ListSessionsResponse(sessions=sessions)

    # ---- Prompt (core) ------------------------------------------------------

    async def prompt(
        self,
        prompt: list[
            TextContentBlock
            | ImageContentBlock
            | AudioContentBlock
            | ResourceContentBlock
            | EmbeddedResourceContentBlock
        ],
        session_id: str,
        **kwargs: Any,
    ) -> PromptResponse:
        """Run Hermes on the user's prompt and stream events back to the editor."""
        state = self.session_manager.get_session(session_id)
        if state is None:
            logger.error("prompt: session %s not found", session_id)
            return PromptResponse(stop_reason="refusal")

        user_text = _extract_text(prompt).strip()
        if not user_text:
            return PromptResponse(stop_reason="end_turn")

        # Intercept slash commands — handle locally without calling the LLM
        if user_text.startswith("/"):
            response_text = self._handle_slash_command(user_text, state)
            if response_text is not None:
                if self._conn:
                    update = acp.update_agent_message_text(response_text)
                    await self._conn.session_update(session_id, update)
                return PromptResponse(stop_reason="end_turn")

        logger.info("Prompt on session %s: %s", session_id, user_text[:100])

        conn = self._conn
        loop = asyncio.get_running_loop()

        if state.cancel_event:
            state.cancel_event.clear()

        tool_call_ids: dict[str, Deque[str]] = defaultdict(deque)
        previous_approval_cb = None

        if conn:
            tool_progress_cb = make_tool_progress_cb(conn, session_id, loop, tool_call_ids)
            thinking_cb = make_thinking_cb(conn, session_id, loop)
            step_cb = make_step_cb(conn, session_id, loop, tool_call_ids)
            message_cb = make_message_cb(conn, session_id, loop)
            approval_cb = make_approval_callback(conn.request_permission, loop, session_id)
        else:
            tool_progress_cb = None
            thinking_cb = None
            step_cb = None
            message_cb = None
            approval_cb = None

        agent = state.agent
        agent.tool_progress_callback = tool_progress_cb
        agent.thinking_callback = thinking_cb
        agent.step_callback = step_cb
        agent.message_callback = message_cb

        if approval_cb:
            try:
                from tools import terminal_tool as _terminal_tool
                previous_approval_cb = getattr(_terminal_tool, "_approval_callback", None)
                _terminal_tool.set_approval_callback(approval_cb)
            except Exception:
                logger.debug("Could not set ACP approval callback", exc_info=True)

        def _run_agent() -> dict:
            try:
                result = agent.run_conversation(
                    user_message=user_text,
                    conversation_history=state.history,
                    task_id=session_id,
                )
                return result
            except Exception as e:
                logger.exception("Agent error in session %s", session_id)
                return {"final_response": f"Error: {e}", "messages": state.history}
            finally:
                if approval_cb:
                    try:
                        from tools import terminal_tool as _terminal_tool
                        _terminal_tool.set_approval_callback(previous_approval_cb)
                    except Exception:
                        logger.debug("Could not restore approval callback", exc_info=True)

        try:
            result = await loop.run_in_executor(_executor, _run_agent)
        except Exception:
            logger.exception("Executor error for session %s", session_id)
            return PromptResponse(stop_reason="end_turn")

        if result.get("messages"):
            state.history = result["messages"]
            # Persist updated history so sessions survive process restarts.
            self.session_manager.save_session(session_id)

        final_response = result.get("final_response", "")
        if final_response and conn:
            update = acp.update_agent_message_text(final_response)
            await conn.session_update(session_id, update)

        usage = None
        if any(result.get(key) is not None for key in ("prompt_tokens", "completion_tokens", "total_tokens")):
            usage = Usage(
                input_tokens=result.get("prompt_tokens", 0),
                output_tokens=result.get("completion_tokens", 0),
                total_tokens=result.get("total_tokens", 0),
                thought_tokens=result.get("reasoning_tokens"),
                cached_read_tokens=result.get("cache_read_tokens"),
            )

        stop_reason = "cancelled" if state.cancel_event and state.cancel_event.is_set() else "end_turn"
        return PromptResponse(stop_reason=stop_reason, usage=usage)

    # ---- Slash commands (headless) -------------------------------------------

    @classmethod
    def _available_commands(cls) -> list[AvailableCommand]:
        commands: list[AvailableCommand] = []
        for spec in cls._ADVERTISED_COMMANDS:
            input_hint = spec.get("input_hint")
            commands.append(
                AvailableCommand(
                    name=spec["name"],
                    description=spec["description"],
                    input=UnstructuredCommandInput(hint=input_hint)
                    if input_hint
                    else None,
                )
            )
        return commands

    async def _send_available_commands_update(self, session_id: str) -> None:
        """Advertise supported slash commands to the connected ACP client."""
        if not self._conn:
            return

        try:
            await self._conn.session_update(
                session_id=session_id,
                update=AvailableCommandsUpdate(
                    sessionUpdate="available_commands_update",
                    availableCommands=self._available_commands(),
                ),
            )
        except Exception:
            logger.warning(
                "Failed to advertise ACP slash commands for session %s",
                session_id,
                exc_info=True,
            )

    def _schedule_available_commands_update(self, session_id: str) -> None:
        """Send the command advertisement after the session response is queued."""
        if not self._conn:
            return
        loop = asyncio.get_running_loop()
        loop.call_soon(
            asyncio.create_task, self._send_available_commands_update(session_id)
        )

    def _handle_slash_command(self, text: str, state: SessionState) -> str | None:
        """Dispatch a slash command and return the response text.

        Returns ``None`` for unrecognized commands so they fall through
        to the LLM (the user may have typed ``/something`` as prose).
        """
        parts = text.split(maxsplit=1)
        cmd = parts[0].lstrip("/").lower()
        args = parts[1].strip() if len(parts) > 1 else ""

        handler = {
            "help": self._cmd_help,
            "model": self._cmd_model,
            "tools": self._cmd_tools,
            "context": self._cmd_context,
            "reset": self._cmd_reset,
            "compact": self._cmd_compact,
            "version": self._cmd_version,
        }.get(cmd)

        if handler is None:
            return None  # not a known command — let the LLM handle it

        try:
            return handler(args, state)
        except Exception as e:
            logger.error("Slash command /%s error: %s", cmd, e, exc_info=True)
            return f"Error executing /{cmd}: {e}"

    def _cmd_help(self, args: str, state: SessionState) -> str:
        lines = ["Available commands:", ""]
        for cmd, desc in self._SLASH_COMMANDS.items():
            lines.append(f"  /{cmd:10s}  {desc}")
        lines.append("")
        lines.append("Unrecognized /commands are sent to the model as normal messages.")
        return "\n".join(lines)

    def _switch_model(self, state: SessionState, new_model: str) -> tuple[str, str]:
        """Switch the model for a session using the shared CLI /model pipeline."""
        current_provider = getattr(state.agent, "provider", None) or "openrouter"
        current_model = getattr(state.agent, "model", "") or state.model or ""
        current_base_url = getattr(state.agent, "base_url", "") or ""
        current_api_key = getattr(state.agent, "api_key", "") or ""

        raw_input = str(new_model or "").strip()
        explicit_provider = ""
        if ":" in raw_input:
            left, right = raw_input.split(":", 1)
            if left.strip() and right.strip():
                explicit_provider = left.strip()
                raw_input = right.strip()

        try:
            from hermes_cli.model_switch import switch_model as shared_switch_model

            result = shared_switch_model(
                raw_input=raw_input,
                current_provider=current_provider,
                current_model=current_model,
                current_base_url=current_base_url,
                current_api_key=current_api_key,
                explicit_provider=explicit_provider,
            )
        except Exception as exc:
            logger.exception("ACP model switch failed during shared pipeline")
            raise RuntimeError(f"ACP model switch failed: {exc}") from exc

        if not result.success:
            raise RuntimeError(result.error_message or "model switch failed")

        switch_model = getattr(state.agent, "switch_model", None)
        if callable(switch_model):
            switch_model(
                result.new_model,
                result.target_provider,
                api_key=result.api_key,
                base_url=result.base_url,
                api_mode=result.api_mode,
            )
        else:
            state.agent.model = result.new_model
            state.agent.provider = result.target_provider
            if result.base_url:
                state.agent.base_url = result.base_url
            if result.api_mode:
                state.agent.api_mode = result.api_mode
            if result.api_key:
                state.agent.api_key = result.api_key
            invalidate = getattr(state.agent, "_invalidate_system_prompt", None)
            if callable(invalidate):
                invalidate()

        state.model = result.new_model
        self.session_manager.save_session(state.session_id)
        logger.info(
            "Session %s: model switched to %s via shared model_switch pipeline",
            state.session_id,
            result.new_model,
        )
        return result.new_model, getattr(state.agent, "provider", None) or result.target_provider

    def _cmd_model(self, args: str, state: SessionState) -> str:
        if not args:
            model = state.model or getattr(state.agent, "model", "unknown")
            provider = getattr(state.agent, "provider", None) or "auto"
            return f"Current model: {model}\nProvider: {provider}"

        new_model, provider_label = self._switch_model(state, args.strip())
        return f"Model switched to: {new_model}\nProvider: {provider_label}"

    def _cmd_tools(self, args: str, state: SessionState) -> str:
        try:
            from model_tools import get_tool_definitions
            toolsets = getattr(state.agent, "enabled_toolsets", None) or ["hermes-acp"]
            tools = get_tool_definitions(enabled_toolsets=toolsets, quiet_mode=True)
            if not tools:
                return "No tools available."
            lines = [f"Available tools ({len(tools)}):"]
            for t in tools:
                name = t.get("function", {}).get("name", "?")
                desc = t.get("function", {}).get("description", "")
                # Truncate long descriptions
                if len(desc) > 80:
                    desc = desc[:77] + "..."
                lines.append(f"  {name}: {desc}")
            return "\n".join(lines)
        except Exception as e:
            return f"Could not list tools: {e}"

    def _cmd_context(self, args: str, state: SessionState) -> str:
        n_messages = len(state.history)
        if n_messages == 0:
            return "Conversation is empty (no messages yet)."
        # Count by role
        roles: dict[str, int] = {}
        for msg in state.history:
            role = msg.get("role", "unknown")
            roles[role] = roles.get(role, 0) + 1
        lines = [
            f"Conversation: {n_messages} messages",
            f"  user: {roles.get('user', 0)}, assistant: {roles.get('assistant', 0)}, "
            f"tool: {roles.get('tool', 0)}, system: {roles.get('system', 0)}",
        ]
        model = state.model or getattr(state.agent, "model", "")
        if model:
            lines.append(f"Model: {model}")
        return "\n".join(lines)

    def _cmd_reset(self, args: str, state: SessionState) -> str:
        state.history.clear()
        self.session_manager.save_session(state.session_id)
        return "Conversation history cleared."

    def _cmd_compact(self, args: str, state: SessionState) -> str:
        if not state.history:
            return "Nothing to compress — conversation is empty."
        try:
            agent = state.agent
            if not getattr(agent, "compression_enabled", True):
                return "Context compression is disabled for this agent."
            if not hasattr(agent, "_compress_context"):
                return "Context compression not available for this agent."

            from agent.model_metadata import estimate_messages_tokens_rough

            original_count = len(state.history)
            approx_tokens = estimate_messages_tokens_rough(state.history)
            original_session_db = getattr(agent, "_session_db", None)

            try:
                # ACP sessions must keep a stable session id, so avoid the
                # SQLite session-splitting side effect inside _compress_context.
                agent._session_db = None
                compressed, _ = agent._compress_context(
                    state.history,
                    getattr(agent, "_cached_system_prompt", "") or "",
                    approx_tokens=approx_tokens,
                    task_id=state.session_id,
                )
            finally:
                agent._session_db = original_session_db

            state.history = compressed
            self.session_manager.save_session(state.session_id)

            new_count = len(state.history)
            new_tokens = estimate_messages_tokens_rough(state.history)
            return (
                f"Context compressed: {original_count} -> {new_count} messages\n"
                f"~{approx_tokens:,} -> ~{new_tokens:,} tokens"
            )
        except Exception as e:
            return f"Compression failed: {e}"

    def _cmd_version(self, args: str, state: SessionState) -> str:
        return f"Hermes Agent v{HERMES_VERSION}"

    # ---- Model switching (ACP protocol method) -------------------------------

    async def set_session_model(
        self, model_id: str, session_id: str, **kwargs: Any
    ) -> SetSessionModelResponse | None:
        """Switch the model for a session (called by ACP protocol)."""
        state = self.session_manager.get_session(session_id)
        if state is None:
            logger.warning("Session %s: model switch requested for missing session", session_id)
            return None

        current_provider = getattr(state.agent, "provider", None) or "openrouter"
        explicit_provider = False
        try:
            from hermes_cli.models import parse_model_input

            parsed_provider, parsed_model = parse_model_input(model_id, current_provider)
            explicit_provider = parsed_model != model_id or parsed_provider != current_provider
        except Exception:
            parsed_provider, parsed_model = current_provider, model_id

        if explicit_provider:
            new_model, provider_label = self._switch_model(state, model_id)
            logger.info(
                "Session %s: model switched via ACP to %s (provider=%s)",
                session_id,
                new_model,
                provider_label,
            )
            return SetSessionModelResponse()

        state.model = model_id
        state.agent.model = model_id
        invalidate = getattr(state.agent, "_invalidate_system_prompt", None)
        if callable(invalidate):
            invalidate()
        self.session_manager.save_session(session_id)
        logger.info("Session %s: model switched via ACP to %s", session_id, model_id)
        return SetSessionModelResponse()

    async def set_session_mode(
        self, mode_id: str, session_id: str, **kwargs: Any
    ) -> SetSessionModeResponse | None:
        """Persist the editor-requested mode so ACP clients do not fail on mode switches."""
        state = self.session_manager.get_session(session_id)
        if state is None:
            logger.warning("Session %s: mode switch requested for missing session", session_id)
            return None
        setattr(state, "mode", mode_id)
        self.session_manager.save_session(session_id)
        logger.info("Session %s: mode switched to %s", session_id, mode_id)
        return SetSessionModeResponse()

    async def set_config_option(
        self, config_id: str, session_id: str, value: str, **kwargs: Any
    ) -> SetSessionConfigOptionResponse | None:
        """Handle ACP config option updates, including model switching."""
        state = self.session_manager.get_session(session_id)
        if state is None:
            logger.warning("Session %s: config update requested for missing session", session_id)
            return None

        # Handle model config option - actually switch the model
        if config_id == "model" and value:
            logger.info("Session %s: model change requested via ACP config to %s", session_id, value)
            new_model, provider_label = self._switch_model(state, value)
            # Return updated config options with the new current model
            return SetSessionConfigOptionResponse(
                config_options=self._get_model_config_options(new_model, state=state)
            )

        # For other config options, just store them
        options = getattr(state, "config_options", None)
        if not isinstance(options, dict):
            options = {}
        options[str(config_id)] = value
        setattr(state, "config_options", options)
        self.session_manager.save_session(session_id)
        logger.info("Session %s: config option %s updated", session_id, config_id)
        return SetSessionConfigOptionResponse(config_options=[])
