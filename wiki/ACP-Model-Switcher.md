# ACP Model Switcher

This page documents how the VS Code ACP model switcher works in this customized Hermes install.

## Goal

Make the ACP model switcher behave the same way as Hermes CLI `/model` logic:
- show only providers/models Hermes considers available
- use the same provider/model resolution rules as CLI
- switch the live session runtime correctly
- preserve provider identity when different providers expose the same model name

## Core principle

ACP should not maintain its own model-switch logic.

The ACP server now reuses the shared CLI pipeline in `hermes_cli.model_switch.switch_model()`.
That means ACP and CLI use the same logic for:
- authenticated provider discovery
- model selection and fallback behavior
- provider switching
- runtime resolution (`api_key`, `base_url`, `api_mode`)
- OpenCode special handling
- Copilot/OpenAI/OpenRouter model routing

## Main file

- `acp_adapter/server.py`

## Shared logic used by ACP

### Listing available providers/models

ACP model list construction now uses:
- `hermes_cli.model_switch.list_authenticated_providers()`
- `hermes_cli.models.curated_models_for_provider()`

This keeps the ACP list aligned with Hermes CLI `/model` provider visibility.

There is also a fallback merge from `~/.hermes/auth.json` so API-key-backed providers in `credential_pool` still appear in ACP/editor sessions where environment detection alone is incomplete.

### Switching models

ACP switching now uses:
- `hermes_cli.model_switch.switch_model()`

The ACP server passes the current session runtime into that shared pipeline:
- current provider
- current model
- current base URL
- current API key
- optional explicit provider from ACP's `provider:model` encoded model id

The result is then applied to the live session with:
- `state.agent.switch_model(...)`

This is important because updating only `state.agent.model` / `provider` fields is not enough. The underlying runtime client must also be rebuilt or swapped correctly.

## Why provider-qualified ACP model IDs are required

Different providers can expose the same bare model name, for example:
- `gpt-5.4`
- `minimax-m2.7`

So ACP advertises model ids in this form:
- `openai-codex:gpt-5.4`
- `copilot:gpt-5.4`
- `opencode-go:minimax-m2.7`
- `openrouter:openai/gpt-5.4`

This prevents ambiguity and lets ACP switch the correct provider/runtime.

## Session response fields used

ACP populates both:
- `models` (`SessionModelState`) — used by the native ACP model dropdown
- `config_options` — mirrors the same options for clients that read config options

## Runtime behavior by provider

### OpenAI Codex
- Uses Codex/ChatGPT account runtime
- Uses live/provider-aware model switch result from shared CLI logic

### GitHub Copilot
- Uses the Copilot runtime resolved by the shared switch pipeline
- Supports GPT and non-GPT models according to Hermes CLI logic

### OpenCode Go
- Uses shared CLI logic plus live runtime swap
- Important for models like `minimax-m2.7` that need `anthropic_messages`
- Correct base URL handling is preserved

## Bugs fixed by this approach

### 1. ACP drift from CLI behavior

Previously ACP had custom logic for provider discovery and switching.
That caused behavior differences from `/model` in Hermes CLI.

Fix:
- ACP now delegates to shared CLI model-switch logic.

### 2. Model name collisions across providers

Previously ACP used bare model IDs, so provider identity was ambiguous.

Fix:
- ACP now uses provider-qualified model IDs.

### 3. Runtime metadata changed without swapping the real client

Previously ACP could update the session's model/provider fields without calling the live agent runtime switch.
That caused failures like:
- wrong provider endpoint being used
- missing Anthropic-compatible client after OpenCode Go MiniMax switch
- `'NoneType' object has no attribute 'messages'`

Fix:
- ACP now applies the resolved switch result through `state.agent.switch_model(...)`.

### 4. OpenAI/Copilot/OpenCode special routing mismatches

Previously ACP could choose API behavior based on stale/default config instead of the selected model.

Fix:
- shared CLI switch logic now determines the runtime result, then ACP applies it.

## Files touched

- `acp_adapter/server.py`
- `tests/acp/test_server.py`

## Tests

Relevant ACP tests live in:
- `tests/acp/test_server.py`

These cover:
- `models` field returned in session responses
- duplicate model IDs kept distinct by provider
- live agent instance is preserved during ACP switching
- provider/runtime switching works through ACP
- credential-pool-backed providers appear in ACP discovery

## Practical summary

If ACP model switching breaks again, the first rule is:

Do not add more custom ACP-only model resolution logic.

Instead:
1. reproduce the issue
2. compare against Hermes CLI `/model`
3. fix the shared model-switch pipeline if needed
4. keep ACP as a thin adapter over the shared CLI behavior
