"""Claude Code CLI (subprocess) provider profile.

Drives turns through a `claude --print --output-format json` subprocess so
Anthropic Max-subscription credentials at ~/.claude/.credentials.json are
spent against the base allowance rather than the OAuth "extras" pool that
the in-process anthropic adapter taps. Mirrors the openai-codex /
codex_app_server pattern but with claude CLI as the upstream binary.
"""

from providers import register_provider
from providers.base import ProviderProfile

claude_code_cli = ProviderProfile(
    name="claude-code-cli",
    aliases=("claude-cli", "claude_code_cli"),
    api_mode="claude_code_cli",
    env_vars=(),  # claude CLI reads ~/.claude/.credentials.json itself
    base_url="",  # no HTTP — subprocess transport
    auth_type="oauth_external",
    supports_health_check=False,  # no /models endpoint
    display_name="Claude Code CLI",
    description="Claude Code CLI (Max subscription, subprocess)",
    signup_url="https://claude.com/claude-code",
    fallback_models=(
        "claude-opus-4-8",
        "claude-sonnet-4-6",
        "claude-haiku-4-5",
    ),
)

register_provider(claude_code_cli)
