"""Phase 7 Block E — cite-or-refuse hard gate.

Run-time enforcement that any DIH-specific claim in the bot's response is
either backed by a `[slug]` citation OR the bot explicitly says "Not in the
brain". A soft-fabricated DIH claim ("there is a note touching on it…"
without an MCP call) is replaced with a transparent refusal message.

Public entry: ``run_gate(user_message, response_text, sender_id, oauth_agent_name)
-> str`` — call from ``claude_code_runtime`` as a response-transform step.
"""

import logging

# Surface every gate decision in the hermes journal at INFO. Pilot-stage
# observability — track which branch fires per turn, tune persona + classifier
# heuristics from the trail without recompiling.
logging.getLogger("agent.block_e").setLevel(logging.INFO)

from agent.block_e.gate import run_gate  # noqa: E402

__all__ = ["run_gate"]
