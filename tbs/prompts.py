"""Listwise prompts shared by the TBS sorter (OrderByLLM wording)."""

from __future__ import annotations

from typing import Sequence


# OrderByLLM (Zhao et al.) NBA listwise prompt — names only, model uses parametric knowledge.
NBA_KNOWLEDGE_COMPARISON_PROMPT = """Given a list of nba players with id at the front:
{keys}
Based on your own knowledge, sort the players in ascending order based on player height (in meters, rounded to two decimal places).
Break ties based on the player name in alphabetical order.
Output a JSON list of integer ids, representing the player from shortest to tallest.
"""

# Rescued version of NBA_KNOWLEDGE_COMPARISON_PROMPT: anchors the model to an explicit,
# stated height estimate per player before it sorts, spells out the tie rule precisely,
# and bans omission/duplication/fabricated ties.
NBA_KNOWLEDGE_COMPARISON_PROMPT_RESCUED = """You will sort NBA players by height, shortest to tallest, using your own knowledge of each player's officially listed height (as commonly reported on NBA.com / Basketball-Reference).

Players (id: name):
{keys}
Do this internally, then output only the final JSON:
1. For each id, recall that player's officially listed height in meters, rounded to exactly two decimal places. Use the same canonical value you would give if asked about that player alone — do not nudge it up or down just to create or avoid a tie with a neighboring player.
2. Sort all ids in ascending order by that height value (shortest first).
3. If two or more players land on the exact same height value, break the tie by comparing their full names alphabetically, A to Z, case-insensitive, comparing the whole string exactly as given (including punctuation and spaces).
4. Every id from the list above must appear in "order" exactly once — no omissions, no duplicates, no ids that were not given to you.

Respond with ONLY this JSON object and nothing else — no markdown code fences, no commentary before or after it:
{{"heights": {{"<id>": <height_meters_2dp>, ...}}, "order": [<id>, <id>, ...]}}
"""


ZHAO_SYSTEM_PROMPT = "You are a helpful agent. Think step by step. Output a JSON object."


def _zhao_numbered_nba_keys(batch: Sequence[str]) -> str:
    """1-based id blocks matching OrderByLLM ``create_numbered_rankings``."""
    return "\n".join(f"id:{i + 1}\n{name}\n" for i, name in enumerate(batch))

