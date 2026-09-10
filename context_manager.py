"""Context management for LLM decisions.

Message layers (rebuilt for every request):
  1. system: constant system prompt (rulebook + contract).
  2. system: rolling window of recent (state -> decision) turns.
  3. system: regenerated RUN MEMORY.
  4. user: the CURRENT state -- NEVER blind-truncated.

Budget priority (highest first):
  1. CURRENT STATE -- always preserved in full.
  2. RUN MEMORY -- preserved (it is tiny and regenerated).
  3. RECENT HISTORY -- disposable, oldest trimmed first, each excerpt capped.

If a pathological single state exceeds the per-request transport cap, it is
compressed STRUCTURALLY (whole lines/blocks removed by priority), never by
blind character slicing: legal actions, the current hand, enemies, player
state, potion/relic state and choice options survive first; repeated static
reference definitions and upgrade previews are dropped first.

History entries store a situation SUMMARY (screen type + key numbers), not
the full raw state blob, so recent decisions stay useful without evicting
the current state.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

# Hard per-request transport guard: an extremely large single state could
# exceed the model context window on its own. When triggered, the state is
# STRUCTURALLY compressed (see _structured_compress), never substring-cut.
CURRENT_STATE_HARD_CAP = 100_000

# Lines whose loss costs the least decision value. They are removed whole,
# in this order, until the text fits the transport cap.
_STATIC_LINE_RE = re.compile(r"\(static reference\)")
_DECK_DEF_LINE_RE = re.compile(r"^\s{2,}\S.*: Name: ")
_PREVIEW_LINE_RE = re.compile(r"^\s*Upgrade preview \(as shown by the game\):")


@dataclass
class DecisionTurn:
    state_excerpt: str
    response: str
    note: str = ""  # e.g. execution result / error feedback


@dataclass
class ContextConfig:
    max_history_turns: int = 8
    max_state_chars: int = 4000  # history excerpt size only
    max_context_chars: int = 24000


def _structured_compress(text: str, limit: int) -> str:
    """Shrink a state text by dropping WHOLE low-value lines, never by
    slicing mid-text. Priority to keep: legal actions, hand, enemies,
    player/potion/relic state, options. Priority to drop: repeated static
    reference definitions, deck definition blocks, upgrade previews."""
    if len(text) <= limit:
        return text
    lines = text.splitlines(keepends=True)

    def size() -> int:
        return sum(len(l) for l in lines)

    # Drop whole low-value line classes until it fits.
    if size() > limit:
        lines = [l for l in lines if not _STATIC_LINE_RE.search(l)]
    if size() > limit:
        lines = [l for l in lines if not _DECK_DEF_LINE_RE.search(l)]
    if size() > limit:
        lines = [l for l in lines if not _PREVIEW_LINE_RE.search(l)]
    if size() > limit:
        # Last resort: drop pile composition lines (composition is still
        # described in RunMemory/history), then blank-line squeeze.
        lines = [l for l in lines if not l.strip().startswith("Composition:")]
    if size() > limit:
        # Absolute last resort: drop the longest lines one by one, keeping
        # the remaining order intact (still whole lines, never a substring).
        indexed = sorted(enumerate(lines), key=lambda t: -len(t[1]))
        dropped = set()
        total = sum(len(l) for l in lines)
        for idx, line in indexed:
            if total <= limit:
                break
            if len(line) < 80:
                break  # don't nuke short structural lines
            dropped.add(idx)
            total -= len(line)
        lines = [l for i, l in enumerate(lines) if i not in dropped]
    return "".join(lines)


@dataclass
class ContextManager:
    config: ContextConfig = field(default_factory=ContextConfig)
    history: list[DecisionTurn] = field(default_factory=list)

    def reset(self) -> None:
        self.history.clear()

    def add_decision(self, state_text: str, response: str, note: str = "") -> None:
        self.history.append(
            DecisionTurn(
                state_excerpt=self._clip(state_text, self.config.max_state_chars),
                response=response,
                note=note,
            )
        )
        # The setting is a turn limit, not a hidden multiplier.  Keeping
        # three times the advertised amount made long runs steadily drag
        # stale tactical context into unrelated decisions.
        limit = self.config.max_history_turns
        if limit <= 0:
            # max_history_turns=0 disables history entirely (note: history
            # slicing must not use [-0:], which is the WHOLE list).
            self.history = []
        elif len(self.history) > limit:
            self.history = self.history[-limit:]

    def build_messages(
        self,
        system_prompt: str,
        run_memory_text: str,
        state_text: str,
    ) -> list[dict[str, str]]:
        messages: list[dict[str, str]] = [{"role": "system", "content": system_prompt}]

        # The current state always goes out whole. Only the pathological
        # transport cap applies, and it compresses structurally.
        current_state = _structured_compress(state_text, CURRENT_STATE_HARD_CAP)

        # Budget: history is trimmed first; RUN MEMORY and the current
        # state are never cut to fit the soft budget.
        budget = self.config.max_context_chars
        budget -= len(system_prompt)
        budget -= len(run_memory_text) + 200
        budget -= len(current_state)

        # Trim OLDEST history first when over budget.
        selected: list[DecisionTurn] = []
        for turn in reversed(self.history):
            cost = len(turn.state_excerpt) + len(turn.response) + len(turn.note) + 60
            if budget - cost < 0:
                break
            budget -= cost
            selected.append(turn)
        selected.reverse()

        if selected:
            messages.append({
                "role": "system",
                "content": (
                    "RECENT DECISIONS (oldest first, context only). Each shows the"
                    " situation you faced, the JSON you answered, and what happened:"
                ),
            })
            for i, turn in enumerate(selected):
                msg = f"[t-{len(selected) - i}] STATE:\n{turn.state_excerpt}\n"
                msg += f"YOUR ANSWER: {turn.response}"
                if turn.note:
                    msg += f"\nRESULT/FEEDBACK: {turn.note}"
                messages.append({"role": "system", "content": msg})

        messages.append({"role": "system", "content": run_memory_text})
        messages.append({
            "role": "user",
            "content": current_state
            + "\n\nAnswer now with EXACTLY ONE valid JSON object and nothing else.",
        })
        return messages

    @staticmethod
    def _clip(text: str, limit: int) -> str:
        """History excerpt cap (fine to clip: history is disposable)."""
        if len(text) <= limit:
            return text
        return text[: limit - 20] + f"\n...[truncated {len(text) - limit} chars]"
