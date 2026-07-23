"""v0 turn classifier: maps each agent utterance to a journey flow state by
keyword scoring. An LLM classifier replaces this in a later phase (the
interface is just `classify(utterance) -> state | None`)."""

from __future__ import annotations

from .flows import JourneyFlow
from .textutil import keyword_matches


class KeywordStateClassifier:
    def __init__(self, flow: JourneyFlow):
        self.flow = flow

    def classify(self, utterance: str) -> str | None:
        best_name: str | None = None
        best_key = (0, 0, "")
        for name, state in sorted(self.flow.states.items()):
            matched = [kw for kw in state.keywords if keyword_matches(kw, utterance)]
            if not matched:
                continue
            # Tie-break on matched keyword mass (more specific wins), then name.
            key = (len(matched), sum(len(k) for k in matched), name)
            if key > best_key:
                best_key = key
                best_name = name
        return best_name
