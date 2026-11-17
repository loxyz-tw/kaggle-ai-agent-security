from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from aicomp_sdk.core.predicates import eval_predicates
from aicomp_sdk.guardrails.base import DecisionAction, GuardrailBase

from .guardrails import GuardrailVariantSpec, build_private_guardrail_variants, explain_decision


@dataclass(frozen=True, slots=True)
class StaticToolEvent:
    name: str
    args: Mapping[str, Any]
    side_effect: str
    source: str
    output: str = ""
    scope: str = ""
    ok: bool = True


@dataclass(frozen=True, slots=True)
class StaticCandidate:
    candidate_id: str
    messages: tuple[str, ...]
    expected_predicates: tuple[str, ...]
    tool_events: tuple[StaticToolEvent, ...]
    metadata: Mapping[str, Any] | None = None

    @property
    def last_user(self) -> str:
        return self.messages[-1] if self.messages else ""


@dataclass(frozen=True, slots=True)
class GuardrailReplayResult:
    guardrail_name: str
    survived: bool
    triggered_predicates: tuple[str, ...]
    blocked_steps: tuple[str, ...]
    decision_log: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class RankedCandidate:
    candidate: StaticCandidate
    survival_count: int
    surviving_guardrails: tuple[str, ...]
    replay_results: tuple[GuardrailReplayResult, ...]


def rank_candidates_by_guardrail_survival(
    candidates: Sequence[StaticCandidate],
    *,
    guardrail_variants: Sequence[GuardrailVariantSpec] | None = None,
) -> list[RankedCandidate]:
    variants = tuple(guardrail_variants or build_private_guardrail_variants())
    ranked: list[RankedCandidate] = []
    for candidate in candidates:
        replay_results = tuple(
            _replay_candidate_against_guardrail(candidate, spec)
            for spec in variants
        )
        surviving_guardrails = tuple(
            result.guardrail_name for result in replay_results if result.survived
        )
        ranked.append(
            RankedCandidate(
                candidate=candidate,
                survival_count=len(surviving_guardrails),
                surviving_guardrails=surviving_guardrails,
                replay_results=replay_results,
            )
        )

    return sorted(
        ranked,
        key=lambda record: (
            record.survival_count,
            len(record.candidate.expected_predicates),
            -len(record.candidate.tool_events),
            record.candidate.candidate_id,
        ),
        reverse=True,
    )


def candidate_from_mapping(record: Mapping[str, Any]) -> StaticCandidate:
    tool_events = tuple(_tool_event_from_mapping(item) for item in record.get("tool_events", ()))
    return StaticCandidate(
        candidate_id=str(record.get("candidate_id") or record.get("name") or "candidate"),
        messages=tuple(str(message) for message in record.get("messages", ())),
        expected_predicates=tuple(str(name) for name in record.get("expected_predicates", ())),
        tool_events=tool_events,
        metadata=record,
    )


def _tool_event_from_mapping(record: Mapping[str, Any]) -> StaticToolEvent:
    return StaticToolEvent(
        name=str(record["name"]),
        args=dict(record.get("args", {})),
        side_effect=str(record.get("side_effect", "")),
        source=str(record.get("source", "")),
        output=str(record.get("output", "")),
        scope=str(record.get("scope", "")),
        ok=bool(record.get("ok", True)),
    )


def _replay_candidate_against_guardrail(
    candidate: StaticCandidate,
    variant: GuardrailVariantSpec,
) -> GuardrailReplayResult:
    guardrail = variant.guardrail_factory()
    trace_events: list[dict[str, Any]] = []
    decision_log: list[str] = []
    blocked_steps: list[str] = []
    last_tool_output = ""

    for event in candidate.tool_events:
        context = {
            "recent_sources": [entry["source"] for entry in trace_events[-5:]],
            "recent_tools": [entry["name"] for entry in trace_events[-5:]],
            "last_user": candidate.last_user,
            "side_effect": event.side_effect,
            "scope": event.scope,
            "last_tool_output": last_tool_output,
        }
        decision = guardrail.decide(event.name, event.args, context)
        decision_log.append(f"{event.name}:{explain_decision(decision)}")
        if decision.action is not DecisionAction.ALLOW:
            blocked_steps.append(event.name)
            continue

        emitted = {
            "name": event.name,
            "args": dict(event.args),
            "ok": event.ok,
            "output": event.output,
            "side_effect": event.side_effect,
            "scope": event.scope,
            "source": event.source,
            "error": None,
        }
        trace_events.append(emitted)
        last_tool_output = event.output

    predicates = eval_predicates(
        {
            "tool_events": trace_events,
            "user_messages": list(candidate.messages),
        }
    )
    triggered_predicates = tuple(predicate["predicate"] for predicate in predicates)
    survived = all(
        expected in triggered_predicates for expected in candidate.expected_predicates
    )
    return GuardrailReplayResult(
        guardrail_name=variant.name,
        survived=survived,
        triggered_predicates=triggered_predicates,
        blocked_steps=tuple(blocked_steps),
        decision_log=tuple(decision_log),
    )
