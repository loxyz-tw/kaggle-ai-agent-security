from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from aicomp_sdk.agents import build_agent_factory
from aicomp_sdk.attacks import AttackAlgorithmBase, AttackCandidate, AttackRunConfig
from aicomp_sdk.evaluation.ops import (
    AttackEvalOptions,
    eval_attack,
    summarize_attack_findings,
)

from .attack_families import AttackFamily


@dataclass(frozen=True, slots=True)
class FamilyEvalResult:
    family_name: str
    guardrail_name: str
    findings_count: int
    unique_cells: int
    raw_score: float
    normalized_score: float
    predicate_signatures: tuple[tuple[tuple[str, int], ...], ...]

    def as_measurement(self) -> dict[str, Any]:
        return {
            "findings_count": self.findings_count,
            "unique_cells": self.unique_cells,
            "raw_score": self.raw_score,
            "normalized_score": self.normalized_score,
            "predicate_signatures": self.predicate_signatures,
        }


def evaluate_family(
    family: AttackFamily,
    *,
    agent_selection: str,
    guardrail_factory: type,
    guardrail_name: str,
    budget_s: float = 20.0,
    max_tool_hops: int = 8,
) -> FamilyEvalResult:
    agent_factory = build_agent_factory(agent_selection)

    class FamilyAttack(AttackAlgorithmBase):
        def run(self, env, config):
            del env, config
            return [AttackCandidate.from_messages(family.messages)]

    findings = eval_attack(
        FamilyAttack,
        budget_s=budget_s,
        agent_factory=agent_factory,
        env_selection="sandbox",
        options=AttackEvalOptions(
            run_config=AttackRunConfig(time_budget_s=budget_s, max_tool_hops=max_tool_hops),
            guardrail_factory=guardrail_factory,
        ),
    )
    summary = summarize_attack_findings(findings)
    return FamilyEvalResult(
        family_name=family.name,
        guardrail_name=guardrail_name,
        findings_count=summary["findings_count"],
        unique_cells=summary["unique_cells"],
        raw_score=summary["score_raw"],
        normalized_score=summary["score"],
        predicate_signatures=tuple(
            tuple((predicate["predicate"], int(predicate["severity"])) for predicate in finding["predicates"])
            for finding in findings
        ),
    )
