"""Ranked fault-hypothesis engine.

Given an anomaly result and the fitted baseline, the engine scores every
fault mode in the knowledge base:

* A signature rule fires when the feature breached in the rule's
  direction; it contributes its weight scaled by log(1 + |z|) so that
  stronger breaches matter more, with diminishing returns.
* Evidence *against* a hypothesis comes from signature features of the
  other fault modes that are healthy — recorded explicitly, never
  hidden, in the spirit of MaintAI's competing-hypotheses requirement.
* The top hypothesis must beat the runner-up by ``margin`` (in score
  points) for the diagnosis to be confident; otherwise the engine
  reports an ambiguous diagnosis so the workflow requests more evidence
  or escalates instead of inventing certainty.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from orbit.detection.baseline import AnomalyResult, BaselineDetector
from orbit.diagnosis.knowledge import (
    ActionPlan,
    FaultMode,
    KNOWLEDGE_BASE,
    SignatureRule,
)


AMBIGUITY_MARGIN = 1.0
MIN_TOP_SCORE = 2.0
# Beyond this z-score a breach is definitive; extra z adds no diagnostic
# information (huge z mostly reflects how tight the healthy baseline was).
Z_SATURATION = 10.0


def _dedupe_by_feature(items: List[EvidenceItem]) -> List[EvidenceItem]:
    seen = set()
    result = []
    for item in items:
        if item.feature in seen:
            continue
        seen.add(item.feature)
        result.append(item)
    return result


@dataclass(frozen=True)
class EvidenceItem:
    feature: str
    value: float
    z: float
    direction: str
    rationale: str

    def as_dict(self) -> Dict[str, object]:
        return {
            "feature": self.feature,
            "value": round(self.value, 4),
            "z": round(self.z, 2),
            "direction": self.direction,
            "rationale": self.rationale,
        }


@dataclass(frozen=True)
class FaultHypothesis:
    mode_id: str
    display_name: str
    score: float
    supporting: Tuple[EvidenceItem, ...]
    contradicting: Tuple[EvidenceItem, ...]
    confidence: str  # "high" | "medium" | "low"

    @property
    def mode(self) -> FaultMode:
        return KNOWLEDGE_BASE[self.mode_id]


@dataclass(frozen=True)
class DiagnosisResult:
    is_confident: bool
    hypotheses: Tuple[FaultHypothesis, ...]
    primary: Optional[FaultHypothesis]
    recommended_action: Optional[ActionPlan]
    ambiguity_note: str = ""

    def as_summary(self) -> Dict[str, object]:
        return {
            "confident": self.is_confident,
            "primary": self.primary.display_name if self.primary else None,
            "action": self.recommended_action.title if self.recommended_action else None,
            "hypotheses": [
                {
                    "mode": h.display_name,
                    "score": round(h.score, 2),
                    "confidence": h.confidence,
                    "n_supporting": len(h.supporting),
                    "n_contradicting": len(h.contradicting),
                }
                for h in self.hypotheses
            ],
            "ambiguity_note": self.ambiguity_note or None,
        }


class DiagnosisEngine:
    """Deterministic, evidence-based fault-hypothesis ranking."""

    def __init__(self, margin: float = AMBIGUITY_MARGIN, min_top_score: float = MIN_TOP_SCORE):
        if margin <= 0:
            raise ValueError("margin must be positive")
        self.margin = float(margin)
        self.min_top_score = float(min_top_score)

    def _rule_score(self, rule: SignatureRule, z_value: float, direction: str) -> float:
        if direction != rule.direction:
            return 0.0
        z_eff = min(abs(z_value), Z_SATURATION)
        return rule.weight * math.log1p(z_eff)

    def diagnose(
        self,
        anomaly: AnomalyResult,
        features: Dict[str, float],
        detector: BaselineDetector,
    ) -> DiagnosisResult:
        if not anomaly.is_anomaly:
            return DiagnosisResult(
                is_confident=False,
                hypotheses=(),
                primary=None,
                recommended_action=None,
                ambiguity_note="no anomaly to diagnose",
            )
        z = detector.z_scores(features)
        breaches = {b.feature: b for b in anomaly.breaches}

        hypotheses: List[FaultHypothesis] = []
        for mode in KNOWLEDGE_BASE.values():
            supporting: List[EvidenceItem] = []
            missing_signature: List[EvidenceItem] = []
            alternatives: List[EvidenceItem] = []
            score = 0.0
            for rule in mode.signature:
                breach = breaches.get(rule.feature)
                if breach is None:
                    missing_signature.append(
                        EvidenceItem(
                            feature=rule.feature,
                            value=float(features[rule.feature]),
                            z=z[rule.feature],
                            direction="normal",
                            rationale=mode.contradicting_note
                            or f"{rule.feature} within baseline contradicts {mode.display_name}.",
                        )
                    )
                    continue
                contribution = self._rule_score(rule, breach.z, breach.direction)
                if contribution > 0:
                    score += contribution
                    supporting.append(
                        EvidenceItem(
                            feature=rule.feature,
                            value=breach.value,
                            z=breach.z,
                            direction=breach.direction,
                            rationale=rule.rationale,
                        )
                    )
            for other in KNOWLEDGE_BASE.values():
                if other.mode_id == mode.mode_id:
                    continue
                for rule in other.signature:
                    breach = breaches.get(rule.feature)
                    if breach is not None:
                        alternatives.append(
                            EvidenceItem(
                                feature=rule.feature,
                                value=breach.value,
                                z=breach.z,
                                direction=breach.direction,
                                rationale=(
                                    f"{rule.feature} breach instead supports "
                                    f"{other.display_name}."
                                ),
                            )
                        )
            contradicting = _dedupe_by_feature(missing_signature) + _dedupe_by_feature(alternatives)
            confidence = "high" if score >= self.min_top_score + self.margin else (
                "medium" if score >= self.min_top_score else "low"
            )
            hypotheses.append(
                FaultHypothesis(
                    mode_id=mode.mode_id,
                    display_name=mode.display_name,
                    score=score,
                    supporting=tuple(supporting),
                    contradicting=tuple(contradicting),
                    confidence=confidence,
                )
            )

        hypotheses.sort(key=lambda h: h.score, reverse=True)
        top, runner_up = hypotheses[0], hypotheses[1]

        confident = (
            top.score >= self.min_top_score
            and top.score - runner_up.score >= self.margin
        )
        if confident:
            action = top.mode.actions[0]
            note = ""
        else:
            action = None
            note = (
                f"Top hypothesis '{top.display_name}' (score {top.score:.2f}) is not "
                f"separated from '{runner_up.display_name}' (score {runner_up.score:.2f}) "
                f"by the required margin {self.margin:.1f}; requesting more evidence "
                "before any intervention."
            )
        return DiagnosisResult(
            is_confident=confident,
            hypotheses=tuple(hypotheses),
            primary=top if top.score >= self.min_top_score else None,
            recommended_action=action,
            ambiguity_note=note,
        )
