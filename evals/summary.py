"""Turns the graded cases of a run into scores.

For each target, prints:
- each layer's pass rate, over the cases that check it, so a failure shows
  where it comes from: a wrong path, a detail made up, a wrong outcome, or a
  wrong answer; and the wrong-path rate (routing failures);
- the pass rate of each split (happy, edge, known_failure, adversarial):
  a model can do well on common requests and still fail the harmful ones;
- the time a case took, and the model calls and tokens it used.

And the cases that passed in all.
"""

from dataclasses import dataclass, field

from .golden import SPLITS, TARGETS

LAYERS = ("routing", "extraction", "outcome", "answer")


@dataclass
class Result:
    """One case, after grading."""

    case: str  # the case's id
    target: str
    split: str
    layers: dict[str, list[str]]  # each layer's problems; empty lists when it passed
    seconds: float  # how long the case took, end to end
    calls: int = 0  # model calls the case made, judge excluded
    tokens: int = 0
    conversation: list[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        """True when no layer found a problem."""
        return not any(self.layers.values())


def _rate(passed: int, total: int) -> str:
    """e.g. "40/43 (93%)"; "-" when nothing was scored."""
    return f"{passed}/{total} ({passed / total:.0%})" if total else "-"


def summary(results: list[Result]) -> list[str]:
    """The lines to print for a run."""
    lines = []
    for target in TARGETS:
        mine = [r for r in results if r.target == target]
        if not mine:
            continue
        lines += [f"\n{target} ({len(mine)} cases)"]
        for layer in LAYERS:
            scored = [r.layers[layer] for r in mine if layer in r.layers]
            if scored:
                lines.append(f"  {layer:14} {_rate(sum(not p for p in scored), len(scored))}")
        if routed := [r.layers["routing"] for r in mine if "routing" in r.layers]:
            lines.append(f"  {'wrong path':14} {_rate(sum(bool(p) for p in routed), len(routed))}")
        for split in SPLITS:
            if in_split := [r for r in mine if r.split == split]:
                lines.append(f"  {split:14} {_rate(sum(r.passed for r in in_split), len(in_split))}")
        lines.append(f"  {'all layers':14} {_rate(sum(r.passed for r in mine), len(mine))}")
        lines.append(f"  {'avg time':14} {sum(r.seconds for r in mine) / len(mine):.1f}s, "
                     f"{sum(r.calls for r in mine) / len(mine):.1f} model calls, "
                     f"{sum(r.tokens for r in mine) / len(mine):,.0f} tokens a case")
    lines.append(f"\nIn all: {_rate(sum(r.passed for r in results), len(results))} passed")
    return lines
