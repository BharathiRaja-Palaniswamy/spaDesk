"""Turns the graded results of a run into the comparison of the two modes.

Prints three things:
- the pass rate and latency of each mode, side by side;
- the cases whose result differs between the modes. The model and the prompt
  are the same in both, so these are where retrieval changed the answer: a
  case that passes with the whole document and fails with RAG is a retrieval
  miss.
- the totals.
"""

from dataclasses import dataclass
from statistics import mean

MODES = ("full", "rag")


@dataclass(frozen=True)
class Result:
    """One case asked in one mode, after grading."""

    case: str  # the case's id
    mode: str  # "full" (whole document) or "rag" (retrieved chunks)
    problems: list[str]  # from checks.grade(); empty when the case passed
    seconds: float  # how long the request took, end to end

    @property
    def passed(self) -> bool:
        """True when no check found a problem."""
        return not self.problems


def print_summary(results: list[Result]) -> None:
    """Prints the side-by-side comparison of the modes for one run."""
    by_mode = {mode: [r for r in results if r.mode == mode] for mode in MODES}

    print(f"\n{'':14}{'full':>10}{'rag':>10}")
    print(f"{'pass rate':14}" + "".join(f"{_rate(by_mode[m]):>10}" for m in MODES))
    print(f"{'avg latency':14}" + "".join(f"{mean(r.seconds for r in by_mode[m]):>9.1f}s" for m in MODES))
    print(f"{'slowest':14}" + "".join(f"{max(r.seconds for r in by_mode[m]):>9.1f}s" for m in MODES))

    # Pair each case's two results, and keep the pairs that disagree.
    full = {r.case: r for r in by_mode["full"]}
    differ = [(full[r.case], r) for r in by_mode["rag"] if r.case in full and full[r.case].passed != r.passed]
    print(f"\nDiffer between modes ({len(differ)}):")
    for f, r in differ:
        failed = r if f.passed else f
        print(f"  {f.case:32} full {_word(f)}  rag {_word(r)}   {'; '.join(failed.problems)}")

    print(f"\n{sum(r.passed for r in results)}/{len(results)} passed in all")


def _rate(results: list[Result]) -> str:
    """e.g. "40/43"."""
    return f"{sum(r.passed for r in results)}/{len(results)}"


def _word(result: Result) -> str:
    """PASS or FAIL."""
    return "PASS" if result.passed else "FAIL"
