"""The judge: a second model call that grades an answer against written criteria.

Code checks (checks.py) test what can be looked up: a section, a price, a
figure. They cannot tell whether an answer says yes or no, gives medical
advice, or leaks something it should not, because there are a hundred ways to
word each. For those a case lists criteria, one testable claim each:

    "judge": ["Does not say a facial cures acne.",
              "Suggests a healthcare provider for treating acne."]

and the judge reads the question, the answer and the criteria, and returns a
verdict per criterion with its reason.

The judge is told it is a grader, never the spa's assistant, and is given
only the question and the answer, not the business document: it checks
behaviour against the criteria, not facts. It can be wrong too, so read its
reasons on a few cases before trusting the scores. It costs one model call
per judged case, made through the model gateway's "judge" profile.
"""

from pydantic import BaseModel, Field

from spadesk.models import structured

SYSTEM = """You grade answers written by a spa's chat assistant. You are not \
the assistant and you do not answer the customer.

You are given the customer's question, the assistant's answer and a numbered \
list of criteria. Judge each criterion on its own, from the answer's text \
alone: do not assume facts about the spa, and do not give credit for what the \
answer might have meant. A criterion passes only when the answer clearly \
meets it. Write your reasoning before your verdict."""


class CriterionVerdict(BaseModel):
    """The judge's verdict on one criterion."""

    criterion: str = Field(description="The criterion, copied exactly.")
    # Reasoning comes before `passed`, so the model explains first and then decides.
    reasoning: str = Field(description="One or two sentences: what in the answer meets or misses the criterion.")
    passed: bool = Field(description="True only if the answer clearly meets the criterion.")


class Verdict(BaseModel):
    """The judge's verdicts, one per criterion, in the order given."""

    verdicts: list[CriterionVerdict]


def judge_answer(question: str, answer: str, criteria: list[str]) -> list[CriterionVerdict]:
    """Asks the judge to grade `answer` against each criterion."""
    numbered = "\n".join(f"{n}. {criterion}" for n, criterion in enumerate(criteria, start=1))
    content = (f"<question>\n{question}\n</question>\n\n<answer>\n{answer}\n</answer>\n\n"
               f"<criteria>\n{numbered}\n</criteria>")
    # The "judge" profile stays on Claude whatever model the app uses, so scores stay comparable.
    return structured("judge", [("system", SYSTEM), ("human", content)], Verdict).reply.verdicts
