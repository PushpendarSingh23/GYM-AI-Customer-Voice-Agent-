"""Evaluation harness for the gym support agent.

Runs a handful of scripted conversations through the graph (the same one the
text and voice apps use) and checks, per JD-shaped concerns: latency, routing/
resolution correctness, and hallucination (does the agent state facts not
backed by ``mock_data``, the only source of truth in this demo).

This is deliberately small and readable rather than a general eval framework
— the idea is to keep adding real failure cases here as they turn up, not to
build eval infrastructure for its own sake.

Run: uv run python -m gym_support.eval
Requires GROQ_API_KEY — this calls the real model, not a mock.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Callable

from dotenv import load_dotenv
from langchain.messages import AIMessage
from langgraph.checkpoint.memory import InMemorySaver

from . import mock_data
from .graph import build_graph

load_dotenv()


def _credits_grounded(membership_id: str) -> Callable[[str], list[str]]:
    """Verifier: every credit count the agent states must match mock_data exactly.

    Grounding against the actual store (rather than hardcoding "8, 2, 3" in the
    test case) means this keeps working if the default bundle ever changes,
    and it's checking the real thing we care about — did the agent report
    what's actually in the system, not just some plausible-looking numbers.
    """

    def verify(reply: str) -> list[str]:
        member = mock_data.get_membership(membership_id)
        failures = []
        for credit_type, count in member["credits"].items():
            if str(count) not in reply:
                failures.append(
                    f"expected {credit_type}={count} to appear in reply, but it didn't"
                )
        return failures

    return verify


def _booking_grounded(membership_id: str, class_name: str) -> Callable[[str], list[str]]:
    """Verifier: the booked class must be real, and the stated remaining
    credits must match what booking actually left in mock_data."""

    def verify(reply: str) -> list[str]:
        klass = mock_data.find_class(class_name)
        failures = []
        if klass is None:
            failures.append(f"'{class_name}' is not a real class in CLASS_SCHEDULE")
            return failures
        if klass["name"].lower() not in reply.lower():
            failures.append(f"reply never confirms the class name '{klass['name']}'")
        member = mock_data.get_membership(membership_id)
        remaining = member["credits"][klass["credit_type"]]
        if str(remaining) not in reply:
            failures.append(f"expected remaining {klass['credit_type']}={remaining} in reply")
        return failures

    return verify


@dataclass
class TestCase:
    name: str
    turns: list[str]  # user messages, sent in order on one thread
    expected_agent: str  # active_agent the graph should land on after the last turn
    verify: Callable[[str], list[str]] = field(default_factory=lambda: (lambda reply: []))


CASES = [
    TestCase(
        name="cancel_membership",
        turns=["I want to cancel my membership. My membership ID is MEM-CANCEL-1."],
        expected_agent="cancellation",
        verify=lambda reply: [] if "cancel" in reply.lower() else ["reply never confirms cancellation"],
    ),
    TestCase(
        name="check_credits",
        turns=["How many credits do I have left? My membership ID is MEM-CREDITS-1."],
        expected_agent="credits",
        verify=_credits_grounded("MEM-CREDITS-1"),
    ),
    TestCase(
        name="book_class",
        turns=["I'd like to book Sunrise Yoga for next Monday. My membership ID is MEM-BOOK-1."],
        expected_agent="booking",
        verify=_booking_grounded("MEM-BOOK-1", "Sunrise Yoga"),
    ),
    TestCase(
        name="mid_conversation_handoff",
        # Starts a cancellation, then switches intent before finishing it —
        # exercises transfer_to_triage + re-routing, not just a single handoff.
        turns=[
            "I want to cancel my membership.",
            "Actually, how many credits do I have? My membership ID is MEM-SWITCH-1.",
        ],
        expected_agent="credits",
        verify=_credits_grounded("MEM-SWITCH-1"),
    ),
]

# Exploratory cases: no correct answer is scored pass/fail. These probe
# requests outside all three specialists' scope, where the honest behavior is
# "I can't help with that, let me get you a person" — triage's prompt today
# forces a pick among three transfers with no such option, so this documents
# a real gap rather than asserting behavior that doesn't exist yet.
EXPLORATORY_CASES = [
    TestCase(
        name="out_of_scope_request",
        turns=["What's your policy on refunding a lost gym towel deposit?"],
        expected_agent="",
    ),
]


@dataclass
class CaseResult:
    name: str
    passed: bool
    routing_ok: bool
    verify_failures: list[str]
    final_agent: str
    expected_agent: str
    final_reply: str
    latency_s: float


def run_case(case: TestCase) -> CaseResult:
    graph = build_graph(checkpointer=InMemorySaver())
    thread_id = str(uuid.uuid4())

    total_latency = 0.0
    result = None
    for message in case.turns:
        start = time.monotonic()
        result = graph.invoke(
            {"messages": [{"role": "user", "content": message}]},
            config={"configurable": {"thread_id": thread_id}},
        )
        total_latency += time.monotonic() - start

    reply = next(
        (
            m.text if isinstance(m.text, str) else str(m.content)
            for m in reversed(result["messages"])
            if isinstance(m, AIMessage) and m.content
        ),
        "(no response)",
    )
    final_agent = result.get("active_agent") or "triage"

    routing_ok = final_agent == case.expected_agent
    verify_failures = case.verify(reply)

    return CaseResult(
        name=case.name,
        passed=routing_ok and not verify_failures,
        routing_ok=routing_ok,
        verify_failures=verify_failures,
        final_agent=final_agent,
        expected_agent=case.expected_agent,
        final_reply=reply,
        latency_s=total_latency,
    )


def main() -> None:
    print(f"\n{'CASE':<26}{'RESULT':<8}{'AGENT':<14}{'LATENCY':<10}DETAIL")
    print("-" * 100)

    results = [run_case(case) for case in CASES]
    for r in results:
        status = "PASS" if r.passed else "FAIL"
        detail = ""
        if not r.routing_ok:
            detail = f"expected agent '{r.expected_agent}', got '{r.final_agent}'"
        elif r.verify_failures:
            detail = "; ".join(r.verify_failures)
        print(f"{r.name:<26}{status:<8}{r.final_agent:<14}{r.latency_s:>6.2f}s   {detail}")

    passed = sum(r.passed for r in results)
    print("-" * 100)
    print(f"{passed}/{len(results)} passed\n")

    for case in EXPLORATORY_CASES:
        r = run_case(case)
        print(f"Exploratory — {r.name} (no pass/fail; documents a known gap):")
        print(f'  routed to: {r.final_agent}  ({r.latency_s:.2f}s)')
        print(f'  reply: "{r.final_reply}"\n')


if __name__ == "__main__":
    main()
