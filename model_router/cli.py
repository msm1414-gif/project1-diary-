"""Command-line interface for the model router.

Usage:
    model-router "Summarize this 2-page memo into 3 bullet points"
    echo "Refactor a 50k-line monorepo to async" | model-router -
    model-router --json "Classify support tickets by urgency"
"""

from __future__ import annotations

import argparse
import json
import sys

import anthropic

from model_router.router import CANDIDATES, recommend_model


def _read_task(arg: str | None) -> str:
    """Resolve the task text from the CLI arg, stdin, or a prompt."""
    if arg == "-":
        return sys.stdin.read()
    if arg:
        return arg
    if not sys.stdin.isatty():
        return sys.stdin.read()
    # Interactive fallback.
    print("Describe the task (Ctrl-D to finish):", file=sys.stderr)
    return sys.stdin.read()


def _format_human(rec) -> str:
    bar = "=" * 56
    lines = [
        bar,
        f"  Recommended : {rec.recommended_model}  ({rec.model_id})",
    ]
    if rec.runner_up:
        lines.append(
            f"  Runner-up   : {rec.runner_up}  ({CANDIDATES[rec.runner_up]})"
        )
    lines += [
        f"  Complexity  : {rec.complexity}",
        f"  Confidence  : {rec.confidence:.0%}",
        bar,
        "",
        "Reasoning:",
        f"  {rec.reasoning}",
    ]
    if rec.considerations:
        lines.append("")
        lines.append("Considerations:")
        lines += [f"  - {c}" for c in rec.considerations]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="model-router",
        description="Decide whether to use Sonnet, Opus, or Fable for a task.",
    )
    parser.add_argument(
        "task",
        nargs="?",
        help="Task description. Use '-' to read from stdin. "
        "If omitted, reads from stdin.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output the raw recommendation as JSON.",
    )
    args = parser.parse_args(argv)

    task = _read_task(args.task)
    if not task.strip():
        parser.error("no task provided")

    try:
        rec = recommend_model(task)
    except anthropic.AuthenticationError:
        print(
            "Error: invalid or missing API key. Set ANTHROPIC_API_KEY.",
            file=sys.stderr,
        )
        return 2
    except TypeError as exc:
        # The SDK raises a TypeError at request time when no auth method can
        # be resolved (no ANTHROPIC_API_KEY, auth token, or profile).
        if "authentication" in str(exc).lower():
            print(
                "Error: no credentials found. Set ANTHROPIC_API_KEY.",
                file=sys.stderr,
            )
            return 2
        raise
    except anthropic.APIError as exc:
        print(f"Error: API request failed: {exc}", file=sys.stderr)
        return 1
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    if args.json:
        payload = rec.model_dump()
        payload["model_id"] = rec.model_id
        print(json.dumps(payload, indent=2, ensure_ascii=False))
    else:
        print(_format_human(rec))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
