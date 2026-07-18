from __future__ import annotations

import argparse
import json
import sys

from .validate import QuestionPackError, distribution, load_candidates, validate_publishable_pack


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate Evater question-bank packs.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate_parser = subparsers.add_parser("validate", help="Validate a question JSON file.")
    validate_parser.add_argument("path")
    validate_parser.add_argument("--publish", action="store_true", help="Apply the strict 100-question publish gate.")

    args = parser.parse_args()
    try:
        candidates = load_candidates(args.path)
        report = validate_publishable_pack(candidates) if args.publish else {
            "count": len(candidates),
            "distribution": distribution(candidates),
            "publishable": False,
        }
        print(json.dumps(report, indent=2))
        return 0
    except QuestionPackError as exc:
        print(str(exc), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
