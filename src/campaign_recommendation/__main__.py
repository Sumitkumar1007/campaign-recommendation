from __future__ import annotations

import argparse
import json

from .pipelines.recommend import run_account_explanation, run_recommendations
from .pipelines.train import run_training
from .settings import load_config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Campaign recommendation CLI")
    parser.add_argument("--config", default=None, help="Optional path to config JSON")

    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Train the baseline recommendation model")
    train_parser.add_argument("--max-rows", type=int, default=None, help="Optional training row cap")

    recommend_parser = subparsers.add_parser("recommend", help="Generate campaign recommendations")
    recommend_parser.add_argument("--model-dir", default=None, help="Optional model artifact directory")
    recommend_parser.add_argument(
        "--input-csv",
        default=None,
        help="Optional base population CSV with apac_card_number, risk, collectable_amount, campaign_identifier, emi_date",
    )

    explain_parser = subparsers.add_parser("explain", help="Explain recommendations for a single account")
    explain_parser.add_argument("--account-id", required=True, help="Loan account identifier")
    explain_parser.add_argument("--model-dir", default=None, help="Optional model artifact directory")
    explain_parser.add_argument(
        "--input-csv",
        default=None,
        help="Optional base population CSV with current-state account inputs",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    config = load_config(args.config)

    if args.command == "train":
        summary = run_training(config=config, max_rows=args.max_rows)
        print(json.dumps(summary, indent=2))
        return

    if args.command == "recommend":
        output_path = run_recommendations(config=config, model_dir=args.model_dir, input_csv=args.input_csv)
        print(json.dumps({"recommendation_output_path": str(output_path)}, indent=2))
        return

    if args.command == "explain":
        payload = run_account_explanation(
            config=config,
            account_id=args.account_id,
            model_dir=args.model_dir,
            input_csv=args.input_csv,
        )
        print(json.dumps(payload, indent=2))
        return

    raise ValueError(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    main()
