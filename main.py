import argparse
from pathlib import Path

from src.bind_score import bind_score_with_paths
from src.data_loader import load_data
from src.evaluate import run_evaluation
from src.features import HORIZONS, build_feature_frame
from src.paths import DATA_DIR, MODELS_DIR, OUTPUT_DIR
from src.train import temporal_split, train_models


def cmd_train(args: argparse.Namespace) -> None:
    report = train_models(args.data_dir, args.models_dir)
    print(f"Training complete. Models saved to {args.models_dir}")
    for t, info in report["horizons"].items():
        test_metrics = info["test_metrics"]
        val_metrics = info["val_metrics"]
        print(
            f"  t={t} best={info['best_model']} "
            f"val P@20={val_metrics['precision_at_top_20']:.3f} "
            f"test ROC-AUC={test_metrics['roc_auc']:.3f} "
            f"test PR-AUC={test_metrics['pr_auc']:.3f} "
            f"test P@20={test_metrics['precision_at_top_20']:.3f}"
        )


def cmd_evaluate(args: argparse.Namespace) -> None:
    submissions, events = load_data(args.data_dir)
    feature_frame = build_feature_frame(submissions, events)
    _, test_ids = temporal_split(submissions)

    report = run_evaluation(feature_frame, test_ids, args.models_dir, args.output_dir)
    print(f"Evaluation report saved to {args.output_dir / 'evaluation_report.json'}")
    for row in report["model_comparison"]:
        print(
            f"  t={row['t']} {row['model']} "
            f"ROC-AUC={row['roc_auc']:.3f} PR-AUC={row['pr_auc']:.3f} "
            f"P@20={row['precision_at_top_20']:.3f}"
        )


def cmd_score(args: argparse.Namespace) -> None:
    score = bind_score_with_paths(
        submission_id=args.submission_id,
        t=args.t,
        models_dir=args.models_dir,
        data_dir=args.data_dir,
    )
    print(f"bind_score(submission_id={args.submission_id}, t={args.t}) = {score:.4f}")


def cmd_run(args: argparse.Namespace) -> None:
    """Train, evaluate, and print a sample score (full pipeline)."""
    cmd_train(args)
    cmd_evaluate(args)
    submissions, _ = load_data(args.data_dir)
    sample_id = int(submissions.iloc[0]["submissionId"])
    score = bind_score_with_paths(
        sample_id, 7, models_dir=args.models_dir, data_dir=args.data_dir
    )
    print(f"Sample: bind_score({sample_id}, 7) = {score:.4f}")


def _add_path_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DATA_DIR,
        help=f"Data directory (default: {DATA_DIR})",
    )
    parser.add_argument(
        "--models-dir",
        type=Path,
        default=MODELS_DIR,
        help=f"Models directory (default: {MODELS_DIR})",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Bind score engine")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Train models for t in {0,7,30}")
    _add_path_args(train_parser)
    train_parser.set_defaults(func=cmd_train)

    eval_parser = subparsers.add_parser("evaluate", help="Evaluate models and feature importance")
    _add_path_args(eval_parser)
    eval_parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help=f"Output directory (default: {OUTPUT_DIR})",
    )
    eval_parser.set_defaults(func=cmd_evaluate)

    score_parser = subparsers.add_parser("score", help="Score a single submission at horizon t")
    score_parser.add_argument("--submission-id", type=int, required=True)
    score_parser.add_argument("--t", type=int, required=True, choices=HORIZONS)
    _add_path_args(score_parser)
    score_parser.set_defaults(func=cmd_score)

    run_parser = subparsers.add_parser(
        "run",
        help="Full pipeline: train, evaluate, and print a sample bind_score",
    )
    _add_path_args(run_parser)
    run_parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help=f"Output directory (default: {OUTPUT_DIR})",
    )
    run_parser.set_defaults(func=cmd_run)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
