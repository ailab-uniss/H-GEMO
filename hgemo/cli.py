"""Command-line interface.

The only entry-point used in the paper is::

    python -m hgemo.cli run --config <yaml> [--fold-idx K]

which runs a single fold of a single configuration file.  The launch
scripts iterate over datasets and folds externally.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from .config import load_config, set_dotted
from .experiment import run_experiment_from_config


def _cmd_run(args: argparse.Namespace) -> int:
    config = load_config(Path(args.config))

    # Apply --override key=value pairs first
    for kv in (args.override or []):
        if "=" not in kv:
            raise SystemExit(f"--override expects key=value, got: {kv!r}")
        key, val = kv.split("=", 1)
        set_dotted(config, key, val)

    if getattr(args, "seed", None) is not None:
        config["seed"] = int(args.seed)
    if getattr(args, "out_dir", None) is not None:
        logging_cfg = config.get("logging")
        if not isinstance(logging_cfg, dict):
            logging_cfg = {}
            config["logging"] = logging_cfg
        logging_cfg["out_dir"] = str(args.out_dir)

    fold_idx = getattr(args, "fold_idx", None)
    out_dir = run_experiment_from_config(config, fold_idx=fold_idx)
    print(str(out_dir))
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="hgemo")
    sub = p.add_subparsers(dest="cmd", required=True)

    pr = sub.add_parser("run", help="Run one experiment config")
    pr.add_argument("--config", required=True, help="Path to YAML config")
    pr.add_argument("--seed", type=int, default=None,
                    help="Override random seed in the config")
    pr.add_argument("--out-dir", default=None,
                    help="Override logging.out_dir in the config")
    pr.add_argument("--fold-idx", type=int, default=None,
                    help="CV fold index (0-based)")
    pr.add_argument("--override", nargs="*", metavar="KEY=VAL",
                    help="Dotted-key overrides (e.g. dataset.name=Emotions)")
    pr.set_defaults(func=_cmd_run)

    args = p.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
