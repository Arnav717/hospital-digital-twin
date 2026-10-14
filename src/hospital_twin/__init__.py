"""AI-Powered Hospital Digital Twin for predictive operations and decision intelligence."""

from __future__ import annotations

import argparse
import subprocess
import sys

from .config import ROOT

__version__ = "1.0.0"


def main() -> None:
    ap = argparse.ArgumentParser(prog="hospital-twin", description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build", help="generate data, train forecasters, optimise, recommend")
    b.add_argument("--days", type=int, default=730, help="days of history to synthesise")
    b.add_argument("--reps", type=int, default=24, help="Monte-Carlo replications")
    e = sub.add_parser("evaluate", help="write reports/evaluation.md")
    e.add_argument("--reps", type=int, default=24)
    a = sub.add_parser("app", help="launch the Streamlit command centre")
    a.add_argument("--port", type=int, default=8501)
    p = sub.add_parser("api", help="launch the REST API")
    p.add_argument("--port", type=int, default=8000)
    args = ap.parse_args()

    if args.cmd == "build":
        from .pipeline import build

        build(days=args.days, reps=args.reps)
    elif args.cmd == "evaluate":
        from .evaluation import evaluate

        evaluate(reps=args.reps)
    elif args.cmd == "app":
        sys.exit(subprocess.call([sys.executable, "-m", "streamlit", "run", str(ROOT / "app" / "Home.py"),
                                  "--server.port", str(args.port)]))
    elif args.cmd == "api":
        import uvicorn

        uvicorn.run("hospital_twin.api.main:app", host="0.0.0.0", port=args.port)
