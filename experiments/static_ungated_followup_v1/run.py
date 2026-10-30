#!/usr/bin/env python3
"""Run the separately recorded ungated counterpart of Safety-first RouteGate."""
import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
from routegate import staticfollowup as U
from routegate import staticassessment as A
from routegate.validation import ValidationError


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stage",required=True,choices=["preflight","init","generate","status","prepare","build","dryrun","judge","join","export","import","report","progress"])
    p.add_argument("--data-root",type=Path,default=Path.home()/"routegate-data")
    p.add_argument("--limit",type=int,help="Generate at most this many new responses, then stop; the same command without this option resumes.")
    p.add_argument("--judge",choices=["sonnet","opus"])
    p.add_argument("--dry-run",action="store_true")
    p.add_argument("--retry-failures",action="store_true")
    p.add_argument("--completed",type=Path)
    p.add_argument("--provenance",type=Path)
    p.add_argument("--check-model",action="store_true",help="Also verify the locally cached model/tokenizer snapshot without loading weights.")
    args=p.parse_args()
    try:
        if args.stage in ("init","preflight"):
            with U.lock(U.root_for(args.data_root),write=args.stage=="init" and not args.dry_run):
                result=U.prepare(args.data_root,write=args.stage=="init" and not args.dry_run)
            if args.check_model:
                from routegate.runtime import resolve_snapshot,GenerationConfig
                resolve_snapshot(GenerationConfig(max_new_tokens=768))
                result["model_snapshot_checked"]=True
        elif args.stage=="generate":
            U.require(not args.dry_run,"Use preflight for a generation dry run.")
            result=U.generate(args.data_root,limit=args.limit)
        elif args.stage=="status":result=U.status(args.data_root)
        else:
            # Separate judge locks permit Sonnet/Opus concurrently; structural stages
            # need both locks, so a join/import cannot race an active judge.
            import contextlib
            write=not args.dry_run and args.stage not in ("dryrun","progress")
            with contextlib.ExitStack() as stack:
                if write:
                    stack.enter_context(U.lock(U.root_for(args.data_root),"assessment",shared=args.stage=="judge"))
                    if args.stage=="judge":
                        U.require(args.judge is not None,"Choose sonnet or opus.")
                        stack.enter_context(U.lock(U.root_for(args.data_root),"judge_"+args.judge))
                    else:
                        for j in ("sonnet","opus"):
                            stack.enter_context(U.lock(U.root_for(args.data_root),"judge_"+j))
                result=A.run_stage(args.data_root,args.stage,judge=args.judge,dry_run=args.dry_run,
                    retry_failures=args.retry_failures,completed=args.completed,provenance=args.provenance)
        print(json.dumps(result,indent=2,sort_keys=True))
        if args.stage=="judge" and result.get("failed",0):
            print("Failed attempts were preserved. Inspect them before --retry-failures.")
            return 2
        return 0
    except (ValidationError,OSError,ValueError,KeyError) as exc:
        print("STOP: "+str(exc),file=sys.stderr)
        return 1


if __name__=="__main__":
    raise SystemExit(main())
