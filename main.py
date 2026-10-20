"""Demo5 commands. No work runs on import."""
import argparse
import os
import sys


DESIGNS = ("lodo", "both_0911", "both_0917", "loso")  # must match src.splits.DESIGNS (checked in tests)


def check(_args):
    import numpy
    import torch
    print(f"python {sys.version.split()[0]}")
    print(f"numpy {numpy.__version__}")
    print(f"torch {torch.__version__} cuda {torch.version.cuda}")
    print(f"cuda_available {torch.cuda.is_available()}")
    print(f"CUDA_VISIBLE_DEVICES {os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>')!r}")
    for i in range(torch.cuda.device_count()):
        free, total = torch.cuda.mem_get_info(i)
        print(f"cuda:{i} {torch.cuda.get_device_name(i)} free {free / 2**30:.1f} of {total / 2**30:.1f} GiB")


def preprocess(args):
    from config import Config
    from src.cache import build_cache
    build_cache(Config(), workers=args.workers)


def train(args):
    from config import Config
    from src.train import run
    overrides = {k: v for k, v in (("encoder_lr", args.encoder_lr), ("head_lr", args.head_lr), ("patience", args.patience)) if v is not None}
    for n, seed in enumerate(args.seed):
        # Re-fingerprinting the raw sessions reads about 1 GB, so do it once per command, before the first run.
        run(Config(), args.candidate, args.val_date, seed, args.run, args.device, args.epochs, args.subset, args.cache,
            verify_sources=(n == 0), overrides=overrides, design=args.design)


def study(args):
    from config import Config
    from src.study import launch
    extra = [x for flag, v in (("--epochs", args.epochs), ("--encoder-lr", args.encoder_lr), ("--head-lr", args.head_lr), ("--patience", args.patience))
             if v is not None for x in (flag, str(v))]
    failed = launch(Config(), args.run, args.gpus, args.per_gpu, args.cache, candidates=args.candidates, extra=extra, design=args.design)
    sys.exit(1 if failed else 0)


def report(args):
    from config import Config
    from src.report import build_report
    print(build_report(Config(), args.runs, args.name))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("check", help="print Python, torch, numpy and GPU visibility").set_defaults(fn=check)
    p = sub.add_parser("preprocess", help="build the window cache (refuses to overwrite)")
    p.add_argument("--workers", type=int, default=8)
    p.set_defaults(fn=preprocess)
    p = sub.add_parser("train", help="fit one candidate on two pool dates, validate on the third, score all splits")
    p.add_argument("--candidate", required=True, choices=("b0", "b1", "c1", "c2"))
    p.add_argument("--val-date", "--fold", dest="val_date", required=True, help="validation date; for --design loso, the test session fold")
    p.add_argument("--design", choices=DESIGNS, default="lodo", help="split design (lodo = the study; others are thesis analyses C and D)")
    p.add_argument("--seed", type=int, nargs="+", default=[0])
    p.add_argument("--run", required=True, help="run id; an existing run directory is never overwritten")
    p.add_argument("--device", required=True, help="cpu or cuda:N (explicit)")
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--subset", type=float, default=None, help="fraction of training windows (smoke runs)")
    p.add_argument("--cache", default=None, help="cache id (default: the only cache)")
    p.add_argument("--encoder-lr", type=float, default=None, help="experiment override of the encoder learning rate")
    p.add_argument("--head-lr", type=float, default=None, help="experiment override of the head learning rate")
    p.add_argument("--patience", type=int, default=None, help="experiment override of the early-stopping patience")
    p.set_defaults(fn=train)
    p = sub.add_parser("study", help="run every candidate, validation date and seed, several jobs at once per GPU")
    p.add_argument("--run", required=True)
    p.add_argument("--gpus", nargs="+", required=True, help="explicit devices, for example cuda:0 cuda:1")
    p.add_argument("--per-gpu", type=int, default=3, help="jobs at once on each GPU (refused if they cannot fit)")
    p.add_argument("--cache", default=None, help="cache id (default: the only cache)")
    p.add_argument("--candidates", nargs="+", choices=("b0", "b1", "c1", "c2"), default=None, help="only these candidates (experiments)")
    p.add_argument("--design", choices=DESIGNS, default="lodo", help="split design (lodo = the study; others are thesis analyses C and D)")
    p.add_argument("--epochs", type=int, default=None, help="experiment override, forwarded to every job")
    p.add_argument("--encoder-lr", type=float, default=None)
    p.add_argument("--head-lr", type=float, default=None)
    p.add_argument("--patience", type=int, default=None)
    p.set_defaults(fn=study)
    p = sub.add_parser("report", help="write tables, ROC figure and REPORT.md from finished runs")
    p.add_argument("--runs", nargs="+", required=True, help="run ids; a later id replaces an earlier one for the same job")
    p.add_argument("--name", required=True, help="folder name under results/")
    p.set_defaults(fn=report)
    args = parser.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main()
