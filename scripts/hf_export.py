"""Build the Hugging Face export tree for shawnghu/nanogpt-fullbatch from the run directories on charmander.

Usage: python hf_export.py <part> [<part> ...]   parts: small, dataset, raw, tokloss, ckpt, all
Idempotent: skips outputs that already exist. Writes into /workspace/shawnghu/nanogpt-fb/hf_export.
"""
import glob
import json
import os
import shutil
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import save_file

ROOT = Path("/workspace/shawnghu/nanogpt-fb")
RUNS = ROOT / "fb/runs"
OUT = ROOT / "hf_export"
FB = RUNS / "full800/main"
NUM_MICROBATCHES = 16272
T = 49152


def cp(src, dst):
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    if not dst.exists():
        shutil.copy2(src, dst)


def run_files(src: Path, dst: Path):
    for name in ("metrics.jsonl", "config.json"):
        if (src / name).exists():
            cp(src / name, dst / name)


# ------------------------------------------------------------------ small files
def small():
    run_files(FB, OUT / "full_batch")
    cp(FB / "dyn/scalars.jsonl", OUT / "full_batch/dynamics/scalars.jsonl")
    for f in sorted(glob.glob(str(FB / "dyn/step_*.npz"))):
        cp(f, OUT / "full_batch/dynamics/arrays" / os.path.basename(f))
    run_files(RUNS / "ladder800/control_B393K", OUT / "small_batch_control")
    ladder = {  # (fixed-set run, fresh-data run) by batch size
        "batch_4.7M": ("ladder800_pod/fixed_B4.7M_L1.2", "ladder800_pod/stream_B4.7M_L1.2"),
        "batch_4.7M_lr1.73": ("ladder800_pod/fixed_B4.7M_L1.73", None),
        "batch_19M": ("ladder800_pod/fixed_B19M_L1.2", "ladder800_pod/stream_B19M_L1.2"),
        "batch_50M": ("ladder800_pod/fixed_B50M_L1.2", "ladder800_pod/stream_B50M_L1.2"),
        "batch_100M": ("ladder800_pod/fixed_B100M_L1.2", None),
        "batch_200M": ("ladder800/fixed_B200M_L1.2", None),
    }
    for name, (fixed, fresh) in ladder.items():
        run_files(RUNS / fixed, OUT / "batch_ladder" / name / "fixed_800M_set")
        if fresh:
            run_files(RUNS / fresh, OUT / "batch_ladder" / name / "fresh_data")
    for d in sorted((RUNS / "sweep").iterdir()):
        run_files(d, OUT / "large_batch_sweep_fresh_data" / d.name)
    for d in sorted((RUNS / "ref_fresh").glob("S*")) if (RUNS / "ref_fresh").exists() else []:
        run_files(d, OUT / "fresh_data_convergence_reference" / f"steps_{d.name[1:]}")
    for f in sorted(glob.glob(str(ROOT / "abl/runs_pod/*.out"))) + [str(ROOT / "abl/runs_pod/summary.json")]:
        cp(f, OUT / "record_ablation/logs" / os.path.basename(f))
    for f in sorted(glob.glob(str(RUNS / "verify*/*.json"))):
        cp(f, OUT / "verification" / Path(f).parent.name / os.path.basename(f))


# ------------------------------------------------------------------ dataset
def dataset():
    caches = sorted(glob.glob(str(ROOT / "cache/packed/*/key.json")))
    src = None
    for k in caches:
        key = json.loads(open(k).read())
        if key["M"] == NUM_MICROBATCHES and key["T"] == T:
            src = Path(k).parent
    assert src is not None, "packed cache for the 800M set not found"
    cp(src / "buf.npy", OUT / "dataset/train_tokens.npy")
    cp(src / "cu.npy", OUT / "dataset/attention_segment_ends.npy")
    buf = np.load(src / "buf.npy", mmap_mode="r")
    gids = np.arange(NUM_MICROBATCHES, dtype=np.int64)
    checksum = 0
    for a in range(0, NUM_MICROBATCHES, 1024):
        rs = buf[a:a + 1024, :T].astype(np.int64).sum(axis=1)
        checksum += int((rs * (gids[a:a + 1024] + 1)).sum())
    info = dict(
        description="The fixed 800M-token training set: the first 16,272 packed microbatches of "
                    "kjj0/fineweb10B-gpt2 (GPT-2 tokens), exactly as used by every fixed-set run.",
        source_dataset="https://huggingface.co/datasets/kjj0/fineweb10B-gpt2 (fineweb_train_*.bin in sorted order)",
        num_microbatches=NUM_MICROBATCHES, tokens_per_microbatch=T, total_tokens=NUM_MICROBATCHES * T,
        train_tokens_layout="uint16 array [16272, 49153]; microbatch k has inputs = row[k, :49152], targets = row[k, 1:]",
        attention_segment_ends_layout="int32 array [16272, 384]; cumulative attention-segment end offsets "
                                      "within the microbatch, padded with 49152",
        packing_rule="every microbatch starts at a BOS token (50256); documents are cut at 3072 tokens; "
                     "the last document is cut to fill the microbatch and its remainder is skipped; "
                     "attention segments are additionally cut every 2560 tokens",
        checksum_definition="sum_k (k + 1) * sum(inputs of microbatch k), exact integer",
        checksum=checksum,
        validation="first 10,485,760 tokens of fineweb_val_000000.bin from the same source dataset",
    )
    (OUT / "dataset/dataset_info.json").write_text(json.dumps(info, indent=2))


# ------------------------------------------------------------------ per-step gradients / updates
def _convert_raw(args):
    src, dst = args
    if Path(dst).exists():
        return dst
    d = torch.load(src, map_location="cpu", weights_only=False)
    tensors = {k: v.contiguous() for k, v in d.items()}
    tmp = dst + ".tmp"
    save_file(tensors, tmp, metadata={"note": "see README: full_batch/gradients|updates"})
    os.replace(tmp, dst)
    return dst


def raw():
    jobs = []
    for kind, outname in (("grad", "gradients"), ("update", "updates")):
        (OUT / "full_batch" / outname).mkdir(parents=True, exist_ok=True)
        for f in sorted(glob.glob(str(FB / f"raw/{kind}_*.pt"))):
            step = int(Path(f).stem.split("_")[1])
            jobs.append((f, str(OUT / "full_batch" / outname / f"step_{step:06d}.safetensors")))
    with ProcessPoolExecutor(16) as ex:
        for i, _ in enumerate(ex.map(_convert_raw, jobs)):
            if i % 50 == 0:
                print("raw", i, len(jobs), flush=True)


# ------------------------------------------------------------------ per-token losses
def tokloss():
    gid = {r: np.load(FB / f"tokloss/global_ids_rank{r}.npy") for r in range(4)}
    n = len(gid[0])
    steps = sorted({int(Path(f).stem.split("_")[1]) for f in glob.glob(str(FB / "tokloss/step_*_rank0.npy"))})
    (OUT / "full_batch/token_losses").mkdir(parents=True, exist_ok=True)
    for s in steps:
        dst = OUT / "full_batch/token_losses" / f"step_{s:06d}.npy"
        if dst.exists():
            continue
        full = np.lib.format.open_memmap(str(dst) + ".tmp.npy", mode="w+", dtype=np.float16, shape=(NUM_MICROBATCHES, T))
        for r in range(4):
            a = np.load(FB / f"tokloss/step_{s:06d}_rank{r}.npy", mmap_mode="r")
            assert a.shape[0] >= n and a.shape[1] == T, (s, r, a.shape)
            full[gid[r]] = a[:n]   # rows beyond n (steps 0-40 only) are instrumentation probe passes: dropped
        full.flush()
        del full
        os.replace(str(dst) + ".tmp.npy", dst)
        print("tokloss", s, flush=True)


# ------------------------------------------------------------------ checkpoints
def ckpt():
    for d in sorted(glob.glob(str(FB / "ckpt/step_*"))):
        step = int(Path(d).name.split("_")[1])
        out = OUT / "full_batch/checkpoints" / f"step_{step:06d}"
        if (out / "DONE").exists():
            continue
        out.mkdir(parents=True, exist_ok=True)
        for r in range(4):
            st = torch.load(f"{d}/rank{r}.pt", map_location="cpu", weights_only=False)
            if r == 0:
                save_file({k: v.contiguous() for k, v in st["params"].items()}, str(out / "model.safetensors"),
                          metadata={"step": str(step)})
                save_file({k: v.contiguous() for k, v in st["buffers"].items()}, str(out / "resume_state/buffers.safetensors"))
            opt_t, opt_meta = {}, {}
            for label, sd in st["optimizer"].items():
                for k, v in sd.items():
                    if torch.is_tensor(v):
                        opt_t[f"{label}/{k}"] = v.contiguous()
                    else:
                        opt_meta[f"{label}/{k}"] = v
            (out / "resume_state").mkdir(exist_ok=True)
            save_file(opt_t, str(out / f"resume_state/optimizer_rank{r}.safetensors"))
            meta = dict(non_tensor_optimizer_state=opt_meta, split_embed=st["split_embed"], step_count=st["step_count"],
                        mlp_quantize_calls=st["mlp_quantize_calls"], world=st["world"], rank=st["rank"])
            (out / f"resume_state/meta_rank{r}.json").write_text(json.dumps(meta, indent=1, default=str))
            save_file({"rng_cpu": st["rng_cpu"], "rng_cuda": st["rng_cuda"]}, str(out / f"resume_state/rng_rank{r}.safetensors"))
            del st
        (out / "DONE").write_text("ok")
        print("ckpt", step, flush=True)


if __name__ == "__main__":
    parts = sys.argv[1:] or ["all"]
    if "all" in parts:
        parts = ["small", "dataset", "tokloss", "ckpt", "raw"]
    for p in parts:
        globals()[p]()
    print("done", parts)
