"""Full-batch / large-batch training of the NanoGPT speedrun WR model (record #360) with a fixed objective.

Launch: torchrun --standalone --nproc_per_node=N train_fb.py key=value ...   (see fullbatch/config.py)
Writes <out_dir>/config.json, <out_dir>/metrics.jsonl (one JSON object per line), <out_dir>/ckpt/.

Preemption-safe: a rolling checkpoint every ckpt_minutes of wall time (and at a step boundary after
SIGTERM/SIGUSR1), atomic (written to ckpt/tmp_step_X, renamed to ckpt/step_X once every rank's file and
meta.json are there). On start the run resumes from the latest complete checkpoint, truncates its jsonl
outputs back to that step, and continues. A finished run leaves <out_dir>/DONE (or HALTED) and keeps only
the milestone checkpoints and the final one; a requeued job that finds the marker exits at once.
"""
import glob
import json
import os
import shutil
import signal
import sys
import time
_T_PROC = time.time()
_STARTUP = {}

os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
import torch
# The CE kernel (perf/kernels/cross_entropy.py) is compiled into the CURRENT device at import time.
torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", "0")))
import numpy as np
import torch._dynamo as dynamo
import torch.distributed as dist

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fullbatch.config import parse_config
from fullbatch.data import build_train_sample
from fullbatch.trainer import FullBatchTrainer

dynamo.config.recompile_limit = 64
INSTR_DIR = os.environ.get("NFB_INSTR_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "instr"))

STOP = {"flag": False}


def _on_signal(signum, frame):
    STOP["flag"] = True


signal.signal(signal.SIGTERM, _on_signal)
signal.signal(signal.SIGUSR1, _on_signal)


# ---------------------------------------------------------------------------------------- checkpoints
def list_ckpts(ckdir):
    out = []
    for d in glob.glob(os.path.join(ckdir, "step_*")):
        meta = os.path.join(d, "meta.json")
        if os.path.exists(meta):
            out.append((json.load(open(meta)), d))
    return sorted(out, key=lambda x: x[0]["step"])


def save_ckpt(t, ckdir, step, milestone, extra, suite_state, keep):
    tmp = os.path.join(ckdir, f"tmp_step_{step:06d}")
    final = os.path.join(ckdir, f"step_{step:06d}")
    if t.master:
        shutil.rmtree(tmp, ignore_errors=True)
        os.makedirs(tmp)
    dist.barrier()
    T0 = time.perf_counter()
    st = t.capture_state()
    st["extra"] = extra
    path = os.path.join(tmp, f"rank{t.rank}.pt")
    torch.save(st, path + ".part")
    os.replace(path + ".part", path)
    if t.master and suite_state is not None:
        torch.save(suite_state, os.path.join(tmp, "suite.pt"))
    del st
    dist.barrier()
    if t.master:
        with open(os.path.join(tmp, "meta.json"), "w") as f:
            json.dump(dict(step=step, milestone=bool(milestone), world=t.world, time=time.time()), f)
        if os.path.exists(final):
            shutil.rmtree(final)
        os.replace(tmp, final)
        rolling = [d for m, d in list_ckpts(ckdir) if not m["milestone"]]
        for d in rolling[:-keep] if keep > 0 else rolling:
            shutil.rmtree(d, ignore_errors=True)
        for d in glob.glob(os.path.join(ckdir, "tmp_step_*")):
            shutil.rmtree(d, ignore_errors=True)
    dist.barrier()
    return time.perf_counter() - T0



# ---------------------------------------------------------------------------------------- partial (mid-step) checkpoints
# A full-batch step is one pass over the whole dataset (minutes); low-prio allocations are often shorter. A
# partial checkpoint saves the partially accumulated gradient of the step in progress (fp32 accumulators,
# loss/token sums, the microbatch index, every model buffer, rng, and the per-microbatch statistics/token
# losses gathered so far), so a preempted step resumes at that microbatch instead of from its start.
class _StopMidStep(Exception):
    pass


def list_partials(ckdir):
    out = []
    for d in glob.glob(os.path.join(ckdir, "partial_*")):
        meta = os.path.join(d, "meta.json")
        if os.path.exists(meta):
            out.append((json.load(open(meta)), d))
    return sorted(out, key=lambda x: (x[0]["step"], x[0]["idx"]))


def save_partial(t, ckdir, step, idx, loss_sum, tokens, mb_state, tok_list):
    tmp = os.path.join(ckdir, "tmp_partial")
    final = os.path.join(ckdir, f"partial_step{step:06d}_mb{idx:06d}")
    if t.master:
        shutil.rmtree(tmp, ignore_errors=True)
        os.makedirs(tmp)
    dist.barrier()
    T0 = time.perf_counter()
    m = t.raw_model
    cpu = lambda x: x.detach().to("cpu", copy=True)
    st = dict(step=step, idx=idx, loss_sum=float(loss_sum.item()), tokens=int(tokens),
              acc=[cpu(v) for v in t.acc.values()],
              buffers={n: cpu(b) for n, b in m.named_buffers() if n.split(".")[-1] not in t._SKIP_BUFFERS},
              mlp_quantize_calls=m.mlp_quantize_calls,
              rng_cpu=torch.get_rng_state(), rng_cuda=torch.cuda.get_rng_state(t.device),
              mb_state=mb_state, tok=(torch.stack(tok_list) if tok_list else None), world=t.world, rank=t.rank)
    path = os.path.join(tmp, f"rank{t.rank}.pt")
    torch.save(st, path + ".part")
    os.replace(path + ".part", path)
    del st
    dist.barrier()
    if t.master:
        with open(os.path.join(tmp, "meta.json"), "w") as f:
            json.dump(dict(step=step, idx=idx, world=t.world, time=time.time()), f)
        os.replace(tmp, final)
        for _, d in list_partials(ckdir):
            if d != final:
                shutil.rmtree(d, ignore_errors=True)
    dist.barrier()
    return time.perf_counter() - T0


def mbstats_state(suite):
    if suite is None:
        return None
    ms = suite.mbstats
    stk = lambda xs: torch.stack(xs).cpu() if xs else None
    return dict(n_tot=ms.n_tot, sq_w=None if ms.sq_w is None else ms.sq_w.cpu(), mb_sq=stk(ms.mb_sq),
                mb_dot_prev=stk(ms.mb_dot_prev), mb_n=list(ms.mb_n), mb_loss=stk(ms.mb_loss))


def restore_mbstats(suite, st, device):
    if suite is None or st is None:
        return
    ms = suite.mbstats
    unstk = lambda x: [] if x is None else list(x.to(device).unbind(0))
    ms.n_tot = st["n_tot"]
    ms.sq_w = None if st["sq_w"] is None else st["sq_w"].to(device)
    ms.mb_sq, ms.mb_dot_prev, ms.mb_loss = unstk(st["mb_sq"]), unstk(st["mb_dot_prev"]), unstk(st["mb_loss"])
    ms.mb_n = list(st["mb_n"])


def truncate_jsonl(path, start_step):
    """Drop step records with step >= start_step (they will be recomputed); keep event lines."""
    if not os.path.exists(path):
        return
    keep = []
    for line in open(path):
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue  # a torn last line
        if "step" in r and isinstance(r["step"], int) and r["step"] >= start_step and "event" not in r:
            continue
        keep.append(line if line.endswith("\n") else line + "\n")
    with open(path + ".tmp", "w") as f:
        f.writelines(keep)
    os.replace(path + ".tmp", path)


# ---------------------------------------------------------------------------------------- halting rule
def load_ref(path):
    pts = {"train_eval": [], "val_loss": []}
    for line in open(path):
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue  # torn last line of a file still being written
        if "step" not in r or "event" in r:
            continue
        for k in pts:
            if k in r and r[k] == r[k]:
                pts[k].append((r["step"] + 1, r[k]))   # x = number of optimizer updates done
    return {k: sorted(v) for k, v in pts.items() if v}


def ref_at(pts, x):
    if not pts or x < pts[0][0] or x > pts[-1][0]:
        return None
    for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
        if x0 <= x <= x1:
            return y0 if x1 == x0 else y0 + (y1 - y0) * (x - x0) / (x1 - x0)
    return pts[-1][1] if x == pts[-1][0] else None


# ---------------------------------------------------------------------------------------- dynamics suite
def make_suite(t, cfg):
    sys.path.insert(0, INSTR_DIR)
    from fb_hooks import Cadence, DynamicsSuite
    if cfg.dyn_preset == "lean":
        cad = Cadence(val=0, norms=1, microbatch_stats=1, spectra=5, line_probe=5, curvature=10, lanczos=100,
                      group_lanczos_every=500, precond_lanczos=100, slq=1000, raw_grad=1, raw_update=1, checkpoint=0)
    elif cfg.dyn_preset == "rich":
        cad = Cadence(val=0, spectra=1, line_probe=1, curvature=2, lanczos=10, group_lanczos_every=50,
                      precond_lanczos=50, slq=100, raw_grad=1, raw_update=1, checkpoint=0)
    else:
        raise SystemExit(f"unknown dyn_preset {cfg.dyn_preset}")
    suite = DynamicsSuite(t, cfg.out_dir, cad)
    excl = {x for x in cfg.dyn_raw_exclude.split(",") if x}
    if excl:
        orig = suite.log.save_raw

        def save_raw(step, kind, names, vec, dtype=torch.bfloat16):
            if kind in ("grad", "update"):
                keep = [(n, v) for n, v in zip(names, vec) if n not in excl]
                names, vec = [n for n, _ in keep], [v for _, v in keep]
            return orig(step, kind, names, vec, dtype=dtype)
        suite.log.save_raw = save_raw
    return suite


def suite_state(suite):
    cpu = lambda v: None if v is None else [x.detach().to("cpu", copy=True) for x in v]
    return dict(prev_G=cpu(suite.prev_G), prev_u=cpu(suite.prev_u), mb_prev_full=cpu(suite.mbstats.prev_full))


def restore_suite(suite, st, device):
    dev = lambda v: None if v is None else [x.to(device) for x in v]
    suite.prev_G, suite.prev_u = dev(st["prev_G"]), dev(st["prev_u"])
    suite.mbstats.prev_full = dev(st["mb_prev_full"])


# ---------------------------------------------------------------------------------------- main
def main():
    cfg = parse_config(sys.argv[1:])
    rank = int(os.environ.get("RANK", 0))
    for marker in ("DONE", "HALTED"):
        if os.path.exists(os.path.join(cfg.out_dir, marker)):
            if rank == 0:
                print(f"{marker} marker present in {cfg.out_dir}; nothing to do", flush=True)
            return
    _STARTUP["imports"] = time.time() - _T_PROC
    t = FullBatchTrainer(cfg)
    _STARTUP["trainer_init"] = time.time() - _T_PROC
    ckdir = os.path.join(cfg.out_dir, "ckpt")
    metrics_path = os.path.join(cfg.out_dir, "metrics.jsonl")
    if t.master:
        os.makedirs(ckdir, exist_ok=True)
        with open(os.path.join(cfg.out_dir, "config.json"), "w") as f:
            f.write(cfg.to_json())

    # ---- resume
    start, extra, suite_st = 0, {}, None
    ck = list_ckpts(ckdir) if cfg.resume else []
    st_box = [ck[-1][0]["step"] if ck else 0]
    dist.broadcast_object_list(st_box, src=0)
    start = st_box[0]
    if start > 0:
        d = os.path.join(ckdir, f"step_{start:06d}")
        st = torch.load(os.path.join(d, f"rank{t.rank}.pt"), weights_only=False)
        t.restore_state(st)
        extra = st.get("extra", {})
        del st
        sp = os.path.join(d, "suite.pt")
        if cfg.dyn and os.path.exists(sp):
            suite_st = torch.load(sp, weights_only=False)
    _STARTUP["resume_load"] = time.time() - _T_PROC
    partial = None
    if cfg.mode == "full" and cfg.resume:
        parts = [(m_, d_) for m_, d_ in list_partials(ckdir) if m_["step"] == start and m_["world"] == t.world]
        pbox = [parts[-1][1] if parts else None]
        dist.broadcast_object_list(pbox, src=0)
        if pbox[0] is not None:
            partial = torch.load(os.path.join(pbox[0], f"rank{t.rank}.pt"), weights_only=False)
            assert partial["world"] == t.world and partial["rank"] == t.rank and partial["step"] == start
    if t.master:
        truncate_jsonl(metrics_path, start)
        truncate_jsonl(os.path.join(cfg.out_dir, "dyn", "scalars.jsonl"), start)
    dist.barrier()
    log = open(metrics_path, "a") if t.master else None

    def emit(rec):
        if t.master:
            log.write(json.dumps(rec) + "\n")
            log.flush()
            print(json.dumps(rec), flush=True)

    suite = make_suite(t, cfg) if cfg.dyn else None
    if suite is not None and suite_st is not None:
        restore_suite(suite, suite_st, t.device)
        del suite_st

    # ---- train-loss sample, token losses, halting reference
    sample = None
    if cfg.train_eval_tokens:
        sample_cpu, _ = build_train_sample(cfg, t.rank, t.world)
        sample = [(None, a.to(t.device), b.to(t.device), c.to(t.device)) for a, b, c in sample_cpu]
    _STARTUP["train_sample"] = time.time() - _T_PROC
    tok_buf = []
    if cfg.token_loss_every and cfg.mode == "full":
        os.makedirs(os.path.join(cfg.out_dir, "tokloss"), exist_ok=True)
        np.save(os.path.join(cfg.out_dir, "tokloss", f"global_ids_rank{t.rank}.npy"), np.asarray(t.data.global_ids))
    # The reference may still be running (its metrics file is synced in while we train): reload it at
    # every eval; matched steps beyond its last eval are simply not compared yet.
    ref = (load_ref(cfg.halt_ref) if os.path.exists(cfg.halt_ref) else {}) if cfg.halt_ref else None
    bad_since = extra.get("bad_since")

    emit(dict(event="resume" if start else "start", start_step=start, world=t.world, dataset_tokens=t.dataset_tokens,
              dataset_checksum=t.dataset_checksum, microbatches_per_rank=(len(t.data) if t.data is not None else None),
              steps=cfg.steps, torch=torch.__version__, slurm_job=os.environ.get("SLURM_JOB_ID"),
              restarts=os.environ.get("SLURM_RESTART_COUNT"), startup_s={k: round(v, 1) for k, v in _STARTUP.items()}, init_s=getattr(t, "init_times", None),
              host=os.uname().nodename, wall=time.time()))

    def evaluate(rec):
        T = time.perf_counter()
        rec.update(t.evaluate())
        if sample is not None:
            rec["train_eval"] = t.eval_loss_on(sample)
        rec["t_eval"] = time.perf_counter() - T

    if cfg.val_at_start and start == 0:
        r0 = dict(step=-1)
        evaluate(r0)
        emit(r0)

    t_start = time.perf_counter()
    last_ckpt = time.time()
    steps_this_proc = 0
    halted = False
    for step in range(start, cfg.steps):
        last = step == cfg.steps - 1
        want_tok = cfg.token_loss_every and cfg.mode == "full" and (step % cfg.token_loss_every == 0 or last)
        if want_tok:
            tok_buf.clear()
            t.mb_hook = lambda info: tok_buf.append(info["token_losses"].to(torch.float16).cpu())
        if partial is not None and partial["step"] == step:
            pst = partial

            def _on_restore(pst=pst):
                m_ = t.raw_model
                named_b = dict(m_.named_buffers())
                for n_, v_ in pst["buffers"].items():
                    named_b[n_].copy_(v_)
                m_.mlp_quantize_calls = pst["mlp_quantize_calls"]
                torch.set_rng_state(pst["rng_cpu"])
                torch.cuda.set_rng_state(pst["rng_cuda"], t.device)
                restore_mbstats(suite, pst["mb_state"], t.device)
                if want_tok and pst["tok"] is not None:
                    tok_buf.extend(pst["tok"].unbind(0))

            t.partial_resume = dict(step=step, idx=pst["idx"], loss_sum=pst["loss_sum"], tokens=pst["tokens"],
                                    acc=pst["acc"], on_restore=_on_restore)
            emit(dict(event="partial_resume", step=step, idx=pst["idx"]))
            partial = None
        if cfg.mode == "full" and (cfg.partial_minutes > 0 or cfg.partial_test_at > 0):
            pstate = dict(last=time.time())

            def _partial_hook(n_done, loss_sum, tokens, step=step, pstate=pstate):
                if n_done % cfg.partial_check_every or n_done >= len(t.data):
                    return
                test = cfg.partial_test_at > 0 and n_done == cfg.partial_test_at and steps_this_proc == 0
                due = cfg.partial_minutes > 0 and time.time() - pstate["last"] > cfg.partial_minutes * 60
                flags = torch.tensor([float(due and t.master), float(STOP["flag"]), float(test)], device=t.device)
                dist.all_reduce(flags, op=dist.ReduceOp.MAX)
                if not bool(flags.any()):
                    return
                secs = save_partial(t, ckdir, step, n_done, loss_sum, tokens,
                                    mbstats_state(suite), list(tok_buf) if want_tok else None)
                pstate["last"] = time.time()
                emit(dict(event="partial_checkpoint", step=step, idx=n_done, seconds=secs))
                if bool(flags[1]) or bool(flags[2]):
                    raise _StopMidStep()

            t.partial_hook = _partial_hook
        try:
            rec = suite.train_step(step) if suite is not None else t.train_step(step)
        except _StopMidStep:
            emit(dict(event="stopping", step_done=step, reason="mid-step"))
            if suite is not None:
                suite.close()
            dist.destroy_process_group()
            return
        finally:
            t.partial_hook = None
        t.mb_hook = None
        rec["step"] = step
        if want_tok:
            np.save(os.path.join(cfg.out_dir, "tokloss", f"step_{step:06d}_rank{t.rank}.npy"),
                    torch.stack(tok_buf).numpy())
            tok_buf.clear()
        rec["elapsed"] = time.perf_counter() - t_start
        due_eval = (cfg.train_eval_every and (step + 1) % cfg.train_eval_every == 0) or \
                   (cfg.val_every and (step + 1) % cfg.val_every == 0) or last
        if due_eval:
            evaluate(rec)
        # halting rule: worse than the fresh-data reference at matched steps, at every eval over halt_span steps
        if ref is not None and due_eval and step + 1 >= cfg.halt_min_step:
            if os.path.exists(cfg.halt_ref):
                ref = load_ref(cfg.halt_ref)
            x = step + 1
            worse = []
            for k in ("train_eval", "val_loss"):
                y = ref_at(ref.get(k), x)
                if y is not None and k in rec:
                    rec[f"ref_{k}"] = y
                    if rec[k] - y > max(cfg.halt_margin, cfg.halt_margin_rel * y):
                        worse.append(k)
            if worse:
                bad_since = bad_since if bad_since is not None else x
                rec["halt_bad"] = worse
                if x - bad_since >= cfg.halt_span:
                    halted = True
            elif any(f"ref_{k}" in rec for k in ("train_eval", "val_loss")):
                bad_since = None
        if step % cfg.log_every == 0 or last or due_eval:
            emit(rec)
        steps_this_proc += 1

        # ---- checkpoint decision (collective)
        flags = torch.tensor([float(time.time() - last_ckpt > cfg.ckpt_minutes * 60 and cfg.ckpt_minutes > 0)
                              if t.master else 0.0, float(STOP["flag"])], device=t.device)
        dist.all_reduce(flags, op=dist.ReduceOp.MAX)
        time_due, stop = bool(flags[0]), bool(flags[1])
        milestone = bool(cfg.ckpt_milestone_every and (step + 1) % cfg.ckpt_milestone_every == 0)
        test_stop = bool(cfg.stop_after_steps and steps_this_proc >= cfg.stop_after_steps and not last)
        if last or halted or time_due or stop or milestone or test_stop:
            sst = suite_state(suite) if (suite is not None and t.master) else None
            secs = save_ckpt(t, ckdir, step + 1, milestone or last or halted, dict(bad_since=bad_since), sst,
                             cfg.ckpt_keep)
            last_ckpt = time.time()
            if t.master:
                for m_, d_ in list_partials(ckdir):
                    if m_["step"] <= step:
                        shutil.rmtree(d_, ignore_errors=True)
            emit(dict(event="checkpoint", step_done=step + 1, seconds=secs, milestone=milestone or last or halted,
                      reason=[r for r, v in (("time", time_due), ("signal", stop), ("milestone", milestone),
                                             ("test_stop", test_stop), ("halt", halted), ("final", last)) if v]))
        if halted:
            emit(dict(event="halted", step_done=step + 1, bad_since=bad_since))
            break
        if stop or test_stop:
            emit(dict(event="stopping", step_done=step + 1, reason="signal" if stop else "stop_after_steps"))
            if suite is not None:
                suite.close()
            dist.destroy_process_group()
            return

    if suite is not None:
        suite.close()
    if t.master:
        # keep milestones and the final checkpoint, drop the rolling ones
        for m, d in list_ckpts(ckdir):
            if not m["milestone"]:
                shutil.rmtree(d, ignore_errors=True)
        with open(os.path.join(cfg.out_dir, "HALTED" if halted else "DONE"), "w") as f:
            f.write(json.dumps(dict(time=time.time())))
    emit(dict(event="end", halted=halted, peak_mem_gb=torch.cuda.max_memory_allocated() / 2**30,
              elapsed=time.perf_counter() - t_start))
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
