# instr/ — learning-dynamics instrumentation for the full-batch run

Files
- `dynamics.py` — model-agnostic pieces on lists of tensors: MicrobatchStats (gradient noise from the
  accumulation loop), HVP (finite differences, "maxrel" step rule, optional Richardson), MaskedHVP /
  ScaledHVP (block and Adam-preconditioned Hessians), lanczos (full reorth, Ritz vectors), slq_density,
  line_probe, matrix_spectra(_distributed) (Gram method), step_norms, DynamicsLogger (jsonl + npz + async raw).
- `fb_hooks.py` — FBOracle (Oracle over fb/'s FullBatchTrainer, with fp8-state freezing) and
  DynamicsSuite (drop-in replacement for `t.train_step(step)` with cadenced probes), Cadence.
- `validate_hvp.py`, `validate_hvp2.py` — toy GPT (70M) with exact double-backward HVPs: FD accuracy
  vs eps / precision / direction, Lanczos convergence. Results: out/validate_hvp*.json.
- `test_fb_oracle2.py` — the same checks on the real WR model through FullBatchTrainer, plus a suite
  end-to-end run. Results: out/fb_oracle2 (1 GPU), out/fb_oracle3 (2 GPUs).

Use in a training script (run from fb/; call torch.cuda.set_device(LOCAL_RANK) before importing fb):

    sys.path.insert(0, "/workspace/shawnghu/nanogpt-fb/instr")
    from fb_hooks import DynamicsSuite, Cadence
    t = FullBatchTrainer(cfg)
    suite = DynamicsSuite(t, cfg.out_dir, Cadence(lanczos=10, slq=100, ...))
    for step in range(cfg.steps):
        rec = suite.train_step(step)      # dict of scalars; arrays go to <out_dir>/dyn/step_*.npz
    suite.close()                          # joins the async raw-artifact writers

Outputs: `<run>/dyn/scalars.jsonl` (one line per step), `<run>/dyn/step_XXXXXX.npz` (arrays: per-microbatch
grad norms/losses/alignment, per-tensor norms and cosines, singular values of W/G/u, line-probe losses,
Ritz values, Lanczos tridiagonals, SLQ nodes/weights, gradient/update mass in top eigvecs),
`<run>/raw/{grad,update}_XXXXXX.pt` (bf16, dense params) and `params_XXXXXX.pt` (fp32).

What the Hessian is: the Hessian of the fp8 TRAINING forward (the function the optimizer actually
differentiates) at frozen fp8 scales, over the dense parameters (n-gram table held fixed), on a fixed
probe subset of the training set. See NOTES.md [instr] for validation numbers and the cost table.
