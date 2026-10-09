# nanogpt-fullbatch

Exact full-batch training of the NanoGPT speedrun world-record model (modded-nanogpt record #360, "ANVIL2"), with instrumentation for learning dynamics.

This is a fork of [KellerJordan/modded-nanogpt](https://github.com/KellerJordan/modded-nanogpt) at commit `4ea6b93`. The upstream README is kept as `README_modded_nanogpt.md`, and every upstream file is unchanged. The data from all runs (metrics, dynamics statistics, full gradients and updates every step, per-token losses, checkpoints, and the exact training set) is at [huggingface.co/datasets/shawnghu/nanogpt-fullbatch](https://huggingface.co/datasets/shawnghu/nanogpt-fullbatch).

## Headline result

On a fixed 800M-token FineWeb set that ordinary small-batch training (393K tokens per step) takes to 3.248 validation loss, exact full-batch training (every step's gradient is over all 800M tokens) reaches a lower training loss but plateaus at about 3.50 validation loss. See the dataset card for details.

## What's here

- `train_fb.py` and `fullbatch/`: the trainer. It reuses the record's model, kernels and ANVIL/Adam optimizer, but:
  - accumulates gradients over arbitrarily many microbatches (fp32 accumulators), so a step can be an exact gradient over a fixed token set (`mode=full`) or a large minibatch (`mode=minibatch`, from a fresh token stream or from the fixed set);
  - uses a fixed objective: the record's final-stage settings for the whole run, the full softmax, tied embeddings, and no hashed n-gram table;
  - checkpoints at step boundaries and mid-step (partial accumulators), and resumes automatically, which makes it usable on preemptible compute.
  All options are in `fullbatch/config.py` and are set as `key=value` on the command line.
- `instr/`: the learning-dynamics instrumentation (`DynamicsSuite`): per-microbatch gradient statistics, norms, cosines, spectra, line probes along the update, finite-difference Hessian-vector products, Lanczos, and raw artifact saving. See `instr/README.md`.
- `verify_fb.py`: correctness checks (accumulation versus one joint backward pass, determinism, microbatch-order invariance, world-size invariance).
- `record_ablation/`: the environment-variable switches (as a patch against upstream) used to ablate the record toward a full-batch-compatible configuration, plus the job scripts.
- `scripts/`: the analysis helpers and the Hugging Face export script. `scripts/cluster/` has the Slurm launchers as they were used; they contain cluster-specific paths.

## Running

The environment is the record's: `torch==2.10` from the cu128 index, on a machine with a CUDA 13 runtime (the patched FlashAttention-3 kernel needs `libcudart.so.13`), plus `pip install -r requirements.txt safetensors`. Then, for example, the main run:

```
torchrun --standalone --nproc_per_node=4 train_fb.py mode=full dataset_tokens=800000000 steps=350 lr_mul=1.2 \
  train_eval_tokens=10027008 train_eval_every=5 dyn=True dyn_preset=lean token_loss_every=10 \
  ckpt_minutes=1.5 partial_minutes=2 out_dir=runs/full800/main
```

Cache locations can be set with `NFB_CACHE_DIR` (packed dataset and canonical-token mask), `NFB_LOCAL_CACHE_DIR` (per-node copy) and `NFB_INSTR_DIR`. In the runs these were hard-coded cluster paths; making them configurable is the only change from the code that ran.
