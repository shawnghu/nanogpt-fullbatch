"""Configuration of a full-batch (or large-batch) run. Every field can be set from the command line
as key=value (python train_fb.py steps=500 mode=full dataset_tokens=50331648 ...)."""
import dataclasses
import json
from dataclasses import dataclass, field


@dataclass
class FBConfig:
    # ---------------- data ----------------
    train_glob: str = "data/fineweb10B/fineweb_train_*.bin"
    val_glob: str = "data/fineweb10B/fineweb_val_*.bin"
    vocab_size: int = 50257          # tokenizer vocab (the model pads to a multiple of 128)
    bos_id: int = 50256
    tokenizer: str = "gpt2"          # only used for the canonical eval mask (gpt2 only)
    # "full": every step is the exact gradient over the fixed set of dataset_tokens tokens.
    # "minibatch": every step uses batch_tokens tokens, from the stream (fresh data) or the fixed set.
    mode: str = "full"
    dataset_tokens: int = 50_331_648     # fixed-set size; rounded down to a multiple of micro_tokens * world
    dataset_skip_microbatches: int = 0   # start the fixed set this many microbatches into the stream
    batch_tokens: int = 0                # minibatch mode: tokens per optimizer step (multiple of micro_tokens*world)
    minibatch_source: str = "stream"     # stream | fixed (shuffled epochs over the fixed set)
    micro_tokens: int = 49152            # tokens per microbatch (per forward/backward on one GPU)
    max_seq_len: int = 3072              # documents are cut to this many tokens (WR final stage)
    attn_cap: int = 2560                 # attention segments are cut at this many tokens (WR VIRTUAL_SEQ_CAP)

    # ---------------- objective (fixed for the whole run) ----------------
    ws_short: int = 5                    # attention windows in 128-token blocks (WR final stage (5, 11))
    ws_long: int = 11
    # YaRN rescaling chains applied once at init, reproducing the WR's rotary state at its final stage.
    yarn_short_chain: str = "1,3,5"
    yarn_long_chain: str = "3,7,11"
    mtp_weights: str = "1.0"             # multi-token-prediction weights; "1.0" = plain next-token loss
    prefix_weight: float = 0.0
    tied_embed: bool = True              # embed tied to lm_head for the whole run (False: untied from step 0)
    ngram: bool = False                  # hashed n-gram embedding table (dense full-batch path; needs 8 GPUs)
    ngram_rows: int = 84_602_880         # WR table size; smaller only to test the plumbing on fewer GPUs

    # ---------------- optimization ----------------
    steps: int = 1000
    anvil_lr: float = 0.023
    adam_lr: float = 0.008
    ngram_lr_mul: float = 70.0
    lr_mul: float = 1.73                 # peak multiplier on both base lrs (WR batch-24 stage: 1.73)
    warmup_frac: float = 0.0             # linear lr warmup over this fraction of steps
    cooldown_frac: float = 0.80          # linear decay to lr_floor over the last cooldown_frac of steps
    lr_floor: float = 0.30               # absolute multiplier at the end of cooldown
    anvil_wd: float = 2.25
    adam_wd: float = 0.005
    # ANVIL rail schedule, as fractions of steps (WR: warmup 240, cooldown 50, engage 514 of 1194 steps)
    rail_warmup_frac: float = 240 / 1194
    rail_cooldown_frac: float = 50 / 1194
    rail_engage_frac: float = 514 / 1194
    # The gradient handed to the optimizer is loss_scale * (mean-token-loss gradient). The WR's is the
    # per-rank token-sum gradient (49152 tokens at batch 24); only Adam's eps (1e-10) sees the scale.
    loss_scale: float = 49152.0

    # ---------------- evaluation / logging ----------------
    val_tokens: int = 10_485_760
    val_chunk_tokens: int = 262_144
    val_every: int = 0                   # steps; 0 = only at the end
    val_at_start: bool = False
    canon_mask: bool = True              # also report the WR's canonical-masked val loss (gpt2 only)
    out_dir: str = "runs/default"
    log_every: int = 1
    save_every: int = 0                  # checkpoint every N steps (0 = only at the end if save_final)
    save_final: bool = False
    seed: int = 0
    compile: bool = True
    # ---------------- train-loss measurement on a fixed sample of the fixed set ----------------
    train_eval_tokens: int = 0           # 0 = off; else ~this many tokens spread over the fixed set
    train_eval_every: int = 0            # steps (also evaluates val at the same steps)
    # ---------------- checkpoint / resume ----------------
    ckpt_minutes: float = 15.0           # rolling checkpoint every this many minutes of wall time (0 = off)
    ckpt_keep: int = 2                   # rolling checkpoints kept
    ckpt_milestone_every: int = 0        # also keep a checkpoint every N steps (0 = none); the final one is kept
    resume: bool = True                  # auto-resume from the latest checkpoint in <out_dir>/ckpt
    stop_after_steps: int = 0            # testing: checkpoint and exit after this many steps in THIS process
    partial_minutes: float = 0.0         # full mode: mid-step partial checkpoint every this many minutes (0 = off)
    partial_check_every: int = 64        # microbatches between (collective) partial-checkpoint checks
    partial_test_at: int = 0             # testing: save a partial at this microbatch of the first step, then exit
    # ---------------- halting rule (compare to a fresh-data reference at matched steps) ----------------
    halt_ref: str = ""                   # reference metrics.jsonl (needs train_eval and val_loss at eval steps)
    halt_margin: float = 0.03            # worse by more than max(halt_margin, halt_margin_rel * ref) ...
    halt_margin_rel: float = 0.0
    halt_span: int = 10                  # ... at every eval over at least this many consecutive steps
    halt_min_step: int = 5
    # ---------------- dynamics instrumentation (instr/DynamicsSuite) ----------------
    dyn: bool = False
    dyn_preset: str = "lean"
    dyn_raw_exclude: str = "value_embeds"   # comma list of params left out of raw G/u saves
    token_loss_every: int = 0            # save per-token training losses every N steps (fp16, per rank)

    @property
    def mtp(self) -> list[float]:
        return [float(x) for x in self.mtp_weights.split(",")]

    @staticmethod
    def chain(s: str) -> list[int]:
        return [int(x) for x in s.split(",")]

    def to_json(self) -> str:
        return json.dumps(dataclasses.asdict(self), indent=1)


def parse_config(argv: list[str]) -> FBConfig:
    cfg = FBConfig()
    types = {f.name: f.type for f in dataclasses.fields(FBConfig)}
    for arg in argv:
        key, _, value = arg.partition("=")
        key = key.lstrip("-")
        if key not in types:
            raise SystemExit(f"unknown config key {key!r}")
        t = types[key]
        if t in (bool, "bool"):
            v = value.lower() in ("1", "true", "yes", "on")
        elif t in (int, "int"):
            v = int(float(value))
        elif t in (float, "float"):
            v = float(value)
        else:
            v = value
        setattr(cfg, key, v)
    return cfg
