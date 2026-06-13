# Ternary Adversarial

**Ternary Adversarial** is a Rust library for stress-testing ternary agent strategies against worst-case perturbations — providing adversarial attack generation, robustness scoring, defense analysis, and adversarial training to harden ternary decision systems.

## Why It Matters

Any decision-making system can be fooled by carefully crafted inputs. In binary systems, adversarial examples are well-studied (image classification, spam detection). Ternary systems {-1, 0, +1} are different: the zero state provides a natural "abstain" option that can be exploited or defended. This crate provides a complete adversarial testing pipeline: generate attacks (gradient-free, random, targeted), measure robustness (per-position vulnerability scores), analyze defenses (which inputs are exploitable), and train against adversaries to produce hardened decision tables.

## How It Works

### Threat Model

A ternary **strategy** is a function `Strategy = fn(&TernaryState) -> Ternary` that maps a vector of ternary values to a ternary decision. An adversary tries to find a minimal perturbation (flip or shift of a small number of positions) that changes the decision.

### Perturbation Kinds

| Kind | Transform | L0 Cost |
|------|-----------|---------|
| Flip | -1 ↔ +1, 0 stays | 1 per position |
| ShiftPositive | -1→0, 0→+1, +1→+1 | 1 per position |
| ShiftNegative | +1→0, 0→-1, -1→-1 | 1 per position |
| Zero | Any → 0 | 1 per position |
| Random | Uniform random | 1 per position |

### Attack Algorithm (Gradient-Free)

For a state S with decision D:

```
for budget in 1..=max_budget:
    for each subset of `budget` positions:
        for each perturbation kind:
            perturbed = perturb(S, positions, kind)
            if strategy(perturbed) != D:
                return AdversarialExample(found!)
```

Worst case: **O(C(N, k) · k · 3^k)** for budget k in N dimensions. Single-position attacks (k=1): **O(N · 5)** = **O(N)**.

### Robustness Scoring

```
robustness = 1 - (states_flipped / states_tested)
```

A strategy with robustness ≥ 0.9 is considered robust; < 0.5 is vulnerable. Per-position scores identify which state components are most exploitable.

**Total states tested**: 3^D for dimension D. Robustness evaluation: **O(3^D · D)** — exhaustive over all states and all single-position perturbations.

### Adversarial Training

Training produces a `DecisionTable` — a lookup mapping every state to a hardened decision:

1. Initialize table from original strategy
2. For each round:
   - Find adversarial examples against current table
   - Correct decisions on adversarial states
   - Re-evaluate robustness
3. Stop when robustness ≥ target_score or max rounds exceeded

Training cost: **O(rounds · 3^D · D)**.

### Defense Report

For each vulnerable state, the report records:
- Which positions are exploitable
- Original vs. flipped decision
- Human-readable description

Per-position vulnerability count: **O(3^D · D)** to compute.

## Quick Start

```rust
use ternary_adversarial::*;

fn my_strategy(state: &TernaryState) -> Ternary {
    let sum: i8 = state.iter().map(|t| t.as_i8()).sum();
    if sum > 0 { Ternary::Positive }
    else if sum < 0 { Ternary::Negative }
    else { Ternary::Zero }
}

let report = RobustnessReport::evaluate(my_strategy, 3);
println!("Robustness: {:.3}", report.score.value());

let defense = DefenseReport::analyze(my_strategy, 3);
println!("Vulnerabilities: {}", defense.vulnerabilities.len());
```

## API

| Module | Key Types |
|--------|-----------|
| `adversary` | `Adversary`, `AdversaryConfig`, `Environment` |
| `attack` | `AttackSuite`, `AttackKind`, `AttackResult` |
| `perturbation` | `Perturbation`, `PerturbationKind` |
| `robustness` | `RobustnessScore`, `RobustnessReport` |
| `defense` | `DefenseReport`, `Vulnerability` |
| `training` | `AdversarialTraining`, `TrainingConfig`, `TrainingLog`, `DecisionTable` |

Core type: `Ternary` (Negative, Zero, Positive), `Strategy = fn(&TernaryState) -> Ternary`, `TernaryState = Vec<Ternary>`.

## Architecture Notes

Ternary Adversarial provides the security testing layer for agent strategies in SuperInstance. In γ + η = C, adversarial attacks exploit γ (growth — finding inputs that cause incorrect expansion) while robustness scoring and training implement η (avoidance — hardening against exploitable weaknesses). The `DecisionTable` output integrates with `ternary-agent` as an alternative to function-based strategies.

See [ARCHITECTURE.md](https://github.com/SuperInstance/SuperInstance/blob/main/ARCHITECTURE.md) for agent security architecture.

## References

1. Goodfellow, I. et al. (2015). "Explaining and Harnessing Adversarial Examples." *ICLR*.
2. Madry, A. et al. (2018). "Towards Deep Learning Models Resistant to Adversarial Attacks." *ICLR*.
3. Athalye, A. et al. (2018). "Obfuscated Gradients Give a False Sense of Security." *ICML*.

## License

MIT
