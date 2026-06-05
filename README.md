# ternary-adversarial

Adversarial testing for ternary agents — stress-testing strategies against worst-case environments.

## Overview

This crate provides tools to evaluate and harden ternary decision strategies against adversarial perturbations. It implements concepts from adversarial machine learning adapted to the ternary (−1, 0, +1) domain.

### Core Components

- **Adversary** — Generates adversarial environments designed to flip a strategy's output
- **Perturbation** — Small changes to inputs (flip, shift, zero, random) designed to change decisions
- **RobustnessScore** — Measures how robust a strategy is to perturbations (0.0–1.0)
- **AttackSuite** — Collection of adversarial attacks: gradient-free, random, targeted, universal
- **DefenseReport** — Analysis of which positions are vulnerable and why
- **AdversarialTraining** — Trains agents against adversaries to improve robustness

## Quick Start

```rust
use ternary_adversarial::*;

// Define a strategy: decide based on sum of ternary values
fn sum_strategy(state: &TernaryState) -> Ternary {
    let sum: i8 = state.iter().map(|t| t.as_i8()).sum();
    if sum > 0 { Ternary::Positive }
    else if sum < 0 { Ternary::Negative }
    else { Ternary::Zero }
}

// Evaluate robustness
let report = robustness::RobustnessReport::evaluate(sum_strategy, 3);
println!("Robustness: {}", report.score);

// Analyze vulnerabilities
let defense = defense::DefenseReport::analyze(sum_strategy, 3);
println!("{}", defense.summary());

// Run adversarial training to harden
let config = training::TrainingConfig::default();
let training = training::AdversarialTraining::new(config);
let (table, logs) = training.train(sum_strategy, 3);
```

## Adversarial ML Theory

### What is Adversarial Testing?

Adversarial testing applies principles from adversarial machine learning to evaluate the robustness of decision strategies. In traditional ML, adversarial examples are carefully crafted inputs designed to cause model misclassification. Here, we adapt these concepts to ternary decision systems.

### Threat Model

A **ternary adversary** operates under the following assumptions:

1. **Black-box access**: The adversary can query the strategy but does not know its internal structure
2. **Perturbation budget**: The adversary can modify at most `k` positions in the input state
3. **L₀ norm constraint**: Changes are measured by the number of positions modified (not magnitude, since ternary values are discrete)

### Attack Strategies

| Attack | Description |
|--------|-------------|
| **Gradient-Free** | Systematically tries all single-position perturbations |
| **Random** | Randomly perturbs positions using seeded RNG |
| **Targeted** | Attempts to force a specific target output |
| **Universal** | Finds perturbations that work across many states |

### Robustness Metrics

**Robustness Score** = 1 − (fraction of states where perturbation flips the decision)

- Score ≥ 0.9: Robust
- Score < 0.5: Vulnerable
- Score = 1.0: Impervious (e.g., constant strategies)

### Adversarial Training

Adversarial training iteratively:

1. Finds adversarial examples that flip the strategy's output
2. Adjusts the decision table to resist those specific perturbations
3. Repeats until robustness reaches a target threshold

This is analogous to adversarial training in deep learning (Goodfellow et al., 2015), but adapted for discrete ternary strategies.

### Defense Strategies

- **Position hardening**: Identify the most vulnerable positions and make them more resistant
- **Decision smoothing**: Ensure nearby states produce consistent decisions
- **Budget-aware design**: Design strategies that resist perturbations up to a given budget

## Properties

- **Pure Rust**, no unsafe code (`#![forbid(unsafe_code)]`)
- **No external dependencies**
- **Zero allocations in hot paths** (where possible)
- **Comprehensive test coverage** (25+ tests)

## License

MIT

## See Also
- **ternary-arena** — related
- **ternary-agent** — related
- **ternary-failure** — related
- **ternary-noise** — related

