//! Attack suite: collection of adversarial attacks.

use crate::{Ternary, TernaryState, Strategy};
use crate::adversary::Environment;
use crate::perturbation::{Perturbation, PerturbationKind};

/// The kind of attack to perform.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AttackKind {
    /// Gradient-free search: systematically try all perturbations.
    GradientFree,
    /// Random perturbations with a given seed.
    /// Random perturbation with seed.
    Random { /// RNG seed
        seed: u64,
    },
    /// Targeted attack: try to flip to a specific target decision.
    Targeted { target: Ternary },
    /// Universal attack: find perturbation that works on many states.
    Universal,
}

/// Result of an attack.
#[derive(Debug, Clone)]
pub struct AttackResult {
    /// The kind of attack that was performed.
    pub kind: AttackKind,
    /// Whether a successful adversarial example was found.
    pub success: bool,
    /// The adversarial environment (if found).
    pub environment: Option<Environment>,
    /// Number of queries made to the strategy.
    pub queries: usize,
}

/// A suite of adversarial attacks.
#[derive(Debug, Clone)]
pub struct AttackSuite {
    /// The kinds of attacks in this suite.
    pub attacks: Vec<AttackKind>,
}

impl AttackSuite {
    /// Create a default suite with all attack types.
    pub fn default_suite() -> Self {
        AttackSuite {
            attacks: vec![
                AttackKind::GradientFree,
                AttackKind::Random { seed: 42 },
                AttackKind::Targeted { target: Ternary::Negative },
                AttackKind::Targeted { target: Ternary::Positive },
                AttackKind::Targeted { target: Ternary::Zero },
            ],
        }
    }

    /// Create a suite with only gradient-free attacks.
    pub fn gradient_free_only() -> Self {
        AttackSuite {
            attacks: vec![AttackKind::GradientFree],
        }
    }

    /// Create a suite with only random attacks.
    pub fn random_only(seed: u64, num_seeds: usize) -> Self {
        AttackSuite {
            attacks: (0..num_seeds)
                .map(|i| AttackKind::Random { seed: seed + i as u64 })
                .collect(),
        }
    }

    /// Run all attacks in the suite against a single state.
    pub fn attack_state(&self, state: &TernaryState, strategy: Strategy) -> Vec<AttackResult> {
        self.attacks.iter()
            .map(|kind| Self::run_attack(kind, state, strategy))
            .collect()
    }

    /// Run a single attack.
    pub fn run_attack(kind: &AttackKind, state: &TernaryState, strategy: Strategy) -> AttackResult {
        let original = strategy(state);

        match kind {
            AttackKind::GradientFree => {
                let mut queries = 0;
                for i in 0..state.len() {
                    for perturb_kind in [PerturbationKind::Flip, PerturbationKind::ShiftPositive, PerturbationKind::ShiftNegative] {
                        let p = Perturbation::new(vec![i], perturb_kind);
                        let perturbed = p.apply(state);
                        queries += 1;
                        let decision = strategy(&perturbed);
                        if decision != original {
                            return AttackResult {
                                kind: *kind,
                                success: true,
                                environment: Some(Environment {
                                    original: state.clone(),
                                    perturbed,
                                    changed_indices: vec![i],
                                    original_decision: original,
                                    perturbed_decision: decision,
                                }),
                                queries,
                            };
                        }
                    }
                }
                AttackResult {
                    kind: *kind,
                    success: false,
                    environment: None,
                    queries,
                }
            }

            AttackKind::Random { seed } => {
                let mut queries = 0;
                let mut rng_state = *seed;
                for _ in 0..100 {
                    rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                    let idx = (rng_state as usize) % state.len();
                    let val_choice = ((rng_state >> 32) as usize) % 3;
                    let new_val = Ternary::all()[val_choice];

                    let mut perturbed = state.clone();
                    perturbed[idx] = new_val;
                    queries += 1;

                    let decision = strategy(&perturbed);
                    if decision != original {
                        return AttackResult {
                            kind: *kind,
                            success: true,
                            environment: Some(Environment {
                                original: state.clone(),
                                perturbed,
                                changed_indices: vec![idx],
                                original_decision: original,
                                perturbed_decision: decision,
                            }),
                            queries,
                        };
                    }
                }
                AttackResult {
                    kind: *kind,
                    success: false,
                    environment: None,
                    queries,
                }
            }

            AttackKind::Targeted { target } => {
                let mut queries = 0;
                if original == *target {
                    return AttackResult {
                        kind: *kind,
                        success: false,
                        environment: None,
                        queries: 0,
                    };
                }
                for i in 0..state.len() {
                    let mut perturbed = state.clone();
                    perturbed[i] = *target;
                    queries += 1;
                    if strategy(&perturbed) == *target {
                        return AttackResult {
                            kind: *kind,
                            success: true,
                            environment: Some(Environment {
                                original: state.clone(),
                                perturbed,
                                changed_indices: vec![i],
                                original_decision: original,
                                perturbed_decision: *target,
                            }),
                            queries,
                        };
                    }
                }
                AttackResult {
                    kind: *kind,
                    success: false,
                    environment: None,
                    queries,
                }
            }

            AttackKind::Universal => {
                // Try each perturbation on all positions
                let mut queries = 0;
                for i in 0..state.len() {
                    for &new_val in &Ternary::all() {
                        if state[i] == new_val {
                            continue;
                        }
                        let mut perturbed = state.clone();
                        perturbed[i] = new_val;
                        queries += 1;
                        if strategy(&perturbed) != original {
                            let dec = strategy(&perturbed);
                            return AttackResult {
                                kind: *kind,
                                success: true,
                                environment: Some(Environment {
                                    original: state.clone(),
                                    perturbed,
                                    changed_indices: vec![i],
                                    original_decision: original,
                                    perturbed_decision: dec,
                                }),
                                queries,
                            };
                        }
                    }
                }
                AttackResult {
                    kind: *kind,
                    success: false,
                    environment: None,
                    queries,
                }
            }
        }
    }

    /// Run the full suite against multiple states.
    pub fn attack_all(&self, states: &[TernaryState], strategy: Strategy) -> Vec<Vec<AttackResult>> {
        states.iter().map(|s| self.attack_state(s, strategy)).collect()
    }

    /// Count successful attacks across all results.
    pub fn success_rate(results: &[AttackResult]) -> f64 {
        if results.is_empty() {
            return 0.0;
        }
        let successes = results.iter().filter(|r| r.success).count();
        successes as f64 / results.len() as f64
    }
}
