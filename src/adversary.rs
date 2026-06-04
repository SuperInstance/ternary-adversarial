//! Adversary: generates adversarial environments designed to break strategies.

use crate::{Ternary, TernaryState, Strategy};

/// Configuration for an adversary.
#[derive(Debug, Clone)]
pub struct AdversaryConfig {
    /// Maximum perturbation budget (number of positions that can be flipped).
    pub budget: usize,
    /// Number of random seeds to try when searching for adversarial examples.
    pub seeds: usize,
    /// Whether to allow perturbing zero-valued positions.
    pub perturb_zeros: bool,
}

impl Default for AdversaryConfig {
    fn default() -> Self {
        AdversaryConfig {
            budget: 1,
            seeds: 100,
            perturb_zeros: true,
        }
    }
}

/// An adversarial environment: a perturbed state paired with metadata.
#[derive(Debug, Clone)]
pub struct Environment {
    /// The original (clean) state.
    pub original: TernaryState,
    /// The adversarially perturbed state.
    pub perturbed: TernaryState,
    /// Indices that were changed.
    pub changed_indices: Vec<usize>,
    /// The strategy's decision on the original state.
    pub original_decision: Ternary,
    /// The strategy's decision on the perturbed state.
    pub perturbed_decision: Ternary,
}

impl Environment {
    /// Whether this adversarial example successfully flipped the decision.
    pub fn is_successful(&self) -> bool {
        self.original_decision != self.perturbed_decision
    }
}

/// An adversary that searches for perturbations that flip a strategy's output.
#[derive(Debug, Clone)]
pub struct Adversary {
    config: AdversaryConfig,
}

impl Adversary {
    /// Create a new adversary with the given configuration.
    pub fn new(config: AdversaryConfig) -> Self {
        Adversary { config }
    }

    /// Create a default adversary.
    pub fn default_adversary() -> Self {
        Adversary::new(AdversaryConfig::default())
    }

    /// Try to find an adversarial example for a given state and strategy.
    pub fn attack_state(&self, state: &TernaryState, strategy: Strategy) -> Option<Environment> {
        let original_decision = strategy(state);

        // Try all single-position perturbations first (budget = 1)
        for budget in 1..=self.config.budget {
            if let Some(env) = self.search_budget(state, strategy, original_decision, budget) {
                return Some(env);
            }
        }
        None
    }

    /// Search for adversarial examples with a specific budget.
    fn search_budget(
        &self,
        state: &TernaryState,
        strategy: Strategy,
        original_decision: Ternary,
        budget: usize,
    ) -> Option<Environment> {
        if budget == 0 || state.is_empty() {
            return None;
        }

        // Systematic search: try flipping each position
        if budget == 1 {
            for (i, val) in state.iter().enumerate() {
                if *val == Ternary::Zero && !self.config.perturb_zeros {
                    continue;
                }
                let perturbed = self.perturb_single(state, i);
                let perturbed_decision = strategy(&perturbed);
                if perturbed_decision != original_decision {
                    return Some(Environment {
                        original: state.clone(),
                        perturbed,
                        changed_indices: vec![i],
                        original_decision,
                        perturbed_decision,
                    });
                }
            }
        }

        // Multi-budget: try combinations (simplified greedy approach)
        if budget > 1 {
            let mut current = state.clone();
            let mut changed = Vec::new();
            for _ in 0..budget {
                let best = self.find_best_flip(&current, strategy, original_decision);
                if let Some(idx) = best {
                    current = self.perturb_single(&current, idx);
                    changed.push(idx);
                }
            }
            let perturbed_decision = strategy(&current);
            if perturbed_decision != original_decision {
                return Some(Environment {
                    original: state.clone(),
                    perturbed: current,
                    changed_indices: changed,
                    original_decision,
                    perturbed_decision,
                });
            }
        }

        None
    }

    /// Find the single flip that most changes the strategy output.
    fn find_best_flip(
        &self,
        state: &TernaryState,
        strategy: Strategy,
        original_decision: Ternary,
    ) -> Option<usize> {
        for (i, val) in state.iter().enumerate() {
            if *val == Ternary::Zero && !self.config.perturb_zeros {
                continue;
            }
            let perturbed = self.perturb_single(state, i);
            if strategy(&perturbed) != original_decision {
                return Some(i);
            }
        }
        // If no flip changes the output, just return the first perturbable index
        state.iter().enumerate()
            .find(|(_, v)| !(**v == Ternary::Zero && !self.config.perturb_zeros))
            .map(|(i, _)| i)
    }

    /// Perturb a single position in the state.
    fn perturb_single(&self, state: &TernaryState, index: usize) -> TernaryState {
        let mut perturbed = state.clone();
        perturbed[index] = match perturbed[index] {
            Ternary::Negative => Ternary::Positive,
            Ternary::Positive => Ternary::Negative,
            Ternary::Zero => Ternary::Positive, // Zero perturbs to positive by default
        };
        perturbed
    }

    /// Attack all states and return successful adversarial examples.
    pub fn attack_all(&self, states: &[TernaryState], strategy: Strategy) -> Vec<Environment> {
        states.iter()
            .filter_map(|s| self.attack_state(s, strategy))
            .collect()
    }

    /// Get the config.
    pub fn config(&self) -> &AdversaryConfig {
        &self.config
    }
}
