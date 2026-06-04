//! Adversarial training: train agents against adversaries to improve robustness.

use crate::{Ternary, TernaryState, Strategy};
use crate::adversary::{Adversary, AdversaryConfig};
use crate::robustness::RobustnessScore;
use crate::all_states;

/// Configuration for adversarial training.
#[derive(Debug, Clone)]
pub struct TrainingConfig {
    /// Number of training rounds.
    pub rounds: usize,
    /// Maximum perturbation budget for the adversary.
    pub budget: usize,
    /// Minimum robustness score to achieve.
    pub target_score: f64,
}

impl Default for TrainingConfig {
    fn default() -> Self {
        TrainingConfig {
            rounds: 10,
            budget: 2,
            target_score: 0.9,
        }
    }
}

/// A log entry from training.
#[derive(Debug, Clone)]
pub struct TrainingLog {
    /// Round number.
    pub round: usize,
    /// Robustness score at this round.
    pub score: RobustnessScore,
    /// Number of adversarial examples found.
    pub adversarial_count: usize,
    /// Strategy decisions that were corrected.
    pub corrections: Vec<(TernaryState, Ternary, Ternary)>,
}

/// Adversarial training engine.
///
/// Since strategies are function pointers (fn), we can't modify them directly.
/// Instead, adversarial training here simulates the process and produces
/// a "hardened" decision table (mapping states to decisions) that can be
/// used as a lookup strategy.
#[derive(Debug)]
pub struct AdversarialTraining {
    config: TrainingConfig,
}

impl AdversarialTraining {
    /// Create a new training instance.
    pub fn new(config: TrainingConfig) -> Self {
        AdversarialTraining { config }
    }

    /// Run adversarial training, returning a decision table and training log.
    ///
    /// The decision table maps each state to a decision. It starts from the
    /// original strategy and iteratively hardens against adversarial examples.
    pub fn train(
        &self,
        strategy: Strategy,
        dim: usize,
    ) -> (DecisionTable, Vec<TrainingLog>) {
        let states = all_states(dim);
        let adversary = Adversary::new(AdversaryConfig {
            budget: self.config.budget,
            seeds: 50,
            perturb_zeros: true,
        });

        // Initialize decision table from original strategy
        let mut table = DecisionTable::from_strategy(strategy, &states);
        let mut logs = Vec::new();

        for round in 0..self.config.rounds {
            let mut adversarial_count = 0;
            let mut corrections = Vec::new();

            let table_clone = table.clone();
            for state in &states {
                let decision = table_clone.decide(state);

                // Use manual perturbation search since we can't pass closures
                let mut found_adversarial = false;
                for i in 0..state.len() {
                    let mut perturbed = state.clone();
                    perturbed[i] = match perturbed[i] {
                        Ternary::Negative => Ternary::Positive,
                        Ternary::Positive => Ternary::Negative,
                        Ternary::Zero => Ternary::Positive,
                    };
                    if table_clone.decide(&perturbed) != decision {
                        found_adversarial = true;
                        break;
                    }
                }

                if found_adversarial {
                    adversarial_count += 1;
                    let original = table_clone.decide(state);
                    // Harden: pick the decision most robust to perturbation
                    let mut counts = [0usize; 3];
                    for &val in &Ternary::all() {
                        for (i, orig) in state.iter().enumerate() {
                            if *orig != val {
                                let mut perturbed = state.clone();
                                perturbed[i] = val;
                                let p_decision = table_clone.decide(&perturbed);
                                match p_decision {
                                    Ternary::Negative => counts[0] += 1,
                                    Ternary::Zero => counts[1] += 1,
                                    Ternary::Positive => counts[2] += 1,
                                }
                            }
                        }
                    }
                    let best = if counts[0] >= counts[1] && counts[0] >= counts[2] {
                        Ternary::Negative
                    } else if counts[1] >= counts[2] {
                        Ternary::Zero
                    } else {
                        Ternary::Positive
                    };

                    if best != original {
                        corrections.push((state.clone(), original, best));
                        table.set(state, best);
                    }
                }
            }

            let score = self.evaluate_table(&table, &states, &adversary);
            logs.push(TrainingLog {
                round,
                score,
                adversarial_count,
                corrections,
            });

            if score.value() >= self.config.target_score {
                break;
            }
        }

        (table, logs)
    }

    fn evaluate_table(
        &self,
        table: &DecisionTable,
        states: &[TernaryState],
        _adversary: &Adversary,
    ) -> RobustnessScore {
        let table_clone = table.clone();
        let mut flipped = 0usize;
        for state in states {
            let decision = table_clone.decide(state);
            for i in 0..state.len() {
                let mut perturbed = state.clone();
                perturbed[i] = match perturbed[i] {
                    Ternary::Negative => Ternary::Positive,
                    Ternary::Positive => Ternary::Negative,
                    Ternary::Zero => Ternary::Positive,
                };
                if table_clone.decide(&perturbed) != decision {
                    flipped += 1;
                    break;
                }
            }
        }
        RobustnessScore::new(1.0 - (flipped as f64 / states.len().max(1) as f64))
    }
}

/// A decision table mapping states to decisions.
///
/// This serves as a "trained" strategy that can be hardened through
/// adversarial training.
#[derive(Debug, Clone)]
pub struct DecisionTable {
    entries: Vec<(TernaryState, Ternary)>,
}

impl DecisionTable {
    /// Create a decision table from a strategy.
    pub fn from_strategy(strategy: Strategy, states: &[TernaryState]) -> Self {
        let entries = states.iter()
            .map(|s| (s.clone(), strategy(s)))
            .collect();
        DecisionTable { entries }
    }

    /// Look up the decision for a state.
    pub fn decide(&self, state: &TernaryState) -> Ternary {
        // Find exact match
        for (s, d) in &self.entries {
            if s.len() == state.len() && s.iter().zip(state.iter()).all(|(a, b)| a == b) {
                return *d;
            }
        }
        // Default: compute sum and decide
        let sum: i8 = state.iter().map(|t| t.as_i8()).sum();
        if sum < 0 { Ternary::Negative }
        else if sum > 0 { Ternary::Positive }
        else { Ternary::Zero }
    }

    /// Set the decision for a state.
    pub fn set(&mut self, state: &TernaryState, decision: Ternary) {
        for (s, d) in &mut self.entries {
            if s.len() == state.len() && s.iter().zip(state.iter()).all(|(a, b)| a == b) {
                *d = decision;
                return;
            }
        }
        self.entries.push((state.clone(), decision));
    }

    /// Get all entries.
    pub fn entries(&self) -> &[(TernaryState, Ternary)] {
        &self.entries
    }

    /// Count entries.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether the table is empty.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}
