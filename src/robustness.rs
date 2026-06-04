//! Robustness scoring: measure how robust a strategy is to perturbations.

use crate::{TernaryState, Strategy, all_states};
use crate::adversary::{Adversary, AdversaryConfig};
use crate::perturbation::{Perturbation, PerturbationKind};

/// A robustness score between 0.0 (completely vulnerable) and 1.0 (perfectly robust).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RobustnessScore(pub f64);

impl RobustnessScore {
    /// Create a new robustness score, clamped to [0, 1].
    pub fn new(score: f64) -> Self {
        RobustnessScore(score.max(0.0).min(1.0))
    }

    /// Get the raw score value.
    pub fn value(&self) -> f64 {
        self.0
    }

    /// Whether the strategy is considered robust (score >= 0.9).
    pub fn is_robust(&self) -> bool {
        self.0 >= 0.9
    }

    /// Whether the strategy is considered vulnerable (score < 0.5).
    pub fn is_vulnerable(&self) -> bool {
        self.0 < 0.5
    }
}

impl std::fmt::Display for RobustnessScore {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:.3}", self.0)
    }
}

/// A detailed robustness report.
#[derive(Debug, Clone)]
pub struct RobustnessReport {
    /// Overall robustness score.
    pub score: RobustnessScore,
    /// Number of states tested.
    pub states_tested: usize,
    /// Number of states where perturbation flipped the decision.
    pub states_flipped: usize,
    /// Robustness per dimension position.
    pub position_scores: Vec<RobustnessScore>,
}

impl RobustnessReport {
    /// Generate a full robustness report for a strategy.
    pub fn evaluate(strategy: Strategy, dim: usize) -> Self {
        let states = all_states(dim);
        let adversary = Adversary::new(AdversaryConfig::default());
        let mut flipped = 0;
        let mut position_flips = vec![0usize; dim];
        let mut position_total = vec![0usize; dim];

        for state in &states {
            let original = strategy(state);
            let vulnerable = adversary.attack_state(state, strategy).is_some();
            if vulnerable {
                flipped += 1;
            }

            // Per-position analysis
            for i in 0..dim {
                let perturbations = vec![
                    Perturbation::new(vec![i], PerturbationKind::Flip),
                    Perturbation::new(vec![i], PerturbationKind::ShiftPositive),
                    Perturbation::new(vec![i], PerturbationKind::ShiftNegative),
                ];
                for p in &perturbations {
                    let perturbed = p.apply(state);
                    if strategy(&perturbed) != original {
                        position_flips[i] += 1;
                    }
                    position_total[i] += 1;
                }
            }
        }

        let score = if states.is_empty() {
            1.0
        } else {
            1.0 - (flipped as f64 / states.len() as f64)
        };

        let position_scores: Vec<RobustnessScore> = (0..dim)
            .map(|i| {
                let s = if position_total[i] == 0 {
                    1.0
                } else {
                    1.0 - (position_flips[i] as f64 / position_total[i] as f64)
                };
                RobustnessScore::new(s)
            })
            .collect();

        RobustnessReport {
            score: RobustnessScore::new(score),
            states_tested: states.len(),
            states_flipped: flipped,
            position_scores,
        }
    }

    /// Quick robustness score (sample-based, for large dimensions).
    pub fn quick_score(strategy: Strategy, states: &[TernaryState]) -> RobustnessScore {
        if states.is_empty() {
            return RobustnessScore::new(1.0);
        }
        let adversary = Adversary::default_adversary();
        let flipped = states.iter()
            .filter(|s| adversary.attack_state(s, strategy).is_some())
            .count();
        RobustnessScore::new(1.0 - (flipped as f64 / states.len() as f64))
    }
}
