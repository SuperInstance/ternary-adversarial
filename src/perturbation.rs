//! Perturbation: small changes to inputs designed to flip agent decisions.

use crate::Ternary;

/// A ternary state vector.
pub type TernaryState = Vec<Ternary>;

/// The kind of perturbation to apply.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PerturbationKind {
    /// Flip a single position (negate: -1 ↔ +1, 0 stays 0 or shifts).
    Flip,
    /// Shift by one step in a specific direction.
    ShiftPositive,
    /// Shift by one step in the negative direction.
    ShiftNegative,
    /// Set to zero.
    Zero,
    /// Random perturbation.
    Random { seed: u64 },
}

/// A specific perturbation applied to a state.
#[derive(Debug, Clone)]
pub struct Perturbation {
    /// Indices affected by this perturbation.
    pub indices: Vec<usize>,
    /// The kind of perturbation.
    pub kind: PerturbationKind,
}

impl Perturbation {
    /// Create a new perturbation.
    pub fn new(indices: Vec<usize>, kind: PerturbationKind) -> Self {
        Perturbation { indices, kind }
    }

    /// Apply this perturbation to a state, returning the perturbed state.
    pub fn apply(&self, state: &TernaryState) -> TernaryState {
        let mut result = state.clone();
        for &idx in &self.indices {
            if idx < result.len() {
                result[idx] = self.transform(result[idx]);
            }
        }
        result
    }

    fn transform(&self, val: Ternary) -> Ternary {
        match self.kind {
            PerturbationKind::Flip => val.flip(),
            PerturbationKind::ShiftPositive => match val {
                Ternary::Negative => Ternary::Zero,
                Ternary::Zero => Ternary::Positive,
                Ternary::Positive => Ternary::Positive,
            },
            PerturbationKind::ShiftNegative => match val {
                Ternary::Positive => Ternary::Zero,
                Ternary::Zero => Ternary::Negative,
                Ternary::Negative => Ternary::Negative,
            },
            PerturbationKind::Zero => Ternary::Zero,
            PerturbationKind::Random { seed } => {
                let idx = (seed as usize) % 3;
                Ternary::all()[idx]
            }
        }
    }

    /// Compute the L0 distance (number of changed positions) after applying.
    pub fn l0_distance(&self, original: &TernaryState, perturbed: &TernaryState) -> usize {
        original.iter()
            .zip(perturbed.iter())
            .filter(|(a, b)| a != b)
            .count()
    }

    /// Generate all single-position perturbations for a state of given dimension.
    pub fn all_single(dim: usize, kind: PerturbationKind) -> Vec<Perturbation> {
        (0..dim).map(|i| Perturbation::new(vec![i], kind)).collect()
    }
}

/// Generate perturbations that maximize decision change (gradient-free search).
pub fn generate_perturbations(
    state: &TernaryState,
    max_budget: usize,
) -> Vec<Perturbation> {
    let mut perturbations = Vec::new();

    // Single-position perturbations
    for i in 0..state.len() {
        perturbations.push(Perturbation::new(vec![i], PerturbationKind::Flip));
        perturbations.push(Perturbation::new(vec![i], PerturbationKind::ShiftPositive));
        perturbations.push(Perturbation::new(vec![i], PerturbationKind::ShiftNegative));
    }

    // Multi-position perturbations (if budget allows)
    if max_budget > 1 && state.len() >= 2 {
        for i in 0..state.len() {
            for j in (i + 1)..state.len() {
                if perturbations.len() > max_budget * state.len() * 4 {
                    break;
                }
                perturbations.push(Perturbation::new(vec![i, j], PerturbationKind::Flip));
            }
        }
    }

    perturbations
}
