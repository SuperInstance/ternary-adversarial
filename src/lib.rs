//! # Ternary Adversarial
//!
//! Adversarial testing for ternary agents — stress-testing strategies against
//! worst-case environments. Provides tools to generate adversarial inputs,
//! measure robustness, and train agents to resist attacks.
//!
//! ## Core Concepts
//!
//! - **Adversary**: Generates adversarial environments designed to break strategies
//! - **Perturbation**: Small changes to inputs designed to flip agent decisions
//! - **RobustnessScore**: Measures strategy robustness to perturbations
//! - **AttackSuite**: Collection of adversarial attacks (gradient-free, random, targeted)
//! - **DefenseReport**: Analysis of vulnerable positions
//! - **AdversarialTraining**: Train agents against adversaries

#![forbid(unsafe_code)]
#![warn(missing_docs)]
#![allow(missing_docs)] // TODO: add docs to all enum variants

mod adversary;
mod attack;
mod perturbation;
mod robustness;
mod defense;
mod training;
#[cfg(test)]
mod tests;

pub use adversary::{Adversary, AdversaryConfig, Environment};
pub use attack::{AttackSuite, AttackKind, AttackResult};
pub use perturbation::{Perturbation, PerturbationKind};
pub use robustness::{RobustnessScore, RobustnessReport};
pub use defense::{DefenseReport, Vulnerability};
pub use training::{AdversarialTraining, TrainingConfig, TrainingLog};

/// A ternary value: Negative (-1), Zero (0), or Positive (+1).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Ternary {
    /// Negative value (-1)
    Negative,
    /// Zero value (0)
    Zero,
    /// Positive value (+1)
    Positive,
}

impl Ternary {
    /// Convert to integer representation.
    pub fn as_i8(self) -> i8 {
        match self {
            Ternary::Negative => -1,
            Ternary::Zero => 0,
            Ternary::Positive => 1,
        }
    }

    /// Convert from integer representation.
    pub fn from_i8(v: i8) -> Option<Self> {
        match v {
            -1 => Some(Ternary::Negative),
            0 => Some(Ternary::Zero),
            1 => Some(Ternary::Positive),
            _ => None,
        }
    }

    /// All three ternary values.
    pub fn all() -> [Ternary; 3] {
        [Ternary::Negative, Ternary::Zero, Ternary::Positive]
    }

    /// Flip the ternary value (negate).
    pub fn flip(self) -> Ternary {
        match self {
            Ternary::Negative => Ternary::Positive,
            Ternary::Zero => Ternary::Zero,
            Ternary::Positive => Ternary::Negative,
        }
    }
}

impl std::fmt::Display for Ternary {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.as_i8())
    }
}

/// A ternary state vector — the input to an agent's strategy.
pub type TernaryState = Vec<Ternary>;

/// A strategy maps a ternary state to a ternary decision.
pub type Strategy = fn(&TernaryState) -> Ternary;

/// Generate all possible ternary states of a given dimension.
pub fn all_states(dim: usize) -> Vec<TernaryState> {
    if dim == 0 {
        return vec![vec![]];
    }
    let smaller = all_states(dim - 1);
    let mut result = Vec::with_capacity(smaller.len() * 3);
    for val in Ternary::all() {
        for state in &smaller {
            let mut extended = state.clone();
            extended.push(val);
            result.push(extended);
        }
    }
    result
}
