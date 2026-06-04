//! Defense report: analysis of which positions are vulnerable and why.

use crate::{Ternary, TernaryState, Strategy, all_states};
use crate::robustness::RobustnessReport;

/// A specific vulnerability found in a strategy.
#[derive(Debug, Clone)]
pub struct Vulnerability {
    /// The state that exposes this vulnerability.
    pub state: TernaryState,
    /// The position(s) that, when perturbed, flip the decision.
    pub vulnerable_positions: Vec<usize>,
    /// The original decision.
    pub original_decision: Ternary,
    /// The flipped decision.
    pub flipped_decision: Ternary,
    /// A human-readable description.
    pub description: String,
}

/// A defense report analyzing strategy vulnerabilities.
#[derive(Debug, Clone)]
pub struct DefenseReport {
    /// The dimension of the ternary state.
    pub dimension: usize,
    /// All found vulnerabilities.
    pub vulnerabilities: Vec<Vulnerability>,
    /// Robustness report.
    pub robustness: RobustnessReport,
    /// Per-position vulnerability count.
    pub position_vulnerability_counts: Vec<usize>,
}

impl DefenseReport {
    /// Generate a defense report for a strategy.
    pub fn analyze(strategy: Strategy, dim: usize) -> Self {
        let states = all_states(dim);
        let mut vulnerabilities = Vec::new();
        let mut position_counts = vec![0usize; dim];

        for state in &states {
            let original = strategy(state);
            let mut vuln_positions = Vec::new();

            for i in 0..dim {
                // Try flipping
                let mut perturbed = state.clone();
                perturbed[i] = perturbed[i].flip();
                if strategy(&perturbed) != original {
                    vuln_positions.push(i);
                    position_counts[i] += 1;
                    continue;
                }

                // Try shifting
                for &val in &Ternary::all() {
                    if state[i] == val {
                        continue;
                    }
                    let mut p2 = state.clone();
                    p2[i] = val;
                    if strategy(&p2) != original {
                        vuln_positions.push(i);
                        position_counts[i] += 1;
                        break;
                    }
                }
            }

            if !vuln_positions.is_empty() {
                // Find the actual flipped decision
                let mut perturbed = state.clone();
                perturbed[vuln_positions[0]] = perturbed[vuln_positions[0]].flip();
                let flipped = strategy(&perturbed);

                let desc = format!(
                    "State [{}] flips from {} to {} when position(s) [{}] change",
                    state.iter().map(|t| t.to_string()).collect::<Vec<_>>().join(","),
                    original,
                    flipped,
                    vuln_positions.iter().map(|i| i.to_string()).collect::<Vec<_>>().join(",")
                );

                vulnerabilities.push(Vulnerability {
                    state: state.clone(),
                    vulnerable_positions: vuln_positions,
                    original_decision: original,
                    flipped_decision: flipped,
                    description: desc,
                });
            }
        }

        let robustness = RobustnessReport::evaluate(strategy, dim);

        DefenseReport {
            dimension: dim,
            vulnerabilities,
            robustness,
            position_vulnerability_counts: position_counts,
        }
    }

    /// Number of vulnerabilities found.
    pub fn vulnerability_count(&self) -> usize {
        self.vulnerabilities.len()
    }

    /// The most vulnerable position (highest count).
    pub fn most_vulnerable_position(&self) -> Option<usize> {
        self.position_vulnerability_counts.iter()
            .enumerate()
            .max_by_key(|(_, count)| *count)
            .map(|(i, _)| i)
    }

    /// Summary string.
    pub fn summary(&self) -> String {
        format!(
            "Defense Report (dim={}): {} vulnerabilities, robustness={}, most vulnerable position={:?}",
            self.dimension,
            self.vulnerability_count(),
            self.robustness.score,
            self.most_vulnerable_position(),
        )
    }
}
