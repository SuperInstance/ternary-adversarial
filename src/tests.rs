#[cfg(test)]
mod tests {
    use crate::*;
    use crate::adversary::{Adversary, AdversaryConfig};
    use crate::attack::{AttackSuite, AttackKind, AttackResult};
    use crate::defense::DefenseReport;
    use crate::perturbation::{Perturbation, PerturbationKind, generate_perturbations};
    use crate::robustness::{RobustnessScore, RobustnessReport};
    use crate::training::{AdversarialTraining, TrainingConfig, DecisionTable};

    // --- Ternary tests ---

    #[test]
    fn ternary_as_i8() {
        assert_eq!(Ternary::Negative.as_i8(), -1);
        assert_eq!(Ternary::Zero.as_i8(), 0);
        assert_eq!(Ternary::Positive.as_i8(), 1);
    }

    #[test]
    fn ternary_from_i8() {
        assert_eq!(Ternary::from_i8(-1), Some(Ternary::Negative));
        assert_eq!(Ternary::from_i8(0), Some(Ternary::Zero));
        assert_eq!(Ternary::from_i8(1), Some(Ternary::Positive));
        assert_eq!(Ternary::from_i8(5), None);
    }

    #[test]
    fn ternary_flip() {
        assert_eq!(Ternary::Negative.flip(), Ternary::Positive);
        assert_eq!(Ternary::Positive.flip(), Ternary::Negative);
        assert_eq!(Ternary::Zero.flip(), Ternary::Zero);
    }

    #[test]
    fn all_states_dim1() {
        let states = all_states(1);
        assert_eq!(states.len(), 3);
    }

    #[test]
    fn all_states_dim2() {
        let states = all_states(2);
        assert_eq!(states.len(), 9); // 3^2
    }

    // --- Strategy helpers ---

    fn sum_strategy(state: &TernaryState) -> Ternary {
        let sum: i8 = state.iter().map(|t| t.as_i8()).sum();
        if sum > 0 { Ternary::Positive }
        else if sum < 0 { Ternary::Negative }
        else { Ternary::Zero }
    }

    fn always_positive(_: &TernaryState) -> Ternary {
        Ternary::Positive
    }

    fn first_element_strategy(state: &TernaryState) -> Ternary {
        if state.is_empty() { Ternary::Zero }
        else { state[0] }
    }

    fn majority_strategy(state: &TernaryState) -> Ternary {
        let pos = state.iter().filter(|t| **t == Ternary::Positive).count();
        let neg = state.iter().filter(|t| **t == Ternary::Negative).count();
        if pos > neg { Ternary::Positive }
        else if neg > pos { Ternary::Negative }
        else { Ternary::Zero }
    }

    // --- Adversary tests ---

    #[test]
    fn adversary_breaks_sum_strategy() {
        // [Positive, Negative] -> sum=0 -> Zero
        // Flip first to Negative: [Neg, Neg] -> Negative (flipped!)
        let adversary = Adversary::default_adversary();
        let state = vec![Ternary::Positive, Ternary::Negative];
        let result = adversary.attack_state(&state, sum_strategy);
        assert!(result.is_some());
        let env = result.unwrap();
        assert!(env.is_successful());
    }

    #[test]
    fn adversary_fails_on_constant_strategy() {
        let adversary = Adversary::default_adversary();
        let state = vec![Ternary::Positive, Ternary::Negative];
        let result = adversary.attack_state(&state, always_positive);
        assert!(result.is_none()); // Can't flip always-positive
    }

    #[test]
    fn adversary_budget_config() {
        let config = AdversaryConfig { budget: 3, seeds: 10, perturb_zeros: false };
        let adversary = Adversary::new(config);
        assert_eq!(adversary.config().budget, 3);
    }

    #[test]
    fn adversary_attack_all() {
        let adversary = Adversary::default_adversary();
        let states = all_states(2);
        let results = adversary.attack_all(&states, sum_strategy);
        assert!(!results.is_empty());
    }

    // --- Perturbation tests ---

    #[test]
    fn perturbation_flip() {
        let state = vec![Ternary::Negative, Ternary::Zero, Ternary::Positive];
        let p = Perturbation::new(vec![0], PerturbationKind::Flip);
        let perturbed = p.apply(&state);
        assert_eq!(perturbed[0], Ternary::Positive);
        assert_eq!(perturbed[1], Ternary::Zero); // unchanged
    }

    #[test]
    fn perturbation_shift_positive() {
        let state = vec![Ternary::Negative];
        let p = Perturbation::new(vec![0], PerturbationKind::ShiftPositive);
        let perturbed = p.apply(&state);
        assert_eq!(perturbed[0], Ternary::Zero);
    }

    #[test]
    fn perturbation_l0_distance() {
        let original = vec![Ternary::Positive, Ternary::Negative, Ternary::Zero];
        let perturbed = vec![Ternary::Negative, Ternary::Negative, Ternary::Zero];
        let p = Perturbation::new(vec![0], PerturbationKind::Flip);
        assert_eq!(p.l0_distance(&original, &perturbed), 1);
    }

    #[test]
    fn generate_perturbations_count() {
        let state = vec![Ternary::Positive, Ternary::Negative];
        let perts = generate_perturbations(&state, 2);
        // 2 positions * 3 kinds + 1 pair = 7
        assert!(perts.len() >= 6);
    }

    // --- Robustness tests ---

    #[test]
    fn robustness_score_clamp() {
        assert_eq!(RobustnessScore::new(-1.0).value(), 0.0);
        assert_eq!(RobustnessScore::new(2.0).value(), 1.0);
    }

    #[test]
    fn robustness_score_classification() {
        assert!(RobustnessScore::new(0.95).is_robust());
        assert!(!RobustnessScore::new(0.95).is_vulnerable());
        assert!(RobustnessScore::new(0.3).is_vulnerable());
        assert!(!RobustnessScore::new(0.3).is_robust());
    }

    #[test]
    fn robustness_report_constant_strategy() {
        let report = RobustnessReport::evaluate(always_positive, 2);
        assert!(report.score.value() >= 0.99); // Should be nearly 1.0 (can't flip constant)
        assert_eq!(report.states_flipped, 0);
    }

    // --- Attack suite tests ---

    #[test]
    fn attack_suite_gradient_free() {
        let suite = AttackSuite::gradient_free_only();
        let state = vec![Ternary::Positive, Ternary::Negative];
        let results = suite.attack_state(&state, sum_strategy);
        assert!(!results.is_empty());
    }

    #[test]
    fn attack_suite_random() {
        let suite = AttackSuite::random_only(42, 5);
        assert_eq!(suite.attacks.len(), 5);
    }

    #[test]
    fn attack_suite_targeted() {
        let suite = AttackSuite::default_suite();
        let state = vec![Ternary::Positive, Ternary::Negative]; // sum=0 -> Zero
        let results = suite.attack_state(&state, sum_strategy);
        // At least some should succeed
        assert!(results.iter().any(|r| r.success));
    }

    #[test]
    fn attack_success_rate() {
        let results = vec![
            AttackResult { kind: AttackKind::GradientFree, success: true, environment: None, queries: 5 },
            AttackResult { kind: AttackKind::GradientFree, success: false, environment: None, queries: 3 },
        ];
        assert_eq!(AttackSuite::success_rate(&results), 0.5);
    }

    // --- Defense report tests ---

    #[test]
    fn defense_report_sum_strategy() {
        let report = DefenseReport::analyze(sum_strategy, 2);
        assert!(report.vulnerability_count() > 0);
        assert!(report.dimension == 2);
        assert!(report.most_vulnerable_position().is_some());
    }

    #[test]
    fn defense_report_summary() {
        let report = DefenseReport::analyze(first_element_strategy, 2);
        let summary = report.summary();
        assert!(summary.contains("Defense Report"));
        assert!(summary.contains("dim=2"));
    }

    // --- Training tests ---

    #[test]
    fn decision_table_from_strategy() {
        let states = all_states(2);
        let table = DecisionTable::from_strategy(sum_strategy, &states);
        assert_eq!(table.len(), 9);
        let state = vec![Ternary::Positive, Ternary::Positive];
        assert_eq!(table.decide(&state), Ternary::Positive);
    }

    #[test]
    fn decision_table_set() {
        let states = all_states(1);
        let mut table = DecisionTable::from_strategy(sum_strategy, &states);
        let state = vec![Ternary::Zero];
        assert_eq!(table.decide(&state), Ternary::Zero);
        table.set(&state, Ternary::Positive);
        assert_eq!(table.decide(&state), Ternary::Positive);
    }

    #[test]
    fn adversarial_training_improves_robustness() {
        let config = TrainingConfig {
            rounds: 20,
            budget: 2,
            target_score: 0.95,
        };
        let training = AdversarialTraining::new(config);
        let (_table, logs) = training.train(first_element_strategy, 2);
        // Training should have produced logs
        assert!(!logs.is_empty());
        // Final score should be >= initial score
        if logs.len() >= 2 {
            assert!(logs.last().unwrap().score.value() >= logs.first().unwrap().score.value());
        }
    }
}
