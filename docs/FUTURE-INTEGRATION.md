# Future Integration: ternary-adversarial

## Current State
Provides adversarial training and defense for ternary strategies: `Adversary` generates perturbations to flip decisions, `AttackSuite` tests multiple attack kinds, `DefenseReport` analyzes vulnerabilities by position, `RobustnessReport` scores overall strategy robustness, and `AdversarialTraining` hardens strategies via iterative attack-defense cycles.

## Integration Opportunities

### With ternary-cell (Room Security Hardening)
ternary-cell's decision function maps cell state to ternary output. ternary-adversarial's `Adversary` finds minimal perturbations that flip cell decisions — revealing which cells are vulnerable to manipulation. `DefenseReport::analyze()` identifies vulnerable positions in the cell grid. `AdversarialTraining` hardens cells by exposing them to attacks during training, so they become robust against adversarial inputs in production.

### With ternary-codes (Adversarial Error Correction)
ternary-codes' error-correcting codes protect against random noise. ternary-adversarial protects against targeted perturbations. Together: encode cell state with ternary-codes' Hamming encoding, then adversarially test whether the encoding can be broken. The code distance sets a lower bound on adversarial robustness — an adversary needs at least d perturbations to overcome distance-d encoding.

### With ternary-steganography (Adversarial Steganography Detection)
ternary-steganography hides data in trit sequences. ternary-adversarial trains detectors to find hidden data. The `AttackSuite` can include steganographic detection as an attack kind: try to detect (and remove) hidden data in incoming messages. Conversely, ternary-steganography can train to resist detection — an adversarial arms race that improves both hiding and finding.

## Potential in Mature Systems
In room-as-codespace, adversarial security is critical. Rooms receive inputs from external sources (user queries, sensor data, other rooms). ternary-adversarial provides the defense layer: every incoming message passes through a robustness check (`RobustnessReport::score()`). Messages that are too close to adversarial perturbations are flagged. Rooms are periodically adversarially trained — PLATO runs attack suites against room decision functions and hardens vulnerable positions.

## Cross-Pollination Ideas
- **ternary-games**: Adversarial game theory — attacker and defender play a zero-sum ternary game. Nash equilibrium strategies are optimal for both.
- **ternary-noise**: Adversarial noise is worst-case noise; random noise is average-case. The gap between them measures strategy sensitivity.
- **ternary-ensemble**: Ensemble adversarial training — attack the ensemble, not individual agents. Ensembles are typically more robust than individuals.

## Dependencies for Next Steps
- Add `CellDefenseReport` mapping ternary-adversarial analysis to cell grid positions
- Integrate adversarial robustness scoring into ternary-cell's conservation phase
- Build adversarial testing pipeline for PLATO room validation
- Define `AdversarialSkillCheck` for construct-core skill validation
