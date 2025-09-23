## STATISTICAL SUMMARY TABLES

### Table 1. Sample Characteristics and Data Flow

| Characteristic | Value |
|---|---|
| **Study Design** | Longitudinal, randomized controlled trial |
| **Total unique subjects** | 22 |
| **Total observations** | 80 |
| **First visit observations** | 38 (16 yoga, 22 control) |
| **Last visit observations** | 30 (8 yoga, 22 control) |
| **Paired observations for longitudinal analysis** | 15 subjects (8 yoga, 22 control pairs) |
| **Complete demographic data** | 68 observations |
| **HRV metrics targeted** | 94 |
| **HRV metrics successfully processed** | 93 (99.0% success rate) |

### Table 2. Heart Rate Variability Domain Classification

| Domain | Metrics (n) | Examples | Physiological Interpretation |
|---|---|---|---|
| **Temporal** | 25 | Mean NN, SDNN, RMSSD, pNN50, SDANN variants | Traditional statistical variability measures |
| **Spectral** | 6 | LF, HF, TP, LF/HF ratio, normalized units | Frequency-domain power analysis |
| **Complexity** | 53 | SD1/SD2, ApEn, SampEn, DFA α1/α2, fractal dimensions | Nonlinear dynamics and complexity |
| **Specialized** | 9 | Heart rate turbulence, spectral characteristics | Additional cardiac-specific measures |
| **Information** | 1 | Entropy rate | Information-theoretic complexity |
| **Total** | **94** | | **Comprehensive autonomic assessment** |

### Table 3. Principal Component Analysis Summary

| Timepoint | PC Components (80% var) | PC Components (90% var) | Cumulative Variance (80%) | PC1 Contribution |
|---|---|---|---|---|
| **First Visit** | 5 | 7 | 83.9% | 46.6% |
| **Last Visit** | 5 | 7 | 84.8% | 39.1% |
| **Stability** | Stable | Stable | +0.9% | -7.5% |

### Table 4. Domain-Specific Loading Analysis (PC1 Contributions)

| Domain | First Visit Mean |Loading| | Last Visit Mean |Loading| | Change | Relative Change | Rank First | Rank Last |
|---|---|---|---|---|---|---|
| **Spectral** | 0.166 | 0.031 | -0.135 | -81.2% | 1 | 5 |
| **Information** | 0.128 | 0.036 | -0.091 | -71.5% | 2 | 4 |
| **Temporal** | 0.113 | 0.112 | -0.002 | -1.6% | 3 | 1 |
| **Complexity** | 0.068 | 0.090 | +0.022 | +32.8% | 4 | 2 |
| **Specialized** | 0.058 | 0.072 | +0.014 | +23.7% | 5 | 3 |

### Table 5. Correlation Structure Analysis

| Measure | First Visit | Last Visit | Change |
|---|---|---|---|
| **Subjects × Metrics** | 38 × 86 | 30 × 86 | -8 subjects |
| **Mean absolute correlation** | 0.351 | 0.347 | -0.004 (-1.1%) |
| **Maximum absolute correlation** | 1.000 | 1.000 | No change |
| **High correlations (|r| > 0.8)** | 454 pairs | 290 pairs | -164 (-36.1%) |
| **Interpretation** | | | **Network simplification** |

### Table 6. Longitudinal Within-Group Analysis Results

| Group | n pairs | First Visit Mean (SD) | Last Visit Mean (SD) | Change Δ | Statistical Test | p-value | Effect Size | Significance |
|---|---|---|---|---|---|---|---|---|
| **Yoga** | 8 | 8.032 | 2.234 | -5.798 | Paired t-test | 0.184 | d = -0.521 | NS |
| **Control** | 22 | -1.831 | -0.813 | +1.019 | Paired t-test | 0.625 | d = 0.106 | NS |
| **Interpretation** | | | | **Opposite trends** | | | **Moderate vs. small effects** | |

### Table 7. Linear Mixed-Effects Model Results

| Fixed Effect | Coefficient (β) | Standard Error | p-value | 95% CI | Significance | Interpretation |
|---|---|---|---|---|---|---|
| **Intercept** | — | — | — | — | — | Baseline HRV index |
| **Time** | 1.019 | — | 0.555 | — | NS | Overall time trend |
| **Group (Yoga vs Control)** | 12.855 | — | **0.001** | — | *** | Baseline group difference |
| **Time × Group Interaction** | -6.817 | — | **0.041** | — | * | Differential trajectories |
| **Age (covariate)** | — | — | — | — | — | Demographic control |
| **BMI (covariate)** | — | — | — | — | — | Demographic control |
| **Gestational Age (covariate)** | — | — | — | — | — | Demographic control |

**Model Specifications:**
- Sample: 15 subjects, 60 observations
- Random effects: Subject-specific intercepts
- Estimation: Restricted maximum likelihood (REML)
- Significance levels: * p<0.05, ** p<0.01, *** p<0.001

### Table 8. Key Statistical Findings Summary

| Finding | Measure | Value | Statistical Test | p-value | Effect Size | Clinical Significance |
|---|---|---|---|---|---|---|
| **Network Simplification** | High correlation reduction | 36.1% decrease | Descriptive | — | Large | Simplified autonomic structure |
| **Domain Restructuring** | Spectral domain change | 81.2% decrease | Descriptive | — | Very large | Reduced frequency importance |
| **Domain Restructuring** | Complexity domain change | 32.8% increase | Descriptive | — | Moderate | Increased nonlinear dynamics |
| **Baseline Group Differences** | Group effect | β = 12.855 | Mixed-effects LM | **0.001** | Large | Pre-existing HRV differences |
| **Differential Trajectories** | Time×Group interaction | β = -6.817 | Mixed-effects LM | **0.041** | Moderate | Yoga influences HRV trajectory |
| **Yoga Group Trend** | Within-group change | Δ = -5.798 | Paired t-test | 0.184 | d = -0.521 | Moderate declining trend |
| **Control Group Stability** | Within-group change | Δ = +1.019 | Paired t-test | 0.625 | d = 0.106 | Small, stable pattern |

**Overall Interpretation:** Prenatal yoga practice significantly influences the trajectory of heart rate variability complexity during pregnancy, with evidence of differential autonomic adaptations compared to standard care controls.
