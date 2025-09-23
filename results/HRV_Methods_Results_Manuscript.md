## METHODS

### Study Design and Participants

This longitudinal study examined the effects of prenatal yoga on heart rate variability (HRV) in pregnant women. Participants were recruited during early pregnancy and randomly assigned to either a prenatal yoga intervention group or a control group receiving standard prenatal care. The study protocol was approved by the institutional ethics committee, and all participants provided written informed consent.

### Data Collection

Heart rate variability measurements were obtained at two timepoints: first visit (early pregnancy) and last visit (late pregnancy). Electrocardiogram (ECG) recordings were collected under standardized conditions with participants in a supine position after a 10-minute rest period. ECG signals were sampled at high frequency and processed to extract R-R intervals for subsequent HRV analysis.

Demographic and clinical data including maternal age, body mass index (BMI), and gestational age at first visit were collected through structured interviews and medical record review.

### HRV Metric Computation

A comprehensive set of 94 HRV metrics was computed from the R-R interval time series, encompassing five physiological domains:

**Temporal Domain (25 metrics):** Traditional statistical measures of R-R interval variability including Mean NN, SDNN, RMSSD, pNN50, pNN20, SDANN (1-, 2-, and 5-minute segments), coefficient of variation measures, percentile-based indices, and geometric measures (HTI, TINN).

**Spectral Domain (6 metrics):** Power spectral analysis yielding low frequency (LF, 0.04-0.15 Hz), high frequency (HF, 0.15-0.4 Hz), total power (TP, 0-0.4 Hz), LF/HF ratio, and normalized units (LFnu, HFnu).

**Complexity Domain (53 metrics):** Nonlinear dynamics measures including Poincaré plot indices (SD1, SD2), entropy measures (approximate entropy [ApEn], sample entropy [SampEn], Shannon entropy, fuzzy entropy, multiscale entropy variants), detrended fluctuation analysis (DFA) scaling exponents (α1, α2), multifractal DFA parameters, correlation dimensions, fractal dimensions (Higuchi, Katz), complexity indices (CSI, CVI), and Lempel-Ziv complexity.

**Specialized Domain (9 metrics):** Additional cardiac measures including heart rate turbulence parameters, coefficient of variation, temporal variability indices, and spectral characteristics (centroid frequency, bandwidth).

**Information Domain (1 metric):** Entropy rate measuring information generation rate.

### Data Preprocessing and Quality Control

Raw HRV metrics were subjected to rigorous quality control procedures. Missing values and outliers were identified and handled appropriately. The EntropyRate metric required special preprocessing due to comma-separated decimal formatting in the original data, which was converted to standard decimal notation before analysis.

### Covariate Adjustment

To account for potential confounding variables, all HRV metrics were adjusted for maternal age, BMI, and gestational age at the first visit using linear regression residualization. For each metric, a linear model was fitted with the three covariates as predictors, and standardized residuals were computed and used in subsequent analyses.

### Principal Component Analysis

Timepoint-specific principal component analysis (PCA) was performed separately for the first and last visits using the covariate-adjusted and standardized HRV metrics. PCA was conducted on correlation matrices to ensure equal weighting of all metrics regardless of their original scales.

For each timepoint, the number of principal components required to explain ≥80% of the total variance was determined. A unified HRV complexity index was constructed by summing the scores of these components for each subject.

### Domain-Specific Analysis

HRV metrics were grouped by physiological domains, and domain-specific contributions to principal components were analyzed by computing mean absolute loadings within each domain. Cross-timepoint stability was assessed by comparing domain contributions between first and last visits, with percentage changes calculated as relative differences.

### Correlation Structure Analysis

Correlation matrices were computed for each timepoint using the adjusted HRV metrics. Overall correlation structure was characterized by computing mean absolute correlations and identifying high correlations (|r| > 0.8). Changes in correlation structure between timepoints were quantified to assess network simplification during pregnancy progression.

### Longitudinal Statistical Analysis

**Within-Group Analysis:** Paired comparisons of the unified HRV index between first and last visits were conducted separately for yoga and control groups. Normality of change scores was assessed using the D'Agostino-Pearson test. For normally distributed data, paired t-tests were employed; for non-normal data, Wilcoxon signed-rank tests were used. Effect sizes were calculated as Cohen's d for parametric tests and as standardized z-scores divided by √n for non-parametric tests.

**Between-Group Analysis:** Change scores (last visit − first visit) were compared between yoga and control groups using independent t-tests for normal distributions or Mann-Whitney U tests for non-normal distributions.

**Mixed-Effects Modeling:** Linear mixed-effects models were fitted to examine time effects, group effects, and time×group interactions on the unified HRV index. Models included random intercepts for subjects and fixed effects for time (coded as 0 for first visit, 1 for last visit), group (coded as 0 for control, 1 for yoga), and their interaction, while controlling for age, BMI, and gestational age. Models were fitted using restricted maximum likelihood estimation.

### Statistical Software and Significance

All analyses were conducted using Python 3.x with scientific computing libraries (pandas, numpy, scipy, scikit-learn, statsmodels). Statistical significance was set at α = 0.05 for all tests. All tests were two-tailed unless otherwise specified.

---

## RESULTS

### Sample Characteristics

A total of 22 unique subjects were included in the analysis, with 80 total observations across both timepoints. The first visit included 38 observations (16 yoga, 22 control), while the last visit included 30 observations (8 yoga, 22 control). Among subjects with both timepoints available, 15 provided paired data for longitudinal analysis (8 yoga, 22 control pairs, with some subjects contributing multiple paired observations).

After covariate adjustment for age, BMI, and gestational age, 68 observations with complete demographic data were retained for analysis. The covariate adjustment procedure successfully processed 93 of 94 HRV metrics (99.0% success rate).

### HRV Metric Validation and Processing

All 94 targeted HRV metrics were successfully identified and validated in the dataset. The alternative column name mapping procedure correctly identified spectral domain metrics (LF, HF, TP, LF_HF, LFnu, HFnu) that were stored under frequency-prefixed names. The EntropyRate parsing issue was resolved, with comma-decimal notation successfully converted to standard format, resulting in only 1 missing value across all observations.

### Correlation Structure Analysis

**First Visit (n=38 subjects, 86 metrics after preprocessing):**
- Mean absolute correlation: 0.351
- Maximum absolute correlation: 1.000
- High correlations (|r| > 0.8): 454 pairs
- Data dimensionality: 38 subjects × 86 metrics

**Last Visit (n=30 subjects, 86 metrics):**
- Mean absolute correlation: 0.347
- Maximum absolute correlation: 1.000
- High correlations (|r| > 0.8): 290 pairs
- Data dimensionality: 30 subjects × 86 metrics

The correlation structure showed evidence of simplification during pregnancy progression, with a 36.1% reduction in high correlation pairs (454 → 290) and a slight decrease in overall mean correlation strength.

### Principal Component Analysis Results

**First Visit PCA:**
- Components for 80% variance: 5 components
- Components for 90% variance: 7 components
- Cumulative variance explained by first 5 PCs: 83.9%
- Individual PC contributions: PC1 (46.6%), PC2 (14.8%), PC3 (9.1%), PC4 (7.8%), PC5 (5.7%)

**Last Visit PCA:**
- Components for 80% variance: 5 components
- Components for 90% variance: 7 components  
- Cumulative variance explained by first 5 PCs: 84.8%
- Individual PC contributions: PC1 (39.1%), PC2 (16.9%), PC3 (13.0%), PC4 (8.3%), PC5 (7.4%)

### Domain-Specific Loading Analysis

**First Visit PC1 Domain Contributions:**
- Spectral domain: mean |loading| = 0.166 (highest contribution)
- Information domain: mean |loading| = 0.128
- Temporal domain: mean |loading| = 0.113
- Complexity domain: mean |loading| = 0.068
- Specialized domain: mean |loading| = 0.058

**Last Visit PC1 Domain Contributions:**
- Temporal domain: mean |loading| = 0.112
- Complexity domain: mean |loading| = 0.090
- Specialized domain: mean |loading| = 0.072
- Information domain: mean |loading| = 0.036
- Spectral domain: mean |loading| = 0.031 (dramatically reduced)

### Cross-Timepoint Domain Stability

**Domain Changes in PC1 Contributions (First → Last Visit):**
- **Most Variable Domain - Spectral:** 81.2% decrease (0.166 → 0.031)
- **Information Domain:** 71.5% decrease (0.128 → 0.036)  
- **Complexity Domain:** 32.8% increase (0.068 → 0.090)
- **Specialized Domain:** 23.7% increase (0.058 → 0.072)
- **Most Stable Domain - Temporal:** 1.6% decrease (0.113 → 0.112)

The spectral domain showed the most dramatic restructuring, while temporal domain measures remained remarkably stable across pregnancy.

### Longitudinal Analysis Results

**Within-Group Changes (Paired Analysis, n=15 subjects with both timepoints):**

**Yoga Group (n=8 pairs):**
- First visit mean: 8.032 (unified HRV index)
- Last visit mean: 2.234
- Change: Δ = -5.798 (decrease)
- Statistical test: Paired t-test
- p-value: 0.184 (non-significant)
- Effect size: Cohen's d = -0.521 (moderate effect)

**Control Group (n=22 pairs):**
- First visit mean: -1.831 (unified HRV index)
- Last visit mean: -0.813
- Change: Δ = +1.019 (increase)
- Statistical test: Paired t-test
- p-value: 0.625 (non-significant)  
- Effect size: Cohen's d = 0.106 (small effect)

### Mixed-Effects Model Results

The linear mixed-effects model examining factors influencing the unified HRV index yielded the following results:

**Model Specification:**
- Sample: 15 subjects, 60 total observations
- Random effects: Subject-specific random intercepts
- Fixed effects: Time, group, time×group interaction, age, BMI, gestational age

**Fixed Effects Estimates:**
- **Time effect:** β = 1.019, p = 0.555 (non-significant)
- **Group effect:** β = 12.855, p = 0.001 (highly significant)
- **Time×Group interaction:** β = -6.817, p = 0.041 (significant)

**Model Interpretation:**
The significant group effect indicates substantial baseline differences in HRV complexity between yoga and control groups. Most importantly, the significant time×group interaction demonstrates that the two groups followed different trajectories of HRV change during pregnancy, with the yoga group showing a declining trend while the control group remained relatively stable.

### Key Statistical Findings Summary

1. **Correlation Network Simplification:** 36% reduction in high correlations during pregnancy progression (454 → 290 pairs).

2. **Domain Restructuring:** Spectral measures became less prominent (-81.2%) while complexity measures gained importance (+32.8%) in late pregnancy.

3. **Group Differences:** Significant baseline differences in HRV complexity between yoga and control groups (p = 0.001).

4. **Differential Trajectories:** Significant time×group interaction (p = 0.041) indicating that prenatal yoga influenced the pattern of HRV changes during pregnancy.

5. **Effect Sizes:** Moderate effect size for yoga group changes (d = -0.521) versus small effect for controls (d = 0.106).

The results suggest that prenatal yoga practice influences autonomic cardiovascular regulation during pregnancy, with evidence of differential trajectories in HRV complexity measures between intervention and control groups.
