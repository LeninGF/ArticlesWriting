# Reviewer Revision Session — 2026-05-25

**Paper:** MDPI Diagnostics — Preeclampsia ML/HIA  
**Branch:** `reviewer-revisions`  
**File:** `preeclampsia-ml-hia.tex` / `bibliography.bib`

---

## Session Commits

| Commit | Message | Items covered |
|--------|---------|---------------|
| `8df594b` | R1.1–R1.4 + HPO acronym (prior session) | R1.1, R1.2, R1.3, R1.4 |
| `850c974` | R1.5+R1.6: HPO random search table; Canvas ensemble equation and artifact-verified composition in Results | R1.5, R1.6 |
| `dacac34` | R1.7: synthetic data methodology; FullPIERS 10% threshold cited (long2025, teichmann2025, elongi2025); SpO2/chest pain evidence (millman2011, elongi2025); em-dash prose cleanup | R1 M&M synthetic, R1 Results FullPIERS, R1 Discussion, R2 SpO2/chest pain |
| `631e39a` | R1/R2 Intro: explicit FullPIERS variable list; synthetic dataset equation reference in body paragraph | R1 Intro, R2 Intro |
| `2cb5f03` | R1/R2 Intro background: clinical PE definition (ISSHP/Brown2018), global prevalence (Vera-Ponce 2025 meta-analysis), ML citations strengthened (Darsareh2024, Rahman2023); fix severe/severe prose repetition | R1 Intro, R2 Intro |
| `4b0796f` | R2 Intro: expand clinical PE definition (ISSHP/Brown2018 three-branch criterion, previously normotensive, SBP/DBP abbreviations, risk factors, labor/postpartum onset, early/late distinction, severe features); integrate Vera-Ponce 2025 | R2 Intro |
| `3f763eb` | R2 font uniformization: remove `\texttt{}` from algorithm names (LightGBM, XGBoost, CatBoost, LR, MLP, RF, Weighted Ensemble) throughout Abstract and body; retain monospace on code identifiers | R2 Manuscript-wide |
| `7e24bbe` | R2 software versions: NumPy>=2.0.2, pandas>=2.3.1 (synth-data); scikit-learn>=1.6.1 (C-AutoML); AWS city (SageMaker Canvas); SHAP>=0.49.0 (Appendix A); MLflow>=2.14.0 (Appendix B) | R2 M&M software versions |
| `d1593cb` | R2 software versions: add Python>=3.9 to synthetic-data section (line 984) | R2 M&M software versions |
| `f7c0d1d` | R2 M&M: add footer note to tab:distributions explaining N, U, Gamma, Bernoulli notation | R2 M&M Table 5 footer |
| `46d90ca` | R2 M&M: add p-values to tab:desc_stats (Mann-Whitney U, two-sided) and tab:variable_distribution (Pearson chi-squared); add \usepackage{multirow} to preamble; create 05-statistical-tests-p-values.ipynb | R2 M&M Table 4 p-values |
| `657904e` | R2 M&M+Discussion: PE prevalence explanation (16.11% record-level, 25.1% patient-level), HGOIA OB/GYN specialty + MSP zonification rationale, performance metric implications, inter-group table refs; Limitations: chronic HTN ICD-10 selection (O10 excluded, O11 included), PE sub-categories as future work | R2 M&M PE incidence/prevalence, inter-group comparisons, chronic HTN, PE sub-categories |
| `bf4845f` | R2 Discussion: add strengths sentence to Limitations paragraph (real-world dataset, C-AutoML reproducibility with MLflow, SHAP, Canvas benchmarking) | R2 Discussion strengths/limitations |
| `0e6893b` | R2 Discussion: add comparison table (`tab:ml_pe_comparison`) and paragraph contextualising HGOIA results against similar-framework studies (Li2021, Liu2022, Zeng2024, AdvancingML2025, BiasFree2024, JMIR2026 review); bridge paragraph in Related Works | R2 Discussion compare with similar frameworks |
| `70d91f3` | R2 references: verify and clean up new comparison-study bibliography entries (DOIs confirmed via Google Scholar) | R2 Discussion references |
| `13a6f07` | R2 Intro: split proteinuria sentence for readability (two distinct clinical points in one long sentence) | R2 Intro prose |

---

# Reviewer Revision Session — 2026-05-28

**Branch:** `reviewer-revisions`  
**Status:** No commits pushed (user requested hold on commits)

---

## Summary of Changes

### 1. Bridge Paragraph Removed (Related Works)
- Removed the `\added[id=R2]{...}` bridge paragraph at lines 596–607 that introduced comparison studies (Liu2022, Zeng2024, AdvancingML2025, BiasFree2024, JMIR2026). Instead of salvaging citations into Related Works, the citations remain only in the Discussion comparison table.
- **Motivation:** Paragraph was redundant with Discussion content.

### 2. PE Prevalence Paragraph Rewritten (M&M — Statistical Characteristics)
- Replaced verbose multi-factor explanation with a concise version:
  - Two valid factors: (i) ICD-10 case-enriched selection design, (ii) HGOIA referral centre effect
  - Removed flawed "record-level inflates prevalence" argument (actually, record-level *deflates* vs patient-level: 16.11% vs 25.1%)
  - Added p-value interpretation: all four predictors show highly significant differences (Mann-Whitney $U$ and Pearson $\chi^2$, $p < 0.001$)

### 3. R12 Combined Author Defined
- Added `\definechangesauthor[name={Reviewers~1\&2},color=violet]{R12}` to preamble
- Fixed invalid `\added[id={R1,R2}]` → `\added[id=R12]` in the synthetic data paragraph

### 4. R12 Synthetic Data Paragraph Improved
- Fixed connective: "are instead retained…reflecting their role" → "are nonetheless retained because"
- Dropped redundant closing clause about `PE_{FullPIERS}` being "only used for secondary risk stratification analyses"

### 5. Notation Uniformized Across Document
- `\mathrm{PE_{FullPIERS}}` → `\mathrm{PE}_{\mathrm{FullPIERS}}` (3 instances)
- `\mathrm{PE}_{MSP}` → `\mathrm{PE}_{\mathrm{MSP}}` (2 instances: equation + prose)

### 6. Results Section — Option A Merge
- Merged the `\added[id=R1]` block (explaining why all models achieve F1≈1.0) with the prose paragraph into one clean `\added` block
- Fixed `\mathrm{PE}\_{\mathrm{MSP}}` notation bug (backslash-underscore produced literal underscore)
- Removed redundant "Therefore, perfect performance should be interpreted as…" sentence from prose paragraph (now covered by `\added` block)
- Deleted commented-out original R1 block (dead code)

### 7. Fix A — Introduction (Critical)
- Corrected the description of synthetic dataset construction: previously stated "outcome labels were derived from the FullPIERS predicted risk scores" — this conflated PE_MSP (diagnostic, MSP rule) with PE_FullPIERS (prognostic, FullPIERS). Now correctly states that features follow FullPIERS distributions but the diagnostic label uses the MSP guideline.

### 8. Fix H — Discussion (Minor)
- "adverse outcome" → "adverse **outcomes**" (plural, matching FullPIERS literature)

### 9. Grammar & Style Fixes (4 issues)
| Fix | Line | Change |
|-----|------|--------|
| Notation | 1070 | `\mathrm{PE}_{MSP}` → `\mathrm{PE}_{\mathrm{MSP}}` |
| Eq. lead-in | 620 | `\eqref{...}` → `Equation~\eqref{...}` |
| Punctuation | 1834 | `; which` → `, which` (ungrammatical semicolon) |
| Clarity | 884 | "85.13%" ambiguous → clarified as 36–49 age group with table reference |

---

## Remaining (Post-Session)
- **Figure 2 DPI** (≥300 dpi) — user to handle manually
- **Compilation status:** Clean — no errors, 38-page PDF produced

## ✅ Completed This Session

### Reviewer 1 — Materials and Methods

#### R1.5 — HPO Explanation (C-AutoML)
- **What was done:** Added `\added[id=R1]` paragraph in the C-AutoML training section explaining:
  - Strategy: *random search* (`ParameterSampler`, seed `42 + 100i + m`)
  - Justification over grid search (Bergstra & Bengio 2012)
  - Cross-reference to `tab:hpo-search-spaces`
- **New table:** `tab:hpo-search-spaces` — all 6 model families (LR, RF, MLP, XGBoost, LightGBM, CatBoost) with full discrete candidate sets, taken directly from `code/autopilot_train.py`
- **New bib entry:** `bergstra2012random` (JMLR 2012, vol. 13, pp. 281–305)
- **Location:** ~lines 1156–1233
- **Cover letter text:** *"We added a paragraph in the C-AutoML training section explaining the hyperparameter optimization strategy: randomized search (Bergstra & Bengio, JMLR 2012) over discrete candidate sets, with a unique random seed per trial (42 + 100i + m) ensuring reproducibility. Table [tab:hpo-search-spaces] lists the complete search space for all six model families (LR, RF, MLP, XGBoost, LightGBM, CatBoost). Randomized search was chosen over grid search because it explores a higher-dimensional region of the hyperparameter space per unit of compute budget, as demonstrated by Bergstra & Bengio (2012). (commit 850c974)"*

#### R1.6 — Canvas Model Run Settings + Weighted Ensemble Structure
- **What was done:** Rewrote the Canvas paragraph (Methods) with `\added[id=R1]` markup:
  - Optimization target: F1 maximization
  - 10 Layer-1 bagged candidates listed by name
  - Two-layer AutoGluon architecture described
  - Greedy forward ensemble selection algorithm (Caruana et al. 2004)
  - **New equation** `eq:ensemble-prediction`: $\hat{p}_\mathrm{ens}(\mathbf{x}) = \sum_k w_k \hat{p}_k(\mathbf{x}),\ w_k = n_k/S$
- **Results section updated:** Replaced imprecise "LightGBM, MLP, RF" with artifact-verified names (LightGBM, Random Forest (Entropy), NeuralNetFastAI), weights $\tfrac{1}{3}$ each, referencing `eq:ensemble-prediction`
- **Moved:** Job-specific artifact details (which models selected, which excluded) from Methods → Results (`sec:performance-aws-sage-maker-canvas`), per paper structure conventions
- **Removed:** Canvas job ID (`Canvas1774655537951`) — deemed irrelevant for the paper
- **New bib entries:** `agtabular2020` (AutoGluon-Tabular, Erickson et al. 2020), `caruana2004ensemble` (ICML 2004, doi:10.1145/1015330.1015432)
- **Cover letter text:** *"The Amazon SageMaker Canvas section was rewritten to specify: (i) optimization target — F1-score maximization; (ii) the 10 Layer-1 bagged candidate models evaluated (LightGBM, LightGBMXT, XGBoost, CatBoost, Random Forest Gini, Random Forest Entropy, Extra Trees Gini, Extra Trees Entropy, NeuralNetFastAI, NeuralNetTorch); (iii) the two-layer AutoGluon greedy forward ensemble selection algorithm (Caruana et al. 2004, ICML); and (iv) the final ensemble composition verified directly from the binary model artifact: LightGBM, Random Forest (Entropy), and NeuralNetFastAI, each with weight 1/3. The ensemble prediction formula is given as Equation (eq:ensemble-prediction): $\\hat{p}_\\mathrm{ens}(\\mathbf{x}) = \\sum_k w_k \\hat{p}_k(\\mathbf{x})$. (commit 850c974)"*

### Canvas Model — Verified From Binary Artifact
The model artifact was downloaded to `code/canvas-model/` and inspected using `strings` + raw byte decoding (AutoGluon not installed). Confirmed:
- **Method:** `autogluon.core.models.greedy_ensemble.ensemble_selection` (Caruana 2004)
- **Ensemble size:** `S = 3`
- **3 selected:** `LightGBM_BAG_L1`, `RandomForestEntr_BAG_L1`, `NeuralNetFastAI_BAG_L1`
- **Weights:** `0.3333...` each (decoded from IEEE 754 bytes `55 55 55 55 55 55 d5 3f`)
- **Score function:** `f1_score`
- **10 candidates trained:** LightGBM, LightGBMXT, XGBoost, CatBoost, RF(Gini), RF(Entropy), ExtraTrees(Gini), ExtraTrees(Entropy), NeuralNetFastAI, NeuralNetTorch

---

## ✅ Completed — Reviewer 1 (continued)

#### R1.1 — Dataset Characteristics Table (`8df594b`)
- **What was added:** `tab:dataset-characteristics` (Table 1) with patient counts at each preprocessing stage, class distribution (PE-positive / PE-negative) for both HGOIA and synthetic cohorts, missing-data rate and imputation strategy (median for numeric, mode for categorical).
- Also updated `tab:desc_stats` caption and `fig:statistics` caption to clarify that statistics reflect the pre-Canvas HGOIA cohort (3 571 PE− / 674 PE+).
- **Cover letter text:** *"We added Table 1 (Dataset Characteristics) reporting patient counts at each preprocessing stage, the class distribution (674 PE-positive out of 4 245 HGOIA records, 27.7% PE-positive in the synthetic cohort), and the missing-data rate with imputation strategy (median imputation for numeric features; mode imputation for categorical features via scikit-learn ColumnTransformer). All descriptive statistics in Table 2 now include an explicit annotation indicating they reflect the pre-Canvas HGOIA cohort. (commit 8df594b)"*

#### R1.2 — C-AutoML Pipeline Flowchart (`8df594b`)
- **What was added:** Figure 5 (`fig:c-automl-flowchart`) — a TikZ flowchart of the full C-AutoML pipeline, showing data ingestion → stratified train-validation split → HPO inner loop (6 model families × 10 iterations = 60 trials, randomized search via `ParameterSampler`) → permutation-importance feature selection → best-model output artifact.
- **Cover letter text:** *"We added Figure 5, a flowchart that makes the preprocessing, feature-selection, model-selection, and hyperparameter-optimization steps of the C-AutoML pipeline visually explicit. The figure shows the complete path from raw CSV input to the serialized best-model artifact, including the stratified split strategy and the 60-trial random-search loop across six model families. (commit 8df594b)"*

#### R1 — Introduction
- **Strengthened background** with ISSHP/Brown2018 clinical PE definition and Vera-Ponce 2025 meta-analysis
- Added ML-prediction citations (Darsareh2024, Rahman2023)
- Added FullPIERS variable list in Intro with explicit cross-reference
- Commits: `631e39a`, `2cb5f03`, `4b0796f`
- **Cover letter text:** *"The Introduction was expanded with: (i) a paragraph providing the ISSHP 2018 three-branch clinical definition of PE (Brown et al. 2018), covering previously normotensive criteria, SBP/DBP thresholds, risk factors, temporal onset (labor/postpartum), early/late onset distinction, and severe features; (ii) updated global prevalence (pooled 4.43%) and mortality figures from the 2025 meta-analysis by Vera-Ponce et al.; (iii) an explicit list of the FullPIERS predictor variables used to generate the synthetic dataset, with cross-reference to the Synthetic Data Generation section; and (iv) two additional ML PE-prediction references (Darsareh et al. 2024; Rahman et al. 2023) that strengthen the motivation for ML-based approaches. All added text is marked with \\added[id=R1/R2]{...}. (commits 631e39a, 2cb5f03, 4b0796f)"*

#### R1 — Materials & Methods: Synthetic Dataset
- Added `\added[id=R1]` paragraph in `sec:synth-data-gener` detailing:
  - FullPIERS risk threshold set at **10%** (not default 50%), with sensitivity rationale
  - All variables computationally generated; timing/repetition questions N/A
  - SpO₂ and chest pain justified as FullPIERS predictors (millman2011spo2, elongi2025fullpiers)
- New bib entries: `millman2011spo2`, `long2025fullpiers`, `teichmann2025fullpiers`, `elongi2025fullpiers`
- Commit: `dacac34`
- **Cover letter text:** *"We added a methodological paragraph in the Synthetic Data Generation section (marked \\added[id=R1]) explaining: (i) the FullPIERS risk threshold was set at 10% rather than the logistic-regression default of 50%, because at 50% the model would classify very few patients as high-risk in a low-prevalence context, defeating the purpose of a screening label; (ii) the justification for including SpO₂ and chest pain — both are predictor variables in the published FullPIERS equation (Millman et al. 2011; Elongi et al. 2025), and their inclusion mirrors that validated prognostic model rather than representing an ad-hoc variable selection; and (iii) all synthetic variable values are computationally generated from parametric distributions and are not clinically collected measurements, so questions of gestational timing and repeated measurement do not apply to this dataset component. (commit dacac34)"*

#### R1 — Results: FullPIERS / LR ~100% Performance
- Added `\added[id=R1]` block explaining:
  - All 6 model families achieve F1=1.000 in iterations 1–3 (structurally expected)
  - Labels assigned by deterministic threshold rule → linear separability guaranteed
  - By design for controlled feasibility testing, not overfitting in classical sense
- Commit: `dacac34`
- **Cover letter text:** *"We added a clarifying paragraph in the Results section explaining why all six model families achieve F1 = 1.000 on the synthetic dataset: the class labels are assigned by a deterministic threshold rule (the MSP equation and the FullPIERS 10% threshold) applied to the same feature values used as model inputs. This guarantees perfect linear separability in the training distribution by construction. The result demonstrates that the pipeline correctly learns the labeling rule — a necessary condition for controlled feasibility testing — and is not a manifestation of overfitting or data leakage in the classical sense, since no memorization of individual samples is required. This behavior is expected and acknowledged as a limitation of synthetic-label evaluation. (commit dacac34)"*

#### R1 — Discussion: Expanded
- Expanded limitations paragraph covering demographic-only dataset, single-center design, synthetic data limitations, and FullPIERS 10% threshold choice
- Commit: `dacac34`
- **Cover letter text:** *"The Discussion was expanded with an explicit limitations paragraph acknowledging: (i) only four demographic variables are available from the HGOIA dataset, and all clinical laboratory variables are synthetic — the model cannot currently be applied as a clinical diagnostic tool; (ii) the single-center, retrospective design limits external validity and generalizability to other populations; (iii) the synthetic dataset is generated from a parametric statistical model calibrated to FullPIERS coefficients and population-level distributions, not from prospective clinical measurements; and (iv) the 10% FullPIERS threshold was chosen to maximize sensitivity for a rare-event screening context and may not be appropriate for other clinical settings. (commits 8df594b, dacac34)"*

---

## ⬜ Remaining — Reviewer 1

*(All R1 items completed as of `4b0796f`)*

---

### Synthetic Dataset — Methodology Plan (from `code/04-synthetic-data-generation.ipynb`)

**What the notebook shows (to be reflected in the paper):**

| Aspect | Detail |
|--------|--------|
| Function | `generar_dataset_preeclampsia()` — Python, NumPy + pandas |
| N samples | n = 2000, fixed seed = 42 |
| Latent groups | Control (80%) / Risk (20%) — assigned before sampling |
| Variables | 10 features per sample; all synthetically drawn |
| Distributions | See `tab:distributions` (already in paper) |
| Non-negative enforcement | `np.clip(..., a_min=0)` on Proteinuria, Platelets, Creatinine, AST |
| Label 1 (MSP rule) | `Preeclampsia_MSP` — equation already in paper as `eq:preeclampsia-msp-rule` |
| Label 2 (FullPIERS) | `FullPIERS_Risk_Pct` — continuous probability %; threshold = **10%** (not default 50%) |
| FullPIERS coefficients | From Lancet 2011 paper (b0=2.68, b1=−5.41×10⁻², b2=1.23, b3=−2.71×10⁻², b4=2.07×10⁻¹, etc.) |
| FullPIERS binary label | `Preeclampsia_FullPIERS_Label` — 1 if FullPIERS_Risk_Pct ≥ 10% |
| Threshold rationale | 10% threshold chosen because the adverse outcome event is uncommon; 50% default would have very low sensitivity in a screening context |

**Paper additions needed:**

1. **Methods (`sec:synth-data-gener`)** — `\added[id=R1]`:
   - State explicitly that FullPIERS risk threshold was set at **10%** (not the logistic default 50%), with rationale: in low-prevalence prognostic models a 50% threshold is too conservative and would miss most high-risk patients
   - Add clarifying sentence that all variables are computationally generated, not clinically collected (`\added[id=R2]`)
   - Justify inclusion of SpO₂ and chest pain: these are FullPIERS predictors (cited in fullpiers_2011); they are part of the published prognostic model, not general illness markers in this context

2. **Results (`sec:synth-data-results`)** — `\added[id=R1]`:
   - Strengthen the existing "perfect performance expected" sentence with explicit explanation:
     - Labels were assigned by a **deterministic rule** (MSP equation) over the same features used for training → linear separability is guaranteed in the training distribution
     - This is by design for *controlled feasibility testing*, not a clinical claim
     - Not overfitting in the classical sense: the separation is structural, not due to memorization
     - FullPIERS-based label track shows the same behavior because the FullPIERS score is itself a linear combination of the features

3. **Discussion** — already has Limitations paragraph (R1.3) mentioning synthetic data; may need one additional sentence cross-referencing the 10% threshold choice

---

---

## ✅ Completed — Reviewer 2

#### R2 — Manuscript-wide: Font Uniformization (`3f763eb`)
- **What was changed:** Removed `\texttt{}` wrapper from all algorithm/model names used as prose throughout the entire manuscript (Abstract, Introduction, Related Work, Methods, Results, Discussion, figure captions):
  - LightGBM (21 + 4 = 25 occurrences), XGBoost (8), CatBoost (6), Weighted Ensemble (4+1), Logistic Regression (1), Multi-Layer Perceptron (1), Random Forest (1) — **42 total replacements**
- **What was kept in `\texttt{}`:** All legitimate code identifiers — `scikit-learn`, `ParameterSampler`, `sklearn`, `NumPy`, `pandas`, `autopilot_train.py`, `Pipeline`, `liblinear`, `lbfgs`, ICD-10 codes, Canvas artifact IDs
- **Why `\replaced{}{}` markup was NOT used:** Font formatting is a typographic correction, not a content change — the words on the page are identical before and after; only the rendering command changed. Using `\replaced` would have produced every algorithm name doubled (struck-through + underlined) 42 times throughout the document, making it unreadable. The `changes` package is for content edits only.
- **Cover letter note to include:** *"Algorithm names (LightGBM, XGBoost, CatBoost, Logistic Regression, Multi-Layer Perceptron, Random Forest, Weighted Ensemble) throughout the manuscript have been set in roman body font, consistent with medical journal conventions. Code and software identifiers (scikit-learn, ParameterSampler, etc.) retain monospace formatting as they refer to specific software objects. No content was altered (commit 3f763eb)."*

#### R2 — Introduction: References aligned with PE prediction/diagnosis guidelines
- **What was done:** Added `\added[id=R2]` block in Introduction para 2 providing the ISSHP 2018 three-branch clinical PE definition (Brown et al. 2018) and integrating the Vera-Ponce 2025 global meta-analysis (pooled prevalence 4.43%, mortality figures, risk factors, closing sentence).
- Para 1 also updated with `\added[id=R2]` to incorporate Vera-Ponce 2025 mortality figures and the word "leading" for PE among hypertensive disorders.
- Commits: `2cb5f03`, `4b0796f`
- **Cover letter text:** *"Following the reviewer's request to align the Introduction with PE prediction/diagnosis guidelines, we added a paragraph providing the ISSHP 2018 clinical definition of PE (Brown et al.) — covering the three diagnostic pathways, the previously-normotensive criterion, SBP/DBP thresholds (\u2265140/90 mmHg on two occasions), risk factors, labor and postpartum onset, early-onset vs. late-onset distinction, and severe features. We also integrated updated global prevalence and mortality figures from the 2025 meta-analysis by Vera-Ponce et al. (pooled prevalence 4.43%, leading cause of maternal mortality). Two recent ML PE-prediction references (Darsareh et al. 2024; Rahman et al. 2023) were added to strengthen the motivation for the proposed approach. All additions are marked with \\added[id=R2]{...}. (commits 2cb5f03, 4b0796f)"*

#### R2 — M&M: SpO2 and chest pain justified (`dacac34`)
- **What was done:** Added `\added[id=R2]` clarifying sentence in the Synthetic Data Generation section justifying the inclusion of SpO\u2082 and chest pain by citing their presence as explicit predictor variables in the published FullPIERS model (Millman et al. 2011; Elongi et al. 2025). The sentence explicitly states these variables are part of the validated FullPIERS equation, not general illness markers.
- Bib entries: `millman2011spo2`, `elongi2025fullpiers`
- **Cover letter text:** *"SpO\u2082 (oxygen saturation) and chest pain are retained because they are explicit predictor variables in the published FullPIERS prognostic equation (Millman et al. 2011; Elongi et al. 2025). Their presence in our synthetic dataset directly mirrors the validated model. A clarifying sentence citing these references has been added to the Synthetic Data Generation section with \\added[id=R2]{...} markup, making the clinical justification explicit. (commit dacac34)"*

##### R2 — Note on not re-running synthetic experiments without SpO₂/chest pain

- **Cover letter text (to include as additional justification):** *"We note that removing SpO₂ and chest pain from the synthetic dataset would not affect the reported results. The primary training labels (Preeclampsia\_MSP) are assigned by the deterministic MSP rule (Equation~[eq:preeclampsia-msp-rule]), which depends exclusively on gestational age, systolic and diastolic blood pressure, proteinuria, platelet count, creatinine, and AST. SpO₂ and chest pain are not label-determining features; they participate only in the secondary FullPIERS prognostic label. Consequently, all six model families would continue to achieve F1=1.000 on the synthetic track with or without these two variables, because the class boundary is structurally encoded in the remaining features. Re-running the full experimental pipeline (60 C-AutoML trials plus Canvas benchmarking) would therefore produce identical conclusions at substantial computational cost, and we respectfully consider that the existing justification — both variables are explicit components of the published FullPIERS equation — is sufficient to support their retention."*

#### R2 — M&M: Software versions + city of manufacturer (`7e24bbe`, `d1593cb`)
- **What was done:** Added inline version numbers throughout the manuscript for all software tools, and city/organization for commercial services only (per clinical journal convention — city applies to proprietary instruments and platforms, not to open-source software libraries).
  - **Synthetic Data Generation** (line ~984): `Python~($\ge$~3.9)`, `\texttt{NumPy}~($\ge$~2.0.2)`, `\texttt{pandas}~($\ge$~2.3.1)`
  - **C-AutoML section** (~line 1248): `\texttt{scikit-learn}~($\ge$~1.6.1)~\cite{sklearn-api}` added (citation already existed elsewhere; no duplicate introduced)
  - **Amazon SageMaker Canvas paragraph** (~line 1196): added `(Amazon Web Services, Inc., Seattle, WA, USA)` immediately after the product name — commercial service, city is applicable
  - **Appendix A** (optional packages): added `\texttt{shap}~$\ge$~0.49.0` with `\cite{lundberg2017shap}` — was missing from the list
  - **Appendix B** (MLOPs section): added `($\ge$~2.14.0)` inline with the existing `MLflow~\cite{mlflow_2018}` reference
- **Why city is given only for Amazon SageMaker Canvas:** The reviewer's request for "city of manufacturer" follows the clinical/biomedical convention for identifying proprietary instruments and commercial software platforms. Open-source libraries (Python, NumPy, pandas, scikit-learn, SHAP, MLflow) are community-developed tools with no single manufacturer or city of origin; for these, version numbers and literature citations are the appropriate identification method. Amazon SageMaker Canvas is a commercial managed cloud service offered by Amazon Web Services, Inc. (Seattle, WA, USA), to which the convention applies.
- **Cover letter text:** *"Following the reviewer's request, we have added explicit version numbers for all software tools used in the study. In the Synthetic Data Generation section, the implementation language and core libraries are now identified as Python ($\ge$ 3.9), NumPy ($\ge$ 2.0.2), and pandas ($\ge$ 2.3.1). In the C-AutoML section, scikit-learn ($\ge$ 1.6.1) is cited with its API reference (Buitinck et al. 2013). The SHAP library ($\ge$ 0.49.0; Lundberg & Lee 2017) has been added to the list of optional packages in Appendix A, and MLflow ($\ge$ 2.14.0; Zaharia et al. 2018) now includes its version in Appendix B. Regarding the city of manufacturer: we note that this convention in clinical journals applies to proprietary instruments and commercial software platforms. The open-source libraries (Python, NumPy, pandas, scikit-learn, SHAP, MLflow) are community-developed tools with no manufacturer location; for these, version numbers and primary literature citations constitute the appropriate identification. Amazon SageMaker Canvas is a commercial managed service operated by Amazon Web Services, Inc. (Seattle, WA, USA), and this designation has been added to the manuscript accordingly. (commits 7e24bbe, d1593cb)"*

#### R2 — M&M: Table 5 footer note — distribution notation (`f7c0d1d`)
- **What was done:** Added `\added[id=R2]{}` footer note below `tab:distributions` (Table 5 — Statistical Distributions for Synthetic Data Generation) explaining all four distribution symbols used in the Control and Risk group columns:
  - $\mathcal{N}(\mu, \sigma)$: Normal distribution, mean $\mu$, standard deviation $\sigma$
  - $\mathcal{U}(a, b)$: Uniform distribution over $[a, b]$
  - $\mathrm{Gamma}(k, \theta)$: Gamma distribution with shape $k$ and scale $\theta$
  - $\mathrm{Bernoulli}(p)$: Bernoulli variable with success probability $p$ (binary outcome)
- **Why footer note, not caption:** MDPI house style places symbol keys below the table as a "Notes:" paragraph; captions describe table content, not notation conventions.
- **Cover letter text:** *"We added a notation legend as a footer note below Table 5 (Statistical Distributions for Synthetic Data Generation), clarifying the distribution symbols used: $\mathcal{N}(\mu, \sigma)$ denotes a Normal distribution with mean $\mu$ and standard deviation $\sigma$; $\mathcal{U}(a, b)$ denotes a Uniform distribution over $[a, b]$; $\mathrm{Gamma}(k, \theta)$ denotes a Gamma distribution with shape $k$ and scale $\theta$; and $\mathrm{Bernoulli}(p)$ denotes a binary Bernoulli variable with success probability $p$. (commit f7c0d1d)"*

#### R2 — M&M: Add p-values to Table 4 (`46d90ca`)
- **What was done:**
  - Created `code/05-statistical-tests-p-values.ipynb` with Mann-Whitney U tests and Pearson chi-squared tests; ran all cells; exported `tab_desc_stats_with_pvalues.tex` and `tab_variable_distribution_with_pvalues.tex`.
  - Added `\usepackage{multirow}` to `preeclampsia-ml-hia.tex` preamble.
  - **`tab:desc_stats`:** Changed column spec from `{lp{2.cm}XXXX}` to `{lp{2.cm}XXXXr}`; added `$p$-value\textsuperscript{a}` header column; added `\added[id=R2]{\multirow{2}{*}{$<$0.001}}` for each variable's two rows; added `\added[id=R2]` footnote explaining the Mann-Whitney U test.
  - **`tab:variable_distribution`:** Changed column spec from `{llXXp{2.cm}X}` to `{llXXp{2.cm}Xr}`; added `$p$-value\textsuperscript{b}` header column; added `\added[id=R2]{\multirow{4}{*}{$<$0.001}}` for each variable block (5 variables including AGE GROUP); added `\added[id=R2]` footnote explaining Pearson chi-squared.
  - All p-values are $p < 0.001$ (highly significant): MWU statistic for Age $U=249156$; BMI $U=364121$; Gestational Weeks $U=616264$; Weight $U=366806$. Chi-squared: Age $\chi^2=1490.33$; BMI $\chi^2=947.43$; Gestational Weeks $\chi^2=428.14$; Weight $\chi^2=1001.46$; Age Group $\chi^2=1572.27$.
  - **Note on `\added` inside tables:** `\added[id=R2]{...}` cannot wrap a full `tabularx` environment (the `changes` package processes it as a macro argument, which breaks booktabs internal commands). The workaround is to mark only the new text content within cells with `\added`, and change the column spec silently.
- **Cover letter text:** *"Following the reviewer's request, we have added a $p$-value column to Table 4. For the descriptive statistics table (`tab:desc_stats`), we applied Mann-Whitney $U$ tests (two-sided) comparing the distribution of each continuous variable between the preeclampsia-positive and preeclampsia-negative groups, yielding a single $p$-value per variable displayed in a merged cell spanning both the No and Yes rows. For the quartile distribution table (`tab:variable_distribution`), we applied Pearson chi-squared tests of independence between the variable bin and preeclampsia status, yielding one $p$-value per variable block spanning all four bin rows. All tests produced $p < 0.001$, confirming highly significant between-group differences for all four continuous variables and for the age-group categorical variable. Statistical analyses are fully reproducible from `code/05-statistical-tests-p-values.ipynb`. (commit 46d90ca)"*

---

#### R2 — M&M + Discussion: PE prevalence, inter-group comparisons, chronic HTN, PE sub-categories (`657904e`)

##### R2 — PE incidence and high prevalence

- **What was done:** Added `\added[id=R2]{...}` paragraph in the Statistical Characteristics subsection (M&M), immediately after the opening paragraph introducing Figure and Tables 2 & 4.
- **Paper text (concise):** States the record-level prevalence of 16.11% (671/4,165 records) and patient-level prevalence of 25.1% (588/2,344 unique patients), explains both exceed the global estimate of 4.43% because HGOIA is a public OB/GYN specialty hospital in Quito and Ecuador's MSP-coordinated zonification model routes high-risk pregnancies to specialised facilities. Notes that precision and F1 are calibrated to the cohort prevalence and would decrease in lower-prevalence settings. Also adds a sentence pointing explicitly to Tables 2 and 4 as inter-group comparisons with statistical significance tests (closing the inter-group comparisons item).
- **Cover letter text (detailed):** *"We report that the record-level PE prevalence in the final ML dataset is 16.11% (671 of 4,165 records); at the patient level, 588 of the 2,344 unique patients (PCTE\_IDE\_x) received a PE diagnosis (25.1%). Both figures substantially exceed the global pooled estimate of approximately 4.43% (Vera-Ponce et al., 2025). The enrichment reflects two compounding factors: (i) HGOIA is a public hospital specialised in gynaecology and obstetrics in Quito, whose clinical focus systematically concentrates obstetric complications; and (ii) Ecuador's public health system operates under a tiered zonification model coordinated by the Ministerio de Salud Pública (MSP), under which patients with complex or high-risk pregnancies are directed toward specialised facilities. The exact catchment area and referral statistics for HGOIA would require confirmation from official MSP records and are acknowledged as a caveat. The elevated base rate directly affects the reported performance metrics: precision and F1 are calibrated to the cohort prevalence and would be expected to decrease if the same models were deployed in a primary-care or general-population setting where PE prevalence is closer to 4–5%. Inter-group comparisons of all model input variables (with Mann-Whitney U and Pearson chi-squared statistical significance tests) are provided in Table 2 (tab:desc_stats) and Table 4 (tab:variable_distribution), both added at the request of Reviewer 2 in the previous revision point. (commit 657904e)"*

##### R2 — Chronic HTN exclusion and PE sub-categories

- **What was done:** Added `\added[id=R2]{...}` paragraph at the end of the Discussion Limitations section (after the existing `\added[id=R1]` limitations block).
- **Paper text:** States that O10 (pure chronic HTN) was absent from both classes by design; O11 (superimposed PE on chronic HTN) is present in the positive class with a partially different physiopathological background — separate O11 analysis recommended as future work. Also states that PE sub-category stratification (early/late, moderate/severe) was not feasible with the available binary labels and is recommended as future work.
- **Cover letter text:** *"Regarding the reviewer's request to exclude patients with pre-existing chronic hypertension: patients with chronic hypertension without superimposed preeclampsia (ICD-10 O10) were not included in either class by design. The positive class was constructed from ICD-10 codes O11, O13, O14, O15, and O16, as documented in Table [tab:cie10-icd10-mapping]; the negative class consists solely of normal pregnancy supervision records (Z340, Z348, Z349). Therefore, pure chronic hypertension cases (O10) are absent from the dataset. However, we acknowledge that cases of superimposed preeclampsia on chronic hypertension (O11) are present in the positive class and carry a partially different physiopathological background from de novo gestational preeclampsia. A dedicated analysis of O11 cases as a separate subgroup is recommended as future work and has been added to the Limitations paragraph.*

*Regarding PE sub-categories (early vs. late onset; moderate vs. severe): we understand the reviewer's concern and recognise that both the HGOIA demographic dataset and the synthetic dataset must be considered separately.*

*For the HGOIA dataset, stratified analysis is not feasible because the data record only a binary PE diagnosis without severity grade or gestational age at onset. Inferring sub-categories from the demographic variables alone would be methodologically unsound: gestational age recorded at a given prenatal visit does not equal gestational age at PE onset, and severity cannot be determined from age, weight, BMI, and gestational weeks.*

*For the synthetic dataset, adding sub-category labels would technically be straightforward — one could define, for example, early-onset PE as PE-MSP-positive with gestational weeks < 34, or severe PE as PE-MSP-positive with SBP ≥ 160 mmHg or DBP ≥ 110 mmHg — because all clinical variables needed for such rules (gestational age, blood pressure, platelets, creatinine, AST) are already present in the synthetic generation function. However, generating synthetic sub-category labels would require a fundamentally different experimental design. The present study is a binary diagnostic classification task: given a set of predictor variables, determine whether PE is present. Introducing sub-categories would change the problem to a multiclass, ordinal, or multi-label classification task, requiring retraining of all models (6 families × 2 training tracks = 12 retrained configurations), new hyperparameter searches, new evaluation protocols, and new interpretability analyses. This would constitute a separate investigation rather than a revision to the current manuscript, whose scope is deliberately focused on binary PE diagnosis as a foundational step.*

*We have acknowledged this limitation in the revised Discussion and recommend sub-category analysis as a direction for future work, where both the HGOIA dataset (once enriched with severity and onset-timing metadata) and the synthetic dataset (with sub-category labels incorporated into the generation function) could be jointly employed. (commit 657904e)"*

---

#### R2 — Discussion: Comparison with similar-framework studies (`0e6893b`)

- **What was done:** Added a 3-sentence bridge paragraph at the end of Related Works introducing recent comparable-framework studies (Liu2022, Zeng2024, AdvancingML2025, BiasFree2024, JMIR2026 review) and cross-referencing the new Discussion comparison table. Added a comparison table (`tab:ml_pe_comparison`) and a 4-sentence contextualisation paragraph in the Discussion, placed after the interpretability paragraph and before the Limitations block. Added 5 new references to `bibliography.bib`.
- **Paper text (Related Works bridge):** *"Several recent studies have further narrowed the comparison to tree-based ensembles operating on demographic and routinely collected clinical variables, reporting AUC values in the 0.80--0.96 range depending on predictor availability [citations]. A 2026 systematic review confirms that performance heterogeneity across studies is largely driven by differences in predictor sets and population characteristics, supporting comparisons restricted to works with broadly similar frameworks and variable types [citation]. Table [tab:ml_pe_comparison] in the Discussion provides a structured comparison of our results against these studies."*
- **Paper text (Discussion):** *"Table [tab:ml_pe_comparison] situates our results among studies that employ comparable model families and predictor types. Despite using only four demographic variables, our best models achieve discrimination metrics (AUC 0.944--0.956, F1 0.802--0.805) that fall within the range reported for tree-based ensembles operating on richer predictor sets [citations]. Direct numerical ranking across studies is discouraged by the substantial heterogeneity in populations, predictor availability, and outcome definitions documented in recent systematic reviews [citations]; the comparison in Table [tab:ml_pe_comparison] is therefore intended to contextualise rather than rank performance."*
- **Cover letter text:** *"Following the reviewer's request to compare our results with studies using similar frameworks and variables, we have added a comparison table and contextualisation paragraph to the Discussion. As the reviewer correctly noted, comparisons must be restricted to studies employing comparable model families and predictor types. Finding an exact variable match is not possible because studies using demographic-only predictor sets for PE prediction are rare; we therefore selected the closest comparators: Li et al. (2021, XGBoost on 38 EHR variables including BMI and blood pressure, AUC 0.955), Liu et al. (2022, ensemble ML on longitudinal EMR trajectories, AUC >0.90), Zeng et al. (2024, ensemble ML on maternal characteristics and first-trimester biomarkers, AUC 0.80--0.86), and a recent 2025 comparison of CatBoost, LightGBM, XGBoost, and RF for early PE prediction (accuracy ~0.90, AUC >0.90). All comparators use tree-based ensemble methods and tabular clinical/demographic data. Our best models (AUC 0.944--0.956) fall within the reported range despite using only four demographic predictors (age, gestational age, BMI, weight), which we note is a substantially narrower feature set. We explicitly caution against direct numerical ranking, citing systematic reviews (Darsareh 2024, Rahman 2023, JMIR 2026) that document performance heterogeneity driven by differences in predictor sets, populations, and outcome definitions. A bridge paragraph in Related Works signposts the new table. Five new references were added to the bibliography. (commit 0e6893b)"*

---


| Item | Task | Status |
|------|------|--------|
| **Manuscript-wide** | Font uniformization: remove `\texttt{}` from model names in Abstract + body | ✅ `3f763eb` |
| **Introduction** | Add references aligned with PE prediction/diagnosis guidelines | ✅ `2cb5f03`, `4b0796f` |
| **M&M** | Software versions + city of manufacturer for all tools | ✅ `7e24bbe`, `d1593cb` |
| **M&M** | Report PE incidence in dataset; discuss implications on performance | ✅ `657904e` |
| **M&M** | Discuss unusually high PE prevalence in cohort (referral-center bias?) | ✅ `657904e` |
| **M&M** | Clarify gestational age (trimester) when each feature was recorded | ⚠️ N/A — see note below |
| **M&M** | Specify if lab variables (platelets, creatinine, proteinuria/24h) are single or repeated | ⚠️ N/A — see note below |
| **M&M** | Inter-group comparisons table (PE vs non-PE for all model variables, with p-values) | ✅ `657904e` (Tables 2 & 4 with p-values + explicit sentence) |
| **M&M** | Consider excluding chronic hypertension patients | ✅ `657904e` (O10 absent by design; O11 acknowledged in Limitations) |
| **M&M** | Remove or justify SpO2 and chest pain (non-PE-specific) | ✅ `dacac34` — justified via FullPIERS citations |
| **M&M** | Regenerate Figure 2 at ≥300 dpi without horizontal stretching | ⬜ |
| **M&M** | Add p-values to Table 4 | ✅ `46d90ca` |
| **M&M** | Add footer note explaining Us and Ns in Table 5 | ✅ `f7c0d1d` |
| **Results** | Clarify FullPIERS / LR ~100% performance (shared with R1) | ✅ `dacac34` |
| **Results** | Apply models to PE sub-categories (early/late, moderate/severe) | ✅ `657904e` (acknowledged as future work in Limitations) |
| **Discussion** | Re-expand after model revision; compare with studies using similar frameworks | ✅ `0e6893b` |
| **Discussion** | State explicitly strengths and limitations | ✅ `657904e`, `bf4845f` |

---

### ⚠️ R2 Items Not Applicable — Reviewer Misidentified Variables as Clinically Collected

**Reviewer 2 asked:**
> "At which gestational age (1st/2nd/3rd trimester) each feature was recorded?"  
> "Were variables such as platelets, creatinine, proteinuria/24h recorded at a specific moment, or were multiple determinations used?"

**Why these cannot be answered:** The reviewer appears to assume that all features in the manuscript (including platelets, creatinine, proteinuria/24h, SpO₂, chest pain) come from clinical data collection. This is incorrect:

- The **HGOIA dataset** contains **only demographic variables**: Age, BMI, Gestational Age (weeks), and Weight. These come from retrospective administrative records and are not laboratory measurements — no timing or repetition question applies.
- The **clinical-lab variables** (SBP, DBP, Proteinuria\_24h\_mg, SpO₂, Platelets, Creatinine\_umol\_L, AST\_U\_L, Chest\_Pain) exist **exclusively in the synthetic dataset**, where each value is **computationally generated** from a parametric statistical distribution (see `code/04-synthetic-data-generation.ipynb` and `tab:distributions` in the paper). They were never clinically collected from any patient.

**Draft response to reviewer (to include in cover letter / response document):**

> We thank the reviewer for raising this important methodological point. We clarify that the clinical laboratory variables the reviewer refers to (platelets, creatinine, 24-h proteinuria, SpO₂, chest pain) do not originate from clinical data collection and are not present in the HGOIA dataset. The HGOIA dataset contains only demographic variables (maternal age, body-mass index, gestational age in weeks, and maternal weight) extracted from administrative records. The laboratory variables appear exclusively in the **synthetic dataset**, where each value is algorithmically generated from a parametric statistical distribution (Normal, Gamma, or Bernoulli) chosen to reflect clinically plausible ranges for each group (Control vs. Risk), as fully described in Table [distributions] and Section [Synthetic Data Generation]. Because these values are simulated rather than measured, concepts such as gestational trimester of measurement, single vs. repeated measurements, or specific measurement timing are not applicable to this component of the study. We have added a clarifying sentence in the Synthetic Data Generation section to make this distinction explicit.

**Paper change required:** Add a sentence in `sec:synth-data-gener` explicitly stating that the synthetic variables are not clinically collected and that timing/repetition questions therefore do not apply. Use `\added[id=R2]` markup.

---

## Original Table Backups (pre-R2 p-value additions)

**Context:** Commit `46d90ca` added a `$p$-value` column to both tables below.
If a reviewer asks to revert or the column needs to be removed, restore the
`tabularx` bodies using one of the two methods below.

### Recovery method 1 — git (most reliable)

```bash
# View the original file at the commit just before the p-value changes
git show f7c0d1d:preeclampsia-ml-hia.tex | grep -A 60 "tab:desc_stats" | head -30
git show f7c0d1d:preeclampsia-ml-hia.tex | grep -A 80 "tab:variable_distribution" | head -45

# To recover the full file at that state:
git checkout f7c0d1d -- preeclampsia-ml-hia.tex
# (then re-apply any subsequent non-table changes manually, or cherry-pick)
```

### Recovery method 2 — paste verbatim originals below

**Label:** `\label{tab:desc_stats}` — placed inside `\begin{table}[H]` that starts
just before `\begin{tabularx}`. The caption begins *"Descriptive statistics of
demographic variables by preeclampsia diagnosis…"*

Replace the current `\begin{tabularx}...\end{tabularx}` block (and the
`\added[id=R2]` footnote line that follows it) with:

```latex
\begin{tabularx}{\textwidth}{lp{2.cm}XXXX}
% R1.2 BEGIN: Sample sizes and statistics are from the pre-deduplication cohort
% (dataset_preeclampsia_cleaned.csv, 4245 records: 3571 normal / 674 PE).
% This is what 02-explore-cleaned-data.ipynb computes. Caption updated to
% clarify the cohort; tab:dataset-characteristics shows the final ML counts.
\toprule
Variable & Preeclampsia & Sample Size & Mean & Median & Std. Dev. \\
\midrule
AGE (YEARS) & No & 3571 & 18.71 & 18.00 & 4.13 \\
          & Yes & 674  & 29.72 & 30.00 & 7.17 \\
\midrule
BMI       & No & 3571 & 25.71 & 25.19 & 4.39 \\
          & Yes & 674  & 33.00 & 32.63 & 6.02 \\
\midrule
GESTATIONAL  & No & 3571 & 27.55 & 27.70 & 7.76 \\
WEEKS                   & Yes & 674  & 33.80 & 35.40 & 4.42 \\
\midrule
WEIGHT    & No & 3571 & 61.20 & 59.50 & 11.67 \\
          & Yes & 674  & 80.51 & 78.75 & 16.17 \\
\bottomrule
\end{tabularx}
```

---

**Label:** `\label{tab:variable_distribution}` — placed inside `\begin{table}[H]`
whose caption begins *"Distribution of pregnancies and preeclampsia prevalence
across demographic concentration quartiles…"*

Replace the current `\begin{tabularx}...\end{tabularx}` block (and the
`\added[id=R2]` footnote line that follows it) with:

```latex
\begin{tabularx}{\textwidth}{llXXp{2.cm}X}
\toprule
Variable & Bin (range) & Total pregnancies & Normal pregnancies & Preeclampsia pregnancies & Preeclampsia prevalence (\%) \\
\midrule
AGE         & (12.999, 17.0] & 1745 & 1704 & 41 & 2.35\% \\
(YEARS)     & (17.0, 18.0]   & 690  & 670  & 20  & 2.90\% \\
            & (18.0, 22.0]   & 797  & 735  & 62  & 7.78\% \\
            & (22.0, 48.0]   & 1013 & 462  & 551 & 54.39\% \\
\midrule
AGE GROUP   & 10–14 years    & 73   & 66   & 7   & 9.59\% \\
            & 15–19 years    & 2877 & 2813 & 64  & 2.22\% \\
            & 20–35 years    & 1100 & 663  & 437 & 39.73\% \\
            & 36–49 years    & 195  & 29   & 166 & 85.13\% \\
\midrule
BMI         & (15.329, 23.18] & 1063 & 1041 & 22  & 2.07\% \\
            & (23.18, 25.93]  & 1064 & 1014 & 50  & 4.70\% \\
            & (25.93, 29.42]  & 1058 & 935  & 123 & 11.63\% \\
            & (29.42, 73.46]  & 1060 & 581  & 479 & 45.19\% \\
\midrule
GESTATIONAL & (3.399, 26.0]   & 1066 & 1055 & 11  & 1.03\% \\
WEEKS       & (26.0, 27.696]  & 1229 & 1062 & 167 & 13.59\% \\
            & (27.696, 35.2]  & 904  & 760  & 144 & 15.93\% \\
            & (35.2, 41.5]    & 1046 & 694  & 352 & 33.65\% \\
\midrule
WEIGHT      & (19.799, 54.0]  & 1086 & 1065 & 21  & 1.93\% \\
            & (54.0, 61.5]    & 1043 & 1002 & 41  & 3.93\% \\
            & (61.5, 71.5]    & 1064 & 937  & 127 & 11.94\% \\
            & (71.5, 157.2]   & 1052 & 567  & 485 & 46.10\% \\
\bottomrule
\end{tabularx}
```

---

## Key Technical Facts (for reference)

### C-AutoML HPO Search Spaces (from `code/autopilot_train.py`)
| Model | Key hyperparameters |
|-------|-------------------|
| LR | C ∈ {0.01,0.1,1,5,10}; solver ∈ {liblinear,lbfgs}; class_weight ∈ {None,balanced} |
| RF | n_estimators ∈ {200,400,600}; max_depth ∈ {None,5,10,20}; min_samples_split ∈ {2,5,10}; min_samples_leaf ∈ {1,2,4} |
| MLP | hidden_layer_sizes ∈ {(64),(128),(64,32),(128,64)}; alpha ∈ {0.0001,0.001,0.01}; lr_init ∈ {0.0005,0.001,0.005} |
| XGBoost | max_depth ∈ {3,4,6,8}; lr ∈ {0.01,0.05,0.1,0.2}; subsample ∈ {0.7,0.85,1.0} |
| LightGBM | num_leaves ∈ {15,31,63}; lr ∈ {0.01,0.05,0.1}; subsample ∈ {0.7,0.85,1.0} |
| CatBoost | depth ∈ {4,6,8}; lr ∈ {0.01,0.05,0.1}; iterations ∈ {200,400,600}; l2_leaf_reg ∈ {1,3,5} |

### Bibliography entries added this session
```
bergstra2012random  — Bergstra & Bengio (JMLR 2012, v13, pp.281–305)
agtabular2020       — Erickson et al., AutoGluon-Tabular (arXiv:2003.06505)
caruana2004ensemble — Caruana et al., Ensemble Selection (ICML 2004, doi:10.1145/1015330.1015432)
millman2011spo2     — Millman et al., SpO2 in FullPIERS context
long2025fullpiers   — Long et al. 2025 FullPIERS
teichmann2025fullpiers — Teichmann et al. 2025 FullPIERS
elongi2025fullpiers — Elongi et al. 2025 FullPIERS
brown2018isshp      — Brown et al. 2018, ISSHP classification, Hypertension 72(1):24–43
vera2025preeclampsia — Vera-Ponce et al. 2025, global prevalence meta-analysis, Front Reprod Health 7:1706009
Darsareh2024        — Darsareh et al. 2024 (ML PE prediction)
Rahman2023          — Rahman et al. 2023 (ML PE prediction)
```

### Label / reference map (key cross-references)
| Label | Content | ~Line |
|-------|---------|-------|
| `eq:ensemble-prediction` | Weighted ensemble prediction formula | ~1138 |
| `tab:hpo-search-spaces` | C-AutoML HPO search spaces | ~1188 |
| `fig:c-automl-flowchart` | C-AutoML training flowchart | ~1237 |
| `tab:dataset-characteristics` | Dataset at each preprocessing stage | ~798 |
| `sec:performance-aws-sage-maker-canvas` | Canvas results subsection | ~1562 |
