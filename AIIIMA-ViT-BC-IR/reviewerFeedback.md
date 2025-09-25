# Reviewer #2

## Questions

1. Please confirm that you:
   a) Will not use ideas from a paper that you review to develop new ones of your own before its publication.
   b) After the review process, destroy all copies of papers and supplementary material associated with the submission.

**Agreement accepted**

3. Please describe the contribution of the paper (a few lines)

Appreciate the "Comprehensive" research work presented in the paper.

1. **First Video Transformer Application to Dynamic Thermal Imaging**: Introduces Video Vision Transformers (ViViT) to Dynamic Infrared Thermography (DIT) protocol for breast cancer detection, representing the "first" systematic application of temporal transformer architecture.

2. **Systematic Evaluation**: Provides systematic evaluation comparing spatial-only (ViT), temporal (TDL+LSTM), and spatio-temporal (ViViT) approaches on the same thermal imaging task, offering insights into which architectural paradigms are most effective for sequential/dynamic thermal data.

3. **Pre-training Advantage**: Demonstrates that pre-trained transformers (ImageNet-1K weights) significantly outperform scratch training for medical thermal imaging, achieving 0.9444 AUC versus 0.8611 AUC.

4. **State-of-the-Art Advancement**: Advances the state-of-the-art for Dynamic Infrared Thermography by achieving 0.9444 AUC compared to previous 3D CNN baseline of 0.9351 AUC, though with higher spatial resolution (224×224 vs 32×32).

4. Please provide detailed and constructive comments for the authors

(a) **There is NO DISCUSSION SECTION in the paper!**

(b) **Domain Gap Issue**: Pre-training on natural images/videos for medical thermal data represents a significant domain gap that's insufficiently addressed in the discussion.

(c) **Overfitting Concerns**: Deep models trained on tiny datasets (46-134 patients) are inherently prone to overfitting. Even after k-fold validation, you are choosing the best performing model instead of averaging prediction probabilities or voting techniques.

**Others:**

1. Generally you should avoid abbreviations in an abstract, but you can use them sparingly if the term is crucial and repeated multiple times.

2. Check the ORCID of the authors.

3. **Clarity Enhancement**: A table outlining models and what they were pretrained on (e.g., ViT models: Pre-trained on ImageNet-1K (natural images), then fine-tuned on thermal data) and what are the training datasets for each of these models (ex: Infrared RGB Images (DIR): 149 patients total, 134 for training, 15 for test) would enhance the clarity.

4. **Figure 1 Improvement**: Figure 1 which provides overview of the study can be enhanced by using graphical and representative deep learning architecture and known machine learning symbols. While the flowchart conveys a lot of information it fails to capture the reader's attention; a graphical overview would be better.

5. **Discussion Points**: A good point of discussion can be: Do we need Dynamic Thermal Imaging?? and such large models?? (https://pmc.ncbi.nlm.nih.gov/articles/PMC12189745/)
   *(There is no discussion section in the paper!!; This makes this article more of a technical report than a research paper; Lacks discussion of what the results mean for clinical practice)*

5. Rate the paper on a scale of 0-5, 5 being the strongest (3-5: accept; 0-2: reject)

**Weak Accept**

6. Rate the paper in terms of its clinical significance and the application of AI in Infrared Imaging for Medical Applications.

**Medium**

---

# Reviewer #3

## Questions

1. Please confirm that you:
   a) Will not use ideas from a paper that you review to develop new ones of your own before its publication.
   b) After the review process, destroy all copies of papers and supplementary material associated with the submission.

**Agreement accepted**

3. Please describe the contribution of the paper (a few lines)

This paper introduces a video transformer-based approach for classifying breast cancer from dynamic infrared thermography (DIT) sequences using the UFF protocol on the DMR-IR dataset (149 patients). It fine-tunes a Video Vision Transformer (ViViT, specifically TimeSformer) pre-trained on Kinetics-400, achieving an AUC of 0.935 in 10-fold cross-validation and 0.944 on a held-out test set of 15 patients, outperforming prior 3D CNN benchmarks. Key contributions include extending ViTs to temporal DIT analysis, evaluating transfer learning's impact, comparing thermal map vs. RGB formats, and claiming the first application of ViViTs in this domain for improved global spatio-temporal modeling.

4. Please provide detailed and constructive comments for the authors

**Technical Innovation:**
The use of ViViT/TimeSformer video transformers and careful benchmarking against established 3D CNN and ViT models represents strong methodological advancement. The systematic comparison of model architectures, use of pre-trained weights, and extensive cross-validation demonstrate technical rigor.

**Introduction and Research Questions:** The background on breast cancer epidemiology, limitations of mammography, and advantages of TG/DIT is concise and supported by relevant citations. The RQs logically build toward evaluating ViTs' superiority, transfer learning, and integration with dynamic protocols.

**Related Works (Section 4):** Good overview of TG protocols, DL in medical imaging, and prior CNN/ViT applications. Citations to [6] (3D CNNs) and [10] (static TG) set clear baselines.

**Small Dataset Size:**
The overall sample is limited (149 patients, 20 frames each). While this is typical for medical imaging, it constrains model generalization and may inflate performance with deep architectures and pre-trained networks. External validation from larger or multicenter datasets would strengthen the findings.

**Temporal acquisition is underspecified.**
DIT sequences have 20 frames, but frame rate/total duration of the thermal stress acquisition is not reported. Without timing, it's impossible to judge whether the sequence spans physiological dynamics that DIT aims to capture. Please report fps, total seconds, and when cooling started/ended relative to capture.

**Temporal signal may be under-utilized.**
ViT(32) and ViViT achieve similar AUC (~0.94), suggesting that the diagnostic signal is largely spatial, or the temporal sampling doesn't capture physiological changes. Run ablations:
1. Frame count (3/4/6/8), contiguous vs uniformly spaced;
2. Early vs late frames;
Report when, if ever, temporal modeling adds measurable value.

**Plain Temperature Data vs. Image Data:**
The study mentions different outcomes based on input data (temperature matrices, IR images). More insight on why performance differs, and on practical implications for clinical deployment (e.g., ease of acquisition, standardization), would add value.

**Discussion (Section 7):** Ties results to RQs, noting ViTs' global modeling edge and transfer learning benefits. Acknowledges small dataset and need for larger validation. Suggestion: Expand on limitations - computational cost of ViViT wrt the other methods.

5. Rate the paper on a scale of 0-5, 5 being the strongest (3-5: accept; 0-2: reject)

**Strong Accept**

6. Rate the paper in terms of its clinical significance and the application of AI in Infrared Imaging for Medical Applications.

**High**

---

# Reviewer #5

## Questions

1. Please confirm that you:
   a) Will not use ideas from a paper that you review to develop new ones of your own before its publication.
   b) After the review process, destroy all copies of papers and supplementary material associated with the submission.

**Agreement accepted**

3. Please describe the contribution of the paper (a few lines)

The paper proposes the first use of Video Vision Transformers (ViViTs) for dynamic infrared thermography (DIT) in breast cancer detection. It adapts transformer-based models to process sequential thermal imaging data, evaluates both pre-trained and scratch-trained architectures, and benchmarks them against CNN and 3D CNN approaches. On the DMR-IR dataset, the method achieves an AUC of 0.94 on a small test set, showing that ViViTs can model temporal dependencies in thermographic sequences effectively and perform comparably or better than existing CNN-based methods.

4. Please provide detailed and constructive comments for the authors

1) **Reference Formatting**: In some places, references are grouped but not sorted consistently (e.g., "[21,17]" instead of "[17,21]"). Springer requires numerical order within the brackets.

2) **Temporal Sampling Justification**: The implementation details (optimizers, augmentation, patch sizes, training epochs) are solid, but the choice to sample only 8 frames for the TimeSformer model raises questions about whether relevant temporal dynamics are lost. This design compromise is not well-justified.

3) **Overfitting Analysis**: The results section shows near-perfect performance for ViT (32×32) pre-trained (F1 = 1.0 in CV), which is suspiciously high given the small dataset size and the inherent noise in thermography. Overfitting is a strong possibility, yet this risk is not critically analyzed.

4) **Statistical Significance**: Comparisons with prior works are made, but statistical significance tests are missing. Given the limited dataset, reporting confidence intervals or performing permutation tests would add rigor.

5. Rate the paper on a scale of 0-5, 5 being the strongest (3-5: accept; 0-2: reject)

**Weak Accept**

6. Rate the paper in terms of its clinical significance and the application of AI in Infrared Imaging for Medical Applications.

**High**
