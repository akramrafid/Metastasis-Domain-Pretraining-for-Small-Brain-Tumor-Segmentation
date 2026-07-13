# Metastasis Detection-Rate Validation Methodology

This document outlines the clinical and statistical methodology for evaluating the models on the unlabeled Stanford BrainMetShare test cohort.

---

## 1. Ground Truth Determination

The Stanford BrainMetShare test cohort (`test_NifTI`, $N = 51$ patients) contains no voxel-level segmentation masks. However, per the study inclusion criteria documented in the accompanying clinical metadata (`AI_Brain_Mets_Primary_Cancer.csv`), every patient registered in this cohort is clinically diagnosed with active brain metastases. 

Therefore, the patient-wise ground truth for metastasis presence is **strictly positive ($1.0$)** for every case in this cohort.

---

## 2. Detection Criteria & Metric Definition

In the absence of spatial annotations, we formulate evaluation as a binary detection task. 

### Voxel-Volume Detection Criterion
A patient volume is classified as **"detected"** by the model if it predicts at least one voxel of metastasis in the entire 3D volume:
$$D_p = \mathbb{I}\left( \sum_{x,y,z} \hat{y}(x,y,z) > 0 \right)$$
where:
- $\hat{y}(x,y,z) \in \{0, 1\}$ is the binary prediction at voxel coordinate $(x,y,z)$.
- $\mathbb{I}(\cdot)$ is the indicator function.
- $D_p \in \{0.0, 1.0\}$ represents the model's patient-wise detection outcome.

### Cohort Detection Rate
The cohort-wide **Detection Rate** is defined as the fraction of patients with positive detections:
$$\text{Detection Rate} = \frac{1}{N} \sum_{p=1}^{N} D_p$$
This metric serves as a patient-level proxy for clinical sensitivity in unlabeled cohorts.

---

## 3. Methodological Limitations

While this validation method allows us to assess the model's performance on fully unannotated clinical test sets, it introduces the following key limitation:
- **Anatomical Correctness Verification**: Since no voxel-level masks are available, we cannot verify that the model's positive predictions align anatomically with the true metastasis locations. A positive prediction could theoretically be a false positive in normal tissue rather than a true detection of a lesion. This limitation is discussed openly in the paper's limitations section.
