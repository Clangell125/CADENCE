# CADENCE: Evolutionary Fragment-Based Discovery of Kinase Inhibitors

**CADENCE** is a computational drug discovery pipeline designed for the rapid generation of novel candidate kinase inhibitors directly from kinase amino acid sequences. By integrating **Convolutional Neural Networks (CNNs)** for binding affinity prediction with a **BRICS-based evolutionary algorithm** for molecular synthesis, CADENCE bypasses the need for computationally expensive 3D protein structures during the early discovery phase.

## 🚀 Key Features

* **Sequence-Driven Discovery:** Generates inhibitor candidates using only the target's amino acid sequence.
* **Evolutionary Synthesis:** Utilizes BRICS-based decomposition and mutation to evolve molecules over 15 generation cycles.
* **Automated Filtering:** Prioritizes "drug-like" candidates by enforcing Lipinski’s Rule of Five.
* **Benchmarked Performance:** Achieves a Mean Squared Error (MSE) of 0.1962 on the Davis dataset, outperforming several established baseline models.

---

## 🏗️ Pipeline Architecture

The CADENCE pipeline consists of three core stages:

### 1. Binding Affinity Prediction

A CNN-based model trained on approximately 30,000 kinase-ligand pairs from the [Davis Dataset](https://doi.org/10.1038/nbt.1990).

* **Input:** 1D amino acid sequences (encoded 1-20) and 2048-bit Morgan Fingerprints for ligands.
* **Architecture:** Parallel convolutional branches with varying kernel sizes (4, 8, 12) to capture multi-scale protein features.
* **Metric:** Predicted affinity is measured on the  scale.

### 2. Candidate Generation (Evolutionary Algorithm)

* **Initialization:** Starts with 1024 randomly sampled ligands.
* **Mutation:** Molecules undergo BRICS-based mutation (65% probability), where fragments are replaced with random pieces from the fragment pool.
* **Selection:** The top 15 distinct "elite" candidates are carried forward, ensuring chemical diversity via Tanimoto similarity filtering.

### 3. Filtering Stage

Candidates are automatically screened to meet drug-likeness criteria:

* **Molecular Weight:**  Da
* **LogP:** 
* **H-Bond Donors:** 
* **H-Bond Acceptors:** 

---

## 📊 Results & Validation

The pipeline was validated against three major cancer-linked kinases: **ABL1**, **HER1**, and **ERBB2**.

| Target | Clinical Relevance | AI-Lead Performance |
| --- | --- | --- |
| **ABL1** | Chronic Myelogenous Leukemia (CML) | Predicted  |
| **HER1** | Rapidly dividing lung cancer cells | Rediscovered **Lapatinib** structure |
| **ERBB2** | Breast and ovarian tumor overexpression | Competitive with current treatments |

> **Note:** The model successfully "learned" binding specificity, demonstrated by the diverging predicted affinities between known strong binders (Gleevec) and negative controls (Aspirin) during training.

---

## 🛠️ Installation & Usage

*(Note: Add your specific repository commands here)*

```bash
git clone https://github.com/your-username/CADENCE.git
cd CADENCE
pip install -r requirements.txt
python run_pipeline.py --target "YOUR_AMINO_ACID_SEQUENCE"

```

## 📝 Citation

If you use this work, please cite the original study:

```bibtex
@article{angell2026cadence,
  title={CADENCE: Evolutionary Fragment-Based Discovery of Novel ABL1, HER1, and ERBB2 Competitive Inhibitors},
  author={Angell, Christopher and Chen, David},
  year={2026}
}

