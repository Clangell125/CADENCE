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

* **Molecular Weight:** Da  
* **LogP:** Lipophilicity
* **H-Bond Donors** 
* **H-Bond Acceptors** 

---

## 📊 Results & Validation

The pipeline was validated against three major cancer-linked kinases: **ABL1**, **HER1**, and **ERBB2**.

| Target | Clinical Relevance | AI-Lead Performance |
| --- | --- | --- |
| **ABL1** | Chronic Myelogenous Leukemia (CML) | Competitive with current treatments  |
| **HER1** | Rapidly dividing lung cancer cells | Competitive with current treatments + Rediscovered **Lapatinib** structure |
| **ERBB2** | Breast and ovarian tumor overexpression | Competitive with current treatments |

> **Note:** The model successfully "learned" binding specificity, demonstrated by the diverging predicted affinities between known strong binders (Gleevec) and negative controls (Aspirin) during training.

---

## 🛠️ Installation & Usage



```bash
# 1. Clone the repository
git clone https://github.com/your-username/CADENCE.git

# 2. Enter the directory
cd CADENCE

# 3. Install all dependencies
pip install tensorflow rdkit-pypi pyqt6 PyTDC scikit-learn pandas numpy tqdm

# 4. Launch the application
python cadence_pro.py

```


## 🚀 Getting Started

To use the CADENCE pipeline for your own target proteins, follow these steps:

### 1. Prerequisites

Ensure you have Python 3.9+ and the necessary bioinformatics libraries installed:

```bash
pip install tensorflow rdkit-pypi pyqt6 PyTDC scikit-learn pandas numpy tqdm

```

### 2. Running the Pipeline

The entire stack is encapsulated in the `cadence_pro.py` script. This script handles the protein encoding, evolutionary molecular generation, and filtering.

1. **Launch the script:**
```bash
python cadence_pro.py

```


2. **Input your Target:** When prompted, paste the **Kinase Amino Acid Sequence** (in FASTA/1D string format).
3. **Monitor Progress:** The script will run through 15 generations of evolutionary refinement.
4. **Review Leads:** The top 15 candidate molecules (in SMILES format) will be exported to a `.csv` file in the `/results` directory.

---

## 🔬 How it Works (Under the Hood)

When you input a sequence into `cadence_pro.py`, the following workflow is triggered:

1. **Protein Feature Extraction:** The sequence is processed through a multi-scale CNN to identify potential binding motifs.
2. **Fragment-Based Assembly:** Using a library of over 300 bioactive fragments, the algorithm begins building molecules.
3. **Survival of the Fittest:** Each generated molecule is scored against your protein sequence. Only molecules with the highest predicted binding affinity () and those passing **Lipinski’s Rule of Five** are kept for the next mutation cycle.


## 📝 Citation

If you use this work, please cite the original study:

```bibtex
@article{angell2026cadence,
  title={CADENCE: Evolutionary Fragment-Based Discovery of Novel ABL1, HER1, and ERBB2 Competitive Inhibitors},
  author={Angell, Christopher and Chen, David},
  year={2026}
}

