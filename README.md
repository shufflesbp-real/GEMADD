# GEMADD
# GEMADD: A Graph-based Explainable Multimodal Approach for Disease Diagnosis


## Table of Contents

- [Requirements](#requirements)
- [Installation](#installation)
- [Datasets](#datasets)
- [Project Structure](#project-structure)
- [Usage](#usage)
- [Citation](#citation)
- [License](#license)

---

## Requirements

### Python Version
- **Python 3.10.12**

### Dependencies
All required packages are listed in `requirements.txt`. Key dependencies include:
- PyTorch
- OpenAI CLIP
- NetworkX
- NumPy, Pandas, scikit-learn

---

## Installation

### 1. Clone the Repository
```bash
git clone <https://github.com/shufflesbp-real/GEMADD/tree/main>
cd <GEMADD>
```
### 2. Install Requirements
```bash
pip install -r requirements.txt
```

This will install all dependencies including the CLIP model from:
```
https://github.com/openai/CLIP.git
```

### 3. Install SRWR Module
Clone and install the Supervised Random Walk with Restart module:
```bash
git clone https://github.com/jinhongjung/srwr.git
cd srwr
grep -v "numpy" requirements.txt | grep -v "scipy" > modified_requirements.txt
pip install -r modified_requirements.txt
```

Make sure the `srwr` module is accessible in your Python path.

### 4. Download VGG19 Image Embeddings
Download the pre-computed VGG19 embeddings from:
- **Download Link:** [VGG19 Embeddings (Google Drive)-150 MB](https://drive.google.com/file/d/1lwfErF8Q0O7Gpyo_U7feCicYNG0m0KLc/view?usp=drive_link)
- Place the downloaded `vgg19_image_embeddings.npz` file in the appropriate data directory

---

## Datasets

### MDDIAL Dataset

The preprocessed MDDIAL dataset is provided in `data-files/mddial/`.

**Original Dataset:** [MDDial GitHub Repository](https://github.com/srijamacherla24/MDDial/tree/main/data)

<!-- **Citation:**
```bibtex
@article{macherla2023mddialmultiturndifferentialdiagnosis,
      title={MDDial: A Multi-turn Differential Diagnosis Dialogue Dataset with Reliability Evaluation}, 
      author={Srija Macherla and Man Luo and Mihir Parmar and Chitta Baral},
      year={2023},
      eprint={2308.08147},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2308.08147}, 
}
``` -->

### ES-MMD Dataset

> **Important:** The ES-MMD dataset is not publicly available. Access must be requested from the authors.

**For access, please refer to:** [KI-MMDG GitHub Repository](https://github.com/NLP-RL/KI-MMDG)

For reproducibility, sample preprocessed files are provided in `data-files/esmmd/` (files starting with `sample_*`).

<!-- **Citation:**
```bibtex
@inproceedings{tiwari-etal-2024-seeing,
    title = "Seeing Is Believing! towards Knowledge-Infused Multi-modal Medical Dialogue Generation",
    author = "Tiwari, Abhisek  and
      Bera, Shreyangshu  and
      Verma, Preeti  and
      Manthena, Jaithra Varma  and
      Saha, Sriparna  and
      Bhattacharyya, Pushpak  and
      Dhar, Minakshi  and
      Tiwari, Sarbajeet",
    editor = "Calzolari, Nicoletta  and
      Kan, Min-Yen  and
      Hoste, Veronique  and
      Lenci, Alessandro  and
      Sakti, Sakriani  and
      Xue, Nianwen",
    booktitle = "Proceedings of the 2024 Joint International Conference on Computational Linguistics, Language Resources and Evaluation (LREC-COLING 2024)",
    month = may,
    year = "2024",
    address = "Torino, Italia",
    publisher = "ELRA and ICCL",
    url = "https://aclanthology.org/2024.lrec-main.1264/",
    pages = "14513--14523",
}
``` -->

---

### Running on MDDIAL Dataset

1. **Update paths in `config.py`** if needed (default paths work if running from main directory)

2. **Run the main script:**
```bash
python main.py
```

**Alternative:** If running from a different directory, update the paths in `data_loader.py` accordingly.

### Running on ES-MMD Dataset

#### Step 1: Prepare Data Files

Ensure all data files are in place:
- ES-MMD dataset files (train/test JSON, symptom co-occurrence, disease-symptom mapping)
- VGG19 embeddings (`vgg19_image_embeddings.npz`)
- Image files in `/all_images/train/` and `test/`

#### Step 2: Update Configuration

Update paths in `config.py`:

#### Step 3: Verify SRWR Module

Ensure SRWR module is accessible otherwise adjust the imports accordingly.

#### Step 4: Run Diagnosis System

```bash
cd esmmd-diagnosis
python main.py
```

Results will be saved to the path specified in `RESULTS_OUTPUT_PATH` in `config.py`.

---

