# GEMADD
# GEMADD: A Graph-based Explainable Multimodal Approach for Disease Diagnosis


## Table of Contents

- [Requirements](#requirements)
- [Installation](#installation)
- [Datasets](#datasets)
- [References](#references)
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


### ES-MMD Dataset

> **Important:** The ES-MMD dataset is not publicly available. Access must be requested from the authors.

**For access, please refer to:** [KI-MMDG GitHub Repository](https://github.com/NLP-RL/KI-MMDG)

For reproducibility, sample preprocessed files are provided in `data-files/esmmd/` (files starting with `sample_*`).
---

### Running on MDDIAL Dataset

1. **Update paths in `config.py`** if needed (default paths work if running from main directory)

2. **Run the main script:**
```bash
python main.py
```

**Alternative:** If running from a different directory, update the paths in `data_loader.py` accordingly.

### Running on ES-MMD Dataset
#### Data Preprocessing

After obtaining the ES-MMD dataset, you need to preprocess the original CSV files into the required JSON format.

#### Step 1: Create Training and Test JSON Files

From the original ES-MMD CSV file, extract the following information for each dialogue:

1. **Dialogue ID** - Unique identifier for each conversation
2. **Symptoms** - Extract from BIO-tagged format
3. **Intent** - Determine if symptom is affirmed (true) or denied (false)
4. **Disease** - Final diagnosis from the dataset
5. **Patient Reported Symptom** - Initial symptom that started the dialogue

**Required JSON Format:**
```
{
"12491": {
"spots or clouds in vision": true,
"symptoms of eye": true,
"patient_reported_symptoms": "elbow pain",
"ground_truth": "Central retinal artery or vein occlusion"
},
"12492": {
"skin rash": true,
"itching of skin": false,
"patient_reported_symptoms": "skin lesion",
"ground_truth": "Acne"
}
}
```


**Key Format Rules:**
- **Keys:** Dialogue IDs (as strings)
- **Values:** Dictionary containing:
  - **Symptom keys:** Symptom names in **lowercase**
  - **Symptom values:** `true` if Intent is "Affirmative", `false` if "Negative"
  - **patient_reported_symptoms:** Initial symptom (lowercase)
  - **ground_truth:** Disease name (original case from dataset)

**Data Split:**
- **Training Set:** 70% of dialogues → Save as `dict_train_esmmd.json`
- **Test Set:** 30% of dialogues → Save as `dict_test_esmmd.json`
###### Create Disease-Symptom Mapping

Create `disease_symptom_esmmd.json` containing all possible symptoms for each disease.

**Required Format:**
```
{
"Acne": [
"skin rash",
"acne or pimples",
"skin lesion",
"abnormal appearing skin"
],
"Diabetes": [
"increased thirst",
"frequent urination",
"fatigue"
]
}
```
**Key Format Rules:**
- **Keys:** Disease names (exact match with ground_truth values)
- **Values:** List of all possible symptoms (lowercase)

**Generation Method:**
- Aggregate all symptoms associated with each disease across the entire dataset
- Remove duplicates
- Ensure symptom names are in lowercase

Refer to `sample_disease_symptom_esmmd.json` in the repository for the complete format.

#### Step 2: Prepare Data Files

Ensure all data files are in place:
- ES-MMD dataset files (train/test JSON, symptom co-occurrence, disease-symptom mapping)
- VGG19 embeddings (`vgg19_image_embeddings.npz`)
- Image files in `/all_images/train/` and `test/`

#### Step 3: Update Configuration

Update paths in `config.py`:

#### Step 4: Verify SRWR Module

Ensure SRWR module is accessible otherwise adjust the imports accordingly.

#### Step 5: Run Diagnosis System

```bash
cd esmmd-diagnosis
python main.py
```

Results will be saved to the path specified in `RESULTS_OUTPUT_PATH` in `config.py`.

---


## References

- Tiwari, A., Bera, S., Verma, P., Manthena, J. V., Saha, S., Bhattacharyya, P., Dhar, M., & Tiwari, S. (2024). *Seeing Is Believing! Towards Knowledge-Infused Multi-modal Medical Dialogue Generation*. In N. Calzolari, M.-Y. Kan, V. Hoste, A. Lenci, S. Sakti, & N. Xue (Eds.), Proceedings of the 2024 Joint International Conference on Computational Linguistics, Language Resources and Evaluation (LREC-COLING 2024) (pp. 14513–14523). ELRA and ICCL. [https://aclanthology.org/2024.lrec-main.1264](https://aclanthology.org/2024.lrec-main.1264)

- Macherla, S., Luo, M., Parmar, M., & Baral, C. (2023). *MDDial: A Multi-turn Differential Diagnosis Dialogue Dataset with Reliability Evaluation*. arXiv preprint arXiv:2308.08147. [https://arxiv.org/abs/2308.08147](https://arxiv.org/abs/2308.08147)


