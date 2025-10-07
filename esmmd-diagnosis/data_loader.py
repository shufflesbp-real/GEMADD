import json
import pandas as pd
import numpy as np
from collections import defaultdict
from config import *
from image_processor import ImageProcessor

class DataLoader:
    """Loads and prepares all required datasets"""
    
    def __init__(self):
        self.dict_patient_train = {}
        self.dict_patient_test = {}
        self.co_occurrence_dict = {}
        self.disease_symptom_dict = {}
        self.p_d_given_s = defaultdict(dict)
        self.min_scores = {}
        self.symp_to_dis = defaultdict(list)
        self.image_embeddings = None
        self.image_processor = None

    def load_all_data(self):
        """Load all required data files"""
        print("Loading data files...")
        
        # Load JSON files
        self._load_json_files()
        
        # Load CSV files
        self._load_csv_files()
        
        # Build symptom-to-disease mapping
        self._build_symptom_disease_mapping()
        
        # Load image embeddings if available
        try:
            self.image_embeddings = np.load(IMAGE_EMBEDDINGS_PATH)
            print(f"✓ Loaded image embeddings")
        except FileNotFoundError:
            print("⚠ Image embeddings not found, image-based diagnosis will be limited")
        
        print("✓ All data loaded successfully")
        return self
    def initialize_image_processor(self):
        
        self.image_processor = ImageProcessor(
            IMAGE_EMBEDDINGS_PATH,
            self.p_d_given_s
        )
        print("✓ Image processor initialized")
    def is_image_input(self, input_str):
        if not isinstance(input_str, str):
            return False
        input_lower = input_str.strip().lower()
        return input_lower.endswith(('.jpg', '.jpeg', '.png')) or '$' in input_str
    def _load_json_files(self):
        """Load all JSON data files"""
        with open(DICT_TRAIN_PATH, "r") as f:
            self.dict_patient_train = json.load(f)
        print(f"✓ Loaded training data: {len(self.dict_patient_train)} samples")
        
        with open(DICT_TEST_PATH, "r") as f:
            self.dict_patient_test = json.load(f)
        print(f"✓ Loaded test data: {len(self.dict_patient_test)} samples")
        
        with open(SYMPTOM_COOCCURRENCE_PATH, "r") as f:
            self.co_occurrence_dict = json.load(f)
        print(f"✓ Loaded symptom co-occurrence data")
        
        with open(DISEASE_SYMPTOM_PATH) as f:
            self.disease_symptom_dict = json.load(f)
        print(f"✓ Loaded disease-symptom mapping: {len(self.disease_symptom_dict)} diseases")
    
    def _load_csv_files(self):
        """Load CSV files and process P(D|S) probabilities"""
        p_d_given_df = pd.read_csv(P_D_GIVEN_S_PATH)
        p_d_given_df = p_d_given_df.rename(columns={'weight': 'P(D|S)'})
        
        # Build P(D|S) dictionary
        for _, row in p_d_given_df.iterrows():
            symptom = row['symptom'].lower()
            disease = row['disease'].lower()
            prob = row['P(D|S)']
            self.p_d_given_s[symptom][disease] = prob
        
        # Calculate minimum scores per disease
        self.min_scores = p_d_given_df.groupby('disease')['P(D|S)'].min().to_dict()
        
        print(f"✓ Loaded P(D|S) probabilities")
    
    def _build_symptom_disease_mapping(self):
        """Build reverse mapping from symptoms to diseases"""
        for disease, symptoms in self.disease_symptom_dict.items():
            for symptom in symptoms:
                self.symp_to_dis[symptom.lower()].append(disease)
        print(f"✓ Built symptom-to-disease mapping")
    
    def get_patient_data(self, dialogue_id, is_test=True):
        """Get patient data for a specific dialogue ID"""
        data_dict = self.dict_patient_test if is_test else self.dict_patient_train
        return data_dict.get(str(dialogue_id), {})


def compute_symptom_statistics(dict_patient_train):
    """
    Compute symptom sign statistics from training data
    Returns symptom_sign and symptom_disease_stats dictionaries
    """
    symptom_sign = defaultdict(lambda: {"pos": 0, "neg": 0})
    symptom_disease_stats = defaultdict(
        lambda: {"pos": defaultdict(int), "neg": defaultdict(int)}
    )
    
    for k, v in dict_patient_train.items():
        disease = dict_patient_train[k]['ground_truth'].lower()
        
        for sym, val in v.items():
            if sym == 'ground_truth':
                continue
            
            if sym == 'patient_reported_symptoms':
                symptom_sign[val.lower()]['pos'] += 1
                symptom_disease_stats[val.lower()]['pos'][disease] += 1
            else:
                if val == True:
                    symptom_sign[sym.lower()]['pos'] += 1
                    symptom_disease_stats[sym.lower()]['pos'][disease] += 1
                else:
                    symptom_sign[sym.lower()]['neg'] += 1
                    symptom_disease_stats[sym.lower()]['neg'][disease] += 1
    
    print(f"✓ Computed symptom statistics for Gini calculation")
    return symptom_sign, symptom_disease_stats
