import json
import pandas as pd
from collections import defaultdict

def load_files():
    """
    Load all the necessary data files.
    """
    with open("/data-files/mddial/dict_patient_train.json", "r") as f:
        dict_patient_train = json.load(f)
    with open("/data-files/mddial/dict_patient_test.json", "r") as f:
        dict_patient_test = json.load(f)
    with open("/data-files/mddial/symptom_co-occurence.json", "r") as f:
        co_occurrence_dict = json.load(f)
    with open('/data-files/mddial/symptom.txt', 'r') as file:
        symptom_list = [line.strip().lower() for line in file]
    with open("/data-files/mddial/disease_symptoms.txt") as f:
        disease_symptom_dict = json.load(f)

    p_d_given_df = pd.read_csv("/data-files/mddial/conditional_probabilities_P_D_given_S.csv")
    p_d_given_s = defaultdict(dict)
    for _, row in p_d_given_df.iterrows():
        symptom = row['symptom']
        disease = row['disease']
        prob = row['P(D|S)']
        p_d_given_s[symptom][disease] = prob

    min_scores_df = pd.read_csv("/data-files/mddial/Mddial_wts_2.csv")
    min_scores_df = min_scores_df.rename(columns={'weight': 'P(D|S)'})
    min_scores = min_scores_df.groupby('disease')['P(D|S)'].min().to_dict()
    min_scores = {k.capitalize(): v for k, v in min_scores.items()}


    symp_to_dis = defaultdict(list)
    for disease, symptoms in disease_symptom_dict.items():
        for symptom in symptoms:
            symp_to_dis[symptom.lower()].append(disease)

    return (dict_patient_train, dict_patient_test, co_occurrence_dict,
            symptom_list, disease_symptom_dict, p_d_given_s, min_scores, symp_to_dis)
