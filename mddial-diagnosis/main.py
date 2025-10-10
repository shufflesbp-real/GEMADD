import networkx as nx
import pandas as pd
from collections import defaultdict
from data_loader import load_files
from utils import get_candidate_disease, should_stop_conversation, modify_ppr
from graph_builder import create_graph
from reasoner import calc_dis_rank_1
from dialogue_manager import generate_responses
from evaluation import calculate_and_print_all_metrics
from error_analysis import analyze_diagnosis_errors, print_error_analysis
import time 
from data_loader import load_files
from graph_visualizer import visualize_single_sample_from_dict
from collections import defaultdict

def main():
    (dict_patient_train, dict_patient_test, co_occurrence_dict,symptom_list, disease_symptom_dict, p_d_given_s, min_scores, symp_to_dis) = load_files()
    symptom_sign = defaultdict(lambda: {"pos": 0, "neg": 0})
    symptom_disease_stats = defaultdict(lambda: {"pos": defaultdict(int), "neg": defaultdict(int)})
    for record in dict_patient_train.values():
        disease = record['ground_truth'].lower()
        for symptom in record.get('pos-symptoms', []):
            symptom = symptom.lower()
            symptom_sign[symptom]['pos'] += 1
            symptom_disease_stats[symptom]['pos'][disease] += 1
        for symptom in record.get('neg-symptoms', []):
            symptom = symptom.lower()
            symptom_sign[symptom]['neg'] += 1
            symptom_disease_stats[symptom]['neg'][disease] += 1
    result_columns = [
        'Dialogue_id', 'Ground Truth', 'Predicted Ans', 'Predicted List',
        'Dialogue Length', 'Match Rate (Precision)', 'Recall (vs. KG)',
        'Recall (vs. Patient Yes)'
    ]
    result_df = pd.DataFrame(columns=result_columns)
    failed_dialogues = []
    data_resources = (co_occurrence_dict, disease_symptom_dict, symptom_sign,,symptom_disease_stats, dict_patient_test, p_d_given_s, min_scores, symp_to_dis)
    start_time = time.time()
    for k, v in dict_patient_test.items():
        try:
            dialogue_id = k
            # print(f"Processing dialogue_id {dialogue_id}")
            patient_symptom = v['patient_reported_symptoms']
            new_graph = nx.DiGraph()
            dialogue_node = f'user query_{dialogue_id}'
            ground_truth = v['ground_truth']
            candidate_diseases = get_candidate_disease(patient_symptom, symp_to_dis)
            possible_set_of_diseases = []
            _, new_graph, possible_set_of_diseases = create_graph(
                new_graph, dialogue_node, patient_symptom, 1, 0,
                candidate_diseases, possible_set_of_diseases, 1, ground_truth,
                p_d_given_s, min_scores, symp_to_dis
            )
            output = calc_dis_rank_1(new_graph, dialogue_id)
            output = modify_ppr(possible_set_of_diseases, output)
            asked_symptoms_by_system = set(patient_symptom)
            dialogue_length = 0 # Default length is 0 if no questions are asked

            if should_stop_conversation(possible_set_of_diseases, output, 0):
                predicted = list(output.keys())[0] if output else "None"
                predicted_list = list(output.keys())
            else:
                # We had set max turns for any conversation turn to 8.
                predicted, predicted_list, asked_symptoms_by_system, dialogue_length = generate_responses(
                    new_graph, dialogue_node, patient_symptom, candidate_diseases,
                    possible_set_of_diseases, dialogue_id, ground_truth, data_resources, turns=8
                )

            x1 = set(s.lower() for s in disease_symptom_dict.get(ground_truth.capitalize(), []))
            x2_yes = set(s.lower() for s in v.get('pos-symptoms', []))
            x3 = asked_symptoms_by_system
            intersection_x1_x3 = x1.intersection(x3)
            intersection_x2yes_x3 = x2_yes.intersection(x3)
            precision = len(intersection_x1_x3) / len(x3) if len(x3) > 0 else 0
            recall_kg = len(intersection_x1_x3) / len(x1) if len(x1) > 0 else 0
            recall_patient = len(intersection_x2yes_x3) / len(x2_yes) if len(x2_yes) > 0 else 0
            result_df.loc[len(result_df)] = [
                dialogue_id, ground_truth, predicted, predicted_list,
                dialogue_length, precision, recall_kg, recall_patient
            ]
        except Exception as e:
            print(f"Skipping dialogue_id {dialogue_id} due to error: {e}")
            failed_dialogues.append(dialogue_id)
    end_time = time.time()  
    elapsed_time = end_time - start_time
    print(f"\nTime taken to process all test dialogues: {elapsed_time:.2f} seconds")
    calculate_and_print_all_metrics(result_df)
    if failed_dialogues:
        print(f"\nCould not process {len(failed_dialogues)} dialogues: {failed_dialogues}")
        
def visualize_sample_graph():
    (dict_patient_train, dict_patient_test, co_occurrence_dict,symptom_list, disease_symptom_dict, p_d_given_s, min_scores, symp_to_dis) = load_files()
    symptom_sign = defaultdict(lambda: {"pos": 0, "neg": 0})
    symptom_disease_stats = defaultdict(lambda: {"pos": defaultdict(int), "neg": defaultdict(int)})
    for record in dict_patient_train.values():
        disease = record['ground_truth'].lower()
        for symptom in record.get('pos-symptoms', []):
            symptom = symptom.lower()
            symptom_sign[symptom]['pos'] += 1
            symptom_disease_stats[symptom]['pos'][disease] += 1
        for symptom in record.get('neg-symptoms', []):
            symptom = symptom.lower()
            symptom_sign[symptom]['neg'] += 1
            symptom_disease_stats[symptom]['neg'][disease] += 1
    data_resources = (co_occurrence_dict, disease_symptom_dict, symptom_sign,symptom_disease_stats, dict_patient_test, p_d_given_s, min_scores, symp_to_dis)
    sample = {
    "ground_truth": "Asthma",
    "pos-symptoms": [
      "cough",
      "chest tightness",
      "shortness of breath"
    ],
    "neg-symptoms": [
      "stuffy nose",
      "runny nose"
    ]
  }    
    G, top_diseases = visualize_single_sample_from_dict(sample, "6", data_resources)
def do_error_analysis():
    (dict_patient_train, dict_patient_test, co_occurrence_dict,symptom_list, disease_symptom_dict, p_d_given_s, min_scores, symp_to_dis) = load_files()
    symptom_sign = defaultdict(lambda: {"pos": 0, "neg": 0})
    symptom_disease_stats = defaultdict(lambda: {"pos": defaultdict(int), "neg": defaultdict(int)})
    for record in dict_patient_train.values():
        disease = record['ground_truth'].lower()
        for symptom in record.get('pos-symptoms', []):
            symptom = symptom.lower()
            symptom_sign[symptom]['pos'] += 1
            symptom_disease_stats[symptom]['pos'][disease] += 1
        for symptom in record.get('neg-symptoms', []):
            symptom = symptom.lower()
            symptom_sign[symptom]['neg'] += 1
            symptom_disease_stats[symptom]['neg'][disease] += 1
    result_columns = [
        'Dialogue_id', 'Ground Truth', 'Predicted Ans', 'Predicted List',
        'Dialogue Length', 'Match Rate (Precision)', 'Recall (vs. KG)',
        'Recall (vs. Patient Yes)'
    ]
    result_df = pd.DataFrame(columns=result_columns)
    failed_dialogues = []
    error_tracking_data = []
    data_resources = (co_occurrence_dict, disease_symptom_dict, symptom_sign,symptom_disease_stats, dict_patient_test, p_d_given_s, min_scores, symp_to_dis)
    start_time = time.time()
    for k, v in dict_patient_test.items():
        try:
            dialogue_id = k
            print(f"Processing dialogue_id {dialogue_id}")
            patient_symptom = v['patient_reported_symptoms']
            new_graph = nx.DiGraph()
            dialogue_node = f'user query_{dialogue_id}'
            ground_truth = v['ground_truth'].lower()
            candidate_diseases = get_candidate_disease(patient_symptom, symp_to_dis)
            was_in_candidates = ground_truth in candidate_diseases
            initial_candidates = candidate_diseases.copy()
            possible_set_of_diseases = []
            _, new_graph, possible_set_of_diseases = create_graph(
                new_graph, dialogue_node, patient_symptom, 1, 0,
                candidate_diseases, possible_set_of_diseases, 1, ground_truth,
                p_d_given_s, min_scores, symp_to_dis
            )
            still_in_possible_after_init = ground_truth in possible_set_of_diseases
            output = calc_dis_rank_1(new_graph, dialogue_id)
            output = modify_ppr(possible_set_of_diseases, output)
            asked_symptoms_by_system = set(patient_symptom)
            dialogue_length = 0
            if should_stop_conversation(possible_set_of_diseases, output, 0):
                predicted = list(output.keys())[0] if output else "None"
                predicted_list = list(output.keys())
            else:
                possible_before_dialogue = possible_set_of_diseases.copy()
                predicted, predicted_list, asked_symptoms_by_system, dialogue_length = generate_responses(
                    new_graph, dialogue_node, patient_symptom, candidate_diseases,
                    possible_set_of_diseases, dialogue_id, ground_truth, data_resources, turns=8
                )
               #checking if ground truth was pruned during dialogue
                was_pruned = (ground_truth in possible_before_dialogue) and (ground_truth not in predicted_list)
            error_type = None
            final_rank = -1
            if isinstance(predicted, str):
                is_correct = (predicted.lower() == ground_truth)
            else:
                is_correct = (predicted[0].lower() == ground_truth if predicted else False)
            if not is_correct:
                if ground_truth in predicted_list:
                    final_rank = predicted_list.index(ground_truth)
                if not was_in_candidates:
                    error_type = 'not_in_candidates'
                elif was_in_candidates and not still_in_possible_after_init:
                    error_type = 'got_pruned'
                elif was_in_candidates and still_in_possible_after_init and ground_truth not in predicted_list:
                    error_type = 'got_pruned'
                elif ground_truth in predicted_list and final_rank > 0:
                    error_type = 'present_but_not_top1'
                else:
                    error_type = 'got_pruned'
            error_tracking_data.append({
                'dialogue_id': dialogue_id,
                'was_in_candidates': was_in_candidates,
                'was_pruned': (was_in_candidates and ground_truth not in predicted_list),
                'final_rank': final_rank,
                'error_type': error_type
            })
            x1 = set(s.lower() for s in disease_symptom_dict.get(ground_truth.capitalize(), []))
            x2_yes = set(s.lower() for s in v.get('pos-symptoms', []))
            x3 = asked_symptoms_by_system
            intersection_x1_x3 = x1.intersection(x3)
            intersection_x2yes_x3 = x2_yes.intersection(x3)
            precision = len(intersection_x1_x3) / len(x3) if len(x3) > 0 else 0
            recall_kg = len(intersection_x1_x3) / len(x1) if len(x1) > 0 else 0
            recall_patient = len(intersection_x2yes_x3) / len(x2_yes) if len(x2_yes) > 0 else 0
            result_df.loc[len(result_df)] = [
                dialogue_id, ground_truth, predicted, predicted_list,
                dialogue_length, precision, recall_kg, recall_patient
            ]
        except Exception as e:
            print(f"Skipping dialogue_id {dialogue_id} due to error: {e}")
            failed_dialogues.append(dialogue_id)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"\nTime taken to process all test dialogues: {elapsed_time:.2f} seconds")
    calculate_and_print_all_metrics(result_df)
    error_analysis = analyze_diagnosis_errors(result_df, error_tracking_data)
    print_error_analysis(error_analysis)
    if failed_dialogues:
        print(f"\nCould not process {len(failed_dialogues)} dialogues: {failed_dialogues}")
if __name__ == '__main__':
    import sys
    # if len(sys.argv) > 1 and sys.argv[1] == '--visualize':
    #     visualize_sample_graph()
    # else:
    #     main()
    main()
    # do_error_analysis()
