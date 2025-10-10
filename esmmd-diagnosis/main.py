import pandas as pd
from data_loader import DataLoader, compute_symptom_statistics
from graph_builder import GraphBuilder
from symptom_selector import SymptomSelector
from pagerank_calculator import PageRankCalculator
from diagnosis_system import DiagnosisSystem
from metrics import DiagnosisMetrics
from config import *
import time


def get_candidate_diseases(symptoms, symp_to_dis):
    common_dis = []
    for symptom in symptoms:
        if symptom in symp_to_dis:
            common_dis.append(set(symp_to_dis[symptom]))
    
    if not common_dis:
        return []
    return list(set.union(*common_dis))

def get_patient_yes_symptoms(patient_data):
    yes_symptoms = set()
    for key, val in patient_data.items():
        if key == 'ground_truth':
            continue
        if key == 'patient_reported_symptoms':
            if not (isinstance(val, str) and (val.startswith('$') or val.lower().endswith(('.jpg', '.jpeg', '.png')))):
                yes_symptoms.add(val.lower())
        elif val == True:
            yes_symptoms.add(key.lower())
    return yes_symptoms


def main():
    start_time = time.time()
    data_loader = DataLoader()
    data_loader.load_all_data()
    data_loader.initialize_image_processor()
    symptom_sign, symptom_disease_stats = compute_symptom_statistics(
        data_loader.dict_patient_train
    )
    graph_builder = GraphBuilder(
        data_loader.p_d_given_s,
        data_loader.min_scores,
        data_loader.symp_to_dis,
        data_loader.disease_symptom_dict
    )
    symptom_selector = SymptomSelector(
        data_loader.co_occurrence_dict,
        symptom_sign,
        symptom_disease_stats
    )
    pagerank_calc = PageRankCalculator()
    diagnosis_system = DiagnosisSystem(
        graph_builder,
        symptom_selector,
        pagerank_calc,
        data_loader
    )
    metrics = DiagnosisMetrics()
    results_list = []
    test_ids = list(data_loader.dict_patient_test.keys())
    total = len(test_ids)
    image_count = 0
    text_count = 0
    skipped_count = 0
    for idx, dialogue_id in enumerate(test_ids):
        if (idx + 1) % 10 == 0:
            print(f"  Progress: {idx+1}/{total} ({100*(idx+1)/total:.1f}%) | Images: {image_count}, Text: {text_count}")  
        patient_data = data_loader.get_patient_data(dialogue_id, is_test=True)
        if not patient_data or 'ground_truth' not in patient_data:
            continue
        ground_truth = patient_data['ground_truth']
        initial_input = None
        for key, val in patient_data.items():
            if key == 'patient_reported_symptoms':
                initial_input = val
                break
        if not initial_input:
            continue
        is_image = data_loader.is_image_input(initial_input)
        try:
            if is_image:
                image_count += 1
                image_disease_dict = data_loader.image_processor.image_to_dis(initial_input)
                if not image_disease_dict:
                    skipped_count += 1
                    continue
                    
                predicted, top_k_preds, dialogue_length, asked_symptoms = diagnosis_system.run_dialogue_from_image(
                    dialogue_id,
                    initial_input,
                    ground_truth,
                    image_disease_dict,
                    patient_data
                )
            else:
                text_count += 1
                initial_symptoms = [initial_input.lower()]
                candidate_diseases = get_candidate_diseases(initial_symptoms, data_loader.symp_to_dis)
                
                if not candidate_diseases:
                    continue
                    
                predicted, top_k_preds, dialogue_length, asked_symptoms = diagnosis_system.run_dialogue(
                    dialogue_id,
                    initial_symptoms,
                    ground_truth,
                    candidate_diseases,
                    patient_data
                )
            symptoms_gt = set(data_loader.disease_symptom_dict.get(ground_truth.lower(), []))
            symptoms_patient_yes = get_patient_yes_symptoms(patient_data)
            metrics.add_result(
                dialogue_id,
                ground_truth,
                predicted,
                dialogue_length,
                symptoms_gt,
                asked_symptoms,
                symptoms_patient_yes,
                top_k_preds
            )
            results_list.append({
                'dialogue_id': dialogue_id,
                'input_type': 'image' if is_image else 'text',
                'initial_input': initial_input,
                'ground_truth': ground_truth.lower(),
                'predicted': predicted if isinstance(predicted, str) else str(predicted),
                'top_3_predictions': str(top_k_preds[:3]),
                'dialogue_length': dialogue_length,
                'correct': ground_truth.lower() == (predicted.lower() if isinstance(predicted, str) else (predicted[0].lower() if isinstance(predicted, list) else "")),
                'num_symptoms_asked': len(asked_symptoms),
                'symptom_precision': len(symptoms_gt.intersection(asked_symptoms)) / len(asked_symptoms) if len(asked_symptoms) > 0 else 0,
                'symptom_recall': len(symptoms_gt.intersection(asked_symptoms)) / len(symptoms_gt) if len(symptoms_gt) > 0 else 0
            })
            
        except Exception as e:
            print(f" Error processing dialogue {dialogue_id}: {e}")
            skipped_count += 1
            continue

    end_time = time.time()
    elapsed_time = end_time - start_time
    
    # print(f"\n  Summary: {text_count} text, {image_count} images, {skipped_count} skipped")
    metrics.print_metrics(elapsed_time=elapsed_time)
    results_df = pd.DataFrame(results_list)
    results_df.to_csv(RESULTS_OUTPUT_PATH, index=False)
    print(f"\nResults saved to: {RESULTS_OUTPUT_PATH}")
    print("\n" + "="*60)
    print("COMPLETED")
    print("="*60)

def main_image_only():
    start_time = time.time()
    data_loader = DataLoader()
    data_loader.load_all_data()
    data_loader.initialize_image_processor()
    symptom_sign, symptom_disease_stats = compute_symptom_statistics(
        data_loader.dict_patient_train
    )
    graph_builder = GraphBuilder(
        data_loader.p_d_given_s,
        data_loader.min_scores,
        data_loader.symp_to_dis,
        data_loader.disease_symptom_dict
    )
    symptom_selector = SymptomSelector(
        data_loader.co_occurrence_dict,
        symptom_sign,
        symptom_disease_stats
    )
    pagerank_calc = PageRankCalculator()
    diagnosis_system = DiagnosisSystem(
        graph_builder,
        symptom_selector,
        pagerank_calc,
        data_loader
    )
    metrics = DiagnosisMetrics()
    results_list = []
    test_ids = list(data_loader.dict_patient_test.keys())
    image_dialogue_ids = []
    for dialogue_id in test_ids:
        patient_data = data_loader.get_patient_data(dialogue_id, is_test=True)
        if not patient_data or 'ground_truth' not in patient_data:
            continue
        initial_input = None
        for key, val in patient_data.items():
            if key == 'patient_reported_symptoms':
                initial_input = val
                break
        if initial_input and data_loader.is_image_input(initial_input):
            image_dialogue_ids.append(dialogue_id)
    total = len(image_dialogue_ids)
    image_count = 0
    skipped_count = 0
    for idx, dialogue_id in enumerate(image_dialogue_ids):
        if (idx + 1) % 10 == 0:
            print(f"  Progress: {idx+1}/{total} ({100*(idx+1)/total:.1f}%) | Processed: {image_count}, Skipped: {skipped_count}")
        patient_data = data_loader.get_patient_data(dialogue_id, is_test=True)
        ground_truth = patient_data['ground_truth']
        initial_input = None
        for key, val in patient_data.items():
            if key == 'patient_reported_symptoms':
                initial_input = val
                break
        try:
            image_count += 1
            image_disease_dict = data_loader.image_processor.image_to_dis(initial_input)
            if not image_disease_dict:
                print(f" No diseases found for image, skipping...")
                skipped_count += 1
                continue
            predicted, top_k_preds, dialogue_length, asked_symptoms = diagnosis_system.run_dialogue_from_image(
                dialogue_id,
                initial_input,
                ground_truth,
                image_disease_dict,
                patient_data
            )
            symptoms_gt = set(data_loader.disease_symptom_dict.get(ground_truth.lower(), []))
            symptoms_patient_yes = get_patient_yes_symptoms(patient_data)
            metrics.add_result(
                dialogue_id,
                ground_truth,
                predicted,
                dialogue_length,
                symptoms_gt,
                asked_symptoms,
                symptoms_patient_yes,
                top_k_preds
            )
            results_list.append({
                'dialogue_id': dialogue_id,
                'input_type': 'image',
                'initial_input': initial_input,
                'ground_truth': ground_truth.lower(),
                'predicted': predicted if isinstance(predicted, str) else str(predicted),
                'top_3_predictions': str(top_k_preds[:3]),
                'dialogue_length': dialogue_length,
                'correct': ground_truth.lower() == (predicted.lower() if isinstance(predicted, str) else (predicted[0].lower() if isinstance(predicted, list) else "")),
                'num_symptoms_asked': len(asked_symptoms),
                'symptom_precision': len(symptoms_gt.intersection(asked_symptoms)) / len(asked_symptoms) if len(asked_symptoms) > 0 else 0,
                'symptom_recall': len(symptoms_gt.intersection(asked_symptoms)) / len(symptoms_gt) if len(symptoms_gt) > 0 else 0
            })
        except Exception as e:
            print(f"Error processing dialogue {dialogue_id}: {e}")
            skipped_count += 1
            continue
    end_time = time.time()
    elapsed_time = end_time - start_time
    metrics.print_metrics(elapsed_time=elapsed_time)
    results_df = pd.DataFrame(results_list)
    image_only_output = RESULTS_OUTPUT_PATH.replace('.csv', '_image_only.csv')
    results_df.to_csv(image_only_output, index=False)
    print(f"\nResults saved to: {image_only_output}")
if __name__ == "__main__":
    # main()
    main_image_only()
