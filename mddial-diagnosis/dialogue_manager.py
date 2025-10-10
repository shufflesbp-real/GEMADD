from symptom_selector import get_entropy_based_symptoms
from reasoner import calc_dis_rank_1
from utils import modify_ppr, should_stop_conversation
from graph_builder import create_graph, prune_diseases

# Threshold for the condition in the symptom inquiry where the response of patient is either not sure or the symptom is not present in ground truth conversation.
DONT_KNOW_THRESHOLD = 0.15
EDGE_WEIGHT_THRESHOLD = 0.005

def calculate_avg_cooccurrence(unknown_symptom, yes_symptoms, co_occurrence_dict):
    if not yes_symptoms:
        return 0.0
    
    cooccurrence_scores = []
    
    for yes_symptom in yes_symptoms:
        if yes_symptom in co_occurrence_dict:
            if unknown_symptom in co_occurrence_dict[yes_symptom]:
                cooccurrence_scores.append(co_occurrence_dict[yes_symptom][unknown_symptom])
        
        if unknown_symptom in co_occurrence_dict:
            if yes_symptom in co_occurrence_dict[unknown_symptom]:
                cooccurrence_scores.append(co_occurrence_dict[unknown_symptom][yes_symptom])
    
    if not cooccurrence_scores:
        return 0.0
    
    return sum(cooccurrence_scores)/len(cooccurrence_scores)
    # return max(cooccurrence_scores)

def generate_doctor_response(symptom, asked_symptoms, diseases, co_occurrence_dict, symptom_sign, symptom_disease_stats):
    entropy_syms = get_entropy_based_symptoms(symptom, diseases, asked_symptoms, co_occurrence_dict, symptom_sign, symptom_disease_stats, top_k=10)
    return entropy_syms[0][0] if entropy_syms else None

def generate_patient_response(symptom, dialogue_id, dict_patient):
    dialogue_id = f'{dialogue_id}'
    data = dict_patient.get(dialogue_id, {})
    
    pos_syms = data.get("pos-symptoms", [])
    neg_syms = data.get("neg-symptoms", [])
    
    if symptom in pos_syms:
        return 1
    elif symptom in neg_syms:
        return -1
    else:
        return 0

def generate_responses(G, dialogue_node, patient_symptom, candidate_diseases, possible_set_of_diseases, dialogue_id, gt, data_resources, turns=8):
    (co_occurrence_dict, disease_symptom_dict, symptom_sign,
     symptom_disease_stats, dict_patient, p_d_given_s, min_scores, symp_to_dis) = data_resources
    
    turn_counter = 1
    f = 0
    asked_symptoms = set(patient_symptom)
    yes_symptoms = set(patient_symptom)
    questions_asked = 0
    last_symptom = None  # Track most recently added symptom for pruning
    
    while turn_counter <= turns:
        doctor_response = generate_doctor_response(patient_symptom, asked_symptoms, possible_set_of_diseases, co_occurrence_dict, symptom_sign, symptom_disease_stats)
        
        if not doctor_response:
            break
        
        questions_asked += 1
        asked_symptoms.add(doctor_response)
        patient_has = generate_patient_response(doctor_response, dialogue_id, dict_patient)
        
        if patient_has == 1:  # Patient says "Yes"
            yes_symptoms.add(doctor_response)
            _, G, possible_set_of_diseases = create_graph(
                G, dialogue_node, [doctor_response], 1, turn_counter,
                candidate_diseases, possible_set_of_diseases, 1, gt,
                p_d_given_s, min_scores, symp_to_dis
            )
            last_symptom = doctor_response
            patient_symptom = doctor_response
            
            G, possible_set_of_diseases = prune_diseases(
                G, possible_set_of_diseases, last_symptom, EDGE_WEIGHT_THRESHOLD
            )
            
        elif patient_has == -1:  # Patient says "No"
            _, G, possible_set_of_diseases = create_graph(
                G, dialogue_node, [doctor_response], -1, turn_counter,
                candidate_diseases, possible_set_of_diseases, 2, gt,
                p_d_given_s, min_scores, symp_to_dis
            )
            
        else:  # Patient is not sure (Don't Know)
            avg_cooccur = calculate_avg_cooccurrence(
                doctor_response,
                yes_symptoms,
                co_occurrence_dict
            )
            
            if avg_cooccur > DONT_KNOW_THRESHOLD:
                yes_symptoms.add(doctor_response)
                _, G, possible_set_of_diseases = create_graph(
                    G, dialogue_node, [doctor_response], 1, turn_counter,
                    candidate_diseases, possible_set_of_diseases, 3, gt,
                    p_d_given_s, min_scores, symp_to_dis
                )
                last_symptom = doctor_response
                patient_symptom = doctor_response
                
            else:
                # Treat as negative
                _, G, possible_set_of_diseases = create_graph(
                    G, dialogue_node, [doctor_response], 0, turn_counter,
                    candidate_diseases, possible_set_of_diseases, 3, gt,
                    p_d_given_s, min_scores, symp_to_dis
                )
        
        # Applying PageRank and check for threshold condition if any disease satisfies.
        output_dict = calc_dis_rank_1(G, dialogue_id)
        output_dict = modify_ppr(possible_set_of_diseases, output_dict)
        
        if should_stop_conversation(possible_set_of_diseases, output_dict, turn_counter):
            f = 1
            if len(possible_set_of_diseases) == 1:
                return list(possible_set_of_diseases)[0], list(possible_set_of_diseases), asked_symptoms, questions_asked
            else:
                keys = list(output_dict.keys())
                return keys[0] if keys else "None", keys, asked_symptoms, questions_asked
        
        turn_counter += 1
    
    # Final ranking after all turns
    if f == 0:
        output_dict = calc_dis_rank_1(G, dialogue_id)
        output_dict = modify_ppr(possible_set_of_diseases, output_dict)
        keys = list(output_dict.keys())
        
        if len(keys) > 1:
            return [keys[0], keys[1]], keys, asked_symptoms, questions_asked
        elif len(keys) == 1:
            return keys[0], keys, asked_symptoms, questions_asked
        else:
            return "None", [], asked_symptoms, questions_asked
