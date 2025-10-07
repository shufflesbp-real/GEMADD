# """
# Main Diagnosis System
# Implements Algorithm 1: Iterative Disease Diagnosis
# """

# import networkx as nx
# from config import *


# class DiagnosisSystem:
#     """Main system orchestrating the diagnosis process"""
    
#     def __init__(self, graph_builder, symptom_selector, pagerank_calc, data_loader):
#         self.graph_builder = graph_builder
#         self.symptom_selector = symptom_selector
#         self.pagerank_calc = pagerank_calc
#         self.data_loader = data_loader
    
#     def should_stop_conversation(self, possible_diseases, ppr_output, turns):
#         """Check stopping criteria (Algorithm 1, line 15)"""
#         if len(possible_diseases) == 1:
#             return True
        
#         keys = list(ppr_output.keys())
#         if len(keys) == 1:
#             return True
        
#         if len(keys) >= 2:
#             first_dis = keys[0]
#             second_dis = keys[1]
#             score_diff = (ppr_output[first_dis] - ppr_output[second_dis]) / ppr_output[first_dis]
#             if score_diff > RANK_THRESHOLD:
#                 return True
        
#         return False
    
#     def modify_ppr(self, possible_diseases, ppr_output):
#         """Filter PPR output to only include possible diseases"""
#         for key in list(ppr_output.keys()):
#             if key not in possible_diseases:
#                 ppr_output.pop(key)
        
#         for k, v in ppr_output.items():
#             ppr_output[k] = round(v, 3)
        
#         return ppr_output
    
#     def generate_patient_response(self, symptom, dialogue_id, patient_data):
#         """Simulate patient response (Algorithm 1, lines 24-32)"""
#         if symptom in patient_data:
#             if patient_data[symptom] == True:
#                 return 1
#             elif patient_data[symptom] == False:
#                 return -1
#             else:
#                 return 0
#         else:
#             return 0
    
#     def calculate_avg_cooccurrence(self, unknown_symptom, yes_symptoms, co_occurrence_dict):
#         """
#         Calculate average co-occurrence between unknown symptom and all "Yes" symptoms
        
#         Args:
#             unknown_symptom: The symptom patient said "Don't Know" to
#             yes_symptoms: Set of symptoms patient confirmed as "Yes"
#             co_occurrence_dict: Symptom co-occurrence dictionary
        
#         Returns:
#             Average co-occurrence score (0.0 if no data available)
#         """
#         if not yes_symptoms:
#             return 0.0
        
#         cooccurrence_scores = []
        
#         for yes_symptom in yes_symptoms:
#             # Check co-occurrence in both directions
#             if yes_symptom in co_occurrence_dict:
#                 if unknown_symptom in co_occurrence_dict[yes_symptom]:
#                     cooccurrence_scores.append(co_occurrence_dict[yes_symptom][unknown_symptom])
            
#             if unknown_symptom in co_occurrence_dict:
#                 if yes_symptom in co_occurrence_dict[unknown_symptom]:
#                     cooccurrence_scores.append(co_occurrence_dict[unknown_symptom][yes_symptom])
        
#         if not cooccurrence_scores:
#             return 0.0
    
#         return sum(cooccurrence_scores) / len(cooccurrence_scores)
    
#     def run_dialogue(self, dialogue_id, initial_symptoms, ground_truth, 
#                     candidate_diseases, patient_data, max_turns=MAX_DIALOGUE_TURNS):
#         """Run complete diagnosis dialogue (Algorithm 1, main loop)"""
#         G = nx.DiGraph()
#         dialogue_node = f"user query_{dialogue_id}"
        
#         asked_symptoms = set(initial_symptoms)
#         yes_symptoms = set(initial_symptoms)  # FIXED: Track yes symptoms properly
        
#         candidate_diseases, G, possible_diseases = self.graph_builder.create_graph(
#             G, dialogue_node, initial_symptoms, 1, 0, 
#             candidate_diseases, set(candidate_diseases), 1, ground_truth.lower()
#         )
        
#         current_symptom = initial_symptoms[0] if initial_symptoms else None
#         turn = 1
#         output_dict = {}
        
#         while turn <= max_turns:
#             entropy_syms = self.symptom_selector.get_entropy_based_symptoms(
#                 current_symptom, possible_diseases, asked_symptoms
#             )
            
#             if not entropy_syms:
#                 break
            
#             doctor_symptom = entropy_syms[0][0]
#             asked_symptoms.add(doctor_symptom)
            
#             patient_response = self.generate_patient_response(
#                 doctor_symptom, dialogue_id, patient_data
#             )
            
#             if patient_response == 1:
#                 yes_symptoms.add(doctor_symptom)  # FIXED
#                 candidate_diseases, G, possible_diseases = self.graph_builder.create_graph(
#                     G, dialogue_node, [doctor_symptom], 1, turn,
#                     candidate_diseases, possible_diseases, 1, ground_truth.lower()
#                 )
#                 current_symptom = doctor_symptom
                
#             elif patient_response == -1:
#                 candidate_diseases, G, possible_diseases = self.graph_builder.create_graph(
#                     G, dialogue_node, [doctor_symptom], -1, turn,
#                     candidate_diseases, possible_diseases, 2, ground_truth.lower()
#                 )
            
#             else: 
#                 avg_cooccur = self.calculate_avg_cooccurrence(
#                     doctor_symptom, 
#                     yes_symptoms,
#                     self.data_loader.co_occurrence_dict
#                 )
#                 if avg_cooccur > DONT_KNOW_THRESHOLD:
#                     response_val = 1
#                     yes_symptoms.add(doctor_symptom)  # FIXED
#                 else:
#                     response_val = -1  # FIXED: Should be -1, not 0
                
#                 candidate_diseases, G, possible_diseases = self.graph_builder.create_graph(
#                     G, dialogue_node, [doctor_symptom], response_val, turn,
#                     candidate_diseases, possible_diseases, 3, ground_truth.lower()
#                 )
            
#             output_dict = self.pagerank_calc.calculate_disease_rank(G, dialogue_id)
#             output_dict = self.modify_ppr(possible_diseases, output_dict)
            
#             if self.should_stop_conversation(possible_diseases, output_dict, turn):
#                 if len(possible_diseases) == 1:
#                     predicted = list(possible_diseases)[0]
#                 else:
#                     predicted = list(output_dict.keys())[0] if output_dict else None
                
#                 return predicted, list(output_dict.keys()), turn, asked_symptoms
            
#             turn += 1
        
#         if not output_dict:
#             output_dict = self.pagerank_calc.calculate_disease_rank(G, dialogue_id)
        
#         keys = list(output_dict.keys())
#         if len(keys) >= 2:
#             predicted = [keys[0], keys[1]]
#         elif len(keys) == 1:
#             predicted = keys[0]
#         else:
#             predicted = None
        
#         return predicted, list(output_dict.keys()), turn-1, asked_symptoms
    
#     def run_dialogue_from_image(self, dialogue_id, image_filename, ground_truth, 
#                                 image_disease_dict, patient_data, max_turns=MAX_DIALOGUE_TURNS):
#         """
#         Run diagnosis dialogue starting from image input
        
#         Uses REAL symptoms from CLIP instead of fake symptom name
#         """
#         G = nx.DiGraph()
#         dialogue_node = f"user query_{dialogue_id}"
        
#         # Extract REAL visual symptoms from image using CLIP
#         visual_symptoms = self.data_loader.image_processor.img_to_sym(image_filename)
        
#         if not visual_symptoms:
#             symptom_name = f"visual_symptom_{image_filename}"
#             visual_symptoms = [symptom_name]
        
#         # Use the first visual symptom (e.g., "skin rash")
#         initial_symptom = visual_symptoms[0].lower()
        
#         asked_symptoms = set([initial_symptom])
#         yes_symptoms = set([initial_symptom])
        
#         # Create graph from image diseases
#         candidate_diseases, G, possible_diseases = self.graph_builder.create_graph_from_image(
#             G, dialogue_node, initial_symptom, image_disease_dict, set()
#         )
        
#         # Convert to list
#         candidate_diseases = [d.lower() for d in candidate_diseases]
#         possible_diseases = [d.lower() for d in possible_diseases]
        
#         turn = 1
#         output_dict = {}
#         current_symptom = initial_symptom
        
#         # Continue with text-based dialogue
#         while turn <= max_turns:
#             entropy_syms = self.symptom_selector.get_entropy_based_symptoms(
#                 current_symptom, possible_diseases, asked_symptoms
#             )
            
#             if not entropy_syms:
#                 break
            
#             doctor_symptom = entropy_syms[0][0]
#             asked_symptoms.add(doctor_symptom)
            
#             patient_response = self.generate_patient_response(
#                 doctor_symptom, dialogue_id, patient_data
#             )
            
#             if patient_response == 1:
#                 yes_symptoms.add(doctor_symptom)
#                 candidate_diseases, G, possible_diseases = self.graph_builder.create_graph(
#                     G, dialogue_node, [doctor_symptom], 1, turn,
#                     candidate_diseases, possible_diseases, 1, ground_truth.lower()
#                 )
#                 current_symptom = doctor_symptom
                
#             elif patient_response == -1:
#                 candidate_diseases, G, possible_diseases = self.graph_builder.create_graph(
#                     G, dialogue_node, [doctor_symptom], -1, turn,
#                     candidate_diseases, possible_diseases, 2, ground_truth.lower()
#                 )
            
#             else:  # Don't know
#                 avg_cooccur = self.calculate_avg_cooccurrence(
#                     doctor_symptom, yes_symptoms, self.data_loader.co_occurrence_dict
#                 )
                
#                 if avg_cooccur > DONT_KNOW_THRESHOLD:
#                     response_val = 1
#                     yes_symptoms.add(doctor_symptom)
#                 else:
#                     response_val = 0
                
#                 candidate_diseases, G, possible_diseases = self.graph_builder.create_graph(
#                     G, dialogue_node, [doctor_symptom], response_val, turn,
#                     candidate_diseases, possible_diseases, 3, ground_truth.lower()
#                 )
            
#             output_dict = self.pagerank_calc.calculate_disease_rank(G, dialogue_id)
#             output_dict = self.modify_ppr(possible_diseases, output_dict)
            
#             if self.should_stop_conversation(possible_diseases, output_dict, turn):
#                 if len(possible_diseases) == 1:
#                     predicted = list(possible_diseases)[0]
#                 else:
#                     predicted = list(output_dict.keys())[0] if output_dict else None
                
#                 return predicted, list(output_dict.keys()), turn, asked_symptoms
            
#             turn += 1
        
#         if not output_dict:
#             output_dict = self.pagerank_calc.calculate_disease_rank(G, dialogue_id)
        
#         keys = list(output_dict.keys())
#         if len(keys) >= 2:
#             predicted = [keys[0], keys[1]]
#         elif len(keys) == 1:
#             predicted = keys[0]
#         else:
#             predicted = None
        
#         return predicted, list(output_dict.keys()), turn-1, asked_symptoms

"""
Main Diagnosis System
Implements Algorithm 1: Iterative Disease Diagnosis
"""

import networkx as nx
from config import *


class DiagnosisSystem:
    """Main system orchestrating the diagnosis process"""
    
    def __init__(self, graph_builder, symptom_selector, pagerank_calc, data_loader):
        self.graph_builder = graph_builder
        self.symptom_selector = symptom_selector
        self.pagerank_calc = pagerank_calc
        self.data_loader = data_loader
    
    def should_stop_conversation(self, possible_diseases, ppr_output, turns):
        """Check stopping criteria (Algorithm 1, line 15)"""
        if len(possible_diseases) == 1:
            return True
        
        keys = list(ppr_output.keys())
        if len(keys) == 1:
            return True
        
        if len(keys) >= 2:
            first_dis = keys[0]
            second_dis = keys[1]
            score_diff = (ppr_output[first_dis] - ppr_output[second_dis]) / ppr_output[first_dis]
            if score_diff > RANK_THRESHOLD:
                return True
        
        return False
    
    def modify_ppr(self, possible_diseases, ppr_output):
        """Filter PPR output to only include possible diseases"""
        for key in list(ppr_output.keys()):
            if key not in possible_diseases:
                ppr_output.pop(key)
        
        for k, v in ppr_output.items():
            ppr_output[k] = round(v, 3)
        
        return ppr_output
    
    def generate_patient_response(self, symptom, dialogue_id, patient_data):
        """Simulate patient response (Algorithm 1, lines 24-32)"""
        if symptom in patient_data:
            if patient_data[symptom] == True:
                return 1
            elif patient_data[symptom] == False:
                return -1
            else:
                return 0
        else:
            return 0
    
    def calculate_avg_cooccurrence(self, unknown_symptom, yes_symptoms, co_occurrence_dict):
        """
        Calculate average co-occurrence between unknown symptom and all "Yes" symptoms
        """
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
    
        return sum(cooccurrence_scores) / len(cooccurrence_scores)
    
    def run_dialogue(self, dialogue_id, initial_symptoms, ground_truth, 
                    candidate_diseases, patient_data, max_turns=MAX_DIALOGUE_TURNS):
        """Run complete diagnosis dialogue (Algorithm 1, main loop) with error tracking"""
        G = nx.DiGraph()
        dialogue_node = f"user query_{dialogue_id}"
        
        # NEW: Track for error analysis
        gt_lower = ground_truth.lower()
        gt_in_initial_candidates = gt_lower in [d.lower() for d in candidate_diseases]
        
        asked_symptoms = set(initial_symptoms)
        yes_symptoms = set(initial_symptoms)
        
        candidate_diseases, G, possible_diseases = self.graph_builder.create_graph(
            G, dialogue_node, initial_symptoms, 1, 0, 
            candidate_diseases, set(candidate_diseases), 1, gt_lower
        )
        
        current_symptom = initial_symptoms[0] if initial_symptoms else None
        turn = 1
        output_dict = {}
        
        while turn <= max_turns:
            entropy_syms = self.symptom_selector.get_entropy_based_symptoms(
                current_symptom, possible_diseases, asked_symptoms
            )
            
            if not entropy_syms:
                break
            
            doctor_symptom = entropy_syms[0][0]
            asked_symptoms.add(doctor_symptom)
            
            patient_response = self.generate_patient_response(
                doctor_symptom, dialogue_id, patient_data
            )
            
            if patient_response == 1:
                yes_symptoms.add(doctor_symptom)
                candidate_diseases, G, possible_diseases = self.graph_builder.create_graph(
                    G, dialogue_node, [doctor_symptom], 1, turn,
                    candidate_diseases, possible_diseases, 1, gt_lower
                )
                current_symptom = doctor_symptom
                
            elif patient_response == -1:
                candidate_diseases, G, possible_diseases = self.graph_builder.create_graph(
                    G, dialogue_node, [doctor_symptom], -1, turn,
                    candidate_diseases, possible_diseases, 2, gt_lower
                )
            
            else: 
                avg_cooccur = self.calculate_avg_cooccurrence(
                    doctor_symptom, yes_symptoms, self.data_loader.co_occurrence_dict
                )
                if avg_cooccur > DONT_KNOW_THRESHOLD:
                    response_val = 1
                    yes_symptoms.add(doctor_symptom)
                else:
                    response_val = -1
                
                candidate_diseases, G, possible_diseases = self.graph_builder.create_graph(
                    G, dialogue_node, [doctor_symptom], response_val, turn,
                    candidate_diseases, possible_diseases, 3, gt_lower
                )
            
            output_dict = self.pagerank_calc.calculate_disease_rank(G, dialogue_id)
            output_dict = self.modify_ppr(possible_diseases, output_dict)
            
            if self.should_stop_conversation(possible_diseases, output_dict, turn):
                if len(possible_diseases) == 1:
                    predicted = list(possible_diseases)[0]
                else:
                    predicted = list(output_dict.keys())[0] if output_dict else None
                
                break
            
            turn += 1
        
        if not output_dict:
            output_dict = self.pagerank_calc.calculate_disease_rank(G, dialogue_id)
        
        keys = list(output_dict.keys())
        if len(keys) >= 2:
            predicted = [keys[0], keys[1]]
        elif len(keys) == 1:
            predicted = keys[0]
        else:
            predicted = None
        
        # NEW: Determine error category
        error_category = None
        pruned_diseases = self.graph_builder.pruned_diseases.get(str(dialogue_id), [])
        
        predicted_lower = predicted.lower() if isinstance(predicted, str) else (predicted[0].lower() if isinstance(predicted, list) and predicted else "")
        
        if gt_lower != predicted_lower:
            # Wrong prediction - categorize the error
            if not gt_in_initial_candidates:
                error_category = "not_in_initial_candidates"
            elif gt_lower in [d.lower() for d in pruned_diseases]:
                error_category = "pruned_during_dialogue"
            elif gt_lower in [k.lower() for k in keys]:
                error_category = "present_but_not_top1"
            else:
                error_category = "other"
        
        # NEW: Return error_category
        return predicted, keys, turn-1, asked_symptoms, error_category
    
    def run_dialogue_from_image(self, dialogue_id, image_filename, ground_truth, 
                                image_disease_dict, patient_data, max_turns=MAX_DIALOGUE_TURNS):
        """Run diagnosis dialogue starting from image input with error tracking"""
        G = nx.DiGraph()
        dialogue_node = f"user query_{dialogue_id}"
        
        # NEW: Track for error analysis
        gt_lower = ground_truth.lower()
        gt_in_initial_candidates = gt_lower in [d.lower() for d in image_disease_dict.keys()]
        
        visual_symptoms = self.data_loader.image_processor.img_to_sym(image_filename)
        
        if not visual_symptoms:
            symptom_name = f"visual_symptom_{image_filename}"
            visual_symptoms = [symptom_name]
        
        initial_symptom = visual_symptoms[0].lower()
        
        asked_symptoms = set([initial_symptom])
        yes_symptoms = set([initial_symptom])
        
        candidate_diseases, G, possible_diseases = self.graph_builder.create_graph_from_image(
            G, dialogue_node, initial_symptom, image_disease_dict, set()
        )
        
        candidate_diseases = [d.lower() for d in candidate_diseases]
        possible_diseases = [d.lower() for d in possible_diseases]
        
        turn = 1
        output_dict = {}
        current_symptom = initial_symptom
        
        while turn <= max_turns:
            entropy_syms = self.symptom_selector.get_entropy_based_symptoms(
                current_symptom, possible_diseases, asked_symptoms
            )
            
            if not entropy_syms:
                break
            
            doctor_symptom = entropy_syms[0][0]
            asked_symptoms.add(doctor_symptom)
            
            patient_response = self.generate_patient_response(
                doctor_symptom, dialogue_id, patient_data
            )
            
            if patient_response == 1:
                yes_symptoms.add(doctor_symptom)
                candidate_diseases, G, possible_diseases = self.graph_builder.create_graph(
                    G, dialogue_node, [doctor_symptom], 1, turn,
                    candidate_diseases, possible_diseases, 1, gt_lower
                )
                current_symptom = doctor_symptom
                
            elif patient_response == -1:
                candidate_diseases, G, possible_diseases = self.graph_builder.create_graph(
                    G, dialogue_node, [doctor_symptom], -1, turn,
                    candidate_diseases, possible_diseases, 2, gt_lower
                )
            
            else:
                avg_cooccur = self.calculate_avg_cooccurrence(
                    doctor_symptom, yes_symptoms, self.data_loader.co_occurrence_dict
                )
                
                if avg_cooccur > DONT_KNOW_THRESHOLD:
                    response_val = 1
                    yes_symptoms.add(doctor_symptom)
                else:
                    response_val = -1
                
                candidate_diseases, G, possible_diseases = self.graph_builder.create_graph(
                    G, dialogue_node, [doctor_symptom], response_val, turn,
                    candidate_diseases, possible_diseases, 3, gt_lower
                )
            
            output_dict = self.pagerank_calc.calculate_disease_rank(G, dialogue_id)
            output_dict = self.modify_ppr(possible_diseases, output_dict)
            
            if self.should_stop_conversation(possible_diseases, output_dict, turn):
                if len(possible_diseases) == 1:
                    predicted = list(possible_diseases)[0]
                else:
                    predicted = list(output_dict.keys())[0] if output_dict else None
                
                break
            
            turn += 1
        
        if not output_dict:
            output_dict = self.pagerank_calc.calculate_disease_rank(G, dialogue_id)
        
        keys = list(output_dict.keys())
        if len(keys) >= 2:
            predicted = [keys[0], keys[1]]
        elif len(keys) == 1:
            predicted = keys[0]
        else:
            predicted = None
        
        # NEW: Determine error category
        error_category = None
        pruned_diseases = self.graph_builder.pruned_diseases.get(str(dialogue_id), [])
        
        predicted_lower = predicted.lower() if isinstance(predicted, str) else (predicted[0].lower() if isinstance(predicted, list) and predicted else "")
        
        if gt_lower != predicted_lower:
            if not gt_in_initial_candidates:
                error_category = "not_in_initial_candidates"
            elif gt_lower in [d.lower() for d in pruned_diseases]:
                error_category = "pruned_during_dialogue"
            elif gt_lower in [k.lower() for k in keys]:
                error_category = "present_but_not_top1"
            else:
                error_category = "other"
        
        # NEW: Return error_category
        return predicted, keys, turn-1, asked_symptoms, error_category
