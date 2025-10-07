import json
from collections import defaultdict, Counter

def get_candidate_disease(symps, symp_to_dis):
    """
    Get candidate diseases based on the initial symptoms.
    A disease is a candidate if it is associated with at least half of the initial symptoms.
    """
    disease_counter = Counter()
    total_symps = 0

    for symptom in symps:
        if symptom in symp_to_dis:
            total_symps += 1
            for disease in symp_to_dis[symptom]:
                disease_counter[disease] += 1

    if total_symps == 0:
        return []

    threshold = total_symps / 2.0
    return [disease for disease, count in disease_counter.items() if count >= threshold]

def should_stop_conversation(possible_set_of_diseases, ppr_output, turns):
  """
  Check if the conversation should stop based on the disease ranking.
  """
  if len(possible_set_of_diseases) == 1:
    return True
  else:
    keys = list(ppr_output.keys())
    if len(keys) < 2:
        return False
    else:
        first_dis = keys[0]
        second_dis = keys[1]
        if ppr_output[first_dis] == 0: # Avoid division by zero
            return False
        score = (ppr_output[first_dis] - ppr_output[second_dis]) / ppr_output[first_dis]
        if score > 0.8:
            return True
        else:
            return False

def modify_ppr(possible_set_of_diseases, ppr_output):
    """
    Filter and round the PPR output.
    """
    # Create a new dictionary to avoid modifying the original during iteration
    filtered_ppr = {}
    for key, value in ppr_output.items():
        if key in possible_set_of_diseases:
            filtered_ppr[key] = round(value, 3)
    return filtered_ppr