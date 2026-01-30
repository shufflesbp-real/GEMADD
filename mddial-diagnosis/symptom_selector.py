### USING ENTROPY INSTEAD OF GINI IMPURITY ###

import math
from collections import defaultdict

def get_top_cooccurring_symptoms(symptoms, asked_symptoms, co_occurrence_dict, top_k=10):
    """
    Get top co-occurring symptoms.
    """
    if not isinstance(symptoms, list):
        symptoms = [symptoms]

    if not symptoms:
        return []

    cooccurring_sets = []
    for symptom in symptoms:
        if symptom in co_occurrence_dict:
            cooccurring = list(co_occurrence_dict[symptom].keys())
            cooccurring_sets.append(set(cooccurring))

    if not cooccurring_sets:
        return []

    common_cooccurring = set.intersection(*cooccurring_sets)
    common_cooccurring -= set(asked_symptoms)

    if not common_cooccurring:
        return []

    scores = {}
    for sym in common_cooccurring:
        score_list = [co_occurrence_dict.get(symptom, {}).get(sym, 0) for symptom in symptoms]
        if score_list:
            scores[sym] = sum(score_list) / len(score_list)

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [s for s, _ in ranked][:top_k]


def calc_gini(symptom, diseases, y_n, symptom_sign, symptom_disease_stats):
    """
    Calculate ENTROPY (replacing Gini impurity, interface unchanged).
    """
    sym_count = symptom_sign[symptom][y_n]
    if sym_count == 0:
        return 0.0

    entropy = 0.0
    for disease in diseases:
        disease = disease.lower()
        dis_count = symptom_disease_stats[symptom][y_n].get(disease, 0)
        if dis_count == 0:
            continue
        p = dis_count / sym_count
        entropy -= p * math.log2(p)

    return entropy


def get_entropy_based_symptoms(symptom, diseases, asked_symptoms,
                               co_occurrence_dict, symptom_sign,
                               symptom_disease_stats, top_k=5):
    common_symptoms = get_top_cooccurring_symptoms(
        symptom, asked_symptoms, co_occurrence_dict
    )

    symp_gini = {}
    for sym in common_symptoms:
        if sym not in asked_symptoms:
            sym = sym.lower()

            gini_yes = calc_gini(sym, diseases, 'pos',
                                 symptom_sign, symptom_disease_stats)
            gini_no = calc_gini(sym, diseases, 'neg',
                                symptom_sign, symptom_disease_stats)

            total = symptom_sign[sym]['pos'] + symptom_sign[sym]['neg']

            weighted_gini = (
                (symptom_sign[sym]['pos'] * gini_yes +
                 symptom_sign[sym]['neg'] * gini_no) / total
                if total > 0 else 0.0
            )

            symp_gini[sym] = {
                'gini_pos': gini_yes,     # now entropy
                'gini_neg': gini_no,      # now entropy
                'weighted_gini': weighted_gini  # expected entropy
            }

    sorted_syms = sorted(
        symp_gini.items(),
        key=lambda x: x[1]['weighted_gini']
    )

    return sorted_syms[:top_k]
