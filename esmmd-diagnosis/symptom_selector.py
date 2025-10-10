from config import TOP_K_COOCCURRENCE, TOP_K_ENTROPY

class SymptomSelector:
    def __init__(self, co_occurrence_dict, symptom_sign, symptom_disease_stats):
        self.co_occurrence_dict = co_occurrence_dict
        self.symptom_sign = symptom_sign
        self.symptom_disease_stats = symptom_disease_stats
    def get_top_cooccurring_symptoms(self, symptoms, asked_symptoms, top_k=TOP_K_COOCCURRENCE):
        if not isinstance(symptoms, list):
            symptoms = [symptoms]
        if not symptoms:
            return []
        cooccurring_sets = []
        for symptom in symptoms:
            if symptom in self.co_occurrence_dict:
                cooccurring = list(self.co_occurrence_dict[symptom].keys())
                cooccurring_sets.append(set(cooccurring))
        if not cooccurring_sets:
            return []
        common_cooccurring = set.intersection(*cooccurring_sets)
        common_cooccurring -= set(asked_symptoms)
        if not common_cooccurring:
            return []
        scores = {}
        for sym in common_cooccurring:
            score_list = []
            for symptom in symptoms:
                symptom_cooccurrence = self.co_occurrence_dict.get(symptom, {})
                if sym in symptom_cooccurrence:
                    score_list.append(symptom_cooccurrence[sym])
            if score_list:
                scores[sym] = sum(score_list) / len(score_list)
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [s for s, _ in ranked][:top_k]

    def calc_gini(self, symptom, diseases, y_n):
        sym_count = self.symptom_sign[symptom][y_n]
        if sym_count == 0:
            return 1.0
        sum_score = 0
        for disease in diseases:
            disease = disease.lower()
            dis_count = self.symptom_disease_stats[symptom][y_n][disease]
            if dis_count == 0:
                continue
            sum_score += (dis_count / sym_count) ** 2
        return 1 - sum_score
        
    def get_entropy_based_symptoms(self, symptom, diseases, asked_symptoms, top_k=TOP_K_ENTROPY):
        common_symptoms = self.get_top_cooccurring_symptoms(symptom, asked_symptoms)
        symp_gini = {}
        for sym in common_symptoms:
            if sym not in asked_symptoms:
                sym = sym.lower()
                gini_yes = self.calc_gini(sym, diseases, 'pos')
                gini_no = self.calc_gini(sym, diseases, 'neg')
                total = self.symptom_sign[sym]['pos'] + self.symptom_sign[sym]['neg']
                weighted_gini = (
                    (self.symptom_sign[sym]['pos'] * gini_yes + 
                     self.symptom_sign[sym]['neg'] * gini_no) / total
                    if total > 0 else 1.0
                )
                symp_gini[sym] = {
                    'gini_pos': gini_yes,
                    'gini_neg': gini_no,
                    'weighted_gini': weighted_gini
                }
        sorted_syms = sorted(symp_gini.items(), key=lambda x: x[1]['weighted_gini'])
        return sorted_syms[:top_k]
