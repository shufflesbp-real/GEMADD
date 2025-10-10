from collections import defaultdict
import numpy as np

class DiagnosisMetrics:
    def __init__(self):
        self.results = []
    def add_result(self, dialogue_id, ground_truth, predicted_disease, 
                   dialogue_length, symptoms_gt, symptoms_asked, 
                   symptoms_patient_yes, top_k_predictions):
        gt_lower = ground_truth.lower()
        rank = None
        for idx, pred in enumerate(top_k_predictions):
            if pred.lower() == gt_lower:
                rank = idx + 1 
                break
        result = {
            'dialogue_id': dialogue_id,
            'ground_truth': gt_lower,
            'predicted_top1': predicted_disease.lower() if isinstance(predicted_disease, str) else (predicted_disease[0].lower() if isinstance(predicted_disease, list) and len(predicted_disease) > 0 else None),
            'dialogue_length': dialogue_length,
            'symptoms_gt': set(s.lower() for s in symptoms_gt),
            'symptoms_asked': set(s.lower() for s in symptoms_asked),
            'symptoms_patient_yes': set(s.lower() for s in symptoms_patient_yes),
            'top_k_predictions': [d.lower() for d in top_k_predictions],
            'rank': rank,
        }
        self.results.append(result)
    
    def calculate_hit_at_k(self, k):
        if not self.results:
            return 0.0
        
        hits = 0
        for result in self.results:
            gt = result['ground_truth']
            top_k_preds = result['top_k_predictions'][:k]
            if gt in top_k_preds:
                hits += 1
        return hits / len(self.results)
    def calculate_mrr(self):
        if not self.results:
            return 0.0
        reciprocal_ranks = []
        for result in self.results:
            if result['rank'] is not None:
                reciprocal_ranks.append(1.0 / result['rank'])
            else:
                reciprocal_ranks.append(0.0) 
        return sum(reciprocal_ranks) / len(reciprocal_ranks)
    
    def calculate_average_dialogue_length(self):
        if not self.results:
            return 0.0
        total_length = sum(r['dialogue_length'] for r in self.results)
        return total_length / len(self.results)
    def calculate_symptom_precision_vs_kg(self):
        if not self.results:
            return 0.0
        precisions = []
        for result in self.results:
            x1 = result['symptoms_gt']
            x3 = result['symptoms_asked']
            
            if len(x3) == 0:
                continue
            intersection = x1.intersection(x3)
            precision = len(intersection) / len(x3)
            precisions.append(precision)
        return sum(precisions) / len(precisions) if precisions else 0.0
    
    def calculate_symptom_recall_vs_kg(self):
        if not self.results:
            return 0.0
        recalls = []
        for result in self.results:
            x1 = result['symptoms_gt']
            x3 = result['symptoms_asked']
            if len(x1) == 0:
                continue
            intersection = x1.intersection(x3)
            recall = len(intersection) / len(x1)
            recalls.append(recall)
        return sum(recalls) / len(recalls) if recalls else 0.0
    
    def calculate_symptom_recall_vs_patient_yes(self):
        if not self.results:
            return 0.0
        recalls = []
        for result in self.results:
            patient_yes = result['symptoms_patient_yes']
            x3 = result['symptoms_asked']
            if len(patient_yes) == 0:
                continue
            intersection = patient_yes.intersection(x3)
            recall = len(intersection) / len(patient_yes)
            recalls.append(recall)
        return sum(recalls) / len(recalls) if recalls else 0.0
    
    def calculate_symptom_f1_vs_kg(self):
        precision = self.calculate_symptom_precision_vs_kg()
        recall = self.calculate_symptom_recall_vs_kg()
        if precision + recall == 0:
            return 0.0
        return 2 * (precision * recall) / (precision + recall)
    
    def calculate_disease_precision(self):
        if not self.results:
            return 0.0
        correct = sum(1 for r in self.results if r['predicted_top1'] == r['ground_truth'])
        total = len(self.results)
        return correct / total if total > 0 else 0.0
    
    def calculate_disease_recall(self):
        return self.calculate_disease_precision()
    
    def calculate_disease_f1(self):
        precision = self.calculate_disease_precision()
        recall = self.calculate_disease_recall()
        if precision + recall == 0:
            return 0.0
        return 2 * (precision * recall) / (precision + recall)
    def get_total_dialogues(self):
        return len(self.results)
    
    def get_all_metrics(self):
        return {
            'total_dialogues': self.get_total_dialogues(),
            'hit_at_1': self.calculate_hit_at_k(1),
            'hit_at_3': self.calculate_hit_at_k(3),
            'hit_at_5': self.calculate_hit_at_k(5),
            'mrr': self.calculate_mrr(),
            'avg_dialogue_length': self.calculate_average_dialogue_length(),
            'symptom_precision_vs_kg': self.calculate_symptom_precision_vs_kg(),
            'symptom_recall_vs_kg': self.calculate_symptom_recall_vs_kg(),
            'symptom_recall_vs_patient_yes': self.calculate_symptom_recall_vs_patient_yes(),
            'symptom_f1_vs_kg': self.calculate_symptom_f1_vs_kg(),
            'disease_precision': self.calculate_disease_precision(),
            'disease_recall': self.calculate_disease_recall(),
            'disease_f1': self.calculate_disease_f1(),
        }
    
    def print_metrics(self, elapsed_time=None):
        metrics = self.get_all_metrics()
        print("--- Evaluation Metrics ---")
        print(f"Total Dialogues Processed: {metrics['total_dialogues']}")
        print("-"*28)
        
        print("\nDiagnosis Accuracy:")
        print(f"Diagnosis Success Rate (Hit@1): {metrics['hit_at_1']:.4f}")
        print(f"Hit@3: {metrics['hit_at_3']:.4f}")
        print(f"Hit@5: {metrics['hit_at_5']:.4f}")
        print(f"Mean Reciprocal Rank (MRR):     {metrics['mrr']:.4f}")
        
        print("\nDialogue Efficiency:")
        print(f"Average Dialogue Length: {metrics['avg_dialogue_length']:.2f} turns")
        
        print("\nSymptom Matching Rates:")
        print(f"Precision (vs. KG Symptoms):      {metrics['symptom_precision_vs_kg']:.4f}")
        print(f"Recall (vs. Patient 'Yes' Symp):  {metrics['symptom_recall_vs_patient_yes']:.4f}")        
        print("="*60 + "\n")
