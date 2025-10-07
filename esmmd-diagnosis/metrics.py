"""
Metrics Module
Implements all evaluation metrics for diagnosis system
"""

from collections import defaultdict
import numpy as np


class DiagnosisMetrics:
    """Calculates diagnosis performance metrics"""
    
    def __init__(self):
        self.results = []
        
    def add_result(self, dialogue_id, ground_truth, predicted_disease, 
                   dialogue_length, symptoms_gt, symptoms_asked, 
                   symptoms_patient_yes, top_k_predictions):
        """
        Add a diagnosis result for evaluation
        
        Args:
            dialogue_id: Unique dialogue identifier
            ground_truth: True disease name
            predicted_disease: Predicted disease name (top-1)
            dialogue_length: Number of conversation turns
            symptoms_gt: Set of symptoms associated with correct disease in KG (x1)
            symptoms_asked: Set of symptoms asked by the system (x3)
            symptoms_patient_yes: Set of symptoms patient said "Yes" to
            top_k_predictions: List of top-k predicted diseases (ordered)
        """
        # Find the rank of ground truth in predictions
        gt_lower = ground_truth.lower()
        rank = None
        for idx, pred in enumerate(top_k_predictions):
            if pred.lower() == gt_lower:
                rank = idx + 1  # 1-indexed
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
            'rank': rank,  # Rank of ground truth (None if not found)
        }
        self.results.append(result)
    
    def calculate_hit_at_k(self, k):
        """
        Hit@K: Proportion of cases where ground truth is in top-K predictions
        
        Args:
            k: Top-k to consider
        """
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
        """
        Mean Reciprocal Rank (MRR)
        MRR = average of (1/rank) where rank is position of ground truth
        """
        if not self.results:
            return 0.0
        
        reciprocal_ranks = []
        for result in self.results:
            if result['rank'] is not None:
                reciprocal_ranks.append(1.0 / result['rank'])
            else:
                reciprocal_ranks.append(0.0)  # Not found in predictions
        
        return sum(reciprocal_ranks) / len(reciprocal_ranks)
    
    def calculate_average_dialogue_length(self):
        """Average number of conversational turns"""
        if not self.results:
            return 0.0
        
        total_length = sum(r['dialogue_length'] for r in self.results)
        return total_length / len(self.results)
    
    def calculate_symptom_precision_vs_kg(self):
        """
        Symptom Precision (vs. KG Symptoms)
        = |intersection(x1, x3)| / |x3|
        Where:
            x1 = symptoms associated with correct disease in KG
            x3 = symptoms asked by proposed system
        """
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
        """
        Symptom Recall (vs. KG Symptoms)
        = |intersection(x1, x3)| / |x1|
        Where:
            x1 = symptoms associated with correct disease in KG
            x3 = symptoms asked by proposed system
        """
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
        """
        Symptom Recall (vs. Patient 'Yes' Symptoms)
        = |intersection(patient_yes, x3)| / |patient_yes|
        Where:
            patient_yes = symptoms patient confirmed as "Yes"
            x3 = symptoms asked by proposed system
        """
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
        """F1 Score for symptom matching (vs KG)"""
        precision = self.calculate_symptom_precision_vs_kg()
        recall = self.calculate_symptom_recall_vs_kg()
        
        if precision + recall == 0:
            return 0.0
        
        return 2 * (precision * recall) / (precision + recall)
    
    def calculate_disease_precision(self):
        """
        Disease Diagnosis Precision (Top-1)
        Precision = TP / (TP + FP)
        """
        if not self.results:
            return 0.0
        
        correct = sum(1 for r in self.results if r['predicted_top1'] == r['ground_truth'])
        total = len(self.results)
        
        return correct / total if total > 0 else 0.0
    
    def calculate_disease_recall(self):
        """
        Disease Diagnosis Recall (Top-1)
        For single-label classification, Recall = Precision
        """
        return self.calculate_disease_precision()
    
    def calculate_disease_f1(self):
        """
        Disease Diagnosis F1 (Top-1)
        """
        precision = self.calculate_disease_precision()
        recall = self.calculate_disease_recall()
        
        if precision + recall == 0:
            return 0.0
        
        return 2 * (precision * recall) / (precision + recall)
    
    def get_total_dialogues(self):
        """Get total number of dialogues processed"""
        return len(self.results)
    
    def get_all_metrics(self):
        """Get all metrics in a single dictionary"""
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
        """Print all metrics in the exact format requested"""
        metrics = self.get_all_metrics()
        
        print("\n" + "="*60)
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
        print(f"Recall (vs. KG Symptoms):         {metrics['symptom_recall_vs_kg']:.4f}")
        print(f"Recall (vs. Patient 'Yes' Symp):  {metrics['symptom_recall_vs_patient_yes']:.4f}")
        print(f"F1-Score (vs. KG Symptoms):       {metrics['symptom_f1_vs_kg']:.4f}")
        
        print("\nDisease Diagnosis Performance:")
        print(f"Precision: {metrics['disease_precision']:.4f}")
        print(f"Recall:    {metrics['disease_recall']:.4f}")
        print(f"F1-Score:  {metrics['disease_f1']:.4f}")
        
        if elapsed_time is not None:
            print("\nExecution Time:")
            print(f"Total Time Taken: {elapsed_time:.2f} seconds ({elapsed_time/60:.2f} minutes)")
        
        print("="*60 + "\n")
