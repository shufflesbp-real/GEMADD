import pandas as pd

def get_rank(row):
    try:
        # Ensure GT is a string for comparison
        gt = str(row['Ground Truth'])
        # Ensure predicted list contains strings
        pred_list = [str(p) for p in row['Predicted List']]
        return pred_list.index(gt)
    except (ValueError, TypeError):
        return -1


def calculate_disease_diagnosis_metrics(result_df):
    metrics = {}
    
    for k in [1, 3, 5]:
        tp = 0  # True Positives
        fp = 0  # False Positives
        fn = 0  # False Negatives
        
        for _, row in result_df.iterrows():
            gt = str(row['Ground Truth']).lower().strip()
            pred_list = [str(p).lower().strip() for p in row['Predicted List']]
            
            # Get top-k predictions
            top_k_preds = pred_list[:k] if len(pred_list) >= k else pred_list
            
            if gt in top_k_preds:
                # Ground truth is in top-k predictions
                tp += 1
                # Count other predictions as false positives
                fp += (len(top_k_preds) - 1)
            else:
                # Ground truth not in top-k predictions
                fn += 1
                # All top-k predictions are false positives
                fp += len(top_k_preds)
        
        # Calculate metrics
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        metrics[f'precision_top_{k}'] = precision
        metrics[f'recall_top_{k}'] = recall
        metrics[f'f1_top_{k}'] = f1
    
    return metrics


def calculate_disease_diagnosis_metrics_strict(result_df):
    tp = 0
    fp = 0
    fn = 0
    
    for _, row in result_df.iterrows():
        gt = str(row['Ground Truth']).lower().strip()
        pred_list = [str(p).lower().strip() for p in row['Predicted List']]
        
        if len(pred_list) > 0:
            top_1_pred = pred_list[0]
            if top_1_pred == gt:
                tp += 1
            else:
                fp += 1
                fn += 1
        else:
            fn += 1
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return {
        'precision_strict': precision,
        'recall_strict': recall,
        'f1_strict': f1
    }


def calculate_and_print_all_metrics(result_df, output_file='/home/Byomakesh/ours-diagnosis/mddial-diagnosis/evaluation_metrics.txt'):
    result_df['GT_Rank'] = result_df.apply(get_rank, axis=1)
    
    # --- Standard Ranking Metrics ---
    total_dialogues = len(result_df)
    hit_at_1 = len(result_df[result_df['GT_Rank'] == 0]) / total_dialogues
    hit_at_3 = len(result_df[result_df['GT_Rank'].between(0, 2)]) / total_dialogues
    hit_at_5 = len(result_df[result_df['GT_Rank'].between(0, 4)]) / total_dialogues
    
    # Filter out failed ranks for MRR calculation
    mrr_df = result_df[result_df['GT_Rank'] != -1]
    mrr = (1 / (mrr_df['GT_Rank'] + 1)).mean() if not mrr_df.empty else 0.0
    
    # --- Disease Diagnosis Classification Metrics ---
    diagnosis_metrics = calculate_disease_diagnosis_metrics(result_df)
    diagnosis_metrics_strict = calculate_disease_diagnosis_metrics_strict(result_df)
    
    # --- Dialogue Efficiency Metrics ---
    avg_dialogue_length = result_df['Dialogue Length'].mean()
    avg_precision = result_df['Match Rate (Precision)'].mean()
    avg_recall_kg = result_df['Recall (vs. KG)'].mean()
    avg_recall_patient = result_df['Recall (vs. Patient Yes)'].mean()
    
    # Create the output string
    output_lines = []
    output_lines.append("\n" + "=" * 60)
    output_lines.append("EVALUATION METRICS - DISEASE DIAGNOSIS SYSTEM")
    output_lines.append("=" * 60)
    output_lines.append(f"\nTotal Dialogues Processed: {total_dialogues}")
    output_lines.append("-" * 60)
    
    # Ranking-based Diagnosis Accuracy
    output_lines.append("\n[ RANKING-BASED DIAGNOSIS ACCURACY ]")
    output_lines.append(f"  Hit@1 (Diagnosis Success Rate):  {hit_at_1:.4f} ({hit_at_1*100:.2f}%)")
    output_lines.append(f"  Hit@3:                            {hit_at_3:.4f} ({hit_at_3*100:.2f}%)")
    output_lines.append(f"  Hit@5:                            {hit_at_5:.4f} ({hit_at_5*100:.2f}%)")
    output_lines.append(f"  Mean Reciprocal Rank (MRR):       {mrr:.4f}")
    
    # Disease Diagnosis Classification Metrics - Strict (Top-1)
    output_lines.append("\n[ DISEASE DIAGNOSIS METRICS - STRICT (Top-1 Only) ]")
    output_lines.append(f"  Precision:  {diagnosis_metrics_strict['precision_strict']:.4f} ({diagnosis_metrics_strict['precision_strict']*100:.2f}%)")
    output_lines.append(f"  Recall:     {diagnosis_metrics_strict['recall_strict']:.4f} ({diagnosis_metrics_strict['recall_strict']*100:.2f}%)")
    output_lines.append(f"  F1-Score:   {diagnosis_metrics_strict['f1_strict']:.4f} ({diagnosis_metrics_strict['f1_strict']*100:.2f}%)")
    
    # Disease Diagnosis Classification Metrics - Top-k
    output_lines.append("\n[ DISEASE DIAGNOSIS METRICS - Top-k Predictions ]")
    for k in [1, 3, 5]:
        output_lines.append(f"\n  Top-{k}:")
        output_lines.append(f"    Precision:  {diagnosis_metrics[f'precision_top_{k}']:.4f} ({diagnosis_metrics[f'precision_top_{k}']*100:.2f}%)")
        output_lines.append(f"    Recall:     {diagnosis_metrics[f'recall_top_{k}']:.4f} ({diagnosis_metrics[f'recall_top_{k}']*100:.2f}%)")
        output_lines.append(f"    F1-Score:   {diagnosis_metrics[f'f1_top_{k}']:.4f} ({diagnosis_metrics[f'f1_top_{k}']*100:.2f}%)")
    
    # Dialogue Efficiency
    output_lines.append("\n[ DIALOGUE EFFICIENCY ]")
    output_lines.append(f"  Average Dialogue Length:  {avg_dialogue_length:.2f} turns")
    
    # Symptom Matching Rates
    output_lines.append("\n[ SYMPTOM MATCHING QUALITY ]")
    output_lines.append(f"  Precision (vs. KG Symptoms):      {avg_precision:.4f} ({avg_precision*100:.2f}%)")
    output_lines.append(f"  Recall (vs. KG Symptoms):         {avg_recall_kg:.4f} ({avg_recall_kg*100:.2f}%)")
    output_lines.append(f"  Recall (vs. Patient 'Yes' Symp):  {avg_recall_patient:.4f} ({avg_recall_patient*100:.2f}%)")
    
    output_lines.append("\n" + "=" * 60)
    
    # Print to console
    for line in output_lines:
        print(line)
    
    # Write to file
    with open(output_file, 'w') as f:
        for line in output_lines:
            f.write(line + '\n')
    
    print(f"\nMetrics saved to: {output_file}")
