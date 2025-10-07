import pandas as pd

def analyze_diagnosis_errors(result_df, error_tracking_data):
    """
    Analyze errors in diagnosis to categorize into three types:
    1. Ground truth disease was not present in initial candidate diseases
    2. Ground truth disease was pruned during dialogue
    3. Ground truth disease present but not ranked as Top-1
    
    Args:
        result_df: DataFrame with diagnosis results
        error_tracking_data: List of dicts with error tracking info for each dialogue
    
    Returns:
        Dictionary with error analysis statistics and detailed DataFrame
    """
    # Filter only incorrect diagnoses
    incorrect_diagnoses = []
    
    for i, row in result_df.iterrows():
        gt = str(row['Ground Truth']).lower().strip()
        pred_list = row['Predicted List']
        
        # Convert predicted list to lowercase strings
        if isinstance(pred_list, list):
            pred_list_lower = [str(p).lower().strip() for p in pred_list]
        else:
            pred_list_lower = []
        
        # Check if diagnosis is wrong (ground truth not at position 0)
        is_wrong = (len(pred_list_lower) == 0 or pred_list_lower[0] != gt)
        
        if is_wrong:
            incorrect_diagnoses.append({
                'dialogue_id': row['Dialogue_id'],
                'ground_truth': gt,
                'predicted': pred_list_lower[0] if pred_list_lower else 'None',
                'predicted_list': pred_list_lower
            })
    
    # Match with error tracking data
    error_categories = {
        'not_in_candidates': 0,
        'got_pruned': 0,
        'present_but_not_top1': 0,
        'unknown': 0
    }
    
    detailed_errors = []
    
    for error_case in incorrect_diagnoses:
        dialogue_id = error_case['dialogue_id']
        gt = error_case['ground_truth']
        
        # Find matching tracking data
        tracking = next((t for t in error_tracking_data if t['dialogue_id'] == dialogue_id), None)
        
        if tracking:
            error_type = tracking['error_type']
            error_categories[error_type] += 1
            
            detailed_errors.append({
                'Dialogue_ID': dialogue_id,
                'Ground_Truth': gt,
                'Predicted_Top1': error_case['predicted'],
                'Error_Type': error_type,
                'Was_In_Candidates': tracking['was_in_candidates'],
                'Was_Pruned': tracking['was_pruned'],
                'Final_Rank': tracking['final_rank']
            })
        else:
            error_categories['unknown'] += 1
            detailed_errors.append({
                'Dialogue_ID': dialogue_id,
                'Ground_Truth': gt,
                'Predicted_Top1': error_case['predicted'],
                'Error_Type': 'unknown',
                'Was_In_Candidates': 'N/A',
                'Was_Pruned': 'N/A',
                'Final_Rank': 'N/A'
            })
    
    # Calculate percentages
    total_errors = len(incorrect_diagnoses)
    error_percentages = {}
    
    if total_errors > 0:
        for error_type, count in error_categories.items():
            error_percentages[error_type] = (count / total_errors) * 100
    
    detailed_df = pd.DataFrame(detailed_errors)
    
    return {
        'total_incorrect': total_errors,
        'error_counts': error_categories,
        'error_percentages': error_percentages,
        'detailed_errors_df': detailed_df
    }


def print_error_analysis(error_analysis):
    """
    Print error analysis in a readable format.
    """
    print("\n" + "="*70)
    print("ERROR ANALYSIS - DIAGNOSIS FAILURES")
    print("="*70)
    
    total = error_analysis['total_incorrect']
    counts = error_analysis['error_counts']
    percentages = error_analysis['error_percentages']
    
    print(f"\nTotal Incorrect Diagnoses: {total}")
    print("-"*70)
    
    print("\nError Breakdown:")
    print(f"  1. Not in Initial Candidates:  {counts['not_in_candidates']:4d} ({percentages.get('not_in_candidates', 0):5.2f}%)")
    print(f"  2. Got Pruned During Dialogue: {counts['got_pruned']:4d} ({percentages.get('got_pruned', 0):5.2f}%)")
    print(f"  3. Present but Not Top-1:      {counts['present_but_not_top1']:4d} ({percentages.get('present_but_not_top1', 0):5.2f}%)")
    
    if counts['unknown'] > 0:
        print(f"  4. Unknown/Untracked:          {counts['unknown']:4d} ({percentages.get('unknown', 0):5.2f}%)")
    
    print("="*70)
    
    # Save detailed error report
    detailed_df = error_analysis['detailed_errors_df']
    if not detailed_df.empty:
        output_file = '/home/Byomakesh/ours-diagnosis/mddial-diagnosis/error_analysis_detailed.csv'
        detailed_df.to_csv(output_file, index=False)
        print(f"\nDetailed error report saved to: {output_file}")
