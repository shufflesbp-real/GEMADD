import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import os
from utils import get_candidate_disease, modify_ppr
from graph_builder import create_graph
from reasoner import calc_dis_rank_1
def visualize_final_diagnosis_graph(G, top_diseases_ranked, dialogue_id, save_path='/home/Byomakesh/ours-diagnosis/mddial-diagnosis/visualization/'):
    fig, ax = plt.subplots(figsize=(14, 14))
    top_1_disease = top_diseases_ranked[0] if top_diseases_ranked else None
    user_node = [node for node in G.nodes() if G.nodes[node].get('type') == 'usernode'][0]
    disease_nodes = [node for node in G.nodes() if G.nodes[node].get('type') == 'disease']
    symptom_nodes = [node for node in G.nodes() if G.nodes[node].get('type') == 'symptom']
    pos = {}
    pos[user_node] = (0, 0)
    n_symptoms = len(symptom_nodes)
    symptom_radius = 1.8
    for i, symptom in enumerate(symptom_nodes):
        angle = 2 * np.pi * i / n_symptoms - np.pi/2
        pos[symptom] = (symptom_radius * np.cos(angle), symptom_radius * np.sin(angle))
    n_diseases = len(disease_nodes)
    disease_radius = 3.2
    if top_1_disease in disease_nodes:
        other_diseases = [d for d in disease_nodes if d != top_1_disease]
        pos[top_1_disease] = (0, disease_radius)
        for i, disease in enumerate(other_diseases):
            angle = 2 * np.pi * (i + 0.5) / n_diseases - np.pi/2
            pos[disease] = (disease_radius * np.cos(angle), disease_radius * np.sin(angle))
    else:
        for i, disease in enumerate(disease_nodes):
            angle = 2 * np.pi * i / n_diseases - np.pi/2
            pos[disease] = (disease_radius * np.cos(angle), disease_radius * np.sin(angle))
            
    edges_to_draw = {
        'user_to_symptom_pos': [],
        'user_to_symptom_neg': [],
        'symptom_to_top1': [],
        'symptom_to_others': []
    }
    for u, v, data in G.edges(data=True):
        edge_weight = data.get('weight', 0)
        if u == user_node and G.nodes[v].get('type') == 'symptom':
            if edge_weight > 0:
                edges_to_draw['user_to_symptom_pos'].append((u, v))
            else:
                edges_to_draw['user_to_symptom_neg'].append((u, v))
        elif G.nodes[u].get('type') == 'symptom' and G.nodes[v].get('type') == 'disease':
            if v == top_1_disease:
                edges_to_draw['symptom_to_top1'].append((u, v))
            else:
                edges_to_draw['symptom_to_others'].append((u, v))
    nx.draw_networkx_edges(
        G, pos,
        edgelist=edges_to_draw['symptom_to_others'],
        edge_color='#D5DBDB',
        width=2.5,
        alpha=0.5,
        arrowsize=15,
        arrowstyle='->',
        connectionstyle='arc3,rad=0.05',
        ax=ax
    )
    nx.draw_networkx_edges(
        G, pos,
        edgelist=edges_to_draw['user_to_symptom_neg'],
        edge_color='#E74C3C',
        width=4.5,
        alpha=0.9,
        arrowsize=20,
        arrowstyle='->',
        ax=ax
    )
    nx.draw_networkx_edges(
        G, pos,
        edgelist=edges_to_draw['user_to_symptom_pos'],
        edge_color='#27AE60',
        width=4.5,
        alpha=0.9,
        arrowsize=20,
        arrowstyle='->',
        ax=ax
    )
    nx.draw_networkx_edges(
        G, pos,
        edgelist=edges_to_draw['symptom_to_top1'],
        edge_color='#D68910',
        width=6.0,
        alpha=1.0,
        arrowsize=25,
        arrowstyle='->',
        connectionstyle='arc3,rad=0.05',
        ax=ax
    )
    if top_1_disease and top_1_disease in disease_nodes:
        other_diseases = [d for d in disease_nodes if d != top_1_disease]
    else:
        other_diseases = disease_nodes
    if other_diseases:
        nx.draw_networkx_nodes(
            G, pos,
            nodelist=other_diseases,
            node_color='#FDEBD0',
            node_size=6000, 
            node_shape='o',
            edgecolors='#AEB6BF',
            linewidths=2.5,
            alpha=0.65,
            ax=ax
        )
    nx.draw_networkx_nodes(
        G, pos,
        nodelist=symptom_nodes,
        node_color='#7FB3D5',
        node_size=10000,
        node_shape='o',
        edgecolors='#1B4F72',
        linewidths=3,
        ax=ax
    )
    nx.draw_networkx_nodes(
        G, pos,
        nodelist=[user_node],
        node_color='#BB8FCE',
        node_size=8000,
        node_shape='o',
        edgecolors='#5B2C6F',
        linewidths=4,
        ax=ax
    )
    if top_1_disease and top_1_disease in disease_nodes:
        nx.draw_networkx_nodes(
            G, pos,
            nodelist=[top_1_disease],
            node_color='#E67E22',
            node_size=8500,
            node_shape='o',
            edgecolors='#78281F',
            linewidths=5,
            ax=ax
        )
    labels = {}
    for node in G.nodes():
        if 'user' in node.lower():
            labels[node] = 'Patient'
        else:
            label = str(node).replace('_', ' ').title()
            if len(label) > 20:
                words = label.split()
                mid = len(words) // 2
                label = ' '.join(words[:mid]) + '\n' + ' '.join(words[mid:])
            labels[node] = label
    nx.draw_networkx_labels(
        G, pos,
        labels=labels,
        font_size=10,
        font_weight='bold',
        font_color='#17202A',
        font_family='sans-serif',
        ax=ax
    )
    edge_labels = {}
    for u, v, data in G.edges(data=True):
        weight = data.get('weight', '')
        if u == user_node:
            if weight == 1:
                edge_labels[(u, v)] = '+1'
            elif weight == -1:
                edge_labels[(u, v)] = '-1'
        elif G.nodes[u].get('type') == 'symptom' and G.nodes[v].get('type') == 'disease':
            if v == top_1_disease and isinstance(weight, float) and weight > 0.001:
                edge_labels[(u, v)] = f"{weight:.3f}"
    nx.draw_networkx_edge_labels(
        G, pos,
        edge_labels=edge_labels,
        font_size=10,
        font_color='#0B5345',
        font_weight='bold',
        bbox=dict(boxstyle='round,pad=0.4', facecolor='white', edgecolor='lightgray', alpha=0.9),
        ax=ax
    )
    ax.set_xlim(-4.5, 4.5)
    ax.set_ylim(-4.5, 4.5)
    ax.axis('off')
    ax.set_aspect('equal')
    plt.tight_layout(pad=0.5)

    os.makedirs(save_path, exist_ok=True)
    filename_png = f'{save_path}diagnosis_graph_dialogue_{dialogue_id}.png'
    plt.savefig(filename_png, dpi=300, bbox_inches='tight', facecolor='white', pad_inches=0.2)
    print(f"PNG saved: {filename_png}")
    plt.show()
    plt.close()

def visualize_single_sample_from_dict(sample_data, sample_id, data_resources):
    (co_occurrence_dict, disease_symptom_dict, symptom_sign,
     symptom_disease_stats, dict_patient, p_d_given_s, min_scores, symp_to_dis) = data_resources
    pos_symptoms = [s.lower() for s in sample_data.get('pos-symptoms', [])]
    neg_symptoms = [s.lower() for s in sample_data.get('neg-symptoms', [])]
    ground_truth = sample_data['ground_truth'].lower()
    G = nx.DiGraph()
    dialogue_node = f'user_node_{sample_id}'
    patient_reported_symptoms = pos_symptoms[:3] if len(pos_symptoms) >= 3 else pos_symptoms
    candidate_diseases = get_candidate_disease(patient_reported_symptoms, symp_to_dis)
    possible_set_of_diseases = []
    _, G, possible_set_of_diseases = create_graph(
        G, dialogue_node, pos_symptoms, 1, 0,
        candidate_diseases, possible_set_of_diseases, 1, ground_truth,
        p_d_given_s, min_scores, symp_to_dis
    )
    if neg_symptoms:
        _, G, possible_set_of_diseases = create_graph(
            G, dialogue_node, neg_symptoms, -1, 1,
            candidate_diseases, possible_set_of_diseases, 2, ground_truth,
            p_d_given_s, min_scores, symp_to_dis
        )
    output = calc_dis_rank_1(G, sample_id)
    output = modify_ppr(possible_set_of_diseases, output)
    top_diseases_ranked = list(output.keys())

    visualize_final_diagnosis_graph(G, top_diseases_ranked, sample_id)
    return G, top_diseases_ranked
