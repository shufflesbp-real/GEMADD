import networkx as nx
import matplotlib.pyplot as plt

def create_graph(G, id, symptom, wt, turn, candidate_diseases, possible_set_of_diseases, cond, gt, p_d_given_s, min_scores, symp_to_dis):
    """
    Create and update the knowledge graph.
    """
    G.add_node(id, type='usernode')
    edge = 'positive' if wt == 1 else 'negative'

    if turn == 0:
        possible_set_of_diseases = candidate_diseases.copy()
        for dis in candidate_diseases:
            G.add_node(dis, type='disease')

    for dis in candidate_diseases:
        valid_connection = False
        for sym in symptom:
            G.add_node(sym, type='symptom')
            G.add_edge(id, sym, weight=wt, edge_type=edge)

            if dis in p_d_given_s.get(sym, {}):
                dis_wt = p_d_given_s[sym][dis]
                if dis_wt > 0.04 or dis == gt:
                    G.add_edge(sym, dis, weight=dis_wt)
                    valid_connection = True
            elif turn == 0 or dis in symp_to_dis.get(sym, []):
                dis_wt = min_scores.get(dis, 0.001)
                G.add_edge(sym, dis, weight=dis_wt)
                valid_connection = True
            elif cond == 3:
                if dis in symp_to_dis.get(sym, []):
                    dis_wt = min_scores.get(dis, 0.001)
                    G.add_edge(sym, dis, weight=dis_wt)
                    valid_connection = True

        if not valid_connection and dis in possible_set_of_diseases and wt == 1:
            possible_set_of_diseases.remove(dis)

    return candidate_diseases, G, possible_set_of_diseases
def prune_diseases(G, possible_set_of_diseases, last_symptom, edge_weight_threshold=0.04):
    """
    Prune diseases from graph according to Algorithm 1, Line 33.
    
    Prune diseases D from G if:
    1. D is not connected to last_symptom (C*)
    2. Edge weight (last_symptom, D) < threshold
    
    Args:
        G: NetworkX graph
        possible_set_of_diseases: Current set of candidate diseases
        last_symptom: Most recently added symptom (C*)
        edge_weight_threshold: Minimum edge weight threshold
    
    Returns:
        Updated possible_set_of_diseases and modified graph G
    """
    diseases_to_prune = []
    
    for disease in list(possible_set_of_diseases):
        # Check if disease is connected to the last symptom
        if not G.has_edge(last_symptom, disease):
            # Disease not connected to C* - mark for pruning
            diseases_to_prune.append(disease)
        else:
            # Check edge weight threshold
            edge_weight = G[last_symptom][disease].get('weight', 0)
            if edge_weight < edge_weight_threshold:
                # Edge weight below threshold - mark for pruning
                diseases_to_prune.append(disease)
    
    # Remove pruned diseases from possible set and optionally from graph
    for disease in diseases_to_prune:
        if disease in possible_set_of_diseases:
            possible_set_of_diseases.remove(disease)
        
        # Optional: Remove disease node from graph completely
        # Uncomment the following lines if you want to remove nodes from graph
        # if G.has_node(disease):
        #     G.remove_node(disease)
    
    return G, possible_set_of_diseases

def visualise_graph(G, turn):
    """
    Visualize the graph using matplotlib.
    """
    plt.figure(figsize=(10, 8))
    pos = nx.spring_layout(G, seed=42, k=0.5, iterations=100)
    color_map = []
    for node in G:
        node_type = G.nodes[node].get("type", "")
        if node_type == "usernode":
            color_map.append("orange")
        elif node_type == "symptom":
            color_map.append("lightgreen")
        elif node_type == "disease":
            color_map.append("lightblue")
        else:
            color_map.append("grey")

    edge_colors = ['green' if G[u][v].get('weight', 0) > 0 else 'red' if G[u][v].get('weight', 0) < 0 else 'blue' for u, v in G.edges()]
    nx.draw(
        G, pos, with_labels=True, node_color=color_map, edge_color=edge_colors,
        node_size=1200, font_size=9, font_weight='bold', width=1.5
    )
    edge_labels = {(u, v): G[u][v].get("weight", "") for u, v in G.edges()}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color="black", font_size=8)

    plt.title(f"Graph after turn: {turn}", fontsize=12)
    plt.axis("off")
    plt.tight_layout()
    plt.show()