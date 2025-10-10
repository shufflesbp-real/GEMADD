import networkx as nx
import matplotlib.pyplot as plt

def create_graph(G, id, symptom, wt, turn, candidate_diseases, possible_set_of_diseases, cond, gt, p_d_given_s, min_scores, symp_to_dis):
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
    diseases_to_prune = []
    for disease in list(possible_set_of_diseases):
        if not G.has_edge(last_symptom, disease):
            diseases_to_prune.append(disease)
        else:
            # Check edge weight threshold
            edge_weight = G[last_symptom][disease].get('weight', 0)
            if edge_weight < edge_weight_threshold:
                diseases_to_prune.append(disease)
    # Removing pruned diseases from possible set and from graph
    for disease in diseases_to_prune:
        if disease in possible_set_of_diseases:
            possible_set_of_diseases.remove(disease)
        if G.has_node(disease):
            G.remove_node(disease)
    return G, possible_set_of_diseases

def visualise_graph(G, turn):
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
