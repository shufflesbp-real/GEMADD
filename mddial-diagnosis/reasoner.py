from srwr.srwr.srwr import SRWR
import networkx as nx

def calc_dis_rank_1(new_graph, dialogue_id):
    """
    In this functino we calculate the disease ranking using Signed Random Walk with Restart (SRWR).
    """
    try:
        srwr = SRWR()
        node_mapping = {node: idx for idx, node in enumerate(new_graph.nodes())}
        graph_nodes = list(new_graph.nodes())
        G_numeric = nx.relabel_nodes(new_graph, node_mapping)
        
        edge_file = f"/output_path/graph_edges_{dialogue_id}.txt"
        with open(edge_file, "w") as f:
            for u, v, data in G_numeric.edges(data=True):
                weight = data.get("weight", 1.0)
                f.write(f"{u} {v} {weight}\n")
        srwr.read_graph(edge_file)
        srwr.normalize()
        
        user_query_node = next((node for node in graph_nodes if node.startswith("user query_")), None)
        seed = node_mapping.get(user_query_node, 0)
        
        c, epsilon, beta, gamma, max_iters = 0.15, 1e-9, 0.6, 0.4, 100
        handles_deadend = True
        rd, rp, rn, residuals = srwr.query(seed, c, epsilon, beta, gamma, max_iters, handles_deadend)
        id_to_node = {v: k for k, v in node_mapping.items()}
        disease_nodes = [node for node, attr in new_graph.nodes(data=True) if attr.get('type') == 'disease']
        filtered_rd = {
            id_to_node[node_id]: score.item()
            for node_id, score in enumerate(rd)
            if node_id in id_to_node and id_to_node[node_id] in disease_nodes
        }
        return dict(sorted(filtered_rd.items(), key=lambda item: item[1], reverse=True))
        
    except Exception as e:
        print(f"[Error] Failed for graph {dialogue_id}: {e}")
        return {}
