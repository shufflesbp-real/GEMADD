"""
PageRank Calculator Module
Computes Personalized PageRank for disease ranking
"""

import os
import networkx as nx
from srwr.srwr.srwr import SRWR
from config import *


class PageRankCalculator:
    """Calculates PageRank scores for disease ranking"""
    
    def __init__(self):
        self.rejected_ids = []
    
    def calculate_disease_rank(self, graph, dialogue_id):
        """Calculate disease rankings using Personalized PageRank"""
        try:
            srwr = SRWR()
            
            node_mapping = {node: idx for idx, node in enumerate(graph.nodes())}
            graph_nodes = list(graph.nodes())
            G_numeric = nx.relabel_nodes(graph, node_mapping)
            
            edge_file = f"{GRAPH_EDGE_DIR}graph_edges_{dialogue_id}.txt"
            with open(edge_file, "w") as f:
                for u, v, data in G_numeric.edges(data=True):
                    weight = data.get("weight", 1.0)
                    f.write(f"{u} {v} {weight}\n")
            
            srwr.read_graph(edge_file)
            srwr.normalize()
            
            user_query_node = next(
                (node for node in graph_nodes if node.startswith("user query_")), 
                None
            )
            seed = node_mapping.get(user_query_node, 0)
            
            rd, rp, rn, residuals = srwr.query(
                seed, PPR_C, PPR_EPSILON, PPR_BETA, PPR_GAMMA, 
                PPR_MAX_ITERS, handles_deadend=True
            )
            
            id_to_node = {v: k for k, v in node_mapping.items()}
            disease_nodes = [
                node for node, attr in graph.nodes(data=True) 
                if attr.get('type') == 'disease'
            ]
            
            filtered_rd = {
                id_to_node[node_id]: score.item()
                for node_id, score in enumerate(rd)
                if node_id in id_to_node and id_to_node[node_id] in disease_nodes
            }
            
            if os.path.exists(edge_file):
                os.remove(edge_file)
            
            return dict(sorted(filtered_rd.items(), key=lambda item: item[1], reverse=True))
            
        except Exception as e:
            print(f"[Error] PageRank failed for dialogue {dialogue_id}: {e}")
            self.rejected_ids.append(dialogue_id)
            return {}
