import networkx as nx
from config import MIN_EDGE_WEIGHT_THRESHOLD

class GraphBuilder:
    def __init__(self, p_d_given_s, min_scores, symp_to_dis, disease_symptom_dict):
        self.p_d_given_s = p_d_given_s
        self.min_scores = min_scores
        self.symp_to_dis = symp_to_dis
        self.disease_symptom_dict = disease_symptom_dict
        self.pruned_diseases = {}
    def create_graph(self, G, user_id, symptoms, weight, turn, 
                    candidate_diseases, possible_diseases, condition, ground_truth):

        dialogue_id = user_id.replace("user query_", "")
        if dialogue_id not in self.pruned_diseases:
            self.pruned_diseases[dialogue_id] = []
        G.add_node(user_id, type='usernode')
        edge_type = 'positive' if weight == 1 else 'negative'
        if turn == 0:
            possible_diseases = candidate_diseases.copy()
            for dis in candidate_diseases:
                G.add_node(dis, type='disease')
        for dis in candidate_diseases:
            valid_connection = False
            for sym in symptoms:
                G.add_node(sym, type='symptom')
                G.add_edge(user_id, sym, weight=weight, edge_type=edge_type)
                if dis in self.p_d_given_s.get(sym, {}):
                    dis_wt = self.p_d_given_s[sym][dis]
                    if dis_wt > MIN_EDGE_WEIGHT_THRESHOLD or dis == ground_truth:
                        G.add_edge(sym, dis, weight=dis_wt)
                        valid_connection = True
                elif turn == 0 or dis in self.symp_to_dis.get(sym, []):
                    dis_wt = self.min_scores.get(dis, 0.001)
                    G.add_edge(sym, dis, weight=dis_wt)
                    valid_connection = True
                elif condition == 3:  # Don't know
                    if weight == 1 and dis in self.symp_to_dis.get(sym, []):
                        dis_wt = self.min_scores.get(dis, 0.001)
                        G.add_edge(sym, dis, weight=dis_wt)
                        valid_connection = True
            if not valid_connection and dis in possible_diseases and weight == 1:
                possible_diseases.remove(dis)
                self.pruned_diseases[dialogue_id].append(dis)
        return candidate_diseases, G, possible_diseases
    def create_graph_from_image(self, G, user_id, symptom_name, 
                               candidate_disease_dict, possible_diseases):
        dialogue_id = user_id.replace("user query_", "")
        if dialogue_id not in self.pruned_diseases:
            self.pruned_diseases[dialogue_id] = []
        G.add_node(user_id, type='usernode')
        possible_diseases = [key.lower() for key in candidate_disease_dict.keys()]
        G.add_node(symptom_name, type='symptom')
        G.add_edge(user_id, symptom_name, weight=1, edge_type='positive')
        for dis, val in candidate_disease_dict.items():
            dis = dis.lower()
            G.add_node(dis, type='disease')
            G.add_edge(symptom_name, dis, weight=round(val, 3))
        
        return candidate_disease_dict.keys(), G, possible_diseases
