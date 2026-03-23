import os
import networkx as nx
import pandas as pd
import json
import pickle

def build_and_save_knowledge_graph(csv_path="chineseconceptnet.csv", save_path="knowledge_graph.pkl"):
    """构建并保存知识图谱"""
    print("开始构建知识图谱...")
    
    # 加载数据
    conceptnet_data = pd.read_csv(csv_path, sep='\t', header=None, quotechar='"', encoding='utf-8')
    knowledge_graph = nx.MultiDiGraph()
    
    # 构建图谱
    for _, row in conceptnet_data.iterrows():
        relation = row[1]
        start = row[2]
        end = row[3]
        edge_data = json.loads(row[4])
        weight = edge_data.get('weight', 1.0)
        knowledge_graph.add_edge(start, end, relation=relation, weight=weight)
    
    # 保存图谱
    with open(save_path, 'wb') as f:
        pickle.dump(knowledge_graph, f)
    
    print(f"知识图谱构建完成并保存至 {save_path}")
    print(f"节点数: {len(knowledge_graph.nodes)}, 边数: {len(knowledge_graph.edges)}")
    
if __name__ == "__main__":
    build_and_save_knowledge_graph()