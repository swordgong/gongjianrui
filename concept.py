import networkx as nx
import pandas as pd
import json

#读取ConceptNet数据
conceptnet_data = pd.read_csv('chineseconceptnet.csv', sep='\t', header=None, quotechar='"', encoding='utf-8')

#打印前几行检查数据结构
print(conceptnet_data.head())

#构建ConceptNet知识图谱
knowledge_graph = nx.MultiDiGraph()

for _, row in conceptnet_data.iterrows():
    relation = row[1]
    start = row[2]
    end = row[3]
    
    #权重信息在第4列
    edge_data = json.loads(row[4])
    weight = edge_data.get('weight', 1.0)  #默认权重为1.0
    
    knowledge_graph.add_edge(start, end, relation=relation, weight=weight)

print("中文ConceptNet知识图谱构建完成！")
print(f"图谱包含 {knowledge_graph.number_of_nodes()} 个节点和 {knowledge_graph.number_of_edges()} 条边")