import pickle
import re
import json

filepath = "./data/pre_process/graph_construction/graph.pkl"
with open(filepath, 'rb') as f:
    graph = pickle.load(f)

# Extract labels
label_dict = {}
for u,v, edge in graph.edges(data=True):
    if edge['label'] not in label_dict:
        label_dict[edge['label']] = []

    label_node = graph.nodes[u]['label']
    if not label_node:
        label_node = graph.nodes[v]['label']
    label_dict[edge['label']].append(label_node)

# Clean labels
label_dict_cleaned = {}
for label, value in label_dict.items():
    new_label = re.sub(r'\[\d+\]', '', label)
    label_dict_cleaned[new_label] = list(set(value))

# Already validated labels
validated_filepath = "./validated_labels_2.json"
with open(validated_filepath, 'r') as f:
    already_validated_labels = json.load(f)

# Ask user to review labels
validated_labels = []
for label,values in label_dict_cleaned.items():

    if "ngap." in label:
        continue

    if label in already_validated_labels:
        continue

    values_list = '\n - '.join(values[:10])
    print(f"{label} :\n - {values_list}")

    user_input = input("Is this label correct? (y/n) ")
    if user_input.lower() == 'y':
        validated_labels.append(label)
    print('\n'*50)

json.dump(validated_labels, open("./validated_labels.json", "w"), indent=4)
