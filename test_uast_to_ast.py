import json

with open('naps_example_trainB.jsonl', 'r') as json_file:
    json_list = list(json_file)

data = []
for json_str in json_list:
    data.append(json.loads(json_str))
uast1 = data[0]['code_tree']
uast2 = data[1]['code_tree']
