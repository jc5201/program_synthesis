import json
import uast_to_ast
import astunparse

with open('naps_example_trainB.jsonl', 'r') as json_file:
    json_list = list(json_file)

data = []
for json_str in json_list:
    data.append(json.loads(json_str))
uast1 = data[0]['code_tree']
uast2 = data[1]['code_tree']

past1 = uast_to_ast.convert_program(uast1)
pcode = astunparse.unparse(past1)
print(pcode)

