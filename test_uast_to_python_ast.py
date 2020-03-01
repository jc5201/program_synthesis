import json
import uast_to_python_ast
import astunparse


def main():
    with open('naps_example_trainB.jsonl', 'r') as json_file:
        json_list = list(json_file)

    data = []
    for json_str in json_list:
        data.append(json.loads(json_str))
    uast1 = data[0]['code_tree']

    pcode1 = convert_uast_to_python_code(uast1)
    print(pcode1)

    for test in data[0]['tests']:
        input = test['input']
        output = test['output']
        if not test_input(input, output, pcode1):
            print("Failed")


def convert_uast_to_python_code(uast):
    py_ast = uast_to_python_ast.uast_to_python_ast(uast)
    pcode = astunparse.unparse(py_ast)
    return pcode


def test_input(input, output, code):
    # TODO: remove global
    global ret
    exec(code + '\n\nglobal ret\nret = main(' + str(input)[1:-1] + ')')
    # print(ret)
    return ret.strip() == output.strip()


main()

