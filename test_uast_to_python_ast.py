import json
import uast_to_python_ast
import astunparse


def main():
    test_trainB()


def test_trainB(skip_partial=True):
    with open('naps.trainB.1.0.jsonl', 'r') as json_file:
        json_list = list(json_file)

    for i, json_str in enumerate(json_list):
        print("Start data#" + str(i))
        data = json.loads(json_str)

        if skip_partial and data['is_partial']:
            print("Skipping data#" + str(i))
            continue
        uast = data['code_tree']
        pcode = convert_uast_to_python_code(uast)

        for j, test in enumerate(data['tests']):
            input = test['input']
            output = test['output']
            if not test_input(input, output, pcode):
                print("Failed: data#" + str(i) + ', test case#' + str(j))
                exit(-1)

def convert_uast_to_python_code(uast):
    py_ast = uast_to_python_ast.uast_to_python_ast(uast)
    pcode = 'import uast_utils' + astunparse.unparse(py_ast)
    return pcode


def test_input(input, output, code):
    # TODO: remove global
    global ret
    exec(code + '\n\nglobal ret\nret = main(' + str(input)[1:-1] + ')')
    print(ret)
    return str(ret).strip() == str(output).strip()


main()

