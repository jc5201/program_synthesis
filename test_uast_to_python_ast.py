import json
import uast_to_python_ast
import astunparse

skip_list = [122, 196, 1410, 1811,          # := in list comp
             327, 734, 786, 1131, 1160, 1779, 1784,    # continue
             704, 707, 714, 806, 832, 877, 936, 968, 1032, 1055, 1072, 1215, 1463,  # := with a[b] or a.b
             984, 1388, 1785,               # concat(str, int)
             1438, 1554,                    # array[char]
             1873, 1986, 2112,              # '10' != '10.0'
             2035,                          # TODO
             ]


def main():
    test_trainB(start_idx=0)


def test_trainB(start_idx=0, skip_partial=True):
    with open('naps.trainB.1.0.jsonl', 'r') as json_file:
        json_list = list(json_file)

    for i, json_str in enumerate(json_list):
        if i < start_idx or i in skip_list:
            continue
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
                print("Failed: data#{}, test case#{}".format(i, j))
                print("expected '{}', got '{}'".format(output, ret))
                exit(-1)

def convert_uast_to_python_code(uast):
    py_ast = uast_to_python_ast.uast_to_python_ast(uast)
    imports = 'import uast_utils\nimport math\nimport re'
    pcode = imports + astunparse.unparse(py_ast)
    return pcode


def test_input(input, output, code):
    # TODO: remove global
    global ret
    cmd = '{}\n\nglobal ret\nret = main({})'.format(code, str(input)[1:-1])
    exec(cmd, globals())
    print(ret)
    return check_equal(ret, output)


def check_equal(a, b):
    if isinstance(b, list):
        if len(a) != len(b):
            return False
        for x, y in zip(a, b):
            if not check_equal(x, y):
                return False
        return True
    elif isinstance(b, str):
        return a.strip() == b.strip()
    else:
        return a == b


main()

