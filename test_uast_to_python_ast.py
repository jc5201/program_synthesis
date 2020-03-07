import json
import uast_to_python_ast
import astunparse
import loader
import logging


def main():
    test_trainB_valid(start_idx=0)


def test_trainB_valid(start_idx=0):
    valid_data = loader.load_naps_valid()
    for i, data in enumerate(valid_data):
        if i < start_idx:
            continue
        logging.debug('Start testing data#{}'.format(i))
        uast = data['code_tree']
        pcode = convert_uast_to_python_code(uast)

        for j, test in enumerate(data['tests']):
            input = test['input']
            output = test['output']
            if not test_input(input, output, pcode):
                logging.error("Failed: data#{}, test case#{}".format(i, j))
                logging.error("expected '{}', got '{}'".format(output, ret))
                exit(-1)
    logging.info("Finished testing naps_valid")


def convert_uast_to_python_code(uast):
    py_ast = uast_to_python_ast.uast_to_python_ast(uast)
    pcode = astunparse.unparse(py_ast)
    return pcode


def test_input(input, output, code):
    # TODO: remove global
    global ret
    cmd = '{}\n\nglobal ret\nret = main({})'.format(code, str(input)[1:-1])
    exec(cmd, globals())
    logging.debug(ret)
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


if __name__ == "__main__":
    logging.getLogger().setLevel(logging.DEBUG)
    logging.getLogger().addHandler(logging.StreamHandler())
    main()

