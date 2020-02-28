import ast
import copy


def uast_to_python_ast(uast) -> ast.AST:
    tree = ast.AST()
    tree.body = convert_program(uast)
    return tree


def convert_program(program) -> list[ast.AST]:
    types = program['types']
    funcs = program['funcs']
    body = []

    if not(isinstance(types, list) and isinstance(funcs, list)):
        raise AssertionError("Invalid UAST: invalid types, funcs in program")

    elif len(types) == 1:
        assert types[0][2] == '__globals__'
        global_variables = list(types[0][2].keys())
        body.append(ast.Global(global_variables))

    for func in funcs:
        if func[2] == '__globals__.__init__':
            for stmt in func[5]:
                assert stmt[0] == 'assign'
                body.append(convert_stmt(stmt))
        elif func[2] == '__main__':
            main_func = copy.deepcopy(func)
            main_func[2] = 'main'
            body.append(convert_func(main_func))
        else:
            body.append(convert_func(func))

    return body


def convert_func(func) -> ast.AST:
    kind, return_type, name, arguments, variables, body = func

    if '.' in name:
        NotImplemented

    stmt_list = [convert_stmt(stmt) for stmt in body]
    return ast.FunctionDef(name, convert_args(arguments), stmt_list)


def convert_args(args) -> ast.arguments:
    args_list = []
    for arg in args:
        assert arg[0] == 'var'
        args_list.append(ast.arg(convert_type(arg[1]), arg[2]))
    return ast.arguments(args=args_list, varargs=None, kwarg=None, defaults=[])


def convert_stmt(stmt) -> ast.AST:
    NotImplemented


def convert_type(type) -> type:
    NotImplemented

