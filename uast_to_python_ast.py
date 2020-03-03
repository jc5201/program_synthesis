import ast
import copy


def uast_to_python_ast(uast) -> ast.AST:
    tree = ast.Module()
    tree.body = convert_program(uast)
    return tree


def convert_program(program) -> list:
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
    return ast.FunctionDef(name, convert_args(arguments), stmt_list, [], None)


def convert_args(args) -> ast.arguments:
    args_list = []
    for arg in args:
        assert arg[0] == 'var'
        args_list.append(ast.arg(arg[2], None))  # TODO: type annotation
    return ast.arguments(args=args_list, vararg=None, kwonlyargs=[],
                         kw_defaults=[], kwarg=None, defaults=[])


def convert_type(t:str) -> type:
    if t.startswith('<'):
        return dict
    elif '*' in t:
        return list
    elif '%' in t:
        return set
    elif '#' in t:
        raise Exception("type __global__ triggered")
    else:
        type_map = {'bool': bool, 'char': chr, 'int': int, 'real': float}
        return type_map[t]


def convert_stmt(stmt) -> ast.AST:
    if stmt[0] == 'if':
        _, _, condition, body_if, body_else = stmt
        if_test = convert_expr(condition)
        if_body = [convert_stmt(line) for line in body_if]
        else_body = [convert_stmt(line) for line in body_else]
        node = ast.If(if_test, if_body, else_body)
        return node
    elif stmt[0] == 'foreach':
        _, _, var, collection, body = stmt
        target = convert_var(var)
        iter = convert_expr(collection)
        for_body = [convert_stmt(line) for line in body]
        node = ast.For(target, iter, for_body, [])
        return node
    elif stmt[0] == 'while':
        _, _, condition, body, body_inc = stmt
        while_test = convert_expr(condition)
        while_body = [convert_stmt(line) for line in body + body_inc]
        node = ast.While(while_test, while_body, [])
        return node
    elif stmt[0] == 'break':
        return ast.Break
    elif stmt[0] == 'continue':
        return ast.Continue
    elif stmt[0] == 'return':
        _, _, expr = stmt
        return ast.Return(convert_expr(expr))
    elif stmt[0] == 'noop':
        return ast.Pass
    else:
        return ast.Expr(convert_expr(stmt))


def convert_expr(expr) -> ast.AST:
    if expr[0] == 'assign':
        _, _, lhs, rhs = expr
        if lhs[0] == 'var':
            return ast.NamedExpr(convert_expr(lhs), convert_expr(rhs))
        else:
            return ast.Assign([convert_expr(lhs)], convert_expr(rhs))
    if expr[0] == 'var':
        return convert_var(expr)
    if expr[0] == 'field':
        _, _, exp, field = expr
        return ast.Attribute(convert_expr(exp), field, ast.Load())
    if expr[0] == 'val':
        _, typ, value = expr
        if typ == 'bool':
            if value == 'true':
                return ast.Constant(True, 'b')
            else:
                return ast.Constant(False, 'b')
        elif typ == 'str':
            return ast.Constant(value, 'u')
        else:
            return ast.Constant(value, 'n')
    if expr[0] == 'invoke':
        _, _, func_name, args = expr
        args = [convert_expr(arg) for arg in args]
        bin_ops = {'+': ast.Add, '-': ast.Sub, '*': ast.Mult, '/': ast.FloorDiv,
                   '%': ast.Mod, '&': ast.BitAnd, '|': ast.BitOr}
        cmp_ops = {'==': ast.Eq, '!=': ast.NotEq, '<': ast.Lt, '>': ast.Gt,
                   '>=': ast.GtE, '<=': ast.LtE}
        bool_ops = {'&&': ast.And, '||': ast.Or}
        if func_name in bin_ops.keys(): # TODO: if float, '/' ast.Div
            return ast.BinOp(args[0], bin_ops[func_name](), args[1])
        elif func_name in cmp_ops.keys():
            return ast.Compare(args[0], [cmp_ops[func_name]()], [args[1]])
        elif func_name in bool_ops.keys():
            return ast.BoolOp(bool_ops[func_name](), args)
        elif func_name == '_ctor':
            if convert_type(expr[1]) == list:
                return ast.List([], ast.Load())
            elif convert_type(expr[1]) == dict:
                return ast.Dict([], [])
            elif convert_type(expr[1]) == set:
                return ast.Set([])
            else:
                func = expr[1]
        elif func_name.endswith('.__init__'):
            func = func_name[:-9]
        elif func_name == 'array_initializer':
            return ast.List(args, ast.Load())
        elif func_name == 'array_push':
            return ast.Call(ast.Attribute(args[0], 'append', ast.Load()), [args[1]], [])
        elif func_name == 'array_index':
            return ast.Subscript(args[0], args[1], ast.Load())
        elif func_name == 'sort':
            func = 'sorted'
        else:
            func = func_name
        print("invoke " + func)
        return ast.Call(ast.Name(func, ast.Load()), args, [])
    if expr[0] == '?:':
        _, _, condition, expr_if, expr_else = expr
        test = convert_expr(condition)
        return ast.IfExp(test, convert_expr(expr_if), convert_expr(expr_else))
    if expr[0] == 'cast':
        _, typ, exp = expr
        func = str(convert_type(typ))  # TODO: check again
        return ast.Call(ast.Name(func), [convert_expr(exp)], [])


def convert_var(var) -> ast.Name:
    return ast.Name(var[2], ast.Load())

