import ast
import copy
import logging

call_utils_func = lambda x, y: ast.Call(ast.Attribute(ast.Name('uast_utils'), x, ast.Load()), y, [])

def uast_to_python_ast(uast) -> ast.AST:
    tree = ast.Module()
    tree.body = convert_program(uast)
    return tree


def convert_program(program) -> list:
    types = program['types']
    funcs = program['funcs']
    body = []
    classes = {}

    imports = ['uast_utils', 'math', 're']
    for i in imports:
        body.append(ast.Import([ast.alias(i, None)]))

    if not(isinstance(types, list) and isinstance(funcs, list)):
        raise AssertionError("Invalid UAST: invalid types, funcs in program")

    for struct in types:
        if struct[1] == '__globals__':
            global_variables = list(types[0][2].keys())
            body.append(get_ast_from_string('__globals__ = dict()'))
            for var in global_variables:
                body.append(get_ast_from_string('__globals__["{}"] = None'.format(var)))
        else:
            class_body = []
            for field, val in struct[2].items():
                class_body.append(ast.Assign([ast.Name(field)], ast.Constant(0, 'n')))
            class_def = ast.ClassDef(struct[1], [], [], class_body, [])
            classes[struct[1]] = class_def

    for func in funcs:
        if func[2] == '__globals__.__init__':
            for stmt in func[5]:
                assert stmt[0] == 'assign'
                body.append(convert_stmt(stmt))
        elif func[2] == '__main__':
            main_func = copy.deepcopy(func)
            main_func[2] = 'main'
            body.append(convert_func(main_func))
        elif func[0] == 'ctor':
            assert func[1][-1] == '#' and func[1][:-1] in classes.keys()
            init_func = copy.deepcopy(func)
            init_func[2] = '__init__'
            init_func[3] = [['var', func[1], 'self']] + init_func[3]
            assert init_func[5][-1][0] == 'return'
            init_func[5] = init_func[5][:-1]
            classes[func[1][:-1]].body.append(convert_func(init_func))

            redirect_func = copy.deepcopy(func)
            redirect_func[5] = [['return', func[1], ['invoke', func[1], func[1][:-1], func[3]]]]
            body.append(convert_func(redirect_func))
        else:
            body.append(convert_func(func))

    body = body + list(classes.values())
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
        if arg[2] == 'this':
            arg[2] = 'self'
        args_list.append(ast.arg(arg[2], None))  # TODO: type annotation
    return ast.arguments(args=args_list, vararg=None, kwonlyargs=[],
                         kw_defaults=[], kwarg=None, defaults=[])


def convert_type(t:str) -> type:
    if t.endswith('*'):
        return list
    elif t.endswith('%'):
        return set
    elif t.endswith('#'):
        raise Exception("type __global__ triggered")
    elif t.startswith('<'):
        return dict
    else:
        type_map = {'bool': bool, 'char': chr, 'int': int, 'real': float}
        return type_map[t]


def convert_type_str(t:str) -> str:
    if t.endswith('*'):
        return 'list'
    elif t.endswith('%'):
        return 'set'
    elif t.endswith('#'):
        raise Exception("type __global__ triggered")
    elif t.startswith('<'):
        return 'dict'
    else:
        type_map = {'bool': 'bool', 'char': 'chr', 'int': 'int', 'real': 'float'}
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
        return ast.Break()
    elif stmt[0] == 'continue':
        return ast.Continue()
    elif stmt[0] == 'return':
        _, _, expr = stmt
        return ast.Return(convert_expr(expr))
    elif stmt[0] == 'noop':
        return ast.Pass()
    else:
        return ast.Expr(convert_expr(stmt))


def convert_expr(expr) -> ast.AST:
    if expr[0] == 'assign':
        _, _, lhs, rhs = expr
        if expr[1] == 'char' and rhs[1] == 'int':
            rhs = ['cast', 'char', rhs]

        if lhs[0] == 'var':
            return ast.NamedExpr(convert_expr(lhs), convert_expr(rhs))
        elif lhs[0] == 'invoke' and lhs[1] == 'char' and lhs[2] == 'array_index':
            array = convert_expr(lhs[3][0])
            idx = convert_expr(lhs[3][1])
            val = convert_expr(rhs)
            return ast.Assign([array], call_utils_func('replace_string_at', [array, idx, val]))
        else:
            return ast.Assign([convert_expr(lhs)], convert_expr(rhs))
    if expr[0] == 'var':
        return convert_var(expr)
    if expr[0] == 'field':
        _, _, exp, field = expr
        if exp[2] == '__globals__':
            return ast.Subscript(ast.Name('__globals__'), ast.Constant(field, 'u'), ast.Load())
        else:
            return ast.Attribute(convert_expr(exp), field, ast.Load())
    if expr[0] == 'val':
        _, typ, value = expr
        if typ == 'bool':
            if value == True:
                return ast.Constant(True, 'b')
            else:
                return ast.Constant(False, 'b')
        elif typ == 'char*':
            if value == None:
                return ast.Constant('', 'u')
            return ast.Constant(value.replace('\\t', '\t').replace('\\n', '\n'), 'u')
        elif typ == 'real':
            return ast.Constant(value, 'f')
        else:
            return ast.Constant(value, 'n')
    if expr[0] == 'invoke':
        _, _, func_name, arguments = expr
        args = [convert_expr(arg) for arg in arguments]
        bin_ops = {'+': ast.Add, '-': ast.Sub, '*': ast.Mult, '/': ast.Div,
                   '%': ast.Mod, '&': ast.BitAnd, '|': ast.BitOr, 'pow': ast.Pow,
                   '<<': ast.LShift, '>>': ast.RShift, '^': ast.BitXor}
        cmp_ops = {'==': ast.Eq, '!=': ast.NotEq, '<': ast.Lt, '>': ast.Gt,
                   '>=': ast.GtE, '<=': ast.LtE}
        bool_ops = {'&&': ast.And, '||': ast.Or}
        if func_name in bin_ops.keys():
            if arguments[0][1] == 'char' and arguments[1][1] == 'int':
                left = ast.Call(ast.Name('ord', ast.Load()), [args[0]], [])
            else:
                left = args[0]
            if arguments[1][1] == 'char' and arguments[0][1] == 'int':
                right = ast.Call(ast.Name('ord', ast.Load()), [args[1]], [])
            else:
                right = args[1]
            if func_name == '/' and arguments[0][1] == 'int' and arguments[1][1] == 'int':
                return ast.BinOp(left, ast.FloorDiv(), right)
            return ast.BinOp(left, bin_ops[func_name](), right)
        elif func_name in cmp_ops.keys():
            if arguments[0][1] == 'char' and arguments[1][1] == 'int':
                left = ast.Call(ast.Name('ord', ast.Load()), [args[0]], [])
            else:
                left = args[0]
            if arguments[1][1] == 'char' and arguments[0][1] == 'int':
                right = ast.Call(ast.Name('ord', ast.Load()), [args[1]], [])
            else:
                right = args[1]
            return ast.Compare(left, [cmp_ops[func_name]()], [right])
        elif func_name in bool_ops.keys():
            return ast.BoolOp(bool_ops[func_name](), args)
        elif func_name == '_ctor':
            if len(args) != 0:
                l = len(args)
                assert expr[1][-1*l:] == '*' * l
                return call_utils_func('list_init', [ast.List(args, ast.Load())])
            if convert_type(expr[1]) == list:
                func = 'list'
            elif convert_type(expr[1]) == dict:
                func = 'dict'
            elif convert_type(expr[1]) == set:
                func = 'set'
            else:
                func = expr[1]
        elif func_name.endswith('.__init__'):
            func = func_name[:-9]
        elif func_name in default_funcs.keys():
            if func_name in strict_args_funcs.keys():
                for i, t in strict_args_funcs[func_name]:
                    if not arguments[i][1] in t:
                        if 'char' in t:
                            args[i] = ast.Call(ast.Name('chr'), [args[i]], [])
                        if t == 'int':
                            args[i] = ast.Call(ast.Name('ord'), [args[i]], [])

            return get_funcs_ast(func_name, args)
        elif func_name == 'array_initializer':
            return ast.List(args, ast.Load())
        elif func_name == 'fill':
            return call_utils_func('array_fill', args)
        elif func_name == 'array_remove_idx':
            return call_utils_func('array_remove_idx', args)
        else:
            func = func_name
        logging.debug("invoke " + func)
        return ast.Call(ast.Name(func, ast.Load()), args, [])
    if expr[0] == '?:':
        _, _, condition, expr_if, expr_else = expr
        test = convert_expr(condition)
        return ast.IfExp(test, convert_expr(expr_if), convert_expr(expr_else))
    if expr[0] == 'cast':
        _, typ, exp = expr
        if typ == 'int' and exp[1] == 'char':
            func = 'ord'
        elif typ == 'char' and exp[1] == 'int':
            func = 'chr'
        else:
            func = convert_type_str(typ)
        return ast.Call(ast.Name(func), [convert_expr(exp)], [])


def convert_var(var) -> ast.Name:
    if var[2] == 'this':
        assert var[1].endswith('#')
        return ast.Name('self', ast.Load())
    else:
        return ast.Name(var[2], ast.Load())


def get_funcs_ast(func_name, args) -> ast.AST:
    func_str = default_funcs[func_name]
    func_ast = get_ast_from_string(func_str)
    return VariableSubstitutor(args).visit(func_ast)


def get_ast_from_string(s):
    tree = ast.parse(s).body[0]
    if isinstance(tree, ast.Expr):
        return tree.value
    else:
        return tree


class VariableSubstitutor(ast.NodeTransformer):
    def __init__(self, args):
        self.args = args

    def visit_Name(self, node):
        if node.id.startswith('x'):
            arg_num = int(node.id[1:])
            assert arg_num < len(self.args)
            return self.args[arg_num]
        else:
            return node


default_funcs = {
    '!': 'not x0',

    'str': 'str(x0)',
    'len': 'len(x0)',
    'sqrt': 'math.sqrt(x0)',
    'log': 'math.log(x0)',
    'atan2': 'math.atan2(x0, x1)',
    'sin': 'math.sin(x0)',
    'cos': 'math.cos(x0)',
    'pow': 'x0 ** x1',
    'round': 'math.floor(x0 + 0.5)',
    'floor': 'math.floor(x0)',
    'ceil': 'math.ceil(x0)',
    'min': 'min(x0, x1)',
    'max': 'max(x0, x1)',
    'abs': 'abs(x0)',
    'reverse': 'list(reversed(x0))',
    'lower': 'x0.lower()',
    'upper': 'x0.upper()',
    'sort': 'sorted(x0)',
    'copy_range': '[i for i in x0[x1:x2]]',
    'contains': 'x1 in x0',
    'concat': 'x0 + x1',

    'array_index': 'x0[x1]',
    'array_push': 'x0.append(x1)',
    'array_pop': 'x0.pop()',
    'array_insert': 'x0.insert(x1, x2)',
    'array_find': 'x0.index(x1) if x1 in x0 else -1',
    'array_find_next': 'x0.index(x1, x2) if x1 in x0[x2:] else -1',
    'array_concat': 'x0 + x1',

    'string_find': 'x0.find(x1)',
    'string_find_last': 'x0.rfind(x1)',
    'string_replace_one': 'x0.replace(x1, x2, 1)',
    'string_replace_all': 'x0.replace(x1, x2)',
    'string_insert': 'x0[:x1] + x2 + x0[x1:]',
    'string_split': "[i for i in re.split('|'.join(map(re.escape, x1)), x0) if i]",
    'string_trim': 'x0.strip()',
    'substring': 'x0[x1:x2]',
    'substring_end': 'x0[x1:]',

    'set_push': 'x0.add(x1)',
    'set_remove': 'x0.remove(x1)',

    'map_has_key': 'x1 in x0',
    'map_get': 'x0[x1] if x1 in x0 else None',
    'map_put': 'x0[x1] = x2',
    'map_keys': 'list(x0.keys())',
    'map_values': 'list(x0.values())',
    'map_remove_key': 'del x0[x1]',

}

strict_args_funcs = {
    'string_find': [(1, ['char', 'char*'])],
}
