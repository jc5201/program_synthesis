import ast
import astunparse
import ast_gen_helper
import torch


bin_ops = {'+': ast.Add(), '-': ast.Sub(), '*': ast.Mult(), '/': ast.Div(),
           '%': ast.Mod(), '&': ast.BitAnd(), '|': ast.BitOr(), 'pow': ast.Pow(),
           '<<': ast.LShift(), '>>': ast.RShift(), '^': ast.BitXor()}
cmp_ops = {'==': ast.Eq(), '!=': ast.NotEq(), '<': ast.Lt(), '>': ast.Gt(),
           '>=': ast.GtE(), '<=': ast.LtE()}
bool_ops = {'&&': ast.And(), '||': ast.Or()}
unary_ops = {'~': ast.Invert(), '!': ast.Not()}


def node_to_ast(tree, text_list):
    return rec_node_to_ast(tree, text_list)

def rec_node_to_ast(node, text_list):
    node_type = ast_gen_helper.token_list[node[0].item()]
    children = node[2]

    if node_type == '<root>':
        ast_node = ast.Module()
        ast_node.body = [rec_node_to_ast(child, text_list) for child in children]
        return ast_node
    elif node_type == 'funcdef':
        assert children[0][0] == '<func_name>'
        assert children[1][0] == '<args_num>'
        assert children[2][0] == '<stmts>'
        func_name = 'func' + str(children[0][2][1].item())
        args_num = children[1][2][0].item()
        args_list = [ast.arg('arg' + str(i), None) for i in range(args_num)]
        ast_args = ast.arguments(args=args_list, vararg=None, kwonlyargs=[],
                                 kw_defaults=[], kwarg=None, defaults=[])
        ast_body = rec_node_to_ast(children[2], text_list)
        ast_node = ast.FunctionDef(func_name, ast_args, ast_body, [], None)
        return ast_node
    elif node_type == 'assign':
        ast_lhs_name = rec_node_to_ast(children[0], text_list)
        ast_rhs = rec_node_to_ast(children[1], text_list)
        ast_node = ast.Assign([ast_lhs_name], ast_rhs)
        return ast_node
    elif node_type == 'return':
        ast_val = rec_node_to_ast(children[0], text_list)
        ast_node = ast.Return(ast_val)
        return ast_node
    elif node_type == 'if':
        ast_iter = rec_node_to_ast(children[0], text_list)
        ast_body = rec_node_to_ast(children[1], text_list)
        ast_body_else = rec_node_to_ast(children[2], text_list)
        ast_node = ast.If(ast_iter, ast_body, ast_body_else)
        return ast_node
    elif node_type == 'for':
        ast_iter = rec_node_to_ast(children[0], text_list)
        ast_var = rec_node_to_ast(children[1], text_list)
        ast_body = rec_node_to_ast(children[2], text_list)
        ast_node = ast.For(ast_var, ast_iter, ast_body, [])
        return ast_node
    elif node_type == 'while':
        ast_cond = rec_node_to_ast(children[0], text_list)
        ast_body = rec_node_to_ast(children[1], text_list)
        ast_node = ast.While(ast_cond, ast_body, [])
        return ast_node
    elif node_type == 'call':
        func_type = ast_gen_helper.token_list[children[0][2][0].item()]
        func_name = children[0][2][1].item()
        if func_type == '<func_type_func>':
            ast_func_name = ast.Name('func' + str(func_name), ast.Load())
        else:
            ast_func_name = ast.Name(ast_gen_helper.builtin_func_list[func_name], ast.Load())
        ast_args = rec_node_to_ast(children[1], text_list)
        ast_node = ast.Call(ast_func_name, ast_args, [])
        return ast_node
    elif node_type == 'var':
        ast_node = rec_node_to_ast(children[1], text_list)
        return ast_node
    elif node_type == 'val':
        const_type = ast_gen_helper.token_list[children[0][2][0].item()]
        const_idx = children[0][2][1].item()
        if const_type == '<const_type_int>':
            ast_node = ast.Constant(ast_gen_helper.const_int_list[const_idx], 'n')
        elif const_type == '<const_type_float>':
            ast_node = ast.Constant(ast_gen_helper.const_float_list[const_idx], 'f')
        else:
            ast_node = ast.Constant(text_list[const_idx], 'u')
        return ast_node
    elif node_type == 'binop': #TODO: add boolop
        op_name = bin_ops[ast_gen_helper.bin_op_list[children[0][2][0].item()]]
        ast_left = rec_node_to_ast(children[1], text_list)
        ast_right = rec_node_to_ast(children[2], text_list)
        ast_node = ast.BinOp(ast_left, op_name, ast_right)
        return ast_node
    elif node_type == 'cmpop':
        op_name = cmp_ops[ast_gen_helper.cmp_op_list[children[0][2][0].item()]]
        ast_left = rec_node_to_ast(children[1], text_list)
        ast_right = rec_node_to_ast(children[2], text_list)
        ast_node = ast.Compare(ast_left, [op_name], [ast_right])
        return ast_node
    elif node_type == 'uop':
        op_name = unary_ops[ast_gen_helper.unary_op_list[children[0][2][0].item()]]
        ast_exp = rec_node_to_ast(children[1], text_list)
        ast_node = ast.UnaryOp(op_name, ast_exp)
        return ast_node
    elif node_type == 'break':
        return ast.Break()
    elif node_type == 'continue':
        return ast.Continue()
    elif node_type == '<var_name>':
        var_type = ast_gen_helper.token_list[children[0].item()]
        var_idx = children[1].item()
        if var_type == '<var_type_var>':
            var_name = 'var' + str(var_idx)
        else:
            var_name = 'arg' + str(var_idx)
        ast_node = ast.Name(var_name)
        return ast_node
    elif node_type == '<func_name>':
        func_type = ast_gen_helper.token_list[children[0].item()]
        func_name = children[1].item()
        if func_type == '<func_type_func>':
            ast_func_name = ast.Name('func' + str(func_name, ast.Load()))
        else:
            ast_func_name = ast.Name(ast_gen_helper.builtin_func_list[func_name], ast.Load())
        return ast_func_name
    else:
        assert False


def ast_to_code(py_ast):
    return astunparse.unparse(py_ast)


def compile_code(code):
    compile(code, 'exec')


def execute_code(code, input):
    global ret
    cmd = '{}\n\nglobal ret\nret = func0({})'.format(code, str(input)[1:-1])
    exec(cmd, globals())
    return ret


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


def evaluate_code(trees, texts, tests):
    scores = []
    for tree, text, test in zip(trees, texts, tests):
        python_ast = node_to_ast(tree, text)
        try:
            code = ast_to_code(python_ast)
        except:
            scores.append(-3.0)
        try:
            compile_code(code)
        except:
            scores.append(-2.0)
        try:
            score = 0
            for case in test:
                input = case['input']
                output = case['output']
                result = execute_code(code, input)
                if check_equal(result, output):
                    score = score + 1.0 / len(inputs)
            scores.append(score)
        except:
            scores.append(-1.0)
    return scores

