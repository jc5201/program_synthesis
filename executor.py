import ast
import astunparse
import ast_gen_helper
import torch

import logging


bin_ops = {'+': ast.Add(), '-': ast.Sub(), '*': ast.Mult(), '/': ast.Div(), '//': ast.FloorDiv(),
           '%': ast.Mod(), '&': ast.BitAnd(), '|': ast.BitOr(), 'pow': ast.Pow(),
           '<<': ast.LShift(), '>>': ast.RShift(), '^': ast.BitXor()}
cmp_ops = {'==': ast.Eq(), '!=': ast.NotEq(), '<': ast.Lt(), '>': ast.Gt(),
           '>=': ast.GtE(), '<=': ast.LtE(), 'Is': ast.Is(), 'IsNot': ast.IsNot(),
           'In': ast.In(), 'NotIn': ast.NotIn()}
bool_ops = {'&&': ast.And(), '||': ast.Or()}
unary_ops = {'~': ast.Invert(), '!': ast.Not()}


def node_to_ast(tree, text_list):
    return rec_node_to_ast(tree, text_list)


def rec_node_to_ast(tree, text_list):
    node_type = ast_gen_helper.token_list[tree[0, 2].item()]

    children_start_idx = list(filter(lambda x: tree[x, 1] == tree[0, 0] and
                                               tree[x, 2] != ast_gen_helper.token_list.index('<End>'),
                                     range(tree.size(0))))
    children_end_idx = []
    for start_idx in children_start_idx:
        end_idx = -1
        for j in range(start_idx + 1, tree.size(0)):
            if tree[j, 1] == tree[0, 0]:
                end_idx = j
                break
        if end_idx == -1:
            assert False
        children_end_idx.append(end_idx)

    # logging.debug('node_to_ast: node_type: {}'.format(node_type))

    if node_type == '<root>':
        ast_node = ast.Module()
        ast_node.body = [rec_node_to_ast(tree[s:e], text_list)
                         for s, e in zip(children_start_idx, children_end_idx)]
        return ast_node
    elif node_type == 'funcdef':
        assert len(children_start_idx) == 1
        assert ast_gen_helper.token_list[children_start_idx[0][2].item()] == '<stmts>'
        func_name = 'func' + str(tree[0, 6 + 2].item())
        args_num = tree[0, 13 + 2].item()
        args_list = [ast.arg('arg' + str(i), None) for i in range(args_num)]
        ast_args = ast.arguments(args=args_list, vararg=None, kwonlyargs=[],
                                 kw_defaults=[], kwarg=None, defaults=[])
        ast_body = rec_node_to_ast(tree[children_start_idx[0]:children_end_idx[0]], text_list)
        ast_node = ast.FunctionDef(func_name, ast_args, ast_body, [], None)
        return ast_node
    elif node_type == 'assign':
        assert len(children_start_idx) == 2
        ast_lhs_name = rec_node_to_ast(tree[children_start_idx[0]:children_end_idx[0]], text_list)
        ast_rhs = rec_node_to_ast(tree[children_start_idx[1]:children_end_idx[1]], text_list)
        ast_node = ast.Assign([ast_lhs_name], ast_rhs)
        return ast_node
    elif node_type == 'return':
        assert len(children_start_idx) == 1
        ast_val = rec_node_to_ast(tree[children_start_idx[0]:children_end_idx[0]], text_list)
        ast_node = ast.Return(ast_val)
        return ast_node
    elif node_type == 'if':
        assert len(children_start_idx) == 3
        assert ast_gen_helper.token_list[children_start_idx[0][2].item()] == '<cond>'
        assert ast_gen_helper.token_list[children_start_idx[1][2].item()] == '<stmts>'
        assert ast_gen_helper.token_list[children_start_idx[2][2].item()] == '<stmts>'
        ast_iter = rec_node_to_ast(tree[children_start_idx[0]:children_end_idx[0]], text_list)
        ast_body = rec_node_to_ast(tree[children_start_idx[1]:children_end_idx[1]], text_list)
        ast_body_else = rec_node_to_ast(tree[children_start_idx[2]:children_end_idx[2]], text_list)
        ast_node = ast.If(ast_iter, ast_body, ast_body_else)
        return ast_node
    elif node_type == 'for':
        assert len(children_start_idx) == 3
        assert ast_gen_helper.token_list[children_start_idx[1][2].item()] == 'var'
        assert ast_gen_helper.token_list[children_start_idx[2][2].item()] == '<stmts>'
        ast_iter = rec_node_to_ast(tree[children_start_idx[0]:children_end_idx[0]], text_list)
        ast_var = rec_node_to_ast(tree[children_start_idx[1]:children_end_idx[1]], text_list)
        ast_body = rec_node_to_ast(tree[children_start_idx[2]:children_end_idx[2]], text_list)
        ast_node = ast.For(ast_var, ast_iter, ast_body, [])
        return ast_node
    elif node_type == 'while':
        assert len(children_start_idx) == 2
        assert ast_gen_helper.token_list[children_start_idx[0][2].item()] == '<cond>'
        assert ast_gen_helper.token_list[children_start_idx[1][2].item()] == '<stmts>'
        ast_cond = rec_node_to_ast(tree[children_start_idx[0]:children_end_idx[0]], text_list)
        ast_body = rec_node_to_ast(tree[children_start_idx[1]:children_end_idx[1]], text_list)
        ast_node = ast.While(ast_cond, ast_body, [])
        return ast_node
    elif node_type == 'call':
        assert len(children_start_idx) == 1
        assert ast_gen_helper.token_list[children_start_idx[0][2].item()] == '<exprs>'
        func_type = ast_gen_helper.token_list[tree[0, 4 + 2].item()]
        if func_type == '<func_type_func>':
            func_idx = tree[0, 6 + 2].item()
            ast_func_name = ast.Name('func' + str(func_idx), ast.Load())
        else:
            func_idx = tree[0, 5 + 2].item()
            ast_func_name = ast.Name(ast_gen_helper.builtin_func_list[func_idx], ast.Load())
        ast_args = rec_node_to_ast(tree[children_start_idx[0]:children_end_idx[0]], text_list)
        ast_node = ast.Call(ast_func_name, ast_args, [])
        return ast_node
    elif node_type == 'var':
        assert len(children_start_idx) == 0
        var_type = ast_gen_helper.token_list[tree[0, 7 + 2].item()]
        if var_type == '<var_type_var>':
            var_name = 'var' + str(tree[0, 9 + 2].item())
        elif var_type == '<var_type_arg>':
            var_name = 'arg' + str(tree[0, 8 + 2].item())
        else:
            assert False
        ast_node = ast.Name(var_name)
        return ast_node
    elif node_type == 'val':
        assert len(children_start_idx) == 0
        const_type = ast_gen_helper.token_list[tree[0, 10 + 2].item()]
        if const_type == '<const_type_int>':
            const_idx = tree[0, 11 + 2].item()
            ast_node = ast.Constant(ast_gen_helper.const_int_list[const_idx], 'n')
        elif const_type == '<const_type_float>':
            const_idx = tree[0, 12 + 2].item()
            ast_node = ast.Constant(ast_gen_helper.const_float_list[const_idx], 'f')
        else:
            const_idx = tree[0, 14 + 2].item()
            ast_node = ast.Constant(text_list[const_idx], 'u')
        return ast_node
    elif node_type == 'binop': #TODO: add boolop
        assert len(children_start_idx) == 2
        op_name = bin_ops[ast_gen_helper.bin_op_list[tree[0, 1 + 2].item()]]
        ast_left = rec_node_to_ast(tree[children_start_idx[0]:children_end_idx[0]], text_list)
        ast_right = rec_node_to_ast(tree[children_start_idx[1]:children_end_idx[1]], text_list)
        ast_node = ast.BinOp(ast_left, op_name, ast_right)
        return ast_node
    elif node_type == 'cmpop':
        assert len(children_start_idx) == 2
        op_name = cmp_ops[ast_gen_helper.cmp_op_list[tree[0, 2 + 2].item()]]
        ast_left = rec_node_to_ast(tree[children_start_idx[0]:children_end_idx[0]], text_list)
        ast_right = rec_node_to_ast(tree[children_start_idx[1]:children_end_idx[1]], text_list)
        ast_node = ast.Compare(ast_left, [op_name], [ast_right])
        return ast_node
    elif node_type == 'uop':
        assert len(children_start_idx) == 1
        op_name = unary_ops[ast_gen_helper.unary_op_list[tree[0, 3 + 2].item()]]
        ast_exp = rec_node_to_ast(tree[children_start_idx[0]:children_end_idx[0]], text_list)
        ast_node = ast.UnaryOp(op_name, ast_exp)
        return ast_node
    elif node_type == 'break':
        assert len(children_start_idx) == 0
        return ast.Break()
    elif node_type == 'continue':
        assert len(children_start_idx) == 0
        return ast.Continue()
    elif node_type == '<stmts>':
        return [rec_node_to_ast(tree[s:e], text_list)
                for s, e in zip(children_start_idx, children_end_idx)]
    elif node_type == '<exprs>':
        return [rec_node_to_ast(tree[s:e], text_list)
                for s, e in zip(children_start_idx, children_end_idx)]
    elif node_type == '<expr>' or node_type == '<cond>':
        assert len(children_start_idx) == 1
        return rec_node_to_ast(tree[children_start_idx[0]:children_end_idx[0]], text_list)
    else:
        assert False


def ast_to_code(py_ast):
    return astunparse.unparse(py_ast)


def compile_code(code):
    compile(code, '<string>', 'exec')


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
        try:
            python_ast = node_to_ast(tree, text)
            code = ast_to_code(python_ast)
        except Exception as ex:
            scores.append(-3.0)
            continue
        try:
            compile_code(code)
        except Exception as ex:
            scores.append(-2.0)
            continue
        try:
            score = 0
            for case in test:
                input = case['input']
                output = case['output']
                result = execute_code(code, input)
                if check_equal(result, output):
                    score = score + 1.0 / len(tests)
            scores.append(score)
        except Exception as ex:
            scores.append(-1.0)
            continue
    return scores

