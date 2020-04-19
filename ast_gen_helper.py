
token_list = ['<pad>', '<root>',
              '<End>', 'funcdef', 'assign', 'return', 'if', 'for', 'while', 'call', 'var', 'val',
              'binop', 'cmpop', 'uop', 'break', 'continue',
              '<cond>', '<stmts>', '<expr>', '<exprs>',
              '<var_name>', '<func_name>', '<op_name>', '<args_num>', '<const>',

              '<var_type_var>', '<var_type_arg>',
              '<const_type_int>', '<const_type_float>', '<const_type_pointer>',
              '<func_type_func>', '<func_type_builtin>']

bin_op_list = ['+', '-', '*', '/', '//', '%', '**', '<<', '>>', '|', '&', '^']

cmp_op_list = ['==', '!=', '<', '<=', '>', '>=', 'Is', 'IsNot', 'In', 'NotIn']

unary_op_list = ['~', '!']

builtin_func_list = ['list', 'dict', 'sorted', 'reversed', 'len',
                     'str', 'int', 'float',
                     'math.max', 'math.min', 'math.floor', 'math.ceil', 'math.abs']

const_int_list = [-1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 100]

const_float_list = [-1.0, 0.0, 0.01, 0.1, 1.0, 2.0, 3.0, 10.0, 100.0, 1000.0]

func_num_limit = 20
arg_num_limit = 10
var_num_limit = 20

