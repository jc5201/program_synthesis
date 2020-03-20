import torch
import torch.nn as nn
import ast_gen_helper


class CodeSynthesisModel:
    def __init__(self, latent_vector_dim=16):
        self.text_encoder = TextEncoder(latent_vector_dim)
        self.discriminator = Discriminator(latent_vector_dim)

        self.step = 0

    def forward_text_encoder(self, x, train=True):
        return self.text_encoder.forward(x)

    def forward_discriminator(self, tree, train=True):
        return self.discriminator.forward(tree)

    def get_steps(self):
        return self.step

    def increase_step(self, amount=1):
        self.step = self.step + amount

    def parameters(self):
        params = []
        params.append({'params': self.text_encoder.parameters()})
        params.append({'params': self.discriminator.parameters()})
        return params


class TextEncoder(nn.Module):
    def __init__(self, latent_vector_dim):
        super(TextEncoder, self).__init__()

        self.vector_token = torch.LongTensor([1999])
        embed_dict_size = 2000
        self.embed_dim = 32
        self.lstm_out_dim = self.embed_dim
        lstm_layer_num = 2

        self.embedding = nn.Embedding(embed_dict_size, self.embed_dim)
        self.lstm = nn.LSTM(self.embed_dim, self.lstm_out_dim, lstm_layer_num)
        self.ff = nn.Sequential(
            nn.Linear(self.lstm_out_dim, self.lstm_out_dim),
            nn.BatchNorm1d(self.lstm_out_dim),
            nn.Linear(self.lstm_out_dim, latent_vector_dim),
            nn.BatchNorm1d(latent_vector_dim)
        )

    def forward(self, xs):
        ol = []
        vl = []
        for x in xs:
            embedded = self.embedding(x).view(-1, 1, self.embed_dim)
            lstm_out, hid = self.lstm(embedded)
            assert lstm_out.size(0) == x.size(0)

            vector_token = self.embedding(self.vector_token).view(1, 1, self.embed_dim)
            v, _ = self.lstm(vector_token, hid)
            ol.append(lstm_out)
            vl.append(v.view(-1))
        return ol, torch.stack(vl)


class Discriminator(nn.Module):
    def __init__(self, note_dim, text_encoder):
        super(Discriminator, self).__init__()
        self.text_encoder = text_encoder

        embed_dict_size = 200
        self.embed_dim = 32
        hidden_dim = 32
        lstm_out_dim = self.text_encoder.lstm_out_dim

        add_prefix = lambda x, y:list(map(lambda e: y + e, x))
        self.token_list = ast_gen_helper.token_list + add_prefix(ast_gen_helper.bin_op_list, 'bin_op_') \
                          + add_prefix(ast_gen_helper.cmp_op_list, 'cmp_op_') \
                          + add_prefix(ast_gen_helper.unary_op_list, 'unary_op_') \
                          + add_prefix(ast_gen_helper.builtin_func_list, 'builtin_func_') \
                          + add_prefix(range(ast_gen_helper.var_num_limit), 'name_var') \
                          + add_prefix(range(ast_gen_helper.arg_num_limit), 'name_arg') \
                          + add_prefix(range(ast_gen_helper.func_num_limit), 'name_func') \
                          + add_prefix(ast_gen_helper.const_int_list, 'const_int_') \
                          + add_prefix(ast_gen_helper.const_float_list, 'const_float_')

        self.embedding = nn.Embedding(embed_dict_size, self.embed_dim)
        self.leaf_ff = nn.Sequential(
            nn.Linear(self.embed_dim * 2, hidden_dim)
        )
        self.node_ff = nn.Sequential(
            nn.Linear(self.embed_dim * 2 + hidden_dim, hidden_dim),
            nn.ReLU()
        )
        self.ptr_ff = nn.Linear(note_dim + lstm_out_dim, hidden_dim)

        self.ff_model = nn.Sequential(
            nn.Linear(self.embed_dim, hidden_dim)
        )
        self.tail_score = nn.Sequential(
            nn.Linear(hidden_dim, 1),
            nn.Tanh()
        )

    def child_sum(self, node):
        node_type = node[0]
        node_children = node[2]

        def long_ind(x):
            torch.LongTensor(self.token_list.index(x))

        if node_type.item() == self.token_list.index('<var_name>'):
            var_type, var_name = node_children[0], node_children[1]
            var_idx = long_ind('name_var0') \
                      + (var_type - long_ind('<var_type_var>')) \
                      * (long_ind('name_arg0') - long_ind('name_var0')) \
                      + var_name
            v = self.embedding(torch.cat((node_type, var_idx))).view(1, -1)
            return self.leaf_ff(v)
        elif node_type.item() == self.token_list.index('<const>'):
            const_type, const_val = node_children[0], node_children[1]
            if self.token_list[const_type.item()] == '<const_type_int>':
                v =


        new_tree = []
        for i, node in enumerate(tree):
            if isinstance(node, list):
                children = [self.child_sum(n) for n in node]
                if len(children) == 0:
                    embedded = self.embedding(torch.LongTensor([2]).view(1, 1)).view(-1)
                    new_tree.append(embedded)
                else:
                    lstm_input = torch.stack(children) if len(children) == 1 else children[0]
                    out, _ = self.child_sum_lstm(lstm_input.view(-1, 1, self.embed_dim))
                    new_tree.append(out[-1, :, :].view(-1))
            else:
                embedded = self.embedding(torch.LongTensor([node]).view(1, 1)).view(-1)
                new_tree.append(embedded)
        return self.child_sum_ff(torch.cat(new_tree).view(1, -1)).view(-1)

    def forward(self, texts, trees):
        lstm_out_list, first_notes = self.text_encoder(texts)

        for i, (note, tree) in enumerate(zip(first_notes, trees)):
            self.child_sum(tree)
            pass

        # each node of tree should be [sort, type, name, args, cond, body, body_else, body_inc, iter_foreach, target]
        code_out = []
        for tree in trees:
            code_out.append(self.child_sum(tree))
        code_out_gathered = torch.stack(code_out)
        ff_out = self.ff_model(code_out_gathered)
        score = self.tail_score(ff_out)
        summary = self.tail_summary(ff_out)
        return score, summary


class Generator(nn.Module):
    def __init__(self, note_dim, text_encoder):
        super(Generator, self).__init__()
        self.text_encoder = text_encoder

        embed_dict_size = 100
        self.embed_dim = 8
        hidden_dim = self.embed_dim * 2 + note_dim
        prime_type_num = 15
        bin_op_num = len(ast_gen_helper.bin_op_list)
        cmp_op_num = len(ast_gen_helper.cmp_op_list)
        unary_op_num = len(ast_gen_helper.unary_op_list)
        func_type_num = 2   # builtin, func (func0 == main)
        builtin_func_num = len(ast_gen_helper.builtin_func_list)
        func_num_limit = ast_gen_helper.func_num_limit
        var_type_num = 2    # var, arg
        arg_num_limit = ast_gen_helper.arg_num_limit
        var_num_limit = ast_gen_helper.var_num_limit
        const_type_num = 3    # int, float, pointer
        const_int_num = len(ast_gen_helper.const_int_list)
        const_float_num = len(ast_gen_helper.const_float_list)
        self.token_list = ast_gen_helper.token_list
        self.leaf_list = ['break', 'continue']

        self.type_embedding = nn.Embedding(embed_dict_size, self.embed_dim)
        self.sibling_mask = nn.Sequential(self.embed_dim + note_dim, self.embed_dim)
        self.predict_type = nn.Sequential(
            nn.Linear(hidden_dim, prime_type_num),
            nn.Softmax()
        )
        self.predict_note = nn.Linear(hidden_dim + self.embed_dim, note_dim)
        self.predict_bin_op = nn.Sequential(nn.Linear(note_dim, bin_op_num), nn.Softmax())
        self.predict_cmp_op = nn.Sequential(nn.Linear(note_dim, cmp_op_num), nn.Softmax())
        self.predict_unary_op = nn.Sequential(nn.Linear(note_dim, unary_op_num), nn.Softmax())
        self.predict_func_type = nn.Sequential(nn.Linear(note_dim, func_type_num), nn.Softmax())
        self.predict_builtin_func = nn.Sequential(nn.Linear(note_dim, builtin_func_num), nn.Softmax())
        self.predict_func_name = nn.Sequential(nn.Linear(note_dim, func_num_limit), nn.Softmax())
        self.predict_var_type = nn.Sequential(nn.Linear(note_dim, var_type_num), nn.Softmax())
        self.predict_arg_name = nn.Sequential(nn.Linear(note_dim, arg_num_limit), nn.Softmax())
        self.predict_var_name = nn.Sequential(nn.Linear(note_dim, var_num_limit), nn.Softmax())
        self.predict_const_type = nn.Sequential(nn.Linear(note_dim, const_type_num), nn.Softmax())
        self.predict_const_int = nn.Sequential(nn.Linear(note_dim, const_int_num), nn.Softmax())
        self.predict_const_float = nn.Sequential(nn.Linear(note_dim, const_float_num), nn.Softmax())
        self.predict_args_num = nn.Sequential(nn.Linear(note_dim, arg_num_limit), nn.Softmax())
        self.ptr_ff = nn.Linear(self.text_encoder.lstm_out_dim + note_dim, 1)

    def predict_const_copy(self, note, lstm_out):
        l = lstm_out.size(0)
        cc = torch.cat([note.expand(l, note.size(1)), lstm_out], dim=1)
        return nn.Softmax(self.ptr_ff(cc).view(1, -1))

    def next_node(self, parent, sibling, note, lstm_out, train):
        assert parent.size()[0] == 1
        mask = self.sibling_mask(torch.cat([parent, note]))
        sibling_notes = list(map(lambda x: x[1], sibling))
        sibling_sum = self.sibling_sum(mask, sibling_notes)
        hidden = torch.cat([parent, sibling_sum, note], dim=1)
        sort_prob = self.predict_type(hidden)
        sort = self.pick(sort_prob, train)
        note = self.predict_note(torch.cat([hidden, self.type_embedding(sort)]))
        if self.token_list[sort.item()] == 'if':
            child = self.gen_child(sort, note, ['<cond>', '<stmts>', '<stmts>'], train)
        elif self.token_list[sort.item()] == 'for':
            child = self.gen_child(sort, note, ['<expr>', '<stmts>'], train)
        elif self.token_list[sort.item()] == 'while':
            child = self.gen_child(sort, note, ['<cond>', '<stmts>'], train)
        elif self.token_list[sort.item()] == 'return':
            child = self.gen_child(sort, note, ['<expr>'], train)
        elif self.token_list[sort.item()] == 'call':
            child = self.gen_child(sort, note, ['<func_name>', '<exprs>'], train)
        elif self.token_list[sort.item()] == 'var':
            child = self.gen_child(sort, note, ['<var_name>'], train)
        elif self.token_list[sort.item()] == 'val':
            child = self.gen_child(sort, note, ['<const>'], train)
        elif self.token_list[sort.item()] == 'binop':
            child = self.gen_child(sort, note, ['<op_name>', '<expr>', '<expr>'], train)
        elif self.token_list[sort.item()] == 'cmpop':
            child = self.gen_child(sort, note, ['<op_name>', '<expr>', '<expr>'], train)
        elif self.token_list[sort.item()] == 'uop':
            child = self.gen_child(sort, note, ['<op_name>', '<expr>'], train)
        elif self.token_list[sort.item()] == 'assign':
            child = self.gen_child(sort, note, ['<var_name>', '<expr>'], train)
        elif self.token_list[sort.item()] == 'funcdef':
            child = self.gen_child(sort, note, ['<func_name>', '<args_num>', '<stmts>'], train)
        # TODO: []
        else:
            child = []
        return sort, note, child

    def gen_child(self, parent_type, parent_note, child_type_list, lstm_out, train):
        assert len(child_type_list) != 0
        child_list = []
        child_note_list = []
        mask = self.sibling_mask(torch.cat([self.type_embedding(parent_type), parent_note]))
        for child_type in child_type_list:
            child = torch.LongTensor([self.token_list.index(child_type)]).view(1, -1)
            note = self.predict_note(torch.cat([self.type_embedding(parent_type),
                                                self.sibling_sum(mask, child_note_list),
                                                parent_note,
                                                self.type_embedding(child)], dim=1))
            if child_type == '<var_name>':
                var_type = self.pick(self.predict_var_type(note), train)
                var_long_rep = torch.LongTensor([self.token_list.index('<var_type_var>')])
                var_type_ = var_long_rep.view(1, 1) + var_type.view(1, 1)
                if self.token_list[var_type_.item()] == '<var_type_var>':
                    var_name = self.pick(self.predict_var_name(note), train)
                else:
                    var_name = self.pick(self.predict_arg_name(note), train)
                aux = [var_type_, var_name]
            elif child_type == '<const>':
                const_type = self.pick(self.predict_const_type(note), train)
                const_long_rep = torch.LongTensor([self.token_list.index('<const_type_int>')])
                const_type_ = const_long_rep.view(1, 1) + const_type.view(1, 1)
                if self.token_list[const_type_.item()] == '<const_type_int>':
                    const_val = self.pick(self.predict_const_int(note), train)
                elif self.token_list[const_type_.item()] == '<const_type_float>':
                    const_val = self.pick(self.predict_const_float(note), train)
                else:
                    const_val = self.pick(self.predict_const_copy(note, lstm_out), train)
                aux = [const_type_, const_val]
            elif child_type == '<op_name>':
                if self.token_list[parent_type.item()] == 'binop':
                    op_name = self.pick(self.predict_bin_op(note), train)
                    aux = [op_name]
                elif self.token_list[parent_type.item()] == 'cmpop':
                    op_name = self.pick(self.predict_cmp_op(note), train)
                    aux = [op_name]
                else:
                    op_name = self.pick(self.predict_unary_op(note), train)
                    aux = [op_name]
            elif child_type == '<func_name>':
                if self.token_list[parent_type.item()] == 'call':
                    func_type = self.pick(self.predict_func_type(note), train)
                    func_long_rep = torch.LongTensor([self.token_list.index('<func_type_func>')])
                    func_type_ = func_long_rep.view(1, 1) + func_type.view(1, 1)
                    if self.token_list[func_type_.item()] == '<func_type_func>':
                        func_name = self.pick(self.predict_func_name(note), train)
                    else:
                        func_name = self.pick(self.predict_builtin_func(note), train)
                    aux = [func_type_, func_name]
                else:
                    func_type = torch.LongTensor([self.token_list.index('<func_type_func>')]).view(1, 1)
                    func_name = self.pick(self.predict_func_name(note), train)
                    aux = [func_type, func_name]
            elif child_type == '<args_num>':
                args_num = self.pick(self.predict_args_num(note), train)
                aux = [args_num]
            else:
                aux = []
            child_list.append((child, note, aux))
            child_note_list.append(note)
        return child_list

    def sibling_sum(self, mask, note_list):
        sibling_num = len(note_list)
        if sibling_num == 0:
            return torch.zeros(1, self.embed_dim)
        masked = mask.expand(sibling_num, self.embed_dim) * torch.stack(note_list)
        return torch.sum(masked, dim=0).view(1, -1)

    def pick(self, prob, train):
        if train:
            dist = torch.distributions.Categorical(prob.view(-1))
            result = dist.sample()
        else:
            _, result = torch.max(prob.view(-1), 0)
        return result

    def forward(self, xs, train=True):
        lstm_out_list, first_notes = self.text_encoder(xs)

        trees_list = []
        for i in range(len(xs)):
            traverse_list = []
            first_note = first_notes[i]
            root_sort = torch.LongTensor([self.token_list.index('<root>')]).view(1, 1)
            root = [self.type_embedding(root_sort), first_note, []]
            traverse_list.append(root)
            trees_list.append(root)
            while len(traverse_list) != 0:
                current = traverse_list[0]
                traverse_list = traverse_list[1:]
                child = current[2]
                while True:
                    next, note, next_child_list = self.next_node(current[0], child, current[1], lstm_out_list[i], train)
                    if next.item() == self.token_list.index('<End>'):
                        break
                    elif len(next_child_list) == 0:
                        child.append([next, note, []])
                    else:
                        n = [next, note, []]
                        for c in next_child_list:
                            cn = [c[0], c[1], c[2]]
                            n[2].append(cn)
                            if len(c[2]) == 0:  # else, there is aux
                                traverse_list.append(cn)
                        child.append(n)

        return trees_list

