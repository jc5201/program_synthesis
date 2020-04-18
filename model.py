import torch
import torch.nn as nn
import ast_gen_helper


class CodeSynthesisModel:
    def __init__(self, note_dim=16):
        lstm_out_dim = 32
        self.text_encoder = TextEncoder(note_dim, lstm_out_dim)
        self.generator = Generator(note_dim, lstm_out_dim)
        self.discriminator = Discriminator(note_dim, lstm_out_dim)

        self.step = 0

    def forward_text_encoder(self, texts, train=True):
        return self.text_encoder.forward(texts)

    def forward_generator(self, texts, trees, train=True):
        lstm_out_list, first_notes = self.forward_text_encoder(texts, train)
        return self.generator.forward(lstm_out_list, first_notes, trees, train)

    def forward_discriminator(self, texts, trees, train=True):
        lstm_out_list, first_notes = self.forward_text_encoder(texts, train)
        return self.discriminator.forward(lstm_out_list, first_notes, trees, train)

    def get_steps(self):
        return self.step

    def increase_step(self, amount=1):
        self.step = self.step + amount

    def parameters(self, text_lr, gen_lr, dis_lr):
        params = []
        params.append({'params': self.text_encoder.parameters(), 'lr': text_lr})
        params.append({'params': self.generator.parameters(), 'lr': gen_lr})
        params.append({'params': self.discriminator.parameters(), 'lr': dis_lr})
        return params

    def set_generator_trainable(self, val):
        for param in self.generator.parameters():
            param.requires_grad = val

    def set_discriminator_trainable(self, val):
        for param in self.discriminator.parameters():
            param.requires_grad = val

    def set_text_encoder_trainable(self, val):
        for param in self.text_encoder.parameters():
            param.requires_grad = val


class TextEncoder(nn.Module):
    def __init__(self, latent_vector_dim, lstm_out_dim):
        super(TextEncoder, self).__init__()

        self.vector_token = torch.LongTensor([1999])
        embed_dict_size = 2000
        self.embed_dim = 32
        self.lstm_out_dim = lstm_out_dim
        lstm_layer_num = 2

        self.embedding = nn.Embedding(embed_dict_size, self.embed_dim)
        self.lstm = nn.LSTM(self.embed_dim, self.lstm_out_dim, lstm_layer_num)
        self.ff = nn.Sequential(
            nn.Linear(self.lstm_out_dim, self.lstm_out_dim),
            nn.Tanh(),
            nn.Linear(self.lstm_out_dim, latent_vector_dim),
            nn.Tanh()
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
        notes = self.ff(torch.stack(vl).view(-1, self.lstm_out_dim))
        return ol, notes


class Discriminator(nn.Module):
    def __init__(self, note_dim, lstm_out_dim):
        super(Discriminator, self).__init__()

        embed_dict_size = 200
        self.embed_dim = 16
        self.hidden_dim = 32
        lstm_out_dim = lstm_out_dim

        add_prefix = lambda x, y:list(map(lambda e: y + e if isinstance(e, str) else y + str(e), x))
        self.token_list = ast_gen_helper.token_list + add_prefix(ast_gen_helper.bin_op_list, 'bin_op_') \
                          + add_prefix(ast_gen_helper.cmp_op_list, 'cmp_op_') \
                          + add_prefix(ast_gen_helper.unary_op_list, 'unary_op_') \
                          + add_prefix(ast_gen_helper.builtin_func_list, 'builtin_func_') \
                          + add_prefix(range(ast_gen_helper.var_num_limit), 'name_var') \
                          + add_prefix(range(ast_gen_helper.arg_num_limit), 'name_arg') \
                          + add_prefix(range(ast_gen_helper.func_num_limit), 'name_func') \
                          + add_prefix(ast_gen_helper.const_int_list, 'const_int_') \
                          + add_prefix(ast_gen_helper.const_float_list, 'const_float_') \
                          + add_prefix(range(ast_gen_helper.arg_num_limit), 'arg_num_')

        self.embedding = nn.Embedding(embed_dict_size, self.embed_dim)
        self.leaf_ff = nn.Sequential(
            nn.Linear(self.embed_dim * 2, self.hidden_dim),
            nn.Linear(self.hidden_dim, self.hidden_dim)
        )
        self.node_ff = nn.Sequential(
            nn.Linear(self.embed_dim * 2 + self.hidden_dim, self.hidden_dim),
            nn.ReLU()
        )
        self.ptr_ff = nn.Linear(note_dim + lstm_out_dim, self.hidden_dim)

        self.ff_model = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.Linear(self.hidden_dim, self.hidden_dim)
        )
        self.tail_score = nn.Sequential(
            nn.Linear(self.hidden_dim, 1)
        )

    def child_sum(self, parent_type, node, lstm_out, first_note):
        node_type = node[0]
        node_children = node[2]

        def long_ind(x):
            return torch.LongTensor([self.token_list.index(x)])

        if node_type.item() == self.token_list.index('<var_name>'):
            var_type, var_name = node_children[0], node_children[1]
            var_idx = long_ind('name_var0') \
                      + (var_type - long_ind('<var_type_var>')) \
                      * (long_ind('name_arg0') - long_ind('name_var0')) \
                      + var_name
            v = self.embedding(torch.cat((node_type.view(1), var_idx.view(1)))).view(1, -1)
            return self.leaf_ff(v)
        elif node_type.item() == self.token_list.index('<const>'):
            const_type, const_val = node_children[0], node_children[1]
            if self.token_list[const_type.item()] == '<const_type_int>':
                idx = long_ind('const_int_-1') + const_val
                v = self.embedding(torch.cat((const_type.view(1), idx.view(1)))).view(1, -1)
                return self.leaf_ff(v)
            elif self.token_list[const_type.item()] == '<const_type_float>':
                idx = long_ind('const_float_-1.0') + const_val
                v = self.embedding(torch.cat((const_type.view(1), idx.view(1)))).view(1, -1)
                return self.leaf_ff(v)
            else:
                v = torch.cat((first_note, lstm_out[const_val.item(), 0, :])).view(1, -1)
                return self.ptr_ff(v)
        elif node_type.item() == self.token_list.index('<op_name>'):
            op_name = node_children[0]
            if self.token_list[parent_type.item()] == 'binop':
                idx = long_ind('bin_op_+') + op_name
                v = self.embedding(torch.cat((node_type.view(1), idx.view(1)))).view(1, -1)
                return self.leaf_ff(v)
            elif self.token_list[parent_type.item()] == 'cmpop':
                idx = long_ind('cmp_op_==') + op_name
                v = self.embedding(torch.cat((node_type.view(1), idx.view(1)))).view(1, -1)
                return self.leaf_ff(v)
            else:
                idx = long_ind('unary_op_~') + op_name
                v = self.embedding(torch.cat((node_type.view(1), idx.view(1)))).view(1, -1)
                return self.leaf_ff(v)
        elif node_type.item() == self.token_list.index('<func_name>'):
            func_type, func_name = node_children[0], node_children[1]
            if self.token_list[func_type.item()] == '<func_type_func>':
                idx = long_ind('name_func0') + func_name
                v = self.embedding(torch.cat((func_type.view(1), idx.view(1)))).view(1, -1)
                return self.leaf_ff(v)
            else:
                idx = long_ind('builtin_func_list') + func_name
                v = self.embedding(torch.cat((func_type.view(1), idx.view(1)))).view(1, -1)
                return self.leaf_ff(v)
        elif node_type.item() == self.token_list.index('<args_num>'):
            args_num = node_children[0]
            idx = long_ind('arg_num_0') + args_num
            v = self.embedding(torch.cat((node_type.view(1), idx.view(1)))).view(1, -1)
            return self.leaf_ff(v)
        else:
            if len(node_children) == 0:
                children = torch.zeros(1, self.hidden_dim)
            else:
                children = torch.stack([self.child_sum(node_type, c, lstm_out, first_note).view(-1) for c in node_children])
            parent_embed = self.embedding(parent_type).view(1, -1)
            node_embed = self.embedding(node_type).view(1, -1)
            children_sum = torch.sum(children, dim=0).view(1, -1)
            return self.node_ff(torch.cat((parent_embed, node_embed, children_sum), dim=1))

    def forward(self, lstm_out_list, first_notes, trees, train):
        tree_sum = []
        for i, (note, tree) in enumerate(zip(first_notes, trees)):
            assert self.token_list[tree[0].item()] == '<root>'
            summary = self.child_sum(tree[0], tree, lstm_out_list[i], first_notes[i, :])
            tree_sum.append(summary)

        ff_out = self.ff_model(torch.stack(tree_sum))
        score = self.tail_score(ff_out)
        return score


class Generator(nn.Module):
    def __init__(self, note_dim, lstm_out_dim):
        super(Generator, self).__init__()

        embed_dict_size = 100
        self.embed_dim = 8
        self.note_dim = note_dim
        hidden_dim = self.embed_dim + self.note_dim * 2
        prime_type_num = 15  # update manually
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
        self.sibling_mask = nn.Linear(self.embed_dim + self.note_dim, self.note_dim)
        self.predict_type = nn.Sequential(
            nn.Linear(hidden_dim, prime_type_num),
            nn.Softmax()
        )
        self.predict_note = nn.Sequential(
            nn.Linear(hidden_dim + self.embed_dim, self.note_dim),
            nn.Tanh()
        )
        self.predict_bin_op = nn.Sequential(nn.Linear(self.note_dim, bin_op_num), nn.Softmax())
        self.predict_cmp_op = nn.Sequential(nn.Linear(self.note_dim, cmp_op_num), nn.Softmax())
        self.predict_unary_op = nn.Sequential(nn.Linear(self.note_dim, unary_op_num), nn.Softmax())
        self.predict_func_type = nn.Sequential(nn.Linear(self.note_dim, func_type_num), nn.Softmax())
        self.predict_builtin_func = nn.Sequential(nn.Linear(self.note_dim, builtin_func_num), nn.Softmax())
        self.predict_func_name = nn.Sequential(nn.Linear(self.note_dim, func_num_limit), nn.Softmax())
        self.predict_var_type = nn.Sequential(nn.Linear(self.note_dim, var_type_num), nn.Softmax())
        self.predict_arg_name = nn.Sequential(nn.Linear(self.note_dim, arg_num_limit), nn.Softmax())
        self.predict_var_name = nn.Sequential(nn.Linear(self.note_dim, var_num_limit), nn.Softmax())
        self.predict_const_type = nn.Sequential(nn.Linear(self.note_dim, const_type_num), nn.Softmax())
        self.predict_const_int = nn.Sequential(nn.Linear(self.note_dim, const_int_num), nn.Softmax())
        self.predict_const_float = nn.Sequential(nn.Linear(self.note_dim, const_float_num), nn.Softmax())
        self.predict_args_num = nn.Sequential(nn.Linear(self.note_dim, arg_num_limit), nn.Softmax())
        self.ptr_ff = nn.Linear(lstm_out_dim + self.note_dim, 1)

    def predict_const_copy(self, note, lstm_out):
        l = lstm_out.size(0)
        cc = torch.cat([note.expand(l, note.size(1)), lstm_out.view(l, -1)], dim=1)
        return nn.Softmax(dim=1)(self.ptr_ff(cc).view(1, -1))

    def next_node(self, parent, sibling, note, lstm_out, train):
        assert parent.size()[0] == 1
        mask = self.sibling_mask(torch.cat([parent, note], dim=1))
        sibling_notes = list(map(lambda x: x[1], sibling))
        sibling_sum = self.sibling_sum(mask, sibling_notes)
        hidden = torch.cat([parent, sibling_sum, note], dim=1)
        sort_prob = self.predict_type(hidden)
        sort = self.pick(sort_prob, train).view(1)
        note = self.predict_note(torch.cat([hidden, self.type_embedding(sort)], dim=1))
        if self.token_list[sort.item()] == 'if':
            child = self.gen_child(sort, note, ['<cond>', '<stmts>', '<stmts>'], lstm_out, train)
        elif self.token_list[sort.item()] == 'for':
            child = self.gen_child(sort, note, ['<expr>', '<var_name>', '<stmts>'], lstm_out, train)
        elif self.token_list[sort.item()] == 'while':
            child = self.gen_child(sort, note, ['<cond>', '<stmts>'], lstm_out, train)
        elif self.token_list[sort.item()] == 'return':
            child = self.gen_child(sort, note, ['<expr>'], lstm_out, train)
        elif self.token_list[sort.item()] == 'call':
            child = self.gen_child(sort, note, ['<func_name>', '<exprs>'], lstm_out, train)
        elif self.token_list[sort.item()] == 'var':
            child = self.gen_child(sort, note, ['<var_name>'], lstm_out, train)
        elif self.token_list[sort.item()] == 'val':
            child = self.gen_child(sort, note, ['<const>'], lstm_out, train)
        elif self.token_list[sort.item()] == 'binop':
            child = self.gen_child(sort, note, ['<op_name>', '<expr>', '<expr>'], lstm_out, train)
        elif self.token_list[sort.item()] == 'cmpop':
            child = self.gen_child(sort, note, ['<op_name>', '<expr>', '<expr>'], lstm_out, train)
        elif self.token_list[sort.item()] == 'uop':
            child = self.gen_child(sort, note, ['<op_name>', '<expr>'], lstm_out, train)
        elif self.token_list[sort.item()] == 'assign':
            child = self.gen_child(sort, note, ['<var_name>', '<expr>'], lstm_out, train)
        elif self.token_list[sort.item()] == 'funcdef':
            child = self.gen_child(sort, note, ['<func_name>', '<args_num>', '<stmts>'], lstm_out, train)
        # TODO: []
        else:
            child = []
        return sort, note, child

    def gen_child(self, parent_type, parent_note, child_type_list, lstm_out, train):
        assert len(child_type_list) != 0
        child_list = []
        child_note_list = []
        mask = self.sibling_mask(torch.cat([self.type_embedding(parent_type), parent_note], dim=1))
        for child_type in child_type_list:
            child = torch.LongTensor([self.token_list.index(child_type)]).view(1, -1)
            note = self.predict_note(torch.cat([self.type_embedding(parent_type),
                                                self.sibling_sum(mask, child_note_list),
                                                parent_note,
                                                self.type_embedding(child.view(1))], dim=1))
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
            return torch.zeros(1, self.note_dim)
        masked = mask.expand(sibling_num, self.note_dim) * torch.cat(note_list, dim=0)
        return torch.sum(masked, dim=0).view(1, -1)

    def pick(self, prob, train):
        if train:
            dist = torch.distributions.Categorical(prob)
            result = dist.sample()
        else:
            _, result = torch.max(prob.view(-1), 0)
        return result

    def forward(self, lstm_out_list, first_notes, train=True):
        trees_list = []
        for i in range(first_notes.size(0)):
            first_note = first_notes[i, :].view(1, -1)
            traverse_list = []
            root_type = torch.LongTensor([self.token_list.index('<root>')]).view(1)
            root = [root_type, first_note, []]
            traverse_list.append(root)
            trees_list.append(root)
            j = 0
            while len(traverse_list) != 0:
                j = j + 1
                if j > 100:
                    break
                current = traverse_list[0]
                traverse_list = traverse_list[1:]
                child = current[2]
                for k in range(10):
                    current_embed = self.type_embedding(current[0]).view(1, -1)
                    next, note, next_child_list = self.next_node(current_embed, child, current[1], lstm_out_list[i], train)
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

