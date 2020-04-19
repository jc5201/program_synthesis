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
        self.embed_dim = 32
        self.note_dim = note_dim
        hidden_dim = self.embed_dim + self.note_dim * 2
        prime_type_num = 24  # update manually   # offset = 2
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
        self.code_lstm = nn.LSTM(self.embed_dim, self.note_dim, 2)
        self.next_note = nn.Linear(self.note_dim * 3, self.note_dim)
        self.ff_attention = nn.Sequential(
            nn.Linear(self.embed_dim * 5, self.embed_dim * 3),
            nn.ReLU(),
            nn.Linear(self.embed_dim * 3, self.embed_dim * 1),
            nn.ReLU()
        )
        self.ff_combine = nn.Sequential(
            nn.Linear(self.embed_dim * 2, self.note_dim),
            nn.ReLU()
        )
        self.predict_type = nn.Sequential(
            nn.Linear(self.note_dim, prime_type_num),
            nn.Softmax()
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

    def predict_const_copy(self, notes, lstm_out, train):
        pointers = []
        for i in range(notes.size(0)):
            l = lstm_out.size(1)
            note = notes[i]
            cc = torch.cat([note.expand(l, note.size(1)), lstm_out.view(l, -1)], dim=1)
            pointers.append(self.pick(nn.Softmax(dim=1)(self.ptr_ff(cc).view(1, -1)), 0, train))
        return torch.cat(pointers, dim=-1)

    def pick(self, prob, offset, train):
        if train:
            dist = torch.distributions.Categorical(prob)
            result = dist.sample()
        else:
            _, result = torch.max(prob.view(-1), 0)
        return result + torch.LongTensor([offset]).view(result.size())

    def tree_attention(self, trees, first_notes, train):
        embeded_code = self.type_embedding(trees[:, :, 2])
        # (batch_size, episode_len, note_dim)
        code_lstm_out, _ = self.code_lstm(embeded_code)
        # (batch_size, episode_len, note_dim)
        notes = [first_notes.unsqueeze(1)]
        for i in range(trees.size(1)):
            next_note = self.next_note(torch.cat([first_notes, notes[0].squeeze(1), code_lstm_out[:, i, :]])).unsqueeze(1)
            notes.append(next_note)

        final_note = notes[trees.size(1)]
        stacked_notes = torch.stack(notes[:-1], dim=1)
        # (batch_size, episode_len, note_dim)
        score = torch.bmm(stacked_notes.reshape(trees.size(0) * trees.size(1), 1, self.note_dim),
                          final_note.reshape(trees.size(0) * trees.size(1), self.note_dim, 1))\
                    .reshape(trees.size(0), trees.size(1))

        return torch.Softmax(1)(score)

    def forward(self, lstm_out_list, first_notes, trees, train=True):
        # idx, parent_idx, type, value1, ...
        idx = trees[:, -1, 0] + torch.LongTensor([1]).expand(trees.size(0), 1)
        parent_idx_list = []
        for i in range(trees.size(0)):
            if trees[i, -1, 2] == self.token_list.index('<End>'):
                prev_parent_idx = trees[i, -1, 1]
                parent_idx_list.append(trees[i, (idx[i, 0] - prev_parent_idx - 1), 1])
        parent_idx = torch.LongTensor(parent_idx_list)

        parent_type_list = [trees[i, parent_idx[i], 2] for i in range(trees.size(0))]
        parent_type = torch.LongTensor(parent_type_list)
        parent_type_embed = self.type_embedding(parent_type)

        attentions = self.tree_attention(trees, first_notes, train)
        sorted_attention_value, sorted_attention_idx = attentions.sort(1, True)

        l2 = []
        for i in range(trees.size(0)):
            l1 = []
            for j in range(5):
                l1.append(parent_type_embed[i, sorted_attention_idx[i, j]])
            torch.stack(l1, dim=-1)
        reordered_trees = torch.stack(l2, dim=-1)
        # (batch_size, 5, embed_dim)
        temp_output = self.ff_attention(reordered_trees.view(trees.size(0), -1))
        temp_output = self.ff_combine(torch.cat([parent_type_embed, temp_output], dim=1))

        type = self.pick(self.predict_type(temp_output), 2, train)
        value1 = self.pick(self.predict_bin_op(temp_output), 0, train)
        value2 = self.pick(self.predict_cmp_op(temp_output), 0, train)
        value3 = self.pick(self.predict_unary_op(temp_output), 0, train)
        value4 = self.pick(self.predict_func_type(temp_output), 31, train)
        value5 = self.pick(self.predict_builtin_func(temp_output), 0, train)
        value6 = self.pick(self.predict_func_name(temp_output), 0, train)
        value7 = self.pick(self.predict_var_type(temp_output), 26, train)
        value8 = self.pick(self.predict_arg_name(temp_output), 0, train)
        value9 = self.pick(self.predict_var_name(temp_output), 0, train)
        value10 = self.pick(self.predict_const_type(temp_output), 28, train)
        value11 = self.pick(self.predict_const_int(temp_output), 0, train)
        value12 = self.pick(self.predict_const_float(temp_output), 0, train)
        value13 = self.pick(self.predict_args_num(temp_output), 0, train)
        value14 = self.predict_const_copy(temp_output, lstm_out_list)

        return torch.cat([idx, parent_idx, type,
                          value1, value2, value3, value4, value5,
                          value6, value7, value8, value9, value10,
                          value11, value12, value13, value14], dim=1)


