import torch
import torch.nn as nn
import math
import ast_gen_helper


class CodeSynthesisModel:
    def __init__(self, note_dim=16):
        lstm_out_dim = 32
        self.text_encoder = TextEncoder(note_dim, lstm_out_dim).cuda()
        self.generator = Generator(note_dim, lstm_out_dim).cuda()
        self.discriminator = Discriminator(note_dim, lstm_out_dim).cuda()

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

        self.vector_token = torch.LongTensor([1999]).cuda()
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
        self.pos_enc_dim = 8
        lstm_out_dim = lstm_out_dim
        self.node_dim = 2 * self.pos_enc_dim + 14 * self.embed_dim + lstm_out_dim
        self.hidden_dim = 32

        self.token_list = ast_gen_helper.all_token_list

        self.embedding = nn.Embedding(embed_dict_size, self.embed_dim)
        self.positional_encoding = PositionalEncoding(self.pos_enc_dim, max_len=200)
        self.leaf_end_ff = nn.Linear(self.embed_dim, self.hidden_dim)
        self.leaf_child_ff = nn.Sequential(
            nn.Linear(self.embed_dim * 14 + lstm_out_dim, self.hidden_dim * 4),
            nn.ReLU(),
            nn.Linear(self.hidden_dim * 4, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim)
        )
        self.child_sum_ff = nn.Sequential(
            nn.Linear(self.hidden_dim * 2, self.hidden_dim * 4),
            nn.ReLU(),
            nn.Linear(self.hidden_dim * 4, self.hidden_dim),
            nn.ReLU()
        )

        self.node_ff = nn.Sequential(
            nn.Linear(self.pos_enc_dim * 2 + self.embed_dim * 14 + lstm_out_dim, self.hidden_dim * 4),
            nn.ReLU(),
            nn.Linear(self.hidden_dim * 4, self.hidden_dim),
            nn.ReLU()
        )
        self.combine_ff = nn.Sequential(
            nn.Linear(self.hidden_dim * 2, self.hidden_dim * 4),
            nn.ReLU(),
            nn.Linear(self.hidden_dim * 4, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim)
        )

        self.tree_attention = nn.Sequential(
            nn.Linear(self.node_dim * 2, self.node_dim),
            nn.Linear(self.node_dim, 1)
        )
        self.tree_att_ff = nn.Sequential(
            nn.Linear(self.hidden_dim * 6, self.hidden_dim * 4),
            nn.ReLU(),
            nn.Linear(self.hidden_dim * 4, self.hidden_dim),
            nn.ReLU())

        self.ff_model = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.ReLU()
        )
        self.tail_score = nn.Sequential(
            nn.Linear(self.hidden_dim, 1)
        )

    def child_sum(self, tree, lstm_out, first_note):
        node_type = tree[0, 2]
        node_children_idx = list(filter(lambda x: tree[x, 1] == tree[0, 0], range(tree.size(0))))
        children_summary = []
        for start_idx in node_children_idx:
            end_idx = -1
            if tree[start_idx, 2].item() == self.token_list.index('<End>'):
                end_idx = start_idx + 1
            else:
                for j in range(start_idx + 1, tree.size(0)):
                    if tree[j, 1] == tree[0, 0]:
                        end_idx = j
                        break
                if end_idx == -1:
                    end_idx = tree.size(1)
            child_tree = tree[start_idx:end_idx, :]
            summary = self.child_sum(child_tree, lstm_out, first_note)
            children_summary.append(summary)

        if len(children_summary) == 0:
            if node_type.item() == self.token_list.index('<End>'):
                return self.leaf_end_ff(self.embedding(node_type)).view(1, -1)
            else:
                embedded = self.embedding(tree[0, 2:-1])
                ptr_embedded = lstm_out[tree[0, -1], 0, :]
                # (15, embed_dim)
                return self.leaf_child_ff(torch.cat([embedded.view(-1), ptr_embedded], dim=0).view(1, -1))

        child_summary = children_summary[0]
        for i in range(1, len(children_summary)):
            child_summary = self.child_sum_ff(torch.cat([child_summary, children_summary[i]], dim=1))

        node_embedded = self.embedding(tree[0, 2:-1])
        node_ptr_embedded = lstm_out[tree[0, -1], 0, :]
        node_tmp = self.node_ff(torch.cat([node_embedded.view(-1), node_ptr_embedded], dim=0).view(1, -1))

        return self.combine_ff(torch.cat([node_tmp, child_summary], dim=1))

    def attentional_recent_discriminator(self, tree, lstm_out, first_note):
        # (episode_len, 17)
        node_embedded = self.embedding(tree[:, 2:-1]).view(tree.size(0), -1)
        # (episode_len, 14 * embed_dim)
        pos_encoding = torch.cat([self.positional_encoding(tree[:, 0]), self.positional_encoding(tree[:, 1])], dim=2).view(tree.size(0), -1)
        # (episode_len, 2 *pe_dim)
        node_vec = torch.cat([pos_encoding, node_embedded, lstm_out[tree[:, -1], 0, :]], dim=1)
        # (episode_len, 2 *pe_dim + 14 * embed_dim + lstm_out_dim)
        attentions = self.tree_attention(torch.cat([node_vec[-1].expand(tree.size(0), node_vec.size(1)), node_vec], dim=1))
        # (episode_len, 1)

        sorted_attention_value, sorted_attention_idx = attentions.sort(1, True)

        reordered_tree = node_vec[sorted_attention_idx[:5], :].squeeze(1)
        if reordered_tree.size(0) < 5:
            pad = torch.zeros(5 - reordered_tree.size(0), reordered_tree.size(1)).cuda()
            reordered_tree = torch.cat([reordered_tree, pad], dim=0)
        # (5, 2 *pe_dim + 14 * embed_dim + lstm_out_dim)

        hidden = self.node_ff(torch.cat([node_vec[-1].unsqueeze(0), reordered_tree], dim=0))
        return self.tree_att_ff(hidden.view(1, -1))

    def forward(self, lstm_out_list, first_notes, trees, train):
        # (batch_size, episode_len, 17)
        tree_sum = []
        for i, (note, tree) in enumerate(zip(first_notes, trees)):
            start_idx = -1
            for j in range(trees.size(1)):
                if tree[j, 2] == self.token_list.index('<root>'):
                    start_idx = j
                    break
            assert start_idx != -1
            summary = self.attentional_recent_discriminator(tree[start_idx:, :], lstm_out_list[i], note)
            tree_sum.append(summary)

        ff_out = self.ff_model(torch.cat(tree_sum, dim=0))
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
            l = lstm_out[i].size(0)
            note = notes[i]
            cc = torch.cat([note.expand(l, notes.size(1)), lstm_out[i].view(l, -1)], dim=1)
            pointers.append(self.pick(nn.Softmax(dim=1)(self.ptr_ff(cc).view(1, -1)), 0, train))
        return torch.cat(pointers, dim=0)

    def pick(self, prob, offset, train):
        if train:
            dist = torch.distributions.Categorical(prob)
            result = dist.sample()
        else:
            _, result = torch.max(prob.view(-1), 0)
        return result.reshape(prob.size(0), 1) + torch.LongTensor([offset]).expand(prob.size(0), 1).cuda()

    def tree_attention(self, trees, first_notes, train):
        embeded_code = self.type_embedding(trees[:, :, 2])
        # (batch_size, episode_len, embed_dim)
        code_lstm_out, _ = self.code_lstm(embeded_code)
        # (batch_size, episode_len, note_dim)
        notes = [first_notes.unsqueeze(1)]
        for i in range(trees.size(1)):
            next_note = self.next_note(torch.cat([first_notes, notes[0].squeeze(1), code_lstm_out[:, i, :]], dim=1)).unsqueeze(1)
            notes.append(next_note)

        final_note = notes[trees.size(1)]
        stacked_notes = torch.cat(notes[:-1], dim=1)
        # (batch_size, episode_len, note_dim)
        score = torch.bmm(stacked_notes.reshape(trees.size(0) * trees.size(1), 1, self.note_dim),
                          final_note.reshape(trees.size(0), 1, self.note_dim).repeat(1, trees.size(1), 1)
                          .reshape(trees.size(0) * trees.size(1), self.note_dim, 1))\
            .reshape(trees.size(0), trees.size(1))

        return nn.Softmax(1)(score)

    def forward(self, lstm_out_list, first_notes, trees, train=True):
        # idx, parent_idx, type, value1, ...
        idx = trees[:, -1, 0].view(trees.size(0), 1) + torch.LongTensor([1]).expand(trees.size(0), 1).cuda()
        parent_idx_list = []
        for i in range(trees.size(0)):
            if trees[i, -1, 2] == self.token_list.index('<End>'):
                prev_parent_idx = trees[i, -1, 1]
                parent_idx_list.append(trees[i, (idx[i, 0] - prev_parent_idx - 1), 1])
            else:
                parent_idx_list.append(trees[i, -1, 0])
        parent_idx = torch.LongTensor(parent_idx_list).view(-1, 1).cuda()

        parent_type_list = [trees[i, parent_idx[i], 2] for i in range(trees.size(0))]
        parent_type = torch.LongTensor(parent_type_list).cuda()
        parent_type_embed = self.type_embedding(parent_type)

        attentions = self.tree_attention(trees, first_notes, train)
        sorted_attention_value, sorted_attention_idx = attentions.sort(1, True)

        l2 = []
        for i in range(trees.size(0)):
            l1 = []
            for j in range(5):
                l1.append(trees[i, sorted_attention_idx[i, j], 2])
            l2.append(self.type_embedding(torch.LongTensor(l1).cuda()).view(5, self.embed_dim))
        reordered_trees = torch.stack(l2)
        # (batch_size, 5, embed_dim)
        temp_output = self.ff_attention(reordered_trees.view(trees.size(0), -1))
        temp_output = self.ff_combine(torch.cat([parent_type_embed, temp_output], dim=1))

        type = self.pick(self.predict_type(temp_output), 2, train)
        value1 = self.pick(self.predict_bin_op(temp_output), 0, train)
        value2 = self.pick(self.predict_cmp_op(temp_output), 0, train)
        value3 = self.pick(self.predict_unary_op(temp_output), 0, train)
        value4 = self.pick(self.predict_func_type(temp_output), 26, train)
        value5 = self.pick(self.predict_builtin_func(temp_output), 0, train)
        value6 = self.pick(self.predict_func_name(temp_output), 0, train)
        value7 = self.pick(self.predict_var_type(temp_output), 21, train)
        value8 = self.pick(self.predict_arg_name(temp_output), 0, train)
        value9 = self.pick(self.predict_var_name(temp_output), 0, train)
        value10 = self.pick(self.predict_const_type(temp_output), 23, train)
        value11 = self.pick(self.predict_const_int(temp_output), 0, train)
        value12 = self.pick(self.predict_const_float(temp_output), 0, train)
        value13 = self.pick(self.predict_args_num(temp_output), 0, train)
        value14 = self.predict_const_copy(temp_output, lstm_out_list, train)

        return torch.cat([idx, parent_idx, type,
                          value1, value2, value3, value4, value5,
                          value6, value7, value8, value9, value10,
                          value11, value12, value13, value14], dim=1)


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model).cuda()
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1).cuda()
        div_term = torch.exp(torch.arange(0, d_model, 2).float().cuda() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[x, :]
