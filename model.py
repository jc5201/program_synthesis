import torch
import torch.nn as nn
import ast


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

        embed_dict_size = 2000
        self.embed_dim = 8
        lstm_hidden_dim = latent_vector_dim
        lstm_layer_num = 2
        self.embedding = nn.Embedding(embed_dict_size, self.embed_dim)
        self.lstm = nn.LSTM(self.embed_dim, lstm_hidden_dim, lstm_layer_num)
        self.ff = nn.Sequential(
            nn.ReLU(),
            nn.Linear(lstm_hidden_dim, latent_vector_dim),
            nn.BatchNorm1d(latent_vector_dim)
        )

    def forward(self, xs):
        l = []
        for x in xs:
            embedded = self.embedding(x).view(-1, 1, self.embed_dim)
            lstm_out, hid = self.lstm(embedded)
            l.append(lstm_out[-1, :, :].view(-1))
        return self.ff(torch.stack(l))


class Discriminator(nn.Module):
    def __init__(self, latent_vector_dim):
        super(Discriminator, self).__init__()

        embed_dict_size = 2000
        self.embed_dim = 16
        hidden_dim = 8

        self.embedding = nn.Embedding(embed_dict_size, self.embed_dim)
        self.child_sum_lstm = nn.LSTM(self.embed_dim, self.embed_dim, 1)
        self.child_sum_ff = nn.Sequential(
            nn.Linear(self.embed_dim * 10, self.embed_dim * 4),
            nn.ReLU(),
            nn.Linear(self.embed_dim * 4, self.embed_dim)
        )
        self.ff_model = nn.Sequential(
            nn.Linear(self.embed_dim, hidden_dim)
        )
        self.tail_score = nn.Sequential(
            nn.Linear(hidden_dim, 1),
            nn.Tanh()
        )
        self.tail_summary = nn.Sequential(
            nn.Linear(hidden_dim, latent_vector_dim),
            nn.BatchNorm1d(latent_vector_dim)
        )

    def child_sum(self, tree):
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

    def forward(self, trees):
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
    def __init__(self, latent_vector_dim, dictionary):
        super(Generator, self).__init__()
        embed_dict_size = 100
        self.embed_dim = 8
        hidden_dim = self.embed_dim * 2 + latent_vector_dim
        sort_num = 15   # <End>, functiondef, assign, return, for, while, if,
                        # break, continue, boolOp, BinOp, UnaryOp, Call,
                        # var, val, args
        self.dictionary = dictionary
        self.token_list = ['<End>', '<root>']
        self.leaf_list = ['break', 'continue']

        self.embedding = nn.Embedding(embed_dict_size, self.embed_dim)
        self.sibling_mask = nn.Sequential(self.embed_dim + latent_vector_dim, self.embed_dim)
        self.predict_sort = nn.Sequential(
            nn.Linear(hidden_dim, sort_num),
            nn.Softmax()
        )
        self.predict_note = nn.Linear(hidden_dim + self.embed_dim, latent_vector_dim)

    def next_node(self, parent, sibling, note, train=True):
        assert parent.size()[0] == 1
        mask = self.sibling_mask(torch.cat([parent, note]))
        sibling_notes = list(map(lambda x: x[1], sibling))
        sibling_sum = self.sibling_sum(mask, sibling_notes)
        hidden = torch.cat([parent, sibling_sum, note], dim=1)
        sort_prob = self.predict_sort(hidden)
        if train:
            dist = torch.distributions.Categorical(sort_prob)
            sort = dist.sample()
        else:
            _, sort = torch.max(sort_prob, 0)
        note = self.predict_note(torch.cat([hidden, self.embedding(sort)]))
        if self.token_list[sort.item()] == 'if':
            child = self.gen_child(sort, note, ['<cond>', '<stmts>', '<stmts>'])
        elif self.token_list[sort.item()] == 'for':
            child = self.gen_child(sort, note, ['<iterable>', '<stmts>'])
        elif self.token_list[sort.item()] == 'while':
            child = self.gen_child(sort, note, ['<cond>', '<stmts>'])
        elif self.token_list[sort.item()] == 'return':
            child = self.gen_child(sort, note, ['<expr>'])
        elif self.token_list[sort.item()] == 'call':
            child = self.gen_child(sort, note, ['<name>', '<exprs>'])
        # TODO: var, val, op
        else:
            child = []
        return sort, note, child

    def gen_child(self, parent_type, parent_note, child_type_list):
        assert len(child_type_list) != 0
        child_list = []
        child_note_list = []
        mask = self.sibling_mask(torch.cat([self.embedding(parent_type), parent_note]))
        for child_type in child_type_list:
            child = torch.LongTensor([self.token_list.index(child_type)]).view(1, -1)
            note = self.predict_note(torch.cat([self.embedding(parent_type),
                                                self.sibling_sum(mask, child_note_list),
                                                parent_note,
                                                self.embedding(child)], dim=1))
            child_list.append((child, note))
            child_note_list.append(note)
        return child_list

    def sibling_sum(self, mask, note_list):
        sibling_num = len(note_list)
        if sibling_num == 0:
            return torch.zeros(1, self.embed_dim)
        masked = mask.expand(sibling_num, self.embed_dim) * torch.stack(note_list)
        return torch.sum(masked, dim=0).view(1, -1)

    def forward(self, xs, train=True):
        trees = []
        for x in xs:
            traverse_list = []
            root_sort = torch.LongTensor([self.token_list.index('<root>')]).view(1, 1)
            root = [self.embedding(root_sort), x, []]
            traverse_list.append(root)
            trees.append(root)
            while len(traverse_list) != 0:
                current = traverse_list[0]
                child = current[2]
                while True:
                    next, note, child_of_next = self.next_node(current[0], child, current[1], train)
                    if next.item() == self.token_list.index('<End>'):
                        break
                    elif len(child_of_next) == 0:
                        child.append([next, note, []])
                    else:
                        n = [next, note, []]
                        for c in child_of_next:
                            cn = [c[0], c[1], []]
                            n[2].append(cn)
                            traverse_list.append(cn)
                        child.append(n)
                traverse_list = traverse_list[1:]

        return trees

