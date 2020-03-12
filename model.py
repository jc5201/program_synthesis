import torch
import torch.nn as nn


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


