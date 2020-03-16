import torch

import logging
import argparse
import os
import shutil
import pickle
import time
import random
import itertools
import json
from torch.utils.tensorboard import SummaryWriter

import loader
import model

log_dir = './log'
model_save_dir = './log'
model_filename = './model.py'
trainA_len = 16410

parser = argparse.ArgumentParser()
parser.add_argument('--mode', type=str, default='train',        # TODO
                    help='what to do (train)')
parser.add_argument('--model', type=str, default='text',        # TODO
                    help='type of train target (text, generator)')
parser.add_argument('--epoch', type=int, default='20',
                    help='num of epoch to train')
parser.add_argument('--batch_size', type=int, default='10',
                    help='batch size')
parser.add_argument('--learning_rate', type=float, default='1e-3',
                    help='learning rate')
parser.add_argument('--cuda', action='store_true',        # TODO
                    help='use cuda')
parser.add_argument('--model-name', type=str, default='',
                    help='name of train model')
parser.add_argument('--disable-terminal-log', action='store_true',
                    help='do no print log to terminal')
parser.add_argument('--log-level', type=str, default='info',        # TODO
                    help='log-level (debug, info, warning, error)')


args = parser.parse_args()

logging.getLogger().setLevel(logging.DEBUG) # TODO: apply args
formatter = logging.Formatter('[%(asctime)s][%(levelname)s|%(filename)s:%(lineno)s] >> %(message)s')
fileHandler = logging.FileHandler('{}/{}_{}.log'.format(log_dir, args.model_name,
                                                   time.strftime('%y%m%d-%H%M%S', time.localtime(time.time()))),
                                  mode='w')
fileHandler.setFormatter(formatter)
logging.getLogger().addHandler(fileHandler)
if not args.disable_terminal_log:
    streamHandler = logging.StreamHandler()
    streamHandler.setFormatter(formatter)
    logging.getLogger().addHandler(streamHandler)

if args.cuda:
    if torch.cuda.is_available():
        device = torch.device('cuda')
        logging.info('use cuda')
    else:
        logging.warn('cuda is not available')


def main():
    model_name = args.model_name
    if not model_name:
        logging.error('model name is empty')
    if model_exists(model_name):
        synth_model = load_model(model_name)
    else:
        synth_model = create_model(model_name)
    optimizer = torch.optim.Adam(synth_model.parameters(), lr=args.learning_rate)
    validation_set = range(trainA_len // 100)
    training_set = list(set(range(trainA_len)) - set(validation_set))
    target_epoch = args.epoch
    batch_size = args.batch_size
    train_target = ['text']
    text_dictionary = loader.load_word_dictionary()
    code_dictionary = loader.load_code_dictionary()
    dataset = loader.load_from_jsonl_file(loader.naps_train_A, [])

    tensorboard_writer = SummaryWriter('runs/' + model_name)

    logging.info('start training. step: {}, batch size:{}, learning rate:{}'.format(
        synth_model.get_steps(), batch_size, args.learning_rate
    ))

    for epoch in range(target_epoch):
        logging.info('start epoch #{}'.format(epoch))
        batches = make_batch(training_set, batch_size)
        loss = []
        for i, batch in enumerate(batches):
            logging.debug('start batch #{}: {}'.format(i, batch))
            optimizer.zero_grad()
            raw_data = load_batch_from(dataset, batch)
            text_data, code_data = ready_data(raw_data, text_dictionary, code_dictionary)
            step_loss = train_vector_representation(synth_model, text_data, code_data, optimizer)

            loss.append(step_loss.detach())
            synth_model.increase_step(batch_size)
            tensorboard_writer.add_scalar('Loss/train', step_loss.item(), synth_model.get_steps())
            logging.debug('finished batch: train_loss: {}'.format(step_loss))

        raw_data = load_batch_from(dataset, validation_set)
        text_data, code_data = ready_data(raw_data, text_dictionary, code_dictionary)
        eval_loss = evaluate_vector_representation(synth_model, text_data, code_data)

        average_loss = sum(loss) / len(loss)
        logging.info('epoch #{}: training_loss: {}, validation_loss: {}'.format(epoch, average_loss, eval_loss))
    loader.save_code_dictionary(code_dictionary)
    save_model(model_name, synth_model)
    tensorboard_writer.close()


def make_batch(training_set, batch_size):
    random.shuffle(training_set)
    return [list(itertools.islice(training_set, batch_size * i, batch_size * (i + 1)))
            for i in range((len(training_set) - 1) // batch_size + 1)]


def load_batch_from(dataset, batch):
    return [dataset[i] for i in batch]


def ready_data(data, text_dictionary, code_dictionary):
    text_tensors = []
    code_trees = []
    for raw_data in data:
        parsed_data = json.loads(raw_data)
        if 'text' in parsed_data.keys():
            text_data = ready_text_for_training(parsed_data['text'], text_dictionary)
        elif 'texts' in parsed_data.keys():
            text_data = ready_text_for_training(parsed_data['texts'][0], text_dictionary)
        else:
            logging.error('unexpected data format')
            raise AssertionError
        code_data = ready_code_for_training(parsed_data['code_tree'], code_dictionary)
        text_tensors.append(text_data)
        code_trees.append(code_data)
    return text_tensors, code_trees


def ready_text_for_training(text_data, dictionary):
    int_rep = list(map(lambda x: dictionary.get(x, 0), text_data))
    return torch.LongTensor(int_rep)


def ready_code_for_training(code_data, dictionary):
    assert isinstance(code_data, dict)
    return uast_to_node(code_data, dictionary)


def uast_to_node(uast, dictionary):
    if isinstance(uast, dict):
        sort = get_after_put_if_not_exist(dictionary, '<program>')
        body = [uast_to_node(node, dictionary) for node in uast['funcs'] + uast['types']]
        return uast_node(sort=sort, body=body)
    elif uast[0] == 'record' or uast[0] == 'struct':
        sort = get_after_put_if_not_exist(dictionary, '<struct>')
        name = get_after_put_if_not_exist(dictionary, uast[1])
        body = [uast_to_node(var, dictionary) for var in uast[2].values()]
        return uast_node(sort=sort, name=name, body=body)
    elif uast[0] == 'func' or uast[0] == 'ctor':
        sort = get_after_put_if_not_exist(dictionary, '<func>')
        type = get_after_put_if_not_exist(dictionary, uast[1])
        name = get_after_put_if_not_exist(dictionary, uast[2])
        args = [uast_to_node(arg, dictionary) for arg in uast[3]]
        body = [uast_to_node(exp, dictionary) for exp in uast[5]]
        return uast_node(sort=sort, type=type, name=name, args=args, body=body)
    elif uast[0] == 'assign':
        sort = get_after_put_if_not_exist(dictionary, '<assign>')
        type = get_after_put_if_not_exist(dictionary, uast[1])
        target = [uast_to_node(uast[2], dictionary)]
        body = [uast_to_node(uast[3], dictionary)]
        return uast_node(sort=sort, type=type, target=target, body=body)
    elif uast[0] == 'var':
        sort = get_after_put_if_not_exist(dictionary, '<var>')
        type = get_after_put_if_not_exist(dictionary, uast[1])
        name = get_after_put_if_not_exist(dictionary, uast[2])
        return uast_node(sort=sort, type=type, name=name)
    elif uast[0] == 'if':
        sort = get_after_put_if_not_exist(dictionary, '<if>')
        cond = [uast_to_node(uast[2], dictionary)]
        body = [uast_to_node(node, dictionary) for node in uast[3]]
        body_else = [uast_to_node(node, dictionary) for node in uast[4]]
        return uast_node(sort=sort, cond=cond, body=body, body_else=body_else)
    elif uast[0] == 'foreach':
        sort = get_after_put_if_not_exist(dictionary, '<foreach>')
        target = [uast_to_node(uast[2], dictionary)]
        iter_foreach = [uast_to_node(uast[3], dictionary)]
        body = [uast_to_node(node, dictionary) for node in uast[4]]
        return uast_node(sort=sort, target=target, iter_foreach=iter_foreach, body=body)
    elif uast[0] == 'while':
        sort = get_after_put_if_not_exist(dictionary, '<while>')
        cond = [uast_to_node(uast[2], dictionary)]
        body = [uast_to_node(node, dictionary) for node in uast[3]]
        body_iter = [uast_to_node(node, dictionary) for node in uast[4]]
        return uast_node(sort=sort, cond=cond, body=body, body_iter=body_iter)
    elif uast[0] == 'break':
        sort = get_after_put_if_not_exist(dictionary, '<break>')
        return uast_node(sort=sort)
    elif uast[0] == 'continue':
        sort = get_after_put_if_not_exist(dictionary, '<continue>')
        return uast_node(sort=sort)
    elif uast[0] == 'return':
        sort = get_after_put_if_not_exist(dictionary, '<return>')
        return uast_node(sort=sort)
    elif uast[0] == 'noop':
        sort = get_after_put_if_not_exist(dictionary, '<noop>')
        return uast_node(sort=sort)
    elif uast[0] == 'field':
        sort = get_after_put_if_not_exist(dictionary, '<field>')
        type = get_after_put_if_not_exist(dictionary, uast[1])
        name = get_after_put_if_not_exist(dictionary, uast[3])
        body = [uast_to_node(uast[2], dictionary)]
        return uast_node(sort=sort, type=type, name=name, body=body)
    elif uast[0] == 'val':
        sort = get_after_put_if_not_exist(dictionary, '<val>')
        type = get_after_put_if_not_exist(dictionary, uast[1])
        val_str = uast[2] if isinstance(uast[2], str) else str(uast[2])
        val_sort = get_after_put_if_not_exist(dictionary, '<val_in>')
        if len(val_str) == 0:
            empty_str = get_after_put_if_not_exist(dictionary, '<empty_str>')
            body = [uast_node(sort=val_sort, body=empty_str)]
        else:
            body = [uast_node(sort=val_sort, body=get_after_put_if_not_exist(dictionary, char)) for char in val_str]
        return uast_node(sort=sort, type=type, body=body)
    elif uast[0] == 'invoke':
        sort = get_after_put_if_not_exist(dictionary, '<invoke>')
        type = get_after_put_if_not_exist(dictionary, uast[1])
        name = get_after_put_if_not_exist(dictionary, uast[2])
        args = [uast_to_node(arg, dictionary) for arg in uast[3]]
        return uast_node(sort=sort, type=type, name=name, args=args)
    elif uast[0] == '?:':
        sort = get_after_put_if_not_exist(dictionary, '<ternary>')
        type = get_after_put_if_not_exist(dictionary, uast[1])
        cond = [uast_to_node(uast[2], dictionary)]
        body = [uast_to_node(uast[3], dictionary)]
        body_else = [uast_to_node(uast[4], dictionary)]
        return uast_node(sort=sort, type=type, cond=cond, body=body, body_else=body_else)
    elif uast[0] == 'cast':
        sort = get_after_put_if_not_exist(dictionary, '<cast>')
        type = get_after_put_if_not_exist(dictionary, uast[1])
        body = [uast_to_node(uast[2], dictionary)]
        return uast_node(sort=sort, type=type, body=body)
    else:
        logging.error("unexpected uast")
        raise AssertionError


def get_after_put_if_not_exist(dictionary, word):
    if word in dictionary.keys():
        return dictionary[word]
    else:
        dictionary[word] = len(dictionary)
        logging.debug('added {} to dictionary'.format(word))
        return dictionary[word]


def uast_node(sort, type=0, name=0, args=0, cond=0, body=0,
              body_else=0, body_iter=0, iter_foreach=0, target=0):
    return [sort, type, name, args, cond, body, body_else, body_iter, iter_foreach, target]


def train_vector_representation(synth_model, input_texts, input_codes, optimizer):
    vec_from_text = synth_model.forward_text_encoder(input_texts, train=False).detach()
    _, vec_from_disc = synth_model.forward_discriminator(input_codes, train=True)
    loss = calculate_vector_rep_loss(vec_from_text, vec_from_disc)
    distorted_vec = torch.cat([vec_from_text[1:, :], vec_from_text[:1, :]])
    loss_wrong_match = calculate_vector_rep_loss(distorted_vec, vec_from_text)
    (loss - torch.clamp(loss_wrong_match, 0, 10)).backward(retain_graph=False)
    optimizer.step()
    optimizer.zero_grad()
    logging.debug('train part1: loss {}, loss_wrong_match {}'.format(loss, loss_wrong_match))

    vec_from_text = synth_model.forward_text_encoder(input_texts, train=True)
    _, vec_from_disc = synth_model.forward_discriminator(input_codes, train=False)
    loss = calculate_vector_rep_loss(vec_from_text, vec_from_disc.detach())
    distorted_vec = torch.cat([vec_from_text[1:, :], vec_from_text[:1, :]])
    loss_wrong_match = calculate_vector_rep_loss(distorted_vec, vec_from_text)
    (loss - torch.clamp(loss_wrong_match, 0, 10)).backward(retain_graph=False)
    optimizer.step()
    optimizer.zero_grad()
    logging.debug('train part2: loss {}, loss_wrong_match {}'.format(loss, loss_wrong_match))

    vec_from_text = synth_model.forward_text_encoder(input_texts, train=False).detach()
    _, vec_from_disc = synth_model.forward_discriminator(input_codes, train=False)
    loss = calculate_vector_rep_loss(vec_from_text, vec_from_disc.detach())
    distorted_vec = torch.cat([vec_from_text[1:, :], vec_from_text[:1, :]])
    loss_wrong_match = calculate_vector_rep_loss(distorted_vec, vec_from_text)
    logging.debug('train part3: loss {}, loss_wrong_match {}'.format(loss, loss_wrong_match))

    return loss


def evaluate_vector_representation(synth_model, input_texts, input_codes):
    vec_from_text = synth_model.forward_text_encoder(input_texts, train=False)
    _, vec_from_disc = synth_model.forward_discriminator(input_codes, train=False)
    return calculate_vector_rep_loss(vec_from_text, vec_from_disc)


def calculate_vector_rep_loss(output, target):
    loss_f = torch.nn.MSELoss()
    return loss_f(output, target)


def model_exists(model_name):
    return os.path.exists(model_save_dir + '/' + model_name + '/' + 'model.pkl')


def create_model(model_name):
    synth_model = model.CodeSynthesisModel()

    model_dir = model_save_dir + '/' + model_name
    if os.path.exists(model_dir):
        if os.path.isdir(model_dir):
            shutil.rmtree(model_dir, ignore_errors=True)
        else:
            os.remove(model_dir)
    os.mkdir(model_dir)
    shutil.copy(model_filename, model_dir)

    return synth_model


def save_model(model_name, synth_model, step=None):
    if step:
        model_path = model_save_dir + '/' + model_name + '/' + 'model_{}.pkl'.format(step)
    else:
        model_path = model_save_dir + '/' + model_name + '/' + 'model.pkl'
    if os.path.exists(model_path):
        os.remove(model_path)
        logging.info('deleted previous model in {}'.format(model_path))
    with open(model_path, 'wb') as f:
        pickle.dump(synth_model, f)
    logging.info('saved model {} into {}'.format(model_name, model_path))


def load_model(model_name, step=None):
    if step:
        model_path = model_save_dir + '/' + model_name + '/' + 'model_{}.pkl'.format(step)
    else:
        model_path = model_save_dir + '/' + model_name + '/' + 'model.pkl'
    with open(model_path, 'rb') as f:
        synth_model = pickle.load(f)
    logging.info('loaded model {} from {}'.format(model_name, model_path))
    return synth_model


if __name__ == '__main__':
    main()

