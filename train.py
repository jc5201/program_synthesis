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
        synth_model, optimizer = load_model(model_name)
    else:
        synth_model, optimizer = create_model(model_name)
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
            text_data = ready_data(raw_data, text_dictionary)
            step_loss = train_vector_representation(synth_model, text_data, optimizer)

            loss.append(step_loss.detach())
            synth_model.increase_step(batch_size)
            tensorboard_writer.add_scalar('Loss/train', step_loss.item(), synth_model.get_steps())
            logging.debug('finished batch: train_loss: {}'.format(step_loss))

        raw_data = load_batch_from(dataset, validation_set)
        text_data = ready_data(raw_data, text_dictionary)
        eval_loss = evaluate_vector_representation(synth_model, text_data)

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


def ready_data(data, text_dictionary):
    text_tensors = []
    for raw_data in data:
        parsed_data = json.loads(raw_data)
        if 'text' in parsed_data.keys():
            text_data = ready_text_for_training(parsed_data['text'], text_dictionary)
        elif 'texts' in parsed_data.keys():
            text_data = ready_text_for_training(parsed_data['texts'][0], text_dictionary)
        else:
            logging.error('unexpected data format')
            raise AssertionError
        text_tensors.append(text_data)
    return text_tensors


def ready_text_for_training(text_data, dictionary):
    int_rep = list(map(lambda x: dictionary.get(x, 0), text_data))
    return torch.LongTensor(int_rep)


def get_after_put_if_not_exist(dictionary, word):
    if word in dictionary.keys():
        return dictionary[word]
    else:
        dictionary[word] = len(dictionary)
        logging.debug('added {} to dictionary'.format(word))
        return dictionary[word]


def train_vector_representation(synth_model, input_texts, optimizer):
    codes = synth_model.forward_generator(input_texts, train=True)
    scores = synth_model.forward_discriminator(input_texts, codes, train=True)
    # generate loss
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    return loss.item()


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
    optimizer = torch.optim.Adam(synth_model.parameters(), lr=args.learning_rate)

    model_dir = model_save_dir + '/' + model_name
    if os.path.exists(model_dir):
        if os.path.isdir(model_dir):
            shutil.rmtree(model_dir, ignore_errors=True)
        else:
            os.remove(model_dir)
    os.mkdir(model_dir)
    shutil.copy(model_filename, model_dir)

    return synth_model, optimizer


def save_model(model_name, synth_model, optimizer, step=None):
    if step:
        model_path = model_save_dir + '/' + model_name + '/' + 'model_{}.pkl'.format(step)
        optim_path = model_save_dir + '/' + model_name + '/' + 'optimizer_{}.pkl'.format(step)
    else:
        model_path = model_save_dir + '/' + model_name + '/' + 'model.pkl'
        optim_path = model_save_dir + '/' + model_name + '/' + 'optimizer.pkl'
    if os.path.exists(model_path):
        os.remove(model_path)
        logging.info('deleted previous model in {}'.format(model_path))
    torch.save(synth_model, model_path)
    torch.save(optimizer, optim_path)
    logging.info('saved model {} into {}'.format(model_name, model_path))


def load_model(model_name, step=None):
    if step:
        model_path = model_save_dir + '/' + model_name + '/' + 'model_{}.pkl'.format(step)
        optim_path = model_save_dir + '/' + model_name + '/' + 'optimizer_{}.pkl'.format(step)
    else:
        model_path = model_save_dir + '/' + model_name + '/' + 'model.pkl'
        optim_path = model_save_dir + '/' + model_name + '/' + 'optimizer.pkl'
    synth_model = torch.load(model_path)
    optimizer = torch.load(optim_path)
    logging.info('loaded model {} from {}'.format(model_name, model_path))
    return synth_model, optimizer


if __name__ == '__main__':
    main()

