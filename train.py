import torch

import logging
import argparse
import os
import shutil
import time
import random
import itertools
import json
from torch.utils.tensorboard import SummaryWriter

import loader
import model
import executor

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
parser.add_argument('--batch-size', type=int, default='10',
                    help='batch size')
parser.add_argument('--learning-rate', type=float, default='1e-3',
                    help='learning rate')
parser.add_argument('--cuda', action='store_true',        # TODO
                    help='use cuda')
parser.add_argument('--model-name', type=str, default='',
                    help='name of train model')
parser.add_argument('--note-dim', type=int, default='16',
                    help='dimension of note vector')
parser.add_argument('--disable-terminal-log', action='store_true',
                    help='do no print log to terminal')
parser.add_argument('--log-level', type=str, default='info',        # TODO
                    help='log-level (debug, info, warning, error)')
parser.add_argument('--save-interval', type=int, default='4000',
                    help='model save interval')


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
    dataset = loader.load_from_jsonl_file(loader.naps_train_A, [])

    tensorboard_writer = SummaryWriter('runs/' + model_name)
    recent_save_step = synth_model.get_steps() // args.save_interval

    logging.info('start training. step: {}, batch size:{}, learning rate:{}'.format(
        synth_model.get_steps(), batch_size, args.learning_rate
    ))

    for epoch in range(target_epoch):
        logging.info('start epoch #{}'.format(epoch))
        batches = make_batch(training_set, batch_size)
        gen_loss_list = []
        disc_loss_list = []
        for i, batch in enumerate(batches):
            logging.debug('start batch #{}: {}'.format(i, batch))
            optimizer.zero_grad()
            raw_data = load_batch_from(dataset, batch)
            text_idx_data, text_data, tests = ready_data(raw_data, text_dictionary)
            gen_loss, dis_loss, target_score = train_vector_representation(synth_model, text_idx_data, text_data, tests, optimizer)

            gen_loss_list.append(gen_loss)
            disc_loss_list.append(dis_loss)
            synth_model.increase_step(batch_size)
            tensorboard_writer.add_scalar('Loss/train_gen', gen_loss, synth_model.get_steps())
            tensorboard_writer.add_scalar('Loss/train_dis', dis_loss, synth_model.get_steps())
            tensorboard_writer.add_scalar('Loss/train_target', target_score, synth_model.get_steps())
            logging.debug('finished batch: gen_loss: {}, dis_loss: {}, target_score: {}'.format(
                gen_loss, dis_loss, target_score))
            if synth_model.get_steps() // args.save_interval > recent_save_step:
                save_model(model_name, synth_model, optimizer, synth_model.get_steps())
                recent_save_step = synth_model.get_steps() // args.save_interval

        raw_data = load_batch_from(dataset, validation_set)
        text_data = ready_data(raw_data, text_dictionary)
        eval_gen_loss, eval_dis_loss = evaluate_vector_representation(synth_model, text_idx_data, text_data, tests)

        average_gen_loss = sum(gen_loss_list) / len(gen_loss_list)
        average_dis_loss = sum(disc_loss_list) / len(disc_loss_list)
        logging.info('epoch #{}: training_loss: {}, {}, validation_loss: {}, {}'.format(
            epoch, average_gen_loss, average_dis_loss, eval_gen_loss, eval_dis_loss))
    save_model(model_name, synth_model, optimizer)
    tensorboard_writer.close()


def make_batch(training_set, batch_size):
    random.shuffle(training_set)
    return [list(itertools.islice(training_set, batch_size * i, batch_size * (i + 1)))
            for i in range((len(training_set) - 1) // batch_size + 1)]


def load_batch_from(dataset, batch):
    return [dataset[i] for i in batch]


def ready_data(data, text_dictionary):
    text_tensors = []
    raw_texts = []
    tests = []
    for raw_data in data:
        parsed_data = json.loads(raw_data)
        if 'text' in parsed_data.keys():
            text_data = ready_text_for_training(parsed_data['text'], text_dictionary)
            raw_texts.append(parsed_data['text'])
        elif 'texts' in parsed_data.keys():
            text_data = ready_text_for_training(parsed_data['texts'][0], text_dictionary)
            raw_texts.append(parsed_data['texts'][0])
        else:
            logging.error('unexpected data format')
            raise AssertionError
        text_tensors.append(text_data)
        tests.append(parsed_data['tests'])
    return text_tensors, raw_texts, tests


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


def train_vector_representation(synth_model, input_texts, raw_input_texts, tests, optimizer):
    synth_model.set_text_encoder_trainable(False)
    synth_model.set_generator_trainable(False)
    synth_model.set_discriminator_trainable(True)
    optimizer.zero_grad()
    codes = synth_model.forward_generator(input_texts, train=True)
    scores = synth_model.forward_discriminator(input_texts, codes, train=True)
    target_score = executor.evaluate_code(codes, raw_input_texts, tests)
    target_score = torch.Tensor(target_score)
    disc_loss = torch.nn.MSELoss()(target_score, scores)
    disc_loss.backward()
    optimizer.step()

    synth_model.set_text_encoder_trainable(True)
    synth_model.set_generator_trainable(True)
    synth_model.set_discriminator_trainable(False)
    optimizer.zero_grad()
    codes = synth_model.forward_generator(input_texts, train=True)
    scores = synth_model.forward_discriminator(input_texts, codes, train=True)
    gen_loss = torch.sum(scores) * -1 / len(input_texts)
    gen_loss.backward()
    optimizer.step()

    return gen_loss.item(), disc_loss.item(), target_score.mean().item()


def evaluate_vector_representation(synth_model, input_texts, raw_input_texts, tests):
    codes = synth_model.forward_generator(input_texts, train=False)
    scores = synth_model.forward_discriminator(input_texts, codes, train=False)
    target_score = executor.evaluate_code(codes, raw_input_texts, tests)
    target_score = torch.Tensor(target_score)
    gen_loss = -1 * torch.sum(scores)
    disc_loss = torch.nn.MSELoss()(target_score, scores)
    return gen_loss, disc_loss


def model_exists(model_name):
    return os.path.exists(model_save_dir + '/' + model_name + '/' + 'model.pkl')


def create_model(model_name):
    synth_model = model.CodeSynthesisModel(args.note_dim)
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

