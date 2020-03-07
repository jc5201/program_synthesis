import json
import logging
import pickle

naps_dir = './'
naps_train_A = naps_dir + 'naps.trainA.1.0.jsonl'
naps_train_B = naps_dir + 'naps.trainB.1.0.jsonl'
naps_valid_source = naps_dir + 'naps.valid.source.code.pkl'

naps_valid_skip_list = [122, 196, 1410, 1811,          # := in list comp
             327, 734, 786, 1131, 1160, 1779, 1784,    # continue
             704, 707, 714, 806, 832, 877, 936, 968, 1032, 1055, 1072, 1215, 1463,  # := with a[b] or a.b
             984, 1388, 1785,               # concat(str, int)
             1438, 1554,                    # array[char]
             1873, 1986, 2112,              # '10' != '10.0'
             2035,                          # TODO
             ]


def load_naps_valid(filename=naps_valid_source):
    with open(filename, 'rb') as f:
        naps_valid = pickle.load(f)
    logging.info("Loaded naps_valid from {}".format(filename))
    return naps_valid


def gen_naps_valid(filename=naps_valid_source, skip_partial=True):
    data = load_from_jsonl_file(naps_train_B)
    valid_data = []
    for i, problem in enumerate(data):
        if i in naps_valid_skip_list:
            logging.debug("skip data#{}: data#{} in skip_list".format(i, i))
            continue
        if skip_partial and problem['is_partial']:
            logging.debug("skip data#{}: data#{} is partial".format(i, i))
            continue
        valid_data.append(problem)
    logging.info("generated naps_valid from valid source codes of naps trainB")
    with open(filename, 'wb') as f:
        pickle.dump(valid_data, f)
    logging.info("Saved naps_valid to {}".format(filename))


def load_from_jsonl_file(filename):
    data = []
    with open(filename, 'r') as json_file:
        while True:
            line = json_file.readline()
            if not line:
                break
            data.append(json.loads(line))

    return data


def load_word_dictionary():
    filename = 'dictionary.pkl'
    with open(filename, 'rb') as f:
        dictionary = pickle.load(f)
    return dictionary


def gen_word_dictionary():
    filename = 'dictionary.pkl'
    dictionary = {'<unk>': 0}

    with open(naps_train_A, 'r') as f:
        i = 0
        while True:
            line = f.readline()
            if not line:
                break
            data = json.loads(line)
            if 'texts' in data.keys():
                for text in data['texts']:
                    for word in text:
                        if word not in dictionary:
                            dictionary[word] = len(dictionary)
            if 'text' in data.keys():
                for word in data['text']:
                    if word not in dictionary:
                        dictionary[word] = len(dictionary)
            i = i+1
            if i % 100 == 0:
                logging.info("trainA data#{}".format(i))

    with open(naps_train_B, 'r') as f:
        i = 0
        while True:
            line = f.readline()
            if not line:
                break
            data = json.loads(line)
            if 'text' in data.keys():
                for word in data['text']:
                    if word not in dictionary:
                        dictionary[word] = len(dictionary)
            i = i+1
            if i % 100 == 0:
                logging.info("trainB data#{}".format(i))

    with open(filename, 'wb') as f:
        pickle.dump(dictionary, f)
    logging.info("Generated word dictionary and saved to {}".format(filename))
