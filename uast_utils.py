import copy


def list_init(args):
    nlist = 0
    for n in reversed(args):
        nlist = [copy.deepcopy(nlist) for _ in range(n)]
    return nlist


def replace_string_at(string, idx, char):
    return string[:idx] + (chr(char) if isinstance(char, int) else char) + string[idx + 1:]
