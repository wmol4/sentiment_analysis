import numpy as np
from os.path import relpath

DO_TEST = False

def to_bit_seq(int_x, length):
    """
    Returns the bit string of x (of exactly the passed length)
    (If the binary representation is too long, will truncate from the right,
    if too short, will add zeros to the front)
    """
    binary_str = bin(int_x)[2:length+2]
    return '0'*(length - len(binary_str)) + binary_str

def symbol_list_to_encoder(symbol_array):
    """
    Make a symbol encoder from an array of symbols.
    symbol_array: An iterable of symbols. (can use `len` and is iterable)
    returns: (b, encoder)
        b: number of bits the encoder fits to
        encoder: the encoder for the symbols, mapping each symbol to it's
                    index + 1 in the input symbol_array.
                    Maps all unknown symbols to 0
    """
    symbol_to_index_dict = {s: i + 1 for i, s in enumerate(symbol_array)}
    return len(symbol_array) + 1, lambda s: symbol_to_index_dict[s] if s in symbol_to_index_dict else 0

def generate_log_m_embedding_function_from_symbol_list(symbol_list):
    """
    Make a specialized log_m_embedding_function from an array of symbols.
    symbol_array: An iterable of symbols. (can use `len` and is iterable)
    returns: (rows, special_log_m_embedding)
    """
    rows, encoder = symbol_list_to_encoder(symbol_list)
    def special_log_m_embedding(in_symbol_iter, maxlen=None, dtype = int):
        return log_m_embedding(in_symbol_iter, maxlen=maxLen, rows = rows, dtype = dtype, symbol_encoder=encoder)
    return special_log_m_embedding, rows

def log_m_embedding(in_symbol_iter, maxlen=None, rows = 7, dtype = int, symbol_encoder = lambda x: ord(x) - 32):
    """
        Encodes an iterator of symbols to columns of bits.
        symbol_encoder: char -> int
            function to map each character to a number
        rows: int
            number of rows.
    """
    embedding = np.asarray([b=='1' for c in in_symbol_iter for b in to_bit_seq(symbol_encoder(c), rows)]\
                           [slice(None,maxlen*rows if maxlen != None else None)],\
                           dtype=dtype)

    a = np.transpose(embedding.reshape((len(embedding)//rows, rows)))
    if(maxlen != None):
        ret = np.zeros((rows, maxlen))
        ret[:a.shape[0], :a.shape[1]] = a
    else:
        ret = a
    return ret

def test_log_m_embedding(testImgLoc = relpath('testImgs'), symbol_encoder = None):
    from os.path import join, isdir
    import matplotlib.pyplot as plt
    from img_helpers import save_embedding_image

    assert isdir(testImgLoc), "Cannot find test image directory {0}".format(testImgLoc)

    all_the_characters = """ !"#$%&'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\\]^_`abcdefghijklmnopqrstuvwxyz{|}~"""
    if(symbol_encoder == None):
        save_embedding_image(log_m_embedding(all_the_characters), join(testImgLoc, 'test'), (9.5, .7))
        for i, c in enumerate(all_the_characters):
            save_embedding_image(log_m_embedding(c), join(testImgLoc, 'test_char_{0}').format(i), (.1, .7))
    else:
        save_embedding_image(log_m_embedding(all_the_characters, symbol_encoder = symbol_encoder), join(testImgLoc, 'test'), (9.5, .7))
        for i, c in enumerate(all_the_characters):
            save_embedding_image(log_m_embedding(c, symbol_encoder = symbol_encoder), join(testImgLoc, 'test_char_{0}').format(i), (.1, .7))

if __name__ == "__main__":
    if(DO_TEST):
        try:
            test_log_m_embedding()
        except Exception as e:
            print("Error in test_log_m_embedding: ", repr(e))
