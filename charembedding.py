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

def log_m_embedding(in_string, maxlen=None, rows = 7, dtype = int, char_encode = lambda x: ord(x) - 32):
    """
        Encodes a string with columns of bits as characters.
        char_encode: char -> int
            function to map each character to a number
        rows: int
            number of rows.
    """
    embedding = np.asarray([b=='1' for c in in_string for b in to_bit_seq(char_encode(c), rows)]\
                           [slice(None,maxlen*rows if maxlen != None else None)],\
                           dtype=dtype)

    a = np.transpose(embedding.reshape((len(embedding)//rows, rows)))
    if(maxlen != None):
        ret = np.zeros((rows, maxlen))
        ret[:a.shape[0], :a.shape[1]] = a
    else:
        ret = a
    return ret

def test_log_m_embedding(testImgLoc = relpath('testImgs'), char_encode = None):
    from os.path import join, isdir
    import matplotlib.pyplot as plt
    from img_helpers import save_embedding_image

    assert isdir(testImgLoc), "Cannot find test image directory {0}".format(testImgLoc)

    all_the_characters = """ !"#$%&'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\\]^_`abcdefghijklmnopqrstuvwxyz{|}~"""
    if(char_encode == None):
        save_embedding_image(log_m_embedding(all_the_characters), join(testImgLoc, 'test'), (9.5, .7))
        for i, c in enumerate(all_the_characters):
            save_embedding_image(log_m_embedding(c), join(testImgLoc, 'test_char_{0}').format(i), (.1, .7))
    else:
        save_embedding_image(log_m_embedding(all_the_characters, char_encode = char_encode), join(testImgLoc, 'test'), (9.5, .7))
        for i, c in enumerate(all_the_characters):
            save_embedding_image(log_m_embedding(c, char_encode = char_encode), join(testImgLoc, 'test_char_{0}').format(i), (.1, .7))

if __name__ == "__main__":
    if(DO_TEST):
        try:
            test_log_m_embedding()
        except Exception as e:
            print("Error in test_log_m_embedding: ", repr(e))
