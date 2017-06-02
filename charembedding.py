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

def test_log_m_embedding(testImgLoc = relpath('testImgs')):
    from os.path import join, isdir
    import matplotlib.pyplot as plt

    assert isdir(testImgLoc), "Cannot find test image directory {0}".format(testImgLoc)

    all_the_characters = """ !"#$%&'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\\]^_`abcdefghijklmnopqrstuvwxyz{|}~"""
    plt.imshow(log_m_embedding(all_the_characters)).write_png(join(testImgLoc, 'test.png'))
    for i, c in enumerate(all_the_characters):
        plt.imshow(log_m_embedding(c)).write_png(join(testImgLoc, 'test_char_{0}.png').format(i))

if __name__ == "__main__":
    if(DO_TEST):
        try:
            test_log_m_embedding()
        except Exception as e:
            print("Error in test_log_m_embedding: ", repr(e))
