import numpy as np
import matplotlib.pyplot as plt
def to_bit_seq(int_x, length):
    """
    Returns the bit string of x (of exactly the passed length)
    (If the binary representation is too long, will truncate from the right,
    if too short, will add zeros to the front)
    """
    binary_str = bin(int_x)[2:length+2]
    return '0'*(length - len(binary_str)) + binary_str

def log_m_embedding(in_string, maxlen=None, char_encode = ord, rows = 8, dtype = int):
    """
        Encodes a string with columns of bits as characters.
        char_encode: char -> int
            function to map each character to a number
        rows: int
            number of rows.
    """
    embedding = np.asarray([b=='1' for c in in_string for b in to_bit_seq(char_encode(c), rows)][slice(None,maxlen)],\
                           dtype=dtype)
    a = np.transpose(embedding.reshape((len(embedding)//rows, rows)))
    if(maxlen != None):
        ret = np.zeros((rows, maxlen))
        ret[:a.shape[0], :a.shape[1]] = a
    else:
        ret = a
    return ret
