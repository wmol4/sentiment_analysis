import matplotlib.pyplot as plt

def save_embedding_image(array, save_name, figsize):
    fig = plt.figure(figsize = figsize, dpi = 100, frameon = False)
    ax = plt.axes([0, 0, 1, 1])
    plt.imshow(array, cmap = 'Greys', interpolation = 'nearest')
    plt.axis('off')
    plt.savefig('{}.png'.format(save_name))
    plt.close()
