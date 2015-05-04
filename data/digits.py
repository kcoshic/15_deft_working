import numpy as np
import struct

digit_folder = '/Users/jkinney/github/15_deft/data/digits/'

def read_binary_image_from_file(file_handle,width,height):
    G = width*height
    image = np.zeros(G)
    for g in range(G):
        success = False
        byte = file_handle.read(1)
        if byte=='':
            break
        image[g] = ord(byte)
        success = True

    return image.reshape(width,height), success

def get_digit_images(num=1, digit='random'):
    # Choose a digit
    if not digit in range(10):
        digit = np.random.choice(10)

    # Open file containing images of that digit
    file_name = '%sdata%d'%(digit_folder,digit)
    f = open(file_name, "rb")
    assert f

    # Read all images from that file
    width = 28
    height = 28
    images = []
    while True:
        image, success = read_binary_image_from_file(f,width,height)
        if success:
            images.append(image)
        else:
            f.close()
            break

    # Return a bunch of images
    indices= np.random.choice(len(images),size=num)
    return [images[i] for i in indices]

if __name__ == '__main__':
    import matplotlib.pyplot as plt

    # Show some randomly selected digits
    plt.close('all')
    num_cols = 5
    num_rows = 1
    K = num_cols*num_rows
    plt.figure(figsize=[10,10])
    images = get_digit_images(K)
    for k in range(K):
        plt.subplot(num_rows, num_cols, k)
        plt.imshow(images[k], cmap='bone', interpolation='nearest')
        plt.xticks([])
        plt.yticks([])

    # Plotting incantation
    plt.ion() # So focus goes back to commandline
    plt.draw() # Needed to avoid "CGContextRef is NULL" exception
    plt.show()
    #plt.tight_layout() # Needed so plot is drawn tollerably