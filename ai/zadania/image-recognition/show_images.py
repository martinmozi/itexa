from loader import MnistDataloader
from os.path import join
import matplotlib
import matplotlib.pyplot as plt
import random

# Prefer a GUI backend on desktop systems, but fall back to file output in headless envs.
try:
    matplotlib.use('Qt5Agg', force=True)
except Exception:
    try:
        matplotlib.use('TkAgg', force=True)
    except Exception:
        matplotlib.use('Agg', force=True)


input_path = './mnist'
training_images_filepath = join(input_path, 'train-images-idx3-ubyte/train-images-idx3-ubyte')
training_labels_filepath = join(input_path, 'train-labels-idx1-ubyte/train-labels-idx1-ubyte')
test_images_filepath = join(input_path, 't10k-images-idx3-ubyte/t10k-images-idx3-ubyte')
test_labels_filepath = join(input_path, 't10k-labels-idx1-ubyte/t10k-labels-idx1-ubyte')


def show_images(images, title_texts):
    cols = 5
    rows = int(len(images)/cols) + 1
    plt.figure(figsize=(30, 20))
    index = 1
    for x in zip(images, title_texts):
        image = x[0]
        title_text = x[1]
        plt.subplot(rows, cols, index)
        plt.imshow(image, cmap=plt.cm.gray)
        if (title_text != ''):
            plt.title(title_text, fontsize=15)
        index += 1
    plt.tight_layout()

    backend = matplotlib.get_backend().lower()
    non_gui_backends = {'agg', 'cairo', 'pdf', 'ps', 'svg', 'template'}
    if backend.split('.')[-1] in non_gui_backends or backend == 'agg':
        output_file = 'mnist_samples.png'
        plt.savefig(output_file, bbox_inches='tight')
        print(f'Obrázky boli uložené do {output_file}. Tento prostredie nemá GUI backend.')
    else:
        plt.show()


mnist_dataloader = MnistDataloader(training_images_filepath, training_labels_filepath, test_images_filepath, test_labels_filepath)
(x_train, y_train), (x_test, y_test) = mnist_dataloader.load_data()


images_2_show = []
titles_2_show = []
# display random training and test images with their labels
for i in range(0, 10):
    r = random.randint(0, len(x_train) - 1)
    images_2_show.append(x_train[r])
    titles_2_show.append('training image [' + str(r) + '] = ' + str(y_train[r]))

for i in range(0, 5):
    r = random.randint(0, len(x_test) - 1)
    images_2_show.append(x_test[r])
    titles_2_show.append('test image [' + str(r) + '] = ' + str(y_test[r]))

show_images(images_2_show, titles_2_show)
