import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy import misc
from scipy.fft import dctn, idctn


### Exercitiul 1

if __name__ == '__main__':
    # Incarcam imaginea si definim matricea pentru cuantizare
    image = misc.ascent()
    Q_jpeg = np.array([
        [16, 11, 10, 16, 24, 40, 51, 61],
        [12, 12, 14, 19, 26, 28, 60, 55],
        [14, 13, 16, 24, 40, 57, 69, 56],
        [14, 17, 22, 29, 51, 87, 80, 62],
        [18, 22, 37, 56, 68, 109, 103, 77],
        [24, 35, 55, 64, 81, 104, 113, 92],
        [49, 64, 78, 87, 103, 121, 120, 101],
        [72, 92, 95, 98, 112, 100, 103, 99]
    ])

    height, width = image.shape
    image_jpeg = np.zeros_like(image)

    tot_comp_nonzero = 0
    tot_comp_jpeg_nonzero = 0

    # Parcurgem imaginea in blocuri de 8x8
    for i in range(0, height, 8):
        for j in range(0, width, 8):
            block = image[i:i + 8, j:j + 8]

            block_dct = dctn(block, type=2)
            quantiz_block = np.round(block_dct / Q_jpeg) * Q_jpeg
            idct_block = idctn(quantiz_block)
            image_jpeg[i:i + 8, j:j + 8] = idct_block

            tot_comp_nonzero += np.count_nonzero(block_dct)
            tot_comp_jpeg_nonzero += np.count_nonzero(quantiz_block)

    plt.figure(figsize=(10, 5))
    plt.suptitle('Exercitiul 1')

    plt.subplot(1, 2, 1)
    plt.imshow(image, cmap=plt.cm.gray)
    plt.title('Imagine Originala')

    plt.subplot(1, 2, 2)
    plt.imshow(image_jpeg, cmap=plt.cm.gray)
    plt.title('Imagine JPEG')

    plt.show()

    print('Componente in frecventa: ' + str(tot_comp_nonzero) +
          '\nComponente in frecventa dupa cuantizare: ' + str(tot_comp_jpeg_nonzero))


### Exercitiul 2

image_color = misc.face()

# Conversia din RGB in YCbCr
def rgb_to_ycbcr(img):
    # Coeficientii de conversie
    matrix = np.array([[0.299, 0.587, 0.114], #Y
                       [-0.1687, -0.3313, 0.5], #Cb
                       [0.5, -0.4187, -0.0813]]) #Cr
    adjust_comp = np.array([0, 128, 128])
    return np.dot(img - adjust_comp, matrix.T)


# Conversia din YCbCr in RGB
def ycbcr_to_rgb(img):
    # Coeficientții de conversie
    matrix = np.array([[1, 0, 1.402], #R
                       [1, -0.344136, -0.714136], #G
                       [1, 1.772, 0]]) #B
    adjust_comp = np.array([0, 128, 128])
    return np.dot(img, matrix.T) + adjust_comp


image_ycbcr = rgb_to_ycbcr(image_color).astype(float)

Q_jpeg = np.array([
    [16, 11, 10, 16, 24, 40, 51, 61],
    [12, 12, 14, 19, 26, 28, 60, 55],
    [14, 13, 16, 24, 40, 57, 69, 56],
    [14, 17, 22, 29, 51, 87, 80, 62],
    [18, 22, 37, 56, 68, 109, 103, 77],
    [24, 35, 55, 64, 81, 104, 113, 92],
    [49, 64, 78, 87, 103, 121, 120, 101],
    [72, 92, 95, 98, 112, 100, 103, 99]
])
Y_jpeg = np.zeros_like(image_ycbcr)

for channel in range(3):
    height, width = image_ycbcr.shape[:2]
    for i in range(0, height, 8):
        for j in range(0, width, 8):

            block = image_ycbcr[i:i + 8, j:j + 8, channel]
            dct_block = dctn(block, type=2)

            quantized_block = np.round(dct_block / Q_jpeg) * Q_jpeg
            idct_block = idctn(quantized_block, type=2)
            Y_jpeg[i:i + 8, j:j + 8, channel] = idct_block


# Facem conversia inapoi in RGB cu conditia ca valorile sa fie in intervalul [0, 255]
image_compressed = ycbcr_to_rgb(Y_jpeg)
image_compressed = np.clip(image_compressed, 0, 255).astype('uint8')

plt.figure(figsize=(10, 5))
plt.suptitle('Exercitiul 2')

plt.subplot(1, 2, 1)
plt.imshow(image_color)
plt.title('Imagine Originala')

plt.subplot(1, 2, 2)
plt.imshow(image_compressed)
plt.title('Imagine JPEG RGB')

plt.show()


### Exercitiul 3

def mse(image_a, image_b):
    err = np.sum((image_a.astype("float") - image_b.astype("float")) ** 2)
    err /= float(image_a.shape[0] * image_a.shape[1] * image_a.shape[2])
    return err


def compress_image(image_ycbcr, Q_jpeg):
    Y_jpeg = np.zeros_like(image_ycbcr)
    for channel in range(3):
        height, width = image_ycbcr.shape[:2]
        for i in range(0, height, 8):
            for j in range(0, width, 8):
                block = image_ycbcr[i:i + 8, j:j + 8, channel]
                dct_block = dctn(block, type=2)
                quantized_block = np.round(dct_block / Q_jpeg) * Q_jpeg
                idct_block = idctn(quantized_block, type=2)
                Y_jpeg[i:i + 8, j:j + 8, channel] = idct_block
    return Y_jpeg


Q_jpeg = np.array([
    [16, 11, 10, 16, 24, 40, 51, 61],
    [12, 12, 14, 19, 26, 28, 60, 55],
    [14, 13, 16, 24, 40, 57, 69, 56],
    [14, 17, 22, 29, 51, 87, 80, 62],
    [18, 22, 37, 56, 68, 109, 103, 77],
    [24, 35, 55, 64, 81, 104, 113, 92],
    [49, 64, 78, 87, 103, 121, 120, 101],
    [72, 92, 95, 98, 112, 100, 103, 99]
])

Q_jpeg_original = Q_jpeg.copy()

adjust_factor = 1.1

# Stabilim pragul MSE
mse_threshold = 0.01

Y_jpeg = compress_image(image_ycbcr, Q_jpeg)
image_compressed = ycbcr_to_rgb(Y_jpeg)
image_compressed = np.clip(image_compressed, 0, 255).astype('uint8')

current_mse = mse(image_color, image_compressed)
print("\nExercitiul 3")
print(f"MSE initial: {current_mse}")

while current_mse > mse_threshold:
    Q_jpeg = (Q_jpeg_original * adjust_factor).astype(int)

    Y_jpeg = compress_image(image_ycbcr, Q_jpeg)
    image_compressed = ycbcr_to_rgb(Y_jpeg)
    image_compressed = np.clip(image_compressed, 0, 255).astype('uint8')

    current_mse = mse(image_color, image_compressed)
    print(f"MSE dupa ajustare: {current_mse}")

    adjust_factor += 0.1

    # Evitam ajutari excesive printr-o conditie
    if adjust_factor > 2:
        print("S-a atins limita de ajustare -> Oprire")
        break


plt.figure(figsize=(10, 5))
plt.suptitle('Exercitiul 3')

plt.subplot(1, 2, 1)
plt.imshow(image_color)
plt.title('Imagine Originala')

plt.subplot(1, 2, 2)
plt.imshow(image_compressed)
plt.title('Imagine JPEG RGB')

plt.show()


### Exercitiul 4

def compress_frame(frame, Q_jpeg):
    # Conversie in YCbCr
    frame_ycbcr = rgb_to_ycbcr(frame).astype(float)
    compressed = np.zeros_like(frame_ycbcr)

    # Facem compresia fiecarui canal
    for channel in range(3):
        height, width = frame_ycbcr.shape[:2]
        for i in range(0, height, 8):
            for j in range(0, width, 8):
                block = frame_ycbcr[i:i + 8, j:j + 8, channel]
                dct_block = dctn(block, type=2)
                quantized_block = np.round(dct_block / Q_jpeg) * Q_jpeg
                idct_block = idctn(quantized_block, type=2)
                compressed[i:i + 8, j:j + 8, channel] = idct_block

    return ycbcr_to_rgb(compressed)


def extract_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    return frames


def join_video (frames, output_path='output.mp4'):
    height, width, layers = frames[0].shape
    video = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), 30, (width, height))
    for frame in frames:
        video.write(frame)
    video.release()


print('\nExercitiul 4\nSe proceseaza videoclipul...')
init_path = 'C:\FMI-CTI\ANUL 4\Procesarea semnalelor\Tema 1\pisica.mp4'
extracted_frames = extract_frames(init_path)

Q_jpeg = np.array([
    [16, 11, 10, 16, 24, 40, 51, 61],
    [12, 12, 14, 19, 26, 28, 60, 55],
    [14, 13, 16, 24, 40, 57, 69, 56],
    [14, 17, 22, 29, 51, 87, 80, 62],
    [18, 22, 37, 56, 68, 109, 103, 77],
    [24, 35, 55, 64, 81, 104, 113, 92],
    [49, 64, 78, 87, 103, 121, 120, 101],
    [72, 92, 95, 98, 112, 100, 103, 99]
])
compressed_frames = [compress_frame(frame, Q_jpeg) for frame in extracted_frames]

join_video(compressed_frames, 'C:\FMI-CTI\ANUL 4\Procesarea semnalelor\Tema 1\pisica_frumoasa.mp4')
print('Videoclipul a fost compresat')
