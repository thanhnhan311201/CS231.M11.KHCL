import numpy as np
import cv2
import matplotlib.pyplot as plt

def calc_histogram(img):
    pixel_counts = np.zeros((256,), dtype=int)
    pixel = np.arange(256)
    flatten_img = img.flatten()
    
    for i in range(len(flatten_img)):
        pixel_counts[flatten_img[i]] += 1

    return pixel, pixel_counts

if __name__ == '__main__':
    img = np.array(cv2.imread('img_test.jpg', 0))

    print('Plot histogram chart with your function.')
    print('Plot histogram chart with cv2.calcHist.')
    try:
        choice = int(input('Your choice: '))
        if choice == 1:
            pixel, pixel_counts = calc_histogram(img)

            # plt.bar(pixel, pixel_counts)
            plt.plot(pixel, pixel_counts)
            plt.title('Histogram chart')
            plt.xlabel('Grayscale values')
            plt.ylabel('Pixel counts')
            plt.savefig('histogram_chart.png')
            plt.show()
        elif choice == 2:
            hist = cv2.calcHist([img], [0], None, [256], [0, 256])

            plt.plot(hist)
            plt.title('Histogram chart')
            plt.xlabel('Grayscale values')
            plt.ylabel('Pixel counts')
            plt.savefig('histogram_chart_via_cv2Function.png')
            plt.show()
        else:
            print('Do nothing!')
    except:
        print('Do nothing!')