import numpy as np
import cv2

def resize_img(img):
    return cv2.resize(img, (600, 400))

def push(img1, img2):
    img1 = resize_img(img1)
    img2 = resize_img(img2)

    for i in range(img1.shape[0]):
        res_img = np.concatenate((img1[:img1.shape[0]-i,:,:], img2[:i,:,:]), axis=0)
        cv2.imshow('push transition', res_img)
        cv2.waitKey(1)

    cv2.waitKey(0)

def wipe(img1, img2):
    img1 = resize_img(img1)
    img2 = resize_img(img2)

    for i in range(img1.shape[1]):
        res_img = np.concatenate((img1[:,:img1.shape[1]-i,:], img2[:,img2.shape[1]-i:,:]), axis=1)
        cv2.imshow('wipe transition', res_img)
        cv2.waitKey(1)
    cv2.waitKey(0)

def split(img1, img2):
    img1 = resize_img(img1)
    img2 = resize_img(img2)

    for i in range(img2.shape[1]//2):
        res_img = np.concatenate((img1[:,:img1.shape[1]//2-i,:], img2[:,img2.shape[1]//2-i:img2.shape[1]//2+i,:], img1[:,img1.shape[1]//2+i:,:]), axis=1)
        cv2.imshow('split transition', res_img)
        cv2.waitKey(1)
    cv2.waitKey(0)

def uncover(img1, img2):
    img1 = resize_img(img1)
    img2 = resize_img(img2)

    for i in range(img1.shape[1]):
        res_img = np.concatenate((img1[:,i:,:], img2[:,img2.shape[1]-i:,:]), axis=1)
        cv2.imshow('uncover transition', res_img)
        cv2.waitKey(1)
    cv2.waitKey(0)

def cover(img1, img2):
    img1 = resize_img(img1)
    img2 = resize_img(img2)

    for i in range(img1.shape[1]):
        res_img = np.concatenate((img1[:,:img1.shape[1]-i,:], img2[:,:i,:]), axis=1)
        cv2.imshow('cover transition', res_img)
        cv2.waitKey(1)
    cv2.waitKey(0)

def comb(img1, img2):
    img1 = resize_img(img1)
    img2 = resize_img(img2)

    for i in range(img1.shape[1]):
        cross_bar_1 = np.concatenate((img1[:49,i:,:], img2[:49,img2.shape[1]-i:,:]), axis=1)
        cross_bar_2 = np.concatenate((img2[50:99,:i,:], img1[50:99,:img1.shape[1]-i,:]), axis=1)
        cross_bar_3 = np.concatenate((img1[100:149,i:,:], img2[100:149,img2.shape[1]-i:,:]), axis=1)
        cross_bar_4 = np.concatenate((img2[150:199,:i,:], img1[150:199,:img1.shape[1]-i,:]), axis=1)
        cross_bar_5 = np.concatenate((img1[200:249,i:,:], img2[200:249,img2.shape[1]-i:,:]), axis=1)
        cross_bar_6 = np.concatenate((img2[250:299,:i,:], img1[250:299,:img1.shape[1]-i,:]), axis=1)
        cross_bar_7 = np.concatenate((img1[300:349,i:,:], img2[300:349,img2.shape[1]-i:,:]), axis=1)
        cross_bar_8 = np.concatenate((img2[350:399,:i,:], img1[350:399,:img1.shape[1]-i,:]), axis=1)
        res_img = np.concatenate((cross_bar_1, cross_bar_2, cross_bar_3, cross_bar_4,
                                cross_bar_5, cross_bar_6, cross_bar_7, cross_bar_8), axis=0)
        cv2.imshow('comb transition', res_img)
        cv2.waitKey(1)
    cv2.waitKey(0)

def dissolve(img1, img2):
    img1 = resize_img(img1)
    img2 = resize_img(img2)

    shuffle_index = []
    for i in range(0, img1.shape[0], 20):
        for j in range(0, img1.shape[1], 20):
            shuffle_index.append((i,j))

    np.random.shuffle(shuffle_index)

    for i, j in shuffle_index:
        img1[i:i+20,j:j+20,:] = img2[i:i+20,j:j+20,:]
        cv2.imshow('dissolve transition', img1)
        cv2.waitKey(1)
    cv2.waitKey(0)

if __name__ == '__main__':
    img1 = cv2.imread('IU.jpg')
    img2 = cv2.imread('ThuyChi.jpg')

    print('What kind of image transition do you want to use?')
    print('1.Push transition.')
    print('2.Wipe transition.')
    print('3.Split transition.')
    print('4.Uncover transition.')
    print('5.Cover transition.')
    print('6.Comb transition.')
    print('7.Dissolve transition.')
    try:
        choice = int(input('Your choice: '))
        if choice == 1:
            push(img1, img2)
        elif choice == 2:
            wipe(img1, img2)
        elif choice == 3:
            split(img1, img2)
        elif choice == 4:
            uncover(img1, img2)
        elif choice == 5:
            cover(img1, img2)
        elif choice == 6:
            comb(img1, img2)
        elif choice == 7:
            dissolve(img1, img2)           
        else:
            print('Do nothing!!!')
    except:
            print('Do nothing!!!')