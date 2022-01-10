import cv2 
import numpy as np
from tqdm import trange
import sys
import matplotlib.pyplot as plt
import time
import argparse

def calc_energy(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    gX = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=5)
    gY = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=5)

    energy = np.sqrt((gX**2) + (gY**2))

    return energy

def detect_seam_map(energy):
    h, w = energy.shape
    M = np.copy(energy)

    for i in range(1, h):
        for j in range(w):
            if j == 0:
                idx_min = np.argmin(M[i-1, j:j+2])
                min = M[i-1, j+idx_min]
            else:
                idx_min = np.argmin(M[i-1, j-1:j+2])
                min = M[i-1 ,j-1 + idx_min]
            M[i, j] += min
    
    return M

def remove_seam(seam_map, img, n):
    M = np.copy(seam_map)
    h, w =  seam_map.shape

    for k in range(n):
        h_M, w_M = M.shape
        mask = np.ones((h_M, w_M), dtype=bool)
        j = np.argmin(M[-1])
        x, y = [len(M)-1, len(M)-2], [j-1, j]
        mask[len(M)-1, j]=False
        for i in range(len(M)-2, -1, -1):
            if j == 0:
                j = j + np.argmin(M[i, 0:j+2])
            else:
                j = j - 1 + np.argmin(M[i, j-1:j+2])
            mask[i,j] = False
            if (i, j+1) == (x[-1], y[-1]):
                y[-1] = y[-1] - 1
            if (i, j-1) != (x[-1], y[-1]):
                x.append(i)
                y.append(j-1)
            x.append(i-1)
            y.append(j)
            
        x,y = np.array(x), np.array(y)
        mask_D = np.zeros_like(M, dtype=bool)
        mask_D[x, y] = True
        M = M[mask].reshape((h, w-k-1))
        y = y - 1
        M = np.pad(M, [(0, 1), (0, 1)], mode='constant')
        M[x, y] = np.sqrt(np.power(M[x, y+1]-M[x, y], 2) + np.power(M[x+1, y]-M[x, y], 2))
        M = M[:-1, :-1]
        mask = np.stack([mask] * 3, axis=2)
        img = img[mask].reshape((h, w-k-1,3))
    
    return img, n
		
def remove_multiple_seam(img, scale):
    n = img.shape[1]-int(img.shape[1]*scale)
    total_seam_deleted = 0
    seam_delete_per_loop = max(1, n//25)

    while total_seam_deleted < n:
        img_ = np.copy(img)
        energy_map = calc_energy(img_)
        seam_map = detect_seam_map(energy_map)
        img, seam_deleted = remove_seam(seam_map, img_, min(seam_delete_per_loop, n-total_seam_deleted))
        total_seam_deleted += seam_deleted
        cv2.imshow('Result',img)
        cv2.waitKey(1)

    return img

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', type=str, default='image.jpg')
    parser.add_argument('--resize', type=str)
    parser.add_argument('--scale', type=float, default=0.8)
    args = vars(parser.parse_args())

    img_path = args['image']
    scale = args['scale']
    
    img = cv2.imread(img_path)
    if args['resize']:
        size = tuple([int(x) for x in args['resize'].split(',')])
        img = cv2.resize(img, size)

    cv2.imshow('Original image', img)
    print("Original image shape: ", img.shape)

    res_img = remove_multiple_seam(img, scale)
    print("Result image shape: ", res_img.shape)
    
    cv2.imwrite(f'result_{img_path}', res_img)
    cv2.waitKey(0)

if __name__ == '__main__':
    main()