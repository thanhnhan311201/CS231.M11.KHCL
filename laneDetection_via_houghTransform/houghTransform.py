import numpy as np
import cv2
import math

def roi(img):
    height, width = img.shape
    mask = np.zeros_like(img)

    polygon = np.array([[
        (0, height * 2 / 5),
        (width, height * 2 / 5),
        (width, height),
        (0, height),
    ]], np.int32)

    cv2.fillPoly(mask, polygon, 255)
    masked = cv2.bitwise_and(img, mask)
    
    return masked

def process_img(original_img):
    # processed_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)
    processed_img = cv2.inRange(original_img, (190,190,190), (255,255,255))
    # cv2.imshow('binary image', processed_img)
    processed_img = cv2.Canny(processed_img, threshold1=100, threshold2=200)
    # cv2.imshow('edge image', processed_img)
    processed_img = roi(processed_img)
    cv2.imshow('processed image', processed_img)
    
    return processed_img

def lineDetect_via_loop(img):
    img = cv2.resize(img, (600,320))
    Edge = process_img(img)

    h, w = np.shape(Edge)
    phu = int(math.sqrt(h**2 + w**2))
    H = np.zeros((360, phu), dtype=int)

    for yi in range(0, h):
        for xi in range(0, w):
            if Edge[yi, xi] > 0:
                for theta_D in range(0, 360):
                    theta = math.radians(theta_D)
                    r = int(xi * math.cos(theta) + yi * math.sin(theta))
                    H[theta_D, r] += 1
    
    print(H)

    eps = 1e-9
    threshold = 120
    h1, w1 = np.shape(H)
    for y in range(0, h1):
        for x in range(0, w1):
            if H[y, x] > threshold:
                theta_y = math.radians(y)

                a = -(math.cos(theta_y) / (math.sin(theta_y) + eps))
                b = x/(math.sin(theta_y) + eps)
                for x1 in range(0, w):
                    y1 = int(a * x1 + b)
                    if y1 < img.shape[0] and y1 > -1:
                        img[y1, x1] = [255,0,0]

    return img

def lineDetect_via_houghLinesP(img):
    processed_img = process_img(img)

    lines = cv2.HoughLinesP(processed_img,1, np.pi/180, 100, minLineLength=10, maxLineGap=50)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(img, (x1, y1), (x2, y2), [255, 0, 0], 3)

    return img

if __name__ == '__main__':
    img = cv2.imread('img.jpg')

    print('1. Line detection with loop.')
    print('2. Line detection with HoughLinesP.')
    print('Which one do you want to apply?')
    try:
        choice = int(input('Your choice: '))
        if choice == 1:
            res_img = lineDetect_via_loop(img)

            cv2.imshow('Result image', res_img)
            cv2.imwrite('resImg_via_loop.jpg', res_img)
            cv2.waitKey(0)
        elif choice == 2:
            res_img = lineDetect_via_houghLinesP(img)

            cv2.imshow('Result image', res_img)
            cv2.imwrite('resImg_via_houghLinesP.jpg', res_img)
            cv2.waitKey(0)
    except:
        print('Do nothing!')

    cv2.destroyAllWindows()