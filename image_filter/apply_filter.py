import numpy as np
import cv2
# from sklearn.preprocessing import StandardScaler

def Conv_with_2f(F, H):
    h, w = H.shape

    G = np.zeros((F.shape[0] - h + 1, F.shape[1] - w + 1))
    for i in range(G.shape[0]):
        for j in range(G.shape[1]):
            G[i, j] = (F[i : i + h, j : j + w] * H[::-1]).sum()

    return G

def Conv_with_4f(F, H):
    h, w = H.shape
    G = np.zeros((F.shape[0] - h + 1, F.shape[1] - w + 1))
    k = h//2

    for i  in range(G.shape[0]):
        for j  in range(G.shape[1]):
            sum = 0
            for u  in range(-k, k+1):
                for v  in range(-k, k+1):
                    sum += H[u + k, v + k] * F[i - u + k, j - v + k]
            G[i, j] = sum

    return G

def Corre_with_2f(F, H):
    h, w = H.shape

    G = np.zeros((F.shape[0] - h + 1, F.shape[1] - w + 1))
    for i in range(G.shape[0]):
        for j in range(G.shape[1]):
            G[i, j] = (F[i : i + h, j : j + w] * H).sum()

    return G

def Corre_with_4f(F, H):
    h, w = H.shape
    G = np.zeros((F.shape[0] - h + 1, F.shape[1] - w + 1))
    k = h//2
    
    for i  in range(G.shape[0]):
        for j  in range(G.shape[1]):
            sum = 0
            for u  in range(-k, k+1):
                for v  in range(-k, k+1):
                    sum += H[u + k, v + k] * F[i + u + k, j + v + k]
            G[i, j] = sum
            
    return G

def scale(x, out_range=(-1, 1), axis=None):
	domain = np.min(x, axis), np.max(x, axis)
	y = (x - (domain[1] + domain[0]) / 2) / (domain[1] - domain[0])
    
	return y * (out_range[1] - out_range[0]) + (out_range[1] + out_range[0]) / 2

def Subtraction(F, H):
    h, w = H.shape

    G = np.zeros((F.shape[0] - h + 1, F.shape[1] - w + 1))
    for i in range(G.shape[0]):
        for j in range(G.shape[1]):
            G[i,j] = (abs(F[i : i + h, j : j + w] - H)).sum()

    # scaler = StandardScaler()
    # print('ok')
    # G = scaler.fit_transform(G)
    # G = scale(G)

    return G

if __name__ == '__main__':
    F = np.array([[2, 0, 3, 0 ,1], [1, 0, 4, 2, 1], [3, 1, 1, 0 ,1], [1, 0, 4, 5, 0], [0, 1, 2, 0, 1]])
    H = np.array([[0, 1, 0], [0, 1, 0], [0, 1, 0]])

    filter_a = np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]])
    filter_b = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]]) * (1 / 9)
    filter_c = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]])
    filter_d = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]])

    print('1. Conv with 2 for.')
    print('2. Conv with 4 for.')
    print('3. Corre with 2 for.')
    print('4. Corre with 4 for.')
    print('5. Apply image with filter conv.')
    print('6. Apply image with filter corre.')
    print('7. Apply image with subtraction.')
    try:
        choice = int(input('Your choice is: '))
        if choice == 1:
            G = Conv_with_2f(F, H)
            print(f'Result is:\n {G}')
        elif choice == 2:
            G = Conv_with_4f(F, H)
            print(f'Result is: \n {G}')
        elif choice == 3:
            G = Corre_with_2f(F, H)
            print(f'Result is: \n {G}')
        elif choice == 4:
            G = Corre_with_4f(F, H)
            print(f'Result is: \n {G}')
        elif choice == 5:
            print('Do you want to apply image with filter a,b,c or d?')
            sub_choice = input('Your choice: ')
            if sub_choice == 'a' or sub_choice == 'A':
                img = np.array(cv2.imread('img_test.jpg', 0))
                res = np.array(Conv_with_2f(img, filter_a), dtype=np.uint8)
                cv2.imshow('Original image', img)
                cv2.imshow('Result image', res)
                cv2.waitKey(0)
            elif sub_choice == 'b' or sub_choice == 'B':
                img = np.array(cv2.imread('img_test.jpg', 0))
                res = np.array(Conv_with_2f(img, filter_b), dtype=np.uint8)
                cv2.imshow('Original image', img)
                cv2.imshow('Result image', res)
                cv2.waitKey(0)
            elif sub_choice == 'c' or sub_choice == 'C':
                img = np.array(cv2.imread('img_test.jpg', 0))
                res = np.array(Conv_with_2f(img, filter_c), dtype=np.uint8)
                cv2.imshow('Original image', img)
                cv2.imshow('Result image', res)
                cv2.waitKey(0)
            elif sub_choice == 'd' or sub_choice == 'D':
                img = np.array(cv2.imread('img_test.jpg', 0))
                res = np.array(Conv_with_2f(img, filter_d), dtype=np.uint8)
                cv2.imshow('Original image', img)
                cv2.imshow('Result image', res)
                cv2.waitKey(0)
            else:
                print('Do nothing!!!')
        elif choice == 6:
            print('Do you want to apply image with filter a,b,c or d?')
            sub_choice = input('Your choice: ')
            if sub_choice == 'a' or sub_choice == 'A':
                img = np.array(cv2.imread('img_test.jpg', 0))
                res = np.array(Corre_with_2f(img, filter_a), dtype=np.uint8)
                cv2.imshow('Original image', img)
                cv2.imshow('Result image', res)
                cv2.waitKey(0)
            elif sub_choice == 'b' or sub_choice == 'B':
                img = np.array(cv2.imread('img_test.jpg', 0))
                res = np.array(Corre_with_2f(img, filter_b), dtype=np.uint8)
                cv2.imshow('Original image', img)
                cv2.imshow('Result image', res)
                cv2.waitKey(0)
            elif sub_choice == 'c' or sub_choice == 'C':
                img = np.array(cv2.imread('img_test.jpg', 0))
                res = np.array(Corre_with_2f(img, filter_c), dtype=np.uint8)
                cv2.imshow('Original image', img)
                cv2.imshow('Result image', res)
                cv2.waitKey(0)
            elif sub_choice == 'd' or sub_choice == 'D':
                img = np.array(cv2.imread('img_test.jpg', 0))
                res = np.array(Corre_with_2f(img, filter_d), dtype=np.uint8)
                cv2.imshow('Original image', img)
                cv2.imshow('Result image', res)
                cv2.waitKey(0)
            else:
                print('Do nothing!!!')
        elif choice == 7:
            img = cv2.imread('pocker_card.png', 0)
            template = cv2.imread('diamond.png', 0)

            res = np.array(Subtraction(img, template), dtype=np.uint8)

            res = res / np.max(res) * 255
            thresh = 100
            res[res < thresh] = 0
            res[res > thresh] = 255
            print(res)
            cv2.imshow('input img', img)
            cv2.imshow('filter', template)
            cv2.imshow('result image', 255 - res)
            cv2.waitKey(0)
        else:
            print('Do nothing!!!')
    except:
        print('Do nothing!!!')