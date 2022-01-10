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
    filter_a = np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]])
    filter_b = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]]) * (1 / 9)
    filter_c = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]])
    filter_d = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]])

    img = cv2.imread('pocker_card.png', 0)
    template = cv2.imread('diamond.png', 0)
    res = np.array(Subtraction(img, template))

    res = res / np.max(res) * 255
    res = res.astype('uint8')
    thresh = 100
    res[res < thresh] = 0
    res[res > thresh] = 255
    print(res)

    cv2.imshow('input img', img)
    cv2.imshow('filter', template)
    cv2.imshow('result image', 255 - res)
    cv2.waitKey(0)