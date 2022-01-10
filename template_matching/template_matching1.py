import numpy as np
import cv2
# import matplotlib.pyplot as plt

def template_matching(F, H):
    """
    Implement template matching algorithm
    """
    F_height, F_width = F.shape[:2]
    H_height, H_width = H.shape[:2]

    G_height, G_width = int(((F_height - H_height)/1)+1), int(((F_width - H_width)/1)+1)
    G_shape = (G_height, G_width, 3)
    G = np.zeros(G_shape, dtype=np.int32)
    #print(G.dtype)

    for i in range(G.shape[0]):
        for j in range(G.shape[1]):
            G[i, j, :] = np.sum((np.absolute(np.subtract(F[i:H_height+i, j:H_width+j, :], H))), dtype=np.int32) / (H.shape[0] * H.shape[1])
            #G[i, j, :] = cv2.subtract(F[i:H_height+i, j:H_width+j, :], H).sum()
            #G[i, j, :]  = cv2.absdiff(F[i:H_height+i, j:H_width+j, :], H).sum()
            #print(G[i, j, :])
            
    return G

def main():
    img = cv2.imread("cards.png")
    img_for_detect = img.copy()
    cv2.imshow("img src", img)
    img = img.astype('int32')
    #print(img.dtype)
    scale = 0.5
    filter = cv2.imread("filter.png")
    filter = cv2.rotate(filter, rotateCode=cv2.ROTATE_180)
    print(filter.shape)
    filter = cv2.resize(filter, (int(filter.shape[1] * scale), int(filter.shape[0] * scale)))
    cv2.imshow("template img", filter)
    filter = filter.astype('int32')
    #print(filter.dtype)
    res = template_matching(img, filter)
    res = res / np.max(res)
    res *= 255
    res = res.astype('uint8')
    res[res < 50] = 0
    res[res >= 50] = 255
    res = 255 - res

    for i in range(res.shape[0]):
        for j in range(res.shape[1]):
            if res[i][j][0] == 255:
                cv2.rectangle(img_for_detect, (j, i), (j + filter.shape[1], i + filter.shape[0]), (0, 255, 0))

    cv2.imshow("result", img_for_detect)

    #print(res)
    #print(res.dtype)
    #res = cv2.matchTemplate(img, filter, cv2.TM_CCOEFF_NORMED)
    #cv2.imshow("result", res)
    cv2.waitKey(0)


if __name__ == "__main__":
    main()