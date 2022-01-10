import cv2

# đọc ảnh foreground
fg = cv2.imread('TS.png')
# print('Kich thuoc theo tung kenh cua foreground: ', fg.shape)

# đọc ảnh effect
eff = cv2.imread('beach.png')
eff = cv2.resize(eff, (639, 360))
# print('Kich thuoc theo tung kenh cua eff: ', eff.shape)

# đọc ảnh mask
mask = cv2.imread('TaylorSwift.png', cv2.IMREAD_UNCHANGED)
# print('Kich thuoc theo tung kenh cua mask: ', mask.shape)

# Hiện thị hai ảnh lên màn hình
# cv2.imshow('Foreground', fg)
# cv2.imshow('Background', eff)
# cv2.imshow('Mask', mask)
# cv2.waitKey(0)

#Sao chép ảnh qua biến mới
result = fg.copy()
alpha = 0.4
result[mask[:,:,3] != 0] = fg[mask[:,:,3] != 0] * alpha + eff[mask[:,:,3] != 0] * (1 - alpha)

cv2.imshow('Result', result)
# cv2.imwrite('result.png', result)
cv2.waitKey(0)