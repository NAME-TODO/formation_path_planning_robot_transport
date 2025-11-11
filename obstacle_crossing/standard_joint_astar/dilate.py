import cv2

map_folder = 'maps/'
dilated_map_folder = 'maps_dilated/'

map_name = 'map_large.png'
kernel_size = (2*5, 2*5) # 2 * dilation radius

img = cv2.imread(map_folder + map_name, cv2.IMREAD_GRAYSCALE)
img = cv2.bitwise_not(img)
_, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, kernel_size)
dilated = cv2.bitwise_not(cv2.dilate(binary, kernel, iterations=1))
cv2.imwrite(dilated_map_folder + map_name, dilated + img)