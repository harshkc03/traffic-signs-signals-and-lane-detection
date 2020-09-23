import matplotlib.pylab as plt
import cv2
import numpy as np

def region_of_interest(img, vertices):
	mask = np.zeros_like(img)
	#channel_count = img.shape[2]
	match_mask_color = 255
	cv2.fillPoly(mask, vertices, match_mask_color)
	masked_image = cv2.bitwise_and(img, mask)
	return masked_image

def drow_the_lines(img, lines):
	img = np.copy(img)
	blank_image = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)

	for line in lines:
		for x1, y1, x2, y2 in line:
			angle = int(np.arctan((y2-y1)/(x2-x1))*(180/3.14))
			if angle < 0:
				if not -85<angle<-35:
					continue 
			else:
				if not 35<angle<85:
					continue
			cv2.line(blank_image, (x1,y1), (x2,y2), (0, 255, 0), thickness=10)

	img = cv2.addWeighted(img, 0.8, blank_image, 1, 0.0)
	return img

image = cv2.imread("data/signs5.png") 

print(image.shape)
height = image.shape[0]
width = image.shape[1]

xtl, ytl = width*0.454, height*0.625
xtr, ytr = width*0.538, height*0.625
xbl, ybl = width*0.219, height*0.996
xbr, ybr = width*0.805, height*0.996

region_of_interest_vertices = [
    (xtl, ytl), (xtr, ytr),
    (xbr, ybr), (xbl, ybl)
]

crop1 = region_of_interest(image,
            np.array([region_of_interest_vertices], np.int32),)

gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
canny_image = cv2.Canny(gray_image, 100,300)
cropped_image = region_of_interest(canny_image,
            np.array([region_of_interest_vertices], np.int32),)
lines = cv2.HoughLinesP(cropped_image,
                    rho=1,
                    theta=np.pi/180,
                    threshold=25,
                    lines=np.array([]),
                    minLineLength=3,
                    maxLineGap=2000)
image_with_lines = drow_the_lines(image, lines)

plt.imshow(image_with_lines)
plt.show()