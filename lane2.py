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

def draw_the_lines(img, lines, height):
    img = np.copy(img)
    blank_image = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)

    for line in lines:
        for x1, y1, x2, y2 in line:

            angle = int(np.arctan((y2-y1)/(x2-x1))*(180/3.14))

            if y1<height*0.684 :
                x = ((height*0.684 - y1)/np.tan(angle*(np.pi/180)))
                if x<0:
                    x = -1*x
                x1 = int(x1 + x)
                y1 = int(height*0.684)

            if y2<height*0.684 :
                x = ((height*0.684 - y2)/np.tan(angle*(np.pi/180)))
                if x<0:
                    x = -1*x
                x2 = int(x2 - x)
                y2 = int(height*0.684)

            if angle < 0:
                if not -85<angle<-35:
                    continue 
            else:
                if not 35<angle<85:
                    continue

            cv2.line(blank_image, (x1,y1), (x2,y2), (0, 255, 0), thickness=2)

    img = cv2.addWeighted(img, 0.8, blank_image, 1, 0.0)
    return img

def auto_canny(image, sigma=0.33):
	# compute the median of the single channel pixel intensities
	v = np.median(image)

	# apply automatic Canny edge detection using the computed median
	lower = int(max(0, (1.0 - sigma) * v))
	upper = int(min(255, (1.0 + sigma) * v))
	edged = cv2.Canny(image, lower, upper)

	# return the edged image
	return edged

#num = input("Enter: ")
num = 3
#image = cv2.imread('lane-test/t3.png')
image = cv2.imread('lane-test/t'+str(num)+'.png')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
print(image.shape)
height = image.shape[0]
width = image.shape[1]

#xtl, ytl = 242, 768   #0.177, 1
#xtr, ytr = 1124, 768  #0.823 , 1
#xbl, ybl = 733, 460   #0.536, 0.598
#xbr, ybr = 633, 460   #0.463, 0.598

xtl, ytl = width*0.177, height
xtr, ytr = width*0.823, height
xbl, ybl = width*0.536, height*0.598
xbr, ybr = width*0.463, height*0.598


region_of_interest_vertices = [
	(xtl, ytl),
	(xtr, ytr),(xbl,ybl)
	,(xbr,ybr)
]

gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
canny_image = auto_canny(gray_image)
cropped_image = region_of_interest(canny_image,
                np.array([region_of_interest_vertices], np.int32),)
lines = cv2.HoughLinesP(cropped_image,
                        rho=1,
                        theta=np.pi/180,
                        threshold=80,
                        lines=np.array([]),
                        minLineLength=4,
                        maxLineGap=100)

image_with_lines = draw_the_lines(image, lines, height)

#x1, y1, x2, y2 = lines[0][0][0], lines[0][0][1], lines[0][0][2], lines[0][0][3]
#print(x1, y1, x2, y2)
#angle = int(np.arctan((y2-y1)/(x2-x1))*(180/3.14))
#print(angle)
#plt.imshow(canny_image)
#plt.show()
plt.imshow(image_with_lines)
plt.show()
