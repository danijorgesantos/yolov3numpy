
import cv2
import numpy as np

pc = 1
bx = 150
by = 150
bh = 550
bw = 370
c1 = 0  # pedestrian
c2 = 1  # car
c3 = 0  # traffic light


y = [pc, bx, by, bh, bw, c1, c2, c3]

print(y)





































































# Python program to explain cv2.rectangle() method  
   
# Reading an image in default mode 
image = cv2.imread('mer.png') 
   
# Window name in which image is displayed 
window_name = 'Image'
  
# Start coordinate, here (5, 5) 
# represents the top left corner of rectangle 
start_point = (bx, by) 
  
# Ending coordinate, here (220, 220) 
# represents the bottom right corner of rectangle 
end_point = (bh, bw) 
  
# Blue color in BGR 
color = (255, 0, 0) 
  
# Line thickness of 2 px 
thickness = 2
  
# Using cv2.rectangle() method 
# Draw a rectangle with blue line borders of thickness of 2 px 
image = cv2.rectangle(image, start_point, end_point, color, thickness) 
  
# Displaying the image  
cv2.imshow(window_name, image)  
cv2.waitKey(0)
cv2.destroyAllWindows()


