# Simple-Lane-Finding-Algorithm
---
**Project: Detect and draw extrapolated lane lines on a video stream**

[//]: # (Image References)

[image1]: ./examples/grayscale.jpg "Grayscale"

---

### Reflection

### 1. Describe your pipeline. As part of the description, explain how you modified the draw_lines() function.

My pipeline consisted of 5 steps. The first step was to conver the image to grayscale. Next I applied Gaussian Smoothing to the image using openCV. I then applied a Canny function to the image in order to find the gradients of the image. After this, I masked the image, blacking out any parts of the image that are not in the defined region of interest. We then apply weighting to the image.

In order to draw a single line on the left and right lanes, I modified the draw_lines() function by first looping through all of the lines drawn and sorting them into the left and the right line. I did this by first taking out any lines that have a slope with absolute value less than 0.5. I then checked if the slope was positive or negative, putting them into the respective vector left_line or right_line according to their slope. From here I took only the right lines and put all of the x values in their own vector and all of the y values in their own vector. I then ran a linear regression on the vectors to get the slope and intercept of the right line. I repeated this same process for the left line. From here I calculated the x and y endpoints of each line and drew them on the image. 



### 2. Identify potential shortcomings with your current pipeline


One potential shortcoming would be if the lanes are very curved, the slope of the lines may be less than the predefined threshold. This would result in these lines not being captured in the regression. The line would most likely not be a good representation of the lane. Also if the lanes are very curved, this linear regression model would not be a good option to use in the first place. 

Another limitation of this pipeline is that the masked region has to be changed depending on the video stream. The camera could be mounted at a different part on the car and it could changed the region that should be masked. 


### 3. Suggest possible improvements to your pipeline

A possible improvement would be to make the lines drawn be actual polynomial equations. This would make it so they could actual follow the contour of the lanes. 

Another potential improvement could be to identify the entire lane as a region as opposed to just identifying the left and right line. 
