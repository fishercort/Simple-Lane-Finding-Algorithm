
# Self-Driving Car Engineer Nanodegree


## Project: **Finding Lane Lines on the Road** 
***
In this project, you will use the tools you learned about in the lesson to identify lane lines on the road.  You can develop your pipeline on a series of individual images, and later apply the result to a video stream (really just a series of images). Check out the video clip "raw-lines-example.mp4" (also contained in this repository) to see what the output should look like after using the helper functions below. 

Once you have a result that looks roughly like "raw-lines-example.mp4", you'll need to get creative and try to average and/or extrapolate the line segments you've detected to map out the full extent of the lane lines.  You can see an example of the result you're going for in the video "P1_example.mp4".  Ultimately, you would like to draw just one line for the left side of the lane, and one for the right.

In addition to implementing code, there is a brief writeup to complete. The writeup should be completed in a separate file, which can be either a markdown file or a pdf document. There is a [write up template](https://github.com/udacity/CarND-LaneLines-P1/blob/master/writeup_template.md) that can be used to guide the writing process. Completing both the code in the Ipython notebook and the writeup template will cover all of the [rubric points](https://review.udacity.com/#!/rubrics/322/view) for this project.

---
Let's have a look at our first image called 'test_images/solidWhiteRight.jpg'.  Run the 2 cells below (hit Shift-Enter or the "play" button above) to display the image.

**Note: If, at any point, you encounter frozen display windows or other confounding issues, you can always start again with a clean slate by going to the "Kernel" menu above and selecting "Restart & Clear Output".**

---

**The tools you have are color selection, region of interest selection, grayscaling, Gaussian smoothing, Canny Edge Detection and Hough Tranform line detection.  You  are also free to explore and try other techniques that were not presented in the lesson.  Your goal is piece together a pipeline to detect the line segments in the image, then average/extrapolate them and draw them onto the image for display (as below).  Once you have a working pipeline, try it out on the video stream below.**

---

<figure>
 <img src="examples/line-segments-example.jpg" width="380" alt="Combined Image" />
 <figcaption>
 <p></p> 
 <p style="text-align: center;"> Your output should look something like this (above) after detecting line segments using the helper functions below </p> 
 </figcaption>
</figure>
 <p></p> 
<figure>
 <img src="examples/laneLines_thirdPass.jpg" width="380" alt="Combined Image" />
 <figcaption>
 <p></p> 
 <p style="text-align: center;"> Your goal is to connect/average/extrapolate line segments to get output like this</p> 
 </figcaption>
</figure>

**Run the cell below to import some packages.  If you get an `import error` for a package you've already installed, try changing your kernel (select the Kernel menu above --> Change Kernel).  Still have problems?  Try relaunching Jupyter Notebook from the terminal prompt.  Also, consult the forums for more troubleshooting tips.**  

## Import Packages


```python
#importing some useful packages
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
%matplotlib inline
```

## Read in an Image


```python
#reading in an image
image = mpimg.imread('test_images/solidWhiteRight.jpg')

#printing out some stats and plotting
print('This image is:', type(image), 'with dimensions:', image.shape)
plt.imshow(image)  # if you wanted to show a single color channel image called 'gray', for example, call as plt.imshow(gray, cmap='gray')
```

    This image is: <class 'numpy.ndarray'> with dimensions: (540, 960, 3)





    <matplotlib.image.AxesImage at 0x129dc07f0>




![png](output_6_2.png)


## Ideas for Lane Detection Pipeline

**Some OpenCV functions (beyond those introduced in the lesson) that might be useful for this project are:**

`cv2.inRange()` for color selection  
`cv2.fillPoly()` for regions selection  
`cv2.line()` to draw lines on an image given endpoints  
`cv2.addWeighted()` to coadd / overlay two images
`cv2.cvtColor()` to grayscale or change color
`cv2.imwrite()` to output images to file  
`cv2.bitwise_and()` to apply a mask to an image

**Check out the OpenCV documentation to learn about these and discover even more awesome functionality!**

## Helper Functions

Below are some helper functions to help get you started. They should look familiar from the lesson!


```python
import math

def grayscale(img):
    """Applies the Grayscale transform
    This will return an image with only one color channel
    but NOTE: to see the returned image as grayscale
    (assuming your grayscaled image is called 'gray')
    you should call plt.imshow(gray, cmap='gray')"""
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Or use BGR2GRAY if you read an image with cv2.imread()
    # return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
def canny(img, low_threshold, high_threshold):
    """Applies the Canny transform"""
    return cv2.Canny(img, low_threshold, high_threshold)

def gaussian_blur(img, kernel_size):
    """Applies a Gaussian Noise kernel"""
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

def region_of_interest(img, vertices):
    """
    Applies an image mask.
    
    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    """
    #defining a blank mask to start with
    mask = np.zeros_like(img)   
    
    #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255
        
    #filling pixels inside the polygon defined by "vertices" with the fill color    
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    
    #returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image


def draw_lines(img, lines, color=[255, 0, 0], thickness=7):
    """
    NOTE: this is the function you might want to use as a starting point once you want to 
    average/extrapolate the line segments you detect to map out the full
    extent of the lane (going from the result shown in raw-lines-example.mp4
    to that shown in P1_example.mp4).  
    
    Think about things like separating line segments by their 
    slope ((y2-y1)/(x2-x1)) to decide which segments are part of the left
    line vs. the right line.  Then, you can average the position of each of 
    the lines and extrapolate to the top and bottom of the lane.
    
    This function draws `lines` with `color` and `thickness`.    
    Lines are drawn on the image inplace (mutates the image).
    If you want to make the lines semi-transparent, think about combining
    this function with the weighted_img() function below
    """
    """
    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(img, (x1, y1), (x2, y2), color, thickness)"""
    
    right_lines = []
    left_lines = []
    
    #loop through all of the lines to sort them out if theyre in the left or right lane
    for line in lines:
        for x1, y1, x2, y2 in line:
            x1, y1, x2, y2 = line[0]  # line = [[x1, y1, x2, y2]]
            
            # Calculate slope
            slope = (y2 - y1) / (x2 - x1)
            
            # Filter lines based on slope, only take slopes over +- 0.5
            if (abs(slope) > 0.5):
                #Right lane - Remember that since the top left corner is (0,0) slope is backwards
                if (slope > 0):
                    right_lines.append(line)
                #Left Lane
                elif (slope < 0):
                    left_lines.append(line)  
    
    
    # Run linear regression to find best fit line for right and left lane lines
    # Right lane lines
    right_lines_x = []
    right_lines_y = []
    
    #Loop through and put all x-values in one vector, and all y-values in another
    for line in right_lines:
        x1, y1, x2, y2 = line[0]
        
        right_lines_x.append(x1)
        right_lines_x.append(x2)
        
        right_lines_y.append(y1)
        right_lines_y.append(y2)
        
    
    right_slope, right_intercept = np.polyfit(right_lines_x, right_lines_y, 1)  # y = m*x + b
    
        
    # Left lane lines
    left_lines_x = []
    left_lines_y = []
    
    #Loop through and put all x-values in one vector, and all y-values in another
    for line in left_lines:
        x1, y1, x2, y2 = line[0]
        
        left_lines_x.append(x1)
        left_lines_x.append(x2)
        
        left_lines_y.append(y1)
        left_lines_y.append(y2)
        

    left_slope, left_intercept = np.polyfit(left_lines_x, left_lines_y, 1)  # y = m*x + b
  
    
    
    # Find end points for right and left lines in order to extrapolate the lines
    #40% of the image
    
    height_thresh = .4
    y1 = img.shape[0]
    y2 = img.shape[0] * (1 - height_thresh)
    
    #Calculate x-values of right line
    right_x1 = (y1 - right_intercept) / right_slope # x = (y-b)/m
    right_x2 = (y2 - right_intercept) / right_slope # x = (y-b)/m
    
    #Calculate x-values of left line
    left_x1 = (y1 - left_intercept) / left_slope # x = (y-b)/m
    left_x2 = (y2 - left_intercept) / left_slope # x = (y-b)/m
        
    # Convert into int because it expects pixels as integers
    y1 = int(y1)
    y2 = int(y2)
    right_x1 = int(right_x1)
    right_x2 = int(right_x2)
    left_x1 = int(left_x1)
    left_x2 = int(left_x2)

    # Draw both lanes on the image
    cv2.line(img, (right_x1, y1), (right_x2, y2), color, thickness)
    cv2.line(img, (left_x1, y1), (left_x2, y2), color, thickness)

def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    """
    `img` should be the output of a Canny transform.
        
    Returns an image with hough lines drawn.
    """
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    draw_lines(line_img, lines)
    return line_img

# Python 3 has support for cool math symbols.

def weighted_img(img, initial_img, α=0.8, β=1., λ=0.):
    """
    `img` is the output of the hough_lines(), An image with lines drawn on it.
    Should be a blank image (all black) with lines drawn on it.
    
    `initial_img` should be the image before any processing.
    
    The result image is computed as follows:
    
    initial_img * α + img * β + λ
    NOTE: initial_img and img must be the same shape!
    """
    return cv2.addWeighted(initial_img, α, img, β, λ)
```

## Test Images

Build your pipeline to work on the images in the directory "test_images"  
**You should make sure your pipeline works well on these images before you try the videos.**


```python
import os
os.listdir("test_images/")
```




    ['solidWhiteCurve.jpg',
     'solidWhiteRight.jpg',
     'solidYellowCurve.jpg',
     'solidYellowCurve2.jpg',
     'solidYellowLeft.jpg',
     'whiteCarLaneSwitch.jpg']



## Build a Lane Finding Pipeline



Build the pipeline and run your solution on all test_images. Make copies into the `test_images_output` directory, and you can use the images in your writeup report.

Try tuning the various parameters, especially the low and high Canny thresholds as well as the Hough lines parameters.


```python
from PIL import Image
import glob
image_list = []
for filename in glob.glob('test_images/*.jpg'): #get all jpg images in the file
    im=cv2.imread(filename)
    image_list.append(im)
    
count = 1
for image1 in image_list:
    # TODO: Build your pipeline that will draw lane lines on the test_images
    # then save them to the test_images_output directory.
    image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
    
    plt.figure()
    #plt.imshow(image1)
    gray = grayscale(image1)
    #plt.imshow(gray, cmap='gray')

    # Define a kernel size and apply Gaussian smoothing
    blur_gray = gaussian_blur(gray, 5)

    # Define our parameters for Canny and apply
    low_threshold = 50
    high_threshold = 150
    edges = canny(blur_gray, low_threshold, high_threshold)

    # This time we are defining a four sided polygon to mask
    vertices = np.array([[(120,image.shape[0]),(460, 320), (510, 320), (950,image.shape[0])]], dtype=np.int32)
    masked_image = region_of_interest(edges, vertices)


    # Define the Hough transform parameters
    # Make a blank the same size as our image to draw on
    rho = 2 # distance resolution in pixels of the Hough grid
    theta = np.pi/180 # angular resolution in radians of the Hough grid
    threshold = 20     # minimum number of votes (intersections in Hough grid cell)
    min_line_length = 10 #minimum number of pixels making up a line
    max_line_gap = 8    # maximum gap in pixels between connectable line segments

    line_img = hough_lines(masked_image, rho, theta, threshold, min_line_length, max_line_gap)

    lines_edges = weighted_img(line_img, image1, α=0.8, β=1., λ=0.)

    
    
    
    
    # Draw the lines on the edge image
    plt.imshow(lines_edges) 
    
    # Write images to output file
    lines_edges = cv2.cvtColor(lines_edges, cv2.COLOR_BGR2RGB)
    cv2.imwrite('test_images_output/image1_' + str(count) + '.jpg', lines_edges)
    
    count = count + 1
    


```


![png](output_16_0.png)



![png](output_16_1.png)



![png](output_16_2.png)



![png](output_16_3.png)



![png](output_16_4.png)



![png](output_16_5.png)


## Test on Videos

You know what's cooler than drawing lanes over images? Drawing lanes over video!

We can test our solution on two provided videos:

`solidWhiteRight.mp4`

`solidYellowLeft.mp4`

**Note: if you get an import error when you run the next cell, try changing your kernel (select the Kernel menu above --> Change Kernel). Still have problems? Try relaunching Jupyter Notebook from the terminal prompt. Also, consult the forums for more troubleshooting tips.**

**If you get an error that looks like this:**
```
NeedDownloadError: Need ffmpeg exe. 
You can download it by calling: 
imageio.plugins.ffmpeg.download()
```
**Follow the instructions in the error message and check out [this forum post](https://discussions.udacity.com/t/project-error-of-test-on-videos/274082) for more troubleshooting tips across operating systems.**


```python
# Import everything needed to edit/save/watch video clips
from moviepy.editor import VideoFileClip
from IPython.display import HTML
```


```python
def process_image(image):
    # NOTE: The output you return should be a color image (3 channel) for processing video below
    # TODO: put your pipeline here,
    # you should return the final output (image where lines are drawn on lanes)
    # then save them to the test_images_output directory.
    
    #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    gray = grayscale(image)
    #plt.imshow(gray, cmap='gray')

    # Define a kernel size and apply Gaussian smoothing
    blur_gray = gaussian_blur(gray, 5)

    # Define our parameters for Canny and apply
    low_threshold = 50
    high_threshold = 150
    edges = canny(blur_gray, low_threshold, high_threshold)

    # This time we are defining a four sided polygon to mask
    vertices = np.array([[(120,image.shape[0]),(460, 320), (510, 320), (965,image.shape[0])]], dtype=np.int32)
    masked_image = region_of_interest(edges, vertices)


    # Define the Hough transform parameters
    # Make a blank the same size as our image to draw on
    rho = 2 # distance resolution in pixels of the Hough grid
    theta = np.pi/180 # angular resolution in radians of the Hough grid
    threshold = 20     # minimum number of votes (intersections in Hough grid cell)
    min_line_length = 10 #minimum number of pixels making up a line
    max_line_gap = 8    # maximum gap in pixels between connectable line segments

    line_img = hough_lines(masked_image, rho, theta, threshold, min_line_length, max_line_gap)

    
    lines_edges = weighted_img(line_img, image, α=0.8, β=1., λ=0.)

    # Create a "color" binary image to combine with line image
    #color_edges = np.dstack((edges, edges, edges)) 

    # Draw the lines on the edge image
   
    #plt.imshow(lines_edges) 
    
    
    
    return lines_edges
```

Let's try the one with the solid white lane on the right first ...


```python
white_output = 'test_videos_output/solidWhiteRight.mp4'
## To speed up the testing process you may want to try your pipeline on a shorter subclip of the video
## To do so add .subclip(start_second,end_second) to the end of the line below
## Where start_second and end_second are integer values representing the start and end of the subclip
## You may also uncomment the following line for a subclip of the first 5 seconds
##clip1 = VideoFileClip("test_videos/solidWhiteRight.mp4").subclip(0,5)
clip1 = VideoFileClip("test_videos/solidWhiteRight.mp4")
white_clip = clip1.fl_image(process_image) #NOTE: this function expects color images!!
%time white_clip.write_videofile(white_output, audio=False)
```

    [MoviePy] >>>> Building video test_videos_output/solidWhiteRight.mp4
    [MoviePy] Writing video test_videos_output/solidWhiteRight.mp4


    
      0%|          | 0/222 [00:00<?, ?it/s][A
      4%|▍         | 9/222 [00:00<00:02, 82.85it/s][A
      9%|▊         | 19/222 [00:00<00:02, 85.78it/s][A
     13%|█▎        | 29/222 [00:00<00:02, 87.13it/s][A
     18%|█▊        | 39/222 [00:00<00:02, 89.20it/s][A
     21%|██        | 47/222 [00:00<00:02, 78.18it/s][A
     24%|██▍       | 54/222 [00:00<00:02, 71.55it/s][A
     27%|██▋       | 61/222 [00:00<00:02, 64.26it/s][A
     31%|███       | 68/222 [00:00<00:02, 58.85it/s][A
     33%|███▎      | 74/222 [00:01<00:02, 58.33it/s][A
     36%|███▋      | 81/222 [00:01<00:02, 61.36it/s][A
     40%|███▉      | 88/222 [00:01<00:02, 61.34it/s][A
     43%|████▎     | 95/222 [00:01<00:02, 57.96it/s][A
     45%|████▌     | 101/222 [00:01<00:02, 55.66it/s][A
     48%|████▊     | 107/222 [00:01<00:02, 53.08it/s][A
     51%|█████     | 113/222 [00:01<00:02, 51.89it/s][A
     54%|█████▎    | 119/222 [00:01<00:02, 50.95it/s][A
     56%|█████▋    | 125/222 [00:02<00:01, 52.18it/s][A
     59%|█████▉    | 132/222 [00:02<00:01, 54.44it/s][A
     62%|██████▏   | 138/222 [00:02<00:01, 54.62it/s][A
     65%|██████▍   | 144/222 [00:02<00:01, 54.20it/s][A
     68%|██████▊   | 150/222 [00:02<00:01, 54.57it/s][A
     70%|███████   | 156/222 [00:02<00:01, 52.86it/s][A
     73%|███████▎  | 162/222 [00:02<00:01, 53.45it/s][A
     76%|███████▌  | 168/222 [00:02<00:01, 51.17it/s][A
     78%|███████▊  | 174/222 [00:02<00:00, 53.35it/s][A
     82%|████████▏ | 181/222 [00:03<00:00, 56.18it/s][A
     84%|████████▍ | 187/222 [00:03<00:00, 54.02it/s][A
     87%|████████▋ | 193/222 [00:03<00:00, 55.30it/s][A
     90%|████████▉ | 199/222 [00:03<00:00, 54.99it/s][A
     92%|█████████▏| 205/222 [00:03<00:00, 56.36it/s][A
     95%|█████████▌| 211/222 [00:03<00:00, 54.36it/s][A
     98%|█████████▊| 217/222 [00:03<00:00, 53.61it/s][A
    100%|█████████▉| 221/222 [00:03<00:00, 58.41it/s][A

    [MoviePy] Done.
    [MoviePy] >>>> Video ready: test_videos_output/solidWhiteRight.mp4 
    
    CPU times: user 2.82 s, sys: 1.02 s, total: 3.85 s
    Wall time: 4.33 s


Play the video inline, or if you prefer find the video in your filesystem (should be in the same directory) and play it in your video player of choice.


```python
HTML("""
<video width="960" height="540" controls>
  <source src="{0}">
</video>
""".format(white_output))
```





<video width="960" height="540" controls>
  <source src="test_videos_output/solidWhiteRight.mp4">
</video>




## Improve the draw_lines() function

**At this point, if you were successful with making the pipeline and tuning parameters, you probably have the Hough line segments drawn onto the road, but what about identifying the full extent of the lane and marking it clearly as in the example video (P1_example.mp4)?  Think about defining a line to run the full length of the visible lane based on the line segments you identified with the Hough Transform. As mentioned previously, try to average and/or extrapolate the line segments you've detected to map out the full extent of the lane lines. You can see an example of the result you're going for in the video "P1_example.mp4".**

**Go back and modify your draw_lines function accordingly and try re-running your pipeline. The new output should draw a single, solid line over the left lane line and a single, solid line over the right lane line. The lines should start from the bottom of the image and extend out to the top of the region of interest.**

Now for the one with the solid yellow lane on the left. This one's more tricky!


```python
yellow_output = 'test_videos_output/solidYellowLeft.mp4'
## To speed up the testing process you may want to try your pipeline on a shorter subclip of the video
## To do so add .subclip(start_second,end_second) to the end of the line below
## Where start_second and end_second are integer values representing the start and end of the subclip
## You may also uncomment the following line for a subclip of the first 5 seconds
##clip2 = VideoFileClip('test_videos/solidYellowLeft.mp4').subclip(0,5)
clip2 = VideoFileClip('test_videos/solidYellowLeft.mp4')
yellow_clip = clip2.fl_image(process_image)
%time yellow_clip.write_videofile(yellow_output, audio=False)
```

    [MoviePy] >>>> Building video test_videos_output/solidYellowLeft.mp4
    [MoviePy] Writing video test_videos_output/solidYellowLeft.mp4


    100%|█████████▉| 681/682 [00:09<00:00, 68.90it/s]


    [MoviePy] Done.
    [MoviePy] >>>> Video ready: test_videos_output/solidYellowLeft.mp4 
    
    CPU times: user 8.1 s, sys: 3.08 s, total: 11.2 s
    Wall time: 10.3 s



```python
HTML("""
<video width="960" height="540" controls>
  <source src="{0}">
</video>
""".format(yellow_output))
```





<video width="960" height="540" controls>
  <source src="test_videos_output/solidYellowLeft.mp4">
</video>




## Writeup and Submission

If you're satisfied with your video outputs, it's time to make the report writeup in a pdf or markdown file. Once you have this Ipython notebook ready along with the writeup, it's time to submit for review! Here is a [link](https://github.com/udacity/CarND-LaneLines-P1/blob/master/writeup_template.md) to the writeup template file.

