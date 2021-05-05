#importing some useful packages
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import math
import os

# Import everything needed to edit/save/watch video clips
from moviepy.editor import VideoFileClip
from IPython.display import HTML

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
    `vertices` should be a numpy array of integer points.
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


def draw_lines(img, lines, color=[255, 0, 0], thickness=6):
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

    # Lines segments belongs to Left lane
    lines_l = []
    # Lines segments belongs to Right lane
    lines_r = []
    
    # Camer X point
    imshape = img.shape
    camera_x = imshape[1] / 2
    
    for line in lines:
        for x1,y1,x2,y2 in line:
            slope = (y2 - y1)/ (x2 - x1)

            # calucalte the angle of each slope
            slope_angle = np.arctan(slope) * (180/np.pi)

            # determine left lane (y1, y2) line offset points
            if (-10 > slope_angle > -85) and (x1 < camera_x) and (x2 < camera_x):
                lines_l.append(line)

            # determine right lane (y1, y2) line offset points
            elif (10 < slope_angle < 85) and (x2 > camera_x) and (x2 > camera_x):
                lines_r.append(line)

    # lane starting, ending Y-cordinate points
    start_Y = 325
    end_Y = 600
    
    if len(lines_l) != 0:
        # Slope, intercept of left lane
        [slope_l, intercept_l] = calculate_avg_line_slope_intecept(lines_l)

        # line equation, y = mx + b   and   x = (y - b) / m
        start_X_left = int((start_Y - intercept_l) / slope_l)
        end_X_left = int((end_Y - intercept_l) / slope_l)

        # left lane
        cv2.line(img, (start_X_left, start_Y), (end_X_left, end_Y), color, thickness)

    if len(lines_r) != 0:
        # Slope, intercept of right lane
        [slope_r, intercept_r] = calculate_avg_line_slope_intecept(lines_r)    

        start_X_right = int((start_Y - intercept_r) / slope_r)
        end_X_right = int((end_Y - intercept_r) / slope_r)

        #right lane
        cv2.line(img, (start_X_right, start_Y), (end_X_right, end_Y), color, thickness)

def calculate_avg_line_slope_intecept(lines):
    """
    Calculate slope and intercept of line

    Get average line points amang set of lines, 
    then calculate the slope and intecept of this line.
    """
    n = 1
    x1Avg = 0
    x2Avg = 0
    y1Avg = 0
    y2Avg = 0
    slope = 0
    intercept = 0

    for line in lines :
        for x1, y1, x2, y2 in line:
            x1Avg = x1Avg + (x1 - x1Avg)/n
            x2Avg = x2Avg + (x2 - x2Avg)/n
            y1Avg = y1Avg + (y1 - y1Avg)/n
            y2Avg = y2Avg + (y2 - y2Avg)/n
            n += 1
    
    if (x1Avg== 0 and y1Avg==0 and x2Avg==0 and y2Avg==0):
        raise Exception ("average calucalation is not correct")
    
    [slope, intercept] = np.polyfit([x1Avg, x2Avg], [y1Avg, y2Avg], 1)
        
    return [slope, intercept]


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

def weighted_img(img, initial_img, α=0.8, β=1., γ=0.):
    """
    `img` is the output of the hough_lines(), An image with lines drawn on it.
    Should be a blank image (all black) with lines drawn on it.
    
    `initial_img` should be the image before any processing.
    
    The result image is computed as follows:
    
    initial_img * α + img * β + γ
    NOTE: initial_img and img must be the same shape!
    """
    return cv2.addWeighted(initial_img, α, img, β, γ)


def process_image(image):
    # NOTE: The output you return should be a color image (3 channel) for processing video below
    # TODO: put your pipeline here,
    # you should return the final output (image where lines are drawn on lanes)

    # Show Gray scale image
    gray_img = grayscale(image)

    # Generate a blur gray image
    kernel_size = 3
    blur_img= gaussian_blur(gray_img, kernel_size)

    # Generate Canny image edges
    low_threshold = 50
    high_threshold = 150
    edges = canny(blur_img, low_threshold, high_threshold)

    # Defining area of intrest ( four sided polygon to mask)
    imshape = image.shape
    vertices = np.array([[(0,imshape[0]),(450, 325), (500, 325), (imshape[1],imshape[0])]], dtype=np.int32)
    masked_edges = region_of_interest(edges, vertices)

    # Define the Hough transform parameters
    # Make a blank the same size as our image to draw on
    rho = 2 # distance resolution in pixels of the Hough grid
    theta = np.pi/180 # angular resolution in radians of the Hough grid (1 degree)
    threshold = 15     # minimum number of votes (intersections in Hough grid cell)
    min_line_length = 40 #minimum number of pixels making up a line
    max_line_gap = 20    # maximum gap in pixels between connectable line segments
 
    # Draw image with hough lines drawn 
    line_img = hough_lines(masked_edges, rho, theta, threshold, min_line_length, max_line_gap)

    # Draw the lines on the edge image
    result = weighted_img(line_img, image)

    return result


def main():

    print("NanoDegree 1st Project: Finding Lane Lines on the Road")

    # list and load test images
    test_images = os.listdir("test_images/")

    for img_file in test_images:
        # reading in an image
        img = mpimg.imread('test_images/' + img_file)

        lined_img = process_image(img)

        plt.imshow(lined_img)
        # plt.show()

        # output file name
        output_file_name = img_file[:len(img_file) - 4] + '-lines.jpg'

        plt.imsave('test_outputs/' + output_file_name, lined_img)
        

    white_output = 'test_videos_output/solidWhiteRight.mp4'
    ## To speed up the testing process you may want to try your pipeline on a shorter subclip of the video
    ## To do so add .subclip(start_second,end_second) to the end of the line below
    ## Where start_second and end_second are integer values representing the start and end of the subclip
    ## You may also uncomment the following line for a subclip of the first 5 seconds
    ##clip1 = VideoFileClip("test_videos/solidWhiteRight.mp4").subclip(0,5)
    clip1 = VideoFileClip("test_videos/solidWhiteRight.mp4")
    white_clip = clip1.fl_image(process_image) #NOTE: this function expects color images!!
    # %time 
    white_clip.write_videofile(white_output, audio=False)


    yellow_output = 'test_videos_output/solidYellowLeft.mp4'
    ## To speed up the testing process you may want to try your pipeline on a shorter subclip of the video
    ## To do so add .subclip(start_second,end_second) to the end of the line below
    ## Where start_second and end_second are integer values representing the start and end of the subclip
    ## You may also uncomment the following line for a subclip of the first 5 seconds
    ##clip2 = VideoFileClip('test_videos/solidYellowLeft.mp4').subclip(0,5)
    clip2 = VideoFileClip('test_videos/solidYellowLeft.mp4')
    yellow_clip = clip2.fl_image(process_image)
    #%time 
    yellow_clip.write_videofile(yellow_output, audio=False)
    
if __name__ == "__main__":
    main()