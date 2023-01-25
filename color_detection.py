import webcolors
import cv2
import numpy as np
def closest_colour(requested_colour):
    min_colours = {}
    for key, name in webcolors.CSS3_HEX_TO_NAMES.items():
        r_c, g_c, b_c = webcolors.hex_to_rgb(key)
        rd = (r_c - requested_colour[0]) ** 2
        gd = (g_c - requested_colour[1]) ** 2
        bd = (b_c - requested_colour[2]) ** 2
        min_colours[(rd + gd + bd)] = name
    return min_colours[min(min_colours.keys())]

def get_colour_name(requested_colour):
#     if color[0][0][0] > 230 and color[0][0][1] > 230 and color[0][0][2] > 220:
#         closest_name = actual_name = 'White'
    #else:
        
    try:
        closest_name = actual_name = webcolors.rgb_to_name(requested_colour)
    except ValueError:
        closest_name = closest_colour(requested_colour)
        actual_name = None
    return actual_name, closest_name
def color_detection(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Define the number of bins for the histogram
    bins = 32

    # Compute the color histogram
    hist = cv2.calcHist([hsv], [0, 1, 2], None, [bins, bins, bins], [0, 256, 0, 256, 0, 256])

    # Normalize the histogram
    cv2.normalize(hist, hist)

    # Get the index of the bin with the highest value
    index = np.unravel_index(np.argmax(hist, axis=None), hist.shape)

    # Get the color of the bin
    color = np.array([index[0] * 180 / bins, index[1] * 256 / bins, index[2] * 256 / bins], dtype=np.uint8)

    # Convert the color from HSV to BGR
    color = cv2.cvtColor(np.array([[color]], dtype=np.uint8), cv2.COLOR_HSV2BGR)

    # Print the dominant color of t-shirt
    print(color)
    requested_colour = color[0][0]

    requested_colour1 = {}
    requested_colour1['R'] = color[0][0][0]
    requested_colour1['G'] = color[0][0][1]
    requested_colour1['B'] = color[0][0][2]


    actual_name, closest_name = get_colour_name(requested_colour)

    return requested_colour1,closest_name