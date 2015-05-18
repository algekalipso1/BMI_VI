
# This file creates Gabor gratings for two purposes. 
# 1) Create the receptive fields of the columns in V1
# 2) Create "mixed effects" stimuli that have single, double, tripple, etc. Gabor patterns at different orientations



from PIL import Image, ImageFilter, ImageDraw
import random
import math
import sys
import os
from images2gif import writeGif
import helper_functions



simple_grating_directory = "/Users/andesgomez/Documents/Stanford/Spring2015/Psych287/project/simple_gratings/"
simple_Gabor_directory = "/Users/andesgomez/Documents/Stanford/Spring2015/Psych287/project/simple_Gabors/"



# For Sinusoidal
width = 25
height = 25
c_per_pixel = .2


# For Gaussian
amp = 1
sd = 5

center_point = (width/2., height/2.)

angle = 210 # in degrees - clockwise starting from the top - just like the orientation selectivity encoding
direction_vector = (math.sin(angle/180.*math.pi), math.cos(angle/180.*math.pi))


# This part produces the sinosoidal wave and only that.

# base = Image.new('RGB', (width, height))
# pixel_base = base.load()
# for i in range(width):
# 	for j in range(height):
# 		abs_x = i - center_point[0]
# 		abs_y = j - center_point[1]
# 		component_in_direction = abs_x*direction_vector[0] + abs_y*direction_vector[1] # dot product
# 		value = math.sin(math.pi*component_in_direction*c_per_pixel)
# 		value = int(127*(value + 1))
# 		pixel_base[i, j] = (value, value, value)

# # cpp -> cycles per pixel
# base.save(simple_grating_directory + "0_8cpp_05_degree.png")


# Now include a Gaussian operator to produce a Gabor filter

base = Image.new('RGB', (width, height))
pixel_base = base.load()

for i in range(width):
	for j in range(height):
		abs_x = i - center_point[0]
		abs_y = j - center_point[1]
		component_in_direction = abs_x*direction_vector[0] + abs_y*direction_vector[1] # dot product
		value = math.sin(math.pi*component_in_direction*c_per_pixel)
		distance_to_center = (abs_x**2 + abs_y**2)**.5
		corresponding_Gaussian_fraction = helper_functions.gaussianFunction(amp, sd, distance_to_center)
		value = value*corresponding_Gaussian_fraction
		value = int(127*(value + 1))
		pixel_base[i, j] = (value, value, value)

#base.show()

base.save(simple_Gabor_directory + "25_25_210degrees_02cpp_Gabor_5sd.png")


# Just to verify the createGaborDataStructure function works properly, print one of them:

# gabor_width = 25
# gabor_height = 25
# gabor_cycles_per_pixel = .2
# gabor_amp = 1
# gabor_sd = 5
# angles = [0, 45, 90, 135, 180, 225, 270, 315]

# all_Gabors = {}
# for i in range(len(angles)):
# 	all_Gabors[i] = helper_functions.createGaborDataStructure(gabor_width, gabor_height, gabor_cycles_per_pixel, gabor_amp, gabor_sd, angles[i])

# base = Image.new('RGB', (25, 25))
# pixel_base = base.load()
# for i, j in all_Gabors[3].keys():
# 	value = int(127*(all_Gabors[3][(i, j)] + 1))
# 	pixel_base[i, j] = (value, value, value)

# base.show()

# Note: The above looks fine. I think 25x25 is about the right size for this spatial frequency.

