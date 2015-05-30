# This is to get familiar with the constraints that will apply to the project.
# V1 is simulated -> 2X2 input/no input and stimulation/no stimulation.

from PIL import Image, ImageFilter, ImageDraw
import random
import math
import sys
import os
from images2gif import writeGif
import helper_functions_for_toy


gifs_directory = "/Users/andesgomez/Documents/Stanford/Spring2015/Psych287/project/gifs/"
image_directory = "/Users/andesgomez/Documents/Stanford/Spring2015/Psych287/project/source_pictures/"


name_of_gif = "gaussian_kenerl_21_" # "input_remains_constant_0"


# The broad idea and constraints:
# An input image is presented. The activation of pixels in the input image is pushed
# into a neural network that uses a columnar architecture and:
# (1) Has Gabor receptive fields for each column
# (2) Exhibits lateral inhibition
# (3) Has a pinwheel arrangement/distribution of orientation selectivity (also known as whorls)

# The simulation must be able to replicate the findings in "THE SENSATIONS PRODUCED 
# BY ELECTRICAL STIMULATION OF THE VISUAL CORTEX" by G. S. BRINDLEY AND W. S. LEWIN
# Specifically:
# 1.- One electrode creates a single small spot of white light ("But for some electrodes it is two or several such spots, or a small cloud").
# 2.- Retinotopic position.
# 3.- With a stronger stimuli (electrical) more phosephenes arise
# 4.- "The phosphenes produced by stimulation through electrodes 2-4 mm apart can be easily distinguished."
# 5.- No observed "flicker fussion frequency"
# 6.- "During voluntary eye movements, the phosphenes move with the eyes. During vestibular reflex eye movements they remain fixed in space."
# 7.- " Phosphenes ordinarily cease immediately when stimulation ceases, but after strong stimulation they sometimes persist for up to 2 min."


# Plan of attack:
# A - Create a datastructure for V1, perhaps for the retinal ganglion cells and intermediate stage (LGN)
# B - Wire up V1 so that units have Gabor receptive fields, and a pinwheel orientation selectivity distribution
# C - Determine how to visualize the output (can you visualize V1 directly?)
# D - Assess the outcomes of stimulation and compare them to the set of constraints shown above.


# Creating a datastructure for V1 - Challenge, it has to be a hexagonal grid
# I will simply use a (row, colum) address for each of the cortical columns
# A detail: The odd rows are shifterd d/2 to the right, and have 1 less columns

rows = 20
columns = 20
default_potential = 0.25
diameter = 10
input_values = 0.0

# Network operation/update parameters
lmbd = 0.1 # this is the lambda parameter for the imput updating
epochs = 10
alpha = 0.02 # .31
beta = .02 # 0.002
t = 1.0
amp = 15
sd  = 10
every_how_many_epochs = 2

# This should actually be an image with edges and stuff
#image_to_use = "wavy_cross"
image_to_use = "0_1cycles_per_pixel_1_0_direction.png"
#image_to_use = "0_037cycles_per_pixel_29_71_direction.png"

input_image = Image.open(image_directory + image_to_use)
input_pixels = input_image.load()


# Parameter description string

name_of_image_without_extension = image_to_use.split('.')[0]

parameter_description = "r" + str(rows) + "c" + str(columns) + "diameter" + str(diameter) + "a" + str(alpha) + "b" + str(beta) + "amp" + str(amp) + "sd" + str(sd) + "image_" + name_of_image_without_extension
name_of_gif += parameter_description


# The datastructure that keeps track of the electric potential (and possibly other variables) in each of the columns
# Orientations are tracked in the second value of the list for the v1_values dic
# v1_input_values tracks the input to the cortical column. These values will be produced with 
# a set of linear filters (Gabor functions) over an imput image. 


v1_values = {}
v1_input_values = {}
for i in range(rows):
	for j in range(columns):
		if (i % 2 == 0 or j < columns - 1):
			v1_values[(i, j)] = [default_potential + random.random()/4., (i + j) % 8]
			v1_input_values[(i, j)] = input_values

available_coordinates = set(v1_values.keys())


# Maybe also make a data structure to keep the visualization values for v1 hypercolimns. 
# The reason is that the values in v1_values are scaled by the softmax function, and
# so the values go from 0 to something that depends on alpha and beta. More funcitonal
# might be a log transformation...


# Now create the Gabor data structures that will be used as filters
# These are to be used on the getInputsFromImage function because these Gabor filters
# provide the receptive fields for the various hypercolumns.

gabor_width = 25
gabor_height = 25
gabor_cycles_per_pixel = .2
gabor_amp = 1
gabor_sd = 5
angles = [0, 45, 90, 135, 180, 225, 270, 315]

all_Gabors = {}
for i in range(len(angles)):
	all_Gabors[i] = helper_functions.createGaborDataStructure(gabor_width, gabor_height, gabor_cycles_per_pixel, gabor_amp, gabor_sd, angles[i])



# The datastructure the keeps track of which columns are connected to each other
v1_ccs_connections = {}
for i in range(rows):
	for j in range(columns):
		if (i % 2 == 0 or j < columns - 1):
			if (i % 2 == 0):
				these_connections = set([(i, j + 1), (i, j - 1), (i - 1, j), (i - 1, j - 1), (i + 1, j), (i + 1, j - 1)])
				these_connections = these_connections.intersection(available_coordinates)
				v1_ccs_connections[(i, j)] = these_connections
			else:
				these_connections = set([(i, j + 1), (i, j - 1), (i - 1, j), (i - 1, j + 1), (i + 1, j), (i + 1, j + 1)])
				these_connections = these_connections.intersection(available_coordinates)
				v1_ccs_connections[(i, j)] = these_connections



#base, pixel_base = helper_functions.visualizeNetwork(v1_values, rows, columns, diameter)
#base, pixel_base = visualizeNetwork(v1_values, rows, columns, diameter)
#base.show()


# Orientation visualization (8 orientations with pointing piece)
# Clockwise orientation movement

orientation_visual = {0:[(0, -1), (0, 0), (0, 1), (0, 2), (0, 3), (-1, 2), (1, 2)], 1: [(-1, -1), (0, 0), (1, 1), (2, 2), (3, 3), (2, 3), (3, 2)],
2:[(-1, 0),(0, 0), (1, 0), (2, 0), (3, 0), (2, 1), (2, -1)], 3: [(-1, 1),(0, 0), (1, -1), (2, -2), (3, -3), (2, -3), (3, -2)],
4: [(0, 1), (0, 0), (0, -1), (0, -2), (0, -3), (-1, -2), (1, -2)], 5: [(1, 1),(0, 0), (-1, -1), (-2, -2), (-3, -2), (-2, -3), (-3, -3)],
6: [(1, 0),(0, 0), (-1, 0), (-2, 0), (-3, 0), (-2, 1), (-2, -1)], 7: [(1, -1), (0, 0), (-1, 1), (-2, 2), (-3, 3), (-3, 2), (-2, 3)]}


base, pixel_base = helper_functions.visualizeNetworkWithOrientations(v1_values, rows, columns, diameter, orientation_visual)
#base.show()

# Simplifying assumption: Only one spatial frequency modeled. Is this a fundamental problem?
# So use only one spatial frequency for Gabor filters orientation selectivity.




# Now do the iterative non-linearity with linear lambda updates interspersed (interleaved)
image_sequence = []
image_sequence += [base]
for epoch in range(epochs):
	#inputs_from_image = helper_functions.getInputsFromImage(input_image, input_pixels, v1_values, all_Gabors, rows, columns)
	inputs_from_image, multiplicated_image = helper_functions.getInputsFromImageAndReturnProductImage(input_image, input_pixels, v1_values, all_Gabors, rows, columns)
	# Update Inputs to network ("input deltas"): Based on current network values compute new inputs and update old inputs with a fraction of the new ones (lambda update)
	input_deltas = helper_functions.computeDeltaInputs(inputs_from_image)
	# Update network input values
	v1_input_values = helper_functions.updateNetInputs(v1_input_values, input_deltas, lmbd) # Change the net_inputs['C'] value in order to add an oscillation
	# "Activate network" which means, apply divisive normalization
	#helper_functions.activateNetworkGlobal(v1_values, v1_input_values, alpha, beta, diameter)
	helper_functions.activateNetworkGaussian(v1_values, v1_input_values, alpha, beta, diameter, amp, sd)
	if epoch % every_how_many_epochs == 0:
		multiplicated_image.show()
		base, pixel_base = helper_functions.visualizeNetworkWithOrientations(v1_values, rows, columns, diameter, orientation_visual)
		image_sequence += [base]


writeGif(gifs_directory + name_of_gif + ".gif", image_sequence, duration=0.2)



