# This file creates a simulation of V1 by using lateral inhibition
# on a hexagonal lattice. The network's units have Gabor receptive fields that take as input 
# the image name_of_gif below. In each iteration there is a lamda update of the unit's 
# net input followed by a re-calculation of the unit's activity using a generalization of the softmax function.


from PIL import Image, ImageFilter, ImageDraw
import random
import math
import sys
import os
from images2gif import writeGif
import helper_functions
import hardcoded_visuals


gifs_of_v1_activity = "/Users/andesgomez/Documents/Stanford/Spring2015/Psych287/project/v1_activity_gifs/"
gifst_of_v1_encoded_pictures = "/Users/andesgomez/Documents/Stanford/Spring2015/Psych287/project/v1_encoding_gifs/"
image_directory = "/Users/andesgomez/Documents/Stanford/Spring2015/Psych287/project/source_pictures/"


name_of_gif = "gaussiann_3_6freq1phase_j_" #


rows = 30
columns = 30
default_potential = 0.25
diameter = 10
input_values = 0.0

# Network operation/update parameters
lmbd = 0.1 # this is the lambda parameter for the imput updating
epochs = 20
alpha = 0.32 # .31
beta = .02 # 0.002
t = 1.0
amp = 15
sd  = 5
every_how_many_epochs = 3


candidate_pictures = os.listdir('../source_pictures')
image_to_use = candidate_pictures[2]


input_image = Image.open(image_directory + image_to_use)
input_pixels = input_image.load()


# Parameter description string

name_of_image_without_extension = image_to_use.split('.')[0]
parameter_description = "lmbd" + str(lmbd) + "r" + str(rows) + "c" + str(columns) + "diameter" + str(diameter) + "a" + str(alpha) + "b" + str(beta) + "amp" + str(amp) + "sd" + str(sd) +"epochs" + str(epochs) + "image_" + name_of_image_without_extension
name_of_gif += parameter_description


# v1_values is a datastructure that keeps track of:
# 0 - the activation values (result of a divisive normalization)
# 1 - friendly 'renormalized' values to allow proper visualization of v1 activation and of encoded activation gifs
# 2 - orientations
# 3 - frequency
# 4 - phase
# You have to set these variables when you initialize the v1_values.

# v1_input_values is a datastructure that keeps track of the inputs from other units or source image

# First define the range of frequencies you can choose from as well as the phases
posible_frquencies = [0.01, 0.05, 0.1, 0.13, 0.16, 0.2]
possible_phases = [0.]


v1_values = {}
v1_input_values = {}
for i in range(rows):
	for j in range(columns):
		if (i % 2 == 0 or j < columns - 1):
			this_frequency = posible_frquencies[random.randint(0,len(posible_frquencies) - 1)]
			this_phase = possible_phases[random.randint(0,len(possible_phases) - 1)]
			v1_values[(i, j)] = [default_potential + random.random()/4., 0, (j) % 8, this_frequency, this_phase]
			v1_input_values[(i, j)] = input_values

available_coordinates = set(v1_values.keys())

available_v1_gabor_parameters = set()
for i, j in v1_values.keys():
	gabor_pars = (v1_values[(i, j)][2], v1_values[(i, j)][3], v1_values[(i, j)][4])
	available_v1_gabor_parameters.add(gabor_pars)


# Create the Gabor data structures that will be used as filters
# These are to be used on the getInputsFromImage function because these Gabor filters
# provide the receptive fields for the various hypercolumns.

gabor_width = 25
gabor_height = 25
gabor_cycles_per_pixel = .2
gabor_amp = 1
gabor_sd = 5
#angles = [0, 45, 90, 135, 180, 225, 270, 315]
angles = [0, 45, 90, 135, 180, 225, 270, 315]


all_Gabors = {}
available_v1_gabor_parameters = list(available_v1_gabor_parameters)
for i in range(len(available_v1_gabor_parameters)):
	ori, freq, pha = available_v1_gabor_parameters[i]
	all_Gabors[(ori, freq, pha)] = helper_functions.createParametrizedGaborDataStructure(gabor_width, gabor_height, freq, gabor_amp, gabor_sd, angles[ori], pha)


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
# This retrieves the hardoded orientation visuals from the hardcoded_visuals functions.
orientation_visual = hardcoded_visuals.getVisuals(0)

# Possibly replace the orientations for the far far more general trick of using the rainbow for coding angle.


base, pixel_base = helper_functions.visualizeNetworkWithOrientations(v1_values, rows, columns, diameter, orientation_visual)
#base.show()

# Simplifying assumption: Only one spatial frequency modeled. Is this a fundamental problem?
# So use only one spatial frequency for Gabor filters orientation selectivity.




# Now do the iterative non-linearity with linear lambda updates interspersed (interleaved)
image_sequence = []
image_sequence += [base]
for epoch in range(epochs):
	inputs_from_image = helper_functions.getInputsFromImage(input_image, input_pixels, v1_values, all_Gabors, rows, columns)
	#inputs_from_image, multiplicated_image = helper_functions.getInputsFromImageAndReturnProductImage(input_image, input_pixels, v1_values, all_Gabors, rows, columns)
	# Update Inputs to network ("input deltas"): Based on current network values compute new inputs and update old inputs with a fraction of the new ones (lambda update)
	input_deltas = helper_functions.computeDeltaInputs(inputs_from_image)
	# Update network input values
	v1_input_values = helper_functions.updateNetInputs(v1_input_values, input_deltas, lmbd)
	# "Activate network" which means, apply divisive normalization
	#v1_values = helper_functions.activateNetworkGlobal(v1_values, v1_input_values, alpha, beta, diameter)
	v1_values = helper_functions.activateNetworkGaussian(v1_values, v1_input_values, alpha, beta, diameter, amp, sd)
	if epoch % every_how_many_epochs == 0:
		#multiplicated_image.show()
		v1_values = helper_functions.updateVisualizationValues(v1_values)
		base, pixel_base = helper_functions.visualizeNetworkWithOrientations(v1_values, rows, columns, diameter, orientation_visual)
		image_sequence += [base]


writeGif(gifs_of_v1_activity + name_of_gif + ".gif", image_sequence, duration=0.2)



