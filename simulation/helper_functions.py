# These are helper functions for the project. Includes visualization,
# Gabor operations and electric stimulation.


from PIL import Image, ImageFilter, ImageDraw
import random
import math
import sys
import os
from images2gif import writeGif


# Math functions
#################################################################################################################################
# Simple Gaussian function that is simply distance-from-origin-dependent
e = math.exp(1)
def gaussianFunction(amp, sd, d):
	return amp*e**(-d**2 / (2*sd**2))

# Euclidean distance in hexagonal array in 2D.
def computeDistance(i, j, ii, jj, diameter):
	dx = (i - ii)*diameter
	if i % 2 != ii % 2:
		if i % 2 == 0:
			dx = (i - ii)*diameter + diameter/2.
		else:
			dx = (i - ii)*diameter - diameter/2.
	dy = diameter*3**.5*(j - jj)/2.
	distance = (dx**2 + dy**2)**.5
	return distance


# Network visualization functions.
#################################################################################################################################
# Visualize v1 
def visualizeNetwork(v1_values, rows, columns, diameter):
	width = int(columns*diameter)
	height = int(diameter*(1 + (rows - 1)*3**.5/2.)) 
	base = Image.new('RGB', (width, height))
	pixel_base = base.load()
	for i in range(width):
		for j in range(height):
			pixel_base[i, j] = (0, 0, 0)
	for i, j in v1_values.keys():
		if (i % 2 == 0):
			this_x = min(int(j*diameter + diameter/2), width - 1)
			this_y = min(int(diameter/2 + diameter*i*3**.5/2.), height - 1)
			pixel_base[this_x, this_y] = (255, 255, 255)
		else:
			this_x = min(int(j*diameter + diameter), width - 1)
			this_y = min(int(diameter/2 + diameter*i*3**.5/2.), height - 1)
			pixel_base[this_x, this_y] = (255, 255, 255)	
	return base, pixel_base


# Visualize v1 
def visualizeNetworkWithOrientations(v1_values, rows, columns, diameter, orientation_visual):
	width = int(columns*diameter)
	height = int(diameter*(1 + (rows - 1)*3**.5/2.)) 
	base = Image.new('RGB', (width, height))
	pixel_base = base.load()
	for i in range(width):
		for j in range(height):
			pixel_base[i, j] = (0, 0, 0)
	for i, j in v1_values.keys():
		brightness = v1_values[(i, j)][1]
		orientation = v1_values[(i, j)][2]
		this_x = min(int(j*diameter + diameter), width - 1)
		this_y = min(int(diameter/2 + diameter*i*3**.5/2.), height - 1)
		if (i % 2 == 0):
			this_x = min(int(j*diameter + diameter/2), width - 1)
			this_y = min(int(diameter/2 + diameter*i*3**.5/2.), height - 1)
		for di, dj in orientation_visual[orientation]:
			if this_x + di >= 0 and this_y + dj >= 0 and this_x + di < width and this_y + dj < height:
				#print this_x, di, this_y, dj
				one_channel = min(max(int(brightness*255), 0), 255)
				rgb_brightness_corrected = (one_channel, one_channel, one_channel)
				pixel_base[this_x + di, this_y + dj] = rgb_brightness_corrected
	return base, pixel_base	


# This function is to update the visualization values (index 2 of v1_input)
def updateVisualizationValues(v1_values):
	list_of_values = [] # I know, this is inefficient, datastructure hierarchy would benefit from being tweaked. That said this only needs to be called when visualizing
	for i, j in v1_values.keys():
		list_of_values.append(v1_values[(i, j)][0])
	maximum_in_list = max(list_of_values)
	for i, j in v1_values.keys():
		v1_values[(i, j)][1] = v1_values[(i, j)][0] / float(maximum_in_list)
	return v1_values



# This should only be ran more than once if there are multiple input sources.
#################################################################################################################################
# This takes an image as input and return the inptut to the cortical
# columns after Gabor filter stuff.
# inputs_from_image has the same format as v1_values: {(a, b):[x, y, z, ...]}
def getInputsFromImage(input_image, input_pixels, v1_values, all_Gabors, rows, columns):
	xx, yy = input_image.size
	row_intervals = xx/rows
	column_intervals = yy/columns
	inputs_from_image = {}
	for i, j in v1_values.keys():
		orientation = v1_values[(i, j)][2]
		frequency = v1_values[(i, j)][3]
		phase = v1_values[(i, j)][4]
		added_input = 0.
		this_x = row_intervals*i
		this_y = column_intervals*j
		for ii, jj in all_Gabors[(orientation, frequency, phase)].keys():
			# go to regions of input_pixels in a way that scales the number of cortical columns to the size of the picture. 
			# In the future maybe do the polar coordinates of truly retinotopic areas of the brain.
			if (this_x + ii) >= 0 and (this_x + ii) < xx:
				if (this_y + jj) >= 0 and (this_y + jj) < yy:
					rr, gg, bb = input_pixels[this_x + ii, this_y + jj]
					added_input += all_Gabors[(orientation, frequency, phase)][(ii, jj)]*(rr + gg + bb) / 3.
		inputs_from_image[(i, j)] = added_input
	return inputs_from_image




# This section uses image, and network values to determine the new delta input values (right before updating the network inputs themselves)
#################################################################################################################################
# Neural Network operations
def computeDeltaInputs(inputs_from_image, v1_values):
	return inputs_from_image


def deltaInputsLocalSimilarity(inputs_from_image, v1_values, v1_ccs_connections, scaling):
	all_hypercolumns = inputs_from_image.keys()
	for i, j in all_hypercolumns:
		center_orientation = v1_values[(i, j)][2]
		these_connections = v1_ccs_connections[(i, j)]
		added_input = 0.
		for ii, jj in these_connections:
			this_value = v1_values[(ii, jj)][0]
			this_orientation = v1_values[(ii, jj)][2]
			angle_difference = 2*math.pi*((center_orientation - this_orientation) % 8)/8.
			cosine_similarity = abs(math.cos(angle_difference))
			added_input += this_value*cosine_similarity*scaling
		inputs_from_image[(i, j)] += added_input
	return inputs_from_image


# Update inputs
#################################################################################################################################
# Using the lambda rule to update network inputs based on the input deltas (computed from image and last network's activation)
def updateNetInputs(v1_input_values, input_deltas, lmbd):
	for i, j in v1_input_values.keys():
		v1_input_values[(i, j)] = (1 - lmbd) * v1_input_values[(i, j)] + lmbd * input_deltas[(i, j)]
	return v1_input_values



# Network activation functions (global, gaussian or mexican hat)
#################################################################################################################################
# Normalization over the entire v1 layer (rather than localized divisive)
def activateNetworkGlobal(v1_values, v1_input_values, alpha, beta, diameter):
	total_number_of_units = len(v1_values.keys())
	total_sum_of_exponents_of_inputs = 0.
	for i, j in v1_input_values.keys():
		total_sum_of_exponents_of_inputs += e**max(min(v1_input_values[(i, j)], 700), -700)
	for i, j in v1_input_values.keys():
		v1_values[(i, j)][0] = e**max(min(v1_input_values[(i, j)], 700), -700) / (alpha + beta*total_sum_of_exponents_of_inputs)
	return v1_values



# The various activation functions
def activateNetworkGaussian(v1_values, v1_input_values, alpha, beta, diameter, amp, sd):
	for i, j in v1_input_values.keys():
		total_sum_of_exponents_of_inputs = 0.
		for ii, jj in v1_input_values.keys():
			distance = computeDistance(i, j, ii, jj, diameter)
			total_sum_of_exponents_of_inputs += (e**max(min(v1_input_values[(ii, jj)], 700), -700))*gaussianFunction(amp, sd, distance)
		v1_values[(i, j)][0] = e**max(min(v1_input_values[(i, j)], 700), -700) / (alpha + beta*total_sum_of_exponents_of_inputs)
	return v1_values


def activateNetworkWithMexicanHatKernel(network, net_inputs, alpha, beta, t, amp1, sd1, amp2, sd2):
	return



#################################################################################################################################
# Creation of "Gabor receptive fields." these are patches like wavelets in real space
# The filters encode frequency, phase and orientation. In the toy network the frequency and phase are held constant.
# But here these will also be possible variables. 
def createGaborDataStructure(width, height, cycles_per_pixel, amp, sd, angle):
	center_point = (width/2., height/2.)
	direction_vector = (math.sin(angle/180.*math.pi), math.cos(angle/180.*math.pi))
	filter_values = {}
	for i in range(width):
		for j in range(height):
			abs_x = int(i - center_point[0])
			abs_y = int(j - center_point[1])
			component_in_direction = abs_x*direction_vector[0] + abs_y*direction_vector[1] # dot product
			value = math.sin(math.pi*component_in_direction*cycles_per_pixel)
			distance_to_center = (abs_x**2 + abs_y**2)**.5
			corresponding_Gaussian_fraction = gaussianFunction(amp, sd, distance_to_center)
			value = value*corresponding_Gaussian_fraction
			filter_values[(abs_x, abs_y)] = value
	return filter_values


# With phase, orientation and frequency requirements
def createParametrizedGaborDataStructure(gabor_width, gabor_height, freq, gabor_amp, gabor_sd, angle, pha):
	center_point = (gabor_width/2., gabor_height/2.)
	direction_vector = (math.sin(angle/180.*math.pi), math.cos(angle/180.*math.pi))
	filter_values = {}
	for i in range(gabor_width):
		for j in range(gabor_height):
			abs_x = int(i - center_point[0])
			abs_y = int(j - center_point[1])
			component_in_direction = abs_x*direction_vector[0] + abs_y*direction_vector[1] # dot product
			value = math.sin(math.pi*component_in_direction*freq + pha)
			distance_to_center = (abs_x**2 + abs_y**2)**.5
			corresponding_Gaussian_fraction = gaussianFunction(gabor_amp, gabor_sd, distance_to_center)
			value = value*corresponding_Gaussian_fraction
			filter_values[(abs_x, abs_y)] = value
	return filter_values


# Creation of elipses to visualize edge detectors.
#################################################################################################################################
def createEllipse(width, height, freq, amp, sd, angle):
	center_point = (width/2., height/2.)
	direction_vector = (math.sin((angle)/180.*math.pi), math.cos((angle)/180.*math.pi))
	ellipse_values = {}
	for i in range(width):
		for j in range(height):
			abs_x = int(i - center_point[0])
			abs_y = int(j - center_point[1])
			component_in_direction = (abs_x*direction_vector[0] + abs_y*direction_vector[1]) * (0.1/freq) # dot product
			distance_to_center = (abs_x**2 + abs_y**2)**.5
			distance_component = distance_to_center*component_in_direction
			corresponding_Gaussian_fraction = gaussianFunction(amp, sd, distance_component)
			ellipse_values[(abs_x, abs_y)] = corresponding_Gaussian_fraction
	return ellipse_values



# These will be functions that will take as input v1_values and return a simulated experience based on that network's activation
#################################################################################################################################


# This just uses the visualization value and linearly adds the actual Gabor filters used in the generation of input values
def visualizeExperienceNaiveAddition(v1_values, all_Gabors, rows, columns, diameter, scale):
	xx = diameter*(rows + 1)
	yy = diameter*(columns + 1)
	experience_image = Image.new('RGB', (xx, yy))
	experience_image_pixels = experience_image.load()
	for tx in range(xx):
		for ty in range(yy):
			experience_image_pixels[tx, ty] = (127, 127, 127)
	inputs_from_image = {}
	for i, j in v1_values.keys():
		v1_this_value = v1_values[(i, j)][0]
		v1_this_visual = v1_values[(i, j)][1]
		orientation = v1_values[(i, j)][2]
		frequency = v1_values[(i, j)][3]
		phase = v1_values[(i, j)][4]
		added_input = 0.
		this_x = (diameter+1)*i 
		this_y = (diameter+1)*j 
		for ii, jj in all_Gabors[(orientation, frequency, phase)].keys():
			if (this_x + ii) >= 0 and (this_x + ii) < xx:
				if (this_y + jj) >= 0 and (this_y + jj) < yy:
					added_input = all_Gabors[(orientation, frequency, phase)][(ii, jj)]*(v1_this_visual)*scale
					rr, gg, bb = experience_image_pixels[this_x + ii, this_y + jj]
					rgbrgb = (rr + gg + bb) / 3.
					rgbrgb += added_input
					rgbrgb = int(max(0, min(255, rgbrgb)))
					experience_image_pixels[this_x + ii, this_y + jj] = (rgbrgb, rgbrgb, rgbrgb)
	return experience_image, experience_image_pixels

# This one skips the Gabors and instead goes to straight lines (or elipses)
def visualizeExperienceNaiveLines(v1_values, all_Ellipsis, rows, columns, diameter, scale):
	xx = diameter*(rows + 1)
	yy = diameter*(columns + 1)
	experience_image = Image.new('RGB', (xx, yy))
	experience_image_pixels = experience_image.load()
	for tx in range(xx):
		for ty in range(yy):
			experience_image_pixels[tx, ty] = (127, 127, 127)
	inputs_from_image = {}
	for i, j in v1_values.keys():
		v1_this_value = v1_values[(i, j)][0]
		v1_this_visual = v1_values[(i, j)][1]
		orientation = v1_values[(i, j)][2]
		frequency = v1_values[(i, j)][3]
		phase = v1_values[(i, j)][4]
		added_input = 0.
		this_x = diameter*(i+ 1) 
		this_y = diameter*(j+ 1)
		for ii, jj in all_Ellipsis[(orientation, frequency, phase)].keys():
			if (this_x + ii) >= 0 and (this_x + ii) < xx:
				if (this_y + jj) >= 0 and (this_y + jj) < yy:
					added_input = all_Ellipsis[(orientation, frequency, phase)][(ii, jj)]*(v1_this_visual)*scale
					rr, gg, bb = experience_image_pixels[this_x + ii, this_y + jj]
					rgbrgb = (rr + gg + bb) / 3.
					rgbrgb += added_input
					rgbrgb = int(max(0, min(255, rgbrgb)))
					experience_image_pixels[this_x + ii, this_y + jj] = (rgbrgb, rgbrgb, rgbrgb)
	return experience_image, experience_image_pixels


# This adds a little direction line everywhere, depending on the degree of activation of surrounding hypercolumns
# and their weighted average orientation.
# The lines are not placed at regulare pre-defined intervals. Rather they are selected at random from the possible
# centers (pixels) on the output image. Samples per block diameter.
def visualizeExperienceField(v1_values, all_Ellipsis, v1_ccs_connections, rows, columns, diameter, scale, samples):
	xx = diameter*(rows + 1)
	yy = diameter*(columns + 1)
	experience_image = Image.new('RGB', (xx, yy))
	experience_image_pixels = experience_image.load()
	for tx in range(xx):
		for ty in range(yy):
			experience_image_pixels[tx, ty] = (0, 0, 0)
	inputs_from_image = {}
	for i, j in v1_values.keys():
		v1_this_value = v1_values[(i, j)][0]
		v1_this_visual = v1_values[(i, j)][1]
		orientation = v1_values[(i, j)][2]
		frequency = v1_values[(i, j)][3]
		phase = v1_values[(i, j)][4]
		added_input = 0.
		this_x = diameter*(i+1) 
		this_y = diameter*(j+1) 

		# find the orientations of neighbours

		# find the weights of such orientations

		# compute the distance effect from there
		# computeDistance(i, j, ii, jj, diameter)

		for ii, jj in all_Ellipsis[(orientation, frequency, phase)].keys():
			if (this_x + ii) >= 0 and (this_x + ii) < xx:
				if (this_y + jj) >= 0 and (this_y + jj) < yy:
					added_input = all_Ellipsis[(orientation, frequency, phase)][(ii, jj)]*(v1_this_visual)*scale
					rr, gg, bb = experience_image_pixels[this_x + ii, this_y + jj]
					rgbrgb = (rr + gg + bb) / 3.
					rgbrgb += added_input
					rgbrgb = int(max(0, min(255, rgbrgb)))
					experience_image_pixels[this_x + ii, this_y + jj] = (rgbrgb, rgbrgb, rgbrgb)
	return experience_image, experience_image_pixels






# Perhaps now try something where the samples are chosen with a probability proportional
# to the network activity around the area.








#################################################################################################################################
# For debugging visualizations

def getInputsFromImageAndReturnProductImage(input_image, input_pixels, v1_values, all_Gabors, rows, columns):
	xx, yy = input_image.size
	multiplied_image = Image.new('RGB', (xx, yy))
	multiplied_image_pixels = multiplied_image.load()
	for i in range(xx):
		for j in range(yy):
			multiplied_image_pixels[i, j] = (127, 127, 127)
	row_intervals = xx/rows
	column_intervals = yy/columns
	inputs_from_image = {}
	for i, j in v1_values.keys():
		current_v1_value = v1_values[(i, j)][0]
		orientation = v1_values[(i, j)][2]
		added_input = 0.
		this_x = row_intervals*i
		this_y = column_intervals*j
		for ii, jj in all_Gabors[orientation].keys():
			# go to regions of input_pixels in a way that scales the number of cortical columns to the size of the picture. 
			# In the future maybe do the polar coordinates of truly retinotopic areas of the brain.
			if (this_x + ii) >= 0 and (this_x + ii) < xx:
				if (this_y + jj) >= 0 and (this_y + jj) < yy:
					rr, gg, bb = input_pixels[this_x + ii, this_y + jj]
					this_pixel_product = all_Gabors[orientation][(ii, jj)]*(rr + gg + bb) / 3.
					added_input += this_pixel_product
					this_pixel_product_to_visualize = all_Gabors[orientation][(ii, jj)]*math.log(current_v1_value + 0.00000000000000001)*3
					image_r, image_g, image_b = multiplied_image_pixels[this_x + ii, this_y + jj]
					multiplied_image_pixels[this_x + ii, this_y + jj] = (int(this_pixel_product_to_visualize + image_r), int(this_pixel_product_to_visualize + image_g), int(this_pixel_product_to_visualize + image_b))
		inputs_from_image[(i, j)] = added_input
	return inputs_from_image, multiplied_image




def multiplyImageByGabor(gabor_datastructure, input_image, input_pixels, offset_x, offset_y):
	xx, yy = input_image.size
	multiplied_image = Image.new('RGB', (xx, yy))
	multiplied_image_pixels = multiplied_image.load()
	for i in range(xx):
		for j in range(yy):
			multiplied_image_pixels[i, j] = (127, 127, 127)
	for ii, jj in gabor_datastructure.keys():
		# go to regions of input_pixels in a way that scales the number of cortical columns to the size of the picture. 
		# In the future maybe do the polar coordinates of truly retinotopic areas of the brain.
		if (offset_x + ii) >= 0 and (offset_x + ii) < xx:
			if (offset_y + jj) >= 0 and (offset_y + jj) < yy:
				rr, gg, bb = input_pixels[offset_x + ii, offset_y + jj]
				this_pixel_product = gabor_datastructure[(ii, jj)]*(rr + gg + bb) / 3.
				image_r, image_g, image_b = multiplied_image_pixels[offset_x + ii, offset_y + jj]
				multiplied_image_pixels[offset_x + ii, offset_y + jj] = (int(this_pixel_product + image_r), int(this_pixel_product + image_g), int(this_pixel_product + image_b))
	return multiplied_image






#input_pixels = input_image.load()
#multiplied_i = multiplyImageByGabor(all_Gabors[3], input_image, input_pixels, 50, 50)
#multiplied_i.show()
