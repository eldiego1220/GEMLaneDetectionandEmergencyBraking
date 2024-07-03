import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pickle
# from combined_thresh import combined_thresh
# from perspective_transform import perspective_transform

# feel free to adjust the parameters in the code if necessary

def line_fit(binary_warped):
	"""
	Find and fit lane lines
	"""
	# Assuming you have created a warped binary image called "binary_warped"
	# Take a histogram of the bottom half of the image
	histogram = np.sum(binary_warped[binary_warped.shape[0]//2:,:], axis=0)
	# Create an output image to draw on and visualize the result
	out_img = (np.dstack((binary_warped, binary_warped, binary_warped))*255).astype('uint8')
	# Find the peak of the left and right halves of the histogram
	# These will be the starting point for the left and right lines
	midpoint = int(histogram.shape[0]/2)
	leftx_base = np.argmax(histogram[100:midpoint]) + 100
	rightx_base = np.argmax(histogram[midpoint:-100]) + midpoint

	# Choose the number of sliding windows
	nwindows = 9
	# Set height of windows
	window_height = int(binary_warped.shape[0]/nwindows)
	# Identify the x and y positions of all nonzero pixels in the image
	nonzero = binary_warped.nonzero()
	nonzeroy = np.array(nonzero[0])
	nonzerox = np.array(nonzero[1]) 
	# Current positions to be updated for each window
	leftx_current = leftx_base
	rightx_current = rightx_base
	# Set the width of the windows +/- margin
	margin = 100
	# Set minimum number of pixels found to recenter window
	minpix = 50
	# Create empty lists to receive left and right lane pixel indices
	left_lane_inds = []
	right_lane_inds = []
	#Lane detected flag
	left_flag = 1
	right_flag = 1

	# Step through the windows one by one
	for window in range(nwindows):
		# Identify window boundaries in x and y (and right and left)
		##TO DO
		start_point_left = (int(leftx_current - margin), int(binary_warped.shape[0]-(window_height+1)-window*window_height))
		end_point_left = (int(leftx_current + margin), int(binary_warped.shape[0]-1-window*window_height))

		start_point_right = (int(rightx_current - margin), int(binary_warped.shape[0]-(window_height+1)-window*window_height))
		end_point_right = (int(rightx_current + margin), int(binary_warped.shape[0]-1-window*window_height))
		####
		# Draw the windows on the visualization image using cv2.rectangle()
		##TO DO
		out_img = cv2.rectangle(out_img, start_point_left, end_point_left, (255, 0, 0), 3)
		out_img = cv2.rectangle(out_img, start_point_right, end_point_right, (255, 0, 0), 3) 


		####
		# Identify the nonzero pixels in x and y within the window
		##TO DO
		left_count = 0
		right_count = 0
		for i in range(np.shape(nonzero)[1]):
			if (nonzerox[i] > leftx_current - margin and nonzerox[i] < leftx_current + margin) and left_flag:
				if (nonzeroy[i] > binary_warped.shape[0]-(window_height+1)-window*window_height and nonzeroy[i] < binary_warped.shape[0]-1-window*window_height):
					left_lane_inds.append(i)
					left_count += 1

			if (nonzerox[i] > rightx_current - margin and nonzerox[i] < rightx_current + margin and right_flag):
				if (nonzeroy[i] > binary_warped.shape[0]-(window_height+1)-window*window_height and nonzeroy[i] < binary_warped.shape[0]-1-window*window_height):
					right_lane_inds.append(i)
					right_count +=1


		####
		# If you found > minpix pixels, recenter next window on their mean position
		##TO DO
					
		if window == 0 and left_count < minpix:
			print("no left lane")
			left_flag = 0
		elif left_count > minpix and left_flag:
			leftx_current = np.mean(nonzerox[left_lane_inds])


		if window == 0 and right_count < minpix:
			print("no right lane")
			right_flag = 0
		elif right_count > minpix and right_flag:
			rightx_current = np.mean(nonzerox[right_lane_inds])

		####

	# Concatenate the arrays of indices
	# left_lane_inds = np.concatenate(left_lane_inds)
	# right_lane_inds = np.concatenate(right_lane_inds)

	# Extract left and right line pixel positions
	leftx = nonzerox[left_lane_inds]
	lefty = nonzeroy[left_lane_inds]
	rightx = nonzerox[right_lane_inds]
	righty = nonzeroy[right_lane_inds]

	# Fit a second order polynomial to each using np.polyfit()
	# If there isn't a good fit, meaning any of leftx, lefty, rightx, and righty are empty,
	# the second order polynomial is unable to be sovled.
	# Thus, it is unable to detect edges.
	try:
	##TODO
		if left_flag and right_flag:
			print("right and left lanes exist")
			left_fit = np.polyfit(lefty, leftx, 2)
			right_fit = np.polyfit(righty, rightx, 2)
			if left_fit[2] > 960 and right_fit[2] > 320:
				print("converged to the right")
				left_fit = [right_fit[0], right_fit[1], right_fit[2] - 750]
			if right_fit[2] < 960 and left_fit[2] < 320:
				print("converged to the left")
				right_fit = [left_fit[0], left_fit[1], left_fit[2] + 750]
			if right_fit[2] < 960 and left_fit[2] > 320:
				print("both lines are incorrect")
				return None

		elif left_flag and not right_flag:
			print("generating right lane")
			left_fit = np.polyfit(lefty, leftx, 2)
			right_fit = [left_fit[0], left_fit[1], left_fit[2] + 750]
		elif not left_flag and right_flag:
			print("generating left lane")
			right_fit = np.polyfit(righty, rightx, 2)
			left_fit = [right_fit[0], right_fit[1], right_fit[2] - 750]
	####
	except TypeError:
		print("Unable to detect lanes")
		return None


	# Return a dict of relevant variables
	ret = {}
	ret['left_fit'] = left_fit
	ret['right_fit'] = right_fit
	ret['nonzerox'] = nonzerox
	ret['nonzeroy'] = nonzeroy
	ret['out_img'] = out_img
	ret['left_lane_inds'] = left_lane_inds
	ret['right_lane_inds'] = right_lane_inds

	return ret


def tune_fit(binary_warped, left_fit, right_fit):
	"""
	Given a previously fit line, quickly try to find the line based on previous lines
	"""
	# Assume you now have a new warped binary image
	# from the next frame of video (also called "binary_warped")
	# It's now much easier to find line pixels!
	nonzero = binary_warped.nonzero()
	nonzeroy = np.array(nonzero[0])
	nonzerox = np.array(nonzero[1])
	margin = 100
	left_lane_inds = ((nonzerox > (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2] - margin)) & (nonzerox < (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2] + margin)))
	right_lane_inds = ((nonzerox > (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + right_fit[2] - margin)) & (nonzerox < (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + right_fit[2] + margin)))
	left_flag = 1
	right_flag = 1

	#check min count
	histogram = np.sum(binary_warped[binary_warped.shape[0]//2:,:], axis=0)
	midpoint = int(histogram.shape[0]/2)
	leftx_current = np.argmax(histogram[100:midpoint]) + 100
	rightx_current = np.argmax(histogram[midpoint:-100]) + midpoint

	nwindows = 9
	window_height = int(binary_warped.shape[0]/nwindows)
	minpix = 50

	left_count = 0
	right_count = 0
	for i in range(np.shape(nonzero)[1]):
		if (nonzerox[i] > leftx_current - margin and nonzerox[i] < leftx_current + margin) and left_flag:
			if (nonzeroy[i] > binary_warped.shape[0]-(window_height+1) and nonzeroy[i] < binary_warped.shape[0]-1):
				left_count += 1

		if (nonzerox[i] > rightx_current - margin and nonzerox[i] < rightx_current + margin and right_flag):
			if (nonzeroy[i] > binary_warped.shape[0]-(window_height+1) and nonzeroy[i] < binary_warped.shape[0]-1):
				right_count +=1
				
	if left_count < minpix:
		print("no left lane")
		left_flag = 0
	if right_count < minpix:
		print("no right lane")
		right_flag = 0

	# Again, extract left and right line pixel positions
	leftx = nonzerox[left_lane_inds]
	lefty = nonzeroy[left_lane_inds]
	rightx = nonzerox[right_lane_inds]
	righty = nonzeroy[right_lane_inds]

	# If we don't find enough relevant points, return all None (this means error)
	min_inds = 10
	if lefty.shape[0] < min_inds or righty.shape[0] < min_inds:
		return None

	# Fit a second order polynomial to each
	if left_flag and right_flag:
			print("right and left lanes exist")
			left_fit = np.polyfit(lefty, leftx, 2)
			right_fit = np.polyfit(righty, rightx, 2)
			# if left_fit[2] > 960 and right_fit[2] > 320:
			# 	print("converged to the right")
			# 	left_fit = [right_fit[0], right_fit[1], right_fit[2] - 750]
			# if right_fit[2] < 960 and left_fit[2] < 320:
			# 	print("converged to the left")
			# 	right_fit = [left_fit[0], left_fit[1], left_fit[2] + 750]
			# if right_fit[2] < 960 and left_fit[2] > 320:
			# 	print("both lines are incorrect")
			# 	return None

	elif left_flag and not right_flag:
		print("generating right lane")
		left_fit = np.polyfit(lefty, leftx, 2)
		right_fit = [left_fit[0], left_fit[1], left_fit[2] + 750]
	elif not left_flag and right_flag:
		print("generating left lane")
		right_fit = np.polyfit(righty, rightx, 2)
		left_fit = [right_fit[0], right_fit[1], right_fit[2] - 750]

	# Generate x and y values for plotting
	ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
	left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
	right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

	# Return a dict of relevant variables
	ret = {}
	ret['left_fit'] = left_fit
	ret['right_fit'] = right_fit
	ret['nonzerox'] = nonzerox
	ret['nonzeroy'] = nonzeroy
	ret['left_lane_inds'] = left_lane_inds
	ret['right_lane_inds'] = right_lane_inds

	return ret


def viz1(binary_warped, ret, save_file=None):
	"""
	Visualize each sliding window location and predicted lane lines, on binary warped image
	save_file is a string representing where to save the image (if None, then just display)
	"""
	# Grab variables from ret dictionary
	left_fit = ret['left_fit']
	right_fit = ret['right_fit']
	nonzerox = ret['nonzerox']
	nonzeroy = ret['nonzeroy']
	out_img = ret['out_img']
	left_lane_inds = ret['left_lane_inds']
	right_lane_inds = ret['right_lane_inds']

	# Generate x and y values for plotting
	ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
	left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
	right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

	out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
	out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]
	plt.imshow(out_img)
	plt.plot(left_fitx, ploty, color='yellow')
	plt.plot(right_fitx, ploty, color='yellow')
	plt.xlim(0, 1280)
	plt.ylim(720, 0)
	if save_file is None:
		plt.show()
	else:
		plt.savefig(save_file)
	plt.gcf().clear()


def bird_fit(binary_warped, ret, save_file=None):
	"""
	Visualize the predicted lane lines with margin, on binary warped image
	save_file is a string representing where to save the image (if None, then just display)
	"""
	# Grab variables from ret dictionary
	left_fit = ret['left_fit']
	right_fit = ret['right_fit']
	nonzerox = ret['nonzerox']
	nonzeroy = ret['nonzeroy']
	left_lane_inds = ret['left_lane_inds']
	right_lane_inds = ret['right_lane_inds']

	# Create an image to draw on and an image to show the selection window
	out_img = (np.dstack((binary_warped, binary_warped, binary_warped))*255).astype('uint8')
	window_img = np.zeros_like(out_img)
	# Color in left and right line pixels
	out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
	out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

	# Generate x and y values for plotting
	ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0])
	left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
	right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

	# Generate a polygon to illustrate the search window area
	# And recast the x and y points into usable format for cv2.fillPoly()
	margin = 100  # NOTE: Keep this in sync with *_fit()
	left_line_window1 = np.array([np.transpose(np.vstack([left_fitx-margin, ploty]))])
	left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx+margin, ploty])))])
	left_line_pts = np.hstack((left_line_window1, left_line_window2))
	right_line_window1 = np.array([np.transpose(np.vstack([right_fitx-margin, ploty]))])
	right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx+margin, ploty])))])
	right_line_pts = np.hstack((right_line_window1, right_line_window2))

	# Draw the lane onto the warped blank image
	cv2.fillPoly(window_img, np.int_([left_line_pts]), (0,255, 0))
	cv2.fillPoly(window_img, np.int_([right_line_pts]), (0,255, 0))
	result = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)

	# cv2.imshow('bird',result)
	# cv2.imwrite('bird_from_cv2.png', result)

	# if save_file is None:
	# 	plt.show()
	# else:
	# 	plt.savefig(save_file)
	# plt.gcf().clear()

	return result


def final_viz(undist, left_fit, right_fit, m_inv):
	"""
	Final lane line prediction visualized and overlayed on top of original image
	"""
	# Generate x and y values for plotting
	ploty = np.linspace(0, undist.shape[0]-1, undist.shape[0])
	left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
	right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

	# Create an image to draw the lines on
	#warp_zero = np.zeros_like(warped).astype(np.uint8)
	#color_warp = np.dstack((warp_zero, warp_zero, warp_zero))
	color_warp = np.zeros((720, 1280, 3), dtype='uint8')  # NOTE: Hard-coded image dimensions

	# Recast the x and y points into usable format for cv2.fillPoly()
	pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
	pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
	pts = np.hstack((pts_left, pts_right))

	# Draw the lane onto the warped blank image
	cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))

	# Warp the blank back to original image space using inverse perspective matrix (Minv)
	newwarp = cv2.warpPerspective(color_warp, m_inv, (undist.shape[1], undist.shape[0]))
	# Combine the result with the original image
	# Convert arrays to 8 bit for later cv to ros image transfer
	undist = np.array(undist, dtype=np.uint8)
	newwarp = np.array(newwarp, dtype=np.uint8)
	result = cv2.addWeighted(undist, 1, newwarp, 0.3, 0)

	return result