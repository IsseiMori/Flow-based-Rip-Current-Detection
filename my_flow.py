# python contours.py --video beach.mp4 --out . --height 480 --window 900


# Unresolved Bugs

import os
import numpy as np 
import cv2
import argparse
import time
import math
import matplotlib.pyplot as plt
from PIL import Image
import copy

def zero_edge_flow(cpu_flow, offset = 20):

	height, width, _ = cpu_flow.shape

	cpu_flow[0:offset,:,0] = 0
	cpu_flow[0:offset,:,1] = 0
	cpu_flow[height-offset:height,:,0] = 0
	cpu_flow[height-offset:height,:,1] = 0
	cpu_flow[:,0:offset,0] = 0
	cpu_flow[:,0:offset,1] = 0
	cpu_flow[:,width-offset:width,0] = 0
	cpu_flow[:,width-offset:width,1] = 0

	return cpu_flow

def remove_outlier(cpu_flow, p=1.5):
	
	q3_x = np.quantile(cpu_flow[:,:,0], (0.75))
	q3_y = np.quantile(cpu_flow[:,:,1], (0.75))
	q1_x = np.quantile(cpu_flow[:,:,0], (0.25))
	q1_y = np.quantile(cpu_flow[:,:,1], (0.25))

	ior_x = q3_x - q1_x
	ior_y = q3_y - q1_y

	cpu_flow[:,:,0] = np.where(cpu_flow[:,:,0] > (q3_x + p * ior_x), 0, cpu_flow[:,:,0])
	cpu_flow[:,:,1] = np.where(cpu_flow[:,:,1] > (q3_y + p * ior_y), 0, cpu_flow[:,:,1])
	cpu_flow[:,:,0] = np.where(cpu_flow[:,:,0] > (q3_y + p * ior_y), 0, cpu_flow[:,:,0])
	cpu_flow[:,:,1] = np.where(cpu_flow[:,:,1] > (q3_x + p * ior_x), 0, cpu_flow[:,:,1])

	return cpu_flow


# return degree angle and normalized magnitude
def calc_angle_from_flow_cpu(cpu_flow):
	cpu_flow_x = cpu_flow[:,:,0]
	cpu_flow_y = cpu_flow[:,:,1]

	cpu_flow_magnitude, cpu_flow_angle = cv2.cartToPolar(
		cpu_flow_x, cpu_flow_y, angleInDegrees=True,
	)

	cv2.normalize(cpu_flow_magnitude, cpu_flow_magnitude, 0.0, 1.0, cv2.NORM_MINMAX)

	return cpu_flow_angle, cpu_flow_magnitude

def calc_bgr_from_angle_magnitude_rip(cpu_flow_angle, cpu_flow_magnitude, rip_bins):

	new_magnitude = cpu_flow_magnitude.copy()
	new_magnitude = np.clip(new_magnitude, 0, np.quantile(new_magnitude, (0.95)))
	new_magnitude = new_magnitude / np.max(new_magnitude)

	# new_magnitude = np.where(new_magnitude < 0., 0, new_magnitude)

	# angle 0-360, 0 left, up 90, right 180, bottom 270
	#   1 
	# 0    2 
	# 5    3
	#   4

	for i in range(0, 6):
		if i in rip_bins:
			angle_low = i * 60
			angle_high = (i + 1) * 60
			new_magnitude = np.where((cpu_flow_angle >= angle_low) & (cpu_flow_angle <= angle_high), 0, new_magnitude)



	cpu_flow_hsv = cv2.merge((
		cpu_flow_angle, 
		np.ones_like(cpu_flow_angle, np.float32),
		new_magnitude
	))

	cpu_flow_bgr = cv2.cvtColor(cpu_flow_hsv, cv2.COLOR_HSV2BGR) * 255
	cpu_flow_bgr = cpu_flow_bgr.astype(np.uint8)

	return cpu_flow_bgr

def add_color_wheel(img, wheel):
	wheel_resized = cv2.resize(wheel, (50, 50))
	img[40:40+wheel_resized.shape[0], 10:10+wheel_resized.shape[1]] = wheel_resized

def unitize_xy(x, y):
	theta = math.atan2(y,x)
	return (math.cos(theta), math.sin(theta))

def flow_to_unit(flow):
	flow_unit = flow.copy()

	for row in range(len(flow_unit)):
		for col in range(len(flow_unit[0])):
			dx = flow[row][col][0]
			dy = flow[row][col][1]
			dx, dy = unitize_xy(dx, dy)
			flow_unit[row][col][0] = dx
			flow_unit[row][col][1] = dy

	return flow_unit

def xy_to_bin(x, y, bin_num):
	theta = math.atan2(y,x)
	bin_dir = (int)((theta + math.pi) / (2 * math.pi) * bin_num)
	return bin_dir % bin_num

# calculate bin for each arrow
def flow_to_bins(flow, bin_num):
	flow_bins = np.zeros((flow.shape[0], flow.shape[1]), dtype=np.uint16)

	for row in range(len(flow)):
		for col in range(len(flow[0])):
			dx = flow[row][col][0]
			dy = flow[row][col][1]

			bin_dir = xy_to_bin(dx, dy, bin_num)
			flow_bins[row][col] = bin_dir

	return flow_bins

def flow_to_bin_weighted(flow, bin_num):
	bin_weighted = np.zeros((flow.shape[0], flow.shape[1], bin_num), dtype=np.float32)

	for row in range(len(flow)):
		for col in range(len(flow[0])):
			dx = flow[row][col][0]
			dy = flow[row][col][1]
			magnitude = math.sqrt(dx * dx + dy * dy)
			bin_dir = xy_to_bin(dx, dy, bin_num)
			bin_weighted[row][col][bin_dir] += magnitude

	return bin_weighted


# increment the bin count
def append_to_bin_hist(bin_hist, flow_bins):
	
	for row in range(len(flow_bins)):
		for col in range(len(flow_bins[0])):
			bin_dir = flow_bins[row][col]
			bin_hist[row][col][bin_dir] += 1

	return bin_hist

# increment the bin count
def remove_from_bin_hist(bin_hist, flow_bins):
	
	for row in range(len(flow_bins)):
		for col in range(len(flow_bins[0])):
			bin_dir = flow_bins[row][col]
			bin_hist[row][col][bin_dir] -= 1

	return bin_hist

def hist_to_max_bin(bin_hist):

	max_bins = np.zeros((bin_hist.shape[0], bin_hist.shape[1]), dtype=np.uint16)

	for row in range(len(bin_hist)):
		for col in range(len(bin_hist[0])):
			bin_dir = np.argmax(bin_hist[row][col])
			max_bins[row][col] = bin_dir

	return max_bins


def bin_to_flow(bin_dir, bin_num):
	angle = bin_dir / float(bin_num) * math.pi * 2 - math.pi + 0.5 / float(bin_num) * math.pi * 2
	dx = math.cos(angle)
	dy = math.sin(angle)
	return (dx, dy)

def max_bin_to_rip(bin_dir, bin_num):
	opposit = bin_dir - bin_num / 2 if bin_dir + bin_num / 2 > bin_num - 1 else bin_dir + bin_num / 2
	opp_near1 = bin_dir - bin_num / 2 + 1 if bin_dir + bin_num / 2 + 1 > bin_num - 1 else bin_dir + bin_num / 2 + 1
	opp_near2 = bin_dir - bin_num / 2 - 1 if bin_dir + bin_num / 2 - 1 > bin_num - 1 else bin_dir + bin_num / 2 - 1

	return int(opposit), int(opp_near1), int(opp_near2)

def mat_mode_bin(flow_bins, bin_num):

	count = np.zeros(bin_num, dtype=np.uint16)

	for row in range(len(flow_bins)):
		for col in range(len(flow_bins[0])):
			bin_dir = flow_bins[row][col]
			count[bin_dir] += 1

	return np.argmax(count)

def mat_mode_bin_min(flow_bins, bin_num, flow, min):

	count = np.zeros(bin_num, dtype=np.uint16)

	for row in range(len(flow_bins)):
		for col in range(len(flow_bins[0])):
			dx = flow[row][col][0]
			dy = flow[row][col][1]
			if math.sqrt(dx*dx+dy*dy) >= min:
				bin_dir = flow_bins[row][col]
				count[bin_dir] += 1

	return np.argmax(count)

def draw_arrows_flow(img, flow, bin_num, vertices_root_pos_2d, dt, wave_dir, min=0):

	img_ret = img.copy()
	img_ret_seg = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
	img_ret_seg.fill(0)

	max_x = np.amax(flow[:][:][0])
	max_y = np.amax(flow[:][:][1])

	max_len = math.sqrt(max_x * max_x + max_y * max_y)

	flow_bins = flow_to_bins(flow, bin_num)
	max_bin = mat_mode_bin_min(flow_bins, bin_num, flow, min)

	if (wave_dir != -1): max_bin = wave_dir

	opposit, opp_near1, opp_near2 = max_bin_to_rip(max_bin, bin_num)

	img_ret = cv2.rectangle(img_ret, (5, 35), (180,95), (0,0,0), -1)

	cv2.arrowedLine(img_ret, (10, 50), (30, 50), (0, 0, 255), 3, tipLength = 0.5)
	cv2.arrowedLine(img_ret, (10, 80), (30, 80), (0, 255, 255), 3, tipLength = 0.5)

	img_ret = cv2.putText(img_ret, "rip current", (40,55), cv2.FONT_HERSHEY_SIMPLEX,  
                   0.6, (255,255,255), 1, cv2.LINE_AA)
	img_ret = cv2.putText(img_ret, "feeder current", (40,85), cv2.FONT_HERSHEY_SIMPLEX,  
                   0.6, (255,255,255), 1, cv2.LINE_AA)


	for row in range(len(flow)):
		for col in range(len(flow[0])):
			dx = flow[row][col][0]
			dy = flow[row][col][1]

			# if math.sqrt(dx*dx+dy*dy) >= min:

			dx, dy = unitize_xy(dx, dy)
			# dx = dx / math.sqrt(4)
			# dy = dy / math.sqrt(4)
			x0 = vertices_root_pos_2d[row][col][0]
			y0 = vertices_root_pos_2d[row][col][1]

			bin_dir = xy_to_bin(dx, dy, bin_num)
			if bin_dir == opposit or bin_dir == opp_near1 or bin_dir == opp_near2:
				if (bin_dir == opposit) : 
					color = (0, 0, 255)
				elif (bin_dir == opp_near1) : 
					color = (0, 255, 255)
				elif (bin_dir == opp_near2) : 
					color = (0, 255, 255)


				cv2.arrowedLine(img_ret, (int(x0), int(y0)), (int(x0 + dx * dt), int(y0 + dy * dt)), color, 3, tipLength = 0.5)

	return img_ret, [opposit, opp_near1, opp_near2]

def draw_arrows_flow_mask(img, flow, bin_num, vertices_root_pos_2d, dt, wave_dir, mask, grid_size, min=0):

	img_ret = img.copy()
	img_ret_seg = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
	img_ret_seg.fill(0)

	max_x = np.amax(flow[:][:][0])
	max_y = np.amax(flow[:][:][1])

	max_len = math.sqrt(max_x * max_x + max_y * max_y)

	flow_bins = flow_to_bins(flow, bin_num)
	max_bin = mat_mode_bin_min(flow_bins, bin_num, flow, min)

	if (wave_dir != -1): max_bin = wave_dir

	opposit, opp_near1, opp_near2 = max_bin_to_rip(max_bin, bin_num)


	for row in range(len(flow)):
		for col in range(len(flow[0])):

			x0 = col*grid_size + grid_size/2
			y0 = row*grid_size + grid_size/2

			if mask[math.floor(y0)][math.floor(x0)] == 0: 
				continue

			dx = flow[row][col][0]
			dy = flow[row][col][1]

			# if math.sqrt(dx*dx+dy*dy) >= min:

			dx, dy = unitize_xy(dx, dy)
			# dx = dx / math.sqrt(4)
			# dy = dy / math.sqrt(4)
			x0 = vertices_root_pos_2d[row][col][0]
			y0 = vertices_root_pos_2d[row][col][1]

			bin_dir = xy_to_bin(dx, dy, bin_num)
			if bin_dir == opposit or bin_dir == opp_near1 or bin_dir == opp_near2:
				if (bin_dir == opposit) : 
					color = (0, 0, 255)
				elif (bin_dir == opp_near1) : 
					color = (0, 255, 255)
				elif (bin_dir == opp_near2) : 
					color = (0, 255, 255)


				cv2.arrowedLine(img_ret, (int(x0), int(y0)), (int(x0 + dx * dt), int(y0 + dy * dt)), color, 3, tipLength = 0.5)

	img_ret = cv2.rectangle(img_ret, (5, 35), (180,95), (0,0,0), -1)
	cv2.arrowedLine(img_ret, (10, 50), (30, 50), (0, 0, 255), 3, tipLength = 0.5)
	cv2.arrowedLine(img_ret, (10, 80), (30, 80), (0, 255, 255), 3, tipLength = 0.5)

	img_ret = cv2.putText(img_ret, "rip current", (40,55), cv2.FONT_HERSHEY_SIMPLEX,  
                   0.6, (255,255,255), 1, cv2.LINE_AA)
	img_ret = cv2.putText(img_ret, "feeder current", (40,85), cv2.FONT_HERSHEY_SIMPLEX,  
                   0.6, (255,255,255), 1, cv2.LINE_AA)


	return img_ret, [opposit, opp_near1, opp_near2]

def draw_arrows_unit_flow(img, flow, bin_num, vertices_root_pos_2d, dt):

	img_ret = img.copy()

	for row in range(len(flow)):
		for col in range(len(flow[0])):
			dx = flow[row][col][0]
			dy = flow[row][col][1]
			dx, dy = unitize_xy(dx, dy)
			x0 = vertices_root_pos_2d[row][col][0]
			y0 = vertices_root_pos_2d[row][col][1]

			cv2.arrowedLine(img_ret, (int(x0), int(y0)), (int(x0 + dx * dt), int(y0 + dy * dt)), (255, 0, 0), 3, tipLength = 0.5)

	return img_ret

#   1 
# 0    2 
# 5    3
#   4
def draw_arrows_bins(img, flow_bins, bin_num, vertices_root_pos_2d, dt, wave_dir):

	img_ret = img.copy()
	img_ret_seg = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
	img_ret_seg.fill(0)

	max_bin = mat_mode_bin(flow_bins, bin_num)

	if (wave_dir != -1): max_bin = wave_dir

	opposit, opp_near1, opp_near2 = max_bin_to_rip(max_bin, bin_num)

	count_max = 0
	count_near1 = 0
	count_near2 = 0
	count_opposite = 0

	for row in range(len(flow_bins)):
		for col in range(len(flow_bins[0])):
			bin_dir = flow_bins[row][col]
			dx, dy = bin_to_flow(bin_dir, bin_num)
			x0 = vertices_root_pos_2d[row][col][0]
			y0 = vertices_root_pos_2d[row][col][1]

			if (bin_dir == max_bin) : 
				color = (0, 0, 0)
				count_max += 1
			elif (bin_dir == opposit) : 
				color = (0, 0, 255)
				count_opposite += 1
			elif (bin_dir == opp_near1) : 
				color = (0, 200, 255)
				count_near1 += 1
			elif (bin_dir == opp_near2) : 
				color = (0, 255, 200)
				count_near2 += 1
			else : color = (128, 128, 128)

			cv2.arrowedLine(img_ret, (int(x0), int(y0)), (int(x0 + dx * dt), int(y0 + dy * dt)), color, 3, tipLength = 0.5)

			if (bin_dir == opposit or bin_dir == opp_near1 or bin_dir == opp_near2): 
				cv2.rectangle(img_ret_seg, (int(x0), int(y0)), (int(x0 + 20), int(y0 + 20)), (255, 255, 255), -1)


	img_ret = cv2.putText(img_ret, str(count_max), (30,50), cv2.FONT_HERSHEY_SIMPLEX,  
                   0.6, (0, 0, 0), 2, cv2.LINE_AA) 
	img_ret = cv2.putText(img_ret, str(count_near1), (30,70), cv2.FONT_HERSHEY_SIMPLEX,  
                   0.6, (0, 0, 255), 2, cv2.LINE_AA) 
	img_ret = cv2.putText(img_ret, str(count_near2), (30,90), cv2.FONT_HERSHEY_SIMPLEX,  
                   0.6, (0, 200, 255), 2, cv2.LINE_AA) 
	img_ret = cv2.putText(img_ret, str(count_opposite), (30,110), cv2.FONT_HERSHEY_SIMPLEX,  
                   0.6, (0, 255, 200), 2, cv2.LINE_AA) 

	return img_ret, img_ret_seg