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

import my_flow

def main(video, outpath, height, window_size, grid_size, bin_size, wave_dir, mask_path):

	# init dict to track time for every stage at each iteration
	timers = {
		"full pipeline": [],
		"reading": [],
		"pre-process": [],
		"optical flow": [],
		"post-process": [],
	}
	
	print("reading ", video)

	filename = os.path.splitext(os.path.basename(video))[0]
	# if not os.path.exists(outpath + "/" + filename):
	# 	os.makedirs(outpath + "/" + filename)

	# init video capture with video
	cap = cv2.VideoCapture(video)

	# get default video FPS
	fps = cap.get(cv2.CAP_PROP_FPS)
	print(fps, " fps")

	# get total number of video frames
	num_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)

	# read the first frame
	ret, frame = cap.read()

	# proceed if frame reading was successful
	if not ret: 
		print(ret)
		return

	# width after resize
	width = math.floor(frame.shape[1] * 
			height / (frame.shape[0]))

	video_out1 = cv2.VideoWriter(outpath + "/" + filename + "_" + str(bin_size) + "_arrows.avi", cv2.VideoWriter_fourcc(*'MJPG'), fps, (width, height)) 
	video_out2 = cv2.VideoWriter(outpath + "/" + filename + "_" + str(bin_size) + "_color.avi", cv2.VideoWriter_fourcc(*'MJPG'), fps, (width, height)) 

	# resize frame
	resized_frame = cv2.resize(frame, (width, height))

	gpu_frame = cv2.cuda_GpuMat()
	gpu_frame.upload(resized_frame)

	previous_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2GRAY)

	gpu_previous = cv2.cuda_GpuMat()
	gpu_previous.upload(previous_frame)

	# create gpu_hsv output for optical flow
	gpu_hsv = cv2.cuda_GpuMat(gpu_frame.size(), cv2.CV_32FC3)
	gpu_hsv_8u = cv2.cuda_GpuMat(gpu_frame.size(), cv2.CV_8UC3)

	gpu_h = cv2.cuda_GpuMat(gpu_frame.size(), cv2.CV_32FC1)
	gpu_s = cv2.cuda_GpuMat(gpu_frame.size(), cv2.CV_32FC1)
	gpu_v = cv2.cuda_GpuMat(gpu_frame.size(), cv2.CV_32FC1)

	# set saturation to 1
	gpu_s.upload(np.ones_like(previous_frame, np.float32))

	cpu_flow_array = []

	color_wheel = cv2.imread("images/colorwheel_hue.png")

	mask_img = cv2.imread(mask_path, 0)
	mask_img_bgr = cv2.cvtColor(mask_img, cv2.COLOR_GRAY2BGR)


	# Calculate number of arrows
	arrow_count_w = math.floor(width / grid_size)
	arrow_count_h = math.floor(height / grid_size)

	print(arrow_count_h, arrow_count_w)

	# 1D array of vertices position
	vertices_root =  np.array([], dtype=np.float32)
	vertices_root_pos_2d = np.zeros((arrow_count_h, arrow_count_w, 2), dtype=np.float32)
	for row in range (0, arrow_count_h):
		for col in range (0, arrow_count_w):
			x0 = col*grid_size + grid_size/2
			y0 = row*grid_size + grid_size/2
			if row == 0 and col == 0: 
				vertices_root =  np.array([x0, y0], dtype=np.float32)
			else:
				vertices_root = np.vstack((vertices_root, np.array([x0, y0], dtype=np.float32)))
			vertices_root_pos_2d[row][col][0] = x0
			vertices_root_pos_2d[row][col][1] = y0

	buffer_flow = np.zeros((window_size, arrow_count_h, arrow_count_w, 2), dtype=np.float32)
	buffer_unit_flow = np.zeros((window_size, arrow_count_h, arrow_count_w, 2), dtype=np.float32)

	sum_flow = np.zeros((arrow_count_h, arrow_count_w, 2), dtype=np.float32)
	sum_unit_flow = np.zeros((arrow_count_h, arrow_count_w, 2), dtype=np.float32)

	# stores the bin direction for each frame for windowsize frames
	buffer_bin = np.zeros((window_size, arrow_count_h, arrow_count_w), dtype=np.uint16)

	# stores the number of frames in each bin for the previous windowsize frames
	bin_hist = np.zeros((arrow_count_h, arrow_count_w, bin_num), dtype=np.uint16)

	# bin counts are weighted based on their magnitude
	buffer_bin_weighted_hist = np.zeros((window_size, arrow_count_h, arrow_count_w, bin_num), dtype=np.uint16)

	sum_bin_weighted_hist = np.zeros((arrow_count_h, arrow_count_w, bin_num), dtype=np.float32)


	lk_params = dict(winSize = (15, 15),
					maxLevel = 5,
					criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

	# convert to gray
	old_gray = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2GRAY)

	frame_count = 0
	while True:

		ret, frame = cap.read()

		# if frame reading was not successful, break
		if not ret:
			break

		resized_frame = cv2.resize(frame, (width, height))


		gray_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2GRAY)

		gpu_frame.upload(resized_frame)
		#gpu_frame = cv2.cuda.resize(gpu_frame, (width, height))
		gpu_current = cv2.cuda.cvtColor(gpu_frame, cv2.COLOR_BGR2GRAY)
		gpu_flow = cv2.cuda_FarnebackOpticalFlow.create(
			4, 0.5, False, 5, 10, 7, 1.5, 0,
		)
		# calculate optical flow
		gpu_flow = cv2.cuda_FarnebackOpticalFlow.calc(
			gpu_flow, gpu_previous, gpu_current, None,
		)
		cpu_flow = gpu_flow.download()
		cpu_flow = my_flow.zero_edge_flow(cpu_flow, 50)
		#cpu_flow = my_flow.remove_outlier(cpu_flow, 1.1)
		'''
		create aggregated flow 
		'''
		cpu_flow_divided = cpu_flow / window_size
		cpu_flow_array.append(cpu_flow_divided)

		if frame_count == 0:
			cpu_flow_average = cpu_flow_divided.copy()
		elif  frame_count < window_size:
			cpu_flow_average += cpu_flow_divided
		else:
			cpu_flow_average += cpu_flow_divided
			cpu_flow_average -= cpu_flow_array[0]
			cpu_flow_array.pop(0)


		cpu_flow_average_angle, cpu_flow_average_magnitude = my_flow.calc_angle_from_flow_cpu(cpu_flow_average)
	
		gpu_previous = gpu_current



		# Optical Flow LK
		new_points, status, error = cv2.calcOpticalFlowPyrLK(old_gray, gray_frame, vertices_root, None, **lk_params)


		# Calculate flow at each point
		flow = np.zeros((arrow_count_h, arrow_count_w, 2), dtype=np.float32)
		for i in range(len(new_points)):
			row = math.floor(i / arrow_count_w)
			col = math.floor(i % arrow_count_w)
			# print(row, col)
			flow[row][col][0] = new_points[i][0] - vertices_root[i][0]
			flow[row][col][1] = new_points[i][1] - vertices_root[i][1]

		# Calculate unit flow
		flow_bins = my_flow.flow_to_bins(flow, bin_num)

		current_buffer_i = frame_count % window_size


		# update total flow
		sum_flow -= buffer_flow[current_buffer_i]
		buffer_flow[current_buffer_i] = flow
		sum_flow += flow


		# threshold_min_mag = min(window_size, frame_count+1) * 0.5
		vis_flow, rip_bins = my_flow.draw_arrows_flow_mask(resized_frame, sum_flow, bin_num, vertices_root_pos_2d, grid_size * 0.8, wave_dir, mask_img, grid_size)

		
		cpu_flow_average_bgr = my_flow.calc_bgr_from_angle_magnitude_rip(cpu_flow_average_angle, cpu_flow_average_magnitude, rip_bins)
		cpu_flow_average_bgr_strong = my_flow.calc_bgr_from_angle_magnitude_rip(cpu_flow_average_angle, np.ones_like(cpu_flow_average_angle, np.float32), rip_bins)

		cpu_flow_average_bgr = (mask_img_bgr > 0) * cpu_flow_average_bgr
		cpu_flow_average_bgr_strong = (mask_img_bgr > 0) * cpu_flow_average_bgr_strong

		cpu_flow_overlay = cpu_flow_average_bgr.copy()
		cv2.addWeighted(cpu_flow_average_bgr, 1, resized_frame, 1, 0, cpu_flow_overlay)
		my_flow.add_color_wheel(cpu_flow_overlay, color_wheel)
		cpu_flow_overlay = cv2.putText(cpu_flow_overlay, str(frame_count), (30,30), cv2.FONT_HERSHEY_SIMPLEX,  
                   0.8, (255,255,255), 1, cv2.LINE_AA)

		vis_flow = cv2.putText(vis_flow, str(frame_count), (30,30), cv2.FONT_HERSHEY_SIMPLEX,  
                   0.8, (255,255,255), 1, cv2.LINE_AA) 

		old_gray = gray_frame.copy()


		# visualization
		cv2.imshow("vis_flow", vis_flow)
		cv2.imshow("flow", cpu_flow_overlay)

		video_out1.write(vis_flow)
		video_out2.write(cpu_flow_overlay)

		k = cv2.waitKey(1)
		if k == 27:
			break

		frame_count += 1

	video_out1.release()
	video_out2.release()

	# release the capture
	cap.release()

	# destroy all windows
	cv2.destroyAllWindows()


if __name__ == "__main__":

	# init argument parser
	parser = argparse.ArgumentParser(description="Rip Currents Detection with CUDA enabled")

	parser.add_argument(
		"--video", help="path to .mp4 video file", required=True, type=str,
	)

	parser.add_argument(
		"--out", help="path and file name of the output file without .mp4", required=True, type=str,
	)

	parser.add_argument(
		"--height", help="resized height of the output", required=False, type=int, default=480,
	)

	parser.add_argument(
		"--window", help="resized height of the output", required=False, type=int, default=900,
	)

	parser.add_argument(
		"--grid", help="grid per pixel", required=False, type=int, default=20,
	)

	parser.add_argument(
		"--bins", help="number of bins", required=False, type=int, default=6,
	)

	parser.add_argument(
		"--wave_dir", help="incoming dir", required=False, type=int, default=-1,
	)

	parser.add_argument(
		"--mask", help="mask file path", required=True, type=str, default="",
	)

	# parsing script arguments
	args = parser.parse_args()
	video = args.video
	outpath = args.out
	height = args.height
	window_size = args.window
	grid_size = args.grid
	bin_num = args.bins
	wave_dir = args.wave_dir
	mask_path = args.mask


	# run pipeline
	main(video, outpath, height, window_size, grid_size, bin_num, wave_dir, mask_path)