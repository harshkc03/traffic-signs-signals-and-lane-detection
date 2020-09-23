import time
import tensorflow as tf
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
	tf.config.experimental.set_memory_growth(physical_devices[0], True)
from absl import app, flags, logging
from absl.flags import FLAGS
import core.utils as utils
from core.yolov4 import filter_boxes
from tensorflow.python.saved_model import tag_constants
from PIL import Image
import cv2
import numpy as np
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

flags.DEFINE_string('framework', 'tf', '(tf, tflite, trt')
flags.DEFINE_string('weights', 'yolov4-tiny-416', 'path to weights file')
flags.DEFINE_integer('size', 416, 'resize images to')
flags.DEFINE_boolean('tiny', True, 'yolo or yolo-tiny')
flags.DEFINE_string('model', 'yolov4', 'yolov3 or yolov4')
flags.DEFINE_string('video', './data/test-video5.mp4', 'path to input video')
flags.DEFINE_float('iou', 0.45, 'iou threshold')
flags.DEFINE_float('score', 0.50, 'score threshold')
flags.DEFINE_string('output', 'vidout.avi', 'path to output video')
flags.DEFINE_string('output_format', 'XVID', 'codec used in VideoWriter when saving video to file')
flags.DEFINE_boolean('dis_cv2_window', False, 'disable cv2 window during the process') # this is good for the .ipynb

def region_of_interest(img, vertices):

	#Code snippet to crop out the region of interest
	mask = np.zeros_like(img)
	match_mask_color = 255
	cv2.fillPoly(mask, vertices, match_mask_color)
	masked_image = cv2.bitwise_and(img, mask)
	return masked_image

def draw_the_lines(img, lines):

	#Make a copy of the image
	img = np.copy(img)

	#Create blank image
	blank_image = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)

	#Iterate through detected lines and draw them
	for line in lines:
		angle = int(np.arctan((line[0][3]-line[0][1])/(line[0][2]-line[0][0]))*(180/3.14))
		if angle < 0:
			if not -85<angle<-35:
				continue 
		else:
			if not 35<angle<85:
				continue
		cv2.line(blank_image, (line[0][0],line[0][1]), (line[0][2],line[0][3]), (0, 255, 0), thickness=5)

	#Combine blank image and copy of the image and return it
	img = cv2.addWeighted(img, 0.8, blank_image, 1, 0.0)
	return img

def auto_canny(image, sigma=0.33):
	# compute the median of the single channel pixel intensities
	v = np.median(image)

	# apply automatic Canny edge detection using the computed median
	lower = int(max(0, (1.0 - sigma) * v))
	upper = int(min(255, (1.0 + sigma) * v))
	edged = cv2.Canny(image, lower, upper)

	# return the edged image
	return edged

def main(_argv):

	#Configure tensorflow and initilize variables
	config = ConfigProto()
	config.gpu_options.allow_growth = True
	session = InteractiveSession(config=config)
	STRIDES, ANCHORS, NUM_CLASS, XYSCALE = utils.load_config(FLAGS)
	input_size = FLAGS.size
	video_path = FLAGS.video
	tot_frames = 0
	tot_time = 0

	#Read video from path
	print("Video from: ", video_path )
	vid = cv2.VideoCapture(video_path)

	#Load YOLO converted model
	saved_model_loaded = tf.saved_model.load(FLAGS.weights, tags=[tag_constants.SERVING])
	infer = saved_model_loaded.signatures['serving_default']
	
	#If output video is True, initialize the required variables
	if FLAGS.output:
		# by default VideoCapture returns float instead of int
		width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
		height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
		fps = int(vid.get(cv2.CAP_PROP_FPS))
		codec = cv2.VideoWriter_fourcc(*FLAGS.output_format)
		out = cv2.VideoWriter(FLAGS.output, codec, fps, (width, height))

	#Loop through video frames
	while True:

		#Variable which will be used for calculating time
		prev_time = time.time()

		#Read a frame from the video
		_, frame = vid.read()

		#If no frame is found, break out of the loop
		if frame is None:
		  print("Done !!")
		  break
		
		#Image pre-processing
		frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
		image = Image.fromarray(frame)
		image_data = (cv2.resize(frame, (input_size, input_size)) / 255)[np.newaxis, ...].astype(np.float32)

		#Perform YOLO detection
		batch_data = tf.constant(image_data)
		pred_bbox = infer(batch_data)
		[(boxes, pred_conf)] = [(value[:, :, 0:4], value[:, :, 4:]) for key, value in pred_bbox.items()]

		boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
			boxes=tf.reshape(boxes, (tf.shape(boxes)[0], -1, 1, 4)),
			scores=tf.reshape(
				pred_conf, (tf.shape(pred_conf)[0], -1, tf.shape(pred_conf)[-1])),
			max_output_size_per_class=50,
			max_total_size=50,
			iou_threshold=FLAGS.iou,
			score_threshold=FLAGS.score
		)
		pred_bbox = [boxes.numpy(), scores.numpy(), classes.numpy(), valid_detections.numpy()]

		#Draw the predicted boxes
		image,pred_names = utils.draw_bbox(frame, pred_bbox)

		#Image pre-processing for lane detection
		result = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
		height = result.shape[0]
		width = result.shape[1]

		xtl, ytl = width*0.454, height*0.625
		xtr, ytr = width*0.538, height*0.625
		xbl, ybl = width*0.219, height*0.996
		xbr, ybr = width*0.805, height*0.996

		region_of_interest_vertices = [
			(xtl, ytl), (xtr, ytr),
			(xbr, ybr), (xbl, ybl)
		]

		gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

		#Apply canny on the grayscale image
		canny_image = auto_canny(gray_image)

		#Crop the region of interest
		cropped_image = region_of_interest(canny_image,
			  np.array([region_of_interest_vertices], np.int32),)

		#Finding out lines using Hough transform
		lines = cv2.HoughLinesP(cropped_image,
				  rho=1,
				  theta=np.pi/180,
				  threshold=25,
				  lines=np.array([]),
				  minLineLength=4,
				  maxLineGap=2000)

		#If lane lines are found, draw them on the image
		if not lines is None:
			image_with_lines = draw_the_lines(result, lines)
		else:
			image_with_lines = result.copy()

		#Debugging/Info printing
		curr_time = time.time()
		exec_time = curr_time - prev_time
		result = np.asarray(image)
		info = ", Time: %.2f ms" %(1000*exec_time)
		tot_frames += 1
		tot_time += exec_time

		print(" ")
		print("Frame no: "+str(tot_frames)+info+", FPS: %.2f " %(1/exec_time))
		print("Predictions: "+str(pred_names))

		if not FLAGS.dis_cv2_window:
			cv2.namedWindow("result", cv2.WINDOW_AUTOSIZE)
			cv2.imshow("result", image_with_lines)
			if cv2.waitKey(1) & 0xFF == ord('q'): break

		#Write the frame on the output video
		if FLAGS.output:
			out.write(image_with_lines)

	#Debugging/Info printing
	print(" ")
	print("Total frames: "+str(tot_frames))
	print("Total time: %.2f s" %(tot_time))
	print("Average time: %.2f ms" %(1000*(tot_time/tot_frames)))
	print("Average FPS: %.2f" %(1/(tot_time/tot_frames)))

if __name__ == '__main__':
	try:
		app.run(main)
	except SystemExit:
		pass
