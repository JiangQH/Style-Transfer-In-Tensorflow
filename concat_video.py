import cv2
import numpy as np
video_out = './'
def concat(video1, video2):
	vidcap1 = cv2.VideoCapture(video1)
	vidcap2 = cv2.VideoCapture(video2)
	fps = vidcap1.get(cv2.cv.CV_CAP_PROP_FPS)
	width = vidcap1.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH)
	height = vidcap1.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT)
	outvideo = cv2.VideoWriter(osp.join(video_out, 'concated.avi'), cv2.cv.CV_FOURCC('M','J','P','G'), fps, (int(width), int(height)))
	while True:
		success1, image1 = vidcap1.read()
		success2, image2 = vidcap2.read()
		if not (success1 and success2):
			break
		# concat them
		concated = np.concatenate((image1, image2), axis=1)
		outvideo.write(concated)
	cv2.destroyAllWindows()
	outvideo.release()
