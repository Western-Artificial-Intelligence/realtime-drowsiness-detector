import cv2
import numpy as np

class Camera:
	def __init__(self, index=0, width=640, height=480):
		self.cap = cv2.VideoCapture(index)
		self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
		self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

	def read(self) -> tuple[bool, np.ndarray]:
		return self.cap.read()
	
	def release(self) -> None:
		self.cap.release()