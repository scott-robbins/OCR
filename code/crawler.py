import matplotlib.pyplot as plt
import scipy.ndimage as ndi 
import scipy.misc as misc 
from tqdm import tqdm
import numpy as np 
import time 
import sys 
import os 

def pre_process_image(arr, threshold):
	im = abs(-1*(arr[:,:,0]/16 + arr[:,:,1]/16 + arr[:,:,2]/16))
	f1 = np.ones((3,3))
	f2 = np.ones((6,6))
	iconv = ndi.convolve(im,f2,origin=0) - ndi.convolve(im,f1,origin=0)
	return abs(ndi.gaussian_laplace(iconv, sigma=threshold))

def collect_points(img_data):
	points = []
	detected = np.zeros((img_data.shape[0],img_data.shape[1],3))
	neighbors = ndi.convolve(img_data,np.ones((8,8)))
	for x in range(img_data.shape[0]):
		for y in range(img_data.shape[1]):	
			try:
				nn = neighbors[x,y]
				if img_data[x,y] > 1 or nn >= 63:
					points.append([x,y])
					detected[x,y] = [1,0,0]
			except IndexError:
				pass
	return ndi.gaussian_laplace(detected,sigma=0.3)[:,:,0] > 0,detected, points


def main():
	test_image = 'PXL_OCR_NAME.jpg'

	# load image 
	raw_im = np.array(plt.imread('../images/%s'%test_image)).astype(np.int32)
	im_arr = pre_process_image(raw_im, raw_im.mean()/23.)

	# trace the solid shapes 
	edges, shape, inner_pts = collect_points(im_arr)
	ratio = len(inner_pts)/float(raw_im.shape[0]*raw_im.shape[1])
	print('[*] %d points found' % len(inner_pts))
	print('[*] %f percent of image excluded' % (100-ratio*100))

	# show image	
	f,ax = plt.subplots(3,1, sharex=True,sharey=True)
	ax[0].imshow(raw_im); 	ax[0].set_title('Original Image')
	ax[1].imshow(shape);	ax[1].set_title('Text Detected')
	ax[2].imshow(edges); 	ax[2].set_title('Text Edges')
	plt.show()


if __name__ == '__main__':
	main()
