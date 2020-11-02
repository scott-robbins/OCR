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
	return ndi.gaussian_laplace(iconv, sigma=threshold)

def video_to_images(video_file):
	if os.path.isdir(os.getcwd()+'/frames'):
		os.system('rm -rf frames')
	os.mkdir('frames')
	cmd1 = 'ffmpeg -i '+video_file+' -vf fps=30 frames/frame%03d.png'
	os.system(cmd1+' >> /dev/null')

def process_frames():
	if os.path.isdir(os.getcwd()+'/outputs'):
		os.system('rm -rf outputs')
	os.mkdir('outputs'); index = 0
	progress = tqdm(total=len(os.listdir('frames')))
	for raw_img in os.listdir('frames'):
		im = os.getcwd()+'/frames/'+raw_img
		im_arr = np.array(plt.imread(im))
		# Now process and save output
		misc.imsave('outputs/frame%03d.png' % index, pre_process_image(im_arr, 4))
		progress.update(1)
		index += 1
	test_gif = 'ffmpeg -i outputs/frame%03d.png -i palette.png -filter_complex "fps=15,scale=480:-1:flags=lanczos[x];[x][1:v]paletteuse" animated.mp4'
	makep = 'ffmpeg -i outputs/frame000.png -vf palettegen=16 palette.png'
	os.system(makep + ' >> /dev/null')
	os.system(test_gif + ' >> /dev/null')
	os.system('rm *.png; rm -rf frames outputs')
