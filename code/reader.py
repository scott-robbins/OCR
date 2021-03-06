import matplotlib.pyplot as plt
import scipy.ndimage as ndi 
import scipy.misc as misc 
from tqdm import tqdm
import numpy as np 
import time 
import sys 
import os 


def load_image(file_name, show):
	if os.path.isfile(file_name):
		im_arr = np.array(plt.imread(file_name))
	else:
		print('[!!] Cannot find %s' % file_name)
	if show:
		plt.title(file_name.split('/')[-1])
		plt.imshow(im_arr)
		plt.show()
	return im_arr


def load_test_images(show):
	images = {}
	img_lib = '/'.join(os.getcwd().split('/')[0:-1])+'/images/'
	if os.path.isdir(img_lib):
		for im in os.listdir(img_lib):
			im_name  = img_lib+im
			img = load_image(im_name, show)
			image = {'arr': img,
					 'name': im_name}
			images[im] = image
	return images

def pre_process_image(img_data, threshold):
	arr = img_data['arr']
	im = abs(-1*(arr[:,:,0]/16 + arr[:,:,1]/16 + arr[:,:,2]/16))
	f1 = np.ones((6,6))
	f2 = np.ones((8,8))
	iconv = ndi.convolve(im,f2,origin=0) - ndi.convolve(im,f1,origin=0)
	return ndi.gaussian_laplace(iconv, sigma=threshold)

def main():
	debug = False

	# Load OCR Test Images
	test_images = load_test_images(debug)
	test_image = test_images['PXL_OCR_TEST.jpg']
	print('Testing OCR Algorithms with image: %s' % test_image['name'].split('/')[-1])
	if '-test' in sys.argv:
		test_array = test_image['arr']
		print(np.array(test_array.shape))

		threshold = 10
		edges = pre_process_image(test_image, threshold)
		plt.imshow(edges > 0, cmap='gray')
		plt.show()

	if '-filter-video' in sys.argv:
		ind = list(reversed(range(150)))
		progress = tqdm(total=len(ind))
		# Attempt 1: Edge detect with blur conv subtract
		for j in ind: 
			misc.imsave('test_conv_%03d.png' %j, abs(pre_process_image(test_image, j/10)))
			progress.update(1)
		test_gif = 'ffmpeg -i test_conv_%03d.png -i palette.png -filter_complex "fps=20,scale=720:-1:flags=lanczos[x];[x][1:v]paletteuse" test2.gif'
		makep = 'ffmpeg -i test_conv_011.png -vf palettegen=16 palette.png'
		os.system(makep + ' >> /dev/null')
		os.system(test_gif + ' >> /dev/null')
		os.system('rm *.png')	

	if '-process-video' in sys.argv and len(sys.argv) >= 2:
		video_in = sys.argv[2]
		print('Processing %s' % video_in)
		# convert to raw images
		import processor 
		processor.video_to_images(video_in)
		processor.process_frames()

if __name__ == '__main__':
	main()
