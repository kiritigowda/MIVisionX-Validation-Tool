import sys
#setup python path for RALI
sys.path.append('/opt/rocm/mivisionx/rali/python/')

import os
from rali import *
from rali_image_iterator import *
from rali_common import *

#batch size = 64
raliList_mode1_64 = ['original', 'warpAffine', 'contrast', 'rain', 
			'brightness', 'colorTemp', 'exposure', 'vignette', 
			'fog', 'snow', 'pixelate', 'SnPNoise', 
			'gamma', 'rotate', 'jitter', 'blend',
			'rotate45+resize', 'rotate45+warpAffine', 'rotate45+contrast', 'rotate45+rain', 
			'rotate45+brightness', 'rotate45+colorTemp', 'rotate45+exposure', 'rotate45+vignette', 
			'rotate45+fog', 'rotate45+snow', 'rotate45+pixelate', 'rotate45+SnPNoise', 
			'rotate45+gamma', 'rotate45+rotate', 'rotate45+jitter', 'rotate45+blend',
			'flip+resize', 'flip+warpAffine', 'flip+contrast', 'flip+rain', 
			'flip+brightness', 'flip+colorTemp', 'flip+exposure', 'flip+vignette', 
			'flip+fog', 'flip+snow', 'flip+pixelate', 'flip+SnPNoise', 
			'flip+gamma', 'flip+rotate', 'flip+jitter', 'flip+blend',			
			'rotate135+resize', 'rotate135+warpAffine', 'rotate135+contrast', 'rotate135+rain', 
			'rotate135+brightness', 'rotate135+colorTemp', 'rotate135+exposure', 'rotate135+vignette', 
			'rotate135+fog', 'rotate135+snow', 'rotate135+pixelate', 'rotate135+SnPNoise', 
			'rotate135+gamma', 'rotate135+rotate', 'rotate135+jitter', 'rotate135+blend']
raliList_mode2_64 = ['original', 'warpAffine', 'contrast', 'rain', 
			'brightness', 'colorTemp', 'exposure', 'vignette', 
			'fog', 'snow', 'pixelate', 'SnPNoise', 
			'gamma', 'rotate', 'jitter', 'blend',
			'warpAffine+original', 'warpAffine+warpAffine', 'warpAffine+contrast', 'warpAffine+rain', 
			'warpAffine+brightness', 'warpAffine+colorTemp', 'warpAffine+exposure', 'warpAffine+vignette', 
			'warpAffine+fog', 'warpAffine+snow', 'pixelate', 'warpAffine+SnPNoise', 
			'warpAffine+gamma', 'warpAffine+rotate', 'warpAffine+jitter', 'warpAffine+blend',
			'fishEye+original', 'fishEye+warpAffine', 'fishEye+contrast', 'fishEye+rain', 
			'fishEye+brightness', 'fishEye+colorTemp', 'fishEye+exposure', 'fishEye+vignette', 
			'fishEye+fog', 'fishEye+snow', 'fishEye+pixelate', 'fishEye+SnPNoise', 
			'fishEye+gamma', 'fishEye+rotate', 'fishEye+jitter', 'fishEye+blend',
			'lensCorrection+original', 'lensCorrection+warpAffine', 'lensCorrection+contrast', 'lensCorrection+rain', 
			'lensCorrection+brightness', 'lensCorrection+colorTemp', 'exposure', 'lensCorrection+vignette', 
			'lensCorrection+fog', 'lensCorrection+snow', 'lensCorrection+pixelate', 'lensCorrection+SnPNoise', 
			'lensCorrection+gamma', 'lensCorrection+rotate', 'lensCorrection+jitter', 'lensCorrection+blend',]
raliList_mode3_64 = ['original', 'warpAffine', 'contrast', 'rain', 
			'brightness', 'colorTemp', 'exposure', 'vignette', 
			'fog', 'snow', 'pixelate', 'SnPNoise', 
			'gamma', 'rotate', 'jitter', 'blend',
			'colorTemp+original', 'colorTemp+warpAffine', 'colorTemp+contrast', 'colorTemp+rain', 
			'colorTemp+brightness', 'colorTemp+colorTemp', 'colorTemp+exposure', 'colorTemp+vignette', 
			'colorTemp+fog', 'colorTemp+snow', 'colorTemp+pixelate', 'colorTemp+SnPNoise', 
			'colorTemp+gamma', 'colorTemp+rotate', 'colorTemp+jitter', 'colorTemp+blend',
			'colorTemp+original', 'colorTemp+warpAffine', 'colorTemp+contrast', 'colorTemp+rain', 
			'colorTemp+brightness', 'colorTemp+colorTemp', 'colorTemp+exposure', 'colorTemp+vignette', 
			'colorTemp+fog', 'colorTemp+snow', 'colorTemp+pixelate', 'colorTemp+SnPNoise', 
			'colorTemp+gamma', 'colorTemp+rotate', 'colorTemp+jitter', 'colorTemp+blend',
			'warpAffine+original', 'warpAffine+warpAffine', 'warpAffine+contrast', 'warpAffine+rain', 
			'warpAffine+brightness', 'warpAffine+colorTemp', 'warpAffine+exposure', 'warpAffine+vignette', 
			'warpAffine+fog', 'warpAffine+snow', 'pixelate', 'warpAffine+SnPNoise', 
			'warpAffine+gamma', 'warpAffine+rotate', 'warpAffine+jitter', 'warpAffine+blend']
#batch size = 16
raliList_mode1_16 = ['original', 'warpAffine', 'contrast', 'rain', 
			'brightness', 'colorTemp', 'exposure', 'vignette', 
			'fog', 'snow', 'pixelate', 'SnPNoise', 
			'gamma', 'rotate', 'jitter', 'blend']
raliList_mode2_16 = ['original', 'warpAffine', 'contrast', 'contrast+rain', 
			'brightness', 'brightness+colorTemp', 'exposure', 'exposure+vignette', 
			'fog', 'fog+snow', 'pixelate', 'pixelate+SnPNoise', 
			'gamma', 'rotate', 'rotate+jitter', 'blend']
raliList_mode3_16 = ['original', 'warpAffine', 'contrast', 'warpAffine+rain', 
			'brightness', 'colorTemp', 'exposure', 'vignette', 
			'fog', 'vignette+snow', 'pixelate', 'gamma',
			'SnPNoise+gamma', 'rotate', 'jitter+pixelate', 'blend']

# Class to initialize Rali and call the augmentations 
class DataLoader(RaliGraph):
	def __init__(self, input_path, rali_batch_size, model_batch_size, input_color_format, affinity, image_validation, h_img, w_img, raliMode, loop_parameter,
				 tensor_layout = TensorLayout.NCHW, reverse_channels = False, multiplier = [1.0,1.0,1.0], offset = [0.0, 0.0, 0.0]):
		RaliGraph.__init__(self, rali_batch_size, affinity)
		self.validation_dict = {}
		self.process_validation(image_validation)
		self.setSeed(0)

		#params for contrast
		self.min_param = RaliIntParameter(0)
		self.max_param = RaliIntParameter(255)
		#param for brightness
		self.alpha_param = RaliFloatParameter(0.0)
		#param for colorTemp		
		self.adjustment_param = RaliIntParameter(0)
		#param for exposure
		self.shift_param = RaliFloatParameter(0.0)
		#param for SnPNoise
		self.sdev_param = RaliFloatParameter(0.0)
		#param for gamma
		self.gamma_shift_param = RaliFloatParameter(0.0)
		#param for rotate
		self.degree_param = RaliFloatParameter(0.0)

		#rali list of augmentation
		self.rali_list = None

		if model_batch_size == 16:
			if raliMode == 1:
				self.jpg_img = self.jpegFileInput(input_path, input_color_format, False, loop_parameter, 0)
				self.input = self.resize(self.jpg_img, h_img, w_img, True)

				self.warped = self.warpAffine(self.input,True)

				self.contrast_img = self.contrast(self.input,True, self.min_param, self.max_param)
				self.rain_img = self.rain(self.input, True)

				self.bright_img = self.brightness(self.input,True, self.alpha_param)
				self.temp_img = self.colorTemp(self.input, True, self.adjustment_param)

				self.exposed_img = self.exposure(self.input, True, self.shift_param)
				self.vignette_img = self.vignette(self.input, True)
				self.fog_img = self.fog(self.input, True)
				self.snow_img = self.snow(self.input, True)

				self.pixelate_img = self.pixelate(self.input, True)
				self.snp_img = self.SnPNoise(self.input, True, self.sdev_param)
				self.gamma_img = self.gamma(self.input, True, self.gamma_shift_param)

				self.rotate_img = self.rotate(self.input, True, self.degree_param)
				self.jitter_img = self.jitter(self.input, True)
				
				self.blend_img = self.blend(self.input, self.contrast_img, True)
			elif raliMode == 2:
				self.jpg_img = self.jpegFileInput(input_path, input_color_format, False, loop_parameter, 0)
				self.input = self.resize(self.jpg_img, h_img, w_img, True)

				self.warped = self.warpAffine(self.input,True)

				self.contrast_img = self.contrast(self.input,True, self.min_param, self.max_param)
				self.rain_img = self.rain(self.contrast_img, True)

				self.bright_img = self.brightness(self.input,True, self.alpha_param)
				self.temp_img = self.colorTemp(self.bright_img, True, self.adjustment_param)

				self.exposed_img = self.exposure(self.input, True, self.shift_param)
				self.vignette_img = self.vignette(self.exposed_img, True)
				self.fog_img = self.fog(self.input, True)
				self.snow_img = self.snow(self.fog_img, True)

				self.pixelate_img = self.pixelate(self.input, True)
				self.snp_img = self.SnPNoise(self.pixelate_img, True, self.sdev_param)
				self.gamma_img = self.gamma(self.input, True, self.gamma_shift_param)

				self.rotate_img = self.rotate(self.input, True, self.degree_param)
				self.jitter_img = self.jitter(self.rotate_img, True)

				self.blend_img = self.blend(self.rotate_img, self.warped, True)
			elif raliMode == 3:
				self.jpg_img = self.jpegFileInput(input_path, input_color_format, False, loop_parameter, 0)
				self.input = self.resize(self.jpg_img, h_img, w_img, True)
				self.warped = self.warpAffine(self.input,True)

				self.contrast_img = self.contrast(self.input,True, self.min_param, self.max_param)
				self.rain_img = self.rain(self.warped, True)

				self.bright_img = self.brightness(self.input,True, self.alpha_param)
				self.temp_img = self.colorTemp(self.input, True, self.adjustment_param)

				self.exposed_img = self.exposure(self.input, True, self.shift_param)
				self.vignette_img = self.vignette(self.input, True)
				self.fog_img = self.fog(self.input, True)
				self.snow_img = self.snow(self.vignette_img, True)

				self.pixelate_img = self.pixelate(self.input, True)
				self.gamma_img = self.gamma(self.input, True, self.gamma_shift_param)
				self.snp_img = self.SnPNoise(self.gamma_img, True, self.sdev_param)

				self.rotate_img = self.rotate(self.input, True, self.degree_param)
				self.jitter_img = self.jitter(self.pixelate_img, True)

				self.blend_img = self.blend(self.snow_img, self.bright_img, True)
		elif model_batch_size == 64:
			if raliMode == 1:
				self.jpg_img = self.jpegFileInput(input_path, input_color_format, False, loop_parameter, 0)
				self.input = self.resize(self.jpg_img, h_img, w_img, False)

				self.rot135_img = self.rotate(self.input, False, 135)
				self.flip_img = self.flip(self.input, False)
				self.rot45_img = self.rotate(self.input, False, 45)

				self.setof16_mode1(self.input, h_img, w_img)
				self.setof16_mode1(self.rot45_img, h_img, w_img)
				self.setof16_mode1(self.flip_img, h_img, w_img)
				self.setof16_mode1(self.rot135_img , h_img, w_img)
				
			elif raliMode == 2:
				self.jpg_img = self.jpegFileInput(input_path, input_color_format, False, loop_parameter, 0)
				self.input = self.resize(self.jpg_img, h_img, w_img, False)

				#self.warpAffine2_img = self.warpAffine(self.input, False, [[1.5,0],[0,1],[None,None]])
				self.warpAffine1_img = self.warpAffine(self.input, False, [[0.5,0],[0,2],[None,None]]) #squeeze
				self.fishEye_img = self.fishEye(self.input, False)
				self.lensCorrection_img = self.lensCorrection(self.input, False, 1.5, 2)

				self.setof16_mode1(self.input, h_img, w_img)
				self.setof16_mode1(self.warpAffine1_img, h_img, w_img)
				self.setof16_mode1(self.fishEye_img, h_img, w_img)
				self.setof16_mode1(self.lensCorrection_img, h_img, w_img)

			elif raliMode == 3:
				self.jpg_img = self.jpegFileInput(input_path, input_color_format, False, loop_parameter, 0)
				self.input = self.resize(self.jpg_img, h_img, w_img, False)

				self.colorTemp1_img = self.colorTemp(self.input, False, 10)
				self.colorTemp2_img = self.colorTemp(self.input, False, 20)
				self.warpAffine2_img = self.warpAffine(self.input, False, [[2,0],[0,1],[None,None]]) #stretch

				self.setof16_mode1(self.input, h_img, w_img)
				self.setof16_mode1(self.colorTemp1_img, h_img, w_img)
				self.setof16_mode1(self.colorTemp2_img, h_img, w_img)
				self.setof16_mode1(self.warpAffine2_img , h_img, w_img)	
		#rali iterator
		if self.build() != 0:
			raise Exception('Failed to build the augmentation graph')
		self.tensor_format =tensor_layout
		self.multiplier = multiplier
		self.offset = offset
		self.reverse_channels = reverse_channels
		self.w = self.getOutputWidth()
		self.h = self.getOutputHeight()
		self.b = self.getBatchSize()
		self.n = self.getOutputImageCount()
		color_format = self.getOutputColorFormat()
		self.p = (1 if color_format is ColorFormat.IMAGE_U8 else 3)
		height = self.h*self.n
		self.out_image = np.zeros((height, self.w, self.p), dtype = "uint8")
		self.out_tensor = np.zeros(( self.b*self.n, self.p, self.h/self.b, self.w,), dtype = "float32")


	def get_input_name(self):
		return self.jpg_img.name(0)

	def process_validation(self, validation_list):
		for i in range(len(validation_list)):
			name, groundTruthIndex = validation_list[i].decode("utf-8").split(' ')
			self.validation_dict[name] = groundTruthIndex

	def get_ground_truth(self):
		return self.validation_dict[self.get_input_name()]

	def setof16_mode1(self, input_image, h_img, w_img):
		self.resized_image = self.resize(input_image, h_img, w_img, True)

		self.warped = self.warpAffine(input_image,True)

		self.contrast_img = self.contrast(input_image,True, self.min_param, self.max_param)
		self.rain_img = self.rain(input_image, True)

		self.bright_img = self.brightness(input_image,True, self.alpha_param)
		self.temp_img = self.colorTemp(input_image, True, self.adjustment_param)

		self.exposed_img = self.exposure(input_image, True, self.shift_param)
		self.vignette_img = self.vignette(input_image, True)
		self.fog_img = self.fog(input_image, True)
		self.snow_img = self.snow(input_image, True)

		self.pixelate_img = self.pixelate(input_image, True)
		self.snp_img = self.SnPNoise(input_image, True, self.sdev_param)
		self.gamma_img = self.gamma(input_image, True, self.gamma_shift_param)

		self.rotate_img = self.rotate(input_image, True, self.degree_param)
		self.jitter_img = self.jitter(input_image, True)
		
		self.blend_img = self.blend(input_image, self.contrast_img, True)

	def updateAugmentationParameter(self, augmentation):
		#values for contrast
		min = int(augmentation*100)
		max = 150 + int((1-augmentation)*100)
		self.min_param.update(min)
		self.max_param.update(max)

		#values for brightness
		alpha = augmentation*1.95
		self.alpha_param.update(alpha)

		#values for colorTemp
		adjustment = (augmentation*99) if ((int(augmentation*100)) % 2 == 0) else (-1*augmentation*99)
		adjustment = int(adjustment)
		self.adjustment_param.update(adjustment)

		#values for exposure
		shift = augmentation*0.95
		self.shift_param.update(shift)

		#values for SnPNoise
		sdev = augmentation*0.7
		self.sdev_param.update(sdev)

		#values for gamma
		gamma_shift = augmentation*5.0
		self.gamma_shift_param.update(gamma_shift)

		#values for rotation
		degree = augmentation * 180.0
		self.degree_param.update(degree)
		
	def get_next_augmentation(self):
		if self.getReaminingImageCount() <= 0:
			#raise StopIteration
			return -1
		if self.run() != 0:
			#raise StopIteration
			return -1
		self.copyToNPArray(self.out_image)
		if(TensorLayout.NCHW == self.tensor_format):
			self.copyToTensorNCHW(self.out_tensor, self.multiplier, self.offset, self.reverse_channels)
		else:
			self.copyToTensorNHWC(self.out_tensor, self.multiplier, self.offset, self.reverse_channels)
		return self.out_image , self.out_tensor

	def get_rali_list(self, raliMode, model_batch_size):
		if model_batch_size == 16:
			if raliMode == 1:
				self.rali_list = raliList_mode1_16
			elif raliMode == 2:
				self.rali_list = raliList_mode2_16
			elif raliMode == 3:
				self.rali_list = raliList_mode3_16
		elif model_batch_size == 64:
			if raliMode == 1:
				self.rali_list = raliList_mode1_64
			elif raliMode == 2:
				self.rali_list = raliList_mode2_64
			elif raliMode == 3:
				self.rali_list = raliList_mode3_64
				
		return self.rali_list
