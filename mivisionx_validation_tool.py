__author__      = "Kiriti Nagesh Gowda"
__copyright__   = "Copyright 2019, AMD MIVisionX"
__credits__     = ["Hansel Yang; Lakshmi Kumar;"]
__license__     = "MIT"
__version__     = "0.9.5"
__maintainer__  = "Kiriti Nagesh Gowda"
__email__       = "Kiriti.NageshGowda@amd.com"
__status__      = "ALPHA"
__script_name__ = "MIVisionX Validation Tool"

import argparse
import os
import sys
import ctypes
import cv2
import time
import numpy
import numpy as np
from numpy.ctypeslib import ndpointer
from rali import *
from rali_image_iterator import *
from inference_control import *
from inference_viewer import *

# global variables
FP16inference = False
verbosePrint = False
labelNames = None
colors =[
        (0,153,0),        # Top1
        (153,153,0),      # Top2
        (153,76,0),       # Top3
        (0,128,255),      # Top4
        (255,102,102),    # Top5
        ];
raliList_mode1 = ['original', 'warpAffine', 'contrast', 'rain', 
			'brightness', 'colorTemp', 'exposure', 'vignette', 
			'fog', 'snow', 'pixelate', 'SnPNoise', 
			'gamma', 'rotate', 'jitter', 'blend']
raliList_mode2 = ['original', 'warpAffine', 'contrast', 'contrast+rain', 
			'brightness', 'brightness+colorTemp', 'exposure', 'exposure+vignette', 
			'fog', 'fog+snow', 'pixelate', 'pixelate+SnPNoise', 
			'gamma', 'rotate', 'rotate+jitter', 'blend']
raliList_mode3 = ['original', 'warpAffine', 'contrast', 'warpAffine+rain', 
			'brightness', 'colorTemp', 'exposure', 'vignette', 
			'fog', 'vignette+snow', 'pixelate', 'gamma',
			'SnPNoise+gamma', 'rotate', 'jitter+pixelate', 'blend']
# Class to initialize Rali and call the augmentations 
class DataLoader(RaliGraph):
    def __init__(self, input_path, batch_size, input_color_format, affinity, image_validation, h_img, w_img, raliMode):
        RaliGraph.__init__(self, batch_size, affinity)
        self.validation_dict = {}
        self.process_validation(image_validation)
        self.setSeed(0)
        if raliMode == 1:
	        self.jpg_img = self.jpegFileInput(input_path, input_color_format, False, 0)
	        self.input = self.resize(self.jpg_img, h_img, w_img, True)
	        
	        self.warped = self.warpAffine(self.input,True)

	        self.contrast_img = self.contrast(self.input,True)
	        self.rain_img = self.rain(self.input, True)

	        self.bright_img = self.brightness(self.input,True)
	        self.temp_img = self.colorTemp(self.input, True)

	        self.exposed_img = self.exposure(self.input, True)
	        self.vignette_img = self.vignette(self.input, True)
	        self.fog_img = self.fog(self.input, True)
	        self.snow_img = self.snow(self.input, True)

	        self.pixelate_img = self.pixelate(self.input, True)
	        self.snp_img = self.SnPNoise(self.input, True, 0.2)
	        self.gamma_img = self.gamma(self.input, True)

	        self.rotate_img = self.rotate(self.input, True)
	        self.jitter_img = self.jitter(self.input, True)
			
	        self.blend_img = self.blend(self.input, self.contrast_img, True)
        elif raliMode == 2:
	    	self.jpg_img = self.jpegFileInput(input_path, input_color_format, False, 0)
	        self.input = self.resize(self.jpg_img, h_img, w_img, True)

	    	self.warped = self.warpAffine(self.input,True)

	    	self.contrast_img = self.contrast(self.input,True)
	    	self.rain_img = self.rain(self.contrast_img, True)

	    	self.bright_img = self.brightness(self.input,True)
	    	self.temp_img = self.colorTemp(self.bright_img, True)

	    	self.exposed_img = self.exposure(self.input, True)
	    	self.vignette_img = self.vignette(self.exposed_img, True)
	    	self.fog_img = self.fog(self.input, True)
	    	self.snow_img = self.snow(self.fog_img, True)

	    	self.pixelate_img = self.pixelate(self.input, True)
	    	self.snp_img = self.SnPNoise(self.pixelate_img, True, 0.2)
	    	self.gamma_img = self.gamma(self.input, True)

	    	self.rotate_img = self.rotate(self.input, True)
	    	self.jitter_img = self.jitter(self.rotate_img, True)

	        self.blend_img = self.blend(self.rotate_img, self.warped, True)
        elif raliMode == 3:
	    	self.jpg_img = self.jpegFileInput(input_path, input_color_format, False, 0)
	        self.input = self.resize(self.jpg_img, h_img, w_img, True)

	    	self.warped = self.warpAffine(self.input,True)

	    	self.contrast_img = self.contrast(self.input,True)
	    	self.rain_img = self.rain(self.warped, True)

	    	self.bright_img = self.brightness(self.input,True)
	    	self.temp_img = self.colorTemp(self.input, True)

	    	self.exposed_img = self.exposure(self.input, True)
	    	self.vignette_img = self.vignette(self.input, True)
	    	self.fog_img = self.fog(self.input, True)
	    	self.snow_img = self.snow(self.vignette_img, True)

	    	self.pixelate_img = self.pixelate(self.input, True)
	    	self.gamma_img = self.gamma(self.input, True)
	    	self.snp_img = self.SnPNoise(self.gamma_img, True, 0.2)

	    	self.rotate_img = self.rotate(self.input, True)
	    	self.jitter_img = self.jitter(self.pixelate_img, True)

	        self.blend_img = self.blend(self.snow_img, self.bright_img, True)

    def get_input_name(self):
        return self.jpg_img.name(0)

    def process_validation(self, validation_list):
		for i in range(len(validation_list)):
			name, groundTruthIndex = validation_list[i].decode("utf-8").split(' ')
			self.validation_dict[name] = groundTruthIndex

    def get_ground_truth(self):
		return self.validation_dict[self.get_input_name()]

# AMD Neural Net python wrapper
class AnnAPI:
	def __init__(self,library):
		self.lib = ctypes.cdll.LoadLibrary(library)
		self.annQueryInference = self.lib.annQueryInference
		self.annQueryInference.restype = ctypes.c_char_p
		self.annQueryInference.argtypes = []
		self.annCreateInference = self.lib.annCreateInference
		self.annCreateInference.restype = ctypes.c_void_p
		self.annCreateInference.argtypes = [ctypes.c_char_p]
		self.annReleaseInference = self.lib.annReleaseInference
		self.annReleaseInference.restype = ctypes.c_int
		self.annReleaseInference.argtypes = [ctypes.c_void_p]
		self.annCopyToInferenceInput = self.lib.annCopyToInferenceInput
		self.annCopyToInferenceInput.restype = ctypes.c_int
		self.annCopyToInferenceInput.argtypes = [ctypes.c_void_p, ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"), ctypes.c_size_t, ctypes.c_bool]
		self.annCopyFromInferenceOutput = self.lib.annCopyFromInferenceOutput
		self.annCopyFromInferenceOutput.restype = ctypes.c_int
		self.annCopyFromInferenceOutput.argtypes = [ctypes.c_void_p, ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"), ctypes.c_size_t]
		self.annRunInference = self.lib.annRunInference
		self.annRunInference.restype = ctypes.c_int
		self.annRunInference.argtypes = [ctypes.c_void_p, ctypes.c_int]
		print('OK: AnnAPI found "' + self.annQueryInference().decode("utf-8") + '" as configuration in ' + library)

# classifier definition
class annieObjectWrapper():
	def __init__(self, annpythonlib, weightsfile):
		self.api = AnnAPI(annpythonlib)
		input_info,output_info,empty = self.api.annQueryInference().decode("utf-8").split(';')
		input,name,n_i,c_i,h_i,w_i = input_info.split(',')
		outputCount = output_info.split(",")
		stringcount = len(outputCount)
		if stringcount == 6:
			output,opName,n_o,c_o,h_o,w_o = output_info.split(',')
		else:
			output,opName,n_o,c_o= output_info.split(',')
			h_o = '1'; w_o  = '1';
		self.hdl = self.api.annCreateInference(weightsfile.encode('utf-8'))
		self.dim = (int(w_i),int(h_i))
		self.outputDim = (int(n_o),int(c_o),int(h_o),int(w_o))

	def __del__(self):
		self.api.annReleaseInference(self.hdl)

	def runInference(self, img_tensor, out):
		# copy input f32 to inference input
		status = self.api.annCopyToInferenceInput(self.hdl, np.ascontiguousarray(img_tensor, dtype=np.float32), img_tensor.nbytes, 0)
		# run inference
		status = self.api.annRunInference(self.hdl, 1)
		# copy output f32
		status = self.api.annCopyFromInferenceOutput(self.hdl, np.ascontiguousarray(out, dtype=np.float32), out.nbytes)
		return out

	def classify(self, img_tensor):
		# create output.f32 buffer
		out_buf = bytearray(self.outputDim[0]*self.outputDim[1]*self.outputDim[2]*self.outputDim[3]*4)
		out = np.frombuffer(out_buf, dtype=numpy.float32)
		# run inference & receive output
		output = self.runInference(img_tensor, out)
		return output

# process classification output function
def processClassificationOutput(inputImage, modelName, modelOutput, batchSize):
	# post process output file
	start = time.time()
	softmaxOutput = np.float32(modelOutput)
	outputList = np.split(softmaxOutput, batchSize)
	topIndex = []
	topLabels = []
	topProb = []
	for i in range(len(outputList)):
		for x in outputList[i].argsort()[-5:]:
			topIndex.append(x)
			topLabels.append(labelNames[x])
			topProb.append(softmaxOutput[x])
	end = time.time()
	if(verbosePrint):
		print '%30s' % 'Processed results in ', str((end - start)*1000), 'ms'

	return topIndex, topProb

# MIVisionX Classifier
if __name__ == '__main__':   
	app = QtGui.QApplication(sys.argv)
	if len(sys.argv) == 1:
		panel = inference_control()
		panel.show()
		app.exec_()
		modelFormat = (str)(panel.model_format)
		modelName = (str)(panel.model_name)
		modelLocation = (str)(panel.model)
		modelBatchSize = (str)(panel.batch)
		raliMode = (int)(panel.mode) + 1
		modelInputDims = (str)(panel.input_dims)
		modelOutputDims = (str)(panel.output_dims)
		label = (str)(panel.label)
		outputDir = (str)(panel.output)
		imageDir = (str)(panel.image)
		imageVal = (str)(panel.val)
		hierarchy = (str)(panel.hier)
		inputAdd = (str)(panel.add)
		inputMultiply = (str)(panel.multiply)
		fp16 = (str)(panel.fp16)
		replaceModel = (str)(panel.replace)
		verbose = (str)(panel.verbose)
	else:
		parser = argparse.ArgumentParser()
		parser.add_argument('--model_format',		type=str, required=True,	help='pre-trained model format, options:caffe/onnx/nnef [required]')
		parser.add_argument('--model_name',			type=str, required=True,	help='model name                             [required]')
		parser.add_argument('--model',				type=str, required=True,	help='pre_trained model file/folder          [required]')
		parser.add_argument('--model_batch_size',	type=str, required=True,	help='n - batch size			             [required]')
		parser.add_argument('--rali_mode',			type=str, required=True,	help='rali mode (1/2/3)			             [required]')
		parser.add_argument('--model_input_dims',	type=str, required=True,	help='c,h,w - channel,height,width           [required]')
		parser.add_argument('--model_output_dims',	type=str, required=True,	help='c,h,w - channel,height,width           [required]')
		parser.add_argument('--label',				type=str, required=True,	help='labels text file                       [required]')
		parser.add_argument('--output_dir',			type=str, required=True,	help='output dir to store ADAT results       [required]')
		parser.add_argument('--image_dir',			type=str, required=True,	help='image directory for analysis           [required]')
		parser.add_argument('--image_val',			type=str, default='',		help='image list with ground truth           [optional]')
		parser.add_argument('--hierarchy',			type=str, default='',		help='AMD proprietary hierarchical file      [optional]')
		parser.add_argument('--add',				type=str, default='', 		help='input preprocessing factor [optional - default:[0,0,0]]')
		parser.add_argument('--multiply',			type=str, default='',		help='input preprocessing factor [optional - default:[1,1,1]]')
		parser.add_argument('--fp16',				type=str, default='no',		help='quantize to FP16 			[optional - default:no]')
		parser.add_argument('--replace',			type=str, default='no',		help='replace/overwrite model   [optional - default:no]')
		parser.add_argument('--verbose',			type=str, default='no',		help='verbose                   [optional - default:no]')
		args = parser.parse_args()
		
		# get arguments
		modelFormat = args.model_format
		modelName = args.model_name
		modelLocation = args.model
		modelBatchSize = args.model_batch_size
		raliMode = (int)(args.rali_mode)
		modelInputDims = args.model_input_dims
		modelOutputDims = args.model_output_dims
		label = args.label
		outputDir = args.output_dir
		imageDir = args.image_dir
		imageVal = args.image_val
		hierarchy = args.hierarchy
		inputAdd = args.add
		inputMultiply = args.multiply
		fp16 = args.fp16
		replaceModel = args.replace
		verbose = args.verbose

	# set verbose print
	if(verbose != 'no'):
		verbosePrint = True

	# set fp16 inference turned on/off
	if(fp16 != 'no'):
		FP16inference = True
	# set paths
	modelCompilerPath = '/opt/rocm/mivisionx/model_compiler/python'
	ADATPath= '/opt/rocm/mivisionx/toolkit/analysis_and_visualization/classification'
	setupDir = '~/.mivisionx-validation-tool'
	analyzerDir = os.path.expanduser(setupDir)
	modelDir = analyzerDir+'/'+modelName+'_dir'
	nnirDir = modelDir+'/nnir-files'
	openvxDir = modelDir+'/openvx-files'
	modelBuildDir = modelDir+'/build'
	adatOutputDir = os.path.expanduser(outputDir)
	inputImageDir = os.path.expanduser(imageDir)
	trainedModel = os.path.expanduser(modelLocation)
	labelText = os.path.expanduser(label)
	hierarchyText = os.path.expanduser(hierarchy)
	imageValText = os.path.expanduser(imageVal)
	pythonLib = modelBuildDir+'/libannpython.so'
	weightsFile = openvxDir+'/weights.bin'
	finalImageResultsFile = modelDir+'/imageResultsFile.csv'
	raliMode = 1

	# get input & output dims
	str_c_i, str_h_i, str_w_i = modelInputDims.split(',')
	c_i = int(str_c_i); h_i = int(str_h_i); w_i = int(str_w_i)
	str_c_o, str_h_o, str_w_o = modelOutputDims.split(',')
	c_o = int(str_c_o); h_o = int(str_h_o); w_o = int(str_w_o)
	modelBatchSizeInt = int(modelBatchSize)
	# input pre-processing values
	Ax=[0,0,0]
	if(inputAdd != ''):
		Ax = [float(item) for item in inputAdd.strip("[]").split(',')]
	Mx=[1,1,1]
	if(inputMultiply != ''):
		Mx = [float(item) for item in inputMultiply.strip("[]").split(',')]

	# check pre-trained model
	if(not os.path.isfile(trainedModel) and modelFormat != 'nnef' ):
		print("\nPre-Trained Model not found, check argument --model\n")
		quit()

	# check for label file
	if (not os.path.isfile(labelText)):
		print("\nlabels.txt not found, check argument --label\n")
		quit()
	else:
		fp = open(labelText, 'r')
		labelNames = fp.readlines()
		fp.close()

	# MIVisionX setup
	if(os.path.exists(analyzerDir)):
		print("\nMIVisionX Validation Tool\n")
		# replace old model or throw error
		if(replaceModel == 'yes'):
			os.system('rm -rf '+modelDir)
		elif(os.path.exists(modelDir)):
			print("OK: Model exists")

	else:
		print("\nMIVisionX Validation Tool Created\n")
		os.system('(cd ; mkdir .mivisionx-validation-tool)')
	# Setup Text File for Demo
	if (not os.path.isfile(analyzerDir + "/setupFile.txt")):
		f = open(analyzerDir + "/setupFile.txt", "w")
		f.write(modelFormat + ';' + modelName + ';' + modelLocation + ';' + modelBatchSize + ';' + modelInputDims + ';' + modelOutputDims + ';' + label + ';' + outputDir + ';' + imageDir + ';' + imageVal + ';' + hierarchy + ';' + str(Ax).strip('[]').replace(" ","") + ';' + str(Mx).strip('[]').replace(" ","") + ';' + fp16 + ';' + replaceModel + ';' + verbose)
		f.close()
	else:
		count = len(open(analyzerDir + "/setupFile.txt").readlines())
		if count < 10:
			with open(analyzerDir + "/setupFile.txt", "r") as fin:
				data = fin.read().splitlines(True)
				modelList = []
				for i in range(len(data)):
					if data[i] != '\n':
						modelList.append(data[i].split(';')[1])
				if modelName not in modelList:
					f = open(analyzerDir + "/setupFile.txt", "a")
					f.write("\n" + modelFormat + ';' + modelName + ';' + modelLocation + ';' + modelBatchSize + ';' + modelInputDims + ';' + modelOutputDims + ';' + label + ';' + outputDir + ';' + imageDir + ';' + imageVal + ';' + hierarchy + ';' + str(Ax).strip('[]').replace(" ","") + ';' + str(Mx).strip('[]').replace(" ","") + ';' + fp16 + ';' + replaceModel + ';' + verbose)
					f.close()
		else:
			with open(analyzerDir + "/setupFile.txt", "r") as fin:
				data = fin.read().splitlines(True)
				delModelName = data[0].split(';')[1]
			delmodelPath = analyzerDir + '/' + delModelName + '_dir'
			if(os.path.exists(delmodelPath)): 
				os.system('rm -rf ' + delmodelPath)
			with open(analyzerDir + "/setupFile.txt", "w") as fout:
			    fout.writelines(data[1:])
			with open(analyzerDir + "/setupFile.txt", "a") as fappend:
				fappend.write("\n" + modelFormat + ';' + modelName + ';' + modelLocation + ';' + modelBatchSize + ';' + modelInputDims + ';' + modelOutputDims + ';' + label + ';' + outputDir + ';' + imageDir + ';' + imageVal + ';' + hierarchy + ';' + str(Ax).strip('[]').replace(" ","") + ';' + str(Mx).strip('[]').replace(" ","") + ';' + fp16 + ';' + replaceModel + ';' + verbose)
				fappend.close()

	# Compile Model and generate python .so files
	if (replaceModel == 'yes' or not os.path.exists(modelDir)):
		os.system('mkdir '+modelDir)
		if(os.path.exists(modelDir)):
			# convert to NNIR
			if(modelFormat == 'caffe'):
				os.system('(cd '+modelDir+'; python '+modelCompilerPath+'/caffe_to_nnir.py '+trainedModel+' nnir-files --input-dims 1,' + modelInputDims + ')')
				os.system('(cd '+modelDir+'; python '+modelCompilerPath+'/nnir_update.py --batch-size ' + modelBatchSize + ' nnir-files nnir-files)')
			elif(modelFormat == 'onnx'):
				os.system('(cd '+modelDir+'; python '+modelCompilerPath+'/onnx_to_nnir.py '+trainedModel+' nnir-files --input_dims 1,' + modelInputDims + ')')
				os.system('(cd '+modelDir+'; python '+modelCompilerPath+'/nnir_update.py --batch-size ' + modelBatchSize + ' nnir-files nnir-files)')
			elif(modelFormat == 'nnef'):
				os.system('(cd '+modelDir+'; python '+modelCompilerPath+'/nnef_to_nnir.py '+trainedModel+' nnir-files --batch-size ' + modelBatchSize + ')')
				#os.system('(cd '+modelDir+'; python '+modelCompilerPath+'/nnir_update.py --batch-size ' + modelBatchSize + ' nnir-files nnir-files)')
			else:
				print("ERROR: Neural Network Format Not supported, use caffe/onnx/nnef in arugment --model_format")
				quit()
			# convert the model to FP16
			if(FP16inference):
				os.system('(cd '+modelDir+'; python '+modelCompilerPath+'/nnir_update.py --convert-fp16 1 --fuse-ops 1 nnir-files nnir-files)')
				print("\nModel Quantized to FP16\n")
			# convert to openvx
			if(os.path.exists(nnirDir)):
				os.system('(cd '+modelDir+'; python '+modelCompilerPath+'/nnir_to_openvx.py nnir-files openvx-files)')
			else:
				print("ERROR: Converting Pre-Trained model to NNIR Failed")
				quit()
			
			# build model
			if(os.path.exists(openvxDir)):
				os.system('mkdir '+modelBuildDir)
			else:
				print("ERROR: Converting NNIR to OpenVX Failed")
				quit()

	os.system('(cd '+modelBuildDir+'; cmake ../openvx-files; make; ./anntest ../openvx-files/weights.bin )')
	print("\nSUCCESS: Converting Pre-Trained model to MIVisionX Runtime successful\n")

	# create inference classifier
	classifier = annieObjectWrapper(pythonLib, weightsFile)

	# check for image val text
	totalImages = 0;
	if(imageVal != ''):
		if (not os.path.isfile(imageValText)):
			print("\nImage Validation Text not found, check argument --image_val\n")
			quit()
		else:
			fp = open(imageValText, 'r')
			imageValidation = fp.readlines()
			fp.close()
			totalImages = len(imageValidation)
	else:
		print("\nFlow without Image Validation Text not implemented, pass argument --image_val\n")
		quit()

	# original std out location 
	orig_stdout = sys.stdout
	# setup results output file
	sys.stdout = open(finalImageResultsFile,'w')	
	print('Image File Name,Ground Truth Label,Output Label 1,Output Label 2,Output Label 3,Output Label 4,Output Label 5,Prob 1,Prob 2,Prob 3,Prob 4,Prob 5')
	sys.stdout = orig_stdout

	#setup for Rali
	batchSize = 1
	loader = DataLoader(inputImageDir, batchSize, ColorFormat.IMAGE_RGB24, Affinity.PROCESS_CPU, imageValidation, h_i, w_i, raliMode)
	imageIterator = ImageIterator(loader, reverse_channels=False,multiplier=Mx,offset=Ax)
	print ('Pipeline created ...')
	print 'Image iterator created ... number of images', imageIterator.imageCount()
	print 'Loader created ...num of images' , loader.getOutputImageCount()

	# process images
	correctTop5 = 0; correctTop1 = 0; wrong = 0; noGroundTruth = 0;
	
	#create output dict for all the images
	guiResults = {}

	viewer = inference_viewer(modelName, totalImages*modelBatchSizeInt)
	viewer.show()
	#image_tensor has the input tensor required for inference
	for x,(image_batch, image_tensor) in enumerate(imageIterator,0):
		viewer.runState = True
		start_main = time.time()
		imageFileName = loader.get_input_name()
		groundTruthIndex = loader.get_ground_truth()
		groundTruthIndex = int(groundTruthIndex)

		#create output list for each image
		augmentedResults = []

		#create images for display
		original_image = image_batch[0:h_i, 0:w_i]
		cloned_image = np.copy(image_batch)
		frame = image_tensor

		#show original image
		width = original_image.shape[0]
		height = original_image.shape[1]
#		viewer.putImage(original_image, width, height)
		viewer.showImage(original_image, width, height)

		# run inference
		start = time.time()
		output = classifier.classify(frame)
		end = time.time()
		if(verbosePrint):
			print '%30s' % 'Executed Model in ', str((end - start)*1000), 'ms'

		for i in range(loader.getOutputImageCount()):
			#using tensor output of RALI as frame 		
			
			# process output and display
			start = time.time()
			topIndex, topProb = processClassificationOutput(frame, modelName, output, modelBatchSizeInt)
			end = time.time()
			if(verbosePrint):
				print '%30s' % 'Processed display in ', str((end - start)*1000), 'ms\n'

			# write image results to a file
			start = time.time()
			sys.stdout = open(finalImageResultsFile,'a')
			print(imageFileName+','+str(groundTruthIndex)+','+str(topIndex[4 + i*4])+
			','+str(topIndex[3 + i*4])+','+str(topIndex[2 + i*4])+','+str(topIndex[1 + i*4])+','+str(topIndex[0 + i*4])+','+str(topProb[4 + i*4])+
			','+str(topProb[3 + i*4])+','+str(topProb[2 + i*4])+','+str(topProb[1 + i*4])+','+str(topProb[0 + i*4]))
			sys.stdout = orig_stdout
			end = time.time()
			if(verbosePrint):
				print '%30s' % 'Image result saved in ', str((end - start)*1000), 'ms'

			# create progress image
			start = time.time()

			# augmentedResults List: 0 = wrong; 1-5 = TopK; -1 = No Ground Truth
			if(groundTruthIndex == topIndex[4 + i*4]):
				correctTop1 = correctTop1 + 1
				correctTop5 = correctTop5 + 1
				augmentedResults.append(1)
			elif(groundTruthIndex == topIndex[3 + i*4] or groundTruthIndex == topIndex[2 + i*4] or groundTruthIndex == topIndex[1 + i*4] or groundTruthIndex == topIndex[0 + i*4]):
				correctTop5 = correctTop5 + 1
				if (groundTruthIndex == topIndex[3 + i*4]):
					augmentedResults.append(2)
				elif (groundTruthIndex == topIndex[2 + i*4]):
					augmentedResults.append(3)
				elif (groundTruthIndex == topIndex[1 + i*4]):
					augmentedResults.append(4)
				elif (groundTruthIndex == topIndex[0 + i*4]):
					augmentedResults.append(5)
			elif(groundTruthIndex == -1):
				noGroundTruth = noGroundTruth + 1
				augmentedResults.append(-1)
			else:
				wrong = wrong + 1
				augmentedResults.append(0)

			# Total progress
			viewer.setTotalProgress(modelBatchSizeInt*x+i+1)
			# Top 1 progress
			viewer.setTop1Progress(correctTop1)
			# Top 5 progress
			viewer.setTop5Progress(correctTop5)
			# Mismatch progress
			viewer.setMisProgress(wrong)
			# No ground truth progress
			viewer.setNoGTProgress(noGroundTruth)

			end = time.time()
			if(verbosePrint):
				print '%30s' % 'Progress image created in ', str((end - start)*1000), 'ms'

			#write type of augmentation on image
			if raliMode == 1:
				raliList = raliList_mode1
			elif raliMode == 2:
				raliList = raliList_mode2
			elif raliMode == 3:
				raliList = raliList_mode3

			text_width, text_height = cv2.getTextSize(raliList[i], cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)[0]
			text_off_x = 5
			text_off_y = (i*h_i)+h_i-7
			box_coords = ((text_off_x, text_off_y), (text_off_x + text_width - 2, text_off_y - text_height - 2))
			cv2.rectangle(cloned_image, box_coords[0], box_coords[1], (192,192,192), cv2.FILLED)
			cv2.putText(cloned_image, raliList[i], (text_off_x, text_off_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 2)	

			#show RALI augmented images
			if augmentedResults[i] == 0:
				cv2.rectangle(cloned_image, (0,(i*(h_i-1)+i)),((w_i-1),(h_i-1)*(i+1) + i), (255,0,0), 4, cv2.LINE_8, 0)
			elif augmentedResults[i] > 0  and augmentedResults[i] < 6:		
				cv2.rectangle(cloned_image, (0,(i*(h_i-1)+i)),((w_i-1),(h_i-1)*(i+1) + i), (0,255,0), 4, cv2.LINE_8, 0)

		#split RALI augmented images into a grid
		image_batch1, image_batch2, image_batch3, image_batch4 = np.vsplit(cloned_image, 4)
		final_image_batch = np.hstack((image_batch1, image_batch2, image_batch3, image_batch4))

    	#show augmented images
		aug_width = final_image_batch.shape[0]
		aug_height = final_image_batch.shape[1]
		#viewer.putAugImage(final_image_batch, width, height)
		viewer.showAugImage(final_image_batch, aug_width, aug_height)
		#viewer.showImage()
		#viewer.update()
		#cv2.namedWindow('augmented_images', cv2.WINDOW_GUI_EXPANDED)
		#cv2.imshow('augmented_images', cv2.cvtColor(final_image_batch, cv2.COLOR_RGB2BGR))
		#app.exec_()
		# exit inference on ESC; pause/play on SPACEBAR; quit program on 'q'
		key = cv2.waitKey(2)
		if not viewer.runState:
			break
		while viewer.pauseState:
			cv2.waitKey(0)
			print 'hi'
			if not viewer.runState:
				print 'hi2'
				break
			
		# if key == 27: 
		# 	break
		# if key == 32:
		# 	if cv2.waitKey(0) == 32:
		# 		continue
		# if key == 113:
		# 	exit(0)

		guiResults[imageFileName] = augmentedResults
		end_main = time.time()
		#print '%30s' % 'Process Batch Time ', str((end_main - start_main)*1000), 'ms'

	print("\nSUCCESS: Images Inferenced with the Model\n")

	# Create ADAT folder and file
	print("\nADAT tool called to create the analysis toolkit\n")
	if(not os.path.exists(adatOutputDir)):
		os.system('mkdir ' + adatOutputDir)
	
	if(hierarchy == ''):
		os.system('python '+ADATPath+'/generate-visualization.py --inference_results '+finalImageResultsFile+
		' --image_dir '+inputImageDir+' --label '+labelText+' --model_name '+modelName+' --output_dir '+adatOutputDir+' --output_name '+modelName+'-ADAT')
	else:
		os.system('python '+ADATPath+'/generate-visualization.py --inference_results '+finalImageResultsFile+
		' --image_dir '+inputImageDir+' --label '+labelText+' --hierarchy '+hierarchyText+' --model_name '+modelName+' --output_dir '+adatOutputDir+' --output_name '+modelName+'-ADAT')
	print("\nSUCCESS: Image Analysis Toolkit Created\n")
	print("Press ESC to exit or close progess window\n")

	# Wait to quit
	# while True:
	# 	key = cv2.waitKey(2)
	# 	if key == 27:
	# 		cv2.destroyAllWindows()
	# 		break   
	# 	# if cv2.getWindowProperty(windowProgress,cv2.WND_PROP_VISIBLE) < 1:        
	# 	# 	break

	outputHTMLFile = os.path.expanduser(adatOutputDir+'/'+modelName+'-ADAT-toolKit/index.html')
	os.system('firefox '+outputHTMLFile)
