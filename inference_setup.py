import sys
import os
import ctypes
import time
import numpy as np
import cv2
import Queue
from numpy.ctypeslib import ndpointer
from PyQt4 import QtGui, uic, QtCore
from PyQt4.QtCore import pyqtSignal

from rali_setup import *

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
		out = np.frombuffer(out_buf, dtype=np.float32)
		# run inference & receive output
		output = self.runInference(img_tensor, out)
		return output

class modelInference(QtCore.QObject):
	def __init__(self, modelName, modelFormat, imageDir, modelLocation, label, hierarchy, imageVal, modelInputDims, modelOutputDims, 
				modelBatchSize, outputDir, inputAdd, inputMultiply, verbose, fp16, replaceModel, loop, rali_mode, origQueue, augQueue, parent=None):

		super(modelInference, self).__init__(parent)
		self.modelCompilerPath = '/opt/rocm/mivisionx/model_compiler/python'
		self.ADATPath= '/opt/rocm/mivisionx/toolkit/analysis_and_visualization/classification'
		self.setupDir = '~/.mivisionx-validation-tool'

		self.analyzerDir = os.path.expanduser(self.setupDir)
		self.modelDir = self.analyzerDir+'/'+modelName+'_dir'
		self.inputImageDir = os.path.expanduser((str)(imageDir))
		self.trainedModel = os.path.expanduser((str)(modelLocation))
		self.labelText = os.path.expanduser(label)
		self.hierarchyText = os.path.expanduser(hierarchy)
		self.imageValText = os.path.expanduser(imageVal)
		self.adatOutputDir = os.path.expanduser(outputDir)
		self.nnirDir = self.modelDir+'/nnir-files'
		self.openvxDir = self.modelDir+'/openvx-files'
		self.modelBuildDir = self.modelDir+'/build'
		self.pythonLib = self.modelBuildDir+'/libannpython.so'
		self.weightsFile = self.openvxDir+'/weights.bin'
		self.finalImageResultsFile = self.modelDir+'/imageResultsFile.csv'
		self.modelBatchSize = modelBatchSize
		self.verbosePrint = False
		self.FP16inference = False
		self.loop = False
		self.classifier = None
		self.labelNames = []
		self.raliEngine = None
		self.rali_mode = rali_mode
		self.origQueue = origQueue
		self.augQueue = augQueue
		str_c_i, str_h_i, str_w_i = modelInputDims.split(',')
		self.c_i = int(str_c_i); self.h_i = int(str_h_i); self.w_i = int(str_w_i)
		str_c_o, str_h_o, str_w_o = modelOutputDims.split(',')
		self.c_o = int(str_c_o); self.h_o = int(str_h_o); self.w_o = int(str_w_o)

		finished = pyqtSignal()

		# set verbose print
		if(verbose != 'no'):
			self.verbosePrint = True

		# set fp16 inference turned on/off
		if(fp16 != 'no'):
			self.FP16inference = True

		#set loop parameter based on user input
		if loop == 'yes':
			self.loop = True
		else:
			self.loop = False
		
		# get input & output dims
		self.modelBatchSizeInt = int(modelBatchSize)
		# input pre-processing values
		Ax=[0,0,0]
		if(inputAdd != ''):
			self.Ax = [float(item) for item in inputAdd.strip("[]").split(',')]
		Mx=[1,1,1]
		if(inputMultiply != ''):
			self.Mx = [float(item) for item in inputMultiply.strip("[]").split(',')]

		# Setup Text File for Demo
		if (not os.path.isfile(self.analyzerDir + "/setupFile.txt")):
			f = open(self.analyzerDir + "/setupFile.txt", "w")
			f.write(modelFormat + ';' + modelName + ';' + modelLocation + ';' + modelBatchSize + ';' + modelInputDims + ';' + modelOutputDims + ';' + label + ';' + outputDir + ';' + imageDir + ';' + imageVal + ';' + hierarchy + ';' + str(Ax).strip('[]').replace(" ","") + ';' + str(Mx).strip('[]').replace(" ","") + ';' + fp16 + ';' + replaceModel + ';' + verbose + ';' + loop)
			f.close()
		else:
			count = len(open(self.analyzerDir + "/setupFile.txt").readlines())
			if count < 10:
				with open(self.analyzerDir + "/setupFile.txt", "r") as fin:
					data = fin.read().splitlines(True)
					modelList = []
					for i in range(len(data)):
						if data[i] != '\n':
							modelList.append(data[i].split(';')[1])
					if modelName not in modelList:
						f = open(self.analyzerDir + "/setupFile.txt", "a")
						f.write("\n" + modelFormat + ';' + modelName + ';' + modelLocation + ';' + modelBatchSize + ';' + modelInputDims + ';' + modelOutputDims + ';' + label + ';' + outputDir + ';' + imageDir + ';' + imageVal + ';' + hierarchy + ';' + str(Ax).strip('[]').replace(" ","") + ';' + str(Mx).strip('[]').replace(" ","") + ';' + fp16 + ';' + replaceModel + ';' + verbose + ';' + loop)
						f.close()
			else:
				with open(self.analyzerDir + "/setupFile.txt", "r") as fin:
					data = fin.read().splitlines(True)
					delModelName = data[0].split(';')[1]
				delmodelPath = self.analyzerDir + '/' + delModelName + '_dir'
				if(os.path.exists(delmodelPath)): 
					os.system('rm -rf ' + delmodelPath)
				with open(self.analyzerDir + "/setupFile.txt", "w") as fout:
				    fout.writelines(data[1:])
				with open(self.analyzerDir + "/setupFile.txt", "a") as fappend:
					fappend.write("\n" + modelFormat + ';' + modelName + ';' + modelLocation + ';' + modelBatchSize + ';' + modelInputDims + ';' + modelOutputDims + ';' + label + ';' + outputDir + ';' + imageDir + ';' + imageVal + ';' + hierarchy + ';' + str(Ax).strip('[]').replace(" ","") + ';' + str(Mx).strip('[]').replace(" ","") + ';' + fp16 + ';' + replaceModel + ';' + verbose + ';' + loop)
					fappend.close()

		self.replaceModel = replaceModel
		self.modelFormat = modelFormat
		self.modelInputDims = modelInputDims
		self.modelOutputDims = modelOutputDims
		self.imageVal = imageVal
		self.stdout = None
		# get correct list for augmentations
		self.raliList = []
		self.setupInference()

	def setupInference(self):
		# check pre-trained model
		if(not os.path.isfile(self.trainedModel) and self.modelFormat != 'nnef' ):
			print("\nPre-Trained Model not found, check argument --model\n")
			quit()

		# check for label file
		if (not os.path.isfile(self.labelText)):
			print("\nlabels.txt not found, check argument --label\n")
			quit()
		else:
			fp = open(self.labelText, 'r')
			#labelNames = fp.readlines()
			self.labelNames = [x.strip('\n') for x in fp.readlines()]
			fp.close()

		# MIVisionX setup
		if(os.path.exists(self.analyzerDir)):
			print("\nMIVisionX Validation Tool\n")
			# replace old model or throw error
			if(self.replaceModel == 'yes'):
				os.system('rm -rf '+self.modelDir)
			elif(os.path.exists(self.modelDir)):
				print("OK: Model exists")
		else:
			print("\nMIVisionX Validation Tool Created\n")
			os.system('(cd ; mkdir .mivisionx-validation-tool)')

		# Compile Model and generate python .so files
		if (self.replaceModel == 'yes' or not os.path.exists(self.modelDir)):
			os.system('mkdir '+self.modelDir)
			if(os.path.exists(self.modelDir)):
				# convert to NNIR
				if(self.modelFormat == 'caffe'):
					os.system('(cd '+self.modelDir+'; python '+self.modelCompilerPath+'/caffe_to_nnir.py '+self.trainedModel+' nnir-files --input-dims 1,' + self.modelInputDims + ')')
					os.system('(cd '+self.modelDir+'; python '+self.modelCompilerPath+'/nnir_update.py --batch-size ' + self.modelBatchSize + ' nnir-files nnir-files)')
				elif(self.modelFormat == 'onnx'):
					os.system('(cd '+self.modelDir+'; python '+self.modelCompilerPath+'/onnx_to_nnir.py '+self.trainedModel+' nnir-files --input_dims 1,' + self.modelInputDims + ')')
					os.system('(cd '+self.modelDir+'; python '+self.modelCompilerPath+'/nnir_update.py --batch-size ' + self.modelBatchSize + ' nnir-files nnir-files)')
				elif(self.modelFormat == 'nnef'):
					os.system('(cd '+self.modelDir+'; python '+self.modelCompilerPath+'/nnef_to_nnir.py '+self.trainedModel+' nnir-files --batch-size ' + self.modelBatchSize + ')')
					#os.system('(cd '+self.modelDir+'; python '+self.modelCompilerPath+'/nnir_update.py --batch-size ' + self.modelBatchSize + ' nnir-files nnir-files)')
				else:
					print("ERROR: Neural Network Format Not supported, use caffe/onnx/nnef in arugment --model_format")
					quit()
				# convert the model to FP16
				if(self.FP16inference):
					os.system('(cd '+self.modelDir+'; python '+self.modelCompilerPath+'/nnir_update.py --convert-fp16 1 --fuse-ops 1 nnir-files nnir-files)')
					print("\nModel Quantized to FP16\n")
				# convert to openvx
				if(os.path.exists(self.nnirDir)):
					os.system('(cd '+self.modelDir+'; python '+self.modelCompilerPath+'/nnir_to_openvx.py nnir-files openvx-files)')
				else:
					print("ERROR: Converting Pre-Trained model to NNIR Failed")
					quit()
				
				# build model
				if(os.path.exists(self.openvxDir)):
					os.system('mkdir '+self.modelBuildDir)
				else:
					print("ERROR: Converting NNIR to OpenVX Failed")
					quit()

		#os.system('(cd '+self.modelBuildDir+'; cmake ../openvx-files; make; ./anntest ../openvx-files/weights.bin )')
		print("\nSUCCESS: Converting Pre-Trained model to MIVisionX Runtime successful\n")

		# create inference classifier
		self.classifier = annieObjectWrapper(self.pythonLib, self.weightsFile)

		# check for image val text
		if(self.imageVal != ''):
			if (not os.path.isfile(self.imageValText)):
				print("\nImage Validation Text not found, check argument --image_val\n")
				quit()
			else:
				fp = open(self.imageValText, 'r')
				imageValidation = fp.readlines()
				fp.close()
		else:
			print("\nFlow without Image Validation Text not implemented, pass argument --image_val\n")
			quit()
		totalImages = len(os.listdir(self.inputImageDir))

		# original std out location 
		self.orig_stdout = sys.stdout
		# setup results output file
		sys.stdout = open(self.finalImageResultsFile,'w')	
		print('Image File Name,Ground Truth Label,Output Label 1,Output Label 2,Output Label 3,Output Label 4,Output Label 5,Prob 1,Prob 2,Prob 3,Prob 4,Prob 5')
		sys.stdout = self.orig_stdout

		# Setup Rali Data Loader. 
		rali_batch_size = 1
		self.raliEngine = DataLoader(self.inputImageDir, rali_batch_size, int(self.modelBatchSizeInt), ColorFormat.IMAGE_RGB24, Affinity.PROCESS_CPU, imageValidation, self.h_i, self.w_i, self.rali_mode, self.loop, 
										TensorLayout.NCHW, False, self.Ax, self.Mx)
		self.raliList = self.raliEngine.get_rali_list(self.rali_mode, int(self.modelBatchSizeInt))

	# process classification output function
	def processClassificationOutput(self, modelOutput):#, labelNames):
		# post process output file
		start = time.time()
		softmaxOutput = np.float32(modelOutput)
		outputList = np.split(softmaxOutput, self.modelBatchSizeInt)
		topIndex = []
		#topLabels = []
		topProb = []
		for i in range(len(outputList)):
			for x in outputList[i].argsort()[-5:]:
				topIndex.append(x)
				#topLabels.append(labelNames[x])
				topProb.append(softmaxOutput[x])
		end = time.time()
		if(self.verbosePrint):
			print '%30s' % 'Processed results in ', str((end - start)*1000), 'ms'

		return topIndex, topProb

	def inference(self, frame):
		output = self.classifier.classify(frame)
		return output

	def runInference(self):
		while True:
			image_batch, image_tensor = self.raliEngine.get_next_augmentation()
			frame = image_tensor
			original_image = image_batch[0:self.h_i, 0:self.w_i]
			cloned_image = np.copy(image_batch)

			#get image file name and ground truth
			
			imageFileName = self.raliEngine.get_input_name()
			groundTruthIndex = self.raliEngine.get_ground_truth()
			groundTruthIndex = int(groundTruthIndex)
			groundTruthLabel = self.labelNames[groundTruthIndex].decode("utf-8").split(' ', 1)
			frame = image_tensor
			original_image = image_batch[0:self.h_i, 0:self.w_i]
			cloned_image = np.copy(image_batch)
			text_width, text_height = cv2.getTextSize(groundTruthLabel[1].split(',')[0], cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)[0]
			text_off_x = (self.w_i/2) - (text_width/2)
			text_off_y = self.h_i-7
			box_coords = ((text_off_x, text_off_y), (text_off_x + text_width - 2, text_off_y - text_height - 2))
			cv2.rectangle(original_image, box_coords[0], box_coords[1], (245, 197, 66), cv2.FILLED)
			cv2.putText(original_image, groundTruthLabel[1].split(',')[0], (text_off_x, text_off_y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,0), 2)

			#Step 7: call python inference. Returns output tensor with 1000 class probabilites
			output = self.inference(frame)
			#Step 8: Process output for each of the 64 images
			for i in range(self.modelBatchSizeInt):
				topIndex, topProb = self.processClassificationOutput(output)

				correctTop5 = 0; correctTop1 = 0; wrong = 0; noGroundTruth = 0;
				#create output dict for all the images
				guiResults = {}
				#to calculate FPS
				avg_benchmark = 0.0
				frameMsecs = 0.0
				frameMsecsGUI = 0.0
				totalFPS = 0.0
				resultPerAugmentation = []
				for iterator in range(self.modelBatchSizeInt):
					resultPerAugmentation.append([0,0,0])

				#create output list for each image
				augmentedResults = []

				#process the output tensor
				resultPerAugmentation, augmentedResults = self.processOutput(correctTop1, correctTop5, augmentedResults, resultPerAugmentation, groundTruthIndex,
																								topIndex, topProb, wrong, noGroundTruth, i, imageFileName)

				augmentationText = self.raliList[i].split('+')
				textCount = len(augmentationText)
				for cnt in range(0,textCount):
					currentText = augmentationText[cnt]
					text_width, text_height = cv2.getTextSize(currentText, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 2)[0]
					text_off_x = (self.w_i/2) - (text_width/2)
					text_off_y = (i*self.h_i)+self.h_i-7-(cnt*text_height)
					box_coords = ((text_off_x, text_off_y), (text_off_x + text_width - 2, text_off_y - text_height - 2))
					cv2.rectangle(cloned_image, box_coords[0], box_coords[1], (245, 197, 66), cv2.FILLED)
					cv2.putText(cloned_image, currentText, (text_off_x, text_off_y), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,0,0), 2) 

				# put augmented image result
				# if augmentedResults[i] == 0:
				# 	cv2.rectangle(cloned_image, (0,(i*(self.h_i-1)+i)),((self.w_i-1),(self.h_i-1)*(i+1) + i), (255,0,0), 4, cv2.LINE_8, 0)
				# elif augmentedResults[i] > 0  and augmentedResults[i] < 6:      
				# 	cv2.rectangle(cloned_image, (0,(i*(self.h_i-1)+i)),((self.w_i-1),(self.h_i-1)*(i+1) + i), (0,255,0), 4, cv2.LINE_8, 0)

			#Step 9: split image as needed
			if self.modelBatchSizeInt == 64:
					image_batch = np.vsplit(cloned_image, 16)
					final_image_batch = np.hstack((image_batch))
			elif self.modelBatchSizeInt == 16:
				image_batch = np.vsplit(cloned_image, 4)
				final_image_batch = np.hstack((image_batch))
			
			self.origQueue.put(original_image)
			self.augQueue.put(final_image_batch)
			
			#return original_image, final_image_batch
			#Step 10: adat generation
			# if adatFlag == False:
			# 	self.inferenceEngine.generateADAT(modelName, hierarchy)
			# 	adatFlag = True
	def setStatistics(self):
		
	def processOutput(self, correctTop1, correctTop5, augmentedResults, resultPerAugmentation, groundTruthIndex, topIndex, topProb, wrong, noGroundTruth, i, imageFileName):
		msFrame = 0.0
		msFrameGUI = 0.0
		start = time.time()
		sys.stdout = open(self.finalImageResultsFile,'a')
		print(imageFileName+','+str(groundTruthIndex)+','+str(topIndex[4 + i*4])+
		','+str(topIndex[3 + i*4])+','+str(topIndex[2 + i*4])+','+str(topIndex[1 + i*4])+','+str(topIndex[0 + i*4])+','+str(topProb[4 + i*4])+
		','+str(topProb[3 + i*4])+','+str(topProb[2 + i*4])+','+str(topProb[1 + i*4])+','+str(topProb[0 + i*4]))
		sys.stdout = self.orig_stdout
		end = time.time()
		msFrame += (end - start)*1000
		if(self.verbosePrint):
			print '%30s' % 'Image result saved in ', str((end - start)*1000), 'ms'

		#data collection for individual augmentation scores
		countPerAugmentation = resultPerAugmentation[i]

		# augmentedResults List: 0 = wrong; 1-5 = TopK; -1 = No Ground Truth
		if(groundTruthIndex == topIndex[4 + i*4]):
			correctTop1 = correctTop1 + 1
			correctTop5 = correctTop5 + 1
			augmentedResults.append(1)
			countPerAugmentation[0] = countPerAugmentation[0] + 1
			countPerAugmentation[1] = countPerAugmentation[1] + 1
		elif(groundTruthIndex == topIndex[3 + i*4] or groundTruthIndex == topIndex[2 + i*4] or groundTruthIndex == topIndex[1 + i*4] or groundTruthIndex == topIndex[0 + i*4]):
			correctTop5 = correctTop5 + 1
			countPerAugmentation[1] = countPerAugmentation[1] + 1
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
			countPerAugmentation[2] = countPerAugmentation[2] + 1

		resultPerAugmentation[i] = countPerAugmentation

		return resultPerAugmentation, augmentedResults

	def generateADAT(modelName, hierarchy):
		# Create ADAT folder and file
		print("\nADAT tool called to create the analysis toolkit\n")
		if(not os.path.exists(self.adatOutputDir)):
			os.system('mkdir ' + self.adatOutputDir)
		
		if(hierarchy == ''):
			os.system('python '+self.ADATPath+'/generate-visualization.py --inference_results '+self.finalImageResultsFile+
			' --image_dir '+self.inputImageDir+' --label '+self.labelText+' --model_name '+modelName+' --output_dir '+self.adatOutputDir+' --output_name '+modelName+'-ADAT')
		else:
			os.system('python '+self.ADATPath+'/generate-visualization.py --inference_results '+self.finalImageResultsFile+
			' --image_dir '+self.inputImageDir+' --label '+self.labelText+' --hierarchy '+self.hierarchyText+' --model_name '+modelName+' --output_dir '+self.adatOutputDir+' --output_name '+modelName+'-ADAT')
		print("\nSUCCESS: Image Analysis Toolkit Created\n")