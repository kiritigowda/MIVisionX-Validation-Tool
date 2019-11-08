import pyqtgraph as pg
import cv2
import numpy as np
from PyQt4 import QtGui, uic
from PyQt4.QtGui import QPixmap
from PyQt4.QtCore import QTime, QTimer, QThread
from inference_setup import *

class InferenceViewer(QtGui.QMainWindow):
    def __init__(self, model_name, model_format, image_dir, model_location, label, hierarchy, image_val, input_dims, output_dims, 
                                    batch_size, output_dir, add, multiply, verbose, fp16, replace, loop, rali_mode, container_logo, parent):
        super(InferenceViewer, self).__init__(parent)
        self.parent = parent

        self.model_name = model_name
        self.model_format = model_format 
        self.image_dir = image_dir
        self.model_location = model_location
        self.label = label
        self.hierarchy = hierarchy
        self.image_val = image_val
        self.input_dims = input_dims
        self.output_dims = output_dims
        self.batch_size = batch_size
        self.batch_size_int = (int)(batch_size)
        self.output_dir = output_dir
        self.add = add
        self.multiply = multiply
        self.verbose = verbose
        self.fp16 = fp16
        self.replace = replace
        self.loop = loop
        self.rali_mode = rali_mode
        inputImageDir = os.path.expanduser(image_dir)
        self.total_images = len(os.listdir(inputImageDir))
        self.imgCount = 0
        self.frameCount = 9
        self.container_index = (int)(container_logo)
        # self.origImageQueue = Queue.Queue()
        # self.augImageQueue = Queue.Queue()

        self.graph = pg.PlotWidget(title="Accuracy vs Time")
        self.x = [0] 
        self.y = [0]
        self.augAccuracy = []
        
        self.time = QTime.currentTime()

        self.runState = False
        self.pauseState = False
        self.progIndex = 0
        self.augIntensity = 0.0
        self.lastIndex = self.frameCount - 1

        self.pen = pg.mkPen('w', width=4)

        self.inferenceEngine = None
        self.receiver_thread = None
        self.AMD_Radeon_pixmap = QPixmap("./data/images/AMD_Radeon.png")
        self.AMD_Radeon_white_pixmap = QPixmap("./data/images/AMD_Radeon-white.png")
        self.MIVisionX_pixmap = QPixmap("./data/images/MIVisionX-logo.png")
        self.MIVisionX_white_pixmap = QPixmap("./data/images/MIVisionX-logo-white.png")
        self.EPYC_pixmap = QPixmap("./data/images/EPYC-blue.png")
        self.EPYC_white_pixmap = QPixmap("./data/images/EPYC-blue-white.png")
        self.docker_pixmap = QPixmap("./data/images/Docker.png")
        self.singularity_pixmap = QPixmap("./data/images/Singularity.png")
        
        self.initUI()
        self.initEngines()
        self.show()
        self.updateTimer = QTimer()
        self.updateTimer.timeout.connect(self.update)
        self.updateTimer.start(40)

    def initUI(self):
        uic.loadUi("inference_viewer.ui", self)
        #self.showMaximized()
        self.setStyleSheet("background-color: white")
        self.name_label.setText("Model: %s" % (self.model_name))
        self.dataset_label.setText("Augmentation set - %d" % (self.rali_mode))
        self.imagesFrame.setStyleSheet(".QFrame {border-width: 20px; border-image: url(./data/images/filmStrip.png);}")
        self.total_progressBar.setStyleSheet("QProgressBar::chunk { background: lightblue; }")
        self.top1_progressBar.setStyleSheet("QProgressBar::chunk { background: green; }")
        self.top5_progressBar.setStyleSheet("QProgressBar::chunk { background: lightgreen; }")
        self.mis_progressBar.setStyleSheet("QProgressBar::chunk { background: red; }")
        self.total_progressBar.setMaximum(self.total_images*self.batch_size_int)

        self.graph.setLabel('left', 'Accuracy', '%')
        self.graph.setLabel('bottom', 'Time', 's')
        self.graph.setYRange(0, 100, padding=0)
        #self.graph.addLegend()
        self.graph.plot(self.x, self.y, pen=self.pen, name='Total')
        self.verticalLayout_2.addWidget(self.graph)
        self.graph.setBackground(None)
        self.graph.setMaximumWidth(550)
        self.graph.setMaximumHeight(300)
        self.level_slider.setMaximum(100)
        self.level_slider.valueChanged.connect(self.setIntensity)
        self.pause_pushButton.setStyleSheet("color: white; background-color: darkBlue")
        self.stop_pushButton.setStyleSheet("color: white; background-color: darkRed")
        self.pause_pushButton.clicked.connect(self.pauseView)
        self.stop_pushButton.clicked.connect(self.closeView)
        self.dark_checkBox.stateChanged.connect(self.setBackground)
        self.verbose_checkBox.stateChanged.connect(self.showVerbose)
        self.dark_checkBox.setChecked(True)

        if self.container_index == 1:
            self.container_logo.setPixmap(self.docker_pixmap)
        elif self.container_index == 2:
            self.container_logo.setPixmap(self.singularity_pixmap)
        else:
            self.container_logo.hide()

        for augmentation in range(self.batch_size_int):
            self.augAccuracy.append([0])

        self.showVerbose()

    def initEngines(self):
        
        self.receiver_thread = QThread()
        # Creating an object for inference.
        self.inferenceEngine = modelInference(self.model_name, self.model_format, self.image_dir, self.model_location, self.label, self.hierarchy, self.image_val,
                                                self.input_dims, self.output_dims, self.batch_size, self.output_dir, self.add, self.multiply, self.verbose, self.fp16, 
                                                self.replace, self.loop, self.rali_mode)
        
        self.inferenceEngine.moveToThread(self.receiver_thread)
        self.receiver_thread.started.connect(self.inferenceEngine.runInference)
        #self.inferenceEngine.finished.connect(self.inferenceEngine.quit)
        #self.inferenceEngine.finished.connect(self.inferenceEngine.deleteLater)
        self.receiver_thread.finished.connect(self.inferenceEngine.deleteLater)
        self.receiver_thread.start()
        self.receiver_thread.terminate()

    def paintEvent(self, event):

        #painter = QtGui.QPainter(self)
        #painter.setRenderHint()
        # update parameters for the augmentation & get 64 augmentations for an image
        augmentation = self.getIntensity()
        #self.raliEngine.updateAugmentationParameter(augmentation)

        original_image, aug_image = self.inferenceEngine.runInference()
        width = original_image.shape[1]
        height = original_image.shape[0]
        self.showImage(original_image, width, height)
        width = aug_image.shape[1]
        height = aug_image.shape[0]
        self.showAugImage(aug_image, width, height)

    
    def resetViewer(self):
        self.imgCount = 0
        del self.x[:]
        self.x.append(0)
        del self.y[:]
        self.y.append(0)
        del self.augAccuracy[:]
        for augmentation in range(self.batch_size_int):
            self.augAccuracy.append([0])

        self.time = QTime.currentTime()
        self.lastTime = 0
        self.progIndex = 0
        self.lastIndex = self.frameCount - 1
        self.graph.clear()

    def setTotalProgress(self, value):
        self.total_progressBar.setValue(value)
        if self.getIndex() == 0:
            self.total_progressBar.setMaximum(self.total_images*self.batch_size)
            self.imgProg_label.setText("Processed: %d of %d" % (value, self.total_images*self.batch_size))
        else:
            self.total_progressBar.setMaximum(self.total_images)
            self.imgProg_label.setText("Processed: %d of %d" % (value, self.total_images))
    
    def setTop1Progress(self, value, total):
        self.top1_progressBar.setValue(value)
        self.top1_progressBar.setMaximum(total)
    
    def setTop5Progress(self, value, total):
        self.top5_progressBar.setValue(value)
        self.top5_progressBar.setMaximum(total)
    
    def setMisProgress(self, value, total):
        self.mis_progressBar.setValue(value)
        self.mis_progressBar.setMaximum(total)
    
    # def setNoGTProgress(self, value):
    #     self.noGT_progressBar.setValue(value)

    def plotGraph(self, accuracy):
        curTime = self.time.elapsed()/1000.0
        if (curTime - self.lastTime > 0.1):
            self.x.append(curTime)
            self.y.append(accuracy)
            self.graph.plot(self.x, self.y, pen=self.pen)
            if self.progIndex:
                self.graph.plot(self.x, self.augAccuracy[self.progIndex-1], pen=pg.mkPen('b', width=4))
            self.lastTime = curTime

    def showImage(self, image, width, height):
        qimage = QtGui.QImage(image, width, height, width*3, QtGui.QImage.Format_RGB888)
        qimage_resized = qimage.scaled(self.image_label.width(), self.image_label.height(), QtCore.Qt.IgnoreAspectRatio)
        index = self.imgCount % self.frameCount
        self.origImage_layout.itemAt(index).widget().setPixmap(QtGui.QPixmap.fromImage(qimage_resized))
        self.origImage_layout.itemAt(index).widget().setStyleSheet("border: 5px solid yellow;");
        self.origImage_layout.itemAt(self.lastIndex).widget().setStyleSheet("border: 0");
        self.imgCount += 1
        self.lastIndex = index

    def showAugImage(self, image, width, height):
        qimage = QtGui.QImage(image, width, height, width*3, QtGui.QImage.Format_RGB888)
        if self.batch_size == 64:
            qimage_resized = qimage.scaled(self.aug_label.width(), self.aug_label.height(), QtCore.Qt.IgnoreAspectRatio)
            pixmap = QtGui.QPixmap.fromImage(qimage_resized)
            self.aug_label.setPixmap(pixmap)
        elif self.batch_size == 16:
            qimage_resized = qimage.scaled(self.aug_label.width(), self.aug_label.height(), QtCore.Qt.KeepAspectRatio)
            pixmap = QtGui.QPixmap.fromImage(qimage_resized)
            self.aug_label.setPixmap(pixmap)

    # def putAugImage(self, image, width, height):
    #     qimage = QtGui.QImage(image, width, height, width*3, QtGui.QImage.Format_RGB888)
    #     qimage_resized = qimage.scaled(self.aug_label.width(), self.aug_label.height(), QtCore.Qt.KeepAspectRatio)
    #     pixmap = QtGui.QPixmap.fromImage(qimage_resized)
    #     self.augImageQueue.put(pixmap)

    # def putImage(self, image, width, height):
    #     qimage = QtGui.QImage(image, width, height, width*3, QtGui.QImage.Format_RGB888)
    #     qimage_resized = qimage.scaled(self.image_label.width(), self.image_label.height(), QtCore.Qt.KeepAspectRatio)
    #     pixmap = QtGui.QPixmap.fromImage(qimage_resized)
    #     self.origImageQueue.put(pixmap)

    # def showImage(self):
    #     if not self.origImageQueue.empty():
    #         origImage = self.origImageQueue.get()
    #         augImage = self.augImageQueue.get()
    #         self.imageList[(self.imgCount % self.frameCount)].setPixmap(origImage)
    #         self.aug_label.setPixmap(augImage)
    #         self.imgCount += 1

    def keyPressEvent(self, event):
        if event.key() == QtCore.Qt.Key_Escape:
            self.closeView()
            
        if event.key() == QtCore.Qt.Key_Space:
            self.pauseView()

    def mousePressEvent(self, event):
        if event.button() == QtCore.Qt.LeftButton:
            mousePos = event.pos()
            if self.aug_label.geometry().contains(mousePos):
                index = self.calculateIndex(mousePos.x(), mousePos.y())
                self.progIndex = index
            else:
                self.progIndex = 0
            
            self.graph.clear()

    def setBackground(self):
        if self.dark_checkBox.isChecked():
            self.setStyleSheet("background-color: #25232F;")
            self.pen = pg.mkPen('w', width=4)
            self.graph.setBackground(None)
            self.origTitle_label.setStyleSheet("color: #C82327;")
            self.controlTitle_label.setStyleSheet("color: #C82327;")
            self.progTitle_label.setStyleSheet("color: #C82327;")
            self.graphTitle_label.setStyleSheet("color: #C82327;")
            self.augTitle_label.setStyleSheet("color: #C82327;")
            self.name_label.setStyleSheet("color: white;")
            self.dataset_label.setStyleSheet("color: white;")
            self.imgProg_label.setStyleSheet("color: white;")
            self.fps_label.setStyleSheet("color: #C82327;")
            self.dark_checkBox.setStyleSheet("color: white;")
            self.verbose_checkBox.setStyleSheet("color: white;")
            self.level_label.setStyleSheet("color: white;")
            self.low_label.setStyleSheet("color: white;")
            self.high_label.setStyleSheet("color: white;")
            self.AMD_logo.setPixmap(self.AMD_Radeon_white_pixmap)
            self.MIVisionX_logo.setPixmap(self.MIVisionX_white_pixmap)
            self.EPYC_logo.setPixmap(self.EPYC_white_pixmap)
        else:
            self.setStyleSheet("background-color: white;")
            self.pen = pg.mkPen('k', width=4)
            self.graph.setBackground(None)
            self.origTitle_label.setStyleSheet("color: 0;")
            self.controlTitle_label.setStyleSheet("color: 0;")
            self.progTitle_label.setStyleSheet("color: 0;")
            self.graphTitle_label.setStyleSheet("color: 0;")
            self.augTitle_label.setStyleSheet("color: 0;")
            self.name_label.setStyleSheet("color: 0;")
            self.dataset_label.setStyleSheet("color: 0;")
            self.imgProg_label.setStyleSheet("color: 0;")
            self.fps_label.setStyleSheet("color: 0;")
            self.dark_checkBox.setStyleSheet("color: 0;")
            self.verbose_checkBox.setStyleSheet("color: 0;")
            self.level_label.setStyleSheet("color: 0;")
            self.low_label.setStyleSheet("color: 0;")
            self.high_label.setStyleSheet("color: 0;")
            self.AMD_logo.setPixmap(self.AMD_Radeon_pixmap)
            self.MIVisionX_logo.setPixmap(self.MIVisionX_pixmap)
            self.EPYC_logo.setPixmap(self.EPYC_pixmap)
            
    def showVerbose(self):
        if self.verbose_checkBox.isChecked():
            self.dataset_label.show()
            self.fps_label.show()
            self.fps_lcdNumber.show()
        else:
            self.dataset_label.hide()
            self.fps_label.hide()
            self.fps_lcdNumber.hide()
        
    def displayFPS(self, fps):
        self.fps_lcdNumber.display(fps)

    def pauseView(self):
        self.pauseState = not self.pauseState
        if self.pauseState:
            self.pause_pushButton.setText('Resume')
        else:
            self.pause_pushButton.setText('Pause')

    def closeView(self):
        self.runState = False

    def startView(self):
        self.runState = True

    def stopView(self):
        self.runState = False

    def getState(self):
        return self.runState

    def isPaused(self):
        return self.pauseState

    def setIntensity(self):
        self.augIntensity = (float)(self.level_slider.value()) / 100.0

    def getIntensity(self):
        return self.augIntensity

    def calculateIndex(self, x, y):
        if self.batch_size == 64:
            imgWidth = self.aug_label.width() / 16.0
        else:
            imgWidth = self.aug_label.width() / 4.0
        imgHeight = self.aug_label.height() / 4.0
        x -= self.aug_label.x()
        y -= self.aug_label.y()
        column = (int)(x / imgWidth)
        row = (int)(y / imgHeight)
        index = 4 * column + row
        return index + 1

    def getIndex(self):
        return self.progIndex
    
    def setAugName(self, name):
        self.name_label.setText(name)

    def storeAccuracy(self, index, accuracy):
        curTime = self.time.elapsed()/1000.0
        if (curTime - self.lastTime > 0.1):
            self.augAccuracy[index].append(accuracy)
