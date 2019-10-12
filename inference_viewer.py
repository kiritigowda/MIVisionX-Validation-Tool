import sys, os
import Queue
#import numpy as np
import pyqtgraph as pg
from PyQt4 import QtGui, uic, QtCore
from PyQt4.QtGui import QImage
from PyQt4.QtCore import QTimer, QObject, QTime

class inference_viewer(QtGui.QMainWindow):
    def __init__(self, model_name, total_images, parent=None):
        super(inference_viewer, self).__init__(parent)
        self.model_name = model_name
        self.total_images = total_images
        self.imgCount = 0
        self.frameCount = 9
        self.imageList = []
        # self.origImageQueue = Queue.Queue()
        # self.augImageQueue = Queue.Queue()
        
        self.initUI()
        
        self.graph = pg.PlotWidget(title="Accuracy vs Time")
        self.graph.setLabel('left', 'Accuracy', '%')
        self.graph.setLabel('bottom', 'Time', 's')
        self.x = [0] 
        self.y = [0]
        self.graph.plot(self.x, self.y)
        self.verticalLayout_2.addWidget(self.graph)
        self.graph.setMaximumWidth(550)
        self.graph.setMaximumHeight(400)
        self.graph.setBackground((255,255,255))
        self.time = QTime.currentTime()
        self.lastTime = 0
        self.imageList.append(self.image_label)
        self.imageList.append(self.image_label2)
        self.imageList.append(self.image_label3)
        self.imageList.append(self.image_label4)
        self.imageList.append(self.image_label5)
        self.imageList.append(self.image_label6)
        self.imageList.append(self.image_label7)
        self.imageList.append(self.image_label8)
        self.imageList.append(self.image_label9)
        # self.imageList.append(self.image_label10)
        # self.imageList.append(self.image_label11)
        # self.imageList.append(self.image_label12)
        # self.imageList.append(self.image_label13)
        # self.imageList.append(self.image_label14)
        # self.imageList.append(self.image_label15)
        # self.imageList.append(self.image_label16)

        self.pause_pushButton.clicked.connect(self.pauseView)
        self.stop_pushButton.clicked.connect(self.closeView)

        self.runState = False
        self.pauseState = False
        
        self.show()
        # self.timer = QTimer(self)
        # QtCore.QTimer.connect(self.timer, QtCore.SIGNAL("timeout()"), self, QtCore.SLOT("showImage()"))
        # self.timer.timeout.connect(self.showImage)
        # self.timer.start(40)

    def initUI(self):
        uic.loadUi("inference_viewer.ui", self)
        self.setStyleSheet("background-color: white")
        self.name_label.setText(self.model_name)
        self.total_progressBar.setStyleSheet("QProgressBar::chunk { background: lightblue; }")
        self.top1_progressBar.setStyleSheet("QProgressBar::chunk { background: green; }")
        self.top5_progressBar.setStyleSheet("QProgressBar::chunk { background: lightgreen; }")
        self.mis_progressBar.setStyleSheet("QProgressBar::chunk { background: red; }")
        self.total_progressBar.setMaximum(self.total_images)
        #self.noGT_progressBar.setStyleSheet("QProgressBar::chunk { background: yellow; }"

    def setTotalProgress(self, value):
        self.total_progressBar.setValue(value)
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
            self.graph.plot(self.x, self.y)
            self.lastTime = curTime

    def showImage(self, image, width, height):
        qimage = QtGui.QImage(image, width, height, width*3, QtGui.QImage.Format_RGB888)
        qimage_resized = qimage.scaled(self.image_label.width(), self.image_label.height(), QtCore.Qt.KeepAspectRatio)
        self.imageList[(self.imgCount % self.frameCount)].setPixmap(QtGui.QPixmap.fromImage(qimage_resized))
        self.imgCount += 1


    def showAugImage(self, image, width, height):
        qimage = QtGui.QImage(image, width, height, width*3, QtGui.QImage.Format_RGB888)
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