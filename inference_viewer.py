import sys, os
from PyQt4 import QtGui, uic
from PyQt4.QtGui import QImage

class inference_viewer(QtGui.QMainWindow):
    def __init__(self, model_name, total_images, parent=None):
        super(inference_viewer, self).__init__(parent)
        self.model_name = model_name
        self.total_images = total_images
        self.imgCount = 0
        self.frameCount = 8
        self.imageList = []
        self.initUI()
        self.imageList.append(self.ui.image_label)
        self.imageList.append(self.ui.image_label2)
        self.imageList.append(self.ui.image_label3)
        self.imageList.append(self.ui.image_label4)
        self.imageList.append(self.ui.image_label5)
        self.imageList.append(self.ui.image_label6)
        self.imageList.append(self.ui.image_label7)
        self.imageList.append(self.ui.image_label8)

    def initUI(self):
        self.ui = uic.loadUi("inference_viewer.ui")
        self.ui.name_label.setText(self.model_name)
        self.ui.total_progressBar.setStyleSheet("QProgressBar::chunk { background: lightblue; }")
        self.ui.top1_progressBar.setStyleSheet("QProgressBar::chunk { background: green; }")
        self.ui.top5_progressBar.setStyleSheet("QProgressBar::chunk { background: lightgreen; }")
        self.ui.mis_progressBar.setStyleSheet("QProgressBar::chunk { background: red; }")
        self.ui.noGT_progressBar.setStyleSheet("QProgressBar::chunk { background: yellow; }")
        self.ui.total_progressBar.setMaximum(self.total_images)
        self.ui.top1_progressBar.setMaximum(self.total_images)
        self.ui.top5_progressBar.setMaximum(self.total_images)
        self.ui.mis_progressBar.setMaximum(self.total_images)
        self.ui.noGT_progressBar.setMaximum(self.total_images)
        self.ui.show()

    def setTotalProgress(self, value):
        self.ui.total_progressBar.setValue(value)
        self.ui.imgProg_label.setText("Processed: %d of %d" % (value, self.total_images))

    def setTop1Progress(self, value):
        self.ui.top1_progressBar.setValue(value)
    
    def setTop5Progress(self, value):
        self.ui.top5_progressBar.setValue(value)
    
    def setMisProgress(self, value):
        self.ui.mis_progressBar.setValue(value)
    
    def setNoGTProgress(self, value):
        self.ui.noGT_progressBar.setValue(value)

    def showImage(self, image):
        width = image.shape[0]
        height = image.shape[1]
        qimage = QtGui.QImage(image, width, height, width*3, QtGui.QImage.Format_RGB888)
        #qImg = QtGui.QImage(image, width, height, QImage.Format_RGB32)
        
        self.imageList[(self.imgCount % self.frameCount)].setPixmap(QtGui.QPixmap.fromImage(qimage))
        self.imgCount += 1
        # self.imageList[(7 % self.frameCount)].setPixmap(QtGui.QPixmap.fromImage(qimage))
        
        