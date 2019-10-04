import sys, os
from PyQt4 import QtGui, uic, QtCore
from PyQt4.QtGui import QImage

class inference_viewer(QtGui.QMainWindow):
    def __init__(self, model_name, total_images, parent=None):
        super(inference_viewer, self).__init__(parent)
        self.model_name = model_name
        self.total_images = total_images
        self.imgCount = 0
        self.frameCount = 16
        self.imageList = []
        self.initUI()
        self.imageList.append(self.image_label)
        self.imageList.append(self.image_label2)
        self.imageList.append(self.image_label3)
        self.imageList.append(self.image_label4)
        self.imageList.append(self.image_label5)
        self.imageList.append(self.image_label6)
        self.imageList.append(self.image_label7)
        self.imageList.append(self.image_label8)
        self.imageList.append(self.image_label9)
        self.imageList.append(self.image_label10)
        self.imageList.append(self.image_label11)
        self.imageList.append(self.image_label12)
        self.imageList.append(self.image_label13)
        self.imageList.append(self.image_label14)
        self.imageList.append(self.image_label15)
        self.imageList.append(self.image_label16)

    def initUI(self):
        uic.loadUi("inference_viewer.ui", self)
        self.name_label.setText(self.model_name)
        self.total_progressBar.setStyleSheet("QProgressBar::chunk { background: lightblue; }")
        self.top1_progressBar.setStyleSheet("QProgressBar::chunk { background: green; }")
        self.top5_progressBar.setStyleSheet("QProgressBar::chunk { background: lightgreen; }")
        self.mis_progressBar.setStyleSheet("QProgressBar::chunk { background: red; }")
        self.noGT_progressBar.setStyleSheet("QProgressBar::chunk { background: yellow; }")
        self.total_progressBar.setMaximum(self.total_images)
        self.top1_progressBar.setMaximum(self.total_images)
        self.top5_progressBar.setMaximum(self.total_images)
        self.mis_progressBar.setMaximum(self.total_images)
        self.noGT_progressBar.setMaximum(self.total_images)
        self.show()

    def setTotalProgress(self, value):
        self.total_progressBar.setValue(value)
        self.imgProg_label.setText("Processed: %d of %d" % (value, self.total_images))

    def setTop1Progress(self, value):
        self.top1_progressBar.setValue(value)
    
    def setTop5Progress(self, value):
        self.top5_progressBar.setValue(value)
    
    def setMisProgress(self, value):
        self.mis_progressBar.setValue(value)
    
    def setNoGTProgress(self, value):
        self.noGT_progressBar.setValue(value)

    def showImage(self, image):
        width = image.shape[0]
        height = image.shape[1]
        qimage = QtGui.QImage(image, width, height, width*3, QtGui.QImage.Format_RGB888)
        qimage_resized = qimage.scaled(self.image_label.width(), self.image_label.height(), QtCore.Qt.KeepAspectRatio)
        self.imageList[(self.imgCount % self.frameCount)].setPixmap(QtGui.QPixmap.fromImage(qimage_resized))
        self.imgCount += 1

    def showAugImage(self, image):
        width = image.shape[0]
        height = image.shape[1]
        qimage = QtGui.QImage(image, width, height, width*3, QtGui.QImage.Format_RGB888)
        qimage_resized = qimage.scaled(self.aug_label.width(), self.aug_label.height(), QtCore.Qt.KeepAspectRatio)
        self.aug_label.setPixmap(QtGui.QPixmap.fromImage(qimage_resized))

    def keyPressEvent(self, event):
        if event.key() == QtCore.Qt.Key_Escape:
            self.close()
            
        if event.key() == QtCore.Qt.Key_Space:
            print 'space'
