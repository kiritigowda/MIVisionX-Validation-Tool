import sys, os
from PyQt4 import QtGui, uic

class inference_control(QtGui.QMainWindow):
    def __init__(self, parent=None):
        super(inference_control, self).__init__(parent)
        self.ui = uic.loadUi("inference_control.ui")
        #self.ui.setStyleSheet("background-color: white")
        self.model_format = ''
        self.model_name = ''
        self.model = ''
        self.input_dims = ''
        self.output_dims = ''
        self.label = ''
        self.output = ''
        self.image = ''
        self.val = ''
        self.hier = ''
        self.add = '0,0,0'
        self.multiply = '1,1,1'
        self.fp16 = 'no'
        self.replace = 'no'
        self.verbose = 'no'

        self.ui.upload_comboBox.activated.connect(self.fromFile)
        self.ui.file_pushButton.clicked.connect(self.browseFile)
        self.ui.output_pushButton.clicked.connect(self.browseOutput)
        self.ui.label_pushButton.clicked.connect(self.browseLabel)
        self.ui.image_pushButton.clicked.connect(self.browseImage)
        self.ui.val_pushButton.clicked.connect(self.browseVal)
        self.ui.hier_pushButton.clicked.connect(self.browseHier)
        self.ui.run_pushButton.clicked.connect(self.runConfig)
        self.ui.file_lineEdit.textChanged.connect(self.checkInput)
        self.ui.name_lineEdit.textChanged.connect(self.checkInput)
        self.ui.idims_lineEdit.textChanged.connect(self.checkInput)
        self.ui.odims_lineEdit.textChanged.connect(self.checkInput)
        self.ui.output_lineEdit.textChanged.connect(self.checkInput)
        self.ui.label_lineEdit.textChanged.connect(self.checkInput)
        self.ui.image_lineEdit.textChanged.connect(self.checkInput)
        self.ui.image_lineEdit.textChanged.connect(self.checkInput)

        self.ui.idims_lineEdit.setPlaceholderText("c,h,w [required]")
        self.ui.odims_lineEdit.setPlaceholderText("c,h,w [required]")
        self.ui.padd_lineEdit.setPlaceholderText("r,g,b [optional]")
        self.ui.pmul_lineEdit.setPlaceholderText("r,g,b [optional]")
        self.ui.val_lineEdit.setPlaceholderText("[optional]")
        self.ui.hier_lineEdit.setPlaceholderText("[optional]")
        self.readSetupFile()
        self.ui.show()

    def browseFile(self):
        if self.ui.format_comboBox.currentText() == 'nnef':
            self.ui.file_lineEdit.setText(QtGui.QFileDialog.getExistingDirectory(self, 'Open Folder', './'))    
        else:
            self.ui.file_lineEdit.setText(QtGui.QFileDialog.getOpenFileName(self, 'Open File', './', '*'))

    def browseOutput(self):
        self.ui.output_lineEdit.setText(QtGui.QFileDialog.getExistingDirectory(self, 'Open Folder', './'))

    def browseLabel(self):
        self.ui.label_lineEdit.setText(QtGui.QFileDialog.getOpenFileName(self, 'Open File', './', '*.txt'))

    def browseImage(self):
        self.ui.image_lineEdit.setText(QtGui.QFileDialog.getExistingDirectory(self, 'Open Folder', './'))

    def browseVal(self):
        self.ui.val_lineEdit.setText(QtGui.QFileDialog.getOpenFileName(self, 'Open File', './', '*.txt'))

    def browseHier(self):
        self.ui.hier_lineEdit.setText(QtGui.QFileDialog.getOpenFileName(self, 'Open File', './', '*.csv'))

    def readSetupFile(self):
        setupDir = '~/.mivisionx-inference-analyzer'
        analyzerDir = os.path.expanduser(setupDir)
        for line in open(analyzerDir + "/setupFile.txt", "r"):
            token = line.split(';')
            if len(token) > 1:
                modelName = token[1]
            self.ui.upload_comboBox.addItem(modelName)
            
    def fromFile(self):
        if self.ui.upload_comboBox.currentIndex() == 0:
            self.ui.name_lineEdit.setEnabled(True)
            self.ui.file_lineEdit.setEnabled(True)
            self.ui.idims_lineEdit.setEnabled(True)
            self.ui.odims_lineEdit.setEnabled(True)
            self.ui.label_lineEdit.setEnabled(True)
            self.ui.output_lineEdit.setEnabled(True)
            self.ui.image_lineEdit.setEnabled(True)
            self.ui.val_lineEdit.setEnabled(True)
            self.ui.hier_lineEdit.setEnabled(True)
            self.ui.padd_lineEdit.setEnabled(True)
            self.ui.pmul_lineEdit.setEnabled(True)
            self.ui.fp16_checkBox.setEnabled(True)
            self.ui.replace_checkBox.setEnabled(True)
            self.ui.verbose_checkBox.setEnabled(True)
            self.ui.file_pushButton.setEnabled(True)
            self.ui.format_comboBox.setEnabled(True)
            self.ui.output_pushButton.setEnabled(True)
            self.ui.label_pushButton.setEnabled(True)
            self.ui.image_pushButton.setEnabled(True)
            self.ui.val_pushButton.setEnabled(True)
            self.ui.hier_pushButton.setEnabled(True)
            self.ui.format_comboBox.setCurrentIndex(0)
            self.ui.name_lineEdit.clear()
            self.ui.file_lineEdit.clear()
            self.ui.idims_lineEdit.clear()
            self.ui.odims_lineEdit.clear()
            self.ui.label_lineEdit.clear()
            self.ui.output_lineEdit.clear()
            self.ui.image_lineEdit.clear()
            self.ui.val_lineEdit.clear()
            self.ui.hier_lineEdit.clear()
            self.ui.padd_lineEdit.clear()
            self.ui.pmul_lineEdit.clear()
            self.ui.fp16_checkBox.setChecked(False)
            self.ui.replace_checkBox.setChecked(False)
            self.ui.verbose_checkBox.setChecked(False)
        else:
            modelName = self.ui.upload_comboBox.currentText()
            setupDir = '~/.mivisionx-inference-analyzer'
            analyzerDir = os.path.expanduser(setupDir)
            for line in open(analyzerDir + "/setupFile.txt", "r"):
                tokens = line.split(';')
                if len(tokens) > 1:
                    name = tokens[1]
                    if modelName == name:
                        if tokens[0] == 'caffe':
                            format = 0
                        elif tokens[0] == 'onnx':
                            format = 1
                        else:
                            format = 2
                        self.ui.format_comboBox.setCurrentIndex(format)
                        self.ui.name_lineEdit.setText(tokens[1])
                        self.ui.file_lineEdit.setText(tokens[2])
                        self.ui.idims_lineEdit.setText(tokens[3])
                        self.ui.odims_lineEdit.setText(tokens[4])
                        self.ui.label_lineEdit.setText(tokens[5])
                        self.ui.output_lineEdit.setText(tokens[6])
                        self.ui.image_lineEdit.setText(tokens[7])
                        self.ui.val_lineEdit.setText(tokens[8])
                        self.ui.hier_lineEdit.setText(tokens[9])
                        self.ui.padd_lineEdit.setText(tokens[10])
                        self.ui.pmul_lineEdit.setText(tokens[11])
                        self.ui.fp16_checkBox.setChecked(True) if tokens[12] == 'yes\n' or tokens[12] == 'yes' else self.ui.fp16_checkBox.setChecked(False)
                        self.ui.replace_checkBox.setChecked(True) if tokens[13] == 'yes\n' or tokens[13] == 'yes' else self.ui.replace_checkBox.setChecked(False)
                        self.ui.verbose_checkBox.setChecked(True) if tokens[14] == 'yes\n' or tokens[14] == 'yes' else self.ui.verbose_checkBox.setChecked(False)
                        self.ui.name_lineEdit.setEnabled(False)
                        self.ui.file_lineEdit.setEnabled(False)
                        self.ui.idims_lineEdit.setEnabled(False)
                        self.ui.odims_lineEdit.setEnabled(False)
                        self.ui.label_lineEdit.setEnabled(False)
                        self.ui.output_lineEdit.setEnabled(False)
                        self.ui.image_lineEdit.setEnabled(False)
                        self.ui.val_lineEdit.setEnabled(False)
                        self.ui.hier_lineEdit.setEnabled(False)
                        self.ui.padd_lineEdit.setEnabled(False)
                        self.ui.pmul_lineEdit.setEnabled(False)
                        self.ui.fp16_checkBox.setEnabled(False)
                        self.ui.output_pushButton.setEnabled(False)
                        self.ui.label_pushButton.setEnabled(False)
                        self.ui.image_pushButton.setEnabled(False)
                        self.ui.val_pushButton.setEnabled(False)
                        self.ui.hier_pushButton.setEnabled(False)
                        self.ui.replace_checkBox.setEnabled(False)
                        self.ui.verbose_checkBox.setEnabled(False)
                        self.ui.file_pushButton.setEnabled(False)
                        self.ui.format_comboBox.setEnabled(False)

    def checkInput(self):
        if not self.ui.file_lineEdit.text().isEmpty() and not self.ui.name_lineEdit.text().isEmpty() \
            and not self.ui.idims_lineEdit.text().isEmpty() and not self.ui.odims_lineEdit.text().isEmpty() \
            and not self.ui.output_lineEdit.text().isEmpty() and not self.ui.label_lineEdit.text().isEmpty() \
            and not self.ui.image_lineEdit.text().isEmpty():
                self.ui.run_pushButton.setEnabled(True)
                self.ui.run_pushButton.setStyleSheet("background-color: lightgreen")
        else:
            self.ui.run_pushButton.setEnabled(False)
            self.ui.run_pushButton.setStyleSheet("background-color: 0")

    def runConfig(self):
        self.model_format = self.ui.format_comboBox.currentText()
        self.model_name = self.ui.name_lineEdit.text()
        self.model = self.ui.file_lineEdit.text()
        self.input_dims = '%s' % (self.ui.idims_lineEdit.text())
        self.output_dims = '%s' % (self.ui.odims_lineEdit.text())
        self.label = self.ui.label_lineEdit.text()
        self.output = self.ui.output_lineEdit.text()
        self.image = self.ui.image_lineEdit.text()
        self.val = self.ui.val_lineEdit.text()
        self.hier = self.ui.hier_lineEdit.text()
        if len(self.ui.padd_lineEdit.text()) < 1:
            self.add = '[0,0,0]'
        else:
            self.add = '[%s]' % (self.ui.padd_lineEdit.text())
        if len(self.ui.pmul_lineEdit.text()) < 1:
            self.multiply = '[1,1,1]'
        else:
            self.multiply = '[%s]' % (self.ui.pmul_lineEdit.text())
        self.fp16 = 'yes' if self.ui.fp16_checkBox.isChecked() else 'no'
        self.replace = 'yes' if self.ui.replace_checkBox.isChecked() else 'no'
        self.verbose = 'yes' if self.ui.verbose_checkBox.isChecked() else 'no'

        self.ui.close()