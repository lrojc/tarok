import sys
from PyQt5 import QtWidgets
from PyQt5 import QtCore
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import QFileDialog
from PyQt5.QtCore import pyqtSlot

from tarok import *
from tarok_gui import Ui_MainWindow
import qimage2ndarray
import random
import numpy as np

class MainWindow(QtWidgets.QMainWindow):

    def __init__(self):
        super(MainWindow, self).__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
                
        self.generiraj_seznam()
    
    @pyqtSlot(int)
    def spremeni_sirino(self, i):
        self.ui.label_image_left.setMaximumWidth(i)
    pass

    @pyqtSlot()
    def generiraj_seznam(self):
        directory=self.ui.ime_direktorija.text()
        for filename in os.listdir(directory):
            if filename.endswith(".jpg"): 
                self.ui.listWidget.addItem(filename)

        self.ui.listWidget.setCurrentRow(0)

    @pyqtSlot()
    def izberi_direktorij(self):
        dlg = QFileDialog()
        dlg.setFileMode(QFileDialog.Directory)
        if dlg.exec_():
            #print(dlg.selectedFiles())
            self.ui.ime_direktorija.setText(dlg.selectedFiles()[0])

    @pyqtSlot(int)
    def print_int(self,int):
        print(int)

        
    @pyqtSlot(str)
    def print_str(self,str):
        print(str)

    @pyqtSlot()
    def perform_previous(self):
        i=self.ui.listWidget.currentRow()
        if i>0:
            self.ui.listWidget.setCurrentRow(i-1)
            #self.perform_analysis()

    @pyqtSlot()
    def perform_next(self):
        i=self.ui.listWidget.currentRow()
        if i<self.ui.listWidget.count():
            self.ui.listWidget.setCurrentRow(i+1)
            #self.perform_analysis()

    def popravi_sirino(self,label,img):
        h,w,c=img.shape
        qh=label.maximumHeight()
        qw=int(w/h*qh)
        label.setMaximumWidth(qw)

    @pyqtSlot()
    def perform_analysis(self):
        directory= self.ui.ime_direktorija.text()
        filename=self.ui.listWidget.currentItem().text()

        img,karta=najdi_karto_devel(directory , filename)
        self.popravi_sirino(self.ui.label_image_left,img)
        self.display_image_on_label(self.ui.label_image_left,img)
        self.display_image_on_label(self.ui.label_image_right,karta)
        

    def display_image_on_label(self,label,frame):
        # https://gist.github.com/bsdnoobz/8464000
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = qimage2ndarray.array2qimage(frame)
        label.setPixmap(QPixmap.fromImage(image))

        
if __name__ == '__main__':

    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
