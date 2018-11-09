import sys
from PyQt5 import QtWidgets
from PyQt5 import QtCore
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import QFileDialog
from PyQt5.QtCore import pyqtSlot, QTimer

from tarok import *
from tarok_gui import Ui_MainWindow
import qimage2ndarray
import random
import numpy as np
import time

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
            #self.perform_analysis() #to sedaj upravlja listWidget

    @pyqtSlot()
    def perform_next(self):
        i=self.ui.listWidget.currentRow()
        if i<self.ui.listWidget.count():
            self.ui.listWidget.setCurrentRow(i+1)
            #self.perform_analysis() #to sedaj upravlja listWidget

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

    @pyqtSlot()
    def stop_timer(self):
        self.timer.stop()
        
    @pyqtSlot()
    def setup_camera(self):
        """Initialize camera.
        """
        self.capture = cv2.VideoCapture(self.ui.ime_videa.text())
        # https://docs.opencv.org/2.4/modules/highgui/doc/reading_and_writing_images_and_video.html#videocapture-get
        #print(self.capture.get(cv2.CAP_PROP_FPS))

        self.timer = QTimer()
        self.timer.timeout.connect(self.display_video_stream)
        self.timer.start(100)

    def display_video_stream(self):
        """Read frame from camera and repaint QLabel widget.
        """
        start = time.time()
        self.capture.grab(); self.capture.grab() ; self.capture.grab()
        
        _, frame = self.capture.retrieve()
        print(self.capture.get(cv2.CAP_PROP_POS_MSEC))

        #_, frame = self.capture.read()
        #frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        #frame = cv2.flip(frame, 1)
        img,karta=najdi_karto_video(frame)

        self.popravi_sirino(self.ui.label_image_left,img)
        self.display_image_on_label(self.ui.label_image_left,img)
        self.display_image_on_label(self.ui.label_image_right,karta)

        end = time.time()
        print(end - start)

        
if __name__ == '__main__':

    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
