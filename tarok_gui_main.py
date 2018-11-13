import sys
from PyQt5 import QtWidgets
from PyQt5 import QtCore
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import QFileDialog
from PyQt5.QtCore import pyqtSlot, QTimer

from tarok import *
from tarok_gui import Ui_MainWindow
import qimage2ndarray
import numpy as np
import time
import cv2

class MainWindow(QtWidgets.QMainWindow):

    def __init__(self):
        super(MainWindow, self).__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.TEngine=TarokEngine()
        #self.generiraj_seznam()
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
        self.fgbg = cv2.createBackgroundSubtractorMOG2()
        
    
    @pyqtSlot(int)
    def spremeni_sirino(self, i):
        self.ui.label_image_left.setMaximumWidth(i)
    pass

    @pyqtSlot()
    def generiraj_seznam(self):
        directory=self.ui.ime_direktorija.text()
        for filename in os.listdir(directory):
            if filename.endswith(self.ui.ending_line.text()): 
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
        if len(img.shape)<3:
            h,w=img.shape
        else:
            h,w,c=img.shape
            
        qh=label.maximumHeight()
        qw=int(w/h*qh)
        label.setMaximumWidth(qw)

    @pyqtSlot()
    def perform_analysis(self):
        directory= self.ui.ime_direktorija.text()
        filename=self.ui.listWidget.currentItem().text()

        img,karta=najdi_karto_devel(directory , filename)
        ime=template_matching_video(karta)
        self.ui.ime_karte.setText(ime)
        
        self.popravi_sirino(self.ui.label_image_left,img)
        self.display_image_on_label(self.ui.label_image_left,img)
        self.display_image_on_label(self.ui.label_image_right,karta)
        

    def display_image_on_label(self,label,frame):
        # https://gist.github.com/bsdnoobz/8464000
        if len(frame.shape)<3:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
        else:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = qimage2ndarray.array2qimage(frame)
        label.setPixmap(QPixmap.fromImage(image))

    @pyqtSlot()
    def stop_timer(self):
        self.timer.stop()

    @pyqtSlot()
    def start_timer(self):
        interval=int(self.ui.sleep_time.text())
        print(interval)
        self.timer.setInterval(interval)
        self.timer.start()
        
    @pyqtSlot()
    def setup_camera(self):
        """Initialize camera.
        """
        self.capture = cv2.VideoCapture(self.ui.ime_videa.text())
        # https://docs.opencv.org/2.4/modules/highgui/doc/reading_and_writing_images_and_video.html#videocapture-get
        #print(self.capture.get(cv2.CAP_PROP_FPS))

        self.timer = QTimer()
        self.timer.timeout.connect(self.display_video_stream)
        self.start_timer()

    @pyqtSlot()
    def display_video_stream(self):
        """Read frame from camera and repaint QLabel widget.
        """
        start = time.time()
        self.capture.grab(); # self.capture.grab() ; self.capture.grab()
        #self.capture.grab(); self.capture.grab() ; self.capture.grab()
        
        retv, frame = self.capture.retrieve()
        if retv==False:
            self.timer.stop()
            return False
        frame = cv2.pyrDown(frame)
        
        video_time=self.capture.get(cv2.CAP_PROP_POS_MSEC)
        

        
        #img,karta=najdi_karto_video(frame)

        #self.TEngine.update(frame)
        #img = self.TEngine.img
        #karta = self.TEngine.karta

        # ime = self.TEngine.ime
        # self.ui.ime_karte.setText(ime)

        frame=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

        sbval=self.ui.spinBox_gauss.value()
        if sbval>1:
            frame = cv2.medianBlur(frame,sbval)

        sbval=self.ui.spinBox_median.value()
        if sbval>1:
            frame = cv2.GaussianBlur(frame,(sbval,sbval),0)

        sbval=self.ui.spinBox_morph.value()
        if sbval>1:
            kernel = np.ones((sbval,sbval),np.uint8)
            frame = cv2.morphologyEx(frame, eval(self.ui.lineEdit_morph.text()), kernel)

        sbval=self.ui.spinBox_gauss_2.value()
        if sbval>1:
            frame = cv2.medianBlur(frame,sbval)

        sbval=self.ui.spinBox_median_2.value()
        if sbval>1:
            frame = cv2.GaussianBlur(frame,(sbval,sbval),0)

        sbval=self.ui.spinBox_morph_2.value()
        if sbval>1:
            kernel = np.ones((sbval,sbval),np.uint8)
            frame = cv2.morphologyEx(frame, eval(self.ui.lineEdit_morph_2.text()), kernel)

            
        self.test_edge_detection(frame)

        # frame = cv2.pyrDown(frame)
        # fgmask = self.fgbg.apply(frame)
        # fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_DILATE, self.kernel)
        # frame=cv2.bitwise_and(frame,frame,mask=fgmask)
        
        
        # self.popravi_sirino(self.ui.label_image_left,frame)
        # self.popravi_sirino(self.ui.label_image_bottom_left,fgmask)
        # self.display_image_on_label(self.ui.label_image_left,frame)
        # self.display_image_on_label(self.ui.label_image_bottom_left,fgmask)

        
        #self.popravi_sirino(self.ui.label_image_left,img)
        #self.display_image_on_label(self.ui.label_image_left,img)
        #self.display_image_on_label(self.ui.label_image_right,karta)

        end = time.time()
        
        self.ui.video_msec_line.setText("{:.1f}".format(video_time/1000))
        self.ui.function_time_line.setText("{:.3f}".format(end-start))

    def test_edge_detection(self,frame):
        laplacian = cv2.Laplacian(frame,cv2.CV_64F)
        laplacian = np.uint8(laplacian)
                
        scharrx = cv2.Sobel(frame,cv2.CV_64F,1,0,ksize=-1)
        scharry = cv2.Sobel(frame,cv2.CV_64F,0,1,ksize=-1)
        scharrxy = np.sqrt(scharrx**2 + scharry**2)
        scharrxy = np.uint8(scharrxy)

        sobelxy = cv2.Sobel(frame,cv2.CV_64F,1,1,ksize=5)
        sobelxy = np.uint8(sobelxy)

        self.popravi_sirino(self.ui.label_image_right,laplacian)
        self.popravi_sirino(self.ui.label_image_left,frame)
        self.popravi_sirino(self.ui.label_image_bottom_left,scharrxy)
        self.popravi_sirino(self.ui.label_karta_1,sobelxy)
        
        self.display_image_on_label(self.ui.label_image_right,laplacian)
        self.display_image_on_label(self.ui.label_image_left,frame)
        self.display_image_on_label(self.ui.label_image_bottom_left,scharrxy)
        self.display_image_on_label(self.ui.label_karta_1,sobelxy)

        
if __name__ == '__main__':

    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
