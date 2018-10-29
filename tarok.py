import cv2
import numpy as np
import sys,os
import matplotlib.pyplot as plt

def _plot_image_correct_color_(img1):
    if len(img1.shape)<3:
        plt.imshow(img1,cmap='gray')
    else:
        img1 = cv2.cvtColor(img1,cv2.COLOR_BGR2RGB)
        plt.imshow(img1)
    plt.xticks([]), plt.yticks([])
    
def display_image(img):
    #Crnobela ali barvna
    if len(img.shape)<3:
        plt.imshow(img,cmap='gray')
        plt.title('grayscale') #Give this plot a title, 
    else:
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        plt.imshow(img)
        plt.title('Colorfull') #Give this plot a title, 
    plt.show()

def display_image2(img1,img2):
    plt.subplot(121)
    if len(img1.shape)<3:
        plt.imshow(img1,cmap='gray')
        #plt.title('grayscale') #Give this plot a title, 
    else:
        img1 = cv2.cvtColor(img1,cv2.COLOR_BGR2RGB)
        plt.imshow(img1)
        #plt.title('Colorfull') #Give this plot a title, 
    plt.subplot(122)
    if len(img2.shape)<3:
        plt.imshow(img2,cmap='gray')
        #plt.title('grayscale') #Give this plot a title, 
    else:
        img2 = cv2.cvtColor(img2,cv2.COLOR_BGR2RGB)
        plt.imshow(img2)
        #plt.title('Colorfull') #Give this plot a title, 
    plt.show()

def display_image3(img1,img2,img3):
    plt.subplot(131)
    _plot_image_correct_color_(img1)
    plt.subplot(132)
    _plot_image_correct_color_(img2)
    plt.subplot(133)
    _plot_image_correct_color_(img3)
    plt.show()
    
def display_image4(img1,img2,img3,img4):
    plt.subplot(141)
    _plot_image_correct_color_(img1)
    plt.subplot(142)
    _plot_image_correct_color_(img2)
    plt.subplot(143)
    _plot_image_correct_color_(img3)
    plt.subplot(144)
    _plot_image_correct_color_(img4)

    plt.show()

def popravi_predznak(lines):
    # Če je rho manjsi od nic zamenjaj predznak in obrni theta za pol kroga
    # Prav tako popravi obliko array-a
    new_lines=np.zeros((len(lines),2))
    for i in range(len(lines)):
        # V resnici je bolje ce predznaka ne spreminjamo...
        #if (lines[i,0,0]<0):
            #lines[i,0,0]*=-1
            #lines[i,0,1]-=np.pi
        new_lines[i]=lines[i][0]
    return new_lines

def je_blizu(line1,line2,dtheta,drho):
    if(abs(line1[0]-line2[0])>dtheta):
        return False
    if(abs(line1[1]-line2[1])>drho):
        return False
    return True

def dobre_crte(lines):
    nf=np.fmod(lines[:,1]+np.pi,np.pi/2)
    nf[nf<np.pi/45]+=np.pi/2
    #    plt.plot(nf)
    #    plt.show()
    #hst,edg=np.histogram(nf,bins=180,weights=np.linspace(3,2,len(nf)))
    hst,edg=np.histogram(nf,bins=180)
    i=np.argmax(hst)
    povp=edg[i]/2+edg[i+1]/2
    
    dobre=[]
    slabe=[]
    i=0
    #print("povprecje " + str (povp))
    for rho, theta in lines:
        #slabe so na stedini, dobre pa ob robovih
        if abs(np.fmod((theta+np.pi),np.pi/2)-povp)<np.pi/45: #(np.pi/60 je 3deg)
            #print("Dobra theta: " + str(theta) + " po enacbi: " + str(np.fmod((theta+np.pi/2),np.pi/2))) 
            dobre.append(i)
        else:
            #print("Slaba theta: " + str(theta) + " po enacbi: " + str(np.fmod((theta+np.pi/2),np.pi/2))) 
            slabe.append(i)
        i+=1
    if len(dobre)==0:
        print("ERROR! Ni dobrih crt")
        return
    return dobre,slabe

def navpicne(lines):
    navpicne=[]
    vodoravne=[]
    i=0
    prva_navp=np.min(lines,0)[1] # v primeru da karta pade pod 45st.
    
    for rho, theta in lines:
        if theta<(prva_navp+np.pi/4) or theta>(prva_navp+3*np.pi/4):
            vodoravne.append(i)
        else:
            navpicne.append(i)
        i+=1

    return navpicne,vodoravne

def _sort_(array,dim):
    return array[np.argsort(array[:,dim])]

def presek(line1,line2):
    r1=line1[0]
    t1=line1[1]
    r2=line2[0]
    t2=line2[1]
    x=-(r1*np.sin(t2)-r2*np.sin(t1))/np.sin(t1-t2)
    y=-(r1*np.cos(t2)-r2*np.cos(t1))/np.sin(t2-t1)
    return x,y

def vogali(lines):
    w=np.linspace(3,1,len(lines))
    nav_i,vod_i=navpicne(lines)
    nav=lines[nav_i]
    nav=_sort_(nav,0) #sortiraj po rho
    #print (nav.size)
    if nav.size>2:
        #pol=np.average(nav[:,0],weights=w[nav_i])
        #pol=np.average(nav[:,0])
        pol=(nav[0,0]+nav[-1,0])/2
    else:
        pol=nav[0,0]
        
    nav_lev=nav[nav[:,0]<=pol]
    nav_des=nav[nav[:,0]>=pol]
    vod=lines[vod_i]
    vod=_sort_(vod,0) #sortiraj po rho
    #print (vod.size)
    if vod.size>2:
        #pol=np.average(vod[:,0],weights=w[vod_i])
        #pol=np.average(vod[:,0])
        pol=(vod[0,0]+vod[-1,0])/2
    else:
        pol=vod[0,0]
        
    vod_zg=vod[vod[:,0]<=pol]
    vod_sp=vod[vod[:,0]>=pol]

    nav_lev=np.median(nav_lev,0)
    nav_des=np.median(nav_des,0)
    vod_zg=np.median(vod_zg,0)
    vod_sp=np.median(vod_sp,0)

    # if (vod_sp[0]-vod_zg[0])<(nav_des[0]-nav_lev[0]):
    #     vod_zg,nav_lev = nav_lev,vod_zg
    #     vod_sp,nav_des = nav_des,vod_sp
    
    robovi=np.array([vod_zg,vod_sp,nav_lev,nav_des])
    
    x0,y0=presek(vod_zg,nav_lev)
    x1,y1=presek(vod_zg,nav_des)
    if x0>x1:
        x0,x1=x1,x0
        y0,y1=y1,y0
        
    x2,y2=presek(vod_sp,nav_lev)
    x3,y3=presek(vod_sp,nav_des)

    if x2>x3:
        x2,x3=x3,x2
        y2,y3=y3,y2

    #vogali = np.float32([[x0,y0],[x1,y1],[x2,y2],[x3,y3]])
    
    if(((x0-x1)**2+(y0-y1)**2)<((x0-x2)**2+(y0-y2)**2)):
        vogali = np.float32([[x0,y0],[x1,y1],[x2,y2],[x3,y3]])
    else:
        vogali = np.float32([[x1,y1],[x3,y3],[x0,y0],[x2,y2]])
    
    #print("Presecicsa")
    #print(vogali)
    return vogali,robovi

def narisi_crte(img,lines,barva,debelina):
    for rho, theta in lines:
        # if (theta<np.pi/4) or (theta>3*np.pi/4):
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a*rho
        y0 = b*rho
        x1 = int(x0 + 10000*(-b))
        y1 = int(y0 + 10000*(a))
        x2 = int(x0 - 10000*(-b))
        y2 = int(y0 - 10000*(a))
        cv2.line(img,(x1,y1),(x2,y2),barva,debelina)
        #display_image(img)
    return img
def merge_images(img1,img2):
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    dst = np.zeros((max(h1, h2), w1+w2,3), dtype="uint8")
    dst[:h1, :w1] = img1
    dst[:h2, w1:w1+w2] = img2
    return dst

def najdi_karto(imagename):
    img = cv2.imread(imagename) #, cv2.IMREAD_IGNORE_ORIENTATION | cv2.IMREAD_COLOR)
    height, width, channels = img.shape 
    if height>width:
        img = cv2.resize(img,(400,800), interpolation = cv2.INTER_CUBIC)
    else:
        img = cv2.resize(img,(800,400), interpolation = cv2.INTER_CUBIC)
        rows,cols,d = img.shape
        M = cv2.getRotationMatrix2D((400,400),90,1)
        img = cv2.warpAffine(img,M,(rows,cols))
        
    img_original=img.copy()
        
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray,(21,21),0)
    #display_image(blur)
    
    median = cv2.medianBlur(gray,21)
    median = cv2.GaussianBlur(median,(15,15),0)
    #display_image(median)
    edges2 = cv2.Canny(blur,60,100,apertureSize = 3)
    edges = cv2.Canny(median,60,100,apertureSize = 3)
    #display_image(edges)
    #display_image(edges2)
        
    N=40
    for i in range(200,10,-5):
        # Krajsaj dozino crte dokler jih ne odcitas N
        lines = cv2.HoughLines(edges,2,np.pi/180,i)
        try:
            if len(lines)>N:
                print("Min dolzina crte:" + str(i))
                print("stevilo crt " + str(len(lines)))
                lines=popravi_predznak(lines)
                dobre,slabe = dobre_crte(lines)
                vse_crte=lines
                lines=lines[dobre]
                nav,vod=navpicne(lines)
                if len(nav)>20 and len(vod)>20: #ce najdemo vsaj 10 navpicnih in vodoravnih
                    break
        except TypeError:
            continue

    pts1,robovi=vogali(lines)
    
    img=narisi_crte(img,vse_crte,(0,0,255),1)
    img=narisi_crte(img,lines,(0,255,255),2)
    img=narisi_crte(img,robovi,(255,0,0),2)
    
    for x, y in pts1:
        cv2.circle(img,(x,y),4,(0,255,0),-1)
                
    pts2 = np.float32([[0,0],[100,0],[0,200],[100,200]])
    M = cv2.getPerspectiveTransform(pts1,pts2)
    icon = cv2.warpPerspective(img_original,M,(100,200))
    #display_image(icon)
    #display_image2(img,icon)
    #print("writing: " + os.path.join('./icons/' ,  imagename[:-6] + ".jpg") )
    #cv2.imwrite(os.path.join('./icons/' ,  imagename[:-6] + ".jpg") ,icon)

    # print("writing: " + os.path.join('./icons2/' ,  imagename[:-6] + ".jpg") )
    # cv2.imwrite(os.path.join('./icons2/' ,  imagename[:-6] + ".jpg") ,icon)
    return img,icon


def shrani_4_slike(lokacija,img1,img2,img3,img4):
    img1=merge_images(img1,img2)
    img1=merge_images(img1,img3)
    img1=merge_images(img1,img4)
    cv2.imwrite(lokacija ,img1)

def najdi_robove_1(img):
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    median = cv2.medianBlur(gray,15)
    edges = cv2.Canny(median,150,220,apertureSize = 3)
    return edges

def najdi_robove_nabor(img):
    # b=img_original[:,:,0]
    # b = cv2.medianBlur(b,15)
    # t,b=cv2.threshold(b,0,85,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    # g=img_original[:,:,1]
    # g = cv2.medianBlur(g,15)
    # t,g=cv2.threshold(g,0,85,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    # r=img_original[:,:,2]
    # r = cv2.medianBlur(r,15)    
    # t,r=cv2.threshold(r,0,85,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    # display_image4(b,g,r,b+g+r)

    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    #display_image(gray)
    gray2=gray.copy()
    max_gr=np.max(gray2)
    gray2[gray2<(max_gr/4)]=np.uint8(max_gr*9)
    #display_image2(gray,gray2)
    gray2=cv2.normalize(gray2,None, 0, 255, cv2.NORM_MINMAX)


    #print(np.min(gray),    np.min(gray2))
    
    median = cv2.medianBlur(gray2,15)
    median = cv2.GaussianBlur(median,(25,25),0)
    #ret3,th3 = cv2.threshold(median,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    th3=median.copy()
    
    #th3 = cv2.GaussianBlur(th3,(15,15),0)
    kernel_small = np.ones((3,3),np.uint8)
    kernel = np.ones((5,5),np.uint8)
    #th3 = cv2.morphologyEx(th3, cv2.MORPH_CLOSE, kernel)
    #th3=cv2.dilate(th3,kernel,iterations = 2)
    #th3=r+g+b
    #th3[th3<255]=0
    #th3 = cv2.morphologyEx(th3, cv2.MORPH_GRADIENT, kernel)

    # th3 = cv2.GaussianBlur(th3,(15,15),0)
    # th3 = cv2.GaussianBlur(th3,(15,15),0)
    # th3=cv2.erode(th3,kernel,iterations = 1)
    #
    #th3 = cv2.medianBlur(th3,15)
    #median = cv2.GaussianBlur(median,(21,21),0)
    #ret3,th3 = cv2.threshold(median,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    #th3=cv2.dilate(th3,kernel_small,iterations = 1)
    #ret3,th3 = cv2.threshold(th3,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    #ret3,th3 = cv2.threshold(th3,0,255,cv2.THRESH_TOZERO_INV+cv2.THRESH_OTSU)
    #ret,th3 = cv2.threshold(th3,180,255,cv2.THRESH_BINARY)
    th3=cv2.dilate(th3,kernel,iterations = 1)
    #th3 = cv2.adaptiveThreshold(th3,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
    #        cv2.THRESH_BINARY,11,2)
    th3 = cv2.medianBlur(gray2,9)
    th3[th3<50]=np.uint(230)
    th3=cv2.dilate(th3,kernel,iterations = 1)
    th3 = cv2.medianBlur(th3,9)
    edges = cv2.morphologyEx(th3, cv2.MORPH_GRADIENT, kernel_small)
    #edges=cv2.dilate(edges,kernel_small,iterations = 1)
    #edges=cv2.erode(edges,kernel_small,iterations = 1)

    #edges = cv2.Canny(th3,150,200,apertureSize = 3)
    edges = cv2.Canny(edges,150,220,apertureSize = 3)
    #edges=th3
    return edges
    
def najdi_karto_devel(imagename):
    img = cv2.imread(imagename) #, cv2.IMREAD_IGNORE_ORIENTATION | cv2.IMREAD_COLOR)
    height, width, channels = img.shape 
    if height>width:
        img = cv2.resize(img,(400,800), interpolation = cv2.INTER_CUBIC)
    else:
        img = cv2.resize(img,(800,400), interpolation = cv2.INTER_CUBIC)
        rows,cols,d = img.shape
        M = cv2.getRotationMatrix2D((400,400),90,1)
        img = cv2.warpAffine(img,M,(rows,cols))
        
    img_original=img.copy()

  

    N=10
    for i in range(110,5,-3):
        # Krajsaj dozino crte dokler jih ne odcitas N
        lines = cv2.HoughLines(edges,1,np.pi/360,i)
        try:
            if len(lines)>N:
                lines=popravi_predznak(lines)
                dobre,slabe = dobre_crte(lines)
                vse_crte=lines
                lines=lines[dobre]
                nav,vod=navpicne(lines)
                if len(nav)>4 and len(vod)>4: #ce najdemo vsaj 10 navpicnih in vodoravnih
                    if(np.max(lines[nav,0])-np.min(lines[nav,0]))>(i/2):
                        if(np.max(lines[vod,0])-np.min(lines[vod,0]))>(i/2):
                            break
        except TypeError:
            continue
        if i<=10:
            print("ERROR! Minimum Hough threshold reached!")
            #display_image4(edges,gray2,median,th3)
            return
    #display_image4(edges,gray2,median,th3)
    print("Min dolzina crte: " + str(i) + " Stevilo crt: " + str(len(lines)) + " " + imagename)

    
    if len(lines)<N:
        print("Stevilo crt je premajhno! -> " + str(len(lines)))

    pts1,robovi=vogali(lines)
    
    img=narisi_crte(img,vse_crte,(0,0,255),1)
    img=narisi_crte(img,lines,(0,255,255),2)
    img=narisi_crte(img,robovi,(255,0,0),2)
    
    for x, y in pts1:
        cv2.circle(img,(x,y),4,(0,255,0),-1)
                
    pts2 = np.float32([[0,0],[100,0],[0,200],[100,200]])
    M = cv2.getPerspectiveTransform(pts1,pts2)
    icon = cv2.warpPerspective(img_original,M,(100,200))
    #display_image(icon)
    #display_image2(img,icon)
    #print("writing: " + os.path.join('./icons/' ,  imagename[:-6] + ".jpg") )
    #cv2.imwrite(os.path.join('./icons/' ,  imagename[:-6] + ".jpg") ,icon)

    # print("writing: " + os.path.join('./icons2/' ,  imagename[:-6] + ".jpg") )
    # cv2.imwrite(os.path.join('./icons2/' ,  imagename[:-6] + ".jpg") ,icon)

    #

    th3=cv2.cvtColor(th3,cv2.COLOR_GRAY2RGB)
    edges=cv2.cvtColor(edges,cv2.COLOR_GRAY2RGB)
    shrani_4_slike('./debug_images/' + imagename[:-6] + ".jpg",th3,edges,img,icon)
    return img,icon


# Za Testiranje knjiznice
if __name__ == "__main__":
    directory = './'
    i=0;
    names={}
    for file in os.listdir(directory):
        filename = os.fsdecode(file)
#        filename = "pik_10_1.jpg"
        if filename.endswith("_1.jpg"): # or filename.endswith("_2.jpg"): 
            
            img,karta=najdi_karto_devel(filename)
            #display_image2(img,karta)
            names[i]=filename[:-6]
            i+=1
            #break
            continue
        else:
            continue

        
