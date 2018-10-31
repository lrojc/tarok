import cv2
import numpy as np
import sys,os
import matplotlib.pyplot as plt
import pickle

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
def shrani_3_slike(lokacija,img1,img2,img3):
    img1=merge_images(img1,img2)
    img1=merge_images(img1,img3)
    cv2.imwrite(lokacija ,img1)
def shrani_2_slike(lokacija,img1,img2):
    img1=merge_images(img1,img2)
    cv2.imwrite(lokacija ,img1)

def najdi_robove_1(img):
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    median = cv2.medianBlur(gray,15)
    edges = cv2.Canny(median,150,200,apertureSize = 3)
    #display_image3(gray,median,edges) 
    return edges

def najdi_robove_2(img):
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    median = cv2.medianBlur(gray,15)
    thresh = cv2.adaptiveThreshold(median,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,11,2)
    thresh = cv2.medianBlur(thresh,5)
    kernel = np.ones((5,5),np.uint8)
    thresh=cv2.dilate(thresh,kernel,iterations = 1)
    thresh=cv2.erode(thresh,kernel,iterations = 1)
    im2, contours,hierarchy=cv2.findContours(thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE )
    cmax=0
    area_coeff=0.5
    for c in contours:

        tmp=cv2.contourArea(c)
        if tmp>1000:
            img_tmp=img.copy()
            epsilon=0.003*cv2.arcLength(c,True)
            approx=cv2.approxPolyDP(c,epsilon,True)
            cv2.drawContours(img_tmp, c, -1, (0,0,255),3)
            cv2.drawContours(img_tmp, [approx], -1, (255,0,0),2)

            x,y,w,h=cv2.boundingRect(c)
            rect=cv2.minAreaRect(c)
            box=cv2.boxPoints(rect)
            box=np.int0(box)

            
            cv2.drawContours(img_tmp, [box], 0, (0,255,255),2) # min rect box
            hull=cv2.convexHull(c)
            cv2.drawContours(img_tmp, [hull], 0, (255,255,0),2) # min rect box
            cv2.rectangle(img_tmp,(x,y),(x+w,y+h),(0,255,0),2)
            area_1=cv2.contourArea(hull)
            area_rect=cv2.contourArea(box)
            area_coeff=(area_rect-area_1)/area_rect

            #display_image(img_tmp)
    
            if (tmp>cmax) and area_coeff<0.1: # mora biti tudi vecji od 0
                cmax=tmp
                cnt=c
            else:
                print(area_1,area_rect,area_coeff)
                #display_image(img_tmp)
            
    cv2.drawContours(img, contours, -1, (0,0,255))
    gray[:]=np.uint8(0)
    #epsilon=0.1*cv2.arcLength(cnt,True)
    #approx=cv2.approxPolyDP(cnt,epsilon,True)
    cv2.drawContours(gray, cnt, -1, 255)
    
    x,y,w,h=cv2.boundingRect(cnt)
    rect=cv2.minAreaRect(cnt)
    box=cv2.boxPoints(rect)
    box=np.int0(box)
    cv2.drawContours(img, [box], 0, (255,0,0))
    cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)

    return img,box,gray,thresh

def najdi_karto_devel(directory,imagename):
    
    img = cv2.imread(directory + imagename) #, cv2.IMREAD_IGNORE_ORIENTATION | cv2.IMREAD_COLOR)
    height, width, channels = img.shape 
    if height>width:
        img = cv2.resize(img,(400,800), interpolation = cv2.INTER_CUBIC)
    else:
        img = cv2.resize(img,(800,400), interpolation = cv2.INTER_CUBIC)
        rows,cols,d = img.shape
        M = cv2.getRotationMatrix2D((400,400),90,1)
        img = cv2.warpAffine(img,M,(rows,cols))
        
    img_original=img.copy()
    img,pts,edges,thresh=najdi_robove_2(img)
    pts1 = np.float32(pts)
    
    for x, y in pts1:
        cv2.circle(img,(x,y),4,(0,255,0),-1)
    
    #print(imagename[:-6])
    #print(pts1)
    print('./debug_images/' + imagename[:-4] + ".jpg")
    nova_visina=200
    nova_sirina=100
    if((pts1[0,0]-pts1[1,0])**2+(pts1[0,1]-pts1[1,1])**2)> \
      ((pts1[2,0]-pts1[1,0])**2+(pts1[2,1]-pts1[1,1])**2): 
       pts2 = np.float32([[0,nova_visina],[0,0],[nova_sirina,0],[nova_sirina,nova_visina]])
    else:
        pts2 = np.float32([[nova_sirina,nova_visina],[0,nova_visina],[0,0],[nova_sirina,0]])
    M = cv2.getPerspectiveTransform(pts1,pts2)
    icon = cv2.warpPerspective(img_original,M,(nova_sirina,nova_visina))
    edges=cv2.cvtColor(edges,cv2.COLOR_GRAY2RGB)
    thresh=cv2.cvtColor(thresh,cv2.COLOR_GRAY2RGB)
    shrani_4_slike('./debug_images/' + imagename[:-4] + ".jpg",thresh,edges,img,icon)
    return img,icon

def najdi_karto_devel2(directory,imagename):
    img = cv2.imread(directory + imagename)

    height, width, channels = img.shape 
    if height>width:
        img = cv2.resize(img,(400,800), interpolation = cv2.INTER_CUBIC)
    else:
        img = cv2.resize(img,(800,400), interpolation = cv2.INTER_CUBIC)
        rows,cols,d = img.shape
        M = cv2.getRotationMatrix2D((400,400),90,1)
        img = cv2.warpAffine(img,M,(rows,cols))
        
    img_original=img.copy()
    
    slika,tocke,edges=najdi_robove_2(img)
    slika=slika.copy()
    
    N=4
    for i in range(110,5,-10):
        # Krajsaj dozino crte dokler jih ne odcitas N
        lines = cv2.HoughLines(edges,2,np.pi/180,i)
        try:
            if len(lines)>N:
                lines=popravi_predznak(lines)
                dobre,slabe = dobre_crte(lines)
                vse_crte=lines
                lines=lines[dobre]
                nav,vod=navpicne(lines)
                if len(nav)>2 and len(vod)>2: #ce najdemo vsaj 10 navpicnih in vodoravnih
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
    print("Min dolzina crte: " + str(i) + " Stevilo crt: " + str(len(lines)) + " " + './debug_images1/' + imagename[:-6] + ".jpg")

    
    if len(lines)<N:
        print("Stevilo crt je premajhno! -> " + str(len(lines)))

    pts1,robovi=vogali(lines)
    #pts1=najdi_robove_2(img)
    
    img=narisi_crte(img,vse_crte,(0,0,255),1)
    img=narisi_crte(img,lines,(0,255,255),2)
    img=narisi_crte(img,robovi,(255,0,0),2)
    
    for x, y in pts1:
        cv2.circle(img,(x,y),4,(0,255,0),-1)

    nova_visina=200
    nova_sirina=100
    pts2 = np.float32([[0,0],[nova_sirina,0],[0,nova_visina],[nova_sirina,nova_visina]])
    M = cv2.getPerspectiveTransform(pts1,pts2)
    icon = cv2.warpPerspective(img_original,M,(nova_sirina,nova_visina))

    #th3=cv2.cvtColor(th3,cv2.COLOR_GRAY2RGB)
    edges=cv2.cvtColor(edges,cv2.COLOR_GRAY2RGB)
    #print('./debug_images1/' + imagename[:-6] + ".jpg")
    shrani_4_slike('./debug_images_1/' + imagename[:-6] + ".jpg",slika,edges,img,icon)
    return img,icon

def naredi_masko(directory,ends,dest):
    #directory = './set_luka_1/'
    i=0;
    names={}
    for file in os.listdir(directory):
        filename = os.fsdecode(file)
        if filename.endswith(ends): # or filename.endswith("_2.jpg"): 
            img,karta=najdi_karto_devel(directory , filename)
            if i==0:
                karta1=karta.copy()
            else:
                karta1=merge_images(karta1,karta)
            names[i]=filename[:-4]
            i+=1
            #break
            continue
        else:
            continue
    cv2.imwrite(dest ,karta1)
    return img,names
def naredi_ikone(directory,ends,dst_dir):
    #directory = './set_luka_1/'
    i=0;
    names={}
    for file in os.listdir(directory):
        filename = os.fsdecode(file)
        if filename.endswith(ends): # or filename.endswith("_2.jpg"): 
            img,karta=najdi_karto_devel(directory , filename)
            cv2.imwrite(dst_dir + filename ,karta)
            i+=1
            continue
        else:
            continue

    return img,names
def empty_run():
    directory = './set_luka_1/'
    i=0;
    names={}
    for file in os.listdir(directory):
        filename = os.fsdecode(file)
#        filename = "pik_10_1.jpg"
        if filename.endswith("_1.jpg") or filename.endswith("_2.jpg"): 
            print(filename)
            img,karta=najdi_karto_devel(directory , filename)
            #display_image2(img,karta)
            names[i]=filename[:-6]
            i+=1
            #break
            continue
        else:
            continue

def template_matching(directory,ends,maska,maska_imena):
    barva=['blue','green','red']
    #img = cv2.imread('maska.jpg',0)
    img=maska
    img2 = img.copy()
    # All the 6 methods for comparison in a list
    methods = ['cv2.TM_CCOEFF', 'cv2.TM_CCOEFF_NORMED', 'cv2.TM_CCORR',
           'cv2.TM_CCORR_NORMED', 'cv2.TM_SQDIFF', 'cv2.TM_SQDIFF_NORMED']
    #for meth in methods:
    meth=methods[0]
    for file in os.listdir(directory):
        filename = os.fsdecode(file)
        if filename.endswith(ends): # or filename.endswith("_2.jpg"):
            template3 = cv2.imread(directory + filename)
            res3=np.zeros((3,5301))
            #display_image3(template3[:,:,0],template3[:,:,1],template3[:,:,2])
            method = eval(meth)
            for i in [0,1,2]:
                template=template3[:,:,i]
                w, h = template.shape[::-1]
                img = img2.copy()
                
                # Apply template Matching
                res = cv2.matchTemplate(img[:,:,i],template,method)
                template2=cv2.flip(template,-1) 
                res2 = cv2.matchTemplate(img[:,:,i],template2,method)
                #print(res[0].shape)

                #if (np.min(res2[0])<np.min(res[0])):
                if (np.max(res2[0])>np.max(res[0])):
                    res=res2
                    print("obrnjen")
                    
                min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
                if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]: # If the method is TM_SQDIFF or TM_SQDIFF_NORMED, take minimum
                    top_left = min_loc
                else:
                    top_left = max_loc
                bottom_right = (top_left[0] + w, top_left[1] + h)
                cv2.rectangle(img,top_left, bottom_right, 255, 2)

                indeks=np.int32(round(top_left[0]/w))
                det_ime=maska_imena[indeks]
                
                if det_ime not in filename:
                    print(filename + "\t" + barva[i] + "\tdetektiran kot:\t" + maska_imena[indeks])

                    plt.subplot(2,1,1),plt.plot(res[0]), plt.title(filename + " " + barva[i]) #, plt.xticks([]), plt.yticks([])
                    plt.subplot(2,2,3),plt.imshow(img),plt.axis([top_left[0],top_left[0]+w,top_left[1],top_left[1]+h])
                    plt.subplot(2,2,4)
                    if np.sum(res-res2)==0:
                        plt.imshow(template2)
                    else:
                        plt.imshow(template)
                    
                    #plt.axis([top_left[0],top_left[0]+w,top_left[1],top_left[1]+h])
                    plt.title('Detected Point'), plt.xticks([]), plt.yticks([])
                    #plt.suptitle(meth)
                    plt.show()
                else:
                    print(filename + "\t" + barva[i] + "\tOK")

                res3[i,:]=res
                
            res=np.reshape(res3, (3,-1))
            res=np.sum(res,0)
            res=np.array([res])
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
            if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]: # If the method is TM_SQDIFF or TM_SQDIFF_NORMED, take minimum
                top_left = min_loc
            else:
                top_left = max_loc
            bottom_right = (top_left[0] + w, top_left[1] + h)
            cv2.rectangle(img,top_left, bottom_right, 255, 2)

            indeks=np.int32(round(top_left[0]/w))
            det_ime=maska_imena[indeks]
            
            if det_ime not in filename:
                print(filename + "\tvsota\tdetektiran kot:\t" + maska_imena[indeks])
                #print(min_val, max_val, min_loc, max_loc)
                plt.subplot(2,1,1),plt.plot(res[0]), plt.title(filename + " vsota") #, plt.xticks([]), plt.yticks([])
                plt.subplot(2,2,1),plt.imshow(img),plt.axis([top_left[0],top_left[0]+w,top_left[1],top_left[1]+h])
                plt.subplot(2,2,2)
                if np.sum(res-res2)==0:
                    plt.imshow(template2)
                else:
                    plt.imshow(template)
                plt.title('Detected Point'), plt.xticks([]), plt.yticks([])
                #plt.suptitle(meth)
                plt.show()
            else:
                print(filename + "\t vsota \tOK")
def template_matching_simple(directory,ends,maska,maska_imena):
    barva=['blue','green','red']
    #img = cv2.imread('maska.jpg',0)
    #img=cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,11,2)
    img=maska
    
    img2 = img.copy()
    # All the 6 methods for comparison in a list
    methods = ['cv2.TM_CCOEFF', 'cv2.TM_CCOEFF_NORMED', 'cv2.TM_CCORR',
           'cv2.TM_CCORR_NORMED', 'cv2.TM_SQDIFF', 'cv2.TM_SQDIFF_NORMED']
    #for meth in methods:
    meth=methods[5]
    for file in os.listdir(directory):
        filename = os.fsdecode(file)
        if filename.endswith(ends): # or filename.endswith("_2.jpg"):
            template3 = cv2.imread(directory + filename)
            res3=np.zeros((3,5301))
            #display_image3(template3[:,:,0],template3[:,:,1],template3[:,:,2])
            method = eval(meth)
            for i in [0,1,2]:

                img=cv2.adaptiveThreshold(img2[:,:,i],255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,11,2)
                img = cv2.medianBlur(img,7)
                template=template3[:,:,i]
                template=cv2.adaptiveThreshold(template,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,11,2)
                template = cv2.medianBlur(template,7)
                w, h = template.shape[::-1]

                
                # Apply template Matching
                res = cv2.matchTemplate(img,template,method)
                template2=cv2.flip(template,-1) 
                res2 = cv2.matchTemplate(img,template2,method)
                #print(res[0].shape)

                if (np.min(res2[0])<np.min(res[0])):
                #if (np.max(res2[0])>np.max(res[0])):
                    res=res2
                    print("obrnjen")
                    
                min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
                if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]: # If the method is TM_SQDIFF or TM_SQDIFF_NORMED, take minimum
                    top_left = min_loc
                else:
                    top_left = max_loc
                bottom_right = (top_left[0] + w, top_left[1] + h)
                cv2.rectangle(img,top_left, bottom_right, 255, 2)

                indeks=np.int32(round(top_left[0]/w))
                det_ime=maska_imena[indeks]
                
                if det_ime not in filename:
                    print(filename + "\t" + barva[i] + "\tdetektiran kot:\t" + maska_imena[indeks])

                    plt.subplot(2,1,1),plt.plot(res[0]), plt.title(filename + " " + barva[i]) #, plt.xticks([]), plt.yticks([])
                    plt.subplot(2,2,3),plt.imshow(img),plt.axis([top_left[0],top_left[0]+w,top_left[1],top_left[1]+h])
                    plt.subplot(2,2,4)
                    if np.sum(res-res2)==0:
                        plt.imshow(template2)
                    else:
                        plt.imshow(template)
                    
                    #plt.axis([top_left[0],top_left[0]+w,top_left[1],top_left[1]+h])
                    plt.title('Detected Point'), plt.xticks([]), plt.yticks([])
                    #plt.suptitle(meth)
                    plt.show()
                else:
                    print(filename + "\t" + barva[i] + "\tOK")

                res3[i,:]=res
                
            res=np.reshape(res3, (3,-1))
            res=np.sum(res,0)
            res=np.array([res])
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
            if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]: # If the method is TM_SQDIFF or TM_SQDIFF_NORMED, take minimum
                top_left = min_loc
            else:
                top_left = max_loc
            bottom_right = (top_left[0] + w, top_left[1] + h)
            cv2.rectangle(img,top_left, bottom_right, 255, 2)

            indeks=np.int32(round(top_left[0]/w))
            det_ime=maska_imena[indeks]
            
            if det_ime not in filename:
                print(filename + "\tvsota\tdetektiran kot:\t" + maska_imena[indeks])
                #print(min_val, max_val, min_loc, max_loc)
                plt.subplot(2,1,1),plt.plot(res[0]), plt.title(filename + " vsota") #, plt.xticks([]), plt.yticks([])
                plt.subplot(2,2,1),plt.imshow(img2),plt.axis([top_left[0],top_left[0]+w,top_left[1],top_left[1]+h])
                plt.subplot(2,2,2)
                if np.sum(res-res2)==0:
                    plt.imshow(template2)
                else:
                    plt.imshow(template)
                plt.title('Detected Point'), plt.xticks([]), plt.yticks([])
                #plt.suptitle(meth)
                plt.show()
            else:
                print(filename + "\t vsota \tOK")

def template_matching_siva(directory,ends,maska,maska_imena):
    img = cv2.imread('maska.jpg',0)
    img_mask=cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,17,2)
    max_val=np.max(img[img_mask>0])
    img[img_mask>0]=np.uint8(max_val)
    img2 = img.copy()
    methods = ['cv2.TM_CCOEFF', 'cv2.TM_CCOEFF_NORMED', 'cv2.TM_CCORR',
           'cv2.TM_CCORR_NORMED', 'cv2.TM_SQDIFF', 'cv2.TM_SQDIFF_NORMED']
    meth=methods[4]
    for file in os.listdir(directory):
        filename = os.fsdecode(file)
        if filename.endswith(ends):
            template = cv2.imread(directory + filename,0)
            template_mask=cv2.adaptiveThreshold(template,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,17,2)
            max_val=np.max(template[template_mask>0])
            svetlost_karte=(template_mask>0).sum()/template_mask.size
            template[template_mask>0]=np.uint8(max_val)
            template = cv2.medianBlur(template,3)
            method = eval(meth)
            w, h = template.shape[::-1]
                
            res = cv2.matchTemplate(img,template,method)
            template2=cv2.flip(template,-1) 
            res2 = cv2.matchTemplate(img,template2,method)
            
            if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
                if (np.min(res2[0])<np.min(res[0])):
                    res=res2
            else:
                if (np.max(res2[0])>np.max(res[0])):
                    res=res2

            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
            if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
                top_left = min_loc
            else:
                top_left = max_loc
            bottom_right = (top_left[0] + w, top_left[1] + h)
            cv2.rectangle(img,top_left, bottom_right, 255, 2)
            indeks=np.int32(round(top_left[0]/w))
            det_ime=maska_imena[indeks]

            detekcija="\tOK"
            if det_ime[:-2] not in filename:
                detekcija= "\tdetektiran kot:\t" + det_ime
                
                plt.subplot(2,1,1),plt.plot(res[0])
                plt.subplot(2,2,3), _plot_image_correct_color_(img)
                plt.xticks(), plt.yticks()
                plt.axis([top_left[0],top_left[0]+w,top_left[1]+h,top_left[1]])
                plt.title(det_ime + " siva") #, plt.xticks([]), plt.yticks([])
                plt.subplot(2,2,4)
                plt.title(filename + " siva") #, plt.xticks([]), plt.yticks([])
                if np.sum(res-res2)==0:
                    _plot_image_correct_color_(template2)
                else:
                    _plot_image_correct_color_(template)
#                plt.title('Detected Point'),
                plt.xticks([]), plt.yticks([])
                plt.subplots_adjust(left=None, bottom=0.07, right=None, top=0.97, wspace=None, hspace=0.35)
                plt.show()

            if "tarok" not in det_ime:
                if svetlost_karte>0.7:
                    print(filename.ljust(20)  + "\tSvetlost: " + str(svetlost_karte) +  "\tplatelc" + detekcija)
                else:
                    print(filename.ljust(20)  + "\tSvetlost: " + str(svetlost_karte) +  "\tbarva" + detekcija)
            else:
                print(filename.ljust(20)  + "\tSvetlost: " + str(svetlost_karte) +  "\ttarok" + detekcija)

def template_matching_siva_z_znaki(directory,ends,maska,maska_imena):
    img = cv2.imread('maska.jpg',0)
    img_mask=cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,17,2)
    max_val=np.max(img[img_mask>0])
    img[img_mask>0]=np.uint8(max_val)
    img2 = img.copy()
    methods = ['cv2.TM_CCOEFF', 'cv2.TM_CCOEFF_NORMED', 'cv2.TM_CCORR',
           'cv2.TM_CCORR_NORMED', 'cv2.TM_SQDIFF', 'cv2.TM_SQDIFF_NORMED']
    meth=methods[4]
    for file in os.listdir(directory):
        filename = os.fsdecode(file)
        if filename.endswith(ends):
            template = cv2.imread(directory + filename,0)
            template_mask=cv2.adaptiveThreshold(template,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,17,2)
            max_val=np.max(template[template_mask>0])
            svetlost_karte=(template_mask>0).sum()/template_mask.size
            template[template_mask>0]=np.uint8(max_val)
            template = cv2.medianBlur(template,3)
            method = eval(meth)
            w, h = template.shape[::-1]
                
            res = cv2.matchTemplate(img,template,method)
            template2=cv2.flip(template,-1) 
            res2 = cv2.matchTemplate(img,template2,method)
            
            if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
                if (np.min(res2[0])<np.min(res[0])):
                    res=res2
            else:
                if (np.max(res2[0])>np.max(res[0])):
                    res=res2

            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
            if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
                top_left = min_loc
            else:
                top_left = max_loc
            bottom_right = (top_left[0] + w, top_left[1] + h)
            cv2.rectangle(img,top_left, bottom_right, 255, 2)
            indeks=np.int32(round(top_left[0]/w))
            det_ime=maska_imena[indeks]

            detekcija="\tOK"
            if det_ime[:-2] not in filename:
                detekcija= "\tdetektiran kot:\t" + det_ime
                
                plt.subplot(2,1,1),plt.plot(res[0])
                plt.subplot(2,2,3), _plot_image_correct_color_(img)
                plt.xticks(), plt.yticks()
                plt.axis([top_left[0],top_left[0]+w,top_left[1]+h,top_left[1]])
                plt.title(det_ime + " siva") #, plt.xticks([]), plt.yticks([])
                plt.subplot(2,2,4)
                plt.title(filename + " siva") #, plt.xticks([]), plt.yticks([])
                if np.sum(res-res2)==0:
                    _plot_image_correct_color_(template2)
                else:
                    _plot_image_correct_color_(template)
#                plt.title('Detected Point'),
                plt.xticks([]), plt.yticks([])
                plt.subplots_adjust(left=None, bottom=0.07, right=None, top=0.97, wspace=None, hspace=0.35)
                plt.show()

            if "tarok" not in det_ime:
                if svetlost_karte>0.7:
                    print(filename.ljust(20)  + "\tSvetlost: " + str(svetlost_karte) +  "\tplatelc" + detekcija)
                else:
                    print(filename.ljust(20)  + "\tSvetlost: " + str(svetlost_karte) +  "\tbarva" + detekcija)
            else:
                print(filename.ljust(20)  + "\tSvetlost: " + str(svetlost_karte) +  "\ttarok" + detekcija)

def pripravi_znake():
    ikone=['src_1_1.jpg','kara_1_1.jpg','pik_9_1.jpg','kriz_9_1.jpg']
    for i in [0,1,2,3]:
        img=cv2.imread("./ikone/" + ikone[i])
        h, w, channels = img.shape
        dw=0.13*w
        dh=0.085*h
        #pts2 = np.float32([[0,height],[0,0],[width,0],[width,height]])
        pts1 = np.float32([[w/2-dw,h/2+dh],
                           [w/2-dw,h/2-dh],
                           [w/2+dw,h/2-dh],
                           [w/2+dw,h/2+dh]])
        pts2 = np.float32([[0,2*dh],
                           [0,0],
                           [2*dw,0],
                           [2*dw,2*dh]])
        M = cv2.getPerspectiveTransform(pts1,pts2)
        icon = cv2.warpPerspective(img,M,(np.int32(2*dw),np.int32(2*dh)))
        cv2.imwrite('./' + ikone[i][:-8] + '.jpg',icon)
        icon=cv2.flip(icon,-1)
        cv2.imwrite('./' + ikone[i][:-8] + '2.jpg',icon)
        display_image(icon)
        
def pripravi_ikone_in_masko():
    directory = './set_luka_1/'
    ends_with="_1.jpg"
    mask_name="maska.jpg"

    img,imena=naredi_masko(directory,ends_with,mask_name)
    f = open('store.pckl', 'wb')
    pickle.dump(imena, f)
    f.close()

    ikone_dir='./ikone/'
    naredi_ikone(directory,ends_with,ikone_dir)
    ends_with="_2.jpg"
    naredi_ikone(directory,ends_with,ikone_dir)
    
def testiraj_masko():
    ikone_dir='./ikone/'
    ends_with="_2.jpg"
    mask_name="maska.jpg"

    f = open('store.pckl', 'rb')
    imena = pickle.load(f)
    f.close()

    mask_img = cv2.imread(mask_name)
    template_matching_siva(ikone_dir,ends_with,mask_img,imena)

if __name__ == "__main__":
    #pripravi_ikone_in_masko()
    #testiraj_masko()

