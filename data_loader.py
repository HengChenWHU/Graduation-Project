
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf
import argparse
import random
import scipy


class Data():
    def __init__(self,args):
        self.BATCH_SIZE=args.BATCH_SIZE
        self.IMG_W=args.IMG_W
        self.IMG_H=args.IMG_H

        #path
        self.rgb_path = args.rgb_path
        self.spare_depth_path = args.spare_depth_path
        self.denth_depth_path = args.denth_depth_path

        self.test_rgb_path=args.test_rgb_path
        self.test_spare_depth_path=args.test_spare_depth_path
        self.test_denth_depth_path=args.test_denth_depth_path

        self.rgb_list=[]
        self.spare_depth_list=[]
        self.denth_depth_list=[]

        self.test_rgb_list=[]
        self.test_spare_depth_list=[]
        self.test_denth_depth_list=[]
        


        self.all_image_list=[]
        self.mini_batches=[]
        #self.imgrgb=[]
        #self.imgrsd=[]
        #self.imgdd=[]
        self.index=0
        self.w_size=5

        self.get_imagelist()
        self.collect_all_images()
        self.get_mini_batch()

        self.localrgb_path="./data/localrgb/"
        
        self.all_imgs=[]
    

    def get_imagelist(self):
        '''
        read rgb,gt,and sp from KITTI datasets
        '''
        for file in os.listdir(self.rgb_path):
            for file2 in os.listdir(self.rgb_path + file):
                for file3 in os.listdir(self.rgb_path + file+'/'+file2):
                    for file4 in os.listdir(self.rgb_path + file+'/'+file2+'/'+file3):
                        if file4=='image_02' or file4=='image_03':
                            for file5 in os.listdir(self.rgb_path + file+'/'+file2+'/'+file3+'/'+file4):
                                if file5=='data':
                                    for file6 in os.listdir(self.rgb_path + file+'/'+file2+'/'+file3+'/'+file4+'/'+file5):
                                        self.rgb_list.append(self.rgb_path + file+'/'+file2+'/'+file3+'/'+file4+'/'+file5+'/'+file6)

        for file in os.listdir(self.denth_depth_path):
            for file2 in os.listdir(self.denth_depth_path + file):
                for file3 in os.listdir(self.denth_depth_path + file+'/'+file2):
                    for file4 in os.listdir(self.denth_depth_path + file+'/'+file2+'/'+file3):
                        for file5 in os.listdir(self.denth_depth_path + file+'/'+file2+'/'+file3+'/'+file4):
                                self.denth_depth_list.append(self.denth_depth_path + file+'/'+file2+'/'+file3+'/'+file4+'/'+file5)
                                
        for file in os.listdir(self.spare_depth_path):
            for file2 in os.listdir(self.spare_depth_path + file):
                for file3 in os.listdir(self.spare_depth_path + file+'/'+file2):
                    for file4 in os.listdir(self.spare_depth_path + file+'/'+file2+'/'+file3):
                        if file4=='image_02' or file4=='image_03':
                            for file5 in os.listdir(self.spare_depth_path + file+'/'+file2+'/'+file3+'/'+file4):
                                self.spare_depth_list.append(self.spare_depth_path + file+'/'+file2+'/'+file3+'/'+file4+'/'+file5)
        
        for file in os.listdir(self.test_rgb_path):
            for file2 in os.listdir(self.test_rgb_path + file):
                for file3 in os.listdir(self.test_rgb_path + file+'/'+file2):
                    for file4 in os.listdir(self.test_rgb_path + file+'/'+file2+'/'+file3):
                        if file4=='image_02' or file4=='image_03':
                            for file5 in os.listdir(self.test_rgb_path + file+'/'+file2+'/'+file3+'/'+file4):
                                if file5=='data':
                                    for file6 in os.listdir(self.test_rgb_path + file+'/'+file2+'/'+file3+'/'+file4+'/'+file5):
                                        self.test_rgb_list.append(self.test_rgb_path + file+'/'+file2+'/'+file3+'/'+file4+'/'+file5+'/'+file6)
        
        for file in os.listdir(self.test_spare_depth_path):
            for file2 in os.listdir(self.test_spare_depth_path + file):
	            for file3 in os.listdir(self.test_spare_depth_path + file+'/'+file2):
	                for file4 in os.listdir(self.test_spare_depth_path + file+'/'+file2+'/'+file3):
	                        for file5 in os.listdir(self.test_spare_depth_path + file+'/'+file2+'/'+file3+'/'+file4):
	                                    self.test_spare_depth_list.append(self.test_spare_depth_path + file+'/'+file2+'/'+file3+'/'+file4+'/'+file5)
        
        for file in os.listdir(self.test_denth_depth_path):
            for file2 in os.listdir(self.test_denth_depth_path + file):
	            for file3 in os.listdir(self.test_denth_depth_path + file+'/'+file2):
	                for file4 in os.listdir(self.test_denth_depth_path + file+'/'+file2+'/'+file3):
	                    if file4=='image_02' or file4=='image_03':
	                        for file5 in os.listdir(self.test_denth_depth_path + file+'/'+file2+'/'+file3+'/'+file4):
	                                    self.test_denth_depth_list.append(self.test_denth_depth_path + file+'/'+file2+'/'+file3+'/'+file4+'/'+file5)
        print("***************train image list check*********************")
        print("rgb"+str(len(self.rgb_list))+"sp"+str(len(self.spare_depth_list))+"gt"+str(len(self.denth_depth_list)))
        print("***************test image list check*********************")
        print("rgb"+str(len(self.test_rgb_list))+"sp"+str(len(self.test_spare_depth_list))+"gt"+str(len(self.test_denth_depth_list)))
        
        rgb_check = self.rgb_list[9].split('/')
        sp_check = self.spare_depth_list[9].split('/')
        gt_check = self.denth_depth_list[9].split('/')

        assert rgb_check[-1] ==sp_check[-1] and sp_check[-1]==gt_check[-1]


    def collect_all_images(self):
            rgb_list2=np.hstack((self.rgb_list))
            spare_depth_list2=np.hstack((self.spare_depth_list))
            denth_depth_list2=np.hstack((self.denth_depth_list))
            self.all_image_list=[rgb_list2,spare_depth_list2,denth_depth_list2]
            
    def get_mini_batch(self):
        n=len(self.all_image_list[0])#
        rgb=self.all_image_list[0]
        spare_depth=self.all_image_list[1]
        denth_depth=self.all_image_list[2]
        
        batch_num = int(n / self.BATCH_SIZE)
        for k in range(0, batch_num):
            mini_batch_rgb = rgb[k*self.BATCH_SIZE:(k+1)*self.BATCH_SIZE]
            mini_batch_spare_depth = spare_depth[k*self.BATCH_SIZE:(k+1)*self.BATCH_SIZE]
            mini_batch_D_denth_depth = denth_depth[k*self.BATCH_SIZE:(k+1)*self.BATCH_SIZE]
            self.mini_batches.append([mini_batch_rgb, mini_batch_spare_depth, mini_batch_D_denth_depth])
            random.shuffle(self.mini_batches)
        


    def read_batch_image(self):
        imgrgb=[]
        imgrsd=[]
        imgdd=[]

        dirrgb=self.mini_batches[self.index][0]
        dirsd=self.mini_batches[self.index][1]
        dirdd=self.mini_batches[self.index][2]
        for i in range(self.BATCH_SIZE):
            #print("**************check_for_rgb**************")
            #print(dirrgb[i])
            rgb_img=scipy.misc.imread(dirrgb[i])
            
            #print("check**********************")
            #print(rgb_img.shape)
            rgb_img=np.resize(rgb_img,(self.IMG_H,self.IMG_W,3))
            #print("check**222222********************")
            #print(rgb_img.shape)

            #rgb_img=cv2.resize(rgb_img,(self.IMG_W,self.IMG_H))
            #rgb_img=self.data_standard(rgb_img)
            #print(dirrgb[i])
            #print("**************check_for_sd**************")
            #print(dirsd[i])
            sd_img=scipy.misc.imread(dirsd[i])
            sd_img=np.resize(sd_img,(self.IMG_H,self.IMG_W,1))
            #print(sd_img)
            #sd_img=cv2.resize(sd_img,(self.IMG_W,self.IMG_H))
            #print("*********************check for once***********************")
            #print(sd_img.shape)
            #sd_img=sd_img[:,:,0]
            #sd_img=np.expand_dims(sd_img, axis=2)
            #sd_img=self.data_standard(sd_img)
            #print(dirsd[i])
            #print("**************check_for_dd**************")
            #print(dirdd[i])
            dd_img=scipy.misc.imread(dirdd[i])
            dd_img=np.resize(dd_img,(self.IMG_H,self.IMG_W,1))
            #dd_img=dd_img+1
            
            
            #dd_img=cv2.resize(dd_img,(1279,415))
            #dd_img=cv2.resize(dd_img,(1242,375))
            #dd_img=dd_img[:,:,0]
            #print("************************************")
            #print(dd_img.sum()/1242/375)
            #dd_img=np.expand_dims(dd_img, axis=2)
            #dd_img=self.data_standard(dd_img)
            #print(dirdd[i])

            imgrgb.append(rgb_img)
            imgrsd.append(sd_img)
            imgdd.append(dd_img)
        return imgrgb,imgrsd,imgdd

    def get_data(self):
    	n=len(self.mini_batches)
    	#name=self.mini_batches[self.index][0]
    	#newname=[]
    	#n2=len(name)
    	#print(n2)
    	#strlen=len(name[0])
    	#print(strlen)
    	#name=str(name)
    	#for j in range(n2):
    	#	newname.append(name[-40-strlen*j:-36-strlen*j]+name[-10-strlen*j:-2-strlen*j])
        #	print(name[-40-strlen*j:-36-strlen*j]+name[-10-strlen*j:-2-strlen*j])
    	#newname=name[-40:-36]+name[-10:-2]
    	
    	imgrgb,imgrsd,imgdd=self.read_batch_image()

    	#lc_imgrgb=self.local_sampe_dataset(imgrgb)
    	#lc_imgrsd=self.local_sampe_dataset(imgrsd)
    	#lc_imgdd=self.local_sampe_dataset(imgdd)

    	imgrgb=np.array(imgrgb)
    	imgrsd=np.array(imgrsd)
    	imgdd=np.array(imgdd)

       #this part can manke sure all the data can be used
    	print("feed batch:",self.index)
    	self.index=self.index+1
    	if self.index>n-1:
    		self.index=0
    	return imgrgb,imgrsd,imgdd

    def data_standard(self,data):
    	data_mean=np.mean(data)
    	data_std=np.std(data)
    	data=(data-data_mean)/data_std
    	return data
    
    def data_argument(self,data):
        a1 = tf.image.random_brightness(data,max_delta=30)
        a2 = tf.image.random_contrast(a1,lower=0.2,upper=1.8)
        a3= tf.image.random_hue(a2,max_delta=0.3)
        return a3


    def read_test_image(self):
        imgrgb=[]
        imgrsd=[]
        imgdd=[]

        n=len(self.test_rgb_list)
        #self.test_spare_depth_list
        #self.test_denth_depth_list

        for i in range(n):
            #rgb_img=cv2.imread(self.test_rgb_list[i])
            rgb_img=cv2.imread(self.test_rgb_list[i])
            rgb_img=np.resize(rgb_img,(self.IMG_H,self.IMG_W,3))
            #rgb_img=cv2.resize(rgb_img,(self.IMG_W,self.IMG_H))
            
            #plt.imshow(rgb_img[:,:,0])
            #plt.axis("off")
            #plt.show()
            #print(dirrgb[i])
            #sd_img=cv2.imread(self.test_spare_depth_list[i])
            sd_img=scipy.misc.imread(self.test_spare_depth_list[i])
            sd_img=np.resize(sd_img,(self.IMG_H,self.IMG_W,1))
            #sd_img=cv2.resize(sd_img,(self.IMG_W,self.IMG_H))
            #sd_img=sd_img[:,:,0]
            #sd_img=np.expand_dims(sd_img, axis=2)
            
            #plt.imshow(sd_img[:,:,0])
            #plt.axis("off")
            #plt.show()
            #print(dirsd[i])
            #dd_img=cv2.imread(self.test_denth_depth_list[i])
            dd_img=scipy.misc.imread(self.test_denth_depth_list[i])
            dd_img=np.resize(dd_img,(self.IMG_H,self.IMG_W,1))
            #dd_img=dd_img+1
            
            self.all_imgs.append(dd_img)
            #dd_img=cv2.resize(dd_img,(1279,415))
            #dd_img=cv2.resize(dd_img,(1242,375))
            #dd_img=dd_img[:,:,0]
            
            #dd_img=np.expand_dims(dd_img, axis=2)
            
            
            
            #plt.imshow(dd_img[:,:,0])
            #plt.axis("off")
            #plt.show()
            #print(dirdd[i])

            imgrgb.append(rgb_img)
            imgrsd.append(sd_img)
            imgdd.append(dd_img)

        return imgrgb,imgrsd,imgdd





    def show_image(self,imgrgb,imgrsd,imgdd):
        for i in range(self.BATCH_SIZE):
            plt.imshow(imgrgb[i][:,:,:])
            plt.axis("off")
            plt.show()
            

            plt.imshow(imgrsd[i][:,:,0])
            plt.axis("off")
            plt.show()

            plt.imshow(imgdd[i][:,:,0])
            plt.axis("off")
            plt.show()
            
    



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='depth completion')
    parser.add_argument('--rgb_path',dest='rgb_path',default="./data/train/rgb/")
    parser.add_argument('--spare_depth_path',dest='spare_depth_path',default="./data/train/sp/")
    parser.add_argument('--denth_depth_path',dest='denth_depth_path',default="./data/train/gt/")
    parser.add_argument('--test_rgb_path',dest='test_rgb_path',default="./data/test/rgb/")
    parser.add_argument('--test_spare_depth_path',dest='test_spare_depth_path',default="./data/test/sp/")
    parser.add_argument('--test_denth_depth_path',dest='test_denth_depth_path',default="./data/test/gt/")


    parser.add_argument('--BATCH_SIZE',dest='BATCH_SIZE',default=2)
    parser.add_argument('--IMG_H',dest='IMG_H',default=375)
    parser.add_argument('--IMG_W',dest='IMG_W',default=1242)
    args = parser.parse_args()
    dataset = Data(args)
    #imgrgb,imgrsd,imgdd=dataset.get_data()
    #dataset.show_image(imgrgb,imgrsd,imgdd)
    ###########################################
    imgrgb,imgrsd,imgdd=dataset.read_test_image()
        
    '''
    a=set()
    b=list()
    print("check the shape")
    for j in range(375):
        for p in range(1242):
            #a.add(imgdd[0][j,p,0])
            b.append(imgdd[0][j,p,0])
            print(imgdd[0][j,p,0])
    #alist=list(a)
    #alist.sort()
    #n=len(alist)
    
    mean1=mean(b)
    
    mean1=mean(b)
    
    len1=len(b)
    #print(n)
    
    all_things=[0 for i in range(len1)]
    
    
    
    ec_number=[]
    for i in b:
        all_things[i]=all_things[i]+1
        
    print((all_things))
    '''
    
  
        
    

    
                    
                    
        
       



