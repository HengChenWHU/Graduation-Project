# -*- coding: utf-8 -*-
"""
Created on Thu Mar  5 21:04:05 2020

@author: Think
"""

import os
import numpy as np
import random
import scipy
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import argparse

class Data_vkitii():
    def __init__(self,args):
        self.BATCH_SIZE=args.BATCH_SIZE
        self.IMG_W=args.IMG_W
        self.IMG_H=args.IMG_H

        #path
        self.rgb_path = args.vkitti_rgb_path
        self.spare_depth_path = args.vkitti_spare_depth_path
        self.denth_depth_path = args.vkitti_denth_depth_path

        self.test_rgb_path=args.vkitti_test_rgb_path
        self.test_spare_depth_path=args.vkitti_test_spare_depth_path
        self.test_denth_depth_path=args.vkitti_test_denth_depth_path

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

        
    def get_imagelist(self):
                '''
                read rgb,gt,and sp from KITTI datasets
                '''
                for file in os.listdir(self.rgb_path):
                    for file2 in os.listdir(self.rgb_path + file):
                        for file3 in os.listdir(self.rgb_path + file+'/'+file2):
                            self.rgb_list.append(self.rgb_path + file+'/'+file2+'/'+file3)
                            
                for file in os.listdir(self.spare_depth_path):
                    for file2 in os.listdir(self.spare_depth_path + file):
                        for file3 in os.listdir(self.spare_depth_path + file+'/'+file2):
                            self.spare_depth_list.append(self.spare_depth_path + file+'/'+file2+'/'+file3)
                
                for file in os.listdir(self.denth_depth_path):
                    for file2 in os.listdir(self.denth_depth_path + file):
                        for file3 in os.listdir(self.denth_depth_path + file+'/'+file2):
                            self.denth_depth_list.append(self.denth_depth_path + file+'/'+file2+'/'+file3)
                
                for file in os.listdir(self.test_rgb_path):
                    for file2 in os.listdir(self.test_rgb_path + file):
                        for file3 in os.listdir(self.test_rgb_path + file+'/'+file2):
                            self.test_rgb_list.append(self.test_rgb_path + file+'/'+file2+'/'+file3)
                            
                for file in os.listdir(self.test_spare_depth_path):
                    for file2 in os.listdir(self.test_spare_depth_path + file):
                        for file3 in os.listdir(self.test_spare_depth_path + file+'/'+file2):
                            self.test_spare_depth_list.append(self.test_spare_depth_path + file+'/'+file2+'/'+file3)
                
                for file in os.listdir(self.test_denth_depth_path):
                    for file2 in os.listdir(self.test_denth_depth_path + file):
                        for file3 in os.listdir(self.test_denth_depth_path + file+'/'+file2):
                            self.test_denth_depth_list.append(self.test_denth_depth_path + file+'/'+file2+'/'+file3)
                
                print("***************train image list check*********************")
                print("rgb"+str(len(self.rgb_list))+"sp"+str(len(self.spare_depth_list))+"gt"+str(len(self.denth_depth_list)))
                print("***************test image list check*********************")
                print("rgb"+str(len(self.test_rgb_list))+"sp"+str(len(self.test_spare_depth_list))+"gt"+str(len(self.test_denth_depth_list)))
                print(self.test_rgb_list[0])

    def collect_all_images(self):
                    rgb_list2=np.hstack((self.rgb_list))
                    spare_list2=np.hstack((self.spare_depth_list))
                    dense_depth_list2=np.hstack((self.denth_depth_list))
                    self.all_image_list=[rgb_list2,spare_list2,dense_depth_list2]
                     
    def get_mini_batch(self):
                n=len(self.all_image_list[0])#
                rgb=self.all_image_list[0]
                spare=self.all_image_list[1]
                denth_depth=self.all_image_list[2]
                
                batch_num = int(n /self.BATCH_SIZE)
        
                for k in range(0, batch_num):
                    mini_batch_rgb = rgb[k*self.BATCH_SIZE:(k+1)*self.BATCH_SIZE]
                    mini_batch_D_denth_depth = denth_depth[k*self.BATCH_SIZE:(k+1)*self.BATCH_SIZE]
                    mini_batch_D_spare_depth = spare[k*self.BATCH_SIZE:(k+1)*self.BATCH_SIZE]
                    self.mini_batches.append([mini_batch_rgb, mini_batch_D_spare_depth,mini_batch_D_denth_depth])
                random.shuffle(self.mini_batches)
                
    def get_data(self):
    	n=len(self.mini_batches)
    	imgrgb,imgrsd,imgdd=self.read_batch_image()
    	imgrgb=np.array(imgrgb)
    	imgrsd=np.array(imgrsd)
    	imgdd=np.array(imgdd)

       #this part can manke sure all the data can be used
    	print("feed batch:",self.index)
    	self.index=self.index+1
    	if self.index>n-1:
    		self.index=0
    	return imgrgb,imgrsd,imgdd
                    
    def read_batch_image(self):
                imgrgb=[]
                imgsp=[]
                imgdd=[]
        
                dirrgb=self.mini_batches[self.index][0]
                dirsp=self.mini_batches[self.index][1]
                dirdd=self.mini_batches[self.index][2]
                for i in range(self.BATCH_SIZE):
        
                    rgb_img=scipy.misc.imread(dirrgb[i])
                    #rgb_img=np.expand_dims(rgb_img,3)
                    #rgb_img=np.resize(rgb_img,(self.IMG_H,self.IMG_W,3))
                    sp_img=scipy.misc.imread(dirsp[i])
                    sp_img=np.expand_dims(sp_img,2)
                    #sp_img=np.resize(sp_img,(self.IMG_H,self.IMG_W,1))
                    dd_img=scipy.misc.imread(dirdd[i])
                    dd_img=np.expand_dims(dd_img,2)
                    #dd_img=np.resize(dd_img,(376,1244,1))
                    #dd_img=np.resize(dd_img,(376,1244,1))
                    imgrgb.append(rgb_img)
                    imgsp.append(sp_img)
                    imgdd.append(dd_img)
                return imgrgb,imgsp,imgdd
            
    def read_test_image(self):
        imgrgb=[]
        imgrsd=[]
        imgdd=[]

        n=len(self.test_rgb_list)
        print("cccccccc")
        print(n)

        for i in range(n):
            #print(self.test_rgb_path[i])
            rgb_img=scipy.misc.imread(self.test_rgb_list[i])
            #rgb_img=np.resize(rgb_img,(self.IMG_H,self.IMG_W,3))

            sd_img=scipy.misc.imread(self.test_spare_depth_list[i])
            sd_img=np.expand_dims(sd_img,2)
            #sd_img=np.resize(sd_img,(self.IMG_H,self.IMG_W,1))

            dd_img=scipy.misc.imread(self.test_denth_depth_list[i])
            dd_img=np.expand_dims(dd_img,2)
            #dd_img=np.resize(dd_img,(376,1244,1))

            imgrgb.append(rgb_img)
            imgrsd.append(sd_img)
            imgdd.append(dd_img)

        return imgrgb,imgrsd,imgdd
    
    def show_image(self,imgrgb,imgrsp,imgdd):
                for i in range(len(imgrgb)):
                    plt.imshow(imgrgb[0][:,:,:])
                    plt.axis("off")
                    plt.show()
                    plt.imshow(imgrsp[0][:,:,0])
                    plt.axis("off")
                    plt.show()
                    plt.imshow(imgdd[0][:,:,0])
                    plt.axis("off")
                    plt.show()
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='depth completion')

    parser.add_argument('--vkitti_rgb_path',dest='vkitti_rgb_path',default="./data/vkitti_1.3.1_rgb/")
    parser.add_argument('--vkitti_spare_depth_path',dest='vkitti_spare_depth_path',default="./data/vkitti_1.3.1_spare_depth/")
    parser.add_argument('--vkitti_denth_depth_path',dest='vkitti_denth_depth_path',default="./data/vkitti_1.3.1_depthgt/")
    
    parser.add_argument('--vkitti_test_rgb_path',dest='vkitti_test_rgb_path',default="./data/vkitti_test_rgb/")
    parser.add_argument('--vkitti_test_spare_depth_path _test',dest='vkitti_test_spare_depth_path',default="./data/vkitti_test_sp/")
    parser.add_argument('--vkitti_test_denth_depth_path',dest='vkitti_test_denth_depth_path',default="./data/vkitti_test_dp/")

    parser.add_argument('--BATCH_SIZE',dest='BATCH_SIZE',default=2)
    parser.add_argument('--IMG_H',dest='IMG_H',default=375)
    parser.add_argument('--IMG_W',dest='IMG_W',default=1242)
    args = parser.parse_args()
    dataset = Data_vkitii(args)
    #imgrgb,imgrsd,imgdd=dataset.get_data()
    #dataset.show_image(imgrgb,imgrsd,imgdd)
    
    imgrgb2,imgrsd2,imgdd2=dataset.read_test_image()
    dataset.show_image(imgrgb2,imgrsd2,imgdd2)
    

            