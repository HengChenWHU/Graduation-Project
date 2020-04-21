import numpy as np
import os
import cv2
import tensorflow as tf

import argparse
from data_loader import Data
from vkitti_reader import Data_vkitii
from model import Model
import matplotlib.pyplot as plt
import scipy.io as sio
import scipy
import numpy as np
import time
import datetime
import math
class Main():
    def __init__(self):
    		self.trainingmode = tf.constant(True,dtype=tf.bool)
    		self.testingmode = tf.constant(False,dtype=tf.bool)
    		self.learning_rate=0.1#0.03#for vkitti
            
    		#self.learning_rate=0.03
          #0.1 for kitti
    
    		self.parser = argparse.ArgumentParser(description='depth completion')
    		self.parser.add_argument('--rgb_path',dest='rgb_path',default="./data/train/rgb/")
    		self.parser.add_argument('--spare_depth_path',dest='spare_depth_path',default="./data/train/sp/")
    		self.parser.add_argument('--denth_depth_path',dest='denth_depth_path',default="./data/train/gt/")
    
    		self.parser.add_argument('--test_rgb_path',dest='test_rgb_path',default="./data/test/rgb/")
    		self.parser.add_argument('--test_spare_depth_path',dest='test_spare_depth_path',default="./data/test/sp/")
    		self.parser.add_argument('--test_denth_depth_path',dest='test_denth_depth_path',default="./data/test/gt/")
          
    		self.parser.add_argument('--vkitti_rgb_path',dest='vkitti_rgb_path',default="./data/vkitti_1.3.1_rgb/")
    		self.parser.add_argument('--vkitti_spare_depth_path',dest='vkitti_spare_depth_path',default="./data/vkitti_1.3.1_spare_depth/")
    		self.parser.add_argument('--vkitti_denth_depth_path',dest='vkitti_denth_depth_path',default="./data/vkitti_1.3.1_depthgt/")
    
    		self.parser.add_argument('--vkitti_test_rgb_path',dest='vkitti_test_rgb_path',default="./data/vkitti_test_rgb/")
    		self.parser.add_argument('--vkitti_test_spare_depth_path _test',dest='vkitti_test_spare_depth_path',default="./data/vkitti_test_sp/")
    		self.parser.add_argument('--vkitti_test_denth_depth_path',dest='vkitti_test_denth_depth_path',default="./data/vkitti_test_dp/")
    
    
    		self.parser.add_argument('--BATCH_SIZE',dest='BATCH_SIZE',default=1)
    		self.parser.add_argument('--learning_rate',dest='lr',default=0.1)
    		self.parser.add_argument('--epoch',dest='epoch',default=100000)#1500 for kitti
    		self.parser.add_argument('--cube_size',dest='cube_size',default=5)
    		self.parser.add_argument('--load_model',dest='load_model',default=False)
    		self.parser.add_argument('--IMG_H',dest='IMG_H',default=375)
    		self.parser.add_argument('--IMG_W',dest='IMG_W',default=1242)
    		self.args = self.parser.parse_args()
    
    		self.test_result=dict()
    		self.train_result=dict()
    		self.test_relative=dict()
    		self.test_average_loss= dict() 
    		self.train_record_step=50
    		self.test_record_step=999
    
    def train(self):
        x = tf.placeholder(tf.float32,[self.args.BATCH_SIZE, self.args.IMG_H, self.args.IMG_W, 3],name='x-input')
        sp=tf.placeholder(tf.float32,[self.args.BATCH_SIZE, self.args.IMG_H, self.args.IMG_W, 1],name='sp-input')
        #y_ = tf.placeholder(tf.float32,[self.args.BATCH_SIZE, 415, 1279, 1],name='y-input')
        y_ = tf.placeholder(tf.float32,[self.args.BATCH_SIZE, self.args.IMG_H, self.args.IMG_W, 1],name='y-input')
        dataset = Data(self.args)
        
        #x=dataset.data_argument(x)#data argument
        #x = tf.image.per_image_standardization(x)
        #x = tf.image.per_image_standardization(x)
        #sp = tf.image.per_image_standardization(sp)
        #y_ = tf.image.per_image_standardization(y_)
        
        #x=dataset.data_argument(x)
        
        #sp = tf.image.per_image_standardization(sp)
        #y_== tf.image.per_image_standardization(y_)
        mask=self.get_mask(y_)
        mask_num=tf.reduce_sum(mask)
        mask_num=mask_num
        
        print("mask_sum")
        print(mask_num)
        
        mask_add=tf.multiply(mask,-1.0)
        mask_add=tf.add(mask_add,1.0)
        #mask_add=tf.multiply(mask_add,0.1)
        
        
        #y_2=tf.multiply(y_,mask)
        #sp2=tf.multiply(sp,mask)
        #x2=tf.multiply(x,mask)
        x2=x
        sp2=sp
        y_2=y_
        
        #sp2=tf.add(sp2,mask_add)
        #y_2=tf.add(y_2,mask_add)
        #x2=tf.add(x2,mask_add)

        
        net=Model(self.args)
        #pre_depth,rgb_depth,sp_depth,igf=net.network3(x2,sp2,net.trainingmode)
        pre_depth,rgb_depth,sp_depth,igs,my_see=net.network(x2,sp2,net.trainingmode)
        
        
        pre_depth2=tf.multiply(pre_depth,mask)
        rgb_depth2=tf.multiply(rgb_depth,mask)
        sp_depth2=tf.multiply(sp_depth,mask)
        
        
        #pre_depth2=pre_depth
        #rgb_depth2=rgb_depth
        #sp_depth2=sp_depth
        
        
        #pre_depth2=pre_depth#tf.add(pre_depth2,mask_add)#############
        #rgb_depth2=rgb_depth#tf.add(rgb_depth2,mask_add)###########
        #sp_depth2=sp_depth#tf.add(sp_depth2,mask_add)###############

        
        
        train_loss=net.loss(pre_depth2,rgb_depth2,sp_depth2,y_2)#/mask_num*self.args.IMG_H*self.args.IMG_W
        
        MAE=net.MAE_loss(pre_depth2,y_2)#/mask_num*self.args.IMG_H*self.args.IMG_W
        #MAE=tf.divide(MAE,mask_num)
        #MAE=(MAE)/self.args.BATCH_SIZE/self.args.IMG_H/self.args.IMG_W
        
        iMAE=net.iMAE_loss(pre_depth2,y_2)#/mask_num*self.args.IMG_H*self.args.IMG_W
        #iMAE=tf.divide(iMAE,mask_num)
        #iMAE=(iMAE)/self.args.BATCH_SIZE/self.args.IMG_H/self.args.IMG_W
        
        iRMSE=net.iRMSE_loss(pre_depth2,y_2)#/mask_num*self.args.IMG_H*self.args.IMG_W
        #iRMSEs=tf.divide(iRMSEs,mask_num)
        #iRMSEs=(iRMSEs)/self.args.BATCH_SIZE/self.args.IMG_H/self.args.IMG_W
        
        RMSE=net.RMSE_loss(pre_depth2,y_2)#/mask_num*self.args.IMG_H*self.args.IMG_W
        #RMSE=tf.divide(RMSE,mask_num)
        #RMSE=(RMSE)/self.args.BATCH_SIZE/self.args.IMG_H/self.args.IMG_W
        
        iRMSE_rgbb=net.iRMSE_loss(rgb_depth2,y_2)#/mask_num*self.args.IMG_H*self.args.IMG_W
        
        erro_map=tf.abs(tf.subtract(pre_depth2,y_2))
        erro_rgb=tf.abs(tf.subtract(rgb_depth2,y_2))
        erro_sp=tf.abs(tf.subtract(sp_depth2,y_2))
        
        
        learning_rate = tf.train.exponential_decay(self.learning_rate,global_step=net.global_step,decay_steps=1000, decay_rate=0.2)
        train_op,regularization_loss=net.optimize(learning_rate)
        
        config = tf.ConfigProto()
        config.gpu_options.per_process_gpu_memory_fraction = 0.9
        config.gpu_options.allow_growth = True

        with tf.Session(config=config) as sess:
            tf.global_variables_initializer().run()
            for i in range(self.args.epoch):
                imgrgb,imgrsd,imgdd=dataset.get_data()
                imgdd=imgdd+1

                #step,youtput,loss_value,op,lr,mask_,erro_map_,rl,rgbp,spp,iMAE_,MAE_,iRMSEs_,RMSE_= sess.run([net.global_step,pre_depth2,train_loss,train_op,learning_rate,mask,erro_map,regularization_loss,rgb_depth2,sp_depth2,iMAE,MAE,iRMSEs,RMSE],feed_dict={x: imgrgb, sp:imgrsd, y_: imgdd})
                #step,youtput,loss_value,op,lr,erro_map_,rl,rgbp,spp,iMAE_,MAE_,iRMSEs_,RMSE_,sp_depth2_,mask_,y_2_,sp2_,mask_add_,mask_num_,igf4= sess.run([net.global_step,pre_depth2,train_loss,train_op,learning_rate,erro_map,regularization_loss,rgb_depth2,sp_depth2,iMAE,MAE,iRMSEs,RMSE,sp_depth2,mask,y_2,sp2,mask_add,mask_num,igf[4]],feed_dict={x: imgrgb, sp:imgrsd, y_: imgdd})
                
                step,loss_value,train_op_,lr,rl,mask_num_=sess.run([net.global_step,train_loss,train_op,learning_rate,regularization_loss,mask_num],feed_dict={x: imgrgb, sp:imgrsd, y_: imgdd})
                youtput,sp2_,y_2_=sess.run([sp_depth2,sp2,y_2],feed_dict={x: imgrgb, sp:imgrsd, y_: imgdd})
                rgb_d2,sp_d2=sess.run([rgb_depth2,sp_depth2],feed_dict={x: imgrgb, sp:imgrsd, y_: imgdd})
                #igf4,igf3,igf2=sess.run([igf[4],igf[3],igf[2]],feed_dict={x: imgrgb, sp:imgrsd, y_: imgdd})
                erro_map_,erro_rgb_,erro_sp_,mask_=sess.run([erro_map,erro_rgb,erro_sp,mask],feed_dict={x: imgrgb, sp:imgrsd, y_: imgdd})
                MAE_,iMAE_,RMSE_,iRMSE_=sess.run([MAE,iMAE,RMSE,iRMSE],feed_dict={x: imgrgb, sp:imgrsd, y_: imgdd})
                
                iRMSE_rgbb_=sess.run([iRMSE_rgbb],feed_dict={x: imgrgb, sp:imgrsd, y_: imgdd})
                
                igs_=sess.run(igs,feed_dict={x: imgrgb, sp:imgrsd, y_: imgdd})
                
                my_see_=sess.run(my_see,feed_dict={x: imgrgb, sp:imgrsd, y_: imgdd})
                
                pre_depth2_=sess.run(pre_depth2,feed_dict={x: imgrgb, sp:imgrsd, y_: imgdd})
                
                sh_my_see_=np.array(my_see_)

                sh_my_see_=sh_my_see_.reshape((40,64))
                sh_my_see_=sh_my_see_*1000
                #print(sh_my_see_)
                
                #imae=0
                #for i in range(pre_depth2_.shape[1]):
                #    for j in range(pre_depth2_.shape[2]):
                #        if pre_depth2_[0,i,j,0]!=0 and y_2_[0,i,j,0]!=0:
                #            a=pre_depth2_[0,i,j,0]
                #            b=y_2_[0,i,j,0]
                #            a=a/1000000.0
                #            b=b/1000000.0
                #            c=a-b
                ##            d=a*b
                #            imae=imae+abs(c/d)
                ##imae=imae/self.args.IMG_H/self.args.IMG_W
                #print("my test nnnnnnnnnnnnnnnnnnnnnn imae",imae)
                
                
               # #rgb_SE_=sess.run(rgb_SE,feed_dict={x: imgrgb, sp:imgrsd, y_: imgdd})
                
                #print(rgb_SE_[0])
                print("Training step: %d, loss: %g  ,ime：%g,mae：%g,irmse：%g,rmse：%g,regularization_loss: %g ,learning_rate :%.8f" % (step, loss_value,iMAE_,MAE_,iRMSE_,RMSE_,rl,lr))
                #print(pre_depth2_)
                if i%50==0:
                    self.train_result.update({i: MAE_}) 
                    #print(mask_)
                if i%600==0 and i!=0:
                    self.test(sess)
                    l_test_rloss=list(self.test_result.values())
                    average_a = np.mean(l_test_rloss)
                    self.test_average_loss.update({i/self.test_record_step: average_a}) 
                '''
                if i%self.train_record_step==0 and i!=0:
                    plt.figure()
                    
                    plt.subplot(431)
                    plt.axis('off') 
                    plt.title('rgb',fontsize='medium',fontweight='bold')
                    plt.imshow(imgrgb[0,:,:,:])
                    
                    
                    plt.subplot(432) 
                    plt.axis('off') 
                    plt.title('spare depth map',fontsize='medium',fontweight='bold')
                    plt.imshow(sp2_[0,:,:,0])
                    
                    plt.subplot(433)
                    plt.axis('off') 
                    plt.title('dense depth map',fontsize='medium',fontweight='bold')
                    plt.imshow(y_2_[0,:,:,0])
                    
                    plt.subplot(434)
                    plt.axis('off') 
                    plt.title('predicted depth map',fontsize='medium',fontweight='bold')
                    plt.imshow(youtput[0,:,:,0])
                    
                    plt.subplot(435)
                    plt.axis('off') 
                    plt.title('rgb depth',fontsize='medium',fontweight='bold')
                    plt.imshow(rgb_d2[0,:,:,0])
                    
                    plt.subplot(436)
                    plt.axis('off') 
                    plt.title('sp map',fontsize='medium',fontweight='bold')
                    plt.imshow(sp_d2[0,:,:,0])
                    
                    plt.subplot(437)
                    plt.axis('off') 
                    plt.title('erro predict',fontsize='medium',fontweight='bold')
                    plt.imshow(erro_map_[0,:,:,0],cmap='hot')
                    
                    plt.subplot(438)
                    plt.axis('off') 
                    plt.title('erro rgb',fontsize='medium',fontweight='bold')
                    plt.imshow(erro_rgb_[0,:,:,0],cmap='hot')
                    
                    plt.subplot(439)
                    plt.axis('off') 
                    plt.title('erro sp',fontsize='medium',fontweight='bold')
                    plt.imshow(erro_sp_[0,:,:,0],cmap='hot')
                    
                    
                    
                    
                    plt.savefig("./train_output/"+"step"+str(i)+"loss"+str(loss_value)+".png") 
                    plt.close() 
                    
                    #sio.savemat("./train_output/imgdd"+str(i)+".mat", {'imgdd':y_2_[0,:,:,0]})
                    #sio.savemat("./train_output/predict"+str(i)+".mat", {'predict':youtput[0,:,:,0]})
                    #sio.savemat("./train_output/mask.mat", {'mask':mask_[0,:,:]})
                    plt.figure()
                    plt.subplot(2,3,1)
                    plt.axis('off') 
                    plt.title('guided image filter1',fontsize='medium',fontweight='bold')
                    plt.imshow(igs_[0][0,:,:,0],cmap="hsv")
                    
                    
                    plt.subplot(2,3,2)
                    plt.axis('off') 
                    plt.title('guided image filter2',fontsize='medium',fontweight='bold')
                    plt.imshow(igs_[1][0,:,:,0],cmap="hsv")
                    
                    plt.subplot(2,3,3)
                    plt.axis('off') 
                    plt.title('guided image filter3',fontsize='medium',fontweight='bold')
                    plt.imshow(igs_[2][0,:,:,0],cmap="hsv")
                    
                    plt.subplot(2,3,4)
                    plt.axis('off') 
                    plt.title('guided image filter4',fontsize='medium',fontweight='bold')
                    plt.imshow(igs_[3][0,:,:,0],cmap="hsv")
                    
                    plt.subplot(2,3,5)
                    plt.axis('off') 
                    plt.title('guided image filter5',fontsize='medium',fontweight='bold')
                    plt.imshow(igs_[4][0,:,:,0],cmap="hsv")
                    
                    plt.savefig("./train_output/"+"step"+str(i)+"igf.png") 
                    plt.close() 
                    
                    
                    #plt.figure()
                    #plt.axis('off') 
                    #plt.title('Squeeze and Excitation',fontsize='medium',fontweight='bold')
                    #plt.imshow(sh_my_see_,cmap="tab20c")
                    #plt.colorbar()
                    #plt.savefig("./train_output/"+"step"+str(i)+"seshow.png") 
                    #plt。close()
                '''
                
                    
                    
                    
                #if i%999==0 and i!=0:
                #    self.record()

            index = list(self.train_result.keys())
            value=list(self.train_result.values())
            plt.figure(3)
            #plt.axis('off') 
            plt.title('train loss',fontsize='medium',fontweight='bold')
            plt.plot(index,value)
            plt.savefig("./train_output/train_loss.png")
            plt.close()
            #self.test_result.clear()

            
    def test(self,sess):
        x = tf.placeholder(tf.float32,[1, self.args.IMG_H, self.args.IMG_W, 3],name='x-input')
        sp=tf.placeholder(tf.float32,[1, self.args.IMG_H, self.args.IMG_W, 1],name='sp-input')#+
        #y_ = tf.placeholder(tf.float32,[1, 415, 1279, 1],name='y-input')
        y_ = tf.placeholder(tf.float32,[1, self.args.IMG_H, self.args.IMG_W, 1],name='y-input')
        dataset = Data(self.args)
        
        mask=self.get_mask(y_)
        mask_num=tf.reduce_sum(mask)
        
        mask=self.get_mask(y_)
        mask_add=tf.multiply(mask,-1.0)
        mask_add=tf.add(mask_add,1.0)
        #mask_add=tf.multiply(mask_add,0.1)
        
        y_2=y_#tf.multiply(y_,mask)
        sp2=sp#tf.multiply(sp,mask)
        x2=x
        #x2=tf.multiply(x,mask)
        
        #sp2=tf.add(sp2,mask_add)
        #y_2=tf.add(y_2,mask_add)
        #x2=tf.add(x2,mask_add)

        
        net=Model(self.args)
        pre_depth,rgb_depth,sp_depth,igs,SE=net.network(x2,sp2,net.testingmode)
        
        
        pre_depth2=tf.multiply(pre_depth,mask)

        rgb_depth2=tf.multiply(rgb_depth,mask)
        sp_depth2=tf.multiply(sp_depth,mask)
        
        #pre_depth2=tf.add(pre_depth2,mask_add)
        #rgb_depth2=tf.add(rgb_depth2,mask_add)
        #sp_depth2=tf.add(sp_depth2,mask_add)

        test_loss=net.test_loss(pre_depth2,y_2)#/mask_num*self.args.IMG_H*self.args.IMG_W
        
        erro_map=tf.abs(tf.subtract(pre_depth2,y_2))
        erro_map_rgb=tf.abs(tf.subtract(rgb_depth2,y_2))
        erro_map_sp=tf.abs(tf.subtract(sp_depth2,y_2))
        #relative_erro_map=tf.div(erro_map,y_)
        #relative_erro_map=tf.reduce_mean(relative_erro_map)
        
        
        
        MAE=net.MAE_loss(pre_depth2,y_2)#/mask_num*self.args.IMG_H*self.args.IMG_W
        rgbMAE=net.MAE_loss(rgb_depth2,y_2)#/mask_num*self.args.IMG_H*self.args.IMG_W
        spMAE=net.MAE_loss(sp_depth2,y_2)#/mask_num*self.args.IMG_H*self.args.IMG_W
        #MAE=(MAE)/self.args.BATCH_SIZE/self.args.IMG_H/self.args.IMG_W
        
        iMAE=net.iMAE_loss(pre_depth2,y_2)#/mask_num*self.args.IMG_H*self.args.IMG_W
        rgbiMAE=net.iMAE_loss(rgb_depth2,y_2)#/mask_num*self.args.IMG_H*self.args.IMG_W
        spiMAE=net.iMAE_loss(sp_depth2,y_2)#/mask_num*self.args.IMG_H*self.args.IMG_W
        #iMAE=(iMAE)/self.args.BATCH_SIZE/self.args.IMG_H/self.args.IMG_W
        
        iRMSEs=net.iRMSE_loss(pre_depth2,y_2)#/mask_num*self.args.IMG_H*self.args.IMG_W
        rgbiRMSEs=net.iRMSE_loss(rgb_depth2,y_2)#/mask_num*self.args.IMG_H*self.args.IMG_W
        spiRMSEs=net.iRMSE_loss(sp_depth2,y_2)#/mask_num*self.args.IMG_H*self.args.IMG_W
        #iRMSEs=(iRMSEs)/self.args.BATCH_SIZE/self.args.IMG_H/self.args.IMG_W
        
        RMSE=net.RMSE_loss(pre_depth2,y_2)#/mask_num*self.args.IMG_H*self.args.IMG_W
        rgbRMSE=net.RMSE_loss(rgb_depth2,y_2)#/mask_num*self.args.IMG_H*self.args.IMG_W
        spRMSE=net.RMSE_loss(sp_depth2,y_2)#/mask_num*self.args.IMG_H*self.args.IMG_W
        #RMSE=(RMSE)/self.args.BATCH_SIZE/self.args.IMG_H/self.args.IMG_W
        MAE_list=[]
        iMAE_list=[]
        RMSE_list=[]
        iRMSEs_list=[]
        
        rgbMAE_list=[]
        rgbiMAE_list=[]
        rgbRMSE_list=[]
        rgbiRMSEs_list=[]
        
        spMAE_list=[]
        spiMAE_list=[]
        spRMSE_list=[]
        spiRMSEs_list=[]
        imgrgb,imgrsd,imgdd=dataset.read_test_image()
        for i in range(len(dataset.test_rgb_list)):
            loss_value,youtput,erro_map_,mask_ = sess.run([test_loss,pre_depth2,erro_map,mask],feed_dict={x: np.expand_dims(imgrgb[i],0), sp:np.expand_dims(imgrsd[i],0),y_: np.expand_dims(imgdd[i],0)})
            MAE_,iMAE_,RMSE_,iRMSE_=sess.run([MAE,iMAE,RMSE,iRMSEs],feed_dict={x: np.expand_dims(imgrgb[i],0), sp:np.expand_dims(imgrsd[i],0),y_: np.expand_dims(imgdd[i],0)})
            
            rgbMAE_,rgbiMAE_,rgbRMSE_,rgbiRMSEs_=sess.run([rgbMAE,rgbiMAE,rgbRMSE,rgbiRMSEs],feed_dict={x: np.expand_dims(imgrgb[i],0), sp:np.expand_dims(imgrsd[i],0),y_: np.expand_dims(imgdd[i],0)})
            
            spMAE_,spiMAE_,spRMSE_,spiRMSEs_=sess.run([spMAE,spiMAE,spRMSE,spiRMSEs],feed_dict={x: np.expand_dims(imgrgb[i],0), sp:np.expand_dims(imgrsd[i],0),y_: np.expand_dims(imgdd[i],0)})
            
            rgb_depth2_,sp_depth2_=sess.run([rgb_depth2,sp_depth2],feed_dict={x: np.expand_dims(imgrgb[i],0), sp:np.expand_dims(imgrsd[i],0),y_: np.expand_dims(imgdd[i],0)})
            
            erro_map_rgb_,erro_map_sp_,sp2_,y_2_=sess.run([erro_map_rgb,erro_map_sp,sp2,y_2],feed_dict={x: np.expand_dims(imgrgb[i],0), sp:np.expand_dims(imgrsd[i],0),y_: np.expand_dims(imgdd[i],0)})
            
            self.test_result.update({i: loss_value}) 
            self.test_relative.update({i: loss_value}) 
            
            imae=0
            imargbe=0
            imaesp=0
            iRMSE=0
            iRMSErgb=0
            iRMSEsp=0
            for i in range(youtput.shape[1]):
                for j in range(youtput.shape[2]):
                        if youtput[0,i,j,0]!=0 and y_2_[0,i,j,0]!=0:
                            a=youtput[0,i,j,0]
                            b=y_2_[0,i,j,0]
                            a=a/1000000.0
                            b=b/1000000.0
                            c=a-b
                            d=a*b
                            imae=imae+abs(c/d)
                        if rgb_depth2_[0,i,j,0]!=0 and y_2_[0,i,j,0]!=0:
                            a1=rgb_depth2_[0,i,j,0]
                            b1=y_2_[0,i,j,0]
                            a1=a1/1000000.0
                            b1=b1/1000000.0
                            c1=a1-b1
                            d1=a1*b1
                            imargbe=imargbe+abs(c1/d1)
                        if sp_depth2_[0,i,j,0]!=0 and y_2_[0,i,j,0]!=0:
                            a2=sp_depth2_[0,i,j,0]
                            b2=y_2_[0,i,j,0]
                            a2=a2/1000000.0
                            b2=b2/1000000.0
                            c2=a2-b2
                            d2=a2*b2
                            imaesp=imaesp+abs(c2/d2)
                        if youtput[0,i,j,0]!=0 and y_2_[0,i,j,0]!=0:
                            a3=youtput[0,i,j,0]
                            b3=y_2_[0,i,j,0]
                            a3=a3/1000000.0
                            b3=b3/1000000.0
                            c3=a3-b3
                            d3=a3*b3
                            iRMSE=iRMSE+(c3/d3)*(c3/d3)
                        if rgb_depth2_[0,i,j,0]!=0 and y_2_[0,i,j,0]!=0:
                            a4=youtput[0,i,j,0]
                            b4=y_2_[0,i,j,0]
                            a4=a4/1000000.0
                            b4=b4/1000000.0
                            c4=a4-b4
                            d4=a4*b4
                            iRMSErgb=iRMSErgb+(c4/d4)*(c4/d4)
                        if sp_depth2_[0,i,j,0]!=0 and y_2_[0,i,j,0]!=0:
                            a5=sp_depth2_[0,i,j,0]
                            b5=y_2_[0,i,j,0]
                            a5=a5/1000000.0
                            b5=b5/1000000.0
                            c5=a5-b5
                            d5=a5*b5
                            iRMSEsp=iRMSEsp+(c5/d5)*(c5/d5)
                            
            imae=imae/self.args.IMG_H/self.args.IMG_W
            imargbe=imargbe/self.args.IMG_H/self.args.IMG_W
            imaesp=imaesp/self.args.IMG_H/self.args.IMG_W
            iRMSE=iRMSE/self.args.IMG_H/self.args.IMG_W
            iRMSE=np.sqrt(iRMSE)
            iRMSErgb=iRMSErgb/self.args.IMG_H/self.args.IMG_W
            iRMSErgb=np.sqrt(iRMSErgb)
            iRMSEsp=iRMSEsp/self.args.IMG_H/self.args.IMG_W
            iRMSEsp=np.sqrt(iRMSEsp)
   
            print('i:%g,MAE_: %g,RMSE_: %g,iMAE_: %g,iRMSE_: %g' % (i,MAE_,RMSE_,imae,iRMSE))
            
        
            if not math.isinf(iMAE_) and not math.isinf(iRMSE_) and not math.isinf(rgbiMAE_) and not math.isinf(rgbiRMSEs_) and not math.isinf(spiMAE_) and not math.isinf(spiRMSEs_):
                MAE_list.append(MAE_)
                iMAE_list.append(imae)
                RMSE_list.append(RMSE_)
                iRMSEs_list.append(iRMSE)
                
                rgbMAE_list.append(rgbMAE_)
                rgbiMAE_list.append(imargbe)
                rgbRMSE_list.append(rgbRMSE_)
                rgbiRMSEs_list.append(iRMSErgb)
                
                spMAE_list.append(spMAE_)
                spiMAE_list.append(imaesp)
                spRMSE_list.append(spRMSE_)
                spiRMSEs_list.append(iRMSEsp)
            if not os.path.exists('./test_output/test_result'+str(i)): 
                os.mkdir('./test_output/test_result'+str(i))
            '''    
            plt.figure(0)
            
            #plt.subplot(321)
            plt.axis('off') 
            #plt.title('rgb',fontsize='medium',fontweight='bold')
            plt.imshow(imgrgb[i][:,:,:])
            plt.savefig("./test_output/test_result"+str(i)+"/rgb.png") 
            plt.close()
            
            plt.figure(1)
            #plt.subplot(322)
            plt.axis('off') 
            #plt.title('spare depth map',fontsize='medium',fontweight='bold')
            plt.imshow(sp2_[0,:,:,0])
            plt.savefig("./test_output/test_result"+str(i)+"/spare.png") 
            plt.close()
            
            plt.figure(2)
            #plt.subplot(323)
            plt.axis('off') 
            #plt.title('dense depth map',fontsize='medium',fontweight='bold')
            plt.imshow(y_2_[0,:,:,0])
            plt.savefig("./test_output/test_result"+str(i)+"/dense.png")
            plt.close()
            
            
            plt.figure(3)
            plt.axis('off') 
            #plt.title('predicted depth map',fontsize='medium',fontweight='bold')
            plt.imshow(youtput[0,:,:,0])
            plt.savefig("./test_output/test_result"+str(i)+"/predict.png")
            plt.close()
            
            plt.figure(4)
            #plt.subplot(325)
            plt.axis('off') 
            #plt.title('erro map',fontsize='medium',fontweight='bold')
            plt.imshow(erro_map_[0,:,:,0],cmap="hot")
            plt.savefig("./test_output/test_result"+str(i)+"/erro_map.png")
            plt.close()
            
            plt.figure(41)
            #plt.subplot(325)
            plt.axis('off') 
            #plt.title('erro map',fontsize='medium',fontweight='bold')
            plt.imshow(erro_map_rgb_[0,:,:,0],cmap="hot")
            plt.savefig("./test_output/test_result"+str(i)+"/erro_map_rgb_.png")
            plt.close()
            
            plt.figure(42)
            #plt.subplot(325)
            plt.axis('off') 
            #plt.title('erro map',fontsize='medium',fontweight='bold')
            plt.imshow(erro_map_sp_[0,:,:,0],cmap="hot")
            plt.savefig("./test_output/test_result"+str(i)+"/erro_map_sp_.png")
            plt.close()
            
            
            plt.figure(5)
            #plt.subplot(326)
            plt.axis('off') 
            #plt.title('mask map',fontsize='medium',fontweight='bold')
            plt.imshow(mask_[0,:,:,0],cmap="hot")
            plt.savefig("./test_output/test_result"+str(i)+"/mask.png")
            plt.close()
            
            plt.figure(6)
            #plt.subplot(326)
            plt.axis('off') 
            #plt.title('mask map',fontsize='medium',fontweight='bold')
            plt.imshow(rgb_depth2_[0,:,:,0])
            plt.savefig("./test_output/test_result"+str(i)+"/rgb_predict.png")
            plt.close()
            
            plt.figure(7)
            #plt.subplot(326)
            plt.axis('off') 
            #plt.title('mask map',fontsize='medium',fontweight='bold')
            plt.imshow(sp_depth2_[0,:,:,0])
            plt.savefig("./test_output/test_result"+str(i)+"/sp_predict.png")
            plt.close()
            
        
            #sio.savemat("./test_output/mat/"+str(i)+'pridict.mat', {'pridict':youtput[0,:,:,0]})
            
            #sio.savemat("./test_output/mat/"+str(i)+'gt.mat', {'gt':imgdd[i][:,:,0]})
            '''
            
        sio.savemat("./test_output/MAS.mat", {'MAS':MAE_list})
        sio.savemat("./test_output/iMAS.mat", {'iMAS':iMAE_list})
        sio.savemat("./test_output/RMSE.mat", {'RMSE':RMSE_list})
        sio.savemat("./test_output/iRMSE.mat", {'iRMSE':iRMSEs_list})
        
        sio.savemat("./test_output/rgbMAS.mat", {'rgbMAS':rgbMAE_list})
        sio.savemat("./test_output/rgbiMAS.mat", {'rgbiMAS':rgbiMAE_list})
        sio.savemat("./test_output/rgbRMSE.mat", {'rgbRMSE':rgbRMSE_list})
        sio.savemat("./test_output/rgbiRMSE.mat", {'rgbiRMSE':rgbiRMSEs_list})
        
        sio.savemat("./test_output/spMAS.mat", {'spMAS':spMAE_list})
        sio.savemat("./test_output/spiMAS.mat", {'spiMAS':spiMAE_list})
        sio.savemat("./test_output/spRMSE.mat", {'spRMSE':spRMSE_list})
        sio.savemat("./test_output/spiRMSE.mat", {'spiRMSE':spiRMSEs_list})
        
        x=len(MAE_list)
        plt.figure(8)
        xx=list(range(1,x+1,1))
        #plt.subplot(326)
        #plt.axis('off') 
        #plt.title('mask map',fontsize='medium',fontweight='bold')
        
        
        plt.scatter(xx,rgbMAE_list,s=3,c='r')
        plt.scatter(xx,spMAE_list,s=3,c='g')
        plt.scatter(xx,MAE_list,s=3,c='b')
        plt.legend(["rgb","sp","rgb+sp"])
        plt.title("MAE")
        plt.xlabel('index')
        plt.ylabel('mm')
        plt.savefig("./test_output/MAE.png")
        plt.close()
        
        plt.figure(9)
        xx=list(range(1,x+1,1))
        #plt.subplot(326)
        #plt.axis('off') 
        #plt.title('mask map',fontsize='medium',fontweight='bold')
        
        
        plt.scatter(xx,rgbiMAE_list,s=3,c='r')
        plt.scatter(xx,spiMAE_list,s=3,c='g')
        plt.scatter(xx,iMAE_list,s=3,c='b')
        plt.legend(["rgb","sp","rgb+sp"])
        #plt.legend(["without SE"])
        plt.title("iMAE")
        mins=min([min(spiMAE_list),min(rgbiMAE_list),min(iMAE_list)])
        maxs=max([max(spiMAE_list),max(rgbiMAE_list),max(iMAE_list)])
        plt.ylim(mins-100, maxs+100)
        
        plt.xlabel('index')
        plt.ylabel('1/km')
        plt.savefig("./test_output/iMAE_list.png")
        plt.close()
        
        plt.figure(10)
        xx=list(range(1,x+1,1))
        #plt.subplot(326)
        #plt.axis('off') 
        #plt.title('mask map',fontsize='medium',fontweight='bold')
        
        
        plt.scatter(xx,rgbMAE_list,s=3,c='r')
        plt.scatter(xx,spMAE_list,s=3,c='g')
        plt.scatter(xx,MAE_list,s=3,c='b')
        plt.legend(["rgb","sp","rgb+sp"])
        plt.title("RMSE")
        plt.xlabel('index')
        plt.ylabel('mm')
        plt.savefig("./test_output/RMSE.png")
        plt.close()
        
        plt.figure(11)
        xx=list(range(1,x+1,1))
        
        
        plt.scatter(xx,rgbiRMSEs_list,s=3,c='r')
        plt.scatter(xx,spiRMSEs_list,s=3,c='g')
        plt.scatter(xx,iRMSEs_list,s=3,c='b')
        plt.title("iRMSE")
        mins=min([min(iRMSEs_list),min(rgbiRMSEs_list),min(spiRMSEs_list)])
        maxs=max([max(iRMSEs_list),max(rgbiRMSEs_list),max(spiRMSEs_list)])
        plt.ylim(0, maxs+100)
        plt.legend(["rgb","sp","rgb+sp"])
        plt.xlabel('index')
        plt.ylabel('1/km')
        plt.savefig("./test_output/iRMSEs.png")
        plt.close()
        
        ave_MAE = np.mean(MAE_list)
        var_MAE = np.var(MAE_list)
        
        ave_iMAE = np.mean(iMAE_list)
        var_iMAE = np.var(iMAE_list)
        
        ave_RMSE = np.mean(RMSE_list)
        var_RMSE = np.var(RMSE_list)
        
        ave_iRMSE = np.mean(iRMSEs_list)
        var_iRMSE = np.var(iRMSEs_list)
######################################################        
        ave_rgbMAE = np.mean(rgbMAE_list)
        var_rgbMAE = np.var(rgbMAE_list)
        
        ave_rgbiMAE = np.mean(rgbiMAE_list)
        var_rgbiMAE = np.var(rgbiMAE_list)
        
        ave_rgbRMSE = np.mean(rgbRMSE_list)
        var_rgbRMSE = np.var(rgbRMSE_list)
        
        ave_rgbiRMSE = np.mean(rgbiRMSEs_list)
        var_rgbiRMSE = np.var(rgbiRMSEs_list)
######################################################        
        ave_spMAE = np.mean(spMAE_list)
        var_spMAE = np.var(spMAE_list)
        
        ave_spiMAE = np.mean(spiMAE_list)
        var_spiMAE = np.var(spiMAE_list)
        
        ave_spRMSE = np.mean(spRMSE_list)
        var_spRMSE = np.var(spRMSE_list)
        
        ave_spiRMSE = np.mean(spiRMSEs_list)
        var_spiRMSE = np.var(spiRMSEs_list)
        
        print("^^^^^^^^^^^^^^^^")
        print(ave_MAE)
        plt.figure(12)
        xx=list(range(0,3))
        #plt.subplot(326)
        #plt.axis('off') 
        #plt.title('mask map',fontsize='medium',fontweight='bold')
        
        
        plt.bar([1],[ave_rgbMAE],width=0.2,color=['r'])
        plt.bar([2],[ave_spMAE],width=0.2,color=['g'])
        plt.bar([3],[ave_MAE],width=0.2,color=['b'])
        plt.xlabel('index')
        plt.ylabel('mm')
        plt.legend(["rgb","sp","rgb+sp"])
        plt.title("MAE")
        plt.savefig("./test_output/MAE_BAR.png")
        plt.close()
        
        plt.bar([1],[ave_rgbiMAE],width=0.2,color=['r'])
        plt.bar([2],[ave_spiMAE],width=0.2,color=['g'])
        plt.bar([3],[ave_iMAE],width=0.2,color=['b'])
        plt.xlabel('index')
        plt.ylabel('1/km')
        plt.legend(["rgb","sp","rgb+sp"])
        plt.title("iMAE")
        mins=min([ave_iMAE,ave_rgbiMAE,ave_spiMAE])
        maxs=max([ave_iMAE,ave_rgbiMAE,ave_spiMAE])
        plt.ylim(0, maxs+30)
        plt.savefig("./test_output/iMAE_BAR.png")
        plt.close()
        
        
        plt.bar([1],[ave_rgbRMSE],width=0.2,color=['r'])
        plt.bar([2],[ave_spRMSE],width=0.2,color=['g'])
        plt.bar([3],[ave_RMSE],width=0.2,color=['b'])
        plt.legend(["rgb","sp","rgb+sp"])
        #plt.legend(["without SE"])
        plt.xlabel('mm')
        plt.ylabel('1/km')
        plt.title("RMSE")
        plt.savefig("./test_output/RMSE_BAR.png")
        plt.close()
        
        
        plt.bar([1],[ave_rgbiRMSE],width=0.2,color=['r'])
        plt.bar([2],[ave_spiRMSE],width=0.2,color=['g'])
        plt.bar([3],[ave_iRMSE],width=0.2,color=['b'])
        mins=min([ave_iRMSE,ave_rgbiRMSE,ave_spiRMSE])
        maxs=max([ave_iRMSE,ave_rgbiRMSE,ave_spiRMSE])
        plt.ylim(0, maxs+30)
        plt.legend(["rgb","sp","rgb+sp"])
        #plt.legend(["without SE"])
        plt.xlabel('index')
        plt.ylabel('1/km')
        plt.title("iRMSE")
        plt.savefig("./test_output/iRMSE_BAR.png")
        plt.close()
        
        
        
        
            
    def get_mask(self,gt):
        mask=tf.less(gt, 50000, name=None)
        mask=tf.cast(mask,tf.float32)
        mask2=tf.greater(gt, 5, name=None)
        mask2=tf.cast(mask2,tf.float32) 
        mask_findal=tf.multiply(mask,mask2)
        #maskp=tf.cast(mask,tf.int32)
        #p=tf.bincount(maskp)
        #num_ture=tf.cast(p[1],tf.float32)
        return mask_findal
    
    
    def four_kind_loss(self):
        pass
        
    def record(self):
        cur=datetime.datetime.now()
        cur.year
        cur.day
        cur.month
        cur.hour
        cur.minute
        Experiment_num=1
        test_loss_sum=0
        test_loss_ave=0
        minist_train_loss=1000
        minist_train_index=-1
        
        for key,value in self.train_result.items():
                if value<minist_train_loss:
                    minist_train_loss=value
                    minist_train_index=key
        
        
        plt.figure()
        plt.subplot(221) 
        plt.scatter(minist_train_index,minist_train_loss,s = 50,color = 'r',lw= 2)#
        plt.text(minist_train_index, minist_train_loss, "minist point", ha='left')
        l_train_i=list(self.train_result.keys())
        l_train_loss=list(self.train_result.values())
        plt.plot(l_train_i, l_train_loss)
        
        plt.subplot(222) 
        l_test_i=list(self.test_result.keys())
        l_test_loss=list(self.test_result.values())
        plt.scatter(l_test_i,l_test_loss,c = 'r',marker = 'x')
        
        plt.subplot(223) 
        l_test_ri=list(self.test_relative.keys())
        l_test_rloss=list(self.test_relative.values())
        plt.scatter(l_test_ri,l_test_rloss,c = 'r',marker = '.')
        
        plt.subplot(224) 
        l_test_avei=list(self.test_average_loss.keys())
        l_test_aveloss=list(self.test_average_loss.values())
        plt.scatter(l_test_avei,l_test_aveloss,c = 'r',marker = '.')

        
        
        plt.savefig("./result_information/"+str(cur.year)+str(cur.month)+str(cur.day)+str(cur.hour)+str(cur.minute)+".png") 
        plt.close()



if __name__ == '__main__':
	tf.reset_default_graph()
	m=Main()
	m.train()




