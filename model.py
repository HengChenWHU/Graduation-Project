import numpy as np
import cv2
import tensorflow as tf
from tflearn.layers.conv import global_avg_pool
import argparse

class Model():
	"""docstring for ClassName"""
	def __init__(self, arg):
		self.arg = arg
		self.trainingmode = tf.constant(True,dtype=tf.bool)
		self.testingmode = tf.constant(False,dtype=tf.bool)
		
		self.global_step = tf.Variable(0, trainable=False)
		self.BATCH_SIZE=arg.BATCH_SIZE
        
		self.IMG_W=arg.IMG_W
		self.IMG_H=arg.IMG_H
        
		self.se_block=[]



	
	def basic_conv_block(self,x,f_num,kernel_size,strides_size,padding_mode,is_training):
		x = tf.layers.conv2d(x,f_num,(kernel_size,kernel_size),strides=(strides_size,strides_size),padding=padding_mode,use_bias=True,kernel_initializer=tf.truncated_normal_initializer() ,bias_initializer=tf.truncated_normal_initializer(),kernel_regularizer=tf.contrib.layers.l2_regularizer(0.1),bias_regularizer=tf.contrib.layers.l2_regularizer(0.1))
		x = tf.layers.batch_normalization(x,training = is_training)
		x = tf.nn.relu(x)
		return x
	
	def basic_deconv_block(self,x,f_num,kernel_size,strides_size,padding_mode,is_training):
		x = tf.layers.conv2d_transpose(x,f_num,(kernel_size,kernel_size),strides=(strides_size,strides_size),padding=padding_mode,use_bias=True,kernel_initializer=tf.truncated_normal_initializer() ,bias_initializer=tf.truncated_normal_initializer(),kernel_regularizer=tf.contrib.layers.l2_regularizer(0.1),bias_regularizer=tf.contrib.layers.l2_regularizer(0.1))
		x = tf.layers.batch_normalization(x,training = is_training)
		x = tf.nn.relu(x)
		return x

	def conv_k3_s2(self,x,filters_num):
		with tf.variable_scope('conv_k3_s2',reuse=tf.AUTO_REUSE):
			x = tf.layers.conv2d(inputs=x,filters=filters_num,kernel_size=(3,3),strides=(2,2),padding='same',use_bias=True,kernel_initializer=tf.truncated_normal_initializer() ,bias_initializer=tf.truncated_normal_initializer(),kernel_regularizer=tf.contrib.layers.l2_regularizer(0.1),bias_regularizer=tf.contrib.layers.l2_regularizer(0.1))
		return x

	def conv_k1_s1(self,x,filters_num):
		x = tf.layers.conv2d(inputs=x,filters=filters_num,kernel_size=(1,1),strides=(1,1),padding='same',use_bias=True,kernel_initializer=tf.truncated_normal_initializer() ,bias_initializer=tf.truncated_normal_initializer(),kernel_regularizer=tf.contrib.layers.l2_regularizer(0.1),bias_regularizer=tf.contrib.layers.l2_regularizer(0.1))
		return x

	def conv_k3_s1(self,x,filters_num):
		x = tf.layers.conv2d(inputs=x,filters=filters_num,kernel_size=(3,3),strides=(1,1),padding='same',use_bias=True,kernel_initializer=tf.truncated_normal_initializer() ,bias_initializer=tf.truncated_normal_initializer(),kernel_regularizer=tf.contrib.layers.l2_regularizer(0.1),bias_regularizer=tf.contrib.layers.l2_regularizer(0.1))
		return x
    
	def conv_k3_s1_ns(self,x,filters_num):
		x = tf.layers.conv2d(inputs=x,filters=filters_num,kernel_size=(3,3),strides=(1,1),padding='valid',use_bias=True,kernel_initializer=tf.truncated_normal_initializer() ,bias_initializer=tf.truncated_normal_initializer(),kernel_regularizer=tf.contrib.layers.l2_regularizer(0.1),bias_regularizer=tf.contrib.layers.l2_regularizer(0.1))
		return x

	def deconv_k3_s2(self,x,filters_num):
		with tf.variable_scope('deconv_k3_s2',reuse=tf.AUTO_REUSE):
			x = tf.layers.conv2d_transpose(inputs=x,filters=filters_num,kernel_size=(3,3),strides=(2,2),padding='same',use_bias=True,kernel_initializer=tf.truncated_normal_initializer() ,bias_initializer=tf.truncated_normal_initializer(),kernel_regularizer=tf.contrib.layers.l2_regularizer(0.1),bias_regularizer=tf.contrib.layers.l2_regularizer(0.1))
		return x

	def deconv_k3_s2_s(self,x,filters_num):
		x = tf.layers.conv2d_transpose(inputs=x,filters=filters_num,kernel_size=(3,3),strides=(2,2),padding='same',use_bias=True,kernel_initializer=tf.truncated_normal_initializer() ,bias_initializer=tf.truncated_normal_initializer(),kernel_regularizer=tf.contrib.layers.l2_regularizer(0.1),bias_regularizer=tf.contrib.layers.l2_regularizer(0.1))
		return x


	def deconv_k1_s1(self,x,filters_num):
		x = tf.layers.conv2d_transpose(inputs=x,filters=filters_num,kernel_size=(1,1),strides=(1,1),padding='same',use_bias=True,kernel_initializer=tf.truncated_normal_initializer() ,bias_initializer=tf.truncated_normal_initializer(),kernel_regularizer=tf.contrib.layers.l2_regularizer(0.1),bias_regularizer=tf.contrib.layers.l2_regularizer(0.1))
		return x

	def deconv_k3_s1(self,x,filters_num):
		x = tf.layers.conv2d_transpose(inputs=x,filters=filters_num,kernel_size=(3,3),strides=(1,1),padding='same',use_bias=True,kernel_initializer=tf.truncated_normal_initializer() ,bias_initializer=tf.truncated_normal_initializer(),kernel_regularizer=tf.contrib.layers.l2_regularizer(0.1),bias_regularizer=tf.contrib.layers.l2_regularizer(0.1))
		return x
		
	def conv_res(self,x,f_num,i,is_training):
		channel_size=x.shape[3]
		if channel_size!=f_num:
			x = self.conv_k3_s2(x,f_num)
		with tf.variable_scope('conv'+str(2*i),reuse=tf.AUTO_REUSE):
			x1 = self.conv_k3_s1(x,f_num)
			x1 = tf.layers.batch_normalization(x1,training = is_training)
			x1 = tf.nn.relu(x1)
			#print('conv'+str(2*i),x1.shape)
		with tf.variable_scope('conv'+str(2*i+1),reuse=tf.AUTO_REUSE):
			x2 = self.conv_k3_s1(x1,f_num)
			x2=x2+x
			x2 = tf.layers.batch_normalization(x2,training = is_training)
			x2 = tf.nn.relu(x2)
			#print('conv'+str(2*i+1),x2.shape)
		return x2

	def Relu(self,x):
		return tf.nn.relu(x)
	def Sigmoid(self,x) :
		return tf.nn.sigmoid(x)
    
	def deconv_res(self,x,f_num,i,is_training):
		channel_size=x.shape[3]
		if channel_size!=f_num:
			x = self.deconv_k3_s2(x,f_num)
		with tf.variable_scope('deconv'+str(2*i),reuse=tf.AUTO_REUSE):
			x1 = self.deconv_k3_s1(x,f_num)
			x1 = tf.layers.batch_normalization(x1,training = is_training)
			x1 = tf.nn.relu(x1)
			#print('deconv'+str(2*i),x1.shape)
		with tf.variable_scope('deconv'+str(2*i+1),reuse=tf.AUTO_REUSE):
			x2 = self.deconv_k3_s1(x1,f_num)
			x2=x2+x
			x2 = tf.layers.batch_normalization(x2,training = is_training)
			x2 = tf.nn.relu(x2)
			#print('deconv'+str(2*i+1),x2.shape)
		return x2
    
	def Global_Average_Pooling(self,x):
		return global_avg_pool(x, name='Global_avg_pooling')
    
	def Fully_connected(self,x, units, layer_name='fully_connected') :
		with tf.name_scope(layer_name) :
			return tf.layers.dense(inputs=x, use_bias=True, units=units)

	def Squeeze_excitation_layer(self,input_x, out_dim, ratio, layer_name):
		with tf.name_scope(layer_name) :
			ratio=8
			squeeze = self.Global_Average_Pooling(input_x)
			excitation = self.Fully_connected(squeeze, units=out_dim / ratio, layer_name=layer_name+'_fully_connected1')
			excitation = self.Relu(excitation)
			excitation = self.Fully_connected(excitation, units=out_dim, layer_name=layer_name+'_fully_connected2')
			excitation = self.Sigmoid(excitation)
			excitation = tf.reshape(excitation, [-1,1,1,out_dim])
			scale = input_x * excitation
			self.se_block.append(excitation[0,0,0,:])
			return scale


	def spp_layer(self,input_, levels=4, name = 'SPP_layer',pool_type = 'max_pool'):   
		shape = input_.get_shape().as_list()
		with tf.variable_scope(name):
			for l in range(levels):
				l = l + 1
				ksize = [1, np.ceil(shape[1]/ l + 1).astype(np.int32), np.ceil(shape[2] / l + 1).astype(np.int32), 1]
				strides = [1, np.floor(shape[1] / l + 1).astype(np.int32), np.floor(shape[2] / l + 1).astype(np.int32), 1]
				if pool_type == 'max_pool':
					pool, maxp1_argmax, maxp1_argmax_mask = self.max_pool(input_, ksize)

					unpool1=self.max_unpool(input_, maxp1_argmax, maxp1_argmax_mask, ksize)
					pool = tf.reshape(pool,(shape[0],-1),)
				if l == 1:
					x_flatten = tf.reshape(pool,(shape[0],-1))
				else:
					x_flatten = tf.concat((x_flatten,pool),axis=1) 
					print("Pool Level {:}: shape {:}".format(l, x_flatten.get_shape().as_list()))
			return x_flatten
        
	def max_pool(self,inp, k):
	     return tf.nn.max_pool_with_argmax_and_mask(inp, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding="SAME")
	
	def max_unpool(self,inp, argmax, argmax_mask, k):
	     return tf.nn.max_unpool(inp, argmax, argmax_mask, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding="SAME")

      
	def add_layer(self,name, l,is_training):
	   shape = l.get_shape().as_list()
	   in_channel = shape[3]
	   with tf.variable_scope(name+"res1",reuse=tf.AUTO_REUSE):
                    
                    c = self.conv_k3_s1(l,in_channel)
                    c = tf.nn.relu(c)
                    c = tf.layers.batch_normalization(c,training = is_training)
                    #c = tf.layers.dropout(c, rate=0.8, training=is_training)
                    
	   with tf.variable_scope(name+"res2",reuse=tf.AUTO_REUSE):
                    
                    c = self.conv_k3_s1(l,in_channel)
                    #c=c+l
                    c = tf.nn.relu(c)
                    c = tf.layers.batch_normalization(c,training = is_training)
                    #c = tf.layers.dropout(c, rate=0.8, training=is_training)
                    l = tf.concat([c, l], 3)
                    return l
   
	def rgb_dense_block(self,rgb_raw,basic_channel,is_training):
              rgb_feature=[]
              rgb_SE=[]
              with tf.variable_scope('basic_channel'):
                  rgb_raw = self.conv_k3_s1(rgb_raw,basic_channel)
                  rgb_raw = tf.nn.relu(rgb_raw)
                  rgb_raw = tf.layers.batch_normalization(rgb_raw,training = is_training)
              with tf.variable_scope('rgb_feature'):
                   d1 = self.add_layer('rgb_dense_layer.{}'.format(1), rgb_raw,is_training)
              with tf.variable_scope('reduce1'):
                   d1 = self.conv_k1_s1(d1,64)
                   
                   d1=d1+rgb_raw
                   d1=self.Squeeze_excitation_layer(d1, 64, 2, "rgb1")
                   print('rgb_f1')
                   print(d1.shape)
                   d2 = self.add_layer('rgb_dense_layer.{}'.format(2), d1,is_training)
                   #d2 = tf.layers.dropout(d2, rate=0.2, training=is_training)
                   
              with tf.variable_scope('reduce2'):
                   d2 = self.conv_k1_s1(d2,64)
                   #d2=d2+d1
                   d2=self.Squeeze_excitation_layer(d2, 64, 2, "rgb2")
                   
                   print('rgb_f2')
                   print(d2.shape)
                   d3 = self.add_layer('rgb_dense_layer.{}'.format(3), d2,is_training)
                   #d3 = tf.layers.dropout(d3, rate=0.2, training=is_training)
              with tf.variable_scope('reduce3'):
                   d3 = self.conv_k1_s1(d3,64)
                   d3=d3+d2
                   d3=self.Squeeze_excitation_layer(d3, 64, 2, "rgb3")
                   print('rgb_f3')
                   print(d3.shape)
                   d4 = self.add_layer('rgb_dense_layer.{}'.format(4), d3,is_training)
                   #d4 = tf.layers.dropout(d4, rate=0.2, training=is_training)
              with tf.variable_scope('reduce4'):
                   d4 = self.conv_k1_s1(d4,64)
                   d4=d4+d3
                   d4=self.Squeeze_excitation_layer(d4, 64, 2, "rgb4")
                   print('rgb_f4')
                   print(d4.shape)
                   d5 = self.add_layer('rgb_dense_layer.{}'.format(5), d4,is_training)
                   #d5 = tf.layers.dropout(d5, rate=0.2, training=is_training)
              with tf.variable_scope('reduce5'):
                   d5 = self.conv_k1_s1(d5,64)
                   d5=d5+d4
                   d5=self.Squeeze_excitation_layer(d5, 64, 2, "rgb5")
                   
                   #d5 = tf.nn.max_pool(d5,ksize=[1, 2, 2, 1],strides=[1, 2, 2, 1],padding='SAME')
                   
              with tf.variable_scope('reduce_channel'):
                   #d5=self.conv_k1_s1(d5,100)
                   print('rgb_f5')
                   print(d5.shape)
                   
                   rgb_feature.append(d1)
                   rgb_feature.append(d2)
                   rgb_feature.append(d3)
                   rgb_feature.append(d4)
                   rgb_feature.append(d5)
                   
                   #rgb_SE.append(rgbexcitation1)
                   #rgb_SE.append(rgbexcitation2)
                   #rgb_SE.append(rgbexcitation3)
                   #rgb_SE.append(rgbexcitation4)
                   #rgb_SE.append(rgbexcitation5)
                   return rgb_feature
	
	def small_fcn(self,img,is_training,name):
               with tf.variable_scope(name):
                    shape = img.get_shape().as_list()
                    in_channel = shape[3]
                    H=shape[1]
                    W=shape[2]
                    with tf.variable_scope("c1"):
                        c1= self.conv_k3_s1(img,in_channel)
                        c1 = tf.nn.relu(c1)
                        c1 = tf.layers.batch_normalization(c1,training = is_training)
                        c1=self.Squeeze_excitation_layer(c1, in_channel, 2, "sp1")
                    
                    with tf.variable_scope("c2"):
                        c2= self.conv_k3_s2(c1,in_channel)
                        c2 = tf.nn.relu(c2)
                        c2 = tf.layers.batch_normalization(c2,training = is_training)
                        c2=self.Squeeze_excitation_layer(c2, in_channel, 2, "sp2")
                        #c2 = tf.layers.dropout(c2, rate=0.5, training=is_training)
                        
                    with tf.variable_scope("c2"):
                        c2_2= self.conv_k3_s2(c2,in_channel)
                        c2_2 = tf.nn.relu(c2_2)
                        c2_2 = tf.layers.batch_normalization(c2_2,training = is_training)
                        c2_2=self.Squeeze_excitation_layer(c2_2, in_channel, 2, "sp3")
                        #c2 = tf.layers.dropout(c2, rate=0.5, training=is_training)
                        
                        
                    with tf.variable_scope("dc1"):
                        c3= self.deconv_k3_s2(c2_2,in_channel)
                        c3 = tf.nn.relu(c3)
                        c3 = tf.layers.batch_normalization(c3,training = is_training)
                        c3=self.Squeeze_excitation_layer(c3, in_channel, 2, "dc1sp1")
                        #c3 = tf.layers.dropout(c3, rate=0.5, training=is_training)
                    with tf.variable_scope("dc2"):
                        c4= self.deconv_k3_s2(c3,in_channel)
                        c4 = tf.nn.relu(c4)
                        c4 = tf.layers.batch_normalization(c4,training = is_training)
                        c4=self.Squeeze_excitation_layer(c4, in_channel, 2, "dc1sp2")
                        
                    with tf.variable_scope("dc3"):
                        c4= self.deconv_k3_s2(c4,in_channel)
                        c4 = tf.nn.relu(c4)
                        c4 = tf.layers.batch_normalization(c4,training = is_training)
                        c4=self.Squeeze_excitation_layer(c4, in_channel, 2, "dc1sp3")
                        #c4 = tf.layers.dropout(c4, rate=0.5, training=is_training)
                        fcn=tf.image.resize_images(c1,(H,W),0)
                        return fcn
            
            
        
                
	def image_filter_block(self,rgb_f,is_training):
               image_filter=[]
               with tf.variable_scope("image_guided_filter"):
                   with tf.variable_scope("igf1"):
                         f1=self.small_fcn(rgb_f[0],is_training,"filter0")
                         print("imf1")
                         print(f1.shape)
                   with tf.variable_scope("igf2"):
                       f2=self.small_fcn(rgb_f[1],is_training,"filter2")
                       f2=f2+f1

                       print("imf2")
                       print(f2.shape)
                   with tf.variable_scope("igf3"):
                       f3=self.small_fcn(rgb_f[2],is_training,"filter3")
                       f3=f3+f2
                       print("imf3")
                       print(f3.shape)
                   with tf.variable_scope("igf4"):
                       f4=self.small_fcn(rgb_f[3],is_training,"filter4")
                       f4=f4+f3
                       print("imf4")
                       print(f4.shape)
                   with tf.variable_scope("igf5"):
                       f5=self.small_fcn(rgb_f[4],is_training,"filter5")
                       f5=f5+f4
                       print("imf5")
                       print(f5.shape)
                   image_filter.append(f1)
                   image_filter.append(f2)
                   image_filter.append(f3)
                   image_filter.append(f4)
                   image_filter.append(f5)
                   return image_filter
    

            
	def sp_dense_block(self,sp_raw,image_guided,basic_channel,is_training,with_imgguided):
               sp_feature=[]
               with tf.variable_scope('basic_channel'):
                  sp_raw = self.conv_k3_s1(sp_raw,basic_channel)
                  sp_raw = tf.nn.relu(sp_raw)
                  sp_raw = tf.layers.batch_normalization(sp_raw,training = is_training)
               with tf.variable_scope('sp_feature'):
                        p1 = self.add_layer('sp_dense_layer.{}'.format(1), sp_raw,is_training)
               with tf.variable_scope('reduce1'):
                        p1 = self.conv_k3_s1(p1,64)
                        p1=p1+sp_raw
                        p1=self.Squeeze_excitation_layer(p1, 64, 2, "sp1")
                        if with_imgguided=='yes':
                            with tf.variable_scope('temp1'):
                                temp=tf.concat([p1,image_guided[0]],3)
                                p1= self.conv_k3_s1(temp,64)
                        print("sp1")
                        print(p1.shape)
                        p2 = self.add_layer('sp_dense_layer.{}'.format(2), p1,is_training)
                        #p2 = tf.layers.dropout(p2, rate=0.2, training=is_training)
               with tf.variable_scope('reduce2'):
                        p2 = self.conv_k3_s1(p2,64)
                        p2=p2+p1
                        p2=self.Squeeze_excitation_layer(p2, 64, 2, "sp2")
                        if with_imgguided=='yes':
                            with tf.variable_scope('temp2'):
                                temp=tf.concat([p2,image_guided[1]],3)
                                p2= self.conv_k3_s1(temp,64)
                        print("sp2")
                        print(p2.shape)
                        p3 = self.add_layer('sp_dense_layer.{}'.format(3), p2,is_training)
                        #p3 = tf.layers.dropout(p3, rate=0.2, training=is_training)
               with tf.variable_scope('reduce3'):
                        p3 = self.conv_k3_s1(p3,64)
                        p3=p3+p2
                        p3=self.Squeeze_excitation_layer(p3, 64, 2, "sp3")
                        if with_imgguided=='yes':
                            with tf.variable_scope('temp3'):
                                temp=tf.concat([p3,image_guided[2]],3)
                                p3= self.conv_k3_s1(temp,64)
                        print("sp3")
                        print(p3.shape)
                        p4 = self.add_layer('sp_dense_layer.{}'.format(4), p3,is_training)
                        #p4 = tf.layers.dropout(p4, rate=0.2, training=is_training)
               with tf.variable_scope('reduce4'):
                        p4 = self.conv_k3_s1(p4,64)
                        p4=p4+p3
                        p4=self.Squeeze_excitation_layer(p4, 64, 2, "sp4")
                        if with_imgguided=='yes':
                            with tf.variable_scope('temp4'):
                                temp=tf.concat([p4,image_guided[3]],3)
                                p4= self.conv_k3_s1(temp,64)
                        print("sp4")
                        print(p4.shape)
                        p5 = self.add_layer('sp_dense_layer.{}'.format(5), p4,is_training)
                        #p5 = tf.layers.dropout(p5, rate=0.2, training=is_training)
               with tf.variable_scope('reduce5'):
                        p5 = self.conv_k3_s1(p5,64)
                        p5=p5+p4
                        p5=self.Squeeze_excitation_layer(p5, 64, 2, "sp5")
                        
                        #p5 = tf.nn.max_pool(p5,ksize=[1, 2, 2, 1],strides=[1, 2, 2, 1],padding='SAME')
                        
                        
               with tf.variable_scope('reduce_channel'):
                        #p5=self.conv_k1_s1(p5,100)
                        if with_imgguided=='yes':
                            with tf.variable_scope('temp5'):
                                temp=tf.concat([p5,image_guided[3]],3)
                                p5= self.conv_k3_s1(temp,64)
                        #p5 = tf.layers.dropout(p5, rate=0.2, training=is_training)
                        print("sp5")
                        print(p5.shape)
                        sp_feature.append(p1)
                        sp_feature.append(p2)
                        sp_feature.append(p3)
                        sp_feature.append(p4)
                        sp_feature.append(p5)
                        return sp_feature
       
	def U_net_block(self,fusion,is_training):
                with tf.variable_scope('U_BLOCK',reuse=tf.AUTO_REUSE):
                        with tf.variable_scope('U_1',reuse=tf.AUTO_REUSE):
                                u1 = self.conv_k3_s1(fusion,512)
                                u1 = tf.nn.relu(u1)
                                #u1=self.Squeeze_excitation_layer(u1, 512, 2, "u1")
                                u1 = tf.layers.batch_normalization(u1,training = is_training)
                                #u1 = tf.layers.dropout(u1, rate=0.8, training=is_training)
                                print("u_1")
                                print(u1.shape)
                        with tf.variable_scope('U_1_1',reuse=tf.AUTO_REUSE):
                                u1_1 = self.conv_k3_s1(fusion,512)
                                u1_1=u1_1+u1
                                u1_1 = tf.nn.relu(u1_1)
                                #u1_1=self.Squeeze_excitation_layer(u1_1, 512, 2, "u1_1")
                                u1_1 = tf.layers.batch_normalization(u1_1,training = is_training)
                                #u1_1 = tf.layers.dropout(u1_1, rate=0.8, training=is_training)
                                print("u_1_1")
                                print(u1_1.shape)
                                
                        with tf.variable_scope('U_2',reuse=tf.AUTO_REUSE):
                                u2 = self.conv_k3_s2(u1_1,256)
                                u2 = tf.nn.relu(u2)
                                u2 = tf.layers.batch_normalization(u2,training = is_training)
                                #u2=self.Squeeze_excitation_layer(u2, 256, 2, "u2")
                                #u2 = tf.layers.dropout(u2, rate=0.8, training=is_training)
                                print("u_2")
                                print(u2.shape)
                        
                        with tf.variable_scope('U_2_2',reuse=tf.AUTO_REUSE):
                                u2_2 = self.conv_k3_s1(u2,256)
                                u2_2=u2_2+u2
                                u2_2 = tf.nn.relu(u2_2)
                                u2_2 = tf.layers.batch_normalization(u2_2,training = is_training)
                                #u2_2=self.Squeeze_excitation_layer(u2_2, 256, 2, "u2_2")
                                #u2_2 = tf.layers.dropout(u2_2, rate=0.8, training=is_training)
                                print("u_2_2")
                                print(u2_2.shape)
                                
                        with tf.variable_scope('U_3',reuse=tf.AUTO_REUSE):
                                u3 = self.conv_k3_s2(u2_2,64)
                                u3 = tf.nn.relu(u3)
                                u3 = tf.layers.batch_normalization(u3,training = is_training)
                                #u3=self.Squeeze_excitation_layer(u3, 64, 2, "u3")
                                #u3 = tf.layers.dropout(u3, rate=0.8, training=is_training)
                                print("u_3")
                                print(u3.shape)
                        
                        with tf.variable_scope('U_3_3',reuse=tf.AUTO_REUSE):
                                u3_3 = self.conv_k3_s1(u3,64)
                                u3_3=u3_3+u3
                                u3_3 = tf.nn.relu(u3_3)
                                u3_3 = tf.layers.batch_normalization(u3_3,training = is_training)
                                #u3_3=self.Squeeze_excitation_layer(u3_3, 64, 2, "u3_3")
                                #u3_3 = tf.layers.dropout(u3_3, rate=0.8, training=is_training)
                                print("u_3_3")
                                print(u3_3.shape)
                                
                        with tf.variable_scope('U_4',reuse=tf.AUTO_REUSE):
                                u4 = self.deconv_k3_s2(u3_3,32)
                                u4 = tf.nn.relu(u4)
                                u4 = tf.layers.batch_normalization(u4,training = is_training)
                                #u4=self.Squeeze_excitation_layer(u4, 32, 2, "u3_3")
                                #u4 = tf.layers.dropout(u4, rate=0.8, training=is_training)
                                print("u_4")
                                print(u4.shape)
                        
                        with tf.variable_scope('U_4_4',reuse=tf.AUTO_REUSE):
                                u4_4 = self.conv_k3_s1(u4,32)
                                u4_4=u4_4+u4
                                u4_4 = tf.nn.relu(u4_4)
                                u4_4 = tf.layers.batch_normalization(u4_4,training = is_training)
                                #u4_4=self.Squeeze_excitation_layer(u4_4, 32, 2, "u4_4")
                                #u4_4 = tf.layers.dropout(u4_4, rate=0.8, training=is_training)
                                
                                u2_2=tf.image.resize_images(u2_2,(48,156),0)
                                u4_4=tf.concat([u4_4, u2_2], 3)
                                
                                print("u_4_4")
                                print(u4_4.shape)
                                
                        with tf.variable_scope('U_5',reuse=tf.AUTO_REUSE):
                                u5 = self.deconv_k3_s2(u4_4,16)
                                u5 = tf.nn.relu(u5)
                                u5 = tf.layers.batch_normalization(u5,training = is_training)
                                #u5=self.Squeeze_excitation_layer(u5, 16, 2, "u5")
                                #u5 = tf.layers.dropout(u5, rate=0.8, training=is_training)
                                print("u_5")
                                print(u5.shape)
                                
                        with tf.variable_scope('U_5_5',reuse=tf.AUTO_REUSE):
                                u5_5 = self.conv_k3_s1(u5,16)
                                u5_5=u5_5+u5
                                u5_5 = tf.nn.relu(u5_5)
                                u5_5 = tf.layers.batch_normalization(u5_5,training = is_training)
                                #u5_5 = tf.layers.dropout(u5_5, rate=0.8, training=is_training)
                                #u5_5=self.Squeeze_excitation_layer(u5_5, 16, 2, "u5_5")
                                
                                u1_1=tf.image.resize_images(u1_1,(96,312),0)
                                u5_5=tf.concat([u5_5, u1_1], 3)
                                print("u_5_5")
                                print(u5_5.shape)
                                
                        with tf.variable_scope('U_6',reuse=tf.AUTO_REUSE):
                                u6 = self.deconv_k3_s2(u5_5,8)
                                u6 = tf.nn.relu(u6)
                                u6 = tf.layers.batch_normalization(u6,training = is_training)
                                #u6=self.Squeeze_excitation_layer(u6, 8, 2, "u6")
                                
                                #u6 = tf.layers.dropout(u6, rate=0.8, training=is_training)
                                print("u_6")
                                print(u6.shape)
                                
                        with tf.variable_scope('U_6_6',reuse=tf.AUTO_REUSE):
                                u6_6 = self.conv_k3_s1(u6,8)
                                u6_6=u6_6+u6
                                u6_6 = tf.nn.relu(u6_6)
                                u6_6 = tf.layers.batch_normalization(u6_6,training = is_training)
                                #u6_6=self.Squeeze_excitation_layer(u6_6, 8, 2, "u6_6")
                                
                                #u6_6 = tf.layers.dropout(u6_6, rate=0.8, training=is_training)
                                print("u_6_6")
                                print(u6_6.shape)
                                
                        with tf.variable_scope('U_7',reuse=tf.AUTO_REUSE):
                                u7 = self.deconv_k3_s2(u6_6,4)
                                u7 = tf.nn.relu(u7)
                                u7 = tf.layers.batch_normalization(u7,training = is_training)
                                #u7=self.Squeeze_excitation_layer(u7, 4, 2, "u7")
                                
                                #u7 = tf.layers.dropout(u7, rate=0.8, training=is_training)
                                print("u_7")
                                print(u7.shape)
                        
                        with tf.variable_scope('U_7_7',reuse=tf.AUTO_REUSE):
                                u7_7 = self.conv_k3_s1(u7,4)
                                u7_7=u7_7+u7
                                u7_7 = tf.nn.relu(u7_7)
                                u7_7 = tf.layers.batch_normalization(u7_7,training = is_training)
                                #u7_7=self.Squeeze_excitation_layer(u7_7, 4, 2, "u7_7")
                                
                                #u7_7 = tf.layers.dropout(u7_7, rate=0.8, training=is_training)
                                print("u_7_7")
                                print(u7_7.shape)
                        
                        with tf.variable_scope('regresion',reuse=tf.AUTO_REUSE):
                                re = self.conv_k1_s1(u7_7,1)
                                re = tf.nn.relu(re)
                                re=tf.image.resize_images(re,(self.IMG_H,self.IMG_W),0)
                                print("regresion")
                                print(re.shape)
                        return re

	def network(self,rgb,sp,is_training):
		print("input_rgb",rgb.shape)
		print("input_sp",sp.shape)
		print("exract feature from rgb images")
		with tf.variable_scope('rgb',reuse=tf.AUTO_REUSE):
                    rgb=self.conv_k3_s2(rgb,32)
                    rgb = tf.nn.relu(rgb)
                    rgb=tf.layers.batch_normalization(rgb,training = is_training)
                    #rgb = tf.layers.dropout(rgb, rate=0.2, training=is_training)
                    print("rgb downsample1")
                    print(rgb.shape)
		with tf.variable_scope('rgb2',reuse=tf.AUTO_REUSE):
                    rgb=self.conv_k3_s2(rgb,64)
                    rgb = tf.nn.relu(rgb)
                    rgb=tf.layers.batch_normalization(rgb,training = is_training)
                    print("rgb downsample2")
                    print(rgb.shape)
                    #rgb = tf.layers.dropout(rgb, rate=0.2, training=is_training)
		with tf.variable_scope('sp',reuse=tf.AUTO_REUSE):
                    sp=self.conv_k3_s2(sp,32)
                    sp = tf.nn.relu(sp)
                    sp=tf.layers.batch_normalization(sp,training = is_training)
                    #sp = tf.layers.dropout(sp, rate=0.2, training=is_training)
		with tf.variable_scope('sp2',reuse=tf.AUTO_REUSE):
                    sp=self.conv_k3_s2(sp,64)
                    sp = tf.nn.relu(sp)
                    sp=tf.layers.batch_normalization(sp,training = is_training)
                    #sp = tf.layers.dropout(sp, rate=0.2, training=is_training)
		with tf.variable_scope('rgb_feature_exract',reuse=tf.AUTO_REUSE):
                    rgb_feature=self.rgb_dense_block(rgb,64,is_training)
                    
		with tf.variable_scope('rgb_predict',reuse=tf.AUTO_REUSE):
                    with tf.variable_scope('deconv1',reuse=tf.AUTO_REUSE):
                        rgb_predict=self.deconv_k3_s2(rgb_feature[4],128)
                        rgb_predict = tf.nn.relu(rgb_predict)
                        rgb_predict=tf.layers.batch_normalization(rgb_predict,training = is_training)
                        print("upsample1")
                        print(rgb_predict.shape)
                    with tf.variable_scope('deconv2',reuse=tf.AUTO_REUSE):
                        rgb_predict=self.deconv_k3_s2(rgb_predict,32)
                        rgb_predict = tf.nn.relu(rgb_predict)
                        rgb_predict=tf.layers.batch_normalization(rgb_predict,training = is_training)
                        print("upsample2")
                        print(rgb_predict.shape)
                    with tf.variable_scope('rgb_regresion',reuse=tf.AUTO_REUSE):
                        rgb_predict=self.conv_k1_s1(rgb_predict,1)
                        rgb_predict = tf.nn.relu(rgb_predict)
                        rgb_predict=tf.image.resize_images(rgb_predict,(self.IMG_H,self.IMG_W),0)
                        print("rgb_predict")
                        print(rgb_predict.shape)
                        
		
       			
		with tf.variable_scope('image_guided_filter',reuse=tf.AUTO_REUSE):
                    igf=self.image_filter_block(rgb_feature,is_training)
		print("exract feature from spare depth images")      
		with tf.variable_scope('sp_feature_exract',reuse=tf.AUTO_REUSE):
                    sp_feature=self.sp_dense_block(sp,igf,64,is_training,'yes')
                    
		with tf.variable_scope('sp_predict',reuse=tf.AUTO_REUSE):
                    with tf.variable_scope('spdeconv1',reuse=tf.AUTO_REUSE):
                        sp_predict=self.deconv_k3_s2(sp_feature[4],128)
                        sp_predict = tf.nn.relu(sp_predict)
                        sp_predict=tf.layers.batch_normalization(sp_predict,training = is_training)
                        print("spupsample1")
                        print(sp_predict.shape)
                    with tf.variable_scope('deconv2',reuse=tf.AUTO_REUSE):
                        sp_predict=self.deconv_k3_s2(sp_predict,32)
                        sp_predict = tf.nn.relu(sp_predict)
                        sp_predict=tf.layers.batch_normalization(sp_predict,training = is_training)
                        print("spupsample2")
                        print(sp_predict.shape)
                    with tf.variable_scope('sp_regresion',reuse=tf.AUTO_REUSE):
                        sp_predict=self.conv_k1_s1(sp_predict,1)
                        sp_predict = tf.nn.relu(sp_predict)
                        sp_predict=tf.image.resize_images(sp_predict,(self.IMG_H,self.IMG_W),0)
                        print("sp_predict")
                        print(sp_predict.shape)
                        
		with tf.variable_scope('fusion_feature_exract',reuse=tf.AUTO_REUSE):
                    fusion=tf.concat([sp_feature[4],rgb_feature[4]],axis=3)
                    #fusion=self.sp_dense_block(sp,igf,64,is_training,'yes')
       
		with tf.variable_scope('fusion_predict',reuse=tf.AUTO_REUSE):
                    with tf.variable_scope('fusiondeconv1',reuse=tf.AUTO_REUSE):
                        fusion=self.deconv_k3_s2(fusion,128)
                        fusion = tf.nn.relu(fusion)
                        fusion=tf.layers.batch_normalization(fusion,training = is_training)
                        print("fusionupsample1")
                        print(fusion.shape)
                        
                    with tf.variable_scope('fusiondeconv2',reuse=tf.AUTO_REUSE):
                        fusion2=self.conv_k3_s1(fusion,128)
                        fusion2=fusion2+fusion
                        fusion2 = tf.nn.relu(fusion2)
                        fusion2=tf.layers.batch_normalization(fusion2,training = is_training)

                        print("fusionupsample2")
                        print(fusion2.shape)
                        
                        
                    with tf.variable_scope('fusiondeconv3',reuse=tf.AUTO_REUSE):
                        fusion3=self.deconv_k3_s2(fusion2,64)
                        fusion3 = tf.nn.relu(fusion3)
                        fusion3=tf.layers.batch_normalization(fusion3,training = is_training)
                        print("fusionupsample2")
                        print(fusion3.shape)
                        
                    with tf.variable_scope('fusiondeconv4',reuse=tf.AUTO_REUSE):
                        fusion4=self.conv_k3_s1(fusion3,64)
                        fusion4=fusion4+fusion3
                        fusion4 = tf.nn.relu(fusion4)
                        fusion4=tf.layers.batch_normalization(fusion4,training = is_training)
                        print("fusionupsample2")
                        print(fusion4.shape)
                    
                    with tf.variable_scope('fusion_regresion1',reuse=tf.AUTO_REUSE):
                        fusion5=self.conv_k3_s1(fusion4,32)
                        fusion5 = tf.nn.relu(fusion5)
                        print("fusion_predict")
                        print(fusion5.shape)
                        
                    with tf.variable_scope('fusion_regresion2',reuse=tf.AUTO_REUSE):
                        fusion6=self.conv_k3_s1(fusion5,8)
                        fusion6 = tf.nn.relu(fusion6)
                        print("fusion_predict2")
                        print(fusion6.shape)
                        
                    with tf.variable_scope('fusion_regresion3',reuse=tf.AUTO_REUSE):
                        fusion7=self.conv_k1_s1(fusion6,1)
                        fusion7 = tf.nn.relu(fusion7)
                        fusion7=tf.image.resize_images(fusion7,(self.IMG_H,self.IMG_W),0)
                        print("fusion_predict")
                        print(fusion7.shape)
    
		return fusion7,rgb_predict,sp_predict,igf,self.se_block
    
    
	def network2(self,x,sp,is_training):
		print("input",x.shape)
		with tf.variable_scope('layer1',reuse=tf.AUTO_REUSE) as scope:
			with tf.variable_scope('rgb',reuse=tf.AUTO_REUSE):
				x=self.basic_conv_block(x,32,3,2,"same",is_training)
				#ttt=self.local_feature_exact(x)
				#print(ttt)
			with tf.variable_scope('sp',reuse=tf.AUTO_REUSE):
				sp=self.basic_conv_block(sp,32,3,2,"same",is_training)
		print("layer1",x.shape)


		with tf.variable_scope('Convlayer2',reuse=tf.AUTO_REUSE) as scope:
			with tf.variable_scope('rgb2',reuse=tf.AUTO_REUSE):
				x=self.conv_res(x,32,2,is_training)
			with tf.variable_scope('sp2',reuse=tf.AUTO_REUSE):
				sp=self.conv_res(sp,32,2,is_training)


		print("Reslayer2",x.shape)
		with tf.variable_scope('Convlayer3',reuse=tf.AUTO_REUSE) as scope:
			with tf.variable_scope('rgb3',reuse=tf.AUTO_REUSE):
				x=self.conv_res(x,64,3,is_training)
			with tf.variable_scope('sp3',reuse=tf.AUTO_REUSE):
				sp=self.conv_res(sp,64,3,is_training)

		print("Reslayer3",x.shape)
		with tf.variable_scope('Convlayer4',reuse=tf.AUTO_REUSE) as scope:
			with tf.variable_scope('rgb4',reuse=tf.AUTO_REUSE):
				x=self.conv_res(x,128,3,is_training)
			with tf.variable_scope('sp4',reuse=tf.AUTO_REUSE):
				sp=self.conv_res(sp,128,3,is_training)


		print("Reslayer4",x.shape)
		with tf.variable_scope('Convlayer5',reuse=tf.AUTO_REUSE) as scope:
			with tf.variable_scope('rgb5',reuse=tf.AUTO_REUSE):
				x=self.conv_res(x,256,3,is_training)
			with tf.variable_scope('sp5',reuse=tf.AUTO_REUSE):
				sp=self.conv_res(sp,256,3,is_training)


		print("Reslayer5",x.shape)
		with tf.variable_scope('Convlayer6',reuse=tf.AUTO_REUSE) as scope:
			with tf.variable_scope('rgb6',reuse=tf.AUTO_REUSE):
				x=self.conv_res(x,512,3,is_training)
			with tf.variable_scope('sp6',reuse=tf.AUTO_REUSE):
				sp=self.conv_res(sp,512,3,is_training)


		print("Reslayer6",x.shape)

		with tf.variable_scope('Convlayer66',reuse=tf.AUTO_REUSE) as scope:
			with tf.variable_scope('rgb66',reuse=tf.AUTO_REUSE):
				x=self.conv_res(x,512,3,is_training)
			with tf.variable_scope('sp66',reuse=tf.AUTO_REUSE):
				sp=self.conv_res(sp,512,3,is_training)

		print("Reslayer6",x.shape)

		with tf.variable_scope('DeConvlayer7',reuse=tf.AUTO_REUSE) as scope:
			with tf.variable_scope('rgb66',reuse=tf.AUTO_REUSE):
				x=self.deconv_res(x,256,3,is_training)
			with tf.variable_scope('sp66',reuse=tf.AUTO_REUSE):
				sp=self.deconv_res(sp,256,3,is_training)

		print("Reslayer7",x.shape)
		with tf.variable_scope('DeConvlayer8',reuse=tf.AUTO_REUSE) as scope:
			with tf.variable_scope('rgb7',reuse=tf.AUTO_REUSE):
				x=self.deconv_res(x,128,3,is_training)
			with tf.variable_scope('sp7',reuse=tf.AUTO_REUSE):
				sp=self.deconv_res(sp,128,3,is_training)


		print("Reslayer8",x.shape)
		with tf.variable_scope('DeConvlayer9',reuse=tf.AUTO_REUSE) as scope:
			with tf.variable_scope('rgb',reuse=tf.AUTO_REUSE):
				x=self.deconv_res(x,64,3,is_training)
			with tf.variable_scope('sp',reuse=tf.AUTO_REUSE):
				sp=self.deconv_res(sp,64,3,is_training)

		print("Reslayer9",x.shape)
		with tf.variable_scope('DeConvlayer10',reuse=tf.AUTO_REUSE) as scope:
			with tf.variable_scope('rgb',reuse=tf.AUTO_REUSE):
				x=self.deconv_res(x,32,3,is_training)
			with tf.variable_scope('sp',reuse=tf.AUTO_REUSE):
				sp=self.deconv_res(sp,32,3,is_training)

			xsp=tf.concat([x,sp],3)

		print("Reslayer10",x.shape)
		with tf.variable_scope('output',reuse=tf.AUTO_REUSE):
			xsp=self.deconv_k3_s2(xsp,1)


		print("output",xsp.shape)
		self.xsp=xsp
		return xsp
    
	def network3(self,rgb,sp,is_training):
		print("input_rgb",rgb.shape)
		print("input_sp",sp.shape)
		print("exract feature from rgb images")
		with tf.variable_scope('rgb',reuse=tf.AUTO_REUSE):
                    rgb=self.conv_k3_s2(rgb,32)
                    rgb = tf.nn.relu(rgb)
                    rgb=tf.layers.batch_normalization(rgb,training = is_training)
                    rgb = tf.layers.dropout(rgb, rate=0.8, training=is_training)
                    print("rgb downsample1")
                    print(rgb.shape)
		with tf.variable_scope('rgb2',reuse=tf.AUTO_REUSE):
                    rgb=self.conv_k3_s2(rgb,32)
                    rgb = tf.nn.relu(rgb)
                    rgb=tf.layers.batch_normalization(rgb,training = is_training)
                    print("rgb downsample2")
                    print(rgb.shape)
                    rgb = tf.layers.dropout(rgb, rate=0.8, training=is_training)
		with tf.variable_scope('sp',reuse=tf.AUTO_REUSE):
                    sp=self.conv_k3_s2(sp,32)
                    sp = tf.nn.relu(sp)
                    sp=tf.layers.batch_normalization(sp,training = is_training)
                    sp = tf.layers.dropout(sp, rate=0.8, training=is_training)
		with tf.variable_scope('sp2',reuse=tf.AUTO_REUSE):
                    sp=self.conv_k3_s2(sp,32)
                    sp = tf.nn.relu(sp)
                    sp=tf.layers.batch_normalization(sp,training = is_training)
		with tf.variable_scope('rgb_feature_exract',reuse=tf.AUTO_REUSE):
                    rgb_feature=self.rgb_dense_block(rgb,64,is_training)
                    
		with tf.variable_scope('rgb_predict',reuse=tf.AUTO_REUSE):
                    with tf.variable_scope('deconv1',reuse=tf.AUTO_REUSE):
                        rgb_predict=self.deconv_k3_s2(rgb_feature[4],128)
                        rgb_predict = tf.nn.relu(rgb_predict)
                        rgb_predict=tf.layers.batch_normalization(rgb_predict,training = is_training)
                        print("upsample1")
                        print(rgb_predict.shape)
                    with tf.variable_scope('deconv2',reuse=tf.AUTO_REUSE):
                        rgb_predict=self.deconv_k3_s2(rgb_predict,32)
                        rgb_predict = tf.nn.relu(rgb_predict)
                        rgb_predict=tf.layers.batch_normalization(rgb_predict,training = is_training)
                        print("upsample2")
                        print(rgb_predict.shape)
                    with tf.variable_scope('rgb_regresion',reuse=tf.AUTO_REUSE):
                        rgb_predict=self.conv_k1_s1(rgb_predict,1)
                        rgb_predict = tf.nn.relu(rgb_predict)
                        rgb_predict=tf.image.resize_images(rgb_predict,(self.IMG_H,self.IMG_W),0)
                        print("rgb_predict")
                        print(rgb_predict.shape)
                        
		
       			
		with tf.variable_scope('image_guided_filter',reuse=tf.AUTO_REUSE):
                    igf=self.image_filter_block(rgb_feature,is_training)
		print("exract feature from spare depth images")      
		with tf.variable_scope('sp_feature_exract',reuse=tf.AUTO_REUSE):
                    sp_feature=self.sp_dense_block(sp,igf,64,is_training,'yes')
                    
		with tf.variable_scope('sp_predict',reuse=tf.AUTO_REUSE):
                    with tf.variable_scope('spdeconv1',reuse=tf.AUTO_REUSE):
                        sp_predict=self.deconv_k3_s2(sp_feature[4],128)
                        sp_predict = tf.nn.relu(sp_predict)
                        sp_predict=tf.layers.batch_normalization(sp_predict,training = is_training)
                        print("spupsample1")
                        print(sp_predict.shape)
                    with tf.variable_scope('deconv2',reuse=tf.AUTO_REUSE):
                        sp_predict=self.deconv_k3_s2(sp_predict,32)
                        sp_predict = tf.nn.relu(sp_predict)
                        sp_predict=tf.layers.batch_normalization(sp_predict,training = is_training)
                        print("spupsample2")
                        print(sp_predict.shape)
                    with tf.variable_scope('sp_regresion',reuse=tf.AUTO_REUSE):
                        sp_predict=self.conv_k1_s1(sp_predict,1)
                        sp_predict = tf.nn.relu(sp_predict)
                        sp_predict=tf.image.resize_images(sp_predict,(self.IMG_H,self.IMG_W),0)
                        print("sp_predict")
                        print(sp_predict.shape)

		with tf.variable_scope('feature_fusion',reuse=tf.AUTO_REUSE):
                    
                    fu=tf.concat([rgb_feature[4], sp_feature[4]], 3)
                    print("fusion shape")
                    print(fu.shape)
                    fu=self.Squeeze_excitation_layer(fu, 128, 2, "SE_BLOCK")
                    print("fusion SE")
                    print(fu.shape)
		print("exract feature from spare depth images")
		with tf.variable_scope('depth_predict',reuse=tf.AUTO_REUSE):
                    depth=self.U_net_block(fu,is_training)
		return depth,rgb_predict,sp_predict

	def loss(self,predictrgb,predictsp,predictfusion,groundtruth):
		self.loss=tf.reduce_mean(tf.abs(tf.subtract(predictrgb,groundtruth)))+tf.reduce_mean(tf.abs(tf.subtract(predictsp,groundtruth)))+tf.reduce_mean(tf.abs(tf.subtract(predictfusion,groundtruth)))
		return self.loss
    
	def MAE_loss(self,predict,groundtruth):
		maeloss=tf.reduce_mean(tf.abs(tf.subtract(predict,groundtruth)))
		return maeloss
    
	def iMAE_loss(self,predict,groundtruth):
		#predict=tf.multiply(predict,tf.constant(0.000001))
		#groundtruth=tf.multiply(groundtruth,tf.constant(0.000001))
       #tf.where(condition,x=None,y=None,name=None)
		predict2=tf.cast(predict,dtype=tf.float64)
		predict2=predict2/1000000.0
        
		groundtruth2=tf.cast(groundtruth,dtype=tf.float64)
		groundtruth2=groundtruth2/1000000.0
		imaeloss=tf.reduce_mean(tf.abs((groundtruth2-predict2)/(groundtruth2*predict2)))
		return imaeloss
    
	def RMSE_loss(self,predict,groundtruth):
		rmseloss=tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(predict,groundtruth))))
		return rmseloss
    
	def iRMSE_loss(self,predict,groundtruth):
		predict2=tf.cast(predict,dtype=tf.float64)
		predict2=predict2/1000000.0
		groundtruth2=tf.cast(groundtruth,dtype=tf.float64)
		groundtruth2=groundtruth2/1000000.0
		rmseloss=tf.sqrt(tf.reduce_mean(tf.square(tf.divide(tf.subtract(groundtruth2,predict2),tf.multiply(predict2,groundtruth2)))))
		#rmseloss2=rmseloss/self.IMG_W=arg.IMG_W/self.IMG_W=arg.IMG_H/self.BATCH_SIZE
		return rmseloss
    
	def test_loss(self,predict,groundtruth):
		test_loss=tf.reduce_mean(tf.abs(tf.subtract(predict,groundtruth)))
		return test_loss

	def optimize(self,learning_rate):
		regularization_loss = tf.reduce_sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
		total_loss=regularization_loss+self.loss
		update_ops=tf.get_collection(tf.GraphKeys.UPDATE_OPS)
		with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
			train_opt = tf.train.AdamOptimizer(learning_rate).minimize(total_loss,global_step=self.global_step)
		return train_opt,regularization_loss
		#	tf.train.GradientDescentOptimizer  AdamOptimizer	


	

if __name__ == '__main__':
	import matplotlib.pyplot as plt
	parser = argparse.ArgumentParser(description='depth completion')
	parser.add_argument('--rgb_path',dest='rgb_path',default="E:/program/self-supervised-depth-completion-master/data/data_rgb/train/2011_09_26_drive_0001_sync/2011_09_26/2011_09_26_drive_0001_sync/image_02/data")
	parser.add_argument('--spare_depth_path',dest='spare_depth_path',default="E:/program/self-supervised-depth-completion-master/data/data_depth_velodyne/train/2011_09_26_drive_0001_sync/proj_depth/velodyne_raw/image_02")
	parser.add_argument('--denth_depth_path',dest='denth_depth_path',default="E:/program/self-supervised-depth-completion-master/data/data_depth_annotated/train/2011_09_26_drive_0001_sync/proj_depth/groundtruth/image_02")
	parser.add_argument('--BATCH_SIZE',dest='BATCH_SIZE',default=2)
	parser.add_argument('--IMG_H',dest='IMG_H',default=300)
	parser.add_argument('--IMG_W',dest='IMG_W',default=300)
	args = parser.parse_args()
	test_rgb='./005.png'
	test_rgb_date=cv2.imread(test_rgb)
	#print(test_rgb_date)
	image_rgb=tf.convert_to_tensor(test_rgb_date)
	image_rgb=tf.to_float(image_rgb, name='ToFloat')
	image_rgb=tf.expand_dims(image_rgb,0)

	model=Model(args)
	testingmode = tf.constant(False,dtype=tf.bool)
	trainingmode = tf.constant(False,dtype=tf.bool)
	output=model.network(image_rgb,image_rgb,testingmode)

	sess= tf.Session()
	sess.run(tf.global_variables_initializer())
	o=sess.run(output)
	print("#################################")
	print(o.shape)
	plt.imshow(o[0,:,:,0])
	plt.axis("off")
	plt.show()





