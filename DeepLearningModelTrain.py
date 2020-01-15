from __future__ import print_function
from hyperopt import Trials, STATUS_OK, tpe
from hyperas import optim
from hyperas.distributions import choice, uniform, conditional
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session 
config = tf.ConfigProto(allow_soft_placement=True) 
config.gpu_options.allow_growth=True 
set_session(tf.Session(config=config)) 
from keras import backend as K
import numpy as np

def sensitivity(y_true, y_pred):
	true_label=K.argmax(y_true, axis=-1)
	pred_label=K.argmax(y_pred, axis=-1)
	INTERESTING_CLASS_ID=2
	sample_mask=K.cast(K.not_equal(true_label, INTERESTING_CLASS_ID), 'int32')
	
	TP_tmp1=K.cast(K.equal(true_label,0),'int32')*sample_mask
	TP_tmp2=K.cast(K.equal(pred_label,0),'int32')*sample_mask
	TP=K.sum(TP_tmp1*TP_tmp2)
	
	FN_tmp1=K.cast(K.equal(true_label,0),'int32')*sample_mask
	FN_tmp2=K.cast(K.not_equal(pred_label,0),'int32')*sample_mask
	FN=K.sum(FN_tmp1*FN_tmp2)
	
	epsilon=0.000000001
	return K.cast(TP,'float')/(K.cast(TP,'float')+K.cast(FN,'float')+epsilon)

def precision(y_true, y_pred):
	true_label=K.argmax(y_true, axis=-1)
	pred_label=K.argmax(y_pred, axis=-1)
	INTERESTING_CLASS_ID=2
	sample_mask=K.cast(K.not_equal(true_label, INTERESTING_CLASS_ID), 'int32')
	
	TP_tmp1=K.cast(K.equal(true_label,0),'int32')*sample_mask
	TP_tmp2=K.cast(K.equal(pred_label,0),'int32')*sample_mask
	TP=K.sum(TP_tmp1*TP_tmp2)

	FP_tmp1=K.cast(K.not_equal(true_label,0),'int32')*sample_mask
	FP_tmp2=K.cast(K.equal(pred_label,0),'int32')*sample_mask
	FP=K.sum(FP_tmp1*FP_tmp2)
	
	epsilon=0.000000001
	return K.cast(TP,'float')/(K.cast(TP,'float')+K.cast(FP,'float')+epsilon)
def f1_score(y_true, y_pred):
	pre=precision(y_true, y_pred)
	sen=sensitivity(y_true, y_pred)
	f1=2*pre*sen/(pre+sen)
	return f1

def data(trainlistpath,testlistpath):
	from keras.utils import np_utils
	import numpy as np
	np.random.seed(1337) 
	from keras.preprocessing.sequence import pad_sequences
	seq_length=700
	domaindata_path='/home/shiqiang/feature_extraction/DeepDomFeatures/train/'
	label_path='/home/shiqiang/feature_extraction/DeepDomFeatures/train/'
	L_file = open(trainlistpath, 'r')
	k=0
	for line in L_file:
		if line.strip() == "":
			continue
		k=k+1
		if k==1:
			chain_name = line.split()[0]
			featurefile=domaindata_path+chain_name+'/'+chain_name+'.fea'
			codeEncodefile=domaindata_path+chain_name+'/'+chain_name+'.txt_encode'
			train_x = np.loadtxt(featurefile, dtype=np.float32)
			xxx=train_x.reshape(1,train_x.shape[0],train_x.shape[1])
			x=pad_sequences(xxx,seq_length,padding='post')
			X_train=x
			train_x2=np.loadtxt(codeEncodefile, dtype=np.float32)
			print(train_x2.shape)
			xxx2=train_x2.reshape(1,train_x2.shape[0])
			X_train2 =xxx2
			labelFile=label_path+chain_name+'/'+chain_name+'.labelvecNEW3'
			yy=np.loadtxt(labelFile, dtype=np.int64)
			Y_train=yy.reshape(1,yy.shape[0],yy.shape[1])
			
		else:
			chain_name = line.split()[0]
			featurefile=domaindata_path+chain_name+'/'+chain_name+'.fea'
			codeEncodefile=domaindata_path+chain_name+'/'+chain_name+'.txt_encode'
			train_x = np.loadtxt(featurefile, dtype=np.float32)
			xxx=train_x.reshape(1,train_x.shape[0],train_x.shape[1])
			x=pad_sequences(xxx,seq_length,padding='post')
			X_train=np.concatenate((X_train,x),axis=0)
			train_x2 = np.loadtxt(codeEncodefile, dtype=np.float32)
			xxx2=train_x2.reshape(1,train_x2.shape[0])
			X_train2=np.concatenate((X_train2,xxx2),axis=0)
			labelFile=label_path+chain_name+'/'+chain_name+'.labelvecNEW3'
			train_y=np.loadtxt(labelFile, dtype=np.int64)
			tt=train_y.reshape(1,train_y.shape[0],train_y.shape[1])
			Y_train=np.concatenate((Y_train,tt),axis=0)
			
	L_file.close()

	domaindata_path='/home/shiqiang/feature_extraction/DeepDomFeatures/train/'
	label_path='/home/shiqiang/feature_extraction/DeepDomFeatures/train/'
	L_file = open(testlistpath, 'r')
	k=0
	for line in L_file:
		if line.strip() == "":
			continue
		k=k+1
		if k==1:
			chain_name = line.split()[0]
			featurefile=domaindata_path+chain_name+'/'+chain_name+'.fea'
			train_x = np.loadtxt(featurefile, dtype=np.float32)
			xxx=train_x.reshape(1,train_x.shape[0],train_x.shape[1])
			x=pad_sequences(xxx,seq_length,padding='post')
			X_test=x
			codeEncodefile=domaindata_path+chain_name+'/'+chain_name+'.txt_encode'
			train_x2=np.loadtxt(codeEncodefile, dtype=np.float32)
			xxx2=train_x2.reshape(1,train_x2.shape[0])
			X_test2 =xxx2
			labelFile=label_path+chain_name+'/'+chain_name+'.labelvecNEW3'
			train_y=np.loadtxt(labelFile, dtype=np.int64)
			y=train_y.reshape(1,train_y.shape[0],train_y.shape[1])
			Y_test=y
		else:
			chain_name = line.split()[0]
			featurefile=domaindata_path+chain_name+'/'+chain_name+'.fea'
			train_x = np.loadtxt(featurefile, dtype=np.float32)
			xxx=train_x.reshape(1,train_x.shape[0],train_x.shape[1])
			x=pad_sequences(xxx,seq_length,padding='post')
			X_test=np.concatenate((X_test,x),axis=0)
			codeEncodefile=domaindata_path+chain_name+'/'+chain_name+'.txt_encode'
			train_x2 = np.loadtxt(codeEncodefile, dtype=np.float32)
			xxx2=train_x2.reshape(1,train_x2.shape[0])
			
			X_test2=np.concatenate((X_test2,xxx2),axis=0)
			labelFile=label_path+chain_name+'/'+chain_name+'.labelvecNEW3'
			train_y=np.loadtxt(labelFile, dtype=np.int64) 
			tt=train_y.reshape(1,train_y.shape[0],train_y.shape[1])
			Y_test=np.concatenate((Y_test,tt),axis=0)
			
	L_file.close()
	return X_train, X_train2, Y_train, X_test, X_test2, Y_test;



def model(X_train, X_train2, Y_train, X_test, X_test2, Y_test,rootpath):
	import os
	import keras
	from keras.models import Sequential, model_from_json
	from keras.layers import Dense, Dropout, Flatten,GRU,LSTM,TimeDistributed
	from keras.preprocessing.sequence import pad_sequences 
	from keras.layers import Convolution2D, MaxPooling2D,Input, Convolution1D
	from keras.layers.core  import RepeatVector
	from keras.models import Model
	from keras import backend as K
	from keras.layers.merge import concatenate,add
	from keras.layers.embeddings import Embedding
	from keras.optimizers import SGD,RMSprop, Adam
	from keras import regularizers
	from sklearn.metrics import confusion_matrix
	from sklearn.utils import class_weight
	from keras.utils import multi_gpu_model
	from keras.callbacks import EarlyStopping, ModelCheckpoint
	from keras.layers.normalization import BatchNormalization
	import numpy as np
	

	left_input=Input(shape=(700,),dtype='float32')
	left_branch=Embedding(input_dim=21,output_dim=50)(left_input)
	
	print(left_branch.shape)
	right_input=Input(shape=(700,25),dtype='float32')
	x=concatenate([left_branch, right_input],axis=2)

	x=BatchNormalization()(x)
	CNN_model_1=Convolution1D(128, 11, activation='relu',padding='same',input_shape=(700,75),kernel_regularizer=regularizers.l1_l2(0.001))(x)
	CNN_model_2=Convolution1D(128, 15, activation='relu', padding='same',input_shape=(700,75),kernel_regularizer=regularizers.l1_l2(0.001))(x)
	CNN_model_3=Convolution1D(128, 21, activation='relu', padding='same',input_shape=(700,75),kernel_regularizer=regularizers.l1_l2(0.001))(x)
	
	
	CNN_Merged_Model= keras.layers.concatenate([CNN_model_1, CNN_model_2,CNN_model_3])
	mergeDim=CNN_Merged_Model.shape[2]
	CNN_Merged_Model=BatchNormalization()(CNN_Merged_Model)
	hidden_units=300
	left= GRU(output_dim =hidden_units, return_sequences=True, input_shape=(700, mergeDim),dropout=0.5)(CNN_Merged_Model)
	right=GRU(output_dim =hidden_units, return_sequences=True, input_shape=(700, mergeDim),go_backwards=True,dropout=0.5)(CNN_Merged_Model)
	GRU_model=add([left,right])

	GRU_model=BatchNormalization()(GRU_model)
	left=GRU(units=hidden_units, recurrent_activation='hard_sigmoid',use_bias=True,kernel_initializer='glorot_uniform',bias_initializer='zeros', activation='tanh',return_sequences=True,dropout=0.5)(GRU_model)
	right=GRU(units=hidden_units, recurrent_activation='hard_sigmoid',use_bias=True,kernel_initializer='glorot_uniform',bias_initializer='zeros', activation='tanh',return_sequences=True,go_backwards=True,dropout=0.5)(GRU_model)
	GRU_model=add([left,right])

	GRU_model=BatchNormalization()(GRU_model)
	left=GRU(units=hidden_units, recurrent_activation='hard_sigmoid',use_bias=True,kernel_initializer='glorot_uniform',bias_initializer='zeros', activation='tanh',return_sequences=True,dropout=0.5)(GRU_model)
	right=GRU(units=hidden_units, recurrent_activation='hard_sigmoid',use_bias=True,kernel_initializer='glorot_uniform',bias_initializer='zeros', activation='tanh',return_sequences=True,go_backwards=True,dropout=0.5)(GRU_model)
	GRU_model=add([left,right])
	GRU_model=Dropout(0.5)(GRU_model)
	
	mainModel= concatenate([GRU_model,CNN_Merged_Model],axis=-1)
	mainModel=BatchNormalization()(mainModel)
	mainModel=TimeDistributed(Dense(512,activation='relu',kernel_regularizer=regularizers.l1_l2(0.001)))(mainModel)
	mainModel=TimeDistributed(Dense(512,activation='relu',kernel_regularizer=regularizers.l1_l2(0.001)))(mainModel)
	
	mainModel=BatchNormalization()(mainModel)
	mainModel=TimeDistributed(Dense(3, activation='softmax'))(mainModel)
	
	model = Model(inputs=[left_input,right_input], outputs=mainModel)
	
	nsamples=X_train.shape[0]
	sample_weights = np.zeros((nsamples,700))
	yclass=np.argmax(Y_train,-1)
	for i in range(0,nsamples):
		for j in range(0,700):
			label=yclass[i,j]
			if label==0:
				sample_weights[i,j]=50
			if label==1:
				sample_weights[i,j]=2
			if label==2:
				sample_weights[i,j]=1
	savepath=rootpath+'weightsnew-{epoch:02d}-{val_loss:.2f}.hdf5'	
	#early_stopping = EarlyStopping(monitor='val_loss',min_delta=0.00002,mode='min', patience=30)
	saveBestModel=ModelCheckpoint(savepath, monitor='val_loss', verbose=1, save_best_only=False,save_weights_only=True, mode='min')
	#cb = [early_stopping,saveBestModel]
	cb = [saveBestModel]
	
	parall_model =multi_gpu_model(model, gpus=2)
	parall_model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08),metrics=['accuracy',f1_score,precision,sensitivity],sample_weight_mode='temporal')
	parall_model.fit([X_train2,X_train], Y_train, batch_size=256, nb_epoch=180, validation_data=([X_test2,X_test],Y_test),callbacks=cb,sample_weight=sample_weights,verbose=1)
	
	return { 'status': STATUS_OK, 'model': parall_model};
	


if __name__ == '__main__':
	for j in range(1,6):
		rootpath='/home/shiqiang/DeepDom/DeepModelTrain/'+'cv'+str(j)+'/'
		print(rootpath)
		trainlist=rootpath+'trainlist.list'
		testlist=rootpath+'testlist.list'
		X_train, X_train2, Y_train, X_test, X_test2, Y_test = data(trainlist,testlist)
		model(X_train, X_train2, Y_train, X_test, X_test2, Y_test,rootpath)
		print("trainning is end.")