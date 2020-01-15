import os
os.environ["CUDA_VISIBLE_DEVICES"]=""
from keras.utils import multi_gpu_model
from keras.models import load_model
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

def data(featurefile,codeEncodefile):
	from keras.utils import np_utils
	import numpy as np
	np.random.seed(1337) 
	from keras.preprocessing.sequence import pad_sequences
	seq_length=700
	train_x = np.loadtxt(featurefile, dtype=np.float32)
	xxx=train_x.reshape(1,train_x.shape[0],train_x.shape[1])
	x=pad_sequences(xxx,seq_length,padding='post')
	X_test=x
	
	train_x2=np.loadtxt(codeEncodefile, dtype=np.float32)
	xxx2=train_x2.reshape(1,train_x2.shape[0])
	X_test2 =xxx2
	return X_test, X_test2;
import sys
featurefile=sys.argv[1]
codefile=sys.argv[2]
X_test, X_test2= data(featurefile,codefile)

path="/home/shiqiang/DeepDom/DeepModelTrain/cv4"
modelPath=os.path.join(path,'weightsnew3-64-0.64.hdf5')
cModel=load_model(modelPath)
feature_network = Model([cModel.layers[0].input,cModel.layers[2].input],cModel.layers[25].output)
feature = feature_network.predict([X_test2,X_test])
for i in range(0,feature.shape[0]):
	xfeatures=feature[i,:,:]
	np.savetxt(sys.argv[3],xfeatures)