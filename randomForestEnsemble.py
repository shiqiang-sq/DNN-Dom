from time import time
import matplotlib.pyplot as plt
import numpy as np

from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.externals import joblib
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTETomek
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix

from imblearn.datasets import fetch_datasets
from imblearn.ensemble import BalancedBaggingClassifier

from imblearn.metrics import classification_report_imbalanced

rootpath='/home/shiqiang/DeepDom/DeepModelTrain/cv4/'

featurepath=rootpath+'xx_train'+'.txt'
xx_train=np.loadtxt(featurepath, dtype=np.float32)
labelpath=rootpath+'yy_train'+'.txt'
yy_train=np.loadtxt(labelpath, dtype=np.float32)

featurepath=rootpath+'xx_test'+'.txt'
xx_test=np.loadtxt(featurepath, dtype=np.float32)
labelpath=rootpath+'yy_test'+'.txt'
yy_test=np.loadtxt(labelpath, dtype=np.float32)

print("loading data is end.")


num_emtimators=100
num_jobs=48
num_features=xx_train.shape[1]

balanced_RF = BalancedBaggingClassifier(
    base_estimator=DecisionTreeClassifier(max_features='auto'),
    random_state=0,n_estimators=num_emtimators, replacement=True, n_jobs=num_jobs)
balanced_RF.fit(xx_train, yy_train)

y_pred = balanced_RF.predict(xx_test)

print('testdataset-BalancedBaggingClassifier:')
print(classification_report_imbalanced(yy_test, y_pred))

y_pred = balanced_RF.predict(xx_train)
print('traindataset-BalancedBaggingClassifier:')
print(classification_report_imbalanced(yy_train, y_pred))


yy_probability=balanced_RF.predict_proba(xx_test)
listFilePath_test=rootpath+'testlist.list'
L_file = open(listFilePath_test, 'r')
domaindata_path='/home/shiqiang/feature_extraction/DeepDomFeatures/train/'
k=0
startLen=0
for line in L_file:
	if line.strip()=="":
		continue
	chain_name=line.split()[0]
	labelPath=domaindata_path+chain_name+'/'+chain_name+'new.label'
	test_label=np.loadtxt(labelPath, dtype=np.int64)
	seqLength=test_label.shape[0]
	if seqLength>700:
		seqLength=700
	endLen=startLen+seqLength
	yy_prob=yy_probability[startLen:endLen,:]
	path=rootpath+'RFscoreTest2/rfscore'+str(k)+'.txt'
	np.savetxt(path,yy_prob,fmt='%0.8f')
	startLen=startLen+seqLength
	k=k+1



yy_probability=balanced_RF.predict_proba(xx_train)
listFilePath_test=rootpath+'trainlist.list'
L_file = open(listFilePath_test, 'r')
domaindata_path='/home/shiqiang/feature_extraction/DeepDomFeatures/train/'
k=0
startLen=0
for line in L_file:
	if line.strip()=="":
		continue
	chain_name=line.split()[0]
	labelPath=domaindata_path+chain_name+'/'+chain_name+'new.label'
	test_label=np.loadtxt(labelPath, dtype=np.int64)
	seqLength=test_label.shape[0]
	if seqLength>700:
		seqLength=700
	endLen=startLen+seqLength
	yy_prob=yy_probability[startLen:endLen,:]
	path=rootpath+'RFscoreTrain2/rfscore'+str(k)+'.txt'
	np.savetxt(path,yy_prob,fmt='%0.8f')
	startLen=startLen+seqLength
	k=k+1