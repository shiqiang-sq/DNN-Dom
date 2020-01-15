from time import time
import matplotlib.pyplot as plt
import numpy as np

from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.externals import joblib
from imblearn.under_sampling import RandomUnderSampler
import sys
from imblearn.combine import SMOTETomek
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix

from imblearn.datasets import fetch_datasets
from imblearn.ensemble import BalancedBaggingClassifier

from imblearn.metrics import classification_report_imbalanced

labelPath=sys.argv[2]
for x in range(1,2):
		featurepath=sys.argv[1]
		xx_test=np.loadtxt(featurepath, dtype=np.float32)		
		modelPath='/home/shiqiang/DeepDom/DeepModelTrain/cv4/'+'balanced_RF-100.model'
		balanced_RF=joblib.load(modelPath)
		y_pred = balanced_RF.predict(xx_test)
		yy_probability=balanced_RF.predict_proba(xx_test)

		test_label=np.loadtxt(labelPath, dtype=np.float32)
		seqLength=test_label.shape[0]
		if seqLength>700:
			seqLength=700
		yy_prob=yy_probability[0:seqLength,:]
		path=sys.argv[3]
		np.savetxt(path,yy_prob,fmt='%0.8f')
