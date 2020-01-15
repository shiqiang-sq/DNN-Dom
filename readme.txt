1-required tools

   a) SCRATCH-1D_1.0
   b) blast 2.8.1
   c) matlab 2017a
   d) keras 2.2.0
   e) Python 3.6.5
   f) TensorFlow 1.8.0
   g) hyperas: https://github.com/ maxpumperla/hyperas
   h) python packages: sklearn, imblearn

2- Main process of method
   
    the predict_main.sh file contains the main process of DNN-Dom.

3- the trained models
    
   a) deep model file: weightsnew3-64-0.64.hdf5
   b) predicted model file: balanced_RF-100.model

4- the training scripts
   a)script file for training deep learning model: DeepLearningModelTrain.py
   b)script file for training p-BRF model: randomForestEnsemble.py (note:  BalancedBaggingClassifier from imblearn.ensemble can be turned into a balanced random forest by passing a sklearn.tree.DecisionTreeClassifier with max_features=¡¯auto¡¯ as a base estimator.)

**************************************************************************************************************
If there are some questions, please contact us: yanw@hust.edu.cn, zdxue@hust.edu.cn or shiqiang@hust.edu.cn.
**************************************************************************************************************

   