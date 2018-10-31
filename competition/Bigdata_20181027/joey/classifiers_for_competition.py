import os
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split,StratifiedKFold
from sklearn import neighbors
from sklearn import metrics
from sklearn import svm
from sklearn import linear_model
from scipy import stats
from yujen.ResultFile import ResultFile

fft = np.load("../yujen/A/saved/fft_all.npy")
indexes = np.load("../yujen/A/saved/select_freq.npy")
train_y = np.load("../yujen/A/saved/labels.npy")
final_o = np.load("../yujen/A/saved/fft_test.npy")
file_names = np.load("../yujen/A/saved/test_files.npy")

def main():
    
    # select frequencies and make train test subset
    train_x = np.log(fft[:,indexes])
    std = np.std(train_x, axis=0)
    mean = np.mean(train_x, axis=0)
    train_x = (train_x - mean)/std
    final_test = np.log(final_o[:,indexes])
    final_test = (final_test - mean)/std

    KFold = StratifiedKFold(n_splits=10,random_state=103, shuffle=True)
    all_svm = []
    val_scores = []
    times = 1 
    all_prediction = []
    for train_idx, val_idx in KFold.split(train_x, train_y):
        print("Split----->",times)
        times += 1
        #SVM
        svc = svm.SVC(C=3.5, kernel="rbf", gamma=7.56e-05,verbose=False)
        svm_clf = svc.fit(train_x[train_idx], train_y[train_idx])
        #Logistic
        mul_lr = linear_model.LogisticRegression(penalty='l2',C=0.01,max_iter=1000,multi_class='multinomial', solver='saga').fit(train_x[train_idx], train_y[train_idx])
        #K neighbors
        KNC = neighbors.KNeighborsClassifier(n_neighbors=1)
        knc_clf = KNC.fit(train_x[train_idx], train_y[train_idx])
        
        #Result for now
        print("Multinomial Logistic regression Train Accuracy :: ", metrics.accuracy_score(train_y[train_idx], mul_lr.predict(train_x[train_idx])))
        print("Multinomial Logistic regression Test Accuracy :: ", metrics.accuracy_score(train_y[val_idx], mul_lr.predict(train_x[val_idx])))
        print("Test Label:: ",train_y[val_idx])
        print("Prediction:: ",mul_lr.predict(train_x[val_idx]) )
        print("Wrong Index:: ", np.where(train_y[val_idx] != mul_lr.predict(train_x[val_idx])))

        print("==================================================================")
        print("Support Vector Machine Train Accuracy :: ", metrics.accuracy_score(train_y[train_idx], svm_clf.predict(train_x[train_idx])))
        print("Support Vector Machine Test Accuracy :: ", metrics.accuracy_score(train_y[val_idx], svm_clf.predict(train_x[val_idx])))
        print("Support Vector Machine Test Label:: ",train_y[val_idx])
        print("Support Vector Machine Prediction:: ",svm_clf.predict(train_x[val_idx]))
        print("Wrong Index:: ", np.where(train_y[val_idx] != svm_clf.predict(train_x[val_idx])))
        print("==================================================================")
        print("K Neighbors Classifier Train Accuracy :: ", metrics.accuracy_score(train_y[train_idx], knc_clf.predict(train_x[train_idx])))
        print("K Neighbors Classifier Test Accuracy :: ", metrics.accuracy_score(train_y[val_idx], knc_clf.predict(train_x[val_idx])))
        print("K Neighbors Classifier Test Label:: ",train_y[val_idx])
        print("K Neighbors Classifier Prediction:: ",knc_clf.predict(train_x[val_idx]))
        print("Wrong Index:: ", np.where(train_y[val_idx] != knc_clf.predict(train_x[val_idx])))

        mul_label = mul_lr.predict(train_x[val_idx])
        svm_label = svm_clf.predict(train_x[val_idx])
        knc_label = knc_clf.predict(train_x[val_idx])

        a = np.stack([knc_label,mul_label,svm_label])#Combine array
        
        mul_final_label = mul_lr.predict(final_test[:])
        svm_final_label = svm_clf.predict(final_test[:])
        knc_final_label = knc_clf.predict(final_test[:])

        final = np.stack([mul_final_label,svm_final_label,knc_final_label])

        for i in range(3):
            all_prediction.append(final[i])
        
        m = stats.mode(a) # Mode&Count
        predicts=m[0] #Get Mode array

        print("==================================================================")
        print("Mode:: ",predicts)
        accuracy = np.equal(predicts, train_y[val_idx]).astype(np.int32)
        accuracy = np.mean(accuracy)
        print("Total Accuracy:: ",accuracy)
        print("Wrong Index:: ", np.where(train_y[val_idx] != predicts))
        val_scores.append(accuracy)
        print("==================================================================")


    print(val_scores)
    accuracy = np.mean(val_scores)
    print(accuracy)
    
    
    mode_for_prediction = stats.mode(all_prediction)
    final_prediction = mode_for_prediction[0]
    print(final_prediction)
    predict_str = []
    for p in final_prediction[0]:
        predict_str.append('type' + str(p))

    print(predict_str)
    result_file = pd.read_csv('/projectA/sample.csv')
    for i,file in enumerate(file_names):
        select = result_file.iloc[:,1] == file
        result_file.loc[select,'type'] = predict_str[i]
    result_file.to_csv('1_125501_A_test.csv', index=False)
    

if __name__ == "__main__":
    main()
