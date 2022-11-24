#PLEASE WRITE THE GITHUB URL BELOW!
# https://github.com/taegyun12

import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

def load_dataset(dataset_path):
	#To-Do: Implement this function
	data_df=pd.read_csv(dataset_path) #data 받아오기
	return data_df

def dataset_stat(dataset_df):	
	#To-Do: Implement this function
	num_feature=dataset_df.shape[1]-1 #feature의 개수
	num_class0=dataset_df.groupby("target").size()[0] #label이 0인 것의 개수
	num_class1=dataset_df.groupby("target").size()[1] #label이 1인 것의 개수
	return num_feature,num_class0,num_class1

def split_dataset(dataset_df, testset_size):
	#To-Do: Implement this function
	dataset_data=dataset_df.iloc[:, :dataset_df.shape[1]-1] #data slicing -> data
	dataset_target=dataset_df.iloc[:,dataset_df.shape[1]-1] #data slicing -> labels
	X_train,X_test,y_train,y_test=train_test_split(dataset_data,dataset_target,test_size=testset_size) #data와 labels를 각각 test와 train용도로 나눔
	return X_train,X_test,y_train,y_test

def decision_tree_train_test(x_train, x_test, y_train, y_test):
	#To-Do: Implement this function
	dt_cls=DecisionTreeClassifier() #classifier 객체 생성
	dt_cls.fit(x_train,y_train) #객체 학습
	dt_pred=dt_cls.predict(x_test) #예상되는 결과값 도출
	acc_score=accuracy_score(y_test,dt_pred) ## 평가
	pre_score=precision_score(y_test,dt_pred)
	rec_score=recall_score(y_test,dt_pred)
	return acc_score,pre_score,rec_score

def random_forest_train_test(x_train, x_test, y_train, y_test):
	#To-Do: Implement this function
	rf_cls = RandomForestClassifier() #classifier 객체 생성
	rf_cls.fit(x_train, y_train) #객체 학습
	rf_pred = rf_cls.predict(x_test) #예상되는 결과값 도출
	acc_score = accuracy_score(y_test, rf_pred) ##평가
	pre_score = precision_score(y_test, rf_pred)
	rec_score = recall_score(y_test, rf_pred)
	return acc_score, pre_score, rec_score

def svm_train_test(x_train, x_test, y_train, y_test):
	#To-Do: Implement this function
	svm_pipe=make_pipeline( #파이프를 만들어 data를 scaling한 후 classifier 객체 생성
		StandardScaler(),
		SVC()
	)
	svm_pipe.fit(x_train, y_train) #객체 학습
	svm_pred=svm_pipe.predict(x_test) #예상되는 결과값 도출
	acc_score = accuracy_score(y_test, svm_pred) ##평가
	pre_score = precision_score(y_test, svm_pred)
	rec_score = recall_score(y_test, svm_pred)
	return acc_score, pre_score, rec_score

def print_performances(acc, prec, recall):
	#Do not modify this function!
	print ("Accuracy: ", acc)
	print ("Precision: ", prec)
	print ("Recall: ", recall)

if __name__ == '__main__':
	#Do not modify the main script!
	data_path = sys.argv[1]
	data_df = load_dataset(data_path)

	n_feats, n_class0, n_class1 = dataset_stat(data_df)
	print ("Number of features: ", n_feats)
	print ("Number of class 0 data entries: ", n_class0)
	print ("Number of class 1 data entries: ", n_class1)

	print ("\nSplitting the dataset with the test size of ", float(sys.argv[2]))
	x_train, x_test, y_train, y_test = split_dataset(data_df, float(sys.argv[2]))

	acc, prec, recall = decision_tree_train_test(x_train, x_test, y_train, y_test)
	print ("\nDecision Tree Performances")
	print_performances(acc, prec, recall)

	acc, prec, recall = random_forest_train_test(x_train, x_test, y_train, y_test)
	print ("\nRandom Forest Performances")
	print_performances(acc, prec, recall)

	acc, prec, recall = svm_train_test(x_train, x_test, y_train, y_test)
	print ("\nSVM Performances")
	print_performances(acc, prec, recall)