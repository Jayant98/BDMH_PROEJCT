import pandas as pd
import csv
import math
import numpy as np
import matplotlib.pyplot as plt
import scipy
from sklearn.utils import resample
from scipy.sparse import coo_matrix
from sklearn import preprocessing
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve,auc,accuracy_score
from random import sample
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score, log_loss
from numpy import argmax
from sklearn.linear_model import LogisticRegression
import statistics 
from sklearn.metrics import confusion_matrix



# Loading the dataset
dataset = (pd.read_csv('dataR2.csv'))

# DATA SHUFFLING CODE
# TO BE DONE ONCE 
'''dataset_shuffled = dataset.sample(frac=1)

dataset_shuffled.to_csv(r'BDMH_ASSIGNMENT_SHUFFLED_DATA.csv',index = False, header=True)'''

# Extracting all features from the dataset
features_in = dataset.drop(['Classification'],axis=1)

# Extracting Labels from the dataset
Labels_in = dataset['Classification']

# Relabling dataset into generic 1-0 format
Labels_in=Labels_in.replace(1,0)
Labels_in=Labels_in.replace(2,1)


# Extracting names of all features into a list
lis_of_features = features_in.columns.values.tolist()


# UNIVARIATE ANALYSIS
print('STARTING UNIVARIATE ANALYSIS')
print("="*60)
print("="*60)
print()

# Select each feature one by one form the list of features
for i in lis_of_features:

	# Calculate the ROC Curve for each feature by comparing the results with the Labels
	specificity_min1, sensitivity, thresh = roc_curve(Labels_in, features_in[i])

	# Calculate the AUC of the ROC 
	roc_auc = auc(specificity_min1, sensitivity)

	# Find the Youden index that maximizes the value J defined below 
	J = sensitivity - specificity_min1
	ix = argmax(J)

	n_iterations = 2000

	# Get the sensitivity and specificity from the ROC that maximizes J
	
	print(sensitivity[ix])
	print(1-specificity_min1[ix])
	
	
	auc_lis = []
	print(i)

	# For each AUC of ROC we calculate the 95% Confidence Interval by finding multiple value of the same over random subsequence of entire data set
	for j in range(n_iterations):
		
		# list to hold indexes of samples in the dataset
		rand_lis = []
		# populating the list
		for k in range(len(Labels_in)):
			rand_lis.append(k)

		# selecting random 50 % of indexes from the list
		sample_indexes = resample(rand_lis, n_samples=(int(.5*len(Labels_in))), stratify=rand_lis)

		
		# selecting correspondind samples and their labels
		label_test = Labels_in[sample_indexes]
		label_features = features_in[i].iloc[sample_indexes]


		# calculating the ROC followed by AUC of the same
		specificity_min1, sensitivity, thresh = roc_curve(label_test, label_features)
		roc_auc = auc(specificity_min1, sensitivity)

		# Adding the value of AUC to the list
		auc_lis.append(roc_auc)

	# Mathematical Formula to get 95% Confidence interval value for the list of AUC values
	alpha = 0.95
	p = ((1.0-alpha)/2.0) * 100
	lower = max(0.0, np.percentile(auc_lis, p))
	p = (alpha+((1.0-alpha)/2.0)) * 100
	upper = min(1.0, np.percentile(auc_lis, p)) 
	print("Confidence interval for the score: [{:0.2f} - {:0.2f}]".format(lower, upper))

	print()


	# Plotting the ROC Curves for each Feature
	plt.figure()
	plt.plot(specificity_min1, sensitivity, color='darkorange', lw=4, label='ROC curve (area = %0.2f)' % roc_auc)
	plt.plot([0, 1], [0, 1], color='darkgreen', lw=4, linestyle='--')
	plt.xlabel('1 - Specificity')
	plt.ylabel('True Positive Rate')
	plt.title("Univariate analysis for variable "+  i)
	plt.legend(loc="lower right")
	plt.show()


# Multivariate Analysis
print("X"*60)
print("X"*60)
print()
print('STARTING MULTIVARIATE ANALYSIS')
print("="*60)
print("="*60)
print()
	

gini_mod= RandomForestClassifier()
gini_mod.fit(features_in,Labels_in);
gini_coeff = gini_mod.feature_importances_
std = np.std([tree.feature_importances_ for tree in gini_mod.estimators_],
             axis=0)
indices = np.argsort(gini_coeff)[::-1]

# Print the feature ranking
print("Feature ranking in decreasing order:")

for f in range(9):
	print("%d. feature is  %s (%f)" % (f + 1, lis_of_features[indices[f]], gini_coeff[indices[f]]))

# Creating a list of classifier used 
classifiers = [
NuSVC(probability=True),
LogisticRegression(max_iter=600),
RandomForestClassifier()
]



#V1 = Glucose, V2 = Resistin, V3 = Age, V4 = BMI - body mass index, V5 = HOMA - homeostasis model assessment for insulin resistance, V6 = Leptin, V7 = Insulin, V8 = Adiponectin, V9 = MCP-1

# Breaking down features into multiple combinations

lis_V1_V2 = ['Glucose', 'Resistin']
lis_V1_V3 = ['Glucose', 'Resistin','Age']
lis_V1_V4 = ['Glucose', 'Resistin','Age','BMI']

lis_V1_V5 = ['Glucose', 'Resistin','Age','BMI','HOMA']
lis_V1_V6 = ['Glucose', 'Resistin','Age','BMI','HOMA','Leptin']
lis_all = [] # This list will be used for all features

# list to hold all the above combinations
lis_cases = [lis_V1_V2,lis_V1_V3,lis_V1_V4,lis_V1_V5,lis_V1_V6,lis_all];

# Main list to hold results for each of the different models used for each of the configurations of Features 
# Structure of This list : Head_lis[ index of classifier used ] [ configuration used ] [ 0 (auc value) 1(sensitivity value ) 2(specificity value) ]
Head_lis = [ [ [ []for num_params in range(3) ]for num_cases in range(6) ] for num_models in range(3) ]

lis_curve = [ [ [ []for num_params in range(1) ]for num_cases in range(6) ] for num_models in range(3) ]
	
# running 500 models (of 3 types) each with 6 configurations	
for runs in range(500):

	print(runs)
	#list for extracting random index 
	rand_lis_control = []
	rand_lis_patient = []
	for i in range(0,52):
		rand_lis_control.append(i)

	for i in range(52,116):
		rand_lis_patient.append(i)

	#getting random indexes
	rand_samples_control = resample(rand_lis_control, n_samples=36,random_state=runs)
	rand_samples_patient = resample(rand_lis_patient, n_samples=45,random_state=runs)

	#extracting training data
	train_control = dataset.iloc[rand_samples_control]
	train_patient = dataset.iloc[rand_samples_patient]


	# Extracting testing data from data by substracting training samples from the main dataset 
	test_control = (dataset[0:52]).merge(train_control, how = 'outer' ,indicator=True).loc[lambda x : x['_merge']=='left_only']
	test_patient = (dataset[52:]).merge(train_patient, how = 'outer' ,indicator=True).loc[lambda x : x['_merge']=='left_only']

	# creating a combined list of training data having samples for both patient and controls 
	train_list = [train_patient,train_control]
	test_list= [test_patient,test_control]

	train_data = (pd.concat(train_list))
	test_data = (pd.concat(test_list))

	# Extracting all features and labels seperately from the training data

	train_labels = train_data['Classification']
	train_features = train_data.drop(['Classification'],axis=1)
	
	# Extracting all features and labels seperately from the testing data

	test_labels = test_data['Classification']
	test_features = test_data.drop(['Classification','_merge'],axis=1)


	train_labels=train_labels.replace(1,0)
	train_labels=train_labels.replace(2,1)

	test_labels=test_labels.replace(1,0)
	test_labels=test_labels.replace(2,1)

	# variable to identify the configuration of features used
	case_ind =0
	
	for cases in lis_cases:
		classifier_ind = 0
		for clf in classifiers:

			# selecting features as per the current configuration
			train_sel_features = train_features[cases];
			test_sel_features = test_features[cases];

			# to check if we have to train model using all features
			if (len(cases)==0):
				train_sel_features = train_features;
				test_sel_features = test_features;	

			# training the model
			clf.fit(train_sel_features, train_labels)
	
			# generating prediction from models
			train_predictions = (clf.predict_proba(test_sel_features))[:,1]
			
			# getting ROC of model

			specificity_min1, sensitivity, thresh = roc_curve(test_labels, train_predictions)
			roc_auc = roc_auc_score(test_labels, train_predictions)
			J = ((sensitivity - specificity_min1))
			ix = argmax(J)

			lis_temp = [specificity_min1, sensitivity, thresh]
			lis_curve[classifier_ind][case_ind][0].append(lis_temp)

			Head_lis[classifier_ind][case_ind][0].append(roc_auc)
			Head_lis[classifier_ind][case_ind][1].append(sensitivity[ix])
			Head_lis[classifier_ind][case_ind][2].append(1-specificity_min1[ix])
			classifier_ind=classifier_ind+1;

		case_ind=case_ind+1
			



## AUC CI
max_per_case = [[]for num_params in range(3) ]
max_auc_case = [[]for num_params in range(3) ]

min_per_case = [[]for num_params in range(3)]
min_auc_case = [[]for num_params in range(3) ]
for run_classi in range(3):

	for run_cases in range(6):
		print(run_classi)
		# for each model and its corresponding feature configurations we calculate 95% CI for AUC sensitivity and specificity
		auc_lis = Head_lis[run_classi][run_cases][0]
		sens_lis = Head_lis[run_classi][run_cases][1]
		spec_lis = Head_lis[run_classi][run_cases][2]

		curve = lis_curve[run_classi][run_cases][0]

		print(classifiers[run_classi].__class__.__name__)
		print()
		print(lis_cases[run_cases])
		print()

		auc_pd = pd.DataFrame(auc_lis)
		mean_auc = 	auc_pd.mean()
		std_auc = auc_pd.std()
		qnorm = scipy.stats.norm.ppf(q=.975)
		sqrt_sd_auc = (std_auc/500)
		sqrt_sd_auc = math.sqrt(sqrt_sd_auc)

		lower_auc = mean_auc - (qnorm*sqrt_sd_auc) 
		upper_auc = mean_auc + (qnorm*sqrt_sd_auc)
		


		## Sensitivity CI

		sens_pd = pd.DataFrame(sens_lis)
		mean_sens = 	sens_pd.mean()
		std_sens = sens_pd.std()
		qnorm = scipy.stats.norm.ppf(q=.975)
		sqrt_sd_sens = math.sqrt((std_sens/500))

		lower_sens = mean_sens - (qnorm*sqrt_sd_sens) 
		upper_sens = mean_sens + (qnorm*sqrt_sd_sens)

		## Specificity CI

		spec_pd = pd.DataFrame(spec_lis)
		mean_spec = 	spec_pd.mean()
		std_spec = spec_pd.std()
		qnorm = scipy.stats.norm.ppf(q=.975)
		sqrt_sd_spec = math.sqrt((std_spec/500))

		lower_spec = mean_spec - (qnorm*sqrt_sd_spec) 
		upper_spec = mean_spec + (qnorm*sqrt_sd_spec)


		print(lower_auc[0])
		print(upper_auc[0])
		print()

		print(lower_sens[0])
		print(upper_sens[0])
		print()

		print(lower_spec[0])
		print(upper_spec[0])
		print()
		

		ind_max = np.argmax(auc_lis)
		max_auc_case[run_classi].append(auc_lis[ind_max])
		max_per_case[run_classi].append(curve[ind_max])

		ind_min = np.argmin(auc_lis)
		min_auc_case[run_classi].append(auc_lis[ind_min])
		min_per_case[run_classi].append(curve[ind_min])


for run_classi in range(3):


	ind_fin_max = np.argmax(max_auc_case[run_classi])
	main_lis_max = max_per_case[run_classi][ind_fin_max]

	ind_fin_min = np.argmin(min_auc_case[run_classi])

	main_lis_min = min_per_case[run_classi][ind_fin_min]
	plt.figure()
	plt.plot(main_lis_max[0], main_lis_max[1], color='darkorange', lw=4, label='ROC curve (area max = %0.2f)' % max_auc_case[run_classi][ind_fin_max])
	plt.plot(main_lis_min[0], main_lis_min[1], color='darkblue', lw=4, label='ROC curve (area min= %0.2f)' % min_auc_case[run_classi][ind_fin_min])
	plt.plot([0, 1], [0, 1], color='darkgreen', lw=4, linestyle='--')
	plt.xlabel('1 - Specificity')
	plt.ylabel('True Positive Rate')
	plt.title("ROC Curves for "+(str)(classifiers[run_classi].__class__.__name__))
	plt.legend(loc="lower right")
	plt.show()

'''
auc_pd = pd.DataFrame(auc_lis)
mean_auc = 	auc_pd.mean()
std_auc = auc_pd.std()
qnorm = scipy.stats.norm.ppf(q=.975)
sqrt_sd_auc = (std_auc/sqrt_500)

lower_auc = mean_auc - (qnorm*sqrt_sd_auc) 
upper_auc = mean_auc + (qnorm*sqrt_sd_auc)

## Sensitivity CI

sens_pd = pd.DataFrame(sens_lis)
mean_sens = 	sens_pd.mean()
std_sens = sens_pd.std()
qnorm = scipy.stats.norm.ppf(q=.975)
sqrt_sd_sens = (std_sens/sqrt_500)

lower_sens = mean_sens - (qnorm*sqrt_sd_sens) 
upper_sens = mean_sens + (qnorm*sqrt_sd_sens)

## Specificity CI

spec_pd = pd.DataFrame(spec_lis)
mean_spec = 	spec_pd.mean()
std_spec = spec_pd.std()
qnorm = scipy.stats.norm.ppf(q=.975)
sqrt_sd_spec = (std_spec/sqrt_500)

lower_spec = mean_spec - (qnorm*sqrt_sd_spec) 
upper_spec = mean_spec + (qnorm*sqrt_sd_spec)

print()
print(i)
print()

print(lower_auc)
print(upper_auc)
print()

print(lower_sens)
print(upper_sens)
print()

print(lower_spec)
print(upper_spec)
print()

#pd.DataFrame(train_predictions).to_csv(r'File Name_nonprob.csv',index = False, header=True)
#train_predictions = clf.predict_proba(x_test_new)


#pd.DataFrame(train_predictions).to_csv(r'File Name.csv',index = False, header=True)


#log_entry = pd.DataFrame([[name, acc*100, ll]], columns=log_cols)
#log = log.append(log_entry)'''



