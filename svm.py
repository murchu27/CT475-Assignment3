#imports
import pandas as pd
import numpy as np
import sys, time
from kFoldCrossValidation import kFoldCrossValidation
from confusionMatrix import confusionMatrix

#SVM model for binary classification
############# binarySVM class written by Michael
class binarySVM:
	def __init__(self, features=2):
		self.features = features
		self.normal = np.zeros(self.features)
		self.intercept = 0 

	def train(self, data, target, pos_label, neg_label):
		self.data = data
		self.target = target
		self.pos_label = pos_label
		self.neg_label = neg_label
		#should write some function that scans data for negative label
		#or force user to specify label

		# use find_sv() to determine support vectors
		sv = self.find_sv()

		# sv[i] contains pairs of class A/B of support vectors indices with minimum distance 
		if self.target[sv[0]] == pos_label:
			pos_sv_index = 0
			neg_sv_index = 1
		else:
			pos_sv_index = 1
			neg_sv_index = 0		

		#normal is parallel to line connecting sv pairs 		
		#for now, arbitrarily choose first pair of vectors
		pos_sv = self.data[sv[pos_sv_index]]
		neg_sv = self.data[sv[neg_sv_index]]

		normal_weights = pos_sv-neg_sv

		#explanation of formula
		"""
		use the formula: y_i(dot(w,x_i) + b)
		for pos_sv, becomes dot(w,x_+) + b = 1 ==> b = 1 - dot(w,x_+)
		for neg_sv, becomes dot(w,x_-) + b = -1 ==> b = -1 - dot(w,x_-)
		then by simul. eqns: 1 - dot(w,x_+) = -1 - dot(w,x_-)
		so, 2 = dot(w,x_+) - dot(w,x_-) ; RHS is a scalar multiple of a
		finally, a = 2/(dot(w,x_+) - dot(w,x_-))
		"""

		normal_dot_pos = np.dot(normal_weights,pos_sv)
		normal_dot_neg = np.dot(normal_weights,neg_sv)
		normal_scale = 2/(normal_dot_pos-normal_dot_neg)

		#a is given by normal_scale; finally, w is given by scale*weights
		self.normal = normal_scale * normal_weights

		#now determine intercept using b = y_i - dot(w,x_i), with i referring to one of the sv's
		self.intercept = 1 - np.dot(self.normal,pos_sv)

	def find_sv(self):
		#find minimum distance between points in the two classes
		m = -1
		min_sample = (0,0)
		l = len(self.data)

		#iterate over all samples in the training data
		for i in range(l):
			sample = self.data[i]

			#we compare this sample with all the samples of the opposite label
			for j in range(i+1,l): #start from i+1 so as not to compare to self, or to previously checked samples
				#check if this sample has the opposite label
				if (self.target[j] == self.pos_label and self.target[i] != self.pos_label) or (self.target[j] != self.pos_label and self.target[i] == self.pos_label):
					other = self.data[j]
					
					#check the distance between the samples
					d = np.linalg.norm(sample - other)
					
					#see if less than or equal to previous minimum (if greater than previous minimum, do nothing)
					if (d < m) or (m == -1): #m = -1 at very beginning
						#if less than previous min, scrap the min_samples to this point, and instead add these samples
						#if equal to previous min, change nothing, but do add samples to min_samples    
						m = d
						min_sample = (i,j)

		#now, min_samples_A and min_samples_B contains the samples of class A and B (respectively) to be used as support vectors
		return(min_sample)

	def predict(self, predictdata):
		#explanation of steps
		"""
		predictdata should an array of data points
		u represents a point predictdata
		w represents the normal
		b represents the intercept
		predict using the following formula: 
		dot(w, u) + b < -1 ==> neg_sample
		dot(w, u) + b > 1 ==> pos_sample
		|dot(w, u) + b| < 1 ==> not sure
		assign each resulting label to a predicttarget array
		"""

		#seems to make a lot of false classfications atm
		#come back to this if there is time

		predicttarget = np.array([])
		self.distances = np.array([])
		for u in predictdata:
			f = np.dot(self.normal, u) + self.intercept

			if f >= 1:
				predicttarget = np.append(predicttarget, self.pos_label)
			elif f <= -1:
				predicttarget = np.append(predicttarget, self.neg_label)
			else:
				predicttarget = np.append(predicttarget, "unsure")

			self.distances = np.append(self.distances, np.sign(f)*(f)/np.linalg.norm(self.normal))

		return predicttarget

# SVM model for classification with multiple classes
############# multiclassSVM class written by Michael (except for a few lines of the predict method)
class multiclassSVM:
	def __init__(self, c=2, features=2):
		self.features = features
		self.c = c
		self.models = np.array([])

	def train(self, data, target, classes):
		self.data = data
		self.target = target
		self.classes = classes
		assert(len(classes)==self.c)

		for i in range(self.c):
			#build an OVA model, using each class as the positive model
			b = binarySVM(self.features)
			b.train(self.data,self.target,classes[i],'other')
			self.models = np.append(self.models,b)


	def predict(self, predictdata):
		#explanation of steps
		"""
		predictdata should an array of data points

		u represents a point predictdata
		w represents the normal
		b represents the intercept

		predict using the following formula: 
		dot(w, u) + b < -1 ==> neg_sample
		dot(w, u) + b > 1 ==> pos_sample
		|dot(w, u) + b| < 1 ==> not sure

		assign each resulting label to a predicttarget array
		"""

		predicttargets = []
		predictions = []

		for model in self.models:
			predicttargets.append(model.predict(predictdata))

		############# this loop written by Mark
		for i in range(len(predicttargets[0])):
			prediction = 'N/A'
			maxScore = 0.0
			for k in range(len(predicttargets)):
				#for predicting data
				if predicttargets[k][i] != 'other' and predicttargets[k][i] != 'unsure':
					if maxScore < self.models[k].distances[i]:
						maxScore = self.models[k].distances[i]
						prediction = predicttargets[k][i]
			predictions.append(prediction)
		return predictions
		#############

	def probability_predict(self, predictdata):

		predicttargets = []
		predictions = []

		for model in self.models:
			predicttargets.append(model.predict(predictdata))

		for i in range(len(predicttargets[0])):
			distances = []
			for k in range(len(predicttargets)):
				#for predicting probabilities
				distances.append(self.models[k].distances[i])

			totalDistances = sum(distances)
			probabilities = []
			for k in distances:
				probabilities.append(k/totalDistances)

		return probabilities


def main():
	############# UI written by Michael
	fname = input("Specify the file name of the dataset (.csv file format): ")
	try:
		f = pd.read_csv(fname).values
	except FileNotFoundError:
		print("File not found, aborting")
		time.sleep(2)
		sys.exit()

	features = eval(input("Specify the number of attributes that each sample has: "))
	data = f[:,:features]
	target = f[:,features]

	classes = eval(input("Specify the number of classes that the classifier will choose between: "))
	class_labels = input("Specify the labels of each class, separated by commas: ").split(",")
	if classes != len(class_labels):
		print("Expected", classes, "classes - instead got", len(class_labels), "classes, aborting")
		time.sleep(2)
		sys.exit()
	for i in range(classes):
		class_labels[i] = class_labels[i].strip(" ")

	split_size = eval(input("Specify the portion of the dataset to be used for testing as an integer value \
							e.g., a value of 3 uses one third of the dataset for testing (note, the remainder of the dataset is used for training): "))
	repetitions = eval("Specify how many times to repeat training (to evaluate average performance): ")
	folds = eval(input("Specify the number of folds to be used during k-Fold cross-validation: "))
	############# end of UI

	############# dataset splitting and classifier evaluation written by Mark
	accuracy = []
	for x in range(repetitions):
		train_data, test_data, train_target, test_target = kFoldCrossValidation().stratifiedFolds(data.tolist(), target.tolist(), split_size, classes)

		model = multiclassSVM(classes, features)
		model.train(np.array(train_data),np.array(train_target),class_labels)
		p = model.predict(test_data)
		pr = model.probability_predict(test_data)

		correct = 0.0
		incorrect = 0.0
		for i in range(len(p)):
			if p[i] == test_target[i]:
				correct += 1
			else:
				incorrect += 1
		accuracy.append(correct/(incorrect+correct))

		matrix, unclassified = confusionMatrix().main(class_labels, p, test_target)
		#print(matrix)
		#print(unclassified)

		myScores, avg = kFoldCrossValidation().main(multiclassSVM(classes, features), data.tolist(), target.tolist(), folds, classes)
		#print(myScores)
		#print(avg)
	
	#print(accuracy)
	#print(np.array(accuracy).mean())
	input("Done")
	############# end of splitting and evalutation

	
if __name__ == "__main__":
	main()