#imports
import numpy as np
import math
import random
from statistics import mean

############# kFoldCrossValidation class written by Mark
class kFoldCrossValidation(object):
	
	def main(self, classifier, arrIn, arrOut, K, N):

		output, folds, classNames = self.splitFolds(arrIn, arrOut, K, N)
		
		loop = 0
		scores = []
		Test = []
		TestOut = []
		for loop in range(K):
			Test = folds[loop]
			TestOut = output[loop]
			Training = []
			TrainingOut = []
			k = 0
			for k in range(len(folds)):
				if not folds[k] == Test:
					Training += folds[k]
					TrainingOut += output[k]
			classifier.train(np.array(Training),np.array(TrainingOut),classNames)
			predictions = classifier.predict(Test)	
			i = 0
			correct = 0 
			incorrect = 0
			for i in range(len(predictions)):
				if predictions[i] == TestOut[i]:
					correct += 1
				else:
					incorrect += 1 
			scores.append((correct / (incorrect + correct)) * 100)
		avg = mean(scores)
		return scores, avg

	def splitArrays(self, arrIn, arrOut, K, N):
		counter = 0
		Array = []
		sortedOut = []
		classNames = []
		sortedIn = []

		i = 0
		for i in range(N):
			Array.append([])
			sortedIn.append([])
			sortedOut.append([])

		j = 0
		for j in range(len(arrOut)):
			try:
				value = classNames.index(arrOut[j])
				Array[value].append(j)
				sortedOut[value].append(arrOut[j])
				sortedIn[value].append(arrIn[j])
			except:
				classNames.append(arrOut[j])
				Array[len(classNames) - 1].append(j)
				sortedOut[len(classNames) - 1].append(arrOut[j])
				sortedIn[len(classNames) - 1].append(arrIn[j])

		return sortedIn, sortedOut, Array, classNames

	def splitFolds(self, arrIn, arrOut, K, N):
		sortedIn, sortedOut, Array, classNames = self.splitArrays(arrIn, arrOut, K, N)
		folds = []
		output = []
		loop = 0
		for x in range(len(sortedIn)):

			shuffleArr = list(zip(sortedIn[x], sortedOut[x]))

			random.shuffle(shuffleArr)

			sortedIn[x], sortedOut[x] = zip(*shuffleArr)
			
		for loop in range(K):
			folds.append([])
			output.append([])	
		k = 0
		for k in range(len(sortedIn)):
			i = 0
			foldNo = 0
			noFoldElems = math.floor(len(sortedIn[k])/K)
			foldChecker = noFoldElems
			remainder = len(sortedIn[k]) % K
			for i in range(len(sortedIn[k])):
				if i <= foldChecker - 1:
					folds[foldNo].append(sortedIn[k][i])
					output[foldNo].append(sortedOut[k][i])
				else:
					if foldNo <= (remainder - 1):
						folds[foldNo].append(sortedIn[k][i])
						output[foldNo].append(sortedOut[k][i])
						foldNo += 1
						foldChecker += noFoldElems + 1
					else: 
						foldNo += 1
						folds[foldNo].append(sortedIn[k][i])
						output[foldNo].append(sortedOut[k][i])
						foldChecker += noFoldElems
		return output, folds, classNames

	def stratifiedFolds(self, ArrIn, ArrOut, K, N):

		sortedIn, sortedOut, Array, classNames = self.splitArrays(ArrIn, ArrOut, K, N)
		TrainingIndexData = []
		TestIndexData = []
		TrainingInData = []
		TestInData = []
		TrainingOutData = []
		TestOutData = []

		for i in range(len(Array)):
			random.shuffle(Array[i])
			length = (math.floor(len(Array[i])/K))*2
			remainder = len(Array[i]) % K

			if remainder > 0:
				length += math.floor(remainder/2)
				if remainder % 2 != 0:
					length += 1

			TrainingIndexData += Array[i][:length]
			TestIndexData += Array[i][length:]

		for elem in TrainingIndexData:

			TrainingInData.append(ArrIn[elem])
			TrainingOutData.append(ArrOut[elem])
		for elem in TestIndexData:
			TestInData.append(ArrIn[elem])
			TestOutData.append(ArrOut[elem])

		return TrainingInData, TestInData, TrainingOutData, TestOutData