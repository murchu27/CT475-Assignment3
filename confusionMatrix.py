############# confusionMatrix class written by Mark
class confusionMatrix(object):
	
	def main(self, classNames, pred, output):

		matrix = []
		unclassified = 0
		for i in range(len(classNames)):
			matrix.append([0, 0, 0])

		for x in range(len(output)):
			
			if pred[x] != 'N/A':
				outIndex = classNames.index(output[x])
				predIndex = classNames.index(pred[x])
				val = matrix[outIndex][predIndex]
				val += 1
				matrix[outIndex][predIndex] = val
			else: 
				unclassified += 1

		return matrix, unclassified