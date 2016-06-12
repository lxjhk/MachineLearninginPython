from numpy import *
import operator
import matplotlib
import matplotlib.pyplot as plt
import os


def createDataSet():
	group = array([[1.0,1.1],[1.0,1.0],[0,0],[0,0.1]])
	labels = ['A','A','B','B']
	return group, labels

def classify0(inX, dataSet, labels, k):
	dataSetSize = dataSet.shape[0]
	diffMat = tile(inX, (dataSetSize,1)) - dataSet
	sqDiffMat = diffMat**2
	sqDistances = sqDiffMat.sum(axis=1)
	distances = sqDistances**0.5
	sortDistIndicies = distances.argsort()
	classCount = {}

	for i in range(k):
		voteIlabel = labels[sortDistIndicies[i]]
		classCount[voteIlabel] = classCount.get(voteIlabel,0) + 1
	sortedClassCount = sorted(classCount.items(),
		key=operator.itemgetter(1), reverse=True)
	return sortedClassCount[0][0]

def file2matrix(filename):
	fr = open(filename)
	numberOfLines = len(fr.readlines())
	returnMat = zeros((numberOfLines,3))
	classLabelVector = []
	fr = open(filename)
	index = 0
	for line in fr.readlines():
		line = line.strip()
		listFromLine = line.split('\t')
		returnMat[index,:] = listFromLine[0:3]
		classLabelVector.append(listFromLine[-1])
		index += 1
	return returnMat,classLabelVector

def MapLabelToVal(labels):
	counter = 0
	labeldic = {}
	labelVal = []
	for element in labels:
		if element in labeldic:
			labelVal.append(labeldic[element])
		else:
			labeldic[element] = counter;
			counter += 1
			labelVal.append(labeldic[element])
	return labelVal

def autoNorm(dataSet):
	minVals = dataSet.min(0)
	maxVals = dataSet.max(0)
	ranges = maxVals - minVals 
	normDataSet = zeros(shape(dataSet))
	m = dataSet.shape[0]
	normDataSet = dataSet - tile(minVals,(m,1))
	normDataSet = normDataSet / tile(ranges, (m,1))
	return normDataSet, ranges, minVals

def datingClassTest():
	hoRatio = 0.10
	datingDataMat, datingLabels = file2matrix('datingTestSet.txt')
	normMat, ranges, minVals = autoNorm(datingDataMat)
	m = normMat.shape[0]
	numTestVecs = int(m*hoRatio)
	errorCount = 0.0
	for i in range(numTestVecs):
		classifierResult = classify0(normMat[i,:], normMat[numTestVecs:m,:],\
		 datingLabels[numTestVecs:m], 3)
		print ("the classifier came back with: %s, the real answer is: %s"\
			% (classifierResult, datingLabels[i]))
		if (classifierResult != datingLabels[i]):
			errorCount +=1.0
	print("the total error rate id :%f" % (errorCount/float(numTestVecs)))

def classifyPerson():
	percentTats = float(input("percentage of time spent playing video games?  "))
	ffMiles = float(input("frequent flier miles earned per year?  "))
	iceCream = float(input("liters of ice cream consumed per year?  "))
	datingDataMat,datingLabels = file2matrix('datingTestSet.txt')
	normMat, ranges, minVals = autoNorm(datingDataMat)
	inArr = array([ffMiles, percentTats, iceCream])
	classifierResult = classify0((inArr - minVals)/ranges, normMat, datingLabels, 3)
	print(classifierResult)

def img2vector(filename):
	returnVect = zeros((1,1024))
	fr = open(filename)
	for i in range(32):
		lineStr = fr.readline()
		for j in range(32):
			returnVect[0,32*i+j] = int(lineStr[j])
	return returnVect

def handwritingClassTest():
	hwLabels = []
	traiiningFileList = os.listdir('digits/trainingDigits')
	m = len(traiiningFileList)
	trainingMat = zeros((m,1024))
	for i in range(m):
		fileNameString = traiiningFileList[i]
		classNumber = int(fileNameString.split('.')[0].split('_')[0])
		hwLabels.append(classNumber)
		trainingMat[i,:] = img2vector('digits/trainingDigits/%s' % fileNameString)
	testFileList = os.listdir('digits/testDigits')
	errorCount = 0.0
	mTest = len(testFileList)
	for i in range(mTest):
		fileNameString = testFileList[i]
		classNumber = int(fileNameString.split('.')[0].split('_')[0])
		vectorUnderTest = img2vector('digits/testDigits/%s' % fileNameString)
		classifierResult = classify0(vectorUnderTest, trainingMat, hwLabels, 3)
		print ("The classifier comes back with %d, the real answer is %d" % (classifierResult, classNumber))
		if (classifierResult != classNumber):
			errorCount += 1.0
	print ("The total number of errors is %d" % errorCount)
	print ("The total error rate is %f" % (errorCount/mTest))


if __name__ == "__main__":
	#group, labels = createDataSet()
	#print(classify0([10,1000], group, labels, 3))

	#datingDataMat, datingLabels = file2matrix("datingTestSet.txt")
	#datingClassTest()
	#classifyPerson()
	#fig = plt.figure()
	#ax = fig.add_subplot(111)
	#ax.scatter(datingDataMat[:,1], datingDataMat[:,2], s=30, c = array(MapLabelToVal(datingLabels)))
	#plt.show()


	#img2vector("digits/testDigits/0_1.txt")
	#handwritingClassTest()

