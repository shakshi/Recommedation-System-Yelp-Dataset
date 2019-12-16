from surprise import Dataset, BaselineOnly
from surprise import accuracy
from surprise.model_selection import train_test_split
from surprise import Reader
from pyspark import SparkContext, SparkConf
import math
import csv, sys, json
from sklearn import linear_model
import numpy
import time

def createRestaurantVector(bdict):

	bvector=[]
	
	alcohol= 0; kids= 0; tv = 0; 
	seating = 0; delivery = 0; groups = 0;
	priceRange = 2; noiseLevel = 1		

	if "attributes" in bdict and bdict["attributes"] != None:

		attr= bdict["attributes"]

		kids = 1 if ("GoodForKids" in attr and attr["GoodForKids"] == True) else 0
		tv = 1 if ("HasTV" in attr and attr["HasTV"] == True) else 0 
		seating = 1 if ("OutdoorSeating" in attr and attr["OutdoorSeating"] == True) else 0 
		delivery = 1 if ("RestaurantsDelivery" in attr and attr["RestaurantsDelivery"] == True) else 0 
		groups = 1 if ("RestaurantsGoodForGroups" in attr and attr["RestaurantsGoodForGroups"] == True) else 0 

		if "Alcohol" in  attr:
			al= attr["Alcohol"]
			if al == "none":
				alcohol= 0
			elif al == "beer_and_wine":
				alcohol= 1
			elif al == "full_bar":
				alcohol= 3
	
		if "RestaurantsPriceRange2" in attr:
			priceRange = int( attr["RestaurantsPriceRange2"] )	#Ranges from 1-4

		if "NoiseLevel" in attr:
			n= attr["NoiseLevel"]

			if n == "quiet":
				noiseLevel= 0
			elif n == "average":
				noiseLevel= 1
			elif n == "loud":
				noiseLevel= 2
			elif n == "very_loud":
				noiseLevel= 3
	
	bvector.append(alcohol); bvector.append(kids); bvector.append(tv);
	bvector.append(seating); bvector.append(delivery);
	bvector.append(groups); bvector.append(priceRange); 
	bvector.append(noiseLevel)
	return bvector

def getBussinessVector(bussid):

	if bussid in bussinessVectorDict:
		bvector= bussinessVectorDict[bussid]
	else:
		#create a bussiness vector for that business
		for row in businessfile:
			if row["business_id"] == bussid:

				bvector= createRestaurantVector(row);
				bussinessVectorDict[bussid] = bvector
				break		
	
	return bvector

def getBusinessVectors(rows):
	global bussinessVectorDict

	'''
	Create a list of bussiness vectors 
	and ratings for each bussiness 
	given x 
	x[0]- bussid
	x[1]- ratings
	'''

	bvectorList = []
	ratingList = []

	for x in rows:
		bussid= x[0]; rating= x[1]

		bvector= getBussinessVector(bussid)				
		bvectorList.append(bvector)
		ratingList.append(rating)

	return bvectorList, ratingList

input_folder = str(sys.argv[1])
test_file = str(sys.argv[2])
output_file = str(sys.argv[3])

conf = SparkConf().setAppName("appName").setMaster("local[*]")
sc = SparkContext(conf=conf)

start= time.time()

reader = Reader(line_format='user item rating', sep=',',  rating_scale = (1, 5), skip_lines=1)
train_dataset = Dataset.load_from_file(input_folder+"yelp_train.csv", reader=reader)
trainset = train_dataset.build_full_trainset()

options = {'method': 'als', 'n_epochs': 20, 'reg_u': 7,'reg_i': 3}
algo = BaselineOnly(bsl_options=options)
algo.fit(trainset)

#train rdd
trainrdd = sc.textFile(input_folder+"yelp_train.csv").map(lambda line: line.split(","))
trainrdd= trainrdd.map(lambda x: (x[0], x[1], x[2]) ).filter(lambda x: x[0]!= "user_id")

users = trainrdd.map(lambda tup : (tup[0], [(tup[1],float(tup[2]))] )).reduceByKey(lambda acc, x: acc+x)
biddict = trainrdd.map(lambda x : x[1]).distinct().zipWithIndex().collectAsMap()   #buss id, index in dict

testrdd = sc.textFile(test_file).map(lambda line: line.split(","))
testdata = testrdd.map(lambda x: (x[0], x[1], x[2]) ).filter(lambda x: x[0]!= "user_id").collect()

businessrdd = sc.textFile(input_folder + "business.json")
businessfile = businessrdd.map(lambda a: json.loads(a)).collect()

bussinessVectorDict = {}

count1 = 0; count2 = 0
count3 = 0; count4 = 0; count5 = 0

error = 0; n=0;

outf = open(output_file,"w")
outf.write("user_id, business_id, prediction")
	
for row in testdata:
	userid= row[0]
	bussid= row[1]
	rating= row[2]

	if bussid in biddict:
		pred = algo.predict(userid, bussid , r_ui=float(rating), verbose=True)		
		outf.write("\n" + pred.uid +","+ pred.iid + ","+ str(pred.est))

		predRating = pred.est
	else:
		blist = users.filter(lambda a: a[0] == userid).collect()
		businessVectorList, ratingList = getBusinessVectors(blist[0][1])
		
		reg = linear_model.LassoLars(alpha=0.7)
		reg.fit(businessVectorList, ratingList)
		mypred = reg.predict( [getBussinessVector(bussid)] )			
		
		outf.write("\n" + pred.uid +","+ pred.iid + ","+ str(float(mypred.item(0))))

		predRating = float(mypred.item(0))

	diff = abs(predRating - float(rating))
	error+= (diff*diff)
	n+=1

	if diff >=0 and diff<1:
		count1 += 1
	elif diff >=1 and diff<2:
		count2 += 1
	elif diff >=2 and diff<3:
		count3 += 1
	elif diff >=3 and diff<4:
		count4 += 1
	else:
		count5 += 1

end= time.time()
error= error/n
error= math.sqrt(error)

print('RMSE', error)

print("Error Distribution:")
print(count1);print(count2)
print(count3); print(count4)
print(count5)

print('Time Taken:', end- start, 's')


