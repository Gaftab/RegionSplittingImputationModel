###################################################
#
#   Script to compute the Weights
#
##################################################
import numpy as np
import csv

ds = "Iris"
cr = 2 #complete rate
weights = []
weights_title = ["sepal_length","sepal_width","petal_length","petal_width","class"]

data_matrix = np.loadtxt(open("../data/"+ds+"_"+str(cr)+"_for_compute_weight.csv"), delimiter=",",skiprows=0)
# data_matrix = np.delete(data_matrix,0,axis=0)
# print(data_matrix)
# print(len(data_matrix))
# transform to matrix
data_matrix=(data_matrix-data_matrix.min())/(data_matrix.max()-data_matrix.min())
# standardization
m, n = data_matrix.shape
# m,n: the number of rows and columns
k = 1 / np.log(m)
yij = data_matrix.sum(axis=0)  
pij = data_matrix / yij
test = pij * np.log(pij)
test = np.nan_to_num(test)

ej = -k * (test.sum(axis=0))
# Calculate the information entropy of each index
wi = (1 - ej) / np.sum(1 - ej)
weights.append(wi)
print(weights)

with open("../data/"+ds+"_weights.csv", 'w', newline='') as result_file:
    writer = csv.writer(result_file)
    writer.writerow(weights_title)
    writer.writerows(weights)