###################################################
#
#   Script to execute the imputation
#
##################################################
import pandas as pd
import math
import csv
import impyute as impy
import numpy as np
import time

attr = ["sepal_length","sepal_width","petal_length","petal_width","class"]
def get_average(records):
    """

    :param records:
    :return:
    """
    return sum(records)/len(records)

def get_variance(records):
    """

    :param records:
    :return:
    """
    average =  get_average(records)
    return sum([(x - average) ** 2 for x in records]) / len(records)

def get_standard_deviation(records):
    """

    :param records:
    :return:
    """
    variance = get_variance(records)
    return math.sqrt(variance)

def get_mse(records_real, records_predict):
    """

    :param records_real:
    :param records_predict:
    :return:
    """
    if len(records_real) == len(records_predict):
        return sum([(x - y) ** 2 for x, y in zip(records_real, records_predict)])
    else:
        return None


def get_rmse(records_real, records_predict):
    """
    :param records_real:
    :param records_predict:
    :return:
    """
    mse = get_mse(records_real, records_predict)
    if mse:
        return math.sqrt(mse)
    else:
        return None

def get_nrmse(records_real, records_predict):
    """

    :param records_real:
    :param records_predict:
    :return:
    """
    mse = get_mse(records_real, records_predict)
    standev = get_standard_deviation(records_real)
    if mse:
        return math.sqrt(mse * standev)

def get_ewmtuples(cr,  mr, data_name,m,n,ict_df):
    weights_file = "../data/"+data_name+"_weights.csv"
    weights_csv = pd.read_csv(weights_file, low_memory=False)
    weights_df = pd.DataFrame(weights_csv)
    interate = []
    if cr == 2:
        wcol = 0
    elif cr == 3:
        wcol = 1
    else:
        wcol = 2
    for i in range(0,m):
        irs = 0
        for j in range(0,n):
            if  pd.isnull(ict_df.iloc[i,j]):
                val = 0
            else:
                val = 1
            ir = weights_df.iloc[wcol,j] * val
            irs += ir
        interate.append([i,irs])

    return interate

def takeSecond(elem):
    return elem[1]

def tuple_order(list,data):
    list_order = sorted(list,key=takeSecond,reverse=True)
    tdata = data.values
    tuple_data = np.array(tdata).tolist()
    # print(tuple_data)
    orderList = []
    orderIndex = []
    for i in range(0,len(list_order)):
        k = list_order[i][0]
        orderIndex.append(k)
        orderList.append(tuple_data[k])
    # print(orderList)
    return orderList, orderIndex

def tuples_part(list,k):
    t = int(len(list)/k)
    part = [ list[i:i+t] for i in range(0,len(list), t)]
    part[k-1]= list[(k-1)*t:]
    return part

def tuple_impute(list,k,data):
    ict_df =pd.DataFrame(list,columns=attr)
    temps = pd.concat([data,ict_df], axis=0,ignore_index=True)
    impy_result = impy.fast_knn(temps, k)
    impy_result.columns=attr
    return impy_result

def cross_validation(list,k,data):
    ict_df =pd.DataFrame(list,columns=attr)
    temps = pd.concat([data,ict_df], axis=0,ignore_index=True)
    impy_result = impy.fast_knn(temps, k)
    impy_result.columns=attr
    return impy_result

def knn_RESI(list,part_num,k,ct_df,M,m,n):
    temp_df = ct_df
    part_impy2_df = pd.DataFrame(columns=attr)
    for i in range(0,part_num):
        part_impy1 = tuple_impute(list[i],k,temp_df)
        temp_df = part_impy1
    part_impy1_df = part_impy1.iloc[M-m:]
    print(part_impy1_df)
    t = len(list[0])
    print("t")
    print(t)
    for i in range(0,part_num-1):
        print("list"+str(i))
        print(list[i])
        CTk_1 =  pd.concat([part_impy1.iloc[:M-m+t*i],part_impy1.iloc[M-m+t*(i+1):]], axis=0,ignore_index=True)
        part_impy2 = tuple_impute(list[i],k,CTk_1)
        part_impy2_df = pd.concat([part_impy2_df,part_impy2.iloc[M-t:]],axis=0,ignore_index=True)
    print("part_impy2_dfk-1")
    print(part_impy2_df)
    print("list[part_num-1]")
    print(list[part_num-1])
    last_part= tuple_impute(list[part_num-1],k,ct_df)
    print("last_part")
    print(last_part)
    part_impy2_df = pd.concat([part_impy2_df,last_part.iloc[M-m:]],axis=0,ignore_index=True)
    print("part_impy2_df")
    print(part_impy2_df)
    part_impy = pd.DataFrame(columns=attr)
    part_impy = pd.concat([part_impy,part_impy1_df],axis=0,ignore_index=True)
    for i in range(0,m):
        for j in range(0,n):
            if part_impy.iloc[i,j] != part_impy2_df.iloc[i,j]:
                part_impy.iloc[i,j] = (part_impy.iloc[i,j] + part_impy2_df.iloc[i,j])/2
    print("part_impy")
    print(part_impy)
    return part_impy

def recover_origsort(data,index):
    temp = pd.DataFrame(columns=attr)
    temp = pd.concat([temp,data], axis=0,ignore_index=True)
    temp["index"] = index
    print(temp)
    temp.sort_values(by="index",ascending=True,inplace=True)
    temp.drop(columns = ["index"],inplace=True)
    # print(temp)
    return temp

ds = "Iris"
cr = 2  #complete rate
mr = 0.2 #missing rate
M = 150
m = 75
n = 5
rmses_title = ["datasets", "RESI", "RESI_time"]

rmses = []

ct_file = "../data/" + ds + "_" + str(cr) + "_CT.csv"
ct_csv = pd.read_csv(ct_file, low_memory=False)
ct_df = pd.DataFrame(ct_csv, columns= attr)

##load the file
ict_file = "../data/" + ds + "_" + str(cr) + "_ICT_" + str(mr) + ".csv"
ict_csv = pd.read_csv(ict_file, low_memory=False)
ict_df = pd.DataFrame(ict_csv)
true_file = "../data/" + ds + "_" + str(cr) + "_ICTtrue.csv"
true_csv = pd.read_csv(true_file, low_memory=False)
true_df = pd.DataFrame(true_csv)

##RESI imputation
tuples_list = get_ewmtuples(cr,mr, ds, m, n, ict_df)
print(tuples_list)
tuples_order = tuple_order(tuples_list,ict_df)
order_data = tuples_order[0]
order_index = tuples_order[1]
part_N = [3,5,7,10,15]
knn_K = [3,5,10,15]
for pn in part_N:
    for kk in knn_K:
        start = time.time()
        subspaces = tuples_part(order_data, pn)
        imputed_data = knn_RESI(subspaces, pn, kk, ct_df,M,m,n)
        end = time.time()
        resi_time = end - start
        print("imputed_data")
        print(imputed_data)


        resi_true = []
        resi_pred = []
        print(str(cr)+"_"+str(mr)+"_"+str(pn)+"_"+str(kk)+"sort results")
        print(imputed_data)
        print(str(cr)+"_"+str(mr)+"_"+str(pn)+"_"+str(kk)+"tuple sequence")
        print(order_index)
        imputed_tuples = recover_origsort(imputed_data,order_index)
        imputed_result = pd.DataFrame( columns=attr)
        imputed_result = pd.concat([imputed_result,imputed_tuples], axis=0,ignore_index=True)
        print("imputed_result")
        print(imputed_result)
        imputed_result.to_csv(
            "../data/resi" + str(ds) + "_" + str(cr) + "_" + str(mr) + "_"+str(pn)+"_"+str(kk)+".csv",
            header=False,
            index=False)
        for i in range(0, m):
            for j in range(0, n):
                if pd.isnull(ict_df.iloc[i,j]):
                    resi_true.append(true_df.iloc[i, j])
                    resi_pred.append(imputed_result.iloc[i,j])
