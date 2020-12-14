# TODO: complete this file.
import numpy as np
import random
from item_response import *
from knn import *
from neural_network import *
from sklearn.impute import KNNImputer
from utils import *
from torch.autograd import Variable
import torch.nn as nn
import torch.utils.data

def bootstrap():
    train_dic = load_train_csv("../data")
    l1 = []
    l2 = []
    for i in range(3):
        size = len(train_dic["user_id"])
        user = []
        question = []
        correct = []
        m = np.zeros([542, 1774]) * np.nan
        for i in range(56688):
            idx = random.randint(0,size-1)
            u = train_dic["user_id"][idx]
            user.append(u)
            q = train_dic["question_id"][idx]
            question.append(q)
            c = train_dic["is_correct"][idx]
            correct.append(c)
            if np.isnan(m[u][q]):
                m[u][q]=c
            else:
                m[u][q]+=c
        train_d = {"user_id":user,"question_id":question,"is_correct":correct}
        l1.append(train_d)
        l2.append(m)
    return l1,l2

def main():
    # load val data
    val_data = load_valid_csv("../data")

    # load bootstrapped data
    l1, l2 = bootstrap()
    # bagging 1 knn by item
    nbrs = KNNImputer(n_neighbors=21)
    # We use NaN-Euclidean distance measure.
    mat = nbrs.fit_transform(l2[0].T)
    mat = mat.T
    p1 = []
    for i in range(len(val_data["user_id"])):
        p1.append(mat[val_data["user_id"][i]][val_data["question_id"][i]])

    val_data = load_valid_csv("../data")

    # load bootstrapped data
    l1, l2 = bootstrap()
    # bagging 1 knn by item
    nbrs = KNNImputer(n_neighbors=15)
    # We use NaN-Euclidean distance measure.
    mat = nbrs.fit_transform(l2[0].T)
    mat = mat.T
    p2 = []
    for i in range(len(val_data["user_id"])):
        p2.append(mat[val_data["user_id"][i]][val_data["question_id"][i]])

    val_data = load_valid_csv("../data")

    # load bootstrapped data
    l1, l2 = bootstrap()
    # bagging 1 knn by item
    nbrs = KNNImputer(n_neighbors=10)
    # We use NaN-Euclidean distance measure.
    mat = nbrs.fit_transform(l2[0].T)
    mat = mat.T
    p3 = []
    for i in range(len(val_data["user_id"])):
        p3.append(mat[val_data["user_id"][i]][val_data["question_id"][i]])

    total = 0
    correct = 0
    p = []
    for i in range(len(p1)):
        a = (p1[i]+p2[i]+p3[i])/3.
        p.append(a>=0.5)
    for i in range(len(p)):
        if p[i] == val_data["is_correct"][i]:
            correct += 1
        total += 1
    print("Acc : ",correct/float(total))

    total = 0
    for i in p1:
        total+= i
    knn_mean = total/len(p1)
    # print("knn mean: ",knn_mean)
    Ex2 = 0
    bias = 0.
    for i in p1:
        bias+=(i-knn_mean)**2
        Ex2+=i**2
    print("knn bias: ",bias/len(p1))
    print("knn var: ",Ex2-knn_mean**2)
    print("=======")

    total = 0
    for i in p2:
        total += i
    irt_mean = total / len(p2)
    # print("irt mean: ", irt_mean)
    Ex2 = 0
    bias = 0.
    for i in p2:
        bias+=(i-irt_mean)**2
        Ex2 += i ** 2
    print("irt bias: ",bias/len(p2))
    print("irt var: ",Ex2 - irt_mean ** 2)
    print("=======")

    total = 0
    for i in p3:
        total += i
    neural_mean = total / len(p3)
    # print("neural net: ", neural_mean )
    Ex2 = 0
    bias = 0.
    for i in p3:
        bias += (i - neural_mean) ** 2
        Ex2 += i ** 2
    print("neural bias:", bias / len(p3))
    print("neural var",Ex2 - neural_mean ** 2)


    total = 0
    for i in p:
        total += i
    mean = total / len(p)
    Ex2 = 0
    bias = 0.
    for i in p:
        bias += (i - mean) ** 2
        Ex2 += i ** 2
    print("ensemble_bias:", bias / len(p))
    print("ensemble_var",Ex2 - mean ** 2)

    # # bagging 2 item response
    # lr = 0.025
    # iterations = 20
    # np.random.seed(1005705621)
    # neg_lld_list = []
    # neg_lld_val_list = []
    # iteration_list = []
    # x = np.random.randint(9, size=542)
    # theta = np.array([float(i) for i in x])
    # x = np.random.randint(9, size=1774)
    # beta = np.array([float(i) for i in x])
    # for i in range(iterations + 1):
    #     neg_lld = -neg_log_likelihood(l1[1], theta=theta, beta=beta)
    #     neg_lld_val = - neg_log_likelihood(val_data, theta=theta, beta=beta)
    #     neg_lld_list.append(neg_lld)
    #     neg_lld_val_list.append(neg_lld_val)
    #     theta, beta = update_theta_beta(l1[1], lr, theta=theta, beta=beta)
    #     iteration_list.append(iterations)
    # p2 = []
    # for i, q in enumerate(val_data["question_id"]):
    #     u = val_data["user_id"][i]
    #     x = (theta[u] - beta[q]).sum()
    #     p_a = sigmoid(x)
    #     p2.append(p_a)
    #
    # # bagging 3 neural net
    # train_matrix = l2[2]
    # valid_data = val_data
    # zero_train_matrix = train_matrix.copy()
    # zero_train_matrix[np.isnan(train_matrix)] = 0
    # zero_train_data = torch.FloatTensor(zero_train_matrix)
    # train_data = torch.FloatTensor(train_matrix)
    #
    #
    # num_question = zero_train_data.shape[1]
    # model = AutoEncoder(num_question, 20)
    # lr = 0.05
    # num_epoch = 15
    # lamb = 0.01
    #
    # p3 = trainforens(model, lr, lamb, train_data, zero_train_data,
    #                 valid_data, num_epoch)
    #
    #
    # total = 0
    # for i in p1:
    #     total+= i
    # knn_mean = total/len(p1)
    # # print("knn mean: ",knn_mean)
    # Ex2 = 0
    # bias = 0.
    # for i in p1:
    #     bias+=(i-knn_mean)**2
    #     Ex2+=i**2
    # print("knn bias: ",bias/len(p1))
    # print("knn var: ",Ex2-knn_mean**2)
    # print("=======")
    #
    # total = 0
    # for i in p2:
    #     total += i
    # irt_mean = total / len(p2)
    # # print("irt mean: ", irt_mean)
    # Ex2 = 0
    # bias = 0.
    # for i in p2:
    #     bias+=(i-irt_mean)**2
    #     Ex2 += i ** 2
    # print("irt bias: ",bias/len(p2))
    # print("irt var: ",Ex2 - irt_mean ** 2)
    # print("=======")
    #
    # total = 0
    # for i in p3:
    #     total += i
    # neural_mean = total / len(p3)
    # # print("neural net: ", neural_mean )
    # Ex2 = 0
    # bias = 0.
    # for i in p3:
    #     bias += (i - neural_mean) ** 2
    #     Ex2 += i ** 2
    # print("neural bias:", bias / len(p3))
    # print("neural var",Ex2 - neural_mean ** 2)
    #
    #
    # total = 0
    # correct = 0
    # p = []
    # for i in range(len(p1)):
    #     a = (p1[i]+p2[i]+p3[i])/3.
    #     p.append(a>=0.5)
    # for i in range(len(p)):
    #     if p[i] == valid_data["is_correct"][i]:
    #         correct += 1
    #     total += 1
    # print("Acc : ",correct/float(total))
    #
    # total = 0
    # for i in p:
    #     total += i
    # mean = total / len(p)
    # Ex2 = 0
    # bias = 0.
    # for i in p:
    #     bias += (i - mean) ** 2
    #     Ex2 += i ** 2
    # print("ensemble_bias:", bias / len(p))
    # print("ensemble_var",Ex2 - mean ** 2)



if __name__ == "__main__":
     main()
