# TODO: complete this file.
from utils import *
import numpy as np
import random


def bootstrap():
    train_dic = load_train_csv("../data")
    l = []
    for i in range(3):
        size = len(train_dic["user_id"])
        user = []
        question = []
        correct = []
        m = np.zeros([542, 1774]) * np.nan
        for i in range(30000):
            idx = random.randint(0,size-1)
            u = train_dic["user_id"][idx]
            user.append(u)
            q = train_dic["question_id"][idx]
            question.append(q)
            c = train_dic["is_correct"][idx]
            correct.append(c)
            m[u][q] += c
        train_d = {"user_id":user,"question_id":question,"is_correct":correct}
        return train_d, m

a,b = bootstrap()
print(len(a["user_id"]))


