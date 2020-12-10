# TODO: complete this file.
from utils import *
import numpy as np
import random


def bootstrap():
    train_dic = load_train_csv("../data")
    l = []
    for i in range(3):
        m = np.zeros([542, 1774]) * np.nan
        user = train_dic["user_id"]
        question = train_dic["question_id"]
        correct = train_dic["is_correct"]
        random.seed(i)
        random.shuffle(user)
        random.seed(i)
        random.shuffle(question)
        random.seed(i)
        random.shuffle(correct)
        for i in range(30000):
            m[user[i]][question[i]] = correct[i]
        l.append(m)
    return l[0],l[1],l[2]


