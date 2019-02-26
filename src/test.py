#!/usr/bin/env python
# coding: utf-8

# In[1]:


data_dir = "../data/"


# In[2]:


ent2id, id2ent = dict(), dict()
for line in open(data_dir + 'newentity2id.txt', 'r'):
    [ent, id] = line.split()
    ent2id[ent] = int(id)
    id2ent[int(id)]  = ent
print("entities num:\t" + str(len(ent2id)))


# In[3]:


import numpy as np
ent_vec = [] #列表
for line in open("../res/entity2vec.bern"): #训练完后最终生成的实体向量矩阵
    vec = [float(i) for i in line.split()] #列表，读取每一行的浮点数
    ent_vec.append(np.array(vec))


# In[4]:


class Counter(dict):
    def __missing__(self, key):
        return False


# In[5]:


is_common = Counter()
kg1, kg2 = [], []
for line in open("../data/common_entities.txt"):
    is_common[ent2id[line.split()[0]]] = True #找到种子实体id，并标记为true
for i in range(24902):
    if(is_common[i]):
        continue
    if(i < 14951):
        kg1.append(i)
    else:
        kg2.append(i)        


# In[9]:


# occupied = Counter()
# distance2entities_pair = []
cnt = 0
rank = 0
from numpy import linalg as LA
hits10 = hits1 = 0
for i in kg1:
    distance = []
    for j in kg2:
        dis = LA.norm(ent_vec[i]-ent_vec[j], 1) #实体i的一行矩阵与实体j的一行矩阵对应值相减的绝对值累加和
        if(dis > 50):
            assert(id2ent[i] + '$' != id2ent[j]) #认为距离大于50的是不可能匹配的，保证带有‘$’的都分配到第2个知识图谱中了
            continue
        distance.append((dis, i, j)) #change to distance2entities_pair afterward，距离<50，加入到distance中
    distance.sort() #对一个kg中的每个实体，计算其与另一个kg中所有实体的距离，选择小于阈值的备选
    for k in range(10): #选中距离最近的前10个
        (dis, i, j) = distance[k]
        if(id2ent[i] + '$' == id2ent[j]):
            hits10 += 1
    for k in range(1): #选中距离最近的前1个
        (dis, i, j) = distance[k]
        if(id2ent[i] + '$' == id2ent[j]):
            hits1 += 1
    for k in range(len(distance)):
        (dis, i, j) = distance[k]
        if(id2ent[i] + '$' == id2ent[j]):
            rank += k + 1
    cnt += 1

    eval_hits10=float(hits10)/cnt
    eval_hits1=float(hits1)/cnt
    eval_rank=float(rank)/cnt
    print(eval_hits10,eval_hits1, eval_rank, cnt)
    evaluation=[eval_hits10,eval_hits1,eval_rank,cnt]
    with open('eval.txt','a+') as f:
        f.writelines(str(evaluation)+'\n')
