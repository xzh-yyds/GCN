import argparse
import javalang
import os
import numpy as np
from sklearn.preprocessing import normalize
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument('--path', type = str, default='./data/project')
args = parser.parse_args()
path = args.path
print('the path of input-data is:', path)


# generate the call graph
os.system('java -jar tool/javacg.jar %s/*.jar > data/calllist.txt' % path)

# 1. 遍历程序，提取出所有java类并编号
class_list = []
for root, _, files in os.walk(path):
    for file in files:
        if file.endswith('.java'):
            file = os.path.join(root, file)
            with open(file, 'r', encoding='gbk') as f:
                code = f.read()
            tree = javalang.parse.parse(code)
            for c in tree.types:
                classname = tree.package.name + '.' + c.name
                class_list.append(classname)

# generate the class name to number
class2num = {}
for i in range(len(class_list)):
    class2num[class_list[i]] = i

classnum = len(class2num)
print('the number of total class is:', classnum)

with open('./data/calllist.txt', 'r', encoding='gbk') as f:
    calllist = f.read()
calllist = calllist.splitlines()


# function: extrac class from the call graph
def Func_Class(name:str):
    cname = name[:name.find(':')]
    return cname

# 2. 生成临界矩阵adj
adj = np.zeros((classnum, classnum), dtype=int)

for r in calllist:
    if r.startswith('C'):
        continue
    cer, cee = r.split()
    cer_m, cee_m = cer[2:], cee[3:]
    cer_c, cee_c = Func_Class(cer_m), Func_Class(cee_m)    
    cer_id, cee_id = class2num.get(cer_c), class2num.get(cee_c)
    if cer_id!=None and cee_id!=None : # 方法级调用图的一条边
        if cer_id!=cee_id: # 不允许自环
            adj[cer_id, cee_id] = 1


# 3. 生成函数调用图，将API进行编号

# store edge of the graph, map[int, set(int)]
e = dict()
methodset = set()
# fill the adjacent matrix C
for line in calllist:
    if line.startswith('C:'):
        continue
    caller, callee = line.split(' ')
    caller, callee = caller[2:], callee[3:]
    caller_class, callee_class = Func_Class(caller), Func_Class(callee)
    caller_id, callee_id = class2num.get(caller_class), class2num.get(callee_class)

    if caller_id != None:
        methodset.add(caller)
    if callee_id != None:
        methodset.add(callee)

    if caller_id != None and callee_id != None:  # both classes are in the application 
        if not caller in e:
            e[caller] = set()
        e[caller].add(callee)

print('the number of total method is:', len(methodset))

# 这里需要注意的是，只有出度不为0并且入度为0的函数才是API
in_func = []
for key in e:
    f = True
    for v in e.values():
        if key in v:
            f = False
    if f:
        in_func.append(key)

APInum = len(in_func)
print('the number of API:', APInum)

# code the in_degree API
func2num = {}
for i in range(len(in_func)):
    func2num[in_func[i]] = i

'''
initialize the adjacent matrix C
'''
C = np.zeros(shape=(classnum, classnum), dtype=int)

'''
initialize the call link matrix EP
'''
EP = np.zeros(shape=(classnum, APInum), dtype=int)

'''
initialize the inherent matrix In
'''
In = np.zeros(shape=(classnum, classnum), dtype=int)


# function dfs, pass all of graph and update the matrix
def dfs(start:str, index:int, S1:set, S2:set):
    if start not in e:
        return
    cer = Func_Class(start)
    sti = class2num.get(cer)

    # 去重，防止出现自递归以及A、B两个类相互递归等情况
    if cer in S1:
        return
    
    S1.add(cer)
    ends = e[start]
    for end in ends:
        cee = Func_Class(end)
        edi = class2num.get(cee)

        # 先更新EP矩阵，不用关注去重问题，这是一个无权重矩阵，只有0和1
        EP[sti, index], EP[edi, index] = 1, 1

        # 如果i和j这两个类已经被存储到调用链上了，那么不用重复添加，这里的调用链是无向的
        h = hash(frozenset({sti, edi}))
        if h in S2:
            continue
        # 如果没有出现，那么就hash一对set，存储到S2中，然后将矩阵权重+1
        else:
            S2.add(h)
            C[sti, edi] += 1
            C[edi, sti] += 1
        dfs(end, index, S1, S2)


# 4. pass all the in_degree API, update the matrix EP and C
for i in range(len(in_func)):
    S1, S2 = set(), set()
    dfs(in_func[i], i, S1, S2)

# print(np.sum(EP))
# print(np.sum(C))


def getFullName(name, tree):
    for imp in tree.imports:
        if imp.path.endswith(name):
            return imp.path
    return tree.package.name + '.' + name

# 5. get the inherent matrix
for root, _, files in os.walk(path):
    for name in files:
        if not name.endswith('.java'):
            continue
        file = os.path.join(root, name)
        with open(file, 'r', encoding='gbk') as f:
            content = f.read()
        tree = javalang.parse.parse(content)
        for c in tree.types:
            fullname = tree.package.name + '.' + c.name
            x_id = class2num[fullname]

            if type(c) == javalang.tree.ClassDeclaration:
                if c.extends != None:
                    y_id = class2num.get(getFullName(c.extends.name, tree))
                    if y_id != None:
                        In[x_id, y_id], In[y_id, x_id] = 1, 1
                if c.implements != None:
                    for inf in c.implements:
                        y_id = class2num.get(getFullName(inf.name, tree))
                        if y_id != None:
                            In[x_id, y_id], In[y_id, x_id] = 1, 1
            elif type(c)==javalang.tree.InterfaceDeclaration:
                if c.extends!=None:
                    for inf in c.extends:
                        y_id = class2num.get(getFullName(inf.name, tree))
                        if y_id != None:
                            In[x_id, y_id], In[y_id, x_id] = 1, 1

adj_new = pd.DataFrame(adj)
adj_new.to_csv('./data/struct.csv', index=False, header=False)

EP_norm, C_norm, In_norm = normalize(EP), normalize(C), normalize(In)
X = pd.DataFrame(np.concatenate((EP_norm, C_norm, In_norm), axis=1))
X.to_csv('./data/content.csv', index=False, header=False)


index2name = pd.DataFrame(data=class_list, index=list(range(0,len(class_list))))
index2name.to_csv('./data/index2name.csv', index=False)