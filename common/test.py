第一题你自己解吧


数值的话

p1 = 1.5
p2 = 1
def sample_x():
    return random.random() * p1
def sample_y():
    return random.random() * p2
#random.random()返回0-1之间的浮点数
tot_sample = 1000000
positive_samples = 0
for i in range(tot_samples):
    x_sample = sample_x()
    y_sample = sample_y()
    if x_sample < y_sample:
        positive_samples += 1
print(positive_samples / tot_samples)

你写这个
integral(0,p1) (1 / p_1) dx integral(-x, 0) (1 / p_2) dy


第二题我写代码


相似性指标：

你就念下面的就差不多了
他就是考量你对树有多少理解

问题1:首先是设计特征的问题，下面的特征都计算出来然后拼接成一个大的向量就可以了：
    树的深度（树有多少层）
    树的节点的总数量
    对称的性质， 比如是不是能够左右翻转成为自己
    最短编辑距离： 将一棵树转化成另一棵树的最小插入/删除的节点的数量
    哈希的值（利用哈希函数将树的向量表示映射到一个值，哈希函数就是从一个东西映射到另一个值上面的函数，）
    最长公共字串（我觉得这个没啥关系啊（这个只能算一个支路上的）) 暴力方法就是O(n^3))的算法，n为字串的长度，（就是两个字串的每个字母都作为开头对比一下），dp熟悉的话，dp的公式是 dp[i][k] = dp[i - 1][k - 1] + 1;

问题2:
距离矩阵
我也不会
字符串的编辑距离就是list的编辑距离
我知道两个字符串的编辑距离，肯定比树的编辑距离简单，您要是知道这个的话不如提一下说您知道这个，然后他会问你dp的问题

问题3:
def calculate_belonging(tree, centers):
    distances = [disrance(tree, center) for center in centers]
    min_distance_idx = disrance.index(min(distances))
    return min_distance_idx

k means#    这个是naive的k means然后加了树的说明
def K-Means(list: trees，int: k)
    #trees 为输入的树的列表,每棵树都以长度为n的向量表示
    #k 为聚类中心的数量
    n=  len(trees)
    dim = len(trees[0])
    centers = random.sampe(trees, k)
    belonged_centers = [] #表示每个树属于哪一个中心节点
    new_centers = [] #表示新计算出来的属于中心的节点，用于判断是否终止
    while(not_equal(new_centers, centers)): #新计算出来的中心和老的计算出来的中心不一样
        new_centers = []
        belonged_centers = []
        for tree in trees:
            belonged_centers.append(calculate_belonging(tree, centers))
        for i, center in enumerate(centers):
            belonged_trees = [tree for tree, curr_center in zip(trees, belonged_centers) if center == curr_center ] #计算属于这个中心的树
            new_center = calculate_mean(belonged_trees) #新的中心是这些树的向量表示的均值
            new_centers.append(new_center)

我憋不住了