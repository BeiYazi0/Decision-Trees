import numpy as np
import math
from collections import Counter
import time


class DecisionNode:
    """Class to represent a nodes or leaves in a decision tree."""

    def __init__(self, left, right, decision_function, class_label=None):
        """
        Create a decision node with eval function to select between left and right node
        NOTE In this representation 'True' values for a decision take us to the left.
        This is arbitrary, but testing relies on this implementation.
        Args:
            left (DecisionNode): left child node
            right (DecisionNode): right child node
            decision_function (func): evaluation function to decide left or right
            class_label (value): label for leaf node
        """
        self.left = left
        self.right = right
        self.decision_function = decision_function
        self.class_label = class_label

    def decide(self, feature):
        """Determine recursively the class of an input array by testing a value
           against a feature's attributes values based on the decision function.

        Args:
            feature: (numpy array(value)): input vector for sample.

        Returns:
            Class label if a leaf node, otherwise a child node.
        """

        if self.class_label is not None:
            return self.class_label

        elif self.decision_function(feature):
            return self.left.decide(feature)

        else:
            return self.right.decide(feature)


def load_csv(data_file_path, class_index=-1):
    """Load csv data in a numpy array.
    Args:
        data_file_path (str): path to data file.
        class_index (int): slice index for data labels.
    Returns:
        features, classes as numpy arrays if class_index is specified,
            otherwise all as nump array.
    """

    handle = open(data_file_path, 'r')
    contents = handle.read()
    handle.close()
    rows = contents.split('\n')
    out = np.array([[float(i) for i in r.split(',')] for r in rows if r])

    if(class_index == -1):
        classes= out[:,class_index]
        features = out[:,:class_index]
        return features, classes
    elif(class_index == 0):
        classes= out[:, class_index]
        features = out[:, 1:]
        return features, classes

    else:
        return out


def build_decision_tree():
    """Create a decision tree capable of handling the sample data contained in the ReadMe.
    It must be built fully starting from the root.
    
    Returns:
        The root node of the decision tree.
    """

    # TODO: finish this.͏︆͏󠄃͏󠄌͏󠄍͏󠄂͏️͏︇͏️͏󠄆
    class_0_node = DecisionNode(None, None, None, 0)
    class_1_node = DecisionNode(None, None, None, 1)
    class_2_node = DecisionNode(None, None, None, 2)
    
    bottom1 = DecisionNode(class_2_node, class_0_node, lambda A: A[2] < 0)
    bottom2 = DecisionNode(class_2_node, class_1_node, lambda A: A[2] < -0.75)
    dt_root_right = DecisionNode(bottom1, bottom2, lambda A: A[1] < -0.3)
    dt_root = DecisionNode(class_0_node, dt_root_right, lambda A: A[0] < 0.06)
    return dt_root


def confusion_matrix(true_labels, classifier_output, n_classes=2):
    """Create a confusion matrix to measure classifier performance.
   
    Classifier output vs true labels, which is equal to:
    Predicted  vs  Actual Values.
    
    Output will sum multiclass performance in the example format:
    (Assume the labels are 0,1,2,...n)
                                     |Predicted|
                     
    |A|            0,            1,           2,       .....,      n
    |c|   0:  [[count(0,0),  count(0,1),  count(0,2),  .....,  count(0,n)],
    |t|   1:   [count(1,0),  count(1,1),  count(1,2),  .....,  count(1,n)],
    |u|   2:   [count(2,0),  count(2,1),  count(2,2),  .....,  count(2,n)],'
    |a|   .............,
    |l|   n:   [count(n,0),  count(n,1),  count(n,2),  .....,  count(n,n)]]
    
    'count' function is expressed as 'count(actual label, predicted label)'.
    
    For example, count (0,1) represents the total number of actual label 0 and the predicted label 1;
                 count (3,2) represents the total number of actual label 3 and the predicted label 2.           
    
    Args:
        classifier_output (list(int)): output from classifier.
        true_labels: (list(int): correct classified labels.
        n_classes: int: number of classes needed due to possible multiple runs with incomplete class sets
    Returns:
        A two dimensional array representing the confusion matrix.
    """

    # TODO: finish this.͏︆͏󠄃͏󠄌͏󠄍͏󠄂͏️͏︇͏️͏󠄆
    c_matrix = [[0]*n_classes for _ in range(n_classes)] # 混淆矩阵，第一个纬度是真实标签，第二个纬度是预测标签
    for idx in range(len(classifier_output)):
        actual, guess  = true_labels[idx], classifier_output[idx]
        c_matrix[actual][guess] += 1
    return c_matrix


def precision(true_labels, classifier_output, n_classes=2, pe_matrix=None):
    """
    Get the precision of a classifier compared to the correct values.
    In this assignment, precision for label n can be calculated by the formula:
        precision (n) = number of correctly classified label n / number of all predicted label n 
                      = count (n,n) / (count(0, n) + count(1,n) + .... + count (n,n))
    Args:
        classifier_output (list(int)): output from classifier.
        true_labels: (list(int): correct classified labels.
        n_classes: int: number of classes needed due to possible multiple runs with incomplete class sets
        pe_matrix: pre-existing numpy confusion matrix
    Returns:
        The list of precision of each classifier output. 
        So if the classifier is (0,1,2,...,n), the output should be in the below format: 
        [precision (0), precision(1), precision(2), ... precision(n)].
    """
    # TODO: finish this.͏︆͏󠄃͏󠄌͏󠄍͏󠄂͏️͏︇͏️͏󠄆
    correct_predict = [0] * n_classes  # 预测正确次数
    predict_count = [0] * n_classes    # 总预测次数
    precision_value = [0] * n_classes  # 预测的准确度
    for idx in range(len(classifier_output)):
        actual, guess = true_labels[idx], classifier_output[idx]
        predict_count[guess] += 1
        if guess == actual:
            correct_predict[guess] += 1

    for i in range(n_classes):
        if correct_predict[i] == 0:
            continue
        precision_value[i] = correct_predict[i]/predict_count[i]
    return precision_value


def recall(true_labels, classifier_output, n_classes=2, pe_matrix=None):
    """
    Get the recall of a classifier compared to the correct values.
    In this assignment, recall for label n can be calculated by the formula:
        recall (n) = number of correctly classified label n / number of all true label n 
                   = count (n,n) / (count(n, 0) + count(n,1) + .... + count (n,n))
    Args:
        classifier_output (list(int)): output from classifier.
        true_labels: (list(int): correct classified labels.
        n_classes: int: number of classes needed due to possible multiple runs with incomplete class sets
        pe_matrix: pre-existing numpy confusion matrix
    Returns:
        The list of recall of each classifier output..
        So if the classifier is (0,1,2,...,n), the output should be in the below format: 
        [recall (0), recall (1), recall (2), ... recall (n)].
    """
    # TODO: finish this.͏︆͏󠄃͏󠄌͏󠄍͏󠄂͏️͏︇͏️͏󠄆
    correct_predict = [0] * n_classes  # 预测正确次数
    true_count = [0] * n_classes       # 真实标签的出现次数 
    recall_value = [0] * n_classes     # 真实标签的召回率
    for idx in range(len(classifier_output)):
        actual, guess  = true_labels[idx], classifier_output[idx]
        true_count[actual] += 1
        if guess == actual:
            correct_predict[actual] += 1

    for i in range(n_classes):
        if correct_predict[i] == 0:
            continue
        recall_value[i] = correct_predict[i]/true_count[i]
    return recall_value


def accuracy(true_labels, classifier_output, n_classes=2, pe_matrix=None):
    """Get the accuracy of a classifier compared to the correct values.
    Balanced Accuracy Weighted:
    -Balanced Accuracy: Sum of the ratios (accurate divided by sum of its row) divided by number of classes.
    -Balanced Accuracy Weighted: Balanced Accuracy with weighting added in the numerator and denominator.

    Args:
        classifier_output (list(int)): output from classifier.
        true_labels: (list(int): correct classified labels.
        n_classes: int: number of classes needed due to possible multiple runs with incomplete class sets
        pe_matrix: pre-existing numpy confusion matrix
    Returns:
        The accuracy of the classifier output.
    """
    # TODO: finish this.͏︆͏󠄃͏󠄌͏󠄍͏󠄂͏️͏︇͏️͏󠄆
    accuracy_value = 0
    if pe_matrix: # 若已经构建好混淆矩阵，直接从混淆矩阵获取正确次数即可
        accuracy_value = np.trace(pe_matrix)
    else:
        for idx in range(len(classifier_output)):
            actual, guess  = true_labels[idx], classifier_output[idx]
            if guess == actual:
                accuracy_value += 1
    return accuracy_value/len(classifier_output)



def gini_impurity(class_vector):
    """Compute the gini impurity for a list of classes.
    This is a measure of how often a randomly chosen element
    drawn from the class_vector would be incorrectly labeled
    if it was randomly labeled according to the distribution
    of the labels in the class_vector.
    It reaches its minimum at zero when all elements of class_vector
    belong to the same class.
    Args:
        class_vector (list(int)): Vector of classes given as 0, 1, 2, ...
    Returns:
        Floating point number representing the gini impurity.
    """
    # TODO: finish this.͏︆͏󠄃͏󠄌͏󠄍͏󠄂͏️͏︇͏️͏󠄆
    vector_len = len(class_vector)
    n_count = {}                   # 分类的出现次数
    for _class in class_vector:
        if n_count.get(_class) == None:
            n_count[_class] = 0
        n_count[_class] += 1
    gini_impurity_value = 1
    for count in n_count.values():
        gini_impurity_value -= (count/vector_len)**2
    return gini_impurity_value


def gini_gain(previous_classes, current_classes):
    """Compute the gini impurity gain between the previous and current classes.
    Args:
        previous_classes (list(int)): Vector of classes given as 0, 1, 2....
        current_classes (list(list(int): A list of lists where each list has
            0, 1, 2, ... values).
    Returns:
        Floating point number representing the gini gain.
    """
    # TODO: finish this.͏︆͏󠄃͏󠄌͏󠄍͏󠄂͏️͏︇͏️͏󠄆
    gini_pre = gini_impurity(previous_classes) # 计算旧分类的基尼不纯度
    gini_cur = 0                     # 当前分类的基尼不纯度
    total = len(previous_classes)
    # 计算当前分类的各队列的基尼不纯度的加权和
    for _classes in current_classes:
        new_size = len(_classes)
        gini_cur += ((new_size/total)*gini_impurity(_classes))
    gini_gain_value = gini_pre - gini_cur
    return gini_gain_value

def entropy(y):
    # 将y值做成字典
    # counter包含键值对，y的取值-y的取值对应的分类个数
    counter = Counter(y)
    res = 0.0
    # 遍历看每一个不同的类别，有多少个样本点
    for num in counter.values():
        p = num / len(y)
        res += -p * math.log(p)
    return res

def split(X, y ,d, value):
    index_a = (X[:, d] <= value) # 定义索引
    index_b = (X[:, d] >value)
    return X[index_a], X[index_b], y[index_a], y[index_b]

def try_spilt(X, y):
    best_entropy = float('inf')
    best_d, best_v = -1, -1 # 维度、阈值
    # 穷搜
    for d in range(X.shape[1]): # 有多少列（特征），shape[0]有多少行，shape[1]有多少列
        sorted_index = np.argsort(X[:, d]) # 返回第d列数据排序后相应的索引
        for i in range(1, len(X)): # 对每一个样本进行遍历
            if (X[sorted_index[i-1], d] != X[sorted_index[i], d]):
                # d这个维度上从1开始，找i-1 和i的中间值
                # 可选值是在d这个维度上的中间值
                v = (X[sorted_index[i-1], d] + X[sorted_index[i], d]) / 2
                # X_l左子树，X_r右子树
                # 进行划分
                X_l, X_r, y_l, y_r = split(X, y ,d, v)
                #  信息熵的和
                e = entropy(y_l) + entropy(y_r)
                # best_entropy:之前搜索过的某一个信息熵
                if e < best_entropy: # 找到更好的划分方式
                    best_entropy, best_d, best_v = e, d, v
    return best_entropy, best_d, best_v

class DecisionTree:
    """Class for automatic tree-building and classification."""

    def __init__(self, depth_limit=22):
        """Create a decision tree with a set depth limit.
        Starts with an empty root.
        Args:
            depth_limit (float): The maximum depth to build the tree.
        """

        self.root = None
        self.depth_limit = depth_limit

    def fit(self, features, classes, attr_num = 0):
        """Build the tree from root using __build_tree__().
        Args:
            features (m x n): m examples with n features.
            classes (m x 1): Array of Classes.
        """

        self.root = self.__build_tree__(features, classes, 0, attr_num)

    def __build_tree__(self, features, classes, depth=0, attr_num = 0):
        """Build tree that automatically finds the decision functions.
        Args:
            features (m x n): m examples with n features.
            classes (m x 1): Array of Classes.
            depth (int): depth to build tree to.
        Returns:
            Root node of decision tree.
        """
        # TODO: finish this.͏︆͏󠄃͏󠄌͏󠄍͏󠄂͏️͏︇͏️͏󠄆
        # 统计各分类的数量
        num_n = Counter(classes)
        keys = list(num_n.keys())
        if len(keys) == 1: # 分类相同可直接确定节点标签
            decision_node = DecisionNode(None, None, None, keys[0])
        elif depth > self.depth_limit: # 当前深度超过决策树的深度限制时，选择分类数量最多的分类确定节点标签
            label = max(num_n, key = lambda key: num_n[key])
            decision_node = DecisionNode(None, None, None, label)
        else:
            break_point = 0             # 最终的正负特征分离值
            num_attributes = len(features[0]) # 属性数量
            num_xs = len(features)        # 分类矢量长度
            p = [_ for _ in range(num_attributes)]
            #  分裂子树时随机抽样attr_num个属性，再进行选择
            if attr_num != 0:
                np.random.shuffle(p)
                p = p[0:attr_num] 
            classes = np.array(classes)
            '''
            best_entropy, attr_index, break_point = try_spilt(features, classes)
            new_neg_split, new_pos_split, new_neg_class, new_pos_class = split(features, classes, attr_index, break_point)
            '''
            best_alpha = 0      # 最大基尼系数增益
            attr_index = -1     # 最大基尼系数增益对应的属性的索引
            new_pos_split = []  # 最终左子树的特征
            new_neg_split = []  # 最终右子树的特征
            new_pos_class = []  # 最终左子树的分类
            new_neg_class = []  # 最终右子树的分类
            # 计算各属性特征的均值作为阈值
            averages = np.mean(features, axis = 0)#(features[:-1] + features[1:])/2
            for idx in p:
            # for average in averages[:,idx]:# 对每个均值判断，找出最佳阈值
                # 用均值对特征进行分割，并保存其对应的分类
                average = averages[idx]
                pos_index = np.where(features[:,idx] > average)
                neg_index = np.where(features[:,idx] <= average)
                pos_feat = features[pos_index]  # 当前属性的正特征
                neg_feat = features[neg_index]  # 当前属性的负特征
                pos_class = classes[pos_index] # 当前属性的正分类
                neg_class = classes[neg_index] # 当前属性的负分类
                
                # 构建新的分类矢量
                new_class = [pos_class, neg_class]
                # 计算先前和当前分类之间的基尼系数增益
                temp_alpha = gini_gain(classes, new_class)
                # 选择使基尼系数增益更大的特征分割及对应的分类
                if temp_alpha > best_alpha:
                    new_pos_split = pos_feat
                    new_neg_split = neg_feat
                    new_pos_class = pos_class
                    new_neg_class = neg_class
                    best_alpha = temp_alpha
                    attr_index = idx
                    break_point = average
            # 增益过小，不分裂
            if best_alpha < 1e-5:
                label = max(num_n, key = lambda key: num_n[key])
                decision_node = DecisionNode(None, None, None, label)
                return decision_node

            # 评估函数（大于分割值时选择左子树）
            func = lambda feature: feature[attr_index] > break_point
            decision_node = DecisionNode(None, None, func)
            # 递归构建子决策树
            decision_node.left = self.__build_tree__(new_pos_split, new_pos_class,  depth+1, attr_num)
            decision_node.right = self.__build_tree__(new_neg_split, new_neg_class,  depth+1, attr_num)
        return decision_node



    def classify(self, features):
        """Use the fitted tree to classify a list of example features.
        Args:
            features (m x n): m examples with n features.
        Return:
            A list of class labels.
        """
        class_labels = []
        # TODO: finish this.͏︆͏󠄃͏󠄌͏󠄍͏󠄂͏️͏︇͏️͏󠄆
        for f_set in features:
            # 使用树对特征进行分类
            val = self.root.decide(f_set)
            class_labels.append(val)
        return class_labels


def generate_k_folds(dataset, k):
    """Split dataset into folds.
    Randomly split data into k equal subsets.
    Fold is a tuple (training_set, test_set).
    Set is a tuple (features, classes).
    Args:
        dataset: dataset to be split.
        k (int): number of subsections to create.
    Returns:
        List of folds.
        => Each fold is a tuple of sets.
        => Each Set is a tuple of numpy arrays.
    """
    folds = []
    # TODO: finish this.͏︆͏󠄃͏󠄌͏󠄍͏󠄂͏️͏︇͏️͏󠄆
    features = dataset[0]  # 数据集的特征
    classes = dataset[1]   # 数据集的分类
    total = len(features)  # 数据集的总量
    n = total // k       # 分割的子集的数据量

    # 将对应的特征和分类组合在一起，以便打乱后其对应关系不变
    shuffle_list = Vectorization().vectorized_glue(features, classes)
    # 随机打乱
    np.random.shuffle(shuffle_list)

    all_index = [_ for _ in range(total)]
    # 分成k份，其中一份作为测试集，构成fold，所有可能取法构成folds用于交叉验证
    for i in range(k):
        test_index = all_index[i*n:(i+1)*n] # 测试集的索引
        train_index = all_index.copy()
        train_index[i*n:(i+1)*n] = []       # 训练集的索引
        train = shuffle_list[train_index]
        test = shuffle_list[test_index]
        train_set = (train[:,:-1], train[:,-1])
        test_set = (test[:,:-1], test[:,-1])
        fold = (train_set, test_set)
        folds.append(fold)
    return folds


class RandomForest:
    """Random forest classification."""

    def __init__(self, num_trees=200, depth_limit=5, example_subsample_rate=.1,
                 attr_subsample_rate=.3):
        """Create a random forest.
         Args:
             num_trees (int): fixed number of trees.
             depth_limit (int): max depth limit of tree.
             example_subsample_rate (float): percentage of example samples.
             attr_subsample_rate (float): percentage of attribute samples.
        """
        self.trees = []
        self.num_trees = num_trees
        self.depth_limit = depth_limit
        self.example_subsample_rate = example_subsample_rate
        self.attr_subsample_rate = attr_subsample_rate

    def fit(self, features, classes):
        """Build a random forest of decision trees using Bootstrap Aggregation.
            features (m x n): m examples with n features.
            classes (m x 1): Array of Classes.
        """
        # TODO: finish this.͏︆͏󠄃͏󠄌͏󠄍͏󠄂͏️͏︇͏️͏󠄆
        m = len(features)
        n = len(features[0])
        # 根据采样率得到子样本的大小
        new_m = int(m * self.example_subsample_rate)
        new_n = int(n * self.attr_subsample_rate)
        # 用于抽样的源
        sample_x_origin = [_ for _ in range(m)]
        #sample_n_origin = [_ for _ in range(n)]
        # 为便于抽样转化为array形式
        features = np.array(features)
        classes = np.array(classes)
        # 子样本的数量num_trees
        for idx in range(self.num_trees):
            # 随机打乱并根据子样本大小选取索引
            np.random.shuffle(sample_x_origin)
            sample_x = sample_x_origin[0:new_m]
            #sample_n = sample_n_origin[0:new_n]
            sample_x.sort()
            #sample_n.sort()
            # 根据索引选取子样本
            new_feat = features[sample_x, :]
            new_class = classes[sample_x]
            # 构建子树
            tree = DecisionTree(self.depth_limit)
            # 传入子样本及属性的抽取数
            tree.fit(new_feat, new_class, new_n)
            # 将子树加入RandomForest
            self.trees.append(tree)


    def classify(self, features):
        """Classify a list of features based on the trained random forest.
        Args:
            features (m x n): m examples with n features.
        Returns:
            votes (list(int)): m votes for each element
        """
        votes = []
        # TODO: finish this.͏︆͏󠄃͏󠄌͏󠄍͏󠄂͏️͏︇͏️͏󠄆
        feat_len = len(features)
        tree_output = []
        
        # 使用子树预测，并收集每个子树的输出
        for i in range(self.num_trees):
            val = self.trees[i].classify(features)
            tree_output.append(val)

        # 根据所有子树的输出确定每个特征对应的分类
        for i in range(feat_len):
            num_n = {}
            # 统计每个子树的输出
            for j in range(self.num_trees):
                val = tree_output[j][i]
                if num_n.get(val) == None:
                    num_n[val] = 0
                num_n[val] += 1
            # 选取输出最多次的分类
            label = max(num_n, key = lambda key: num_n[key])
            votes.append(label)
        return votes


class ChallengeClassifier:
    """Challenge Classifier used on Challenge Training Data."""

    def __init__(self, n_clf=0, depth_limit=0, example_subsample_rt=0.0, \
                 attr_subsample_rt=0.0, max_boost_cycles=0):
        """Create a boosting class which uses decision trees.
        Initialize and/or add whatever parameters you may need here.
        Args:
             num_clf (int): fixed number of classifiers.
             depth_limit (int): max depth limit of tree.
             attr_subsample_rate (float): percentage of attribute samples.
             example_subsample_rate (float): percentage of example samples.
        """
        self.num_clf = n_clf
        self.depth_limit = depth_limit
        self.example_subsample_rt = example_subsample_rt
        self.attr_subsample_rt=attr_subsample_rt
        self.max_boost_cycles = max_boost_cycles
        # TODO: finish this.͏︆͏󠄃͏󠄌͏󠄍͏󠄂͏️͏︇͏️͏󠄆
        self.forest = RandomForest(200, 3, .1, .1)


    def fit(self, features, classes):
        """Build the boosting functions classifiers.
            Fit your model to the provided features.
        Args:
            features (m x n): m examples with n features.
            classes (m x 1): Array of Classes.
        """
        # TODO: finish this.͏︆͏󠄃͏󠄌͏󠄍͏󠄂͏️͏︇͏️͏󠄆
        self.forest.fit(features, classes)

    def classify(self, features):
        """Classify a list of features.
        Predict the labels for each feature in features to its corresponding class
        Args:
            features (m x n): m examples with n features.
        Returns:
            A list of class labels.
        """
        # TODO: finish this.͏︆͏󠄃͏󠄌͏󠄍͏󠄂͏️͏︇͏️͏󠄆
        class_labels = self.forest.classify(features)
        return class_labels



class Vectorization:
    """Vectorization preparation for Assignment 5."""

    def __init__(self):
        pass

    def non_vectorized_loops(self, data):
        """Element wise array arithmetic with loops.
        This function takes one matrix, multiplies by itself and then adds to
        itself.
        Args:
            data: data to be added to array.
        Returns:
            Numpy array of data.
        """

        non_vectorized = np.zeros(data.shape)
        for row in range(data.shape[0]):
            for col in range(data.shape[1]):
                non_vectorized[row][col] = (data[row][col] * data[row][col] +
                                            data[row][col])
        return non_vectorized

    def vectorized_loops(self, data):
        """Array arithmetic using vectorization.
        This function takes one matrix, multiplies by itself and then adds to
        itself.
        Args:
            data: data to be sliced and summed.
        Returns:
            Numpy array of data.
        """
        # TODO: finish this.͏︆͏󠄃͏󠄌͏󠄍͏󠄂͏️͏︇͏️͏󠄆
        vectorized = data * (data + 1)
        return vectorized

    def non_vectorized_slice(self, data):
        """Find row with max sum using loops.
        This function searches through the first 100 rows, looking for the row
        with the max sum. (ie, add all the values in that row together).
        Args:
            data: data to be added to array.
        Returns:
            Tuple (Max row sum, index of row with max sum)
        """
        max_sum = 0
        max_sum_index = 0
        for row in range(100):
            temp_sum = 0
            for col in range(data.shape[1]):
                temp_sum += data[row][col]

            if temp_sum > max_sum:
                max_sum = temp_sum
                max_sum_index = row

        return (max_sum, max_sum_index)

    def vectorized_slice(self, data):
        """Find row with max sum using vectorization.
        This function searches through the first 100 rows, looking for the row
        with the max sum. (ie, add all the values in that row together).
        Args:
            data: data to be sliced and summed.
        Returns:
            Tuple (Max row sum, index of row with max sum)
        """
        # TODO: finish this.͏︆͏󠄃͏󠄌͏󠄍͏󠄂͏️͏︇͏️͏󠄆
        # 搜索前100行
        sum100 = data[0:100].sum(axis=1)
        max_sum = np.max(sum100)                     # 最大行和
        max_sum_index = np.where(max_sum == sum100)  # 对应的索引
        return (max_sum, max_sum_index[0])


    def non_vectorized_flatten(self, data):
        """Display occurrences of positive numbers using loops.
         Flattens down data into a 1d array, then creates a dictionary of how
         often a positive number appears in the data and displays that value.
         ie, [(1203,3)] = integer 1203 appeared 3 times in data.
         Args:
            data: data to be added to array.
        Returns:
            Dictionary [(integer, number of occurrences), ...]
        """
        unique_dict = {}
        flattened = data.flatten()
        for item in flattened:
            if item > 0:
                if item in unique_dict:
                    unique_dict[item] += 1
                else:
                    unique_dict[item] = 1

        return unique_dict.items()

    def vectorized_flatten(self, data):
        """Display occurrences of positive numbers using vectorization.
         Flattens down data into a 1d array, then creates a dictionary of how
         often a positive number appears in the data and displays that value.
         ie, [(1203,3)] = integer 1203 appeared 3 times in data.
         Args:
            data: data to be added to array.
        Returns:
            Dictionary [(integer, number of occurrences), ...]
        """
        # TODO: finish this.͏︆͏󠄃͏󠄌͏󠄍͏󠄂͏️͏︇͏️͏󠄆
        # 水平方向展开data
        flattened = data.flatten() 
        # 返回大于0的元素
        return_list = flattened[flattened > 0]
        # 统计各个数出现的频率
        return Counter(return_list).items()


    def non_vectorized_glue(self, data, vector, dimension='c'):
        """Element wise array arithmetic with loops.
        This function takes a multi-dimensional array and a vector, and then combines
        both of them into a new multi-dimensional array. It must be capable of handling
        both column and row-wise additions.
        Args:
            data: multi-dimensional array.
            vector: either column or row vector
            dimension: either c for column or r for row
        Returns:
            Numpy array of data.
        """
        if dimension == 'c' and len(vector) == data.shape[0]:
            non_vectorized = np.ones((data.shape[0],data.shape[1]+1), dtype=float)
            non_vectorized[:, -1] *= vector
        elif dimension == 'r' and len(vector) == data.shape[1]:
            non_vectorized = np.ones((data.shape[0]+1,data.shape[1]), dtype=float)
            non_vectorized[-1, :] *= vector
        else:
            raise ValueError('This parameter must be either c for column or r for row')
        for row in range(data.shape[0]):
            for col in range(data.shape[1]):
                non_vectorized[row, col] = data[row, col]
        return non_vectorized

    def vectorized_glue(self, data, vector, dimension='c'):
        """Array arithmetic without loops.
        This function takes a multi-dimensional array and a vector, and then combines
        both of them into a new multi-dimensional array. It must be capable of handling
        both column and row-wise additions.
        Args:
            data: multi-dimensional array.
            vector: either column or row vector
            dimension: either c for column or r for row
        Returns:
            Numpy array of data.
        """
        m = len(vector)
        # 将向量转化为numpy数组m*1
        vector_array = np.array([vector])
        if dimension == 'c' and data.shape[0] == m:   # 列向量，将其转置后拼接
            vectorized = np.concatenate((data, np.transpose(vector_array)),axis=1)
        elif dimension == 'r' and data.shape[1] == m: # 行向量，直接拼接
            vectorized = np.concatenate((data, vector_array),axis=0)
        else:
            raise ValueError('This parameter must be either c for column or r for row')
        return vectorized

    def non_vectorized_mask(self, data, threshold):
        """Element wise array evaluation with loops.
        This function takes a multi-dimensional array and then populates a new
        multi-dimensional array. If the value in data is below threshold it
        will be squared.
        Args:
            data: multi-dimensional array.
            threshold: evaluation value for the array if a value is below it, it is squared
        Returns:
            Numpy array of data.
        """
        non_vectorized = np.zeros_like(data, dtype=float)
        for row in range(data.shape[0]):
            for col in range(data.shape[1]):
                val = data[row, col]
                if val >= threshold:
                    non_vectorized[row, col] = val
                    continue
                non_vectorized[row, col] = val**2

        return non_vectorized

    def vectorized_mask(self, data, threshold):
        """Array evaluation without loops.
        This function takes a multi-dimensional array and then populates a new
        multi-dimensional array. If the value in data is below threshold it
        will be squared. You are required to use a binary mask for this problem
        Args:
            data: multi-dimensional array.
            threshold: evaluation value for the array if a value is below it, it is squared
        Returns:
            Numpy array of data.
        """
        vectorized = np.where(data >= threshold, data, data**2)  
        return vectorized

