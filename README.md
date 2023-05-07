# 决策树

机器学习是⼈⼯智能的⼀个⼦领域，决策树是⼀种监督机器学习。在监督学习中，代理将观察样本的输⼊和输出并学习将输⼊映射到输出的函数。

我们将完成以下内容：

1. 通过手绘决策树来感知决策树的构建过程
2. 混淆矩阵、准确率、精确率、召回率
3. 基尼不纯度
4. 构建决策树，使它能够较好地拟合数据集，并对测试数据进行预测
5. kFlod - 交叉验证
6. 随机森林
7. 针对过拟合问题的 "Boosting" 算法

## 拉取仓库

```
git clone https://github.com/BeiYazi0/Decision-Trees
```

## 设置

创建并激活环境

```
conda create -n ai_env python=3.7
conda activate ai_env
```

安装依赖

```
cd Decision-Trees
pip install -r requirements.txt
```

## Jupyter

在`notebook.ipynb`中提供了进一步的说明。运行:

```
jupyter notebook
```
