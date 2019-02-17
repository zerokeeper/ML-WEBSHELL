# ML-WEBSHELL

利用机器学习做的webshell检测demo

黑样本使用的是ysrc/webshell-sample的php样本

白样本是wordpress



使用1-gram分词的正则匹配规则，基于函数调用特征

'\b\w+\b\(|\'\w+\''

分别使用了朴素贝叶斯、决策树、支持向量机、k近邻算法做预测

结果如下

![ml-webshell](https://github.com/zerokeeper/ML-WEBSHELL/blob/master/ml-webshell.png)
