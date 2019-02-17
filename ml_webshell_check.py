#coding=utf-8

import os
import numpy as np
from sklearn import tree
from sklearn import linear_model
from sklearn.naive_bayes import GaussianNB
from sklearn import svm
from sklearn import neighbors
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer

# 1-gram算法的正则匹配规则，基于函数调用特征
r_token_pattern=r'\b\w+\b\(|\'\w+\''

def load_file(file_path):
	t = ""
	with open(file_path) as f:
		for line in f:
			line = line.strip('\n')
			t += line
	return t
 
def load_files(path):
	files_list = []
	for parent, dirs, files in os.walk(path):
		for file in files:
			if file.endswith('.php'):
				file_path = parent + '/' + file
				# print "[*]Loading: %s" % file_path
				t = load_file(file_path)
				files_list.append(t)
	return files_list





# ngram_range设置为(2,2)表示以基于单词切割的2-gram算法生成词汇表因而token_pattern的正则为匹配单个单词，decode_error设置为忽略其他异常字符的影响，
# ngram_range设置为(1,1)表示以基于函数和字符串常量的1-gram算法生成词汇表因而token_pattern的正则为匹配函数调用特征

webshell_bigram_vectorizer = CountVectorizer(ngram_range=(1, 1), decode_error="ignore",
                                    token_pattern = r_token_pattern,min_df=1)
# 加载WebShell黑样本

webshell_files_list=load_files("data/black/webshell-sample-master/php")
# 将现有的词袋特征进行向量化
x1=webshell_bigram_vectorizer.fit_transform(webshell_files_list).toarray()
y1=[1]*len(x1)

# 定义词汇表
vocabulary=webshell_bigram_vectorizer.vocabulary_

# vocabulary参数是使用黑样本生成的词汇表vocabulary将白样本特征化
wp_bigram_vectorizer = CountVectorizer(ngram_range=(1, 1), decode_error="ignore",
                                    token_pattern = r_token_pattern,min_df=1,vocabulary=vocabulary)
wp_files_list=load_files("data/white/wordpress")
x2=wp_bigram_vectorizer.transform(wp_files_list).toarray()
y2=[0]*len(x2)

# 拼接数组

X=np.concatenate((x1,x2))
y=np.concatenate((y1, y2))

# print x,y

#划分为训练集和测试集数据,利用随机种子random_state采样30%的数据作为测试集。
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,random_state=33)



# 朴素贝叶斯
gnb = GaussianNB()
gnb.fit(X_train, y_train)
y_predict=gnb.predict(X_test)
score = np.mean(y_predict == y_test)
print '朴素贝叶斯准确率：',score

# 决策树

dtc = tree.DecisionTreeClassifier()
dtc.fit(X_train,y_train)
y_predict = dtc.predict(X_test)
score = np.mean(y_predict==y_test)
print '决策树准确率：',score


# 逻辑回归
# lr = linear_model.LinearRegression()
# lr.fit(X_train,y_train)
# y_predict = lr.predict(X_test)
# # 逻辑回归出来的结果不是0和1而是浮点数一个范围,效果不好
# # print y_predict
# score = lr.score(X_test,y_test)
# print '逻辑回归准确率：',score


# 支持向量机
svc = svm.SVC()
svc.fit(X_train,y_train)
y_predict = svc.predict(X_test)
score = np.mean(y_predict==y_test)
print '支持向量机准确率：',score

# k近邻

knn = neighbors.KNeighborsClassifier()
knn.fit(X_train,y_train)
y_predict = knn.predict(X_test)
score = np.mean(y_predict==y_test)
print 'k近邻准确率：',score