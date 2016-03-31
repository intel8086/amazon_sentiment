# amazon_sentiment
典型的文本分类，过程如下：

1.数据加载：
  使用pandas 从sqlite模块中读取数据，并对文本分类。划分测试集和数据集。

2.数据清洗和提feature
  对文本使用分词工具，stopwords集等做数据清洗，然后使用CountVectorizer和TfidfTransformer,将其转换为训练矩阵。

3.使用LR,SVM和朴素贝叶斯进行分类

LR部分结果：
             precision    recall  f1-score   support

   positive       0.80      0.69      0.74     16379
   negative       0.94      0.97      0.96     88784

avg / total       0.92      0.93      0.92    105163

SVM部分结果：
             precision    recall  f1-score   support

   positive       0.79      0.70      0.74     16632
   negative       0.93      0.96      0.95     88531

avg / total       0.91      0.93      0.91    105163
