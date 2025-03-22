import os
import matplotlib.pyplot as plt
import sys
# print("python version: {}". format(sys.version))
import numpy as np
# print("numpy version: {}". format(np.__version__))
import pandas as pd
# print("pandas version: {}". format(pd.__version__))
# Model Algorithms
import lightgbm as lgb
# print("lightgbm version: {}". format(lgb.__version__))
# Common Model Helpers
import sklearn
# print("sklearn version: {}". format(sklearn.__version__))
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score,precision_score, recall_score, f1_score, roc_auc_score
# 读取原始数据
data_path = './datasets//'

train = pd.read_csv(os.path.join(data_path, 'train.csv'))
test = pd.read_csv(os.path.join(data_path, 'test.csv'))
train

# 训练特征使用除了id和目标变量之外的所有特征
ycol = 'label'
features = [x for x in train.columns if x not in [ycol, 'id']]
features

# 问题类型:确定本课题为9分类问题，所以设定代码中的num_class=9，5折
NFOLD = 5
num_class = 9
random_state = 2021
KF = StratifiedKFold(n_splits=NFOLD, shuffle=True, random_state=random_state)

# 自定义一个多分类的准确率验证函数
def custom_accuracy_eval(y_hat, data, num_class=num_class):
    y_true = data.get_label()
    y_hat = y_hat.reshape(num_class, -1).T
    return 'accuracy', accuracy_score(y_true, np.argmax(y_hat, axis=1)), True

params_lgb = {
    'boosting':'gbdt',
    'objective':'multiclass',
    'num_class': num_class,
    'metric':'multi_logloss',
    'first_metric_only':True,
    'force_row_wise': True,
    'random_state':random_state,
    'learning_rate':0.05,
    'subsample':0.8,
    'subsample_freq':3,
    'colsample_bytree':0.8,
    'max_depth':6,
    'num_leaves':31,
    'n_jobs':-1,
    'verbose': -1,
}

oof_lgb = np.zeros([len(train), num_class])
predictions_lgb = np.zeros([len(test), num_class])
df_importance_list = []

# 五折交叉验证
for fold_, (trn_idx, val_idx) in enumerate(KF.split(train[features], train[ycol])):
    print('-------------------- fold {} --------------------'.format(fold_+1))
    trn_data = lgb.Dataset(train.iloc[trn_idx][features], label=train.iloc[trn_idx][ycol])
    val_data = lgb.Dataset(train.iloc[val_idx][features], label=train.iloc[val_idx][ycol], reference=trn_data)

    clf_lgb = lgb.train(
        params=params_lgb,
        train_set=trn_data,
        valid_sets=[trn_data, val_data],
        valid_names=('train', 'val'),
        num_boost_round=50000,
        early_stopping_rounds=200,
        verbose_eval=100,
        feval=custom_accuracy_eval,
    )

    oof_lgb[val_idx] = clf_lgb.predict(train.iloc[val_idx][features], num_iteration=clf_lgb.best_iteration)
    predictions_lgb[:] += (clf_lgb.predict(test[features], num_iteration=clf_lgb.best_iteration) / NFOLD)

    df_importance = pd.DataFrame({
        'column': features,
        'importance_split': clf_lgb.feature_importance(importance_type='split'),
        'importance_gain': clf_lgb.feature_importance(importance_type='gain'),
    })
    df_importance_list.append(df_importance)


valid_accuracy_score = accuracy_score(train[ycol], np.argmax(oof_lgb, axis=1))
valid_accuracy_score

# 特征重要性
df_importance = pd.concat(df_importance_list)
df_importance = df_importance.groupby('column').agg('mean').reset_index()
df_importance.sort_values('importance_gain', ascending=False)

# 做出最终预测
test['label'] = np.argmax(predictions_lgb, axis=1)
test[['id', 'label']].to_csv('./submit.csv', index=False)

# 计算各项评估指标
accuracy = accuracy_score(train[ycol], np.argmax(oof_lgb, axis=1))
precision = precision_score(train[ycol], np.argmax(oof_lgb, axis=1), average='macro')
recall = recall_score(train[ycol], np.argmax(oof_lgb, axis=1), average='macro')
f1 = f1_score(train[ycol], np.argmax(oof_lgb, axis=1), average='macro')
auc = roc_auc_score(train[ycol], oof_lgb, average='macro', multi_class='ovo')
# 获取LightGBM模型的学习率
learning_rate = params_lgb['learning_rate']

# 添加到 evaluation DataFrame
evaluation = pd.DataFrame({
    'Model': ['LightGBM'],
    '学习率': [learning_rate],
    '准确率': [accuracy],
    '精确率': [precision],
    '召回率': [recall],
    'F1 值': [f1],
    'AUC值': [auc],
    '5折交叉验证的score': [valid_accuracy_score]  # 您已经计算得到的 5 折交叉验证的得分
})
# 保存为 evaluation.csv 文件
evaluation.to_csv('evaluation.csv', index=False)


# 生成可视化图
# 混交矩阵
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

cm = confusion_matrix(train[ycol], np.argmax(oof_lgb, axis=1))
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
# plt.show()
plt.savefig('./results/hunjiaojuzhen.png')

#ROC曲线
from sklearn.metrics import roc_curve, auc

fpr, tpr, thresholds = roc_curve(train[ycol], oof_lgb[:, 1], pos_label=1)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (AUC = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
# plt.show()
plt.savefig('./results/roc.png')

# #特征重要性可视化
# plt.figure(figsize=(9, 18))  # 调整图表大小
#
# # 绘制特征重要性图表并旋转标签
# ax = sns.barplot(x='importance_gain', y='column', data=df_importance.sort_values('importance_gain', ascending=False))
# plt.title('Feature Importance (Gain)')
# plt.xlabel('Feature Importance')
# plt.ylabel('Features')
# plt.yticks(rotation=45)  # 旋转标签
#
# plt.tight_layout()  # 调整布局，防止标签重叠
# plt.savefig('./results/tezheng.png')
# # plt.show()

#Leraning Curve学习曲线
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve

def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None, n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5)):
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    return plt

# 使用你的数据和模型
# 请确保定义了 X_train 和 y_train，以及你的分类器 clf
X_train = train[features]  # 假设 'features' 是您想要用于训练的特征列表
y_train = train[ycol]  # 假设 'ycol' 是标签所在的列名称
clf = lgb.LGBMClassifier(**params_lgb)  # 使用您之前定义的 params_lgb 参数实例化分类器
clf.fit(X_train, y_train)  # 在您的数据上拟合分类器
title = "Learning Curves (Your Classifier)"
# 绘制学习曲线
plot_learning_curve(clf, title, X_train, y_train, ylim=(0.7, 1.01), cv=KF, n_jobs=-1)

plt.savefig('./results/learning_curve.png')




