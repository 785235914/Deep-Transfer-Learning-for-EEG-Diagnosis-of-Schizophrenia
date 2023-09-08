from tensorflow import image
import keras
print(keras.__version__)
from keras.preprocessing import image
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from keras.wrappers.scikit_learn import KerasClassifier
import os
import torchvision.models as models
ResNet50 = models.resnet50(pretrained=True)
import cv2
from keras.layers import Reshape
from keras.optimizers import Adam, RMSprop, SGD
from skopt import BayesSearchCV
from keras.layers import LSTM, Dense, Flatten
from keras.models import Model
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
import numpy as np
from keras.applications import DenseNet121,DenseNet169,DenseNet201
from keras.applications import InceptionV3
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.applications.xception import Xception
from keras.applications.nasnet import NASNetLarge, NASNetMobile
from keras.applications import ResNet50,ResNet101,ResNet152
from keras.applications import MobileNet,MobileNetV2,MobileNetV3Small,MobileNetV3Large
from keras.applications import VGG16,VGG19
import time
from keras.layers import Reshape
from keras.optimizers import Adam, RMSprop, SGD
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import regularizers
from keras.callbacks import ReduceLROnPlateau
from keras.layers import BatchNormalization

# 指定文件夹路径
# 指定文件夹路径
healthy_save_path = '/Users/yuanyi/学习/毕业论文/毕业论文题目/Pycharm_Code/healthy25-new'
schizophrenia_save_path = '/Users/yuanyi/学习/毕业论文/毕业论文题目/Pycharm_Code/schizophrenia25-new'


# 假设你已经完成了小波时频图的生成并保存在本地文件夹中
# 定义“healthy”和“schizophrenia”文件夹路径
healthy_folder = healthy_save_path
schizophrenia_folder = schizophrenia_save_path

def load_images(folder):
    images = []
    labels = []
    for filename in os.listdir(folder):
        if filename.endswith(".png"):
            img_path = os.path.join(folder, filename)
            # img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # 以灰度模式读取图片
            img = cv2.imread(img_path, cv2.IMREAD_COLOR)  # 以RGB模式读取图片
            img_resized = cv2.resize(img, (224, 224))  # 调整图像大小到64x64
            img_array = img_resized / 255.0  # 将像素值归一化到[0, 1]
            images.append(img_array)
            labels.append(0 if folder == healthy_folder else 1)
    return np.array(images), np.array(labels)

# 加载和处理图像
X_healthy, y_healthy = load_images(healthy_folder)
X_schizophrenia, y_schizophrenia = load_images(schizophrenia_folder)

# 将数据集合并
X = np.concatenate((X_healthy, X_schizophrenia), axis=0)
y = np.concatenate((y_healthy, y_schizophrenia), axis=0)

# 打乱数据集
indices = np.arange(X.shape[0])
np.random.shuffle(indices)
X = X[indices]
y = y[indices]

start_time = time.time()

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
# 原始数据集的形状
input_shape = (224, 224, 3)


def create_model(optimizer='adam',activation='sigmoid',units=128, dropout=0.2,
                 recurrent_dropout=0.2,learning_rate=0.001,epochs=100, batch_size=64):
    # 加载预训练的MobileNet模型
    base_model = MobileNet(weights='imagenet', include_top=False, input_shape=input_shape)

    # 设定不更新预训练模型的权重
    for layer in base_model.layers:
        layer.trainable = False

    # 获取base_model的输出，然后添加新的层
    base_model_output = base_model.output

    # 获取模型最后一层的输出形状
    output_shape = base_model.output_shape
    print('Output Shape: ', output_shape)

    # 输出形状是一个形状元组，其形式为 (None, height, width, depth)
    # 其中 height, width, depth 是特征图的高度，宽度和深度（也就是特征的数量）
    # 这里我们只关心深度，因为它是我们要传入 LSTM 的特征数量

    features_dim = output_shape[-1]  # 提取深度维度
    print('Features Dimension: ', features_dim)

    # 添加Reshape层
    # 我们可以使用从 base_model 获取的 features_dim 来重新塑造特征
    x = Reshape((-1, features_dim))(base_model_output)
    print('Reshape : ', x)

    # 添加LSTM层
    x = LSTM(units, dropout=dropout, recurrent_dropout=recurrent_dropout)(x)
    # 批量归一化 (Batch Normalization): 在模型中添加批量归一化层可以帮助改善训练速度和模型的性能。
    x = BatchNormalization()(x)
    x = Dense(1, activation='sigmoid')(x)

    # 构建最终模型
    model = Model(inputs=base_model.input, outputs=x)

    if optimizer == 'adam':
        optimizer = Adam(learning_rate=learning_rate)
    elif optimizer == 'rmsprop':
        optimizer = RMSprop(learning_rate=learning_rate)
    elif optimizer == 'sgd':
        optimizer = SGD(learning_rate=learning_rate)

    # 编译模型
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

    # 训练模型
    # 在模型训练过程中添加这两个回调
    model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=epochs, batch_size=batch_size)
    return model

# 定义LSTM模型为KerasClassifier，并传递新函数
lstm_model = KerasClassifier(build_fn=create_model, epochs=20, batch_size=64, verbose=0)

# 定义LSTM模型的超参数搜索空间

lstm_param_dist = {
    'epochs': [100],
    'batch_size': [128],
    'units': [128],
    'dropout': [0.2],
    'recurrent_dropout': [0.2],
    'activation': ['sigmoid'],
    'optimizer': ['rmsprop'],
    'learning_rate': [0.001] # 添加学习率作为超参数
    }
#
# lstm_param_dist = {'units': 128,
#                    'recurrent_dropout': 0.2,
#                    'optimizer': 'rmsprop',
#                    'learning_rate': 0.001,
#                    'epochs': 100,
#                    'dropout': 0.2,
#                    'batch_size': 128,
#                    'activation': 'sigmoid'}

# 定义k折交叉验证
kfold = StratifiedKFold(n_splits=10, shuffle=True)

# 使用随机搜索进行LSTM模型参数调优
random_search_lstm = RandomizedSearchCV(lstm_model, param_distributions=lstm_param_dist, n_iter=50, cv=kfold, verbose=2)
random_search_lstm.fit(X_train, y_train)
random_search_lstm.best_estimator_.model.save('LabTestModel25s/224-InceptionResNetV2-Lstm-Random-Model.h5')

# 输出LSTM模型的最佳参数和得分
print("Best parameters: ", random_search_lstm.best_params_)
print("Best accuracy: ", random_search_lstm.best_score_)


lstm_best_params = random_search_lstm.best_params_

# 使用最佳参数创建LSTM模型
lstm_best_model = create_model(optimizer=lstm_best_params['optimizer'],
                               activation=lstm_best_params['activation'],
                               units=lstm_best_params['units'],
                               dropout=lstm_best_params['dropout'],
                               recurrent_dropout=lstm_best_params['recurrent_dropout'],
                               learning_rate=lstm_best_params['learning_rate'],
                               epochs=lstm_best_params['epochs'],
                               batch_size=lstm_best_params['batch_size'])
# 训练模型
lstm_best_model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=lstm_best_params['epochs'], batch_size=lstm_best_params['batch_size'], verbose=1)

# 获取训练集上的损失和精确度
train_loss, train_accuracy = lstm_best_model.evaluate(X_train, y_train, batch_size=lstm_best_params['batch_size'])

# 获取验证集上的损失和精确度
val_loss, val_accuracy = lstm_best_model.evaluate(X_test, y_test, batch_size=lstm_best_params['batch_size'])

# lstm_best_model.save('LabTestModel25s/224-MobileNet-Lstm-Best-Model.h5')

# lstm_best_model.save('ModelROCPlots/MobileNet(0.86)-LSTM_Best_Model(224-25s).h5')

print("Train Loss: ", train_loss)
print("Train Accuracy: ", train_accuracy)
print("Validation Loss: ", val_loss)
print("Validation Accuracy: ", val_accuracy)

# 输出Transfer-CNN 模型的最佳参数和得分
print("Best NASNetMobile-25s parameters: ", random_search_lstm.best_params_)
print("Best NASNetMobile-25s accuracy: ", random_search_lstm.best_score_)

end_time = time.time()
elapsed_time = end_time - start_time

hours, rem = divmod(elapsed_time, 3600)
minutes, seconds = divmod(rem, 60)
print("Total execution time: {} seconds".format(end_time - start_time))
print("Total execution time: {:0>2}:{:0>2}:{:05.2f}".format(int(hours),int(minutes),seconds))


from sklearn.metrics import roc_auc_score, f1_score
from sklearn.metrics import precision_score, recall_score, accuracy_score
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

best_model = lstm_best_model
lstm_best_model.save('ModelROCPlots/MobileNet(0.86)-LSTM_Best_Model(224-25s).h5')

# 使用模型进行预测
y_pred = best_model.predict(X_test)
y_train_pred = best_model.predict(X_train)

# 计算ROC曲线的各个点
fpr, tpr, _ = roc_curve(y_test, y_pred)
fpr_train, tpr_train, _ = roc_curve(y_train, y_train_pred)

# 计算AUC
roc_auc = auc(fpr, tpr)
roc_auc_train = auc(fpr_train, tpr_train)

# 绘制ROC曲线
plt.figure()
plt.plot(fpr, tpr, color='darkorange', label='Test ROC curve (area = %0.2f)' % roc_auc)
plt.plot(fpr_train, tpr_train, color='red', label='Train ROC curve (area = %0.2f)' % roc_auc_train)
plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('MobileNet - Receiver Operating Characteristic')
plt.legend(loc="lower right")

save_path = '/Users/yuanyi/学习/毕业论文/毕业论文题目/Pycharm_Code/ModelROCPlots'
# 保存图像
plt.savefig(f'{save_path}/MobileNet(0.90) - Receiver Operating Characteristic.png')

plt.show()


from sklearn.metrics import precision_recall_curve, auc
from sklearn.model_selection import learning_curve
import matplotlib.pyplot as plt
import numpy as np

# 假设 y_test 是测试集的真实标签，y_scores 是模型的预测分数

# 精确度-召回率曲线
precision, recall, _ = precision_recall_curve(y_test, y_pred)
plt.figure()
plt.plot(recall, precision, label='Precision-Recall curve')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('MobileNet - Precision-Recall curve')
plt.legend(loc="lower left")

# 保存图像
plt.savefig(f'{save_path}/MobileNet(0.90) - Precision-Recall Curve.png')

plt.show()


# 将预测结果二值化，即将预测结果转换为0或1
y_pred = np.where(y_pred > 0.5, 1, 0)
# print("y_pred: ", y_pred)

# 计算准确率
accuracy = accuracy_score(y_test, y_pred)

# 计算F1得分
f1 = f1_score(y_test, y_pred)

# 计算AUC得分
auc = roc_auc_score(y_test, y_pred)

# 计算精度
precision = precision_score(y_test, y_pred)

# 计算召回率
recall = recall_score(y_test, y_pred)

print("Accuracy: ", accuracy)
print("F1 Score: ", f1)
print("AUC: ", auc)
print("Precision: ", precision)
print("Recall: ", recall)


# 224-MobieNet
# Train Loss:  0.00048483689897693694
# Train Accuracy:  1.0
# Validation Loss:  0.39872947335243225
# Validation Accuracy:  0.9212828278541565
# Best MobieNet-CNN parameters:  {'units': 128, 'recurrent_dropout': 0.2, 'optimizer': 'rmsprop', 'learning_rate': 0.001, 'epochs': 100, 'dropout': 0.2, 'batch_size': 128, 'activation': 'sigmoid'}
# Best MobieNet-CNN accuracy:  0.998734176158905

# Accuracy:  0.9067055393586005
# F1 Score:  0.9170984455958548
# AUC:  0.9065488565488564
# Precision:  0.9267015706806283
# Recall:  0.9076923076923077
# Train Loss:  0.00013510501594282687
# Train Accuracy:  1.0
# Validation Loss:  0.48682934045791626
# Validation Accuracy:  0.9067055583000183
# Best NASNetMobile-25s parameters:  {'units': 128, 'recurrent_dropout': 0.2, 'optimizer': 'rmsprop', 'learning_rate': 0.001, 'epochs': 100, 'dropout': 0.2, 'batch_size': 128, 'activation': 'sigmoid'}
# Best NASNetMobile-25s accuracy:  0.9987500011920929