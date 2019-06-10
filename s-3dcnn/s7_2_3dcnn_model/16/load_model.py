from keras.models import load_model
import keras.backend as K
from keras.utils import np_utils
import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt
#from spectral import *
import cv2
import sys
sys.path.append('../')
from data_pretreat import handle_data, get_rawdata, normalize, zero_pad, bf

K.set_image_data_format('channels_last')
K.set_learning_phase(0)

train_scale=0.3
kernel_size=7
n_classes=16
bf=False
model_path='s7_2_3dcnn_model'

# 获得数据
x_train,y_train,x_test,y_test=handle_data(train_scale,kernel_size,bf)
f = open(model_path+'.txt','a')
f.writelines('trainning scale:%f\n'%(train_scale))
f.writelines('kernel size:%d\n'%(kernel_size))
f.close()

# expand_dims
x_train=np.expand_dims(x_train,axis=4)
x_test=np.expand_dims(x_test,axis=4)

#one-hot
y_train=np_utils.to_categorical(y_train)[:,1:17]
y_eval =np_utils.to_categorical(y_test)[:,1:17]
print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_eval.shape)

model=load_model(model_path+'.h5')

# 测试集准确率
def cal_test_acc():
    f = open(model_path+'.txt','a')
    preds = model.evaluate(x_test, y_eval)
    # print ("Loss = " + str(preds[0]) + '\n')
    # print ("Test Accuracy = " + str(preds[1]) + '\n')
    f.writelines('Load model and test accuracy:\n')
    f.writelines("Loss = " + str(preds[0]) + '\n')
    f.writelines("Test Accuracy = " + str(preds[1]) + '\n')
    f.close()


# 数据处理
def data_process():
# 分类报告
    f = open(model_path+'.txt','a')
    
    y_pred = (np.argmax(model.predict(x_test),axis=1)+1).astype('float32')
    y_true = np.squeeze(y_test)
    # target_names = ['Alfalfa','Corn-notill','Corn-mintill','Corn','Grass-pasture','Grass-trees',
    # 'Grass-pasture-mowed','Hay-windrowed','Oats','Soybean-notill','Soybean-mintill','Soybean-clean',
    # 'Wheat','Woods','Buildings-Grass-Trees-Drives','Stone-Steel-Towers']
    classify_report = metrics.classification_report(y_true, y_pred)
    f.writelines('classify_report:\n')
    print(classify_report, file=f)

    # Overall Accuracy
    overall_accuracy = metrics.accuracy_score(y_true, y_pred)
    # print('overall_accuracy: {0:f}'.format(overall_accuracy))
    f.writelines('overall_accuracy: %f \n' %(overall_accuracy))

    # Average Accuracy
    acc_for_each_class = metrics.precision_score(y_true, y_pred, average='macro')
    # print('acc_for_each_class : ', acc_for_each_class)
    f.writelines('average_accuracy: %f \n' %(acc_for_each_class))

    # kappa score
    kappa = metrics.cohen_kappa_score(y_true, y_pred)
    # print('kappa score : ', kappa)
    f.writelines('kappa score: %f \n' %(kappa))
    f.close()

    # 绘制混淆矩阵
    confusion_matrix = metrics.confusion_matrix(y_true, y_pred)
    #print('confusion_matrix : \n', confusion_matrix)
    classes = list(set(y_true))
    classes.sort()
    plt.style.use('ggplot')
    plt.imshow(confusion_matrix, cmap=plt.get_cmap('PuBuGn'))
    indices = range(len(confusion_matrix))
    plt.xticks(indices, list(range(1,17)))
    plt.yticks(indices, list(range(1,17)))
    plt.colorbar()
    plt.xlabel('True')
    plt.ylabel('Pred')
    for first_index in range(len(confusion_matrix)):
        for second_index in range(len(confusion_matrix[first_index])):
            plt.text(first_index, second_index, confusion_matrix[first_index][second_index], 
            fontsize=6, style='oblique', ha='center', va='center', wrap=True)
    plt.savefig('confusion_matrix.png',dpi=300)


# 绘制原图和识别图
def draw_pic(kernel_size,bf):
    x_raw,y_gt = get_rawdata()
    x_raw=normalize(x_raw)
    if bf:
        x_raw = bf(x_raw)
    # 保存rgb图像和原始gt图像 用spectral库
    # save_rgb('rgb.jpg', x_raw, [29, 19, 9])
    # save_rgb('gt.jpg', y_gt, colors=spy_colors)

    # 三幅图 rgb，gt，pred
    plt.subplot(2,2,1)
    img_rgb = np.zeros((x_raw.shape[0],x_raw.shape[1],3))
    img_rgb[:,:,0]=x_raw[:,:,9]
    img_rgb[:,:,1]=x_raw[:,:,19]
    img_rgb[:,:,2]=x_raw[:,:,29]
    plt.grid(False)
    plt.imshow(img_rgb)

    # gt
    plt.subplot(2,2,3)
    plt.grid(False)
    plt.imshow(y_gt,cmap=plt.get_cmap('Spectral_r'))

    # pred
    plt.subplot(2,2,4)
    plt.grid(False)
    pad = (kernel_size - 1) // 2
    data_x_pad = zero_pad(x_raw,pad)
    y_pred_matrix = np.zeros(y_gt.shape)
    for i in range(y_gt.shape[0]):
        for j in range(y_gt.shape[1]):
            if y_gt[i,j] != 0:
                x_cube = np.zeros((1,kernel_size,kernel_size,x_raw.shape[2]))
                x_cube[0,:,:,:] = data_x_pad[i:i+kernel_size,j:j+kernel_size,:]
                x_cube = np.expand_dims(x_cube,axis=4)
                y_pred_matrix[i,j] = (np.argmax(model.predict(x_cube),axis=1)+1).astype('float32')[0]

    plt.imshow(y_pred_matrix,cmap=plt.get_cmap('Spectral_r'))
    plt.savefig('gt_vs_pred.png',dpi=300)

def bf_vs_gt():
    x_raw,_ = get_rawdata()
    x_raw=normalize(x_raw)
    img_one = x_raw[:, :, 34]
    plt.subplot(1, 2, 1)
    plt.imshow(img_one)
    img_new = cv2.bilateralFilter(img_one, 7, 50, 50)
    plt.subplot(1, 2, 2)
    plt.imshow(img_new)
    plt.savefig('bf_vs_gt.png')


if __name__ == '__main__':
    cal_test_acc()
    data_process()
    draw_pic(kernel_size,bf)
    # bf_vs_gt()
