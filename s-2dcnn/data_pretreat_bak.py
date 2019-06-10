from scipy.io import loadmat
import numpy as np

def handle_data(train_scale=0.5):
    data_corrected=loadmat('Indian_pines_corrected.mat')
    data_gt=loadmat('Indian_pines_gt.mat')
    indian_pines_corrected=data_corrected['indian_pines_corrected']
    indian_pines_gt=data_gt['indian_pines_gt']
    #print(indian_pines.shape)
    #print(indian_pines_corrected.shape)
    #print(indian_pines_gt.shape)
    data_x=indian_pines_corrected.reshape(21025,200)
    data_y=indian_pines_gt.reshape(21025,1)
    #print(data_x.shape)
    #print(data_y.shape)

    #训练集比例
    #train_scale=0.5
    train_class=16
    train_x=np.zeros((0,200))
    train_y=np.zeros((0,1))
    test_x=np.zeros((0,200))
    test_y=np.zeros((0,1))


    for i in range (1,17):
        #print(data_y[0:100,0])
        np.random.seed()
        #提取每类数据
        temp_y=np.array(data_y[:,0]==i).astype(int).reshape(-1,1)
        temp_y=np.multiply(temp_y,data_y)
        y=temp_y[np.flatnonzero(temp_y),:]
        x=data_x[np.flatnonzero(temp_y),:]
        #print(y.shape)
        #print(x[:,1])

        #打乱数据集
        permutation = np.random.permutation(y.shape[0])
        x = x[permutation, :]
        #print(x[:,1])

        #构造训练集和测试集
        train_set_number=int(y.shape[0]*train_scale)
        train_x_batch=x[0:train_set_number,:]
        test_x_batch=x[train_set_number:,:]
        train_y_batch=y[0:train_set_number,:]
        test_y_batch=y[train_set_number:,:]

        train_x=np.concatenate((train_x,train_x_batch), axis=0)
        train_y=np.concatenate((train_y,train_y_batch), axis=0)
        test_x=np.concatenate((test_x,test_x_batch), axis=0)
        test_y=np.concatenate((test_y,test_y_batch), axis=0)   

    #打乱训练集
    np.random.seed()
    per_train_all = np.random.permutation(train_y.shape[0])
    train_x = train_x[per_train_all, :]
    train_y = train_y[per_train_all, :]
    
    return train_x,train_y,test_x,test_y
