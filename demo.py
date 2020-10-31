import argparse
import scipy.io
import torch
import numpy as np
import os
from torchvision import datasets
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from collections import Counter
#######################################################################
# Evaluate
parser = argparse.ArgumentParser(description='Demo')
parser.add_argument('--query_index',default='12', type=int, help='test_image_index')
parser.add_argument('--test_dir',default='./person',type=str, help='./test_data')
parser.add_argument('--camera',default='c1', type=str, help='test_image_index')
opts = parser.parse_args()

data_dir = opts.test_dir
image_datasets = {x: datasets.ImageFolder( os.path.join(data_dir,x) ) for x in ['gallery','query']}
#print(image_datasets)
#####################################################################
#Show result
def imshow(path, title=None):
    """Imshow for Tensor."""
    im = plt.imread(path)
    plt.imshow(im)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated


def sort_img(qf, ql, qc, gf, gl, gc):
    query = qf.view(-1,1)#维度n*1
    score = torch.mm(gf,query)#矩阵相乘
    score = score.squeeze(1).cpu()#np.squeeze这个函数的作用是去掉矩阵里维度为1的维度
    score = score.numpy()
    # predict index
    index = np.argsort(score)  #from small to large
    #numpy.argsort(a, axis=-1, kind='quicksort', order=None)
    #返回数组排序后对应的下标。kind是排序算法，axis是排序的轴。
    index = index[::-1]
    # index = index[0:2000]
    # good index
    query_index = np.argwhere(gl==ql)#满足gl==ql的位置
    
    #same camera#  
    camera_index = np.argwhere(gc==qc) 
    junk_index = np.intersect1d(query_index, camera_index)#交集
    mask = np.in1d(index, junk_index, invert=True)
    index = index[mask]
    return index


def reindex(num):
    if num <= 200:
        return num,'c1'
    elif num <= 400:
        return num-200,'c2'
    else:
        return num-400,'c3'
    
def main():
        ######################################################################
    result = scipy.io.loadmat('pytorch_result.mat')
    query_feature = torch.FloatTensor(result['query_f'])
    query_cam = result['query_cam'][0]
    query_label = result['query_label'][0]
    gallery_feature = torch.FloatTensor(result['gallery_f'])
    gallery_cam = result['gallery_cam'][0]
    gallery_label = result['gallery_label'][0]
    
    multi = os.path.isfile('multi_query.mat')
    
    if multi:
        m_result = scipy.io.loadmat('multi_query.mat')
        mquery_feature = torch.FloatTensor(m_result['mquery_f'])
        mquery_cam = m_result['mquery_cam'][0]
        mquery_label = m_result['mquery_label'][0]
        mquery_feature = mquery_feature.cuda()
    
    query_feature = query_feature.cuda()
    gallery_feature = gallery_feature.cuda()
    
    #######################################################################
    
# sort the images
    indexid = (int(opts.camera[1])-1)*200+int(opts.query_index)
    root_path = opts.test_dir+'\\query\\'
    ids = os.listdir(root_path)
    #print(ids)
    i=ids.index('%04d'%indexid) #c1 1 --0001
    #i = opts.query_index-1
    index = sort_img(query_feature[i],query_label[i],query_cam[i],gallery_feature,gallery_label,gallery_cam)
    
    ########################################################################
    # Visualize the rank result
    
    query_path, _ = image_datasets['query'].imgs[i]
    query_label = query_label[i]
    print(query_path)
    print('Top 10 images are as follow:')
    ids = []
    try: # Visualize Ranking Result 
        # Graphical User Interface is needed
        fig = plt.figure(figsize=(16,4))
        ax = plt.subplot(1,11,1)
        ax.axis('off')
        imshow(query_path,'query%s'%opts.camera+'_ID%s'%opts.query_index)
        for i in range(10):
            ax = plt.subplot(1,11,i+2)
            ax.axis('off')
            img_path, _ = image_datasets['gallery'].imgs[index[i]]
            label = gallery_cam[index[i]]
            imshow(img_path)
            ID = str(reindex(gallery_label[index[i]])[0])
            ax.set_title('C%s_'%label +'ID%s'%ID, color='green')
        for j in range(350):
            ids.append(gallery_label[index[j]])
        c = Counter(ids)
        d = c.most_common(4)
        print('Most common:'+str(d))
        index1 = d[0][0] if d[0][1] > 80 else 999
        index2 = d[1][0] if d[1][1] > 58 else 999 #999 for none
        if index1 == 999:
            print('No such target was found!')
        else:
            print('ID:'+str(reindex(index1)[0])+' Camera:'+str(reindex(index1)[1]))
        if index2 == 999:
            print('No such target was found!')  
        else:
            print('ID:'+str(reindex(index2)[0])+' Camera:'+str(reindex(index2)[1]))
    except RuntimeError:
        for i in range(10):
            img_path = image_datasets.imgs[index[i]]
            print(img_path[0])
        print('If you want to see the visualization of the ranking result, graphical user interface is needed.')
    
    fig.savefig("show.png")
    return reindex(index1),reindex(index2)
    
if __name__ == '__main__':
    main()