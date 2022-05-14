#!/usr/bin/env python
# coding: utf-8

# # PCA 主成分分析

# In[10]:


get_ipython().system('pwd')


# ## 4.DCNN(Deep Convolutional Neural Network)特徴量．
# - PyTorchでVGG16などで最終レイヤーの１つ手前の4096次元ベクトルを特徴量として用いる．
# - 画像に対して，標準化（平均を引いた後に，標準偏差で割る）は実施するものの，標準化後のL2正規化については，PCAの性質（スケーリングに対して敏感）を勘案して，保留する．
# 
# -  課題 [1a](https://mm.cs.uec.ac.jp/sys/text/kadai2-4pt.html) の内容を参考にする．
# -  課題 [3a](https://mm.cs.uec.ac.jp/local/homework22.html#3)の内容を参考にする．

# In[11]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from IPython import display
import time
import os

# proxyの設定．
# datasetをダウンロードするので，学内マシンからは通常必要．
os.environ["http_proxy"] = "http://proxy.uec.ac.jp:8080/"
os.environ["https_proxy"] = "http://proxy.uec.ac.jp:8080/"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1" # "0":GPU0, "1":GPU1, "0,1":GPUを２つとも使用

import numpy as np
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import torchvision.models as models

# modelsからvgg16をロード
vgg16 = models.vgg16(pretrained=True,progress=True)
softmax=nn.Softmax(dim=1)
# pretrained=True とすると，学習済みパラメータも読み込まれる．
# ~/.cache/torch/checkpoints/ に読み込まれます．VGG16は550MBもあるので，不要になったら消去しましょう．
# ls でダウンロードされていることを確認してみます．
get_ipython().system(' ls -l ~/.cache/torch/checkpoints/')


# In[12]:


# 画像の変換は, TorchVisionを使うと簡単にできます．
import torchvision.transforms as transforms

from PIL import Image

# 念のため，VGG16で認識してみます．
# 2位以下はResNetと若干異なっていますが，1位はあっているはずです．
image_size = (224, 224) 
# ImageNetデータセットであるため，平均及び標準偏差を用いて，正規化
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
# Composeは複数の変換を連続して行うもの．リサイズして，テンソル変換して，正規化する．
image_transform = transforms.Compose([transforms.Resize(image_size),transforms.ToTensor(),normalize])

# ファイルを開く
img = Image.open('./imgdata/001001.jpg')

# 表示
plt.imshow(mpimg.imread("./imgdata/001001.jpg"))

# 画像を変換
img = image_transform(img)

# データの次元を拡張する
img = img.unsqueeze(0)

# 推論モードにセット
# batch normalization を使ったモデルで認識する場合は，eval modeへの切替えは必須
vgg16.eval()

# 勾配計算はしない
with torch.no_grad():
    out=softmax(vgg16(img)).numpy()[0]

# top5取り出し  
top5   =np.sort(out)[:-6:-1]   
top5idx=np.argsort(out)[:-6:-1]

# 認識結果の top-5 の結果の表示
SYNSET_FILE='./synset_words.txt'  # ImageNet1000 種類のカテゴリ名が書かれたファイル．
synset=open(SYNSET_FILE).read().split('\n')

# top5出力
for i in range(5):
    print("[%d] %.8f %s" % (i+1,top5[i],synset[top5idx[i]]))


# In[13]:


class Flatten(nn.Module):
    def forward(self, input):
        return input.reshape(input.size(0), -1) # 今回はviewをreshapeにしないとハマる！理由は不明．
    
vgg16fc7 = torch.nn.Sequential(
    vgg16.features,
    vgg16.avgpool,
    Flatten(),
    *list(vgg16.classifier.children())[:-3]  # 最後の3つのlayer(relu,dropout,fc1000)を削除
)
# fc7 (fc4096)が最終出力になっている
# print(vgg16fc7)


# 次に学習画像の読み込みです．DataLoaderを使う方法もありますが，ここでは一気に100枚読み込んで，まとめてfc7特徴を抽出してしまいましょう．

# In[14]:


import glob
imglist = glob.glob("./imgdata/*.jpg") # imglistには，処理対象の画像ファイル群
imglist = imglist[:100] # 100枚に限定
print(len(imglist)) # 画像

# 画像をimgsに読み込みます．
in_size=224

# 初期化
imgs = np.zeros((0,in_size,in_size,3), dtype=np.float32)
#imgs = np.empty((in_size,in_size,3), dtype=np.float32)
#imgs = np.zeros([4,4,4])

for i,img_path in enumerate(imglist):
    if i%100==0:
        print("reading {}th image".format(i))
    x = np.array(Image.open(img_path).resize((in_size,in_size)), dtype=np.float32)
    #print(x.shape)
    
    if x.shape == (in_size, in_size):
        x0 = np.empty((in_size,in_size,2), dtype=np.float32)
        #print(x0.shape)
        x = np.dstack([x, x0])

    x = np.expand_dims(x, axis=0)
    #print(x.shape) #(1, 224, 224, 3)
    imgs = np.vstack((imgs,x))
    
mean=np.array([0.485, 0.456, 0.406], dtype=np.float32)
std=np.array([0.229, 0.224, 0.225], dtype=np.float32)
imgs=(imgs/255.0-mean)/std # 標準化
imgs=imgs.transpose(0,3,1,2)  # HWC -> CHW
img=torch.from_numpy(imgs)
print(imgs.shape)


# In[15]:


print(img.shape)


# In[16]:


# 100枚処理するので，GPUを使います．
device = 'cuda' if torch.cuda.is_available() else 'cpu'
vgg16fc7 = vgg16fc7.to(device)

print(device)


# In[17]:


vgg16fc7.eval()
with torch.no_grad():
    fc=vgg16fc7(img.to(device)).cpu().numpy()
    # gpuで処理した結果を cpuに戻して，numpy形式にします．
print(fc.shape)     # shapeの表示
print(fc[0])


# ### L2正規化しないと失敗すると分かった．問4でも，L2正規化してみる．

# In[18]:


L2norm = np.linalg.norm(fc, ord=2, axis=1) # L2ノルムを計算する関数 np.lialg.norm
print(L2norm.shape)
print(L2norm[0])

for n in range(len(L2norm)):
    fc[n] = fc[n]/L2norm[n]

print(fc[0])


# In[19]:


for n in range(len(L2norm)):
    print(np.linalg.norm(fc[n])) #nanが含まれてしまうこともあったが，間欠事象．１に近いのでOK


# In[20]:


np.save("./feature/dcnn.npy", fc) # 標準化まで（正規化はしない）値を書き出し


# ### 特徴量マップ (100, 4096)次元，L2正規化しないと，min, maxの値が暴れてPCAに失敗してしまう．

# In[21]:


import numpy as np

# テスト読み出し
feature = np.load("./feature/dcnn.npy")
print(type(feature))
print(feature[0]) # 最初のデータを表示
print(len(feature))
print(feature.shape)


# ### 0番目の画像を表示確認

# In[22]:


x = imglist[0]
plt.imshow(mpimg.imread(x))


# ## 特徴量マップfeatureに対して，PCAする．

# In[14]:


import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

feature = np.load("./feature/dcnn.npy")
feature = feature*1000 # ゼロに近い値の割り算が発生し，PCA出来ないので，1000倍
print(np.min(feature), np.max(feature))
print(feature.shape)

X_df = pd.DataFrame(feature)
X_df


# ### 95%で主成分分析してみる．35次元と判明．4096次元から大幅に削減．

# In[15]:


pca = PCA(n_components=0.95, svd_solver='full') # PCAインスタンスを生成
pca.fit(feature) #PCA実行
feat95 = pca.fit_transform(X_df)
print(feat95.shape)
print(feat95)

# 主成分の寄与率を出力する
# print('各次元の寄与率: {0}'.format(pca.explained_variance_ratio_))
# print('累積寄与率: {0}'.format(sum(pca.explained_variance_ratio_)))


# In[23]:


column_num = len(feat95[1])
print(column_num)

# 主成分得点 95%
feat95_map = pd.DataFrame(feat95, columns=["PC{}".format(x) for x in range(column_num)])
feat95_map


# In[24]:


plt.bar([n for n in range(1, len(pca.explained_variance_ratio_)+1)], pca.explained_variance_ratio_)


# ### 95%の次元でクラスタリングしてみる．問題文から，k-means法をk=5とk=10で実施する．
# 
# ### k=5のときを考える．

# In[26]:


from sklearn.cluster import KMeans

pred95 = KMeans(n_clusters=5, random_state=0).fit_predict(feat95)
print(pred95.shape)
print(pred95)


# ### 表示関数の定義

# In[25]:


import cv2

def imshowROWCOL(img, row=8, col=5):
    row = row
    col = col

    plt.figure(figsize=(30,30))

    num = 0

    # 6x6の枠を用意し，画像数が足りなくてもエラーが出ないように最小値をとる
    while num < min(len(img),row*col): 
        #img_sub = Image.open(img[num]) # 画像の配列は０から始まるので，この次にインクリメント
        img_sub = cv2.imread(img[num])
        img_sub = cv2.cvtColor(img_sub, cv2.COLOR_BGR2RGB)
        
        num += 1
        plt.subplot(row, col, num)
        plt.imshow(img_sub)
        plt.axis('off')
        


# In[76]:


classNum = 0

idx = np.where(pred95==classNum) # クラス数と一致するインデックスを，groupsの中から抽出．
idx = idx[0]
list(idx)
#print(len(idx))

showimglist = []
for n in range(len(idx)):
    showimglist.append(imglist[idx[n]])

#print(showimglist)
imshowROWCOL(showimglist)
    
idx = np.array(idx).flatten() # クラスの個数を計算するためにnumpyに変換して，flatten
print("クラス", classNum, "の個数：", len(idx))


# In[77]:


classNum = 1

idx = np.where(pred95==classNum) # クラス数と一致するインデックスを，groupsの中から抽出．
idx = idx[0]
list(idx)
#print(len(idx))

showimglist = []
for n in range(len(idx)):
    showimglist.append(imglist[idx[n]])

#print(showimglist)
imshowROWCOL(showimglist)
    
idx = np.array(idx).flatten() # クラスの個数を計算するためにnumpyに変換して，flatten
print("クラス", classNum, "の個数：", len(idx))


# In[78]:


classNum = 2

idx = np.where(pred95==classNum) # クラス数と一致するインデックスを，groupsの中から抽出．
idx = idx[0]
list(idx)
#print(len(idx))

showimglist = []
for n in range(len(idx)):
    showimglist.append(imglist[idx[n]])

#print(showimglist)
imshowROWCOL(showimglist)
    
idx = np.array(idx).flatten() # クラスの個数を計算するためにnumpyに変換して，flatten
print("クラス", classNum, "の個数：", len(idx))


# In[79]:


classNum = 3

idx = np.where(pred95==classNum) # クラス数と一致するインデックスを，groupsの中から抽出．
idx = idx[0]
list(idx)
#print(len(idx))

showimglist = []
for n in range(len(idx)):
    showimglist.append(imglist[idx[n]])

#print(showimglist)
imshowROWCOL(showimglist)
    
idx = np.array(idx).flatten() # クラスの個数を計算するためにnumpyに変換して，flatten
print("クラス", classNum, "の個数：", len(idx))


# In[80]:


classNum = 4

idx = np.where(pred95==classNum) # クラス数と一致するインデックスを，groupsの中から抽出．
idx = idx[0]
list(idx)
#print(len(idx))

showimglist = []
for n in range(len(idx)):
    showimglist.append(imglist[idx[n]])

#print(showimglist)
imshowROWCOL(showimglist)
    
idx = np.array(idx).flatten() # クラスの個数を計算するためにnumpyに変換して，flatten
print("クラス", classNum, "の個数：", len(idx))


# ### k=10のとき

# In[81]:


from sklearn.cluster import KMeans

pred95_c10 = KMeans(n_clusters=10, random_state=0).fit_predict(feat95)
print(pred95_c10.shape)
print(pred95_c10)


# In[85]:


classNum = 0

idx = []
idx = np.where(pred95_c10==classNum) # クラス数と一致するインデックスを，groupsの中から抽出．
idx = idx[0]
list(idx)
#print(len(idx))

showimglist = []
for n in range(len(idx)):
    showimglist.append(imglist[idx[n]])

#print(showimglist)
imshowROWCOL(showimglist)
    
idx = np.array(idx).flatten() # クラスの個数を計算するためにnumpyに変換して，flatten
print("クラス", classNum, "の個数：", len(idx))


# In[86]:


classNum = 1

idx = []
idx = np.where(pred95_c10==classNum) # クラス数と一致するインデックスを，groupsの中から抽出．
idx = idx[0]
list(idx)
#print(len(idx))

showimglist = []
for n in range(len(idx)):
    showimglist.append(imglist[idx[n]])

#print(showimglist)
imshowROWCOL(showimglist)
    
idx = np.array(idx).flatten() # クラスの個数を計算するためにnumpyに変換して，flatten
print("クラス", classNum, "の個数：", len(idx))


# In[87]:


classNum = 2

idx = []
idx = np.where(pred95_c10==classNum) # クラス数と一致するインデックスを，groupsの中から抽出．
idx = idx[0]
list(idx)
#print(len(idx))

showimglist = []
for n in range(len(idx)):
    showimglist.append(imglist[idx[n]])

#print(showimglist)
imshowROWCOL(showimglist)
    
idx = np.array(idx).flatten() # クラスの個数を計算するためにnumpyに変換して，flatten
print("クラス", classNum, "の個数：", len(idx))


# In[88]:


classNum = 3

idx = []
idx = np.where(pred95_c10==classNum) # クラス数と一致するインデックスを，groupsの中から抽出．
idx = idx[0]
list(idx)
#print(len(idx))

showimglist = []
for n in range(len(idx)):
    showimglist.append(imglist[idx[n]])

#print(showimglist)
imshowROWCOL(showimglist)
    
idx = np.array(idx).flatten() # クラスの個数を計算するためにnumpyに変換して，flatten
print("クラス", classNum, "の個数：", len(idx))


# In[89]:


classNum = 4

idx = []
idx = np.where(pred95_c10==classNum) # クラス数と一致するインデックスを，groupsの中から抽出．
idx = idx[0]
list(idx)
#print(len(idx))

showimglist = []
for n in range(len(idx)):
    showimglist.append(imglist[idx[n]])

#print(showimglist)
imshowROWCOL(showimglist)
    
idx = np.array(idx).flatten() # クラスの個数を計算するためにnumpyに変換して，flatten
print("クラス", classNum, "の個数：", len(idx))


# In[90]:


classNum = 5

idx = []
idx = np.where(pred95_c10==classNum) # クラス数と一致するインデックスを，groupsの中から抽出．
idx = idx[0]
list(idx)
#print(len(idx))

showimglist = []
for n in range(len(idx)):
    showimglist.append(imglist[idx[n]])

#print(showimglist)
imshowROWCOL(showimglist)
    
idx = np.array(idx).flatten() # クラスの個数を計算するためにnumpyに変換して，flatten
print("クラス", classNum, "の個数：", len(idx))


# In[91]:


classNum = 6

idx = []
idx = np.where(pred95_c10==classNum) # クラス数と一致するインデックスを，groupsの中から抽出．
idx = idx[0]
list(idx)
#print(len(idx))

showimglist = []
for n in range(len(idx)):
    showimglist.append(imglist[idx[n]])

#print(showimglist)
imshowROWCOL(showimglist)
    
idx = np.array(idx).flatten() # クラスの個数を計算するためにnumpyに変換して，flatten
print("クラス", classNum, "の個数：", len(idx))


# In[92]:


classNum = 7

idx = []
idx = np.where(pred95_c10==classNum) # クラス数と一致するインデックスを，groupsの中から抽出．
idx = idx[0]
list(idx)
#print(len(idx))

showimglist = []
for n in range(len(idx)):
    showimglist.append(imglist[idx[n]])

#print(showimglist)
imshowROWCOL(showimglist)
    
idx = np.array(idx).flatten() # クラスの個数を計算するためにnumpyに変換して，flatten
print("クラス", classNum, "の個数：", len(idx))


# In[93]:


classNum = 8

idx = []
idx = np.where(pred95_c10==classNum) # クラス数と一致するインデックスを，groupsの中から抽出．
idx = idx[0]
list(idx)
#print(len(idx))

showimglist = []
for n in range(len(idx)):
    showimglist.append(imglist[idx[n]])

#print(showimglist)
imshowROWCOL(showimglist)
    
idx = np.array(idx).flatten() # クラスの個数を計算するためにnumpyに変換して，flatten
print("クラス", classNum, "の個数：", len(idx))


# In[94]:


classNum = 9

idx = []
idx = np.where(pred95_c10==classNum) # クラス数と一致するインデックスを，groupsの中から抽出．
idx = idx[0]
list(idx)
#print(len(idx))

showimglist = []
for n in range(len(idx)):
    showimglist.append(imglist[idx[n]])

#print(showimglist)
imshowROWCOL(showimglist)
    
idx = np.array(idx).flatten() # クラスの個数を計算するためにnumpyに変換して，flatten
print("クラス", classNum, "の個数：", len(idx))


# ### 特徴量マップfeatureを再読み込み．

# In[2]:


import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

feature = np.load("./feature/dcnn.npy")
feature = feature*1000 # ゼロに近い値の割り算が発生し，PCA出来ないので，1000倍
print(np.min(feature), np.max(feature))
print(feature.shape)

X_df = pd.DataFrame(feature)
X_df


# ### 90%で主成分分析する．4096次元から，24次元に削減

# In[3]:


pca = PCA(n_components=0.90, svd_solver='full') # PCAインスタンスを生成
pca.fit(feature) #PCA実行
feat90 = pca.fit_transform(X_df)
print(feat90.shape)
print(feat90)

# 主成分の寄与率を出力する
# print('各次元の寄与率: {0}'.format(pca.explained_variance_ratio_))
# print('累積寄与率: {0}'.format(sum(pca.explained_variance_ratio_)))


# In[6]:


column_num = len(feat90[1])
print(column_num)

# 主成分得点 90%
feat90_map = pd.DataFrame(feat90, columns=["PC{}".format(x) for x in range(column_num)])
feat90_map

plt.bar([n for n in range(1, len(pca.explained_variance_ratio_)+1)], pca.explained_variance_ratio_)


# ### 90%の次元でクラスタリングする．k-means法をk=5とk=10で実施する．
# 
# ### k=5のときを考える．

# In[23]:


from sklearn.cluster import KMeans

pred90 = KMeans(n_clusters=5, random_state=0).fit_predict(feat90)
print(pred90.shape)
print(pred90)


# In[26]:


classNum = 0

idx = np.where(pred90==classNum) # クラス数と一致するインデックスを，groupsの中から抽出．
idx = idx[0]
list(idx)
#print(len(idx))

showimglist = []
for n in range(len(idx)):
    showimglist.append(imglist[idx[n]])

#print(showimglist)
imshowROWCOL(showimglist)
    
idx = np.array(idx).flatten() # クラスの個数を計算するためにnumpyに変換して，flatten
print("クラス", classNum, "の個数：", len(idx))


# In[27]:


classNum = 1

idx = np.where(pred90==classNum) # クラス数と一致するインデックスを，groupsの中から抽出．
idx = idx[0]
list(idx)
#print(len(idx))

showimglist = []
for n in range(len(idx)):
    showimglist.append(imglist[idx[n]])

#print(showimglist)
imshowROWCOL(showimglist)
    
idx = np.array(idx).flatten() # クラスの個数を計算するためにnumpyに変換して，flatten
print("クラス", classNum, "の個数：", len(idx))


# In[28]:


classNum = 2

idx = np.where(pred90==classNum) # クラス数と一致するインデックスを，groupsの中から抽出．
idx = idx[0]
list(idx)
#print(len(idx))

showimglist = []
for n in range(len(idx)):
    showimglist.append(imglist[idx[n]])

#print(showimglist)
imshowROWCOL(showimglist)
    
idx = np.array(idx).flatten() # クラスの個数を計算するためにnumpyに変換して，flatten
print("クラス", classNum, "の個数：", len(idx))


# In[29]:


classNum = 3

idx = np.where(pred90==classNum) # クラス数と一致するインデックスを，groupsの中から抽出．
idx = idx[0]
list(idx)
#print(len(idx))

showimglist = []
for n in range(len(idx)):
    showimglist.append(imglist[idx[n]])

#print(showimglist)
imshowROWCOL(showimglist)
    
idx = np.array(idx).flatten() # クラスの個数を計算するためにnumpyに変換して，flatten
print("クラス", classNum, "の個数：", len(idx))


# In[30]:


classNum = 4

idx = np.where(pred90==classNum) # クラス数と一致するインデックスを，groupsの中から抽出．
idx = idx[0]
list(idx)
#print(len(idx))

showimglist = []
for n in range(len(idx)):
    showimglist.append(imglist[idx[n]])

#print(showimglist)
imshowROWCOL(showimglist)
    
idx = np.array(idx).flatten() # クラスの個数を計算するためにnumpyに変換して，flatten
print("クラス", classNum, "の個数：", len(idx))


# ### クラス数kが少ないと，複数の種類の画像が１クラスに混入する．90%の次元では，ノイズの影響は限定的．

# ### 90%の次元で，k=10のときを考える．

# In[32]:


from sklearn.cluster import KMeans

pred90 = KMeans(n_clusters=10, random_state=0).fit_predict(feat90)
print(pred90.shape)
print(pred90)


# In[33]:


classNum = 0

idx = np.where(pred90==classNum) # クラス数と一致するインデックスを，groupsの中から抽出．
idx = idx[0]
list(idx)
#print(len(idx))

showimglist = []
for n in range(len(idx)):
    showimglist.append(imglist[idx[n]])

#print(showimglist)
imshowROWCOL(showimglist)
    
idx = np.array(idx).flatten() # クラスの個数を計算するためにnumpyに変換して，flatten
print("クラス", classNum, "の個数：", len(idx))


# In[34]:


classNum = 1

idx = np.where(pred90==classNum) # クラス数と一致するインデックスを，groupsの中から抽出．
idx = idx[0]
list(idx)
#print(len(idx))

showimglist = []
for n in range(len(idx)):
    showimglist.append(imglist[idx[n]])

#print(showimglist)
imshowROWCOL(showimglist)
    
idx = np.array(idx).flatten() # クラスの個数を計算するためにnumpyに変換して，flatten
print("クラス", classNum, "の個数：", len(idx))


# In[35]:


classNum = 2

idx = np.where(pred90==classNum) # クラス数と一致するインデックスを，groupsの中から抽出．
idx = idx[0]
list(idx)
#print(len(idx))

showimglist = []
for n in range(len(idx)):
    showimglist.append(imglist[idx[n]])

#print(showimglist)
imshowROWCOL(showimglist)
    
idx = np.array(idx).flatten() # クラスの個数を計算するためにnumpyに変換して，flatten
print("クラス", classNum, "の個数：", len(idx))


# In[36]:


classNum = 3

idx = np.where(pred90==classNum) # クラス数と一致するインデックスを，groupsの中から抽出．
idx = idx[0]
list(idx)
#print(len(idx))

showimglist = []
for n in range(len(idx)):
    showimglist.append(imglist[idx[n]])

#print(showimglist)
imshowROWCOL(showimglist)
    
idx = np.array(idx).flatten() # クラスの個数を計算するためにnumpyに変換して，flatten
print("クラス", classNum, "の個数：", len(idx))


# In[37]:


classNum = 4

idx = np.where(pred90==classNum) # クラス数と一致するインデックスを，groupsの中から抽出．
idx = idx[0]
list(idx)
#print(len(idx))

showimglist = []
for n in range(len(idx)):
    showimglist.append(imglist[idx[n]])

#print(showimglist)
imshowROWCOL(showimglist)
    
idx = np.array(idx).flatten() # クラスの個数を計算するためにnumpyに変換して，flatten
print("クラス", classNum, "の個数：", len(idx))


# In[38]:


classNum = 5


idx = np.where(pred90==classNum) # クラス数と一致するインデックスを，groupsの中から抽出．
idx = idx[0]
list(idx)
#print(len(idx))

showimglist = []
for n in range(len(idx)):
    showimglist.append(imglist[idx[n]])

#print(showimglist)
imshowROWCOL(showimglist)
    
idx = np.array(idx).flatten() # クラスの個数を計算するためにnumpyに変換して，flatten
print("クラス", classNum, "の個数：", len(idx))


# In[39]:


classNum = 6

idx = np.where(pred90==classNum) # クラス数と一致するインデックスを，groupsの中から抽出．
idx = idx[0]
list(idx)
#print(len(idx))

showimglist = []
for n in range(len(idx)):
    showimglist.append(imglist[idx[n]])

#print(showimglist)
imshowROWCOL(showimglist)
    
idx = np.array(idx).flatten() # クラスの個数を計算するためにnumpyに変換して，flatten
print("クラス", classNum, "の個数：", len(idx))


# In[40]:


classNum = 7

idx = np.where(pred90==classNum) # クラス数と一致するインデックスを，groupsの中から抽出．
idx = idx[0]
list(idx)
#print(len(idx))

showimglist = []
for n in range(len(idx)):
    showimglist.append(imglist[idx[n]])

#print(showimglist)
imshowROWCOL(showimglist)
    
idx = np.array(idx).flatten() # クラスの個数を計算するためにnumpyに変換して，flatten
print("クラス", classNum, "の個数：", len(idx))


# In[41]:


classNum = 8

idx = np.where(pred90==classNum) # クラス数と一致するインデックスを，groupsの中から抽出．
idx = idx[0]
list(idx)
#print(len(idx))

showimglist = []
for n in range(len(idx)):
    showimglist.append(imglist[idx[n]])

#print(showimglist)
imshowROWCOL(showimglist)
    
idx = np.array(idx).flatten() # クラスの個数を計算するためにnumpyに変換して，flatten
print("クラス", classNum, "の個数：", len(idx))


# In[42]:


classNum = 9

idx = np.where(pred90==classNum) # クラス数と一致するインデックスを，groupsの中から抽出．
idx = idx[0]
list(idx)
#print(len(idx))

showimglist = []
for n in range(len(idx)):
    showimglist.append(imglist[idx[n]])

#print(showimglist)
imshowROWCOL(showimglist)
    
idx = np.array(idx).flatten() # クラスの個数を計算するためにnumpyに変換して，flatten
print("クラス", classNum, "の個数：", len(idx))


# ### kの数を増加させた時の変化
# - 表現量が豊かな画像は，クラス数kが増加するにつれて，より精度高く，画像の分類が出来るように見える
# - しかし，クラス数を増やしても，白黒の画像（特許図面）は全然分割されない．
#     - かろうじて，四角形の表と，その他の特許図面が分類されたように見えるが，精度が判然としない．
#     - 恐らく，ImageNet（カラー）の性質に推察される．

# ## 比較実験として，PCAせず，4096次元のままk-meansを行う．

# In[51]:


import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

feature = np.load("./feature/dcnn.npy")
feature = feature*1000 # ゼロに近い値の割り算が発生し，PCA出来ないので，1000倍
print(np.min(feature), np.max(feature))
print(feature.shape)

X_df = pd.DataFrame(feature)
X_df


# In[52]:


column_num = len(feature[1])
print(column_num)

# 4096次元の特徴量マップを表示
feat_map = pd.DataFrame(feature, columns=["PC{}".format(x) for x in range(column_num)])
feat_map


# ### 4096次元(PCA無し)でクラスタリングする．k-means法をk=5とk=10で実施する．
# 
# ### k=5のときを考える．

# In[53]:


from sklearn.cluster import KMeans

pred95 = KMeans(n_clusters=5, random_state=0).fit_predict(feat_map)
print(pred95.shape)
print(pred95)


# In[54]:


classNum = 0

idx = []
idx = np.where(pred95==classNum) # クラス数と一致するインデックスを，groupsの中から抽出．
idx = idx[0]
list(idx)
#print(len(idx))

showimglist = []
for n in range(len(idx)):
    showimglist.append(imglist[idx[n]])

#print(showimglist)
imshowROWCOL(showimglist)
    
idx = np.array(idx).flatten() # クラスの個数を計算するためにnumpyに変換して，flatten
print("クラス", classNum, "の個数：", len(idx))


# In[55]:


classNum = 1

idx = []
idx = np.where(pred95==classNum) # クラス数と一致するインデックスを，groupsの中から抽出．
idx = idx[0]
list(idx)
#print(len(idx))

showimglist = []
for n in range(len(idx)):
    showimglist.append(imglist[idx[n]])

#print(showimglist)
imshowROWCOL(showimglist)
    
idx = np.array(idx).flatten() # クラスの個数を計算するためにnumpyに変換して，flatten
print("クラス", classNum, "の個数：", len(idx))


# In[56]:


classNum = 2

idx = []
idx = np.where(pred95==classNum) # クラス数と一致するインデックスを，groupsの中から抽出．
idx = idx[0]
list(idx)
#print(len(idx))

showimglist = []
for n in range(len(idx)):
    showimglist.append(imglist[idx[n]])

#print(showimglist)
imshowROWCOL(showimglist)
    
idx = np.array(idx).flatten() # クラスの個数を計算するためにnumpyに変換して，flatten
print("クラス", classNum, "の個数：", len(idx))


# In[57]:


classNum = 3

idx = []
idx = np.where(pred95==classNum) # クラス数と一致するインデックスを，groupsの中から抽出．
idx = idx[0]
list(idx)
#print(len(idx))

showimglist = []
for n in range(len(idx)):
    showimglist.append(imglist[idx[n]])

#print(showimglist)
imshowROWCOL(showimglist)
    
idx = np.array(idx).flatten() # クラスの個数を計算するためにnumpyに変換して，flatten
print("クラス", classNum, "の個数：", len(idx))


# In[58]:


classNum = 4

idx = []
idx = np.where(pred95==classNum) # クラス数と一致するインデックスを，groupsの中から抽出．
idx = idx[0]
list(idx)
#print(len(idx))

showimglist = []
for n in range(len(idx)):
    showimglist.append(imglist[idx[n]])

#print(showimglist)
imshowROWCOL(showimglist)
    
idx = np.array(idx).flatten() # クラスの個数を計算するためにnumpyに変換して，flatten
print("クラス", classNum, "の個数：", len(idx))


# ### 4096次元(PCA無し)でクラスタリングする．k=10のときを考える．

# In[60]:


from sklearn.cluster import KMeans

pred95 = KMeans(n_clusters=10, random_state=0).fit_predict(feat_map)
print(pred95.shape)
print(pred95)


# In[61]:


classNum = 0

idx = []
idx = np.where(pred95==classNum) # クラス数と一致するインデックスを，groupsの中から抽出．
idx = idx[0]
list(idx)
#print(len(idx))

showimglist = []
for n in range(len(idx)):
    showimglist.append(imglist[idx[n]])

#print(showimglist)
imshowROWCOL(showimglist)
    
idx = np.array(idx).flatten() # クラスの個数を計算するためにnumpyに変換して，flatten
print("クラス", classNum, "の個数：", len(idx))


# In[62]:


classNum = 1

idx = []
idx = np.where(pred95==classNum) # クラス数と一致するインデックスを，groupsの中から抽出．
idx = idx[0]
list(idx)
#print(len(idx))

showimglist = []
for n in range(len(idx)):
    showimglist.append(imglist[idx[n]])

#print(showimglist)
imshowROWCOL(showimglist)
    
idx = np.array(idx).flatten() # クラスの個数を計算するためにnumpyに変換して，flatten
print("クラス", classNum, "の個数：", len(idx))


# In[63]:


classNum = 2

idx = []
idx = np.where(pred95==classNum) # クラス数と一致するインデックスを，groupsの中から抽出．
idx = idx[0]
list(idx)
#print(len(idx))

showimglist = []
for n in range(len(idx)):
    showimglist.append(imglist[idx[n]])

#print(showimglist)
imshowROWCOL(showimglist)
    
idx = np.array(idx).flatten() # クラスの個数を計算するためにnumpyに変換して，flatten
print("クラス", classNum, "の個数：", len(idx))


# In[64]:


classNum = 3

idx = []
idx = np.where(pred95==classNum) # クラス数と一致するインデックスを，groupsの中から抽出．
idx = idx[0]
list(idx)
#print(len(idx))

showimglist = []
for n in range(len(idx)):
    showimglist.append(imglist[idx[n]])

#print(showimglist)
imshowROWCOL(showimglist)
    
idx = np.array(idx).flatten() # クラスの個数を計算するためにnumpyに変換して，flatten
print("クラス", classNum, "の個数：", len(idx))


# In[65]:


classNum = 4

idx = []
idx = np.where(pred95==classNum) # クラス数と一致するインデックスを，groupsの中から抽出．
idx = idx[0]
list(idx)
#print(len(idx))

showimglist = []
for n in range(len(idx)):
    showimglist.append(imglist[idx[n]])

#print(showimglist)
imshowROWCOL(showimglist)
    
idx = np.array(idx).flatten() # クラスの個数を計算するためにnumpyに変換して，flatten
print("クラス", classNum, "の個数：", len(idx))


# In[66]:


classNum = 5

idx = []
idx = np.where(pred95==classNum) # クラス数と一致するインデックスを，groupsの中から抽出．
idx = idx[0]
list(idx)
#print(len(idx))

showimglist = []
for n in range(len(idx)):
    showimglist.append(imglist[idx[n]])

#print(showimglist)
imshowROWCOL(showimglist)
    
idx = np.array(idx).flatten() # クラスの個数を計算するためにnumpyに変換して，flatten
print("クラス", classNum, "の個数：", len(idx))


# In[67]:


classNum = 6

idx = []
idx = np.where(pred95==classNum) # クラス数と一致するインデックスを，groupsの中から抽出．
idx = idx[0]
list(idx)
#print(len(idx))

showimglist = []
for n in range(len(idx)):
    showimglist.append(imglist[idx[n]])

#print(showimglist)
imshowROWCOL(showimglist)
    
idx = np.array(idx).flatten() # クラスの個数を計算するためにnumpyに変換して，flatten
print("クラス", classNum, "の個数：", len(idx))


# In[68]:


classNum = 7

idx = []
idx = np.where(pred95==classNum) # クラス数と一致するインデックスを，groupsの中から抽出．
idx = idx[0]
list(idx)
#print(len(idx))

showimglist = []
for n in range(len(idx)):
    showimglist.append(imglist[idx[n]])

#print(showimglist)
imshowROWCOL(showimglist)
    
idx = np.array(idx).flatten() # クラスの個数を計算するためにnumpyに変換して，flatten
print("クラス", classNum, "の個数：", len(idx))


# In[69]:


classNum = 8

idx = []
idx = np.where(pred95==classNum) # クラス数と一致するインデックスを，groupsの中から抽出．
idx = idx[0]
list(idx)
#print(len(idx))

showimglist = []
for n in range(len(idx)):
    showimglist.append(imglist[idx[n]])

#print(showimglist)
imshowROWCOL(showimglist)
    
idx = np.array(idx).flatten() # クラスの個数を計算するためにnumpyに変換して，flatten
print("クラス", classNum, "の個数：", len(idx))


# In[70]:


classNum = 9

idx = []
idx = np.where(pred95==classNum) # クラス数と一致するインデックスを，groupsの中から抽出．
idx = idx[0]
list(idx)
#print(len(idx))

showimglist = []
for n in range(len(idx)):
    showimglist.append(imglist[idx[n]])

#print(showimglist)
imshowROWCOL(showimglist)
    
idx = np.array(idx).flatten() # クラスの個数を計算するためにnumpyに変換して，flatten
print("クラス", classNum, "の個数：", len(idx))


# ### 比較・考察
# - 4096次元でk-meansした場合と，PCA95%の35次元とを比較すると，確かにクラス分類における微差は見られたものの，分類の傾向に違いは見られなかった．PCAによる次元削減の効果は非常に高い．

# ### PCAの効果を考えるために，PCA3次元まで削減する．

# In[84]:


import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

feature = np.load("./feature/dcnn.npy")
feature = feature*1000 # ゼロに近い値の割り算が発生し，PCA出来ないので，1000倍
print(np.min(feature), np.max(feature))
print(feature.shape)

X_df = pd.DataFrame(feature)
X_df


# In[85]:


pca = PCA(n_components=3, svd_solver='full') # PCAインスタンスを生成
pca.fit(feature) #PCA実行
feat3 = pca.fit_transform(X_df)
print(feat3.shape)
#print(feat90)

# 主成分の寄与率を出力する
# print('各次元の寄与率: {0}'.format(pca.explained_variance_ratio_))
# print('累積寄与率: {0}'.format(sum(pca.explained_variance_ratio_)))


# ### PCA3次元で主成分分析し，k=3でクラスタリングする．

# In[86]:


from sklearn.cluster import KMeans

pred3 = KMeans(n_clusters=5, random_state=0).fit_predict(feat3)
print(pred3.shape)
print(pred3)


# In[87]:


classNum = 0

idx = []
idx = np.where(pred3==classNum) # クラス数と一致するインデックスを，groupsの中から抽出．
idx = idx[0]
list(idx)
#print(len(idx))

showimglist = []
for n in range(len(idx)):
    showimglist.append(imglist[idx[n]])

#print(showimglist)
imshowROWCOL(showimglist)
    
idx = np.array(idx).flatten() # クラスの個数を計算するためにnumpyに変換して，flatten
print("クラス", classNum, "の個数：", len(idx))


# In[91]:


classNum = 1

idx = []
idx = np.where(pred3==classNum) # クラス数と一致するインデックスを，groupsの中から抽出．
idx = idx[0]
list(idx)
#print(len(idx))

showimglist = []
for n in range(len(idx)):
    showimglist.append(imglist[idx[n]])

#print(showimglist)
imshowROWCOL(showimglist)
    
idx = np.array(idx).flatten() # クラスの個数を計算するためにnumpyに変換して，flatten
print("クラス", classNum, "の個数：", len(idx))


# In[92]:


classNum = 2

idx = []
idx = np.where(pred3==classNum) # クラス数と一致するインデックスを，groupsの中から抽出．
idx = idx[0]
list(idx)
#print(len(idx))

showimglist = []
for n in range(len(idx)):
    showimglist.append(imglist[idx[n]])

#print(showimglist)
imshowROWCOL(showimglist)
    
idx = np.array(idx).flatten() # クラスの個数を計算するためにnumpyに変換して，flatten
print("クラス", classNum, "の個数：", len(idx))


# In[93]:


classNum = 3

idx = []
idx = np.where(pred3==classNum) # クラス数と一致するインデックスを，groupsの中から抽出．
idx = idx[0]
list(idx)
#print(len(idx))

showimglist = []
for n in range(len(idx)):
    showimglist.append(imglist[idx[n]])

#print(showimglist)
imshowROWCOL(showimglist)
    
idx = np.array(idx).flatten() # クラスの個数を計算するためにnumpyに変換して，flatten
print("クラス", classNum, "の個数：", len(idx))


# In[94]:


classNum = 4

idx = []
idx = np.where(pred3==classNum) # クラス数と一致するインデックスを，groupsの中から抽出．
idx = idx[0]
list(idx)
#print(len(idx))

showimglist = []
for n in range(len(idx)):
    showimglist.append(imglist[idx[n]])

#print(showimglist)
imshowROWCOL(showimglist)
    
idx = np.array(idx).flatten() # クラスの個数を計算するためにnumpyに変換して，flatten
print("クラス", classNum, "の個数：", len(idx))


# ### PCA3次元で主成分分析した場合，想定通り，クラスの混入が多く発生した．
# ### k=5の結果から分かるように，PCAで次元数を削減し過ぎると，クラス分類がうまく行かない．上記を勘案し，PCA3次元において，k=10に上げる効果が見込まれないため，k=10の場合は省略する．
