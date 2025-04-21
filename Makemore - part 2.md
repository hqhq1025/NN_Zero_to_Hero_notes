#深度学习 #Zero2hero

这节课要基于一篇论文，使用 mlp 改进模型

# 前期准备

## 导入相关库

```python
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt # for making figures
%matplotlib inline
```

## 读取 name 文件

```python
# read in all the words
words = open('names.txt', 'r').read().splitlines()
words[:8]
```

## 创建映射

```python
# build the vocabulary of characters and mappings to/from integers
chars = sorted(list(set(''.join(words))))
stoi = {s:i+1 for i,s in enumerate(chars)}
stoi['.'] = 0
itos = {i:s for s,i in stoi.items()}
print(itos)
```

创建字符到整数、整数到字符的映射

# 创建数据集

```python
# build the dataset

block_size = 3 # context length: how many characters do we take to predict the next one?
X, Y = [], []
for w in words:
  
  #print(w)
  context = [0] * block_size
  for ch in w + '.':
    ix = stoi[ch]
    X.append(context)
    Y.append(ix)
    #print(''.join(itos[i] for i in context), '--->', itos[ix])
    context = context[1:] + [ix] # crop and append
  
X = torch.tensor(X)
Y = torch.tensor(Y)
```

这里要根据 word，分出 input 和 label

这里的 blocksize 其实就相当于 n-gram 里的 n，表示下一个预测的单词要基于前 n 个单词的情况

具体解释一下代码：

1. 设置 block size 大小，创建 X Y 列表，分别是 input 和 label
2. 开始遍历所有单词
3. 首先初始化 context，具体是 n 个 0 构成的 list
4. 遍历单词中的字母
5. ix 是当前字母映射后得到的整数，这里作为 label 存到 y 中，同时把 context 作为 input 存入 x
6. 更新 context，这里的规则需要注意，相当于是拿一个长度为 n 的 slide 在单词上滑动，这里相当于是向右滑动一次，因此使用切片，切除原有的 context 中的第一位，把当前的 letter 拼接到 context 中，实现移动
7. 最后，把 X Y 转化为 tensor 类型

# Embedding 嵌入层

这里是对于我而言比较新的内容，我先根据自己的理解解释一下为什么要引入 embedding

在 p1 的部分中，由于我们只是预测字母，因此维度有限，仅仅是 27

但如果要用来预测单词的话，则将面临巨大的维度，无法实现，因此需要转换思路

我们选择将**词汇或符号**映射到连续向量空间中的方法，用矩阵来表示，这样可以大幅降低维度，同时，对于从没出现过的词语，我们可以看学习后的向量与哪些词语比较接近，即可大概了解含义

> **Embedding 的优势与意义：**
> 
> ### **（1）泛化能力强：**
> 
> - 即使模型没有见过某个词序列，但如果词向量接近，它仍然能合理地预测其出现概率。
> 
> ### **（2）捕捉语义关系：**
> 
> - 词嵌入的空间结构捕捉了**语义关系**：
>     - 同义词、近义词距离近；
>     - 不同词性、不同含义的词语距离远。
> 
> ### **（3）提升效率：**
> 
> - 大幅降低模型参数量，减少计算负担。
> - 低维稠密向量提升了计算效率和存储效率。

以下是具体代码：
## 初始化

```python
C = torch.randn((27, 2))
emb = C[X]
W1 = torch.randn((6, 100))
b1 = torch.randn(100)
```
首先，这里随机初始化一个`C`，`C` 是一个**嵌入矩阵**，维度为 27,2，表示从 27 维映射成 2 维
27 代表的是 letter 的个数，2 叫做**嵌入维度**
`emb` 这里是在做**嵌入查找**，也就是把 `X` 这个 input tensor其中的每个 letter 经过 C 的映射，得到映射后的tensor
`W1` 是全链接层的权重，`b1`是全链接层的偏置
这里的 `C W1 b1` 全是后续需要更新的参数
## 非线性激活

```python
h = torch.tanh(emb.view(-1, 6) @ W1 + b1)
```
 这里使用 tanh 作为激活函数，为了让 `emb` 与 `W1` 相乘，需要把 `emb` 展开成一个二维的 tensor
 这里的 -1 的这个参数表示让 torch 自己来设置这个位置的参数（根据前后的计算，并非随机）；而 6 是因为 `W1` 的形状，或者是是因为 `emb` 的形状是 xxxx 2 3
 然后在加上  `b1` ，然后通过 tanh 得到 `h`, 这里的`h`的维度是 228146,100
## 输出层

```python
W2 = torch.randn((100, 27))

b2 = torch.randn(27)

logits = h @ W2 + b2
```
 这里再来一层,把 100 维映射到 27 维,得到 logits,用于预测下一个 letter 是啥
## 归一化

```python
 counts = logits.exp()
 
 prob = counts / counts.sum(1, keepdims=True)
```
这里对 logits 取 exp, 把他转化成对应的未归一化的权重
[[Makemore - part 1]]
[[../动手学深度学习/3.4 softmax 回归]]
然后对每行进行归一化,得到 prob,也就是每个类的概率
这相当于作用了一次 **softmax**
## 损失函数
```python
loss = -prob[torch.arange(32), Y].log().mean()
```
 这里我们的loss 使用 **cross entropy**
# 完整实现
## 初始化

```python
g = torch.Generator().manual_seed(2147483647) # for reproducibility
C = torch.randn((27, 10), generator=g)
W1 = torch.randn((30, 200), generator=g)
b1 = torch.randn(200, generator=g)
W2 = torch.randn((200, 27), generator=g)
b2 = torch.randn(27, generator=g)

parameters = [C, W1, b1, W2, b2]
```
这里的`parameters`包含了所有的参数,用于后续一通操作
## 开启梯度记录

```python
for p in parameters:
	p.requires_grad = True
```
## 小批量 mini batch
由于完整的跑下来那么多数据迭代周期很长,虽然得到的梯度,也就是修正的方向很精确
为了在时间和准确率上做出兼顾,我们采取 mini batch 的方法, 也就是把数据分为小份的 batch,快速的跑完,得到一个近似的更新方向之后进行更新. 这样反复多次的更新会优于几次的精准更新
```python
ix = torch.randint(0, Xtr.shape[0], (32,))
```
这里我们用随机数生成一个 ix, 用于从 X 中随机抽取一部分 data, 作为我们的 mini batch
这里我们选择的是先生成随机的 index, 然后再以此为索引进行数据的抽取
这个随机抽取的过程会在每次循环的初始进行
## 学习率调优 Learning Rate Finder

```python
lre = torch.linspace(-3, 0, 1000)
lrs = 10**lre
```
这里原先似乎确实没学过
首先是用`linespace`这个方法把-3 到 0 切 1000 份
然后用`lre`作为指数,以 10 为底数,得到 lrs,作为学习率

>因为学习率对训练过程影响极大，但它的“有效值域”通常是**以 10 为底的对数尺度分布**的
如果你用线性空间（如 0 到 1）探索，那前面 90% 都没用，因为学习率太小了根本不起作用

然后我们进行训练,在第 i次训练中采用 lrs 的第 i 项,使每次训练的 lrs 逐步变化,记录 loss 以及对应的 lrs
用数据做出 loss-lr 的图表,我们可以通过曲线找到 loss 下降最快的那个学习率,以此定位效率/效果最好的 lr 的大概区间
最终确定我们选择的 lr
## 三种数据集
由于我们的数据有限,并且单纯的对于一个数据集训练会面临 **过拟合** 的问题
因此,我们需要对于有限的数据进行一些切分,把它分为不同功能的数据集:**训练集,** **验证集**, **测试集**
这三者的比例大概是 80%, 10%, 10%
训练集用于训练参数,验证集用于训练超参数,测试集用于测试模型的泛化能力
[[../动手学深度学习/4.4 模型选择、欠拟合与过拟合]]

接下来来动手构建一下数据集, 更改一下以前的代码
```python
# build the dataset
block_size = 3 # context length: how many characters do we take to predict the next one?

def build_dataset(words):  
  X, Y = [], []
  for w in words:

    #print(w)
    context = [0] * block_size
    for ch in w + '.':
      ix = stoi[ch]
      X.append(context)
      Y.append(ix)
      #print(''.join(itos[i] for i in context), '---%3E', itos[ix])
      context = context[1:] + [ix] # crop and append

  X = torch.tensor(X)
  Y = torch.tensor(Y)
  print(X.shape, Y.shape)
  return X, Y

import random
random.seed(42)
random.shuffle(words)
n1 = int(0.8*len(words))
n2 = int(0.9*len(words))

Xtr, Ytr = build_dataset(words[:n1])
Xdev, Ydev = build_dataset(words[n1:n2])
Xte, Yte = build_dataset(words[n2:])
```
这里我们定义了一个构建数据集的函数,前面大部分步骤差不多,主要是把 word 拆分成 inputs 和对应的 label, 也就是 `X` 和 `Y`
然后我们用基于 random seed, 进行一个打乱, 然后按照刚才说的 80:10:10 的比例对于 `X` 和 `Y` 进行 slice, 得到三个数据集
## 训练模型

```python
for i in range(200000):
  
  # minibatch construct
  ix = torch.randint(0, Xtr.shape[0], (32,))
  
  # forward pass
  emb = C[Xtr[ix]] # (32, 3, 10)
  h = torch.tanh(emb.view(-1, 30) @ W1 + b1) # (32, 200)
  logits = h @ W2 + b2 # (32, 27)
  loss = F.cross_entropy(logits, Ytr[ix])
  print(loss.item())
  
  # backward pass
  for p in parameters:
    p.grad = None
  loss.backward()
  
  # update
  #lr = lrs[i]
  lr = 0.1 if i < 100000 else 0.01
  for p in parameters:
    p.data += -lr * p.grad

  # track stats
  #lri.append(lre[i])
  stepi.append(i)
  lossi.append(loss.log10().item())

print(loss.item())
```
这里是把训练过程中用的数据改成 Xtr 和 Ytr 了 也就是训练集
此外,还对于 lr 进行了一些修改
由于这里的循环次数很大,因此设置在 10000 次以后 lr 设置为 0.01, 相当于做一些精细的调整吧
## 评估损失

这里我们用验证集来评估一下此时模型的损失
```python
emb = C[Xdev] # (32, 3, 2)
h = torch.tanh(emb.view(-1, 30) @ W1 + b1) # (32, 100)
logits = h @ W2 + b2 # (32, 27)
loss = F.cross_entropy(logits, Ydev)
loss
```
在 AK 的视频中, 此时训练集和验证集的 loss 很接近,说明此时是**欠拟合**的,因此我们可以做一些调整让模型从测试集中学到更多的特征
[[../动手学深度学习/4.4 模型选择、欠拟合与过拟合]]
### 修改隐藏层
于是, 我们就尝试改变模型的结构: 改变隐藏层的**维度**, 相当于给隐藏层加入更多的**神经元**
需要注意的是, 这个改动不仅是针对 W, 同时也要改变 b 等参数, 不然无法满足矩阵运算
刚才说了验证集是帮助我们"训练"超参数的(超参数就是人为设置的那些参数,比如模型结构,lr,batch size等等) 其实上面的更改维度就是训练超参数的一种体现
### 修改嵌入层
在修改了隐藏层的维度后,我们发现模型的提升有限,因此寻找下一个优化点:嵌入层
我们通过嵌入层对于 letter 进行一个映射, 从一个 int 转换成一个嵌入向量
然而, 我们此时设置的嵌入向量的维度仅仅是 2 维, 因此我们猜测可能是维度太低, 表达能力有限,限制了模型的发挥
### 过拟合 vs 欠拟合 : 我的一些疑惑
一边是 train 与 eval 的 loss 差的太少,欠拟合了; 真差多了之后又过拟合了. 到底什么才算一个好的模型的 loss 呢?
我询问 gpt, 得到了以下回答:

>一个理想的状态具备以下特征：
>1. 训练集 loss 持续下降，测试集 loss 也同步下降
>2. eval loss 达到较低水平后开始震荡或略微上升
>3. 此时 eval accuracy 或其他指标在稳定高点
>4. 型没有明显的过度波动、过拟合、不稳定现象
换句话说就是：
> **训练误差下降，泛化误差也在下降，然后我们在泛化误差“最低点”附近保存模型。**
其实感觉主要就是 train 和 eval 的 loss **没有太大差距**, 并且 **loss 都比较低**,应该是个比较理想的状态
AK 当时应该不满足后者这个条件
# 从模型中采样
和 part 1 一样，训练完了模型，我们像看一下实际的预测效果，因此通过采样得到单词
## 为什么要采样
这个其实是我的一个疑惑：为什么不直接用**模型输出的最大概率对应的字母**，而是**基于概率分布通过采样得到字母**
以下是 GPT 给我的回答:
### 贪心策略
我说的那种方法其实是**贪心策略** 这种策略在 CS 50 AI 中曾经学习过, 也常常出现在各种算法中
然而我们可以知道, 贪心只是一种**局部最优**, 并非全局最优, 并不能得到最好的结果
[[../CS50 AI/Week 0 - Search/Lecture - Search]]
### 生成结果重复单调
- 生成的文本“千篇一律”；
- 出现死循环，比如不停生成相同的几个字符；
- 模型学到的概率分布没法体现出来（只取了最大那个，忽略了不确定性）
### 无法体现语言的多样性
比如在给定上下文 “th”，后面可能接 “e”、“a”、“i” 都是合理的词汇开头。但如果你始终只取最大值，你永远只会生成 “the”，而不会生成 “this”、“that”、“thin”等其他可能。
## 代码实现

```python
# sample from the model
g = torch.Generator().manual_seed(2147483647 + 10)

for _ in range(20):
    
    out = []
    context = [0] * block_size # initialize with all ...
    while True:
      emb = C[torch.tensor([context])] # (1,block_size,d)
      h = torch.tanh(emb.view(1, -1) @ W1 + b1)
      logits = h @ W2 + b2
      probs = F.softmax(logits, dim=1)
      ix = torch.multinomial(probs, num_samples=1, generator=g).item()
      context = context[1:] + [ix]
      out.append(ix)
      if ix == 0:
        break
    
    print(''.join(itos[i] for i in out))

```
### 设置随机种子
跟之前的方法一样
[[Makemore - part 1]]
### 初始化
这里设置 `out` 为一个空的 list, 用于存储后续输出的单词
`Context` 也就是用于预测下一个单词的 **上下文** 初始化为 0, 对应的是起始和终止符
### 映射
通过 emb 矩阵, 将 context 映射成嵌入向量
### 运算
用上面训练好参数的 W 和 b, 对于这里的 emb 进行同样的运算, 得到 logits, 并计算出对应的概率
### 采样
使用之前讲过的 `multinomial` 方法, 依据概率采样, 得到预测的下一个单词
[[Makemore - part 1]]
### 滑动窗口
把 ix 填充到 context 尾部, 滑动并更新 context
# 结语

至此, part 2 部分完结
后续准备再精读一下本次参考的那篇论文: A Neural Probabilistic Language Model
难得的在期中考试周拿出那么多时间来学习深度学习, 但是很沉浸很开心!
或许学习 AI 是取代平时玩游戏, 作为对我生活的一种调剂 (开始但是并不轻松)
希望期中也能取得好成绩!