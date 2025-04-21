这里主要是要讲 nlp 相关的内容
#深度学习 #Zero2hero

# 读取文件

```python
words = open('names.txt', 'r').read().splitlines()
```

首先，读取 `txt` 文件，按列分开，这里的 `txt` 文件是 Karpathy 从网上爬下来的人名，这里主要是利用 n-gram 相关的知识进行下一个字符的预测

# 统计“字频”

## 原始方法：词典 + 元组

### 统计

```python
b = {}
for w in words:
  chs = ['<S>'] + list(w) + ['<E>']
  for ch1, ch2 in zip(chs, chs[1:]):
    bigram = (ch1, ch2)
    b[bigram] = b.get(bigram, 0) + 1
```

这里重新创建一个字典 `b`，用于映射并统计

通过遍历 `words` 列表中的每一个词语 `word`，对每个 `word` 进行操作：把 `word`拆成单个字母，并在首位加上 `S`、`E` 表示首位

每次取出连续的两个字符，组成一个 `tuple`，存入 `b` 字典中，更新频率

这里有两个小技巧：

1. 取出连续字符使用 `zip` + 切片实现
2. 利用 `b.get` 这个方法，避免需要额外考虑字典中不包含的情况

### 排序

```python
sorted(b.items(), key = lambda kv: -kv[1])
```

这里进行一个排序，便于展示，其中排序的参照是 `key`，统计次数由高到低展示

## 现代方法：torch 张量

现在我们可以引入 torch，使用现代的方法进行统计并展示

由于 torch 的张量只能是数字，因此需要建立字母到数字的一个映射，然后再来统计

### 创建张量

```python
import torch
N = torch.zeros((27, 27), dtype=torch.int32)
```

首先，导入 torch 库

然后创建一个 27x27 的张量，其中的数据类型是 int，因为映射的话整数足矣

### 为何是 27

至于为什么是 27，而不是 26+2=28，是因为这里把 S、E 这俩起止符号做了一些调整，变成了 . 这个符号，因为在传统方法统计完，花词频图的时候，发现(E, S)这个组合，或者说 (E, x) 其中 x 是任意的字符，因为单词到 E 就没了，所以起始用 28 个有些浪费，因此就简化为了 27 个

### 创建映射

```python
chars = sorted(list(set(''.join(words))))
stoi = {s:i+1 for i,s in enumerate(chars)}
stoi['.'] = 0
itos = {i:s for s,i in stoi.items()}
```

首先，`''.join(words)` ，把所有的单词内部拼起来，去掉空格之类的

然后运用 set，实现去重，并转化为 list，最后把 list 进行排序，便于后面映射

stoi 是建立一个字典，通过 enumerate 取出 list 中的元素以及对应的 index，建立映射

对于分隔符. 将其映射为 0

然后创建一个反向的 itos 映射，用于把统计结果转换回来，后面画图要用

### 统计

```python
for w in words:
  chs = ['.'] + list(w) + ['.']
  for ch1, ch2 in zip(chs, chs[1:]):
    ix1 = stoi[ch1]
    ix2 = stoi[ch2]
    N[ix1, ix2] += 1
```

这里使用 tensor 之后，就不用再创建字典了

这里的统计方式跟前面其实差不多，只是在 tensor 对应的 (i,j) 位置更新频率即可

### 作图

```python
import matplotlib.pyplot as plt
%matplotlib inline

plt.figure(figsize=(16,16))
plt.imshow(N, cmap='Blues')
for i in range(27):
    for j in range(27):
        chstr = itos[i] + itos[j]
        plt.text(j, i, chstr, ha="center", va="bottom", color='gray')
        plt.text(j, i, N[i, j].item(), ha="center", va="top", color='gray')
plt.axis('off');
```

这里导入 matplotlib 这个库，用于作图

首先指定 figure 的大小，指定展示方式 imshow

然后遍历每个张量每个位置的值，对应的就是 (i,j) 对应的字母组合的出现频率，以这个频率为深浅，进行绘图

# 计算概率

## 归一化

这里需要先把频率转化为概率，具体的方法是 频率/总频率 ，然后得到一个概率

```python
p = N[0].float()
p = p / p.sum()
p
```

这里的 N[0]是指第 0 个字符后，所有字符出现的次数

然后对于 p 求和，然后更新 p，实现归一化，得到字符出现的频率

## 随机采样

现在已经得到了 p 这个分布，p 是一个概率向量，元素之和为 1

后面需要用 multinomial 函数进行随机采样，大概就是根据 p 分布来随机取这些数字，然后再映射回对应的字母，以此实现对下一个字母的预测

```python
g = torch.Generator().manual_seed(2147483647)
ix = torch.multinomial(p, num_samples=1, replacement=True, generator=g).item()
itos[ix]
```

这里用到采样，需要考虑随机相关的事情，这与 seed 有关

为了保证采样的结果是可以复现的，我们这里手动指定一个固定的种子，用这个种子来作为这里的generator参数辅助采样

> 只要种子不变，每次运行都会得到相同的采样结果

multinomial 具体的参数含义：

p 是概率分布

num_samples 是输出的个数（这里仅仅预测下一个字母，因此使用 1）

replacement 表示是否放回，这里设置为 True，表示进行**有放回的采样**，也就是每次采样后得到的元素可以放回

generator 是用来控制随机数生成器的，已经通过 manual_seed 设置好了固定的随机种子

## 再谈归一化 —— keepdims

刚才我们只是取了整个概率分布 P 中的一行/列来进行尝试和示范，现在我们希望直接对 P 进行操作，对每个字母后的字母分布的概率情况统一进行归一化

```python
P = (N+1).float()
P /= P.sum(1, keepdims=True)
```

这里开启了 keepdims，也就是加和时保留维度

这里涉及到了 torch 中 tensor 的求和机制以及乘除中的广播机制，其实他讲那么半天我还是不是很明白为什么一定要 keepdims，包括对于 tensor 的各个维度的顺序及表达其实还是有点懵

经过询问 gpt 后，现在了解到的是，在归一化中使用 keepdims 是标准操作，可以保证后面利用广播机制时不会有问题，也能明确 shape，算是一个好习惯吧

## 尝试生成单词（人名）

```python
g = torch.Generator().manual_seed(2147483647)

for i in range(5):
  
  out = []
  ix = 0
  while True:
    p = P[ix]
    ix = torch.multinomial(p, num_samples=1, replacement=True, generator=g).item()
    out.append(itos[ix])
    if ix == 0:
      break
  print(''.join(out))
```

现在可以尝试用刚才学到的概率分布、随机采样来进行下一个单词的预测，从而实现生成单词/人名了（因为概率其实是从一堆人名中统计而来）

这里具体实现的方式是while，看什么时候采样结果是 . ，也就是“停止符”，这时便代表预测的单词已经结束

```
mor.
axx.
minaymoryles.
kondlaisah.
anchshizarie.
```

可以看一下生成的结果，非常奇怪，主要是因为这里采用的是 bigram，也就是通过前一个字母预测后一个字母，它的效果一般，大概因为无法“学习”到单词的全貌吧

# 计算损失 Loss

由于预测表现不佳，我们想要优化这个模型

为了有更加明确的优化方向，我们首先需要定义什么叫“优”

相应地，我们可以通过量化这个模型到底有多差，来反映模型质量

因此，我们需要定义 Loss 函数，用来量化模型质量

## 负对数 -log

```python
log_likelihood = 0.0
n = 0

for w in words:
#for w in ["andrejq"]:
  chs = ['.'] + list(w) + ['.']
  for ch1, ch2 in zip(chs, chs[1:]):
    ix1 = stoi[ch1]
    ix2 = stoi[ch2]
    prob = P[ix1, ix2]
    logprob = torch.log(prob)
    log_likelihood += logprob
    n += 1
    #print(f'{ch1}{ch2}: {prob:.4f} {logprob:.4f}')

print(f'{log_likelihood=}')
nll = -log_likelihood
print(f'{nll=}')
print(f'{nll/n}')
```

现在的问题来到了 Loss 函数的计算规则该如何制定，这里我们选择 -log 作为 Loss 函数，具体原因如下：

1. 概率之间是乘法的关系，而 log 作用后可以把乘法变为加法
2. log 会让差别很小的概率有明显的区别
3. 这个其实就是交叉熵损失，主要用于分类问题
4. 取负号是因为，通常来说，loss 越低，模型越好，因此加个负号后正好符合

这里我们通过遍历单词中每个字母，获取前后字母组合对应的概率，经过 log 后累加，最后再取负

于是变得到了初步的 loss

## 拆分 data 与 label

```python
# create the training set of bigrams (x,y)
xs, ys = [], []

for w in words[:1]:
  chs = ['.'] + list(w) + ['.']
  for ch1, ch2 in zip(chs, chs[1:]):
    ix1 = stoi[ch1]
    ix2 = stoi[ch2]
    print(ch1, ch2)
    xs.append(ix1)
    ys.append(ix2)
    
xs = torch.tensor(xs)
ys = torch.tensor(ys)
```

这里由于我们需要计算 loss，因此也需要相应地一些有 label 的数据用来判断预测的准确性

这里以现有的单词为例：遍历单词，取出字母对，其中第一个字母是 data，第二个字母是 label

因为我们想要实现的是根据这个单词的首字母，预测整个单词，因此用后一个字母作为前一个input 的 label

然后把这里的 input 和对应的 label 分别存进对应的 list 之中，在这里就是 xs 和 ys

## 独热编码 onehot

之前说了我们采用字符和数字间的映射，用数字表示字母

然而，数字虽然相对方便处理，但是仍有一定的局限性：数字有天然的顺序，可能会对计算机有一定的误导

对于这种实际上是离散分布的情况，我们选择 onehot 编码的方式，用onehot 向量来表示分类

对于 onehot 向量，之前其实在其他课程遇到过很多很多次了，其实就是有 n 个数，对应 n 个类别，某向量表示的类别的位置是 1，其余都是0

onehot 有很多好处：

1. 避免数字编码的顺序性误导，清晰表示离散类别
2. 让神经网络输入层可以接收数值向量，进行有效计算
3. 与矩阵乘法配合良好，实现高效的类别映射
4. 是 embedding 层的基础概念，后续可以扩展到更高效的表示方法。（这个我不太了解，gpt 给的）

```python
import torch.nn.functional as F
xenc = F.one_hot(xs, num_classes=27).float()
xenc
```

这里使用 onehot 编码（在这里是 F 的一个方法），把 inputs 转化为 onehot 向量，其中每个向量有 27 个维度（对应 27 种字母）

同时设置为 float 类型，方便后续运算

## 矩阵乘法

```python
W = torch.randn((27, 1))
xenc @ W
```

这里随机生成了一个矩阵 W，利用 @ 这个运算符计算 xenc 向量和 W 矩阵的乘法

这里需要注意矩阵的shape，这里 xenc 的shape 是5x27 因此我们这里矩阵也应该是 27x 某值，这里设置的是 1

## Softmax

```python
logits = xenc @ W # log-counts
counts = logits.exp() # equivalent N
probs = counts / counts.sum(1, keepdims=True)
probs
```

这里似乎是在手动实现一个 softmax？

首先我们得到了矩阵乘法后得到的 logits 概率

然而，由于 W 是随机生成的，有正有负，为了后续进行 normalize，再加上概率也不能为负，所以我们统一进行 exp 运算，把所有元素变为正数

然后再像之前一样做一下归一化，就能得到一个概率矩阵了

## 计算细节

```python

nlls = torch.zeros(5)
for i in range(5):
  # i-th bigram:
  x = xs[i].item() # input character index
  y = ys[i].item() # label character index
  print('--------')
  print(f'bigram example {i+1}: {itos[x]}{itos[y]} (indexes {x},{y})')
  print('input to the neural net:', x)
  print('output probabilities from the neural net:', probs[i])
  print('label (actual next character):', y)
  p = probs[i, y]
  print('probability assigned by the net to the the correct character:', p.item())
  logp = torch.log(p)
  print('log likelihood:', logp.item())
  nll = -logp
  print('negative log likelihood:', nll.item())
  nlls[i] = nll

print('=========')
print('average negative log likelihood, i.e. loss =', nlls.mean().item())
```

这里作者用了一大段来打印很多东西，主要是训练时的细节，比如预测概率、准确性等等

前面其实说的都差不多，这里我对 nll 加深了理解，特地记录一下

这里的 nll 其实是指 negative log likelihood，也就是对 likelihood 取负对数

这里取出 p，也就是对应正确标签 y 的概率，也就是这里说的 likelihood

然后进行负对数运算，得到我们这里的 nll 对应的是 loss

这里取负对数的原因其实和上面 softmax 差不多：

1. **概率相乘容易数值下溢，用 log 转成相加。**
2. **优化目标是最小化 loss，而 likelihood 本身是最大化的目标，所以加一个负号，变成常用的 “最小化问题”。**

以下是 likelihood 和 loss 的关系：

- Likelihood = 预测正确答案的概率，值在 (0, 1)
- Log likelihood = 对概率取对数，值是负的。
- Negative log likelihood (NLL) = - log likelihood = 损失函数

主要是为了简化运算（乘法变加法），防止数值下溢（相乘后概率过小导致直接舍没了）

# 优化

## 小尝试

### 初始化

```python
# randomly initialize 27 neurons' weights. each neuron receives 27 inputs
g = torch.Generator().manual_seed(2147483647)
W = torch.randn((27, 27), generator=g, requires_grad=True)
```

首先，需要初始化，一个是 g，用来保持随机结果不变

然后是 W，是模型要学的参数矩阵，需要进行随机初始化

由于这里要开始优化了，需要使用梯度下降这一方法，因此这个 W 矩阵需要 requires_grad=True

### 前向传播

```python
# forward pass
xenc = F.one_hot(xs, num_classes=27).float() # input to the network: one-hot encoding
logits = xenc @ W # predict log-counts
counts = logits.exp() # counts, equivalent to N
probs = counts / counts.sum(1, keepdims=True) # probabilities for next character
loss = -probs[torch.arange(5), ys].log().mean()
```

其实这些上面基本都有，只是写一块了

1. 变成 onehot 向量
2. 矩阵乘法
3. 对原始得分（logits 到底是啥）先 exp 再归一化，得到概率
4. 对预测正确（在 probs 中寻找（x,y）数对对应的概率）的概率取负对数，得到这里的 loss

`probs[torch.arange(5), ys]` 这个值得一提，其实是一种方便的写法，直接可以取好几个，我也说不太清楚，看下面 gpt 的解释吧

![image.png](attachment:e286a3aa-e809-4ac3-90b3-c5e544c0f1ee:image.png)

### 反向传播

```python
# backward pass
W.grad = None # set to zero the gradient
loss.backward()
```

万事俱备，现在我们把 W 的梯度清零，然后对 loss 进行反向传播

### 更新参数

```python
W.data += -0.1 * W.grad
```

反向传播后，其实已经计算并记住了梯度，我们可以选择根据梯度来更新 W 的参数，其中这里的 0.1

其实是自己设置的值（这个是学习率吗？），用来调整参数更新快慢

这里学习率需要取负值，因为梯度是函数上升最快的方向，然而这里的函数是 loss，我们希望它下降，因此沿着反方向走就好

## 完整的训练过程

### 创建数据集 初始化

```python
# create the dataset
xs, ys = [], []
for w in words:
  chs = ['.'] + list(w) + ['.']
  for ch1, ch2 in zip(chs, chs[1:]):
    ix1 = stoi[ch1]
    ix2 = stoi[ch2]
    xs.append(ix1)
    ys.append(ix2)
xs = torch.tensor(xs)
ys = torch.tensor(ys)
num = xs.nelement()
print('number of examples: ', num)

# initialize the 'network'
g = torch.Generator().manual_seed(2147483647)
W = torch.randn((27, 27), generator=g, requires_grad=True)
```

### 梯度下降

```python
# gradient descent
for k in range(1):
  
  # forward pass
  xenc = F.one_hot(xs, num_classes=27).float() # input to the network: one-hot encoding
  logits = xenc @ W # predict log-counts
  counts = logits.exp() # counts, equivalent to N
  probs = counts / counts.sum(1, keepdims=True) # probabilities for next character
  loss = -probs[torch.arange(num), ys].log().mean() + 0.01*(W**2).mean()
  print(loss.item())
  
  # backward pass
  W.grad = None # set to zero the gradient
  loss.backward()
  
  # update
  W.data += -50 * W.grad
```

### 看看结果如何

```python
# finally, sample from the 'neural net' model
g = torch.Generator().manual_seed(2147483647)

for i in range(5):
  
  out = []
  ix = 0
  while True:
    
    # ----------
    # BEFORE:
    #p = P[ix]
    # ----------
    # NOW:
    xenc = F.one_hot(torch.tensor([ix]), num_classes=27).float()
    logits = xenc @ W # predict log-counts
    counts = logits.exp() # counts, equivalent to N
    p = counts / counts.sum(1, keepdims=True) # probabilities for next character
    # ----------
    
    ix = torch.multinomial(p, num_samples=1, replacement=True, generator=g).item()
    out.append(itos[ix])
    if ix == 0:
      break
  print(''.join(out))
```

这里我们的 W 已经经过了多轮的训练，得到了优化，现在来看一下效果如何

我们和之前一样，依旧是经过一系列操作得到概率 p，根据这个 p 来进行随机采样，预测下一个字母，最终得到完整单词

这一节课就差不多了，虽说好多东西在各种课程里也都听了，但是细致地一点点抠下来还是有很大的收获提升和感悟的，希望能跟着这个系列继续学下去！

完成于 2025.4.11 凌晨 1：02 于雁北的自习室中

附一张记录：
![[../../source/2db6a44ea56290eccd52257c486f53b8.jpg]]