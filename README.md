[toc]

# AFM模型详解

## 背景

> ctr预估中，为了提高预测准确性，通常需要引入特征交互。但是在稀疏数据集中，直接采用Poly模型以product的方式来显示交互，只能观测到少量交叉特征，模型的泛化能力较差。为了解决Poly模型的泛化性问题，提出了FM模型，利用特征隐向量的内积来实现特征交互。通过学习每个特征的隐向量，FM模型可计算每个特征组合的权重。
>
> 虽然FM的效果很好，但是使用相同的权重来对所有的特征组合进行建模，FM的效果会受到影响。在现实应中，不同的特征通常有不同的影响力，并不是所有的特征对于目标变量包含有用的信号，作用小的特征间的组合应该赋予较小的权重。
>
> 基于此，本文提出了AFM模型，它通过区分特征组合的重要性提升FM的效果，引入注意力机制使得不同的特征组合对预估任务贡献不同的作用。在公开数据集上的实验表明，FM引入注意力机制不仅能带来更好的预估效果；还能洞察哪些特征交互对于预估任务贡献更多。

## 模型

img

> AFM模型结构如图所示，Sparse Input和Embedding Layer与FM一样，Embedding Layer把输入特征中非零部分特征embed成一个dense vector。剩下的三层为重点，如下：

### Pair-wise Interaction Layer

> 受FM模型用内积来建模特征间交互的启发，论文提出了Pair-wise Interaction Layer。将输入的m个向量通过element-wise product操作扩展到m(m-1)/2个组合向量。如下：

img

> 其中，<a href="https://www.codecogs.com/eqnedit.php?latex=\odot" target="_blank"><img src="https://latex.codecogs.com/svg.latex?\odot" title="\odot" /></a>定义了两个向量的elemet-wise product操作。
>
> Pair-wise Interaction Layer的输入是所有嵌入向量，输出也是一组向量。输出是任意两个嵌入向量的element-wise product。任意两个嵌入向量都组合得到一个Interacted vector，所以m个嵌入向量得到m(m-1)/2个向量。

> 此外，通过定义pair-wise interaction layer，可以在神经网络架构下表明FM模型。
>
> 首先，用求和池化（sum pooling）来压缩<a href="https://www.codecogs.com/eqnedit.php?latex=f_{PI}(\varepsilon)" target="_blank"><img src="https://latex.codecogs.com/svg.latex?f_{PI}(\varepsilon)" title="f_{PI}(\varepsilon)" /></a>，然后使用全连接层来将压缩结果映射到预估分数：

img

> 其中，<a href="https://www.codecogs.com/eqnedit.php?latex=p\in&space;R^{k}" target="_blank"><img src="https://latex.codecogs.com/svg.latex?p\in&space;R^{k}" title="p\in R^{k}" /></a>和<a href="https://www.codecogs.com/eqnedit.php?latex=b\in&space;R" target="_blank"><img src="https://latex.codecogs.com/svg.latex?b\in&space;R" title="b\in R" /></a>表示预估层的权重和偏差。假设将<a href="https://www.codecogs.com/eqnedit.php?latex=p" target="_blank"><img src="https://latex.codecogs.com/svg.latex?p" title="p" /></a>设置为1，<a href="https://www.codecogs.com/eqnedit.php?latex=b" target="_blank"><img src="https://latex.codecogs.com/svg.latex?b" title="b" /></a>设置为0，就是FM模型。

### Attention-based Pooling Layer

> Attention机制广泛应用在神经网络建模中，主要思想是在压缩不同部分到一个single representation时，允许不同部分贡献不同，论文通过在组合特征向量上做加权求和从而实现attention机制，如下：

img

> 其中，<a href="https://www.codecogs.com/eqnedit.php?latex=a_{ij}" target="_blank"><img src="https://latex.codecogs.com/svg.latex?a_{ij}" title="a_{ij}" /></a>是特征组合权重<a href="https://www.codecogs.com/eqnedit.php?latex=w_{ij}" target="_blank"><img src="https://latex.codecogs.com/svg.latex?w_{ij}" title="w_{ij}" /></a>的attention score，表示不同的特征组合对于最终预测的贡献程度。可以看到：
>
> 1）Attention-based Pooling Layer的输入是Pair-wise Interaction Layer的输出。它包含m(m-1)/2个向量，每个向量的维度是k(k是嵌入向量的维度)，m是embedding layer中嵌入向量的个数；
>
> 2）Attention-based Pooling Layer的输出是一个k维向量。它对Interacted vector使用Attention Score进行了weighted sum pooling（加权求和池化）操作。

> **Attention score的学习是一个问题，**常规做法就是通过最小化loss来学习，但是存在一个问题：对于训练数据中没有共现过的特征们，它们组合的attention分数无法估计。因此论文进一步提出attention network，用多层感知器MLP来参数化attention分数。
>
> Attention network的输入是两个特征的组合向量(在嵌入空间中编码了他们的组合信息)，定义如下：

img

> 可以看到，Attention Network实际上是一个one layer MLP，激活函数使用ReLU，它的输入是两个嵌入向量element-wise product之后的结果(interacted vector，用来在嵌入空间中对组合特征进行编码)；它的输出是组合特征对应的Attention Score。最后，通过softmax函数来归一化attention分数。

> Attention-based Pooling Layer的输出是一个k维向量，它通过区分不同特征组合的重要性，在嵌入空间中压缩所有特征组合。最后project到预测分数。AFM模型的形式如下所示：

img

### Overfitting Prevention

> AFM模型比FM模型有更强的表达能力，更容易过拟合。因此，可以考虑dropout和L2正则化防止过拟合。
>
> 1）对于Pair-wise Interaction Layer，采用dropout避免co-adaptations：AFM对所有的特征组合进行建模，但并不是每一个特征组合都有用，Pair-wise Interaction Layer的神经元容易彼此之间co-adapt，然后导致过拟合；
>
> 2）对于Attention network（one layer MLP），对权重矩阵W使用L2正则化防止过拟合；最终的目标函数，如下所示：

img

## 实验

> 论文通过实验解答了下面三个问题：
>
> 1）超参数dropout、L2正则化对于AFM模型的影响；
>
> 2）Attention network是否能够学习到特征交互的重要性？
>
> 3）与典型的预估模型相比，AFM表现如何？

### 1) Hyper-parameter Investigation(RQ1)

#### dropout

> 将dropout ratio设置为合适的值，FM和AFM的效果均有提升，其中AFM在Frappe、MovieLens两个数据集上的最佳dropout ratio分别是0.2、0.5。

img

#### L2正则化

> L2正则化系数大于0时，AFM模型的效果是提升的，同时也说明仅在Pair-wise Interaction Layer使用dropout无法完全避免AFM过拟合。

img

### 2) Impact of the Attention Network(RQ2)

> 下图说明了不同的attention factors对于AFM效果的影响，从图中可以看出，在不同的attention factors下，AFM的表现相对平稳。

img

> 论文比较了AFM和FM每轮迭代的训练集误差和测试集误差，AFM比FM要收敛的更快。在Frappe上，AFM的训练集误差和测试集误差都要小于FM；在MovieLens上，AFM的训练误差要稍微高于FM，但测试误差要低于FM。

img

### 3) Performance Comparison(RQ3)

> 本节对比了不同模型在测试集上的效果，表2总结了embedding为256条件下的各模型的最好效果，如下所示。

Img

> 根据表2可以看到：
>
> 1）首先，AFM在所有模型中的效果最好。使用不超过0.1m的额外参数下，AFM的效果比LibFM提升8.6%；在使用更少模型参数下，AFM比效果第二好的Wide&Deep提升4.3%；
>
> 2）HOFM通过建模更高阶的特征交互，效果略微好于FM，但参数却多一倍；
>
> 3）DeepCross由于严重的过拟合问题，效果最差。dropout技术在DeepCross中效果一般，可能是使用了BN的原因。DeepCross模型是所有涉及模型中层数最多的，说明更深层次的学习并不一定有帮助，因为深层网络更容易过拟合，也更难优化；

## 代码

https://github.com/Jesse-csj/TensorFlow_Practice/tree/master/ctr_of_recommendation/AFM_Demo