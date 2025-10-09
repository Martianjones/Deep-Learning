# Deep-Learning

编译平台VS Code，使用语言python 3.9

记录本人学习深度学习的过程，参考书目可在zh.d2l.ai获取

## 1. 引言

### 1.1 机器学习中的关键组件

1、可以用来学习的数据(data);

2、如何转换数据的模型(model);

3、一个目标函数(objective function),用来量化模型的有效性;

4、调整模型参数以优化目标函数的算法。

### 1.2 机器学习中的 各种问题

#### 1.2.1 监督学习

监督学习擅长在“给定输入特征”的情况下预测标签。每个“特征-标签”对都称为一个样本。

监督学习的学习过程一般可以分为三大步骤：

1、从已知大量数据样本中随机选取一个子集，为每个样本获取真实标签。有时，这些样本已有标签；有时这些样本可能需要被人工标记。这些输入和相应的标签一起构成了训练数据集；

2、选择有监督的学习算法，它将训练数据作为输入，并输出一个“已完成学习的模型”；

3、将之前没有见过样本特征放到这个“已完成学习的模型”中，使用模型的输出作为相应的预测。

回归 （regression）是最简单的监督学习任务之一。

#### 1.2.2 无监督学习

这类数据中不含有“目标”的机器学习问题通常被为 *无监督学习* （unsupervised learning）

无监督学习可以用于回答下列问题：

* *聚类* （clustering）问题：没有标签的情况下，我们是否能给数据分类呢？比如，给定一组照片，我们能把它们分成风景照片、狗、婴儿、猫和山峰的照片吗？同样，给定一组用户的网页浏览记录，我们能否将具有相似行为的用户聚类呢？
* *主成分分析* （principal component analysis）问题：我们能否找到少量的参数来准确地捕捉数据的线性相关属性？比如，一个球的运动轨迹可以用球的速度、直径和质量来描述。再比如，裁缝们已经开发出了一小部分参数，这些参数相当准确地描述了人体的形状，以适应衣服的需要。另一个例子：在欧几里得空间中是否存在一种（任意结构的）对象的表示，使其符号属性能够很好地匹配?这可以用来描述实体及其关系，例如“罗马”  “意大利”  “法国”  “巴黎”。
* *因果关系* （causality）和 *概率图模型* （probabilistic graphical models）问题：我们能否描述观察到的许多数据的根本原因？例如，如果我们有关于房价、污染、犯罪、地理位置、教育和工资的人口统计数据，我们能否简单地根据经验数据发现它们之间的关系？
* *生成对抗性网络* （generative adversarial networks）：为我们提供一种合成数据的方法，甚至像图像和音频这样复杂的非结构化数据。潜在的统计机制是检查真实和虚假数据是否相同的测试，它是无监督学习的另一个重要而令人兴奋的领域。

#### 1.2.3 强化学习

在强化学习问题中，智能体（agent）在一系列的时间步骤上与环境交互。 在每个特定时间点，智能体从环境接收一些 *观察* （observation），并且必须选择一个 *动作* （action），然后通过某种机制（有时称为执行器）将其传输回环境，最后智能体从环境中获得 *奖励* （reward）。

## 2. 预备知识

### 2.1 数据操作

#### 2.1.1 入门

张量表示一个由数值组成的数组，这个数组可能有多个维度。 具有一个轴的张量对应数学上的 *向量* （vector）； 具有两个轴的张量对应数学上的 *矩阵* （matrix）； 具有两个轴以上的张量没有特殊的数学名称。张量是矩阵向更高维度的推广。

##### 2.1.1.1 张量基本属性

张量具有以下几个关键属性：

* **阶（Rank）** ：张量的阶是指它的维度数量。例如，一个标量（单个数字）是0阶张量，一个向量（数字数组）是1阶张量，而一个矩阵（数字的二维数组）是2阶张量。
* **形状（Shape）** ：张量的形状描述了它在每个维度上的大小。例如，一个形状为(3, 5)的矩阵表示它有3行5列。
* **数据类型（Type）** ：张量中元素的数据类型，如float32、float64等。

##### 2.1.1.2 张量的几何解释

从几何角度来看，张量中的元素可以被视为某个高维空间中点的坐标。例如，一个2阶张量可以表示一个平面上的点集，而3阶张量可以表示一个立体空间中的点集。

##### 2.1.1.3 张量的运算

深度学习模型的训练和预测涉及大量的张量运算，包括但不限于：

* **逐元素运算** ：如加法、减法、乘法等，这些运算是对张量中每个元素独立进行的。
* **广播** ：当对形状不同的张量进行运算时，较小的张量会被“广播”以匹配较大张量的形状。
* **点积** ：也称为内积或张量积，是将两个张量的对应元素相乘后求和的运算。
* **变形** ：改变张量的形状而不改变其数据。
* **转置** ：在矩阵运算中，转置是将矩阵的行列互换的操作。

#### 2.1.2 运算符

对于任意具有相同形状的张量， 常见的标准算术运算符 `+`、-、`*`、`/`和 `**`都可以被升级为按元素运算。(**是求幂符号)

#### 2.1.3 广播机制

在上面的部分中，我们看到了如何在相同形状的两个张量上执行按元素操作。 在某些情况下，即使形状不同，我们仍然可以通过调用  *广播机制* （broadcasting mechanism）来执行按元素操作。 这种机制的工作方式如下：

* 通过适当复制元素来扩展一个或两个数组，以便在转换之后，两个张量具有相同的形状；
* 对生成的数组执行按元素操作。

#### 2.1.4 索引和切片

张量中的元素可以通过索引访问。与任何python数组一样：第一个元素的索引是0，最后一个元素是-1；可以指定范围以包含第一个元素和最后一个之前的元素。

除读取以外，我们还可以通过指定索引来将元素写入矩阵

#### 2.1.5 节省内存

运行一些操作可能会导致为新结果分配内存。 一个变量在经过重新赋值之后，会为其分配新的内存空间。

这可能是不可取的，原因有两个：

1. 首先，我们不想总是不必要地分配内存。在机器学习中，我们可能有数百兆的参数，并且在一秒内多次更新所有参数。通常情况下，我们希望原地执行这些更新；
2. 如果我们不原地更新，其他引用仍然会指向旧的内存位置，这样我们的某些代码可能会无意中引用旧的参数。

我们可以使用切片表示法，将操作的结果分配给先前分配的数组，例如Y[:]=`<expression>`

#### 2.1.6 转换为其他python对象

可以使用numpy()方法转换torch的张量，同时可以使用torch.tensor()方法来进行修改

要将大小为1的张量转换为python标量，可以调用item函数或python的内置函数

### 2.2 数据预处理

#### 2.2.1 读取数据集

使用pandas包来读取

#### 2.2.4 处理缺失值

注意，“NaN”项代表缺失值。 为了处理缺失的数据，典型的方法包括*插值法*和 *删除法* ， 其中插值法用一个替代值弥补缺失值，而删除法则直接忽略缺失值。

通过位置索引 `iloc`，我们将 `data`分成 `inputs`和 `outputs`， 其中前者为 `data`的前两列，而后者为 `data`的最后一列。 对于 `inputs`中缺少的数值，我们用同一列的均值替换“NaN”项。

对于 `inputs`中的类别值或离散值，我们将“NaN”视为一个类别。 由于“巷子类型”（“Alley”）列只接受两种类型的类别值“Pave”和“NaN”， `pandas`可以自动将此列转换为两列“Alley_Pave”和“Alley_nan”。 巷子类型为“Pave”的行会将“Alley_Pave”的值设置为1，“Alley_nan”的值设置为0。 缺少巷子类型的行会将“Alley_Pave”和“Alley_nan”分别设置为0和1。

### 2.3 线性代数

#### 2.3.1 标量

严格来说，仅包含一个数值被称为 *标量* （scalar）。

本书采用了数学表示法，其中标量变量由普通小写字母表示（例如，x、y和 z）。 本书用R表示所有（连续）*实数*标量的空间，之后将严格定义 *空间* （space）是什么， 但现在只要记住表达式x∈R是表示x是一个实值标量的正式形式。 符号∈称为“属于”，它表示“是集合中的成员”。

标量由只有一个元素的张量表示。

#### 2.3.2 向量

向量可以被视为标量值组成的列表。 这些标量值被称为向量的 *元素* （element）或 *分量* （component）。当向量表示数据集中的样本时，它们的值具有一定的现实意义。人们通过一维张量表示向量。我们可以使用下标来引用向量的任一元素。

向量只是一个数字数组，就像每个数组都有一个长度一样，每个向量也是如此。向量的长度通常称为向量的 *维度* （dimension）。

当用张量表示一个向量（只有一个轴）时，我们也可以通过 `<span class="pre">.shape</span>`属性访问向量的长度。 形状（shape）是一个元素组，列出了张量沿每个轴的长度（维数）。 对于只有一个轴的张量，形状只有一个元素。

为了清楚起见，我们在此明确一下： *向量*或*轴*的维度被用来表示*向量*或*轴*的长度，即向量或轴的元素数量。 然而，张量的维度用来表示张量具有的轴数。 在这个意义上，张量的某个轴的维数就是这个轴的长度。

#### 2.3.3 矩阵

正如向量将标量从零阶推广到一阶，矩阵将向量从一阶推广到二阶。

矩阵是有用的数据结构：它们允许我们组织具有不同模式的数据。尽管单个向量的默认方向是列向量，但在表示表格数据集的矩阵中， 将每个数据样本作为矩阵中的行向量更为常见。

#### 2.3.4 张量

就像向量是标量的推广，矩阵是向量的推广一样，我们可以构建具有更多轴的数据结构。 张量（本小节中的“张量”指代数对象）是描述具有任意数量轴的维数组的通用方法。

#### 2.3.5 张量算法的基本性质

标量、向量、矩阵和任意数量轴的张量（本小节中的“张量”指代数对象）有一些实用的属性。

具体而言，两个矩阵的按元素乘法称为 *Hadamard积* （Hadamard product）

将张量乘以或加上一个标量不会改变张量的形状，其中张量的每个元素都将与标量相加或相乘。

#### 2.3.6 降维

我们可以对任意张量进行的一个有用的操作是计算其元素的和。

我们可以使用sum函数来表示任意形状张量的元素和。

默认情况下，调用求和函数会沿所有的轴降低张量的维度，使它变为一个标量。 我们还可以指定张量沿哪一个轴来通过求和降低维度。

##### 2.3.6.1 非降维求和

但是，有时在调用函数来计算总和或均值时保持轴数不变会很有用。例如，由于 `sum_A`在对每行进行求和后仍保持两个轴，我们可以通过广播将 `A`除以 `sum_A`。cumsum函数不会沿任何轴降低输入张量的维度，该函数会进行逐行求和。

#### 2.3.7 点积

我们已经学习了按元素操作、求和及平均值。 另一个最基本的操作之一是点积。给定两个向量x,y，他们的点积是相同位置按元素乘积的和

点积在很多场合都很有用。将两个向量规范化得到单位长度后，点积表示它们夹角的余弦。

#### 2.3.8 矩阵-向量积

矩阵向量积Ax是一个长度为m的列向量，其第i个元素是点积aiTx；

在代码中使用张量表示矩阵-向量积，我们使用与点积相同的 `dot`函数。当我们为矩阵A和向量x调用dot时，会执行矩阵-向量积。需要注意的时矩阵的列数必须与x的维数相同。

#### 2.3.9 矩阵-矩阵乘法

在掌握点积和矩阵-向量积的知识后， 那么 **矩阵-矩阵乘法** （matrix-matrix multiplication）应该很简单。可以想象是由多个列向量与矩阵相乘得到一个新的矩阵。

#### 2.3.10 范数

线性代数中最有用的一些运算符是 *范数* （norm）。 非正式地说，向量的*范数*是表示一个向量有多大。 即范数是向量长度或大小的一种度量方式。这里考虑的 *大小* （size）概念不涉及维度，而是分量的大小。在线性代数中，向量范数是将向量映射到标量的函数f。给定任意向量x，向量范数要满足一些属性。第一个性质是：如果我们按常熟银子α缩放向量的所有元素，其范数也会按相同常数因子的绝对值缩放；

第二个性质是熟悉的三角不等式：f(x+y)<=f(x)+f(y)

第三个性质是范数必须非负，范数为零当且仅当向量全由0组成

##### 2.3.10.1 范数和目标

在深度学习中，我们经常试图解决优化问题： *最大化*分配给观测数据的概率; *最小化*预测和真实观测之间的距离。 用向量表示物品（如单词、产品或新闻文章），以便最小化相似项目之间的距离，最大化不同项目之间的距离。 目标，或许是深度学习算法最重要的组成部分（除了数据），通常被表达为范数。

### 2.4 微积分

在2500年前，古希腊人把一个多边形分成三角形，并把它们的面积相加，才找到计算多边形面积的方法。 为了求出曲线形状（比如圆）的面积，古希腊人在这样的形状上刻内接多边形。内接多边形的等长边越多，就越接近圆。 这个过程也被称为 *逼近法* （method of exhaustion）。

事实上，逼近法就是 *积分* （integral calculus）的起源。

在深度学习中，我们“训练”模型，不断更新它们，使它们在看到越来越多的数据时变得越来越好。 通常情况下，变得更好意味着最小化一个 *损失函数* （loss function）， 即一个衡量“模型有多糟糕”这个问题的分数。 最终，我们真正关心的是生成一个模型，它能够在从未见过的数据上表现良好。 但“训练”模型只能将模型与我们实际能看到的数据相拟合。 因此，我们可以将拟合模型的任务分解为两个关键问题：

* *优化* （optimization）：用模型拟合观测数据的过程；
* *泛化* （generalization）：数学原理和实践者的智慧，能够指导我们生成出有效性超出用于训练的数据集本身的模型。

#### 2.4.1 导数和微分

我们首先讨论导数的计算，这是几乎所有深度学习优化算法的关键步骤。 在深度学习中，我们通常选择对于模型参数可微的损失函数。 简而言之，对于每个参数， 如果我们把这个参数*增加*或*减少*一个无穷小的量，可以知道损失会以多快的速度增加或减少。

#### 2.4.2 偏导数

到目前为止，我们只讨论了仅含一个变量的函数的微分。 在深度学习中，函数通常依赖于许多变量。 因此，我们需要将微分的思想推广到 *多元函数* （multivariate function）上。

#### 2.4.3 梯度

我们可以连结一个多元函数对其所有变量的偏导数，以得到该函数的 *梯度* （gradient）向量。

#### 2.4.4 链式法则

然而，上面方法可能很难找到梯度。 这是因为在深度学习中，多元函数通常是 *复合* （composite）的， 所以难以应用上述任何规则来微分这些函数。 幸运的是，链式法则可以被用来微分复合函数。

### 2.5 自动微分

正如 [2.4节](https://zh.d2l.ai/chapter_preliminaries/calculus.html#sec-calculus)中所说，求导是几乎所有深度学习优化算法的关键步骤。 虽然求导的计算很简单，只需要一些基本的微积分。 但对于复杂的模型，手工进行更新是一件很痛苦的事情（而且经常容易出错）。

深度学习框架通过自动计算导数，即 *自动微分* （automatic differentiation）来加快求导。 实际中，根据设计好的模型，系统会构建一个 *计算图* （computational graph）， 来跟踪计算是哪些数据通过哪些操作组合起来产生输出。 自动微分使系统能够随后反向传播梯度。 这里， *反向传播* （backpropagate）意味着跟踪整个计算图，填充关于每个参数的偏导数。

#### 2.5.1 非标量变量的反向传播

当 `y`不是标量时，向量 `y`关于向量 `x`的导数的最自然解释是一个矩阵。 对于高阶和高维的 `y`和 `x`，求导的结果可以是一个高阶张量。

然而，虽然这些更奇特的对象确实出现在高级机器学习中（包括深度学习中）， 但当调用向量的反向计算时，我们通常会试图计算一批训练样本中每个组成部分的损失函数的导数。 这里，我们的目的不是计算微分矩阵，而是单独计算批量中每个样本的偏导数之和。

#### 2.5.2 分离计算

有时，我们希望将某些计算移动到记录的计算图之外。 例如，假设 `y`是作为 `x`的函数计算的，而 `z`则是作为 `y`和 `x`的函数计算的。 想象一下，我们想计算 `z`关于 `x`的梯度，但由于某种原因，希望将 `y`视为一个常数， 并且只考虑到 `x`在 `y`被计算后发挥的作用。

这里可以分离 `y`来返回一个新变量 `u`，该变量与 `y`具有相同的值， 但丢弃计算图中如何计算 `y`的任何信息。 换句话说，梯度不会向后流经 `u`到 `x`。 因此，下面的反向传播函数计算 `z=u*x`关于 `x`的偏导数，同时将 `u`作为常数处理， 而不是 `z=x*x*x`关于 `x`的偏导数。

#### 2.5.3 Python控制流的梯度计算

使用自动微分的一个好处是： 即使构建函数的计算图需要通过Python控制流（例如，条件、循环或任意函数调用），我们仍然可以计算得到的变量的梯度。 在下面的代码中，`while`循环的迭代次数和 `if`语句的结果都取决于输入 `a`的值。

### 2.6 概率

简单地说，机器学习就是根据概率做出预测。

#### 2.6.1 基本概率论

在统计学中，我们把从概率分布中抽取样本的过程称为 *抽样* （sampling）。 笼统来说，可以把 *分布* （distribution）看作对事件的概率分配。将概率分配给一些离散选择的分布称为 *多项分布* （multinomial distribution）。

为了抽取一个样本，即掷骰子，我们只需传入一个概率向量。 输出是另一个相同长度的向量：它在索引处的值是采样结果中出现的次数。在估计一个骰子的公平性时，我们希望从同一分布中生成多个样本。如果用Python的for循环来完成这个任务，速度会巨慢。因此我们使用深度学习框架的函数同时抽取多个样本，得到我们想要的任意形状的独立样本数组。

##### 2.6.1.1 概率论公理

在处理骰子掷出时，我们将集合S={1，2，3，4，5，6} 称为 *样本空间* （sample space）或 *结果空间* （outcome space）， 其中每个元素都是 *结果* （outcome）。  *事件* （event）是一组给定样本空间的随机结果。*概率* （probability）可以被认为是将集合映射到真实值的函数。

在给定的样本空间S中，事件A的概率， 表示为P(A)，满足以下属性：

* 对于任意事件，其概率从不会是负数
* 整个样本空间的概率为1
* 对于 *互斥* （mutually exclusive）事件（对于所有都有）的任意一个可数序列，序列中任意一个事件发生的概率等于它们各自发生的概率之和，即。

##### 2.6.1.2 随机变量

在我们掷骰子的随机实验中，我们引入了 *随机变量* （random variable）的概念。 随机变量几乎可以是任何数量，并且它可以在随机实验的一组可能性中取一个值。为了简化符号，一方面，我们可以将P(A)表示为随机变量X上的 *分布* （distribution）： 分布告诉我们获得某一值的概率。 另一方面，我们可以简单用P(a)表示随机变量取值a的概率。

请注意， *离散* （discrete）随机变量（如骰子的每一面） 和 *连续* （continuous）随机变量（如人的体重和身高）之间存在微妙的区别。

#### 2.6.2 处理多个随机变量

很多时候，我们会考虑多个随机变量。 比如，我们可能需要对疾病和症状之间的关系进行建模。 给定一个疾病和一个症状，比如“流感”和“咳嗽”，以某个概率存在或不存在于某个患者身上。 我们需要估计这些概率以及概率之间的关系，以便我们可以运用我们的推断来实现更好的医疗服务。

再举一个更复杂的例子：图像包含数百万像素，因此有数百万个随机变量。 在许多情况下，图像会附带一个 *标签* （label），标识图像中的对象。 我们也可以将标签视为一个随机变量。 我们甚至可以将所有元数据视为随机变量，例如位置、时间、光圈、焦距、ISO、对焦距离和相机类型。 所有这些都是联合发生的随机变量。 当我们处理多个随机变量时，会有若干个变量是我们感兴趣的。

##### 2.6.2.1 联合概率

第一个被称为 *联合概率* （joint probability）P(A=a,B=b)。 给定任意值a和b，联合概率可以回答：A=a和B=b同时满足的概率是多少？ 请注意，对于任何a和b的取值，P(A=a,B=b)<=P(A=a)。

##### 2.6.2.2 条件概率

P(B=b|A=a)表示：它是B=b的概率，前提是A=a已发生。

##### 2.6.2.3 贝叶斯定理

##### 2.6.2.4 边际化

为了能进行事件概率求和，我们需要 *求和法则* （sum rule）， 即的概率相当于计算的所有可能选择，并将所有选择的联合概率聚合在一起。这也称为 *边际化* （marginalization）。 边际化结果的概率或分布称为 *边际概率* （marginal probability） 或 *边际分布* （marginal distribution）。

##### 2.6.2.5 独立性

另一个有用属性是 *依赖* （dependence）与 *独立* （independence）。 如果两个随机变量A和B是独立的，意味着事件A的发生跟事件B的发生无关。 在这种情况下，统计学家通常将这一点表述为A⊥B。 根据贝叶斯定理，马上就能同样得到P(A|B)=P(A)。 在所有其他情况下，我们称A和B依赖。

#### 2.6.3 期望和方差

### 2.7 查阅文档

#### 2.7.1 查找模块中的所有函数和类

为了知道模块中可以调用哪些函数和类，可以调用dir函数。 通常可以忽略以"__"(双下划线)开始和结束的函数，它们时Python中的特殊对象，或以单个"_"(单下划线)开始的函数，他们通常是内部函数。根据剩余的函数名或属性名，我们可能会猜测这个模块提供了各种生成随机数的方法，包括从均匀分布、正态分布和多项分布中采样。

#### 2.7.2 查找特定函数和类的用法

有关如何使用给定函数或类的更具体说明，可以调用 `help`函数。

## 线性神经网络

### 3.1 线性回归

 *回归* （regression）是能为一个或多个自变量与因变量之间关系建模的一类方法。 在自然科学和社会科学领域，回归经常用来表示输入和输出之间的关系。

在机器学习领域中的大多数任务通常都与 *预测* （prediction）有关。 当我们想预测一个数值时，就会涉及到回归问题。 常见的例子包括：预测价格（房屋、股票等）、预测住院时间（针对住院病人等）、 预测需求（零售销量等）。 但不是所有的*预测*都是回归问题。 在后面的章节中，我们将介绍分类问题。分类问题的目标是预测数据属于一组类别中的哪一个。

#### 3.1.1 线性回归的基本元素

*线性回归* （linear regression）可以追溯到19世纪初， 它在回归的各种标准工具中最简单而且最流行。 线性回归基于几个简单的假设： 首先，假设自变量x和因变量y之间的关系是线性的， 即y可以表示为x中元素的加权和，这里通常允许包含观测值的一些噪声； 其次，我们假设任何噪声都比较正常，如噪声遵循正态分布。

为了解释 *线性回归* ，我们举一个实际的例子： 我们希望根据房屋的面积（平方英尺）和房龄（年）来估算房屋价格（美元）。 为了开发一个能预测房价的模型，我们需要收集一个真实的数据集。 这个数据集包括了房屋的销售价格、面积和房龄。 在机器学习的术语中，该数据集称为 *训练数据集* （training data set） 或 *训练集* （training set）。 每行数据（比如一次房屋交易相对应的数据）称为 *样本* （sample）， 也可以称为 *数据点* （data point）或 *数据样本* （data instance）。 我们把试图预测的目标（比如预测房屋价格）称为 *标签* （label）或 *目标* （target）。 预测所依据的自变量（面积和房龄）称为 *特征* （feature）或 *协变量* （covariate）。

##### 3.1.1.1 线性模型

线性假设是指目标（房屋价格）可以表示为特征（面积和房龄）的加权和，如下面的式子：

<math xmlns="http://www.w3.org/1998/Math/MathML" display="block">
  <mrow data-mjx-texclass="ORD">
    <mi data-mjx-auto-op="false">price</mi>
  </mrow>
  <mo>=</mo>
  <msub>
    <mi>w</mi>
    <mrow data-mjx-texclass="ORD">
      <mrow data-mjx-texclass="ORD">
        <mi data-mjx-auto-op="false">area</mi>
      </mrow>
    </mrow>
  </msub>
  <mo>⋅</mo>
  <mrow data-mjx-texclass="ORD">
    <mi data-mjx-auto-op="false">area</mi>
  </mrow>
  <mo>+</mo>
  <msub>
    <mi>w</mi>
    <mrow data-mjx-texclass="ORD">
      <mrow data-mjx-texclass="ORD">
        <mi data-mjx-auto-op="false">age</mi>
      </mrow>
    </mrow>
  </msub>
  <mo>⋅</mo>
  <mrow data-mjx-texclass="ORD">
    <mi data-mjx-auto-op="false">age</mi>
  </mrow>
  <mo>+</mo>
  <mi>b</mi>
  <mo>.</mo>
</math>

中的W age和 W area称为 *权重* （weight），权重决定了每个特征对我们预测值的影响。 称为 *偏置* （bias）、 *偏移量* （offset）或 *截距* （intercept）。 偏置是指当所有特征都取值为0时，预测值应该为多少。 即使现实中不会有任何房子的面积是0或房龄正好是0年，我们仍然需要偏置项。 如果没有偏置项，我们模型的表达能力将受到限制。 严格来说， [(3.1.1)](https://zh.d2l.ai/chapter_linear-networks/linear-regression.html#equation-eq-price-area)是输入特征的一个  *仿射变换* （affine transformation）。 仿射变换的特点是通过加权和对特征进行 *线性变换* （linear transformation）， 并通过偏置项来进行 *平移* （translation）。

给定一个数据集，我们的目标是寻找模型的权重和偏置， 使得根据模型做出的预测大体符合数据里的真实价格。 输出的预测值由输入特征通过*线性模型*的仿射变换决定，仿射变换由所选权重和偏置确定。

而在机器学习领域，我们通常使用的是高维数据集，建模时采用线性代数表示法会比较方便。

当我们的输入包含个特征时，我们将预测结果 （通常使用“尖角”符号表示的估计值）表示为：

<math xmlns="http://www.w3.org/1998/Math/MathML" display="block">
  <mrow data-mjx-texclass="ORD">
    <mover>
      <mi>y</mi>
      <mo stretchy="false">^</mo>
    </mover>
  </mrow>
  <mo>=</mo>
  <msub>
    <mi>w</mi>
    <mn>1</mn>
  </msub>
  <msub>
    <mi>x</mi>
    <mn>1</mn>
  </msub>
  <mo>+</mo>
  <mo>.</mo>
  <mo>.</mo>
  <mo>.</mo>
  <mo>+</mo>
  <msub>
    <mi>w</mi>
    <mi>d</mi>
  </msub>
  <msub>
    <mi>x</mi>
    <mi>d</mi>
  </msub>
  <mo>+</mo>
  <mi>b</mi>
  <mo>.</mo>
</math>

将所有特征放到向量中， 并将所有权重放到向量中， 我们可以用点积形式来简洁地表达模型：

<math xmlns="http://www.w3.org/1998/Math/MathML" display="block">
  <mrow data-mjx-texclass="ORD">
    <mover>
      <mi>y</mi>
      <mo stretchy="false">^</mo>
    </mover>
  </mrow>
  <mo>=</mo>
  <msup>
    <mrow data-mjx-texclass="ORD">
      <mi mathvariant="bold">w</mi>
    </mrow>
    <mi mathvariant="normal">⊤</mi>
  </msup>
  <mrow data-mjx-texclass="ORD">
    <mi mathvariant="bold">x</mi>
  </mrow>
  <mo>+</mo>
  <mi>b</mi>
  <mo>.</mo>
</math>

在 [(3.1.3)](https://zh.d2l.ai/chapter_linear-networks/linear-regression.html#equation-eq-linreg-y)中， 向量对应于单个数据样本的特征。 用符号表示的矩阵 可以很方便地引用我们整个数据集的个样本。 其中，的每一行是一个样本，每一列是一种特征。

对于特征集合，预测值 可以通过矩阵-向量乘法表示为：

<math xmlns="http://www.w3.org/1998/Math/MathML" display="block">
  <mrow data-mjx-texclass="ORD">
    <mrow data-mjx-texclass="ORD">
      <mover>
        <mrow data-mjx-texclass="ORD">
          <mi mathvariant="bold">y</mi>
        </mrow>
        <mo stretchy="false">^</mo>
      </mover>
    </mrow>
  </mrow>
  <mo>=</mo>
  <mrow data-mjx-texclass="ORD">
    <mi mathvariant="bold">X</mi>
  </mrow>
  <mrow data-mjx-texclass="ORD">
    <mi mathvariant="bold">w</mi>
  </mrow>
  <mo>+</mo>
  <mi>b</mi>
</math>

这个过程中的求和将使用广播机制 （广播机制在 [2.1.3节](https://zh.d2l.ai/chapter_preliminaries/ndarray.html#subsec-broadcasting)中有详细介绍）。 给定训练数据特征和对应的已知标签， 线性回归的目标是找到一组权重向量和偏置： 当给定从的同分布中取样的新样本特征时， 这组权重向量和偏置能够使得新样本预测标签的误差尽可能小。

虽然我们相信给定预测的最佳模型会是线性的， 但我们很难找到一个有个样本的真实数据集，其中对于所有的，完全等于。 无论我们使用什么手段来观察特征和标签， 都可能会出现少量的观测误差。 因此，即使确信特征与标签的潜在关系是线性的， 我们也会加入一个噪声项来考虑观测误差带来的影响。

在开始寻找最好的 *模型参数* （model parameters）w和b之前， 我们还需要两个东西： （1）一种模型质量的度量方式； （2）一种能够更新模型以提高模型预测质量的方法。

##### 3.1.1.2 损失函数

在我们开始考虑如何用模型 *拟合* （fit）数据之前，我们需要确定一个拟合程度的度量。  *损失函数* （loss function）能够量化目标的*实际*值与*预测*值之间的差距。 通常我们会选择非负数作为损失，且数值越小表示损失越小，完美预测时的损失为0。 回归问题中最常用的损失函数是平方误差函数。 当样本的预测值为，其相应的真实标签为时， 平方误差可以定义为以下公式：

<math xmlns="http://www.w3.org/1998/Math/MathML" display="block">
  <msup>
    <mi>l</mi>
    <mrow data-mjx-texclass="ORD">
      <mo stretchy="false">(</mo>
      <mi>i</mi>
      <mo stretchy="false">)</mo>
    </mrow>
  </msup>
  <mo stretchy="false">(</mo>
  <mrow data-mjx-texclass="ORD">
    <mi mathvariant="bold">w</mi>
  </mrow>
  <mo>,</mo>
  <mi>b</mi>
  <mo stretchy="false">)</mo>
  <mo>=</mo>
  <mfrac>
    <mn>1</mn>
    <mn>2</mn>
  </mfrac>
  <msup>
    <mrow data-mjx-texclass="INNER">
      <mo data-mjx-texclass="OPEN">(</mo>
      <msup>
        <mrow data-mjx-texclass="ORD">
          <mover>
            <mi>y</mi>
            <mo stretchy="false">^</mo>
          </mover>
        </mrow>
        <mrow data-mjx-texclass="ORD">
          <mo stretchy="false">(</mo>
          <mi>i</mi>
          <mo stretchy="false">)</mo>
        </mrow>
      </msup>
      <mo>−</mo>
      <msup>
        <mi>y</mi>
        <mrow data-mjx-texclass="ORD">
          <mo stretchy="false">(</mo>
          <mi>i</mi>
          <mo stretchy="false">)</mo>
        </mrow>
      </msup>
      <mo data-mjx-texclass="CLOSE">)</mo>
    </mrow>
    <mn>2</mn>
  </msup>
  <mo>.</mo>
</math>

常数不会带来本质的差别，但这样在形式上稍微简单一些 （因为当我们对损失函数求导后常数系数为1）。 由于训练数据集并不受我们控制，所以经验误差只是关于模型参数的函数。

由于平方误差函数中的二次方项， 估计值和观测值之间较大的差异将导致更大的损失。 为了度量模型在整个数据集上的质量，我们需计算在训练集个样本上的损失均值（也等价于求和）。

<math xmlns="http://www.w3.org/1998/Math/MathML" display="block">
  <mi>L</mi>
  <mo stretchy="false">(</mo>
  <mrow data-mjx-texclass="ORD">
    <mi mathvariant="bold">w</mi>
  </mrow>
  <mo>,</mo>
  <mi>b</mi>
  <mo stretchy="false">)</mo>
  <mo>=</mo>
  <mfrac>
    <mn>1</mn>
    <mi>n</mi>
  </mfrac>
  <munderover>
    <mo data-mjx-texclass="OP">∑</mo>
    <mrow data-mjx-texclass="ORD">
      <mi>i</mi>
      <mo>=</mo>
      <mn>1</mn>
    </mrow>
    <mi>n</mi>
  </munderover>
  <msup>
    <mi>l</mi>
    <mrow data-mjx-texclass="ORD">
      <mo stretchy="false">(</mo>
      <mi>i</mi>
      <mo stretchy="false">)</mo>
    </mrow>
  </msup>
  <mo stretchy="false">(</mo>
  <mrow data-mjx-texclass="ORD">
    <mi mathvariant="bold">w</mi>
  </mrow>
  <mo>,</mo>
  <mi>b</mi>
  <mo stretchy="false">)</mo>
  <mo>=</mo>
  <mfrac>
    <mn>1</mn>
    <mi>n</mi>
  </mfrac>
  <munderover>
    <mo data-mjx-texclass="OP">∑</mo>
    <mrow data-mjx-texclass="ORD">
      <mi>i</mi>
      <mo>=</mo>
      <mn>1</mn>
    </mrow>
    <mi>n</mi>
  </munderover>
  <mfrac>
    <mn>1</mn>
    <mn>2</mn>
  </mfrac>
  <msup>
    <mrow data-mjx-texclass="INNER">
      <mo data-mjx-texclass="OPEN">(</mo>
      <msup>
        <mrow data-mjx-texclass="ORD">
          <mi mathvariant="bold">w</mi>
        </mrow>
        <mi mathvariant="normal">⊤</mi>
      </msup>
      <msup>
        <mrow data-mjx-texclass="ORD">
          <mi mathvariant="bold">x</mi>
        </mrow>
        <mrow data-mjx-texclass="ORD">
          <mo stretchy="false">(</mo>
          <mi>i</mi>
          <mo stretchy="false">)</mo>
        </mrow>
      </msup>
      <mo>+</mo>
      <mi>b</mi>
      <mo>−</mo>
      <msup>
        <mi>y</mi>
        <mrow data-mjx-texclass="ORD">
          <mo stretchy="false">(</mo>
          <mi>i</mi>
          <mo stretchy="false">)</mo>
        </mrow>
      </msup>
      <mo data-mjx-texclass="CLOSE">)</mo>
    </mrow>
    <mn>2</mn>
  </msup>
  <mo>.</mo>
</math>

在训练模型时，我们希望寻找一组参数（）， 这组参数能最小化在所有训练样本上的总损失。如下式：

<math xmlns="http://www.w3.org/1998/Math/MathML" display="block">
  <msup>
    <mrow data-mjx-texclass="ORD">
      <mi mathvariant="bold">w</mi>
    </mrow>
    <mo>∗</mo>
  </msup>
  <mo>,</mo>
  <msup>
    <mi>b</mi>
    <mo>∗</mo>
  </msup>
  <mo>=</mo>
  <munder>
    <mi>argmin</mi>
    <mrow data-mjx-texclass="ORD">
      <mrow data-mjx-texclass="ORD">
        <mi mathvariant="bold">w</mi>
      </mrow>
      <mo>,</mo>
      <mi>b</mi>
    </mrow>
  </munder>
  <mtext> </mtext>
  <mi>L</mi>
  <mo stretchy="false">(</mo>
  <mrow data-mjx-texclass="ORD">
    <mi mathvariant="bold">w</mi>
  </mrow>
  <mo>,</mo>
  <mi>b</mi>
  <mo stretchy="false">)</mo>
  <mo>.</mo>
</math>

##### 3.1.1.3 解析解

线性回归刚好是一个很简单的优化问题。 与我们将在本书中所讲到的其他大部分模型不同，线性回归的解可以用一个公式简单地表达出来， 这类解叫作解析解（analytical solution）。 首先，我们将偏置合并到参数中，合并方法是在包含所有参数的矩阵中附加一列。 我们的预测问题是最小化。 这在损失平面上只有一个临界点，这个临界点对应于整个区域的损失极小点。 将损失关于的导数设为0，得到解析解：

<math xmlns="http://www.w3.org/1998/Math/MathML" display="block">
  <msup>
    <mrow data-mjx-texclass="ORD">
      <mi mathvariant="bold">w</mi>
    </mrow>
    <mo>∗</mo>
  </msup>
  <mo>=</mo>
  <mo stretchy="false">(</mo>
  <msup>
    <mrow data-mjx-texclass="ORD">
      <mi mathvariant="bold">X</mi>
    </mrow>
    <mi mathvariant="normal">⊤</mi>
  </msup>
  <mrow data-mjx-texclass="ORD">
    <mi mathvariant="bold">X</mi>
  </mrow>
  <msup>
    <mo stretchy="false">)</mo>
    <mrow data-mjx-texclass="ORD">
      <mo>−</mo>
      <mn>1</mn>
    </mrow>
  </msup>
  <msup>
    <mrow data-mjx-texclass="ORD">
      <mi mathvariant="bold">X</mi>
    </mrow>
    <mi mathvariant="normal">⊤</mi>
  </msup>
  <mrow data-mjx-texclass="ORD">
    <mi mathvariant="bold">y</mi>
  </mrow>
  <mo>.</mo>
</math>

像线性回归这样的简单问题存在解析解，但并不是所有的问题都存在解析解。 解析解可以进行很好的数学分析，但解析解对问题的限制很严格，导致它无法广泛应用在深度学习里。

##### 3.1.1.4 随机梯度下降

即使在我们无法得到解析解的情况下，我们仍然可以有效地训练模型。 在许多任务上，那些难以优化的模型效果要更好。 因此，弄清楚如何训练这些难以优化的模型是非常重要的。

本书中我们用到一种名为 *梯度下降* （gradient descent）的方法， 这种方法几乎可以优化所有深度学习模型。 它通过不断地在损失函数递减的方向上更新参数来降低误差。

梯度下降最简单的用法是计算损失函数（数据集中所有样本的损失均值） 关于模型参数的导数（在这里也可以称为梯度）。 但实际中的执行可能会非常慢：因为在每一次更新参数之前，我们必须遍历整个数据集。 因此，我们通常会在每次需要计算更新的时候随机抽取一小批样本， 这种变体叫做 *小批量随机梯度下降* （minibatch stochastic gradient descent）。

在每次迭代中，我们首先随机抽样一个小批量， 它是由固定数量的训练样本组成的。 然后，我们计算小批量的平均损失关于模型参数的导数（也可以称为梯度）。 最后，我们将梯度乘以一个预先确定的正数，并从当前参数的值中减掉。

我们用下面的数学公式来表示这一更新过程（表示偏导数）：

<math xmlns="http://www.w3.org/1998/Math/MathML" display="block">
  <mo stretchy="false">(</mo>
  <mrow data-mjx-texclass="ORD">
    <mi mathvariant="bold">w</mi>
  </mrow>
  <mo>,</mo>
  <mi>b</mi>
  <mo stretchy="false">)</mo>
  <mo stretchy="false">←</mo>
  <mo stretchy="false">(</mo>
  <mrow data-mjx-texclass="ORD">
    <mi mathvariant="bold">w</mi>
  </mrow>
  <mo>,</mo>
  <mi>b</mi>
  <mo stretchy="false">)</mo>
  <mo>−</mo>
  <mfrac>
    <mi>η</mi>
    <mrow>
      <mo stretchy="false">|</mo>
      <mrow data-mjx-texclass="ORD">
        <mi data-mjx-variant="-tex-calligraphic" mathvariant="script">B</mi>
      </mrow>
      <mo stretchy="false">|</mo>
    </mrow>
  </mfrac>
  <munder>
    <mo data-mjx-texclass="OP">∑</mo>
    <mrow data-mjx-texclass="ORD">
      <mi>i</mi>
      <mo>∈</mo>
      <mrow data-mjx-texclass="ORD">
        <mi data-mjx-variant="-tex-calligraphic" mathvariant="script">B</mi>
      </mrow>
    </mrow>
  </munder>
  <msub>
    <mi>∂</mi>
    <mrow data-mjx-texclass="ORD">
      <mo stretchy="false">(</mo>
      <mrow data-mjx-texclass="ORD">
        <mi mathvariant="bold">w</mi>
      </mrow>
      <mo>,</mo>
      <mi>b</mi>
      <mo stretchy="false">)</mo>
    </mrow>
  </msub>
  <msup>
    <mi>l</mi>
    <mrow data-mjx-texclass="ORD">
      <mo stretchy="false">(</mo>
      <mi>i</mi>
      <mo stretchy="false">)</mo>
    </mrow>
  </msup>
  <mo stretchy="false">(</mo>
  <mrow data-mjx-texclass="ORD">
    <mi mathvariant="bold">w</mi>
  </mrow>
  <mo>,</mo>
  <mi>b</mi>
  <mo stretchy="false">)</mo>
  <mo>.</mo>
</math>

总结一下，算法的步骤如下： （1）初始化模型参数的值，如随机初始化； （2）从数据集中随机抽取小批量样本且在负梯度的方向上更新参数，并不断迭代这一步骤。 对于平方损失和仿射变换，我们可以明确地写成如下形式:

<math xmlns="http://www.w3.org/1998/Math/MathML" display="block">
  <mtable displaystyle="true" columnalign="right" columnspacing="0em" rowspacing="3pt">
    <mtr>
      <mtd>
        <mtable displaystyle="true" columnalign="right left" columnspacing="0em" rowspacing="3pt">
          <mtr>
            <mtd>
              <mrow data-mjx-texclass="ORD">
                <mi mathvariant="bold">w</mi>
              </mrow>
            </mtd>
            <mtd>
              <mi></mi>
              <mo stretchy="false">←</mo>
              <mrow data-mjx-texclass="ORD">
                <mi mathvariant="bold">w</mi>
              </mrow>
              <mo>−</mo>
              <mfrac>
                <mi>η</mi>
                <mrow>
                  <mo stretchy="false">|</mo>
                  <mrow data-mjx-texclass="ORD">
                    <mi data-mjx-variant="-tex-calligraphic" mathvariant="script">B</mi>
                  </mrow>
                  <mo stretchy="false">|</mo>
                </mrow>
              </mfrac>
              <munder>
                <mo data-mjx-texclass="OP">∑</mo>
                <mrow data-mjx-texclass="ORD">
                  <mi>i</mi>
                  <mo>∈</mo>
                  <mrow data-mjx-texclass="ORD">
                    <mi data-mjx-variant="-tex-calligraphic" mathvariant="script">B</mi>
                  </mrow>
                </mrow>
              </munder>
              <msub>
                <mi>∂</mi>
                <mrow data-mjx-texclass="ORD">
                  <mrow data-mjx-texclass="ORD">
                    <mi mathvariant="bold">w</mi>
                  </mrow>
                </mrow>
              </msub>
              <msup>
                <mi>l</mi>
                <mrow data-mjx-texclass="ORD">
                  <mo stretchy="false">(</mo>
                  <mi>i</mi>
                  <mo stretchy="false">)</mo>
                </mrow>
              </msup>
              <mo stretchy="false">(</mo>
              <mrow data-mjx-texclass="ORD">
                <mi mathvariant="bold">w</mi>
              </mrow>
              <mo>,</mo>
              <mi>b</mi>
              <mo stretchy="false">)</mo>
              <mo>=</mo>
              <mrow data-mjx-texclass="ORD">
                <mi mathvariant="bold">w</mi>
              </mrow>
              <mo>−</mo>
              <mfrac>
                <mi>η</mi>
                <mrow>
                  <mo stretchy="false">|</mo>
                  <mrow data-mjx-texclass="ORD">
                    <mi data-mjx-variant="-tex-calligraphic" mathvariant="script">B</mi>
                  </mrow>
                  <mo stretchy="false">|</mo>
                </mrow>
              </mfrac>
              <munder>
                <mo data-mjx-texclass="OP">∑</mo>
                <mrow data-mjx-texclass="ORD">
                  <mi>i</mi>
                  <mo>∈</mo>
                  <mrow data-mjx-texclass="ORD">
                    <mi data-mjx-variant="-tex-calligraphic" mathvariant="script">B</mi>
                  </mrow>
                </mrow>
              </munder>
              <msup>
                <mrow data-mjx-texclass="ORD">
                  <mi mathvariant="bold">x</mi>
                </mrow>
                <mrow data-mjx-texclass="ORD">
                  <mo stretchy="false">(</mo>
                  <mi>i</mi>
                  <mo stretchy="false">)</mo>
                </mrow>
              </msup>
              <mrow data-mjx-texclass="INNER">
                <mo data-mjx-texclass="OPEN">(</mo>
                <msup>
                  <mrow data-mjx-texclass="ORD">
                    <mi mathvariant="bold">w</mi>
                  </mrow>
                  <mi mathvariant="normal">⊤</mi>
                </msup>
                <msup>
                  <mrow data-mjx-texclass="ORD">
                    <mi mathvariant="bold">x</mi>
                  </mrow>
                  <mrow data-mjx-texclass="ORD">
                    <mo stretchy="false">(</mo>
                    <mi>i</mi>
                    <mo stretchy="false">)</mo>
                  </mrow>
                </msup>
                <mo>+</mo>
                <mi>b</mi>
                <mo>−</mo>
                <msup>
                  <mi>y</mi>
                  <mrow data-mjx-texclass="ORD">
                    <mo stretchy="false">(</mo>
                    <mi>i</mi>
                    <mo stretchy="false">)</mo>
                  </mrow>
                </msup>
                <mo data-mjx-texclass="CLOSE">)</mo>
              </mrow>
              <mo>,</mo>
            </mtd>
          </mtr>
          <mtr>
            <mtd>
              <mi>b</mi>
            </mtd>
            <mtd>
              <mi></mi>
              <mo stretchy="false">←</mo>
              <mi>b</mi>
              <mo>−</mo>
              <mfrac>
                <mi>η</mi>
                <mrow>
                  <mo stretchy="false">|</mo>
                  <mrow data-mjx-texclass="ORD">
                    <mi data-mjx-variant="-tex-calligraphic" mathvariant="script">B</mi>
                  </mrow>
                  <mo stretchy="false">|</mo>
                </mrow>
              </mfrac>
              <munder>
                <mo data-mjx-texclass="OP">∑</mo>
                <mrow data-mjx-texclass="ORD">
                  <mi>i</mi>
                  <mo>∈</mo>
                  <mrow data-mjx-texclass="ORD">
                    <mi data-mjx-variant="-tex-calligraphic" mathvariant="script">B</mi>
                  </mrow>
                </mrow>
              </munder>
              <msub>
                <mi>∂</mi>
                <mi>b</mi>
              </msub>
              <msup>
                <mi>l</mi>
                <mrow data-mjx-texclass="ORD">
                  <mo stretchy="false">(</mo>
                  <mi>i</mi>
                  <mo stretchy="false">)</mo>
                </mrow>
              </msup>
              <mo stretchy="false">(</mo>
              <mrow data-mjx-texclass="ORD">
                <mi mathvariant="bold">w</mi>
              </mrow>
              <mo>,</mo>
              <mi>b</mi>
              <mo stretchy="false">)</mo>
              <mo>=</mo>
              <mi>b</mi>
              <mo>−</mo>
              <mfrac>
                <mi>η</mi>
                <mrow>
                  <mo stretchy="false">|</mo>
                  <mrow data-mjx-texclass="ORD">
                    <mi data-mjx-variant="-tex-calligraphic" mathvariant="script">B</mi>
                  </mrow>
                  <mo stretchy="false">|</mo>
                </mrow>
              </mfrac>
              <munder>
                <mo data-mjx-texclass="OP">∑</mo>
                <mrow data-mjx-texclass="ORD">
                  <mi>i</mi>
                  <mo>∈</mo>
                  <mrow data-mjx-texclass="ORD">
                    <mi data-mjx-variant="-tex-calligraphic" mathvariant="script">B</mi>
                  </mrow>
                </mrow>
              </munder>
              <mrow data-mjx-texclass="INNER">
                <mo data-mjx-texclass="OPEN">(</mo>
                <msup>
                  <mrow data-mjx-texclass="ORD">
                    <mi mathvariant="bold">w</mi>
                  </mrow>
                  <mi mathvariant="normal">⊤</mi>
                </msup>
                <msup>
                  <mrow data-mjx-texclass="ORD">
                    <mi mathvariant="bold">x</mi>
                  </mrow>
                  <mrow data-mjx-texclass="ORD">
                    <mo stretchy="false">(</mo>
                    <mi>i</mi>
                    <mo stretchy="false">)</mo>
                  </mrow>
                </msup>
                <mo>+</mo>
                <mi>b</mi>
                <mo>−</mo>
                <msup>
                  <mi>y</mi>
                  <mrow data-mjx-texclass="ORD">
                    <mo stretchy="false">(</mo>
                    <mi>i</mi>
                    <mo stretchy="false">)</mo>
                  </mrow>
                </msup>
                <mo data-mjx-texclass="CLOSE">)</mo>
              </mrow>
              <mo>.</mo>
            </mtd>
          </mtr>
        </mtable>
      </mtd>
    </mtr>
  </mtable>
</math>

公式 [(3.1.10)](https://zh.d2l.ai/chapter_linear-networks/linear-regression.html#equation-eq-linreg-batch-update)中的和都是向量。 在这里，更优雅的向量表示法比系数表示法（如）更具可读性。 表示每个小批量中的样本数，这也称为 *批量大小* （batch size）。 表示 *学习率* （learning rate）。 批量大小和学习率的值通常是手动预先指定，而不是通过模型训练得到的。 这些可以调整但不在训练过程中更新的参数称为 *超参数* （hyperparameter）。  *调参* （hyperparameter tuning）是选择超参数的过程。 超参数通常是我们根据训练迭代结果来调整的， 而训练迭代结果是在独立的 *验证数据集* （validation dataset）上评估得到的。

在训练了预先确定的若干迭代次数后（或者直到满足某些其他停止条件后）， 我们记录下模型参数的估计值，表示为。 但是，即使我们的函数确实是线性的且无噪声，这些估计值也不会使损失函数真正地达到最小值。 因为算法会使得损失向最小值缓慢收敛，但却不能在有限的步数内非常精确地达到最小值。

线性回归恰好是一个在整个域中只有一个最小值的学习问题。 但是对像深度神经网络这样复杂的模型来说，损失平面上通常包含多个最小值。 深度学习实践者很少会去花费大力气寻找这样一组参数，使得在*训练集*上的损失达到最小。 事实上，更难做到的是找到一组参数，这组参数能够在我们从未见过的数据上实现较低的损失， 这一挑战被称为 *泛化* （generalization）。

##### 3.1.1.5 用模型进行预测

给定“已学习”的线性回归模型， 现在我们可以通过房屋面积和房龄来估计一个（未包含在训练数据中的）新房屋价格。 给定特征估计目标的过程通常称为 *预测* （prediction）或 *推断* （inference）。

本书将尝试坚持使用*预测*这个词。 虽然*推断*这个词已经成为深度学习的标准术语，但其实*推断*这个词有些用词不当。 在统计学中，*推断*更多地表示基于数据集估计参数。 当深度学习从业者与统计学家交谈时，术语的误用经常导致一些误解。

#### 3.1.2 矢量化加速

在训练我们的模型时，我们经常希望能够同时处理整个小批量的样本。 为了实现这一点，需要我们对计算进行矢量化， 从而利用线性代数库，而不是在Python中编写开销高昂的for循环。为了说明矢量化为什么如此重要，我们考虑对向量相加的两种方法。 我们实例化两个全为1的10000维向量。 在一种方法中，我们将使用Python的for循环遍历向量； 在另一种方法中，我们将依赖对 `+`的调用。

由于在本书中我们将频繁地进行运行时间的基准测试，所以我们定义一个计时器，现在我们可以对工作负载进行基准测试。首先，我们使用for循环，每次执行一位的加法。或者，我们使用重载的 `+`运算符来计算按元素的和。结果很明显，第二种方法比第一种方法快得多。 矢量化代码通常会带来数量级的加速。 另外，我们将更多的数学运算放到库中，而无须自己编写那么多的计算，从而减少了出错的可能性。

#### 3.1.3 正态分布与平方损失

接下来，我们通过对噪声分布的假设来解读平方损失目标函数。

正态分布和线性回归之间的关系很密切。 正态分布（normal distribution），也称为 *高斯分布* （Gaussian distribution）， 最早由德国数学家高斯（Gauss）应用于天文学研究。 简单的说，若随机变量具有均值和方差（标准差），其正态分布概率密度函数如下：

<math xmlns="http://www.w3.org/1998/Math/MathML" display="block">
  <mi>p</mi>
  <mo stretchy="false">(</mo>
  <mi>x</mi>
  <mo stretchy="false">)</mo>
  <mo>=</mo>
  <mfrac>
    <mn>1</mn>
    <msqrt>
      <mn>2</mn>
      <mi>π</mi>
      <msup>
        <mi>σ</mi>
        <mn>2</mn>
      </msup>
    </msqrt>
  </mfrac>
  <mi>exp</mi>
  <mo data-mjx-texclass="NONE">⁡</mo>
  <mrow data-mjx-texclass="INNER">
    <mo data-mjx-texclass="OPEN">(</mo>
    <mo>−</mo>
    <mfrac>
      <mn>1</mn>
      <mrow>
        <mn>2</mn>
        <msup>
          <mi>σ</mi>
          <mn>2</mn>
        </msup>
      </mrow>
    </mfrac>
    <mo stretchy="false">(</mo>
    <mi>x</mi>
    <mo>−</mo>
    <mi>μ</mi>
    <msup>
      <mo stretchy="false">)</mo>
      <mn>2</mn>
    </msup>
    <mo data-mjx-texclass="CLOSE">)</mo>
  </mrow>
  <mo>.</mo>
</math>

均方误差损失函数（简称均方损失）可以用于线性回归的一个原因是： 我们假设了观测中包含噪声，其中噪声服从正态分布。 噪声正态分布如下式:

<math xmlns="http://www.w3.org/1998/Math/MathML" display="block">
  <mi>y</mi>
  <mo>=</mo>
  <msup>
    <mrow data-mjx-texclass="ORD">
      <mi mathvariant="bold">w</mi>
    </mrow>
    <mi mathvariant="normal">⊤</mi>
  </msup>
  <mrow data-mjx-texclass="ORD">
    <mi mathvariant="bold">x</mi>
  </mrow>
  <mo>+</mo>
  <mi>b</mi>
  <mo>+</mo>
  <mi>ϵ</mi>
  <mo>,</mo>
</math>

因此，我们现在可以写出通过给定x的观测到特定y的 *似然* （likelihood）：

<math xmlns="http://www.w3.org/1998/Math/MathML" display="block">
  <mi>P</mi>
  <mo stretchy="false">(</mo>
  <mi>y</mi>
  <mo>∣</mo>
  <mrow data-mjx-texclass="ORD">
    <mi mathvariant="bold">x</mi>
  </mrow>
  <mo stretchy="false">)</mo>
  <mo>=</mo>
  <mfrac>
    <mn>1</mn>
    <msqrt>
      <mn>2</mn>
      <mi>π</mi>
      <msup>
        <mi>σ</mi>
        <mn>2</mn>
      </msup>
    </msqrt>
  </mfrac>
  <mi>exp</mi>
  <mo data-mjx-texclass="NONE">⁡</mo>
  <mrow data-mjx-texclass="INNER">
    <mo data-mjx-texclass="OPEN">(</mo>
    <mo>−</mo>
    <mfrac>
      <mn>1</mn>
      <mrow>
        <mn>2</mn>
        <msup>
          <mi>σ</mi>
          <mn>2</mn>
        </msup>
      </mrow>
    </mfrac>
    <mo stretchy="false">(</mo>
    <mi>y</mi>
    <mo>−</mo>
    <msup>
      <mrow data-mjx-texclass="ORD">
        <mi mathvariant="bold">w</mi>
      </mrow>
      <mi mathvariant="normal">⊤</mi>
    </msup>
    <mrow data-mjx-texclass="ORD">
      <mi mathvariant="bold">x</mi>
    </mrow>
    <mo>−</mo>
    <mi>b</mi>
    <msup>
      <mo stretchy="false">)</mo>
      <mn>2</mn>
    </msup>
    <mo data-mjx-texclass="CLOSE">)</mo>
  </mrow>
  <mo>.</mo>
</math>

现在，根据极大似然估计法，参数和的最优值是使整个数据集的*似然*最大的值：

<math xmlns="http://www.w3.org/1998/Math/MathML" display="block">
  <mi>P</mi>
  <mo stretchy="false">(</mo>
  <mrow data-mjx-texclass="ORD">
    <mi mathvariant="bold">y</mi>
  </mrow>
  <mo>∣</mo>
  <mrow data-mjx-texclass="ORD">
    <mi mathvariant="bold">X</mi>
  </mrow>
  <mo stretchy="false">)</mo>
  <mo>=</mo>
  <munderover>
    <mo data-mjx-texclass="OP">∏</mo>
    <mrow data-mjx-texclass="ORD">
      <mi>i</mi>
      <mo>=</mo>
      <mn>1</mn>
    </mrow>
    <mrow data-mjx-texclass="ORD">
      <mi>n</mi>
    </mrow>
  </munderover>
  <mi>p</mi>
  <mo stretchy="false">(</mo>
  <msup>
    <mi>y</mi>
    <mrow data-mjx-texclass="ORD">
      <mo stretchy="false">(</mo>
      <mi>i</mi>
      <mo stretchy="false">)</mo>
    </mrow>
  </msup>
  <mrow data-mjx-texclass="ORD">
    <mo stretchy="false">|</mo>
  </mrow>
  <msup>
    <mrow data-mjx-texclass="ORD">
      <mi mathvariant="bold">x</mi>
    </mrow>
    <mrow data-mjx-texclass="ORD">
      <mo stretchy="false">(</mo>
      <mi>i</mi>
      <mo stretchy="false">)</mo>
    </mrow>
  </msup>
  <mo stretchy="false">)</mo>
  <mo>.</mo>
</math>

根据极大似然估计法选择的估计量称为 *极大似然估计量* 。 虽然使许多指数函数的乘积最大化看起来很困难， 但是我们可以在不改变目标的前提下，通过最大化似然对数来简化。 由于历史原因，优化通常是说最小化而不是最大化。 我们可以改为*最小化负对数似然-logP(y|X)*。 由此可以得到的数学公式是：

<math xmlns="http://www.w3.org/1998/Math/MathML" display="block">
  <mo>−</mo>
  <mi>log</mi>
  <mo data-mjx-texclass="NONE">⁡</mo>
  <mi>P</mi>
  <mo stretchy="false">(</mo>
  <mrow data-mjx-texclass="ORD">
    <mi mathvariant="bold">y</mi>
  </mrow>
  <mo>∣</mo>
  <mrow data-mjx-texclass="ORD">
    <mi mathvariant="bold">X</mi>
  </mrow>
  <mo stretchy="false">)</mo>
  <mo>=</mo>
  <munderover>
    <mo data-mjx-texclass="OP">∑</mo>
    <mrow data-mjx-texclass="ORD">
      <mi>i</mi>
      <mo>=</mo>
      <mn>1</mn>
    </mrow>
    <mi>n</mi>
  </munderover>
  <mfrac>
    <mn>1</mn>
    <mn>2</mn>
  </mfrac>
  <mi>log</mi>
  <mo data-mjx-texclass="NONE">⁡</mo>
  <mo stretchy="false">(</mo>
  <mn>2</mn>
  <mi>π</mi>
  <msup>
    <mi>σ</mi>
    <mn>2</mn>
  </msup>
  <mo stretchy="false">)</mo>
  <mo>+</mo>
  <mfrac>
    <mn>1</mn>
    <mrow>
      <mn>2</mn>
      <msup>
        <mi>σ</mi>
        <mn>2</mn>
      </msup>
    </mrow>
  </mfrac>
  <msup>
    <mrow data-mjx-texclass="INNER">
      <mo data-mjx-texclass="OPEN">(</mo>
      <msup>
        <mi>y</mi>
        <mrow data-mjx-texclass="ORD">
          <mo stretchy="false">(</mo>
          <mi>i</mi>
          <mo stretchy="false">)</mo>
        </mrow>
      </msup>
      <mo>−</mo>
      <msup>
        <mrow data-mjx-texclass="ORD">
          <mi mathvariant="bold">w</mi>
        </mrow>
        <mi mathvariant="normal">⊤</mi>
      </msup>
      <msup>
        <mrow data-mjx-texclass="ORD">
          <mi mathvariant="bold">x</mi>
        </mrow>
        <mrow data-mjx-texclass="ORD">
          <mo stretchy="false">(</mo>
          <mi>i</mi>
          <mo stretchy="false">)</mo>
        </mrow>
      </msup>
      <mo>−</mo>
      <mi>b</mi>
      <mo data-mjx-texclass="CLOSE">)</mo>
    </mrow>
    <mn>2</mn>
  </msup>
  <mo>.</mo>
</math>

现在我们只需要假设是某个固定常数就可以忽略第一项， 因为第一项不依赖于w和b。 现在第二项除了常数外，其余部分和前面介绍的均方误差是一样的。 幸运的是，上面式子的解并不依赖于。 因此，在高斯噪声的假设下，最小化均方误差等价于对线性模型的极大似然估计。

#### 3.1.4 从线性回归到深度网络

到目前为止，我们只谈论了线性模型。 尽管神经网络涵盖了更多更为丰富的模型，我们依然可以用描述神经网络的方式来描述线性模型， 从而把线性模型看作一个神经网络。 首先，我们用“层”符号来重写这个模型。

##### 3.1.4.1 神经网络图

深度学习从业者喜欢绘制图表来可视化模型中正在发生的事情。 在 [图3.1.2](https://zh.d2l.ai/chapter_linear-networks/linear-regression.html#fig-single-neuron)中，我们将线性回归模型描述为一个神经网络。 需要注意的是，该图只显示连接模式，即只显示每个输入如何连接到输出，隐去了权重和偏置的值。

在 [图3.1.2](https://zh.d2l.ai/chapter_linear-networks/linear-regression.html#fig-single-neuron)所示的神经网络中，输入为， 因此输入层中的 *输入数* （或称为 *特征维度* ，feature dimensionality）为。 网络的输出为，因此输出层中的*输出数*是1。 需要注意的是，输入值都是已经给定的，并且只有一个*计算*神经元。 由于模型重点在发生计算的地方，所以通常我们在计算层数时不考虑输入层。 也就是说， [图3.1.2](https://zh.d2l.ai/chapter_linear-networks/linear-regression.html#fig-single-neuron)中神经网络的*层数*为1。 我们可以将线性回归模型视为仅由单个人工神经元组成的神经网络，或称为单层神经网络。

对于线性回归，每个输入都与每个输出（在本例中只有一个输出）相连， 我们将这种变换（ [图3.1.2](https://zh.d2l.ai/chapter_linear-networks/linear-regression.html#fig-single-neuron)中的输出层） 称为 *全连接层* （fully-connected layer）或称为 *稠密层* （dense layer）。 下一章将详细讨论由这些层组成的网络。

##### 3.1.4.2 生物学

线性回归发明的时间（1795年）早于计算神经科学，所以将线性回归描述为神经网络似乎不合适。 当控制学家、神经生物学家沃伦·麦库洛奇和沃尔特·皮茨开始开发人工神经元模型时， 他们为什么将线性模型作为一个起点呢？ 我们来看一张图片 [图3.1.3](https://zh.d2l.ai/chapter_linear-networks/linear-regression.html#fig-neuron)： 这是一张由 *树突* （dendrites，输入终端）、  *细胞核* （nucleus，CPU）组成的生物神经元图片。  *轴突* （axon，输出线）和 *轴突端子* （axon terminal，输出端子） 通过 *突触* （synapse）与其他神经元连接。

树突中接收到来自其他神经元（或视网膜等环境传感器）的信息。 该信息通过*突触权重*来加权，以确定输入的影响（即，通过相乘来激活或抑制）。 来自多个源的加权输入以加权和的形式汇聚在细胞核中， 然后将这些信息发送到轴突中进一步处理，通常会通过进行一些非线性处理。 之后，它要么到达目的地（例如肌肉），要么通过树突进入另一个神经元。

当然，许多这样的单元可以通过正确连接和正确的学习算法拼凑在一起， 从而产生的行为会比单独一个神经元所产生的行为更有趣、更复杂， 这种想法归功于我们对真实生物神经系统的研究。

当今大多数深度学习的研究几乎没有直接从神经科学中获得灵感。 我们援引斯图尔特·罗素和彼得·诺维格在他们的经典人工智能教科书 *Artificial Intelligence:A Modern Approach* ([Russell and Norvig, 2016](https://zh.d2l.ai/chapter_references/zreferences.html#id141 "Russell, S. J., &amp; Norvig, P. (2016). Artificial intelligence: a modern approach. Malaysia; Pearson Education Limited,.")) 中所说的：虽然飞机可能受到鸟类的启发，但几个世纪以来，鸟类学并不是航空创新的主要驱动力。 同样地，如今在深度学习中的灵感同样或更多地来自数学、统计学和计算机科学。

### 3.2 线性回归的从零开始实现

在了解线性回归的关键思想之后，我们可以开始通过代码来动手实现线性回归了。 在这一节中，我们将从零开始实现整个方法， 包括数据流水线、模型、损失函数和小批量随机梯度下降优化器。 虽然现代的深度学习框架几乎可以自动化地进行所有这些工作，但从零开始实现可以确保我们真正知道自己在做什么。 同时，了解更细致的工作原理将方便我们自定义模型、自定义层或自定义损失函数。 在这一节中，我们将只使用张量和自动求导。 在之后的章节中，我们会充分利用深度学习框架的优势，介绍更简洁的实现方式。

#### 3.2.1 生成数据集

为了简单起见，我们将根据带有噪声的线性模型构造一个人造数据集。 我们的任务是使用这个有限样本的数据集来恢复这个模型的参数。 我们将使用低维数据，这样可以很容易地将其可视化。 在下面的代码中，我们生成一个包含1000个样本的数据集， 每个样本包含从标准正态分布中采样的2个特征。 我们的合成数据集是一个矩阵X∈R 1000×2。

我们使用线性模型参数w=[2,-3,4]T、b=4.2 和噪声项μ生成数据集及其标签：

<math xmlns="http://www.w3.org/1998/Math/MathML" display="block">
  <mrow data-mjx-texclass="ORD">
    <mi mathvariant="bold">y</mi>
  </mrow>
  <mo>=</mo>
  <mrow data-mjx-texclass="ORD">
    <mi mathvariant="bold">X</mi>
  </mrow>
  <mrow data-mjx-texclass="ORD">
    <mi mathvariant="bold">w</mi>
  </mrow>
  <mo>+</mo>
  <mi>b</mi>
  <mo>+</mo>
  <mrow data-mjx-texclass="ORD">
    <mi>ϵ</mi>
  </mrow>
  <mo>.</mo>
</math>

μ可以视为模型预测和标签时的潜在观测误差。 在这里我们认为标准假设成立，即μ服从均值为0的正态分布。 为了简化问题，我们将标准差设为0.01。 下面的代码生成合成数据集。

通过生成第二个特征 `features[:,1]`和 `labels`的散点图， 可以直观观察到两者之间的线性关系。

#### 3.2.2 读取数据集

回想一下，训练模型时要对数据集进行遍历，每次抽取一小批量样本，并使用它们来更新我们的模型。 由于这个过程是训练机器学习算法的基础，所以有必要定义一个函数， 该函数能打乱数据集中的样本并以小批量方式获取数据。

在下面的代码中，我们定义一个 `data_iter`函数， 该函数接收批量大小、特征矩阵和标签向量作为输入，生成大小为 `batch_size`的小批量。 每个小批量包含一组特征和标签。

通常，我们利用GPU并行运算的优势，处理合理大小的“小批量”。 每个样本都可以并行地进行模型计算，且每个样本损失函数的梯度也可以被并行计算。 GPU可以在处理几百个样本时，所花费的时间不比处理一个样本时多太多。

我们直观感受一下小批量运算：读取第一个小批量数据样本并打印。 每个批量的特征维度显示批量大小和输入特征数。 同样的，批量的标签形状与 `batch_size`相等。

当我们运行迭代时，我们会连续地获得不同的小批量，直至遍历完整个数据集。 上面实现的迭代对教学来说很好，但它的执行效率很低，可能会在实际问题上陷入麻烦。 例如，它要求我们将所有数据加载到内存中，并执行大量的随机内存访问。 在深度学习框架中实现的内置迭代器效率要高得多， 它可以处理存储在文件中的数据和数据流提供的数据。

#### 3.2.3 初始化模型参数

在我们开始用小批量随机梯度下降优化我们的模型参数之前， 我们需要先有一些参数。 在下面的代码中，我们通过从均值为0、标准差为0.01的正态分布中采样随机数来初始化权重， 并将偏置初始化为0。

在初始化参数之后，我们的任务是更新这些参数，直到这些参数足够拟合我们的数据。 每次更新都需要计算损失函数关于模型参数的梯度。 有了这个梯度，我们就可以向减小损失的方向更新每个参数。 因为手动计算梯度很枯燥而且容易出错，所以没有人会手动计算梯度。 我们使用 [2.5节](https://zh.d2l.ai/chapter_preliminaries/autograd.html#sec-autograd)中引入的自动微分来计算梯度。

#### 3.2.4 定义模型

接下来，我们必须定义模型，将模型的输入和参数同模型的输出关联起来。 回想一下，要计算线性模型的输出， 我们只需计算输入特征X和模型权重w的矩阵-向量乘法后加上偏置b。 注意，上面的Xw是一个向量，而是b一个标量。 回想一下 [2.1.3节](https://zh.d2l.ai/chapter_preliminaries/ndarray.html#subsec-broadcasting)中描述的广播机制： 当我们用一个向量加一个标量时，标量会被加到向量的每个分量上。

#### 3.2.5 定义损失函数

因为需要计算损失函数的梯度，所以我们应该先定义损失函数。 这里我们使用 [3.1节](https://zh.d2l.ai/chapter_linear-networks/linear-regression.html#sec-linear-regression)中描述的平方损失函数。 在实现中，我们需要将真实值 `y`的形状转换为和预测值 `y_hat`的形状相同。

#### 3.2.6 定义优化算法

正如我们在 [3.1节](https://zh.d2l.ai/chapter_linear-networks/linear-regression.html#sec-linear-regression)中讨论的，线性回归有解析解。 尽管线性回归有解析解，但本书中的其他模型却没有。 这里我们介绍小批量随机梯度下降。

在每一步中，使用从数据集中随机抽取的一个小批量，然后根据参数计算损失的梯度。 接下来，朝着减少损失的方向更新我们的参数。 下面的函数实现小批量随机梯度下降更新。 该函数接受模型参数集合、学习速率和批量大小作为输入。每 一步更新的大小由学习速率 `lr`决定。 因为我们计算的损失是一个批量样本的总和，所以我们用批量大小（`batch_size`） 来规范化步长，这样步长大小就不会取决于我们对批量大小的选择。

#### 3.2.7 训练

现在我们已经准备好了模型训练所有需要的要素，可以实现主要的训练过程部分了。 理解这段代码至关重要，因为从事深度学习后， 相同的训练过程几乎一遍又一遍地出现。 在每次迭代中，我们读取一小批量训练样本，并通过我们的模型来获得一组预测。 计算完损失后，我们开始反向传播，存储每个参数的梯度。 最后，我们调用优化算法 `sgd`来更新模型参数。

概括一下，我们将执行以下循环：

* 初始化参数
* 重复以下训练，直到完成
  * 计算梯度
  * 更新参数

在每个 *迭代周期* （epoch）中，我们使用 `data_iter`函数遍历整个数据集， 并将训练数据集中所有样本都使用一次（假设样本数能够被批量大小整除）。 这里的迭代周期个数 `num_epochs`和学习率 `lr`都是超参数，分别设为3和0.03。 设置超参数很棘手，需要通过反复试验进行调整。 我们现在忽略这些细节，以后会在 [11节](https://zh.d2l.ai/chapter_optimization/index.html#chap-optimization)中详细介绍。

### 3.3 线性回归的简洁实现

在过去的几年里，出于对深度学习强烈的兴趣， 许多公司、学者和业余爱好者开发了各种成熟的开源框架。 这些框架可以自动化基于梯度的学习算法中重复性的工作。 在 [3.2节](https://zh.d2l.ai/chapter_linear-networks/linear-regression-scratch.html#sec-linear-scratch)中，我们只运用了： （1）通过张量来进行数据存储和线性代数； （2）通过自动微分来计算梯度。 实际上，由于数据迭代器、损失函数、优化器和神经网络层很常用， 现代深度学习库也为我们实现了这些组件。

本节将介绍如何通过使用深度学习框架来简洁地实现 [3.2节](https://zh.d2l.ai/chapter_linear-networks/linear-regression-scratch.html#sec-linear-scratch)中的线性回归模型。

#### 3.3.1 生成数据集

#### 3.3.2 读取数据集

我们可以调用框架中现有的API来读取数据。 我们将 `features`和 `labels`作为API的参数传递，并通过数据迭代器指定 `batch_size`。 此外，布尔值 `is_train`表示是否希望数据迭代器对象在每个迭代周期内打乱数据。
