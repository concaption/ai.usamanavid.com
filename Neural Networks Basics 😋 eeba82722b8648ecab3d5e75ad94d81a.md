# Neural Networks Basics 😋

Course: Neural Networks and Deep Learning
Course Number: 1
Course Week: 2
Cover Picture Credits: https://www.vecteezy.com/vector-art/192250-child-lover-ai-robot-vector
Date: Apr 26, 2020
Day: 2
Topics: Logistic Regression as a Neural Network, Python and Vectorization
Week URL: https://www.coursera.org/learn/neural-networks-deep-learning/home/week/2

---

[Coursera | Online Courses & Credentials From Top Educators. Join for Free | Coursera](https://www.coursera.org/learn/neural-networks-deep-learning/home/week/2)

## **Key Concepts**

---

- Build a logistic regression model, structured as a shallow neural network
- Implement the main steps of an ML algorithm, including making predictions, derivative computation, and gradient descent.
- Implement computationally efficient, highly vectorized, versions of models.
- Understand how to compute derivatives for logistic regression, using a backpropagation mindset.
- Become familiar with Python and Numpy
- Work with iPython Notebooks
- Be able to implement vectorization across multiple training examples

# Logistic Regression as a Neural Network 🤓

---

## Binary Classification

⇒ Process entire training set without using an explicit loop to loop over the entire training set

Organizing a computation of a neural network:

- Forward Propagation (followed by)
- Backward Propagation

Logistic Regression ⇒ an algorithm for binary classifications e.g

1 ⇒ Cat

0 ⇒ Non-Cat

Images are three separate matrices corresponding to R G and B color channels

$$(64 \times 64) \times 3\ \ for\ \ RGB$$

To turn the pixel intensity values into a feature vector, we unroll all the pixel values into an input feature vector

⇒⇒⇒ To unroll it define a feature vector x corresponding to this image and list all the red pixels one by one, then green and then blue in a single row.

$$n\ or\ n_{x} = (64 \times 64) \times 3 = 12288$$

where "n" is dimension of input feature vector x 

INPUT ⇒ vector of dimension n_x

OUTPUT ⇒ y

### Notation

[Neural%20Networks%20Basics%20%F0%9F%98%8B%20eeba82722b8648ecab3d5e75ad94d81a/Standard_notations_for_Deep_Learning.pdf](Neural%20Networks%20Basics%20%F0%9F%98%8B%20eeba82722b8648ecab3d5e75ad94d81a/Standard_notations_for_Deep_Learning.pdf)

Standard notations for Deep Learning

(x, y) ————-→ Single training example

where x= n_x dimensional feature vector

$$x\ \epsilon\ \mathbb{R}^{n_{x}}\quad\quad and \quad \quad y\ \epsilon\ \{0,1\} $$

m = no. of training examples

$$Training\ set=\{(x^{(1)},y^{(1)}), (x^{(2)},y^{(2)}), (x^{(3)},y^{(3)}), ... ,(x^{(m)},y^{(m)}),\}$$

To output all of training examples into a more compact notation, we are going to define a matrix capital X as defined by tacking your training set inputs x1, x2, and so on and stacking them in different column for each example.

$$X=\left[\begin{array}{ccc}  | & | & &| \\  x^{(1)} & x^{(2)} & \cdots & x^{(m)} \\  | & | & &|  \end{array}\right]$$

Which has m # of columns and n_x # of rows.

How about output y?

Let us define Y;

$$Y=\left[\begin{array}{ccc} y^{(1)} & y^{(2)}& y^{(3)} & \cdots & y^{m} \end{array}\right]$$

$$Y\ \epsilon\ \mathbb{R}^{(1, m)}$$

```python
Y.shape()
```

⇒ (1,m)

It is a useful convention to take the data of different training examples and stack them in different columns like we done here in X and Y.

## Logistic Regression

An algorithm in supervised learning problems used for binary classification problems.

**GIVEN:** The feature vector X.

**TO FIND:**

$$\hat{y}=P(y=1 | x)$$

$$x\ \epsilon\ \mathbb{R}^{n_{x}}$$

So, parameters of logistic regression algorithm will be n_x dimensional vector

$$\omega\ \epsilon\ \mathbb{R}^{n_{x}} \quad,\quad b\ \epsilon\ \mathbb{R}$$

OUTPUT⇒

Linear function → Used for linear regression but not a good algorithm for binary classification.

$$\hat{y}=\omega^{T} X+b$$

So, 

$$\hat{y}=\sigma(\omega^{T} X+b)$$

![Neural%20Networks%20Basics%20%F0%9F%98%8B%20eeba82722b8648ecab3d5e75ad94d81a/Untitled.png](Neural%20Networks%20Basics%20%F0%9F%98%8B%20eeba82722b8648ecab3d5e75ad94d81a/Untitled.png)

$$\sigma(z)=\frac{1}{1+e^{-z}}$$

$$If\ z\ is\ large\ \sigma(z) \approx \frac{1}{1+0}=1\\ If\ z\ large\ negative\ number\ \sigma(z)=\frac{1}{1+e^{-z}} \approx \frac{1}{1+Big\ number} \approx 0$$

In this course parameter b will be a separate real number. But in some other courses we may see a different approach where a separate feature

$$x_{0}=1\quad,\quad x\ \epsilon\ \mathbb{{R}^{n_x+1}}$$

$$\hat{y}=\sigma (\theta^TX)$$

$$\left.\theta=\left[\begin{array}{c} \theta_{0} \\ \theta_{1} \\ \theta_{2} \\ \vdots \\ \theta_{n_{x}} \end{array}\right]\right\} $$

## Logistic Regression Cost Function

$$\begin{aligned} &\hat{y}=\sigma\left(w^{T} x+b\right), \text { where } \sigma(z)=\frac{1}{1+e^{-z}}\\ &\text { Given }\left\{\left(x^{(1)}, y^{(1)}\right), \ldots,\left(x^{(m)}, y^{(m)}\right)\right\}, \text { want } \hat{y}^{(i)} \approx y^{(i)} \end{aligned}$$

Supper script "(i)" means "associated with ith training example"

One could do this

$$\mathcal{L}(\hat{y},y)=one\ half\ of\ square\ error=\frac{1}{2}(\hat{y}-y)^2$$

but in logistic regression people don't usually do this because optimization problem becomes little bit complex. Because for multiple training sets the optimization problem becomes non convex. Gradient decent may not find a global optimum  

![Neural%20Networks%20Basics%20%F0%9F%98%8B%20eeba82722b8648ecab3d5e75ad94d81a/Untitled%201.png](Neural%20Networks%20Basics%20%F0%9F%98%8B%20eeba82722b8648ecab3d5e75ad94d81a/Untitled%201.png)

![Neural%20Networks%20Basics%20%F0%9F%98%8B%20eeba82722b8648ecab3d5e75ad94d81a/Untitled%202.png](Neural%20Networks%20Basics%20%F0%9F%98%8B%20eeba82722b8648ecab3d5e75ad94d81a/Untitled%202.png)

So, in logistic regression we actually define an different loss fuction that plays a similar role as squared error but will give us an optimization problem that is convex.  

So, in logistic regression we use the following loss function;

$$\mathcal{L}(\hat{y}, y)=-(y\log\hat{y}+(1-y) \log (1-\hat{y}))$$

For intuition let us look at two cases:

**CASE # 1 If y=1**

$$\mathcal{L}(\hat{y}, y)=-\log\hat{y}$$

As we want to get small loss function we want this term to be as small as possible and to get it log y-hat should be as bigger possible but we certainly cannot get y-hat bigger t hen 1. 

**CASE # 2 If y=0**

$$\mathcal{L}(\hat{y}, y)=-\log(1-\hat{y})$$

As we want to get small loss function, we want this term to be as small as possible and to get it log 1 minus y-hat should be as bigger possible but we certainly cannot get 1 minus y-hat equal to 1 except when y-hat is zero. 

Loss function was defined wrt to a single training example while the cost function defines how well we are doing on all training example so cost function is average of the loss function.

$$\text { Cost function: } J(\omega, b)=\frac{1}{m} \sum_{i=1}^{m}\mathcal{L}\left(\hat{y}^{(i)}, y^{(i)}\right)=-\frac{1}{m} \sum_{i=1}^{m}\left[y^{(i)} \log {\hat{y}}^{(i)}+\left(1-y^{(i)}\right) \log \left(1- \hat{y}^{(i)}\right)\right]$$

Cost function is the cost of parameters so, in training logistic regression model we are going to find the parameters w and b which minimize the overall cost function.

The loss function computes the error for a single training example; the cost function is the average of the loss functions of the entire training set.

## Gradient Descent

Gradient descent algorithm to train to or to learn the parameter w and b on your training set.

$$\begin{array}{c} \text { Recap: } \hat{y}=\sigma\left(w^{T} x+b\right), \sigma(z)=\frac{1}{1+e^{-z}} \\ \\ J(w, b)=\frac{1}{m} \sum_{i=1}^{m} \mathcal{L}\left(\hat{y}^{(i)}, y^{(i)}\right)=-\frac{1}{m} \sum_{i=1}^{m} \left( y^{(i)} \log \hat{y}^{(i)}+(1-y^{(i)}) \log(1-\hat{y}^{(i)})\right) \end{array}$$

want to find w and b that minimize the cost function J(w, b).

In practice w can be much higher dimensional but for the purpose plotting, lets illustrate w as a single real number.

The cost function J(w, b) is then some surface above these horizontal axes w and b so the height of the surface represents the value of J( w, b) at a certain point. And we want to find the values of w and b that correspond to the minimum of the cost function J.

![Neural%20Networks%20Basics%20%F0%9F%98%8B%20eeba82722b8648ecab3d5e75ad94d81a/Untitled%203.png](Neural%20Networks%20Basics%20%F0%9F%98%8B%20eeba82722b8648ecab3d5e75ad94d81a/Untitled%203.png)

Illustration of gradient descent

Cost function J ⇒ a convex function

To find good values for the parameter we will initialize w and b to some initial value.

for logistic regression almost any initialization method work. Usually, you initialize the value to zero. Random initialization also works.

Gradient descent takes a step and moves towards the steepest downhill direction. After a number of iterations, we eventually approach global optimum.

To make it easy to understand let us ignore b for now and make the parameter one dimensional.

So, gradient descent does this to the parameter

![Neural%20Networks%20Basics%20%F0%9F%98%8B%20eeba82722b8648ecab3d5e75ad94d81a/Untitled%204.png](Neural%20Networks%20Basics%20%F0%9F%98%8B%20eeba82722b8648ecab3d5e75ad94d81a/Untitled%204.png)

$$Repeat\ \{\\ \\ \omega:=\omega-\alpha\ \frac{d\ J(\omega)}{d\ \omega}  \\ \}$$

alpha = learning rate and control how bigger step is taken on each iteration

While writing code the variable name **dw** would be used to denote derivative of cost function wrt dw. So, we will use the notation like this

$$\omega:=\omega-\alpha\ d\omega$$

When the w will be more than global optimum the slope would be positive so, the w would become smaller when learning rate times derivative of cost function will be subtracted from the original value and vice versa.

![Neural%20Networks%20Basics%20%F0%9F%98%8B%20eeba82722b8648ecab3d5e75ad94d81a/Untitled%205.png](Neural%20Networks%20Basics%20%F0%9F%98%8B%20eeba82722b8648ecab3d5e75ad94d81a/Untitled%205.png)

## Derivatives

Nothing to be worried about it. Calculus is easy.

## Computation Graph

The computation graph organizes a computation with this blue arrow, left-to-right computation. Let's refer to the next video how you can do the backward red arrow right-to-left computation of the derivatives

![Neural%20Networks%20Basics%20%F0%9F%98%8B%20eeba82722b8648ecab3d5e75ad94d81a/Untitled%206.png](Neural%20Networks%20Basics%20%F0%9F%98%8B%20eeba82722b8648ecab3d5e75ad94d81a/Untitled%206.png)

## Derivatives with a Computation Graph

Let us say we want to compute the value of derivative of J wrt to v. It means the rate of change of cost function wrt to v.

So indeed, terminology of backpropagation, what we're seeing is that if you want to compute the derivative of this final output variable, which usually is a variable you care most about, with respect to v, then we've done one step of backpropagation.

![Neural%20Networks%20Basics%20%F0%9F%98%8B%20eeba82722b8648ecab3d5e75ad94d81a/Untitled%207.png](Neural%20Networks%20Basics%20%F0%9F%98%8B%20eeba82722b8648ecab3d5e75ad94d81a/Untitled%207.png)

 So we call it one step backwards in this graph. Now let's look at another example. What is dJ/da? In other words, if we bump up the value of a, how does that affect the value of J?

![Neural%20Networks%20Basics%20%F0%9F%98%8B%20eeba82722b8648ecab3d5e75ad94d81a/Untitled%208.png](Neural%20Networks%20Basics%20%F0%9F%98%8B%20eeba82722b8648ecab3d5e75ad94d81a/Untitled%208.png)

Chain Rule

Actually, we want to find the derivative of the cost fuction (which is usually at the end of propagation) wrt to weights which are at the start.

In python code, to give a short name to the derivative following notation is used.

![Neural%20Networks%20Basics%20%F0%9F%98%8B%20eeba82722b8648ecab3d5e75ad94d81a/Untitled%209.png](Neural%20Networks%20Basics%20%F0%9F%98%8B%20eeba82722b8648ecab3d5e75ad94d81a/Untitled%209.png)

## Logistic Regression Gradient Descent

(for single training example)

 **Recap for logistic regression:**

$$\begin{aligned} z&=w^{T} x+b\\ \hat{y}&=a=\sigma(z)\\ \mathcal{L}(a, y)&=-(y \log (a)+(1-y) \log (1-a)) \end{aligned}$$

**Forward Propagation:**

$$\begin{array}{l} x_{1} \\ w_{1} \\ x_{2} \\ w_{2} \\ \mathrm{b} \end{array} \rightarrow \fbox{$z=w_{1} x_{1}+w_{2} x_{2}+b$} \longrightarrow \fbox{$a=\sigma(z)$} \longrightarrow\fbox{$\mathcal{L}(\mathrm{a}, y)$}$$

![Neural%20Networks%20Basics%20%F0%9F%98%8B%20eeba82722b8648ecab3d5e75ad94d81a/Untitled%2010.png](Neural%20Networks%20Basics%20%F0%9F%98%8B%20eeba82722b8648ecab3d5e75ad94d81a/Untitled%2010.png)

**Back Propagation:**

$$\fbox{$\begin{array}{l} w_{1} :=w_1-\alpha\ dw_1\\ w_{2} :=w_2-\alpha\ dw_2\\ \mathrm{b}:=\mathrm{b}-\alpha\ d\mathrm{b} \end{array}$} \longleftarrow \fbox{$\begin{array}{l} dw_{1} =x_1.dz\\ dw_{2}=x_2.dz \\ d\mathrm{b}=dz \end{array}$}\longleftarrow \fbox{$d\mathrm{z}=\frac{d\ \mathcal{L}(\mathrm{a},y)}{d\mathrm{z}}=\frac{d\mathcal{L}}{da}.\frac{da}{dz}=a\left(-\frac{y}{a}+\frac{a-y}{1-a}\right)(1-a)=a-y$} \longleftarrow \fbox{$d\mathrm{a}=\frac{d\ \mathcal{L}(\mathrm{a},y)}{d\mathrm{a}}=-\frac{y}{a}+\frac{a-y}{1-a}$} \longleftarrow\fbox{$\mathcal{L}(\mathrm{a}, y)$}$$

## Gradient Descent on m Examples

Let us initialize,

$$\begin{aligned}J&=0 ;\\ d \omega_{1}&=0 ;\\ d \omega_{2}&=0 ;\\ d b&=0\end{aligned}$$

$$\text{For }i=1\text{ to }m$$

$$\begin{array}{l} z^{(i)}=\omega^{\top} x^{(i)}+b \\ a^{(i)}=\sigma\left(z^{(i)}\right) \\ J+=-\left[y^{(i)} \log a^{(i)}+\left(1-y^{(i)}\right)\log \left(1-a^{(i)}\right)\right] \\ d z^{(i)}=a^{(i)}-y^{(i)} \\ d \omega_{1}+=x_{1}^{(i)} d z^{(i)} \\ d w_{2}+=x_{2}^{(i)} d z^{(i)} \quad\quad |\ n=2 \\ d b+=d z^{(i)} \end{array}$$

To get the mean,

$$\begin{array}{l} J /=m \\ d\omega_{1} /=m\\ d \omega_{2} /=m\\ d b/=m \end{array}$$

$$\begin{array}{l} \omega_{1}:=w_{1}-\alpha d \omega_{1} \\ \omega_{2}:=\omega_{2}-\alpha d \omega_{2} \\ b:=b-\alpha d b \end{array}$$

To do with this method we need to apply 2 for loops. For loops make the algorithm slow. That is why we need vectorization.

## Derivation of dL/dz

(Optional)

If you're curious, here is the derivation for

$$\frac{d L}{d z}=a-y$$

$$\text{Note that in this part of the course, Andrew refers to } \frac{d L}{d z} \text{ as } d z$$

By the chain rule:

$$\frac{d L}{d z}=\frac{d L}{d a} \times \frac{d a}{d z}$$

***STEP 1: dL/da***

$$\begin{aligned} &L=-(y \times \log (a)+(1-y) \times \log (1-a))\\ &\frac{d L}{d a}=-y \times \frac{1}{a}-(1-y) \times \frac{1}{1-a} \times-1 \end{aligned}$$

We're taking the derivative with respect to a. Remember that there is an additional -1 in the last term when we take the derivative of (1−*a*) with respect to *a* (remember the Chain Rule).

$$\frac{d L}{d a}=\frac{-y}{a}+\frac{1-y}{1-a}$$

We'll give both terms the same denominator:

$$\frac{d L}{d a}=\frac{-y \times(1-a)}{a \times(1-a)}+\frac{a \times(1-y)}{a \times(1-a)}$$

Clean up the terms:

$$\frac{d L}{d a}=\frac{-y+a y+a-a y}{a(1-a)}$$

So now we have:

$$\frac{d L}{d a}=\frac{a-y}{a(1-a)}$$

***STEP 2: da/dz***

$$\frac{d a}{d z}=\frac{d}{d z} \sigma(z)$$

The derivative of a sigmoid has the form:

![Neural%20Networks%20Basics%20%F0%9F%98%8B%20eeba82722b8648ecab3d5e75ad94d81a/Untitled%2011.png](Neural%20Networks%20Basics%20%F0%9F%98%8B%20eeba82722b8648ecab3d5e75ad94d81a/Untitled%2011.png)

$$\frac{d}{d z} \sigma(z)=\sigma(z) \times(1-\sigma(z))$$

You can look up why this derivation is of this form. For example, google "derivative of a sigmoid", and you can see the derivation in detail.

Recall that σ(z)=a, because we defined "a", the activation, as the output of the sigmoid activation function. So, we can substitute into the formula to get:

$$\frac{d a}{d z}=a(1-a)$$

***STEP 3: dL/dz***

We'll multiply step 1 and step 2 to get the result.

$$\frac{d L}{d z}=\frac{d L}{d a} \times \frac{d a}{d z}\\ \text{From step 1: }\frac{d L}{d a}=\frac{a-y}{a(1-a)}\\ \text{From step 2: }\frac{d a}{d z}=a(1-a)\\ \frac{d L}{d z}=\frac{a-y}{a(1-a)} \times a(1-a)\\ \text{Notice that we can cancel factors to get this:}\\ \frac{d L}{d z}=a-y\\ \text{In Andrew's notation, he's referring to }\frac{d L}{d z}\text{ as } d z\\ \text{So,  } d z=a-y$$

[Derivative of sigmoid function $\sigma (x) = \frac{1}{1+e^{-x}}$](https://math.stackexchange.com/questions/78575/derivative-of-sigmoid-function-sigma-x-frac11e-x)

[Derivative Calculator](https://www.derivative-calculator.net/)

[Deriving the Sigmoid Derivative for Neural Networks](https://beckernick.github.io/sigmoid-derivative-neural-network/)

# Python and Vectorization

---

![Neural%20Networks%20Basics%20%F0%9F%98%8B%20eeba82722b8648ecab3d5e75ad94d81a/Untitled%2012.png](Neural%20Networks%20Basics%20%F0%9F%98%8B%20eeba82722b8648ecab3d5e75ad94d81a/Untitled%2012.png)

## Vectorization

Vectorization ⇒ art of getting rid of excessive/explicit for loops.

In deep learning era , data set is large so to run the operation on data through loops take a lot of time while vectorization uses the advantage of parallel computing.

$$\begin{aligned} \omega&=\left[\begin{array}{l} \vdots \\ \vdots \end{array}\right] \quad \omega \in \mathbb{R}^{n_{x}}\\ x&=\left[\begin{array}{c} \vdots \\ \vdots \end{array}\right] \quad x \in \mathbb{R}^{n_{x}} \\ \\ z&=\underline{\omega^{\top} x}+b \end{aligned}$$

***Non-vectorized approach***

***Vectorized approach***

$$z=0\\ \text{for } i \text{ in } \sin (n-x){:} \\ \quad \quad z+=\omega [i] i \times x [j]\\\ \\z+=b$$

$$z=\underbrace{n p \cdot \operatorname{dot}(\omega, x)}_{\omega^{\top} x}+b$$

[https://gist.github.com/muqadir1/4cade3e49f12893f0179d29d1550e7b6](https://gist.github.com/muqadir1/4cade3e49f12893f0179d29d1550e7b6)

![Neural%20Networks%20Basics%20%F0%9F%98%8B%20eeba82722b8648ecab3d5e75ad94d81a/Untitled%2013.png](Neural%20Networks%20Basics%20%F0%9F%98%8B%20eeba82722b8648ecab3d5e75ad94d81a/Untitled%2013.png)

$$\left.\begin{array}{l} \text { GPU } \\ \text { CPU } \end{array}\right\} \begin{array}{r} \text { SIMD - Sigle Instruction } \\ \text { Multiple Data. } \end{array}$$

## More Vectorization Examples

> **Whenever possible, avoid explicit for-loops.**

While using matrixes we can avoid two explicit For loops by using vectorization.

$$u=A v$$

***Non-vectorized approach***

***Vectorized approach***

$$\begin{aligned} &u_{i}=\sum_{j} A_{i j} v_{j}\\ &u=n p \cdot zeros((n, 1))\\ &\quad\text { for i } \cdots \quad \leftarrow\\ &\quad \quad \begin{array}{l} \text { for } j \cdots \quad \leftarrow \\ \quad u[i]+=A [ i ][j] * v[i] \end{array} \end{aligned}$$

$$z=\underbrace{n p \cdot \operatorname{dot}(\omega, x)}$$

![Neural%20Networks%20Basics%20%F0%9F%98%8B%20eeba82722b8648ecab3d5e75ad94d81a/Untitled%2014.png](Neural%20Networks%20Basics%20%F0%9F%98%8B%20eeba82722b8648ecab3d5e75ad94d81a/Untitled%2014.png)

$$\begin{aligned} &J=0, \quad d w 1=0, \quad d w 2=0, \quad d b=0\\ &\text { for } i=1 \text { to } \mathrm{m}:\\ &\quad\quad z^{(i)}=w^{T} x^{(i)}+b\\ &\quad\quad a^{(i)}=\sigma\left(z^{(i)}\right)\\ &\quad\quad J+=-\left[y^{(i)} \log \hat{y}^{(i)}+\left(1-y^{(i)}\right) \log \left(1-\hat{y}^{(i)}\right)\right]\\ &\quad\quad \mathrm{d} z^{(i)}=a^{(i)}\left(1-a^{(i)}\right)\\ &\quad\quad \mathrm{d} w_{1}+=x_{1}^{(i)} \mathrm{d} z^{(i)}\\ &\quad\quad \mathrm{d} w_{2}+=x_{2}^{(i)} \mathrm{d} z^{(i)}\\ &\quad\quad \mathrm{db}+\mathrm{d} z^{(i)}\\ &J=J / m, \quad d w_{1}=d w_{1} / m, \quad d w_{2}=d w_{2} / m, \quad db=d b / m \end{aligned}$$

![Neural%20Networks%20Basics%20%F0%9F%98%8B%20eeba82722b8648ecab3d5e75ad94d81a/Untitled%2015.png](Neural%20Networks%20Basics%20%F0%9F%98%8B%20eeba82722b8648ecab3d5e75ad94d81a/Untitled%2015.png)

$$\begin{aligned} &J=0, \quad d w = np\cdot zeros((n_{x},1)), \quad d b=0\\ &\text { for } i=1 \text { to } \mathrm{m}:\\ &\quad\quad z^{(i)}=w^{T} x^{(i)}+b\\ &\quad\quad a^{(i)}=\sigma\left(z^{(i)}\right)\\ &\quad\quad J+=-\left[y^{(i)} \log \hat{y}^{(i)}+\left(1-y^{(i)}\right) \log \left(1-\hat{y}^{(i)}\right)\right]\\ &\quad\quad \mathrm{d} z^{(i)}=a^{(i)}\left(1-a^{(i)}\right)\\ &\quad\quad \mathrm{d} w+=x^{(i)} \mathrm{d} z^{(i)}\\ &\quad\quad \mathrm{db}+\mathrm{d} z^{(i)}\\ &J=J / m, \quad d w =d w / m \quad db=d b / m \end{aligned}$$

## Vectorizing Logistic Regression

how to vectorize the implementation of logistic regression, without using even a single for loop.

We have m training example. In order to carry out forward propagation on the entire training set we have to get the following outputs.

$$\begin{aligned} z^{(1)}=w^{T} x^{(1)}+b  \qquad \quad z^{(2)}&=w^{T} x^{(2)}+b \qquad  \quad z^{(3)}=w^{T} x^{(3)}+b\\ a^{(1)}=\sigma\left(z^{(1)}\right) \qquad \quad a^{(2)}&=\sigma\left(z^{(2)}\right) \qquad  \quad a^{(3)}=\sigma\left(z^{(3)}\right) \end{aligned}$$

It turns out that we can do it in two lines of code without using For loops. We have

$$X=\left[\begin{array}{ccc}  | & | & &| \\  x^{(1)} & x^{(2)} & \cdots & x^{(m)} \\  | & | & &|  \end{array}\right]\qquad\qquad (n_x,m)$$

and we need to find

$$\left[z^{(1)} z^{(1)} \cdots\ z^{(m)}\right]=\omega^{\top} X+\underbrace{\left[b^{(1)}\quad b^{(2)}\cdots\ b^{(m)}\right]}_{1\times m}$$

$$=\left[w^{\top} x^{(1)}+b \qquad w^{\top} x^{(1)}+b\quad \cdots\quad  w^{\top} x^{m}+b\right]$$

which turns out to be a (1 by m) matrix made up of terms that we needed. All this calculation can be done in python using the code. 

$$z=n p \cdot \operatorname{dot}(\omega \cdot T, x)+\underbrace{b}_{(1,1) \in \mathbb{R}}$$

As b is  a 1 by 1 real number, while adding python automatically expands it to 1 by m vector. This property is called **broadcasting**.

To be done in the programming assignment ⇒

$$A=\left[a^{(1)}\quad a^{(2)}\ldots\ a^{(m)}\right]=\sigma(z)$$

## Vectorizing Logistic Regression's Gradient Output

how to use vectorization to also perform the gradient computations for all M training samples.

$$\begin{aligned} &\begin{array}{l} d z^{(1)}=a^{(1)}-y^{(1)} \qquad d z^{(2)}=a^{(2)}-y^{(2)}\qquad \cdots\\ d z=\left[d z^{(1)}\quad d_{1}^{(2)} \ldots\ d z^{(m)}\right] \end{array}\\ &A=\left[a^{(1)}\quad a^{(2)} \ldots\ a^{(n)}\right] \cdot Y=\left[y^{(1)}\quad y^{(1)} \ldots\ y^{(m)}\right]\\ &d Z=A-Y=\left[\begin{array}{llll} a^{(1)}- y^{(1)} &\ a^{(2)}-y^{(2)} & \cdots &\ a^{(m)}-y^{(m)} \end{array}\right] \end{aligned}$$

There were two For loops in updating the weights

1. First FOR loops which we got rid of previously was loop along each parameter of single training example. We solved it by using a vector w of on training example which contained all the weights of the single example.
2. Second FOR loop was the loop that was used to update the parameters while looping along every training example. We have to eliminate that loop now.

$$\begin{array}{l} {\left[\begin{array}{c} \rightarrow d \omega=0 \\ d \omega+=x^{(1)} d z^{(1)} \\ d \omega+=x^{(2)} d z^{(2)} \\ \vdots \\ d \omega+=x^{(m)} d z^{(m)} \\ d \omega /=m \end{array}\right.} & \begin{array}{l} \rightarrow d b=0 \\ d b+=d z^{(1)} \\ d b+=d z^{(2)} \\ \qquad \vdots \\ d b+=d z^{(m)} \\ d b /=m \end{array} \end{array}$$

For vectorized implementation we have to sum all the weights or biases gotten through different examples and dividing by m (#  of training examples). For biases,

$$\begin{aligned} d b &=\frac{1}{m} \sum_{i=1}^{m} d z^{(i)} \\ &=\frac{1}{m} \text { np}\cdot \operatorname{sum}(d z) \end{aligned}$$

And for weights,

$$\begin{aligned} d w &=\frac{1}{m}\left[\begin{array}{ccc}  | & | & &| \\  x^{(1)} & x^{(2)} & \cdots & x^{(m)} \\  | & | & &|  \end{array}\right] \left[\begin{array}{c} d z^{(1)} \\ \vdots \\ d z^{(m)}\\ \end{array}\right] \\ &=\frac{1}{m}\left[x^{(1)}dz^{(1)}+\cdots+x^{(m)}dz^{(m)}\right] \\ d w &=\frac{1}{m} X d z^{\top} \end{aligned}$$

So, so far, we have the following in the non-vectorized form and we have to vectorize it.

$$\begin{aligned} &\mathrm{J}=0,\qquad \underbrace{\mathrm{d} w_{1}=0, \mathrm{d} w_{2}=0}_{\mathrm{d} w=0},\qquad \mathrm{db}=0\\ &\begin{aligned} \text { for } i=1 &\text { to } m: \\ &z^{(i)}=w^{T} x^{(i)}+b\\ &a^{(i)}=\sigma\left(z^{(i)}\right)\\ &J+=-\left[y^{(i)} \log a^{(i)}+\left(1-y^{(i)}\right) \log \left(1-a^{(i)}\right)\right]\\ &\mathrm{d} z^{(i)}=a^{(i)}-y^{(i)}\\ &\left\{\begin{array}{l} d w_{1}+=x_{1}^{(i)} d z^{(i)} \\ d w_{2}+=x_{2}^{(i)} d z^{(i)} \\ \end{array}\right\} d \omega+=x^{(i)} * d z^{(i)}\\ &d b+=d z^{(i)} \end{aligned}\\ &\begin{array}{l} \mathrm{J}=\mathrm{J} / \mathrm{m}, \qquad \underbrace{\mathrm{d} w_{1}=\mathrm{d} w_{1} / \mathrm{m},\mathrm{d} w_{2}=\mathrm{d} w_{2} / \mathrm{m}}_{ w=\mathrm{d} w/ \mathrm{m}}, \\ \mathrm{d} \mathrm{b}=\mathrm{db} / \mathrm{m} \end{array} \end{aligned}$$

This is how it is done in vectorized forms

$$\begin{aligned} z&=w^{T} x+b\\ &=n p \cdot d o t(\omega \cdot T, x)+b\\ A&=\sigma(z)\\ d z&=A-Y\\ d w&=\frac{1}{m}X d z^{T}\\ d b&=\frac{1}{m} n p \cdot sum (d z)\\ w:&=w-\alpha d w\\ b:&=b-\alpha d b \end{aligned}$$

These few lines of codes include forward propagation, backward propagation and updating parameters for one iteration.

![Neural%20Networks%20Basics%20%F0%9F%98%8B%20eeba82722b8648ecab3d5e75ad94d81a/5.jpg](Neural%20Networks%20Basics%20%F0%9F%98%8B%20eeba82722b8648ecab3d5e75ad94d81a/5.jpg)

⇒ For multiple iteration we have to apply a loop (this loop cannot be eliminated by using vectorization)

## Broadcasting in Python

## A note on python/numpy vectors

## Quick tour of Jupyter/iPython Notebooks

## Explanation of logistic regression cost function (optional)

$$\hat{y}=\sigma(\omega^Tx+b)\\ \text{Where}\quad \sigma(z)=\frac{1}{1+e^{-z}}\\ \text{Interpret}\quad \hat{y}=p(y=1 | x)$$

There would be two cases in case of logistic regression the probability of these cases would be predicted as :

$$\left.\begin{array}{lll} \text { If } & y=1: & p(y | x)=\hat{y} \\ \text { If } & y=0: & p(y | x)=1-\hat{y} \end{array}\right\} \quad p(y | x)$$

These two equations can be transformed into single equation and we can put the value of y and verify here.

$$p(y | x)=\hat{y}^{y}(1-\hat{y})^{(1-y)}\qquad\leftarrow$$

As this a large number maximizing its log is same as maximizing it. So,

$$\log p(y | x)=\log \hat{y}^{y}(1-\hat{y})^{(1-y)}=y \log \hat{y}+(1-y) \log (1-\hat{y})$$

$$\uparrow\quad \log p(y | x)=-(y \log \hat{y}+(1-y) \log (1-\hat{y})) =-\mathcal{L}(\hat{y}, y)\quad \downarrow$$

### For m training examples