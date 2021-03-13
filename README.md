## ROBT 407 | Homework 1 | Task 0.3
##### Danissa Sandykbayeva

Here I am just importing all the necessary libraries that I'm going to use in this task.


```python
import matplotlib.pyplot as plt
import numpy as np
np.random.seed(5)
```

#### 3.1 Generate a data set of size 20 as directed by Exercise 1.4 of LFD, and plot the examples ${(x_n, y_n)}$ as well as the target function f on a plane. Be sure to mark the examples from different classes differently, and add labels to the axes of the plot.
Now I am creating a linearly-separable 2D dataset of 20 random points - 'D', and a list of labels corresponding to each point - 'L' 


```python
## Generating 2-D Dataset 'D' with 20 samples with lables '+1' or '-1'
X0 = (np.random.rand(10))*4+2.3
Y0 = (np.random.rand(10))*4-2.3
X1 = (np.random.rand(10))*4-2.3
Y1 = (np.random.rand(10))*4+2.3
D = []
for i in range(10):
    D.append([X0[i],Y0[i]])
    D.append([X1[i],Y1[i]])
L = [1,-1,1,-1,1,-1,1,-1,1,-1,1,-1,1,-1,1,-1,1,-1,1,-1]
```

This is how final generated points look on the scatter plot, and we can again see that they are clearly linearly separable.


```python
for i in range (20):
    if L[i] == 1:
        plt.scatter(D[i][0],D[i][1],c='b',marker='x')
    else:
        plt.scatter(D[i][0],D[i][1],c='r',marker='o')
plt.title('Dataset D')
plt.xlabel('$X_n$')
plt.ylabel('$Y_n$')
plt.show()
```


![png](output_6_0.png)


#### 3.2 Run Perceptron Learning Algorithm (PLA) on the data set above. Report the number of updates that PLA takes before converging. Plot the examples ${(x_n, y_n)}$, the target function f , and the final hypothesis g in the same figure. Comment on whether f is close to g.

Now I am here defining a function **pla** (**data** - training dataset, **labels** - vector with labels, **t** - number of iterrations/tries to learn) for the **PLA - Perceptron Learning Algorithm**. <br> Where: **W** - vector of weights, **n_error** - error simply computed as the number of incorrect predictions in a single try divided by the total number of predictions made *(this error is just for the purpose of demonstrating algorithm's preformance)*<br><br>As it was showed in the lectures the algorithm compares current prediction with the label and if it is not correct, it adjusts the weights accordingly ($w_{t+1}=w_t+y_n(t)*x_n(t)$) and returns final weights after going through the specified number of iterations.


```python
def pla(data,labels,t):
    # Initializing weights to 0.0 to start with
    W=[0.0 for i in range (2)]
    # Learning loop
    for n_tries in range (t):
        n_error=0
        # This error is purely to keep track of the performance of the hypothesis
        for i in range (len(D)):
            if(np.sign(np.dot(data[i],W))!=labels[i]):
                # Changing current weights if an error in prediction is made
                W=W+np.dot(data[i],labels[i])
                n_error += 1
        print('Try number: %d, error:%.3f' % (n_tries+1,n_error/len(D)))
    return W    
```


```python
weights=pla(D,L,5)
weights
```

    Try number: 1, error:0.050
    Try number: 2, error:0.000
    Try number: 3, error:0.000
    Try number: 4, error:0.000
    Try number: 5, error:0.000
    




    array([ 3.18797268, -1.97703492])



As it can seen from the error values above it takes less than 5 tries for the algorithm to learn the correct hypothesis. Below is the plot of points with final decision boundary (the final hypothesis g)


```python
x_hyperplane = np.linspace(-2,6)
slope = - weights[0]/weights[1]
y_hyperplane = slope * x_hyperplane
plt.plot(x_hyperplane, y_hyperplane, '-',label='final hypothesis g(X)')
plt.legend()
for i in range (20):
    if L[i] == 1:
        plt.scatter(D[i][0],D[i][1],c='b',marker='x')
    else:
        plt.scatter(D[i][0],D[i][1],c='r',marker='o')
plt.title('Dataset D with the learned decision boundary')
plt.xlabel('$X_n$')
plt.ylabel('$Y_n$')
plt.show()
plt.show()
print('Final hypothesis: g(X) = %.5f*X' % slope)
```


![png](output_11_0.png)


    Final hypothesis: g(X) = 1.61250*X
    

As it is seen from the graph above the algorithm perfectly learns on the training data of 20 points and satisfies the target function f(x).
#### 3.3 Repeat everything in (3.2) with another randomly generated data set of size 20. Compare your results with (3.2)


```python
## Generating 2-D Dataset 'D' with 20 samples with lables '+1' or '-1'
X0 = (np.random.rand(10))*2-1.5
Y0 = (np.random.rand(10))*2+1.5
X1 = (np.random.rand(10))*2+1.5
Y1 = (np.random.rand(10))*2-1.5
D = []
for i in range(10):
    D.append([X0[i],Y0[i]])
    D.append([X1[i],Y1[i]])
L = [1,-1,1,-1,1,-1,1,-1,1,-1,1,-1,1,-1,1,-1,1,-1,1,-1]

weights=pla(D,L,5)
weights

x_hyperplane = np.linspace(-3,7)
slope = - weights[0]/weights[1]
y_hyperplane = slope * x_hyperplane
plt.plot(x_hyperplane, y_hyperplane, '-',label='final hypothesis g(X)')
plt.legend()
for i in range (20):
    if L[i] == 1:
        plt.scatter(D[i][0],D[i][1],c='b',marker='x')
    else:
        plt.scatter(D[i][0],D[i][1],c='r',marker='o')
plt.title('Dataset D with the learned decision boundary')
plt.xlabel('$X_n$')
plt.ylabel('$Y_n$')
plt.show()
print('Final hypothesis: g(X) = %.5f*X' % slope)
```

    Try number: 1, error:0.050
    Try number: 2, error:0.000
    Try number: 3, error:0.000
    Try number: 4, error:0.000
    Try number: 5, error:0.000
    


![png](output_13_1.png)


    Final hypothesis: g(X) = 0.32278*X
    

#### 3.4 Repeat everything in (3.2) with another randomly generated data set of size 100. Compare your results with (3.2)


```python
## Generating 2-D Dataset 'D' with 100 samples with lables '+1' or '-1'
X0 = (np.random.rand(50))*5+2.5
Y0 = (np.random.rand(50))*5-2.5
X1 = (np.random.rand(50))*5-2.5
Y1 = (np.random.rand(50))*5+2.5
D = []
for i in range(50):
    D.append([X0[i],Y0[i]])
    D.append([X1[i],Y1[i]])

    L = np.ones(100)
for k in range (100):
    if k%2!=0:
        L[k]=-1;

weights=pla(D,L,5)
weights

x_hyperplane = np.linspace(-3,7)
slope = - weights[0]/weights[1]
y_hyperplane = slope * x_hyperplane
plt.plot(x_hyperplane, y_hyperplane, '-',label='final hypothesis g(X)')
plt.legend()
for i in range (100):
    if L[i] == 1:
        plt.scatter(D[i][0],D[i][1],c='b',marker='x')
    else:
        plt.scatter(D[i][0],D[i][1],c='r',marker='o')
plt.title('Dataset D (100 points) with the learned decision boundary')
plt.xlabel('$X_n$')
plt.ylabel('$Y_n$')
plt.show()
print('Final hypothesis: g(X) = %.5f*X' % slope)
```

    Try number: 1, error:0.020
    Try number: 2, error:0.000
    Try number: 3, error:0.000
    Try number: 4, error:0.000
    Try number: 5, error:0.000
    


![png](output_15_1.png)


    Final hypothesis: g(X) = 0.61453*X
    

Even though the amount pf the data samples is significantly larger as long as they are linearly separables the algorithm learns very fast and separates data perfectly.

#### 3.5 Repeat everything in (3.2) with another randomly generated data set of size 1000. Compare your results with (3.2)


```python
## Generating 2-D Dataset 'D' with 1000 samples with lables '+1' or '-1'
X0 = (np.random.uniform(-2000,2000,500))*30.2+50000.5
Y0 = (np.random.uniform(-2000,2000,500))*30.2-50000.5
X1 = (np.random.uniform(-2000,2000,500))*30.2-50000.5
Y1 = (np.random.uniform(-2000,2000,500))*30.2+50000.5
D = []
for i in range(500):
    D.append([X0[i],Y0[i]])
    D.append([X1[i],Y1[i]])

    L = np.ones(1000)
for k in range (1000):
    if k%2!=0:
        L[k]=-1;

weights=pla(D,L,10)
weights

x_hyperplane = np.linspace(-100000,100000)
slope = - weights[0]/weights[1]
y_hyperplane = slope * x_hyperplane
plt.plot(x_hyperplane, y_hyperplane, '-',label='final hypothesis g(X)')
plt.legend()
for i in range (1000):
    if L[i] == 1:
        plt.scatter(D[i][0],D[i][1],c='b',marker='x')
    else:
        plt.scatter(D[i][0],D[i][1],c='r',marker='o')
plt.title('Dataset D (1000 points) with the learned decision boundary')
plt.xlabel('$X_n$')
plt.ylabel('$Y_n$')
plt.show()
print('Final hypothesis: g(X) = %.5f*X' % slope)
```

    Try number: 1, error:0.027
    Try number: 2, error:0.024
    Try number: 3, error:0.029
    Try number: 4, error:0.026
    Try number: 5, error:0.028
    Try number: 6, error:0.025
    Try number: 7, error:0.027
    Try number: 8, error:0.026
    Try number: 9, error:0.030
    Try number: 10, error:0.026
    


![png](output_18_1.png)


    Final hypothesis: g(X) = 0.39562*X
    

In the case with 1000 samples there are few 'noise values' or values that make it impossible to perfectly lineraly separate all samples. But the algorithm still reaches minimum error almost at the same speed as in with 20 samples. <br><br>*Also this was nearly impossible to recreate with so many random numbers so that it won't look like two giant squares, since the density of the points is too high*
