# DongKyu Kim
# ECE 471 CGML Assignment 1
# Professor Curro

# library imports
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tqdm import tqdm

# hyper parameters
iteration = 1000
learning_rate = 0.02
lambdah = 0.0001 # regularization constant
display = 100
sigma_noise = 0.1


samples = 1000
percent_train = 0.75
N_train = samples*percent_train
N_test = samples-N_train
layers = [2,20,20,1]

# Data Generation
class Data(object):
    def __init__(self,N):
        np.random.seed(31415)
        temp = np.random.uniform(0,4*np.pi,size=(N,1))
        self.label = np.random.randint(2,size=(N,1))
        self.x1 = temp*np.cos(temp+label*np.pi)+sigma_noise*np.random.normal(size=(N,1))
        self.x2 = temp*np.sin(temp+label*np.pi)+sigma_noise*np.random.normal(size=(N,1))
        self.N = N
# Model
class My_Model(object):
    def __init__(self,sess,data,layers,learning_rate,iteration,lambdah):
        self.sess = sess
        self.data = data
        self.layers = layers
        self.learning_rate = learning_rate
        self.iteration = iteration
        self.lambdah = lambdah
        self.build_model()

    def build_model(self):
        self.x = tf.placeholder(tf.float32,[self.data.N,2])
        self.y = tf.placeholder(tf.float32,[self.data.N,1])
        self.w = {}
        self.b = {}
        for i in range(0,len(self.layers)-1):
            self.w[i] = tf.get_variable('w'+i,[layer[i],layer[i+1]],tf.float32,tf.random_normal_initializer())
            self.b[i] = tf.get_variable('b'+i,[layer[i+1],1],tf.float32,tf.zeros_initializer())
        # first layer
        self.y_hat = tf.nn.relu(tf.add(tf.matmul(self.x,self.w[0]),self.b[0]))
        for i in range(1,len(self.layers)-1):
            self.y_hat = tf.nn.relu(tf.add(tf.matmul(self.y_hat,self.w[i]),self.b[i]))

# tensor definition
x = tf.placeholder(tf.float32, [N,1])
y = tf.placeholder(tf.float32, [N,1])
y_hat = f(x)
loss = tf.reduce_mean(tf.pow(y_hat - y, 2))
optim = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
init = tf.global_variables_initializer()

# session run
sess = tf.Session()
sess.run(init)
data = Data()
for _ in tqdm(range(0, iteration)):
    sess.run([loss, optim], feed_dict={x: data.x, y: data.y})

# results
print("Parameter estimates:")
storage = []
for var in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES):
    print(
        var.name.rstrip(":0"),
        np.array_str(np.array(sess.run(var)).flatten(), precision=3))
    storage.append(sess.run(var))
[w_model,mu_model,sigma_model,b_model] = storage

# Useful Functions
def gaussian_model(x,w,mu,sigma,b):
    phi = tf.exp(-(tf.transpose(x)-mu)**2/sigma**2)
    return tf.transpose(tf.matmul(tf.transpose(w),phi)+b)

def f(x):
    w = tf.get_variable('w',[M,1],tf.float32,tf.random_normal_initializer())
    mu = tf.get_variable('mu',[M,1],tf.float32,tf.random_uniform_initializer())
        # as x is drawn from uniform(0,1) it makes sense that mu is in the
        # range as well
    sigma = tf.get_variable('sigma',[M,1],tf.float32,tf.random_normal_initializer(),constraint=lambda x:tf.abs(x))
        # since technically sigma should be positive, sigma is forced to be.
    b = tf.get_variable('b',[],tf.float32,tf.zeros_initializer())
    return gaussian_model(x,w,mu,sigma,b)

def norm_dist(x,mu,sigma):
    return 1/(sigma*np.sqrt(2*np.pi))*np.exp(-(x-mu)**2/(2*sigma**2))

#plotting
x_plot = np.linspace(-0.2,1.2,300,dtype=np.float32)
y_truth = np.sin(2*np.pi*x_plot)
y_model = gaussian_model(x_plot,w_model,mu_model,sigma_model,b_model)

    #plot 1
plt.figure(1)
plt.subplot()
plt.plot(x_plot,y_truth,'b',label='noiseless sine')
plt.plot(x_plot,sess.run(y_model),'--r',label='model')
plt.plot(data.x,data.y,'go',label='data points')
plt.xlabel('x')
plt.xticks(np.arange(0,1.2,step=0.5))
plt.ylabel('y')
plt.axis([0,1,-1.2,1.2])
plt.title('Linear Regression of a Noisy Sinewave using Gaussian Basis Function')
plt.legend()
plt.show()

    #plot 2
plt.figure(2)
plt.subplot()
for i in range(1,M+1):
    plt.plot(x_plot,norm_dist(x_plot,mu_model[i-1],sigma_model[i-1]),
        label='Basis '+str(i)+': mu ='+str(mu_model[i-1])[1:6]+',sigma ='
        +str(sigma_model[i-1])[1:6])
plt.xlabel('x')
plt.xticks(np.arange(0,1.2,step=0.5))
plt.ylabel('y')
plt.axis([-0.2,1.2,0,plt.axis()[3]])
plt.title('Gaussian Bases for the Manifold')
plt.legend()
plt.show()
