'''

What is Tensorflow?

Tensorflow is a library for number calculating that is created and maintained by Google. It is used for machine learning tasks. A fuck ton of companies ranging from startups to fortune 500 motherfuckers are using this in production.

Tensorflow allows you to do some computations on your computer. The main abstraction behind all of the magic is stateful data flow graphs. Tensorflow uses Tensors. What are those you ask?

According to the site for Tensorflow, a tensor is a typed multi dimensional array . A tensor is a matrix jacked on steroids. More dimensions and built in functionalities.

install tensorflow with --> pip install tensorflow


'''
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

v1 = tf.Variable(0.0)
p1 = tf.placeholder(tf.float32)
new_val = tf.add(v1, p1)
update = tf.assign(v1, new_val)



'''
Simple Linear Regression in Tensorflow

    Formula for linear regression.

            Y = aX + b

Y is the dependent variable, and X is the independent variable. Slope and intercept is used to model data into a motherfucking line.


'''

# Random X values
X = np.random.rand(100).astype(np.float32)
a = 50.0
b = 40.0
Y = a * X + b




Y = np.vectorize(lambda y: y + np.random.normal(loc=0.0, scale=0.05))(Y)



'''
This is our starting point. We start 'a' at 1, and 'b' at 1. We get another line made with those two simple points.  'y = mx + b'
'''
a_var = tf.Variable(1.0)
b_var = tf.Variable(1.0)
y_var = a_var * X + b_var


'''
What we mean by 'loss' is the mean squared error. That is what we are trying to minimize.
'''
loss = tf.reduce_mean(tf.square(y_var - Y))

'''
'optimizer' is a gradient descent function that has a learning rate of 0.5. 'train' takes in that optimizer with the current amount of loss.
'''
optimizer = tf.train.GradientDescentOptimizer(0.5)
train = optimizer.minimize(loss)


## Number of training steps needed to train the linear regression model.
TRAINING_STEPS = 100000

# Results that we will have.
results = []



# Run the tensorflow session.
with tf.Session() as sess:

    # Initialize all of the global variables.
    sess.run(tf.global_variables_initializer())


    '''

    These are the starting variables. They will be starting at 1.0.
        print(sess.run(a_var))
        print(sess.run(b_var))

    We calculate the loss and use that magic number to find the value we want using regression. Se above 'loss' variable to see how we got the number. tf.reduce_mean was used.
        print(sess.run(loss))

    Final training function that we run. Element 0 has the training function, and element 1 has the slope, while element 2 has the y intercept.
        print(sess.run([train, a_var, b_var]))

    '''

    # Number of times to train.
    for step in range(TRAINING_STEPS):
        results.append(sess.run([train, a_var, b_var])[1:])


#Last element in the array of results.
final_pred = results[-1]

# Final A.
a_hat = final_pred[0]

# Final B.
b_hat = final_pred[1]

# Final Y
y_hat = a_hat * X + b_hat

print("a:", a_hat, "b:", b_hat)


'''

Lets try making sense of what is happening. If you look at it, we are trying to find the target values for a, and b. a == 50 and b == 40. We use gradient descent to get a training function that get from a mean loss number. After that we use sess.run with the training function and the starting variables to get the ideal numbers we are looking for. So lets say we started with 1 for the slope, and 1 for the y intercept. Our ideal numbers are 40 and 50. We keep on rerunning for 100 times until the numbers are nearly identical! There you go.

The predicted function is = y_hat
The given one is = Y

'''
