#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 15 10:27:00 2019

@author: huasongzhang
"""
from keras.models import Model
from keras.models import Sequential
from keras.layers import Dense
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D 
import random



def lorenz(x, y, z, r, s=10, b=2.667):
    '''
    Given:
       x, y, z: a point of interest in three dimensional space
       s, r, b: parameters defining the lorenz attractor
    Returns:
       x_dot, y_dot, z_dot: values of the lorenz attractor's partial
           derivatives at the point x, y, z
    '''
    x_dot = s*(y - x)
    y_dot = r*x - y - x*z
    z_dot = x*y - b*z
    r_dot = 0
    return x_dot, y_dot, z_dot, r_dot


dt = 0.01
num_steps = 800

# Need one more for the initial values
#xs = np.empty((num_steps*100,))
#ys = np.empty((num_steps*100,))
#zs = np.empty((num_steps*100,))

# Step through "time", calculating the partial derivatives at the current point
# and using them to estimate the next point
R = np.array([10,28,40])
for j in range(3):
    r = R[j]
    globals()['input'+str(j+1)] = np.empty((num_steps*100,4))
    globals()['output'+str(j+1)] = np.empty((num_steps*100,4))
    for k in range(100):
    # Set initial values
        xs = np.empty((num_steps+1,))
        ys = np.empty((num_steps+1,))
        zs = np.empty((num_steps+1,))
        x0 = (np.random.random(3)-0.5)*30;
        xs[0], ys[0], zs[0] = (x0[0], x0[1], x0[2]);
        for i in range(num_steps-1):
            x_dot, y_dot, z_dot, r_dot = lorenz(xs[i], ys[i], zs[i], r)
            xs[i + 1] = xs[i] + (x_dot * dt)
            ys[i + 1] = ys[i] + (y_dot * dt)
            zs[i + 1] = zs[i] + (z_dot * dt)
        b = np.column_stack((xs,ys,zs,r*np.ones(num_steps+1,)))
        globals()['input'+str(j+1)][num_steps*k:num_steps*(k+1),:] = b[0:num_steps,:]
        globals()['output'+str(j+1)][num_steps*k:num_steps*(k+1),:] = b[1:num_steps+1,:]
        

train_input = np.vstack((input1,input2,input3));
train_output = np.vstack((output1,output2,output3));

#%%
# Plot
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.plot(input1[:,0], input1[:,1], input1[:,2], lw=0.5)
ax.set_xlabel("X Axis")
ax.set_ylabel("Y Axis")
ax.set_zlabel("Z Axis")
ax.set_title("Lorenz Attractor")

plt.show()

fig = plt.figure()
ax = fig.gca(projection='3d')
ax.plot(input2[:,0], input2[:,1], input2[:,2], lw=0.5)
ax.set_xlabel("X Axis")
ax.set_ylabel("Y Axis")
ax.set_zlabel("Z Axis")
ax.set_title("Lorenz Attractor")

plt.show()

fig = plt.figure()
ax = fig.gca(projection='3d')
ax.plot(input3[:,0], input3[:,1], input3[:,2], lw=0.5)
ax.set_xlabel("X Axis")
ax.set_ylabel("Y Axis")
ax.set_zlabel("Z Axis")
ax.set_title("Lorenz Attractor")

plt.show()

#%%
model = Sequential()
#model.add(Dense(units=128, activation='relu',input_dim = 4))
model.add(Dense(units=128, activation='sigmoid',input_dim = 4))
model.add(Dense(units=128, activation='relu'))
#model.add(Dense(units=128, activation='tanh'))
model.add(Dense(units=4, activation='linear'))
#model.add(Dense(units=4, activation='softmax'))
#model.compile(loss='mean_squared_error',
#              optimizer='sgd',
#              metrics=['accuracy'])
model.compile(loss='categorical_crossentropy',
              optimizer='sgd',
              metrics=['accuracy'])
model.fit(train_input, train_output, epochs=100)
#loss_and_metrics = model.evaluate(test_input, test_output, batch_size=100)
#classes = model.predict(x_test, batch_size=128)
#%%
steps = 800

Rt = np.array([17,35])
for j in range(2):
    r = Rt[j]
    globals()['test'+str(j+1)] = np.empty((steps,4))
    globals()['pre'+str(j+1)] = np.empty((steps,4))
    # Set initial values
    xs = np.empty((steps,))
    ys = np.empty((steps,))
    zs = np.empty((steps,))
    prediction = np.empty((steps,4))
    x0 = (np.random.random(3)-0.5)*30;
    xs[0], ys[0], zs[0] = (x0[0], x0[1], x0[2]);
    prediction[0,:] = np.array((x0[0],x0[1],x0[2],r))
    for i in range(steps-1):
        x_dot, y_dot, z_dot, z_dot = lorenz(xs[i], ys[i], zs[i], r)
        xs[i + 1] = xs[i] + (x_dot * dt)
        ys[i + 1] = ys[i] + (y_dot * dt)
        zs[i + 1] = zs[i] + (z_dot * dt)
        prediction[i+1,:] = model.predict([[prediction[i,:]]])
    b = np.column_stack((xs,ys,zs,r*np.ones(steps,)))
    globals()['test'+str(j+1)] = b
    globals()['pre'+str(j+1)] = prediction


#%%
# Plot
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.plot(test1[:,0], test1[:,1], test1[:,2], lw=0.5)
ax.plot(pre1[:,0], pre1[:,1], pre1[:,2], lw=0.5)
ax.set_xlabel("X Axis")
ax.set_ylabel("Y Axis")
ax.set_zlabel("Z Axis")
ax.set_title("Lorenz Attractor")

plt.show()

#%%
# Plot
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.plot(test2[:,0], test2[:,1], test2[:,2], lw=0.5)
ax.plot(pre2[:,0], pre2[:,1], pre2[:,2], lw=0.5)
ax.set_xlabel("X Axis")
ax.set_ylabel("Y Axis")
ax.set_zlabel("Z Axis")
ax.set_title("Lorenz Attractor")

plt.show()

#%%
fig, ax = plt.subplots(3,2,figsize=(6, 9))
ax[0,0].plot(test1[:,0],'r')
ax[0,0].plot(pre1[:,0],'b')
ax[1,0].plot(test1[:,1],'r')
ax[1,0].plot(pre1[:,1],'b')
ax[2,0].plot(test1[:,2],'r')
ax[2,0].plot(pre1[:,2],'b')
ax[0,1].plot(test2[:,0],'r')
ax[0,1].plot(pre2[:,0],'b')
ax[1,1].plot(test2[:,1],'r')
ax[1,1].plot(pre2[:,1],'b')
ax[2,1].plot(test2[:,2],'r',label = 'true trajectory')
ax[2,1].plot(pre2[:,2],'b',label = 'prediction')
ax[2,1].legend()
fig.suptitle('Comparison of predictions and true trajectories')
fig.savefig('lorenz.jpg')



    
