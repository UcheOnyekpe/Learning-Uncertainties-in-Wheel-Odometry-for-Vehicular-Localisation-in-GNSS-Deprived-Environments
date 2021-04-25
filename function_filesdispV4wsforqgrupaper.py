# -*- coding: utf-8 -*-
"""
Created on Thu Jul 23 03:29:06 2020

@author: onyekpeu
"""

# -*- coding: utf-8 -*-
"""
Created on Sat May 30 04:44:25 2020

@author: onyekpeu
"""
#from __future__ import absolute_import
#from __future__ import division
#from __future__ import print_function
#from learning_rate_schedule import * #import ExponentialDecay
import tensorflow.keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM, GRU, SimpleRNN, Bidirectional
from tensorflow.keras.layers import Dropout
from tensorflow.keras.callbacks import EarlyStopping, History, LearningRateScheduler
from tensorflow.keras import regularizers
#from keras.layers import Recurrent_Dropout
# from keras.utils.vis_utils import plot_model
from tensorflow.keras import backend as K
from tensorflow.keras import optimizers 
import time
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from vincenty import vincenty
import tensorflow as tf

import abc
import math

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import ops
from tensorflow.python.keras.utils import generic_utils
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import random_ops


def LSTM_model(x,y, input_dim,output_dim, seq_dim, batch_size, num_epochs, dropout, h2, learning_rate, l1_, l2_, nfr, decay_rate, momentum, decay_steps):
    start=time.time()
    regressor = Sequential()
    regressor.add(LSTM(units =h2,input_shape = (x.shape[1], input_dim), activation="tanh", recurrent_activation="softmax", use_bias=True, kernel_initializer="glorot_uniform", recurrent_initializer="orthogonal", recurrent_dropout=dropout, kernel_regularizer=regularizers.l1_l2(l1=l1_, l2=l2_) , return_sequences = True))
    regressor.add(Dropout(dropout))
    regressor.add(LSTM(units = h2, activation="tanh", recurrent_activation="sigmoid", use_bias=True, kernel_initializer="glorot_uniform", recurrent_initializer="orthogonal", recurrent_dropout=dropout, kernel_regularizer=regularizers.l1_l2(l1=l1_, l2=l2_)))
    adamax=optimizers.Adam(lr=learning_rate)#, beta_1=0.9, beta_2=0.99)     
    regressor.add(Dense(units = output_dim, activation='linear'))
    regressor.compile(optimizer = adamax, loss = 'mean_absolute_error')
    def exp_decay(epoch):
        lrate=learning_rate*np.exp(-decay_rate*epoch)
        return lrate
    lr_rt=LearningRateScheduler(exp_decay)
    loss_history=History()
    callbacks_list=[loss_history, lr_rt]    
    print(regressor.summary())
    history = regressor.fit(x, y, epochs = num_epochs, callbacks=callbacks_list, batch_size = batch_size, validation_split=0.005) #iterates 50 times    
    plt.plot(history.history['loss'], label='train')
    plt.legend()
    plt.savefig('LSTM_LOSS'+ str(nfr))
    plt.show()
    end=time.time()
    Computation_time=end-start
    return Computation_time, regressor


def seq_data_man(data, batch_size, seq_dim, input_dim, output_dim):
    X,Y,Z=data
    X=np.array(X)
    Y=np.array(Y)
    Z=np.array(Z)
    lx=len(X)
#    print(X.shape)
#    print(Y.shape)
#    print(Z.shape)
    x = []# input , 60 stocks price from previous
    y = []# stock prices for the next day
    z=[]
    for i in range(seq_dim,lx):#(timesteps, upperbound in the data)
        x.append(X[i-seq_dim:i, 0:(input_dim)])# append adds elements to the end of the list. i-60:i takes the values from i-60 to i
        y.append(Y[i-1, 0:output_dim])
        z.append(Z[i-1, 0:output_dim])
    x, y, z = np.array(x), np.array(y), np.array(z)
    return (x, y, z)


def sample_freq(data,sf):
    k=[]
    for i in range(sf,len(data),sf):
        s=data[i]
        k.append(s)
    return np.array(k)



def calib1(data1):
    locPred=np.array(data1)
    Acc1=locPred[:,16:17]
    Acc2=locPred[:,17:18]
    gyro1=locPred[:,14:15] 
    Brkpr=locPred[:,26:27] 
    Acc1_bias=np.mean(Acc1,axis=0)
    Acc2_bias=np.mean(Acc2,axis=0)
    gyro1_bias=np.mean(gyro1,axis=0)
    Brkpr_bias=np.mean(Brkpr,axis=0)
    return Acc1_bias, Acc2_bias, gyro1_bias, Brkpr_bias#, sa_bias, sa1_bias, sa2_bias

def normalise(T1, Tmx, Tmn,Z):
    return (Z*(T1-Tmn))/(Tmx-Tmn)

def inv_normalise(N, Tmx, Tmn, Z):
    return (N*(Tmx-Tmn)/Z)+Tmn
#def unnormalise(T1, Tmx, Tmn):
#    return (T1-np.mean(T1,axis=0))/np.std(T1,axis=0)

def absolute_disp(lat, long):
#    long=(long/60)*(-1)
#    lat=lat/60
    k=[]
    for i in range(1, len(lat)):
        lat1=lat[i-1]
        lat2=lat[i]
        lon1=long[i-1]
        lon2=long[i]
        kk=vincenty((lat1,lon1), (lat2, lon2))
#        kk=kk*1609344
        k.append(kk)
    return np.reshape(k,(len(k),1))


def Get_Cummulative(num):
    l=[]
    l.append(num[0])
    for i in range(len(num)-1):
        g=l[i]+num[i+1]
        l.append(g)
    return (np.array(l))#/1000


def sample_freq1(data,sf):
    k=[]
#    x=[]
#    print(len(data))
    for i in range(sf,len(data),sf):
#        print (i)
#        print(k.shape)
        k.append(data[i-sf:i])
    s=np.reshape(k,(len(k),sf,1))
#    print(s.shape)
    return s

def maxmin17(dat_to_norm,sf, Acc1_bias,gyro1_bias):
    locPred=np.array(dat_to_norm)
    # Acc1=locPred[:,16:17]
    dist1=locPred[:,2:3]
    dist2=locPred[:,3:4]



    rl=locPred[:,12:13]
    rr=locPred[:,13:14]


    gy=locPred[:,14:15] 
    dist1=locPred[:,2:3]
    dist2=locPred[:,3:4]
    
    rr=sample_freq(rr,sf)
    rl=sample_freq(rl,sf)    

    

    dist11=sample_freq(dist1,sf)
    dist21=sample_freq(dist2,sf)    
    dista=absolute_disp(dist11, dist21)
    dist=dista*1000  
    

    
    r1=np.mean((dist/rr[1:]), axis=0)
    r2=np.mean((dist/rl[1:]), axis=0)  
    
    rr_=rr*r1#r1#*0.0910
    rl_=rl*r2#r2#*0.0900
    disptl=rl_#integrate(rl,sf)
    disptr=rr_#integrate(rr,sf)        
    dispt1=(disptl+disptr)/2


    return max(rr), min(rr), r1, r2, max(dist), min(dist), max(gy), min(gy)


def data_process13tr(dat, seq_di, input_di, output_di,sf, Acc1_bia, Acc2_bia, gyro1_bia, batch_siz, amx, amn, r1, r2, dgmx, dgmn, gymx, gymn, Z, mode):
#    maxm=False
    if mode=='IDNN':
    
        Xin=np.zeros((1,input_di*seq_di))
        Yin=np.zeros((1,output_di))
        Zin=np.zeros((1,output_di))
        Ain=np.zeros((1,output_di))
    elif mode=='MLNN':
        Xin=np.zeros((1,input_di))
        Yin=np.zeros((1,output_di)) 
        Zin=np.zeros((1,output_di))
        Ain=np.zeros((1,output_di))
    else:
        Xin=np.zeros((1,seq_di, input_di))
        Yin=np.zeros((1,output_di))
        Zin=np.zeros((1,output_di))
        Ain=np.zeros((1,output_di))
        
    for i in range(len(dat)):
        locPred=np.array(dat[i])
        # Acc1=locPred[:,16:17]
        dist1=locPred[:,2:3]
        dist2=locPred[:,3:4]
 
        
        rl=locPred[:,12:13]
        rr=locPred[:,13:14]
        
        dist11=sample_freq(dist1,sf)
        dist21=sample_freq(dist2,sf)
        dista=absolute_disp(dist11, dist21)
        dist=dista*1000
        
        rr_=sample_freq(rr,sf)
        rl_=sample_freq(rl,sf)


        rr_=rr_*r1
        rl_=rl_*r2         


        dispt1=(rl_[1:]+rr_[1:])/2  
   
        rr=normalise(rr[1:], amx, amn, Z)
        rl=normalise(rl[1:], amx, amn, Z)


        
        rr=sample_freq1(rr,sf)
        rl=sample_freq1(rl,sf)




        xcn=np.concatenate((rr, rl), axis=2)  

        x=xcn[:len(dist)]
        y=dist-dispt1#[:len(dist]
        z=dispt1#[:len(dist)]
        
        

        if mode=='MLNN':

            Xin=np.concatenate((Xin,xcn[seq_di:]), axis=0) 
            Yin=np.concatenate((Yin,dist[seq_di:]), axis=0)
            Zin=np.concatenate((Zin,dispt1[seq_di:]), axis=0) 
            Ain=np.concatenate((Ain,dist[seq_di:]), axis=0) 
            Input_data=Xin[1:] 
            Output_data=Yin[1:]
            INS=Zin[1:]
        elif mode=='IDNN':
            x=np.reshape(x,(len(x),seq_di*input_di))
            Xin=np.concatenate((Xin,x), axis=0) 
            Yin=np.concatenate((Yin,y), axis=0)
            Zin=np.concatenate((Zin,z), axis=0) 
            Ain=np.concatenate((Ain,dist), axis=0) 
            Input_data=Xin[1:] 
            Output_data=Yin[1:]
            INS=Zin[1:]
        else:
            Xin=np.concatenate((Xin,x), axis=0) 
            Yin=np.concatenate((Yin,y), axis=0) 
            Zin=np.concatenate((Zin,z), axis=0)  
            Ain=np.concatenate((Ain,dist), axis=0) 
        Input_data=Xin[1:] 
        Output_data=Yin[1:]
        INS=Zin[1:]
        GPS=Ain[1:]
    return  GPS, INS,  Input_data, Output_data

def data_process13t(dat, seq_di, input_di, output_di,sf, Acc1_bia, Acc2_bia, gyro1_bia, batch_siz, amx, amn, r1, r2, dgmx, dgmn, gymx, gymn, Z, mode):
#    maxm=False
    if mode=='IDNN':
    
        Xin=np.zeros((1,input_di*seq_di))
        Yin=np.zeros((1,output_di))
        Zin=np.zeros((1,output_di))
        Ain=np.zeros((1,output_di))
    elif mode=='MLNN':
        Xin=np.zeros((1,input_di))
        Yin=np.zeros((1,output_di)) 
        Zin=np.zeros((1,output_di))
        Ain=np.zeros((1,output_di))
    else:
        Xin=np.zeros((1,seq_di, input_di))
        Yin=np.zeros((1,output_di))
        Zin=np.zeros((1,output_di))
        Ain=np.zeros((1,output_di))
        
        
    for i in range(len(dat)):
        locPred=np.array(dat[i])
        # Acc1=locPred[:,16:17]
        dist1=locPred[:,2:3]
        dist2=locPred[:,3:4]
        rl=locPred[:,12:13]
        rr=locPred[:,13:14]
        
        dist11=sample_freq(dist1,sf)
        dist21=sample_freq(dist2,sf)
        dista=absolute_disp(dist11, dist21)
        dist=dista*1000
        
        rr_=sample_freq(rr,sf)
        rl_=sample_freq(rl,sf)


        rr_=rr_*r1
        rl_=rl_*r2         


        dispt1=(rl_+rr_)/2  
   
        rr=normalise(rr[1:], amx, amn, Z)
        rl=normalise(rl[1:], amx, amn, Z)

        # ak=normalise(ak[1:], gymx, gymn, Z)
        
        rr=sample_freq1(rr,sf)
        rl=sample_freq1(rl,sf)

        # ak=sample_freq1(ak,sf)


        xcn=np.concatenate((rr[:len(dist)], rl[:len(dist)]), axis=2)  

        # dx3=xcn#dispt1
        x=xcn
        y=(dist-dispt1[1:])[:len(xcn)]
        z=(dispt1[1:])[:len(xcn)]
        

        # x,y,z=seq_data_man((dx3, dist-dispt1[1:], dispt1[1:]), batch_siz, seq_di, input_di, output_di)

        

        if mode=='MLNN':

            Xin=np.concatenate((Xin,xcn[seq_di:]), axis=0) 
            Yin=np.concatenate((Yin,dist[seq_di:]), axis=0)
            Zin=np.concatenate((Zin,dispt1[seq_di:]), axis=0) 
            Ain=np.concatenate((Ain,dist[seq_di:]), axis=0) 
            Input_data=Xin[1:] 
            Output_data=Yin[1:]
            INS=Zin[1:]
        elif mode=='IDNN':
            x=np.reshape(x,(len(x),seq_di*input_di))
            Xin=np.concatenate((Xin,x), axis=0) 
            Yin=np.concatenate((Yin,y), axis=0)
            Zin=np.concatenate((Zin,z), axis=0) 
            Ain=np.concatenate((Ain,dist), axis=0) 
            Input_data=Xin[1:] 
            Output_data=Yin[1:]
            INS=Zin[1:]
        else:
            Xin=np.concatenate((Xin,x), axis=0) 
            Yin=np.concatenate((Yin,y), axis=0) 
            Zin=np.concatenate((Zin,z), axis=0)  
            Ain=np.concatenate((Ain,dist), axis=0) 
        Input_data=Xin[1:] 
        Output_data=Yin[1:]
        INS=Zin[1:]
        GPS=Ain[1:]
    return  GPS, INS,  Input_data, Output_data

def get_graphv2(s,t, labels, labelt, labelx, labely, labeltitle,no):#s, ins, t=pred
    plt.plot(s, label=labels)#.format(**)+'INS')
    plt.ylabel(labely)
    plt.xlabel(labelx)
    plt.plot(t, label=labelt)
    plt.legend()
    plt.grid(b=True)
#    plt.ylim(0, )
#    plt.xlim(0,len(s))
    plt.title(labeltitle, fontdict={'fontsize': 15, 'fontweight': 500}, loc='center')
#    plt.savefig(labeltitle+ str(no))
    plt.show() 
def position_plot(gpy, gpx, iny, inx, py, px, outage_length):
    plt.plot(gpx[:outage_length],gpy[:outage_length], label='GPS Trajectory', c='black')#, lw=0.01)
    plt.plot(inx[:outage_length],iny[:outage_length], label='INS Estimation', c='red')#, lw=0.01)
    plt.plot(px[:outage_length],py[:outage_length], label='LSTM INS Solution', c='blue')#, lw=0.01)
#    plt.scatter([px,py], label='Proposed Solution', c='blue', lw=0.05)
    plt.xlabel('East displacement [m]')
    plt.ylabel('North displacement [m]')
#    plt.title('Position')
    plt.legend(loc='best')
    plt.axis('equal')
    plt.show()    
def get_graph(s,t, labels, labelt, labelx, labely, labeltitle,no):#s, ins, t=pred
    plt.plot(np.array([0,1,2,3,4,5,6,7,8,9,10]),np.concatenate((np.zeros((1,1)),s)), label=labels)#.format(**)+'INS')
    plt.ylabel(labely)
    plt.xlabel(labelx)
    plt.plot(np.array([0,1,2,3,4,5,6,7,8,9,10]),np.concatenate((np.zeros((1,1)),t)), label=labelt)#np.array([1,2,3,4,5,6,7,8,9,10])
    plt.legend()
    plt.grid(b=True)
#    plt.ylim(0, )
    plt.xlim(0,len(s)+1)
    print(len(labeltitle))
    plt.title(labeltitle, fontdict={'fontsize': 15, 'fontweight': 500}, loc='center')
    if len(labeltitle)==88:
        labeltitle=('Displacement CRSE for the Sharp Cornering and Successive Left and Right Turns Scenario')
    elif len(labeltitle)==87:
        labeltitle=('Displacement CAE for the Sharp Cornering and Successive Left and Right Turns Scenario')
    if len(labeltitle)==66:
        labeltitle=('Displacement CRSE for the Quick Changes in Acceleration Scenario')
    elif len(labeltitle)==65:
        labeltitle=('Displacement CAE for the Quick Changes in Acceleration Scenario')    
    plt.savefig(labeltitle+ str(no))
    plt.show()  
    
def get_crse(x,y,z,t, label, mode,no):#x=gps, y=ins, z=pred
    eins=np.sqrt(np.power(x,2))#np.sqrt(np.power(x-y,2))
    epred=np.sqrt(np.power(x-z,2))#np.sqrt(np.power(x-z,2))
    crse_ins=Get_Cummulative(eins[:t])
    crse_pred=Get_Cummulative(epred[:t])
#    get_graph(crse_ins, crse_pred, 'INS DR', mode, 'Time (s)', 'CRSE (m)', 'Displacement CRSE for the ' +label,no)
    return crse_ins[-1], crse_pred[-1]

def get_cae(x,y,z,t, label, mode,no):
    eins=x#x-y
    epred=x-z
    caeins=Get_Cummulative(eins[:t])
    caepred=Get_Cummulative(epred[:t])   
#    get_graph(caeins, caepred, 'INS DR', mode, 'Time (s)', 'CAE (m)', 'Displacement CAE for the ' +label,no)
    return np.sqrt(np.power(caeins[-1],2)), np.sqrt(np.power(caepred[-1],2))         
    
def get_aeps(x,y,z,t, label, mode,no):
    eins=np.sqrt(np.power(x,2))#np.sqrt(np.power(x-y,2))
    epred=np.sqrt(np.power(z,2))#np.sqrt(np.power(x-z,2))
    return (eins/t)[-1], (epred/t)[-1]
def get_aeir(x,y,z,t, label, mode,no):
    eins=x#x-y
    epred=x-z
    eir_ins=[]
    eir_ins.append(0)
    eir_pred=[]
    eir_pred.append(0)            
    for i in range (1,len(epred)):
        eir_ins.append(((eins[i]-eins[i-1])/eins[i-1])*100)
        eir_pred.append(((epred[i]-epred[i-1])/epred[i-1])*100) 
#    eir_ins=np.array(eir_ins)
#    eir_pred=np.array(eir_pred)
#    print(eir_ins.shape)
#    print(len(x))
    eir_ins=np.reshape(eir_ins,(len(x),1))
    eir_pred=np.reshape(eir_pred,(len(x),1))
    aeir_ins=np.mean(np.array(eir_ins))
    aeir_pred=np.mean(np.array(eir_pred))
    
    aeir_ins=np.sqrt(np.power(aeir_ins,2))
    aeir_pred=np.sqrt(np.power(aeir_pred,2))
#    print(aeir_ins.shape)
#    print(aeir_pred.shape)
    #get_graph(eir_ins, eir_pred, 'INS DR', mode, 'Time (s)', 'EIR (%)', 'EIR for the ' +label,no)
    return aeir_ins, aeir_pred  
def get_perfmetric(cet, cetp):
    mean=np.mean(cet, axis=0)
    mini=np.amin(cet, axis=0) 
    stdv=np.std(cet, axis=0)
    maxi=np.amax(cet, axis=0)
    meanp=np.mean(cetp, axis=0)
    minip=np.amin(cetp, axis=0) 
    stdvp=np.std(cetp, axis=0) 
    maxip=np.amax(cetp, axis=0)  
    perf_metr=np.concatenate((np.reshape(maxi,(1,1)),np.reshape(mini,(1,1)), np.reshape(mean,(1,1)), np.reshape(stdv,(1,1))), axis=1)#, np.reshape(np.sum(mabe[:(100/Ts)]),(1,1)), np.reshape(np.sum(mabe[:(300/Ts)]),(1,1)), np.reshape(np.sum(mabe[:(600/Ts)]),(1,1)), np.reshape(np.sum(mabe[:(900/Ts)]),(1,1))), axis=1)#, np.reshape(a90,(1,1)), np.reshape(a60,(1,1)), np.reshape(a30,(1,1)), np.reshape(a10),(1,1)), axis=1)   
    perf_metrp=np.concatenate((np.reshape(maxip,(1,1)),np.reshape(minip,(1,1)), np.reshape(meanp,(1,1)), np.reshape(stdvp,(1,1))), axis=1)#, np.reshape(np.sum(mabe[:(100/Ts)]),(1,1)), np.reshape(np.sum(mabe[:(300/Ts)]),(1,1)), np.reshape(np.sum(mabe[:(600/Ts)]),(1,1)), np.reshape(np.sum(mabe[:(900/Ts)]),(1,1))), axis=1)#, np.reshape(a90,(1,1)), np.reshape(a60,(1,1)), np.reshape(a30,(1,1)), np.reshape(a10),(1,1)), axis=1)   
    return perf_metr, perf_metrp
def get_dist_covrd(grth):
    mean=np.mean(grth, axis=0)
    mini=np.amin(grth, axis=0) 
    stdv=np.std(grth, axis=0)
    maxi=np.amax(grth, axis=0)
    perf_metr=np.concatenate((np.reshape(maxi,(1,1)),np.reshape(mini,(1,1)), np.reshape(mean,(1,1)), np.reshape(stdv,(1,1))), axis=1)#, np.reshape(np.sum(mabe[:(100/Ts)]),(1,1)), np.reshape(np.sum(mabe[:(300/Ts)]),(1,1)), np.reshape(np.sum(mabe[:(600/Ts)]),(1,1)), np.reshape(np.sum(mabe[:(900/Ts)]),(1,1))), axis=1)#, np.reshape(a90,(1,1)), np.reshape(a60,(1,1)), np.reshape(a30,(1,1)), np.reshape(a10),(1,1)), axis=1)   
    return perf_metr    
    
    
    
    
def predictcs(xthr,ythr, ithr, regress,  seq_dim,input_di, mode, Ts, mx, mn, Z, label, outage):
    xthr=xthr[:int((np.floor(len(xthr)/(int(outage/Ts)))*(int(outage/Ts))))]
    crset=np.zeros((int(len(xthr)/int(outage/Ts)),1))
    crsetdr=np.zeros((int(len(xthr)/int(outage/Ts)),1))
    caet=np.zeros((int(len(xthr)/int(outage/Ts)),1))
    caetdr=np.zeros((int(len(xthr)/int(outage/Ts)),1))
    aepst=np.zeros((int(len(xthr)/int(outage/Ts)),1))
    aepstdr=np.zeros((int(len(xthr)/int(outage/Ts)),1))
    aeirt=np.zeros((int(len(xthr)/int(outage/Ts)),1))
    aeirtdr=np.zeros((int(len(xthr)/int(outage/Ts)),1))
    dist_covrd=np.zeros((int(len(xthr)/int(outage/Ts)),1))
#    crset,crsetdr, caet, caetdr, aepst, aepstdr, aeirt, aeirtdr=cet, cet, cet, cet,cet, cet, cet, cet
#    cetp=np.zeros((int(len(xthr)/int(outage/Ts)),1))
    cccPred=np.zeros((int(outage/Ts),1)) 
    cccins=np.zeros((int(outage/Ts),1)) 
    cccgps=np.zeros((int(outage/Ts),1)) 
           
    for w in range (0,len(xthr),int(outage/Ts)):
        xtest=xthr[w:w+int(outage/Ts)]
#        ythr=inv_normalise(ythr, mx, mn, Z)        
        ytest=ythr[w:w+int(outage/Ts)]

        ins=ithr[w:w+int(outage/Ts)]
        yyTest1=np.array(ytest)
        
        newP=regress.predict(xtest)
#        newP=inv_normalise(newP3, mx, mn, Z)

        crse_ins, crse_pred=get_crse(yyTest1,ins,newP,int(outage/Ts), label, mode,w)
        cae_ins, cae_pred=get_cae(yyTest1,ins,newP,int(outage/Ts), label, mode,w)
        aeps_ins, aeps_pred=get_aeps(yyTest1,ins,newP,int(outage/Ts), label, mode,w)
        aeir_ins, aeir_pred=get_aeir(yyTest1,ins,newP,int(outage/Ts), label, mode,w)

        crset[int(w/(100/Ts)),0]=float(crse_pred)
        caet[int(w/(100/Ts)),0]=float(cae_pred)
        aepst[int(w/(100/Ts)),0]=float(aeps_pred)
        aeirt[int(w/(100/Ts)),0]=float(aeir_pred)
        crsetdr[int(w/(100/Ts)),0]=float(crse_ins)
        caetdr[int(w/(100/Ts)),0]=float(cae_ins)
        aepstdr[int(w/(100/Ts)),0]=float(aeps_ins)
        aeirtdr[int(w/(100/Ts)),0]=float(aeir_ins)
        dist_covrd[int(w/(100/Ts)),0]=float(sum(yyTest1))
        
        newPreds=np.concatenate((cccPred, np.reshape(newP,(len(newP),1))),axis=1)
        cccPred=newPreds

        INS=np.concatenate((cccins, np.reshape(ins,(len(newP),1))),axis=1)
        cccins=INS  

        GPS=np.concatenate((cccgps, np.reshape(ytest,(len(newP),1))),axis=1)
        cccgps=GPS   
    perf_metr_crsep, perf_metr_crsedr=get_perfmetric(crset, crsetdr)
    perf_metr_caep, perf_metr_caedr=get_perfmetric(caet, caetdr)
    perf_metr_aepsp, perf_metr_aepsdr=get_perfmetric(aepst, aepstdr)
    perf_metr_aeirp, perf_metr_aeirdr=get_perfmetric(aeirt, aeirtdr)
    dist_travld=get_dist_covrd(dist_covrd)
#    perf_metr_aeirp, perf_metr_aeirdr=perf_metr_crsep, perf_metr_crsedr

    return dist_travld, perf_metr_crsep, perf_metr_crsedr, perf_metr_caep, perf_metr_caedr, perf_metr_aepsp, perf_metr_aepsdr, perf_metr_aeirp, perf_metr_aeirdr, newPreds[:,1:], INS[:,1:], GPS[:,1:]#, cet, cetp#opt_runs, cm_runs, cm_runsPred 

def predclean(inp1, inp2):
    if inp2-inp1 >1.5:
        inp2=1.5
    elif inp2-inp1<-1.5:
        inp2=-1.5
    return inp2


def predictrl(xthr,ythr, ithr, regress, seq_dim, Ts, label): #function  helps to predict and analyse the predicted results 
    xthr=xthr[:int((np.floor(len(xthr)/(int(1200/Ts)))*(int(1200/Ts))))] # rounds the input dataset to  the nearest dataset. dataset fed into this function is already sampled at 1 hz
    cet=np.zeros((int(len(xthr)/int(1200/Ts)),4)) # creates an array for the storage of the error at the tenth second of each 10 seconds data sequence for the Neural Network prediction error
    cetp=np.zeros((int(len(xthr)/int(1200/Ts)),4)) # creates an array for the storage of the error at the tenth second of each 10 seconds data sequence for the INS physical model error
    ccc=np.zeros((int(1200/Ts),1)) #creates an array for the storage of the predicted errors
           
    for w in range (0,len(xthr),int(1200/Ts)):
        xtest=xthr[w:w+int(1200/Ts)] #breaks the input features of the test set into sequences of 10 seconds long and feeds it through in batches for prediction and analysis  
        
        ytest=ythr[w:w+int(1200/Ts)] #breaks the output of the test set into sequences of 10 seconds long and feeds it through in batches for analysis  
        ins=ithr[w:w+int(1200/Ts)]
        yyTest1=np.array(ytest)
        

        nne=ytest[0]
        newP1=[]
        aa=[]
        aaz=[]
        for k in range(0, len(xtest)-seq_dim):
            aa=xtest[k,:,:]
            aaz=aa.reshape(1,seq_dim,2)
            newP1=regress.predict(aaz)
            newP1=predclean(nne,newP1)
            nne=newP1
            for f in range(1,seq_dim+1):
                xtest[k+f,-f,0:1]=newP1
        newP=np.array(xtest[:,-1,0:1])
 

#        newP1=[]
#        aa=[]
#        aaz=[]
#        for k in range(0, len(xtest)-seq_dim):
#            aa=xtest[k,:,:]
#            aaz=aa.reshape(1,4,4)
#            newP1=regress.predict(aaz)
#            for f in range(1,4+1):
#                xtest[k+f,-f,-1]=newP1
#        newP=np.array(xtest[:,-1,0:1])       
        
#    
#    ypred1=un_normalise(d2max, d2min, Y_pred1)
#    
#    yTest1=un_normalise(d2max, d2min, aa2)  
#        newP=regress.predict(xtest)# predicts ins/gps position error
#        newP=np.array(newP)
   
        mabe=(newP-yyTest1) # computes the Neural network (NN) prediction  error
        mabe=np.sqrt(mabe**2)# computes the root mean square of the ins/gps error
        yom=ins-yyTest1
        ycr=np.sqrt(yom**2)#computes the root mean square of the ins/gps physical model error
        ycr=Get_Cummulative(ycr)# adds up the ins/gps errors within the 10 seconds also known as crse for the ins physical model

#        plt.plot(Get_Cummulative(ycr), label='INS Model Displacement Error')#.format(**)+'INS')
#        plt.ylabel('Displacement Error')
#        plt.xlabel('Time (secs)')
#        plt.plot(Get_Cummulative(mabe), label='LSTM Model Dispalcement Error')
#        plt.legend()
#        plt.title(label, fontdict={'fontsize': 15, 'fontweight': 500}, loc='center')
#        plt.show()                 
        crse= Get_Cummulative(mabe)# adds up the NN predicted errors within the 10 seconds
       
                     
        ce30p=ycr[:int(300/Ts)]
        cetp[int(w/(1200/Ts)),0]=float(ce30p[-1])#indexes the predicted position error at the tenth second

        ce60p=ycr[:int(600/Ts)]
        cetp[int(w/(1200/Ts)),1]=float(ce60p[-1])#indexes the predicted position error at the tenth second

        ce90p=ycr[:int(900/Ts)]
        cetp[int(w/(1200/Ts)),2]=float(ce90p[-1])#indexes the predicted position error at the tenth second

        ce120p=ycr[:int(1200/Ts)]
        cetp[int(w/(1200/Ts)),3]=float(ce120p[-1])#indexes the predicted position error at the tenth second

        

        ce30=crse[:int(300/Ts)]
        cet[int(w/(1200/Ts)),0]=float(ce30[-1]) #indexes the INS physical model error at the tenth second
        
        ce60=crse[:int(600/Ts)]
        cet[int(w/(1200/Ts)),1]=float(ce60[-1]) #indexes the INS physical model error at the tenth second

        ce90=crse[:int(900/Ts)]
        cet[int(w/(1200/Ts)),2]=float(ce90[-1]) #indexes the INS physical model error at the tenth second

        ce120=crse[:int(1200/Ts)]
        cet[int(w/(1200/Ts)),3]=float(ce120[-1]) #indexes the INS physical model error at the tenth second

        newPreds=np.concatenate((ccc, np.reshape(newP,(len(newP),1))),axis=1) #stores the predicted errors
        ccc=newPreds

    mean30=np.mean(cet[:,0], axis=0) #mean of the NN prediction crse across the 10 s sequences
    mini30=np.amin(cet[:,0], axis=0) # minimum of the NN prediction crse across the 10 s sequences
    stdv30=np.std(cet[:,0], axis=0) # standard deviation of the NN prediction crse across the 10 s sequences
    maxi30=np.amax(cet[:,0], axis=0) # maximum of NN  prediction crse across the 10 s sequences

    mean60=np.mean(cet[:,1], axis=0) #mean of the NN prediction crse across the 10 s sequences
    mini60=np.amin(cet[:,1], axis=0) # minimum of the NN prediction crse across the 10 s sequences
    stdv60=np.std(cet[:,1], axis=0) # standard deviation of the NN prediction crse across the 10 s sequences
    maxi60=np.amax(cet[:,1], axis=0) # maximum of NN  prediction crse across the 10 s sequences
    
    mean90=np.mean(cet[:,2], axis=0) #mean of the NN prediction crse across the 10 s sequences
    mini90=np.amin(cet[:,2], axis=0) # minimum of the NN prediction crse across the 10 s sequences
    stdv90=np.std(cet[:,2], axis=0) # standard deviation of the NN prediction crse across the 10 s sequences
    maxi90=np.amax(cet[:,2], axis=0) # maximum of NN  prediction crse across the 10 s sequences

    mean120=np.mean(cet[:,3], axis=0) #mean of the NN prediction crse across the 10 s sequences
    mini120=np.amin(cet[:,3], axis=0) # minimum of the NN prediction crse across the 10 s sequences
    stdv120=np.std(cet[:,3], axis=0) # standard deviation of the NN prediction crse across the 10 s sequences
    maxi120=np.amax(cet[:,3], axis=0) # maximum of NN  prediction crse across the 10 s sequences

    
    meanp30=np.mean(cetp[:,0], axis=0) #mean of the INS physical model crse across the 10 s sequences
    minip30=np.amin(cetp[:,0], axis=0) # minimum of the INS physical model crse across the 10 s sequences
    stdvp30=np.std(cetp[:,0], axis=0)   # standard deviation of the INS physical model crse across the 10 s sequences
    maxip30=np.amax(cetp[:,0], axis=0)  # maximum of INS physical model crse across the 10 s sequences  

    meanp60=np.mean(cetp[:,1], axis=0) #mean of the INS physical model crse across the 10 s sequences
    minip60=np.amin(cetp[:,1], axis=0) # minimum of the INS physical model crse across the 10 s sequences
    stdvp60=np.std(cetp[:,1], axis=0)   # standard deviation of the INS physical model crse across the 10 s sequences
    maxip60=np.amax(cetp[:,1], axis=0)  # maximum of INS physical model crse across the 10 s sequences 

    meanp90=np.mean(cetp[:,2], axis=0) #mean of the INS physical model crse across the 10 s sequences
    minip90=np.amin(cetp[:,2], axis=0) # minimum of the INS physical model crse across the 10 s sequences
    stdvp90=np.std(cetp[:,2], axis=0)   # standard deviation of the INS physical model crse across the 10 s sequences
    maxip90=np.amax(cetp[:,2], axis=0)  # maximum of INS physical model crse across the 10 s sequences 

    meanp120=np.mean(cetp[:,3], axis=0) #mean of the INS physical model crse across the 10 s sequences
    minip120=np.amin(cetp[:,3], axis=0) # minimum of the INS physical model crse across the 10 s sequences
    stdvp120=np.std(cetp[:,3], axis=0)   # standard deviation of the INS physical model crse across the 10 s sequences
    maxip120=np.amax(cetp[:,3], axis=0)  # maximum of INS physical model crse across the 10 s sequences     

    perf_metr30=np.concatenate((np.reshape(maxi30,(1,1)),np.reshape(mini30,(1,1)), np.reshape(mean30,(1,1)), np.reshape(stdv30,(1,1))), axis=1)#, 
    perf_metrp30=np.concatenate((np.reshape(maxip30,(1,1)),np.reshape(minip30,(1,1)), np.reshape(meanp30,(1,1)), np.reshape(stdvp30,(1,1))), axis=1)#

    perf_metr60=np.concatenate((np.reshape(maxi60,(1,1)),np.reshape(mini60,(1,1)), np.reshape(mean60,(1,1)), np.reshape(stdv60,(1,1))), axis=1)#, 
    perf_metrp60=np.concatenate((np.reshape(maxip60,(1,1)),np.reshape(minip60,(1,1)), np.reshape(meanp60,(1,1)), np.reshape(stdvp60,(1,1))), axis=1)#

    perf_metr90=np.concatenate((np.reshape(maxi90,(1,1)),np.reshape(mini90,(1,1)), np.reshape(mean90,(1,1)), np.reshape(stdv90,(1,1))), axis=1)#, 
    perf_metrp90=np.concatenate((np.reshape(maxip90,(1,1)),np.reshape(minip90,(1,1)), np.reshape(meanp90,(1,1)), np.reshape(stdvp90,(1,1))), axis=1)#

    perf_metr120=np.concatenate((np.reshape(maxi120,(1,1)),np.reshape(mini120,(1,1)), np.reshape(mean120,(1,1)), np.reshape(stdv120,(1,1))), axis=1)#, 
    perf_metrp120=np.concatenate((np.reshape(maxip120,(1,1)),np.reshape(minip120,(1,1)), np.reshape(meanp120,(1,1)), np.reshape(stdvp120,(1,1))), axis=1)#

    return perf_metr30, perf_metrp30, perf_metr60, perf_metrp60, perf_metr90, perf_metrp90, perf_metr120, perf_metrp120, newPreds[:,1:], cet, cetp



