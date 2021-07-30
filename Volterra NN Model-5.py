#import scipy as sp
import matplotlib.pyplot as plt
import multiprocessing as mp
import numpy as np
import math

#from scipy import interpolate

# Volterra Model with Continuous Piecewise Linear Parameter Functions

#global model parameters
N=6
T=1
k0=2
l0=3
m0=2
t0=4
p=3 #convolution matrix size
actPos=1
actNeg=.05

k1=k0+1


#mesh spacing
dt=T/t0
kdt=T/k0
ldt=T/l0
mdt=T/m0

kPt=np.linspace(0,T,k0+1)
lPt=np.linspace(0,T,l0+1)
mPt=np.linspace(0,T,m0+1)
tPt=np.linspace(0,T,t0+1)

#parameter initialization
mu=np.random.uniform(-1,1,(k0+1,l0+1,N,N))
gamma=np.random.uniform(-1,1,(m0+1,N))
#mu=np.array([[[[(j+1+4)*math.cos((i+1)*k*3.14159/4) for j in range(0,N)]
#                for i in range (0,N)]
#               for l in range(0,l0+1)] for k in range(0,k0+1)])
#gamma=np.array([[(2*i-1)*m/10 for i in range(0,N)] for m in range(0,m0+1)])
#mu=np.full((k0+1,l0+1,N,N),-2)
#gamma=np.full((m0+1,N),1)

#convolution matrix parameters
mu=np.random.uniform(-1,1())

#print("mu=",mu)


#functions
def loss(out,tar):
    return np.dot((out-tar),(out-tar))

def lossDer(out,tar):
    return 2*(out-tar)

def dlossHat(uOut,uTar,dumu,dugamma):
    dlossmu=np.zeros((k0+1,l0+1,N,N))
    dlossgamma=np.zeros((m0+1,N))
    dlossHatVal=lossDer(uOut, uTar)
    for i in range(N):
        for j in range(N):
            for k in range(k0+1):
                for l in range(l0+1):
                    dlossmu[k,l,i,j]=np.dot(dlossHatVal,dumu[k,l,i,j])
        for m in range(m0+1):
            dlossgamma[m,i]=np.dot(dlossHatVal,dugamma[m,i])
    return dlossmu,dlossgamma

def dloss(xOut,xTar,uOut,dumu,dugamma):
    dlossmu=np.zeros((k0+1,l0+1,N,N))
    dlossgamma=np.zeros((m0+1,N))
    dlossDerVal=lossDer(xOut,xTar)
    guVal=np.array([actFDer(uOut[i]) for i in range(N)])
    # for i in range(0,N):
    #     for j in range(0,N):
    #         for k in range(0,k0+1):
    #             for l in range(0,l0+1):
    #                 dlossmu[k,l,i,j]=np.dot(dlossDerVal,
    #                                         [guVal[i1]*dumu[k,l,i,j,i1]
    #                                          for i1 in range(0,N)])
    #     for m in range(0,m0+1):
    #         dlossgamma[m,i]=np.dot(dlossDerVal,
    #                                [guVal[i1]*dugamma[m,i,i1]
    #                                 for i1 in range(0,N)])
    dlossmu=np.apply_along_axis(
        np.dot, 4,np.apply_along_axis(
            np.multiply,4,dumu,guVal),dlossDerVal)
    dlossgamma=np.apply_along_axis(
        np.dot,2,np.apply_along_axis(
            np.multiply,2,dugamma,guVal),dlossDerVal)
    # print(dlossmu-np.apply_along_axis(np.dot, 4,
    #                                   np.apply_along_axis(
    #                                       np.multiply,4,dumu,guVal),
    #                                   dlossDerVal))
    return dlossmu,dlossgamma
    
def baseEl(width,h,t):
    v=t-h*width
    if abs(v)>width:
        return 0
    elif v>=0:
        return (1-v/width)
    return(1+v/width)
    
def alpha(tVal):
    return np.exp(-tVal)

def bias(t,gamVal):
    m=math.floor(t/mdt)
    if m<m0:
        return(baseEl(mdt,m,t)*gamVal[m]+baseEl(mdt, m+1, t)*gamVal[m+1])
    return(baseEl(mdt,m,t)*gamVal[m])

def rho(t,uIn,gamVal):
    h=math.floor(t/mdt)
#    print("t=",t,", t/mdt=",t/mdt,", h=",h)
    biasVal=(h-t/mdt+1)*gamVal[h]
    if h<m0:
        biasVal+=(t/mdt-h)*gamVal[h+1]
#    print("bias=",biasVal,", gamma=",gamVal)
    return (alpha(t)*uIn+biasVal)

def actF(uVal):
    if uVal>=0:
        return actPos*uVal
    return actNeg*uVal

def actFInv(xVal):
    if xVal>=0:
        return 1/actPos*xVal
    return 1/actNeg*xVal

def actFDer(uVal):
    if uVal>=0:
        return actPos
    return actNeg

def simpInt(t,pts,ds,k,uVal):
    val=0.0
    nIntv=len(pts)-1
#    print(nIntv)
    for p in range(0,nIntv,2):
            val+=ds/3*(baseEl(kdt,k,t+pts[p])*actF(uVal[p])
                         +4*baseEl(kdt,k,t+pts[p+1])*actF(uVal[p+1])
                         +baseEl(kdt,k,t+pts[p+2])*actF(uVal[p+2]))
    return val
        
def simpIntCall(argList,valQ):
    valQ.put([argList[0],
              simpInt(argList[1][0],argList[1][1],argList[1][2],argList[1][3],
                      argList[1][4])])
    
#Simpson's rule to compute components of R at t with increments of .5dt
#sInt: number of parabolic segments to use per .5dt interval
def RFunVal(uArray,sInt=1):
#    q=mp.Queue()
    #construct array of t points with increments of .5dt
    ds=dt/sInt
    sPt=np.array([np.linspace(0,.5*dt,2*sInt+1),
                  np.linspace(.5*dt,dt,2*sInt+1)])
#    print("sInt=", sInt, len(sPt1),len(sPt2))
    # sPt=np.block([sPt1,sPt2])
    # tHalfPt=np.array([tPt,[(t+.5*dt) for t in tPt]])
    # tBothPt=np.ravel(tPt2.T)[0:-1]
    RArray=np.zeros((2*t0+1,k0+1,l0+1,N))
    uVal=np.array([[[uArray[t]*(1-s/dt)+uArray[t+1]*s/dt
                     for s in sPt[i]] for t in range(t0)] for i in range(2)])
    # uVal1=np.array([[uArray[t]*(1-s/dt)+uArray[t+1]*s/dt
    #                  for s in sPt1] for t in range(t0)])
    # uVal2=np.array([[uArray[t]*(1-s/dt)+uArray[t+1]*s/dt
    #                  for s in sPt2] for t in range(t0)])
#    uVal=np.array([uVal1,uVal2])
 
    curVal=np.zeros((2,k0+1,l0+1,N))
    for j in range(N):
        # print("j=",j)
        for h in range(t0):
            # print("***************")
            # print("h=",h)
            for k in range(max(0,math.floor(tPt[h]/kdt)-1),
                           min(k0+1,math.ceil(tPt[h]/kdt)+2)):
                for l in range(max(0,math.floor(tPt[h]/ldt)-1),
                               min(l0+1,math.ceil(tPt[h]/ldt)+2)):
                    ### use multiprocesses
                    # a=[[[[e,j],[tPt[h],sPt[e],ds,k,uVal[e,h,:,j]]] for j in range(N)]
                    #     for e in range(2)]
                    # proc=[[mp.Process(target=simpIntCall,args=(a[e][j],q))
                    #     for j in range(N)] for e in range(2)]
                    # for e in range(2):
                    #     for p in proc[e]:
                    #         p.start()
                    # # for s in range(2):
                    # #     for p in proc[s]:
                    # #         p.join()
                    # for e in range(2):
                    #     for j in range(N):
                    #         val=q.get()
                    #         print(val)
                    #         RArray[2*h+val[0][0]+1,k,l,val[0][1]]=baseEl(
                    #             ldt,l,(tPt[h+e]+tPt[h+1])/2)*val[1]
                    # use single process
                    # print("k=",k,", l=",l)
    
                        # curVal[0,k,l,j]+=simpInt(tPt[h],sPt[0],ds,k,uVal[0,h,:,j])
                        # curVal[1,k,l,j]+=simpInt(tPt[h],sPt[1],ds,k,uVal[1,h,:,j])
                        #integrate over first half subinterval
                        for subInt in range(0,2*sInt,2):
                            curVal[0,k,l,j]+=ds/3*(baseEl(kdt,k,tPt[h]+sPt[0,subInt])
                                                  *actF(uVal[0,h,subInt,j])
                                                  +4*baseEl(kdt,k,tPt[h]
                                                            +sPt[0,subInt+1])
                                                  *actF(uVal[0,h,subInt+1,j])
                                                  +baseEl(kdt,k,tPt[h]
                                                          +sPt[0,subInt+2])
                                                  *actF(uVal[0,h,subInt+2,j]))
                        RArray[2*h+1,k,l,j]=(
                            baseEl(ldt,l,(tPt[h]+tPt[h+1])/2)*curVal[0,k,l,j])
                        #integrate over second half subinterval
                        for subInt in range(0,2*sInt,2):
                            curVal[0,k,l,j]+=ds/3*(baseEl(kdt,k,tPt[h]+sPt[1,subInt])
                                                  *actF(uVal[1,h,subInt,j])
                                                  +4*baseEl(kdt,k,tPt[h]
                                                            +sPt[1,subInt+1])
                                                  *actF(uVal[1,h,subInt+1,j])
                                                  +baseEl(kdt,k,tPt[h]
                                                          +sPt[1,subInt+2])
                                                  *actF(uVal[1,h,subInt+2,j]))
                        RArray[2*h+2,k,l,j]=(
                            baseEl(ldt,l,tPt[h+1])*curVal[1,k,l,j])
    return RArray

def g(s,uVal,muVal):
    gArray=np.zeros((l0+1,N,N))
    for l in range(l0+1):
        for k in range(max(0,math.floor(s/kdt)-1),
                       min(l+2,k0+1,math.ceil(s/kdt)+2)):
            for j in range(N):
                actuVal=actF(uVal[j])
                for i in range(N):
                    gArray[l,i,j]+=muVal[k,l,i,j]*baseEl(kdt,k,s)*actuVal
    return gArray

def gDer(s,uVal,deruVal,muVal):
#    print("deruVal=",deruVal)
    gDuArray=np.zeros((l0+1,N,N))
    for l in range(l0+1):
        for k in range(max(0,math.floor(s/kdt)-1),
                       min(l+2,k0+1,math.ceil(s/kdt)+2)):
            for j in range(N):
                actDeruVal=actFDer(uVal[j])
                for i in range(N):
                    gDuArray[l,i,j]+=(muVal[k,l,i,j]*baseEl(kdt,k,s)
                                      *actDeruVal*deruVal[j])
#    print("gd=",gDuArray)
    return gDuArray

#Runge-Kutta solution to model parameter dependencies
###Only solution values at T are needed
###Uses R_{k,l,i,j}=R_{k,l,i,i}
def duRunKut(uArray,muVal,gamVal):
    yThetaArray=np.zeros((k0+1,l0+1,N,N,l0+1,N,N))
    uThetaArray=np.zeros((k0+1,l0+1,N,N,N))
    yGammaArray=np.zeros((m0+1,N,l0+1,N,N))
    uGammaArray=np.zeros((m0+1,N,N))
    RArray=RFunVal(uArray)
    #print("RArray=",RArray)
    kTheta1=np.zeros((k0+1,l0+1,N,N,l0+1,N,N))
    
    kTheta2=np.zeros((k0+1,l0+1,N,N,l0+1,N,N))

    kTheta3=np.zeros((k0+1,l0+1,N,N,l0+1,N,N))
    
    kTheta4=np.zeros((k0+1,l0+1,N,N,l0+1,N,N))
    
    uTheta2=np.zeros((k0+1,l0+1,N,N,N))

    uTheta3=np.zeros((k0+1,l0+1,N,N,N))
    uTheta4=np.zeros((k0+1,l0+1,N,N,N))

#    print("*** duRunKut ***")
#    print("mu=",mu)
#    print("gamma=",gamma)
    for h in range(1,t0+1):
#        print("h=",h)
        psi23=np.array([baseEl(ldt,l,tPt[h]-.5*dt) for l in range(l0+1)])
        psi4=np.array([baseEl(ldt,l,tPt[h]) for l in range(l0+1)])
        bet23=np.array([baseEl(mdt,m,tPt[h]-.5*dt) for m in range(m0+1)])
        bet4=np.array([baseEl(mdt,m,tPt[h]) for m in range(m0+1)])
        uGamma2=np.zeros((m0+1,N,N))
        uGamma3=np.zeros((m0+1,N,N))
        uGamma4=np.zeros((m0+1,N,N))
        kGamma1=np.zeros((m0+1,N,l0+1,N,N))
        kGamma2=np.zeros((m0+1,N,l0+1,N,N))
        kGamma3=np.zeros((m0+1,N,l0+1,N,N))
        kGamma4=np.zeros((m0+1,N,l0+1,N,N))
        # uGamma2=np.zeros((m0+1,N,N))
        #kGamma1=np.zeros((m0+1,N,N))
        for i in range(N):
#            print("i=",i)
            uGamma2[:,i,i]=bet23
            uGamma3[:,i,i]=bet23
            uGamma4[:,i,i]=bet4
            uGammaArray[:,i,i]=bet4
            uTheta2[:,:,i,:,i]=RArray[2*h-1]
            uTheta3[:,:,i,:,i]=RArray[2*h-1]
            uTheta4[:,:,i,:,i]=RArray[2*h]
            uThetaArray[:,:,i,:,i]=RArray[2*h]
            kTheta1[:,:,i,:]=np.array([[[gDer(tPt[h]-1,uArray[h-1],
                                              uThetaArray[k,l,i,j],muVal)
                                         for j in range(N)]
                                         for l in range(l0+1)]
                                         for k in range(k0+1)])



            kGamma1[:,i]=np.array([gDer(tPt[h]-dt,uArray[h-1],
                                        uGammaArray[m,i],muVal)
                                   for m in range(m0+1)])
 #            for k in range(0,k0+1):
 #                for l in range(0,l0+1):    
 #                    for j in range(0,N):
 #                       uTheta2[k,l,i,j,i]=RArray[2*h-1,k,l,j]
 #                        # print("(h,k,l,i,j)=", [h,k,l,i,j],
 #                        #       " gDer=",gDer(tPt[h]-dt,uArray[h-1],
 #                        #          uThetaArray[k,l,i,j],muVal))
 #                        kTheta1[k,l,i,j]=gDer(tPt[h]-dt,uArray[h-1],
 #                                 uThetaArray[k,l,i,j],muVal)
 #           for m in range(0,m0+1):
                # print("m=",m)
                # print("uGamma2in=",uGamma2)
                # uGamma2[m,i]=np.zeros((N))
                # kGamma1[m,i]=np.zeros((N))
                # print("uGamma2res=",uGamma2)
                # print("bet23=",bet23)
                # uGamma2[m,i,i]=bet23[m]
                # print("uGamma2out=",uGamma2)
 #               kGamma1[m,i]=gDer(tPt[h]-dt,uArray[h-1],
 #                                       uGammaArray[m,i],muVal)
 #       print("new kTheta1=",kTheta1)
        for l in range(max(0,math.floor((tPt[h]-.5*dt)/ldt)-1),
            min(l0+1,math.ceil((tPt[h]-.5*dt)/ldt)+2)):
            for j in range(N):      
                uTheta2+=psi23[l]*(yThetaArray[:,:,:,:,l,:,j]
                                   +.5*dt*kTheta1[:,:,:,:,l,:,j])
                uGamma2+=psi23[l]*(yGammaArray[:,:,l,:,j]
                                   +.5*dt*kGamma1[:,:,l,:,j])
        kTheta2[:,:,i,:]=np.array([[[gDer(tPt[h]-.5*dt,
                                          .5*uArray[h-1]+.5*uArray[h],
                                          uTheta2[k,l,i,j],muVal)
                                     for j in range(N)]
                                     for l in range(l0+1)]
                                     for k in range(k0+1)])
        kGamma2[:,i]=np.array([gDer(tPt[h]-.5*dt,.5*uArray[h-1]+.5*uArray[h],
                                    uGamma2[m,i],muVal)
                               for m in range(m0+1)])
                #print("newuTheta2=",uTheta2)
#         for i in range(N):
#             for k in range(k0+1):
#                 for l in range(l0+1):    
#                     for j in range(N):
# #                       uTheta3[k,l,i,j,i]=RArray[2*h-1,k,l,j]
#                         kTheta2[k,l,i,j]=gDer(tPt[h]-.5*dt,
#                                               .5*uArray[h-1]+.5*uArray[h],
#                                  uTheta2[k,l,i,j],muVal)
            # for m in range(m0+1):
            #     # uGamma3[m,i]=np.zeros((N))
            #     # kGamma2[m,i]=np.zeros((N))
            #     # print("h=",h,", m=",m)
            #     # print("uArray=",uArray,", midpoint=",.5*uArray[h-1]+.5*uArray[h])
            #     # print("uGamma2=",uGamma2)
                
            #     kGamma2[m,i]=gDer(tPt[h]-.5*dt,.5*uArray[h-1]+.5*uArray[h],
            #                             uGamma2[m,i],muVal)
 #               print("kGamma2=",kGamma2)
        for l in range(max(0,math.floor((tPt[h]-.5*dt)/ldt)-1),
            min(l0+1,math.ceil((tPt[h]-.5*dt)/ldt)+2)):
            for j in range(N):      
                uTheta3+=psi23[l]*(yThetaArray[:,:,:,:,l,:,j]
                                   +.5*dt*kTheta2[:,:,:,:,l,:,j])
                uGamma3+=psi23[l]*(yGammaArray[:,:,l,:,j]
                                   +.5*dt*kGamma2[:,:,l,:,j])
        kTheta3[:,:,i,:]=np.array([[[gDer(tPt[h]-.5*dt,
                                          .5*uArray[h-1]+.5*uArray[h],
                                          uTheta3[k,l,i,j],muVal)
                                     for j in range(N)]
                                    for l in range(l0+1)]
                                   for k in range(k0+1)])
        kGamma3[:,i]=np.array([gDer(tPt[h]-.5*dt,.5*uArray[h-1]+.5*uArray[h],
                                    uGamma3[m,i],muVal)
                               for m in range(m0+1)])
 #        for i in range(N):
 #            for k in range(k0+1):
 #                for l in range(l0+1):    
 #                    for j in range(N):
 # #                       uTheta4[k,l,i,j,i]=RArray[2*h,k,l,j]
 #                        kTheta3[k,l,i,j]=gDer(tPt[h]-.5*dt,
 #                                              .5*uArray[h-1]+.5*uArray[h],
 #                                 uTheta3[k,l,i,j],muVal)
            # for m in range(m0+1):
            #     # uGamma4[m,i]=np.zeros((N))
            #     # kGamma3[m,i]=np.zeros((N))
                
            #     kGamma3[m,i]=gDer(tPt[h]-.5*dt,.5*uArray[h-1]+.5*uArray[h],
            #                             uGamma3[m,i],muVal)
#                print("kGamma3=",kGamma3)
        for l in range(max(0,math.floor(tPt[h]/ldt)-1),
            min(l0+1,math.ceil(tPt[h]/ldt)+2)):
            for j in range(N):      
                uTheta4+=psi4[l]*(yThetaArray[:,:,:,:,l,:,j]
                                  +dt*kTheta3[:,:,:,:,l,:,j])
                uGamma4+=psi4[l]*(yGammaArray[:,:,l,:,j]
                                   +dt*kGamma3[:,:,l,:,j])
        kTheta4[:,:,i,:]=np.array([[[gDer(tPt[h],uArray[h],
                                          uTheta4[k,l,i,j],muVal)
                                     for j in range(N)]
                                    for l in range(l0+1)]
                                   for k in range(k0+1)])
        kGamma4[:,i]=np.array([gDer(tPt[h],uArray[h],uGamma4[m,i],muVal)
                               for m in range(m0+1)])
#         for i in range(N):
#             for k in range(k0+1):
#                 for l in range(l0+1):    
#                     for j in range(N):
# #                        uThetaArray[k,l,i,j,i]=RArray[2*h,k,l,j]
#                         kTheta4[k,l,i,j]=gDer(tPt[h],uArray[h],
#                                  uTheta4[k,l,i,j],muVal)
#             for m in range(0,m0+1):
#                 # uGammaArray[m,i]=np.zeros((N))
#                 # kGamma4[m,i]=np.zeros((N))
                
#                 kGamma4[m,i]=gDer(tPt[h],uArray[h],
#                                         uGamma4[m,i],muVal)
        yThetaArray+=dt/6*(kTheta1+2*kTheta2+2*kTheta3+kTheta4)
        yGammaArray+=dt/6*(kGamma1+2*kGamma2+2*kGamma3+kGamma4)
#        print("yTheta=",yThetaArray[:,:,:,:,1,:,1])
        for l in range(max(0,math.floor(tPt[h]/ldt)-1),
            min(l0+1,math.ceil(tPt[h]/ldt)+2)):
            for j in range(N):
 #               print("[l,j]=",[l,j],", yGammaArray[l]=",yGammaArray[:,:,l,:,j])
                uThetaArray+=psi4[l]*yThetaArray[:,:,:,:,l,:,j]
                uGammaArray+=psi4[l]*yGammaArray[:,:,l,:,j]
#        print("kTheta1=",kTheta1)
#        print("kGamma1=",kGamma1)
#        print("uTheta2=",uTheta2)
#        print("uTheta3=",uTheta3)
#        print("uTheta4=",uTheta4)
#        print("uThetaArray=",uThetaArray)
#        print("uGammaArray=",uGammaArray)
    return uThetaArray,uGammaArray


#Runge-Kutta solution to model
def uRunKut(muVal,gamVal):
    yArray=np.zeros((len(tPt),l0+1,N,N))   
    uArray=np.zeros((len(tPt),N))
    uArray[0]=rho(0,u0,gamVal)
#    print("uArray=",uArray)
    h=1
    for t in tPt[1:]:
 #       print("uSoln t=",t)
        k1=g(t-dt,uArray[-1],muVal)
        rho23=rho(t-.5*dt,u0,gamVal)
        rho4=rho(t,u0,gamVal)
        psi23=np.array([baseEl(ldt,l,t-.5*dt) for l in range(l0+1)])
        psi4=np.array([baseEl(ldt,l,t) for l in range(l0+1)])
        u2=rho23
        u3=rho23
        u4=rho4
        uArray[h]=rho4
        for l in range(max(0,math.floor((t-.5*dt)/ldt)-1),
                           min(l0+1,math.ceil((t-.5*dt)/ldt)+2)):
            for j in range(N):            
                u2+=psi23[l]*(yArray[h-1,l,:,j]+.5*dt*k1[l,:,j])
        k2=g(t-.5*dt,u2,muVal)
        
        for l in range(max(0,math.floor((t-.5*dt)/ldt)-1),
                           min(l0+1,math.ceil((t-.5*dt)/ldt)+2)):
            for j in range(N):      
                u3+=psi23[l]*(yArray[h-1,l,:,j]+.5*dt*k2[l,:,j])
        k3=g(t-.5*dt,u3,muVal)
        
        for l in range(max(0,math.floor(t/ldt)-1),
                           min(l0+1,math.ceil(t/ldt)+2)):
            for j in range(N):      
                u4+=psi4[l]*(yArray[h-1,l,:,j]+dt*k3[l,:,j])
        k4=g(t,u4,muVal)
        yArray[h]=yArray[h-1]+1/6*dt*(k1+2*k2+2*k3+k4)
        
        for j in range (N):
            for l in range(l0+1):
                uArray[h]+=psi4[l]*yArray[h,l,:,j]
        h+=1
    return uArray


if __name__=="__main__":

    #training/testing set
    #objective: identify which coordinates are 1=nonnegative 0=negative
    trainSize=2
    #use identical training elements
    trainIn=np.full((trainSize,N),2)
    
    # trainIn=np.random.uniform(-1,1,(trainSize,N))
    trainOut=np.full((trainSize,N),1)
    
    
    for h in range(trainSize):
        for i in range(N):
            if trainIn[h,i]<0:
                trainOut[h,i]=0
    
    testSize=2
    #use identical test elements
    # testIn=np.full((testSize,N),2)
    
    testIn=np.random.uniform(-1,1,(testSize,N))
    testOut=np.full((testSize,N),1)
    for h in range(testSize):
        for i in range(N):
            if testIn[h,i]<0:
                testOut[h,i]=0
            
    #train
    print("***** train *******")
    dispInt=1
    for h in range(trainSize):
        lossTot=0.0
        x0=trainIn[h]
        x1=trainOut[h]
        u0=np.array([actFInv(x0[i]) for i in range(N)])
        u1=np.array([actFInv(x1[i]) for i in range(N)])
     #   print("[x0,x1]=[",x0,x1,"], [u0,u1]=[",u0,u1,"]")
    
        dlmuOld=np.zeros((k0+1,l0+1,N,N))
        dlgammaOld=np.zeros((m0+1,N))
    #U solution
        uSoln=uRunKut(mu,gamma)
    #print("uSoln=",uSoln)
    
    #print(len(uSoln),len(tPt))
    #RVal=RFunVal(uSoln,2)
    
    #print(RFunVal)
    #plt.plot(tPt,uSoln[:,0])
        dumuVal,dugammaVal=duRunKut(uSoln,mu,gamma)
        # print("duGamma=",dugammaVal)
        xOut=np.array([actF(uSoln[-1,i]) for i in range(N)])
        #dlmu,dlgamma=dlossHat(uSoln[-1],u1,dumuVal, dugammaVal)
        dlmu,dlgamma=dloss(xOut,x1,u1,dumuVal, dugammaVal)
        lossTot+=loss(xOut,x1)
        if h%dispInt==0:
            print("*** update ",h," ***")
            print("[xIn,uIn]=[",x0,u0,"]")
            print("[xOut,uOut]=[",xOut,uSoln[-1],"]")
            print("loss average=",lossTot/dispInt)
            lossTot=0.0
    # mu+=np.random.uniform(-.2,.2,(k0+1,l0+1,N,N))
    # gamma+=np.random.uniform(-.2,.2,(m0+1,N))
        mu-=.15*dlmu+.01*dlmuOld
        gamma-=.05*dlgamma+.01*dlgammaOld
    
        dlmuOld=dlmu
        dlgammaOld=dlgamma
        
    #test
    print("***** test *******")
    for h in range(testSize):
        x0=testIn[h]
    #    x1=testOut[h]
        u0=np.array([actFInv(x0[i]) for i in range(N)])
    #    u1=np.array([actFInv(x1) for i in range(0,N)])
        uSoln=uRunKut(mu,gamma)
    #    u=np.array([actFInv(x1[i]) for i in range(0,N)])
    #    print(uSoln)
        xSoln=np.array([[actF(uSoln[t,i]) for i in range(N)]
                        for t in range(0,t0+1)])
        plt.plot(tPt,xSoln)
    #    print(xSoln[1])
        print("*** element ",h," ***")
        print("[xIn,uIn]=[",x0,u0,"]")
        print("[xOut,uOut]=[",xSoln[-1],uSoln[-1],"]")
        print("loss=",loss(xSoln[-1],testOut[h]))




