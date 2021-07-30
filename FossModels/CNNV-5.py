# -*- coding: utf-8 -*-
"""
Created on Sat Mar 27 11:59:09 2021

@author: mikil
"""

import matplotlib.pyplot as plt
import numpy as np
import math

#global parameters
actWt=[.1,1]

def mul(num):
    p=1
    for m in num:
        p*=m
    # print('p=',p)
    return p

def im2col(image,kerRow,kerCol,st):

    """
    Returns:
      col: (new_h*new_w,kerRow*KerCol*imageChan) matrix,
            each column is a cube that will convolve with a filter
            new_h=(imageRow-kerRow)//st+1, new_w=(imagrCol-kerCol)//st+1
    """

    chan,row,col = image.shape
    new_h = (row-kerRow)//st+1
    new_w = (col-kerCol)//st+1
    col = np.zeros([new_h*new_w,chan*kerRow*kerCol])

    for i in range(new_h):
       for j in range(new_w):
           patch = image[...,i*st:i*st+kerRow,j*st:j*st+kerCol]
           col[i*new_w+j,:] = np.reshape(patch,-1)
    return col

def col2im(ma,h_prime,w_prime,C):
    """
      Args:
      ma: (h_prime*w_prime*w,F) matrix, each col should be reshaped to C*h_prime*w_prime when C>0, or h_prime*w_prime when C = 0
      h_prime: reshaped filter height
      w_prime: reshaped filter width
      C: reshaped filter channel, if 0, reshape the filter to 2D, Otherwise reshape it to 3D
    Returns:
      if C == 0: (F,h_prime,w_prime) matrix
      Otherwise: (F,C,h_prime,w_prime) matrix
    """
    F = ma.shape[1]
    if(C == 1):
        out = np.zeros([F,h_prime,w_prime])
        for i in range(F):
            col = ma[:,i]
            out[i,:,:] = np.reshape(col,(h_prime,w_prime))
    else:
        out = np.zeros([F,C,h_prime,w_prime])
        for i in range(F):
            col = ma[:,i]
            out[i,:,:] = np.reshape(col,(C,h_prime,w_prime))

    return out

#convolution, zero padded
def conv(image,filt,b,st=1):
   imChan,imRow,imCol=image.shape
   inChan,outChan,kerRow,kerCol=filt.shape
   r=(imRow-1)//st+1
   c=(imCol-1)//st+1
   out=np.zeros((outChan,r,c))
 
   imPad=np.pad(image,((0,0),((kerRow-1)//2,(kerRow-1)//2),
                        ((kerCol-1)//2,(kerCol-1)//2)))
   imMat=im2col(imPad,kerRow,kerCol,st)
   filtCol=np.reshape(filt,(outChan,-1))
   outCol=imMat.dot(filtCol.T)+b
   out=col2im(outCol,r,c,1)
   return out
    
#activation function
def act(layer):
    chan,row,col=layer.shape
    out=np.zeros((chan,row,col))
    
    for c in range(chan):
        for i in range(row):
            for j in range(col):
                if layer[c,i,j]>=0:
                    out[c,i,j]=actWt[1]*layer[c,i,j]
                else:
                    out[c,i,j]=actWt[0]*layer[c,i,j]
    return out

def actInv(layer):
    chan,row,col=layer.shape
    out=np.zeros((chan,row,col))
    
    for c in range(chan):
        for i in range(row):
            for j in range(col):
                if layer[c,i,j]>=0:
                    out[c,i,j]=layer[c,i,j]/actWt[1]
                else:
                    out[c,i,j]=layer[c,i,j]/actWt[0]
    return out

def actD(layer):
    chan,row,col=layer.shape
    out=np.zeros((chan,row,col))
    
    for c in range(chan):
        for i in range(row):
            for j in range(col):
                if layer[c,i,j]>=0:
                    out[c,i,j]=actWt[1]
                else:
                    out[c,i,j]=actWt[0]
    return out

def funVals(par,tNum):
    parInts=par.shape[0]
    # print(parDim)
    pInd=[int(h*parInts/tNum) for h in range(tNum)]
    return np.array([par[q] for q in pInd])

def trapInt(vals,ds):
    return ds/2*(np.sum(vals,axis=0)+np.sum(vals[1:-1],axis=0))

def trapIntArr(vals,ds):
    sMax=vals.shape[0]
    out=np.zeros((sMax,*vals.shape[1:]))
    out[1]=ds/2*(vals[0]+vals[1])
    vPre=vals[1]
    for s in range(1,sMax-1):
        out[s+1]=out[s]+ds/2*(vPre+vals[s+1])
        vPre=vals[s+1]
    return out

def loss(out,tar):
   # print('out Shape:',out.shape,', tar Shape:',tar.shape)
   l=np.einsum('oij,oij',(out-tar),(out-tar))
   # print(l)
   return l

def lossDer(out,tar):
    return 2*(out-tar)
 
def dloss(out,tar,uOut,duIn,duFilt,duBias):
   iBase,ic,oc,iDim1,iDim2=duIn.shape[0:5]
   fBase=duFilt.shape[0:2]
   fDim1,fDim2=duFilt.shape[4:6]
   bBase=duBias.shape[0]
   g=actD(uOut)
   dlossx=lossDer(out, tar)/mul(out.shape)
   # print('dlossx=',dlossx)
   # print('iBase=',iBase,', ic=',ic,', oc=',oc,', iDim=',(iDim1,iDim2))
   # print('fBase=',fBase,', fDim=',(fDim1,fDim2),', bBase=',bBase)
   dgIn=np.array([[[[[g*duIn[q,i,o,r,c] for c in range(iDim2)] for r in range(iDim1)]
                    for o in range(oc)] for i in range(ic)] for q in range(iBase)])
   dgFilt=np.array([[[[[[g*duFilt[l1,l2,o1,o2,r,c] for c in range(fDim2)]
                       for r in range(fDim1)] for o2 in range(oc)] for o1 in range(ic)]
                    for l2 in range(fBase[1])] for l1 in range(fBase[0])])
   dgBias=np.array([[g*duBias[m,o] for o in range(oc)] for m in range(bBase)])
   # print('dgBias=',dgBias)
   dlossI=np.einsum('xyz,qiorcxyz->qiorc',dlossx,dgIn)
   dlossF=np.einsum('xyz,kloprcxyz->kloprc',dlossx,dgFilt)
   dlossB=np.einsum('xyz,moxyz->mo',dlossx,dgBias)
   # print('dlossIn=',dlossIn)
   # print('dlossFilt=',dlossFilt)
   # print('dlossB=',dlossB)
   # print('dlossF shape: ',dlossF.shape)
   return dlossI,dlossF,dlossB

# def poolFwrd(x0,pDim,s):
#    ic,iRow,iCol=x0.shape
#    oRow=math.ceil(iRow/s)
#    oCol=math.ceil(iCol/s)
#    xOut=np.zeros(ic,oRow,oCol)
   
#    return xOut,indMax

class CLayer:
   
   def __init__(self,fWt,bVal,s,p):
      """
      Parameters
      ----------
      fWt: array (ic,oc,fH,fW)
      bVal: array (oc)
      s: stride value
      p: padding value
      -------
      ic: input channels
      oc: output channels
      fH: filter height
      fW: filter width
      """
      self.fWt=fWt
      self.bVal=bVal
      self.s,self.p=s,p
      
   def forward(self,inp):
      ic,inH,inW=inp.shape
      oc,fH,fW=self.fWt.shape[1:]
      outH=int((inH+2*self.p[0]-fH)/self.s)
      outC=int((inW+2*self.p[1]-fW)/self.s)
      
      inpP=np.pad(inp,((0,0),(0,0),(self.p,self.p),(self.p,self.p)),mode='constant')
      inMat=im2col(inpP,fH,fW,self.s)
      fCol=np.reshape(self.fWt,(oc,-1))
      outCol=inMat.dot(fCol.T)+self.bVal
      
      return col2im(outCol,outH,outC,1)
      
   

class VBlock: 
   
   
   def __init__(self,ic,oc,inDim,layDim,m0,tNum):
      """
      Parameters
      ----------
      ic : number of input channels
      oc : number of output channels
      inDim : (q0,inH,inW)
         dimensions for input filter
      layDim : (l0P,l0C,layH,layW)
         dimensions for interlayer filter
      tNum : number of t-subintervals
      -------
      ic: input channels
      oc: output channels
      q0: number of input filter basis functions
      inH, inW: input filter height, input filter width
      l0P,l0C: number of layer filter basis functions
      layH, layW: layer filter height, layer filter width
      m0: number of bias basis functions
      """
      self.iDim=(inDim[0],ic,oc,*inDim[1])
      self.lDim=(*layDim[0],oc,oc,*layDim[1])
      self.bDim=(m0,oc)
      self.tNum=tNum
      self.dt=1/tNum
      self.iWt=np.random.randn(*self.iDim)
      self.lWt=np.random.randn(*self.lDim)
      self.bWt=np.random.uniform(-1,1,self.bDim)
      
      self.x=None
      self.u=None
      self.dx=None
      self.dxIn=None
      
      iVals=self.constBasisVals(self.iWt,tNum)
      lVals=self.constBasisVals([self.constBasisVals(
         self.lWt[int(s*layDim[0,0]/tNum)],tNum) for s in range(tNum)])
      bVals=self.constBasisVals(self.bWt,tNum)
      b0=np.zero(self.bDim)
      
      self.iLayer=[CLayer(iVals[h],bVals[h],1,((inDim[1,0]-1)/2,(inDim[1,1]-1)/2))
              for h in range(tNum)]
      self.lLayer=[[CLayer(lVals[h,i],b0,1,((layDim[1,0]-1)/2,(layDim[1,1]-1)/2))
              for h in range(tNum)] for i in range(tNum)]
      
   def constBasisVals(self,baseWt,tNum):
      baseNum=baseWt.shape[0]
      baseInd=[int(h*baseNum/tNum) for h in range(tNum)]
      
      return np.array([baseWt[i] for i in baseInd])
   
   def forward(self,x0,train:bool):
      b0=np.zero(self.bDim)
      oc=self.iDim[2]
      uH,uW=x0.shape[1:]
      u0=actInv(x0)
      
      x=np.zeros((self.tNum,oc,uH,uW))
      u=np.zeros((self.tNum,oc,uH,uW))
      g=np.zeros((self.tNum,oc,uH,uW))
      
      x[0]=x0
      
      u[0]=self.iLayer[0].forward(u0)
      
      for t in range(self.tNum-1):
         zInt=np.array([self.lLayer[s,t].forward(x[s])])

      return None

def uSoln(u0,ds,inFilt,pFilt,b):
    inChan,row,col=u0.shape
    tMax,outChan=inFilt.shape[0],inFilt.shape[2]
    u=np.zeros((tMax,outChan,row,col))
    x=np.zeros((tMax,outChan,row,col))
    # y=np.zeros((tMax,outChan,row,col))
    bzero=np.zeros((outChan))
    # print('outChan=',outChan,', u shape:',u.shape)
    
    # print('inFilt shape=',inFilt.shape,', u0 shape:',u0.shape)
    u[0]=conv(u0,inFilt[0],bzero)
    #print('u0',u[0])
    for h in range(tMax-1):
        #print('h=',h)
        x[h]=act(u[h])
        zInt=np.array([conv(x[s],pFilt[s,h],b[h]/((h+1)*ds))
                       for s in range(h+1)])
        z=trapInt(zInt,ds)
        u[h+1]=z+conv(u0,inFilt[h+1],bzero)
        # print('actU=',actU)
        # print('y[',h,']=',y[h])
#        print('zInt=',zInt)
        # print('z=',z)
        # print('u[',h+1,']=',u[h+1])
    x[tMax-1]=act(u[tMax-1])
    return x,u

def uSolnD(u0,x,u,ds,N,inData,preData,biasData):
   row,col=u.shape[2:]
   iBase=inData[0]
   pBase=preData[0]
   bBase=biasData[0]
   iFilt=inData[1]
   pFilt=preData[1]
   # bias=biasData[1]
   #    print(iFilt.shape)
   tMax,iChan,oChan=iFilt.shape[0:3]
   iDim=iFilt.shape[3:]
   pDim=pFilt.shape[4:]
   # bDim=bias.shape[2:]
   qdt=T/iBase
   ldt=(T/pBase[0],T/pBase[1])
   mdt=T/bBase
   
   bzero=np.zeros((outChan))
    
   # print('tMax=',tMax,', inBase=',iBase,', curBase=',cBase,', preBase=',pBase,
   #        ', biasBase=',bBase,', inChan=',iChan,', outChan=',oChan)
   # print('inDim=',iDim,', curDim=',cDim,', preDim=',pDim,', biasDim=',bDim)
    #array of integrals of x
   g=np.array([actD(u[h]) for h in range(tMax)])
   # print('g Shape:',g.shape)
    #   print('uInt=',uInt)
   duIn=np.zeros((iBase,iChan,oChan,*iDim,oChan,row,col))
   duPre=np.zeros((*pBase,oChan,oChan,*pDim,oChan,row,col))
   duBias=np.zeros((bBase,oChan,oChan,row,col))
   dux0=np.zeros((iChan,row,col,oChan,row,col))
   for oc in range(oChan):
      # print('oc=',oc)
      for ic in range(iChan):
         # print('ic=',ic)
         for q in range(iBase):
            tStart=int(q*qdt/dt)
            # print('q=',q)
            for i in range(iDim[0]):
               for j in range(iDim[1]):
                  varPar=np.zeros((iBase,iChan,oChan,*iDim))
                  varIndex=(q,ic,oc,i,j)
                  varPar[varIndex]=1
                  var=funVals(varPar,tMax)
                  # print(var.shape)
                  
                  du=np.zeros((tMax,oChan,row,col))
                  # dy=np.zeros((tMax,oChan,row,col))
                  # print('u Shape:',u[0].shape)
                  du[tStart]=conv(u0,var[tStart],bzero)
                  # print('du shape:',du.shape,', tMax=',tMax)
                  #print('duOld=',duOld)
                  for h in range(tStart,tMax-1):
                     dzInt=np.zeros((h+1,oChan,row,col))
                     dzInt[tStart:h+1]=np.array([conv(g[s]*du[s],pFilt[s,h],bzero)
                                                 for s in range(tStart,h+1)])
                     dz=trapInt(dzInt,ds)
                     du[h+1]=dz+conv(u0,var[h+1],bzero)
                  duIn[varIndex]=du[tMax-1]
                  #print('duIn[',varIndex,']=',duIn[varIndex])
      for oc2 in range(oChan):
         # print('oc2=',oc2)
      #    for k in range(cBase):
      #       tStart=max(0,int(k*kdt/dt))
      #       print('k=',k)
      #       for i in range(cDim[0]):
      #          for j in range(cDim[1]):
      #             varPar=np.zeros((cBase,oChan,oChan,*cDim))
      #             varIndex=(k,oc,oc2,i,j)
      #             varPar[varIndex]=1
      #             var=funVals(varPar,tMax,ds*tMax)
               
      #             #print('k=',k,', tStart=',tStart,', j=',j,', var=',var)
      #             du=np.zeros((tMax,oChan,row,col))
      #             dy=np.zeros((tMax,oChan,row,col))
      #             for h in range(tStart,tMax-1):
      #                # print('du[',h,']=',du[h])
      #                # print('var[',h,']=',var[h])
      #                dy[h]=conv(dx[h]*du[h],cFilt[h])+conv(x[h],var[h])
      #                dzInt=np.zeros((h+1,oChan,row,col))
      #                #print('in cFilt dyShape:',dy.shape)
      #                dzInt[tStart:h+1]=np.array([conv(dy[s],pFilt[s,h])
      #                                            for s in range(tStart,h+1)])
      #                du[h+1]=trapInt(dzInt,ds)
      #             duCur[varIndex]=du[tMax-1]
      #             #print('duCur[',varIndex,']=',duCur[varIndex])
      #          #print('dx[tMax-1]=',dx[tMax-1])
         for l1 in range(pBase[0]):
            # print('l1=',l1,', pBase=',pBase)
            tStart1=max(0,int(l1*ldt[0]/ldt[1])-1)
            # tEnd1=min(pBase[1],int((l1+1)*ldt[0]/ldt[1])+2)
            # tStart1=0
            # tEnd1=pBase[1]
            
            for l2 in range(tStart1,pBase[1]):
               tStart2=max(0,int(l2*ldt[1]/dt))
               # print('tStart1=',tStart1,', tEnd1=',tEnd1)
               # print('l1=',l1,', l2=',l2,', ldt[1]=',ldt[1],
               #       ', dt=',dt)
               # print('tStartAdj',max(0,int(tStart1*ldt[1]/dt)),', tStart2=',tStart2)
               for i in range(pDim[0]):
                  for j in range(pDim[1]):
                     varPar=np.zeros((*pBase,oChan,oChan,*pDim))
                     varIndex=(l1,l2,oc,oc2,i,j)
                     varPar[varIndex]=1
                     var=np.array([funVals(varPar[min(pBase[0]-1,
                                                  int(s*dt*pBase[0]/T))],
                                           tMax) for s in range(tMax)])
                     # print('var=',var)
                     du=np.zeros((tMax,oChan,row,col))
                     # dy1=np.zeros((tMax,oChan,row,col))
                     # dy2=np.zeros((tMax,oChan,row,col))
                     for h in range(tStart2,tMax-1):
                        # dy1[h]=conv(dx[h]*du[h],cFilt[h])
                        # dy2[h]=conv(x[h],cFilt[h])
                        dzInt=np.zeros((h+1,oChan,row,col))
                        # print('du[',h,']=',du[h])
                        dzInt[tStart2:h+1]=np.array([conv(g[s]*du[s],pFilt[s,h],bzero)
                                                    +conv(x[s],var[s,h],bzero)
                                                    for s in range(tStart2,h+1)])
                        # print('dzInt=',np.transpose(dzInt))
                        du[h+1]=trapInt(dzInt,ds)
                        # print('du[',h+1,']=',du[h+1],', var[:,',h,']=',np.transpose(var[:,h]))
                     duPre[varIndex]=du[tMax-1]
                     # print('duPre[',*varIndex,']=',duPre[varIndex])
      for m in range(bBase):
         tStart=max(0,int(m*mdt/dt)-1)
         # print('m=',m)
         varPar=np.zeros((bBase,oChan))
         varIndex=(m,oc)
         varPar[varIndex]=1
         var=funVals(varPar,tMax)
         du=np.zeros((tMax,oChan,row,col))
         # dy=np.zeros((tMax,oChan,row,col))
         # du[tStart]=var[tStart]
         # print('varStart=',var[tStart])
         for h in range(tStart,tMax-1):
            # dy[h]=conv(dx[h]*du[h],cFilt[h])
            # print('h=',h,', dy=',dy[h],', dx=',dx[h],', du=',du[h],
            #       ', cFilt',cFilt[h])
            dzInt=np.zeros((h+1,oChan,row,col))
            dzInt[tStart:h+1]=np.array([conv(g[s]*du[s],pFilt[s,h],bzero)
                                        +conv(x[s],pFilt[s,h],var[h]/((h+1)*ds))
                                        for s in range(tStart,h+1)])
            du[h+1]=trapInt(dzInt,ds)
         duBias[varIndex]=du[tMax-1]
         # print('duBias[',varIndex,']=',duBias[varIndex])
   for ic in range(iChan):
         for i in range(row):
            for j in range(col):
               varPar=np.zeros((iChan,row,col))
               varIndex=(ic,i,j)
               varPar[varIndex]=1
               varPar=actInv(varPar)
               # var=funVals(varPar,tMax,ds*tMax)
               # print(var.shape)
               
               du=np.zeros((tMax,oChan,row,col))
               # dy=np.zeros((tMax,oChan,row,col))
               # print('u Shape:',u[0].shape)
               du[0]=conv(varPar,iFilt[0],bzero)
               # print('du shape:',du.shape,', tMax=',tMax)
               #print('duOld=',duOld)
               for h in range(tMax-1):
                  dzInt=np.zeros((h+1,oChan,row,col))
                  dzInt[:h+1]=np.array([conv(g[s]*du[s],pFilt[s,h],bzero)
                                              for s in range(h+1)])
                  dz=trapInt(dzInt,ds)
                  du[h+1]=dz+conv(varPar,iFilt[h+1],bzero)
               # print('du Shape:',du[tMax-1].shape)
               dux0[varIndex]=du[tMax-1]
               # print('duIn[',varIndex,']=',duIn[varIndex]
   return duIn,duPre,duBias,dux0
                
                               

if __name__=="__main__":
   # imTest=np.array([np.arange(1,26).reshape((5,5))])
   # kTest=np.array([[[[1,2,-1],[3,4,0],[0,1,0]]],
   #                  [[[1,2,-1],[3,4,0],[0,1,0]]]])
    # print(imTest)
    # print(kTest)
    # convTest=conv(imTest,kTest,0)
    # print(convTest)
    
   inChan=1
   N=(5,5)
   T=1
   t0=10
   t1=t0+1
   tpt=np.linspace(0,T,t1)
   dt=T/t0
   outChan=1
   filtDim=(outChan,outChan,3,3)
   iFiltDim=(inChan,outChan,3,3)
   l0=(2,2) #num subint for prevFilt (arg1,arg2)
   m0=1 #num subint for bias
   q0=1 #num subInt for input filter
 
    #initialize parameters
    # xIn=np.random.uniform(-1,1,(inChan,*N))
    # cFiltPar=np.random.randn(k0,*cFiltDim)
   pFiltPar=np.random.randn(*l0,*filtDim)
   # print('pFiltPar=',pFiltPar)
   biasPar=np.random.uniform(-1,1,(m0,outChan))
   iFiltPar=np.random.randn(q0,*iFiltDim)
    
    # xIn=np.full((inChan,*N),1)
   # xIn=np.array([[[(i+ic)/(j+1)*.4 for j in range(N[1])] for i in range(N[0])]
   #                for ic in range(inChan)])
   # xTar=np.full((outChan,*N),1)
   # pFiltPar=np.array([[[[[[(l1+1)*(i+j)/(1+o1+o2+l2)*.3 for i in range(filtDim[3])]
   #                      for j in range(filtDim[2])] for o2 in range(filtDim[1])]
   #                    for o1 in range(filtDim[0])] for l2 in range(l0[1])]
   #                    for l1 in range(l0[0])])
    # filtPar=np.full((*l0,*filtDim),.2)
   # biasPar=np.full((m0,outChan),-.1)
   # iFiltPar=np.full((q0,*iFiltDim),.1)
   # parTotal=0
   # for l1 in range(l0[1]):
   #    lEnd=min(l0[0],int(l1*T/l0[0])+1)
      
      
   #          #print('l1=',l1,', pBase=',pBase)
   #          tStart1=max(0,int(l1*ldt[0]/dt)-1)
   #          tEnd1=min(pBase[1],int((l1+1)*ldt[0]/dt)+1)
   #          for l2 in range(tStart1,tEnd1):
   #             tStart2=max(0,int(l2*ldt[1]/dt)-1)
   #             # print('l1=',l1,', l2=',l2,', tStart1=',tStart1,', tStart2=',tStart2)
   #             for i in range(pDim[0]):
   #                for j in range(pDim[1]):
    
   parTotal=(((outChan**2)*mul(l0)*mul(filtDim[2:]))
              +(outChan*m0)+(inChan*outChan*mul(iFiltDim[2:])*q0))
   print('Number of parameters <',parTotal)
    
    
   pFilt=np.array([funVals(pFiltPar[min(l0[0]-1,int(s*dt*l0[0]/T))],t1)
                    for s in range(t1)])
   biasVec=funVals(biasPar,t1)
   iFilt=funVals(iFiltPar,t1)
    
   trainSize=2
   testSize=0
   noise=.1
   trainOut=np.zeros((trainSize,outChan,*N))
   trainIn=np.zeros((trainSize,inChan,*N))
   rCon=15
   cCon=10
   for elem in range(trainSize):
      actNodes=np.random.randint(0,mul(N))
      newElemOut=np.zeros((outChan,*N))
      newElemIn=np.zeros((inChan,*N))
      for a in range(actNodes):
         i=np.random.randint(0,N[0])
         j=np.random.randint(0,N[1])
         for oc in range(outChan):
            newElemOut[oc,i,j]=1
         for ic in range(inChan):
            newElemIn[ic,:]+=np.array([[math.exp(-rCon*(r-i)**2)
                                        *math.exp(-cCon*(c-j)**2)
                                   for c in range(N[1])] for r in range(N[0])])
      for r in range(N[0]):
         for c in range(N[1]):
            for ic in range(inChan):
               newElemIn[ic,r,c]=max(0,min(np.random.uniform(-noise,noise)
                                           +min(newElemIn[ic,r,c],1),1))
      trainOut[elem]=newElemOut
      trainIn[elem]=newElemIn
   # print('trainIn=',trainIn)
   # print('trainOut=',trainOut)
            
   # trainOutInd=[(np.random.randint(0,N[0]),np.random.randint(0,N[1]))
   #                                 for a in range(trainSize)]
   # trainIn=(np.array([[np.random.uniform(-.2,.5,N)*noise
   #                     +np.array([[math.exp(-.2*(i-trainOutInd[a][0])**2
   #                                          -.1*(j-trainOutInd[a][1])**2)
   #                                 for j in range(N[1])] for i in range(N[0])])
   #                     for ic in range(inChan)] for a in range(trainSize)]))
   # trainOut=np.zeros((trainSize,outChan,*N))
   # # print('trainOut Shape:',trainOut.shape)
   # for a in range(trainSize):
   #    for oc in range(outChan):
   #       trainOut[a,oc][trainOutInd[a][0],trainOutInd[a][1]]=1
   #       # print('trainOutInd[',a,']=',trainOutInd[a],', trainOut:',trainOut[a])
   #    # trainOut[a,:,trainOutInd[a][0],trainOut[a[1]]]=np.full((outChan),1)
   #    # print('trainOut=',trainOut[a])
   dispInt=20
   repeat=3
   lrnRate=.3/parTotal
   decay=40
   mom=.01/parTotal
   mem=4
   # dliFiltList=[]
   # dlpFiltList=[]
   # dlBiasList=[]
   dliFiltOld=np.zeros((q0,*iFiltDim))
   dlpFiltOld=np.zeros((*l0,*filtDim))
   dlBiasOld=np.zeros((m0,outChan))
   # dliFiltList=[dliFiltOld for d in range(mem)]
   # dlpFiltList=[dlpFiltOld for d in range(mem)]
   # dlBiasList=[dlBiasOld for d in range(mem)]
   lossTot=0.0
   lossCur=0.0
   iterNum=[]
   lossHist=[]
   iterAveNum=[]
   lossAveHist=[]
#   plt.show()
   # plt.ion()
   # fig,ax = plt.subplots(figsize=(150,100))
   # # plt.xlim(0,100) 
   # # plt.ylim(0,10)
   # # axes.set_xlim(0, 100)
   # # axes.set_ylim(0, +20)

   # lossPlot, = ax.plot(iterNum,lossHist)

   # plt.ylim([0,15])
   # plt.show()
   # plt.ion() ## Note this correction
   fig=plt.figure()
   maxLoss=1.0
   plt.axis([0,trainSize,0,maxLoss])
   plt.show()
   # ax.set(xlim=(0, 100), ylim=(0,15))
   for h1 in range(trainSize):
      x0=trainIn[h1]
      x1=trainOut[h1]
      iFiltParOld=iFiltPar
      pFiltParOld=pFiltPar
      biasParOld=biasPar
      for h in range(repeat):
 
         # print('xIn=',x0)
         # print('xOut=',x1)
         uIn=actInv(x0)
         
         #NN output
         xOut,uOut=uSoln(uIn,dt,iFilt,pFilt,biasVec)
         
       #print("uSoln=",uSoln)
       
       #print(len(uSoln),len(tPt))
       #RVal=RFunVal(uSoln,2)
       
       #print(RFunVal)
       #plt.plot(tPt,uSoln[:,0])
         duiFilt,dupFilt,duBias,duxIn=uSolnD(uIn,xOut,uOut,dt,N,
                                             (q0,iFilt),(l0,pFilt),(m0,biasVec))
         # print('duXin=',duxIn,'***')
         dliFilt,dlpFilt,dlBias=dloss(xOut[-1],x1, uOut[-1], duiFilt, dupFilt, duBias)
         # print('dliFilt Shape:',dliFilt.shape,', dlpFilt Shape:',dlpFilt.shape)
         # print('dlpFilt=',dlpFilt)
         lossCur=loss(xOut[-1],x1)
 
         if h==0:
            # print("*** pre-update ",h," ***")
            # print('xIn=',x0)
            # print('xOut=',xOut[-1])
            print('** iter ',h1,' **')
            print("inital loss=",math.sqrt(lossCur/mul(N)))
            # print('trainOut=',x1)
            # print('diff=',x1-xOut[-1])
         if h==repeat-1:
            # print("*** pre-update ",h," ***")
            # print('xIn=',x0)
            # print('xOut=',xOut[-1])
            print("post loss=",math.sqrt(lossCur/mul(N)))
            # print('trainOut=',x1)
            # print('diff=',x1-xOut[-1])
            if math.sqrt(lossCur/mul(N))>maxLoss:
               maxLoss=math.sqrt(lossCur/mul(N))
            lossHist.append(math.sqrt(lossCur/mul(N)))
            iterNum.append(h1)
            # print(lossHist,iterNum)
            plt.axis([0,trainSize,0,maxLoss])
            plt.scatter(iterNum,lossHist,s=2);
            plt.plot(iterAveNum,lossAveHist,'r-',linewidth=1)
            # plt.plot(iterNum,lossHist)
            # plt.show()
            # lossPlot.set_xdata(iterNum)
            # lossPlot.set_ydata(lossHist)
            # fig.canvas.draw()
            plt.show()
            # plt.pause(.0001)
         if not np.isnan(lossCur):
            lossTot+=math.sqrt(lossCur/mul(N))
            # print(pFiltPar.shape,pFiltParOld.shape,dlpFilt.shape,dlpFiltOld.shape)
            iFiltPar-=(lrnRate*dliFilt/(1+decay*lrnRate*math.log(h1+1,2))
                       -mom*dliFiltOld/(1+decay*lrnRate*(math.log(h1+1,2))**2))
            pFiltPar-=(lrnRate*dlpFilt/(1+decay*lrnRate*math.log(h1+1,2))
                       -mom*dlpFiltOld/(1+decay*lrnRate*(math.log(h1+1,2))**2))
            biasPar-=(lrnRate*dlBias/(1+decay*lrnRate*math.log(h1+1,2))
                      -mom*dlBiasOld/(1+decay*lrnRate*(math.log(h1+1,2))**2))
            # dliFiltList.pop(0)
            # dlpFiltList.pop(0)
            # dlBiasList.pop(0)
            # dliFiltList.append(dliFilt)
            # dlpFiltList.append(dlpFilt)
            # dlBiasList.append(dlBias)
            # iFiltPar-=(lrnRate*sum(dliFiltList)/mem)/(1
            #                                             +decay*lrnRate*math.log(h1+1,2))
            # pFiltPar-=(lrnRate*sum(dlpFiltList)/mem)/(1
            #                                             +decay*lrnRate*math.log(h1+1,2))
            # biasPar-=(lrnRate*sum(dlBiasList)/mem)/(1
            #                                          +decay*lrnRate*math.log(h1+1,2))
            duiFiltOld=duiFilt
            dupFiltOld=dupFilt
            duBiasOld=duBias
            
            pFilt=np.array([funVals(pFiltPar[min(l0[0]-1,int(s*dt*l0[0]/T))],t1)
                            for s in range(t1)])
            biasVec=funVals(biasPar,t1)
            iFilt=funVals(iFiltPar,t1)
         else:
            print('**** overflow: h1=',h1,', h=',h,
                  ', xIn=',x0,', xOut=',xOut[-1],' ****')
            print('iFilt=',iFiltPar,', iFiltParOld=', iFiltParOld)
            print('pFilt=',pFiltPar,', iFiltParOld=', pFiltParOld)
            print('biasPar=',biasPar,', biasParOld=', biasParOld)
            lossTot+=math.sqrt(lossCur/mul(N))
            iFiltPar=iFiltParOld
            pFiltPar=pFiltParOld
            biasPar=biasParOld
            # duiFiltOld=duiFilt
            # dupFiltOld=dupFilt
            # duBiasOld=duBias
            pFilt=np.array([funVals(pFiltPar[min(l0[0]-1,int(s*dt*l0[0]/T))],t1)
                            for s in range(t1)])
            biasVec=funVals(biasPar,t1)
            iFilt=funVals(iFiltPar,t1)
            break
            # iFiltPar=iFiltParOld
            # pFiltPar=pFiltParOld
            # biasPar=biasParOld
            # dliFilt=np.zeros((q0,*iFiltDim))
            # dlpFilt=np.zeros((*l0,*filtDim))
            # dlBias=np.zeros((m0,outChan))
            # pFilt=np.array([funVals(pFiltPar[min(l0[0]-1,int(s*dt*l0[0]/T))],t1,T)
            #                 for s in range(t1)])
            # biasVec=funVals(biasPar,t1,T)
            # iFilt=funVals(iFiltPar,t1,T)
            # break
      if h1%dispInt==0:
         with np.printoptions(precision=2, suppress=True):
            print("**** update ****")
            # print('xIn=',x0)
            # print('xOut=',xOut[-1])
            iterAveNum.append(h1)
            lossAveHist.append(lossTot/(repeat*dispInt))
            # plt.axis([0,1000,0,10])
            # plt.scatter(iterNum,lossHist,s=2);
            # plt.plot(iterAveNum,lossAveHist)
            print("loss average=",lossTot/(repeat*dispInt))
            print('trainIn=\n',trainIn[h1])
            print('trainOut=\n',trainOut[h1])
            print('NN Out=\n',xOut[-1])
            print('diff=\n',trainOut[h1]-xOut[-1])
            lossTot=0.0
   # print('iFiltPar=',iFiltPar)
   # print('pFiltPar=',pFiltPar)
   # print('biasPar=',biasPar)

       
