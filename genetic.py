#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 17 12:17:08 2021

@author: douglas
"""
import numpy as np
import copy
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
import copy

class GeneticPrograming:
     
    def __init__(self, aridade, maxDeep,variables,functions,constantes,y):
         
         self.Aridade = aridade
         self.MaxDeep = maxDeep
         self.Variables = variables
         self.functions = functions
         self.Constantes = constantes
         
         self.out = y
         # used to generate matriz
         self.numberOfFunctions = len(functions)
         self.numberArg = 3
         self.numberCon = len(constantes)
         self.numberVar = len(variables)
         self.numberOut = len(y)
         # functions
         self.evalFunc = mean_absolute_error
         
    # to do         
    def crate_candidates(self):
        
        self.candidates = []
        self.solEval = []
        
        
        
        for n in range(self.num_cand):
            # possible solution
            c = np.nan
            flag = 0
            while( ~np.isfinite(np.sum(c)) | (flag < 3 ) ) :
                f = self.recCreation2()
                mat= []
                self.createMatrix(f,mat) # unstack list
                mat = np.array(mat)
                c = self.evaluateTree(mat) # calculates the result
                
                if np.isfinite(np.sum(c)):
                    temp = self.funcEval(c)
                    flag = mat.shape[0]
                
                
            self.candidates.append(mat)
            self.solEval.append(temp)
            
    def run(self,num_ind,num_iter):
        self.best = []
        self.pop = []
        
        self.num_iter = num_iter
        self.num_cand = num_ind
        
        self.mutation = 0.3 # probability of a mutation
        
        self.crate_candidates()
        
        self.best.append(np.min(np.array(self.solEval)))
        self.pop.append(np.mean(np.array(self.solEval)))
        
        #print(self.candidates )
        
        for n_iter in range(self.num_iter): # for to each generation
            if (n_iter%100 == 0):
                print('Generation', n_iter)
                
            for n_ind in range(self.num_cand):# to select make matting or mutation
                # mutation branch
                
                if np.random.uniform() < self.mutation:
                    c = np.nan
                    while(~np.isfinite(np.sum(c))):
                        posSol = self.mutData(copy.copy(self.candidates[n_ind]))
                        
                        try:
                            c = self.evaluateTree(posSol)       
                        except:
                            c = np.nan
                            
                        if np.isfinite(np.sum(c)):
                            temp = self.funcEval(c)
                    # save new solution
                    if (temp < self.solEval[n_ind]):
                        self.solEval[n_ind] = temp 
                        self.candidates[n_ind] = posSol 
                else:
                    # cross 
                    luck = np.log(1/np.array(self.solEval))
                    luck = ( luck-np.min(luck) )/ (np.max(luck)- np.min(luck))
                    totluck = np.sum(luck)
                    luck = luck/totluck
                    draw = np.random.uniform()
                    #print(luck)
                    n_i = 0
                    sumValues = luck[n_i]
                    while(draw > sumValues):
                        n_i+=1
                        sumValues += luck[n_i]
                        
                    mat1 = copy.copy(self.candidates[n_ind] )
                    mat2 = copy.copy(self.candidates[n_i] )
                    [sol1,sol2] = self.crossData(mat1,mat2)
                    
                    # solving erorr in evaluating new solution
                    try:
                        c = self.evaluateTree(sol1)       
                    except:
                        c = np.nan
                        
                    if np.isfinite(np.sum(c)):
                        temp = self.funcEval(c)
                        sol = sol1
                    else:
                        try:
                            c = self.evaluateTree(sol2) 
                        except:
                            c = np.nan   
                        
                        if np.isfinite(np.sum(c)):
                            temp = self.funcEval(c)
                            sol = sol2
                    
                    if (temp < self.solEval[n_ind]) & np.isfinite(np.sum(c)) :
                        self.solEval[n_ind] = temp 
                        self.candidates[n_ind] = sol
                    
            #saving current status
            self.best.append(np.min(np.array(self.solEval)))
            self.pop.append(np.mean(np.array(self.solEval)))
            #print(self.candidates )
                    
                    
    # function to cut matriz in cross over operation
    def cutMat(self,mat):
        cutPoint1 = np.random.randint(1,len(mat))
        #print('cut',cutPoint1)
        # get only the cut point to make calculation
        mat1Head = mat[:cutPoint1,:]
        mat1cut = mat[cutPoint1:,:]
        # get the end point
        vec1 = np.sum(mat1cut[:, 1:(self.Aridade+1)] == 2,axis=1) 
        
        endPos = 1
        buffer = 0
        for val in vec1:
            
            if (val == 0) & (buffer == 0) : # stop
                break
            elif (val == 0) & (buffer > 0):
                buffer -= 1
            elif (val==2):
                buffer+=1
            endPos+=1
        
        '''
        vec1 = np.sum(mat1cut[:, 1:(self.Aridade+1)] == 2,axis=1) == 0
        vec1 = vec1*np.arange(1,len(vec1)+1)
        vec1 = vec1[vec1 > 0]
        endPos = np.min(vec1) 
        '''        
        mat1tail = mat1cut[endPos:,:]
        mat1cut = mat1cut[:endPos,:]
        return(mat1Head,mat1cut,mat1tail)
    
    
    def mutData (self,mat1): 
        slice1h,slice1c,slice1t = self.cutMat(mat1)
        slice2c = []
        
        self.createMatrix(self.recCreation2(),slice2c)
        slice2c = np.array(slice2c)
        # ---------- part 1 -----------
        
        vec1 = np.sum(slice1h[:, 1:(self.Aridade+1)] == 2,axis=1) == 2
        l_pos = 0
        m_size1 = slice1h.shape[0]
        
        c1_size = slice1c.shape[0]
        c2_size = slice2c.shape[0]
        
        for val in vec1:
            # we have two 2
            if (val==True):
                # the value saved is biiger them the current matriz size
                if(slice1h[l_pos,-1] > m_size1-l_pos):
                    # fill with the new value
                    slice1h[l_pos,-1] = slice1h[l_pos,-1] - c1_size + c2_size
            l_pos+=1
        
        sol1 = np.r_[slice1h,slice2c,slice1t]
        return(sol1)
        
        
    # setting matriz afte the 
    def crossData(self, mat1,mat2):
        # random select cut point
        slice1h,slice1c,slice1t = self.cutMat(mat1)
        slice2h,slice2c,slice2t = self.cutMat(mat2)
        
        
        #--------------- to be a function -----------------------
        
        # ---------- part 1 -----------
        
        vec1 = np.sum(slice1h[:, 1:(self.Aridade+1)] == 2,axis=1) == 2
        l_pos = 0
        m_size1 = slice1h.shape[0]
                
        c1_size = slice1c.shape[0]
        c2_size = slice2c.shape[0]

        for val in vec1:            
            # we have two 2
            if (val==True):
                # the value saved is biiger them the current matriz size                
                if(slice1h[l_pos,-1] > m_size1-l_pos):

                    # fill with the new value
                    slice1h[l_pos,-1] = slice1h[l_pos,-1] - c1_size + c2_size
            l_pos+=1
        
        #print('end', slice1h)
        # ---------- part 2 -----------
        
        vec2 = np.sum(slice2h[:, 1:(self.Aridade+1)] == 2,axis=1) == 2
        l_pos = 0
        m_size2 = slice2h.shape[0]
        
        c1_size = slice1c.shape[0]
        c2_size = slice2c.shape[0]
        
        for val in vec2:
            # we have two 2
            if (val==True):
                # the value saved is biiger them the current matriz size
                if(slice2h[l_pos,-1] > m_size2-l_pos):
                    # fill with the new value
                    slice2h[l_pos,-1] = slice2h[l_pos,-1] - c2_size + c1_size
            l_pos+=1
            
            
        
        #print('slice1',slice2c)
        #print('slice1',slice1c)
                
        # creating new solution after cross
        sol1 = np.r_[slice1h,slice2c,slice1t]
        sol2 = np.r_[slice2h,slice1c,slice2t]
        
        return([sol1,sol2])
        
        
    # function to evaluate y against x
    def funcEval(self,y_hat):
        return(self.evalFunc(self.out,y_hat))
        
    
    def evaluateTree(self,mat):
        results = []
        n_count = []
        n_count.append(0)
        for v in mat[0,1:self.Aridade+1]:          
            # insert new branch
            if (v == 2):                
                #print(int(mat[0,self.Aridade+1+n_count[-1]]))
                results.append( self.evaluateTree(mat[int(mat[0,self.Aridade+1+n_count[-1]]):,:]) )                    
            elif (v == 0): # constant
                results.append(self.Constantes[mat[0,self.Aridade+1+n_count[-1]]] )
            # insert variable
            else:# (v== 1):
                results.append(self.Variables[mat[0,self.Aridade+1+n_count[-1]]])
        
            n_count[-1] = n_count[-1]+1
            
        
        
        n_count.pop(-1)
        #if (mat[0,0] == 0):
            
        results[1] = np.ones(self.numberOut)*results[1]
        results[0] = np.ones(self.numberOut)*results[0]
        
        # -- to do add geometric mean operator -- 
        
        #print(results[0], results[1])
        if (self.functions[mat[0,0]] is np.sum):
            return (  self.functions[mat[0,0]] (results,axis=0) ) # returns the operation 
        else:           
            return (  self.functions[mat[0,0]] (results[0],results[1]) ) # returns the operation 
                
    def evaluateTreeInput(self,mat,variables):
        results = []
        n_count = []
        n_count.append(0)
        for v in mat[0,1:self.Aridade+1]:          
            # insert new branch
            if (v == 2):                
                #print(int(mat[0,self.Aridade+1+n_count[-1]]))
                results.append( self.evaluateTreeInput(mat[int(mat[0,self.Aridade+1+n_count[-1]]):,:],variables) )                    
            elif (v == 0): # constant
                results.append(self.Constantes[mat[0,self.Aridade+1+n_count[-1]]] )
            # insert variable
            else:# (v== 1):
                results.append(variables[mat[0,self.Aridade+1+n_count[-1]]])
        
            n_count[-1] = n_count[-1]+1
            
        
        
        n_count.pop(-1)
        #if (mat[0,0] == 0):
        numberOut = len(variables[0])
        results[1] = np.ones(numberOut)*results[1]
        results[0] = np.ones(numberOut)*results[0]
        
        # -- to do add geometric mean operator -- 
        
        #print(results[0], results[1])
        if (self.functions[mat[0,0]] is np.sum):
            return (  self.functions[mat[0,0]] (results,axis=0) ) # returns the operation 
        else:           
            return (  self.functions[mat[0,0]] (results[0],results[1]) ) # returns the operation 
        
    
    
    def createMatrix(self,obj,mat):
        if isinstance(obj, tuple):
            [self.createMatrix(item,mat) for item in obj]
        elif isinstance(obj, np.ndarray):
            mat.append(obj)
        
    def getSizeOfNestedList(self,listOfElem):
        count=0
        if isinstance(listOfElem, tuple):
            count+=np.sum([self.getSizeOfNestedList(item) for item in listOfElem])
            return(count)
        elif isinstance(listOfElem, np.ndarray):
            return(count+1)
        
    def recCreation(self):
        fun = np.random.randint(self.numberOfFunctions)
        arg =[]
        valorArg = []
        passed = 0
        passed = copy.copy(passed)
        #print (hex(id(passed)))
        #print (passed)
        pos = 0
        for v in np.random.randint(self.numberArg,size=self.Aridade):          
            # insert new branch
            if (v == 2):
                # run the first time
                if (passed == 0):
                    otherLine = self.recCreation()                    
                    pos = self.getSizeOfNestedList(otherLine)+1
                    valorArg.append(1) # will always be the next line
                    arg.append(v)
                    #print('otherLine', otherLine)
                    #print('pos', pos)
                    #print(hex(id(pos)))
                    passed +=1
                else:
                    #print('otherLine', otherLine)
                    #print('pos', pos)
                    #print (hex(id(pos)))
                    otherLine = (self.recCreation(),otherLine)                    
                    valorArg.append(pos)
                    arg.append(v)                    
                    # to be updated
                    pos = pos+self.getSizeOfNestedList(otherLine[0])
            # insert constant                
            if (v == 0):
                valorArg.append(np.random.randint(self.numberCon ))
                arg.append(v)
            # insert variable
            if (v== 1):
                valorArg.append(np.random.randint(self.numberVar ))
                arg.append(v)
        
        
        line = np.r_[np.array(fun), np.array(arg), np.array(valorArg)]
        #print(line)
        try:
            return ( line, otherLine )
        except:
            #print('err')
            return(line)
        
        
        
        
    def recCreation2(self):
        fun = np.random.randint(self.numberOfFunctions)
        arg =[]
        valorArg = []
        passed = []
        pos = []        
    
        passed.append(0)
        pos.append(0)
        
        #print (passed)
        #print (hex(id(passed)))
        
        for v in np.random.randint(self.numberArg,size=self.Aridade):          
            # insert new branch
            if (v == 2):
                # run the first time
                if (passed[-1] == 0):
                    otherLine = self.recCreation2()                    
                    pos[-1] = self.getSizeOfNestedList(otherLine)+1
                    valorArg.append(1) # will always be the next line
                    arg.append(v)
                    #print('otherLine', otherLine)
                    #print('pos', pos[-1])
                    #print(hex(id(pos[-1])))
                    passed[-1] +=1
                else:
                    #print('otherLine', otherLine)
                    #print('pos', pos[-1])
                    #print (hex(id(pos)))
                    valorArg.append(pos[-1])
                    arg.append(v)                    
                    otherLine = (otherLine,self.recCreation2())                 
                                        
                    # to be updated
                    pos[-1] = pos[-1]+self.getSizeOfNestedList(otherLine[1])
            # insert constant                
            if (v == 0):
                valorArg.append(np.random.randint(self.numberCon ))
                arg.append(v)
            # insert variable
            if (v== 1):
                valorArg.append(np.random.randint(self.numberVar ))
                arg.append(v)
        
        #print (passed)
        passed.pop(-1)
        pos.pop(-1)
        
        line = np.r_[np.array(fun), np.array(arg), np.array(valorArg)]
        #print(line)
        try:
            return ( line, otherLine )
        except:
            #print('err')
            return(line)