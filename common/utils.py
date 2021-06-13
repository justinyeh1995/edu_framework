#!/usr/bin/env python
# coding: utf-8
import os, sys
def blockPrinting(func):
    def func_wrapper(*args, **kwargs):
        # block all printing to the console
        sys.stdout = open(os.devnull, 'w')
        # call the method in question
        value = func(*args, **kwargs)
        # enable all printing to the console
        sys.stdout = sys.__stdout__
        # pass the return value of the method back
        return value

    return func_wrapper

class Str2CodeAdaptor:
    '''
    By inherenting this class, we will be able to :  
    
      1. insert public and private variables into a class by assigning variable and experssion 
      2. add variable into the top-level global variable pool. 
      
    '''
    @staticmethod 
    def set_global_var(__globals__, var, value):
        assert type(var) == str 
        assert type(__globals__) == dict
        Str2CodeAdaptor.tmp = value
        __globals__.update({var: Str2CodeAdaptor.tmp})
        # exec(f'{var}=Str2CodeAdaptor.tmp', globals())
        del Str2CodeAdaptor.tmp 
    @staticmethod
    def add_var_to_obj(obj, var, value, declare='public'):
        assert declare=='public' or declare == 'private'
        assert type(var) == str
        tmp = value
        if declare == 'public':
            exec(f'obj.{var}=tmp')
        else:
            exec(f'obj._{var}=tmp')
    @staticmethod
    def add_var(env, var, value):
        '''
        This function allows the call to `set_global_var` and `add_var_to_obj` with the determination of `env`, 
        where env can be `globals()` or a object. 
        '''
        if type(env) == dict:
            Str2CodeAdaptor.set_global_var(env, var, value)
        else:
            Str2CodeAdaptor.add_var_to_obj(env, var, value)
            
    def set_var(self, var, value, declare='public'):
        assert declare=='public' or declare == 'private'
        assert type(var) == str 
        tmp = value
        if declare == 'public':
            exec(f'self.{var}=tmp')
        else:
            exec(f'self._{var}=tmp')
    