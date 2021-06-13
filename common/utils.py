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
    def set_global_var(var, value):
        assert type(var) == str 
        exec('Str2CodeAdaptor.tmp = value')
        exec(f'{var}=Str2CodeAdaptor.tmp', globals())
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
    def set_var(self, var, value, declare='public'):
        assert declare=='public' or declare == 'private'
        assert type(var) == str 
        tmp = value
        if declare == 'public':
            exec(f'self.{var}=tmp')
        else:
            exec(f'self._{var}=tmp')
    