# -*- coding: utf-8 -*-
"""
Created on Wed Feb 20 11:51:44 2019

@author: Andy
"""

import numpy as np
import pandas as pd
from pgmpy.models import BayesianModel
from pgmpy.estimators import MaximumLikelihoodEstimator, BaseEstimator
from pgmpy.factors.discrete import TabularCPD


class CustomFunctions():
    
    def __init__(self, model):
        self.model = model
        
    
    def custom_fit(self, model,state_counts, estimator=None, state_names=[], complete_samples_only=True, **kwargs):
        if estimator is None:
            estimator = MaximumLikelihoodEstimator
        else:
            if not issubclass(estimator,BaseEstimator):
                raise TypeError("Estimator object should be a valid pgmpy estimator.")
        
        
        cpds_list = self.custom_get_parameters(state_counts,**kwargs)
        model.add_cpds(*cpds_list)
        
    def create_state_names(self, state_counts):
        state_names = dict()
        for df,key in zip(state_counts,self.model.nodes()):
            print(key)
            df = df.reindex(sorted(df.columns),axis=1)
            df = df.reindex(sorted(df.index),axis=0)
            value = (np.asarray(df.index.tolist()))
            state_names[key]= value
        return state_names

    def custom_get_parameters(self, state_counts):
        parameters = []
    
        state_names = self.create_state_names(state_counts)
        
        for node in (self.model.nodes()):
            cpd = self.custom_estimate_cpd(node,state_counts[self.model.nodes().index(node)], state_names)
            parameters.append(cpd)
            
        return parameters

    def custom_estimate_cpd(self,node, state_counts, state_names):
        state_counts = state_counts.reindex(sorted(state_counts.columns), axis=1)
        state_counts = state_counts.reindex(sorted(state_counts.index), axis=0)
        print(state_counts.columns)
        print(state_counts.index)
    
        state_counts.loc[:,(state_counts==0).all()]=1
        
        parents = self.model.get_parents(node)
        if len(parents) ==0:
            parents_cardinalities = []
        else:
            parents_cardinalities = [len(state_counts.columns.tolist())]
        node_cardinality = len(state_counts.index.tolist())
        
        
        cpd = TabularCPD(node,node_cardinality, np.array(state_counts), evidence = parents,
                         evidence_card=parents_cardinalities, state_names=state_names)
        cpd.normalize()
        return (cpd)