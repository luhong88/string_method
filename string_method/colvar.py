import numpy as np

class Colvar(object):
    
    @staticmethod
    def read_restraint_def(file_loc):
        """
        Read in the restraint definitions from a restraint text file.
        """
        
        restraint_list= []
                
        with open(file_loc, 'r') as infile:
            for line in infile:
                split_line= line.split()
                
                name= split_line[0]
                kappa= float(split_line[1])
                per= split_line[2]
                
                restraint_list.append([name, kappa, per])
        
        return restraint_list
        
    @staticmethod
    def read_group_cv_def(file_loc):
    
        with open(file_loc, 'r') as infile:
            group_cv_def= infile.read()
        
        return group_cv_def
        
    
    def __init__(self, name, kappa, per):
        """
        Initialize a CV definition
        """
        
        self.name= name
        self.kappa= kappa
        self.per= per
