from __future__ import division
from .colvar import Colvar
from .image_ import Image_
from .string_ import String_
import numpy as np
import os
import pickle

def string_production(job_settings):
    status= job_settings['status']
    
    # read in the group and CV definitions
    group_cv_str= Colvar.read_group_cv_def(job_settings['root_dir']+'/'+job_settings['input_dir']+'/'+job_settings['group+cv_def_file'])
    assert group_cv_str, 'group and CV definition file is empty!'
    Image_.group_cv_list= group_cv_str

    # read in the restraint definitions
    restraint_list= Colvar.read_restraint_def(job_settings['root_dir']+'/'+job_settings['input_dir']+'/'+job_settings['restraint_def_file'])
    assert len(restraint_list), 'restraint definition file is empty!'
    # store the restraint definitions in a list of colvar objects
    restraint_list= [Colvar(*res) for res in restraint_list]
    Image_.restraint_list= restraint_list

    # read in the initial condition file; each row is an image, and each column is a colvar
    cntr_list= np.loadtxt(job_settings['root_dir']+'/'+job_settings['input_dir']+'/'+job_settings['init_cond_file'])
    assert cntr_list.shape == (job_settings['num_img'], len(restraint_list)), 'Inconsistent initial condition!'

    #
    if job_settings['check_tessellation'] == job_settings['check_tessellation'] == False:
        print('Are you sure you do not want to check either sigma or tessellation?')

    # copy the job setting file into the input file folder
    os.system('cp job_settings.py %s/%s/' %(job_settings['root_dir'], job_settings['input_dir']))

    if status == 'new':
        
        # generate output_directory
        assert not os.path.isdir(job_settings['root_dir']+'/output'), 'output directory already exists!'
        os.mkdir(job_settings['root_dir']+'/output')
        job_settings['output_dir']= job_settings['root_dir']+'/output'
        
        # check that files required in the job settings exist
        for file_name in ['group+cv_def_file', 'restraint_def_file', 'init_cond_file', 'topo_file', 'mdp_file']:
            assert os.path.isfile(job_settings['root_dir']+'/'+job_settings['input_dir']+'/'+job_settings[file_name]), file_name+' does not exist!'
        if job_settings['index_file'] != 'None': assert os.path.isfile(job_settings['root_dir']+'/'+job_settings['input_dir']+'/'+job_settings['index_file']), file_name+' does not exist!'
        
        if job_settings['fix_endpoints'] == True:
            img_range= range(1, job_settings['num_img'] - 1)
        elif job_settings['fix_endpoints'] == False:
            img_range= range(job_settings['num_img'])
        else:
            raise ValueError('job setting for "fix_endpoints" is invalid')
        for index in img_range:
            assert os.path.isfile(job_settings['root_dir']+'/'+job_settings['input_dir']+'/%d.%s' %(index, job_settings['init_structure_format'])), '%d.%s does not exist!' %(index, job_settings['init_structure_format'])
        
        # read in the directory of the initial structure files
        str_file_list= ['../../../'+job_settings['input_dir']+'/'+'%d.%s' %(index, job_settings['init_structure_format']) for index in range(job_settings['num_img'])]
        
        # generate the image objects
        Image_.js= job_settings
        Image_.cntr_list= [cntr_list]
        img_list= [Image_(str_file_list[img_ind], img_ind) for img_ind in range(job_settings['num_img'])]
        
        # generate the string object
        str_= String_(img_list, job_settings['root_dir'])
        String_.js= job_settings
        
        # running the string calculation
        for itr in range(job_settings['num_itr']):
            str_.evolve()
            str_.smooth()
            str_.reparametrize()
            if job_settings['replica_exchange'] == True: str_.replica_exchange()
            str_.dump_string()
            str_.itr+= 1
            str_.dump_status()


    if status == 'restart':
        
        job_settings['output_dir']= job_settings['root_dir']+'/output'
        pickle_file= '%s/string.p' %job_settings['output_dir']
        assert os.path.isfile(pickle_file), 'Restart file does not exist!'
        str_= pickle.load(open(pickle_file, 'rb'))
        
        # restore static attributes
        String_.js= job_settings
        Image_.js= job_settings
        
        Image_.cntr_list= [cntr_list]
        current_itr= str_.itr
        for itr in range(current_itr):
            cntr_file= '%s/string_%d.npy' %(job_settings['output_dir'], itr)
            cntr= np.load(cntr_file)
            Image_.cntr_list.append(cntr)
        
        assert current_itr < job_settings['num_itr'], 'Target iteration number already completed!'
        
        # running the string calculation
        for itr in range(current_itr, job_settings['num_itr']):
            str_.evolve()
            str_.smooth()
            str_.reparametrize()
            if job_settings['replica_exchange'] == True: str_.replica_exchange()
            str_.dump_string()
            str_.itr+= 1
            str_.dump_status()
        

    if status == 'umbrella':
        
        job_settings['output_dir']= job_settings['root_dir']+'/output'
        pickle_file= '%s/string.p' %job_settings['output_dir']
        assert os.path.isfile(pickle_file), 'Restart file does not exist! The umbrella sampling mode cannot be used without restart!'
        str_= pickle.load(open(pickle_file, 'rb'))
        
        # restore static attributes
        String_.js= job_settings
        Image_.js= job_settings
        
        Image_.cntr_list= [cntr_list]
        current_itr= str_.itr
        for itr in range(current_itr):
            cntr_file= '%s/string_%d.npy' %(job_settings['output_dir'], itr)
            cntr= np.load(cntr_file)
            Image_.cntr_list.append(cntr)
            
        str_.umbrella()