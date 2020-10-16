from __future__ import division
# do not use multithreading; it causes racing conditions
#from concurrent.futures import ProcessPoolExecutor
import multiprocessing
import logging

from .image_ import Image_
import numpy as np
import time
import pickle
import os
import sys


class String_(object):
    """
    see Maragliano et al. (2006) J. Chem. Phys. for the algorithm
    """
    
    js= None
        
    def __init__(self, img_list, root_dir):
        self.itr= 0
        self.img_list= img_list
        self.root_dir= root_dir
        self.num_img= len(img_list)
        self.new_cntr= None
        
        self.two_pi_per_lst= self.img_list[0].two_pi_per_lst
        self.shift= self.img_list[0].shift
        
    def evolve(self):
        """
        Evolve the string
        """
        
        # first check if the output directory exists
        itr_dir= String_.js['output_dir']+'/iter_%d' %self.itr
        if os.path.isdir(itr_dir):
            os.system('rm -r %s' %itr_dir)
            sys.stderr.write('Output folder already exists and will be overwritten, itr: %d\n' %self.itr)
        os.mkdir(itr_dir)
        
        # evolve the images in parallel with multiprocess
        jobs= []
        result_queue= multiprocessing.Queue()
        multiprocessing.log_to_stderr(logging.DEBUG)
        logger= multiprocessing.get_logger()
        logger.setLevel(logging.INFO)
        
        for img in self.img_list:
            proc= multiprocessing.Process(target= img.evolve, args= (result_queue, self.itr))
            jobs.append(proc)
            proc.start()
        
        # fetch results from the queue
        new_img_list= []

        for proc in jobs:
            new_img= result_queue.get()
            assert not isinstance(new_img, Exception), 'Exception found in %s' %proc
            new_img_list.append(new_img)

        for proc in jobs:
            proc.join()
        
        self.img_list= new_img_list


        # the following code is for concurrent.futures, which somehow doesn't work on Wynton
        # evolve the images in parallel with multiprocess
        #pool= ProcessPoolExecutor(self.num_img)
        #futures= [pool.submit(img.evolve, (self.itr,)) for img in self.img_list]
        #    
        #while True:
        #    complete= [future.done() for future in futures]
        #    if np.all(complete):
        #        self.img_list= [future.result() for future in futures]
        #        break
        #    else:
        #        time.sleep(60)
        
        # extract simulation results
        current_cntr= Image_.cntr_list[-1]
        self.new_cntr= np.zeros_like(current_cntr)
        current_gradient= [img.gradient for img in self.img_list]
        current_M= [img.M for img in self.img_list]
        dt= String_.js['euler_time']
        
        # compute the projection operator for each image (except for the two end points)
        # for the two end points, insert identity matrix (for indexing purpose only)
        project=[]
        num_cv= len(self.new_cntr[0])
        project.append(np.eye(num_cv))
        for ind in range(1, self.num_img - 1):
            operator=np.eye(num_cv)
            
            disp_vec= current_cntr[ind + 1] - current_cntr[ind]
            if self.two_pi_per_lst.size == 0:
                disp_proj= np.dot(disp_vec, np.dot(current_M[ind], current_gradient[ind]))
            else:
                min_disp_vec= Image_.min_abs(disp_vec, disp_vec + self.shift, disp_vec - self.shift)
                disp_proj= np.dot(min_disp_vec, np.dot(current_M[ind], current_gradient[ind]))
            
            sigma= 1 if disp_proj >= 0 else -1
            
            disp_vec= current_cntr[ind + sigma] - current_cntr[ind]
            if self.two_pi_per_lst.size != 0:
                disp_vec= Image_.min_abs(disp_vec, disp_vec + self.shift, disp_vec - self.shift)
                
            disp_vec_unit= disp_vec/np.linalg.norm(disp_vec)
            disp_vec_outer= np.outer(disp_vec_unit, disp_vec_unit)
            
            operator-= disp_vec_outer
            project.append(operator)
        project.append(np.eye(num_cv))
        
        
        # Evolve the string using forward Euler
        # the two end points evolve by gradient descent
        self.new_cntr[0]= current_cntr[0] - dt*current_gradient[0]
        self.new_cntr[-1]= current_cntr[-1] - dt*current_gradient[-1]
        for ind in range(1, self.num_img - 1):
            self.new_cntr[ind]= current_cntr[ind] - dt*np.dot(np.dot(project[ind], current_M[ind]), current_gradient[ind])
        
        # write out the displacement vector (w/o the dt)
        if String_.js['clean_up'] == False:
            displacement= []
            displacement.append(current_gradient[0])
            for ind in range(1, self.num_img - 1):
                displacement.append(np.dot(np.dot(project[ind], current_M[ind]), current_gradient[ind]))
            displacement.append(current_gradient[-1])
            
            np.save(itr_dir+'/displacement.npy', np.asarray(displacement))
            np.save(itr_dir+'/current_cntr.npy', np.asarray(current_cntr))
        
    def smooth(self):
        """
        Smooth the string
        """
        
        s= String_.js['smooth']
        
        temp_cntr= np.zeros_like(self.new_cntr)
                
        for ind in range(1, self.num_img - 1):
            if self.two_pi_per_lst.size == 0:
                temp_cntr[ind]= (1 - s)*self.new_cntr[ind] + (s/2.)*(self.new_cntr[ind + 1] + self.new_cntr[ind - 1])
            else:
                dist_next_img= self.new_cntr[ind + 1] - self.new_cntr[ind]
                dist_prev_img= self.new_cntr[ind - 1] - self.new_cntr[ind]
                
                next_img= Image_.min_abs(dist_next_img, dist_next_img + self.shift, dist_next_img - self.shift) + self.new_cntr[ind]
                prev_img= Image_.min_abs(dist_prev_img, dist_prev_img + self.shift, dist_prev_img - self.shift) + self.new_cntr[ind]
                
                temp_cntr[ind]= (1 - s)*self.new_cntr[ind] + (s/2.)*(next_img + prev_img)
        
        self.new_cntr[1:-1]= temp_cntr[1:-1]
        
        if String_.js['clean_up'] == False:
            itr_dir= String_.js['output_dir']+'/iter_%d' %self.itr
            np.save(itr_dir+'/smoothed_cntr.npy', np.asarray(self.new_cntr))
        
    def reparametrize(self):
        """
        Reparametrization of the string
        """
                
        img_disp_vec= np.array([self.new_cntr[ind + 1] - self.new_cntr[ind] for ind in range(self.num_img - 1)])
        if self.two_pi_per_lst.size != 0: img_disp_vec= np.array([Image_.min_abs(disp, disp + self.shift, disp - self.shift) for disp in img_disp_vec])
            
        segment_lengths= np.array([np.linalg.norm(img_disp_vec[ind]) for ind in range(self.num_img - 1)])
        segment_length_upto= np.add.accumulate(segment_lengths)
        segment_length_upto= np.insert(segment_length_upto, 0, 0.)
        img_disp_vec_unit= img_disp_vec/(np.asarray([segment_lengths]).T)
        
        new_img_length_upto= np.array([ind*segment_length_upto[-1]/(self.num_img - 1) for ind in range(1, self.num_img - 1)])
        
        temp_cntr= np.zeros_like(self.new_cntr)
        for ind in range(1, self.num_img - 1):
            old_img_ind= int(np.argwhere(segment_length_upto < new_img_length_upto[ind - 1])[-1])
            temp_cntr[ind]= self.new_cntr[old_img_ind] + (new_img_length_upto[ind - 1] - segment_length_upto[old_img_ind])*img_disp_vec_unit[old_img_ind]
            
            if self.two_pi_per_lst.size != 0:
                old_temp_disp= temp_cntr[ind] - self.new_cntr[ind]
                temp_cntr[ind]= Image_.min_abs(old_temp_disp, old_temp_disp + self.shift, old_temp_disp - self.shift) + self.new_cntr[ind]
        
        self.new_cntr[1:-1] = temp_cntr[1:-1]
        # write new centers to image
        Image_.set_cntr_list(self.new_cntr)
    
    def replica_exchange(self):
        """
        Perform replica exchange
        """
        
        bias_cntr= Image_.cntr_list[-2] # not -1, since we assume that the cntr_list has already been updated for the next iteration
        structure_cntr= [img.current_cntr for img in self.img_list]
        kappa= np.asarray([res.kappa for res in Image_.restraint_list])
        
        ind_after_exchange= list(range(self.num_img))
        
        # randomize the neighbor list for each iteration
        neighbor_list= np.asarray(String_.js['RE_NL'])
        np.random.shuffle(neighbor_list)
        
        success_exchange= 0
        for pair in neighbor_list:
            # the first image in the neighbor list is labeled i; the second j
            i, j= [ind_after_exchange[ind] for ind in pair]
            
            Eixi= np.sum(0.5*kappa*(bias_cntr[i]-structure_cntr[i])**2)
            Ejxj= np.sum(0.5*kappa*(bias_cntr[j]-structure_cntr[j])**2)
            Eixj= np.sum(0.5*kappa*(bias_cntr[i]-structure_cntr[j])**2)
            Ejxi= np.sum(0.5*kappa*(bias_cntr[j]-structure_cntr[i])**2)
            
            ln_prob= -String_.js['RE_temp']*((Eixj - Eixi) - (Ejxj - Ejxi))/String_.js['RT']
            ln_r= np.log(np.random.random())
            
            if ln_r <= ln_prob:
                ind_after_exchange[i], ind_after_exchange[j]= ind_after_exchange[j], ind_after_exchange[i]
                success_exchange+= 1
        
        accept_ratio= float(success_exchange)/len(neighbor_list)
        
        # update the exchanged_index attribute for each image
        for ind, exchanged_ind in enumerate(ind_after_exchange):
            self.img_list[ind].exchanged_index= exchanged_ind
        
        # write log file
        out_str_list= ['%d' %(ind) for ind in ind_after_exchange] + ['%.2f' %accept_ratio]
        out_str='\t'.join(out_str_list) + '\n'
        
        output_file= String_.js['output_dir']+'/RE_log.txt'
        with open(output_file, 'a') as outfile:
            outfile.write(out_str)
        
        # save the parameters used to do replica exchange
        itr_dir= String_.js['output_dir']+'/iter_%d' %self.itr
        if String_.js['clean_up'] == False:
            np.save(itr_dir+'/bias_cntr.npy', bias_cntr)
            np.save(itr_dir+'/structure_cntr.npy', structure_cntr)
            np.save(itr_dir+'/kappa.npy', kappa)
        
    def dump_string(self):
        """
        Write out the latest string iteration as an npy file for analysis
        """
        
        outfile= '%s/string_%d.npy' %(String_.js['output_dir'], self.itr)
        np.save(outfile, Image_.cntr_list[-1])
    
    def dump_status(self):
        """
        Write out the current state for restart; use pickle
        """
        
        pickle.dump(self, open('%s/string.p' %String_.js['output_dir'], 'wb'))
    
    def umbrella(self):
        """
        A mode for running umbrella sampling calculations, starting from a completed iteration
        """
        
        itr_for_US= String_.js['US_iter']
        assert 0 <= itr_for_US <= self.itr, 'US_iter = %d is invalid!' %itr_for_US
        US_continuation= False
        
        # first check if the output directory exists
        US_dir= String_.js['output_dir']+'/iter_%d_US' %itr_for_US
        if os.path.isdir(US_dir):
            US_continuation= True
            sys.stdout.write('Output folder already exists and existing simulations will be continued: %s\n' %US_dir)
        else: os.mkdir(US_dir)
        
        
        jobs= []
        
        multiprocessing.log_to_stderr(logging.DEBUG)
        logger= multiprocessing.get_logger()
        logger.setLevel(logging.INFO)
        
        for img in self.img_list:
            p= multiprocessing.Process(target= img.umbrella, args= (itr_for_US, US_continuation))
            jobs.append(p)
            p.start()
        
        for p in jobs: p.join()
        
        # the following code is for concurrent.futures, which somehow doesn't work on Wynton
        #pool= ProcessPoolExecutor(self.num_img)
        #futures= [pool.submit(img.umbrella, (itr_for_US, US_continuation)) for img in self.img_list]
            
        #while True:
        #    complete= [future.done() for future in futures]
        #    if np.all(complete):
        #        break
        #    else:
        #        time.sleep(60)
        
        
        
