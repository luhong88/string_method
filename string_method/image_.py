from __future__ import division
import numpy as np
import os
import sys
import time

class Image_(object):
    restraint_list= None
    group_cv_list= None
    cntr_list= None
    js= None
    
    @staticmethod
    def set_cntr_list(new_list):
        Image_.cntr_list.append(new_list)
    
    @staticmethod
    def min_abs(arr1, arr2, arr3):
        """
        Compare arr1, arr2, and arr3 element-wise, and then return the value with the smallest magnitude
        """
        
        min_arr= np.where(np.absolute(arr1) <= np.absolute(arr2), arr1, arr2)
        min_arr= np.where(np.absolute(min_arr) <= np.absolute(arr3), min_arr, arr3)
        
        return min_arr
    
    def __init__(self, img_file, index):
        self.current_cntr= Image_.cntr_list[0][index] # assuming that initial structures match the initial image positions
        self.img_file= img_file
        self.index= index
        self.exchanged_index= index # differ from self.index if replica exchange is turned on
        self.status= None
        self.gradient= None
        self.M= None
        
        per_lst= np.asarray([res.per for res in Image_.restraint_list])
        self.two_pi_per_lst= np.where(per_lst == '2pi')[0]
        self.shift= np.zeros_like(Image_.cntr_list[0][0])
        self.shift[self.two_pi_per_lst]+= 2*np.pi
    
    
    def write_plumed(self, init_cntr, init_kappa, final_cntr, final_kappa, time, itr_img_dir, US_step= 0):
        """
        Write plumed input file
        
        The argument US_step is only relevant for the umbrella sampling mode
        """
        
        restraint_def_list= []
        
        time_with_zero= np.insert(time, 0, 0)
        steps= (time_with_zero/Image_.js['md_step_size']).astype(int)
        steps_acc= np.add.accumulate(steps)
        
        for res_ind, res in enumerate(Image_.restraint_list):            
            restraint_def_header= '\nrestraint-%s: ...\n  MOVINGRESTRAINT\n  ARG=%s\n' %(res.name, res.name)
            
            for plumed_step, sim_time in enumerate(steps_acc):
                cntr= init_cntr[res_ind] if plumed_step == 0 else final_cntr[res_ind]
                kappa= init_kappa[res_ind] if plumed_step == 0 else final_kappa[res_ind]
                
                restraint_def_body= '  STEP%d=%d AT%d=%f KAPPA%d=%f\n' \
                               %(plumed_step, sim_time, plumed_step, cntr, plumed_step, kappa)
                restraint_def_header+= restraint_def_body
            
            restraint_def= restraint_def_header+'...\n'
            restraint_def_list.append(restraint_def)
        
        res_name_list= ','.join([res.name for res in Image_.restraint_list])
        if self.status == 'umbrella':
            print_str='\nPRINT ARG=%s STRIDE=%d FILE=step%d_%s.dat\n' %(res_name_list, Image_.js['US_plumed_STRIDE'], US_step, self.status)
        else:
            print_str='\nPRINT ARG=%s STRIDE=%d FILE=%s.dat\n' %(res_name_list, Image_.js['plumed_STRIDE'], self.status)
            
        output_str= Image_.group_cv_list + ''.join(restraint_def_list) + print_str
        
        if self.status == 'umbrella':
            with open(itr_img_dir+'/step%d_plumed_%s.dat' %(US_step, self.status), 'w') as outfile: outfile.write(output_str)
        else:
            with open(itr_img_dir+'/plumed_%s.dat' %self.status, 'w') as outfile: outfile.write(output_str)
        
    def write_mdp(self, time, itr_img_dir, US_step= 0):
        """
        Write the mdp files used to generate GROMACS tpr files
        
        The argument US_step is only relevant for the umbrella sampling mode
        """
        
        if self.status == 'umbrella':
            with open(Image_.js['root_dir']+'/'+Image_.js['input_dir']+'/'+Image_.js['US_mdp_file'], 'r') as infile:
                mdp_settings= infile.read()
        else:
            with open(Image_.js['root_dir']+'/'+Image_.js['input_dir']+'/'+Image_.js['mdp_file'], 'r') as infile:
                mdp_settings= infile.read()
        
        total_time= np.sum(time)
        num_steps= int(total_time/Image_.js['md_step_size'])
        time_settings= '\ndt = %f\nnsteps = %d\n' %(Image_.js['md_step_size'], num_steps)
        
        mdp_str= ''.join([time_settings, mdp_settings])
        
        if self.status == 'umbrella':
            with open(itr_img_dir+'/step%d_%s.mdp' %(US_step, self.status), 'w') as outfile:
                outfile.write(mdp_str)
        else:
            with open(itr_img_dir+'/%s.mdp' %self.status, 'w') as outfile:
                outfile.write(mdp_str)
        
    def write_sbatch(self, itr, itr_img_dir, input_structure_file, US_step= 0):
        """
        Write sbatch file for submission to queue
        
        The argument US_step is only relevant for the umbrella sampling mode
        """

        If (Image_.js['cluster'] == 'wynton-CPU') and Image_.js['status'] == 'umbrella':
            raise NotImplementedError('Running umbrella sampling on Wynton CPU is not advised and not currently implemented')
        
        sbatch_settings_list=[]
        sbatch_settings_list.append('#!/bin/bash\n')
        
        if Image_.js['cluster'] == 'midway2':
            num_cpu= Image_.js['num_node']*Image_.js['cpu_per_node']
                        
            if Image_.js['account'] != 'None':
                sbatch_settings_list.append('\n#SBATCH --account=%s\n' %Image_.js['account'])
            
            if self.status == 'umbrella':
                sbatch_settings_list.append('#SBATCH --job-name=%d-%03d\n#SBATCH --output=step%d_%s.out\n#SBATCH --error=step%d_%s.err\n#SBATCH --partition=%s\n#SBATCH --nodes=%d\n' \
                                            %(itr, self.index, US_step, self.status, US_step, self.status, Image_.js['partition'], Image_.js['num_node']))
            else:
                sbatch_settings_list.append('#SBATCH --job-name=%d-%03d\n#SBATCH --output=%s.out\n#SBATCH --error=%s.err\n#SBATCH --partition=%s\n#SBATCH --nodes=%d\n' \
                                            %(itr, self.index, self.status, self.status, Image_.js['partition'], Image_.js['num_node']))
            if Image_.js['exclusive'] == True:
                sbatch_settings_list.append('#SBATCH --exclusive\n')
            else:
                sbatch_settings_list.append('#SBATCH --ntasks=%d\n' %num_cpu)

            if Image_.js['qos'] != 'None':
                sbatch_settings_list.append('#SBATCH --qos=%s\n' %Image_.js['qos'])
            if Image_.js['exclude'] != 'None':
                sbatch_settings_list.append('#SBATCH --exclude=%s\n' %Image_.js['exclude'])
            
            # the module setting should be modified according to the cluster in question
            #module_settings= '\nmodule purge\nmodule load gromacs/5.1.4-cuda-7.5+intelmpi-5.1+intel-16.0\nmodule load plumed\n'
            module_settings= '\nmodule purge\nsource /project2/dinner/gromacs/sourceme.sh\n'
            
        elif Image_.js['cluster'] == 'wynton-GPU':
            sbatch_settings_list.append('#$ -cwd\n')
            
            sbatch_settings_list.append('#$ -q %s\n' %Image_.js['partition'])
            sbatch_settings_list.append('#$ -pe mpi_onehost %d\n' %Image_.js['num_node'])
            
            sbatch_settings_list.append('#$ -N str-%d-%03d\n' %(itr, self.index))
            if self.status == 'umbrella':
                sbatch_settings_list.append('#$ -o step%d_%s.out\n#$ -e step%d_%s.err\n' %(US_step, self.status, US_step, self.status))
            else:
                sbatch_settings_list.append('#$ -o %s.out\n#$ -e %s.out\n' %(self.status, self.status))
            
            sbatch_settings_list.append('#S -l h_rt=%s\n' %Image_.js['wall_time'])
            
            if Image_.js['exclude'] != 'None':
                sbatch_settings_list.append('#$ -l h=%s\n' %Image_.js['exclude'])
                
            # the module setting should be modified according to the cluster in question
            module_settings= '\nexport GMX_MAXBACKUP=-1\nexport PATH=$PATH:$HOME/.local/bin\nexport OMP_NUM_THREADS=4\nexport CUDA_VISIBLE_DEVICES=$SGE_GPU\n\nmodule use $HOME/software/modules\nmodule purge\nmodule load cuda\nmodule load mpi\nmodule load plumed\nmodule load gromacs\n'
        
        elif Image_.js['cluster'] == 'wynton-CPU':
            sbatch_settings_list.append('#$ -cwd\n')
            
            sbatch_settings_list.append('#$ -q %s\n' %Image_.js['partition'])
            sbatch_settings_list.append('#$ -pe smp %d\n' %Image_.js['num_cores'])
            
            sbatch_settings_list.append('#$ -N str-%d-%03d\n' %(itr, self.index))
            sbatch_settings_list.append('#$ -o %s.out\n#$ -e %s.out\n' %(self.status, self.status))
            
            sbatch_settings_list.append('#S -l h_rt=%s\n' %Image_.js['wall_time'])
            
            if Image_.js['exclude'] != 'None':
                sbatch_settings_list.append('#$ -l h=%s\n' %Image_.js['exclude'])
                
            # the module setting should be modified according to the cluster in question
            module_settings= '\nexport GMX_MAXBACKUP=-1\nexport PATH=$PATH:$HOME/.local/bin\nexport LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/software\nexport CUDA_LIB_PATH=$CUDA_LIB_PATH:$HOME/software\nexport OMP_NUM_THREADS=$NSLOTS\n\nmodule use $HOME/software/modules\nmodule purge\nmodule load mpi\nmodule load plumed\nmodule load gromacs\n'
        
        # combine the sbatch setting strings
        sbatch_settings= ''.join(sbatch_settings_list)
        
        # command for generating the tpr file
        if Image_.js['cluster'] == 'midway2': gen_tpr_head= 'mpirun -np 1'
        elif Image_.js['cluster'] == 'wynton-GPU': gen_tpr_head= ''
        elif Image_.js['cluster'] == 'wynton-CPU': gen_tpr_head= ''
        
        if self.status == 'umbrella':
            gen_tpr_body= 'gmx_mpi grompp -f step%d_%s.mdp -o step%d_%s.tpr -c %s -p ../../../%s/%s' \
                          %(US_step, self.status, US_step, self.status, input_structure_file, Image_.js['input_dir'], Image_.js['topo_file'])
            if US_step > 1:
                gen_tpr_body+= ' -t %s' %self.cpt_file
        else:
            gen_tpr_body= 'gmx_mpi grompp -f %s.mdp -o %s.tpr -c %s -p ../../../%s/%s' \
                          %(self.status, self.status, input_structure_file, Image_.js['input_dir'], Image_.js['topo_file'])
        
        if Image_.js['index_file'] == 'None': gen_tpr_index= ''
        else: gen_tpr_index= '-n ../../../%s/%s' %(Image_.js['input_dir'], Image_.js['index_file'])
        
        gen_tpr= ' '.join([gen_tpr_head, gen_tpr_body, gen_tpr_index, '\n'])
        
        #command for running the simulation
        if Image_.js['cluster'] == 'midway2': run_job_head= 'mpirun -np %d' %num_cpu
        elif Image_.js['cluster'] == 'wynton-GPU': run_job_head= 'mpirun -n %d' %Image_.js['num_node']
        elif Image_.js['cluster'] == 'wynton-CPU': run_job_head= 'mpirun -n $NHOSTS' # the $NHOSTS variable should be 1
        
        if self.status == 'umbrella':
            run_job_body= 'gmx_mpi mdrun -v -deffnm step%d_%s -plumed step%d_plumed_%s.dat -dlb no' %(US_step, self.status, US_step, self.status)
        else:
            run_job_body= 'gmx_mpi mdrun -v -deffnm %s -plumed plumed_%s.dat -dlb no' %(self.status, self.status)
        
        if Image_.js['cluster'] == 'midway2': run_job_ntomp= '-ntomp 1'
        elif Image_.js['cluster'] == 'wynton-GPU': run_job_ntomp= '-ntomp 4' # the number of CPUs used; ntomp=OMP_NUM_THREADS=4 is recommended for wynton-GPU
        elif Image_.js['cluster'] == 'wynton-CPU': run_job_ntomp= '-ntomp $NSLOTS'

        run_job= ' '.join([run_job_head, run_job_body, run_job_ntomp, '\n'])
        
        
        sbatch_str= ''.join([sbatch_settings, module_settings, gen_tpr, run_job])
        
        if Image_.js['cluster'] == 'midway2': submit_file_type= 'sbatch'
        elif Image_.js['cluster'] == 'wynton-GPU': submit_file_type= 'sh'
        elif Image_.js['cluster'] == 'wynton-CPU': submit_file_type= 'sh'
        
        if self.status == 'umbrella':
            with open(itr_img_dir+'/step%d_%s.%s' %(US_step, self.status, submit_file_type), 'w') as outfile:
                outfile.write(sbatch_str)
        else:
            with open(itr_img_dir+'/%s.%s' %(self.status, submit_file_type), 'w') as outfile:
                outfile.write(sbatch_str)
        
    def check(self, itr, itr_img_dir, job_id):
        """
        Check whether the job is finished, and raise an error if the simulation crashed
        
        This function does not work with the umbrella sampling mode (since it doesn't require checking)
        """
        
        if Image_.js['cluster'] == 'midway2':
            input_file_names_space= 'plumed_{status:s}.dat {status:s}.mdp {status:s}.sbatch'.format(status= self.status)
        elif (Image_.js['cluster'] == 'wynton-GPU') or (Image_.js['cluster'] == 'wynton-CPU'):
            input_file_names_space= 'plumed_{status:s}.dat {status:s}.mdp {status:s}.sh'.format(status= self.status)
        
        output_file_names_space= '{status:s}.tpr {status:s}.err {status:s}.out {status:s}.dat'.format(status= self.status)
        
        # check that the input files exist
        for input_file in input_file_names_space.split():
            assert os.path.isfile(itr_img_dir+'/'+input_file), 'input file %s does not exist (itr: %d, img: %d)!' %(input_file, itr, self.index)
        
        # check if the job has completed
        while True:
            time.sleep(60)
            
            # don't do this; can overload the slurm database
            #job_status= os.popen('sacct -j %d --format=State -n -P' %job_id).read().split('\n')
            #job_failed= 'FAILED' in job_status
            #assert not job_failed, 'SBATCH job failed (itr: %d, img: %d)!' %(itr, self.index)
            #job_cancelled= 'CANCELLED' in job_status
            #assert not job_cancelled, 'SBATCH job cancelled (itr: %d, img: %d)!' %(itr, self.index)
            #job_running= 'RUNNING' in job_status
            #if job_running: continue
            #job_completed= 'COMPLETED' in job_status
            
            # returns an empty string if the job is done (whether failed or completed)
            if Image_.js['cluster'] == 'midway2':
                job_is_running= os.popen('squeue -h -j %d 2>/dev/null' %job_id).read()
            elif (Image_.js['cluster'] == 'wynton-GPU') or (Image_.js['cluster'] == 'wynton-CPU'):
                job_is_running= os.popen('qstat -j %d 2>/dev/null' %job_id).read()
            
            if not job_is_running:
                # first check that the simulation did not crash
                # when gromacs crashes due to numerical problems it will write out pdb files that contain the problem region of the simulation box
                err_str_files= os.popen('ls %s/step*.pdb 2>/dev/null' %itr_img_dir).read()
                if err_str_files: raise RuntimeError('simulation appears to have crashed (itr: %d, img: %d)!' %(itr, self.index))
                
                # a successfully completed job should have the following output files
                for output_file in output_file_names_space.split():
                    assert os.path.isfile(itr_img_dir+'/'+output_file), 'output file %s does not exist (itr: %d, img: %d)!' %(output_file, itr, self.index)
                # check the gro output file separately
                wait_sec_limit= 300
                wait_sec= 0
                while True:
                    if os.path.isfile(itr_img_dir+'/%s.gro' %self.status):
                        break
                    else:
                        wait_sec+= 20
                        time.sleep(20)
                        if wait_sec >= wait_sec_limit:
                            raise RuntimeError('output file %s.gro does not exist (itr: %d, img: %d)!' %(self.status, itr, self.index))
                break
            else:
                # if a job has not finished, check that it is not hanging by checking the error message in the err file
                # this behavior appears to have been fixed in GROMACS 2019 and later versions
                if os.path.isfile(itr_img_dir+'/%s.err' %self.status):
                    with open(itr_img_dir+'/%s.err' %self.status, 'r') as infile:
                        error_log= infile.read()
                    if 'Fatal error:' in error_log or 'application called MPI_Abort' in error_log:
                        os.system('scancel %d' %job_id)
                        raise RuntimeError('SBATCH job has hung (itr: %d, img: %d)!' %(itr, self.index))
        
    def get_cv_traj(self, itr_img_dir):
        """
        Read the CV trajectories from the simulation
        """
        
        time.sleep(10)
        traj_file= itr_img_dir+'/%s.dat' %self.status
        try:
            traj= np.loadtxt(traj_file)[:, 1:]
        except:
            raise RuntimeError('Unable to read file %s' %traj_file)
        
        return traj
        
    def clean_up(self, itr, itr_img_dir):
        """
        Delete unused files from each run
        """
        assert os.path.isdir(itr_img_dir), 'Image output folder does not exist, itr: %d, img: %d' %(itr, self.index)
        
        itr_dir= Image_.js['output_dir']+'/iter_%d' %itr
        cp_status= os.system('cp %s/%s.gro %s/%d.gro' %(itr_img_dir, self.status, itr_dir, self.index))
        if cp_status != 0: raise RuntimeError('Unable to copy equil image structure, itr: %d, img: %d' %(itr, self.index))
        
        rm_status= os.system('rm -r %s' %itr_img_dir)
        if rm_status != 0: raise RuntimeError('Unable to delete image output folder, itr: %d, img: %d' %(itr, self.index))
        
    def compute_gradient(self, cv_traj):
        """
        Compute the local free energy gradient
        """
        # the cv traj contains relax+burn+sample; we only need the sampling run portion
        sample_len= int(Image_.js['sample_t']/(Image_.js['md_step_size']*Image_.js['plumed_STRIDE']))
        
        kappas= np.asarray([res.kappa for res in Image_.restraint_list])
        centers= Image_.cntr_list[-1][self.index]
        
        dist_traj= centers - cv_traj[-sample_len:]
        if self.two_pi_per_lst.size == 0:
            gradient= kappas*np.mean(dist_traj, axis= 0)
        else:
            dist_traj_min= Image_.min_abs(dist_traj, dist_traj + self.shift, dist_traj - self.shift)
            
            gradient= kappas*np.mean(dist_traj_min, axis= 0)
        
        self.gradient= gradient
        
        
    def compute_M(self, cv_traj):
        """
        Compute the metric tensor
        
        Here we use the identity matrix (i.e., ignore the metric tensor)
        """
        
        dim= len(Image_.cntr_list[-1][self.index])
        
        self.M= np.eye(dim)
        
    def run(self, itr, itr_img_dir, equil_multiplier, US_step= 0, first_equil= False):
        """
        Generate and manage a job
        
        The argument US_step is only relevant for the umbrella sampling mode
        """
        
        times= {}
        times['pull+equil']= np.array([Image_.js['pull_t'], Image_.js['equil_t']])
        # use a longer pull+equil time if it is the first equiliration run of the first iteration
        if first_equil == True: times['pull+equil']*= Image_.js['first_equil_multiplier']
            
        times['relax+burn+sample']= np.array([Image_.js['relax_t'], Image_.js['burn_t'], Image_.js['sample_t']])
        try:
            times['umbrella']= 1000*np.array([Image_.js['US_t']]) # convert US_t (unit: ns) to ps
        except KeyError:
            pass # if the umbrella sampling parameters are undefined, I'm probably not running umbrella sampling!
        
        # determine the initial and final cntr and kappa depending on self.status
        if self.status == 'pull+equil':
            init_cntr= self.current_cntr
            final_cntr= Image_.cntr_list[-1][self.index]
            
            init_kappa= np.array([res.kappa for res in Image_.restraint_list])*equil_multiplier
            final_kappa= init_kappa
            # return kappa to normal at the end of equilibration
            if (self.exchanged_index != self.index) and (equil_multiplier < 1.): final_kappa= init_kappa/Image_.js['RE_equil_multiplier']
            
            #print('image %d: initial kappa' %self.index, flush= True)
            #print(init_kappa, flush= True)
            #print('image %d: final kappa' %self.index, flush= True)
            #print(final_kappa, flush= True)

        elif self.status == 'relax+burn+sample':
            init_cntr= Image_.cntr_list[-1][self.index]
            final_cntr= init_cntr
            
            final_kappa= np.array([res.kappa for res in Image_.restraint_list])
            init_kappa= final_kappa*equil_multiplier
            
        elif self.status == 'umbrella':
            init_cntr= final_cntr= Image_.cntr_list[itr][self.index] # the index here is correct, since the cntr_list include the initial string
            init_kappa= final_kappa= np.array([res.kappa for res in Image_.restraint_list])
            
        else:
            raise ValueError('Unkown Image object status')
        
        self.write_plumed(init_cntr, init_kappa, final_cntr, final_kappa, times[self.status], itr_img_dir, US_step= US_step)
        self.write_mdp(times[self.status], itr_img_dir, US_step= US_step)
        self.write_sbatch(itr, itr_img_dir, self.img_file, US_step= US_step)
        
        # submit job; first need to change to the directory containing the sbatch file
        current_dir= os.getcwd()
        os.chdir(itr_img_dir)
        
        # generate the submit command
        if Image_.js['cluster'] == 'midway2':
            submit_command_head= 'sbatch --parsable '
            submit_file_type= '.sbatch'
        elif Image_.js['cluster'] == 'wynton-GPU':
            submit_command_head= 'qsub -terse '
            submit_file_type= '.sh'
        
        if self.status == 'umbrella':
            submit_file= 'step%d_%s' %(US_step, self.status)
        else:
            submit_file= self.status
        
        submit_command= ''.join([submit_command_head, submit_file, submit_file_type])
        sbatch_out= os.popen(submit_command).read()
        
         # throw an error if the sbatch output cannot be converted to a number
        try:
            if Image_.js['cluster'] == 'midway2': job_id= int(sbatch_out[:-1])
            elif Image_.js['cluster'] == 'wynton-GPU': job_id= int(sbatch_out)
        except:
            raise RuntimeError('Unable to submit job, iter_%d/img_%d' %(itr, self.index))
        os.chdir(current_dir)
        
        # if doing umbrella sampling, end the program here
        if self.status == 'umbrella': return None
        
        self.check(itr, itr_img_dir, job_id)
        
        # update image attributes
        cv_traj= self.get_cv_traj(itr_img_dir)
        
        if self.two_pi_per_lst.size == 0:
            self.current_cntr= cv_traj[-1]
        else:
            old_cntr= Image_.cntr_list[-1][self.index]
            new_cntr= cv_traj[-1]
            dist= new_cntr - old_cntr
            
            dist_min= Image_.min_abs(dist, dist + self.shift, dist - self.shift)
            
            self.current_cntr= dist_min + old_cntr
        
        self.img_file= '%s.gro' %self.status
        
    def evolve(self, itr):
        """
        Equilibrate the system and then compute the gradient and M tensor
        """
        
        debug= False

        # skip the simulation if the end points are set to be fixed
        end_points= [0, Image_.js['num_img'] - 1]
        if Image_.js['fix_endpoints'] == True and self.index in end_points:
            dim= len(Image_.cntr_list[-1][self.index])
            self.gradient= np.zeros(dim)
            self.M= np.eye(dim)
            
            return self
        
        # first check if the output directory exists
        itr_img_dir= Image_.js['output_dir']+'/iter_%d/img_%d' %(itr, self.index)
        if os.path.isdir(itr_img_dir):
            os.system('rm -r %s' %itr_img_dir)
            print('Output folder already exists and will be deleted, itr: %d, img: %d' %(itr, self.index))
        os.mkdir(itr_img_dir)
        
        # set the initial structure files excpet for the first iteration
        if itr > 0:
            # use exchanged_index to fetch the appropriate structure after replica exchange
            if os.path.isfile('%s/iter_%d/%d.gro' %(Image_.js['output_dir'], itr - 1, self.exchanged_index)):
                self.img_file= '../../iter_%d/%d.gro' %(itr - 1, self.exchanged_index)
            elif os.path.isfile('%s/iter_%d/img_%d/relax+burn+sample.gro' %(Image_.js['output_dir'], itr - 1, self.exchanged_index)):
                self.img_file= '../../iter_%d/img_%d/relax+burn+sample.gro' %(itr - 1, self.exchanged_index)
            else:
                raise RuntimeError('Cannot find the initial structure file, itr: %d, img: %d (exchanged img index: %d)' %(itr, self.index, self.exchanged_index))
        
        equilibration= 1
        kappas= np.array([res.kappa for res in Image_.restraint_list])
        kappa_multiplier=Image_.js['equil_multiplier']
        equil_fail_reason= 'None'
        # Iterate over the equilibration step
        while True:
            if debug: print('image %d: equil round %d, equil limit %d' %(self.index, equilibration, Image_.js['equil_limit']), flush= True)
            if equilibration > Image_.js['equil_limit']:
                # instead of raising an error, let's simply continue
                #raise RuntimeError('Failed to equilibrate img %d, itr %d within %d rounds due to %s failure' %(self.index, itr, Image_.js['equil_limit'], equil_fail_reason))
                print('Failed to equilibrate img %d, itr %d within %d rounds due to %s failure' %(self.index, itr, Image_.js['equil_limit'], equil_fail_reason), flush= True)
                break
            
            # drag the initial structure to the center, then equilibrate at the new center
            self.status= 'pull+equil'
            if (self.exchanged_index != self.index) and (equilibration == 1):
                # use a softer restraint if there was a replica exchange
                if debug: print('image %d: equil multiplier %f' %(self.index, Image_.js['RE_equil_multiplier']), flush= True)
                self.run(itr, itr_img_dir, Image_.js['RE_equil_multiplier'])
            elif (itr == 1) and (equilibration == 1):
                # use a longer pull+equil time if it is the first equiliration run of the first iteration
                self.run(itr, itr_img_dir, kappa_multiplier**equilibration, first_equil= True)
            else:
                if debug: print('image %d: equil multiplier %f' %(self.index, kappa_multiplier**equilibration), flush= True)
                self.run(itr, itr_img_dir, kappa_multiplier**equilibration)
            
            # the structure is equilibrated if within some sigmas of center/within tessellation bound
            dist_to_cntr= self.current_cntr - Image_.cntr_list[-1][self.index]
            if self.two_pi_per_lst.size == 0:
                dist_to_cntr= np.absolute(dist_to_cntr)
            else:
                dist_to_cntr= np.absolute(Image_.min_abs(dist_to_cntr, dist_to_cntr + self.shift, dist_to_cntr - self.shift))
            if debug:
                print('image %d: computed dist_co_cntr' %self.index, flush= True)
                print(dist_to_cntr, flush= True)
            
            
            if Image_.js['check_sigma']:
                sigmas= Image_.js['equil_tolerance']*np.sqrt(Image_.js['RT']/(kappas*kappa_multiplier))
                within_sigma= all([dist < sigma for dist, sigma in zip(dist_to_cntr, sigmas)])

                if debug:
                    print('image %d: check sigma failure' %self.index, flush= True)
                    print('image %d: computed sigmas' %self.index, flush= True)
                    print(sigmas, flush= True)
                    print('image %d: checked distance' %self.index, flush= True)
                    print([dist < sigma for dist, sigma in zip(dist_to_cntr, sigmas)], flush= True)
                
                if not within_sigma:
                    equilibration+= 1
                    equil_fail_reason= 'sigma'

                    if debug:
                        print('image %d: sigma failure; new equilibration = %d' %(self.index, equilibration), flush= True)
                        print(sigmas, flush= True)
                        print(dist_to_cntr, flush= True)
                    continue
                    
            if Image_.js['check_tessellation']:
                if debug: print('image %d: check tessellation failure' %self.index, flush= True)
                if self.index == 0:
                    neighbor_list= [1]
                elif self.index == Image_.js['num_img'] - 1:
                    neighbor_list= [Image_.js['num_img'] - 2]
                else:
                    neighbor_list= [self.index - 1, self.index + 1]
                dist_to_neighbors= [self.current_cntr - Image_.cntr_list[-1][neighbor_ind] for neighbor_ind in neighbor_list]
                
                if self.two_pi_per_lst.size == 0:
                    norm_to_neighbors= [np.linalg.norm(dist) for dist in dist_to_neighbors]
                else:
                    norm_to_neighbors= [np.linalg.norm(Image_.min_abs(dist, dist + self.shift, dist - self.shift)) for dist in dist_to_neighbors]
                norm_to_self= np.linalg.norm(dist_to_cntr)
                within_tessellation= all([norm_to_self < norm for norm in norm_to_neighbors])
                if debug: 
                    print('image %d: checked tessellation' %self.index, flush= True)
                    print([norm_to_self < norm for norm in norm_to_neighbors], flush= True)
                if not within_tessellation:
                    equilibration+= 1
                    equil_fail_reason= 'tessellation'
                    if debug: print('image %d: tessellation failure; new equilibration = %d' %(self.index, equilibration), flush= True)
                    continue
            
            if debug: print('image %d: passed both sigma and tessellation tests' %self.index)
            break
        
        #reduce the force constant, equilibrate at the new force constant, and sample local free energy gradient
        self.status= 'relax+burn+sample'; self.run(itr, itr_img_dir, kappa_multiplier**equilibration)
        
        cv_traj= self.get_cv_traj(itr_img_dir)
        
        self.compute_gradient(cv_traj)
        self.compute_M(cv_traj)
        
        if Image_.js['clean_up']: self.clean_up(itr, itr_img_dir)
        
        return self
        
    def umbrella(self, itr_for_US, US_continuation):
        """
        Run umbrella sampling calcuations for an existing iteration
        """
        
        self.status= 'umbrella'
        
        US_step= 1 # for counting continuations
        
        # skip the simulation if the end points are set to be fixed
        end_points= [0, Image_.js['num_img'] - 1]
        if Image_.js['fix_endpoints'] == True and self.index in end_points:
            return None
        
        # first check if the output directory exists
        itr_img_dir= Image_.js['output_dir']+'/iter_%d_US/img_%d' %(itr_for_US, self.index)
        if not os.path.isdir(itr_img_dir):
            os.mkdir(itr_img_dir)
            US_continuation= False
        else:
            existing_files= os.listdir(itr_img_dir)
            # set the current step number
            while 'step%d_%s.gro' %(US_step, self.status) in existing_files: US_step+= 1
            if US_step == 1: US_continuation= False
            
        # set the initial structure files
        if US_continuation == False:
            # use exchanged_index to fetch the appropriate structure after replica exchange
            if os.path.isfile('%s/iter_%d/%d.gro' %(Image_.js['output_dir'], itr_for_US, self.exchanged_index)):
                self.img_file= '../../iter_%d/%d.gro' %(itr_for_US, self.exchanged_index)
            elif os.path.isfile('%s/iter_%d/img_%d/relax+burn+sample.gro' %(Image_.js['output_dir'], itr_for_US, self.exchanged_index)):
                self.img_file= '../../iter_%d/img_%d/relax+burn+sample.gro' %(itr_for_US, self.exchanged_index)
            else:
                raise RuntimeError('Cannot find the initial structure file for iter_%d_US, initial continuation step 1' %itr_for_US)
        else:
            if os.path.isfile('%s/iter_%d_US/img_%d/step%d_%s.gro' %(Image_.js['output_dir'], itr_for_US, self.index, US_step - 1, self.status)):
                self.img_file= 'step%d_%s.gro' %(US_step - 1, self.status)
                self.cpt_file= 'step%d_%s.cpt' %(US_step - 1, self.status) # cpt files for continuation simulations
            else:
                raise RuntimeError('Cannot find the initial structure file for iter_%d_US, continuation step %d' %(itr_for_US, US_step))
                
        self.run(itr_for_US, itr_img_dir, 0, US_step= US_step)
        
        
        
