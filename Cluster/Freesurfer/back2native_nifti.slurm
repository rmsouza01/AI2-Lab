#!/bin/bash
#SBATCH --ntasks=1           # number of MPI processes
#SBATCH --mem-per-cpu=4GB    # memory; default unit is megabytes
#SBATCH --time=0-4:00       # time (DD-HH:MM)
#SBATCH --output=back2native_nifti_cs.%A_%a.out




# Adapt the script to your needs according to the comments in this script. Actionable items are all in UPPER CASE

export FREESURFER_HOME=/work/harris_lab/freesurfer/freesurfer # SET THIS PATH TO WHERE FREESURFER IS INSTALLED
source $FREESURFER_HOME/SetUpFreeSurfer.sh
export SUBJECTS_DIR=/home/roberto.medeirosdeso/freesurfer-test # SPECIFY THE FOLDER WHERE YOU WANT TO SAVE THE RESULTS

# Printing some info
echo "Running at `date`"
echo "Running on `hostname`"

cd $SLURM_SUBMIT_DIR 
echo "Working directory is `pwd`"

for dir in /home/roberto.medeirosdeso/freesurfer-test/*
   do 
      cd $dir/mri
      subj_name="$(basename $dir)"
      echo "$subj_name"
      # ADAPT TO THE FILES THAT YO UWOULD LIKE TO CONVERT BACK TO NATIVE SPACE AND SAVE AS NIFTI
      mri_label2vol --seg aseg.mgz --temp rawavg.mgz --o aseg-in-rawavg.mgz --regheader aseg.mgz 
      mri_convert aseg-in-rawavg.mgz /home/roberto.medeirosdeso/aseg/$subj_name.nii.gz
   done	
