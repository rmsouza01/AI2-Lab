#!/bin/bash
#SBATCH --ntasks=1           # number of MPI processes
#SBATCH --mem-per-cpu=4GB    # memory; default unit is megabytes
#SBATCH --time=0-24:00       # time (DD-HH:MM)
#SBATCH --output=freesurfer_cs.%A_%a.out



# Adapt the script to your needs according to the comments in this script. Actionable items are all in UPPER CASE

export FREESURFER_HOME=/work/harris_lab/freesurfer/freesurfer # SET THIS PATH TO WHERE FREESURFER IS INSTALLED
source $FREESURFER_HOME/SetUpFreeSurfer.sh
export SUBJECTS_DIR=/home/roberto.medeirosdeso/freesurfer-test # SPECIFY THE FOLDER WHERE YOU WANT TO SAVE THE RESULTS

# Printing some info
echo "Running at `date`"
echo "Running on `hostname`"

cd $SLURM_SUBMIT_DIR 
echo "Working directory is `pwd`"

INPUT_FILE=`sed "${SLURM_ARRAY_TASK_ID}q;d" /work/harris_lab/roberto/Freesurfer-scripts/subjects_list.txt` # CHANGE THE TXT FILE TO THE TXT FILE WITH THE LIST OF FILES YOU WANT TO PROCESS


subj_name=$(basename ${INPUT_FILE} .nii.gz) # THIS LINE EXTRACTS THE FILE EXTENSION OF THE FILE NAME, E.G., NAME.NII.GZ -> NAME. tHIS LINE NEEEDS TO CHANGE IF FILE IS IN ANOTHER FORMAT



# Running freesurfer's recon-all command
echo $subj_name
recon-all -cw256 -i ${INPUT_FILE} -subjid $subj_name
recon-all -all -s $subj_name

# Printing the processing end time
echo "Finished at `date`"
