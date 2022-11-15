# Thalamus_ICLR

Algorithm Thalamus code for ICLR submission. 


Packages required for datasets used, please follow each repository instructions for instalation in the same project folder. Cloning each repo is sufficient.

Neurogym for cognitive tasks: https://github.com/neurogym/neurogym

and 

Continual leanring for split MNIST dataset: https://github.com/GMvandeVen/continual-learning

python Run.py to run Thalamus on a series of tasks. 

Note that the files are actively being organized and will change within a few days.

It takes the following arguments (or can just omit and use default values):
* dataset: string, pass 'neurogym' or 'split_mnist' to use either dataset. Default 'neurogym'
* experiment_name: String. Determines which folder to create and uses this string to identify files saved for the run.
* no_of_taks: Determines now many tasks to train on. Will automatically test on any remaining tasks using latent updates.
* Seed: sets seed for RNG.
* var1 through 4. Can be assigned to a variety of variables in the simulation.

Output will be saved in ./files/experiment_name'

Logs are saved as training_log.npy, testing_log.npy, and config.npy.

Also plots with the current accuracy on tasks, Thalamus latents, and the weight and latent update counts 
Another plot for split MNIST shows accuracy tested with latent updates on all five tasks throught training. 

The analysis folder has the Jupyter notebooks to analyze results and produce the full paper figures. 
These are being updated and organized actively.
