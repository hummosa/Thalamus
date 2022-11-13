# Thalamus_ICLR
Algorithm Thalamus code for ICLR submission. 
THIS ENTIRE CODE REPO IS PROVIDED FOR TRANSPARANCY BUT IS ACTIVELY BEING ORGANIZED AND CLEANED UP!

Packages required for datasets used:
Neurogym for cognitive tasks: https://github.com/neurogym/neurogym
and 
Continual leanring for split MNIST dataset: https://github.com/GMvandeVen/continual-learning

python Run.py to run Thalamus on a series of tasks. 
Lines 73 to 75 determine which dataset to run:
### Dataset
dataset_name = 'neurogym' 
OR:
dataset_name = 'split_mnist'

Output will be saved in ./files/experiment_name'

Logs are saved as training_log.npy, testing_log.npy, and config.npy.
Also plots with the current accuracy on tasks, Thalamus latents, and the weight and latent update counts 
Another plot for split MNIST shows accuracy tested with latent updates on all five tasks throught training. 

The analysis folder has the Jupyter notebooks to analyze results and produce the full paper figures. 
These are being updated and organized actively.



