# Measurement Pred

Code for replicating "pure prediction" measurment predictors from [Benchmarks for Detecting Measurement Tampering](https://arxiv.org/abs/2308.15605) (see especially section 3.2.1 for training setup and Appendix S for training details)

To run training with slurm: \
`python train.py --multirun` \
(add slurm configs in conf/hydra/launcher, or overwrite in 
command line, see https://hydra.cc/docs/plugins/submitit_launcher/ for more details)

To run a short test: \
`python train.py --multirun --config-name pythia_diamonds_local_test`

TODO - instrutions for loading from huggingface