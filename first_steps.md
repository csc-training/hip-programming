## Accessing LUMI

Are you able to `ssh` to LUMI? If not, have you followed the instructions [here](https://docs.lumi-supercomputer.eu/firststeps/)?

If you haven't added the ssh-key correctly or cannot otherwise `ssh` to LUMI, you can use the [web interface](https://www.lumi.csc.fi/public/).

See the [documentation](https://docs.lumi-supercomputer.eu/firststeps/loggingin-webui/) for more help.

## Getting the course material

You can clone this git repository with `git clone https://github.com/csc-training/hip-programming.git`.

This way you get a local access to the lectures, as well as the exercises (which you need to run on LUMI).

## Using slurm

Supercomputers like LUMI are shared resources, meaning multiple users are using them at the same time.
To run something on LUMI, you need to use SLURM to submit a job.

Read the [LUMI documentation](https://docs.lumi-supercomputer.eu/runjobs/) on running jobs to find out more.

## Motivation for the course

Why do we teach GPU programming? Why should you learn to program GPUs?

Because most of the Top 500 supercomputers use (and derive most of their compute cabability from) GPUs
--> if you use any of these supercomputers, you cannot avoid using GPUs.

Why are most of the Top 500 supercomputers using GPUs?

1. Because GPUs are designed and optimized to solve problems commonly encountered in HPC and ML/AI: floating point operations, matrix multiplications.
2. Because of power limitations: performance per Watt is much greater for GPUs than CPUs.
