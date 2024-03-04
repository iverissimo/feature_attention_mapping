

# Feature based attention mapping experiment

*under development*

This repository requires certain dependencies, such as [exptools2](https://github.com/VU-Cog-Sci/exptools2) (for communication with the eyetracker) and [prfpy](https://github.com/VU-Cog-Sci/prfpy.git) (for prf fitting). Experimental design code is built ith python 3.6 and resorts to [Psychopy](https://www.psychopy.org/) functions, while the analysis code requires python 3.10 or higher. 

To run a session, `cd` into the `experiment` folder and write
`python main.py <sub_num> <run_num>`

To run analysis, first install relevant modules from local folder by `cd` into the `analysis` folder and writing
`pip install -e .` 

