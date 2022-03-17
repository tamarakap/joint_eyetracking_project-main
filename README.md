### Latest Results (validation set, split0)
<table>
        <tr>
        All statistics are over all valid frames only.  
        </tr>
        <tr>
                <td colspan="2" align="center"> <img src="https://github.com/shariliu/joint_eyetracking_project/blob/main/assets/collage.png"  alt="collage" width = 300px height = 400px ></td>
        </tr>
        <tr>
                <td><img src="https://github.com/shariliu/joint_eyetracking_project/blob/main/assets/frame_by_frame_00.png" alt="0" width = 1000px height = 100px></td>
                <td><img src="https://github.com/shariliu/joint_eyetracking_project/blob/main/assets/frame_by_frame_01.png" alt="1" width = 1000px height = 100px></td>
        </tr>
        <tr>
                <td><img src="https://github.com/shariliu/joint_eyetracking_project/blob/main/assets/frame_by_frame_02.png" alt="0" width = 1000px height = 100px></td>
                <td><img src="https://github.com/shariliu/joint_eyetracking_project/blob/main/assets/frame_by_frame_03.png" alt="1" width = 1000px height = 100px></td>
        </tr>
        <tr>
                <td><img src="https://github.com/shariliu/joint_eyetracking_project/blob/main/assets/frame_by_frame_04.png" alt="0" width = 1000px height = 100px></td>
                <td><img src="https://github.com/shariliu/joint_eyetracking_project/blob/main/assets/frame_by_frame_05.png" alt="1" width = 1000px height = 100px></td>
        </tr>
        <tr>
                <td><img src="https://github.com/shariliu/joint_eyetracking_project/blob/main/assets/frame_by_frame_06.png" alt="0" width = 1000px height = 100px></td>
                <td><img src="https://github.com/shariliu/joint_eyetracking_project/blob/main/assets/frame_by_frame_07.png" alt="1" width = 1000px height = 100px></td>
        </tr>
</table>

structure of project is as follows:

    ├── assets                  # contains assets for README
    ├── datasets                # put all your (raw or not) datasets here 
    ├── face_classifier         # contains all specific code for face classification, separated from main project files on purpose.
        ├── fc_data.py          # creates face classifier dataset
        ├── fc_eval.py          # face classifier evaluation
        ├── fc_model.py         # face classifier model
        ├── fc_train.py         # face classifier training  script
    ├── models                  # put all your model files here
    ├── plots                   # not used
    ├── statistics              # code for analyzing multi-variant video dataset statistics
    ├── tests                   # pytests
    ├── train.py                # main training loop, keep as generic as possible and minimal
    ├── test.py                 # a sandbox for testing various models performance
    ├── visualize.py            # all visualizations should go here
    ├── parsers.py              # various looking videos coding formats are parsed using the classes in this file
    ├── logger.py               # logging functions should go here (including prints)
    ├── models.py               # all torch models go here
    ├── data.py                 # all torch data loaders should go here
    ├── preprocess.py           # all functions related to processing a raw dataset go here (such as lookit - mostly code from icatcher+, or any other dataset)
    ├── options.py              # use to parse command line arguments. 


### coding conventions:
- indent: 4 spaces
- system paths: Use Path from pathlib only (no os functions)
- git push: Never push to master(main). when you want to edit, branch out, commit there, and merge when you are done.
- feel free to edit any file ! don't be shy. Changes you make affect everyone when you merge, just keep that in mind.
- code was mostly taken from Xincheng's and Peng's efforts, modified for usage as a team.
- keep project structure as generic as possible. don't open a util.py, etc.

### conda
Use conda with the environment.yml file to synchronize environments:

`conda env create --prefix /path/to/virtual/environment -f "/path/to/environment.yml"`
