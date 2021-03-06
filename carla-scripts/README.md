# How to setup the Python environment for CARLA + CIL
This guide provides the exact steps to set up a Conda environment that supports running CARLA, keras_segmentation, and Python 3.6 together.
1. Create a new Conda environment using Python 3.6
2. Run `conda install -c anaconda tensorflow-gpu=1.13.1`
3. Run `conda install -c conda-forge keras==2.2.4`
4. Run `easy_install carla-scripts/carla/dist/carla-0.9.7-py3.5-linux-x86_64.egg` (replace with path to your carla egg-file)
5. Run `pip install --editable git+https://github.com/divamgupta/image-segmentation-keras#egg=keras_segmentation --upgrade` or
`pip install --editable src/keras-segmentation --upgrade` if you have the repo locally already
6. Run `conda install -c anaconda networkx`
7. Run `pip install pygame`
IMPORTANT: Ensure that you put `import tensorflow` before any `import carla` in your scripts.

## Upgrading to a new CARLA version
1. It's recommended that you clone your Conda environment in case you need to go back. Run `conda create --name new_env --clone old_env`, replacing the names.
2. Install the new version of the CARLA library for Python: `easy_install carla-scripts/carla/dist/carla-0.9.7-py3.5-linux-x86_64.egg` (replace with path to your carla egg-file)

# File structure
* carla/: Taken from CARLA's own PythonAPI folder.
  * agents: Modified versions of CARLA's Python-based autopilots.
  * dist: The Python API egg-files, not needed as we used `easy_install` to install them already.
  