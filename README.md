# lwa-exoplanet-pipeline

>Environment:
>- Red Hat Enterprise 8.6
>- g++ 8.5.0 (Makefile c++ version: c++2a)
>- cuda-11.8 (in /usr/local)
>- libraries that may need to be downloaded will depend on LDFLAGS in Makefile

## Installation

The LWA exoplanet pipeline depends on CUDA. Make sure your machine is cuda enabled and has those libraries.

The pipeline also uses IDG, which needs to be downloaded and installed separately from the [IDG repo](https://gitlab.com/astron-idg/idg).

## Include files

The lwa pipeline is dependent on a couple of included files containing the RCI-Memory-Lender and GPU kernels written for the pipeline. The *hard-coded, relative* include paths are in the Makefile as `CALIBRATION_INC_PATH` and `LENDER_INC_PATH`. 

It is assumed that the three repos, [rci-memory-lender](https://github.com/Radio-Camera-Initiative/rci-memory-lender), [lwa-exoplanet-pipeline](https://github.com/Radio-Camera-Initiative/lwa-exoplanet-pipeline) and [calibration-application](https://github.com/Radio-Camera-Initiative/calibration-application) are within the same directory. If you are on wario, these files are in `/fastpool/mlaures/` and their respective repos.

## Running the pipeline

Your machine (if it is not Wario) needs the files from three repos:
- this repo, the lwa-exoplanet-pipeline
- the calibration-application repo 
- and the rci-memory-lender repo. 

The include paths *and* the `apply.o` hard-coded path need to be changed to where you keep these files. Paths already include the repo name, do not take it out unless you have renamed the repo.

From this repo, lwa-exoplanet-pipeline, run the following commands
```
$ make
$ ./gridding /path/to/ms/files/*.ms
```
These commands will run the pipeline with all the ms files that it finds in the path you provided. 

## Seeing Images

Image results are indexed from 1. This is calibrated specifically for LWA data, other data may not be readable. 

You will need Python installed, as well as numpy and matplotlib. To see an image, you can run the Python script `img.py` which will automatically show the `1_image.npy` file. To see another file, add it as a parameter when running the script.