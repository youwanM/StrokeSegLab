# User Guide
>**⚠ This application is for research purpose only !**

## Graphical Interface
This section explains how to use the graphical interface (GUI) of the application

The GUI can be launched either by double-clicking the executable or by running it from the command line without any arguments

When opening the application, a pop-up window appears on the main screen :
> This application is for research purpose only !

You cannot use the application until you close this message by clicking "OK" or the close button

The main window is composed of a menu bar and, below it, a large frame containing several fields.

There are two modes available: *Prediction*, which is the default mode, and *Brain Extraction Only*.  
You can switch between them using the dropdown menu at the bottom of the window.

In both modes, at the top of the window, you’ll find the input path, which is the only required field. You can either click on *Select input folder* or *Select input file* depending on the type of input you want to process. 

The application handles BIDS and not BIDS input directory and file. If you process a file located in a BIDS directory, the outputs will be saved in the *derivatives* folder. Otherwise, they will be placed in an *output* folder located in the parent directory of your file. If you process a directory, BIDS or not, outputs will be save in the *derivatives* folder. In prediction mode, the application processes subjects as follows : 
- With a T1/FLAIR model, only subjects that have both a T1 and a FLAIR image are processed
- With a mono-channel model, all subjects with a T1 image are processed
- When *Keep MNI* mode is enabled, MNI-preprocessed files are prioritized first, followed by brain-extracted files if available 
- When *Keep MNI* mode is disabled, brain-extracted files are prioritized, and MNI-preprocessed files are ignored.

**You just need to select the root BIDS folder, the application will automatically find and organize the files from both *rawdata* and *derivatives***

In Prediction mode : 
The *Suffix* field allows you to specify the suffix for the prediction output file. 
The *Model* dropdown lets you choose the model. On the right, it shows whether the model uses only T1 or both T1 and FLAIR. If no models are available, a red message appears on the right, and you won't be able to run a prediction.
The *Open viewer* field lets you choose whether to open a viewer and select which viewer to use. The viewer will display the segmentation result for the first predicted subject.
The *Output in MNI space* fields lets you choose whether you want to have the output in the MNI space or the subject space. You can’t have both at the same time. If you want MNI space, the app will look for the preprocessed MNI image. If it doesn’t find it, it will generate it during preprocessing.
