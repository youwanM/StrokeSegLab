# User Guide

>**⚠ This application is for research purpose only !**

## Introduction

This application is designed to perform automated segmentation and detection of stroke lesions in brain MRI images. It provides both a graphical interface (GUI) and a command line interface (CLI) to facilitate user interaction, allowing you to process data, manage models, and review results efficiently.

## Graphical Interface
This section explains how to use the graphical interface (GUI) of the application

### Launching the Application

The GUI can be launched either by double-clicking the executable or by running it from the command line without any arguments

When opening the application, a pop-up window appears on the main screen :
> This application is for research purpose only !

You cannot use the application until you close this message by clicking "OK" or the close button. If you don’t want to see the message again, you can check the “Do not show again” option. If later you want to see it at startup, there is a “Restore warning window” option in the Options menu.

### Main Window
The main window is composed of a menu bar and, below it, a large frame containing several fields.

There are two modes available: *Prediction*, which is the default mode, and *Brain Extraction Only*.  
You can switch between them using the dropdown menu at the bottom of the window.

### Input and Output Management

In both modes, at the top of the window, you’ll find the input path, which is the only required field. You can either click on *Select input folder* or *Select input file* depending on the type of input you want to process. 

The application handles BIDS and not BIDS input directory and file. If you process a file located in a BIDS directory, the outputs will be saved in the *derivatives* folder. Otherwise, they will be placed in an *output* folder located in the parent directory of your file. If you process a directory, BIDS or not, outputs will be save in the *derivatives* folder. In prediction mode, the application processes subjects as follows : 
- With a T1/FLAIR model, only subjects that have both a T1 and a FLAIR image are processed
- With a mono-channel model, all subjects with a T1 image are processed
- When *Keep MNI* mode is enabled, MNI-preprocessed files are prioritized first, followed by brain-extracted files if available 
- When *Keep MNI* mode is disabled, brain-extracted files are prioritized, and MNI-preprocessed files are ignored.

**You just need to select the root BIDS folder, the application will automatically find and organize the files from both *rawdata* and *derivatives***


### Prediction Mode

Prediction Mode is used to perform automated segmentation and to detect the presence of stroke lesions in the brain. This mode leverages trained models to analyze input images and generate prediction results for each subject

The *Suffix* field allows you to specify the suffix for the prediction output file.

The *Model* dropdown lets you choose the model. On the right, it shows whether the model uses only T1 or both T1 and FLAIR. If no models are available, a red message appears on the right, and you won't be able to run a prediction.

The *Open viewer* field lets you choose whether to open a viewer and select which viewer to use. The viewer will display the segmentation result for the first predicted subject. ITK-SNAP, FSLeyes, and medInria are the three viewers supported by the application. Only those installed on your machine will appear in the dropdown menu.

The *Output in MNI space* fields lets you choose whether you want to have the output in the MNI space or the subject space. You can’t have both at the same time. If you want MNI space, the app will look for the preprocessed MNI image. If it doesn’t find it, it will generate it during preprocessing.

### Brain Extraction Only Mode

Brain Extraction Only mode performs only the brain extraction step of preprocessing. The resulting brain-extracted images are saved in BIDS format, following the same output logic as in Prediction mode. The output files will use the suffix "BET" to indicate brain extraction.

### Running and Monitoring

Once the required fields are set (such as input path and model, if applicable), you can start the process by clicking the Run button. This applies to both Prediction and Brain Extraction Only modes. Information and log messages will be displayed to keep you updated on the progress. Please note that processing can take a significant amount of time. A Stop button will appear during execution, allowing you to interrupt the process if needed. Stopping may take several tens of seconds depending on the current progress.

### Options Menu

The following options are available from the Options menu in the menu bar at the top of the main window:

- **Save brain extracted image**: This checkbox is available only in Prediction mode. When enabled, it allows you to save the brain-extracted image during the prediction process.
- **Threshold**: Available only in Prediction mode. This option opens a window with a slider and a field, allowing you to adjust the segmentation threshold.
- **Save probability map**: This checkbox is available only in Prediction mode. When enabled, it allows you to save the probability map (before thresholding) during the prediction process.
- **Import a model**: Opens a window with three buttons: Select Model (to choose the model file), Import (to import the selected model), and Close (to close the window). A success message in green or an error message in red will be displayed. Only ONNX (.onnx) model files can be imported.

### Help Menu

The following options are available from the Help menu in the menu bar at the top of the main window:

- **Help**: Opens a window displaying the user documentation (this guide).
- **About**: Opens a window with the About section, which includes three parts:
  - **Developers**: Lists the developers involved in the project.
  - **Licence**: Provides information about the licenses.
  - **Publications**: Lists publications related to the application.

## Command Line Interface

This section explains how to use the application from the command line (CLI).
### Usage

This section explains how to use the application from the command line (CLI).

The application is launched by running `python stroke_seg.py` or `StrokeSeg.exe` if you have the Windows executbale. If no arguments are provided, the graphical interface will open by default. If you specify `--gui`, it will open the GUI with all the specified option prefilled. 

In CLI mode, you must specify either `--input` (for processing data) or `--import-model` (for model management). These options cannot be used together.

**Prediction Mode**: Segment and detect stroke lesions in brain MRI images. Requires `--input`. The `--model` option can be used to specify either the name of an already imported model or the path to a model file (useful for testing a model before importing it). Other options allow customization of output and processing.

**Brain Extraction Only Mode**: Use `--input` and `--only-preproc` to perform only the brain extraction step. No other options should be set.

**Model Management**:
- Import a new ONNX model with `--import-model /path/to/model.onnx`.
- List available models with `--import-model`.

### Options
- `--input`: Required for prediction or brain extraction. Path to the input image(s) or folder.
- `--import-model`: Required for importing a new ONNX model or listing available models. No other options can be set with this argument.
- Other options (optional, depending on mode):
  - `--model` or `-m`: Name of an imported model or path to a model file (for prediction)
  - `--suffix` or `-s`: Output file suffix
  - `--viewer` or `-V`: Viewer name (ITK-SNAP, FSLeyes, medInria)
  - `--only-preproc`: Run only brain extraction
  - `--save-preproc`: Save all preprocessing steps
  - `--keep-mni`: Save input/output images in MNI space
  - `--threshold` or `-t`: Segmentation threshold
  - `--pmap`: Save probability map
  - `-v` : Enable verbose mode (set logginh level to DEBUG instead of INFO)

### Monitoring and Logs
During execution, log messages are displayed in the terminal to keep you informed about progress, errors, and important information for each subject processed. Processing can take a significant amount of time depending on the data and options selected.


