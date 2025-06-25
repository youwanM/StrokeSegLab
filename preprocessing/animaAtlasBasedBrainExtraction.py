#!/usr/bin/python3
# Warning: works only on unix-like systems, not windows where "python animaAtlasBasedBrainExtraction.py ..." has to be run

import sys
import argparse

if sys.version_info[0] > 2:
    import configparser as ConfParser
else:
    import ConfigParser as ConfParser

import glob
import os
from shutil import copyfile, rmtree
from subprocess import call, check_output
import uuid

configFilePath = os.path.join(os.path.expanduser("~"), ".anima",  "config.txt")
if not os.path.exists(configFilePath):
    print('Please create a configuration file for Anima python scripts. Refer to the README')
    quit()

configParser = ConfParser.RawConfigParser()
configParser.read(configFilePath)

animaDir = configParser.get("anima-scripts", 'anima')
animaExtraDataDir = configParser.get("anima-scripts", 'extra-data-root')
animaPyramidalBMRegistration = os.path.join(animaDir, "animaPyramidalBMRegistration")
animaDenseSVFBMRegistration = os.path.join(animaDir, "animaDenseSVFBMRegistration")
animaTransformSerieXmlGenerator = os.path.join(animaDir, "animaTransformSerieXmlGenerator")
animaApplyTransformSerie = os.path.join(animaDir, "animaApplyTransformSerie")
animaConvertImage = os.path.join(animaDir, "animaConvertImage")
animaMaskImage = os.path.join(animaDir, "animaMaskImage")
animaCreateImage = os.path.join(animaDir, "animaCreateImage")
animaMorphologicalOperations = os.path.join(animaDir, "animaMorphologicalOperations")

# Argument parsing
parser = argparse.ArgumentParser(
    description="Computes the brain mask of images given in input by registering a known atlas on it. Their output is prefix_brainMask.nrrd and prefix_masked.nrrd")

parser.add_argument('-L', '--large-fov', action='store_true',
                    help="Specify additional processing to handle large FOV of the input image (typically for babies)."
                         "The atlas must include in that case a large FOV T1w image named: Reference_T1_largeFOV.nrrd")
parser.add_argument('-S', '--second-step', action='store_true',
                    help="Perform second step of atlas based cropping (might crop part of the external part of the brain)")

parser.add_argument('-i', '--input', type=str, required=True, help='File to process')
parser.add_argument('-a', '--atlas', type=str, help='Atlas folder (default: use the adult one in anima scripts data '
                                                    '- icc_atlas folder)')
parser.add_argument('-m', '--mask', type=str, help='Output path of the brain mask (default is inputName_brainMask.nrrd)')
parser.add_argument('-b', '--brain', type=str, help='Output path of the masked brain (default is inputName_masked.nrrd)')
parser.add_argument('-K', '--keep-intermediate-folder', action='store_true',
                    help='Keep intermediate folder after script end')

args = parser.parse_args()

numImages = len(sys.argv) - 1

atlasDir = os.path.join(animaExtraDataDir,"icc_atlas")
if args.atlas:
    atlasDir = args.atlas

atlasImage = os.path.join(atlasDir,"Reference_T1.nrrd")
atlasLargeFOVImage = os.path.join(atlasDir,"Reference_T1_largeFOV.nrrd")
atlasLargeFOVHeadMask = os.path.join(atlasDir,"Reference_T1_HeadMask.nrrd")
atlasImageFromMasked = os.path.join(atlasDir,"Reference_T1_from_masked.nrrd")
iccImage = os.path.join(atlasDir,"BrainMask.nrrd")
iccImageFromMasked = os.path.join(atlasDir,"BrainMask_from_masked.nrrd")

brainImage = args.input

if not os.path.exists(brainImage):
    sys.exit("Error: the image \"" + brainImage + "\" could not be found.")

print("Brain masking image: " + brainImage)

# Get floating image prefix
brainImagePrefix = os.path.splitext(brainImage)[0]
if os.path.splitext(brainImage)[1] == '.gz':
    brainImagePrefix = os.path.splitext(brainImagePrefix)[0]

brainMask = args.mask if args.mask else brainImagePrefix + "_brainMask.nrrd"
maskedBrain = args.brain if args.brain else brainImagePrefix + "_masked.nrrd"
intermediateFolder = os.path.join(os.path.dirname(args.input), 'brain_extract_' + str(uuid.uuid1()))

if not os.path.isdir(intermediateFolder):
    os.mkdir(intermediateFolder)

brainImagePrefix = os.path.join(intermediateFolder, os.path.basename(brainImagePrefix))

# Decide on whether to use large image setting or small image setting
command = [animaConvertImage, "-i", brainImage, "-I"]
convert_output = check_output(command, universal_newlines=True)
size_info = convert_output.split('\n')[1].split('[')[1].split(']')[0]
large_image = False
for i in range(0, 3):
    size_tmp = int(size_info.split(', ')[i])
    if size_tmp >= 350:
        large_image = True
        break

pyramidOptions = ["-p", "4", "-l", "1"]
if large_image:
    pyramidOptions = ["-p", "5", "-l", "2"]

# If large FOV image, use the large FOV part of the atlas
fovOptions = []
if args.large_fov is True:
    command = [animaPyramidalBMRegistration, "-m", atlasLargeFOVImage, "-r", brainImage,
               "-o", brainImagePrefix + "_lfov_rig.nrrd",
               "-O", brainImagePrefix + "_lfov_rig_tr.txt", "--sp", "3"] + pyramidOptions
    call(command)

    command = [animaPyramidalBMRegistration, "-m", atlasLargeFOVImage, "-r", brainImage,
               "-o", brainImagePrefix + "_lfov_aff.nrrd",
               "-O", brainImagePrefix + "_lfov_aff_tr.txt", "-i", brainImagePrefix + "_lfov_rig_tr.txt", "--sp", "3",
               "--ot", "2"] + pyramidOptions
    call(command)

    command = [animaTransformSerieXmlGenerator, "-i", brainImagePrefix + "_lfov_aff_tr.txt",
               "-o", brainImagePrefix + "_lfov_aff_tr.xml"]
    call(command)

    command = [animaApplyTransformSerie, "-i", atlasLargeFOVHeadMask, "-t", brainImagePrefix + "_lfov_aff_tr.xml",
               "-o", brainImagePrefix + "_lfov_cropMask.nrrd", "-g", brainImage, "-n", "nearest"]
    call(command)

    command = [animaMaskImage, "-i", brainImage, "-m", brainImagePrefix + "_lfov_cropMask.nrrd",
               "-o", brainImagePrefix + "_lfov_cropped.nrrd"]
    call(command)

    brainImage = brainImagePrefix + "_lfov_cropped.nrrd"
    fovOptions = ["-i", brainImagePrefix + "_lfov_aff_tr.txt"]

# Rough mask with whole brain
command = [animaPyramidalBMRegistration, "-m", atlasImage, "-r", brainImage, "-o", brainImagePrefix + "_rig.nrrd",
           "-O", brainImagePrefix + "_rig_tr.txt", "--sp", "3"] + pyramidOptions + fovOptions
call(command)

command = [animaPyramidalBMRegistration, "-m", atlasImage, "-r", brainImage, "-o", brainImagePrefix + "_aff.nrrd",
           "-O", brainImagePrefix + "_aff_tr.txt", "-i", brainImagePrefix + "_rig_tr.txt", "--sp", "3", "--ot",
           "2"] + pyramidOptions
call(command)

command = [animaCreateImage, "-g", atlasImage, "-b", "1", "-o", brainImagePrefix + "_baseCropMask.nrrd"]
call(command)

command = [animaTransformSerieXmlGenerator, "-i", brainImagePrefix + "_aff_tr.txt",
           "-o", brainImagePrefix + "_aff_tr.xml"]
call(command)

command = [animaApplyTransformSerie, "-i", brainImagePrefix + "_baseCropMask.nrrd",
           "-t", brainImagePrefix + "_aff_tr.xml", "-g", brainImage, "-o",
           brainImagePrefix + "_cropMask.nrrd", "-n", "nearest"]
call(command)

command = [animaMaskImage, "-i", brainImage, "-m", brainImagePrefix + "_cropMask.nrrd",
           "-o", brainImagePrefix + "_c.nrrd"]
call(command)

command = [animaDenseSVFBMRegistration, "-r", brainImagePrefix + "_c.nrrd", "-m", brainImagePrefix + "_aff.nrrd",
           "-o", brainImagePrefix + "_nl.nrrd", "-O", brainImagePrefix + "_nl_tr.nrrd", "--tub", "2"] + pyramidOptions
call(command)

command = [animaTransformSerieXmlGenerator, "-i", brainImagePrefix + "_aff_tr.txt", "-i",
           brainImagePrefix + "_nl_tr.nrrd", "-o", brainImagePrefix + "_nl_tr.xml"]
call(command)

command = [animaApplyTransformSerie, "-i", iccImage, "-t", brainImagePrefix + "_nl_tr.xml", "-g", brainImage, "-o",
           brainImagePrefix + "_rough_brainMask.nrrd", "-n", "nearest"]
call(command)

command = [animaMaskImage, "-i", brainImage, "-m", brainImagePrefix + "_rough_brainMask.nrrd", "-o",
           brainImagePrefix + "_rough_masked.nrrd"]
call(command)

brainImageRoughMasked = brainImagePrefix + "_rough_masked.nrrd"

if args.second_step is True:
    # Fine mask with masked brain
    command = [animaMorphologicalOperations, "-i", brainImagePrefix + "_rough_brainMask.nrrd",
               "-o", brainImagePrefix + "_rough_brainMask_dil.nrrd", "-a", "dil", "-r", "5", "-R"]
    call(command)

    command = [animaMaskImage, "-i", brainImage, "-m", brainImagePrefix + "_rough_brainMask_dil.nrrd", "-o",
               brainImageRoughMasked]
    call(command)

    command = [animaPyramidalBMRegistration, "-m", atlasImageFromMasked, "-r", brainImageRoughMasked, "-o",
               brainImagePrefix + "_masked_aff.nrrd", "-O", brainImagePrefix + "_masked_aff_tr.txt", "-i",
               brainImagePrefix + "_aff_tr.txt", "--sp", "3", "--ot", "2"] + pyramidOptions
    call(command)

    command = [animaDenseSVFBMRegistration, "-r", brainImageRoughMasked, "-m", brainImagePrefix + "_masked_aff.nrrd", "-o",
               brainImagePrefix + "_masked_nl.nrrd", "-O", brainImagePrefix + "_masked_nl_tr.nrrd", "--tub", "2"] + pyramidOptions
    call(command)

    command = [animaTransformSerieXmlGenerator, "-i", brainImagePrefix + "_masked_aff_tr.txt", "-i",
               brainImagePrefix + "_masked_nl_tr.nrrd", "-o", brainImagePrefix + "_masked_nl_tr.xml"]
    call(command)

    command = [animaApplyTransformSerie, "-i", iccImageFromMasked, "-t", brainImagePrefix + "_masked_nl_tr.xml",
               "-g", brainImage, "-o", brainMask, "-n", "nearest"]
    call(command)

    command = [animaMaskImage, "-i", brainImage, "-m", brainMask, "-o", maskedBrain]
    call(command)
else:
    command = [animaConvertImage, "-i", brainImageRoughMasked, "-o", maskedBrain]
    call(command)
    command = [animaConvertImage, "-i", brainImagePrefix + "_rough_brainMask.nrrd", "-o", brainMask]
    call(command)

if not args.keep_intermediate_folder:
    rmtree(intermediateFolder)