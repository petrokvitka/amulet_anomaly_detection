#!/usr/bin/env python

"""
This script creates a database with MFCCs created from the audio files
for the DCGAN training.

@author: Anastasiia Petrova
@contact: petrokvitka@gmail.com

"""

import os
import argparse
import sys
import textwrap
import matplotlib.pyplot as plt

import librosa

print("Imports are ready!")

def parse_args():
    """
    This function reads the input from user.
    :returns args: list of parameters which were set by the user or defaults
    """
    parser = argparse.ArgumentParser(prog = 'create_mfccs_database', description = textwrap.dedent('''
        This script creates the database with MFCCs from the audio files for the DCGAN training.
            '''),
        epilog = 'That is what you need to make this script work for  you. Enjoy it!')

    # ---------- parameters for training ----------
    parser.add_argument('--input_dir', help = 'Set the directory with input files for the transformation.')
    parser.add_argument('--input_file', help = 'Pass the file in wav format for the transformation.')
    parser.add_argument('--output_dir', help = 'Set the output directory where the database will be saved.')
    parser.add_argument('--prefix', help = 'Provide a prefix fo the MFCCs, for example bad, good, defect...', default = "train")

    args = parser.parse_args()
    return args


def check_existing_input(args):
    if not args.input_dir and not args.input_file:
        print("Please provide a directory with wav files or a singla wav file!\nExit is forced!")
        sys.exit()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
        print("A new directory {} is created!".format(args.output_dir))

    if not os.path.exists(os.path.join(args.output_dir, args.prefix)):
        os.makedirs(os.path.join(args.output_dir, args.prefix))
        print("A new directory {} is created!".format(os.path.join(args.output_dir, args.prefix)))


def check_wav_length(wav, sr):
    if len(wav)%sr != 0:
        wav = wav[:len(wav) - (len(wav)%sr)]
    return wav

def main_script():
    """
    This function does all the essential work.
    """
    args = parse_args()
    check_existing_input(args)

    wavs = []
    wavs_names = []
    if args.input_dir:
        for f in os.listdir(args.input_dir):
            if f.endswith(".wav"):
                wav, sr = librosa.load(os.path.join(args.input_dir, f))

                wav = check_wav_length(wav, sr)

                wavs.append(wav)
                wavs_names.append(os.path.basename(f))
            else:
                continue

    else:
        wav, sr = librosa.load(args.input_file)

        wav = check_wav_length(wav, sr)

        wavs.append(wav)
        wavs_names.append(os.path.basename(args.input_file))

    file_names = []
    for i in range(0, len(wavs)):
        for j in range(0, len(wavs[i]), sr):
            mfcc = librosa.feature.mfcc(wavs[i][j : j + sr])

            wav_name = wavs_names[i].replace('.wav', '')
            img_name = str(wav_name + "_{}.jpg".format(j))
            file_names.append(img_name)

            plt.imsave(os.path.join(args.output_dir, args.prefix, img_name), arr = mfcc, format = "jpg")

            j += sr

    f = open(os.path.join(args.output_dir, "{}_files.txt".format(args.prefix)), "w")
    f.write("\n".join(file_names))
    f.close()


if __name__ == '__main__':
    main_script()
