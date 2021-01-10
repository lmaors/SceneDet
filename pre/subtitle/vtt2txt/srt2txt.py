#!/usr/bin/python3
# -*- coding: UTF-8 -*-
"""
    Converter from SRT to TXT format.

    Usage: srt2txt.py [-h] [-v] [-i INPUTFILE] [-o [OUTPUTFILE]] [-j [JOIN]] [-c [CLEAN]]

      -h, --help                    show this help message and exit
      -v, --version                 show version of the program
      -i, --inputfile INPUTFILE     input file name
      -o, --outputfile OUTPUTFILE   output file name
      -j, --join 0|1                join lines (0 - no, 1 - yes)
      -c, --clean 0|1               clean SRT file of HTML markup (0 - no, 1 - yes) and exit

    Example:
    python3 srt2txt.py -i The-Fate-of-the-First-Stars-Space-Time.srt
    :copyright: (c) 2019 by Doctor_Che
    :license: GPLv3, see LICENSE for more details.
"""

import sys
import argparse
import os.path
import re

VERSION = "0.2.1"


def create_parser():
    # Create a parser class
    parser = argparse.ArgumentParser(
        prog='srt2txt.py',
        description='''Program for translating SRT subtitle files into TXT text files''',
        epilog='''(c) Doctor_Che 2018. The author of the program, as always, does not bear any responsibility for anything''',
        add_help=False
        )


    parser.add_argument('-h', '--help', action='help', help="Справка")
    parser.add_argument('-v', '--version',
                        action='version',
                        help='Print version number',
                        version='%(prog)s {}'.format(VERSION))
    parser.add_argument('-i', '--inputfile',
                        default="inputfile.srt",
                        type=argparse.FileType('r'),
                        help="input file")
    parser.add_argument('-o', '--outputfile',
                        nargs='?',
                        type=argparse.FileType('w'),
                        help="output file")
    parser.add_argument('-j', '--join',
                        nargs='?',
                        default=0,
                        type=int,
                        help="Combining strings into sentences")
    parser.add_argument('-c', '--clean',
                        nargs='?',
                        default=0,
                        type=int,
                        help="Cleaning a file from HTML markup")
    return parser


def convert_srt_to_txt(text, join=False):
    """
    Removing information lines from file
    :param text: String in SRT format
    :param join: Merge lines into sentences
    :return: String in TXT format
    """
    lines = text.split('\n')
    result = []
    for line in lines:
        if not line.strip():  # Skipping empty lines
            continue
        elif line.strip().isdigit():  # Skip lines containing only numbers
            continue
        elif (line.startswith("WEBVTT") or
              line.startswith("Kind: captions") or
              line.startswith("Language: en")):  # Skipping lines containing service information
            continue
        # We skip lines with the format "00:00:00,000 --> 00:00:03,090"
        elif re.match(r"\d{2}:\d{2}:\d{2}.\d{3} --> \d{2}:\d{2}:\d{2}.\d{3}", line.strip()):
            continue
        else:
            result.append(line.strip())
    if join:
        out = join_lines(result)  # Combining strings into sentences
    else:
        out = "\n".join(result)  # Combining strings without parsing into sentences
    return out


def join_lines(lst):
    """
    Merge lines into sentences
    :param lst: List of strings
    :return: String with sentences
    """
    out = ""
    for line in lst:
        if line.startswith("[") and line.endswith("]"):
            out = f"{out}\n{line}\n"
        elif out.endswith("."):
            out = f"{out}\n{line}"
        else:
            out = f"{out} {line}"
    return out


def clean_srt(text_srt):
    """
    从文本中删除HTML标记
    """
    return re.sub(r"</?font.*?>", "", text_srt)


def main():
    output_file = ""
    try:
        parser = create_parser()
        namespace = parser.parse_args(sys.argv[1:])

        text_srt = namespace.inputfile.read()  #Reading the source text from the file

        base, _ = os.path.splitext(namespace.inputfile.name)

        if namespace.clean:
            text_srt_cleaned = clean_srt(text_srt)  # Removing HTML markup from text
            cleaned_file = f"{base}_cleaned.srt"  # Get the path for the cleaned file
            with open(cleaned_file, "w") as fout:
                fout.write(text_srt_cleaned)
        else:
            text_txt = convert_srt_to_txt(text_srt, namespace.join)

            # Writing the converted text to a file
            if namespace.outputfile:
                output_file = namespace.outputfile.name
                namespace.outputfile.write(text_txt)
            else:
                output_file = f"{base}.txt"  # Get the path for the output file
                with open(output_file, "w") as fout:
                    # merge strings
                    text_txt = text_txt.replace('\n', ' ').replace('\r', ' ').replace('&gt;',' ').lower()
                    fout.write(text_txt)

    except IOError:
        print("An error occurred during conversion")
    else:
        print("File conversion was successful")
        print(f"Saved file: '{output_file}'")


if __name__ == '__main__':
    main()
