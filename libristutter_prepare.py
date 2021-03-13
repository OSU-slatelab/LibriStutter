import os
import csv
import json
import string
import logging
import torchaudio
from g2p_en import G2p
from speechbrain.utils.data_utils import get_all_files, download_file
from speechbrain.dataio.dataio import read_audio

logger = logging.getLogger(__name__)
SAMPLE_RATE = 16000

G2P = G2p()
NOT_PHONEMES = [" ", ".", ",", "!", "?"]

# SPEAKERS = [
#    78, 87, 103, 198, 289, 311, 412, 445, 625, 831, 1088, 1334, 1502, 1743,
#    1867, 1926, 1963, 2007, 2092, 2289, 2436, 2518, 2836, 2893, 2952, 2989,
#    3240, 3699, 4018, 4137, 4788, 4830, 4853, 5390, 5393, 5456, 5561, 5678,
#    6000, 6437, 6925, 7067, 7113, 7517, 7780, 8095, 8098, 8238, 8468, 8975,
# ]
SPEAKERS = [
    103, 198, 289, 311, 412, 445, 1088, 1334, 1502, 1743, 1867, 1926, 1963,
    2007, 2092, 2289, 2436, 2518, 2836, 2893, 2952, 2989, 3240, 3699, 4018,
    4137, 4788, 4830, 4853, 5390, 5393, 5456, 5561, 5678,
]

# The indices correspond to the number in the annotation.
STUTTER_TYPE = [
    ".", "interjection", "sound_rep", "word_rep", "phrase_rep", "prolongation"
]


def prepare_libristutter(
    data_folder,
    train_manifest,
    valid_manifest,
    test_manifest,
    valid_spk_count=2,
    test_spk_count=2,
    max_len=10,
):
    "Create the .json manifest files necessary for DynamicItemDataset"

    # Make sure a sample of the files are correct
    if not check_folder(data_folder):
        raise ValueError("{data_folder} doesn't contain LibriStutter")

    # Look for prior preparation
    if skip_prep(train_manifest, valid_manifest, test_manifest):
        logger.info("Preparation already completed, skipping.")
        return

    # Collect files. Include dir separators to prevent spurious matches
    valid_speakers = SPEAKERS[: valid_spk_count]
    test_speakers = SPEAKERS[valid_spk_count: valid_spk_count + test_spk_count]
    extension_matches = [".flac"]
    valid_speaker_matches = [f"/{s}/" for s in valid_speakers]
    test_speaker_matches = [f"/{s}/" for s in test_speakers]
    train_filelist = get_all_files(
        data_folder,
        match_and=extension_matches,
        exclude_or=valid_speaker_matches + test_speaker_matches,
    )
    valid_filelist = get_all_files(
        data_folder,
        match_and=extension_matches,
        match_or=valid_speaker_matches,
    )
    test_filelist = get_all_files(
        data_folder,
        match_and=extension_matches,
        match_or=test_speaker_matches,
    )

    # Create json files
    create_manifest(train_manifest, train_filelist, max_len)
    create_manifest(valid_manifest, valid_filelist, max_len)
    create_manifest(test_manifest, test_filelist, max_len)


def skip_prep(*filenames):
    "Check all passed files"
    for filename in filenames:
        if not os.path.isfile(filename):
            return False
    return True


def check_folder(data_folder):
    "Check for main folders to exist"
    #for part in [1, 2, 3]:
    for part in [1, 2]:
        folder = os.path.join(data_folder, f"LibriStutter Part {part}")
        if not (
            os.path.isdir(folder)
            and os.path.isdir(os.path.join(folder, "LibriStutter Annotations"))
            and os.path.isdir(os.path.join(folder, "LibriStutter Audio"))
        ):
            return False
    return True


def create_manifest(json_file, filelist, max_len):
    "Read all files and create manifest file with annotations"
    manifest = {}
    for filepath in filelist:

        # Load to compute length for sorting etc.
        audio, rate = torchaudio.load(filepath)
        length = audio.size(1) / rate

        # Split path into folders
        path_parts = filepath.split(os.path.sep)
        uttid = path_parts[-1][:-len(".flac")]
        rel_path = os.path.join("{data_root}", *path_parts[-5:])

        # Compute annotation path
        annotation_path_parts = path_parts.copy()
        annotation_path_parts[-1] = uttid + ".csv"
        annotation_path_parts[-4] = "LibriStutter Annotations"
        annot_path = os.path.sep + os.path.join(*annotation_path_parts)

        # Create entry
        annotation_list = compute_annotations(rel_path, length, annot_path, max_len)
        for i, entry in enumerate(annotation_list):
            manifest[f"{uttid}-{i}"] = {**entry}

    # Write annotations to file
    with open(json_file, "w") as w:
        json.dump(manifest, w, indent=2)

    logger.info(f"Finished creating {json_file}")


def compute_annotations(rel_path, length, annotation_path, max_len):
    "Read annotations from file and create annotations list. May involve chunking."
    annotation_list = []
    annotation_types = "words", "breaks", "stutter_type", "phonemes"
    stored_stutter = ""

    # Divide up length evenly into smallest number of chunks that still fits
    max_chunk_length = length / (length // max_len + 1)
    with open(annotation_path) as annotation_csv:
        csv_reader = csv.reader(annotation_csv)
        chunk_start = 0.0
        annotation = {t: [] for t in annotation_types}
        for row in csv_reader:
            word_start = float(row[1]) - chunk_start

            # Format all the data
            word = row[0]
            # Make these have only two decimal places
            break_time = "{:.2f}".format(word_start)
            # Convert stutter type from number to string
            stutter = STUTTER_TYPE[int(row[3])]
            # Use G2P system to convert words to phonemes
            phonemes = word2phonemes(word.lower())

            # Append the annotations for this chunk to list
            # Last thing cannot be a stutter, cuz it affects next word
            if word_start > max_chunk_length and not stored_stutter:
                # Append final break (end of word)
                annotation["breaks"].append(break_time)
                annotation["length"] = word_start
                annotation["wav"] = format_wav_annotation(
                    rel_path, chunk_start, chunk_start + word_start
                )
                annotation_list.append(annotation)

                # Prepare for next chunk
                chunk_start += word_start
                word_start = 0.0
                break_time = "0.00"
                annotation = {t: [] for t in annotation_types}

            # Append this word to prev/post if its a stutter
            if row[0] == "STUTTER":
                # prolongation stutter affects PREV word.
                # Skip adding the beginning of the word to breaks
                # so that the previous word includes the stutter.
                if stutter == "prolongation":
                    # Change prev word stutter type to prolongation
                    if annotation["stutter_type"]:
                        annotation["stutter_type"][-1] = stutter

                # In all other cases, the stutter affects the NEXT word.
                else:
                    stored_stutter = stutter

                    # Still append the break here, the NEXT word shouldn't.
                    annotation["breaks"].append(break_time)

            # Previous word was STUTTER, adjust this word
            elif stored_stutter:
                # Skip adding the beginning of the word to breaks
                # cuz this word will glom onto the STUTTER
                annotation["words"].append(word)
                annotation["stutter_type"].append(stored_stutter)
                annotation["phonemes"].append(phonemes)

                # Reset stored stutter so next word is normal
                stored_stutter = ""

            # Normal word, add items as ususal
            else:
                annotation["words"].append(word)
                annotation["breaks"].append(break_time)
                annotation["stutter_type"].append(stutter)
                annotation["phonemes"].append(phonemes)

            # Sometimes annotations go past end of utterance, this means
            # the wrong transcription is used, so just return nothing
            if float(row[2]) > length:
                print("Illegal annotation for utt:", rel_path)
                return []

        # Append final break (end of last word)
        word_end_time = float(row[2]) - chunk_start
        annotation["breaks"].append("{:.2f}".format(word_end_time))

    # Append last annotation
    annotation["length"] = length - chunk_start
    annotation["wav"] = format_wav_annotation(rel_path, chunk_start, length)
    annotation_list.append(annotation)

    # Combine arrays into strings for json readability
    for annotations in annotation_list:
        for key in annotation_types:
            annotations[key] = " ".join(annotations[key])

    return annotation_list


def format_wav_annotation(rel_path, start, stop):
    return {
        "file": rel_path,
        "start": int(start * SAMPLE_RATE),
        "stop": int(stop * SAMPLE_RATE),
    }


def word2phonemes(word):
    "Convert all words to phonemes other than 'STUTTER'."
    if word == "&" or word == "+":
        word = "and"
    elif word == "/":
        word = "purr"
    elif word == "*":
        word = "times"
    return ".".join(p.strip("012") for p in G2P(word) if p not in NOT_PHONEMES)
