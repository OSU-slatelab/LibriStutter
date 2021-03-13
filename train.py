"""
Recipe for training a disfluency detection system on the LibriStutter dataset.

Authors
 * Peter Plantinga 2021
"""
import sys
import torch
import string
import speechbrain as sb
from hyperpyyaml import load_hyperpyyaml
from speechbrain.dataio.dataio import length_to_mask
from libristutter_prepare import prepare_libristutter
from speechbrain.utils.data_utils import batch_pad_right
from speechbrain.dataio.batch import PaddedBatch, PaddedData

SAMPLE_RATE = 16000
SEGMENT_LENGTH = 4.0
SEGMENT_HOP = 1.0
POST_SEGMENT_WORD_COUNT = 3


def time_reduce(x, factor):
    "Concatenate frames to reduce the time dimension"

    # Ensure time-dimension is divisible by factor
    frames_to_remove = x.size(1) % factor

    # Remove frames randomly
    leftmost = torch.randint(frames_to_remove + 1, (1,)).item()
    rightmost = x.size(1) - frames_to_remove + leftmost
    x = x[:, leftmost:rightmost]

    # Pick random arrangement of concatenated frames
    shape = [x.size(0), x.size(1) // factor, x.size(2) * factor]
    return x.view(shape)


class DetectBrain(sb.Brain):
    """Use attentional model to predict words in segments"""

    def compute_feats(self, wavs, lens, stage):
        feats = self.hparams.compute_feats(wavs)
        feats = time_reduce(feats, factor=self.hparams.time_reduce)
        feats = self.hparams.normalize(
            feats, lens, epoch=self.hparams.counter.current
        )

        # Augment
        if stage == sb.Stage.TRAIN:
            feats = self.hparams.spec_augment(feats)

        return feats

    def embed_words(self, word_lists):
        """Returns prompted words in format suitable to attentional model.

        Both sentences and words have variable length. This method
        sums embedded characters to achieve fixed-length word vectors,
        then returns a padded sentence tensor.

        Arguments
        ---------
        word_lists : list
            List of PaddedData objects for variable length words
        """
        sentences = []
        for word_list in word_lists:
            words, lens = word_list
            words = words.to(self.device)
            words_embedded = self.modules.phoneme_embedding(words)
            words_embedded, _ = self.modules.phn2word_embedding(
                words_embedded, lengths=lens
            )
            mask = length_to_mask(lens, words.size(1), device=self.device)
            masked_words = words_embedded * mask.unsqueeze(-1)
            # We need time dim last for padding
            sentences.append(masked_words.sum(dim=1).transpose(0, 1))

        # Combine, then put time dim in expected place
        tensors, lengths = batch_pad_right(sentences)
        return tensors.transpose(1, 2), lengths

    def compute_forward(self, batch, stage):
        batch = batch.to(self.device)
        wavs, lens = batch.sig
        feats = self.compute_feats(wavs, lens, stage)

        # Use encoder on features for easier recognition
        encoded_feats, _ = self.modules.encoder(feats)

        # Combine phoneme embeddings to form word embedding
        sentences, sentence_lengths = self.embed_words(batch.phonemes_encoded)

        # Use attentional model to make predictions for each word
        outputs, _ = self.modules.prediction_model(
            sentences, encoded_feats, lens
        )
        existence_output = self.modules.existence_output(outputs)

        predictions = {
            "existence": existence_output,
        }

        return predictions

    def compute_objectives(self, predictions, batch, stage):

        word_existences, existence_lengths = batch.word_existences
        existence_loss = sb.nnet.losses.bce_loss(
            predictions["existence"], word_existences, existence_lengths
        )
        if stage != sb.Stage.TRAIN:
            # Round existences to nearest integer
            rounded_targets = torch.round(word_existences)

            # Flatten arrays
            flat_predictions = torch.sigmoid(predictions["existence"]).view(-1)
            flat_targets = rounded_targets.view(-1)
            self.existence_metrics.append(
                batch.id, flat_predictions, flat_targets
            )

        # stutter_target, stutter_lengths = batch.stutter_encoded
        # stutter_loss = sb.nnet.losses.nll_loss(
        #     predictions["stutter"], stutter_target, stutter_lengths
        # )
        # self.stutter_metrics.append(...)
        stutter_loss = 0

        return existence_loss + stutter_loss

    def on_stage_start(self, stage, epoch):
        "Gets called when a stage (either training, validation, test) starts."
        if stage != sb.Stage.TRAIN:
            self.existence_metrics = sb.utils.metric_stats.BinaryMetricStats()
            # self.stutter_metrics = sb.utils.metric_stats.BinaryMetricStats()

    def on_stage_end(self, stage, stage_loss, epoch):
        """Gets called at the end of a stage."""
        if stage == sb.Stage.TRAIN:
            self.train_loss = stage_loss
        else:
            stage_stats = {"loss": stage_loss}
            stage_stats["exist"] = self.existence_metrics.summarize(
                field="F-score", threshold=0.5
            ) * 100
            # stage_stats["stutter"] = self.stutter_metrics.summarize()

        if stage == sb.Stage.VALID:
            self.hparams.train_logger.log_stats(
                stats_meta={"epoch": epoch},
                train_stats={"loss": self.train_loss},
                valid_stats=stage_stats,
            )
            self.checkpointer.save_and_keep_only(
                meta=stage_stats, min_keys=["loss"],
            )

        elif stage == sb.Stage.TEST:
            self.hparams.train_logger.log_stats(
                stats_meta={"Epoch loaded": self.hparams.counter.current},
                test_stats=stage_stats,
            )


def segmenting_collate_fn(examples):
    """Takes a set of examples and segments them based on word boundaries.

    Arguments
    ---------
    examples : list
        List of data dicts from dataset class, with keys:
         * "sig": The full-length audio signal.
         * "splits": List of timings of word boundaries.
         * "phonemes_encoded": List of words (in phonetic form).

    Returns
    -------
    batch : PaddedBatch
        The compiled batch with each entry corresponding to a segment.
    """
    segments = []
    for example in examples:

        # Iterate segments
        length = len(example["sig"]) / SAMPLE_RATE
        max_index = int(max(length - SEGMENT_LENGTH, 0) / SEGMENT_HOP) + 1
        for seg_index in range(max_index):
            segments.append(compute_segment(example, seg_index, length))

    return PaddedBatch(segments)


def compute_segment(example, seg_index, length):
    """From one example and segment index, compute everything for that segment.

    Arguments
    ---------
    example : dict
        One element of collate_fn list, see ``segmenting_collate_fn``
    seg_index : int
        The index, using SEGMENT_HOP to determine starting location.
    length : float
        The full length of utterance, segment can't go past this.

    Returns
    -------
    dict
        Entry suitable for passing to PaddedBatch
    """
    # Compute segment wav tensor
    seg_start = seg_index * SEGMENT_HOP
    seg_end = min(seg_start + SEGMENT_LENGTH, length)
    sig = example["sig"][
        int(seg_start * SAMPLE_RATE):int(seg_end * SAMPLE_RATE)
    ]

    # Iterate splits to find if word is inside segment
    word_existences = []
    word_inputs = []
    stutters = []
    post_count = 0
    joint_times = zip(example["splits"][:-1], example["splits"][1:])
    for i, (start, end) in enumerate(joint_times):

        # Word is before the segment.
        if end < seg_start:
            word_existences.append(0.0)

        # Word is after the segment. Add a few after end, then stop.
        elif start > seg_end:
            if post_count < POST_SEGMENT_WORD_COUNT:
                word_existences.append(0.0)
                post_count += 1
            else:
                break

        # Part of the word in the segment, label is ratio.
        elif start < seg_start:
            word_existences.append((end - seg_start) / (end - start))
        elif end > seg_end:
            word_existences.append((seg_end - start) / (end - start))

        # Whole word is inside the segment.
        else:
            word_existences.append(1.0)

        if i < len(example["phonemes_encoded"]):
            word_inputs.append(example["phonemes_encoded"][i])
        else:
            print("PHONEMES TOO SHORT")
            print(example["id"])
        stutters.append(example["stutter_encoded"][i])

    return {
        "id": example["id"] + f"_{seg_index}",
        "sig": sig,
        "phonemes_encoded": PaddedData(*batch_pad_right(word_inputs)),
        "word_existences": torch.FloatTensor(word_existences),
        "stutter_encoded": torch.LongTensor(stutters),
    }


def dataio_prep(hparams):
    "Prepare datasets and data pipelines"
    phoneme_encoder = sb.dataio.encoder.TextEncoder()
    stutter_encoder = sb.dataio.encoder.TextEncoder()

    @sb.utils.data_pipeline.takes("wav")
    @sb.utils.data_pipeline.provides("sig")
    def audio_pipeline(wav):
        return sb.dataio.dataio.read_audio(wav)

    @sb.utils.data_pipeline.takes("phonemes")
    @sb.utils.data_pipeline.provides("phoneme_list", "phonemes_encoded")
    def word_pipeline(phonemes):
        word_list = [p.split(".") for p in phonemes.strip().split()]
        yield [p for word in word_list for p in word]
        yield [phoneme_encoder.encode_sequence_torch(w) for w in word_list]

    @sb.utils.data_pipeline.takes("breaks")
    @sb.utils.data_pipeline.provides("splits")
    def split_pipeline(breaks):
        splits = [float(f) for f in breaks.strip().split()]
        return torch.FloatTensor(splits)

    @sb.utils.data_pipeline.takes("stutter_type")
    @sb.utils.data_pipeline.provides("stutter_list", "stutter_encoded")
    def stutter_pipeline(stutter_type):
        stutter_list = stutter_type.strip().split()
        yield stutter_list
        yield stutter_encoder.encode_sequence_torch(stutter_list)

    datasets = {}
    for dataset in ["train", "valid", "test"]:
        datasets[dataset] = sb.dataio.dataset.DynamicItemDataset.from_json(
            json_path=hparams[f"{dataset}_manifest"],
            replacements={"data_root": hparams["data_folder"]},
            dynamic_items=[
                audio_pipeline, word_pipeline, split_pipeline, stutter_pipeline
            ],
            output_keys=[
                "id", "sig", "phonemes_encoded", "splits", "stutter_encoded"
            ],
        )

    hparams["dataloader_opts"]["collate_fn"] = segmenting_collate_fn

    phoneme_encoder.update_from_didataset(
        datasets["train"], output_key="phoneme_list"
    )
    stutter_encoder.update_from_didataset(
        datasets["train"], output_key="stutter_list"
    )

    return datasets, phoneme_encoder, stutter_encoder


if __name__ == "__main__":
    # Load hyperparameters file with command-line overrides
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])
    with open(hparams_file) as fin:
        hparams = load_hyperpyyaml(fin, overrides)

    # Create experiment directory
    sb.create_experiment_directory(
        experiment_directory=hparams["output_folder"],
        hyperparams_to_save=hparams_file,
        overrides=overrides,
    )

    # Prepare data
    prepare_libristutter(
        hparams["data_folder"],
        hparams["train_manifest"],
        hparams["valid_manifest"],
        hparams["test_manifest"],
    )
    datasets, phoneme_encoder, stutter_encoder = dataio_prep(hparams)

    # Initialize trainer
    detect_brain = DetectBrain(
        modules=hparams["modules"],
        opt_class=hparams["opt_class"],
        run_opts=run_opts,
        hparams=hparams,
        checkpointer=hparams["checkpointer"],
    )
    detect_brain.phoneme_encoder = phoneme_encoder
    detect_brain.stutter_encoder = stutter_encoder

    # Fit dataset
    detect_brain.fit(
        epoch_counter=detect_brain.hparams.counter,
        train_set=datasets["train"],
        valid_set=datasets["valid"],
        train_loader_kwargs=hparams["dataloader_opts"],
        valid_loader_kwargs=hparams["dataloader_opts"],
    )

    # Evaluate best checkpoint, using lowest or highest value on validation
    detect_brain.evaluate(
        datasets["test"],
        test_loader_kwargs=hparams["dataloader_opts"],
    )
