import argparse
from pathlib import Path
import sys
import os


def parse_arguments_for_training():
    """
    parse command line arguments for training
    TODO: add option to parse from configuration file
    :return:
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment_name", help="The name of the experiment.")
    parser.add_argument("dataset_folder", help="The path to the folder containing the data")
    parser.add_argument("--number_of_classes", type=int, default=3, help="number of classes to predict")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size to train with")
    parser.add_argument("--image_size", type=int, default=100, help="All images will be resized to this size")
    parser.add_argument("--frames_per_datapoint", type=int, default=10, help="Number of frames in each datapoint")
    parser.add_argument("--frames_stride_size", type=int, default=2, help="Stride between frames")
    parser.add_argument("--eliminate_transitions", action="store_true",
                        help="If true, does not use frames where transitions occur (train only!)")
    parser.add_argument("--architecture", type=str, choices=["fc", "icatcher_vanilla", "icatcher+", "rnn"],
                        default="icatcher+",
                        help="Selects architecture to use")
    parser.add_argument("--loss", type=str, choices=["cat_cross_entropy"], default="cat_cross_entropy",
                        help="Selects loss function to optimize")
    parser.add_argument("--optimizer", type=str, choices=["adam", "SGD"], default="adam")
    parser.add_argument("--lr", type=float, default=1e-5, help="Initial learning rate")
    parser.add_argument("--lr_policy", type=str, choices=["lambda", "plateau", "multi_step", "cyclic"],
                        default="plateau",
                        help="learning rate scheduler policy")
    parser.add_argument("--lr_decay_rate", type=int, default=0.98, help="Decay rate for lamda lr policy.")
    parser.add_argument("--continue_train", action="store_true", help="Continue training from latest iteration")
    parser.add_argument("--filter_validation", type=str,
                        help="if present, uses this file to filter out certain data from validation set")
    parser.add_argument("--use_disjoint", action="store_true",
                        help="if true, uses only disjoint subjects videos, else uses only subjects who appeared in train set")
    parser.add_argument("--rand_augment", default=False, action="store_true",
                        help="if true, uses RandAugment for training augmentations")
    parser.add_argument("--horiz_flip", default=False, action="store_true",
                        help="if true, horizontally flips images in training (and flips labels as well)")
    parser.add_argument("--number_of_epochs", type=int, default=100, help="Total number of epochs to train model")
    parser.add_argument("--seed", type=int, default=42, help="Random seed to train with")
    parser.add_argument("--gpu_id", type=str, default=-1, help="GPU ids to use, comma delimited (or -1 for cpu)")
    parser.add_argument("--port", type=str, default="12355", help="port to use for ddp")
    parser.add_argument("--tensorboard", action="store_true", help="Activates tensorboard logging")
    parser.add_argument("--log", action="store_true", help="Logs into a file instead of stdout")
    parser.add_argument("-v", "--verbosity", type=str, choices=["debug", "info", "warning"], default="info",
                        help="Selects verbosity level")
    args = parser.parse_args()
    args.dataset_folder = Path(args.dataset_folder)
    # add some useful arguments for the rest of the code
    args.distributed = len(args.gpu_id.split(",")) > 1
    args.world_size = len(args.gpu_id.split(","))
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    args.experiment_path = Path("runs", args.experiment_name)
    args.experiment_path.mkdir(exist_ok=True, parents=True)
    with open(Path(args.experiment_path, 'commandline_args.txt'), 'w') as f:
        f.write('\n'.join(sys.argv[1:]))
    return args


def parse_arguments_for_testing():
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=str, help="the source to use (path to video file, folder or webcam id)")
    parser.add_argument("model", type=str, help="path to model that will be used for predictions")
    parser.add_argument("--fc_model", type=str, help="path to face classifier model that will be used for deciding "
                                                     "which crop should we select from every frame")
    parser.add_argument("--source_type", type=str, default="file", choices=["file", "webcam"],
                        help="selects source of stream to use.")
    parser.add_argument("--video_filter", type=str,
                        help="either a file consisting of a list of files used to filter videos in a folder,"
                             " or a folder of files to filter the videos with. if file and suffix is .tsv,"
                             " will assume certain structure using the lookit dataset")
    parser.add_argument("--raw_dataset_path", type=str, help="path to raw dataset (required if video_filter is a .tsv file")
    parser.add_argument("--output_annotation", type=str, help="folder to output annotations to")
    parser.add_argument("--on_off", action="store_true",
                        help="left/right/away annotations will be swapped with on/off (only works with icatcher+)")
    # Set up text output file, using https://osf.io/3n97m/ - PrefLookTimestamp coding standard
    parser.add_argument("--output_format", type=str, default="PrefLookTimestamp", choices=["PrefLookTimestamp",
                                                                                           "raw_output",
                                                                                           "compressed"])
    parser.add_argument("--output_file_suffix", type=str, default=".txt", help="the output file suffix")
    parser.add_argument("--architecture", type=str, choices=["fc", "icatcher_vanilla", "icatcher+", "rnn"],
                        default="icatcher+",
                        help="Selects architecture to use")
    parser.add_argument("--loss", type=str, choices=["cat_cross_entropy"], default="cat_cross_entropy",
                        help="Selects loss function to optimize")
    parser.add_argument("--image_size", type=int, default=100, help="All images will be resized to this size")
    parser.add_argument("--output_video_path", help="if present, annotated video will be saved to this folder")
    parser.add_argument("--show_output", action="store_true", help="show results online in a separate window")
    parser.add_argument("--per_channel_mean", nargs=3, metavar=("Channel1_mean", "Channel2_mean", "Channel3_mean"),
                        type=float, help="supply custom per-channel mean of data for normalization")
    parser.add_argument("--per_channel_std", nargs=3, metavar=("Channel1_std", "Channel2_std", "Channel3_std"),
                        type=float, help="supply custom per-channel std of data for normalization")
    parser.add_argument("--gpu_id", type=int, default=-1, help="GPU id to use, use -1 for CPU.")
    parser.add_argument("--log",
                        help="If present, writes log to this path")
    parser.add_argument("-v", "--verbosity", type=str, choices=["debug", "info", "warning"], default="info",
                        help="Selects verbosity level")
    args = parser.parse_args()
    args.model = Path(args.model)
    assert args.model.is_file()
    if args.video_filter:
        args.video_filter = Path(args.video_filter)
        assert args.video_filter.is_file() or args.video_filter.is_dir()
    if args.raw_dataset_path:
        args.raw_dataset_path = Path(args.raw_dataset_path)
    if args.output_annotation:
        args.output_filepath = Path(args.output_annotation)
        args.output_filepath.mkdir(exist_ok=False, parents=True)
        if not args.output_filepath.is_dir():
            print("--output_filepath argument must point to a folder.")
            raise AssertionError
    if args.output_video_path:
        args.output_video_path = Path(args.output_video_path)
        if not args.output_video_path.is_dir():
            print("--output_video_path argument must point to a folder.")
            raise AssertionError
    if args.log:
        args.log = Path(args.log)
    if args.on_off:
        if args.output_format != "raw_output":
            print("On off mode can only be used with raw output format. Pass raw_output with the --output_format flag.")
            raise AssertionError
    if not args.per_channel_mean:
        args.per_channel_mean = [0.485, 0.456, 0.406]
    if not args.per_channel_std:
        args.per_channel_std = [0.229, 0.224, 0.225]
    if args.gpu_id == -1:
        args.device = "cpu"
    else:
        import os
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
        args.device = "cuda:{}".format(0)
    return args


def parse_arguments_for_visualizations():
    parser = argparse.ArgumentParser()
    parser.add_argument("output_folder", type=str, default="output", help="path to output results.")
    parser.add_argument("raw_dataset_folder", type=str, help="path to raw dataset folder")
    parser.add_argument("dataset_folder", type=str, help="path to preprocessed dataset folder")
    parser.add_argument("human_codings_folder", type=str, help="the codings from human1")
    parser.add_argument("human2_codings_folder", type=str, help="the codings from human12")
    parser.add_argument("machine_codings_folder", type=str, help="the codings from machine")
    parser.add_argument("--human_coding_format",
                        type=str,
                        default="PrefLookTimestamp",
                        choices=["PrefLookTimestamp",
                                 "lookit",
                                 "compressed",
                                 "princeton"])
    parser.add_argument("--machine_coding_format",
                        type=str,
                        default="PrefLookTimestamp",
                        choices=["PrefLookTimestamp",
                                 "lookit",
                                 "compressed",
                                 "princeton"])
    parser.add_argument("--log", help="If present, writes log to this path")
    parser.add_argument("-v", "--verbosity", type=str, choices=["debug", "info", "warning"], default="info",
                        help="Selects verbosity level")
    args = parser.parse_args()
    args.output_folder = Path(args.output_folder)
    args.output_folder.mkdir(parents=True, exist_ok=True)
    args.dataset_folder = Path(args.dataset_folder)
    assert args.dataset_folder.is_dir()
    args.human_codings_folder = Path(args.human_codings_folder)
    assert args.human_codings_folder.is_dir()
    args.human2_codings_folder = Path(args.human2_codings_folder)
    assert args.human2_codings_folder.is_dir()
    args.machine_codings_folder = Path(args.machine_codings_folder)
    assert args.machine_codings_folder.is_dir()
    args.raw_video_folder = Path(args.dataset_folder, "raw_videos")
    assert args.raw_video_folder.is_dir()
    args.faces_folder = Path(args.dataset_folder, "faces")
    assert args.faces_folder.is_dir()
    return args


def parse_arguments_for_preprocess():
    parser = argparse.ArgumentParser()
    parser.add_argument("raw_dataset_path", type=str, help="path to raw dataset to preprocess")
    parser.add_argument("output_folder", type=str, help="path to put preprocessed dataset")
    parser.add_argument("--raw_dataset_type", type=str, choices=["lookit", "princeton", "generic"], default="lookit",
                        help="the type of dataset to preprocess")
    parser.add_argument("--fc_model", type=str, default="models/face_classifier_weights_best.pt", help="path to face classifier model if it was trained")
    parser.add_argument("--split_type", type=str, choices=["split0_train", "split0_test", "all"], default="split0_train")
    parser.add_argument("--one_video_per_child_policy", choices=["include_all", "unique_only", "unique_only_in_val", "unique_only_in_train"], type=str,
                        default="unique_only_in_val", help="some videos are of the same child, this policy dictates what to do with those.")
    parser.add_argument("--train_val_disjoint", action="store_true", help="if true, train and validation sets will never contain same child")
    parser.add_argument("--val_percent", type=float, default=0.2, help="desired percent of validation set")
    parser.add_argument("--face_detector_confidence", type=float, default=0.7, help="confidence threshold for face detector")
    parser.add_argument("--gpu_id", type=int, default=-1, help="Which GPU to use (or -1 for cpu)")
    parser.add_argument("--log", help="if present, writes log to this path")
    parser.add_argument("--seed", type=int, default=43, help="random seed (controls split selection)")
    parser.add_argument("-v", "--verbosity", type=str, choices=["debug", "info", "warning"], default="info",
                        help="Selects verbosity level")
    args = parser.parse_args()
    args.raw_dataset_path = Path(args.raw_dataset_path)
    if not args.raw_dataset_path.is_dir():
        raise NotADirectoryError
    args.output_folder = Path(args.output_folder)
    args.output_folder.mkdir(parents=True, exist_ok=True)
    args.video_folder = args.output_folder / "raw_videos"
    args.faces_folder = args.output_folder / "faces"
    args.label_folder = args.output_folder / "coding_first"
    args.label2_folder = args.output_folder / "coding_second"
    args.multi_face_folder = args.output_folder / "multi_face"
    args.face_data_folder = args.output_folder / "infant_vs_others"
    args.fc_model = Path(args.fc_model)
    args.face_model_file = Path("models", "face_model.caffemodel")
    args.config_file = Path("models", "config.prototxt")
    if args.gpu_id == -1:
        args.device = "cpu"
    else:
        import os
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
        args.device = "cuda:{}".format(0)
    return args
