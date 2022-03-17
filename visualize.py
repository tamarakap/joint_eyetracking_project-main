import random
import seaborn as sns
from sklearn.metrics import confusion_matrix
import itertools
import parsers
import os
import matplotlib.pyplot as plt
import pickle
import numpy as np
from tqdm import tqdm
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter, AutoMinorLocator)
import csv
import logging
import cv2
from pathlib import Path
from options import parse_arguments_for_visualizations
import textwrap


def label_to_color(label):
    mapping = {"left": (0.5, 0.6, 0.9),
               "right": (0.9, 0.6, 0.5),
               "away": "lightgrey",
               "invalid": "white",
               "lblue": (0.5, 0.6, 0.9),
               "lred": (0.9, 0.6, 0.5),
               "lgreen": (0.6, 0.8, 0.0),
               "lorange": (0.94, 0.78, 0.0),
               "lyellow": (0.9, 0.9, 0.0),
               "mblue": (0.12, 0.41, 0.87)}
    return mapping[label]


def calculate_confusion_matrix(label, pred, save_path, mat=None, class_num=3):
    """
    creates a plot of the confusion matrix given the gt labels abd the predictions.
    if mat is supplied, ignores other inputs and uses that.
    :param label: the labels
    :param pred: the predicitions
    :param save_path: path to save plot
    :param mat: a numpy 3x3 array representing the confusion matrix
    :param class_num: number of classes
    :return:
    """
    if mat is None:
        mat = np.zeros([class_num, class_num])
        pred = np.array(pred)
        label = np.array(label)
        logging.info('# datapoint: {}'.format(len(label)))
        for i in range(class_num):
            for j in range(class_num):
                mat[i][j] = sum((label == i) & (pred == j))
    total_acc = (mat.diagonal().sum() / mat.sum()) * 100
    norm_mat = mat / np.sum(mat, -1, keepdims=True)
    fig, ax = plt.subplots(figsize=(3, 3))
    ax = sns.heatmap(norm_mat, ax=ax, vmin=0, vmax=1, annot=True, fmt='.2%', cbar=False, cmap='Blues')
    ax.set_xticklabels(['away', 'left', 'right'])
    ax.set_yticklabels(['away', 'left', 'right'])
    plt.axis('equal')
    plt.tight_layout(pad=0.1)
    plt.savefig(save_path)
    logging.info('acc:{:.4f}%'.format(total_acc))
    logging.info('confusion matrix: {}'.format(mat))
    logging.info('normalized confusion matrix: {}'.format(norm_mat))
    return norm_mat, total_acc


def confusion_mat(targets, preds, classes, normalize=False, plot=False, title="Confusion Matrix", cmap=plt.cm.Blues):
    cm = confusion_matrix(targets, preds)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    if plot:
        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)

        fmt = '.2f' if normalize else 'd'
        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, format(cm[i, j], fmt), horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.savefig(title + ".png")
        plt.cla()
        plt.clf()

    return cm


def plot_learning_curve(train_perfs, val_perfs, save_dir, isLoss=False):
    epochs = np.arange(1, len(train_perfs) + 1)
    plt.plot(epochs, train_perfs, label="Training set")
    plt.plot(epochs, val_perfs, label="Validation set")
    plt.xlabel("Epochs")
    metric_name = "Loss" if isLoss else "Accuracy"
    plt.ylabel(metric_name)
    plt.title(metric_name, fontsize=16, y=1.002)
    plt.legend()
    plt.savefig(os.path.join(save_dir, 'learning_curve_%s.png' % metric_name))
    plt.cla()
    plt.clf()


def print_data_img_name(dataloaders, dl_key, selected_idxs=None):
    dataset = dataloaders[dl_key].dataset
    fp_prefix = os.path.join(face_data_folder, dl_key)
    if selected_idxs is None:
        for fname, lbl in dataset.samples:
            print(fname.strip(fp_prefix))
    else:
        for i, tup in enumerate(dataset.samples):
            if i in selected_idxs:
                print(Path(tup[0]).parent.stem, Path(tup[0]).stem)


def time_ms_to_frame(time_ms, fps=30):
    return int(time_ms / 1000.0 * fps)


def frame_to_time_ms(frame, fps=30):
    return int(frame * 1000 / fps)


def compare_two_coding_files(coding1, coding2):
    """
    given two codings in the format described below, returns some relevant metrics in a dictionary
    format:
    [list of [frame number, valid, class], first annotated frame, last annotated frame]
    :param coding1: the "target"
    :param coding2: the "prediction"
    note: naming the codings as target and prediction is just a convention;
    all functions and metrics are symmetric w.r.t. the codings.
    :return:
    """
    start1 = coding1[1]
    end1 = coding1[2]
    start2 = coding2[1]
    end2 = coding2[2]
    coding1_np = np.array([x[0] for x in coding1[0]])
    coding2_np = np.array([x[0] for x in coding2[0]])

    classes = {"away": 0, "left": 1, "right": 2}
    ON_CLASSES = ["left", "right"]
    same_count = 0
    diff_count = 0
    times_coding1 = {"left": [],
                     "right": [],
                     "away": [],
                     "invalid": []}

    times_coding2 = {"left": [],
                     "right": [],
                     "away": [],
                     "invalid": []}

    coding1_by_label = [0, 0, 0]
    coding2_by_label = [0, 0, 0]

    left_right_agree = 0
    c1_left_right_total = 0
    c2_left_right_total = 0

    valid_labels_coding1 = 0
    valid_labels_coding2 = 0
    start = min(start1, start2)
    end = max(end1, end2)
    total_frames_coding1 = end1 - start1
    total_frames_coding2 = end2 - start2
    total_transitions_coding1 = len(coding1[0]) - len([x for x in coding1[0] if x[2] not in classes.keys()])
    total_transitions_coding2 = len(coding2[0]) - len([x for x in coding2[0] if x[2] not in classes.keys()])
    for frame_index in range(start, end):
        if frame_index < coding1[0][0][0]:
            coding1_label = [None, None, "invalid"]
            times_coding1["invalid"].append(frame_index)
        else:
            target_q_np = np.nonzero(frame_index >= coding1_np)[0][-1]
            coding1_label = coding1[0][target_q_np]
            if coding1_label[1] == 1:
                assert coding1_label[2] in classes.keys()
                times_coding1[coding1_label[2]].append(frame_index)
                valid_labels_coding1 += 1
            else:
                coding1_label = [None, None, "invalid"]
                times_coding1["invalid"].append(frame_index)
        if frame_index < coding2[0][0][0]:
            coding2_label = [None, None, "invalid"]
            times_coding2["invalid"].append(frame_index)
        else:
            inferred_q_np = np.nonzero(frame_index >= coding2_np)[0][-1]
            coding2_label = coding2[0][inferred_q_np]
            if coding2_label[1] == 1:
                assert coding2_label[2] in classes.keys()
                times_coding2[coding2_label[2]].append(frame_index)
                valid_labels_coding2 += 1
            else:
                coding2_label = [None, None, "invalid"]
                times_coding2["invalid"].append(frame_index)

        if coding1_label[2] != "invalid":
            assert coding1_label[2] in classes.keys()
            coding1_by_label[classes[coding1_label[2]]] += 1
        if coding2_label[2] != "invalid":
            assert coding2_label[2] in classes.keys()
            coding2_by_label[classes[coding2_label[2]]] += 1
        if coding1_label[2] != "invalid" and coding2_label[2] != "invalid":
            assert coding1_label[2] in classes.keys()
            if coding1_label[2] == coding2_label[2]:
                same_count += 1
            else:
                diff_count += 1
        if coding1_label[2] in ON_CLASSES:
            if coding1_label[2] == coding2_label[2]:
                left_right_agree += 1
            c1_left_right_total += 1
        if coding2_label[2] in ON_CLASSES:
            c2_left_right_total += 1

    accuracy = 100 * same_count / (same_count + diff_count)

    frac_coding1_valid = valid_labels_coding1 / total_frames_coding1
    frac_coding2_valid = valid_labels_coding2 / total_frames_coding2

    num_coding1_valid = valid_labels_coding1
    num_coding2_valid = valid_labels_coding2

    coding1_on_vs_away = (coding1_by_label[0] + coding1_by_label[1]) / sum(coding1_by_label)
    coding2_on_vs_away = (coding2_by_label[0] + coding2_by_label[1]) / sum(coding2_by_label)
    c1_left_right_accuracy = left_right_agree / c1_left_right_total
    c2_left_right_accuracy = left_right_agree / c2_left_right_total

    metrics = {"accuracy": accuracy,
               "frac_coding1_valid": frac_coding1_valid,
               "frac_coding2_valid": frac_coding2_valid,
               "num_coding1_valid": num_coding1_valid,
               "num_coding2_valid": num_coding2_valid,
               "coding1_on_vs_away": coding1_on_vs_away,
               "coding2_on_vs_away": coding2_on_vs_away,
               "left_right_accuracy_c1": c1_left_right_accuracy,
               "left_right_accuracy_c2": c2_left_right_accuracy,
               "coding1_by_label": coding1_by_label,
               "coding2_by_label": coding2_by_label,
               "times_coding1": times_coding1,
               "times_coding2": times_coding2,
               "valid_range_coding1": [start1, end1],
               "valid_range_coding2": [start2, end2],
               "total_transitions_coding1": total_transitions_coding1,
               "total_transitions_coding2": total_transitions_coding2}
    return metrics


def compare_coding_files(human_coding_file, human_coding_file2, machine_coding_file, args):
    if args.machine_coding_format == "PrefLookTimestamp":
        parser = parsers.PrefLookTimestampParser(30)
    else:
        parser = parsers.PrincetonParser(30, start_time_file=Path(args.raw_dataset_folder, "start_times_visitA.csv"))
    machine, mstart, mend = parser.parse(machine_coding_file, file_is_fullpath=True)

    if args.human_coding_format == "PrefLookTimestamp":
        parser = parsers.PrefLookTimestampParser(30)
    else:
        parser = parsers.PrincetonParser(30, start_time_file=Path(args.raw_dataset_folder, "start_times_visitA.csv"))
    human, start1, end1 = parser.parse(human_coding_file, file_is_fullpath=True)
    human2, start2, end2 = parser.parse(human_coding_file2, file_is_fullpath=True)
    # machine = machine[:-1]
    metrics = {}
    metrics["human1_vs_machine"] = compare_two_coding_files([human, start1, end1], [machine, mstart, mend])
    metrics["human1_vs_human2"] = compare_two_coding_files([human, start1, end1], [human2, start2, end2])
    # total_h1_annotated_frames = end1 - start1
    # total_h2_annotated_frames = end2 - start2
    # total_m_annotated_frames = mend - mstart
    return metrics


def save_metrics_csv(sorted_IDs, all_metrics, inference):
    with open(f'iCatcher/plots/CSV_reports/{inference}', 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile, delimiter=',')
        header = ["video ID", f'{inference} label path', "Accuracy"]
        csv_writer.writerow(header)
        for ID in sorted_IDs:
            row = []
            row.append(ID)
            bucket_root = "gaze-coding/iCatcher/pre-trained-inference/"
            row.append(bucket_root + all_metrics[ID][inference]['filename'])
            row.append(all_metrics[ID][inference]['accuracy'])
            csv_writer.writerow(row)


def select_frames_from_video(ID, video_folder, start, end):
    """
    selects 9 random frames from a video for display
    :param ID: the video id (no extension)
    :param video_folder: the raw video folder
    :param start: where annotation begins
    :param end: where annotation ends
    :return: an image grid of 9 frames and the corresponding frame numbers
    """
    imgs_np = np.ones((480*3, 640*3, 3))
    for video_file in Path(video_folder).glob("*"):
        if ID in video_file.name:
            imgs = []
            cap = cv2.VideoCapture(str(video_file))
            frame_selections = np.random.choice(np.arange(start, end//2), size=9, replace=False)
            for i in range(start, end//2):  # to avoid end of video
                ret, frame = cap.read()
                if i in frame_selections:
                    imgs.append(frame[..., ::-1])
                    if len(imgs) >= 9:
                        break
            imgs_np = np.array(imgs)
            imgs_np = make_gridview(imgs_np)
    return imgs_np, frame_selections


def sample_luminance(ID, args, start, end, num_samples=10):
    total_luminance = 0
    sampled = 0
    for video_file in args.raw_video_folder.glob("*"):
        if ID in video_file.stem:
            cap = cv2.VideoCapture(str(video_file))
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_no = int(start / 1000 * fps)

            for i in range(num_samples):
                cap.set(cv2.CAP_PROP_FRAME_COUNT, frame_no - 1)
                ret, frame = cap.read()
                b, g, r = cv2.split(frame)

                total_luminance += 0.2126 * np.sum(r) + 0.7152 * np.sum(g) + 0.0722 * np.sum(b)
                sampled += 1
                frame_no += (((end - start) / num_samples) / 1000 * fps)

    return total_luminance / sampled


def generate_frame_by_frame_comparisons(sorted_IDs, all_metrics, args):
    GRAPH_CLASSES = ["left", "right", "away", "invalid"]
    widths = [20, 1, 5]
    heights = [1]
    skip = 10
    gs_kw = dict(width_ratios=widths, height_ratios=heights)
    # fig, axs = plt.subplots(len(sorted_IDs), 3, figsize=(30, 45), gridspec_kw=gs_kw)
    # if len(axs.shape) == 1:
    #     axs = np.expand_dims(axs, axis=0)
    # plt.suptitle(f'Frame by frame comparisons per video', fontsize=40)
    # color_gradient = list(Color("red").range_to(Color("green"), 100))
    frame_by_frame_path = Path(args.output_folder, "frame_by_frame")
    frame_by_frame_path.mkdir(exist_ok=True, parents=True)
    for i, target_ID in enumerate(tqdm(sorted_IDs)):
        fig, axs = plt.subplots(1, 3, figsize=(24.0, 8.0), gridspec_kw=gs_kw)
        timeline, accuracy, sample_frame = axs  # won't work with single video...
        plt.suptitle('Frame by frame comparisons: {}'.format(target_ID + ".mp4"))
        start1, end1 = all_metrics[target_ID]["human1_vs_human2"]["valid_range_coding2"]
        start2, end2 = all_metrics[target_ID]["human1_vs_human2"]["valid_range_coding1"]
        start3, end3 = all_metrics[target_ID]["human1_vs_machine"]["valid_range_coding2"]
        start = min(start1, start2, start3)
        end = max(end1, end2, end3)
        timeline.set_title("Frames: {} - {}".format(str(start), str(end)))

        times1 = all_metrics[target_ID]["human1_vs_human2"]["times_coding2"]
        times2 = all_metrics[target_ID]["human1_vs_human2"]["times_coding1"]
        times3 = all_metrics[target_ID]["human1_vs_machine"]["times_coding2"]
        times = [times1, times2, times3]
        video_label = ["human 2", "human 1", "machine"]

        for j, vid_label in enumerate(video_label):
            for label in GRAPH_CLASSES:
                timeline.barh(vid_label, skip, left=times[j][label][::skip],
                              height=1, label=label,
                              color=label_to_color(label))
            timeline.set_xlabel("Frame #")
            if j == 0:
                timeline.legend(loc='upper right')
                accuracy.set_title("Accuracy")

        # for j, name in enumerate(["times_coding1", "times_coding2"]):
        #     times = all_metrics[target_ID]["human1_vs_machine"][name]
        #     video_label = "human" if j==0 else "machine"
        #     skip = 10  # frame comparison resolution. Increase to speed up plotting
        #     for label in GRAPH_CLASSES:
        #         timeline.barh(video_label, skip, left=times[label][::skip],
        #                       height=1, label=label,
        #                       color=label_to_color(label))
        #     timeline.set_xlabel("Frame #")
        #     if j == 0:
        #         timeline.legend(loc='upper right')
        #         accuracy.set_title("Accuracy")
        inference = ["human1_vs_human2", "human1_vs_machine"]
        accuracies = [all_metrics[target_ID][entry]['accuracy'] for entry in inference]
        # colors = [color_gradient[int(acc * 100)].rgb for acc in accuracies]

        accuracy.bar(range(len(inference)), accuracies, color="black")
        accuracy.set_xticks(range(len(inference)))
        accuracy.set_xticklabels(inference, rotation=45, ha="right")
        accuracy.set_ylim([0, 100])
        accuracy.set_ylabel("Accuracy")
        # sample_frame_index = min(
        #     [all_metrics[target_ID][ICATCHER]['times_target'][label][0] for label in VALID_CLASSES])
        imgs, times = select_frames_from_video(target_ID, args.raw_video_folder, start, end)
        sample_frame.imshow(imgs)
        sample_frame.set_axis_off()
        # sample_frame_index = (end - start) / 2.0
        # sample_frame.imshow(get_frame_from_video(target_ID, sample_frame_index, args.raw_video_folder))
        longstring = 'Sample frames from video at frames: {}'.format(times)
        formatted_longstring = "\n".join(textwrap.wrap(longstring, 40))
        sample_frame.set_title(formatted_longstring)
        plt.tight_layout()

        plt.savefig(Path(frame_by_frame_path, 'frame_by_frame_{}.png'.format(target_ID + ".mp4")))
    # plt.subplots_adjust(left=0.075, bottom=0.075, right=0.925, top=0.925, wspace=0.2, hspace=0.8)
        plt.cla()
        plt.clf()


def generate_collage_plot(sorted_IDs, all_metrics, save_path):
    """
    plots one image with various selected stats
    :param sorted_IDs: ids of videos sorted by accuracy score
    :param all_metrics: all metrics per video
    :param save_path: where to save the image
    :return:
    """
    classes = {"away": 0, "left": 1, "right": 2}
    # fig, axs = plt.subplots(3, 2, figsize=(10, 12))
    fig = plt.figure(figsize=(10, 12))

    # accuracies plot
    accuracy_bar = fig.add_subplot(3, 2, (1, 2))  # three rows, two columns
    # accuracy_bar = axs[0, :]
    accuracies_hvh = [all_metrics[ID]["human1_vs_human2"]['accuracy'] for ID in sorted_IDs]
    mean_hvh = np.mean(accuracies_hvh)
    accuracies_hvm = [all_metrics[ID]["human1_vs_machine"]['accuracy'] for ID in sorted_IDs]
    mean_hvm = np.mean(accuracies_hvm)
    labels = sorted_IDs
    width = 0.35  # the width of the bars
    x = np.arange(len(labels))
    accuracy_bar.bar(x - width / 2, accuracies_hvh, width, color=label_to_color("lorange"), label='Human vs Human')
    accuracy_bar.bar(x + width / 2, accuracies_hvm, width, color=label_to_color("mblue"), label='Human vs Machine')
    accuracy_bar.set_ylabel('Accuracy')
    accuracy_bar.set_xlabel('Video')
    accuracy_bar.set_title('Accuracy per video')
    accuracy_bar.set_xticks(x)
    # accuracy_bar.bar_label(rects1, padding=3)
    # accuracy_bar.bar_label(rects2, padding=3)
    accuracy_bar.axhline(y=mean_hvh, color=label_to_color("lorange"), linestyle='-', label="mean (" + str(mean_hvh)[:4] + ")")
    accuracy_bar.axhline(y=mean_hvm, color=label_to_color("mblue"), linestyle='-', label="mean (" + str(mean_hvm)[:4] + ")")
    accuracy_bar.set_ylim([0, 100])
    accuracy_bar.legend()

    # target valid plot
    transitions_bar = fig.add_subplot(3, 2, 3)  # three rows, two columns
    width = 0.66  # the width of the bars
    x = np.arange(len(sorted_IDs))
    # target_valid_scatter = axs[1, 0]
    # target_valid_scatter.plot([0, 1], [0, 1], transform=target_valid_scatter.transAxes, color="black", label="Ideal trend")
    # transitions_bar.set_xlim([0, 3])
    # transitions_bar.set_ylim([0, 3])
    #
    transitions_h1 = [100*all_metrics[ID]["human1_vs_human2"]['total_transitions_coding1'] /
                          all_metrics[ID]["human1_vs_human2"]['num_coding1_valid'] for ID in sorted_IDs]
    transitions_h2 = [100*all_metrics[ID]["human1_vs_human2"]['total_transitions_coding2'] /
                          all_metrics[ID]["human1_vs_human2"]['num_coding2_valid'] for ID in sorted_IDs]
    transitions_m = [100*all_metrics[ID]["human1_vs_machine"]['total_transitions_coding2'] /
                          all_metrics[ID]["human1_vs_machine"]['num_coding2_valid'] for ID in sorted_IDs]

    transitions_bar.bar(x - width / 3, transitions_h1, width=(width / 3) - 0.1, label="Human 1", color=label_to_color("lorange"))
    transitions_bar.bar(x, transitions_h2, width=(width / 3) - 0.1, label="Human 2", color=label_to_color("lgreen"))
    transitions_bar.bar(x + width / 3, transitions_m, width=(width / 3) - 0.1, label="Machine", color=label_to_color("mblue"))
    transitions_bar.set_xticks(x)
    transitions_bar.set_title('# Transitions per 100 frames')
    transitions_bar.legend()
    transitions_bar.set_ylabel('# Transitions per 100 frames')
    transitions_bar.set_xlabel('Video')

    # on away plot
    on_away_scatter = fig.add_subplot(3, 2, 4)  # three rows, two columns
    # on_away_scatter = axs[1, 1]
    on_away_scatter.plot([0, 1], [0, 1], transform=on_away_scatter.transAxes, color="black", label="Ideal trend")
    on_away_scatter.set_xlim([0, 1])
    on_away_scatter.set_ylim([0, 1])
    x_target_away_hvh = [all_metrics[ID]["human1_vs_human2"]['coding1_on_vs_away'] for ID in sorted_IDs]
    y_target_away_hvh = [all_metrics[ID]["human1_vs_human2"]['coding2_on_vs_away'] for ID in sorted_IDs]
    x_target_away_hvm = [all_metrics[ID]["human1_vs_machine"]['coding1_on_vs_away'] for ID in sorted_IDs]
    y_target_away_hvm = [all_metrics[ID]["human1_vs_machine"]['coding2_on_vs_away'] for ID in sorted_IDs]
    on_away_scatter.scatter(x_target_away_hvh, y_target_away_hvh, color=label_to_color("lorange"), label='Human vs Human')
    for i in range(len(sorted_IDs)):
        on_away_scatter.annotate(i, (x_target_away_hvh[i], y_target_away_hvh[i]))
    on_away_scatter.scatter(x_target_away_hvm, y_target_away_hvm, color=label_to_color("mblue"), label='Human vs Machine')
    for i in range(len(sorted_IDs)):
        on_away_scatter.annotate(i, (x_target_away_hvm[i], y_target_away_hvm[i]))
    on_away_scatter.set_xlabel("Human 1")
    on_away_scatter.set_ylabel("Human 2 or Machine")
    on_away_scatter.set_title("Percent looking on the screen")
    on_away_scatter.legend()

    # label distribution plot
    # label_scatter = fig.add_subplot(3, 2, 5)  # three rows, two columns
    # # label_scatter = axs[2, 0]
    # label_scatter.plot([0, 1], [0, 1], transform=label_scatter.transAxes, color="black", label="Ideal trend")
    # label_scatter.set_xlim([0, 1])
    # label_scatter.set_ylim([0, 1])
    # for i, label in enumerate(sorted(classes.keys())):
    #     y_labels = [y[i] / sum(y) for y in [all_metrics[ID]["human1_vs_human2"]['coding1_by_label'] for ID in sorted_IDs]]
    #     x_labels = [y[i] / sum(y) for y in [all_metrics[ID]["human1_vs_human2"]['coding2_by_label'] for ID in sorted_IDs]]
    #     label_scatter.scatter(x_labels, y_labels, color=label_to_color(label), label="hvh: " + label, marker='^')
    #     for n in range(len(sorted_IDs)):
    #         label_scatter.annotate(n, (x_labels[n], y_labels[n]))
    # for i, label in enumerate(sorted(classes.keys())):
    #     y_labels = [y[i] / sum(y) for y in [all_metrics[ID]["human1_vs_machine"]['coding1_by_label'] for ID in sorted_IDs]]
    #     x_labels = [y[i] / sum(y) for y in [all_metrics[ID]["human1_vs_machine"]['coding2_by_label'] for ID in sorted_IDs]]
    #     label_scatter.scatter(x_labels, y_labels, color=label_to_color(label), label="hvh: " + label, marker='o')
    #     for n in range(len(sorted_IDs)):
    #         label_scatter.annotate(n, (x_labels[n], y_labels[n]))
    #
    # label_scatter.set_xlabel('Human 1 label proportion')
    # label_scatter.set_ylabel('Human 2 / Machine labels proportion')
    # label_scatter.set_title('labels distribution')
    # label_scatter.legend()  # loc='upper center'

    # label distribution bar plot
    label_bar = fig.add_subplot(3, 2, (5, 6))  # three rows, two columns
    # label_bar = axs[2, 1]
    ticks = range(len(sorted_IDs))
    bottoms_h1 = np.zeros(shape=(len(sorted_IDs)))
    bottoms_h2 = np.zeros(shape=(len(sorted_IDs)))
    bottoms_m = np.zeros(shape=(len(sorted_IDs)))
    width = 0.66
    patterns = [None, "O", "*"]
    for i, label in enumerate(sorted(classes.keys())):
        label_counts_h1 = [y[i] / sum(y) for y in [all_metrics[ID]["human1_vs_human2"]['coding1_by_label'] for ID in sorted_IDs]]
        label_counts_h2 = [y[i] / sum(y) for y in [all_metrics[ID]["human1_vs_human2"]['coding2_by_label'] for ID in sorted_IDs]]
        label_counts_m = [y[i] / sum(y) for y in [all_metrics[ID]["human1_vs_machine"]['coding2_by_label'] for ID in sorted_IDs]]

        label_bar.bar(x - width/3, label_counts_h1, bottom=bottoms_h1, width=(width / 3)-0.1, label=label,
                      color=label_to_color(label), edgecolor='black', hatch=patterns[0])
        label_bar.bar(x, label_counts_h2, bottom=bottoms_h2, width=(width / 3)-0.1, label=label,
                      color=label_to_color(label), edgecolor='black', hatch=patterns[1])
        label_bar.bar(x + width/3, label_counts_m, bottom=bottoms_m, width=(width / 3)-0.1, label=label,
                      color=label_to_color(label), edgecolor='black', hatch=patterns[2])
        if i == 0:
            from matplotlib.patches import Patch
            artists = [Patch(facecolor=label_to_color("away"), label="Away"),
                       Patch(facecolor=label_to_color("left"), label="Left"),
                       Patch(facecolor=label_to_color("right"), label="Right"),
                       Patch(facecolor="white", edgecolor='black', hatch=patterns[0], label="Human 1"),
                       Patch(facecolor="white", edgecolor='black', hatch=patterns[1], label="Human 2"),
                       Patch(facecolor="white", edgecolor='black', hatch=patterns[2], label="Machine")]
            label_bar.legend(handles=artists, bbox_to_anchor=(0.95, 1.0), loc='upper left')
        bottoms_h1 += label_counts_h1
        bottoms_h2 += label_counts_h2
        bottoms_m += label_counts_m
    label_bar.xaxis.set_major_locator(MultipleLocator(1))
    label_bar.set_xticks(ticks)
    label_bar.set_title('Proportion of looking left, right, and away per video')
    label_bar.set_ylabel('Proportion')
    label_bar.set_xlabel('Video')

    plt.subplots_adjust(left=0.1, bottom=0.075, right=0.9, top=0.925, wspace=0.2, hspace=0.5)
    # plt.suptitle(f'{inference} evaluation', fontsize=24)
    plt.savefig(Path(save_path, "collage.png"))
    plt.cla()
    plt.clf()



    # scatter_plots = [target_valid_scatter, on_away_scatter, label_scatter]
    # for ax in scatter_plots:
    #     if ax == target_valid_scatter:
    #         #             pass
    #         ax.set_xlim([0, 100])
    #         ax.set_ylim([0, 100])
    #     else:
    #         ax.set_xlim([0, 1])
    #         ax.set_ylim([0, 1])
    #     ax.plot([0, 1], [0, 1], transform=ax.transAxes, color="black", label="Ideal trend")
    # x = np.arange(len(sorted_IDs))
    # labels = range(len(sorted_IDs))

    # accuracies = [all_metrics[ID][inference]['accuracy'] for ID in sorted_IDs]
    # mean = np.mean(accuracies)
    # accuracy_bar.bar(x, accuracies, color='purple')
    # accuracy_bar.axhline(y=mean, color='black', linestyle='-', label="mean (" + str(mean)[:4] + ")")
    # accuracy_bar.set_title('Accuracy (over mutually valid frames)')
    # accuracy_bar.set_ylim([0, 100])
    # accuracy_bar.set_xticks(ticks)
    # accuracy_bar.set_xticklabels(labels, rotation='vertical', fontsize=8)
    # accuracy_bar.set_xlabel('Video')
    # accuracy_bar.set_ylabel('Accuracy')
    #
    # accuracy_bar.legend()

    # ID_index = axs[0, 1]
    # cell_text = [[i, sorted_IDs[i]] for i in range(len(sorted_IDs))]
    # ID_index.table(cell_text, loc='center', fontsize=18)
    # ID_index.set_title("Video index to ID")


def plot_inference_accuracy_vs_human_agreement(sorted_IDs, all_metrics, args):
    plt.figure(figsize=(8.0, 6.0))
    plt.scatter([all_metrics[id]["human1_vs_human2"]["accuracy"] for id in sorted_IDs],
                [all_metrics[id]["human1_vs_machine"]["accuracy"] for id in sorted_IDs])
    plt.xlim([0, 100])
    plt.ylim([0, 100])
    plt.xlabel("Human accuracy")
    plt.ylabel("iCatcher accuracy")
    plt.title("iCatcher accuracy vs human accuracy for all doubly coded videos")
    plt.savefig(Path(args.output_folder, 'iCatcher_acc_vs_human_acc.png'))
    plt.cla()
    plt.clf()


def plot_luminance_vs_accuracy(sorted_IDs, all_metrics, args):
    plt.figure(figsize=(8.0, 6.0))
    plt.scatter([sample_luminance(id, args, *all_metrics[id]["human1_vs_machine"]['valid_range_coding2']) for id in sorted_IDs],
                [all_metrics[id]["human1_vs_machine"]["accuracy"] for id in sorted_IDs])
    # plt.xlim([0, 1])
    # plt.ylim([0, 1])
    plt.xlabel("Luminance")
    plt.ylabel("iCatcher accuracy")
    plt.title("iCatcher accuracy versus mean video luminance for all doubly coded videos")
    plt.savefig(Path(args.output_folder, 'iCatcher_lum_vs_acc.png'))
    plt.cla()
    plt.clf()


def get_face_pixel_density(id, faces_folder):
    """
    given a video id, calculates the average face area in pixels using pre-processed crops
    :param ids: video id
    :param faces_folder: the folder containing all crops and their meta data as created by "preprocess.py"
    :return:
    """
    face_areas = []
    face_labels = np.load(Path(faces_folder, id, 'face_labels_fc.npy'))
    for i, face_id in enumerate(face_labels):
        if face_id >= 0:
            box_file = Path(faces_folder, id, "box", "{:05d}_{:01d}.npy".format(i, face_id))
            '{name}/box/{frame_number + i:05d}_{face_label_seg[i]:01d}.npy.format()'
            box = np.load(box_file, allow_pickle=True).item()
            face_area = (box['face_box'][1] - box['face_box'][0]) * (box['face_box'][3] - box['face_box'][2])
            face_areas.append(face_area)
    return np.mean(face_areas)


def get_face_location_std(id, faces_folder):
    """
    given a video id, calculates the standard deviation of the face location in that video
    :param ids: video id
    :param faces_folder: the folder containing all crops and their meta data as created by "preprocess.py"
    :return:
    """
    movements = []
    face_labels = np.load(Path(faces_folder, id, 'face_labels_fc.npy'))
    for i, face_id in enumerate(face_labels):
        if face_id >= 0:
            box_file = Path(faces_folder, id, "box", "{:05d}_{:01d}.npy".format(i, face_id))
            '{name}/box/{frame_number + i:05d}_{face_label_seg[i]:01d}.npy.format()'
            box = np.load(box_file, allow_pickle=True).item()
            face_loc = np.array([box['face_box'][1] - box['face_box'][0], box['face_box'][3] - box['face_box'][2]])
            movements.append(face_loc)
    movements = np.array(movements)
    stds = np.mean(np.std(movements, axis=0))
    return stds


def plot_face_pixel_density_vs_accuracy(sorted_IDs, all_metrics, args):
    plt.figure(figsize=(8.0, 6.0))
    densities = [all_metrics[x]["stats"]["avg_face_pixel_density"] for x in sorted_IDs]
    plt.scatter(densities, [all_metrics[id]["human1_vs_machine"]["accuracy"] for id in sorted_IDs])
    plt.xlabel("Face pixel denisty")
    plt.ylabel("iCatcher accuracy")
    plt.title("iCatcher accuracy versus average face pixel density per video")
    plt.savefig(Path(args.output_folder, 'iCatcher_face_density_acc.png'))
    plt.cla()
    plt.clf()


def plot_face_location_vs_accuracy(sorted_IDs, all_metrics, args):
    plt.figure(figsize=(8.0, 6.0))
    stds = [all_metrics[x]["stats"]["avg_face_loc_std"] for x in sorted_IDs]
    plt.scatter(stds, [all_metrics[id]["human1_vs_machine"]["accuracy"] for id in sorted_IDs])
    plt.xlabel("Face location std in pixels")
    plt.ylabel("iCatcher accuracy")
    plt.title("iCatcher accuracy versus face location pixel std")
    plt.savefig(Path(args.output_folder, 'iCatcher_face_location_std_acc.png'))
    plt.cla()
    plt.clf()


def create_cache_metrics(args, force_create=False):
    """
    creates cache that can be used instead of parsing the annotation files from scratch each time
    :return:
    """
    metric_save_path = Path(args.output_folder, "metrics.p")
    if metric_save_path.is_file() and not force_create:
        all_metrics = pickle.load(open(metric_save_path, "rb"))
    else:
        machine_annotation = []
        human_annotation = []
        human_annotation2 = []

        # Get a list of all machine annotation files
        for file in Path(args.human_codings_folder).glob("*"):
            human_annotation.append(file.stem)
            human_ext = file.suffix
        for file in Path(args.human2_codings_folder).glob("*"):
            human_annotation2.append(file.stem)
            human2_ext = file.suffix
        for file in Path(args.machine_codings_folder).glob("*"):
            machine_annotation.append(file.stem)
            machine_ext = file.suffix

        coding_intersect = set(human_annotation2).intersection(set(human_annotation))
        coding_intersect = coding_intersect.intersection(set(machine_annotation))
        # sort the file paths alphabetically to pair them up
        coding_intersect = sorted(list(coding_intersect))
        assert len(coding_intersect) > 0
        all_metrics = {}
        for i, code_file in enumerate(coding_intersect):
            logging.info("computing stats: {} / {}".format(i, len(coding_intersect) - 1))
            human_coding_file = Path(args.human_codings_folder, code_file + human_ext)
            human_coding_file2 = Path(args.human2_codings_folder, code_file + human2_ext)
            machine_coding_file = Path(args.machine_codings_folder, code_file + machine_ext)
            key = human_coding_file.stem
            all_metrics[key] = compare_coding_files(human_coding_file, human_coding_file2, machine_coding_file, args)
            # other stats
            all_metrics[key]["stats"] = {}
            all_metrics[key]["stats"]["avg_face_pixel_density"] = get_face_pixel_density(key, args.faces_folder)
            all_metrics[key]["stats"]["avg_face_loc_std"] = get_face_location_std(key, args.faces_folder)
        # Store in disk for faster access next time:
        pickle.dump(all_metrics, open(metric_save_path, "wb"))
    return all_metrics


def put_text(img, class_name):
    font = cv2.FONT_HERSHEY_SIMPLEX
    top_left_corner_text = (10, 30)
    font_scale = 1
    font_color = (255, 255, 255)
    line_type = 2

    cv2.putText(img, class_name,
                top_left_corner_text,
                font,
                font_scale,
                font_color,
                line_type)
    return img


def put_arrow(img, class_name, face):
    arrow_start_x = int(face[0] + 0.5 * face[2])
    arrow_end_x = int(face[0] + 0.1 * face[2] if class_name == "left" else face[0] + 0.9 * face[2])
    arrow_y = int(face[1] + 0.8 * face[3])
    img = cv2.arrowedLine(img,
                          (arrow_start_x, arrow_y),
                          (arrow_end_x, arrow_y),
                          (0, 255, 0),
                          thickness=3,
                          tipLength=0.4)
    return img


def put_rectangle(popped_frame, face):
    color = (0, 255, 0)  # green
    thickness = 2
    popped_frame = cv2.rectangle(popped_frame,
                                 (face[0], face[1]), (face[0] + face[2], face[1] + face[3]),
                                 color,
                                 thickness)
    return popped_frame


def make_gallery(array, save_path, ncols=3):
    nindex, height, width, intensity = array.shape
    nrows = nindex//ncols
    assert nindex == nrows*ncols
    # want result.shape = (height*nrows, width*ncols, intensity)
    result = (array.reshape(nrows, ncols, height, width, intensity)
              .swapaxes(1,2)
              .reshape(height*nrows, width*ncols, intensity))
    plt.imshow(result)
    plt.savefig(save_path)
    plt.cla()
    plt.clf()


def make_gridview(array, ncols=3):
    nindex, height, width, intensity = array.shape
    nrows = nindex//ncols
    assert nindex == nrows*ncols
    # want result.shape = (height*nrows, width*ncols, intensity)
    result = (array.reshape(nrows, ncols, height, width, intensity)
              .swapaxes(1,2)
              .reshape(height*nrows, width*ncols, intensity))
    return result


def sandbox(metrics):
    for key in metrics.keys():
        acc_human = metrics[key]["human1_vs_human2"]["accuracy"]
        if acc_human >= 100.0:
            print("{}, human accuracy: {}".format(key.stem + ".mp4", acc_human))
        acc_machine = metrics[key]["human1_vs_machine"]["accuracy"]
        if acc_machine <= 60.0:
            print("{}, machine accuracy: {}".format(key.stem + ".mp4", acc_machine))


if __name__ == "__main__":
    args = parse_arguments_for_visualizations()
    if args.log:
        args.log.parent.mkdir(parents=True, exist_ok=True)
        logging.basicConfig(filename=args.log, filemode='w', level=args.verbosity.upper())
    else:
        logging.basicConfig(level=args.verbosity.upper())

    all_metrics = create_cache_metrics(args, force_create=False)
    # sort by accuracy
    sorted_ids = sorted(list(all_metrics.keys()),
                        key=lambda x: all_metrics[x]["human1_vs_machine"]["accuracy"])

    # sandbox(all_metrics)
    generate_collage_plot(sorted_ids, all_metrics, args.output_folder)
    # generate_frame_comparison(random.sample(sorted_ids, min(len(sorted_ids), 8)), all_metrics, args)
    generate_frame_by_frame_comparisons(sorted_ids, all_metrics, args)
    plot_face_pixel_density_vs_accuracy(sorted_ids, all_metrics, args)
    plot_face_location_vs_accuracy(sorted_ids, all_metrics, args)
    plot_inference_accuracy_vs_human_agreement(sorted_ids, all_metrics, args)
    plot_luminance_vs_accuracy(sorted_ids, all_metrics, args)


# def get_open_gaze_label(time_ms, ID):
#     for video_path, label_list in OPEN_GAZE_LABELS.items():
#         if ID in video_path:
#             try:
#                 return [time_ms, 0, label_list[time_ms_to_frame(time_ms)]]
#             except IndexError:
#                 return [time_ms, "none", 0]
#     raise ValueError("ID not found in opengaze labels")

# def parse_open_gaze(ID):
#     labels = []
#     for video_path, label_list in OPEN_GAZE_LABELS.items():
#         if ID in video_path:
#             for frame_idx, label in enumerate(label_list):
#
#                 if frame_idx == 0:
#                     labels.append([0, 1, label])
#                 else:
#                     if label != labels[-1][2]:
#                         labels.append([frame_to_time_ms(frame_idx), 1, label])
#             return labels
#     return
#     raise ValueError("ID not found in opengaze labels")
