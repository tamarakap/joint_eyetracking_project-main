import shutil
from pathlib import Path
import cv2
import numpy as np
import time
import logging
import visualize
from PIL import Image
import face_classifier.fc_model
import face_classifier.fc_data
import face_classifier.fc_eval
import torch
from tqdm import tqdm
import options
import parsers
import video
import csv


def build_video_dataset(raw_dataset_path, tsv_location):
    video_dataset = {}
    # search for videos
    for file in Path(raw_dataset_path / "videos").glob("*"):
        if file.is_file():
            video_dataset[file] = {"video_id": "-".join(file.stem.split("_")[2].split("-")[1:]),
                                   "video_path": file,
                                   "video_suffix": file.suffix,
                                   "in_tsv": False,
                                   "has_1coding": False,
                                   "has_2coding": False,
                                   "first_coding_file": None,
                                   "second_coding_file": None,
                                   "child_id": None,
                                   "split": None}
    # parse tsv file
    rows = []
    with open(tsv_location) as file:
        tsv_file = csv.reader(file, delimiter="\t")
        header = next(tsv_file)
        for row in tsv_file:
            rows.append(row)
    # fill video dataset with information from tsv
    video_id = header.index("videoID")
    child_id = header.index("childID")
    which_dataset = header.index("which.dataset")  # train, val or test video
    tsv_videos = [row[video_id] for row in rows]
    for entry in video_dataset.values():
        if entry["video_id"] in tsv_videos:
            entry["in_tsv"] = True
            index = tsv_videos.index(entry["video_id"])
            entry["child_id"] = rows[index][child_id]
            entry["split"] = rows[index][which_dataset]
    # fill video dataset with information from folders
    first_coding_files = [f for f in Path(raw_dataset_path / "annotations" / 'coder1').glob("*.txt")]
    first_coding_files_video_ids = ["-".join(f.stem.split("_")[2].split("-")[1:]) for f in first_coding_files]
    second_coding_files = [f for f in Path(raw_dataset_path / "annotations" / 'coder2').glob("*.txt")]
    second_coding_files_video_ids = ["-".join(f.stem.split("_")[2].split("-")[1:]) for f in second_coding_files]
    for entry in video_dataset.values():
        if entry["video_id"] in first_coding_files_video_ids:
            index = first_coding_files_video_ids.index(entry["video_id"])
            entry["has_1coding"] = True
            entry["first_coding_file"] = first_coding_files[index]
        if entry["video_id"] in second_coding_files_video_ids:
            index = second_coding_files_video_ids.index(entry["video_id"])
            entry["has_2coding"] = True
            entry["second_coding_file"] = second_coding_files[index]
        if entry["has_2coding"]:  # just a small sanity check
            assert entry["has_1coding"]
    return video_dataset


def preprocess_raw_lookit_dataset(args, force_create=False):
    """
    Organizes the raw videos downloaded from the Lookit platform.
    It puts the videos with annotations into raw_videos folder and
    the annotation from the first and second human annotators into coding_first and coding_second folders respectively.
    :param force_create: forces creation of files even if they exist
    :return:
    """
    np.random.seed(seed=args.seed)  # seed the random generator
    args.video_folder.mkdir(parents=True, exist_ok=True)
    args.label_folder.mkdir(parents=True, exist_ok=True)
    args.label2_folder.mkdir(parents=True, exist_ok=True)
    args.faces_folder.mkdir(parents=True, exist_ok=True)
    tsv_file = Path(args.raw_dataset_path / "prephys_split0_videos.tsv")
    video_dataset = build_video_dataset(args.raw_dataset_path, tsv_file)
    # print some stats
    with open(tsv_file, 'r') as tsv_fp:
        tsv_videos = len(tsv_fp.readlines())
    valid_videos = len(video_dataset.keys())
    valid_videos_in_tsv = len([x for x in video_dataset.values() if x["in_tsv"] and x["has_1coding"]])
    doubly_coded = len([x for x in video_dataset.values() if x["in_tsv"] and x["has_2coding"]])
    unique_children = len(np.unique([x["child_id"] for x in video_dataset.values() if x["in_tsv"] and x["has_1coding"]]))
    logging.info("tsv videos: {},"
                 " found videos: {},"
                 " found videos in tsv: {},"
                 " doubly coded: {},"
                 " unique children: {}".format(tsv_videos, valid_videos, valid_videos_in_tsv, doubly_coded, unique_children))

    # filter out videos according to split type
    if args.split_type == "all":
        videos = [x for x in video_dataset.values() if x["in_tsv"] and x["has_1coding"]]
    elif args.split_type == "split0_train":
        videos = [x for x in video_dataset.values() if x["in_tsv"] and x["has_1coding"] and (x["split"] == "1_train" or x["split"] == "1_validate")]
    elif args.split_type == "split0_test":
        videos = [x for x in video_dataset.values() if x["in_tsv"] and x["has_1coding"] and x["split"] == "2_test"]
    else:
        raise NotImplementedError
    videos = np.array(videos)

    # filter out videos according to one_video_per_child_policy and train_val_disjoint
    if args.one_video_per_child_policy == "include_all":
        double_coded = [x for x in videos if x["has_2coding"]]
        threshold = int(len(videos) * args.val_percent)
        val_set = np.random.choice(double_coded, size=threshold, replace=False)
        if args.train_val_disjoint:
            val_children_id = [x["child_id"] for x in val_set]
            train_set = [x for x in videos if x["child_id"] not in val_children_id and x not in val_set]
            train_set = np.array(train_set)
        else:
            train_set = [x for x in videos if x not in val_set]
    elif args.one_video_per_child_policy == "unique_only":
        video_children_id = [x["child_id"] for x in videos]
        _, indices = np.unique(video_children_id, return_index=True)
        unique_videos = videos[indices]
        double_coded = [x for x in unique_videos if x["has_2coding"]]
        threshold = min(int(len(unique_videos) * args.val_percent), len(double_coded))
        val_set = np.random.choice(double_coded, size=threshold, replace=False)
        train_set = np.array([x for x in unique_videos if x not in val_set])
    elif args.one_video_per_child_policy == "unique_only_in_val":
        video_children_id = [x["child_id"] for x in videos]
        _, unique_indices = np.unique(video_children_id, return_index=True)
        double_coded_and_unique_videos = [x for x in videos[unique_indices] if x["has_2coding"]]
        threshold = min(int(len(videos) * args.val_percent), len(double_coded_and_unique_videos))
        val_set = np.random.choice(double_coded_and_unique_videos, size=threshold, replace=False)
        if args.train_val_disjoint:
            val_children_id = [x["child_id"] for x in val_set]
            train_set = [x for x in videos if x["child_id"] not in val_children_id and x not in val_set]
            train_set = np.array(train_set)
        else:
            train_set = np.array([x for x in videos if x not in val_set])
    elif args.one_video_per_child_policy == "unique_only_in_train":
        val_set = [x for x in videos if x["has_2coding"]]
        val_children_id = [x["child_id"] for x in val_set]
        if args.train_val_disjoint:
            temp_train_set = [x for x in videos if x["child_id"] not in val_children_id and x not in val_set]
        else:
            temp_train_set = [x for x in videos if x not in val_set]
        temp_train_video_children_id = [x["child_id"] for x in temp_train_set]
        _, unique_indices = np.unique(temp_train_video_children_id, return_index=True)
        train_set = np.array(temp_train_video_children_id[unique_indices])
    else:
        raise NotImplementedError

    # create final structured dataset
    logging.info('[preprocess_raw] training set: {} validation set: {}'.format(len(train_set), len(val_set)))

    for video_file in train_set:
        video_file_path = video_file["video_path"]
        coding_file_path = video_file["first_coding_file"]
        assert video_file_path.is_file()
        assert coding_file_path.is_file()
        shutil.copyfile(video_file_path, Path(args.video_folder / (video_file["video_id"] + video_file_path.suffix)))
        shutil.copyfile(coding_file_path, Path(args.label_folder / (video_file["video_id"] + coding_file_path.suffix)))

    for video_file in val_set:
        video_file_path = video_file["video_path"]
        coding_file_path = video_file["first_coding_file"]
        second_coding_file_path = video_file["second_coding_file"]
        assert video_file_path.is_file()
        assert coding_file_path.is_file()
        shutil.copyfile(video_file_path, Path(args.video_folder / (video_file["video_id"] + video_file_path.suffix)))
        shutil.copyfile(coding_file_path, Path(args.label_folder / (video_file["video_id"] + coding_file_path.suffix)))
        shutil.copyfile(second_coding_file_path, Path(args.label2_folder / (video_file["video_id"] + second_coding_file_path.suffix)))


def preprocess_raw_generic_dataset(args, force_create=False):
    args.video_folder.mkdir(parents=True, exist_ok=True)
    args.label_folder.mkdir(parents=True, exist_ok=True)
    args.label2_folder.mkdir(parents=True, exist_ok=True)
    args.faces_folder.mkdir(parents=True, exist_ok=True)

    raw_videos_path = Path(args.raw_dataset_path / 'videos')
    raw_coding_first_path = Path(args.raw_dataset_path / 'coding_first')
    raw_coding_second_path = Path(args.raw_dataset_path / 'coding_second')

    videos = [f.stem for f in raw_videos_path.glob("*.mp4")]
    coding_first = ["_".join(f.stem.split("_")[:-1]) for f in raw_coding_first_path.glob("*")]
    coding_second = ["_".join(f.stem.split("_")[:-1]) for f in raw_coding_second_path.glob("*")]
    coding_ext = next(raw_coding_first_path.glob("*")).suffix

    logging.info('[preprocess_raw] coding_first: {}'.format(len(coding_first)))
    logging.info('[preprocess_raw] coding_second: {}'.format(len(coding_second)))
    logging.info('[preprocess_raw] videos: {}'.format(len(videos)))

    training_set = set(videos).intersection(set(coding_first))
    test_set = set(videos).intersection(set(coding_first)).intersection(set(coding_second))
    for i, file in enumerate(sorted(list(training_set))):
        if not Path(args.video_folder, (file + '.mp4')).is_file() or force_create:
            shutil.copyfile(raw_videos_path / (file + '.mp4'), args.video_folder / (file + '.mp4'))
        if not Path(args.label_folder, (file + coding_ext)).is_file() or force_create:
            real_file = next(raw_coding_first_path.glob(file+"*"))
            shutil.copyfile(real_file, args.label_folder / (file + coding_ext))

    for i, file in enumerate(sorted(list(test_set))):
        if not Path(args.video_folder, (file + '.mp4')).is_file() or force_create:
            shutil.copyfile(raw_videos_path / (file + '.mp4'), args.video_folder / (file + '.mp4'))
        if not Path(args.label_folder, (file + coding_ext)).is_file() or force_create:
            real_file = next(raw_coding_first_path.glob(file + "*"))
            shutil.copyfile(real_file, args.label_folder / (file + coding_ext))
        if not Path(args.label2_folder, (file + coding_ext)).is_file() or force_create:
            real_file = next(raw_coding_second_path.glob(file + "*"))
            shutil.copyfile(real_file, args.label2_folder / (file + coding_ext))


def preprocess_raw_princeton_dataset(args, force_create=False):
    """
    Organizes the raw videos from marchman/princeton.
    It puts the videos with annotations into raw_videos folder and
    the annotation from the first and second human annotators into coding_first and coding_second folders respectively.
    :param force_create: forces creation of files even if they exist
    :return:
    :param args:
    :param force_create:
    :return:
    """
    args.video_folder.mkdir(parents=True, exist_ok=True)
    args.label_folder.mkdir(parents=True, exist_ok=True)
    args.label2_folder.mkdir(parents=True, exist_ok=True)
    args.faces_folder.mkdir(parents=True, exist_ok=True)
    videos = [f.stem for f in Path(args.raw_dataset_path / 'Mov').glob("*.mov")]
    coding_first = [f.stem for f in Path(args.raw_dataset_path / 'VCX').glob("*.vcx")]
    coding_second = [f.stem for f in Path(args.raw_dataset_path / 'VCX2').glob("*.vcx")]

    logging.info('[preprocess_raw] coding_first: {}'.format(len(coding_first)))
    logging.info('[preprocess_raw] coding_second: {}'.format(len(coding_second)))
    logging.info('[preprocess_raw] videos: {}'.format(len(videos)))

    training_set = set(videos).intersection(set(coding_first))
    test_set = set(videos).intersection(set(coding_first)).intersection(set(coding_second))
    # val_percent = 20  # 20%
    # train_factor = (100 - val_percent) / 100
    # if i < (int(len(intersection) * train_factor)):
    for i, file in enumerate(sorted(list(training_set))):
        if not Path(args.video_folder, (file + '.mov')).is_file() or force_create:
            shutil.copyfile(args.raw_dataset_path / 'Mov' / (file + '.mov'), args.video_folder / (file+'.mov'))
        if not Path(args.label_folder, (file + '.vcx')).is_file() or force_create:
            shutil.copyfile(args.raw_dataset_path / 'VCX' / (file + ".vcx"), args.label_folder / (file + ".vcx"))

    for i, file in enumerate(sorted(list(test_set))):
        if not Path(args.video_folder, (file + '.mov')).is_file() or force_create:
            shutil.copyfile(args.raw_dataset_path / 'Mov' / (file + '.mov'), args.video_folder / (file+'.mov'))
        if not Path(args.label_folder, (file + '.vcx')).is_file() or force_create:
            shutil.copyfile(args.raw_dataset_path / 'VCX' / (file + ".vcx"), args.label_folder / (file + ".vcx"))
        if not Path(args.label2_folder, (file + '.vcx')).is_file() or force_create:
            shutil.copyfile(args.raw_dataset_path / 'VCX2' / (file + ".vcx"), args.label2_folder / (file + ".vcx"))


def detect_face_opencv_dnn(net, frame, conf_threshold):
    """
    Uses a pretrained face detection model to generate facial bounding boxes,
    with the format [x, y, width, height] where [x, y] is the lower left coord
    :param net:
    :param frame:
    :param conf_threshold:
    :return:
    """
    frameHeight = frame.shape[0]
    frameWidth = frame.shape[1]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), [104, 117, 123], False, False)
    net.setInput(blob)
    detections = net.forward()
    bboxes = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > conf_threshold:
            x1 = max(int(detections[0, 0, i, 3] * frameWidth), 0)  # left side of box
            y1 = max(int(detections[0, 0, i, 4] * frameHeight), 0)  # top side of box
            if x1 >= frameWidth or y1 >= frameHeight:  # if they are larger than image size, bbox is invalid
                continue
            x2 = min(int(detections[0, 0, i, 5] * frameWidth), frameWidth)  # either right side of box or frame width
            y2 = min(int(detections[0, 0, i, 6] * frameHeight), frameHeight)  # either the bottom side of box of frame height
            bboxes.append([x1, y1, x2-x1, y2-y1])  # (left, top, width, height)
    return bboxes


def process_dataset_lowest_face(args, gaze_labels_only=False, force_create=False):
    """
    process the dataset using the "lowest" face mechanism
    creates:
        face crops (all recognized face crops that abide a threshold value 0.7)
        boxes (the crop parameters - area / location etc)
        gaze_labels (the label of the crop as annotated by first human)
        face_labels (index of the face that was selected using lowest face mechanism)
    :param gaze_labels_only: only creates gaze labels.
    :param force_create: forces creation of files even if they exist
    :return:
    """
    classes = {"away": 0, "left": 1, "right": 2}
    video_list = sorted(list(args.video_folder.glob("*")))
    net = cv2.dnn.readNetFromCaffe(str(args.config_file), str(args.face_model_file))
    for video_file in video_list:
        st_time = time.time()
        logging.info("[process_lkt_legacy] Proccessing %s" % video_file.name)
        cur_video_folder = Path.joinpath(args.faces_folder / video_file.stem)
        cur_video_folder.mkdir(parents=True, exist_ok=True)
        img_folder = Path.joinpath(args.faces_folder, video_file.stem, 'img')
        img_folder.mkdir(parents=True, exist_ok=True)
        box_folder = Path.joinpath(args.faces_folder, video_file.stem, 'box')
        box_folder.mkdir(parents=True, exist_ok=True)

        frame_counter = 0
        no_face_counter = 0
        no_annotation_counter = 0
        valid_counter = 0
        gaze_labels = []
        face_labels = []

        cap = cv2.VideoCapture(str(video_file))
        # make sure target fps is around 30
        vfr, meta_data = video.is_video_vfr(video_file, get_meta_data=True)
        fps = video.get_fps(video_file)
        if vfr:
            logging.warning("video file: {} has variable frame rate".format(str(video_file)))
            logging.info(str(meta_data))
            frame_info, vfr_frame_counter, _ = video.get_frame_information(video_file)
        else:
            print("video fps: {}".format(fps))

        if args.raw_dataset_type == "princeton":
            assert abs(fps - 30) < 0.1
            parser = parsers.PrincetonParser(30,
                                             ".vcx",
                                             args.label_folder,
                                             Path(args.raw_dataset_path, "start_times_visitA.csv"))
        elif args.raw_dataset_type == "generic":
            ext = next(Path(args.label_folder).glob("*")).suffix
            parser = parsers.PrefLookTimestampParser(fps=fps,
                                                     labels_folder=args.label_folder,
                                                     ext=ext,
                                                     return_time_stamps=vfr)
        elif args.raw_dataset_type == "lookit":
            ext = next(Path(args.label_folder).glob("*")).suffix
            parser = parsers.LookitParser(fps=fps,
                                          labels_folder=args.label_folder,
                                          ext=ext,
                                          return_time_stamps=vfr)
        else:
            raise NotImplementedError
        responses, _, end = parser.parse(video_file.stem)
        ret_val, frame = cap.read()
        while ret_val:
            if responses:
                logging.info("[process_lkt_legacy] Processing frame: {}".format(frame_counter))
                if vfr:
                    frame_stamp = frame_info[frame_counter]
                else:
                    frame_stamp = frame_counter
                if responses[0][0] <= frame_stamp <= end:  # only iterate on annotated frames
                    # find closest (previous) response this frame belongs to
                    q = [index for index, val in enumerate(responses) if frame_stamp >= val[0]]
                    response_index = max(q)
                    if responses[response_index][1]:  # make sure response is valid
                        gaze_class = responses[response_index][2]
                        assert gaze_class in classes
                        gaze_labels.append(classes[gaze_class])
                        if not gaze_labels_only:
                            bbox = detect_face_opencv_dnn(net, frame, args.face_detector_confidence)
                            if not bbox:
                                no_face_counter += 1
                                face_labels.append(-2)
                                logging.info("[process_lkt_legacy] Video %s: Face not detected in frame %d" %
                                             (video_file.name, frame_counter))
                            else:
                                # select lowest face, probably belongs to kid: face = min(bbox, key=lambda x: x[3] - x[1])
                                selected_face = 0
                                min_value = bbox[0][3] - bbox[0][1]
                                # gaze_class = responses[response_index][2]
                                for i, face in enumerate(bbox):
                                    if bbox[i][3] - bbox[i][1] < min_value:
                                        min_value = bbox[i][3] - bbox[i][1]
                                        selected_face = i
                                    crop_img = frame[face[1]:face[1] + face[3], face[0]:face[0] + face[2]]
                                    # resized_img = cv2.resize(crop_img, (100, 100))
                                    resized_img = crop_img  # do not lose information in pre-processing step!
                                    face_box = np.array([face[1], face[1] + face[3], face[0], face[0] + face[2]])
                                    img_shape = np.array(frame.shape)
                                    ratio = np.array([face_box[0] / img_shape[0], face_box[1] / img_shape[0],
                                                      face_box[2] / img_shape[1], face_box[3] / img_shape[1]])
                                    face_size = (ratio[1] - ratio[0]) * (ratio[3] - ratio[2])
                                    face_ver = (ratio[0] + ratio[1]) / 2
                                    face_hor = (ratio[2] + ratio[3]) / 2
                                    face_height = ratio[1] - ratio[0]
                                    face_width = ratio[3] - ratio[2]
                                    feature_dict = {
                                        'face_box': face_box,
                                        'img_shape': img_shape,
                                        'face_size': face_size,
                                        'face_ver': face_ver,
                                        'face_hor': face_hor,
                                        'face_height': face_height,
                                        'face_width': face_width
                                    }
                                    img_filename = img_folder / f'{frame_counter:05d}_{i:01d}.png'
                                    if not img_filename.is_file() or force_create:
                                        cv2.imwrite(str(img_filename), resized_img)
                                    box_filename = box_folder / f'{frame_counter:05d}_{i:01d}.npy'
                                    if not box_filename.is_file() or force_create:
                                        np.save(str(box_filename), feature_dict)
                                valid_counter += 1
                                face_labels.append(selected_face)
                                # logging.info(f"valid frame in class {gaze_class}")
                    else:
                        no_annotation_counter += 1
                        gaze_labels.append(-2)
                        face_labels.append(-2)
                        logging.info("[process_lkt_legacy] Skipping since frame is invalid")
                else:
                    no_annotation_counter += 1
                    gaze_labels.append(-2)
                    face_labels.append(-2)
                    logging.info("[process_lkt_legacy] Skipping since frame not in range of annotation")
            else:
                no_annotation_counter += 1
                gaze_labels.append(-2)
                face_labels.append(-2)
                logging.info("[process_lkt_legacy] Skipping frame since parser reported no annotation")
            ret_val, frame = cap.read()
            frame_counter += 1
        if vfr:
            assert frame_counter == vfr_frame_counter
        # save gaze labels
        gaze_labels = np.array(gaze_labels)
        gaze_labels_filename = Path.joinpath(args.faces_folder, video_file.stem, 'gaze_labels.npy')
        if not gaze_labels_filename.is_file() or force_create:
            np.save(str(gaze_labels_filename), gaze_labels)
        if not gaze_labels_only:
            # save face labels
            face_labels = np.array(face_labels)
            face_labels_filename = Path.joinpath(args.faces_folder, video_file.stem, 'face_labels.npy')
            if not face_labels_filename.is_file() or force_create:
                np.save(str(face_labels_filename), face_labels)
        logging.info(
            "[process_lkt_legacy] Total frame: {}, No face: {}, No annotation: {}, Valid: {}".format(
                frame_counter, no_face_counter, no_annotation_counter, valid_counter))
        ed_time = time.time()
        logging.info('[process_lkt_legacy] Time used: %.2f sec' % (ed_time - st_time))


def generate_second_gaze_labels(args, force_create=False, visualize_confusion=False):
    """
    Processes the second annotator labels
    :param force_create: forces creation of files even if they exist
    :param visualize_confusion: if true, visualizes confusion between human annotators.
    :return:
    """
    classes = {"away": 0, "left": 1, "right": 2}
    video_list = list(args.video_folder.glob("*"))
    suffix = next(Path(args.label_folder).glob("*")).suffix
    for video_file in video_list:
        logging.info("[gen_2nd_labels] Video: %s" % video_file.name)
        if (args.label2_folder / (video_file.stem + suffix)).exists():
            fps = video.get_fps(video_file)
            vfr, meta_data = video.is_video_vfr(video_file, get_meta_data=True)
            if vfr:
                logging.warning("video file: {} has variable frame rate".format(str(video_file)))
                logging.info(str(meta_data))
                frame_info, vfr_frame_counter, _ = video.get_frame_information(video_file)
            if args.raw_dataset_type == "princeton":
                assert abs(fps - 30) < 0.1
                parser = parsers.PrincetonParser(30,
                                                 suffix,
                                                 args.label2_folder,
                                                 Path(args.raw_dataset_path, "start_times_visitA.csv"))
            elif args.raw_dataset_type == "generic":
                parser = parsers.PrefLookTimestampParser(fps=fps,
                                                         labels_folder=args.label2_folder,
                                                         ext=suffix,
                                                         return_time_stamps=vfr)
            elif args.raw_dataset_type == "lookit":
                parser = parsers.LookitParser(fps=fps,
                                              labels_folder=args.label2_folder,
                                              ext=suffix,
                                              return_time_stamps=vfr)
            else:
                raise NotImplementedError
            responses, _, end = parser.parse(video_file.stem)
            gaze_labels = np.load(str(Path.joinpath(args.faces_folder, video_file.stem, 'gaze_labels.npy')))
            gaze_labels_second = []
            for frame in range(gaze_labels.shape[0]):
                if vfr:
                    frame_stamp = frame_info[frame]
                else:
                    frame_stamp = frame
                if responses[0][0] <= frame_stamp <= end:  # only iterate on annotated frames
                    q = [index for index, val in enumerate(responses) if frame_stamp >= val[0]]
                    response_index = max(q)
                    if responses[response_index][1]:
                        gaze_class = responses[response_index][2]
                        assert gaze_class in classes
                        gaze_labels_second.append(classes[gaze_class])
                    else:
                        gaze_labels_second.append(-2)
                else:
                    gaze_labels_second.append(-2)
            gaze_labels_second = np.array(gaze_labels_second)
            gaze_labels_second_filename = Path.joinpath(args.faces_folder, video_file.stem, 'gaze_labels_second.npy')
            if not gaze_labels_second_filename.is_file() or force_create:
                np.save(str(gaze_labels_second_filename), gaze_labels_second)
        else:
            logging.info('[gen_2nd_labels] No label!')
    if visualize_confusion:
        visualize_human_confusion_matrix(Path(args.output_folder, "confusion.pdf"))


def visualize_human_confusion_matrix(path):
    """
    wrapper for calculating and visualizing confusion matrix with human annotations
    :return:
    """
    labels = []
    preds = []
    video_list = list(args.video_folder.glob("*"))
    for video_file in video_list:
        gaze_labels_second_filename = Path.joinpath(args.faces_folder, video_file.stem, 'gaze_labels_second.npy')
        if gaze_labels_second_filename.is_file():
            gaze_labels = np.load(str(Path.joinpath(args.faces_folder, video_file.stem, 'gaze_labels.npy')))
            gaze_labels_second = np.load(str(gaze_labels_second_filename))
            idxs = np.where((gaze_labels >= 0) & (gaze_labels_second >= 0))
            labels.extend(list(gaze_labels[idxs]))
            preds.extend(list(gaze_labels_second[idxs]))
    # human_dir = Path('plots', 'human')
    # human_dir.mkdir(exist_ok=True, parents=True)
    _, _ = visualize.calculate_confusion_matrix(labels, preds, path)


def gen_lookit_multi_face_subset(force_create=False):
    """
    Generates multi-face labels for training the face classifier
    :param force_create: forces creation of files even if they exist
    :return:
    """
    args.multi_face_folder.mkdir(exist_ok=True, parents=True)
    names = [f.stem for f in Path(args.video_folder).glob('*')]
    face_hist = np.zeros(10)
    total_datapoint = 0
    for name in names:
        logging.info(name)
        src_folder = args.faces_folder / name
        dst_folder = args.multi_face_folder / name
        dst_folder.mkdir(exist_ok=True, parents=True)
        (dst_folder / 'img').mkdir(exist_ok=True, parents=True)
        (dst_folder / 'box').mkdir(exist_ok=True, parents=True)
        face_labels = np.load(src_folder / 'face_labels.npy')
        files = list((src_folder / 'img').glob(f'*.png'))
        filenames = [f.stem for f in files]
        filenames = sorted(filenames)
        num_datapoint = 0
        for i in range(len(face_labels)):
            frame_str = "{:05d}".format(i)
            frame_face_str = frame_str + "_0"
            if not frame_face_str in filenames:
                face_hist[0] += 1
            else:
                faces = [frame_face_str]
                j = 1
                while (frame_str + "_{}".format(j)) in filenames:
                    faces.append(frame_str + "_{}".format(j))
                    j += 1
                face_hist[j] += 1
                if j > 1:
                    num_datapoint += 1
                    total_datapoint += 1
                    for face in faces:
                        dst_face_file = (dst_folder / 'img' / f'{name}_{face}.png')
                        if not dst_face_file.is_file() or force_create:
                            shutil.copy((src_folder / 'img' / (face + '.png')), dst_face_file)
                        dst_box_file = (dst_folder / 'box' / f'{name}_{face}.npy')
                        if not dst_box_file.is_file() or force_create:
                            shutil.copy((src_folder / 'box' / (face + '.npy')), dst_box_file)
                # if num_datapoint >= 1000:
                #     break
        logging.info('# multi-face datapoint:{}'.format(num_datapoint))
    logging.info('total # multi-face datapoint:{}'.format(total_datapoint))
    logging.info(face_hist)


def process_dataset_face_classifier(args, force_create=False):
    """
    further process a dataset using a trained face baby vs adult classifier and nearest patch mechanism
    :param args: comman line arguments
    :param force_create: forces creation of files even if they exist
    :return:
    """

    ## todo: remove test code from here
    # val_infant_files = [f.stem for f in (face_data_folder / 'val' / 'infant').glob('*.png')]
    # val_others_files = [f.stem for f in (face_data_folder / 'val' / 'others').glob('*.png')]
    # num_correct = 0
    # total = len(val_infant_files) + len(val_others_files)
    # for f in val_infant_files:
    #     if f[-1] == f[-3]:
    #         num_correct += 1
    # for f in val_others_files:
    #     if f[-1] != f[-3]:
    #         num_correct += 1
    # logging.info("\n[process_lkt] {}, {}, {}".format(num_correct, total, num_correct / total))

    # emulate command line arguments
    # replace with what was used to train the face classifier!
    class Args:
        def __init__(self):
            self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
            self.rotation = False
            self.cropping = False
            self.hor_flip = False
            self.ver_flip = False
            self.color = False
            self.erasing = False
            self.noise = False
            self.model = "vgg16"
            self.dropout = 0.0

    fc_args = Args()
    model, input_size = face_classifier.fc_model.init_face_classifier(fc_args,
                                                                      model_name=fc_args.model,
                                                                      num_classes=2,
                                                                      resume_from=args.fc_model)
    data_transforms = face_classifier.fc_eval.get_fc_data_transforms(fc_args, input_size)
    model.to(args.device)
    model.eval()
    video_files = sorted(list(args.video_folder.glob("*")))
    for video_file in tqdm(video_files):
        face_labels_fc_filename = Path.joinpath(args.faces_folder, video_file.stem, 'face_labels_fc.npy')
        if not face_labels_fc_filename.is_file() or force_create:
            logging.info(video_file.stem)
            files = list((args.faces_folder / video_file.stem / 'img').glob(f'*.png'))
            filenames = [f.stem for f in files]
            filenames = sorted(filenames)
            idx = 0
            face_labels = np.load(str(Path.joinpath(args.faces_folder, video_file.stem, 'face_labels.npy')))
            face_labels_fc = []
            hor, ver = 0.5, 1
            for frame in tqdm(range(face_labels.shape[0])):
                if face_labels[frame] < 0:
                    face_labels_fc.append(face_labels[frame])
                else:
                    faces = []
                    centers = []
                    while idx < len(filenames) and (int(filenames[idx][:5]) == frame):
                        img = Image.open(args.faces_folder / video_file.stem / 'img' / (filenames[idx] + '.png')).convert(
                            'RGB')
                        box = np.load(args.faces_folder / video_file.stem / 'box' / (filenames[idx] + '.npy'),
                                      allow_pickle=True).item()
                        centers.append([box['face_hor'], box['face_ver']])
                        img = data_transforms['val'](img)
                        faces.append(img)
                        idx += 1
                    centers = np.stack(centers)
                    faces = torch.stack(faces).to(args.device)
                    output = model(faces)
                    _, preds = torch.max(output, 1)
                    preds = preds.cpu().numpy()
                    idxs = np.where(preds == 0)[0]
                    centers = centers[idxs]
                    if centers.shape[0] == 0:
                        face_labels_fc.append(-1)
                    else:
                        dis = np.sqrt((centers[:, 0] - hor) ** 2 + (centers[:, 1] - ver) ** 2)
                        i = np.argmin(dis)
                        face_labels_fc.append(idxs[i])
                        hor, ver = centers[i]
            face_labels_fc = np.array(face_labels_fc)
            np.save(str(face_labels_fc_filename), face_labels_fc)


def report_dataset_stats(args):
    """
    prints out a list of training and test videos according to the heuristic that doubly coded videos are test set.
    :return:
    """
    all_videos = []
    test_videos = []
    raw_videos = []
    for path in sorted(args.label_folder.glob("*")):
        all_videos.append(path.name)
    for path in sorted(args.label2_folder.glob("*")):
        test_videos.append(path.name)
    for path in sorted(args.video_folder.glob("*")):
        raw_videos.append(path)
    train_videos = [x for x in all_videos if x not in test_videos]
    logging.info("train videos: [{0}]".format(', '.join(map(str, train_videos))))
    logging.info("test videos: [{0}]".format(', '.join(map(str, test_videos))))
    visualize_human_confusion_matrix(Path(args.output_folder, "confusion.pdf"))


if __name__ == "__main__":
    args = options.parse_arguments_for_preprocess()
    if args.log:
        args.log.parent.mkdir(parents=True, exist_ok=True)
        logging.basicConfig(filename=args.log, filemode='w', level=args.verbosity.upper())
    else:
        logging.basicConfig(level=args.verbosity.upper())

    if args.raw_dataset_type == "lookit":
        preprocess_raw_lookit_dataset(args, force_create=False)
    elif args.raw_dataset_type == "princeton":
        preprocess_raw_princeton_dataset(args, force_create=False)
    elif args.raw_dataset_type == "generic":
        preprocess_raw_generic_dataset(args, force_create=False)
    else:
        raise NotImplementedError

    process_dataset_lowest_face(args, gaze_labels_only=False, force_create=False)
    generate_second_gaze_labels(args, force_create=False, visualize_confusion=False)
    report_dataset_stats(args)
    # gen_lookit_multi_face_subset(force_create=False)
    # uncomment next line if face classifier was trained:
    process_dataset_face_classifier(args, force_create=False)
