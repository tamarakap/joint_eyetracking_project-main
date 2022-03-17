import cv2
from pathlib import Path
import numpy as np
from preprocess import detect_face_opencv_dnn, build_video_dataset
import options
import visualize
import logging
import face_classifier
import torch
import models
import data
from PIL import Image


class FaceClassifierArgs:
    def __init__(self, device):
        self.device = device
        self.rotation = False
        self.cropping = False
        self.hor_flip = False
        self.ver_flip = False
        self.color = False
        self.erasing = False
        self.noise = False
        self.model = "vgg16"
        self.dropout = 0.0


def prep_frame(popped_frame, bbox, class_text, face):
    """
    prepares a frame for visualization by adding text, rectangles and arrows.
    :param popped_frame: the frame for which to add the gizmo's to
    :param bbox: if this is not None, adds arrow and bounding box
    :param class_text: the text describing the class
    :param face: bounding box of face
    :return:
    """
    popped_frame = visualize.put_text(popped_frame, class_text)
    if bbox is not None:
        popped_frame = visualize.put_rectangle(popped_frame, face)
        if not class_text == "away" and not class_text == "off" and not class_text == "on":
            popped_frame = visualize.put_arrow(popped_frame, class_text, face)
    return popped_frame


def select_face(bboxes, frame, fc_model, fc_data_transforms, hor, ver):
    """
    selects a correct face from candidates bbox in frame
    :param bboxes: the bounding boxes of candidates
    :param frame: the frame
    :param fc_model: a classifier model, if passed it is used to decide.
    :param fc_data_transforms: the transformations to apply to the images before fc_model sees them
    :param hor: the last known horizontal correct face location
    :param ver: the last known vertical correct face location
    :return: the cropped face and its bbox data
    """
    if fc_model:
        centers = []
        faces = []
        for box in bboxes:
            crop_img = frame[box[1]:box[1] + box[3], box[0]:box[0] + box[2]]
            face_box = np.array([box[1], box[1] + box[3], box[0], box[0] + box[2]])
            img_shape = np.array(frame.shape)
            ratio = np.array([face_box[0] / img_shape[0], face_box[1] / img_shape[0],
                              face_box[2] / img_shape[1], face_box[3] / img_shape[1]])
            face_ver = (ratio[0] + ratio[1]) / 2
            face_hor = (ratio[2] + ratio[3]) / 2

            centers.append([face_hor, face_ver])
            img = crop_img
            img = fc_data_transforms['val'](img)
            faces.append(img)
        centers = np.stack(centers)
        faces = torch.stack(faces).to(args.device)
        output = fc_model(faces)
        _, preds = torch.max(output, 1)
        preds = preds.cpu().numpy()
        idxs = np.where(preds == 0)[0]
        if idxs.size == 0:
            bbox = None
        else:
            centers = centers[idxs]
            dis = np.sqrt((centers[:, 0] - hor) ** 2 + (centers[:, 1] - ver) ** 2)
            i = np.argmin(dis)
            # crop_img = faces[idxs[i]]
            bbox = bboxes[idxs[i]]
            # hor, ver = centers[i]
    else:
        # todo: improve face selection mechanism
        bbox = min(bboxes, key=lambda x: x[3] - x[1])  # select lowest face in image, probably belongs to kid
        # crop_img = frame[bbox[1]:bbox[1] + bbox[3], bbox[0]:bbox[0] + bbox[2]]
    return bbox


def extract_crop(frame, bbox, opt):
    """
    extracts a crop from a frame using bbox, and transforms it
    :param frame: the frame
    :param bbox: opencv bbox 4x1
    :param opt: command line options
    :return: the crop and the 5x1 box features
    """
    if bbox is None:
        return None, None
    img_shape = np.array(frame.shape)
    face_box = np.array([bbox[1], bbox[1] + bbox[3], bbox[0], bbox[0] + bbox[2]])
    crop = frame[bbox[1]:bbox[1] + bbox[3], bbox[0]:bbox[0] + bbox[2]]

    test_transforms = data.DataTransforms(opt.image_size).transformations["test"]
    # crop2 = cv2.resize(crop, (opt.image_size, opt.image_size)) * 1. / 255
    # crop2 = np.expand_dims(crop2, axis=0)
    # crop2 -= np.array(opt.per_channel_mean)
    # crop2 /= (np.array(opt.per_channel_std) + 1e-6)
    crop = test_transforms(Image.fromarray(crop))
    crop = crop.permute(1, 2, 0).unsqueeze(0).numpy()
    ratio = np.array([face_box[0] / img_shape[0], face_box[1] / img_shape[0],
                      face_box[2] / img_shape[1], face_box[3] / img_shape[1]])
    face_size = (ratio[1] - ratio[0]) * (ratio[3] - ratio[2])
    face_ver = (ratio[0] + ratio[1]) / 2
    face_hor = (ratio[2] + ratio[3]) / 2
    face_height = ratio[1] - ratio[0]
    face_width = ratio[3] - ratio[2]
    my_box = np.array([face_size, face_ver, face_hor, face_height, face_width])
    return crop, my_box


def predict_from_video(opt):
    """
    perform prediction on a stream or video file(s) using a network.
    output can be of various kinds, see options for details.
    :param opt:
    :return:
    """
    # initialize
    opt.frames_per_datapoint = 10
    opt.frames_stride_size = 2
    sequence_length = 9
    loc = -5
    classes = {'noface': -2, 'nobabyface': -1, 'away': 0, 'left': 1, 'right': 2}
    reverse_classes = {-2: 'away', -1: 'away', 0: 'away', 1: 'left', 2: 'right'}
    logging.info("using the following values for per-channel mean: {}".format(opt.per_channel_mean))
    logging.info("using the following values for per-channel std: {}".format(opt.per_channel_std))
    face_detector_model_file = Path("models", "face_model.caffemodel")
    config_file = Path("models", "config.prototxt")
    path_to_primary_model = opt.model
    primary_model = models.GazeCodingModel(opt).to(opt.device)
    if opt.device == 'cpu':
        state_dict = torch.load(str(path_to_primary_model), map_location=torch.device(opt.device))
    else:
        state_dict = torch.load(str(path_to_primary_model))
    try:
        primary_model.load_state_dict(state_dict)
    except RuntimeError:  # deal with old models that were encapsulated with "net"
        from collections import OrderedDict
        new_dict = OrderedDict()
        for i in range(len(state_dict)):
            k, v = state_dict.popitem(False)
            new_k = '.'.join(k.split(".")[1:])
            new_dict[new_k] = v
        primary_model.load_state_dict(new_dict)
    primary_model.eval()

    if opt.fc_model:
        fc_args = FaceClassifierArgs(opt.device)
        fc_model, fc_input_size = face_classifier.fc_model.init_face_classifier(fc_args,
                                                                                model_name=fc_args.model,
                                                                                num_classes=2,
                                                                                resume_from=opt.fc_model)
        fc_model.eval()
        fc_model.to(opt.device)
        fc_data_transforms = face_classifier.fc_eval.get_fc_data_transforms(fc_args,
                                                                            fc_input_size)
    else:
        fc_model = None
        fc_data_transforms = None
    # load face extractor model
    face_detector_model = cv2.dnn.readNetFromCaffe(str(config_file), str(face_detector_model_file))
    # set video source
    if opt.source_type == 'file':
        video_path = Path(opt.source)
        if video_path.is_dir():
            logging.warning("Video folder provided as source. Make sure it contains video files only.")
            video_paths = list(video_path.glob("*"))
            if opt.video_filter:
                if opt.video_filter.is_file():
                    if opt.video_filter.suffix == ".tsv":
                        video_dataset = build_video_dataset(opt.raw_dataset_path, opt.video_filter)
                        filter_files = [x for x in video_dataset.values() if
                                        x["in_tsv"] and x["has_1coding"] and x["split"] == "2_test"]
                        video_ids = [x["video_id"] for x in filter_files]
                        filter_files = [x["video_path"].stem for x in filter_files]
                    else:
                        with open(opt.video_filter_file, "r") as f:
                            filter_files = f.readlines()
                            filter_files = [Path(line.rstrip()).stem for line in filter_files]
                else:  # directory
                    filter_files = [x.stem for x in opt.video_filter.glob("*")]
                video_paths = [x for x in video_paths if x.stem in filter_files]
            video_paths = [str(x) for x in video_paths]
        elif video_path.is_file():
            video_paths = [str(video_path)]
        else:
            raise NotImplementedError
    else:
        video_paths = [int(opt.source)]
    for i in range(len(video_paths)):
        video_path = Path(str(video_paths[i]))
        answers = []
        image_sequence = []
        box_sequence = []
        frames = []
        frame_count = 0
        last_class_text = ""  # Initialize so that we see the first class assignment as an event to record
        logging.info("predicting on : {}".format(video_paths[i]))
        cap = cv2.VideoCapture(video_paths[i])
        # Get some basic info about the video
        width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        resolution = (int(width), int(height))
        framerate = int(cap.get(cv2.CAP_PROP_FPS))
        # If creating annotated video output, set up now
        if opt.output_video_path:
            fourcc = cv2.VideoWriter_fourcc(*"MP4V")  # may need to be adjusted per available codecs & OS
            my_suffix = video_path.suffix
            if not my_suffix:
                my_suffix = ".mp4"
            my_video_path = Path(opt.output_video_path, video_path.stem + "_output{}".format(my_suffix))
            video_output = cv2.VideoWriter(str(my_video_path), fourcc, framerate, resolution, True)
        if opt.output_annotation:
            if opt.output_format == "compressed":
                if video_ids:
                    my_output_file_path = Path(opt.output_annotation, video_ids[i])
            else:
                my_output_file_path = Path(opt.output_annotation, video_path.stem + opt.output_file_suffix)
                output_file = open(my_output_file_path, "w", newline="")
            if opt.output_format == "PrefLookTimestamp":
                # Write header
                output_file.write(
                    "Tracks: left, right, away, codingactive, outofframe\nTime,Duration,TrackName,comment\n\n")
        # iterate over frames
        ret_val, frame = cap.read()
        hor, ver = 0.5, 1  # used for improved selection of face
        while ret_val:
            frames.append(frame)
            cv2_bboxes = detect_face_opencv_dnn(face_detector_model, frame, 0.7)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # network was trained on RGB images.
            if not cv2_bboxes:
                answers.append(classes['noface'])  # if face detector fails, treat as away and mark invalid
                image = np.zeros((1, opt.image_size, opt.image_size, 3), np.float64)
                my_box = np.array([0, 0, 0, 0, 0])
                box_sequence.append(my_box)
                image_sequence.append((image, True))
            else:
                selected_bbox = select_face(cv2_bboxes, frame, fc_model, fc_data_transforms, hor, ver)
                crop, my_box = extract_crop(frame, selected_bbox, opt)
                if selected_bbox is None:
                    answers.append(classes['nobabyface'])  # if selecting face fails, treat as away and mark invalid
                    image = np.zeros((1, opt.image_size, opt.image_size, 3), np.float64)
                    image_sequence.append((image, True))
                    my_box = np.array([0, 0, 0, 0, 0])
                    box_sequence.append(my_box)
                else:
                    assert crop.size != 0  # what just happened?
                    answers.append(classes['left'])  # if face detector succeeds, treat as left and mark valid
                    image_sequence.append((crop, False))
                    box_sequence.append(my_box)
                    hor, ver = my_box[2], my_box[1]
            if len(image_sequence) == sequence_length:  # we have enough frames for prediction, predict for middle frame
                popped_frame = frames[loc]
                frames.pop(0)
                if not image_sequence[sequence_length // 2][1]:  # if middle image is valid
                    if opt.architecture == "icatcher+":
                        to_predict = {"imgs": torch.tensor([x[0] for x in image_sequence[0::2]], dtype=torch.float).squeeze().permute(0, 3, 1, 2).to(opt.device),
                                      "boxs": torch.tensor(box_sequence[::2], dtype=torch.float).to(opt.device)
                                      }
                        with torch.set_grad_enabled(False):
                            outputs = primary_model(to_predict)
                            _, prediction = torch.max(outputs, 1)
                            int32_pred = prediction.cpu().numpy()[0]
                    else:
                        raise NotImplementedError
                    answers[loc] = int32_pred
                image_sequence.pop(0)
                box_sequence.pop(0)
                class_text = reverse_classes[answers[loc]]
                if opt.on_off:
                    class_text = "off" if class_text == "away" else "on"
                # If show_output or output_video is true, add text label, bounding box for face, and arrow showing direction
                if opt.show_output:
                    prepped_frame = prep_frame(popped_frame, my_box, class_text, selected_bbox)
                    cv2.imshow('frame', prepped_frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                if opt.output_video_path:
                    prepped_frame = prep_frame(popped_frame, my_box, class_text, selected_bbox)
                    video_output.write(prepped_frame)
                # handle writing output to file
                if opt.output_annotation:
                    if opt.output_format == "raw_output":
                        output_file.write("{}, {}\n".format(frame_count + loc + 1, class_text))
                    elif opt.output_format == "PrefLookTimestamp":
                        if class_text != last_class_text:  # Record "event" for change of direction if code has changed
                            frame_ms = int((frame_count + loc + 1) * (1000. / framerate))
                            output_file.write("{},0,{}\n".format(frame_ms, class_text))
                            last_class_text = class_text
                logging.info("frame: {}, class: {}".format(str(frame_count + loc + 1), class_text))
            ret_val, frame = cap.read()
            frame_count += 1
        # finished processing a video file, cleanup
        if opt.show_output:
            cv2.destroyAllWindows()
        if opt.output_video_path:
            video_output.release()
        if opt.output_annotation:  # write footer to file
            if opt.output_format == "PrefLookTimestamp":
                start_ms = int((1000. / framerate) * (sequence_length // 2))
                end_ms = int((1000. / framerate) * frame_count)
                output_file.write("{},{},codingactive\n".format(start_ms, end_ms))
                output_file.close()
            elif opt.output_format == "compressed":
                np.savez(my_output_file_path, answers)
        cap.release()


if __name__ == '__main__':
    args = options.parse_arguments_for_testing()
    if args.log:
        args.log.parent.mkdir(parents=True, exist_ok=True)
        logging.basicConfig(filename=args.log, filemode='w', level=args.verbosity.upper())
    else:
        logging.basicConfig(level=args.verbosity.upper())
    predict_from_video(args)
