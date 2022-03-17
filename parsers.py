import xml.etree.ElementTree as ET
import logging
from pathlib import Path
import numpy as np
import csv


class BaseParser:
    def __init__(self):
        self.classes = {'away': 0, 'left': 1, 'right': 2}

    def parse(self, file):
        """
        returns a list of lists. each list contains the frame number (or timestamps), valid_flag, class
        where:
        frame number is zero indexed (or if timestamp, starts from 0.0)
        valid_flag is 1 if this frame has valid annotation, and 0 otherwise
        class is either away, left, right or off.

        list should only contain frames that have changes in class (compared to previous frame)
        i.e. if the video is labled ["away","away","away","right","right"]
        then only frame 0 and frame 3 will appear on the output list.

        :param file: the label file to parse.
        :return: None if failed, else: list of lists as described above, the frame which codings starts, and frame at which it ends
        """
        raise NotImplementedError


class TrivialParser(BaseParser):
    """
    A trivial toy parser that labels all video as "left" if input "file" is not None
    """
    def __init__(self):
        super().__init__()

    def parse(self, file):
        if file:
            return [[0, 1, "left"]]
        else:
            return None


class LookitParser(BaseParser):
    """
    a parser that parses Lookit format, a slightly different version of PrefLookTimestampParser.
    """
    def __init__(self, fps, labels_folder=None, ext=None, return_time_stamps=False):
        super().__init__()
        self.fps = fps
        self.return_time_stamps = return_time_stamps
        if ext:
            self.ext = ext
        if labels_folder:
            self.labels_folder = Path(labels_folder)
        self.classes = ["away", "left", "right"]
        self.exclude = ["outofframe", "preview", "instructions"]
        self.special = ["codingactive"]

    def load_and_sort(self, label_path):
        # load label file
        labels = np.genfromtxt(open(label_path, "rb"), dtype='str', delimiter=",", skip_header=3)
        # sort by time
        times = labels[:, 0].astype(np.int)
        sorting_indices = np.argsort(times)
        sorted_labels = labels[sorting_indices]
        return sorted_labels

    def parse(self, file, file_is_fullpath=False):
        if file_is_fullpath:
            label_path = Path(file)
        else:
            label_path = Path(self.labels_folder, file + self.ext)
        labels = self.load_and_sort(label_path)
        # initialize
        output = []
        prev_class = "none"
        prev_frame = -1
        # loop over legitimate class labels
        for i in range(len(labels)):
            frame = int(labels[i, 0])
            if labels[i, 2] in self.classes:
                cur_class = labels[i, 2]
                if prev_class != cur_class:
                    assert frame > prev_frame  # how can two labels be different but point to same time?
                output.append([frame, True, cur_class])
                prev_class = cur_class
                prev_frame = frame
            elif labels[i, 2] in self.special:
                assert False  # we do not permit codingactive label. though this can be easily supported.
        # extract "exclude" regions
        exclude_regions = self.find_exclude_regions(labels)
        merged_exclude_regions = self.merge_overlapping_intervals(exclude_regions)
        # loop over exclude regions and fix output
        for region in merged_exclude_regions:
            region_start = region[0]
            region_end = region[1]
            # deal with labels before region
            q = [index for index, value in enumerate(output) if value[0] < region_start]
            if q:
                last_overlap = max(q)
                prev_class = output[last_overlap][2]
                output.insert(last_overlap + 1, [region_start, False, prev_class])
            # deal with labels inside region
            q = [index for index, value in enumerate(output) if region_start <= value[0] < region_end]
            if q:
                for index in q:
                    output[index][1] = False
            # deal with last label inside region
            q = [index for index, value in enumerate(output) if value[0] <= region_end]
            if q:
                last_overlap = max(q)
                prev_class = output[last_overlap][2]
                output.insert(last_overlap + 1, [region_end, True, prev_class])
        # finish work
        if not self.return_time_stamps:  # convert to frame numbers
            for entry in output:
                entry[0] = int(int(entry[0]) * self.fps / 1000)
        start = int(output[0][0])
        annotations_end = int(output[-1][0])
        trial_times = self.get_trial_end_times(labels)
        trial_end = trial_times[-1]
        return output, start, trial_end

    def find_exclude_regions(self, sorted_labels):
        regions = []
        for entry in sorted_labels:
            if entry[2] in self.exclude:
                regions.append([int(entry[0]), int(entry[0]) + int(entry[1])])
        return regions

    def get_trial_end_times(self, sorted_labels):
        trials = []
        for i in range(len(sorted_labels)):
            if sorted_labels[i, 2] == "end":
                if not self.return_time_stamps:  # convert to frame numbers
                    frame = int(int(sorted_labels[i, 0]) * self.fps / 1000)
                else:
                    frame = int(sorted_labels[i, 0])
                trials.append(frame)
        return trials

    def merge_overlapping_intervals(self, arr):
        merged = []
        if arr:
            arr.sort(key=lambda interval: interval[0])
            merged.append(arr[0])
            for current in arr:
                previous = merged[-1]
                if current[0] <= previous[1]:
                    previous[1] = max(previous[1], current[1])
                else:
                    merged.append(current)
        return merged


class PrefLookTimestampParser(BaseParser):
    """
    a parser that can parse PrefLookTimestamp as described here:
    https://osf.io/3n97m/
    """
    def __init__(self, fps, labels_folder=None, ext=None, return_time_stamps=False):
        super().__init__()
        self.fps = fps
        self.return_time_stamps = return_time_stamps
        if ext:
            self.ext = ext
        if labels_folder:
            self.labels_folder = Path(labels_folder)

    def parse(self, file, file_is_fullpath=False):
        """
        Parses a label file from the lookit dataset, see base class for output format
        :param file: the file to parse
        :param file_is_fullpath: if true, the file represents a full path and extension,
         else uses the initial values provided.
        :return:
        """
        codingactive_counter = 0
        classes = {"away": 0, "left": 1, "right": 2}
        if file_is_fullpath:
            label_path = Path(file)
        else:
            label_path = Path(self.labels_folder, file + self.ext)
        labels = np.genfromtxt(open(label_path, "rb"), dtype='str', delimiter=",", skip_header=3)
        output = []
        start, end = 0, 0
        for entry in range(labels.shape[0]):
            if self.return_time_stamps:
                frame = int(labels[entry, 0])
                dur = int(labels[entry, 1])
            else:
                frame = int(int(labels[entry, 0]) * self.fps / 1000)
                dur = int(int(labels[entry, 1]) * self.fps / 1000)
            class_name = labels[entry, 2]
            valid_flag = 1 if class_name in classes else 0
            if class_name == "codingactive":  # indicates the period of video when coding was actually performed
                codingactive_counter += 1
                start, end = frame, dur
                frame = dur  # if codingactive: add another annotation signaling invalid frames from now on
            frame_label = [frame, valid_flag, class_name]
            output.append(frame_label)
        assert codingactive_counter <= 1  # current parser doesnt support multiple coding active periods
        output.sort(key=lambda x: x[0])
        if end == 0:
            end = int(output[-1][0])
        if len(output) > 0:
            return output, start, end
        else:
            return None


class PrincetonParser(BaseParser):
    """
    A parser that can parse vcx files that are used in princeton laboratories
    """
    def __init__(self, fps, ext=None, labels_folder=None, start_time_file=None):
        super().__init__()
        self.fps = fps
        if ext:
            self.ext = ext
        if labels_folder:
            self.labels_folder = Path(labels_folder)
        self.start_times = None
        if start_time_file:
            self.start_times = self.process_start_times(start_time_file)

    def process_start_times(self, start_time_file):
        start_times = {}
        with open(start_time_file, newline='') as csvfile:
            my_reader = csv.reader(csvfile, delimiter=',')
            next(my_reader, None)  # skip the headers
            for row in my_reader:
                numbers = [int(x) for x in row[1].split(":")]
                time_stamp = numbers[0]*60*60*self.fps + numbers[1]*60*self.fps + numbers[2]*self.fps + numbers[3]
                start_times[Path(row[0]).stem] = time_stamp
        return start_times

    def parse(self, file, file_is_fullpath=False):
        """
        parse a coding file, see base class for output format
        :param file: coding file to parse
        :param file_is_fullpath: if true, the file is a full path with extension, else uses values from initialization
        :return:
        """
        if file_is_fullpath:
            label_path = Path(file)
        else:
            label_path = Path(self.labels_folder, file + self.ext)
        if not label_path.is_file():
            logging.warning("For the file: " + str(file) + " no matching xml was found.")
            return None
        return self.xml_parse(label_path, True)

    def xml_parse(self, input_file, encode=False):
        tree = ET.parse(input_file)
        root = tree.getroot()
        counter = 0
        frames = {}
        current_frame = ""
        for child in root.iter('*'):
            if child.text is not None:
                if 'Response ' in child.text:
                    current_frame = child.text
                    frames[current_frame] = []
                    counter = 16
                else:
                    if counter > 0:
                        counter -= 1
                        frames[current_frame].append(child.text)
            else:
                if counter > 0:
                    if child.tag == 'true':
                        frames[current_frame].append(1)  # append 1 for true
                    else:
                        frames[current_frame].append(0)  # append 0 for false
        responses = []
        for key, val in frames.items():
            [num] = [int(s) for s in key.split() if s.isdigit()]
            responses.append([num, val])
        sorted_responses = sorted(responses)
        if encode:
            encoded_responses = []
            # response_hours = [int(x[1][6]) for x in sorted_responses]
            # if not response_hours.count(response_hours[0]) == len(response_hours):
            #     logging.warning("response")
            for response in sorted_responses:
                frame_number = int(response[1][4]) +\
                               int(response[1][10]) * self.fps +\
                               int(response[1][8]) * 60 * self.fps +\
                               int(response[1][6]) * 60 * 60 * self.fps
                if self.start_times:
                    start_time = self.start_times[input_file.stem]
                    frame_number -= start_time
                assert frame_number < 60 * 60 * self.fps
                encoded_responses.append([frame_number, response[1][14], response[1][16]])
            sorted_responses = encoded_responses
        # replace offs with aways, they are equivalent
        for i, item in enumerate(sorted_responses):
            if item[2] == 'off':
                item[2] = 'away'
                sorted_responses[i] = item
        start = sorted_responses[0][0]
        end = sorted_responses[-1][0]
        return sorted_responses, start, end
