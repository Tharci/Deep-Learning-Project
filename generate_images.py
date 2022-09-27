import cv2
import time
import os

# How to use:
# 1. Define FRAMES_DISTANCE
# 2. Put videos in the 'videos' directory next to the script
# 3. Specify time ranges for each video (optional)
#    - the file has to have the same name as the video and .txt extension
#    - add line for every selected time range in format: "HH:MM:SS - HH:MM:SS"
#    - example: "00:00:00 - 00:11:33"
# 4. Run the script
# 5. Images will be generated to the 'frames' directory, next to the script
#
# Warning:
# - mp4 is the only supported extension at the moment

FRAMES_DISTANCE = 10000
VIDEOS_PATH = "videos"
FRAMES_OUTPUT = "frames"


def prepare_videos():
    try:
        os.mkdir(FRAMES_OUTPUT)
    except:
        pass

    videos = os.listdir(VIDEOS_PATH)
    videos = filter(lambda name: name.endswith("mp4"), videos)
    videos = list(map(lambda name: "%s/%s" % (VIDEOS_PATH, name), videos))

    print("Found videos %s" % videos)
    print("Preparing images...\n")

    for video_file in videos:
        time_ranges = load_video_times(video_file)
        generate_images(video_file, time_ranges)


def load_video_times(file):
    times_name = "%s.txt" % os.path.splitext(file)[0]
    try:
        times_file = open(times_name, "r")
    except:
        print("Could not load video time ranges. Using the whole video...")
        return None

    time_ranges = []
    for line in times_file:
        line = line.replace(" ", "").replace("\n", "")
        start, end = line.split("-")
        print("%s - %s" % (start, end))
        start = time_str_to_timestamp(start)
        end = time_str_to_timestamp(end)
        time_ranges.append((start, end))
    return time_ranges


def time_str_to_timestamp(time_str):
    split = time_str.split(":")
    hours = int(split[0])
    minutes = int(split[1])
    seconds = int(split[2])
    return ((hours * 60 + minutes) * 60 + seconds) * 1000


def generate_images(file, time_ranges):
    print("Converting '%s'" % file)
    start_time = time.time()
    video = cv2.VideoCapture(file)
    video_name = os.path.basename(file)
    last_timestamp = 0
    while True:
        success = video.grab()
        if not success:
            break
        timestamp = video.get(cv2.CAP_PROP_POS_MSEC)
        if timestamp > last_timestamp + FRAMES_DISTANCE and is_timestamp_in_range(timestamp, time_ranges):
            success, image = video.retrieve()
            cv2.imwrite("%s/%s_%d.jpg" % (FRAMES_OUTPUT, video_name, timestamp), image)
            h, w, _ = image.shape
            # imageLeft = image[:, :h]
            # imageRight = image[:, w-h:]
            # cv2.imwrite("%s/%s_%d_l.jpg" % (FRAMES_OUTPUT, video_name, timestamp), imageLeft)
            # cv2.imwrite("%s/%s_%d_r.jpg" % (FRAMES_OUTPUT, video_name, timestamp), imageRight)
            last_timestamp = timestamp
    print("Execution time: %s seconds\n" % (time.time() - start_time))


def is_timestamp_in_range(timestamp, time_ranges):
    if not time_ranges:
        return True

    for times in time_ranges:
        if times[0] < timestamp < times[1]:
            return True
    return False


prepare_videos()
