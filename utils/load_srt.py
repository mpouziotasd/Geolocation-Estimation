import os
import re
from datetime import datetime

def get_srt_metadata(url):
    folder_path = os.path.dirname(url) + "/"
    file_name = url.split('/')[-1].split('.')[0]
    file_path = folder_path + file_name + ".SRT"
    captions = {}

    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()
        frame_cnt = None
        for line in lines:
            if line.startswith('<font size="28">FrameCnt:'):
                match = re.search(r'FrameCnt: (\d+)', line)
                if match:
                    frame_cnt = int(match.group(1)) - 1
                    captions[frame_cnt] = {}
            elif line.startswith('[iso:') and frame_cnt is not None:
                caption_values = re.findall(r'\[(.*?)\]', line)
                if caption_values:
                    for caption_value in caption_values:
                        parameters_values = caption_value.split(' ')
                        for i in range(0, len(parameters_values), 2):
                            if i + 1 < len(parameters_values):
                                key = parameters_values[i].strip(':')
                                value = parameters_values[i + 1]
                                captions[frame_cnt][key] = value
            try:
                timestamp = datetime.strptime(line.strip(), "%Y-%m-%d %H:%M:%S.%f").strftime("%H:%M:%S.%f")
                captions[frame_cnt]['timestamp'] = timestamp
            except ValueError:
                continue
        file.close()

    return captions