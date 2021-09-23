import glob, os, cv2
import numpy as np



def progmeter(curInd=None, totalInd=None, title=None):
    if title is not None:
            process = "\r[%s] %.2f%%:[%d/%d]: |%-50s|" % (title, curInd/totalInd*100, curInd, totalInd, '#' * int(curInd /totalInd * 50))
    else:
            process = "\r%.2f%%:[%d/%d]: |%-50s|" % (curInd/totalInd*100, curInd, totalInd, '#' * int(curInd /totalInd * 50))
    process += "\n" if curInd == totalInd else ""
    print(process, end='', flush=True)

def unrar_file(path):
    from unrar import rarfile
    l_file = glob.glob(os.path.join(path, '*.rar'))
    for file in l_file:
        rar_file = rarfile.RarFile(file)
        rar_file.extractall(path)

def read_videoRecog(path):
    """
    int      car_label;
    int      car_xmin;
    int      car_xmax;
    int      car_ymin;
    int      car_ymax;
    float    car_prob;
    int      car_id;
    float    speed;
    int     plate_color;
    int     plate_structure;
    int     plate_xmin;
    int     plate_xmax;
    int     plate_ymin;
    int     plate_ymax;
    float   plate_prob;
    char    license[MAX_LICENSENUM];
	float   license_prob;
    """
    from struct import unpack
    with open(path, 'rb') as fp:
        data = fp.read()
        pos = 0
        info = []
        while pos < len(data):
            frame_info = {}
            index = unpack('<i', data[pos:pos+4])[0]
            frame_info['index'] = index
            pos += 4
            num = unpack("<i", data[pos:pos + 4])[0]
            frame_info['obj_num'] = num
            pos += 4
            frame_info['obj_info'] = []
            print('index:[%d] obj_num:[%d]' % (index, num))
            for i in range(num):
                obj_info = unpack("<5ifif6if16cf", data[pos: pos + 80])
                frame_info['obj_info'].append(obj_info)
                pos += 80
                # print(obj_info)
            info.append(frame_info)
    return info

def read_videoRecog_2(path):
    """
    int      car_label;
    int      car_xmin;
    int      car_xmax;
    int      car_ymin;
    int      car_ymax;
    float    car_prob;
    int      car_id;
    float    speed;
    int     plate_color;
    int     plate_structure;
    int     plate_xmin;
    int     plate_xmax;
    int     plate_ymin;
    int     plate_ymax;
    float   plate_prob;
    char    license[MAX_LICENSENUM];
	float   license_prob;
	float score
    """
    from struct import unpack
    with open(path, 'rb') as fp:
        data = fp.read()
        pos = 0
        info = []
        while pos < len(data):
            frame_info = {}
            index = unpack('<i', data[pos:pos+4])[0]
            frame_info['index'] = index
            pos += 4
            num = unpack("<i", data[pos:pos + 4])[0]
            frame_info['obj_num'] = num
            pos += 4
            frame_info['obj_info'] = []
            # print('index:[%d] obj_num:[%d]' % (index, num))
            for i in range(num):
                obj_info = unpack("<5ifif6if16cff", data[pos: pos + 84])
                frame_info['obj_info'].append(obj_info)
                pos += 84
                print('\t', obj_info)
            info.append(frame_info)
    return info

def read_para(path):
    import json
    with open(path, 'r') as fp:
        data = json.load(fp)
    return data['dataPara']['videoStart']

def cal_dist(x1, y1, x2, y2):
    return np.sqrt(pow(x1-x2, 2) + pow(y1-y2, 2))

def show_image(wnd_name, img, delay):
    cv2.namedWindow(wnd_name, cv2.WINDOW_NORMAL)
    cv2.imshow(wnd_name, img)
    cv2.waitKey(delay)

def show_image_record(path, id_=None):
    path_video = os.path.join(path, 'ch00.h264')
    path_record = os.path.join(path, 'Data', 'ch00_videoRecog.txt')
    path_para = os.path.join(path, 'Data', 'dataPara.json')

    info = read_videoRecog(path_record)
    start_ind = read_para(path_para)[0]

    cap = cv2.VideoCapture(path_video)
    if not cap.isOpened():
        exit()
    while True:
        frame_no = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        ret, frame = cap.read()
        if not ret:
            break
        if frame_no < start_ind:
            continue
        obj_num = info[frame_no - start_ind]['obj_num']
        obj_info = info[frame_no - start_ind]['obj_info']
        print("[%d] : %d" % (frame_no, obj_num))
        for item in obj_info:
            obj_id = item[6]
            if id_ is not None and obj_id != id_:
                continue
            cv2.rectangle(frame, (item[1], item[3]), (item[2], item[4]), (0,0,255), 3)
        cv2.putText(frame, str(frame_no), (100, 100), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=3, color=(0,0,255), thickness=3)
        show_image("frame", frame, 0)

def show_image_record_2(path, id_=None):
    path_video = os.path.join(path, 'ch00.h264')
    path_record = os.path.join(path, 'ch00_videoRecog-res.txt')
    path_para = os.path.join(path, 'Data', 'dataPara.json')

    info = read_videoRecog_2(path_record)
    start_ind = read_para(path_para)[0]

    cap = cv2.VideoCapture(path_video)
    if not cap.isOpened():
        exit()
    while True:
        frame_no = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        ret, frame = cap.read()
        if not ret:
            break
        if frame_no < start_ind:
            continue
        obj_num = info[frame_no - start_ind]['obj_num']
        obj_info = info[frame_no - start_ind]['obj_info']
        print("[%d] : %d" % (frame_no, obj_num))
        for item in obj_info:
            obj_id = item[6]
            speed = item[7]
            score = item[-1]
            if id_ is not None and obj_id != id_:
                continue
            cv2.rectangle(frame, (item[1], item[3]), (item[2], item[4]), (0,0,255), 3)
            cv2.putText(frame, "speed = %.2f(km/h) score = %.2f" % (speed, score), (item[1], item[3]), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=3,
                        color=(0, 0, 255), thickness=3)

        cv2.putText(frame, str(frame_no), (100, 100), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=3, color=(0,0,255), thickness=3)
        show_image("frame", frame, 0)

def save_bbox_info(path, path_out, id=None):
    def get_id_info(info_id, id):
        for i, item in enumerate(info_id):
            if id == item['id']:
                return i
        return -1

    path_record = os.path.join(path, 'Data', 'ch00_videoRecog.txt')
    path_para = os.path.join(path, 'Data', 'dataPara.json')

    info = read_videoRecog(path_record)
    offset = read_para(path_para)[0]
    info_id = []
    for item in info:
        index = item['index']
        obj_num = item['obj_num']
        for obj in item['obj_info']:
            id = obj[6]
            bbox = obj[1:5] # l,r,t,b
            ind = get_id_info(info_id, id)
            if id != -1:
                if ind != -1:
                    info_id[ind]['frame'].append(offset + index)
                    info_id[ind]['bbox'].append(bbox)
                else:
                    tmp = {'id':id, 'frame':[offset + index], 'bbox':[bbox]}
                    info_id.append(tmp)

    for item in info_id:
        id = item['id']
        p_out = os.path.join(path_out, path[-23:])
        if not os.path.exists(p_out):
            os.makedirs(p_out)
        with open(os.path.join(p_out, str(id) + '.txt'), 'w') as fp:
            for frame, bbox in zip(item['frame'], item['bbox']):
                str_info = "%d,%d,%d,%d,%d\n" % (frame, bbox[0],  bbox[2], bbox[1],bbox[3]) # l,t,r,b
                fp.write(str_info)

def read_path_id(path):
    with open(path, 'r') as fp:
        data = fp.readlines()
        info = []
        for line in data:
            tmp = [int(float(x)) for x in line.split('\n')[0].split(',')]
            info.append({'frame':int(tmp[0]), 'bbox':tmp[1:]})
    return info

def show_id_bbox(path_video, path_id):
    info = read_path_id(path_id)

    cap = cv2.VideoCapture(path_video)
    if not cap.isOpened():
        exit()
    while True:
        frame_no = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        ret, frame = cap.read()
        if not ret:
            break
        if frame_no < info[0]['frame'] or (frame_no - info[0]['frame']) >= len(info):
            continue
        item = info[frame_no - info[0]['frame']]
        cv2.rectangle(frame, (item['bbox'][0], item['bbox'][2]), (item['bbox'][1], item['bbox'][3]), (0,0,255), 3)
        show_image('frame', frame, 10)

def show_bbox_by_path(path_id, shape = (2160, 4096)):
    info = read_path_id(path_id)
    for item in info:
        frame = item['frame']
        bbox = item['bbox']
        l,t,r,b = [int(float(x)) for x in bbox]
        print(frame, ' : ', bbox)
        img = np.zeros(shape, dtype='uint8')
        cv2.rectangle(img,  (l,t), (r,b), 255, 3)
        show_image('frame', img, 0)

def show_bbox_by_info(info, shape = (1440, 2560)):
    for item in info:
        frame = item['frame']
        bbox = item['bbox']
        img = np.zeros(shape, dtype='uint8')
        cv2.rectangle(img,  (item['bbox'][0], item['bbox'][1]), (item['bbox'][2], item['bbox'][3]), 255, 3)
        show_image('frame', img, 0)

def show_merge_info(info, info_, shape=(2160, 4096, 3)):
    assert len(info) == len(info_)
    frame_num = len(info)
    for ind in range(frame_num):
        frame, bbox = info[ind]['frame'], info[ind]['bbox']
        frame_, bbox_ = info_[ind]['frame'], info_[ind]['bbox']
        img = np.zeros(shape, dtype='uint8')

        cv2.rectangle(img,  (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 255, 255), 3)
        cv2.rectangle(img,  (bbox_[0], bbox_[1]), (bbox_[2], bbox_[3]), (0, 0, 255), 3)
        show_image('frame', img, 0)

def show_merge_point(info, info_):
    import matplotlib.pyplot as plt
    figure = plt.figure()
    frame_num = len(info)
    for i in range(frame_num):
        plt.scatter(info[i]['frame'], info[i]['bbox'][0], color = 'r')
        plt.scatter(info_[i]['frame'], info_[i]['bbox'][0], color = 'b')
    plt.show()

def read_path_mingfu(path, path_out):
    def get_ind(id, l_id):
        for i, item in enumerate(l_id):
            if id == item:
                return i
        return -1
    import struct
    # info = [ {'id':-1, 'obj':[{'frame':-1, 'bbox':[-1,-1,-1,-1]}]} ]
    l_id = []
    info = []
    with open(path, 'rb') as fp:
        data = fp.read()
        pos = 0
        while pos < len(data):
            frame_num, obj_num = struct.unpack('<2i', data[pos:pos+8])
            pos += 8
            # print("[%d] : %d" % (frame_num, obj_num))
            for i in range(obj_num):
                id, l, t, w, h = struct.unpack('<i4f', data[pos:pos + 20])
                # print("\t[%d] [%.2f, %.2f, %.2f, %.2f]" % (id, l, t, w, h))
                pos += 20
                ind = get_ind(id, l_id)
                if ind != -1:
                    info[ind]['frame'].append(frame_num)
                    info[ind]['bbox'].append([l, t, l+w, t+h])
                else:
                    l_id.append(id)
                    id_info = {'id':id, 'frame':[frame_num], 'bbox':[[l,t,l+w, t+h]]}
                    info.append(id_info)

        for item in info:
            id = item['id']
            p_out = os.path.join(path_out, str(id) + ".txt")
            if os.path.exists(p_out):
                mode = 'a+'
            else:
                mode = 'w'
            with open(p_out, mode) as fp:
                for frame, bbox in zip(item['frame'], item['bbox']):
                    l,t,r,b = bbox
                    str_info = "%d,%f,%f,%f,%f\n" % (frame, l,t,r,b)  # l,t,r,b
                    fp.write(str_info)


def read_multi_mingfu(path, path_out):
    l_file = os.listdir(path)
    for i, file in enumerate(l_file):
        progmeter(i+1, len(l_file), "[ID]")
        p_file = os.path.join(path, file)
        try:
            read_path_mingfu(p_file, path_out)
        except:
            continue



if __name__ == "__main__":
    # path = r'D:\Data\Data-Speed\tri_20210821_023114_952\Data\ch00_videoRecog.txt'
    # info = read_videoRecog(path)

    # path = r'D:\Data\Data-Speed\tri_20210823_034532_427'
    # path_out = './data'
    # save_bbox_info(path, path_out)

    # path_id = './data/tri_20210823_001031_381/0.txt'
    # info = read_path_id(path_id)
    # print(info)

    # path = r'D:\Projects\卡尔曼滤波\data\info'
    # path_out = r'D:\Projects\卡尔曼滤波\data\id_info'
    # read_path_mingfu(path, path_out)
    # read_multi_mingfu(path, path_out)

    # path = r'./data/id_info/4.txt'
    # show_bbox_by_path(path)
    pass
