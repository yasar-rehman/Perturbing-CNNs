# Database Generation for Face-Anti Spoofing
# The following script will clip x sequential or random frames from videos and store that in a separate folders or
# database with corresponding labels
# we use h5py to make this dataset

# Database = REPLY_ATTACK
# Classes: Real, Wrapped Photo Attack, Cut Photo Attack, Video Attack
# We first take a frame, and then detect the face within the frame. After detection of face, we resize the face part to
# include some of the background area within the frame. After while, we either store that into a separate directory or
# in h5py file.


from imutils.face_utils import rect_to_bb, shape_to_np
import imutils
import numpy as np
import cv2
import os
import dlib
detector = dlib.get_frontal_face_detector()
face_cascade = cv2.CascadeClassifier('/opencv/opencv-3.3.0/data/haarcascades_cuda/haarcascade_frontalface_default.xml')


def video_data(filepath, frame_length):

    video_matrix = []  # define video matrix to store video frames
    database_matrix = []  # define database matrix to store videos along with frames
    cap = cv2.VideoCapture(filepath)
    video_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))-4
    video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    print('length of the video = %g ---- height x width = %d x %d --- fps =%g' % (
        video_length, video_height, video_width, video_fps))
    counter = 1
    starting_point = 0  # select the starting frame

    # read all frames of a video
    while cap.isOpened():

        ret, frame = cap.read()
        if counter >= starting_point:
            video_matrix.append(frame)

        if counter != video_length:

            counter += 1
        else:
            break
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    # print(np.asarray(video_matrix).shape)

    # select random frames from the fetched videos

    batch = 0
    while batch < frame_length:
        if video_length < frame_length:
            dummy = np.random.randint(video_length)
        else:
            dummy = np.random.randint(video_length)

        selected_frames = video_matrix[dummy]
        miniframe = selected_frames

        # miniframe = cv2.cvtColor(selected_frames, cv2.COLOR_BGR2GRAY) # not required in dlib based detector
        cv2.imshow('frame', miniframe)
        cv2.waitKey(1)
        faces = detector(miniframe)
        print(faces)
        if len(faces) == 0:
            print(len(faces))
            sub_face = cv2.resize(selected_frames, (400, 400), interpolation=cv2.INTER_AREA)
            print(batch)
        else:
            maxArea = 300
            for f in faces:
                (x, y, w, h) = np.abs(rect_to_bb(f))
                print(w*h)
                if w*h > maxArea:
                    im_size = np.int(np.sqrt(w * h))
                        #cv2.rectangle(frame, (x, y), (x + w+30, y + h+30), (255, 255, 255))
                    # Save just the rectangle faces in SubRecFaces
                    sub_face = cv2.resize(selected_frames[y:y + h, x:x + w], (im_size, im_size))
                    #FaceFileName = "/home/yaurehman2/Documents/Researchfirstyear/keras_research/face_" + str(batch) + ".jpg"
                    #cv2.imwrite(FaceFileName, sub_face)
                    # Display the image
                    cv2.imshow('Result', selected_frames)
                    cv2.imshow('frame', sub_face)
                    cv2.waitKey(1)
        batch += 1

        database_matrix.append(sub_face)
        print(np.asarray(database_matrix).shape)
    return np.asarray((database_matrix)), len(video_matrix)


#Database storage function

def store_db(path,data_matrix,labels):
    path=os.path.expanduser(path)
    if os.path.isdir(path): #check whether the file path exist or not
        raise ValueError("The path already exist")
    else:
        os.mkdir(path)
    for f in data_matrix.shape[0]:
        FaceFile_Name = path + ".jpg"
        #store the file in the requested path
        cv2.imwrite(FaceFile_Name, data_matrix[f,:])


# this function can make a script file containing paths and labels for database

def get_dataset_reply(paths, filename, protocol_file, parent_directory):  # this function get the dataset from the given path

    for protocol_file in protocol_file:
        full_path = os.path.join(paths, protocol_file)
        with open(filename, "a+") as file1:
            for path in full_path.split(':'):
                if path != '':
                    path_exp = os.path.expanduser(path)
                    #open the file name pointed by the path_exp
                    with open(path_exp, 'r') as f:
                        for line in f.readlines()[0:]:
                            # determine whether the input path points to real face video or fake one
                            path_type = line.strip().split('/')
                            if path_type[1] != 'real':
                                pair = line.strip().split('_')

                                if pair[1] in {'print','highdef'}\
                                        and pair[5] == 'photo' and pair[6][:-4] in {'adverse','controlled'}:
                                    d = os.path.expanduser(os.path.join(parent_directory, line))
                                    d = d[:-1]
                                    file1.write('%s %d \n' % (d, int(1)))

                                # elif pair[1] == ('print' or 'mobile')\
                                #         and pair[5] == 'photo' and pair[6][:-4] == 'controlled':
                                #     d = os.path.expanduser(os.path.join(parent_directory, line))
                                #     d = d[:-1]
                                #     file1.write('%s %d \n' % (d, int(4)))

                                # elif pair[1] == ('highdef')\
                                #         and pair[5] == 'photo' and pair[6][:-4] == ('controlled' or 'adverse'):
                                #     d = os.path.expanduser(os.path.join(parent_directory, line))
                                #     d = d[:-1]
                                #     file1.write('%s %d \n' % (d, int(5)))

                                elif pair[1] in {'mobile','highdef'} and pair[5] in {'video','photo'}\
                                        and pair[6][:-4] in {'adverse','controlled'}:
                                    d = os.path.expanduser(os.path.join(parent_directory, line))
                                    d = d[:-1]
                                    file1.write('%s %d \n' % (d, int(3)))

                                # elif pair[1] == 'mobile'and pair[5] == 'video' and pair[6][:-4] == 'controlled':
                                #     d = os.path.expanduser(os.path.join(parent_directory, line))
                                #     d = d[:-1]
                                #     file1.write('%s %d \n' % (d, int(10)))
                                #
                                # elif pair[1] == 'highdef'and pair[5] == 'video'\
                                #         and pair[6][:-4] == ('adverse' or 'controlled'):
                                #     d = os.path.expanduser(os.path.join(parent_directory, line))
                                #     d = d[:-1]
                                #     file1.write('%s %d \n' % (d, int(11)))
                            else:
                                d = os.path.expanduser(os.path.join(parent_directory, line))
                                d = d[:-1]
                                file1.write('%s %d \n' % (d, int(0)))


def database_reply(pair_filename, parent_path, child_path, textfile1):
    counter = 0  # counter for counting the number of images
    with open(pair_filename, 'r') as f:
        for line in f.readlines()[0:]:
            pair = line.strip().split()  # split the pairs when there is a space

            # get the data using the path and extract x frames
            vid, _ = video_data(pair[0], 100)
            shape_index = np.asarray(vid).shape  # get the shape of the videos
            labels_pairs = np.zeros(
                (shape_index[0]))  # generate labels equal to the data length
            # create labels
            if pair[1] == '0':
                labels_pairs[0:] = 0
            elif pair[1] == '1':
                labels_pairs[0:] = 1
            elif pair[1] == '3':
                labels_pairs[0:] = 3
            # elif pair[1] == '5':
            #     labels_pairs[0:] = 5
            # elif pair[1] == '9':
            #     labels_pairs[0:] = 9
            # elif pair[1] == '10':
            #     labels_pairs[0:] = 10
            # elif pair[1] == '11':
            #     labels_pairs[0:] = 11
            # print(shape_index[0])

            assert shape_index[0] == labels_pairs.shape[0]

            # After creating labels and data matrix it's time to store them in a designated path
            # Every folder we create should contain a frames corresponding to fake and real images of a single subject
            # And the ground truth will be stored in a separate txt file.

            dir = os.path.expanduser(parent_path)
            if os.path.exists(dir):
                print("Path already exist")
            else:
                # make a path to store the data
                os.mkdir(dir)
            path1 = os.path.join(dir, child_path)
            if os.path.exists(path1):
                print("path already exist")
            else:
                os.mkdir(path1)

            textfilepath = os.path.join(os.path.expanduser(textfile1), child_path) + '.txt'

            tmp_path = os.path.expanduser(path1) + '/'
            print(tmp_path)
            for j in range(shape_index[0]):
                storefile = tmp_path + str(counter) + '.jpg'
                cv2.imwrite(storefile, vid[j])
                with open(textfilepath, 'a+') as d:
                    d.write('%s \t %d \n' % (storefile, labels_pairs[j]))
                counter += 1


# first make a list of train and test text files to store the corresponding paths and labels

train_filepath = os.path.expanduser('~/Documents/Newwork/REPLY_Train_Mod_corr')    # file path for training
test_filepath = os.path.expanduser('~/Documents/Newwork/REPLY_Test_Mod_corr')
dev_filepath = os.path.expanduser('~/Documents/Newwork/REPLY_Dev_Mod_corr')

# Directory containing training and testing data
parent_directory = '~/Documents/Newwork/Reply_Attack'

path1 = '~/Documents/Newwork/protocols/'
protocol_file1 =['real-train.txt','attack-grandtest-allsupports-train.txt']

path2 = '~/Documents/Newwork/protocols/'
protocol_file2 =['real-test.txt','attack-grandtest-allsupports-test.txt']


path3 = '~/Documents/Newwork/protocols/'
protocol_file3 =['real-devel.txt','attack-grandtest-allsupports-devel.txt']

path_list = [path1, path2, path3]
file_path = [train_filepath, test_filepath, dev_filepath]
protocol_list = [protocol_file1, protocol_file2, protocol_file3]

for i in range(3):
    get_dataset_reply(path_list[i], file_path[i], protocol_list[i], parent_directory)


parent_path = '~/Documents/Newwork/REPLY_ATTACK_FACE_Mod_corr' # main directory where to store images
child_path1 = 'train'
child_path2 = 'test'
child_path3 = 'devel'


textfile1_path = '~/Documents/Newwork/REPLY_ATTACK_FACE_Mod_corr' # main path where to store the extracted images paths

# pair_filename# refer to paths from where to take the raw videos for frames extractions
pair_filename1 = train_filepath

pair_filename2 = test_filepath

pair_filename3 = dev_filepath

pair_list = [pair_filename1, pair_filename2, pair_filename3]
child_paths = [child_path1, child_path2, child_path3]

for i in range(3):
    database_reply(pair_list[i], parent_path, child_paths[i], textfile1_path)
#get the file name
