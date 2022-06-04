# Database Generation for Face-Anti Spoofing
# The following script will clip x sequential or random frames from videos and store that in a separate folders or
# database with corresponding labels
# we use h5py to make this dataset

# Database = OULU
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
from imutils.face_utils import rect_to_bb, shape_to_np, FaceAligner
from imutils.face_utils.helpers import FACIAL_LANDMARKS_IDXS

detector = dlib.get_frontal_face_detector()
face_cascade = cv2.CascadeClassifier('/home/yaurehman2/virtualenv-py2/opencv/opencv-3.3.0/data/haarcascades_cuda/haarcascade_frontalface_default.xml')

path = '/home/yaurehman2/Documents/stereo_face_liveness/shape_predictor_68_face_landmarks.dat'

predictor = dlib.shape_predictor(path)


def video_data(filepath, frame_length):
    video_matrix = [] # define video matrix to store video frames
    database_matrix=[] # define database matrix to store videos along with frames
    cap = cv2.VideoCapture(filepath)
    video_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))-4
    video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    print('length of the video = %g ---- height x width = %d x %d --- fps =%g' % (
        video_length, video_height, video_width, video_fps))
    counter = 1
    starting_point=0 #select the starting frame
    #read all frames of a video
    while (cap.isOpened()):

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

    #select random frames from the fetched videos

    batch = 0
    while batch < frame_length:
        if video_length < frame_length:
            dummy = np.random.randint(video_length)
        else:
            dummy=np.random.randint(video_length)

        selected_frames = video_matrix[dummy]



        # miniframe = cv2.cvtColor(selected_frames, cv2.COLOR_BGR2GRAY)
        miniframe = selected_frames
        faces = detector(miniframe)





        if len(faces) !=0:
            # print(len(faces))
            # sub_face = cv2.resize(selected_frames, (400, 400), interpolation=cv2.INTER_AREA)
            # print(batch)
            f = faces[0]
            (x, y, w, h) = np.abs(rect_to_bb(f))

            land_marks = predictor(miniframe, f)  # detect the landmarks in the face
            land_marks = shape_to_np(land_marks)  # convert  the landmarks in tuples of x and y
            # for (j,z) in land_marks:
            #     cv2.circle(miniframe, (j,z), 2, (0,255,0), -1)
            #
            # cv2.imshow('win', miniframe)
            # cv2.waitKey()




            im_size = np.int(np.sqrt(w * h))
            #cv2.rectangle(frame, (x, y), (x + w+30, y + h+30), (255, 255, 255))
            # Save just the rectangle faces in SubRecFaces
            sub_face = cv2.resize(selected_frames[y:y + h, x:x + w], (im_size, im_size))
            database_matrix.append(sub_face)
            batch += 1
        else:

            sub_face = cv2.resize(selected_frames, (400, 400), interpolation=cv2.INTER_AREA)
            database_matrix.append(sub_face)
            batch += 1





        # print(np.asarray(database_matrix).shape)
    return np.asarray((database_matrix)), len(video_matrix)




#
#Database storage function

def store_db(path, data_matrix, labels):
    path=os.path.expanduser(path)
    if os.path.isdir(path): #check whether the file path exist or not
        raise ValueError("The path already exist")
    else:
        os.mkdir(path)
    for f in data_matrix.shape[0]:
        FaceFile_Name = path + ".jpg"
        #store the file in the requested path
        cv2.imwrite(FaceFile_Name,data_matrix[f,:])


#this function can make a script file containing paths and labels for CASIA-FASD database

def get_dataset(paths,filename): #this function get the dataset from the given path


    file1 = open(paths, "r")
    file2 = open(filename, "w")

    database = paths.strip().split('/')
    st_file_path = filename.strip().split('/')


    if database[-1] in {'Train_1.txt'} and st_file_path[-1] in {'train_1'} :
        database_path = os.path.expanduser('~/Documents/Newwork/OULU_NPU/Train_files/')

    if database[-1] in {'Dev_1.txt'} and st_file_path[-1] in {'dev_1'}:
        database_path = os.path.expanduser('~/Documents/Newwork/OULU_NPU/Dev_files/')

    if database[-1] in {'Test_1.txt'} and st_file_path[-1] in {'test_1'}:
        database_path = os.path.expanduser('~/Documents/Newwork/OULU_NPU/Test_files/')


    for f in file1:
        x = f.strip().split(',')

        file_path = os.path.join(database_path + x[1] + '.avi')


        if f[-2] in {'1'}:
            file2.write('%s  %d \n' %(file_path, int(0)))

        elif f[-2] in {'2'}:
            file2.write('%s  %d \n' % (file_path, int(1)))

        elif f[-2] in {'3'}:
            file2.write('%s  %d \n' % (file_path, int(2)))

        elif f[-2] in {'4'}:
            file2.write('%s  %d \n' % (file_path, int(3)))

        elif f[-2] in {'5'}:
            file2.write('%s  %d \n' % (file_path, int(4)))

    file1.close()
    file2.close()





# Directory containing training, development and testing data

path1 = os.path.expanduser('~/Documents/Newwork/OULU_NPU/Protocols/Protocol_4/Train_1.txt')
path2 = os.path.expanduser('~/Documents/Newwork/OULU_NPU/Protocols/Protocol_4/Dev_1.txt')
path3 = os.path.expanduser('~/Documents/Newwork/OULU_NPU/Protocols/Protocol_4/Test_1.txt')



# first make a list of train and test text files to store the corresponding paths and labels

train_filepath = os.path.expanduser('~/Documents/Newwork/OULU_protocol4/train_1') # file path for training
dev_filepath = os.path.expanduser('~/Documents/Newwork/OULU_protocol4/dev_1')
test_filepath = os.path.expanduser('~/Documents/Newwork/OULU_protocol4/test_1')

path_file = [path1, path2, path3]
tr_ts_f_p = [train_filepath, dev_filepath, test_filepath]

for i in range(len(path_file)):
    get_dataset(path_file[i],tr_ts_f_p[i])
#
#
parent_path = '~/Documents/Newwork/OULU_FACE_Protocol4'
# #
child_path1 = 'train_1'
child_path2 = 'dev_1'
child_path3 = 'test_1'
# #
textfile1 = '~/Documents/Newwork/OULU_FACE_Protocol4'
# #
ch_paths = [child_path1, child_path2, child_path3]
# #
pair_filename = [train_filepath, dev_filepath, test_filepath]
# #
counter = 0     # counter for counting the number of images
# get the file name

for i in range(len(pair_filename)):
    with open(pair_filename[i],'r') as f:
        for line in f.readlines()[0:]:
            pair = line.strip().split()     # split the pairs when there is a space

            # get the data using the path and extract x frames
            vid, _ = video_data(pair[0], 20)
            shape_index = np.asarray(vid).shape         # get the shape of the videos
            labels_pairs = np.zeros((shape_index[0]))     # generate labels equal to the data length
            #create labels
            if pair[1] == '0':
                labels_pairs[0:] = 0

            elif pair[1] == '1':
                labels_pairs[0:] = 1

            elif pair[1] == '2':
                labels_pairs[0:] = 2

            elif pair[1] == '3':
                labels_pairs[0:] = 3

            elif pair[1] == '4':
                labels_pairs[0:] = 4



            # print(shape_index[0])

            assert shape_index[0] == labels_pairs.shape[0]

            # After creating labels and data matrix it's time to store them in a designated path
            # Every folder we create should contain a frames corresponding to fake and real images of a each subject
            # And the ground truth will be stored in a separate txt file.

            dir = os.path.expanduser(parent_path)
            if os.path.exists(dir):
                print ("Path already exist: %s:" %dir)
            else:
                # make a path to store the data
                os.mkdir(dir)

            path1_1 = os.path.join(dir, ch_paths[i])
            print(path1_1)
            if os.path.exists(path1_1):
                print("path already exist: %s", path1_1)
            else:
                os.mkdir(path1_1)

            textfilepath = os.path.join(os.path.expanduser(textfile1), ch_paths[i]) + '.txt'

            tmp_path = os.path.expanduser(path1_1) + '/'

            print(tmp_path)
            for j in range(shape_index[0]):
                storefile = tmp_path + str(counter) + '.jpg'
                # print(tmp_path + str(counter) + '.jpg')
                cv2.imwrite(storefile, vid[j])
                with open(textfilepath,'a+') as d:
                    d.write('%s \t %d \n' %(storefile, labels_pairs[j]))
                counter += 1



