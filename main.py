import os
import scipy.io
import pandas as pd
import numpy as np
import json
import cv2
from PIL import Image

def get_rows(img_names, labels):
    rows = []
    # enumerate(img_names) -> iterador con estructura (indice, array(['nombre_imagen.jpg'])) 
    for index, img_name in enumerate(img_names):
        for label in labels[index]:
            # print(index)
            # print(img_name.item())
            row = [img_name.item()]
            row.extend(label) # concatena las etiquetas
            rows.append(row)
    return rows

# Convierte el archivo LabelTrainAll.mat en un DataFrame de pandas
def make_train_data():
    '''
    readme-train.txt

    MAFA training set
    1) images folder puts the 25876 image files; 
    2) the label is stored in LabelTrainAll.mat,
    3) the format is stored in a 18d array (x,y,w,h, x1,y1,x2,y2, x3,y3,w3,h3, occ_type, occ_degree, gender, race, orientation, x4,y4,w4,h4),  where        
        (a) (x,y,w,h) is the bounding box of a face, 
        (b) (x1,y1,x2,y2) is the position of two eyes.
        (c) (x3,y3,w3,h3) is the bounding box of the occluder. Note that (x3,y3) is related to the face bounding box position (x,y)
        (d) occ_type stands for the occluder type and has: 1 for simple, 2 for complex and 3 for human body.
        (e) occ_degree stands for the number of occluded face parts
        (f) gender and race stand for the gender and race of one face
        (g) orientation stands for the face orientation/pose, and has: 1-left, 2-left frontal, 3-frontal, 4-right frontal, 5-right
        (h) (x4,y4,w4,h4) is the bounding box of the glasses and is set to (-1,-1,-1,-1) when no glasses.  Note that (x4,y4) is related to the face bounding box position (x,y)

    If any question, please contact me. (geshiming@iie.ac.cn)
    '''

    train = scipy.io.loadmat('LabelTrainAll.mat')
    # print(train.keys())
    train_labels = train['label_train']
    train_labels = train_labels[0]
    img_names = train_labels['imgName']
    labels = train_labels['label'] # 21 labels
    train_columns = [ 'image_name'
                ,'x'
                ,'y'
                ,'w'
                ,'h'
                ,'x1'
                ,'y1'
                ,'x2'
                ,'y2'
                ,'x3'
                ,'y3'
                ,'w3'
                ,'h3'
                ,'occ_type'
                ,'occ_degree'
                ,'gender'
                ,'race'
                ,'orientation'
                ,'x4'
                ,'y4'
                ,'w4'
                ,'h4']
    rows = get_rows(img_names, labels)
    return pd.DataFrame(data=rows, columns=train_columns)

# Convierte el archivo LabelTestAll.mat en un DataFrame de pandas
def make_test_data():
    '''
    MAFA-Label-Test/readme-test.txt

    MAFA testing set
    1) images folder puts the 4935 image files; 
    2) the label is stored in LabelTestAll.mat,
    3) the format is stored in a 18d array (x,y,w,h,face_type,x1,y1,w1,h1, occ_type, occ_degree, gender, race, orientation, x2,y2,w2,h2),  where              
        (a) (x,y,w,h) is the bounding box of a face, 
        (b) face_type stands for the face type and has: 1 for masked face, 2 for unmasked face and 3 for invalid face.
        (c) (x1,y1,w1,h1) is the bounding box of the occluder. Note that (x1,y1) is related to the face bounding box position (x,y)
        (d) occ_type stands for the occluder type and has: 1 for simple, 2 for complex and 3 for human body.
        (e) occ_degree stands for the number of occluded face parts
        (f) gender and race stand for the gender and race of one face
        (g) orientation stands for the face orientation/pose, and has: 1-left, 2-left frontal, 3-frontal, 4-right frontal, 5-right
        (h) (x2,y2,w2,h2) is the bounding box of the glasses and is set to (-1,-1,-1,-1) when no glasses.  Note that (x2,y2) is related to the face bounding box position (x,y)

    If any question, please contact me. (geshiming@iie.ac.cn)
    '''

    test = scipy.io.loadmat('LabelTestAll.mat')
    # print(test.keys())
    test_labels = test['LabelTest']
    test_labels = test_labels[0]
    img_names = test_labels['name']
    labels = test_labels['label']
    test_columns = [ 'image_name',
            'x',
            'y',
            'w',
            'h',
            'face_type',
            'x1',
            'y1',
            'w1',
            'h1', 
            'occ_type', 
            'occ_degree', 
            'gender', 
            'race', 
            'orientation', 
            'x2',
            'y2',
            'w2',
            'h2']
    rows = get_rows(img_names, labels)
    return pd.DataFrame(data=rows, columns=test_columns)

occluder_type = {
  1: "Simple",
  2: "Complex",
  3: "Human body",
  -1: "-1"
}
occluder_degree = {
    1: "Not mask",
    2: "Mouth",
    3: "Fully",
    -1:"-1" 
}


# <object-class> - integer number of object from 0 to (classes-1)
# <x> <y> <width> <height> - float values relative to width and height of image, it can be equal from (0.0 to 1.0]
# for example: <x> = <absolute_x> / <image_width> or <height> = <absolute_height> / <image_height>
# atention: <x> <y> - are center of rectangle (are not top-left corner)

def bb_to_yolo(img, x,y,w,h):
    # img_height, img_width = img.shape[:2] # (height, width) cv2 -> height, width PIL -> width, height xd
    img_width, img_height = img # width, height = im.size 

    yolo_x = (x+(w /2)) / img_width
    yolo_y = (y+(h /2)) / img_height

    yolo_w = w / img_width
    yolo_h = h / img_height

    if yolo_x > 1.0:
        w = img_width - x
        yolo_x = (x+(w /2)) / img_width
    if yolo_y > 1.0:
        h = img_height - y
        yolo_y = (y+(h /2)) / img_height

    return yolo_x, yolo_y, yolo_w, yolo_h


def yolo_to_bb(img, x,y,w,h):
    img_width, img_height = img # width, height = im.size 
    # img_height, img_width = img.shape[:2] # (height, width)
    x, y = (x - (w/2)) * img_width, (y - (h/2))*img_height
    w = w * img_width
    h = h * img_height
    return x, y, w, h


def draw_bounding_box(row, mode):
    img = cv2.imread(mode+'/images/'+row['image_name'])
    img_width, img_height = img.shape[:2]
    
    x = int(row[1])
    y = int(row[2])
    w = int(row[3])
    h = int(row[4])

    _, label_name = get_label(row)
    cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2) # x,y -> top-left, x+w, y+h -> botton-right 
    cv2.putText(img,'occluder_type: '+(row['occ_type']),(x,y+h+10),0,0.3,(0,255,0))
    cv2.putText(img,'occluder_degree: '+(row['occ_degree']),(x,y+h+20),0,0.3,(0,255,0))
    cv2.putText(img,'label: '+label_name, (x,y+h+30),0,0.3,(0,255,0))

    cv2.circle(img, (x+int(w/2), y+int(h/2)), radius=4, color=(0, 0, 255), thickness=-1)

    return img

def resize_and_padding(img, resize):
    # source: https://jdhao.github.io/2017/11/06/resize-image-to-square-with-padding/
    old_size = img.shape[:2] # old_size is in (height, width) format

    ratio = float(resize)/max(old_size)
    new_size = tuple([int(x*ratio) for x in old_size])

    # new_size should be in (width, height) format

    img = cv2.resize(img, (new_size[1], new_size[0]))

    delta_w = resize - new_size[1]
    delta_h = resize - new_size[0]
    top, bottom = delta_h//2, delta_h-(delta_h//2)
    left, right = delta_w//2, delta_w-(delta_w//2)

    color = [0, 0, 0]
    new_img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT,
        value=color)
    return new_img

# <object-class> - integer number of object from 0 to (classes-1)
# <x> <y> <width> <height> - float values relative to width and height of image, it can be equal from (0.0 to 1.0]
# for example: <x> = <absolute_x> / <image_width> or <height> = <absolute_height> / <image_height>
# atention: <x> <y> - are center of rectangle (are not top-left corner)

def get_label(row):
    if (row['occ_type']=="Simple" or row['occ_type']=="Complex") and row['occ_degree'] == "Fully":
        label, label_name = 1, 'Mask' # Mask
    elif row['occ_type']=="Simple" or row['occ_type'] =="Complex": # and (row['occ_degree']) != "Not mask":
        label, label_name = 2, 'Mask incorrect' # Mask incorrect
    else:
        label, label_name = 0, 'No mask' # No mask
    return label, label_name

def mafa_to_yolo_labels(df, mode):
    label_path = mode+'/labels/'
    image_path = mode+'/images/'
    os.mkdir(label_path)
    for index, row in df.iterrows():
        with open(label_path+row.image_name[:-4]+'.txt','a') as f:
            try:
                # img = cv2.imread(image_path+row.image_name) lento, solo necesito saber las dimensiones no cargar la imagen
                img = Image.open(image_path+row.image_name) 
                x, y, w, h = bb_to_yolo(img.size, row.x, row.y, row.w, row.h)
                img.close()

                label, _ = get_label(row)
                if 0.0 <= (x or y or w or h) < 1.0: # Muchas anotaciones de los test estan mal y las x, y son mayores que el tamanio de la imagen 
                    f.write("%i %f %f %f %f\n"%(label, x, y, w, h))
            except FileNotFoundError:
                print("Image "+ image_path+row.image_name + " doesn't exist.")


def visualize_dataset(df, mode):
    for index, row in df.iterrows():
        # print(row['image_name'])
        # img = cv2.imread('train/images/'+row['image_name'])
        str_type = row['occ_type']
        str_degree = row['occ_degree']
        img = draw_bounding_box(row, mode)
        _, label_name = get_label(row)
        # img = resize_and_padding(img, 416)
        cv2.imshow('type: '+str_type+' degree: '+str_degree+' label: '+label_name, img)
        print(row['image_name'])
        key = cv2.waitKey(0)
        if key == ord('a'):
            cv2.imshow('type: '+str_type+' degree:'+str_degree, prev) 
            print(row['image_name'])
            key = cv2.waitKey(0)
        elif key == 27: # escape
            break
        cv2.destroyAllWindows()
        prev = img

def visualize_img(row, mode):
    img = draw_bounding_box(row, mode)
    cv2.imshow('img', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()



def draw_yolo_bounding_box(row, mode):
    img = cv2.imread(mode+'/images/'+row['img_name'])
    print(mode+'/images/'+row['img_name'])
    img_width, img_height = img.shape[:2]
    
    x = float(row[1])
    y = float(row[2])
    w = float(row[3])
    h = float(row[4])

    # formato yolo a escala real
    # x = (row.x+(row.w /2)) / img_width
    # y = (row.y+(row.h /2)) / img_height
    # w = row.w / img_width
    # h = row.h / img_height
    
    x = int(x * img_width)
    y = int(y * img_height)
    w = int(w * img_width)
    h = int(h * img_height)

    cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
    cv2.circle(img, (x, y), radius=4, color=(0, 0, 255), thickness=-1)

    # cv2.putText(img,'occluder_type: '+(row['occ_type']),(x,y+h+10),0,0.3,(0,255,0))
    # cv2.putText(img,'occluder_degree: '+(row['occ_degree']),(x,y+h+20),0,0.3,(0,255,0))

    # cv2.circle(img, (x+int(w/2), y+int(h/2)), radius=4, color=(0, 0, 255), thickness=-1)
    cv2.imshow('img', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def get_yolo_labels(mode):
    directory = os.fsencode(mode+'images/labels')
    for file in os.listdir(directory):
        filename = os.fsdecode(file)
        if filename.endswith(".txt"): 
            path = os.path.join(directory, filename)
            with open(path, "r") as f:
                for line in f:
                    line = f.readline().split(' ')
                    img_name = filename[:-4]+'.jpg'
                    x = line[1]
                    y = line[2]
                    w = line[3]
                    h = line[4][:-1]
                    row = pd.Series(data = [img_name, x, y, w, h], index=['img_name', 'x', 'y', 'w', 'h'])

            continue
        else:
            continue

def add_label_column(df):
    label_list = []
    for _, row in df.iterrows():
        _, label_name = get_label(row)
        label_list.append(label_name)
    return label_list

def data_check(df):
    print('Dataset files: ', len(df))
    print('Number of Mask :', (df['label'] == 'Mask').sum())
    print('Number of Mask incorrect :', (df['label'] == 'Mask incorrect').sum())
    print('Number of No mask:', (df['label'] == 'No mask').sum())

def train_fix_label(df):
    df[df['occ_type']=='-1'].replace({'occ_type': 'Simple', 'occ_degree': 'Fully'})
    df[df['occ_degree']=='-1']
    df.loc[521,'occ_type'] = 'Simple'
    df.loc[521,'occ_degree'] = 'Fully'
    df.loc[1381,'occ_type'] = 'Simple'
    df.loc[1381,'occ_degree'] = 'Fully'
    return df

def test_fix_label(df):
    # row = test.loc[4448]
    # img = Image.open('test/images/'+row.image_name)
    # _, img_height = img.size # width, height = im.size 
    # df.loc[4448, 'h'] = img_height - row.y
    df = df.drop(index=[1627, 5851, 5852, 5853, 5854, 7202, 4898, 159])
    return df

train = make_train_data()
train = train.replace({'occ_type': occluder_type, 'occ_degree': occluder_degree})
train = train_fix_label(train)
test = make_test_data()
test = test.replace({'occ_type': occluder_type, 'occ_degree': occluder_degree})
# test = test_fix_label(test)
def make_labels():
    print('Making yolo labels for training data...')
    mafa_to_yolo_labels(train, 'train')
    print('Done')

    print('Making yolo labels for test data...')
    mafa_to_yolo_labels(test, 'test')
    print('Done')


def bbox(filename):
    if filename[:3] == 'tra':
        mode = 'train'
        df = train[train['image_name'] == filename]
        df
        visualize_dataset(df, mode)
    else:
        mode = 'test'
        df = test[test['image_name'] == filename]
        df
        visualize_dataset(df, mode)


def debug(filename):
    print(test[test['image_name'] == filename])
    img = Image.open('test/images/'+filename)
    print(img.size)
    bbox(filename)

make_labels()
# debug('test_00001626.jpg')
# img_name = 'test_00003494.jpg'
# img = Image.open('test/images/'+img_name)
# x, y, w, h = yolo_to_bb(img.size, x, y, w, h)
# row = pd.Series(data = [img_name, x, y, w, h], index=['image_name', 'x', 'y', 'w', 'h'])
# cv2.imshow('img', draw_bounding_box(row, 'test'))
# cv2.waitKey(0)
# cv2.destroyAllWindows()