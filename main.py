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

def get_occluder_names(df):
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
    df = df.replace({'occ_type': occluder_type, 'occ_degree': occluder_degree})
    return df


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
    # img_width, img_height = img # width, height = im.size # PIL
    img_height, img_width = img # (height, width) # CV2 # height, width, channels = img.shape
    if x > 0:
        x = (x - (w/2)) * img_width
    if y > 0:
        y = (y - (h/2)) * img_height
    w = w * img_width
    h = h * img_height
    x = int(x)
    y = int(y)
    w = int(w)
    h = int(h)
    return x, y, w, h

def draw_bounding_box(row, mode):
    img = cv2.imread(mode+'/images/'+row['image_name'])
    
    x = int(row[1])
    y = int(row[2])
    w = int(row[3])
    h = int(row[4])

    _, label_name = get_label(row)
    cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2) # x,y -> top-left, x+w, y+h -> botton-right 
    cv2.putText(img,'occluder_type: '+(row['occ_type']),(x,y+h+10),0,0.3,(0,255,0))
    cv2.putText(img,'occluder_degree: '+(row['occ_degree']),(x,y+h+20),0,0.3,(0,255,0))
    cv2.putText(img,'label: '+label_name, (x,y+h+30),0,0.3,(0,255,0))

    # cv2.circle(img, (x+int(w/2), y+int(h/2)), radius=4, color=(0, 0, 255), thickness=-1) 

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
    try:
        os.mkdir(label_path)
    except:
        for f in os.listdir(label_path):
            os.remove(os.path.join(label_path, f))

    for index, row in df.iterrows():
        with open(label_path+row.image_name[:-4]+'.txt','a') as f:
            try:
                # img = cv2.imread(image_path+row.image_name) lento, solo necesito saber las dimensiones no cargar la imagen
                img = Image.open(image_path+row.image_name)
                x, y, w, h = bb_to_yolo(img.size, row.x, row.y, row.w, row.h)
                img.close()

                label, _ = get_label(row)
                for i in x, y, w, h:
                    write = True
                    if not(0.0 <= i < 1.0):
                        write = False
                        break # Muchas anotaciones de los test estan mal y las x, y son mayores que el tamanio de la imagen 
                if (write):
                    f.write("%i %f %f %f %f\n"%(label, x, y, w, h))
            except FileNotFoundError:
                print("Image "+ image_path+row.image_name + " doesn't exist.")
                print(index)

    with open(mode+'/images.txt', 'w') as f:
        for item in df.image_name.unique():
            img_path = '../MAFAtoYOLO/'+mode+'/images/'+item
            # img_path = item
            f.write("%s\n" % img_path)

def visualize_dataset(df, mode):
    for index, row in df.iterrows():
        str_type = row['occ_type']
        str_degree = row['occ_degree']
        img = draw_bounding_box(row, mode)
        _, label_name = get_label(row)
        # img = resize_and_padding(img, 640)
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

def draw_yolo_bounding_box(img, row):
    # img = cv2.imread(mode+'/images/'+row['img_name'])
    img_size = img_width, img_height = img.shape[:2]
    print(img_size)
    
    label = row[0] 

    x = float(row[1])
    y = float(row[2])
    w = float(row[3])
    h = float(row[4])

    print(x, y, w, h)
    x, y, w, h = yolo_to_bb(img_size, x, y, w, h)
    print(x, y, w, h)

    img = cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)
    img = cv2.circle(img, (x, y), radius=4, color=(0, 0, 255), thickness=-1)
    img = cv2.putText(img, 'label: '+ label, (x,y+h+10), 0,0.3, (0,255,0))


    return img

def get_yolo_labels(label_path):
    rows = []
    with open(label_path, 'r') as f:
        for line in f:
            line = line.split(' ')
            img_name = label_path.split('\\')[-1][:-4]+'.jpg'
            label = line[0]
            x = line[1]
            y = line[2]
            w = line[3]
            h = line[4][:-1]
            row = [label, x, y, w, h]
            rows.append(row)
    return pd.DataFrame(data=rows, columns=['label', 'x','y','w','h'])

def add_label_column(df):
    label_list = []
    for _, row in df.iterrows():
        _, label_name = get_label(row)
        label_list.append(label_name)
    return label_list

def data_check(df):
    total = len(df)

    mask = (df['label'] == 'Mask').sum()
    mask_incorrect = (df['label'] == 'Mask incorrect').sum()
    no_mask = (df['label'] == 'No mask').sum()

    print('Dataset files: ', total)
    print('Number of Mask : %i / %i, %f %%' % (mask, total, mask*100/total))
    print('Number of Mask incorrect :  %i / %i, %f %%' % (mask_incorrect, total, mask_incorrect*100/total))
    print('Number of No mask :  %i / %i, %f %%' % (no_mask, total, no_mask*100/total))
    print('Number of No Mask + Mask incorrect :  %i / %i, %f %%' % ((no_mask+mask_incorrect), total, (no_mask+mask_incorrect)*100/total))

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

def create_yolo_labels(train, test):
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
        visualize_dataset(df, mode)
    else:
        mode = 'test'
        df = test[test['image_name'] == filename]
        visualize_dataset(df, mode)


def debug(filename):
    print(test[test['image_name'] == filename])
    img = Image.open('test/images/'+filename)
    print(img.size)
    bbox(filename)


def main():
    # 1 - Pasar las anotaciones del .mat a un pandas dataframe
    train = make_train_data()
    test = make_test_data()
    # 2 - Cambiar las anotaciones numericas accion real
    train = get_occluder_names(train)
    test = get_occluder_names(test)
    # 3 - Corregir imagenes mal anotadas
    train = train_fix_label(train)

    train.insert(loc=5, column='label', value=add_label_column(train))
    test.insert(loc=5, column='label', value=add_label_column(test))

    # visualize_dataset(df, 'train')
    # print('Training data stats')
    # data_check(train)

    # print('Test data stats')
    # data_check(test)

    bbox_number = train.groupby(['image_name']).size()

    # Quiero que el dataset este compuesto por todas las imagenes las cuales:
    # Tenga mas de una bbox 
    more_than_one = bbox_number[bbox_number > 1]
    train_mask_multiple = train[train['image_name'].isin(more_than_one.index)]
    print(train_mask_multiple)
    # Las mascarillas esten incorrectas
    train_mask_incorrect = train[train['label']=='Mask incorrect']
    # print('LEN TRAIN MASK INCORRECT', len(train_mask_incorrect))
    # Las personas no lleven mascarillla
    train_no_mask = train[train['label']=='No mask']
    # Y un % de de las imagenes en las que solo sale una mascarilla
    one = bbox_number[bbox_number == 1]
    train_mask = train[train['image_name'].isin(one.index)]
    print(train_mask)

    print('BEFORE: Training data stats')
    data_check(train)

    train_mask = train_mask.sample(frac = 0.15) # 1/5
    # Unimos todas las separaciones y este sera el dataset final
    train = pd.concat([train_mask_multiple, train_mask, train_mask_incorrect, train_no_mask], ignore_index=True)
    train = train.drop_duplicates()

    print('AFTER: Training data stats')
    data_check(train)

    print('Test data stats')
    data_check(test)

    # 4 - Pasar los dataframe al formato que usa YOLO para las anotaciones 
    create_yolo_labels(train, test)

def visualize_yolo_labels(img_path, df):
    img = cv2.imread(img_path)
    for _, row in df.iterrows():
        img = draw_yolo_bounding_box(img, row)
    cv2.imshow('img', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
    # wider_path = 'C:\\Users\\Rodrigo\\PFG\\datasets\\WIDER-FaceMask\\COVID-mask-detection_WIDER.tar\\COVID-mask-detection_WIDER\\Wider-mask-detection\\'
    # sceneraios_path = wider_path+'labels\\train\\'
    # print(os.listdir(sceneraios_path))
    # for scenario in os.listdir(sceneraios_path):
    #     directory = os.listdir(sceneraios_path+'\\'+scenario+'\\')
    #     for filename in directory:
    #         label_path = sceneraios_path+scenario+'\\'+filename
    #         print(label_path)
    #         df = get_yolo_labels(label_path)
    #         img_name = label_path.split('\\')[-1][:-4]+'.jpg'
    #         img_path = wider_path+'images\\train\\'+scenario+'\\'+img_name
    #         visualize_yolo_labels(img_path, df)
