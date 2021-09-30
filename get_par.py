import pytesseract
from pytesseract import Output
import cv2
from pdf2image import convert_from_path
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from tesserocr import PyTessBaseAPI, RIL, PSM, iterate_level
import ctypes

import pandas as pd
matplotlib.use('TkCairo')

import os
import time

os.environ["TESSDATA_PREFIX"] = '/home/andrei/MySpace/tessdata'

H=2896
W=2048
COLUMN_THRESH=0.5
LIB_PATH = '/home/andrei/MySpace/python/ocr_new/ocr/ocr/core/header_foter_detect.so'

def get_par(pdf_path, page_n, h=H, w=W, ):
    pil_imgs = convert_from_path(pdf_path, size=(w, h), first_page=page_n, last_page=page_n, grayscale=True)
    imgs=[np.array(i) for i in pil_imgs]
    
    rgb_im_pill = convert_from_path(pdf_path, size=(w, h), first_page=page_n, last_page=page_n, grayscale=True)
    rgb_im=[np.array(i) for i in rgb_im_pill][0]
    
    res = []
    words = []
    with PyTessBaseAPI(path='/home/andrei/MySpace/tessdata', psm=PSM.AUTO) as api:
        api.SetImageBytes(
                       imagedata=imgs[0].tobytes(),
                       width=W,
                       height=H,
                       bytes_per_pixel=1,
                       bytes_per_line=W)
        boxes = api.GetComponentImages(RIL.WORD, True)
        api.Recognize()
        level = getattr(RIL, 'WORD')
        iter = api.GetIterator()
        for r in iterate_level(iter, level):
            element = r.GetUTF8Text(level)
            word_attributes = {}
            if element and not element.isspace():
                res.append(r.BoundingBox(level))
                words.append(element)


    bboxes = np.array(res, dtype='int32')

        
    def draw_bbox(bbox, img):
        img[bbox[1]:bbox[3], bbox[0]:bbox[2]] = 255

    def calc_mean_len(gaps, mat, result):
        w_hist = mat[gaps[0]:gaps[1], :].sum(axis=0)

        w_bin = (w_hist != 0) * 1

        w_bin_dif = -np.diff(w_bin)

        minus = (w_bin_dif < 0).nonzero()[0]
        ones = (w_bin_dif > 0).nonzero()[0]
        words_sizes = minus[1:] - ones[:-1]

        result.append(words_sizes)

    img = np.full((H, W), 0, dtype='uint8')

    np.apply_along_axis(draw_bbox, 1, bboxes, img)


    h_hist = img.sum(axis=1)
    h_bin = h_hist == 0
    h_bin = h_bin*1
    h_bin_dif = np.diff(h_bin*1)

    minus = (h_bin_dif < 0).nonzero()[0]
    ones = (h_bin_dif > 0).nonzero()[0]

    df = pd.DataFrame(data={'word': words,
     'x1': bboxes[:,0],
     'y1': bboxes[:,1],
     'x2': bboxes[:,2],
     'y2': bboxes[:,3]})

    return bboxes, imgs[0], df, rgb_im

def draw_bbox(bbox, img):
    img[bbox[1]:bbox[3], bbox[0]:bbox[2]] = 255

def get_par_from_bbox(bboxes, words, h=H, w=W):
    def draw_bbox(bbox, img):
        img[bbox[1]:bbox[3], bbox[0]:bbox[2]] = 255

    def calc_mean_len(gaps, mat, result):
        w_hist = mat[gaps[0]:gaps[1], :].sum(axis=0)

        w_bin = (w_hist != 0) * 1

        w_bin_dif = -np.diff(w_bin)

        minus = (w_bin_dif < 0).nonzero()[0]
        ones = (w_bin_dif > 0).nonzero()[0]
        words_sizes = minus[1:] - ones[:-1]

        result.append(words_sizes)

    def calc_areas_ratio(img, x, y):
        s1 = np.sum(img[y[0, 0]:y[-1, 1], x[0, 0]:x[0, 1]])
        s2 = np.sum(img[y[0, 0]:y[-1, 1], x[1, 0]:x[1, 1]])

        ratio = s1 / s2

        return ratio

    def divider_1d(axis_projection):
        x_binary = (axis_projection != 0) * 1
        x_bin_dif = np.zeros((x_binary.shape[0] + 1), dtype=x_binary.dtype)
        x_bin_dif[0] = x_binary[0]
        x_bin_dif[-1] = x_binary[-1] * (-1)
        x_bin_dif[1:x_binary.shape[0]] = np.diff(x_binary)
        x_start = (x_bin_dif > 0).nonzero()[0]
        x_end = (x_bin_dif < 0).nonzero()[0]
        x_interval = np.hstack(
            [x_start[:, np.newaxis], x_end[:, np.newaxis]])
        return x_interval

    def divider_2d(img, x1, y1, x2, y2):
        x_proj = np.sum(img[y1:y2, x1:x2], axis=0)
        y_proj = np.sum(img[y1:y2, x1:x2], axis=1)

        x_interval = divider_1d(x_proj) + x1
        y_interval = divider_1d(y_proj) + y1

        return x_interval, y_interval

    def _divide_by_cols(_img, x1, y1, x2, y2):
        # !!! MUST BE REFACTORED !!!
        # devide if it is possible
        img = _img[y1:y2, x1:x2]
        x_proj_para = np.sum(img, axis=0)
        x_proj = np.heaviside(x_proj_para - (1 - 0.9) * np.mean(x_proj_para),
                              0.9 * np.mean(x_proj_para)).astype('int32')
        x_binary = (x_proj != 0) * 1
        x_bin_dif = np.zeros((x_binary.shape[0] + 1), dtype=x_binary.dtype)
        x_bin_dif[0] = x_binary[0]
        x_bin_dif[-1] = x_binary[-1] * (-1)
        x_bin_dif[1:x_binary.shape[0]] = np.diff(x_binary)
        x_start = (x_bin_dif > 0).nonzero()[0][1:]
        x_end = (x_bin_dif < 0).nonzero()[0][:-1]
        x_interval = np.hstack(
            [x_end[:, np.newaxis], x_start[:, np.newaxis]])

        for interval_ in x_interval:
            slice_ = x_proj_para[interval_[0]:interval_[1]]
            index_ = slice_.argmin() + interval_[0]
            img[:, index_] = 0

        if len(x_interval) > 0:
            return True
        else:
            return False

    def split(img, x1, y1, x2, y2, paragraphs):
    
        x_interval, y_interval = divider_2d(img, x1, y1, x2, y2)

        # case 1
        if (len(x_interval) == 1) and (len(y_interval) == 1):
            divided = _divide_by_cols(img, x1, y1, x2, y2)
            if divided:
                split(img, x1, y1, x2, y2, paragraphs)
            else:
                paragraphs.append([x_interval[0, 0], y_interval[0, 0],
                                   x_interval[0, 1], y_interval[0, 1]])

        # case 2
        elif (len(x_interval) == 1) and (len(y_interval) > 1):
            x_intrvl, _ = divider_2d(img,
                                     x_interval[0, 0], y_interval[0, 0],
                                     x_interval[0, 1], y_interval[0, 1])

            n_start = len(x_intrvl)

            for i in range(1, y_interval.shape[0]):
                x_intrvl, _ = divider_2d(img,
                                         x_interval[0, 0], y_interval[0, 0],
                                         x_interval[0, 1], y_interval[i, 1])
                n = len(x_intrvl)
                if n != n_start:
                    n_start = i - 1
                    break
                elif i == (y_interval.shape[0] - 1):
                    n_start = 0

            split(img,
                  x_interval[0, 0], y_interval[0, 0],
                  x_interval[0, 1], y_interval[n_start, 1],
                  paragraphs)
            split(img,
                  x_interval[0, 0], y_interval[n_start + 1, 0],
                  x_interval[0, 1], y_interval[-1, 1],
                  paragraphs)

        # case 3
        elif (len(x_interval) > 1) and (len(y_interval) == 1):
            # try to split by y-axis
            ratio = calc_areas_ratio(img, x_interval, y_interval)
            _, y_intrvl = divider_2d(img,
                                     x_interval[0, 0], y_interval[0, 0],
                                     x_interval[0, 1], y_interval[-1, 1])
            n = len(y_intrvl)
            if (n > 1) and (ratio < COLUMN_THRESH):
                # split by y-axis
                for y_interval_ in y_intrvl:
                    img[y_interval_[0], x_interval[0, 0]:x_interval[-1, 1]] = 0

                split(img,
                      x_interval[0, 0], y_interval[0, 0],
                      x_interval[-1, 1], y_interval[-1, 1],
                      paragraphs)
            else:
                split(img,
                      x_interval[0, 0], y_interval[0, 0],
                      x_interval[0, 1], y_interval[0, 1],
                      paragraphs)
                split(img,
                      x_interval[1, 0], y_interval[0, 0],
                      x_interval[-1, 1], y_interval[0, 1],
                      paragraphs)

        # case 4
        elif (len(x_interval) > 1) and (len(y_interval) > 1):
            ratio = calc_areas_ratio(img, x_interval, y_interval)
            # by columns
            if ratio > COLUMN_THRESH:
                split(img,
                      x_interval[0, 0], y_interval[0, 0],
                      x_interval[0, 1], y_interval[-1, 1],
                      paragraphs)
                split(img,
                      x_interval[1, 0], y_interval[0, 0],
                      x_interval[-1, 1], y_interval[-1, 1],
                      paragraphs)

            # by rows
            else:
                # find max y-axis gap
                gaps = y_interval[1:, 0] - y_interval[:-1, 1]
                maxpag_ind = gaps.argmax()
                split(img,
                      x_interval[0, 0], y_interval[0, 0],
                      x_interval[1, 1], y_interval[maxpag_ind, 1],
                      paragraphs)
                split(img,
                      x_interval[0, 0], y_interval[maxpag_ind+1, 0],
                      x_interval[1, 1], y_interval[-1, 1],
                      paragraphs)

                if len(x_interval)>2:
                    split(img,
                        x_interval[2, 0], y_interval[0, 0],
                        x_interval[-1, 1], y_interval[maxpag_ind, 1],
                        paragraphs)
                    split(img,
                        x_interval[2, 0], y_interval[maxpag_ind+1, 0],
                        x_interval[-1, 1], y_interval[-1, 1],
                        paragraphs)

                

    # bboxes = df[['x1', 'y1', 'x2', 'y2']].to_numpy()

    img_w = np.full((H, W), 0, dtype='uint8')
    if bboxes.shape[0] == 0:
        bboxes = np.array([[0, 0, H, W]], dtype='int32')
    np.apply_along_axis(draw_bbox, 1, bboxes, img_w)

    # img_w[1300:1500, 1800:2048] = 0 
    # img_w[1190:1216, 1000:1600] = 255

    h_k = (bboxes[:, 3] - bboxes[:, 1]).mean()
    kernel = np.ones((int(h_k), int(h_k) * 2), dtype='uint8')
    img = cv2.dilate(img_w, kernel)
    img = cv2.erode(img, kernel)

    paragraphs = []
    split(img, 0, 0, W, H, paragraphs)
    paragraphs = np.array(paragraphs, dtype='int32')

    df_w = pd.DataFrame(data=words, columns=['word'])
    df = pd.DataFrame(columns=['x1', 'y1', 'x2', 'y2', 'par_bel', 'par_ord'], index=np.arange(0,len(bboxes)), data=-1)
    df[['x1', 'y1', 'x2', 'y2']] = bboxes
    par_bel = df[['par_bel']].to_numpy()
    bboxes_extra = df[['x1', 'y1', 'x2', 'y2', 'par_bel', 'par_ord']].to_numpy()

    lib = ctypes.pydll.LoadLibrary(LIB_PATH)
    lib.head_foot_det.restype = None

    def assign_word_to_par(bboxes_extra, paragraphs):
        for i, p in enumerate(paragraphs):
            ix = ((bboxes_extra[:, 2] <= p[0]) | (bboxes_extra[:, 0] >= p[2]))
            iy = ((bboxes_extra[:, 3] <= p[1]) | (bboxes_extra[:, 1] >= p[3]))
            mask = ~(ix | iy)

            bboxes_extra[mask, 4] = i
            y1 = bboxes_extra[mask][:,1].astype('int32')
            y2 = bboxes_extra[mask][:,3].astype('int32')
            x1 = bboxes_extra[mask][:,0].astype('int32')

            y1_index = y1.argsort()
            y1 = y1[y1_index]
            y2 = y2[y1_index]
            x1 = x1[y1_index]

            # print(y1)

            line_id = np.full(len(y1), 0, dtype='int32')
            lib.head_foot_det(ctypes.py_object(y1),
                      ctypes.py_object(y2),
                      ctypes.py_object(line_id))

            seq = np.arange(len(line_id), dtype='int32')
            # seq = seq[y1_index]

            # for i, f in enumerate(zip(y1, line_id, y1_index)):
            #     print(f[0], f[1], f[2], i)
            
            
            divs = np.unique(line_id, return_index=True)[1]
            bords = np.empty((len(divs), 2), dtype='int32')
            bords[:-1,:] = np.hstack([divs[:-1][:, np.newaxis], divs[1:][:, np.newaxis]])
            bords[-1,0] = divs[-1]
            bords[-1,1] = len(line_id)
            # print(bords)
            # exit(1)

            if len(bords)==0:
                seq = seq[np.argsort(x1)]
            else:
                for i, b in enumerate(bords):
                    tmp = seq[b[0]:b[1]]
                    if i <=2:
                        print(tmp, y1[b[0]:b[1]], y1_index[np.argsort(x1[b[0]:b[1]])] + b[0], np.argsort(x1[b[0]:b[1]]))
                        print('--')
                    # seq[b[0]:b[1]] = y1_index[np.argsort(x1[b[0]:b[1]])] + b[0]
                    seq[b[0]:b[1]] = y1_index[np.argsort(x1[b[0]:b[1]]) + b[0]]
                    # seq[b[0]:b[1]] = np.arange(b[1]-b[0], dtype='int32') + prev
                    # prev += b[1]-b[0]
                    # seq[(b[0] <= i) & (i < b[1])] = i
            print(seq[:bords[2,1]])
            up_seq = y1_index[seq]
            print(up_seq[:bords[2,1]])
            print(seq)
            # print(y1_index)

            exit(1)
            # seq[:9] = -1
            bboxes_extra[mask, 5] = seq

    assign_word_to_par(bboxes_extra, paragraphs)

    df[['x1', 'y1', 'x2', 'y2', 'par_bel', 'par_ord']] = bboxes_extra

    return img, paragraphs, df



if __name__ == '__main__':
    pdf_path = '/home/andrei/MySpace/python/Define/pdf/columns/Short Form Agreement - UK and US (Feb 2020) CAP Reviewed.pdf'
    # pdf_path = '/home/andrei/MySpace/python/Define/pdf/06.pdf'
    bboxes, img_text, df_s, rgb_im = get_par(pdf_path, 4)
    print(df_s)

    img, paragraphs, df = get_par_from_bbox(bboxes, None)

    # img1, img2, img3 = get_par_v2(bboxes)

    data = df[['x1', 'y1', 'x2', 'y2', 'par_bel', 'par_ord']].to_numpy()

    for i, p in enumerate(paragraphs):
        cv2.rectangle(img_text, (p[0], p[1]),
                              (p[2], p[3]), (100, 100, 100), 10)

        for i, p in enumerate(paragraphs):
            if i < (len(paragraphs) - 1):
                p2 = paragraphs[i + 1]
                cv2.arrowedLine(img_text, (p[0] + (p[2] - p[0])//2, p[1] + (p[3] - p[1])//2),
                                    (p2[0] + (p2[2] - p2[0])//2, p2[1] +
                                     (p2[3] - p2[1])//2), (0, 0, 0), 10)
    
    for i, P in enumerate(data):
        p = P[:4]
        t = P[5]
        if t < 10:
            cv2.putText(img_text, str(t), (p[0] + (p[2] - p[0])//2, p[1] + (p[3] - p[1])//2), cv2.FONT_HERSHEY_SIMPLEX, 
                   2, (0,150,0), 2, cv2.LINE_AA)

    
    import matplotlib
    import matplotlib.pyplot as plt
    matplotlib.use('TkCairo')
    f, ax = plt.subplots(1,2)
    ax[0].imshow(img, 'gray')
    ax[1].imshow(img_text, 'gray')
    print(len(df))
    
    print(len(df[df['par_bel'] == -1]))
    print(np.sort(df['par_bel'].unique()))
    print(df[df['par_bel']==0].to_string())

    # print(df[df['par_bel'] == 1])
    
    # ax[2].imshow(img3, 'gray')
    plt.show()
