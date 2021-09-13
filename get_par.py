import pytesseract
from pytesseract import Output
import cv2
from pdf2image import convert_from_path
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TkCairo')

import os
import time

os.environ["TESSDATA_PREFIX"] = '/home/andrei/MySpace/tessdata'

H=2896
W=2048

def get_par(pdf_path, page_n, h=H, w=W):
    pil_imgs = convert_from_path(pdf_path, size=(w, h), first_page=page_n, last_page=page_n, grayscale=True)
    imgs=[np.array(i) for i in pil_imgs]
    
    
    d = pytesseract.image_to_data(imgs[0], output_type=Output.DICT)
    n_boxes = len(d['level'])
    bboxes = np.array([(d['left'][i], d['top'][i], d['width'][i] + d['left'][i], d['height'][i] + d['top'][i]) for i in range(n_boxes) if d['word_num'][i] > 0])
    
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

    # fonts_sizes=ones - minus
    fonts_sizes_gaps = minus[1:] - ones[:-1]
    fonts_sizes = ones - minus
    mean_font_size = np.mean((fonts_sizes[1:] + fonts_sizes_gaps) * 0.5)
    
    words_sizes = []
    gaps = np.hstack([minus[:, np.newaxis], ones[:, np.newaxis]])
    np.apply_along_axis(calc_mean_len, 1, gaps, img, words_sizes)
    words_sizes = np.concatenate(words_sizes)
    mean_words_sizes = np.mean(words_sizes)
    
    kernel = np.ones((int(mean_font_size), int(mean_words_sizes)))
    img = cv2.dilate(img, kernel)

    return img



if __name__ == '__main__':
    pdf_path = '/home/andrei/MySpace/python/Define/pdf/01.pdf'
    img = get_par(pdf_path, 1)
    
    import matplotlib
    import matplotlib.pyplot as plt
    matplotlib.use('TkCairo')
    plt.imshow(img)
    plt.show()
