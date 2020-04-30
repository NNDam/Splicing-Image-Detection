from skimage.feature import local_binary_pattern
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC, LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle
from scipy import fftpack
import numpy as np
from progressbar import progressbar
import glob
import cv2
import pywt

def extract_chromatic_channel(bgr_img):
    # Extract 2 chromatic channes from BGR image
    # Input: BGR Image
    # Output: CrCb channels 
    ycrcb_image = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2YCR_CB)
    return ycrcb_image[:, :, 1:]


def block_processing(cb_image, block_size, stride):
    # Divide image into multiple overlap blocks
    # Input: Cr or Cb channel
    # Output: List of blocks
    height, width, _ = cb_image.shape
    img_blocks = []
    for i in range(0, height - block_size, stride):
        for j in range(0, width - block_size, stride):
            img_blocks.append(cb_image[i: i + block_size, \
                j: j + block_size])
    return np.array(img_blocks)

def extract_lbp_dct(blocks, n_points = 8, radius = 1):
    # Extract feature vector from given blocks
    # Input: List of blocks response with given image
    # Output: Feature vector of given image
    n_blocks, block_size, _, _ = blocks.shape
    CR_feature = np.zeros((n_blocks, block_size, block_size))
    CB_feature = np.zeros((n_blocks, block_size, block_size))
    for idx, block in enumerate(blocks):
        CR_lbp          = local_binary_pattern(block[:, :, 0], n_points, radius)
        CR_lbp          = np.float32(CR_lbp)
        CR_feature[idx] = cv2.dct(CR_lbp)
        CB_lbp          = local_binary_pattern(block[:, :, 1], n_points, radius)
        CB_lbp          = np.float32(CB_lbp)
        CB_feature[idx] = cv2.dct(CB_lbp)
    CR_feature = np.std(CR_feature, axis = 0).flatten()
    CB_feature = np.std(CB_feature, axis = 0).flatten()
    return np.concatenate([CR_feature, CB_feature], axis = 0)

def extract_feature(cb_image, block_size, stride):
    # Extract feature from given CrCb channels
    # Input: CrCb channels
    # Output: Feature vector or given original image
    img_blocks = block_processing(cb_image, block_size, stride)
    feature    = extract_lbp_dct(img_blocks)
    return feature 

def read_and_extract_feature(list_img, block_sizes, strides):
    # Read and extract feature vector from given list images
    total_img = len(list_img)
    dim = 0
    for i in range(len(block_sizes)):
        dim += block_sizes[i] ** 2
    features = np.zeros((total_img, 2*dim))
    for idx in progressbar(range(len(list_img))):
        im         = list_img[idx]
        bgr_img    = cv2.imread(im)
        cb_image   = extract_chromatic_channel(bgr_img)
        tmp        = 0
        for i, bz in enumerate(block_sizes):
            features[idx, tmp: tmp + 2*bz**2] = extract_feature(cb_image, bz, strides[i])
            tmp += 2*bz ** 2
    return features

def process_dataset(folders_real, folders_fake, block_sizes = [32], strides = [16]):
    # Process CASIA dataset
    # Label: 0 - fake image
    #        1 - real image
    list_real = []
    list_fake = []
    for fdr in folders_real:
        list_real += glob.glob(fdr)
    for fdf in folders_fake:
        list_fake += glob.glob(fdf)
    Y_train = np.zeros((len(list_real) + len(list_fake), ), dtype = np.float32)
    Y_train[: len(list_real)] = 1.0
    X_train = read_and_extract_feature(list_real + list_fake, block_sizes = block_sizes, strides = strides)
    return X_train, Y_train

if __name__ == '__main__':
    folder_real = ['CASIA2/Au/*.jpg']
    folder_fake = ['CASIA2/Tp/*.jpg', 'CASIA2/Tp/*.tif']
    print('Build SVM model ...')
    X, Y = process_dataset(folder_real, folder_fake)
    print('Build SVM model ...')
    X, Y = shuffle(X, Y)
    scaler  = StandardScaler()
    X_train = scaler.fit_transform(X)
    clf = LinearSVC()
    scores = cross_val_score(clf, X_train, Y, cv=5, scoring='f1_macro')
    print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
