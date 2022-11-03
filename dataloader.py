import os
import sys
import pickle
import tarfile
import numpy as np
from urllib.request import urlretrieve
from sklearn.preprocessing import OneHotEncoder

def read_batch(src):
    '''Unpack the pickle files'''
    with open(src, 'rb') as f:
        if sys.version_info.major == 2:
            data = pickle.load(f)
        else:
            data = pickle.load(f, encoding='latin1') # Contains the numpy array
    return data

def process_cifar():
    '''Read data into RAM'''
    print('Preparing train set...')
    train_list = [read_batch('./cifar-10-batches-py/data_batch_{0}'.format(i + 1)) for i in range(5)]
    x_train = np.concatenate([x['data'] for x in train_list])
    y_train = np.concatenate([y['labels'] for y in train_list])
    print('Preparing test set...')
    tst = read_batch('./cifar-10-batches-py/test_batch')
    x_test = tst['data']
    y_test = np.asarray(tst['labels'])
    return x_train, x_test, y_train, y_test

def maybe_download_cifar(src="http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"):
    '''Load the training and testing data'''
    try:
        return process_cifar()
    except:
        # Catch the exception that file doesn't exist & Download
        print('Data does not exist. Downloading ' + src)
        filename = src.split('/')[-1]
        filepath = os.path.join("./",filename)
        def _recall_func(num,block_size,total_size):
            sys.stdout.write('\r>> downloading %s %.1f%%' % (filename,float(num*block_size)/float(total_size)*100.0))
            sys.stdout.flush()
        fname, h = urlretrieve(src, filepath,_recall_func)
        file_info = os.stat(filepath)
        print('Successfully download.',filename,file_info.st_size,'bytes')
        print('Extracting files...')
        with tarfile.open(fname) as tar:
            def is_within_directory(directory, target):
                
                abs_directory = os.path.abspath(directory)
                abs_target = os.path.abspath(target)
            
                prefix = os.path.commonprefix([abs_directory, abs_target])
                
                return prefix == abs_directory
            
            def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
            
                for member in tar.getmembers():
                    member_path = os.path.join(path, member.name)
                    if not is_within_directory(path, member_path):
                        raise Exception("Attempted Path Traversal in Tar File")
            
                tar.extractall(path, members, numeric_owner=numeric_owner) 
                
            
            safe_extract(tar)
        os.remove(fname)
        return process_cifar()
    
def load_cifar(channel_first=True, one_hot=False):
    # Raw data
    x_train, x_test, y_train, y_test = maybe_download_cifar()
    # Scale pixel intensity
    x_train = x_train / 255.0
    x_test = x_test / 255.0
    # Reshape
    x_train = x_train.reshape(-1, 3, 32, 32)
    x_test = x_test.reshape(-1, 3, 32, 32)
    # Channel last
    if not channel_first:
        x_train = np.swapaxes(x_train, 1, 3)
        x_test = np.swapaxes(x_test, 1, 3)
    # One-hot encode y
    if one_hot:
        y_train = np.expand_dims(y_train, axis=-1)
        y_test = np.expand_dims(y_test, axis=-1)
        enc = OneHotEncoder(categorical_features='all')
        fit = enc.fit(y_train)
        y_train = fit.transform(y_train).toarray()
        y_test = fit.transform(y_test).toarray()
    # dtypes
    x_train = x_train.astype(np.float32)
    x_test = x_test.astype(np.float32)
    y_train = y_train.astype(np.int32)
    y_test = y_test.astype(np.int32)
    return x_train, x_test, y_train, y_test