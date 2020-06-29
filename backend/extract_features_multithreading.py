import face_recognition
import numpy as np
import os
import time
import threading
import logging
import pandas as pd
from sklearn.decomposition import PCA

def encode_from_dir(path, encodings):
    for subdir, dirs, files in os.walk(path):
        if len(files) > 1:
            _, name = subdir.split("/");
            name = name.replace("_", " ");
            for i, file in enumerate(files):
                file_path = os.path.join(subdir, file);
                image = face_recognition.load_image_file(file_path);
                try:
                    encoding = face_recognition.face_encodings(image)[0]
                    encodings.append((file_path, encoding.tolist(), name))
                except IndexError:
                    pass

def thread_function(name, chunk):
    logging.info("Thread %s: starting", name)
    logging.info("First subdir in chunk {}: {}".format(name, chunk[0]))
    logging.info("Last subdir in chunk {}: {}".format(name, chunk[-1]))
    encodings = [];
    for i, path in enumerate(chunk):
        encode_from_dir(path, encodings);
    with open('encodings/{}.txt'.format(name), 'w+') as f:
        for encoding in encodings:
            f.write("{}\n".format(encoding))
    logging.info("Thread %s: finishing", name)

rootdir = 'fotos'
chunks = [];
n = 12;
threads_n = 8;
dirs = [ os.path.join(rootdir, subdir) for subdir in os.listdir(rootdir) ]
chunk_size = len(dirs) // (2 * n);
for i in range(0, 8):
    if i == n - 1:
        chunks.append(dirs[chunk_size*i:])
    else:
        chunks.append(dirs[chunk_size * i : chunk_size * (i+1)]);

start = time.time();
format = "%(asctime)s: %(message)s"
logging.basicConfig(format=format, level=logging.INFO,
                    datefmt="%H:%M:%S")
threads = [ None ] * threads_n;
for i, _ in enumerate(threads):
    threads[i] = threading.Thread(target=thread_function, args=(i,chunks[i],))
for thread in threads:
    thread.start()
logging.info("Main    : wait for the thread to finish")
for thread in threads:
    thread.join();
logging.info("Main    : all done")
end = time.time();
print("total time:", end - start);
