import os
import numpy as np

from nvidia.dali import pipeline_def
import nvidia.dali.fn as fn
import nvidia.dali.types as types

batch_size=2
sequence_length=8
initial_prefetch_size=16
# video_directory = os.path.join(os.environ['DALI_EXTRA_PATH'], "db", "video", "sintel", "video_files")
# video_files=["v2-25.mp4", "v2-25.mp4", "v2-25.mp4", "v2-25.mp4", "v2-25.mp4", "v2-25.mp4", "v2-25.mp4", "v2-25.mp4", "v2-25.mp4", "v2-25.mp4", "v2-25.mp4", "v2-25.mp4", "v2-25.mp4", "v2-25.mp4"]

video_files = ["v2-25.mp4"] * 100

n_iter=6

@pipeline_def
def video_pipe(filenames):
    videos = fn.readers.video(device="gpu", filenames=filenames, sequence_length=sequence_length,
                              shard_id=0, num_shards=1, random_shuffle=True, initial_fill=initial_prefetch_size, skip_vfr_check=True)
    return videos

pipe = video_pipe(batch_size=batch_size, num_threads=2, device_id=0, filenames=video_files, seed=123456)
pipe.build()
for i in range(n_iter):
    pipe_out = pipe.run()
    sequences_out = pipe_out[0].as_cpu().as_array()
    print(sequences_out.shape)
