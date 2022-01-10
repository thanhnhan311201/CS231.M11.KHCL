import argparse
import os
import os.path as osp
import imageio
import tqdm
import cv2
import numpy as np

def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('video_file')
    parser.add_argument('-r', '--rate', type=float, help='frame rate')
    args = parser.parse_args()

    video_file = args.video_file
    rate = args.rate

    out_dir = osp.splitext(osp.basename(video_file))[0]
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    reader = imageio.get_reader(video_file)
    meta_data = reader.get_meta_data()
    fps = meta_data['fps']
    n_frames = meta_data['nframes']

    for i, img in tqdm.tqdm(enumerate(reader), total=n_frames):
        if rate is None or i % int(round(fps / rate)) == 0:
            # img[:,:,0] -= 5
            # img[:,:,2] -= 5
            imageio.imsave(osp.join(out_dir, '%08d.jpg' % i), img)


if __name__ == '__main__':
    main()

# id = 1
# import os
# all_imgs = os.listdir('camera_cal/')
# for img in all_imgs:
#     os.rename(f'camera_cal/{img}', f'camera_cal/calibration{id}.jpg')
#     id += 1