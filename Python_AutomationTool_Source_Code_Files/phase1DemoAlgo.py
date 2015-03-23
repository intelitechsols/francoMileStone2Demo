
__author__      = "InteliTech"
__copyright__   = "Copyright 2015, InteliTech Solutions"



import ConMatchTrack
import numpy as np
import helper

import argparse
import cv2
from numpy import empty, nan

import os
import sys
import time


Tracker = ConMatchTrack.CMT()

parser = argparse.ArgumentParser(description='Track an object.')

parser.add_argument('vidpath', nargs='?', help='The input path.')
parser.add_argument('--logoimgpath', dest='logoimgpath' , help='The Logo Image input path.')
parser.add_argument('--c', dest='challenge', action='store_true', help='Enter the challenge authentication mode.')
parser.add_argument('--no-scale', dest='estimate_scale', action='store_false', help='Enable/Disable scale estimation')
parser.add_argument('--rotation', dest='enable_rotest', action='store_true', help='Enable/Disable rotation estimation')
parser.add_argument('--BoundBox', dest='BoundBox', help='Specify dimensions of initial bounding box.')
parser.add_argument('--pausetime', dest='pausetime', action='store_true', help='Specify the pause time')
parser.add_argument('--output-folder', dest='output_folder', help='Specify a folder for output data.')
parser.add_argument('--nogui', dest='nogui', action='store_true',
                    help='turn of GUI display (Use in combination with --output-folder ) to just derive frame by frame results  .')
parser.add_argument('--preview', dest='preview', action='store_const', const=True, default=None, help='Force preview')
parser.add_argument('--no-preview', dest='preview', action='store_const', const=False, default=None,
                    help='Disable preview')
parser.add_argument('--skipframe', dest='skipframe', action='store', default=None, help='Skip the first n frames', type=int)


arguments = parser.parse_args()
#arguments.vidpath="video3.mp4"         # for quick testing only
#arguments.logoimgpath="logo-coke3.jpg"


Tracker.estimate_scale = arguments.estimate_scale
Tracker.enable_rotest = arguments.enable_rotest

if arguments.pausetime:
    pausetimer = 0
else:
    pausetimer = 10

if arguments.output_folder is not None:
    if not os.path.exists(arguments.output_folder):
        os.mkdir(arguments.output_folder)
    elif not os.path.isdir(arguments.output_folder):
        raise Exception(arguments.output_folder + ' may exists, certainly  not as a directory')

if arguments.challenge:
    with open('images.txt') as f:
        images = [line.strip() for line in f]

    init_region = np.genfromtxt('region.txt', delimiter=',')
    num_frames = len(images)

    results = empty((num_frames, 4))
    results[:] = nan

    results[0, :] = init_region

    frame = 0

    img0 = cv2.imread(images[frame])
    img_gray0 = cv2.cvtColor(img0, cv2.COLOR_RGB2GRAY)
    img_draw = np.copy(img0)

    tl, br = (helper.array_to_int_tuple(init_region[:2]), helper.array_to_int_tuple(init_region[:2] + init_region[2:4]))

    try:
        Tracker.initialise(im_gray0, tl, br)
        while frame < num_frames:
            img = cv2.imread(images[frame])
            img_gray = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)
            Tracker.process_frame(img_gray)
            results[frame, :] = Tracker.bb

            # Advance frame number
            frame += 1
    except:
        pass  # Swallow errors

    np.savetxt('output.txt', results, delimiter=',')

else:
    # Clean up
    cv2.destroyAllWindows()

    preview = arguments.preview

    if arguments.vidpath is not None:

        # If a path to a file was given, assume it is a single video file
        if os.path.isfile(arguments.vidpath):
            Capture = cv2.VideoCapture(arguments.vidpath)

            # Skip first frames if required
            if arguments.skipframe is not None:
                Capture.set(cv2.cv.CV_CAP_PROP_POS_FRAMES, arguments.skipframe)


        # Otherwise assume it is a format string for reading images
        else:
            Capture = helper.FileVideoCapture(arguments.vidpath)

            # Skip first frames if required
            if arguments.skipframe is not None:
                Capture.frame = 1 + arguments.skipframe

        # By default do not show preview in both cases
        if preview is None:
            preview = False

    else:
        # If no input path was specified, open camera device
        Capture = cv2.VideoCapture(0)
        if preview is None:
            preview = True

    # Check if videocapture is working
    if not Capture.isOpened():
       print 'Cannot open video input.'
       sys.exit(1)


    while preview:
        response, img = Capture.read()
        cv2.imshow('Preview', img)
        k = cv2.waitKey(10)
        if not k == -1:
            break

    # Read first frame or Logo image
    if arguments.logoimgpath is not None:
        #Capture2 = cv2.VideoCapture(arguments.logoimgpath)
        #response, im0 = Capture2.read() #only consider response
        img0=cv2.imread(arguments.logoimgpath)
        img_gray0 = cv2.cvtColor(img0, cv2.COLOR_RGB2GRAY)
        img_draw = np.copy(img0)
    else:
        response, img0 = Capture.read()
        img_gray0 = cv2.cvtColor(img0, cv2.COLOR_RGB2GRAY)
        img_draw = np.copy(img0)

    print img_gray0.shape

    if arguments.BoundBox is not None:
        # Try to disassemble user specified bounding box
        values = arguments.BoundBox.split(',')
        try:
            values = [int(v) for v in values]
        except:
            raise Exception('Error: Unable to identify Co-ordinates if initial bounding box')
        if len(values) != 4:
            raise Exception('Error: Bounding box is exactly 4 elements')
        BoundBox = np.array(values)

        # Convert to point representation, adding singleton dimension
        BoundBox = helper.bb2pts(BoundBox[None, :])

        # Squeeze
        BoundBox = BoundBox[0, :]

        tl = BoundBox[:2]
        br = BoundBox[2:4]
    else:
        # Get rectangle input from user
        (tl, br) = helper.get_rect(img_draw)

    print 'using', tl, br, 'as initial bbox'

    Tracker.initialise(img_gray0, tl, br)

    frame = 1
    while True:
        # Read image
        response, img = Capture.read()
        if not response:
            break
        img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        img_draw = np.copy(img)

        tic = time.time()
        Tracker.process_frame(img_gray)
        toc = time.time()

        # Display results

        # Draw updated estimate
        if Tracker.has_result:
            cv2.line(img_draw, Tracker.tl, Tracker.tr, (255, 0, 0), 4)
            cv2.line(img_draw, Tracker.tr, Tracker.br, (255, 0, 0), 4)
            cv2.line(img_draw, Tracker.br, Tracker.bl, (255, 0, 0), 4)
            cv2.line(img_draw, Tracker.bl, Tracker.tl, (255, 0, 0), 4)

        helper.draw_keypoints(Tracker.tracked_keypoints, img_draw, (0, 255, 0))

        if arguments.output_folder is not None:
            # Original image
            cv2.imwrite('{0}/input_{1:08d}.png'.format(arguments.output_folder, frame), img)
            # Output image
            cv2.imwrite('{0}/output_{1:08d}.png'.format(arguments.output_folder, frame), img_draw)

            # Keypoints
            with open('{0}/keypoints_{1:08d}.csv'.format(arguments.output_folder, frame), 'w') as f:
                f.write('x y\n')
                np.savetxt(f, Tracker.tracked_keypoints[:, :2], fmt='%.2f')

            # Outlier
            with open('{0}/outliers_{1:08d}.csv'.format(arguments.output_folder, frame), 'w') as f:
                f.write('x y\n')
                np.savetxt(f, Tracker.outliers, fmt='%.2f')

            # Votes
            with open('{0}/votes_{1:08d}.csv'.format(arguments.output_folder, frame), 'w') as f:
                f.write('x y\n')
                np.savetxt(f, Tracker.votes, fmt='%.2f')

            # Bounding box
            with open('{0}/BoundBox_{1:08d}.csv'.format(arguments.output_folder, frame), 'w') as f:
                f.write('x y\n')
                # Duplicate entry tl is not a mistake, as it is used as a drawing instruction
                np.savetxt(f, np.array((Tracker.tl, Tracker.tr, Tracker.br, Tracker.bl, Tracker.tl)), fmt='%.2f')

        if not arguments.nogui:
            cv2.imshow('main', img_draw)

            # Check key input
            k = cv2.waitKey(pausetimer)
            key = chr(k & 255)
            if key == 'q':
                break
            if key == 'd':
                import ipdb;

                ipdb.set_trace()

        # Remember image
        img_prev = img_gray

        # Advance frame number
        frame += 1

        print '{5:04d}: center: {0:.2f},{1:.2f} scale: {2:.2f}, active: {3:03d}, {4:04.0f}ms'.format(Tracker.center[0],
                                                                                                     Tracker.center[1],
                                                                                                     Tracker.scale_estimate,
                                                                                                     Tracker.active_keypoints.shape[
                                                                                                         0],
                                                                                                     1000 * (toc - tic),
                                                                                                     frame)
