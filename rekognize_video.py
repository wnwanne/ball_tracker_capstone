from _collections import deque
import boto3
import math
import cv2
from datetime import datetime


def analyzeVideo(video, model, min_confidence):

    global left, top, height, width, x_basket, y_basket, basket_box
    rekognition = boto3.client('rekognition')
    vid = cv2.VideoCapture(video)
    fps = vid.get(cv2.CAP_PROP_FPS)  # frame rate
    frame_width = int(vid.get(3))
    frame_height = int(vid.get(4))
    resolution = (frame_width, frame_height)
    pts = deque()
    fontface = cv2.FONT_HERSHEY_SIMPLEX
    fontscale = 1
    fontcolor = (0, 0, 0)
    thickness = 3
    now = datetime.now()
    filename = 'Demo Media//Outputs//ouputvid-{}.avi'.format(now)
    made_shots = 0
    shots_taken = 0

    # Define the codec and create VideoWriter object.The output is stored in 'outputs-"date and time".mp4' file.
    # Define the fps to be equal to 10. Also frame size is passed.
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(filename, fourcc, fps, resolution)

    while vid.isOpened():
        frameId = vid.get(1)  # current frame number
        print("Processing frame id: {}".format(frameId))
        print('frame_Rate: {}'.format(fps))

        # capture frame by frame
        ret, frame = vid.read()

        # if it doesnt have a frame then break
        if not ret:
            break

        # if has frame then encode it as jpg
        hasFrame, imageBytes = cv2.imencode(".jpg", frame)

        # run each frame against model
        if (hasFrame):

            response = rekognition.detect_custom_labels(
                Image={
                    'Bytes': imageBytes.tobytes(),
                },
                ProjectVersionArn=model,
                MinConfidence=min_confidence
            )
            # Get image shape
            imgHeight, imgWidth, imgChannel  = frame.shape

            # calculate bounding boxes for each detected custom label
            for customLabel in response['CustomLabels']:
                # only looking for basket
                if customLabel['Name'] == 'basket':
                    left_basket = imgWidth * customLabel['Geometry']['BoundingBox']['Left']
                    top_basket = imgHeight * customLabel['Geometry']['BoundingBox']['Top']
                    width_basket = imgWidth * customLabel['Geometry']['BoundingBox']['Width']
                    height_basket = imgHeight * customLabel['Geometry']['BoundingBox']['Height']
                    x_basket = left_basket + (width_basket / 2)
                    y_basket = top_basket + (height_basket / 2)
                    print('Label {}'.format(customLabel['Name']))
                    print('x coor: {}, y coor: {}'.format(x_basket, y_basket))

                    # draw bounding boxes around basket
                    basket_top_left = (int(left_basket), int(top_basket))
                    basket_bot_right = (int(left_basket) + int(width_basket), int(top_basket) + int(height_basket))
                    basket_box = cv2.rectangle(frame, basket_top_left, basket_bot_right, color=(0, 0, 0), thickness=3)

                # only looking for the ball
                if customLabel['Name'] == 'ball':
                    if 'Geometry' in customLabel:
                        left = imgWidth * customLabel['Geometry']['BoundingBox']['Left']
                        top = imgHeight * customLabel['Geometry']['BoundingBox']['Top']
                        width = imgWidth * customLabel['Geometry']['BoundingBox']['Width']
                        height = imgHeight * customLabel['Geometry']['BoundingBox']['Height']
                        x_ball = left + (width/2)
                        y_ball = top + (height/2)
                        coords = (int(x_ball), int(y_ball))

                        # update the points queue
                        pts.appendleft(coords)

                        print('Label {}'.format(customLabel['Name']))
                        print('x coor: {}, y coor: {}'.format(x_ball, y_ball))

                        # draw bounding boxes around image
                        im_top_left = (int(left), int(top))
                        im_bot_right = (int(left) + int(width), int(top) + int(height))
                        cv2.rectangle(frame, im_top_left, im_bot_right,
                                      color=(0, 0, 0), thickness=3)




        # loop over the set of tracked points if ball if above basket threshold
        # (hardcoded this--NEED TO REVISIT LATER)
        # pts[i][1] is "y" and it represents the position of the ball
        # pts[i][0] is "x" and it represents the position of the ball
        # if y < pts[-1][1] means it is above the first detection of the ball when being shot we want to trace it
        # or if x = to the basket's xcoord and above the baskets ycoord keep tracing
            for i in range(1, len(pts)):
                if pts[i][1] < pts[-1][1] or (pts[i][0] == x_basket and pts[i][1] < y_basket):
            # if either of the tracked points are None, ignore them
                    if pts[i - 1] is None or pts[i] is None:
                        continue
                    # otherwise draw the connecting lines
                    cv2.line(frame, pts[i - 1], pts[i], (44, 255, 20), thickness)
                    # compute and draw launch angle
                    (x1, y1) = pts[-1]
                    (x2, y2) = pts[-2]
                    angle = int(math.atan((y1 - y2) / (x1 - x2)) * 180 / math.pi)
                    cv2.putText(frame, str(angle), (int(x1-60), int(y1)),
                                fontface, fontscale, color=fontcolor, thickness=thickness)


        # write the video to a file and show the video
        out.write(frame)
        cv2.imshow('frame', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()
    vid.release()
    out.release()


def main():
    # video = "/Users/nwannw/Documents/AWS/Capstone/winnie_shooting.mp4"
    video = 'Demo Media/winnie_shooting2.mp4'
    model = 'arn:aws:rekognition:us-east-1:333527701433:project/winnie_test_training/version/' \
            'winnie_test_training.2020-04-30T22.35.42/1588300542347'

    new_model = 'arn:aws:rekognition:us-east-1:333527701433:project/capstone_training/version/' \
                'capstone_training.2020-07-01T13.02.04/1593622924884'

    newest_model = 'arn:aws:rekognition:us-east-1:333527701433:project/capstone_try2/version/' \
                   'capstone_try2.2020-07-06T16.44.31/1594068271491'

    min_confidence = 95

    analyzeVideo(video, model, min_confidence)


if __name__ == "__main__":
    main()
