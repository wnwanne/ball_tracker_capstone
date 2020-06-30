from _collections import deque
import boto3
import math
import cv2
from datetime import datetime


def analyzeVideo(video, model, min_confidence):

    rekognition = boto3.client('rekognition')
    vid = cv2.VideoCapture(video)
    fps = vid.get(cv2.CAP_PROP_FPS)  # frame rate
    frame_width = int(vid.get(3))
    frame_height = int(vid.get(4))
    resolution = (frame_width, frame_height)
    pts = deque()
    now = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
    fontface = cv2.FONT_HERSHEY_SIMPLEX
    fontscale = 1
    fontcolor = (0, 0, 0)
    thickness = 3


    # Define the codec and create VideoWriter object.The output is stored in 'outputs-"date and time".mp4' file.
    # Define the fps to be equal to 10. Also frame size is passed.
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter("out-{}.avi".format(now), fourcc, 10, resolution)

    while (vid.isOpened()):
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

        #run each frame against model
        if (hasFrame):

            response = rekognition.detect_custom_labels(
                Image={
                    'Bytes': imageBytes.tobytes(),
                },
                ProjectVersionArn=model,
                MinConfidence=min_confidence
            )

            imgHeight, imgWidth, imgChannel  = frame.shape

            # calculate and display bounding boxes for each detected custom label
            for customLabel in response['CustomLabels']:
                # print('Label ' + str(customLabel['Name']))
                # print('Confidence ' + str(customLabel['Confidence']))
                if customLabel['Name'] == 'ball':
                    if 'Geometry' in customLabel:
                        box = customLabel['Geometry']['BoundingBox']
                        left = imgWidth * box['Left']
                        top = imgHeight * box['Top']
                        width = imgWidth * box['Width']
                        height = imgHeight * box['Height']
                        x = left + (width/2)
                        y = top - (height/2)
                        coords = (int(x), int(y))

                        # update the points queue
                        pts.appendleft(coords)


                        print('Label {}'.format(customLabel['Name']))
                        print('x coor: {}, y coor: {}'.format(x, y))

                        cv2.line(frame, (int(left), int(top)), (int(left) + int(width), int(top)), color=(0, 0, 0), thickness=3)
                        cv2.line(frame, (int(left) + int(width), int(top)),
                                 (int(left) + int(width), int(top) + int(height)), color=(0, 0, 0), thickness=3)
                        cv2.line(frame, (int(left) + int(width), int(top) + int(height)),
                                 (int(left), int(top) + int(width)), color=(0, 0, 0), thickness=3)
                        cv2.line(frame, (int(left), int(top) + int(width)), (int(left), int(top)), color=(0, 0, 0), thickness=3)


        # loop over the set of tracked points
        for i in range(1, len(pts)):
            # if either of the tracked points are None, ignore them
            if pts[i - 1] is None or pts[i] is None:
                continue

            # otherwise, compute the thickness of the line and draw the connecting lines
            cv2.line(frame, pts[i - 1], pts[i], (44, 255, 20), thickness)
            # compute and draw launch angle
            (x1, y1) = pts[-1]
            (x2, y2) = pts[-2]
            angle = int(math.atan((y1 - y2) / (x1 - x2)) * 180 / math.pi)
            cv2.putText(frame, str(angle), (int(x1-60), int(y1)),
                        fontface, fontscale, color=fontcolor, thickness=thickness)





        out.write(frame)
        cv2.imshow('frame', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()
    vid.release()
    out.release()

def main():
    video = "Demo Media/winnie_shooting3.mp4"
    model = 'arn:aws:rekognition:us-east-1:333527701433:project/winnie_test_training/version/' \
            'winnie_test_training.2020-04-30T22.35.42/1588300542347'
    min_confidence = 99

    analyzeVideo(video, model, min_confidence)


if __name__ == "__main__":
    main()