from _collections import deque
import boto3
import numpy as np
import cv2
from datetime import datetime
from PIL import Image, ImageDraw, ExifTags, ImageColor, ImageFont

model = 'arn:aws:rekognition:us-east-1:333527701433:project/winnie_test_training/version/' \
            'winnie_test_training.2020-04-30T22.35.42/1588300542347'

def analyzeVideo(video, model, min_confidence):

    rekognition = boto3.client('rekognition')
    vid = cv2.VideoCapture(video)
    fps = vid.get(cv2.CAP_PROP_FPS)  # frame rate
    frame_width = int(vid.get(3))
    frame_height = int(vid.get(4))
    pts = deque()
    now = datetime.now().strftime("%d/%m/%Y %H:%M:%S")


    # Define the codec and create VideoWriter object.The output is stored in 'outputs-"date and time".mp4' file.
    # Define the fps to be equal to 10. Also frame size is passed.
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    out = cv2.VideoWriter('Demo Media/Outputs/outputs-{}.avi'.format(now), fourcc, 10, (frame_width, frame_height))

    while (vid.isOpened()):
        frameId = vid.get(1)  # current frame number
        print("Processing frame id: {}".format(frameId))
        print('frame_Rate: {}'.format(fps))

        # capture frame by frame
        ret, frame = vid.read()

        # if it doesnt have a frame then break
        if ret != True:
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

            # # convert from cv2 to PIL
            # image_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            imgHeight, imgWidth, imgChannel  = frame.shape
            # draw = ImageDraw.Draw(image_pil)

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

                        # fnt = ImageFont.truetype('/Library/Fonts/Arial.ttf', 50)
                        # draw.text((left, top), customLabel['Name'], fill='#00d400', font=fnt)
                        # draw.point((x,y), fill=255)

                        print('label')
                        print('Left: ' + '{0:.0f}'.format(left))
                        print('Top: ' + '{0:.0f}'.format(top))
                        print('Label Width: ' + "{0:.0f}".format(width))
                        print('Label Height: ' + "{0:.0f}".format(height))
                        print('x coor: {}, y coor: {}'.format(x,y))
                        print(pts)

                        points = (
                            (left, top),
                            (left + width, top),
                            (left + width, top + height),
                            (left, top + height),
                            (left, top))

                        # draw.line(points, fill='#00d400', width=5)
                        cv2.line(frame, (int(left), int(top)), (int(left) + int(width), int(top)), color=(0, 0, 0), thickness=3)
                        cv2.line(frame, (int(left) + int(width), int(top)),
                                 (int(left) + int(width), int(top) + int(height)), color=(0, 0, 0), thickness=3)
                        cv2.line(frame, (int(left) + int(width), int(top) + int(height)),
                                 (int(left), int(top) + int(width)), color=(0, 0, 0), thickness=3)
                        cv2.line(frame, (int(left), int(top) + int(width)), (int(left), int(top)), color=(0, 0, 0), thickness=3)



        # numpy_image = np.array(image_pil)
        # opencv_image = cv2.cvtColor(numpy_image, cv2.COLOR_RGB2BGR)





        # loop over the set of tracked points
        for i in range(1, len(pts)):
            # if either of the tracked points are None, ignore them
            if pts[i - 1] is None or pts[i] is None:
                continue

            # otherwise, compute the thickness of the line and draw the connecting lines
            thickness = 2
            cv2.line(frame, pts[i - 1], pts[i], (44, 255, 20), thickness)

        out.write(frame)
        cv2.imshow('frame', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    vid.release()
    out.release()

def main():
    video = "Demo Media/winnie_shooting2.mov"
    model = 'arn:aws:rekognition:us-east-1:333527701433:project/winnie_test_training/version/' \
            'winnie_test_training.2020-04-30T22.35.42/1588300542347'
    min_confidence = 98

    analyzeVideo(video, model, min_confidence)


if __name__ == "__main__":
    main()