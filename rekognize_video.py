import json
import boto3
import numpy as np
import cv2
import math
# from sort import *
import io
import time
from PIL import Image, ImageDraw, ExifTags, ImageColor, ImageFont

model = 'arn:aws:rekognition:us-east-1:333527701433:project/winnie_test_training/version/' \
            'winnie_test_training.2020-04-30T22.35.42/1588300542347'

def analyzeVideo(video, model, min_confidence):

    rekognition = boto3.client('rekognition')
    vid = cv2.VideoCapture(video)
    fps = vid.get(cv2.CAP_PROP_FPS)  # frame rate
    frame_width = int(vid.get(3))
    frame_height = int(vid.get(4))

    current_time = time.time()

    # Define the codec and create VideoWriter object.The output is stored in 'outpy.avi' file.
    # Define the fps to be equal to 10. Also frame size is passed.
    out = cv2.VideoWriter('Demo Media/Outputs/outputs-{}.mp4'.format(current_time), cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 10, (frame_width, frame_height))

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

            # convert from cv2 to PIL
            image_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            imgWidth, imgHeight = image_pil.size
            draw = ImageDraw.Draw(image_pil)

            # calculate and display bounding boxes for each detected custom label
            for customLabel in response['CustomLabels']:
                # print('Label ' + str(customLabel['Name']))
                # print('Confidence ' + str(customLabel['Confidence']))
                if 'Geometry' in customLabel:
                    box = customLabel['Geometry']['BoundingBox']
                    left = imgWidth * box['Left']
                    top = imgHeight * box['Top']
                    width = imgWidth * box['Width']
                    height = imgHeight * box['Height']
                    x = left + (width/2)
                    y = top - (height/2)

                    fnt = ImageFont.truetype('/Library/Fonts/Arial.ttf', 50)
                    draw.text((left, top), customLabel['Name'], fill='#00d400', font=fnt)
                    draw.point((x,y), fill=255)

                    print('label')
                    print('Left: ' + '{0:.0f}'.format(left))
                    print('Top: ' + '{0:.0f}'.format(top))
                    print('Label Width: ' + "{0:.0f}".format(width))
                    print('Label Height: ' + "{0:.0f}".format(height))
                    print('x coor: {}, y coor: {}'.format(x,y))

                    points = (
                        (left, top),
                        (left + width, top),
                        (left + width, top + height),
                        (left, top + height),
                        (left, top))
                    draw.line(points, fill='#00d400', width=5)

        numpy_image = np.array(image_pil)
        opencv_image = cv2.cvtColor(numpy_image, cv2.COLOR_RGB2BGR)

        out.write(opencv_image)
        cv2.imshow('frame', opencv_image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    vid.release()
    out.release()

def main():
    video = "Demo Media/winnie_shooting.mov"
    model = 'arn:aws:rekognition:us-east-1:333527701433:project/winnie_test_training/version/' \
            'winnie_test_training.2020-04-30T22.35.42/1588300542347'
    min_confidence = 99

    analyzeVideo(video, model, min_confidence)


if __name__ == "__main__":
    main()