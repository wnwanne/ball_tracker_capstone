import cv2
import boto3

# get video from folder and read it in
vidcap = cv2.VideoCapture('videos/winnie_shooting_trimmed2.mp4')
success,image = vidcap.read()
count = 0

# config s3 settings
s3 = boto3.resource(
    's3',
    region_name='us-east-1',
    aws_access_key_id=KEY_ID,
    aws_secret_access_key=ACCESS_KEY
  )

#if vid read was a success
while success:
  #write image to folder
  cv2.imwrite("extracted_frames/frameA%d.jpg" % count, image)     # save frame as JPEG file
  success,image = vidcap.read()
  print('Wrote a new frame: ', success)
  count += 1




  # content = "String content to write to a new S3 file"
  # s3.Object('my-bucket-name', 'newfile.txt').put(Body=content)