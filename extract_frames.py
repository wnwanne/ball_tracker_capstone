import cv2


# get video from folder and read it in
vid_path = 'Demo Media/winnie_shooting2.mov'

vidcap = cv2.VideoCapture(vid_path)
success,image = vidcap.read()
count = 0

#if vid read was a success
while success:
  #write image to folder
  cv2.imwrite( "Demo Media/Frames/frameA%d.jpg" % count, image) # save frame as JPEG file
  success,image = vidcap.read()
  print('Wrote a new frame: ', success)
  count += 1
  if count > 100:
    success = False




  # content = "String content to write to a new S3 file"
  # s3.Object('my-bucket-name', 'newfile.txt').put(Body=content)