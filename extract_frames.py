import cv2


# get video from folder and read it in
vid_path = "/Users/nwannw/Documents/AWS/Capstone/winnie_shooting_miss.mp4"

vidcap = cv2.VideoCapture(vid_path)
success,image = vidcap.read()
count = 0

#if vid read was a success
while success:
  #write image to folder
  cv2.imwrite( "/Users/nwannw/Documents/AWS/Capstone/frame%d.jpg" % count, image) # save frame as JPEG file
  success,image = vidcap.read()
  print('Wrote a new frame: ', success)
  count += 1
  # if count > 100:
  #   success = False

