import cv2
from PIL import Image
import math
import clip
import torch
from flask import Flask, request, jsonify

device = "cuda" if torch.cuda.is_available() else "cpu"
# load OpenAI CLIP model
model, preprocess = clip.load("ViT-B/32", device=device)
app = Flask(__name__)

# define the API endpoint
@app.route('/search_videos', methods=['POST'])
def search_videos():
    # get the search item from the request parameters
    search_item = request.form.get('search_item')

    # get the video files from the request parameters
    video_files = request.files.getlist('video_files')

    # store the matching videos
    matching_videos = []

    # loop through all videos
    for video_file in video_files:
        # no. of frames to skip
        n = 120

        # store the video frames
        video_frames = []

        # open the video
        capture = cv2.VideoCapture(video_file)
        fps = capture.get(cv2.CAP_PROP_FPS)

        current_frame = 0
        # read the current frame
        ret, frame = capture.read()
        while capture.isOpened() and ret:
            ret,frame = capture.read()

            if ret:
              video_frames.append(Image.fromarray(frame[:, :, ::-1]))

            # skip n frames
            current_frame += n
            capture.set(cv2.CAP_PROP_POS_FRAMES, current_frame)

        # ENCODE THE FRAMES
        batch_size = 256
        batches = math.ceil(len(video_frames) / batch_size)

        # store the encoded features 
        video_features = torch.empty([0, 512], dtype=torch.float16).to(device)

        # process each batch
        for i in range(batches):
          # get the relevant frames
          batch_frames = video_frames[i*batch_size : (i+1)*batch_size]
          
          # preprocess the frames for the batch
          batch_preprocessed = torch.stack([preprocess(frame) for frame in batch_frames]).to(device)
          
          # encode with CLIP and normalize
          with torch.no_grad():
            batch_features = model.encode_image(batch_preprocessed)
            batch_features /= batch_features.norm(dim=-1, keepdim=True)

          # append the batch to the list containing all features
          video_features = torch.cat((video_features, batch_features))

        # determine if video contains the search item
        if contain_search_item(video_frames, video_features, search_item):
           matching_videos.append(video_file)
           # break

    # return the list of matching videos 
    return matching_videos


def contain_search_item(video_frames, video_features, search_query):
  # encode and normalize the search query using CLIP
  with torch.no_grad():
    text_features = model.encode_text(clip.tokenize(search_query).to(device))
    text_features /= text_features.norm(dim=-1, keepdim=True)

  # compute the similarity between the search query and each frame 
  similarities = (100.0 * video_features @ text_features.T)
  values, best_photo_idx = similarities.topk(1, dim=0)

  for frame_id in best_photo_idx:
    frame = video_frames[frame_id]
  
  return frame


if __name__ == '__main__':
    app.run(debug=True)

