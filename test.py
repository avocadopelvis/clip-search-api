import requests

# set the search item
search_item = 'a duck'
# provide a list of videos
video_files = ["video.mp4", "video1.mp4"]

# send a POST request to the API endpoint
response = requests.post('http://localhost:5000/search_videos', 
                         data={'search_item': search_item},
                         files=[("videos", open(video, "rb")) for video in video_files])


# print the list of matching videos returned by the API
print(response.json())
