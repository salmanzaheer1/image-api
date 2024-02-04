# -*- coding: utf-8 -*-
#
# Script for teeth whitening in images.
# Requires Dlib's facial landmark predictor.
#

import os
import argparse
import cv2
import dlib
import numpy as np
from skimage import io
from PIL import Image
from scipy.spatial import distance
import IsobarImg
import requests
import json
from io import BytesIO
import random
import datetime
import base64

DEBUG = False

MAR = 0.30
FACE_IMAGE_SIZE = 10000

# Predictor path - Set your predictor file path here
PREDICTOR_PATH = "shape_predictor_68_face_landmarks.dat"

parser = argparse.ArgumentParser(description='Teeth whitening editor')
parser.add_argument('file', help='Image file')
parser.add_argument('-a', metavar='alpha', default='1.0', type=float, help='Alpha value range: 1.0-3.0')
parser.add_argument('-b', metavar='beta', default='20', type=int, help='Beta value range: 0-100')
args = parser.parse_args()

alpha = args.a
beta = args.b
image_path = args.file

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(PREDICTOR_PATH)

def mouth_aspect_ratio(mouth):
    D = distance.euclidean(mouth[33], mouth[51])
    # D1 = distance.euclidean(mouth[50], mouth[58])
    # D2 = distance.euclidean(mouth[51], mouth[57])
    # D3 = distance.euclidean(mouth[52], mouth[56])
    D1 = distance.euclidean(mouth[61], mouth[67])
    D2 = distance.euclidean(mouth[62], mouth[66])
    D3 = distance.euclidean(mouth[63], mouth[65])
    mar = (D1+D2+D3)/(3*D)
    print("mar={}".format(mar))
    return mar;
    
def alphaBlend(img1, img2, mask):
    """ alphaBlend img1 and img 2 (of CV_8UC3) with mask (CV_8UC1 or CV_8UC3)
    """
    if mask.ndim==3 and mask.shape[-1] == 3:
        alpha = mask/255.0
    else:
        alpha = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)/255.0
    blended = cv2.convertScaleAbs(img1*(1-alpha) + img2*alpha)
    return blended
    
def shape2np(s):
    num = len(s.parts())
    np_points = np.zeros((num,2), np.int32)
    idx = 0
    for p in s.parts():
        np_points[idx] = (p.x, p.y)
        idx = idx + 1
    return np_points
def upload_file_to_github(repo, token, file_path):
    """
    Uploads a file to GitHub and returns the URL of the uploaded file.

    Parameters:
    - repo: Repository name
    - path: Path in the repository where the file will be uploaded
    - token: GitHub Personal Access Token
    - file_path: Local file path to the file to upload
    """
    username = "DaniyalAhmadKhan-LUMS"  # GitHub username
    image_name = os.path.basename(file_path)
    timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    name, extension = os.path.splitext(image_name)
    image_name_dt = f"{name}_{timestamp}{extension}"
    path = f"images/{image_name_dt}"
    api_url = f"https://api.github.com/repos/{username}/{repo}/contents/{path}"
    headers = {
        "Authorization": f"token {token}",
        "Accept": "application/vnd.github.v3+json",
    }

    with open(file_path, "rb") as file:
        content = base64.b64encode(file.read()).decode("utf-8")

    data = {
        "message": f"Upload {image_name}",
        "committer": {
            "name": "Daniyal Ahmad Khan",  # Replace with your name
            "email": "22100215@lums.edu.pk"  # Replace with your email
        },
        "content": content
    }

    response = requests.put(api_url, headers=headers, json=data)

    if response.status_code in [200, 201]:
        print("Upload successful.")
        return response.json()["content"]["download_url"]
    else:
        print("Failed to upload file:", response.json())
        return None

def main():
    filename = os.path.basename(image_path)
    publicname = os.path.dirname(image_path)[:-6]
    dirname = os.path.join(publicname, 'result', filename)
    rootname = os.path.dirname(image_path)[:-13]
    if not os.path.exists(dirname):
        os.makedirs(dirname)

    img = io.imread(image_path)
    h,w,c = img.shape
    if c == 4:
        img = img[:,:,:3]
    io.imsave(dirname+"/before.jpg", img)
    res = IsobarImg.beautifyImage(dirname+"/before.jpg")
    res.save(dirname+"/after.jpg")

    faces = detector(img, 1)

    if len(faces)==0:
        print("Face not found")
        return

    max_face = 0
    max_face_id = 0
    for f, d in enumerate(faces):
        face_box = (d.bottom()-d.top())*(d.right()-d.left())
        if face_box > max_face:
            max_face = face_box
            max_face_id = f
    
    for f, d in enumerate(faces):
        if f == max_face_id:
            shape = predictor(img, d)
            break
    
    if (d.bottom()-d.top())*(d.right()-d.left()) < FACE_IMAGE_SIZE:
        print("Face too small:{}".format(max_face))
        return

    np_points = shape2np(shape)

    # detect an open mouth
    if (mouth_aspect_ratio(np_points)<MAR):
        print("Mouth not open")
        return

    # crop face
    crop_img = img[d.top():d.bottom(),d.left():d.right()]
    if crop_img.size == 0:
        return
    if DEBUG:
        io.imsave(dirname+"/face.jpg", crop_img)
    
    # crop mouth
    mouth_max_point = np.max(np_points[60:], axis=0)
    mouth_min_point = np.min(np_points[60:], axis=0)
    if DEBUG:
        io.imsave(dirname+"/mouth.jpg", img[mouth_min_point[1]:mouth_max_point[1], mouth_min_point[0]:mouth_max_point[0]])

    # mouth: 48-67
    # teeth: 60-67
    # create blank image
    mask = np.zeros((d.bottom()-d.top(), d.right()-d.left()), np.uint8)


    # create teeth mask
    cv2.fillConvexPoly(mask, np.int32(np_points[60:]-(d.left(), d.top())), 1)
    if DEBUG:
        cv2.imwrite(dirname+"/mask.jpg", mask)
    crop_jpg_with_mask= cv2.bitwise_and(crop_img, crop_img, mask = mask)

    # smoothing mask
    blur_mask = cv2.GaussianBlur(crop_jpg_with_mask,(21,21), 11.0)
    if DEBUG:
        io.imsave(dirname+'/blur_mask.jpg', blur_mask)
    
    # convert rgb2rgba
    crop_png = cv2.cvtColor(crop_img, cv2.COLOR_RGB2RGBA)
    np_alpha = blur_mask[:, :, 0]/255.0
    crop_png[:, :, 3] = blur_mask[:, :, 0]
    
    crop_png_with_brightness = cv2.convertScaleAbs(crop_png, alpha=alpha, beta=beta)

    if DEBUG:
        io.imsave(dirname+"/brightness.png", crop_png_with_brightness)
    # output
    output = np.zeros(crop_img.shape, crop_img.dtype)
    np_alpha = np_alpha.reshape(crop_img.shape[0], crop_img.shape[1], 1)
    output[:, :, :] = (1.0 - np_alpha) * crop_png[:, :, :3] + np_alpha * crop_png_with_brightness[:, :, :3]
    if DEBUG:
        io.imsave(dirname+"/output.jpg", output)

    img[d.top():d.bottom(),d.left():d.right()] = output
    io.imsave(dirname+"/whitening.jpg", img)
    res = IsobarImg.beautifyImage(dirname+"/whitening.jpg")
    res.save(dirname+"/after.jpg")
    original_image_path = dirname+"/after.jpg"
    # Example usage
    token = 'YOUR_GITHUB_DEVELOPER_TOKEN'  # Replace with your actual GitHub Personal Access Token
    repo = 'public-images'  # Repository name
    file_path = original_image_path  

    url_git = upload_file_to_github(repo, token, file_path)
    if url_git:
        print("Uploaded File URL:", url_git)
    else:
        print("Image upload failed.")



    url = "https://stablediffusionapi.com/api/v3/img2img"

    payload = json.dumps({
    "key": "3fQGgT40tNDQOGc9NJpfRpFQjwvV5wYQWeljNKoo4ZpeL0OUyOcOKzTHlwNW",
    "prompt": "Detective in a trench coat, investigating in a neon-lit, rain-soaked alley, filled with mystery.",
    "negative_prompt": None,
    "init_image": url_git,
    "width": "768",
    "height": "768",
    "samples": "1",
    "num_inference_steps": "30",
    "safety_checker": "no",
    "enhance_prompt": "yes",
    "guidance_scale": 7.5,
    "strength": 0.7,
    "base64": "no",
    "seed": None,
    "webhook": None,
    "track_id": None
    })

    headers = {
    'Content-Type': 'application/json'
    }

    response = requests.request("POST", url, headers=headers, data=payload)

    assert response.ok, response.content
    result = response.json()
    print(result["output"])



    # # Output the result
    # result = response.json()
    # print(response.status_code, result)
    # print(result["output"]["output_images"])


    # Directory where images will be saved
    save_directory = dirname
    os.makedirs(save_directory, exist_ok=True)

    # Function to download and save an image from a URL
    def download_image(url, save_path):
        response = requests.get(url)
        if response.status_code == 200:
            with open(save_path, 'wb') as file:
                file.write(response.content)
            print(f'Image saved as {save_path}')
        else:
            print(f'Failed to download the image from {url}. Status code: {response.status_code}')

    # Loop through all image URLs and download them
    for i, url in enumerate(result["output"]):
        save_path = os.path.join(save_directory, f'image_{i}.png')
        download_image(url, save_path)


if __name__ == "__main__":
    main()