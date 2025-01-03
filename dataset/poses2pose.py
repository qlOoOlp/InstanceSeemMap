import sys
import os
from PIL import Image
import numpy as np

def check_dir(input_dir):
    if not os.path.exists(input_dir):
        os.makedirs(input_dir)

def poses2pose(poses_file, pose_dir):
    # poses.txt 파일에서 pose 정보 읽기
    with open(poses_file, "r") as file:
        poses = file.readlines()
    # pose 정보를 각각의 파일에 저장
    # for idx, pose_info in enumerate(poses):
    #     pose_filename = f"{idx:06}.txt"
    #     pose_filepath = os.path.join(pose_dir, pose_filename)
    #     with open(pose_filepath, "w") as pose_file:
    #         pose_file.write(pose_info)
    for idx, pose_info in enumerate(poses):
        # pose_info = pose_info.split('\t')#-1] #! custom_0513의 경우에는 각 줄 마지막에 기존 이미지 파일명이 적혀있어서 탈락시킴
        # pose_info = '\t'.join(pose_info)

        # pose 파일명 생성 (000000.txt, 000001.txt, ...)
        pose_filename = f"{idx:06}.txt"
        # pose 파일 경로 설정
        pose_filepath = os.path.join(pose_dir, pose_filename)
        # pose 정보를 파일에 쓰기
        with open(pose_filepath, "w") as pose_file:
            pose_file.write(pose_info)
    print(f"poses.txt -> {len(poses[1:])} pose.txt done")







def rename_images(input_dir):
    # 지정한 디렉토리에서 모든 파일 목록을 가져옴
    files = os.listdir(input_dir)
    # 파일들을 정렬 (이름순)
    files.sort(key=lambda x: int(os.path.splitext(x.split('_')[0])[0]))
    
    # 이미지 파일만 골라내기 위해 확장자 필터링 (예: jpg, png)
    image_files = [file for file in files if file.endswith(('.jpg', '.jpeg', '.png'))]
    
    # 이미지 파일의 이름을 인덱스로 변경
    for index, file in enumerate(image_files):
        # 원본 파일 경로
        old_path = os.path.join(input_dir, file)
        # 새 파일명 (인덱스)과 원래 확장자
        new_filename = f"{index:06}{os.path.splitext(file)[1]}"
        new_path = os.path.join(input_dir, new_filename)
        
        # 파일 이름 변경
        os.rename(old_path, new_path)
    print(f"Renamed '{input_dir}'")

def convert_png_to_npy(input_dir, output_dir):
# 입력 디렉토리 내의 모든 파일 목록 가져오기

    file_list = os.listdir(input_dir)
    # PNG 파일만 선택하여 처리
    png_files = [f for f in file_list if f.endswith('.png')]
    
    for png_file in png_files:
        # PNG 파일의 경로
        png_path = os.path.join(input_dir, png_file)
        img_array = _convert_png_to_npy(png_path)

        # NPY 파일의 경로
        npy_file = os.path.splitext(png_file)[0] + '.npy'
        npy_path = os.path.join(output_dir, npy_file)
        
        # NumPy 배열을 NPY 파일로 저장
        np.save(npy_path, img_array)
    print(img_array)
    print(f"Converted {input_dir} to {output_dir}")


def _convert_png_to_npy(png_path):
    img = Image.open(png_path)
    ans = np.array(img)
    ans = ans/1000 #!
    return np.array(ans)
        


if __name__ =="__main__":
    if len(sys.argv) != 2:
        print("Usage: python script.py <data_dir>")
        sys.exit(1)
    data_path=sys.argv[1]


    # os.rename(data_path+"/depth", data_path+"/depth_img")
    # check_dir(data_path+"/depth")


    poses_file=data_path+"/poses.txt"
    pose_dir=data_path+"/pose"
    # rgb_dir=data_path+"/rgb"
    # depth_in=data_path+"/depth_img"
    # depth_dir=data_path+"/depth"

    check_dir(pose_dir)
    poses2pose(poses_file, pose_dir)
    # rename_images(rgb_dir)
    # rename_images(depth_in)
    # convert_png_to_npy(depth_in,depth_dir)