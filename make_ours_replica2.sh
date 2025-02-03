#!/bin/bash

# 실행할 Python 파일 경로
script_path="application.buildmap.buildmap"
# 하이퍼파라미터 정의
scenes=("office_2_1" "office_3_1" "office_4_1" "room_0_1" "room_1_1")
# for psp in "${threshold_pixelSize_post[@]}"; do
for scene in "${scenes[@]}"; do
    # Python 명령 실행
    python -m "$script_path" --version "tb0025_ps50" --scene-id "$scene" --dataset-type "replica" --seem-type "bbox" --depth-sample-rate "1" --no-submap
done

echo -e "\nExecution completed."
