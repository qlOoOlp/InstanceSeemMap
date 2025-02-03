#!/bin/bash

# 실행할 Python 파일 경로
script_path="application.buildmap.buildmap"
# 하이퍼파라미터 정의
scenes=("1LXtFkjw3qL_1_none" "5LpN3gDmAk7_4" "UwV83HsGsw3_1")
# for psp in "${threshold_pixelSize_post[@]}"; do
for scene in "${scenes[@]}"; do
    # Python 명령 실행
    python -m "$script_path" --version "tb0025_ps50" --scene-id "$scene" --seem-type "bbox" --depth-sample-rate "1" --no-submap
done

echo -e "\nExecution completed."
