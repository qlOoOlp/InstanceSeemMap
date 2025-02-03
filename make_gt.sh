#!/bin/bash

# 실행할 Python 파일 경로
script_path="application.buildmap.buildmap"
# 하이퍼파라미터 정의
scenes=("1LXtFkjw3qL_1_none" "5LpN3gDmAk7_4" "UwV83HsGsw3_1")
# for psp in "${threshold_pixelSize_post[@]}"; do
for scene in "${scenes[@]}"; do
    # Python 명령 실행
    python -m "$script_path" --version "gt2" --scene-id "$scene" --only-gt --depth-sample-rate 1
done

echo -e "\nExecution completed."

#!/bin/bash

# 실행할 Python 파일 경로
script_path="application.buildmap.buildmap"
# 하이퍼파라미터 정의
scenes=("apartment_1_4" "apartment_2_2" "frl_apartment_1_1" "office_2_1" "office_3_1" "office_4_1" "room_0_1" "room_1_1")
# for psp in "${threshold_pixelSize_post[@]}"; do
for scene in "${scenes[@]}"; do
    # Python 명령 실행
    python -m "$script_path" --version "gt2" --scene-id "$scene" --dataset-type "replica" --only-gt --depth-sample-rate 1
done

echo -e "\nExecution completed."
