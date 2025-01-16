#!/bin/bash

# 실행할 Python 파일 경로
script_path="application.buildmap.buildmap"
evaluation_script="tuning_eval"

# JSON 파일 경로
output_file="/nvme0n1/hong/VLMAPS/InstanceSeemMap/eval_outputs/results4.json"

# JSON 파일 초기화
if [ ! -f "$output_file" ]; then
    echo "{}" > "$output_file"
fi

# 하이퍼파라미터 정의
min_depth=(0.1) #(0.1 1)
min_size_denoising_after_projection=(5)
threshold_pixelSize=(5) # 100

threshold_semSim=(0.8) # 0.95
threshold_bbox=(0.6) # 0.8
threshold_semSim_post=(0.8) # 0.95
threshold_geoSim_post=(0.2 0.4 0.6) # 0.8
# threshold_pixelSize_post=(5 25 50 100)
psp=25
# 총 조합 수 계산
total_combinations=$((${#min_depth[@]} * ${#min_size_denoising_after_projection[@]} * ${#threshold_pixelSize[@]} * ${#threshold_semSim[@]} * ${#threshold_bbox[@]} * ${#threshold_semSim_post[@]} * ${#threshold_geoSim_post[@]}))
echo "Total combinations: $total_combinations"

# tqdm 스타일 진행률 표시를 위해 `progress_bar` 함수 정의
progress_bar() {
    local total=$1
    local current=$2
    local percent=$(( 100 * current / total ))
    local progress=$(( current * 50 / total ))
    local spaces=$(( 50 - progress ))
    printf "\rProgress: |%.*s%*s| %d%% (%d/%d)" $progress "##################################################" $spaces "" $percent $current $total
}

combination_index=0
# for psp in "${threshold_pixelSize_post[@]}"; do
for sdp in "${min_size_denoising_after_projection[@]}"; do
    for ps in "${threshold_pixelSize[@]}"; do
        for ss in "${threshold_semSim[@]}"; do
            for tb in "${threshold_bbox[@]}"; do
                for ssp in "${threshold_semSim_post[@]}"; do
                    for gsp in "${threshold_geoSim_post[@]}"; do
                        for d in "${min_depth[@]}"; do
                            # 고유 키 생성
                            combination_index=$((combination_index + 1))
                            key="$combination_index"

                            # Python 명령 실행
                            python -m "$script_path" --version "optimize4" --scene-id "apartment_0_1" --vlm "seem" --seem-type "bbox" --dataset-type "replica" --using-size --min-depth "$d" --min-size-denoising-after-projection "$sdp" --threshold-pixelSize "$ps" --threshold-semSim "$ss" --threshold-bbox "$tb" --threshold-semSim-post "$ssp" --threshold-geoSim-post "$gsp" --threshold-pixelSize-post "$psp"

                            # Python 평가 실행
                            python -m "$evaluation_script" \
                                --output-file "$output_file" \
                                --key "$key" \
                                --min-depth "$d" \
                                --min-size-denoising-after-projection "$sdp" \
                                --threshold-pixelSize "$ps" \
                                --threshold-semSim "$ss" \
                                --threshold-bbox "$tb" \
                                --threshold-semSim-post "$ssp" \
                                --threshold-geoSim-post "$gsp" \
                                --threshold-pixelSize-post "$psp" \
                                --version "optimize4" \

                            # 진행률 업데이트
                            progress_bar "$total_combinations" "$combination_index"
                        done
                    done
                done
            done
        done
    done
done

echo -e "\nExecution completed."

# #!/bin/bash

# # 실행할 Python 파일 경로
# script_path="application.buildmap.buildmap"
# script_path2="tuning_eval"

# # JSON 파일 경로
# output_file="/nvme0n1/hong/VLMAPS/InstanceSeemMap/eval_outputs/results.json"

# # JSON 파일 초기화
# if [ ! -f "$output_file" ]; then
#     echo "{}" > "$output_file"
# fi

# # 하이퍼파라미터 정의
# min_depth=(0.1 1)
# min_size_denoising_after_projection=(5 25 50)
# threshold_pixelSize=(5 25 50 100)
# threshold_semSim=(0.8 0.85 0.9 0.95)
# threshold_bbox=(0.2 0.4 0.6 0.8)
# threshold_semSim_post=(0.8 0.85 0.9 0.95)
# threshold_geoSim_post=(0.2 0.4 0.6 0.8)
# threshold_pixelSize_post=(5 25 50 100)

# # 총 조합 수 계산
# total_combinations=$((${#min_depth[@]} * ${#min_size_denoising_after_projection[@]} * ${#threshold_pixelSize[@]} * ${#threshold_semSim[@]} * ${#threshold_bbox[@]} * ${#threshold_semSim_post[@]} * ${#threshold_geoSim_post[@]} * ${#threshold_pixelSize_post[@]}))
# echo "Total combinations: $total_combinations"

# # tqdm 스타일 진행률 표시를 위해 `progress_bar` 함수 정의
# progress_bar() {
#     local total=$1
#     local current=$2
#     local percent=$(( 100 * current / total ))
#     local progress=$(( current * 50 / total ))
#     local spaces=$(( 50 - progress ))
#     printf "\rProgress: |%.*s%*s| %d%% (%d/%d)" $progress "##################################################" $spaces "" $percent $current $total
# }


# combination_index=0
# for d in "${min_depth[@]}"; do
#     for sdp in "${min_size_denoising_after_projection[@]}"; do
#         for ps in "${threshold_pixelSize[@]}"; do
#             for ss in "${threshold_semSim[@]}"; do
#                 for tb in "${threshold_bbox[@]}"; do
#                     for ssp in "${threshold_semSim_post[@]}"; do
#                         for gsp in "${threshold_geoSim_post[@]}"; do
#                             for psp in "${threshold_pixelSize_post[@]}"; do
#                                 # 고유 키 생성
#                                 combination_index=$((combination_index + 1))
#                                 key="$combination_index"

#                                 # Python 명령 실행
#                                 python -m "$script_path" --scene-id "frl_apartment_1_1" --vlm "seem" --seem-type "bbox" --dataset-type "replica" --version "optimize" --using-size --min-depth "$d" --min-size-denoising-after-projection "$sdp" --threshold-pixelSize "$ps" --threshold-semSim "$ss" --threshold-bbox "$tb" --threshold-semSim-post "$ssp" --threshold-geoSim-post "$gsp" --threshold-pixelSize-post "$psp"

#                                 # Python 평가 실행
#                                 python -m "$script_path2" "
#                                 # 진행률 업데이트
#                                 progress_bar "$total_combinations" "$combination_index"
#                             done
#                         done
#                     done
#                 done
#             done
#         done
#     done
# done

# echo -e "\nExecution completed."a