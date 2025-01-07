import subprocess
import itertools
import time
import json
import tqdm
import os
from application.evaluation.metrics import evaluation

# 실행할 Python 파일 경로
script_path = "application.buildmap.buildmap"

# 각 하이퍼파라미터의 후보 정의
min_depth = [0.1, 1]
min_size_denoising_after_projection = [5, 25, 50]
threshold_pixelSize = [0, 5, 25, 50, 100]
threshold_semSim = [0.8, 0.85, 0.9, 0.95]
threshold_bbox = [0.2, 0.4, 0.6, 0.8]
threshold_semSim_post = [0.8, 0.85, 0.9, 0.95]
threshold_geoSim_post = [0.2, 0.4, 0.6, 0.8]
threshold_pixelSize_post = [0, 5, 25, 50, 100]

# 모든 조합 생성
args_combinations = list(itertools.product(
    min_depth,
    min_size_denoising_after_projection,
    threshold_pixelSize,
    threshold_semSim,
    threshold_bbox,
    threshold_semSim_post,
    threshold_geoSim_post,
    threshold_pixelSize_post,
))

# JSON 파일 경로
output_file = "/nvme0n1/hong/VLMAPS/InstanceSeemMap/eval_outputs/results.json"

# JSON 파일에 데이터 추가
def append_to_json(output_file, key, value):
    # 파일이 없으면 초기화
    if not os.path.exists(output_file):
        with open(output_file, 'w') as f:
            json.dump({}, f, indent=4)

    # 기존 데이터 읽기
    with open(output_file, 'r') as f:
        data = json.load(f)

    # 데이터 추가
    data[key] = value

    # 파일에 다시 쓰기
    with open(output_file, 'w') as f:
        json.dump(data, f, indent=4)

# 작업자 함수 정의
def run_script(arg_combination):
    """각 조합에 대해 스크립트를 실행"""
    args = [
        "--scene-id", "frl_apartment_1_1",
        "--vlm", "seem",
        "--seem-type", "bbox",
        "--dataset-type", "replica",
        "--version", "optimize",
        "--using-size",
        "--min_depth", str(arg_combination[0]),
        "--min_size_denoising_after_projection", str(arg_combination[1]),
        "--threshold_pixelSize", str(arg_combination[2]),
        "--threshold_semSim", str(arg_combination[3]),
        "--threshold_bbox", str(arg_combination[4]),
        "--threshold_semSim_post", str(arg_combination[5]),
        "--threshold_geoSim_post", str(arg_combination[6]),
        "--threshold_pixelSize_post", str(arg_combination[7]),
    ]
    result = subprocess.run(["python", "-m", script_path] + args, capture_output=True, text=True)

    # 평가 실행
    config = {
        "data_path": "/nvme0n1/hong/VLMAPS/InstanceSeemMap/Data",
        "data_type": "habitat_sim",
        "dataset_type": "replica",
        "vlm": "ours",
        "version": "b1209_ps50aft_s",
        "scene_ids": ["frl_apartment_1_1"],
        "gt_version": "gt",
        "visualize": False,
        "bool_save": True
    }
    eval = evaluation(config)
    eval_results = eval.evaluate()
    eval_result = list[eval_results[0].values()]
    # 결과 반환
    return {"params": arg_combination, "results": eval_result}

# 순차 실행
if __name__ == "__main__":
    start_time = time.time()

    # 기존 JSON 파일에서 마지막 인덱스 가져오기
    # if os.path.exists(output_file):
    #     with open(output_file, 'r') as f:
    #         existing_data = json.load(f)
    #     last_index = max(map(int, existing_data.keys())) if existing_data else -1
    # else:
        last_index = -1

    ppbar = tqdm.tqdm(total=len(args_combinations))

    for i, arg_combination in enumerate(args_combinations):
        key = str(last_index + i + 1)  # 새로운 키 값
        result = run_script(arg_combination)

        # JSON 파일에 저장
        append_to_json(output_file, key, result)
        ppbar.update(1)  # 한 작업 완료 시 진행률 업데이트

    print(f"Execution completed in {time.time() - start_time:.2f} seconds")



# import subprocess
# import itertools
# import multiprocessing
# import time

# # 실행할 Python 파일 경로
# script_path = "application.buildmap.buildmap"#"/nvme0n1/hong/VLMAPS/InstanceSeemMap/application/buildmap/buildmap.py"#"/nvme0n1/hong/VLMAPS/InstanceSeemMap/tun.py"

# # 각 하이퍼파라미터의 후보 정의
# min_depth = [0.1, 1]
# min_size_denoising_after_projection = [5, 25, 50]
# threshold_pixelSize = [0, 5, 25, 50, 100]
# threshold_semSim = [0.8, 0.85, 0.9, 0.95]
# threshold_bbox = [0.2, 0.4, 0.6, 0.8]
# threshold_semSim_post = [0.8, 0.85, 0.9, 0.95]
# threshold_geoSim_post = [0.2, 0.4, 0.6, 0.8]
# threshold_pixelSize_post = [0, 5, 25, 50, 100]
# # 모든 조합 생성 (5^5 = 3125개)
# args_combinations = list(itertools.product(min_depth, min_size_denoising_after_projection, threshold_pixelSize, threshold_semSim, threshold_bbox, threshold_semSim_post, threshold_geoSim_post, threshold_pixelSize_post))

# # 작업자 함수 정의
# def run_script(arg_combination):
#     """각 조합에 대해 스크립트를 실행"""
#     args = [
#         "--scene-id", "frl_apartment_1_1",
#         "--vlm", "seem",
#         "--seem-type", "bbox",
#         "--dataset-type", "replica",
#         "--version", "optimize",
#         "--using-size",
#         "--min_depth", arg_combination[0],
#         "--min_size_denoising_after_projection", arg_combination[1],
#         "--threshold_pixelSize", arg_combination[2],
#         "--threshold_semSim", arg_combination[3],
#         "--threshold_bbox", arg_combination[4],
#         "--threshold_semSim_post", arg_combination[5],
#         "--threshold_geoSim_post", arg_combination[6],
#         "--threshold_pixelSize_post", arg_combination[7],
#     ]
#     result = subprocess.run(["python -m", script_path] + args, capture_output=True, text=True)
#     return {"args": args, "stdout": result.stdout, "stderr": result.stderr}

# # 병렬 실행 관리
# def parallel_execution(args_combinations, max_processes=3):
#     """3개씩 병렬 실행"""
#     pool = multiprocessing.Pool(max_processes)
#     results = pool.map(run_script, args_combinations)
#     pool.close()
#     pool.join()
#     return results

# # 실행
# if __name__ == "__main__":
#     start_time = time.time()
#     results = parallel_execution(args_combinations, max_processes=3)

#     # 결과 출력
#     for res in results:
#         print("Args:", res["args"])
#         print("STDOUT:", res["stdout"])
#         print("STDERR:", res["stderr"])
#     print(f"Execution completed in {time.time() - start_time:.2f} seconds")


# # import subprocess
# # import itertools
# # import multiprocessing
# # import time
# # from tqdm import tqdm

# # # 실행할 Python 파일 경로
# # script_path = "/nvme0n1/hong/VLMAPS/InstanceSeemMap/tun.py"

# # # 각 하이퍼파라미터의 후보 정의
# # arg1_values = ["value1", "value2", "value3", "value4", "value5"]
# # arg2_values = ["option1", "option2", "option3", "option4", "option5"]
# # arg3_values = ["choiceA", "choiceB", "choiceC", "choiceD", "choiceE"]
# # arg4_values = ["paramX", "paramY", "paramZ", "paramW", "paramV"]
# # arg5_values = ["alpha", "beta", "gamma", "delta", "epsilon"]

# # # 모든 조합 생성 (5^5 = 3125개)
# # args_combinations = list(itertools.product(arg1_values, arg2_values, arg3_values, arg4_values, arg5_values))

# # # 작업자 함수 정의
# # def run_script(arg_combination):
# #     """각 조합에 대해 스크립트를 실행"""
# #     args = [
# #         "--arg1", arg_combination[0],
# #         "--arg2", arg_combination[1],
# #         "--arg3", arg_combination[2],
# #         "--arg4", arg_combination[3],
# #         "--arg5", arg_combination[4]
# #     ]
# #     result = subprocess.run(["python", script_path] + args, capture_output=True, text=True)
# #     return {"args": args, "stdout": result.stdout, "stderr": result.stderr}

# # # 병렬 실행 관리
# # def parallel_execution(args_combinations, max_processes=3):
# #     """3개씩 병렬 실행"""
# #     results = []
# #     with multiprocessing.Pool(max_processes) as pool:
# #         # tqdm을 사용해 진행 상황 표시
# #         with tqdm(total=len(args_combinations)) as pbar:
# #             for res in pool.imap_unordered(run_script, args_combinations):
# #                 results.append(res)
# #                 pbar.update(1)  # 한 작업 완료 시 진행률 업데이트
# #     return results

# # # 실행
# # if __name__ == "__main__":
# #     start_time = time.time()
# #     results = parallel_execution(args_combinations, max_processes=3)

# #     # 결과 출력
# #     for res in results:
# #         print("Args:", res["args"])
# #         print("STDOUT:", res["stdout"])
# #         print("STDERR:", res["stderr"])
# #     print(f"Execution completed in {time.time() - start_time:.2f} seconds")
