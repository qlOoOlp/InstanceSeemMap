import subprocess
import itertools
import multiprocessing
import time

# 실행할 Python 파일 경로
script_path = "path/to/your_script.py"

# 각 하이퍼파라미터의 후보 정의
arg1_values = ["value1", "value2", "value3", "value4", "value5"]
arg2_values = ["option1", "option2", "option3", "option4", "option5"]
arg3_values = ["choiceA", "choiceB", "choiceC", "choiceD", "choiceE"]
arg4_values = ["paramX", "paramY", "paramZ", "paramW", "paramV"]
arg5_values = ["alpha", "beta", "gamma", "delta", "epsilon"]

# 모든 조합 생성 (5^5 = 3125개)
args_combinations = list(itertools.product(arg1_values, arg2_values, arg3_values, arg4_values, arg5_values))

# 작업자 함수 정의
def run_script(arg_combination):
    """각 조합에 대해 스크립트를 실행"""
    args = [
        "--arg1", arg_combination[0],
        "--arg2", arg_combination[1],
        "--arg3", arg_combination[2],
        "--arg4", arg_combination[3],
        "--arg5", arg_combination[4]
    ]
    result = subprocess.run(["python", script_path] + args, capture_output=True, text=True)
    return {"args": args, "stdout": result.stdout, "stderr": result.stderr}

# 병렬 실행 관리
def parallel_execution(args_combinations, max_processes=3):
    """3개씩 병렬 실행"""
    pool = multiprocessing.Pool(max_processes)
    results = pool.map(run_script, args_combinations)
    pool.close()
    pool.join()
    return results

# 실행
if __name__ == "__main__":
    start_time = time.time()
    results = parallel_execution(args_combinations, max_processes=3)

    # 결과 출력
    for res in results:
        print("Args:", res["args"])
        print("STDOUT:", res["stdout"])
        print("STDERR:", res["stderr"])
    print(f"Execution completed in {time.time() - start_time:.2f} seconds")
