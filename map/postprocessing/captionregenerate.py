import os
import re, json
import copy
import time
import random
import numpy as np
from tqdm import tqdm

# OpenAI 예외 (SDK 버전에 따라 일부가 없을 수 있어 존재하는 것만 사용)
import openai
from openai import RateLimitError
try:
    from openai import APIConnectionError, APITimeoutError, InternalServerError
except Exception:
    APIConnectionError = APITimeoutError = InternalServerError = tuple()

from map.utils.gpt import GPTPrompt, parse_object_goal_instruction


def _convert_numpy(o):
    """json.dumps(default=...)에서 사용할 NumPy 변환기"""
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"Object of type {o.__class__.__name__} is not JSON serializable")


def _to_native(obj):
    """최종 저장 전 전체 자료구조를 순수 파이썬 타입으로 변환(키는 문자열)"""
    if isinstance(obj, dict):
        return {str(k): _to_native(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_native(x) for x in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    return obj


def _call_with_retry(messages,
                     max_retries=8,
                     base_delay=1.0,
                     max_delay=15.0):
    """
    OpenAI 호출을 레이트리밋/일시적 오류에서 자동 재시도.
    - 지수 백오프 + 지터 사용
    - 성공 시 즉시 반환, 실패 시 마지막 예외 재발생
    """
    attempt = 0
    while True:
        try:
            return parse_object_goal_instruction(messages=messages)
        except RateLimitError as e:
            # 분당 토큰 한도 초과 등: 대기 후 재시도
            wait = min(max_delay, base_delay * (2 ** attempt))
            wait += random.uniform(0, 0.5 * wait)  # 지터
            print(f"[RateLimit] 대기 {wait:.2f}s 후 재시도 (attempt {attempt+1}/{max_retries}): {e}")
            time.sleep(wait)
        except (APIConnectionError, APITimeoutError, InternalServerError) as e:
            # 네트워크/타임아웃/서버 오류: 백오프로 재시도
            wait = min(max_delay, base_delay * (2 ** attempt))
            wait += random.uniform(0, 0.5 * wait)
            print(f"[Transient] 대기 {wait:.2f}s 후 재시도 (attempt {attempt+1}/{max_retries}): {e}")
            time.sleep(wait)
        except openai.OpenAIError as e:
            # 기타 OpenAI 오류는 치명적일 수 있어 즉시 중단
            print(f"[OpenAIError] 중단: {e}")
            raise
        except Exception as e:
            # 비예상 오류: 중단
            print(f"[UnknownError] 중단: {e}")
            raise

        attempt += 1
        if attempt >= max_retries:
            raise RuntimeError(f"최대 재시도 횟수({max_retries}) 초과")


class caption_regenerator:
    def __init__(self, config, save_dir, caption_path):
        self.config = config
        with open(caption_path, "r", encoding="utf-8") as f:
            self.prev_info = json.load(f)
        # self.prev_info = caption_extractor.inst_dict
        self.root_dir = config['root_path']
        self.scene_id = config['scene_id']
        self.version = config['version']
        self.save_dir = save_dir
        self.new_info = copy.deepcopy(self.prev_info)
        self.data_dir = os.path.join(
            self.root_dir, config['data_type'], config['dataset_type'], config['scene_id'],
            'map', f"{config['scene_id']}_{config['version']}"
        )

    def process(self):
        prompt_obj = GPTPrompt()

        for inst_id, inst_val in tqdm(self.prev_info.items(), desc="Regenerating Captions"):
            # 요약 입력용 JSON 문자열 (NumPy 안전 직렬화)
            info = json.dumps(inst_val, indent=4, ensure_ascii=False, default=_convert_numpy)
            print(self.new_info[inst_id])
            messages = prompt_obj.to_summarize_with_cate()[:]
            messages.append({"role": "user", "content": info})

            # ---- 재시도 로직으로 안전 호출 ----
            caption_text = _call_with_retry(
                messages,
                max_retries=8,     # 필요시 조정
                base_delay=1.0,    # 필요시 조정
                max_delay=15.0     # 필요시 조정
            )
            self.new_info[inst_id]["caption"] = caption_text
            print(self.new_info[inst_id])
            # 기존 captions는 제거(없어도 에러 안 나게)
            self.new_info[inst_id].pop("captions", None)
            print(self.new_info[inst_id])
            raise Exception("stop here")
            # 과도한 폭주 방지용 가벼운 간격 (선택)
            time.sleep(0.2)

        # 최종 저장 전 전체 구조를 순수 파이썬 타입으로 변환
        final_payload = _to_native(self.new_info)
        out_path = os.path.join(self.save_dir, "final_inst_data.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(final_payload, f, ensure_ascii=False, indent=4)
        print(f"[OK] 저장 완료: {out_path}")
