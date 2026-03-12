import pickle
import openai
import os, time, re, random, base64
from openai import OpenAI
from openai import RateLimitError, APIError, APIConnectionError, APITimeoutError




# 환경 변수에서 API 키를 읽도록 유지
os.environ.get("OPENAI_API_KEY")

# 자동 재시도는 비활성화하고, 재시도는 우리 코드에서만 관리
_client = OpenAI(max_retries=0)

def _parse_retry_after_seconds(err_msg: str):
    """'Please try again in 394ms' 또는 'try again in 2s'에서 대기 시간을 초 단위로 파싱."""
    if not err_msg:
        return None
    m = re.search(r'try again in\s+(\d+)\s*ms', err_msg, flags=re.I)
    if m:
        return max(0.0, int(m.group(1)) / 1000.0)
    m = re.search(r'try again in\s+([\d\.]+)\s*s', err_msg, flags=re.I)
    if m:
        return max(0.0, float(m.group(1)))
    return None



def _retry_after_seconds_from_error(e):
    # Retry-After 헤더 우선
    try:
        resp = getattr(e, "response", None)
        if resp and getattr(resp, "headers", None):
            ra = resp.headers.get("retry-after") or resp.headers.get("Retry-After")
            if ra:
                try:
                    return float(ra)
                except ValueError:
                    pass
    except Exception:
        pass
    # 메시지 안의 'try again in 394ms/2s' 패턴
    msg = str(getattr(e, "message", "")) or str(e)
    m = re.search(r'try again in\s+(\d+)\s*ms', msg, flags=re.I)
    if m: return max(0.0, int(m.group(1))/1000.0)
    m = re.search(r'try again in\s+([\d\.]+)\s*s', msg, flags=re.I)
    if m: return max(0.0, float(m.group(1)))
    return None

def parse_object_goal_instruction(
    messages,
    *,
    model: str = "gpt-4o-mini",
    max_tokens: int = 150,
    temperature: float = 0.2,
    retry_forever: bool = True,   # ← 기본을 '무한 대기'로
    max_attempts: int = 10,       # retry_forever=False일 때만 사용
):
    """
    retry_forever=True  → 성공할 때까지 무한 재시도(대기).
    retry_forever=False → max_attempts 회까지 재시도 후 실패.
    """
    base_backoff = 0.8
    max_backoff  = 60.0  # 개별 대기 상한(초)
    attempt = 0

    try:
        while True:
            attempt += 1
            try:
                resp = _client.chat.completions.create(
                    model=model,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                )
                return (resp.choices[0].message.content or "").strip()

            except RateLimitError as e:
                wait_s = _retry_after_seconds_from_error(e)
                if wait_s is None:
                    wait_s = (0.5 + random.random()) * min(max_backoff, base_backoff * (2 ** (attempt - 1)))
                # print(f"[parse][attempt {attempt}] RateLimit → sleep {wait_s:.2f}s")
                time.sleep(wait_s)

            except APIConnectionError as e:
                wait_s = (0.5 + random.random()) * min(max_backoff, base_backoff * (2 ** (attempt - 1)))
                # print(f"[parse][attempt {attempt}] APIConnectionError: {e} → sleep {wait_s:.2f}s")
                time.sleep(wait_s)

            except APIError as e:
                status = getattr(e, "status_code", None)
                # 5xx는 재시도, 4xx는 즉시 실패(키/권한/요청형식 문제는 무한 대기해봤자 해결 안 됨)
                if status and 500 <= status < 600:
                    wait_s = (0.5 + random.random()) * min(max_backoff, base_backoff * (2 ** (attempt - 1)))
                    # print(f"[parse][attempt {attempt}] APIError {status}: {e} → sleep {wait_s:.2f}s")
                    time.sleep(wait_s)
                else:
                    print(f"[parse] APIError {status}: {e}")
                    raise

            # 무한 대기가 아닌 경우 시도 횟수 제한
            if (not retry_forever) and (attempt >= max_attempts):
                raise RuntimeError("parse_object_goal_instruction: retry attempts exceeded.")

    except KeyboardInterrupt:
        # 사용자가 수동 중단 시 깔끔히 빠져나오도록
        raise RuntimeError("parse_object_goal_instruction: interrupted by user.")



def encode_image_to_base64(image_path):
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

class GPTPrompt:
    def __init__(self):
        self.system_prompt_info = f"""Identify and describe instances. Input and output must be in JSON format.
                                    The input field 'captions' contains a list of image captions aiming to identify the instance.
                                    Output 'color' as a visual color of the identified instance
                                    Output 'material' making up the identified instance
                                    Output 'caption' as a concise description of the identified instance."""

        self.example_1_info = """{
                                    "inst_category": "bed",
                                    "room": "bedroom",
                                    "captions": [
                                        "The bed in the scene is a twin bed, featuring a white and pink color scheme. The bed is made with a white sheet and a pink blanket, giving it a classic and elegant appearance. The bed is positioned in the corner of the room, with a window nearby, allowing for natural light to enter the space.",
                                        "The bed in the scene is a small, old-fashioned bed with a white and pink color scheme. It is covered with a white sheet, and there is a pink blanket on top of it. The bed appears to be made with a floral pattern, adding a touch of vintage charm to the room. The bed is situated in the corner of the room, surrounded by a window and a chair.",
                                        "The bed in the scene is a twin-sized bed with a floral bedspread. It is positioned in the corner of the room, close to the window. The bed appears to be old-fashioned, adding a vintage touch to the bedroom."
                                    ]
                                }"""
        
        self.response_1_info = """{
                                    "inst_category": "bed",
                                    "room": "bedroom",
                                    "color": "white, pink",
                                    "material": "sheet, blanket",
                                    "captions": "The twin-sized, old-fashioned bed with a white and pink floral bedspread sits in the corner near a window, adding vintage charm.",
                                }"""
        
        self.example_2_info = """{
                                    "inst_category": "sofa",
                                    "room": "living room",
                                    "captions": [
                                        "The sofa in the scene is a brown leather couch, which is placed in the living room. The couch is positioned in front of a window, and it appears to be well-maintained and comfortable.",
                                        "The sofa in the scene is an old-fashioned, antique-style couch. It is made of brown leather and has a vintage appearance. The couch is placed in the corner of the room, and it appears to be well-maintained and preserved.",
                                        "The sofa in the scene is a brown leather couch. It is placed in front of a window, and a doll is sitting on it. The couch appears to be of good quality and is well-maintained, adding a touch of elegance to the room."
                                    ]
                                }"""
        
        self.response_2_info = """{
                                    "inst_category": "sofa",
                                    "room": "living room",
                                    "color": "brown",
                                    "material": "leather",
                                    "captions": "The well-maintained brown leather couch, with a vintage charm, sits in front of a window, adding elegance to the room.",
                                }"""
        
        self.example_3_info = """{
                                "inst_category": "curtain",
                                "room": "bedroom",
                                "captions": [
                                    "The curtain in the scene is white and made of lace. It is hanging in front of a window, covering it partially. The lace pattern adds a delicate and elegant touch to the room.",
                                    "The curtain in the scene is white and has a lace pattern. It is hanging in front of a window, covering it partially. The curtain appears to be old and worn, suggesting that it has been in place for a long time.",
                                    "There is one curtain in the scene, and it is white. The curtain is hanging in front of a window, and it appears to be sheer, which suggests that it is made of a lightweight and delicate material. The sheer curtain allows some light to pass through while still providing privacy and a sense of coziness to the room."
                                ]
                            }"""
        
        self.response_3_info = """{
                                    "inst_category": "curtain",
                                    "room": "bedroom",
                                    "color": "white",
                                    "material": "lace",
                                    "captions": "The white lace curtain, sheer and slightly worn, hangs in front of a window, adding a delicate and cozy touch.",
                                }"""
        
        self.example_4_info = """{
                                    "inst_category": "chest_of_drawers",
                                    "room": "bedroom",
                                    "captions": [
                                        "The chest of drawers in the scene is made of wood and is placed in the corner of the room. It is a small wooden dresser with a vintage appearance, likely made of high-quality materials.",
                                        "The chest of drawers in the scene is made of wood and is placed in the corner of the room. It is a small, wooden piece of furniture that adds a touch of warmth and character to the space."
                                    ]
                                }"""
        
        self.response_4_info = """{
                                    "inst_category": "chest_of_drawers",
                                    "room": "bedroom",
                                    "color": "brown",
                                    "material": "wood",
                                    "captions": "The small wooden chest of drawers sits in the corner, adding warmth and vintage charm to the room.",
                                }"""
        
        self.example_5_info = """{
                                    "inst_category": "sink",
                                    "room": "kitchen",
                                    "captions": [
                                        "The sink in the scene is white and made of porcelain. It is located in the middle of the kitchen, under the window.",
                                        "The sink in the scene is made of wood and is located in the kitchen.",
                                        "The sink in the scene is white and made of porcelain. It is a single sink, located in a bathroom setting."
                                        ]
                                }"""
        
        self.response_5_info = """{
                                    "inst_category": "sink",
                                    "room": "kitchen",
                                    "color": "white",
                                    "material": "porcelain",
                                    "captions": "The white porcelain sink, located under a window, is placed in the kitchen, adding a clean and classic touch.",
                                }"""

    def to_summarize_with_cate(self):
        prompt_json = [
            {
                "role": "system",
                "content": self.system_prompt_info
            },
            {
                "role": "user",
                "content": self.example_1_info
            },
            {
                "role": "assistant",
                "content": self.response_1_info
            },
            {
                "role": "user",
                "content": self.example_2_info
            },
            {
                "role": "assistant",
                "content": self.response_2_info
            },
            {
                "role": "user",
                "content": self.example_3_info
            },
            {
                "role": "assistant",
                "content": self.response_3_info
            },
            {
                "role": "user",
                "content": self.example_4_info
            },
            {
                "role": "assistant",
                "content": self.response_4_info
            },
            {
                "role": "user",
                "content": self.example_5_info
            },
            {
                "role": "assistant",
                "content": self.response_5_info
            }
        ]
        return prompt_json
    

    def query(self, query, json_data):
        prompt_json = [
            {"role": "system",
            "content": (
                "You are a strict selector. "
                "OUTPUT FORMAT RULES:\n"
                "- Return ONLY a bare integer (the instance_id).\n"
                "- No quotes. No code fences. No JSON. No extra text.\n"
                # "- If no suitable match exists, return exactly: -1"
            )},
            {"role": "user",
            "content": (
                f"From the following JSON list, find the item that best matches the query "
                f"and return only its instance_id as a bare integer.\n\n"# If there is no suitable match, return -1.\n\n"
                f"Query: {query}\n\nJSON:\n{json_data}"
            )}
        ]
        return prompt_json



    # def make_query(self, query_type, img_dir, instance_category, room_category, inst_content=None, inst_caption=None):
    #     base64_image = encode_image_to_base64(img_dir)

    #     # ====== 허용 표현 세트 ======
    #     VERB_VARIANTS = ['Find', 'Look for', 'Search for', 'Seek']
    #     ATTR_CLAUSE_VARIANTS = ['with', 'featuring', 'showcasing', 'characterized by', 'sporting']
    #     # 룸 위치 표현 세트(여기서만 고른다; 임의로 만들지 말 것)
    #     def allowed_room_phrases(rc: str):
    #         return [
    #             f"in the {rc}",
    #             f"inside the {rc}",
    #             f"within the {rc}",
    #             f"at the {rc}",
    #             f"located in the {rc}",
    #             f"found in the {rc}",
    #         ]

    #     # 참조 텍스트(설명/캡션 통합)
    #     refs = []
    #     if inst_content:
    #         refs.extend([str(x) for x in inst_content] if isinstance(inst_content, (list, tuple)) else [str(inst_content)])
    #     if inst_caption:
    #         refs.append(str(inst_caption))
    #     reference_text = "\n".join([f"- {r}" for r in refs]) if refs else "None."

    #     # 공통 형식 규칙(완전한 1문장 + 마침표)
    #     common_sentence_rules = (
    #         "Write EXACTLY ONE complete sentence that ENDS WITH A PERIOD.\n"
    #         "No bullet points. No markdown. No explanations. No prefixes.\n"
    #         "Do NOT use quotes or brackets. Avoid words like 'object', 'instance', or 'image'."
    #     )
    #     # 카테고리 토큰 고정 규칙
    #     literal_tokens_rules = (
    #         f'Use the object token EXACTLY as provided: "{instance_category}" — no synonyms, no pluralization, '
    #         "no paraphrasing, no casing changes."
    #     )
    #     room_token_rule = (
    #         f'Use the room token EXACTLY as provided: "{room_category}" — no synonyms, no paraphrasing, '
    #         "no casing changes, no adjectives."
    #     )

    #     # 리스트를 프롬프트에 넣기 위한 문자열화
    #     def list_to_lines(xs):  # 보기 좋게 줄바꿈
    #         return "\n  - " + "\n  - ".join(xs)

    #     # ================= 타입별 프롬프트 =================
    #     if query_type == "object":
    #         prompt = f"""
    #             You are given an image and a target object category.

    #             Target object category (use EXACTLY as given): "{instance_category}"

    #             Task:
    #             - Generate a short, natural search query that helps a user find this target in the current image.
    #             - Start the sentence by choosing EXACTLY ONE verb from this list:{list_to_lines(VERB_VARIANTS)}
    #             - Optionally add ONE concise attribute clause by choosing EXACTLY ONE connector from this list:{list_to_lines(ATTR_CLAUSE_VARIANTS)}
    #             {literal_tokens_rules}
    #             {common_sentence_rules}
    #             Do NOT mention any room or broader scene context.

    #             Good patterns (adapt to visible evidence, keep exact token):
    #             - <VERB> a sleek {instance_category} {{ATTR_CLAUSE}} metal legs.
    #             - <VERB> a {instance_category} {{ATTR_CLAUSE}} a glass top near the window.
    #             """.strip()

    #     elif query_type == "room":
    #         room_phrases = allowed_room_phrases(room_category)
    #         prompt = f"""
    #             You are given an image and two target categories.

    #             Object: "{instance_category}" (use EXACTLY)
    #             Room: "{room_category}" (use EXACTLY)

    #             Task:
    #             - Include BOTH the object and the room.
    #             - Start with EXACTLY ONE verb from:{list_to_lines(VERB_VARIANTS)}
    #             - Choose EXACTLY ONE room phrase from:{list_to_lines(room_phrases)}
    #             {literal_tokens_rules}
    #             {room_token_rule}
    #             {common_sentence_rules}

    #             Examples (pick ONE verb + ONE room phrase; keep exact tokens):
    #             - <VERB> a wooden {instance_category} inside the {room_category}.
    #             - <VERB> a {instance_category} {{ATTR_CLAUSE}} a marble top located in the {room_category}.
    #             """.strip()

    #     elif query_type == "caption":
    #         prompt = f"""
    #             You are given an image, a target object token, and optional reference descriptions.

    #             Object: "{instance_category}" (use EXACTLY)
    #             References:
    #             {reference_text}

    #             Task:
    #             - If any reference clearly matches a visible {instance_category}, base the query on it; otherwise rely on the image.
    #             - Start with EXACTLY ONE verb from:{list_to_lines(VERB_VARIANTS)}
    #             - Optionally add ONE attribute clause using:{list_to_lines(ATTR_CLAUSE_VARIANTS)}
    #             {literal_tokens_rules}
    #             {common_sentence_rules}
    #             DO NOT mention ANY room or location phrase.

    #             Example:
    #             - <VERB> a {instance_category} {{ATTR_CLAUSE}} a dark wooden frame against the wall.
    #             """.strip()

    #     elif query_type == "mixed":
    #         room_phrases = allowed_room_phrases(room_category)
    #         prompt = f"""
    #             You are given an image and two tokens.

    #             Object: "{instance_category}" (use EXACTLY)
    #             Room: "{room_category}" (use EXACTLY)
    #             References (optional):
    #             {reference_text}

    #             Task:
    #             - Start with EXACTLY ONE verb from:{list_to_lines(VERB_VARIANTS)}
    #             - Include AT LEAST ONE concrete attribute using EXACTLY ONE connector from:{list_to_lines(ATTR_CLAUSE_VARIANTS)}
    #             - Include the room using EXACTLY ONE phrase from:{list_to_lines(room_phrases)}
    #             {literal_tokens_rules}
    #             {room_token_rule}
    #             {common_sentence_rules}

    #             Patterns (choose ONE verb + ONE attr connector + ONE room phrase):
    #             - <VERB> a light blue {instance_category} {{ATTR_CLAUSE}} plush cushions located in the {room_category}.
    #             - <VERB> a rectangular {instance_category} {{ATTR_CLAUSE}} a black frame inside the {room_category}.
    #             """.strip()

    #     elif query_type == "abs":
    #         prompt = f"""
    #             You are given an image and a hidden object token.

    #             Hidden object token: "{instance_category}"  (DO NOT output this token)

    #             Task:
    #             - Produce ONE abstract, affordance-based query (e.g., 'something to sit on', 'somewhere to sleep', 'something to wash hands').
    #             - Start with EXACTLY ONE verb from:{list_to_lines(VERB_VARIANTS)}
    #             - Optionally add ONE concise attribute clause using:{list_to_lines(ATTR_CLAUSE_VARIANTS)}
    #             {common_sentence_rules}
    #             DO NOT mention ANY room or location phrase.
    #             DO NOT mention the hidden object token.

    #             Examples:
    #             - <VERB> something to sit on {{ATTR_CLAUSE}} plush cushions.
    #             - <VERB> something to store clothes.
    #             """.strip()

    #     elif query_type == "mixed_a":
    #         room_phrases = allowed_room_phrases(room_category)
    #         prompt = f"""
    # You are given an image, a hidden object token, and optional references.

    # Hidden object token: "{instance_category}"  (DO NOT output this token)
    # References (optional):
    # {reference_text}

    # Task:
    # - Produce ONE affordance-based query (abstract; no object token).
    # - Start with EXACTLY ONE verb from:{list_to_lines(VERB_VARIANTS)}
    # - Include AT LEAST ONE concise attribute using EXACTLY ONE connector from:{list_to_lines(ATTR_CLAUSE_VARIANTS)}
    # - Include the room using EXACTLY ONE phrase from:{list_to_lines(room_phrases)}
    # {common_sentence_rules}
    # {room_token_rule}

    # Patterns (choose ONE verb + ONE attr connector + ONE room phrase):
    # - <VERB> something to relax on {{ATTR_CLAUSE}} soft upholstery located in the {room_category}.
    # - <VERB> somewhere to work {{ATTR_CLAUSE}} a dark wooden surface inside the {room_category}.
    # """.strip()

    #     else:
    #         raise ValueError(f"Unknown query_type: {query_type}")

    #     prompt_json = [
    #         {"role": "system",
    #         "content": ("You are a careful vision-language assistant. "
    #                     "Follow the format strictly. Use ONLY one choice from each provided list. "
    #                     "Keep EXACT object/room tokens when required.")},
    #         {"role": "user",
    #         "content": [
    #             {"type": "text", "text": prompt},
    #             {"type": "image_url",
    #             "image_url": {"url": f"data:image/png;base64,{base64_image}", "detail": "high"}}]}
    #     ]
    #     return prompt_json











    def make_query(self, query_type, img_dir, instance_category, room_category, inst_content=None, inst_caption=None):
        base64_image = encode_image_to_base64(img_dir)

        # ====== 허용 표현 세트 ======
        VERB_VARIANTS = ['Find', 'Look for', 'Search for', 'Seek']
        ATTR_CLAUSE_VARIANTS = ['with', 'featuring', 'showcasing', 'characterized by', 'sporting']

        # 룸 위치 표현 세트(여기서만 고른다; 임의로 만들지 말 것)
        def allowed_room_phrases(rc: str):
            return [
                f"in the {rc}",
                f"inside the {rc}",
                f"within the {rc}",
                f"at the {rc}",
                f"located in the {rc}",
                f"found in the {rc}",
            ]

        # 참조 텍스트(설명/캡션 통합) — inst_caption은 사용하지 않는다
        refs = []
        if inst_content:
            refs.extend([str(x) for x in inst_content] if isinstance(inst_content, (list, tuple)) else [str(inst_content)])
        reference_text = "\n".join([f"- {r}" for r in refs]) if refs else "None."

        # 공통 형식 규칙(완전한 1문장 + 마침표)
        common_sentence_rules = (
            "Write EXACTLY ONE complete sentence that ENDS WITH A PERIOD.\n"
            "No bullet points. No markdown. No explanations. No prefixes.\n"
            "Do NOT use quotes or brackets. Avoid words like 'object', 'instance', or 'image'."
        )
        # 카테고리 토큰 고정 규칙
        literal_tokens_rules = (
            f'Use the object token EXACTLY as provided: "{instance_category}" — no synonyms, no pluralization, '
            "no paraphrasing, no casing changes."
        )
        room_token_rule = (
            f'Use the room token EXACTLY as provided: "{room_category}" — no synonyms, no paraphrasing, '
            "no casing changes, no adjectives."
        )

        # 리스트를 프롬프트에 넣기 위한 문자열화
        def list_to_lines(xs):  # 보기 좋게 줄바꿈
            return "\n  - " + "\n  - ".join(xs)

        # ================= 타입별 프롬프트 =================
        if query_type == "object":
            prompt = f"""
    You are given an image and a target object category.

    Target object category (use EXACTLY as given): "{instance_category}"

    Task:
    - Generate a short, natural search query that helps a user find this target in the current image.
    - Start the sentence by choosing EXACTLY ONE verb from this list:{list_to_lines(VERB_VARIANTS)}
    - Optionally add ONE concise attribute clause by choosing EXACTLY ONE connector from this list:{list_to_lines(ATTR_CLAUSE_VARIANTS)}
    {literal_tokens_rules}
    {common_sentence_rules}
    Do NOT mention any room or broader scene context.

    Good patterns (adapt to visible evidence, keep exact token):
    - <VERB> a sleek {instance_category} {{ATTR_CLAUSE}} metal legs.
    - <VERB> a {instance_category} {{ATTR_CLAUSE}} a glass top near the window.
    """.strip()

        elif query_type == "room":
            room_phrases = allowed_room_phrases(room_category)
            prompt = f"""
    You are given an image and two target categories.

    Object: "{instance_category}" (use EXACTLY)
    Room: "{room_category}" (use EXACTLY)

    Task:
    - Include BOTH the object and the room.
    - Start with EXACTLY ONE verb from:{list_to_lines(VERB_VARIANTS)}
    - Choose EXACTLY ONE room phrase from:{list_to_lines(room_phrases)}
    {literal_tokens_rules}
    {room_token_rule}
    {common_sentence_rules}

    Examples (pick ONE verb + ONE room phrase; keep exact tokens):
    - <VERB> a wooden {instance_category} inside the {room_category}.
    - <VERB> a {instance_category} {{ATTR_CLAUSE}} a marble top located in the {room_category}.
    """.strip()

        elif query_type == "caption":
            # 색/재질 + 고유 특징(구분력 있는 디테일) 최소 2개 속성 강제
            prompt = f"""
    You are given an image and a target object token.

    Object: "{instance_category}" (use EXACTLY)
    Optional references:
    {reference_text}

    Task:
    - Generate ONE short, natural search query that describes the visible "{instance_category}" ONLY from the image (use references if they clearly match).
    - Start with EXACTLY ONE verb from:{list_to_lines(VERB_VARIANTS)}
    - Include **AT LEAST TWO** concrete attributes **if visible**:
    (i) **ONE color or material** word, and
    (ii) **ONE distinctive feature** that helps disambiguate among same-category items
        (e.g., ice maker, built-in water dispenser, sliding doors, elegant handles,
        brass doorknob, patterned glass window, rounded edge, beveled edge, metal legs,
        glass/wooden top, glossy/matte/stainless-steel finish, woodgrain texture).
    - Use EXACTLY ONE connector for the attribute clause from:{list_to_lines(ATTR_CLAUSE_VARIANTS)}
    {literal_tokens_rules}
    {common_sentence_rules}
    DO NOT mention ANY room or location phrase.
    Prefer concrete, visual cues over vague wording.

    Example:
    - <VERB> a {instance_category} {{ATTR_CLAUSE}} a stainless steel finish and a built-in water dispenser.
    """.strip()

        elif query_type == "mixed":
            room_phrases = allowed_room_phrases(room_category)
            prompt = f"""
    You are given an image and two tokens.

    Object: "{instance_category}" (use EXACTLY)
    Room: "{room_category}" (use EXACTLY)
    Optional references:
    {reference_text}

    Task:
    - Start with EXACTLY ONE verb from:{list_to_lines(VERB_VARIANTS)}
    - Include the room using EXACTLY ONE phrase from:{list_to_lines(room_phrases)}
    - Include **AT LEAST TWO** concrete attributes **if visible**:
    (i) **ONE color or material** word, and
    (ii) **ONE distinctive feature** that helps disambiguate among same-category items
        (e.g., ice maker, built-in water dispenser, sliding doors, elegant handles,
        brass doorknob, patterned glass window, rounded edge, beveled edge, metal legs,
        glass/wooden top, glossy/matte/stainless-steel finish, woodgrain texture).
    - Use EXACTLY ONE connector for the attribute clause from:{list_to_lines(ATTR_CLAUSE_VARIANTS)}
    {literal_tokens_rules}
    {room_token_rule}
    {common_sentence_rules}

    Patterns (choose ONE verb + ONE attr connector + ONE room phrase):
    - <VERB> a {instance_category} {{ATTR_CLAUSE}} a glossy white finish and sliding doors located in the {room_category}.
    - <VERB> a {instance_category} {{ATTR_CLAUSE}} a speckled granite surface and a rounded edge inside the {room_category}.
    """.strip()

        elif query_type == "abs":
            prompt = f"""
    You are given an image and a hidden object token.

    Hidden object token: "{instance_category}"  (DO NOT output this token)

    Task:
    - Produce ONE abstract, affordance-based query (e.g., 'something to sit on', 'somewhere to sleep', 'something to wash hands').
    - Start with EXACTLY ONE verb from:{list_to_lines(VERB_VARIANTS)}
    - Optionally add ONE concise attribute clause using:{list_to_lines(ATTR_CLAUSE_VARIANTS)}
    {common_sentence_rules}
    DO NOT mention ANY room or location phrase.
    DO NOT mention the hidden object token.

    Examples:
    - <VERB> something to sit on {{ATTR_CLAUSE}} plush cushions.
    - <VERB> something to store clothes.
    """.strip()

        elif query_type == "mixed_a":
            room_phrases = allowed_room_phrases(room_category)
            prompt = f"""
    You are given an image and a hidden object token.

    Hidden object token: "{instance_category}"  (DO NOT output this token)
    Optional references:
    {reference_text}

    Task:
    - Produce ONE affordance-based query (abstract; no object token).
    - Start with EXACTLY ONE verb from:{list_to_lines(VERB_VARIANTS)}
    - Include the room using EXACTLY ONE phrase from:{list_to_lines(room_phrases)}
    - Include **AT LEAST TWO** concrete attributes **if visible**:
    (i) **ONE color or material** word, and
    (ii) **ONE distinctive feature** that helps disambiguate among same-category items
        (e.g., ice maker, built-in water dispenser, sliding doors, elegant handles,
        brass doorknob, patterned glass window, rounded edge, beveled edge, metal legs,
        glass/wooden top, glossy/matte/stainless-steel finish, woodgrain texture).
    - Use EXACTLY ONE connector for the attribute clause from:{list_to_lines(ATTR_CLAUSE_VARIANTS)}
    {common_sentence_rules}
    {room_token_rule}

    Patterns (choose ONE verb + ONE attr connector + ONE room phrase):
    - <VERB> something to keep food cold {{ATTR_CLAUSE}} a stainless steel finish and an ice maker located in the {room_category}.
    - <VERB> somewhere to store dishes {{ATTR_CLAUSE}} sliding doors and elegant handles inside the {room_category}.
    """.strip()

        else:
            raise ValueError(f"Unknown query_type: {query_type}")

        prompt_json = [
            {"role": "system",
            "content": ("You are a careful vision-language assistant. "
                        "Follow the format strictly. Use ONLY one choice from each provided list. "
                        "Keep EXACT object/room tokens when required. "
                        "Prefer concrete, visually verifiable attributes.")},
            {"role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{base64_image}", "detail": "high"}}
            ]}
        ]
        return prompt_json



















    def make_queries(self, data_dir, targets): 

        query_inst_dict = {}

        for target_id, target_val in targets.items():
            inst_info = {"msgs":{}}
            frame_path = os.path.join(data_dir, f"{int(target_val['frame_id']):06d}.png")
            frame_id = target_val['frame_id']
            instance_category = target_val['instance_category']
            room_category = target_val['room_category']
            inst_info["instance_category"] = instance_category
            inst_info["room_category"] = room_category
            inst_info["frame_id"] = frame_id
            for query_type in target_val["query_type"]:
                prompt_json = self.make_query(query_type, frame_path, instance_category, room_category)
                inst_info["msgs"][query_type] = prompt_json
            query_inst_dict[target_id] = inst_info
        return query_inst_dict

# Usage example
if __name__ == "__main__":
    prompt_obj = GPTPrompt()
    json_data = prompt_obj.to_summarize_with_cate()
    print(json_data)







