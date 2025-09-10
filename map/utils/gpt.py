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

# def parse_object_goal_instruction(
#     messages,
#     *,
#     model: str = "gpt-4o-mini",
#     max_tokens: int = 28,
#     temperature: float = 0.2,
#     timeout: int = 60,
# ):
#     """
#     Chat Completions 호출 + 재시도(지수 백오프 + 지터).
#     타입별 max_tokens/temperature를 외부에서 주입 가능.
#     """
#     max_attempts = 10
#     base_backoff = 0.8
#     max_backoff  = 20.0

#     for attempt in range(1, max_attempts + 1):
#         try:
#             resp = _client.chat.completions.create(
#                 model=model,
#                 messages=messages,
#                 max_tokens=max_tokens,
#                 temperature=temperature,
#                 timeout=timeout,  # 초
#             )
#             return (resp.choices[0].message.content or "").strip()

#         except RateLimitError as e:
#             wait_s = _parse_retry_after_seconds(str(e))
#             if wait_s is None:
#                 wait_s = (0.5 + random.random()) * min(max_backoff, base_backoff * (2 ** (attempt - 1)))
#             time.sleep(wait_s)
#             continue

#         except (APITimeoutError, APIConnectionError) as e:
#             wait_s = (0.5 + random.random()) * min(max_backoff, base_backoff * (2 ** (attempt - 1)))
#             time.sleep(wait_s)
#             continue

#         except APIError as e:
#             status = getattr(e, "status_code", None)
#             if status and 500 <= status < 600:
#                 wait_s = (0.5 + random.random()) * min(max_backoff, base_backoff * (2 ** (attempt - 1)))
#                 time.sleep(wait_s)
#                 continue
#             raise

#     raise RuntimeError("parse_object_goal_instruction: retry attempts exceeded.")



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
    max_tokens: int = 80,
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





# def parse_object_goal_instruction(messages):
#     """
#     Parse language instruction into a series of landmarks
#     Example: "first go to the kitchen and then go to the toilet" -> ["kitchen", "toilet"]
#     """

#     # openai_key = os.environ["OPENAI_KEY"]
#     # openai.api_key = openai_key
#     os.environ.get("OPENAI_API_KEY")

#     client = openai.OpenAI()
#     # response = client.chat.completions.create(
#     #     model="gpt-4o-mini",
#     #     messages=messages,
#     #     max_tokens=300,
#     # )


#     response = client.chat.completions.create(
#         model="gpt-4o-mini",
#         messages=messages,
#         max_tokens=10,
#         temperature=0,  # 형식 안정화
#     )




#     text = response.choices[0].message.content # .strip("\n")
#     return text

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
    
    # def query(self, query, json_data):
    #     prompt_json = [
    #         {"role": "system", "content": "You are a helpful assistant that selects the most appropriate item from a list of JSON objects based on the user's query."},
    #         {"role": "user", "content": f"From the following JSON list, find the item that best matches the query and return only its instance_id. If there is no suitable match, do not return anything.: '{query}'\n\n{json_data}"}
    #     ]   
    #     return prompt_json

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

        # 참조 텍스트(설명/캡션 통합)
        refs = []
        if inst_content:
            refs.extend([str(x) for x in inst_content] if isinstance(inst_content, (list, tuple)) else [str(inst_content)])
        if inst_caption:
            refs.append(str(inst_caption))
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
            prompt = f"""
                You are given an image, a target object token, and optional reference descriptions.

                Object: "{instance_category}" (use EXACTLY)
                References:
                {reference_text}

                Task:
                - If any reference clearly matches a visible {instance_category}, base the query on it; otherwise rely on the image.
                - Start with EXACTLY ONE verb from:{list_to_lines(VERB_VARIANTS)}
                - Optionally add ONE attribute clause using:{list_to_lines(ATTR_CLAUSE_VARIANTS)}
                {literal_tokens_rules}
                {common_sentence_rules}
                DO NOT mention ANY room or location phrase.

                Example:
                - <VERB> a {instance_category} {{ATTR_CLAUSE}} a dark wooden frame against the wall.
                """.strip()

        elif query_type == "mixed":
            room_phrases = allowed_room_phrases(room_category)
            prompt = f"""
                You are given an image and two tokens.

                Object: "{instance_category}" (use EXACTLY)
                Room: "{room_category}" (use EXACTLY)
                References (optional):
                {reference_text}

                Task:
                - Start with EXACTLY ONE verb from:{list_to_lines(VERB_VARIANTS)}
                - Include AT LEAST ONE concrete attribute using EXACTLY ONE connector from:{list_to_lines(ATTR_CLAUSE_VARIANTS)}
                - Include the room using EXACTLY ONE phrase from:{list_to_lines(room_phrases)}
                {literal_tokens_rules}
                {room_token_rule}
                {common_sentence_rules}

                Patterns (choose ONE verb + ONE attr connector + ONE room phrase):
                - <VERB> a light blue {instance_category} {{ATTR_CLAUSE}} plush cushions located in the {room_category}.
                - <VERB> a rectangular {instance_category} {{ATTR_CLAUSE}} a black frame inside the {room_category}.
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
    You are given an image, a hidden object token, and optional references.

    Hidden object token: "{instance_category}"  (DO NOT output this token)
    References (optional):
    {reference_text}

    Task:
    - Produce ONE affordance-based query (abstract; no object token).
    - Start with EXACTLY ONE verb from:{list_to_lines(VERB_VARIANTS)}
    - Include AT LEAST ONE concise attribute using EXACTLY ONE connector from:{list_to_lines(ATTR_CLAUSE_VARIANTS)}
    - Include the room using EXACTLY ONE phrase from:{list_to_lines(room_phrases)}
    {common_sentence_rules}
    {room_token_rule}

    Patterns (choose ONE verb + ONE attr connector + ONE room phrase):
    - <VERB> something to relax on {{ATTR_CLAUSE}} soft upholstery located in the {room_category}.
    - <VERB> somewhere to work {{ATTR_CLAUSE}} a dark wooden surface inside the {room_category}.
    """.strip()

        else:
            raise ValueError(f"Unknown query_type: {query_type}")

        prompt_json = [
            {"role": "system",
            "content": ("You are a careful vision-language assistant. "
                        "Follow the format strictly. Use ONLY one choice from each provided list. "
                        "Keep EXACT object/room tokens when required.")},
            {"role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{base64_image}", "detail": "high"}}]}
        ]
        return prompt_json





    # def make_query(self, query_type, img_dir, instance_category, room_category, inst_content=None, inst_caption=None):
    #     base64_image = encode_image_to_base64(img_dir)

    #     # 참조 텍스트 구성 (descriptive/caption 통합)
    #     refs = []
    #     if inst_content:
    #         refs.extend([str(x) for x in inst_content] if isinstance(inst_content, (list, tuple)) else [str(inst_content)])
    #     if inst_caption:
    #         refs.append(str(inst_caption))
    #     reference_text = "\n".join([f"- {r}" for r in refs]) if refs else "None."

    #     # 공통 형식 규칙: 완전한 1문장, 마침표로 종료, 불필요 요소 금지
    #     common_sentence_rules = """
    # Write EXACTLY ONE complete sentence that ENDS WITH A PERIOD.
    # No bullet points. No markdown. No explanations. No prefixes.
    # Do NOT use quotes or brackets. Avoid words like "object", "instance", or "image".
    # """.strip()

    #     # 명시적 토큰 사용(동의어/대소문자/복수형 금지)
    #     literal_tokens_rules = f"""
    # Use the object category token EXACTLY as provided: "{instance_category}" — no synonyms, no pluralization, no paraphrasing, no casing changes.
    # """.strip()

    #     # 룸 표현 다양화: 허용된 표현 중 하나만 사용
    #     def allowed_room_phrases(rc):
    #         return (
    #             f'["in the {rc}", "inside the {rc}", "within the {rc}", '
    #             f'"at the {rc}", "located in the {rc}"]'
    #         )

    #     # ================= 타입별 프롬프트 =================

    #     if query_type == "object":
    #         prompt = f"""
    # You are given an image and a target object category.

    # Target object category (use EXACTLY as given): "{instance_category}"

    # Task:
    # Generate a short, natural search query that helps a user find this target in the current image.
    # {literal_tokens_rules}
    # {common_sentence_rules}
    # Do NOT mention any room or broader scene context.

    # Good examples (keep the exact category token, adapt adjectives to what is plausibly visible):
    # - Find a sleek {instance_category} with metal legs.
    # - Find a {instance_category} with a glass top near the window.
    # """.strip()

    #     elif query_type == "room":
    #         # 룸은 반드시 사용 + 허용 표현 세트 중 하나만
    #         prompt = f"""
    # You are given an image and two target categories.

    # Target object category (use EXACTLY as given): "{instance_category}"
    # Target room category (use EXACTLY as given): "{room_category}"

    # Task:
    # Generate a search query that includes BOTH the object and the room.
    # {literal_tokens_rules}
    # {common_sentence_rules}

    # Room constraints:
    # - Use the room token EXACTLY as provided: "{room_category}" (no synonyms, no paraphrasing, no casing changes, no adjectives).
    # - Use EXACTLY ONE of the following location phrases (choose the most natural; do NOT invent new ones):
    # {allowed_room_phrases(room_category)}

    # Examples (keep exact tokens; pick ONE room phrase):
    # - Find a wooden {instance_category} inside the {room_category}.
    # - Find a {instance_category} with a marble top located in the {room_category}.
    # - Find a compact {instance_category} within the {room_category}.
    # """.strip()

    #     elif query_type == "caption":
    #         # 캡션/레퍼런스 활용 + 룸 언급 금지
    #         prompt = f"""
    # You are given an image, a target object category, and reference descriptions.

    # Target object category (use EXACTLY as given): "{instance_category}"
    # Reference Descriptions:
    # {reference_text}

    # Task:
    # 1) If any reference clearly matches a visible {instance_category}, base the query on it; otherwise rely on the image alone.
    # 2) Generate a visually grounded search query for the {instance_category}.
    # {literal_tokens_rules}
    # {common_sentence_rules}
    # DO NOT mention ANY room or location phrase.

    # Example:
    # - Find a {instance_category} with a dark wooden frame against the wall.
    # """.strip()

    #     elif query_type == "mixed":
    #         # 오브젝트+룸+디스크립티브(최소 1개 속성) + 허용된 룸 표현
    #         prompt = f"""
    # You are given an image and two target categories.

    # Target object category (use EXACTLY as given): "{instance_category}"
    # Target room category (use EXACTLY as given): "{room_category}"
    # Reference Descriptions (optional):
    # {reference_text}

    # Task:
    # 1) Describe the {instance_category} with at least ONE concrete visual attribute visible in the image OR from references
    # (e.g., color, material, design, relative position) using a "with ..." or "featuring ..." clause.
    # 2) Include the room using EXACTLY ONE allowed location phrase.
    # {literal_tokens_rules}
    # {common_sentence_rules}

    # Room constraints:
    # - Use the room token EXACTLY as provided: "{room_category}" (no synonyms, no paraphrasing, no casing changes, no adjectives).
    # - Use EXACTLY ONE of the following location phrases (choose the most natural; do NOT invent new ones):
    # {allowed_room_phrases(room_category)}

    # Examples (keep exact tokens; include ≥1 attribute; pick ONE room phrase):
    # - Find a light blue {instance_category} with plush cushions located in the {room_category}.
    # - Find a rectangular {instance_category} with a black frame inside the {room_category}.
    # """.strip()

    #     elif query_type == "abs":
    #         # 추상(affordance) + 룸 정보 금지(항상 금지)
    #         prompt = f"""
    # You are given an image and a hidden target object category.

    # Hidden object category: "{instance_category}"  (DO NOT write this token in the output)

    # Task:
    # Produce ONE abstract, affordance-based search query that matches the function of the hidden category evident in the image
    # (e.g., "Find something to sit on", "Find somewhere to sleep", "Find something to wash hands").
    # {common_sentence_rules}
    # DO NOT mention ANY room or location phrase.
    # DO NOT mention the object token "{instance_category}".

    # Examples:
    # - Find something to sit on.
    # - Find something to store clothes.
    # """.strip()

    #     elif query_type == "mixed_a":
    #         # 추상(affordance) + 룸 + 디스크립티브(최소 1개 속성 필수) + 허용된 룸 표현
    #         prompt = f"""
    # You are given an image, a hidden target object category, and optional reference descriptions.

    # Hidden object category: "{instance_category}"  (DO NOT write this token in the output)
    # Reference Descriptions (optional, use ONLY if they plausibly match the visible target):
    # {reference_text}

    # Task:
    # Produce ONE abstract, affordance-based search query that:
    # 1) NEVER mentions the token "{instance_category}".
    # 2) Includes AT LEAST ONE concise descriptive attribute consistent with the image or references
    # (e.g., color, material, shape, relative position) using a "with ..." or "featuring ..." clause.
    # 3) Includes the room using EXACTLY ONE allowed location phrase.
    # {common_sentence_rules}

    # Room constraints:
    # - Use the room token EXACTLY as provided: "{room_category}" (no synonyms, no paraphrasing, no casing changes, no adjectives).
    # - Use EXACTLY ONE of the following location phrases (choose the most natural; do NOT invent new ones):
    # {allowed_room_phrases(room_category)}

    # Affordance templates (choose ONE that best fits the category and image):
    # - "Find something to <verb> ..."
    # - "Find somewhere to <verb> ..."

    # Examples (abstract + ≥1 attribute + ONE room phrase):
    # - Find something to sit on with plush cushions located in the {room_category}.
    # - Find somewhere to work at with a dark wooden surface inside the {room_category}.
    # - Find something to store dishes with glass doors within the {room_category}.
    # """.strip()

    #     else:
    #         raise ValueError(f"Unknown query_type: {query_type}")

    #     prompt_json = [
    #         {
    #             "role": "system",
    #             "content": (
    #                 "You are a careful vision-language assistant. "
    #                 "Follow the format and constraints strictly. "
    #                 "Ensure the output is ONE COMPLETE SENTENCE that ENDS WITH A PERIOD."
    #             ),
    #         },
    #         {
    #             "role": "user",
    #             "content": [
    #                 {"type": "text", "text": prompt},
    #                 {
    #                     "type": "image_url",
    #                     "image_url": {"url": f"data:image/png;base64,{base64_image}", "detail": "high"},
    #                 },
    #             ],
    #         },
    #     ]
    #     return prompt_json




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








    
    # def make_query(self, query_type, img_dir, instance_category, room_category, inst_content=None, inst_caption=None):
    #     base64_image = encode_image_to_base64(img_dir)
        
    #     #TODO0814: 지정받은 instance category에 대해서 caption을 뽑도록 프롬프트 수정해야됨
    #     #TODO0814: Room은 지정안받아도 되는지 이야기 해봐야될듯
    #     prompt = ""
    #     reference_text = """
    #                     Ex1. The dark wood cabinet, likely walnut or mahogany, features a glossy finish, intricate carvings, and sturdy construction, suggesting high-quality craftsmanship from the late 19th or early 20th century.
    #                     Ex2. The large dark wood door, featuring an aged finish and glass panels, stands prominently in the hallway, adding character and elegance.
    #                     Ex3. TThe flat-screen TV monitor with a dark brown wooden frame is mounted on the wall, providing a modern touch to the hallway.
    #                     """
        
    #     if query_type == "object":
    #         prompt = f"""
    #                     You are given an image. Your task is to analyze the image and identify a visually prominent object.
    #                     Generate a short natural language query for a user to find this object.

    #                     The query can be:
    #                     - A specific target like "find a sofa"
    #                     - Or more abstract like "find something to sit on" or "find something to sleep"

    #                     Focus only on the object itself. Do NOT include room information or surrounding context.
    #                 """
    #     elif query_type == "room":
    #         prompt = """
    #                     You are given an image. Your task is to:
    #                     1. Identify the type of room shown in the image (e.g., bedroom, bathroom, kitchen, hallway).
    #                     2. Identify one prominent object in that room.
    #                     3. Generate a natural language search query.

    #                     Output must follow this exact format:
    #                     Find a [object] in the [room type].

    #                     Do NOT include:
    #                     - Bullet points
    #                     - Markdown formatting
    #                     - "Search query:" or any other prefixes
    #                     - Explanations or justification
    #                     - Descriptions beyond the target object and room

    #                     Good Examples:
    #                     Find a cozy bed in the bedroom.
    #                     Find a white sink in the bathroom.
    #                     Find a vintage chair in the living room.

    #                     Now generate only one sentence in that format.
    #                 """
    #     elif query_type == "caption":
    #         prompt = f"""
    #                     You are given an image and a list of object descriptions. Each description refers to a specific instance of an object in the scene.

    #                     Your task is to look at the image and, based on the visual elements you detect,
    #                     select the most visually relevant instance from the reference text,
    #                     and generate a highly detailed and specific search query describing that object.

    #                     Use natural language. Your goal is to make the query both *visually grounded* and *textually enriched*.

    #                     Reference Descriptions:
    #                     {reference_text}

    #                     Be specific, natural, and visually grounded. Do not describe the instance itself — focus on locating the target object using the instance as context.

    #                     - "Find a chair positioned next to the window covered with lace curtain."
    #                 """
    #     elif query_type == "mixed":
    #         prompt = f"""
    #                     You are given an image and a set of detailed descriptions of objects or furniture in the scene.

    #                     Your task:
    #                     1. Identify the most visually prominent object in the image.
    #                     2. Match it to the most relevant reference description.
    #                     3. Infer the type of room (e.g., bedroom, kitchen, hallway) from the image.
    #                     4. Generate a natural search query that includes:
    #                     - Object type
    #                     - Descriptive attributes (e.g., color, material, design)
    #                     - Room/location where the object is found

    #                     Important:
    #                     - Use adjectives to describe the object.
    #                     - **Do NOT use adjectives for the room type.** Keep it simple, like "in the bedroom" or "in the living room".

    #                     ### Example Output:
    #                     Find a vintage wooden bed with a colorful quilt in the bedroom.
    #                 """
            
    #     prompt_json = [
    #         {"role": "system", "content": "You are a helpful vision-language assistant."},
    #         {
    #             "role": "user",
    #             "content": [
    #                 {"type": "text", "text": prompt},
    #                 {
    #                     "type": "image_url",
    #                     "image_url": {
    #                         "url": f"data:image/png;base64,{base64_image}",
    #                         "detail": "high"
    #                     }
    #                 }
    #             ]
    #         }
    #     ]
    #     return prompt_json 
    
    # # 이미지를 받아서 query_type에 따라 적절한 쿼리를 생성하는 함수
    # def make_queries_rand_pick(self, img_dir, inst_json, pkl_file, cnt_dict): 
    #     with open(pkl_file, 'rb') as f:
    #         inst_dict = pickle.load(f)

    #     extracted_inst_list = []
    #     inst_id_list = list(inst_dict.keys())
        
    #     query_inst_dict = {}
    #     query_dict = {key: [] for key in cnt_dict}

    #     for query_type, cnt in cnt_dict.items():
    #         for _ in range(cnt):
    #             inst_info = {}
    #             # random inst_id 추출
    #             random_inst_id = random.choice(inst_id_list)
    #             inst_id_list.remove(random_inst_id)
    #             if random_inst_id in extracted_inst_list:
    #                 continue

    #             prompt_json, frame_num  = self.make_query(query_type, img_dir, category)

    #             query_dict_ = dict()
    #             query_dict_["inst_id"] = random_inst_id
    #             query_dict_["prompt"] = prompt_json
    #             query_dict[query_type].append(query_dict_) 
                
    #             inst_info["inst_id"] = random_inst_id
    #             inst_info["frame"] = frame_num
    #             inst_info["q_type"] = query_type
    #             query_inst_dict[random_inst_id] = inst_info

    #     return query_inst_dict, query_dict



