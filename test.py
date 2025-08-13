import numpy as np

# 0이 아닌 값들만 추출
inst_room_mask = np.array([[0,1,2],[0,2,2],[3,4,1]])
nonzero_values = inst_room_mask[inst_room_mask != 0]

if nonzero_values.size > 0:
    # np.bincount로 빈도 계산 후 최빈값 추출
    most_freq_value = np.bincount(nonzero_values).argmax()
else:
    most_freq_value = None  # 0 제외한 값이 없는 경우
print(most_freq_value)