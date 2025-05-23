import numpy as np
from pathlib import Path
import cv2
import hydra
from utils.utils import load_config
from map.navigation.robot.habitat_lang_robot import HabitatLanguageRobot
from map.utils.llm_utils import parse_object_goal_instruction
from map.utils.matterport3d_categories import mp3dcat
from map.utils.mapping_utils import (
    cvt_pose_vec2tf,
)

def main() -> None:
    config = load_config('config/navigation.yaml')
    robot = HabitatLanguageRobot(config)
    robot.setup_scene(config["scene_id"])
    instr = "Go to the red counter."
    object_categories = parse_object_goal_instruction(instr)
    print(f"instruction: {instr}")
    robot.empty_recorded_actions()
    init_pose = np.array([[1.0, 0.0, 0.0, 8.37611164209252],
                            [0.0, 1.0, 0.0, -1.2843499183654785],
                            [0.0, 0.0, 1.0, 7.724077207991911],
                            [0.0, 0.0, 0.0, 1.0]])
    robot.set_agent_state(init_pose) #! np.arrray()
    for cat_i, cat in enumerate(object_categories):
        print(f"Navigating to category {cat}")
        actions_list = robot.move_to_object(300)
    recorded_actions_list = robot.get_recorded_actions()
    robot.set_agent_state(robot.get_agent_tf())
    for action in recorded_actions_list:
        robot.test_step(robot.sim, action, vis=config.nav.vis)
    # robot.map.init_categories(mp3dcat[1:-1])

if __name__ == "__main__":
    main()