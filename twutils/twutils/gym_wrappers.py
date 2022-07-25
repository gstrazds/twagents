import gym
# from typing import List, Dict, Optional, Any
# import numpy as np
from twutils.twlogic import parse_ftwc_recipe

class ScoreToRewardWrapper(gym.RewardWrapper):
    """ Converts returned cumulative score into per-step incrmenttal reward.
    Compatible only with vector envs (TW gym env wrapper produces such by default)
    """
    def __init__(self, env):
        super().__init__(env)
        self._prev_score = []

    def reset(self, **kwargs):
        obs, infos = self.env.reset(**kwargs)
        assert isinstance(obs, (list, tuple))   # for use only with vector envs (TW gym env wrapper produces such by default)
        self._prev_score = [0] * len(obs)
        if 'game_score' not in infos:
            infos['game_score'] = self._prev_score
        return obs, infos

    # gym.RewardWrapper
    # def step(self, action):
    #     observation, score, done, info = self.env.step(action)
    #     return observation, self.reward(score), done, info
    def step(self, action):
        observation, score, done, infos = self.env.step(action)
        #if 'game_score' in infos:
        #     assert infos['game_score'] == score, f"{infos['game_score']} should== {score}" #FAILS: infos from prev step
        # else:
        infos['game_score'] = score
        return observation, self.reward(score), done, infos

    def reward(self, score):
        assert isinstance(score, (list, tuple))
        assert len(score) == len(self._prev_score)
        _reward = []
        for _i in range(len(score)):
            _reward.append(score[_i] - self._prev_score[_i])
            self._prev_score[_i] = score[_i]
        return tuple(_reward)


def normalize_feedback_vs_obs_description(act:str, obs:str, feedback:str, description:str):
    obs = obs.strip()
    new_feedback = None
    if act == None or act == 'start':   # when resetting the game
        if obs == description and obs == feedback:
            new_feedback = 'You look around'
        elif feedback:
            clean_feedback = feedback.strip()
            clean_descr = description.strip()
            # print("clean_feedback", clean_feedback)
            # print("clean_descr", clean_descr)
            if clean_feedback.endswith(clean_descr):
                new_feedback = clean_feedback[0:-len(clean_descr)]  # chop off the redundant tail end

    elif obs != description.strip():
        # print("ConsistentFeedbackWrapper: obs != description")
        # print(f"<<{obs}>>")
        # print(f">>{infos['description']}<<")
        pass
    elif obs != feedback.strip():
        # print("ConsistentFeedbackWrapper: obs != feedback")
        # print(f"<<{obs}>>")
        # print(f">>{infos['feedback']}<<")
        pass
    else:
        if act.startswith("go "):
            new_feedback = f'You {act} and look around'
        elif act in ['east', 'west', 'north', 'south']:
            new_feedback = f'You go {act} and look around'
        elif act.startswith('examine') or act.startswith('look at') or act.startswith('read'):
            new_feedback = f'You {act}'
        elif act.startswith("open "):
            new_feedback = f'You {act}'
        else:
            # print(f"ConsistentFeedbackWrapper ignoring act=|{act}|")
            pass
    return new_feedback


INSTRUCTIONS_TOKEN = "---------"

def simplify_feedback(feedback_str: str):
    if not feedback_str:
        return ''
    feedback_str = feedback_str.strip()
    if "cook a delicious meal" in feedback_str and "cookbook in the kitchen" in feedback_str:
        feedback_str = f"{INSTRUCTIONS_TOKEN} Do : find kitchen , read cookbook , eat meal ;"
    elif feedback_str.endswith(" and look around"):  # this is a preprocessed feedback msg from QaitGymEnvWrapper
        feedback_str = feedback_str[:-16]+"."   # remove " and look around" (because it's redundant)
    elif "all following ingredients and follow the directions to prepare" in feedback_str:
        ingredients, directions = parse_ftwc_recipe(feedback_str)
        if ingredients or directions:
            feedback_str = "You read the recipe"
            if ingredients:
                feedback_str += f" {INSTRUCTIONS_TOKEN} Acquire : " + " , ".join(ingredients) + " ;"
            if directions:
                feedback_str += f" {INSTRUCTIONS_TOKEN} Do : " + " , ".join(directions) + " ;"
    elif "our score has" in feedback_str:  # strip out useless lines about 'score has gone up by one point"
        feedback_lines = feedback_str.split('\n')
        output_lines = []
        for line in feedback_lines:
            line = line.strip()
            if "score has " in line:
                continue
            if "ou scored " in line:  # You scored x out of a possible y ...
                continue
            if "*** the end ***" in line.lower():
                continue
            if "would you like to quit" in line.lower():
                continue
            if "would you like to restart" in line.lower():
                continue
            if line:
                if line.endswith(" Not bad."):  # You eat the meal. Not bad.
                    line = line[0:-9]
                elif "dding the meal to your " in line:  # Adding the meal to your inventory.
                    line = "You prepare the meal."
                output_lines.append(line)
        feedback_str = "\n".join(output_lines)
    return feedback_str


class ConsistentFeedbackWrapper(gym.Wrapper):
    """ Simplifies/normalizes the strings returned in infos['feedback'].
    Compatible only with vector envs (TW gym env wrapper produces such by default)
    """

    def __init__(self, env):
        super().__init__(env)

    # def reset(self, **kwargs):
    #     observation, infos = self.env.reset(**kwargs)
    #     return observation, infos
    def reset(self, **kwargs):
        observation, infos = self.env.reset(**kwargs)
        for idx, obs in enumerate(observation):
            new_feedback = normalize_feedback_vs_obs_description(None,
                    obs, infos['feedback'][idx], infos['description'][idx])
        if new_feedback:
            print(f"MODIFYING infos['feedback'] : '{new_feedback}' <-- orig: {infos['feedback'][idx]}")
            infos['feedback'][idx] = new_feedback
        return observation, infos

    def step(self, action):
        observation, reward, done, infos = self.env.step(action)
        #print(f"ConsistentInfoWrapper: {len(infos['facts'])} {len(observation)} {len(observation[0])}")
        assert isinstance(observation, (list, tuple))   # use with vector envs (TW gym wrapper produces such by default)
        assert 'feedback' in infos, f"infos should include feedback {infos.keys()}"
        assert 'description' in infos, f"infos should include description {infos.keys()}"
        for idx, obs in enumerate(observation):
            new_feedback = normalize_feedback_vs_obs_description(action[idx],
                    obs, infos['feedback'][idx], infos['description'][idx])
            if new_feedback:
                print(f"ConsistenFeedbackWrapper MODIFYING infos['feedback'] : '{new_feedback}' <-- orig:", infos['feedback'][idx])
                infos['feedback'][idx] = new_feedback
            else:
                pass
                #print(f"NOT MODIFYING infos['feedback'] :\n"
                #      f" ----- observation: {obs}\n"
                #      f" ----- infos[feedback]: {infos['feedback'][idx]}\n"
                #      f" ----- infos[description] {infos['description'][idx]}")
        return observation, reward, done, infos
