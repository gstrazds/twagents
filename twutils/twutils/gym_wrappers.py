import gym
# from typing import List, Dict, Optional, Any
# import numpy as np

class ScoreToRewardWrapper(gym.RewardWrapper):
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


class ConsistentFeedbackWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)

    # def reset(self, **kwargs):
    #     observation, infos = self.env.reset(**kwargs)
    #     return observation, infos
    def reset(self, **kwargs):
        observation, infos = self.env.reset(**kwargs)
        for idx, obs in enumerate(observation):
            if obs == infos['description'][idx] and obs == infos['feedback'][idx]:
                print("MODIFYING infos['feedback'] : 'You look around' <-- orig:", infos['feedback'][idx])
                infos['feedback']['idx'] = 'You look around'
        return observation, infos

    def step(self, action):
        observation, reward, done, infos = self.env.step(action)
        #print(f"ConsistentInfoWrapper: {len(infos['facts'])} {len(observation)} {len(observation[0])}")
        assert isinstance(observation, (list, tuple))   # use with vector envs (TW gym wrapper produces such by default)
        assert 'feedback' in infos, f"infos should include feedback {infos.keys()}"
        assert 'description' in infos, f"infos should include description {infos.keys()}"
        for idx, obs in enumerate(observation):
            obs = obs.strip()
            new_feedback = None
            if obs != infos['description'][idx].strip():
                #print("ConsistentFeedbackWrapper: obs != description")
                #print(f"<<{obs}>>")
                #print(f">>{infos['description']}<<")
                pass
            elif obs != infos['feedback'][idx].strip():
                #print("ConsistentFeedbackWrapper: obs != feedback")
                # print(f"<<{obs}>>")
                # print(f">>{infos['feedback']}<<")
                pass
            else:
                act = action[idx]
                if act.startswith("go "):
                    new_feedback = f'You {act} and look around'
                elif act in ['east', 'west', 'north', 'south']:
                    new_feedback = f'You go {act} and look around'
                elif act.startswith('examine') or act.startswith('look at') or act.startswith('read'):
                    new_feedback = f'You {act}'
                elif act.startswith("open "):
                    new_feedback = f'You {act}'
                else:
                    #print(f"ConsistentFeedbackWrapper ignoring act=|{act}|")
                    pass
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

