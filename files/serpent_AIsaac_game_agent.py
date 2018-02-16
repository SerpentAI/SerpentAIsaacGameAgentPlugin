from serpent.game_agent import GameAgent

from serpent.frame_grabber import FrameGrabber

from serpent.input_controller import KeyboardKey

import serpent.cv

from .helpers.frame_processing import frame_to_hearts
from .helpers.terminal_printer import TerminalPrinter
from .helpers.ppo import SerpentPPO

import itertools
import collections

import time
import random
import os
import pickle

import numpy as np

import skimage.io
import skimage.filters
import skimage.morphology
import skimage.measure
import skimage.draw
import skimage.segmentation
import skimage.color

import pyperclip

from datetime import datetime


class SerpentAIsaacGameAgent(GameAgent):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.frame_handlers["PLAY"] = self.handle_play

        self.frame_handler_setups["PLAY"] = self.setup_play

        self.printer = TerminalPrinter()

    @property
    def bosses(self):
        return {
            "MONSTRO": "1010"
        }

    def setup_play(self):
        self.first_run = True

        move_inputs = {
            "MOVE UP": [KeyboardKey.KEY_W],
            "MOVE LEFT": [KeyboardKey.KEY_A],
            "MOVE DOWN": [KeyboardKey.KEY_S],
            "MOVE RIGHT": [KeyboardKey.KEY_D],
            "MOVE TOP-LEFT": [KeyboardKey.KEY_W, KeyboardKey.KEY_A],
            "MOVE TOP-RIGHT": [KeyboardKey.KEY_W, KeyboardKey.KEY_D],
            "MOVE DOWN-LEFT": [KeyboardKey.KEY_S, KeyboardKey.KEY_A],
            "MOVE DOWN-RIGHT": [KeyboardKey.KEY_S, KeyboardKey.KEY_D],
            "DON'T MOVE": []
        }

        shoot_inputs = {
            "SHOOT UP": [KeyboardKey.KEY_UP],
            "SHOOT LEFT": [KeyboardKey.KEY_LEFT],
            "SHOOT DOWN": [KeyboardKey.KEY_DOWN],
            "SHOOT RIGHT": [KeyboardKey.KEY_RIGHT],
            "DON'T SHOOT": []
        }

        self.game_inputs = dict()

        for move_label, shoot_label in itertools.product(move_inputs, shoot_inputs):
            label = f"{move_label.ljust(20)}{shoot_label}"
            self.game_inputs[label] = move_inputs[move_label] + shoot_inputs[shoot_label]

        self.run_count = 0
        self.run_reward = 0

        self.observation_count = 0

        self.performed_inputs = collections.deque(list(), maxlen=8)

        self.reward_10 = collections.deque(list(), maxlen=10)
        self.reward_100 = collections.deque(list(), maxlen=100)
        self.reward_1000 = collections.deque(list(), maxlen=1000)

        self.average_reward_10 = 0
        self.average_reward_100 = 0
        self.average_reward_1000 = 0

        self.top_reward = 0
        self.top_reward_run = 0

        self.previous_time_alive = 0

        self.time_alive_10 = collections.deque(list(), maxlen=10)
        self.time_alive_100 = collections.deque(list(), maxlen=100)
        self.time_alive_1000 = collections.deque(list(), maxlen=1000)

        self.average_time_alive_10 = 0
        self.average_time_alive_100 = 0
        self.average_time_alive_1000 = 0

        self.top_time_alive = 0
        self.top_time_alive_run = 0

        self.death_check = False
        self.just_relaunched = False

        self.frame_buffer = None

        self.ppo_agent = SerpentPPO(
            frame_shape=(100, 100, 4),
            game_inputs=self.game_inputs
        )

        try:
            self.ppo_agent.agent.restore_model(directory=os.path.join(os.getcwd(), "datasets", "aisaac"))
            self.restore_metadata()
        except Exception:
            pass

        # Warm Agent?
        game_frame_buffer = FrameGrabber.get_frames([0, 2, 4, 6], frame_type="PIPELINE")
        self.ppo_agent.generate_action(game_frame_buffer)

        self.health = collections.deque(np.full((16,), 24), maxlen=16)
        self.boss_health = collections.deque(np.full((16,), 654), maxlen=16)


        self.boss_skull_image = None

        self.started_at = datetime.utcnow().isoformat()
        self.run_timestamp = None

    def handle_play(self, game_frame):
        if self.first_run:
            self._goto_boss(boss_key=self.bosses["MONSTRO"], items=["c330", "c92", "c92", "c92"])
            self.first_run = False

            self.run_count += 1
            self.run_timestamp = time.time()

            return None

        self.printer.add("")
        self.printer.add("Serpent.AI Lab - AIsaac")
        self.printer.add("Reinforcement Learning: Training a PPO Agent")
        self.printer.add("")
        self.printer.add(f"Stage Started At: {self.started_at}")
        self.printer.add(f"Current Run: #{self.run_count}")
        self.printer.add("")

        hearts = frame_to_hearts(game_frame, self.game)

        # Check for Curse of Unknown
        if not len(hearts):
            self.input_controller.tap_key(KeyboardKey.KEY_R, duration=1.5)
            self._goto_boss(boss_key=self.bosses["MONSTRO"], items=["c330", "c92", "c92", "c92"])

            return None

        self.health.appendleft(24 - hearts.count(None))
        self.boss_health.appendleft(self._get_boss_health(game_frame))

        reward, is_alive = self.reward_aisaac([None, None, game_frame, None])
        self.run_reward += reward

        self.printer.add(f"Current Reward: {round(reward, 2)}")
        self.printer.add(f"Run Reward: {round(self.run_reward, 2)}")
        self.printer.add("")

        if self.frame_buffer is not None:
            self.observation_count += 1

            if self.ppo_agent.agent.batch_count == 2047:
                self.printer.flush()
                self.printer.add("Updating AIsaac Model With New Data... ")
                self.printer.flush()

                self.input_controller.tap_key(KeyboardKey.KEY_ESCAPE)
                self.ppo_agent.observe(reward, terminal=(not is_alive or self._is_boss_dead(game_frame)))
                self.input_controller.tap_key(KeyboardKey.KEY_ESCAPE)

                self.frame_buffer = None

                return None
            else:
                self.ppo_agent.observe(reward, terminal=(not is_alive or self._is_boss_dead(game_frame)))

        self.printer.add(f"Observation Count: {self.observation_count}")
        self.printer.add(f"Current Batch Size: {self.ppo_agent.agent.batch_count}")
        self.printer.add("")

        if is_alive:
            if random.random() <= 0.05:
                self.printer.add("Randomness strikes!")
                self.printer.flush()

                for i in range(3):
                    random_game_input = self.game_inputs[random.choice(list(self.game_inputs))]
                    self.input_controller.handle_keys(random_game_input)

                    time.sleep(0.125)

                self.frame_buffer = None

                return None

            self.death_check = False

            self.printer.add(f"Average Rewards (Last 10 Runs): {round(self.average_reward_10, 2)}")
            self.printer.add(f"Average Rewards (Last 100 Runs): {round(self.average_reward_100, 2)}")
            self.printer.add(f"Average Rewards (Last 1000 Runs): {round(self.average_reward_1000, 2)}")
            self.printer.add("")
            self.printer.add(f"Top Run Reward: {round(self.top_reward, 2)} (Run #{self.top_reward_run})")
            self.printer.add("")
            self.printer.add(f"Previous Run Time Alive: {round(self.previous_time_alive, 2)}")
            self.printer.add("")
            self.printer.add(f"Average Time Alive (Last 10 Runs): {round(self.average_time_alive_10, 2)}")
            self.printer.add(f"Average Time Alive (Last 100 Runs): {round(self.average_time_alive_100, 2)}")
            self.printer.add(f"Average Time Alive (Last 1000 Runs): {round(self.average_time_alive_1000, 2)}")
            self.printer.add("")
            self.printer.add(f"Top Time Alive: {round(self.top_time_alive, 2)} (Run #{self.top_time_alive_run})")
            self.printer.add("")
            self.printer.add("Latest Inputs:")
            self.printer.add("")

            for i in self.performed_inputs:
                self.printer.add(i)

            self.printer.flush()

            self.frame_buffer = FrameGrabber.get_frames([0, 2, 4, 6], frame_type="PIPELINE")

            action, label, game_input = self.ppo_agent.generate_action(self.frame_buffer)

            self.performed_inputs.appendleft(label)
            self.input_controller.handle_keys(game_input)
        else:
            self.input_controller.handle_keys([])

            if not self.death_check:
                self.death_check = True

                self.printer.flush()
                return None
            else:
                self.printer.flush()
                self.run_count += 1

                self.reward_10.appendleft(self.run_reward)
                self.reward_100.appendleft(self.run_reward)
                self.reward_1000.appendleft(self.run_reward)

                self.average_reward_10 = float(np.mean(self.reward_10))
                self.average_reward_100 = float(np.mean(self.reward_100))
                self.average_reward_1000 = float(np.mean(self.reward_1000))

                if self.run_reward > self.top_reward:
                    self.top_reward = self.run_reward
                    self.top_reward_run = self.run_count - 1

                self.previous_time_alive = time.time() - self.run_timestamp

                self.run_reward = 0

                self.time_alive_10.appendleft(self.previous_time_alive)
                self.time_alive_100.appendleft(self.previous_time_alive)
                self.time_alive_1000.appendleft(self.previous_time_alive)

                self.average_time_alive_10 = float(np.mean(self.time_alive_10))
                self.average_time_alive_100 = float(np.mean(self.time_alive_100))
                self.average_time_alive_1000 = float(np.mean(self.time_alive_1000))

                if self.previous_time_alive > self.top_time_alive:
                    self.top_time_alive = self.previous_time_alive
                    self.top_time_alive_run = self.run_count - 1

                if not self.run_count % 10:
                    self.ppo_agent.agent.save_model(directory=os.path.join(os.getcwd(), "datasets", "aisaac", "ppo_model"), append_timestep=False)
                    self.dump_metadata()

                self.health = collections.deque(np.full((16,), 24), maxlen=16)
                self.boss_health = collections.deque(np.full((16,), 654), maxlen=16)

                self.performed_inputs.clear()

                self.frame_buffer = None

                self.input_controller.tap_key(KeyboardKey.KEY_R, duration=1.5)
                self._goto_boss(boss_key=self.bosses["MONSTRO"], items=["c330", "c92", "c92", "c92"])

                self.run_timestamp = time.time()

    def reward_aisaac(self, frames, **kwargs):
        damage_multiplier = 1 / len(set(self.health))

        reward = 0
        is_alive = self.health[0] + self.health[1]

        if is_alive:
            reward += 0.5

            if self.health[0] < self.health[1]:
                factor = self.health[1] - self.health[0]
                reward -= factor * 0.25

                return reward, is_alive

        if self.boss_health[0] < self.boss_health[1]:
            reward += (0.5 * damage_multiplier)

        return reward, is_alive

    def dump_metadata(self):
        metadata = dict(
            started_at=self.started_at,
            run_count=self.run_count - 1,
            observation_count=self.observation_count,
            reward_10=self.reward_10,
            reward_100=self.reward_100,
            reward_1000=self.reward_1000,
            average_reward_10=self.average_reward_10,
            average_reward_100=self.average_reward_100,
            average_reward_1000=self.average_reward_1000,
            top_reward=self.top_reward,
            top_reward_run=self.top_reward_run,
            time_alive_10=self.time_alive_10,
            time_alive_100=self.time_alive_100,
            time_alive_1000=self.time_alive_1000,
            average_time_alive_10=self.average_time_alive_10,
            average_time_alive_100=self.average_time_alive_100,
            average_time_alive_1000=self.average_time_alive_1000,
            top_time_alive=self.top_time_alive,
            top_time_alive_run=self.top_time_alive_run
        )

        with open("datasets/aisaac/metadata.json", "wb") as f:
            f.write(pickle.dumps(metadata))

    def restore_metadata(self):
        with open("datasets/aisaac/metadata.json", "rb") as f:
            metadata = pickle.loads(f.read())

        self.started_at = metadata["started_at"]
        self.run_count = metadata["run_count"]
        self.observation_count = metadata["observation_count"]
        self.reward_10 = metadata["reward_10"]
        self.reward_100 = metadata["reward_100"]
        self.reward_1000 = metadata["reward_1000"]
        self.average_reward_10 = metadata["average_reward_10"]
        self.average_reward_100 = metadata["average_reward_100"]
        self.average_reward_1000 = metadata["average_reward_1000"]
        self.top_reward = metadata["top_reward"]
        self.top_reward_run = metadata["top_reward_run"]
        self.time_alive_10 = metadata["time_alive_10"]
        self.time_alive_100 = metadata["time_alive_100"]
        self.time_alive_1000 = metadata["time_alive_1000"]
        self.average_time_alive_10 = metadata["average_time_alive_10"]
        self.average_time_alive_100 = metadata["average_time_alive_100"]
        self.average_time_alive_1000 = metadata["average_time_alive_1000"]
        self.top_time_alive = metadata["top_time_alive"]
        self.top_time_alive_run = metadata["top_time_alive_run"]

    def _goto_boss(self, boss_key="1010", items=None):
        self.input_controller.tap_key(KeyboardKey.KEY_SPACE)
        time.sleep(1)
        self.input_controller.tap_key(KeyboardKey.KEY_GRAVE)
        time.sleep(0.5)

        if items is not None:
            for item in items:
                pyperclip.copy(f"giveitem {item}")
                self.input_controller.tap_keys([KeyboardKey.KEY_LEFT_CTRL, KeyboardKey.KEY_V], duration=0.1)
                time.sleep(0.1)
                self.input_controller.tap_key(KeyboardKey.KEY_ENTER)
                time.sleep(0.1)

        pyperclip.copy(f"goto s.boss.{boss_key}")
        self.input_controller.tap_keys([KeyboardKey.KEY_LEFT_CTRL, KeyboardKey.KEY_V], duration=0.1)
        time.sleep(0.1)

        self.input_controller.tap_key(KeyboardKey.KEY_ENTER)
        time.sleep(0.1)
        self.input_controller.tap_key(KeyboardKey.KEY_ENTER)
        time.sleep(0.5)
        self.input_controller.tap_key(KeyboardKey.KEY_ENTER)
        time.sleep(0.5)
        self.input_controller.tap_key(KeyboardKey.KEY_ENTER)
        time.sleep(0.2)

    def _get_boss_health(self, game_frame):
        gray_boss_health_bar = serpent.cv.extract_region_from_image(
            game_frame.grayscale_frame,
            self.game.screen_regions["HUD_BOSS_HP"]
        )

        try:
            threshold = skimage.filters.threshold_otsu(gray_boss_health_bar)
        except ValueError:
            threshold = 1

        bw_boss_health_bar = gray_boss_health_bar > threshold

        return bw_boss_health_bar[bw_boss_health_bar > 0].size

    def _is_boss_dead(self, game_frame):
        gray_boss_skull = serpent.cv.extract_region_from_image(
            game_frame.grayscale_frame,
            self.game.screen_regions["HUD_BOSS_SKULL"]
        )

        if self.boss_skull_image is None:
            self.boss_skull_image = gray_boss_skull

        is_dead = False

        if skimage.measure.compare_ssim(gray_boss_skull, self.boss_skull_image) < 0.5:
            is_dead = True

        self.boss_skull_image = gray_boss_skull

        return is_dead
