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

        # Warm Agent?
        game_frame_buffer = FrameGrabber.get_frames([0, 2, 4, 6], frame_type="PIPELINE")
        self.ppo_agent.generate_action(game_frame_buffer)

        self.health = collections.deque(np.full((8,), 6), maxlen=8)
        self.boss_health = collections.deque(np.full((8,), 654), maxlen=8)

        self.boss_skull_image = None

        self.started_at = datetime.utcnow().isoformat()
        self.run_timestamp = None

    def handle_play(self, game_frame):
        if self.first_run:
            self._goto_boss(boss_key=self.bosses["MONSTRO"], items=["c330"])
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
            self._goto_boss(boss_key=self.bosses["MONSTRO"], items=["c330"])

            return None

        self.health.appendleft(24 - hearts.count(None))
        self.boss_health.appendleft(self._get_boss_health(game_frame))

        reward, is_alive = self.reward_aisaac([None, None, game_frame, None])
        self.run_reward += reward

        self.printer.add(f"Current Reward: {reward}")
        self.printer.add(f"Run Reward: {self.run_reward}")
        self.printer.add("")

        if self.frame_buffer is not None:
            self.ppo_agent.observe(reward, terminal=(not is_alive or self._is_boss_dead(game_frame)))
            self.observation_count += 1

        self.printer.add(f"Observation Count: {self.observation_count}")
        self.printer.add("")

        if is_alive:
            self.death_check = False

            self.printer.add(f"Average Rewards (Last 10 Runs): {self.average_reward_10}")
            self.printer.add(f"Average Rewards (Last 100 Runs): {self.average_reward_100}")
            self.printer.add(f"Average Rewards (Last 1000 Runs): {self.average_reward_1000}")
            self.printer.add("")
            self.printer.add(f"Top Run Reward: {self.top_reward} (Run #{self.top_reward_run})")
            self.printer.add("")
            self.printer.add(f"Previous Run Time Alive: {self.previous_time_alive}")
            self.printer.add("")
            self.printer.add(f"Average Time Alive (Last 10 Runs): {self.average_time_alive_10}")
            self.printer.add(f"Average Time Alive (Last 100 Runs): {self.average_time_alive_100}")
            self.printer.add(f"Average Time Alive (Last 1000 Runs): {self.average_time_alive_1000}")
            self.printer.add("")
            self.printer.add(f"Top Time Alive: {self.top_time_alive} (Run #{self.top_time_alive_run})")
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

                self.health = collections.deque(np.full((8,), 6), maxlen=8)
                self.boss_health = collections.deque(np.full((8,), 654), maxlen=8)

                self.performed_inputs.clear()

                self.frame_buffer = None

                self.input_controller.tap_key(KeyboardKey.KEY_R, duration=1.5)
                self._goto_boss(boss_key=self.bosses["MONSTRO"], items=["c330"])

                self.run_timestamp = time.time()

    def reward_aisaac(self, frames, **kwargs):
        reward = 0
        is_alive = self.health[0] + self.health[1]

        if is_alive:
            reward += 0.05

            if self.health[0] < self.health[1]:
                factor = self.health[1] - self.health[0]
                reward -= 0.025 * factor

        if self.boss_health[0] < self.boss_health[1]:
            reward += 0.95

        return reward, is_alive

    def _goto_boss(self, boss_key="1010", items=None):
        self.input_controller.tap_key(KeyboardKey.KEY_SPACE)
        time.sleep(1)
        self.input_controller.tap_key(KeyboardKey.KEY_GRAVE)
        time.sleep(0.5)

        if items is not None:
            for item in items:
                pyperclip.copy(f"giveitem {item}")
                self.input_controller.tap_keys([KeyboardKey.KEY_LEFT_CTRL, KeyboardKey.KEY_V])
                self.input_controller.tap_key(KeyboardKey.KEY_ENTER)
                time.sleep(0.1)

        pyperclip.copy(f"goto s.boss.{boss_key}")
        self.input_controller.tap_keys([KeyboardKey.KEY_LEFT_CTRL, KeyboardKey.KEY_V])

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
