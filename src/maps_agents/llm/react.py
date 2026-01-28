import requests
import json
import os
import re
import time
import random
from collections import deque
from typing import Callable, List, Dict, Any, Optional, Tuple

import numpy as np
import yaml
from requests.exceptions import (
    ChunkedEncodingError,
    ConnectionError,
    HTTPError,
    ReadTimeout,
    RequestException,
)

from maps_agents.eval import AbstractAgent
from maps_agents.eval.state_interface import MapPydanticGameResponse
from map_py.shared_constants import GAMEPLAY_RULES, SANDBOX_GAMEPLAY_RULES, SANDBOX_INSTR_WITH_DOC_REF
from map_py.observations_and_actions.shared_constants import ACTIONS_BY_DIFFICULTY
from maps_agents.eval.resource_interface import Resource, ResourceCost
from maps_agents.eval.utils import EvalConfig

# Shared session for connection pooling
_SESSION = requests.Session()


def call_llm_openrouter(
    messages,
    model: str,
    api_key: str,
    temperature: float = 0.2,
    timeout: Tuple[float, float] = (10.0, 180.0),  # (connect, read)
    max_retries: int = 4,
    backoff_base_s: float = 0.8,
) -> Tuple[str, ResourceCost]:
    """
    Call OpenRouter's chat/completions endpoint with retry logic.
    `messages` is a list of dicts: [{"role": "system"|"user"|"assistant", "content": "..."}]

    Args:
        messages: List of message dicts
        model: OpenRouter model name
        api_key: OpenRouter API key
        temperature: Sampling temperature
        timeout: Tuple of (connect_timeout, read_timeout) in seconds
        max_retries: Maximum number of retry attempts
        backoff_base_s: Base backoff time in seconds for exponential backoff

    Returns:
        Tuple of (content: str, token_usage: ResourceCost)
    """
    url = "https://openrouter.ai/api/v1/chat/completions"

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
    }

    last_exc: Optional[BaseException] = None

    for attempt in range(1, max_retries + 1):
        try:
            resp = _SESSION.post(
                url,
                headers=headers,
                json=payload,
                timeout=timeout,
            )

            # Check for server errors (5xx) which are retryable
            if resp.status_code >= 500:
                raise HTTPError(
                    f"Server error {resp.status_code}: {resp.text[:500]}",
                    response=resp
                )

            # Raise for other non-2xx errors (4xx are not retryable)
            resp.raise_for_status()

            # Parse JSON defensively
            try:
                data = resp.json()
            except json.JSONDecodeError as e:
                text_snippet = resp.text[:800] if resp.text else ""
                raise ValueError(
                    f"OpenRouter returned non-JSON response. "
                    f"status={resp.status_code} snippet={text_snippet}"
                ) from e

            # Defensive extraction
            choices = data.get("choices")
            if not choices:
                raise ValueError(f"Missing choices in response: keys={list(data.keys())}")

            message = choices[0].get("message", {})
            content = message.get("content")
            if content is None:
                raise ValueError(f"Missing message.content in response: {message}")

            usage = data.get("usage", {}) or {}
            token_usage = ResourceCost(model=model, costs={
                Resource.TOKENS_IN: int(usage.get("prompt_tokens", 0) or 0),
                Resource.TOKENS_OUT: int(usage.get("completion_tokens", 0) or 0),
            })

            return content, token_usage

        except (ChunkedEncodingError, ConnectionError, ReadTimeout) as e:
            # Retryable network failures
            last_exc = e
            if attempt == max_retries:
                break

            # Exponential backoff with jitter
            sleep_s = backoff_base_s * (2 ** (attempt - 1))
            sleep_s *= (0.8 + 0.4 * random.random())  # jitter in [0.8, 1.2]
            print(f"Retry {attempt}/{max_retries} after {type(e).__name__}, sleeping {sleep_s:.1f}s")
            time.sleep(sleep_s)

        except HTTPError as e:
            # Only retry 5xx errors
            if hasattr(e, 'response') and e.response is not None and e.response.status_code >= 500:
                last_exc = e
                if attempt == max_retries:
                    break

                sleep_s = backoff_base_s * (2 ** (attempt - 1))
                sleep_s *= (0.8 + 0.4 * random.random())
                print(f"Retry {attempt}/{max_retries} after HTTP {e.response.status_code}, sleeping {sleep_s:.1f}s")
                time.sleep(sleep_s)
            else:
                # 4xx errors are not retryable
                raise

        except RequestException as e:
            # Other requests exceptions - fail fast
            raise

    # Exhausted retries
    assert last_exc is not None
    raise last_exc

class ReactAgent(AbstractAgent):
    def __init__(
        self,
        *,
        horizon: int,
        difficulty: str,
        config_path: Optional[str | os.PathLike] = None,
        learnings_path: Optional[str | os.PathLike] = None
    ):
        """
        ReAct-style agent for MAPs using OpenRouter.

        Args:
            horizon: planning horizon (days)
            difficulty: difficulty string (e.g., "easy", "normal", "hard")
            config_path: path to config file
            learnings_path: path to save learnings
        """
        self.api_key = os.getenv("OPENROUTER_API_KEY")
        self.horizon = horizon
        self.difficulty = difficulty
        self.message_history: List[str] = []
        if config_path is None:
            config_path = "src/maps_agents/llm/react_config.yaml"
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        self.config = config
        self.model = config['llm_model']
        self.temperature = config['temperature']
        self.max_history_length = config['max_history_length']
        self.use_placement_heuristic = config.get('use_placement_heuristic', False)

        if (config['generate_learnings'] or config['use_learnings']):
            self.learnings_path = learnings_path or os.path.join("src/maps_agents/llm/learnings", f"{self.model}_{self.difficulty}.txt")
            os.makedirs(os.path.dirname(self.learnings_path), exist_ok=True)

        if config['generate_learnings']:
            with open(self.learnings_path, "w") as f:
                f.write("")
            self.learnings = []     
        elif config['use_learnings']:
            with open(self.learnings_path, "r") as f:
                self.learnings = f.readlines()

        ResourceCost.register_custom_model("anthropic/claude-4.5-sonnet", 3, 15)
        ResourceCost.register_custom_model("qwen/qwen3-vl-235b-a22b-instruct", 0.2, 1.2)

        self.resource_cost = ResourceCost(model=self.model)
        self.prev_action = "N/A: No previous action."

        actions_list = ", ".join(ACTIONS_BY_DIFFICULTY[self.difficulty])
        self.system_prompt = self.config['system_prompt']
        self.system_prompt = self.system_prompt.replace("{DOCS}", GAMEPLAY_RULES)
        self.system_prompt = self.system_prompt.replace("{SANDBOX_INSTR_WITH_DOC_REF}", SANDBOX_INSTR_WITH_DOC_REF)
        self.system_prompt = self.system_prompt.replace("{SANDBOX_DOCS}", SANDBOX_GAMEPLAY_RULES)
        self.system_prompt = self.system_prompt.replace("{horizon}", str(self.horizon))
        self.system_prompt = self.system_prompt.replace("{difficulty}", self.difficulty)
        self.system_prompt = self.system_prompt.replace("{actions_list}", actions_list)
        self.system_prompt_msg = {"role": "system", "content": self.system_prompt}

        self.dialogue_template = self.config['dialogue'][0]['content']

    @staticmethod
    def get_agent(agent_config_path: Optional[str | os.PathLike], eval_config: EvalConfig) -> 'ReactAgent':
        return ReactAgent(
            horizon=eval_config.horizon,
            difficulty=eval_config.difficulty,
            config_path=agent_config_path,
        )

    @property
    def observation_type(self) -> str:
        return 'pydantic'

    def get_action_resource_usage(self, reset: bool = True) -> ResourceCost:
        resource_cost = self.resource_cost
        if reset:
            self.resource_cost = ResourceCost(model=self.model)
        return resource_cost

    @staticmethod
    def extract_only_action_name(action : str):
        try:
            parenthesis = action.index("(") 
            action = action[:parenthesis].strip()
        except:
            pass
        return action

    # gpt-5-nano likes produces lists prefixed with "-"
    @staticmethod
    def parse_react_output(llm_output: str, tag="") -> str:
        try:
            if isinstance(llm_output, list) and len(llm_output) > 0:
                llm_output = llm_output[0]
            lines = llm_output.split('\n')
            action_kw = f'Action{tag}: '
            action = [ReactAgent.extract_only_action_name(line[len(action_kw):].strip(' -')) for line in lines if line.startswith(action_kw)]
            if len(action) != 1:
                print(f'Warning! Wrong number of actions? {action}') 
            action = action[0]
            arg_kw = f'Action Input{tag}: '
            args = [line[len(arg_kw):].strip(' -') for line in lines if line.startswith(arg_kw)]
            if len(args) != 1:
                print(f'Warning! Wrong number of proposed args? {args}')
            args = args[0] if args else ''
            if '#' in args:
                args = args[:args.index('#')]  # Strip out comment if present

            args = args.strip()
            # If it's been put inside of a list of tuple, then strip. Nano sometimes gets stuck doing this.
            if (args.startswith('(') and args.endswith(')')) or \
                (args.startswith('[') and args.endswith(']')):
                args = args[1:-1]

            return f'{action}({args})'
        except:
            return None

    @staticmethod
    def parse_react_multi_output(llm_output: str, max_proposals: int) -> list[str]:
        actions: List[str] = []
        for i in range(max_proposals):
            action = ReactAgent.parse_react_output(llm_output, tag = str(i+1))
            if action is not None:
                actions.append(action)
        return actions

    @staticmethod
    def extract_learnings(llm_output: str) -> str:
        LEARNINGS_HEADER = "Learnings Summary: "
        LEARNINGS_FOOTER = "Thought:"
        return llm_output[llm_output.rfind(LEARNINGS_HEADER) + len(LEARNINGS_HEADER):llm_output.rfind(LEARNINGS_FOOTER)].strip()

    @staticmethod
    def get_neighbors(pos: Tuple[int, int], grid_size: int = 20) -> List[Tuple[int, int]]:
        """Get valid neighboring positions (4-connected)."""
        x, y = pos
        neighbors = []
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < grid_size and 0 <= ny < grid_size:
                neighbors.append((nx, ny))
        return neighbors

    @staticmethod
    def get_place_attraction_xy(entity_type: str, state: Dict[str, Any]) -> Tuple[int, int]:
        """
        Compute optimal x, y coordinates for placing a ride or shop using BFS heuristic.

        Args:
            entity_type: 'ride' or 'shop'
            state: Park state dictionary from obs.model_dump() (pydantic format)

        Returns:
            Tuple of (x, y) coordinates for placement
        """
        TOP = 5  # Top-N positions to consider

        # Build grid: 0=empty, 1=path, 2=water, 3=ride, 4=shop, 5=entrance, 6=exit
        grid = np.zeros((20, 20), dtype=int)

        # Pydantic format: paths is a list of Path objects with x, y fields
        for path in state.get("paths", []):
            grid[path['x'], path['y']] = 1

        # Pydantic format: waters is a list of Water objects with x, y fields
        for water in state.get("waters", []):
            grid[water['x'], water['y']] = 2

        # Pydantic format: rides is a Rides object with ride_list containing Ride objects
        rides_data = state.get("rides", {})
        ride_list = rides_data.get("ride_list", []) if isinstance(rides_data, dict) else []
        for ride in ride_list:
            grid[ride['x'], ride['y']] = 3

        # Pydantic format: shops is a Shops object with shop_list containing Shop objects
        shops_data = state.get("shops", {})
        shop_list = shops_data.get("shop_list", []) if isinstance(shops_data, dict) else []
        for shop in shop_list:
            grid[shop['x'], shop['y']] = 4

        # Pydantic format: entrance and exit are tuples (x, y), not dicts
        entrance = state.get("entrance")
        exit_pos = state.get("exit")

        # Handle tuple format (x, y) from pydantic
        if entrance and isinstance(entrance, (list, tuple)):
            entrance_x, entrance_y = entrance[0], entrance[1]
            grid[entrance_x, entrance_y] = 5
        else:
            entrance_x, entrance_y = 0, 0

        if exit_pos and isinstance(exit_pos, (list, tuple)):
            grid[exit_pos[0], exit_pos[1]] = 6

        # BFS from entrance to find valid positions adjacent to paths
        valid_options = []
        entrance_pos = (entrance_x, entrance_y)
        visited = set()
        visited.add(entrance_pos)
        queue = deque([entrance_pos])

        while queue and len(valid_options) < TOP:
            current = queue.popleft()
            for neighbor in ReactAgent.get_neighbors(current):
                # If neighbor is a path, add to queue and visited
                if neighbor not in visited and grid[neighbor[0], neighbor[1]] == 1:
                    visited.add(neighbor)
                    queue.append(neighbor)
                # If neighbor is empty, add to valid options (but not from entrance directly)
                elif neighbor not in visited and grid[neighbor[0], neighbor[1]] == 0 and current != entrance_pos:
                    visited.add(neighbor)
                    valid_options.append(neighbor)

        # Handle case where no valid positions found
        if not valid_options:
            return (entrance_x, entrance_y)

        # Score positions: rides prefer water (+1), penalize empty (-1); shops opposite
        if entity_type == 'ride':
            scoring = {0: -1, 1: 0, 2: 1, 3: 0, 4: 0, 5: 0, 6: 0}
        else:  # shop
            scoring = {0: 1, 1: 0, 2: -1, 3: 0, 4: 0, 5: 0, 6: 0}

        scores = {
            option: sum([scoring[grid[n[0], n[1]]] for n in ReactAgent.get_neighbors(option)])
            for option in valid_options
        }
        sorted_options = sorted(valid_options, key=lambda x: scores[x], reverse=True)
        return sorted_options[0]

    @staticmethod
    def apply_placement_heuristic(action: str, state: Dict[str, Any]) -> str:
        """
        Replace coordinates in place/move actions with heuristic-computed values.

        Args:
            action: Action string like "place(x=5, y=10, type=\"ride\", ...)"
            state: Park state dictionary from obs.model_dump()

        Returns:
            Modified action string with heuristic coordinates
        """
        if not action:
            return action

        # Extract action name (before the parenthesis)
        action_lower = action.lower()

        # Check if this is a place or move action
        is_place = action_lower.startswith('place(')
        is_move = action_lower.startswith('move(')

        if not is_place and not is_move:
            return action  # Not a place/move action

        # Extract the type parameter to determine entity type (ride, shop, or staff)
        type_match = re.search(r'type\s*=\s*["\']?(ride|shop|staff)["\']?', action_lower)
        if not type_match:
            return action  # Can't determine entity type

        entity_type = type_match.group(1)

        # Only apply heuristic for rides and shops, not staff
        if entity_type not in ('ride', 'shop'):
            return action

        # Get heuristic coordinates
        new_x, new_y = ReactAgent.get_place_attraction_xy(entity_type, state)

        # Determine which coordinate keys to replace
        if is_move:
            # For move actions, replace new_x and new_y
            action = re.sub(r'\bnew_x\s*=\s*\d+', f'new_x={new_x}', action)
            action = re.sub(r'\bnew_y\s*=\s*\d+', f'new_y={new_y}', action)
        else:
            # For place actions, replace x and y (but not new_x/new_y)
            # Use negative lookbehind to avoid matching new_x/new_y
            action = re.sub(r'(?<!new_)\bx\s*=\s*\d+', f'x={new_x}', action)
            action = re.sub(r'(?<!new_)\by\s*=\s*\d+', f'y={new_y}', action)

        return action

    def react_step(self, obs: Any, info: Dict[str, Any], sandbox_steps_left: Optional[int] = None) -> str:
        """
        Single ReAct step:
        - builds state string and history
        - calls OpenRouter
        - parses into an action call string

        Returns:
            action: str
        """
        # State string
        py_json = obs.model_dump()
        if 'error' in info:
            state_str = (f"Park State : {json.dumps(py_json)}\n"
                         f"NOTE: While attempting the action `{self.prev_action}` the error `{info['error']}` occurred.")
        else:
            state_str = f"Park State : {json.dumps(py_json)}"

        # Messages
        if self.config['use_learnings']:
            learnings = "\n".join([f"Learning{i}: {learning}" for i, learning in enumerate(self.learnings)])
            system_prompt = self.system_prompt.replace("{LEARNINGS}", learnings)
            system_prompt_msg = {"role": "system", "content": system_prompt}
        else:
            system_prompt_msg = {"role": "system", "content": self.system_prompt}

        user_msg_content = self.dialogue_template.replace("{state_str}", state_str)
        if self.config['generate_learnings']:
            assert sandbox_steps_left is not None, "sandbox_steps_left must be provided when generate_learnings is true"
            user_msg_content = user_msg_content.replace("{sandbox_steps_left}", str(sandbox_steps_left))
        new_user_msg = {"role": "user", "content": user_msg_content}
        
        # Call LLM
        llm_output, token_usage = call_llm_openrouter(
            messages=[system_prompt_msg, *self.message_history, new_user_msg],
            model=self.model,
            api_key=self.api_key,
            temperature=self.temperature,
        )
        self.resource_cost += token_usage

        # Parse into "action_name(arg1=..., ...)"
        action = ReactAgent.parse_react_output(llm_output)

        # Apply placement heuristic if enabled
        if self.use_placement_heuristic and action:
            action = ReactAgent.apply_placement_heuristic(action, py_json)

        if self.config['generate_learnings']:
            learning = ReactAgent.extract_learnings(llm_output)
            self.learnings.append(learning)
            with open(self.learnings_path, "a") as f:
                f.write(learning + "\n")

        # Build history block for next step
        self.message_history.append(new_user_msg)
        self.message_history.append({"role": "assistant", "content": llm_output.strip()})


        # print(">>> system prompt: ", self.system_prompt)
        # for msg in self.message_history:
        #     role, content = msg['role'], msg['content']
        #     print(">>> --------------------------------")
        #     print(f">>> {role}: {content}")
        # print(f"================================================")

        # * 2 because we have one user message and one assistant message per step
        if len(self.message_history) > (self.max_history_length * 2):
            self.message_history = self.message_history[2:]  # Remove the oldest user and assistant messages

        self.prev_action = action
        return action

    def act(self, game_response: MapPydanticGameResponse, run_id: int, logging_id: Optional[str] = None, sandbox_steps_left: Optional[int] = None) -> str:
        return self.react_step(game_response.obs, game_response.info, sandbox_steps_left=sandbox_steps_left)

get_agent = ReactAgent.get_agent