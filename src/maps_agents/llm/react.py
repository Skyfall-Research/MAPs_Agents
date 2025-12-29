import requests
import json
import os
from typing import Callable, List, Dict, Any, Optional, Tuple
import yaml

from maps_agents.eval import AbstractAgent
from maps_agents.eval.state_interface import MapPydanticGameResponse
from map_py.shared_constants import GAMEPLAY_RULES, SANDBOX_GAMEPLAY_RULES, SANDBOX_INSTR_WITH_DOC_REF
from map_py.observations_and_actions.shared_constants import ACTIONS_BY_DIFFICULTY
from maps_agents.eval.resource_interface import Resource, ResourceCost
from maps_agents.eval.utils import EvalConfig


def call_llm_openrouter(messages, model: str, api_key: str, temperature: float = 0.2) -> Tuple[str, ResourceCost]:
    """
    Thin wrapper around OpenRouter's chat/completions endpoint.
    `messages` is a list of dicts: [{"role": "system"|"user"|"assistant", "content": "..."}]
    
    Returns:
        Tuple of (content: str, token_usage: Dict[str, int]) where token_usage contains
        'input_tokens' and 'output_tokens'
    """
    resp = requests.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json={
            "model": model,
            "messages": messages,
            "temperature": temperature,
        },
        timeout=60,
    )
    resp.raise_for_status()
    data = resp.json()
    content = data["choices"][0]["message"]["content"]
    usage = data.get("usage", {})
    token_usage = ResourceCost(model=model, costs={
        Resource.TOKENS_IN: usage.get("prompt_tokens", 0),
        Resource.TOKENS_OUT: usage.get("completion_tokens", 0),
    })
    return content, token_usage

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
        if self.config['generate_learnings']:
            learning = ReactAgent.extract_learnings(llm_output)
            self.learnings.append(learning)
            with open(self.learnings_path, "a") as f:
                f.write(learning + "\n")

        # Build history block for next step
        self.message_history.append(new_user_msg)
        self.message_history.append({"role": "assistant", "content": llm_output.strip()})


        print(">>> system prompt: ", self.system_prompt)
        for msg in self.message_history:
            role, content = msg['role'], msg['content']
            print(">>> --------------------------------")
            print(f">>> {role}: {content}")
        print(f"================================================")

        # * 2 because we have one user message and one assistant message per step
        if len(self.message_history) > (self.max_history_length * 2):
            self.message_history = self.message_history[2:]  # Remove the oldest user and assistant messages

        self.prev_action = action
        return action

    def act(self, game_response: MapPydanticGameResponse, run_id: int, logging_id: Optional[str] = None, sandbox_steps_left: Optional[int] = None) -> str:
        return self.react_step(game_response.obs, game_response.info, sandbox_steps_left=sandbox_steps_left)

get_agent = ReactAgent.get_agent