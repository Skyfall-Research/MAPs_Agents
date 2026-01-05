import requests
import json
import base64
import io
from typing import Callable, List, Dict, Any, Optional, Tuple
import yaml
import numpy as np
from PIL import Image
import os

from map_py.shared_constants import GAMEPLAY_RULES, GAMEPLAY_RULES_ACTIONS_ONLY
from map_py.observations_and_actions.shared_constants import ACTIONS_BY_DIFFICULTY

from maps_agents.eval import AbstractAgent
from maps_agents.eval.state_interface import MapPydanticAndImageGameResponse
from maps_agents.llm.react import ReactAgent
from maps_agents.eval.resource_interface import Resource, ResourceCost
from maps_agents.eval.utils import EvalConfig


import base64, io
from typing import Literal, Optional, Tuple
import numpy as np
from PIL import Image

def encode_image_to_base64(
    image_array: np.ndarray,
    *,
    format: Literal["jpeg", "png", "webp"] = "png",
    jpeg_quality: int = 85,
    webp_quality: int = 80,
    lossless_webp: bool = False,
    max_size: Optional[Tuple[int, int]] = None,  # (max_w, max_h)
    add_data_url_prefix: bool = False,
) -> str:
    """
    Encode an RGB uint8 numpy image (H, W, 3) to base64.

    - format="png" is usually best for UI/text/sim renders.
    - format="jpeg" can be smaller for photo-like content.
    """
    if not isinstance(image_array, np.ndarray):
        raise ValueError(f"Expected numpy array, got {type(image_array)}")
    if image_array.ndim != 3 or image_array.shape[2] != 3:
        raise ValueError(f"Expected shape (H, W, 3), got {image_array.shape}")

    # Normalize dtype if needed (optional convenience)
    if image_array.dtype != np.uint8:
        # Accept float in [0,1] or int types, convert safely
        if np.issubdtype(image_array.dtype, np.floating):
            arr = np.clip(image_array, 0.0, 1.0)
            image_array = (arr * 255.0 + 0.5).astype(np.uint8)
        else:
            image_array = np.clip(image_array, 0, 255).astype(np.uint8)

    pil_image = Image.fromarray(image_array, mode="RGB")

    # Optional resize to reduce payload (keeps aspect ratio)
    if max_size is not None:
        max_w, max_h = max_size
        pil_image.thumbnail((max_w, max_h), Image.Resampling.LANCZOS)

    buffer = io.BytesIO()

    fmt = format.lower()
    if fmt == "jpeg":
        pil_image.save(buffer, format="JPEG", quality=jpeg_quality, optimize=True)
        mime = "image/jpeg"
    elif fmt == "png":
        pil_image.save(buffer, format="PNG", optimize=True)
        mime = "image/png"
    elif fmt == "webp":
        pil_image.save(
            buffer,
            format="WEBP",
            quality=webp_quality,
            lossless=lossless_webp,
            method=6,
        )
        mime = "image/webp"
    else:
        raise ValueError(f"Unsupported format: {format}")

    b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:{mime};base64,{b64}" if add_data_url_prefix else b64



def call_vlm_openrouter(
    messages: List[Dict[str, Any]],
    model: str,
    api_key: str,
    temperature: float = 0.2
) -> Tuple[str, ResourceCost]:
    """
    Call OpenRouter's chat/completions endpoint with multimodal support.

    Messages can contain either simple string content or structured content
    with text and images.

    Args:
        messages: List of message dicts with structure:
            {
                "role": "system"|"user"|"assistant",
                "content": str | [{"type": "text"|"image_url", ...}]
            }
        model: OpenRouter model name (e.g., "anthropic/claude-3.5-sonnet")
        api_key: OpenRouter API key
        temperature: Sampling temperature

    Returns:
        Tuple of (response_content: str, resource_cost: ResourceCost)

    Raises:
        requests.HTTPError: If the API call fails
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
        timeout=120,  # Longer timeout for vision models
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


class ReactVLMAgent(AbstractAgent):
    def __init__(
        self,
        *,
        horizon: int,
        difficulty: str,
        config_path: Optional[str | os.PathLike] = None,
    ):
        """
        ReAct-style VLM agent for MAPs using OpenRouter.

        Args:
            model: OpenRouter vision model name (e.g., "anthropic/claude-3.5-sonnet").
                   If None, reads from config file.
            api_key: OpenRouter API key
            horizon: Planning horizon (days)
            difficulty: Difficulty string ("easy" or "medium")
            temperature: VLM sampling temperature. If None, reads from config file.
            max_history_length: Max number of history message pairs to keep.
                                If None, reads from config file.
            config_path: Path to vlm_config.yaml
        """
        # Load config
        if config_path is None:
            config_path = "src/maps_agents/vlm/vlm_config.yaml"
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        self.config = config

        # Read settings from config if not provided
        self.model = config['vlm_model']
        self.temperature = config['temperature']
        self.max_history_length = config['max_history_length']

        ResourceCost.register_custom_model("anthropic/claude-4.5-sonnet", 3, 15)

        super().__init__(name=f"ReactVLMAgent({self.model})")

        self.api_key = os.getenv("OPENROUTER_API_KEY")
        self.horizon = horizon
        self.difficulty = difficulty
        self.message_history: List[Dict[str, Any]] = []

        # Resource tracking
        self.resource_cost = ResourceCost(model=self.model)
        self.prev_action = "N/A: No previous action."

        # Build system prompt using template-based approach
        actions_list = ", ".join(ACTIONS_BY_DIFFICULTY[self.difficulty])
        self.system_prompt = self.config['system_prompt']
        self.system_prompt = self.system_prompt.replace("{DOCS}", GAMEPLAY_RULES)
        self.system_prompt = self.system_prompt.replace("{horizon}", str(self.horizon))
        self.system_prompt = self.system_prompt.replace("{difficulty}", self.difficulty)
        self.system_prompt = self.system_prompt.replace("{actions_list}", actions_list)
        self.system_prompt_msg = {"role": "system", "content": self.system_prompt}

        # Extract dialogue template from config
        self.dialogue_template = self.config['dialogue'][0]['content']

    @staticmethod
    def get_agent(agent_config_path: Optional[str | os.PathLike], eval_config: EvalConfig) -> 'ReactVLMAgent':
        return ReactVLMAgent(
            horizon=eval_config.horizon,
            difficulty=eval_config.difficulty,
            config_path=agent_config_path,
        )

    @property
    def observation_type(self) -> str:
        return 'pydantic_and_image'

    def _build_user_message(
        self,
        state_str: str,
        image_base64: str,
    ) -> Dict[str, Any]:
        """
        Build a multimodal user message with text and image using config template.

        Args:
            state_str: Current state observation string
            image_base64: Base64-encoded image

        Returns:
            Message dict with multimodal content (text + image)
        """
        # Use dialogue template from config
        text_content = self.dialogue_template.replace("{state_str}", state_str)

        return {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": text_content
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{image_base64}"
                    }
                }
            ]
        }

    def _build_text_only_user_message(self, state_str: str) -> Dict[str, str]:
        """
        Build text-only user message for history (no image).

        Args:
            state_str: Current state observation string

        Returns:
            Message dict with text-only content
        """
        text_content = self.dialogue_template.replace("{state_str}", state_str)
        return {"role": "user", "content": text_content}

    def _build_assistant_response(self, vlm_output: str) -> Dict[str, str]:
        """Build assistant message dictionary."""
        return {"role": "assistant", "content": vlm_output.strip()}

    def react_step(
        self,
        pydantic_obs: Any,
        image: np.ndarray,
        info: Dict[str, Any]
    ) -> str:
        """
        Single ReAct step with vision support.

        Args:
            pydantic_obs: Pydantic observation (FullParkObs)
            image: RGB image array (H, W, 3) uint8
            info: Info dictionary from environment

        Returns:
            action: Action string

        Raises:
            ValueError: If image format is invalid
        """
        # Encode image to base64
        try:
            image_base64 = encode_image_to_base64(image)
        except ValueError as e:
            raise ValueError(f"Invalid image format: {e}")

        # Build state string
        py_json = pydantic_obs.model_dump()
        if 'error' in info:
            state_str = (
                f"Park State : {json.dumps(py_json)}\n"
                f"NOTE: While attempting the action `{self.prev_action}` "
                f"the error `{info['error']}` occurred."
            )
        else:
            state_str = f"Park State : {json.dumps(py_json)}"

        # Build messages
        system_msg = {"role": "system", "content": self.system_prompt}

        # Build current user message with image (for API call only, not stored in history)
        current_user_msg_with_image = self._build_user_message(state_str, image_base64)

        # Call VLM with text-only history + current multimodal message
        vlm_output, token_usage = call_vlm_openrouter(
            messages=[system_msg, *self.message_history, current_user_msg_with_image],
            model=self.model,
            api_key=self.api_key,
            temperature=self.temperature,
        )
        self.resource_cost += token_usage

        # Parse into "action_name(arg1=..., ...)"
        action = ReactAgent.parse_react_output(vlm_output)

        # Append text-only version to history (no image)
        text_only_user_msg = self._build_text_only_user_message(state_str)
        self.message_history.append(text_only_user_msg)
        self.message_history.append(self._build_assistant_response(vlm_output))

        # Debug output
        # print(">>> system prompt: ", self.system_prompt)
        # for msg in self.message_history:
        #     role, content = msg['role'], msg['content']
        #     print(">>> --------------------------------")
        #     print(f">>> {role}: {content}")
        # print(f"================================================")

        # * 2 because we have one user message and one assistant message per step
        if len(self.message_history) > (self.max_history_length * 2):
            self.message_history = self.message_history[2:]  # Remove oldest user and assistant messages

        self.prev_action = action
        return action

    def act(
        self,
        game_inputs: MapPydanticAndImageGameResponse,
        run_id: int,
        logging_id: Optional[str] = None
    ) -> str:
        """
        Generate an action based on multimodal observation.

        Args:
            game_inputs: Game response with pydantic_and_image observation
            run_id: Run identifier for tracking
            logging_id: Optional logging identifier

        Returns:
            Action string

        Raises:
            ValueError: If observation format is incorrect
        """
        # Extract observation
        obs_dict = game_inputs.obs

        # Validate observation format
        if not isinstance(obs_dict, dict):
            raise ValueError(
                f"Expected dict observation for pydantic_and_image, "
                f"got {type(obs_dict)}"
            )
        if 'pydantic_obs' not in obs_dict or 'image' not in obs_dict:
            raise ValueError(
                "Observation must contain 'pydantic_obs' and 'image' keys"
            )

        pydantic_obs = obs_dict['pydantic_obs']
        image = obs_dict['image']

        # Validate image format
        if not isinstance(image, np.ndarray):
            raise ValueError(f"Image must be numpy array, got {type(image)}")

        return self.react_step(pydantic_obs, image, game_inputs.info)

    def get_action_resource_usage(self, reset: bool = True) -> ResourceCost:
        """
        Get cumulative resource usage.

        Args:
            reset: Whether to reset internal counters

        Returns:
            ResourceCost object with token usage
        """
        resource_cost = self.resource_cost
        if reset:
            self.resource_cost = ResourceCost(model=self.model)
        return resource_cost

    # Static methods for code reuse
    @staticmethod
    def extract_only_action_name(action: str):
        """Extract action name from action string. Delegates to ReactAgent."""
        from maps_agents.llm.react import ReactAgent
        return ReactAgent.extract_only_action_name(action)

    @staticmethod
    def parse_react_output(vlm_output: str, tag="") -> str:
        """Parse ReAct output to extract action. Delegates to ReactAgent."""
        from maps_agents.llm.react import ReactAgent
        return ReactAgent.parse_react_output(vlm_output, tag)

    @staticmethod
    def parse_react_multi_output(vlm_output: str, max_proposals: int) -> List[str]:
        """Parse multi-proposal ReAct output. Delegates to ReactAgent."""
        from maps_agents.llm.react import ReactAgent
        return ReactAgent.parse_react_multi_output(vlm_output, max_proposals)


get_agent = ReactVLMAgent.get_agent