# Local imports
from maps_agents.llm.react import ReactAgent
from maps_agents.eval.utils import EvalConfig 
from maps_agents.eval.state_interface import GameResponse, OBSERVATION_TYPE_TO_GAME_RESPONSE
from maps_agents.eval.agent_interface import AbstractAgent, ActionGenerationError, ResourceCost

# External imports
from map_py.mini_amusement_park import MiniAmusementPark


def generate_learnings(layout_name, difficulty, num_steps):
    eval_config = EvalConfig(run_idx=0, difficulty=difficulty)
    agent: AbstractAgent = ReactAgent.get_agent("src/maps_agents/llm/react_generate_learnings_config.yaml", eval_config)

    if not agent.supports_config(eval_config):
        raise ValueError("Agent does not support the current eval config.")

    game = MiniAmusementPark(
        host="localhost", 
        port="3000", 
        return_raw_in_info=True,
        observation_type=agent.observation_type,
        layout=layout_name,
        difficulty=difficulty
    )
    game.reset()

    game.sandbox_action(f"set_sandbox_mode(sandbox_steps={num_steps})")

    curr_state, curr_raw_state = game.get_observation_and_raw_state()
    info = {'raw_state': curr_raw_state}

    # Initial reward is set to zero
    agent_input = OBSERVATION_TYPE_TO_GAME_RESPONSE[agent.observation_type](obs=curr_state, info=info, terminated=False, truncated=False, reward=0)

    learning_steps = 0
    while learning_steps < num_steps:
        action = agent.act(agent_input, game.park_id, sandbox_steps_left=num_steps - learning_steps)
        if game.is_sandbox_action(action):
            game.sandbox_action(action)
        else:
            game.step(action)
            learning_steps += 1

if __name__ == "__main__":
    generate_learnings(layout_name="diagonal_squares", difficulty="easy", num_steps=100)