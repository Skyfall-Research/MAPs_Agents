import argparse
from maps_agents import ReactAgent, ReactVLMAgent, RLAgent
from maps_agents.eval.evaluator import evaluate_agent

AGENT_MAP = {
    'react': (ReactAgent,
    'react_vlm': (ReactVLMAgent,
    'rl': (RLAgent,
}



def get_default_config(agent_type: str) -> str:
    """Get the default config path for an agent type."""
    config_map = {
        'react': 'src/maps_agents/llm/react_generate_learnings_config.yaml',
        'react_vlm': 'src/maps_agents/vlm/vlm_config.yaml',
        'rl': 'src/maps_agents/rl/config.yaml',
    }
    return config_map.get(agent_type.lower(), config_map['react'])


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate an agent on the Mini Amusement Park Simulator',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        '--agent',
        type=str,
        choices=['react', 'react_vlm', 'rl'],
        default='react',
        help='Type of agent to evaluate'
    )
    
    parser.add_argument(
        '--config',
        type=str,
        default=None,
        help='Path to agent configuration file. If not provided, uses default for agent type.'
    )
    
    parser.add_argument(
        '--samples-per-start',
        type=int,
        default=1,
        help='Number of evaluation runs per starting layout'
    )
    
    parser.add_argument(
        '--input-layouts',
        type=str,
        nargs='+',
        default=['diagonal_squares'],
        help='List of input layouts to evaluate on'
    )
    
    parser.add_argument(
        '--difficulty',
        type=str,
        choices=['easy', 'medium'],
        default='easy',
        help='Game difficulty level'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose output'
    )
    
    parser.add_argument(
        '--max-processes',
        type=int,
        default=0,
        help='Maximum number of parallel processes (0 for sequential execution)'
    )
    
    parser.add_argument(
        '--port',
        type=int,
        default=3000,
        help='Port number for the game server connection'
    )
    
    parser.add_argument(
        '--exp-name',
        type=str,
        default=None,
        help='Experiment name (default: auto-generated with timestamp)'
    )
    
    args = parser.parse_args()
    
    # Get agent constructor
    agent_constructor = get_agent_constructor(args.agent)
    
    # Get config path (use default if not provided)
    config_path = args.config if args.config is not None else get_default_config(args.agent)
    
    # Run evaluation
    evaluate_agent(
        agent_constructor=agent_constructor,
        agent_config_path=config_path,
        samples_per_start=args.samples_per_start,
        input_layouts=args.input_layouts,
        difficulty=args.difficulty,
        verbose=args.verbose,
        max_processes=args.max_processes,
        port=args.port,
        exp_name=args.exp_name
    )


if __name__ == '__main__':
    main()