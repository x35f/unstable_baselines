overwrite_args = {
  "env_name": "HalfCheetah-v2", 
  "agent": {
    "q_network": {
      "learning_rate": 1e-3
    }, 
    "policy_network": {
      "learning_rate": 1e-3
    }, 
    "entropy": {
      "automatic_tuning": False
    }
  }
}
