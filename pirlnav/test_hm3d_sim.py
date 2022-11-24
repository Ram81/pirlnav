import habitat_sim

backend_cfg = habitat_sim.SimulatorConfiguration()
backend_cfg.scene_id = "data/scene_datasets/hm3d/val/00802-wcojb4TFT35/wcojb4TFT35.basis.glb"
backend_cfg.scene_dataset_config_file = "data/scene_datasets/hm3d/val/hm3d_annotated_basis.scene_dataset_config.json"

sem_cfg = habitat_sim.CameraSensorSpec()
sem_cfg.uuid = "semantic"
sem_cfg.sensor_type = habitat_sim.SensorType.SEMANTIC

agent_cfg = habitat_sim.agent.AgentConfiguration()
agent_cfg.sensor_specifications = [sem_cfg]

sim_cfg = habitat_sim.Configuration(backend_cfg, [agent_cfg])
sim = habitat_sim.Simulator(sim_cfg)

