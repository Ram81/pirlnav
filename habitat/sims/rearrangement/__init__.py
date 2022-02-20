from habitat.core.registry import registry
from habitat.core.simulator import Simulator

# from habitat.sims.habitat_simulator.actions import (
#     HabitatSimV1ActionSpaceConfiguration,
# )


def _try_register_rearrangement_sim():
    try:
        import habitat_sim  # noqa: F401

        has_habitat_sim = True
    except ImportError as e:
        has_habitat_sim = False
        rearrangement_sim_import_error = e

    if has_habitat_sim:
        from habitat.sims.rearrangement.actions import (  # noqa: F401
            RearrangementSimV0ActionSpaceConfiguration,
        )
    else:

        @registry.register_simulator(name="RearrangementSim-v0")
        class RearrangementSimImportError(Simulator):
            def __init__(self, *args, **kwargs):
                raise rearrangement_sim_import_error
