from habitat.core.registry import registry
from habitat.core.simulator import Simulator


def _try_register_pickplace_sim():
    try:
        import habitat_sim  # noqa: F401

        has_habitat_sim = True
    except ImportError as e:
        has_habitat_sim = False
        rearrangement_sim_import_error = e

    if has_habitat_sim:
        from habitat.sims.pickplace.actions import (  # noqa: F401
            PickPlaceSimV0ActionSpaceConfiguration,
        )
    else:

        @registry.register_simulator(name="PickPlaceSim-v0")
        class PickPlaceSimImportError(Simulator):
            def __init__(self, *args, **kwargs):
                raise rearrangement_sim_import_error
