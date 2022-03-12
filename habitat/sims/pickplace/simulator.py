#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Dict, List, Optional

import numpy as np
import math
import magnum as mn

from collections import defaultdict

from habitat_sim.nav import NavMeshSettings
from habitat_sim.utils.common import quat_from_coeffs, quat_to_magnum, quat_to_coeffs, quat_from_magnum
from habitat_sim.physics import MotionType

from habitat.core.registry import registry
from habitat.core.simulator import (
    Config,
    Observations
)
from habitat.sims.habitat_simulator.habitat_simulator import HabitatSim
from habitat.tasks.pickplace.pickplace import ObjectStateSpec


@registry.register_simulator(name="PickPlaceSim-v0")
class PickPlaceSim(HabitatSim):
    r"""Simulator wrapper over habitat-sim with
    object rearrangement functionalities.
    """

    def __init__(self, config: Config) -> None:
        self.did_reset = False
        super().__init__(config=config)
        self.nearest_object_id = -1
        self.gripped_object_id = -1
        self.gripped_object_transformation = None
        self.max_distance = 2.0
        self.agent_object_handle = "cylinderSolid_rings_1_segments_12_halfLen_1_useTexCoords_false_useTangents_false_capEnds_true"

        agent_id = self.habitat_config.DEFAULT_AGENT_ID
        agent_config = self._get_agent_config(agent_id)

        self.navmesh_settings = NavMeshSettings()
        self.navmesh_settings.set_defaults()
        self.navmesh_settings.agent_radius = agent_config.RADIUS
        self.navmesh_settings.agent_height = agent_config.HEIGHT
        self.default_agent_id = agent_id
        self.ISLAND_RADIUS_LIMIT = 2.0

    def reconfigure(self, config: Config) -> None:
        self.remove_existing_objects()
        super().reconfigure(config)
        self._initialize_objects(config)

    def reset(self):
        sim_obs = super().reset()
        if self._update_agents_state():
            sim_obs = self.get_sensor_observations(agent_ids=self.default_agent_id)

        self._prev_sim_obs = sim_obs
        self._prev_sim_obs["gripped_object_id"] = -1
        self._prev_sim_obs["object_under_cross_hair"] = -1
        self._prev_step_data = defaultdict(int)
        self.did_reset = True
        self.gripped_object_id = -1
        self.nearest_object_id = -1
        return self._sensor_suite.get_observations(sim_obs)

    def remove_existing_objects(self):
        # first remove all existing objects
        existing_object_ids = self.get_existing_object_ids()

        for obj_id in existing_object_ids:
            self.remove_object(obj_id)
            object_ = self.get_object_from_scene(obj_id)
            if object_ is not None:
                self.remove_contact_test_object(object_.object_handle)
        self.remove_contact_test_object(self.agent_object_handle)
        self.clear_recycled_object_ids()
        self.clear_scene_objects()

    def _initialize_objects(self, sim_config):
        # Add contact test object for agent
        self.add_contact_test_object(self.agent_object_handle)
        objects = sim_config.objects
        obj_attr_mgr = self.get_object_template_manager()

        self.obj_id_to_semantic_obj_id_map = defaultdict(int)

        if objects is not None:
            # Sort objects by object id
            object_map = {}
            for object_ in objects:
                object_map[object_.object_id] = object_

            for key in sorted(object_map.keys()):
                object_ = object_map[key]
                object_handle = object_.object_template.split('/')[-1]
                object_template = "data/test_assets/objects/{}".format(object_handle)
                object_pos = object_.position
                rotation = quat_from_coeffs(object_.rotation)
                object_rotation = quat_to_magnum(rotation)

                object_template_id = obj_attr_mgr.load_object_configs(
                    object_template
                )[0]
                object_attr = obj_attr_mgr.get_template_by_ID(object_template_id)
                obj_attr_mgr.register_template(object_attr)

                object_id = self.add_object_by_handle(object_attr.handle)
                self.add_contact_test_object(object_attr.handle)

                self.set_translation(object_pos, object_id)
                self.set_rotation(object_rotation, object_id)

                object_.object_handle = "data/test_assets/objects/{}".format(object_handle)
                self.add_object_in_scene(object_id, object_)

                self.set_object_motion_type(MotionType.DYNAMIC, object_id)
                self.obj_id_to_semantic_obj_id_map[object_id] = object_.semantic_object_id
                self.set_object_semantic_id(object_.semantic_object_id, object_id)


    def get_resolution(self):
        resolution = self._default_agent.agent_config.sensor_specifications[
            0
        ].resolution
        return mn.Vector2(list(map(int, resolution)))

    def clear_scene_objects(self):
        self._scene_objects = []

    def add_object_in_scene(self, objectId, data):
        data.object_id = objectId
        self._scene_objects.append(data)

    def update_object_in_scene(self, prevObjectId, newObjectId):
        for index in range(len(self._scene_objects)):
            if self._scene_objects[index].object_id == prevObjectId:
                self._scene_objects[index].object_id = newObjectId

    def get_object_from_scene(self, objectId):
        for index in range(len(self._scene_objects)):
            if self._scene_objects[index].object_id == objectId:
                return self._scene_objects[index]
        return None
    
    def check_object_exists_in_scene(self, object_id):
        exists = object_id in self.get_existing_object_ids()
        return exists
    
    def is_collision(self, handle, translation, is_navigation_test = False):
        collision_filter = 1
        collision_mask = -1
        if is_navigation_test:
            collision_mask = 1
        return self.pre_add_contact_test(
            handle, translation, is_navigation_test, collision_filter, collision_mask
        )

    def is_agent_colliding(self, action, agentTransform):
        stepSize = 0.15
        if action == "move_forward":
            position = agentTransform.backward * (-1 * stepSize)
            newPosition = agentTransform.translation + position
            filteredPoint = self.pathfinder.try_step(
                agentTransform.translation,
                newPosition
            )
            filterDiff = filteredPoint - newPosition
            # adding buffer of 0.05 y to avoid collision with navmesh
            offsets = [0, 0.05, -0.05, 0.1, -0.1, -0.2]
            is_colliding = False
            for offset in offsets:
                finalPosition = newPosition + filterDiff + mn.Vector3(0.0, offset, 0.0)
                collision = self.is_collision(self.agent_object_handle, finalPosition, True)
                if collision:
                    is_colliding = True
            return {
                "collision": is_colliding,
            }
        elif action == "move_backward":
            position = agentTransform.backward * stepSize
            newPosition = agentTransform.translation + position
            filteredPoint = self.pathfinder.try_step(
                agentTransform.translation,
                newPosition
            )
            filterDiff = filteredPoint - newPosition
            # adding buffer of 0.05 y to avoid collision with navmesh
            offsets = [0, 0.05, -0.05, 0.1, -0.1, -0.2]
            is_colliding = False
            for offset in offsets:
                finalPosition = newPosition + filterDiff + mn.Vector3(0.0, offset, 0.0)
                collision = self.is_collision(self.agent_object_handle, finalPosition, True)
                if collision:
                    is_colliding = True
            return {
                "collision": is_colliding,
            }
        return {
            "collision": False
        }

    def draw_bb_around_nearest_object(self, object_id):
        if object_id == -1:
            if self.nearest_object_id != -1 and self.gripped_object_id != self.nearest_object_id:
                self.set_object_bb_draw(False, self.nearest_object_id)
                self.nearest_object_id = object_id
        else:
            if self.nearest_object_id != -1 and self.gripped_object_id != self.nearest_object_id:
                self.set_object_bb_draw(False, self.nearest_object_id)
                self.nearest_object_id = -1
            object_ = self.get_object_from_scene(object_id)
            if object_.is_receptacle == True:
                return
            if self.nearest_object_id != object_id:
                self.nearest_object_id = object_id
                self.set_object_bb_draw(True, self.nearest_object_id, 0)

    def step(self, action: int):
        dt = 1.0 / 20.0
        self._num_total_frames += 1
        collided = False

        agent_config = self._default_agent.agent_config
        action_spec = agent_config.action_space[action]

        crosshair_pos = np.array(self.get_resolution()) // 2
        crosshair_pos = list(map(int, crosshair_pos))

        ray = self.unproject(crosshair_pos)
        cross_hair_point = ray.direction
        ref_point = self._default_agent.body.object.absolute_translation

        nearest_object_id = self.find_nearest_object_under_crosshair(
            cross_hair_point, ref_point, self.get_resolution(), self.max_distance
        )
        object_under_cross_hair = nearest_object_id
        self._prev_step_data["nearest_object_id"] = nearest_object_id
        self._prev_step_data["gripped_object_id"] = self.gripped_object_id

        if action_spec.name == "grab_or_release_object_under_crosshair":
            self._prev_step_data["action"] = action_spec.name
            self._prev_step_data["is_grab_action"] = self.gripped_object_id == -1
            self._prev_step_data["is_release_action"] = self.gripped_object_id != -1
            # already gripped an object
            if self.gripped_object_id != -1:
                ref_transform = self._default_agent.body.object.transformation
                ray_hit_info = self.find_floor_position_under_crosshair(
                    cross_hair_point, ref_transform,
                    self.get_resolution(), action_spec.actuation.amount
                )

                floor_position = ray_hit_info.point
                if floor_position is None:
                    return True

                new_object_position = mn.Vector3(
                    floor_position.x, floor_position.y, floor_position.z
                )
                scene_object = self.get_object_from_scene(self.gripped_object_id)

                # find no collision point
                count = 0
                contact = self.is_collision(scene_object.object_handle, new_object_position)
                while contact and count < 4:
                    new_object_position = mn.Vector3(
                        new_object_position.x,
                        new_object_position.y + 0.25,
                        new_object_position.z,
                    )
                    contact = self.is_collision(scene_object.object_handle, new_object_position)
                    count += 1

                snapped_point = self.pathfinder.snap_point(new_object_position)
                snapped_point = np.array(new_object_position)
                is_point_on_separate_island = self.island_radius(snapped_point) < self.ISLAND_RADIUS_LIMIT

                if not (contact or is_point_on_separate_island):
                    new_object_id = self.add_object_by_handle(
                        scene_object.object_handle
                    )
                    self.set_translation(new_object_position, new_object_id)
                    self.set_object_semantic_id(scene_object.semantic_object_id, new_object_id)

                    self.update_object_in_scene(self.gripped_object_id, new_object_id)
                    self._prev_step_data["new_object_translation"] = np.array(new_object_position).tolist()
                    self._prev_step_data["new_object_id"] = new_object_id
                    self._prev_step_data["object_handle"] = scene_object.object_handle
                    self._prev_step_data["gripped_object_id"] = self.gripped_object_id
                    self.gripped_object_id = -1
            elif nearest_object_id != -1:
                self.gripped_object_transformation = self.get_transformation(
                    nearest_object_id
                )
                object_ = self.get_object_from_scene(nearest_object_id)
                if object_.is_receptacle != True:
                    self.remove_object(nearest_object_id)
                    self.gripped_object_id = nearest_object_id
                self._prev_step_data["gripped_object_id"] = self.gripped_object_id
        elif action_spec.name == "no_op":
            pass
        else:
            agent_transform = self._default_agent.body.object.transformation
            data = self.is_agent_colliding(action_spec.name, agent_transform)
            if not data["collision"]:
                self._default_agent.act(action)
                collided = data["collision"]
                self._last_state = self._default_agent.get_state()

        # step world physics
        super().step_world(dt)

        # self.draw_bb_around_nearest_object(object_under_cross_hair)

        # obtain observations
        self._prev_sim_obs = self.get_sensor_observations(agent_ids=self.default_agent_id, draw_crosshair=False)
        self._prev_sim_obs["collided"] = collided
        self._prev_sim_obs["gripped_object_id"] = self.gripped_object_id
        self._prev_sim_obs["object_under_cross_hair"] = object_under_cross_hair
        self._prev_step_data["collided"] = collided

        observations = self._sensor_suite.get_observations(self._prev_sim_obs)
        return observations

    def is_object_out_of_bounds(self):
        object_ids = self.get_existing_object_ids()
        for object_id in object_ids:
            position = self.get_translation(object_id)
            snapped_position = self.pathfinder.snap_point(position)

            if np.isnan(snapped_position.x) or math.isnan(snapped_position.x):
                return True
        return False

    def restore_object_states(self, object_states: Dict = {}):
        object_ids = []
        for object_state in object_states:
            object_id = object_state.object_id
            translation = object_state.translation
            object_rotation = object_state.rotation

            object_translation = mn.Vector3(translation)
            if isinstance(object_rotation, list):
                object_rotation = quat_from_coeffs(object_rotation)

            object_rotation = quat_to_magnum(object_rotation)
            self.set_translation(object_translation, object_id)
            self.set_rotation(object_rotation, object_id)
            object_ids.append(object_id)
        return object_ids
    
    def get_current_object_states(self):
        existing_object_ids = self.get_existing_object_ids()
        object_states = []
        for object_id in existing_object_ids:
            translation = self.get_translation(object_id)
            rotation = self.get_rotation(object_id)
            rotation = quat_from_magnum(rotation)
            scene_object = self.get_object_from_scene(object_id)

            object_state = ObjectStateSpec(
                object_id=object_id,
                translation=np.array(translation).tolist(),
                rotation=quat_to_coeffs(rotation).tolist(),
                object_handle=scene_object.object_handle,
            )
            object_states.append(object_state)
        return object_states
    
    def get_agent_pose(self):
        agent_translation = self._default_agent.body.object.translation
        agent_rotation = self._default_agent.body.object.rotation
        sensor_data = {}
        for sensor_key, v in self._default_agent._sensors.items():
            rotation = quat_from_magnum(v.object.rotation)
            rotation = quat_to_coeffs(rotation).tolist()
            translation = v.object.translation
            sensor_data[sensor_key] = {
                "rotation": rotation,
                "translation": np.array(translation).tolist()
            }
        
        return {
            "position": np.array(agent_translation).tolist(),
            "rotation": quat_to_coeffs(quat_from_magnum(agent_rotation)).tolist(),
            "sensor_data": sensor_data
        }
    
    def restore_sensor_states(self, sensor_data: Dict = {}):
        for sensor_key, v in self._default_agent._sensors.items():
            rotation = None
            if sensor_key in sensor_data.keys():
                rotation = sensor_data[sensor_key]["rotation"]
            else:
                rotation = sensor_data["rgb"]["rotation"]
            rotation = quat_from_coeffs(rotation)
            agent_rotation = quat_to_magnum(rotation)
            v.object.rotation = agent_rotation
    
    def get_sensor_states(self):
        sensor_states = {}
        for sensor_key, v in self._default_agent._sensors.items():
            rotation = v.object.rotation
            rotation = quat_from_magnum(rotation)
            rotation = quat_to_coeffs(rotation)
            sensor_states[sensor_key] = {
                "rotation": rotation
            }
        return sensor_states

    def add_objects_by_handle(self, objects):
        for object_ in objects:
            object_handle = object_.object_handle
            object_id = self.add_object_by_handle(object_handle)
            self.set_object_semantic_id(object_.semantic_object_id, object_id)

    def step_from_replay(self, action: int, replay_data: Dict = {}):
        dt = 1.0 / 20.0
        self._num_total_frames += 1
        collided = False

        agent_config = self._default_agent.agent_config
        action_spec = agent_config.action_space[action]

        if action_spec.name == "grab_or_release_object_under_crosshair":
            action_data = replay_data.action_data
            if action_data is not None and action_data.released_object_id != -1:
                if replay_data.is_release_action:
                    # Fetch object handle and drop point from replay
                    new_object_position = mn.Vector3(action_data.released_object_position)
                    scene_object = self.get_object_from_scene(action_data.released_object_id)
                    new_object_id = self.add_object_by_handle(
                        scene_object.object_handle
                    )
                    self.set_translation(new_object_position, new_object_id)
                    self.set_object_semantic_id(scene_object.semantic_object_id, new_object_id)

                    self.update_object_in_scene(new_object_id, action_data.released_object_id)
                    self.gripped_object_id = replay_data.gripped_object_id
                elif replay_data.is_grab_action:
                    self.gripped_object_transformation = self.get_transformation(
                        action_data.grab_object_id
                    )
                    self.remove_object(action_data.grab_object_id)
                    self.gripped_object_id = replay_data.gripped_object_id
        elif action_spec.name == "no_op":
            self.restore_object_states(replay_data.object_states)
        else:
            if replay_data.agent_state is not None:
                if action_spec.name == "look_up" or action_spec.name == "look_down":
                    sensor_data = replay_data.agent_state.sensor_data
                    self.restore_sensor_states(sensor_data)
                else:
                    position = np.array(replay_data.agent_state.position)
                    rotation = np.array(replay_data.agent_state.rotation)
                    success = self.set_agent_state(
                        position, rotation, reset_sensors=False
                    )
                collided = replay_data.collision
                self._last_state = self._default_agent.get_state()
            else:
                collided = replay_data.collision
                if not collided:
                    self._default_agent.act(action)

        # self.draw_bb_around_nearest_object(replay_data.object_under_cross_hair)

        # obtain observations
        self._prev_sim_obs = self.get_sensor_observations(agent_ids=self.default_agent_id, draw_crosshair=False)
        self._prev_sim_obs["collided"] = collided
        self._prev_sim_obs["gripped_object_id"] = self.gripped_object_id

        observations = self._sensor_suite.get_observations(self._prev_sim_obs)
        return observations

    def get_observations_at(
        self,
        position: Optional[List[float]] = None,
        rotation: Optional[List[float]] = None,
        sensor_states: Optional[List[Dict]] = None,
        object_states: Optional[List[Dict]] = None,
        keep_agent_at_new_pose: bool = False,
    ) -> Optional[Observations]:
        current_state = self.get_agent_state()
        current_object_states = self.get_current_object_states()
        current_sensor_states = self.get_sensor_states()
        if position is None or rotation is None:
            success = True
        else:
            success = self.set_agent_state(
                np.array(position), np.array(rotation), reset_sensors=False
            )
        if sensor_states is not None:
            self.restore_sensor_states(sensor_states)

        object_to_re_add = []
        if object_states is not None:
            current_state_objects = self.restore_object_states(object_states)

            for object_id in self.get_existing_object_ids():
                if object_id not in current_state_objects:
                    self.remove_object(object_id)
                    object_to_re_add.append(self.get_object_from_scene(object_id))

        if success:
            sim_obs = self.get_sensor_observations(agent_ids=self.default_agent_id)

            self._prev_sim_obs = sim_obs

            observations = self._sensor_suite.get_observations(sim_obs)
            if not keep_agent_at_new_pose:
                self.set_agent_state(
                    current_state.position,
                    current_state.rotation,
                    reset_sensors=False,
                )
                self.add_objects_by_handle(object_to_re_add)
                self.restore_object_states(current_object_states)
                self.restore_sensor_states(current_sensor_states)
                
            return observations
        else:
            return None
