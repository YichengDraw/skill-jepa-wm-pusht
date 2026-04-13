from __future__ import annotations

from dataclasses import dataclass


@dataclass
class HierarchicalPlan:
    actions: object
    skill: object
    subgoal: object
    high_level_costs: object
    low_level_costs: object
    skill_consistency: object


class HierarchicalPlanner:
    def __init__(self, high_level_planner, low_level_planner, subgoal_resolver=None) -> None:
        self.high_level_planner = high_level_planner
        self.low_level_planner = low_level_planner
        self.subgoal_resolver = subgoal_resolver

    def plan(self, current_z, current_s, goal_z) -> HierarchicalPlan:
        high_plan = self.high_level_planner.plan(current_z=current_z, goal_z=goal_z)
        subgoal_s = self.subgoal_resolver(high_plan.subgoal) if self.subgoal_resolver is not None else None
        low_plan = self.low_level_planner.plan(
            spatial_tokens=current_s,
            target_z=high_plan.subgoal,
            target_skill=high_plan.first_skill,
            target_s=subgoal_s,
        )
        return HierarchicalPlan(
            actions=low_plan.actions,
            skill=high_plan.first_skill,
            subgoal=high_plan.subgoal,
            high_level_costs=high_plan.costs,
            low_level_costs=low_plan.costs,
            skill_consistency=low_plan.skill_consistency,
        )
