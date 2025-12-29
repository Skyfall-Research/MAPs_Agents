"""
Simplified Hierarchical Policy for MultiDiscrete Action Spaces.

This policy works with SimpleParkActionSpace and MapsSimpleGymObservationSpace,
providing hierarchical sampling with simpler masking (no coordinate/price masks).
"""

from typing import Any, Dict
import torch as th
import torch.nn.functional as F
from stable_baselines3.common.policies import MultiInputActorCriticPolicy
from stable_baselines3.common.distributions import CategoricalDistribution
from map_py.observations_and_actions.simple_gym_obs import NORMALIZATION_CONFIG
from map_py.observations_and_actions.shared_constants import MAP_CONFIG


def denormalize_with_config_tensor(value: th.Tensor, config_key: str) -> th.Tensor:
    """
    Invert normalize_with_config, operating on a batched torch Tensor.

    Args:
        value:       normalized tensor (...), values in [0,1]
        config_key:  key into NORMALIZATION_CONFIG

    Returns:
        denormalized tensor of same shape/device as `value`
    """
    use_log, max_value = NORMALIZATION_CONFIG[config_key]

    if max_value == 1.0:
        raise ValueError(
            f"Normalization max=1.0 for {config_key} is 1.0; cannot invert."
        )
    if max_value < -1.0:
        raise ValueError(
            f"Normalization max={max_value} for {config_key} is less than -1.0; cannot invert."
        )

    max_value_t = th.as_tensor(max_value, dtype=value.dtype, device=value.device)

    if use_log:
        # y = log1p(x) / log1p(M)  →  log1p(x) = y * log1p(M)  →  x = expm1(...)
        max_value_eff = th.log1p(max_value_t)   # log1p(M)
        unscaled = value * max_value_eff        # log1p(x)
        return th.expm1(unscaled)               # x
    else:
        # y = x / M → x = y * M
        return value * max_value_t


def get_entity_existence_cache_simple(
    rides_vector: th.Tensor,
    shops_vector: th.Tensor,
    staff_vector: th.Tensor
) -> Dict[str, th.Tensor]:
    """
    Compute entity existence from vector representations.

    Args:
        rides_vector: [B, 19] = 12 counts + 7 metrics
        shops_vector: [B, 15] = 12 counts + 3 metrics
        staff_vector: [B, 14] = 12 counts + 2 costs

    Returns:
        Dictionary with:
        - 'type': [B, 3] tensor (ride/shop/staff existence)
        - 'subtype': [B, 9] tensor (all subtype existence)
        - 'subclass': [B, 9, 4] tensor (subtype+color combinations)
    """
    batch_size = rides_vector.shape[0]
    device = rides_vector.device

    ride_counts = rides_vector[:, :12].view(batch_size, 3, 4)   # [B, 3, 4]
    shop_counts = shops_vector[:, :12].view(batch_size, 3, 4)   # [B, 3, 4]
    staff_counts = staff_vector[:, :12].view(batch_size, 3, 4)  # [B, 3, 4]

    # TYPE EXISTENCE [B, 3]
    type_exists = th.zeros(batch_size, 3, dtype=th.bool, device=device)
    type_exists[:, 0] = (ride_counts > 0.0).any(dim=(1, 2))
    type_exists[:, 1] = (shop_counts > 0.0).any(dim=(1, 2))
    type_exists[:, 2] = (staff_counts > 0.0).any(dim=(1, 2))

    # SUBTYPE EXISTENCE [B, 9]
    subtype_exists = th.zeros(batch_size, 9, dtype=th.bool, device=device)
    subtype_exists[:, 0:3] = (ride_counts > 0.0).any(dim=2)
    subtype_exists[:, 3:6] = (shop_counts > 0.0).any(dim=2)
    subtype_exists[:, 6:9] = (staff_counts > 0.0).any(dim=2)

    # SUBCLASS EXISTENCE [B, 9, 4]
    subclass_exists = th.zeros(batch_size, 9, 4, dtype=th.bool, device=device)
    subclass_exists[:, 0:3, :] = ride_counts > 0.0
    subclass_exists[:, 3:6, :] = shop_counts > 0.0
    subclass_exists[:, 6:9, :] = staff_counts > 0.0

    return {
        "type": type_exists,
        "subtype": subtype_exists,
        "subclass": subclass_exists,
    }


class SimpleHierarchicalPolicy(MultiInputActorCriticPolicy):
    """
    Simplified hierarchical policy for the simple action/observation spaces.

    Works with:
    - SimpleParkActionSpace: 5 actions, 3 parameters (type, subtype, subclass)
    - MapsSimpleGymObservationSpace: 5 vectors (rides, shops, staff, guests, park)

    No coordinate or price masking - those are handled by decode_action.
    """

    def __init__(self, *args, difficulty: str = "easy", **kwargs):
        super().__init__(*args, **kwargs)

        self.nvec = self.action_space.nvec  # e.g. [5, 3, 9, 4]
        self.difficulty = difficulty

        # Action masks: which parameters are used for each action
        # shape: [n_action_types, n_params]
        mask_tensor = th.zeros(int(self.nvec[0]), len(self.nvec) - 1, dtype=th.bool, device=self.device)
        for action_type, mask_list in self.action_space.action_masks.items():
            mask_tensor[action_type, :] = th.tensor(mask_list, dtype=th.bool, device=self.device)
        self.register_buffer("param_mask_by_type", mask_tensor)

        # Parameter mappings
        self.dim_to_param_mapping = self.action_space.dim_to_param_mapping
        self.param_to_dim_mapping = self.action_space.param_to_dim_mapping

        self.staff_subtype_index = 2

        # Difficulty-based action type masking
        if self.difficulty == "easy":
            # Mask research
            action_type_mask = [False, False, False, False, True, False]
        elif self.difficulty == "medium":
            # Mask only wait
            action_type_mask = [False, False, False, False, False, False]
        else:
            # No masking
            action_type_mask = [False, False, False, False, False, False]

        self.register_buffer(
            "action_type_mask",
            th.tensor(action_type_mask, dtype=th.bool, device=self.device)
        )

        # Subtype masks for each type
        self.register_buffer(
            "ride_mask",
            th.tensor(
                [True, True, True, False, False, False, False, False, False],
                dtype=th.bool,
                device=self.device,
            ),
        )
        self.register_buffer(
            "shop_mask",
            th.tensor(
                [False, False, False, True, True, True, False, False, False],
                dtype=th.bool,
                device=self.device,
            ),
        )
        self.register_buffer(
            "staff_mask",
            th.tensor(
                [False, False, False, False, False, False, True, True, True],
                dtype=th.bool,
                device=self.device,
            ),
        )

        # Building costs for affordability checks
        ride_building_costs = th.zeros(3, 4, dtype=th.long, device=self.device)
        shop_building_costs = th.zeros(3, 4, dtype=th.long, device=self.device)

        ride_subtypes = ["carousel", "ferris_wheel", "roller_coaster"]
        shop_subtypes = ["drink", "food", "specialty"]
        subclasses = ["yellow", "blue", "green", "red"]

        for local_idx, subtype_name in enumerate(ride_subtypes):
            for subclass_idx, subclass_name in enumerate(subclasses):
                cost = MAP_CONFIG["rides"][subtype_name][subclass_name]["building_cost"]
                ride_building_costs[local_idx, subclass_idx] = cost

        for local_idx, subtype_name in enumerate(shop_subtypes):
            for subclass_idx, subclass_name in enumerate(subclasses):
                cost = MAP_CONFIG["shops"][subtype_name][subclass_name]["building_cost"]
                shop_building_costs[local_idx, subclass_idx] = cost

        self.register_buffer("ride_building_costs", ride_building_costs)
        self.register_buffer("shop_building_costs", shop_building_costs)

        # Minimum costs per type/subtype
        min_ride_cost = ride_building_costs.min()
        min_shop_cost = shop_building_costs.min()
        self.register_buffer(
            "min_type_costs",
            th.tensor([min_ride_cost, min_shop_cost, 0], dtype=th.long, device=self.device),
        )

        min_ride_subtype_costs = ride_building_costs.min(dim=1)[0]  # [3]
        min_shop_subtype_costs = shop_building_costs.min(dim=1)[0]  # [3]
        staff_costs = th.zeros(3, dtype=th.long, device=self.device)
        self.register_buffer(
            "min_subtype_costs",
            th.cat([min_ride_subtype_costs, min_shop_subtype_costs, staff_costs]),
        )

    def forward(self, obs, deterministic: bool = False):
        """
        Forward pass with hierarchical sampling and simplified masking.

        Masking order:
        1. Action type: difficulty + entity existence + revenue
        2. Type: affordability (place) + existence (move/remove/modify)
        3. Subtype: static (ride/shop/staff) + affordability (place) + existence (mrm)
        4. Subclass: affordability (place) + existence (mrm)
        """
        # Feature extraction
        features = self.extract_features(obs)
        latent_pi, latent_vf = self.mlp_extractor(features)
        values = self.value_net(latent_vf).flatten()  # ensure shape [B]

        action_logits = self.action_net(latent_pi)  # [B, sum(nvec)]
        batch_size = action_logits.shape[0]

        actions = th.zeros(batch_size, len(self.nvec), dtype=th.long, device=self.device)

        type_logits = action_logits[:, : self.nvec[0]]          # [B, 5]
        param_logits_flat = action_logits[:, self.nvec[0] :]    # [B, 3+9+4]

        # Sample action type
        type_dist = self.get_valid_action_type_dist(type_logits, obs)
        action_type = type_dist.mode() if deterministic else type_dist.sample()  # [B]
        actions[:, 0] = action_type
        log_prob = type_dist.log_prob(action_type)  # [B]

        # Param mask by action type
        param_mask = self.param_mask_by_type[action_type]  # [B, 3]

        # Mask modify-staff subtype
        staff_mask = (action_type == 3)
        full_mask = th.zeros_like(param_logits_flat, dtype=th.bool)
        full_mask[staff_mask, self.staff_subtype_index] = True
        param_logits_flat = param_logits_flat.masked_fill(full_mask, -1e9)

        # Split param logits: [type(3), subtype(9), subclass(4)]
        param_logits_list = list(
            th.split(param_logits_flat, self.nvec[1:].tolist(), dim=1)
        )

        # Existence cache for move/remove/modify
        move_remove_modify = (action_type >= 1) & (action_type <= 3)
        existence_cache = None
        if move_remove_modify.any():
            existence_cache = get_entity_existence_cache_simple(
                obs["rides_vector"],
                obs["shops_vector"],
                obs["staff_vector"],
            )

        # === HIERARCHICAL PARAMETER SAMPLING ===
        for dim_idx, n in enumerate(self.nvec[1:]):
            logits_dim = param_logits_list[dim_idx]

            # TYPE PARAM: affordability + existence
            if dim_idx == self.param_to_dim_mapping["type"]:
                place = action_type == 0

                # Affordability for place
                if place.any():
                    money = denormalize_with_config_tensor(
                        obs["park_vector"][:, 3], "money"
                    )
                    money_expanded = money.unsqueeze(1)
                    cannot_afford = money_expanded < self.min_type_costs.unsqueeze(0)  # [B,3]
                    type_mask = cannot_afford & place.unsqueeze(1)
                    param_logits_list[dim_idx] = param_logits_list[dim_idx].masked_fill(
                        type_mask, -1e9
                    )
                    logits_dim = param_logits_list[dim_idx]

                # Type existence for move/remove/modify
                if move_remove_modify.any() and existence_cache is not None:
                    type_existence = existence_cache["type"]  # [B,3]
                    existence_mask = ~type_existence & move_remove_modify.unsqueeze(1)
                    param_logits_list[dim_idx] = param_logits_list[dim_idx].masked_fill(
                        existence_mask, -1e9
                    )
                    logits_dim = param_logits_list[dim_idx]

            # Sample this dimension
            dim_dist = CategoricalDistribution(int(n)).proba_distribution(logits_dim)
            selected_param = dim_dist.mode() if deterministic else dim_dist.sample()
            actions[:, 1 + dim_idx] = selected_param

            # After sampling TYPE: mask SUBTYPE
            if dim_idx == self.param_to_dim_mapping["type"]:
                selected_type = selected_param  # [B]

                # Static subtype mask (ride/shop/staff)
                subtype_mask = (
                    (selected_type == 0).unsqueeze(1) & ~self.ride_mask.unsqueeze(0)
                ) | (
                    (selected_type == 1).unsqueeze(1) & ~self.shop_mask.unsqueeze(0)
                ) | (
                    (selected_type == 2).unsqueeze(1) & ~self.staff_mask.unsqueeze(0)
                )

                subtype_dim = self.param_to_dim_mapping["subtype"]
                param_logits_list[subtype_dim] = param_logits_list[subtype_dim].masked_fill(
                    subtype_mask, -1e9
                )

                # Subtype affordability (place only, attractions only)
                place = action_type == 0
                needs_affordability = place & (selected_type < 2)
                if needs_affordability.any():
                    money = denormalize_with_config_tensor(
                        obs["park_vector"][:, 3], "money"
                    )
                    money_expanded = money.unsqueeze(1)
                    cannot_afford_subtype = (
                        money_expanded < self.min_subtype_costs.unsqueeze(0)
                    )  # [B,9]
                    affordability_mask = (
                        cannot_afford_subtype & needs_affordability.unsqueeze(1)
                    )
                    param_logits_list[subtype_dim] = param_logits_list[subtype_dim].masked_fill(
                        affordability_mask, -1e9
                    )

                # Subtype existence (move/remove/modify)
                if move_remove_modify.any() and existence_cache is not None:
                    subtype_existence = existence_cache["subtype"]  # [B,9]
                    existence_mask = ~subtype_existence & move_remove_modify.unsqueeze(1)
                    param_logits_list[subtype_dim] = param_logits_list[subtype_dim].masked_fill(
                        existence_mask, -1e9
                    )

            # After sampling SUBTYPE: mask SUBCLASS
            if dim_idx == self.param_to_dim_mapping["subtype"]:
                selected_subtype = selected_param  # [B]

                # Subclass affordability (place only, attractions only)
                place = action_type == 0
                is_ride = selected_type == 0
                is_shop = selected_type == 1
                needs_affordability = place & (is_ride | is_shop)

                if needs_affordability.any():
                    money = denormalize_with_config_tensor(
                        obs["park_vector"][:, 3], "money"
                    )
                    # Map global subtype idx to local (0..2) for rides/shops
                    local_subtype = th.where(
                        is_shop, selected_subtype - 3, selected_subtype
                    ).clamp(0, 2)

                    building_costs_rides = self.ride_building_costs[local_subtype, :]
                    building_costs_shops = self.shop_building_costs[local_subtype, :]

                    building_costs = th.where(
                        is_ride.unsqueeze(1),
                        building_costs_rides,
                        building_costs_shops,
                    )

                    money_expanded = money.unsqueeze(1)
                    cannot_afford = money_expanded < building_costs  # [B,4]
                    affordability_mask = cannot_afford & needs_affordability.unsqueeze(1)

                    subclass_dim = self.param_to_dim_mapping["subclass"]
                    param_logits_list[subclass_dim] = param_logits_list[subclass_dim].masked_fill(
                        affordability_mask, -1e9
                    )

                # Subclass existence (move/remove/modify)
                if move_remove_modify.any() and existence_cache is not None:
                    batch_indices = th.arange(batch_size, device=self.device)
                    subclass_existence = existence_cache["subclass"][
                        batch_indices, selected_subtype, :
                    ]  # [B,4]
                    existence_mask = ~subclass_existence & move_remove_modify.unsqueeze(1)
                    subclass_dim = self.param_to_dim_mapping["subclass"]
                    param_logits_list[subclass_dim] = param_logits_list[subclass_dim].masked_fill(
                        existence_mask, -1e9
                    )

        # Zero out unused params in returned actions
        actions[:, 1:] = actions[:, 1:] * param_mask.long()

        # Compute param log_prob with masking
        for dim_idx, (n, logits_dim) in enumerate(zip(self.nvec[1:], param_logits_list)):
            log_probs_dim = F.log_softmax(logits_dim, dim=1)
            acts_dim = actions[:, dim_idx + 1].unsqueeze(-1)
            lp_dim = log_probs_dim.gather(1, acts_dim).squeeze(1)
            lp_dim = lp_dim * param_mask[:, dim_idx].float()
            log_prob += lp_dim

        return actions, values, log_prob

    def get_valid_action_type_dist(self, type_logits: th.Tensor, obs: Dict[str, Any]):
        """
        Apply static + dynamic feasibility masks to type logits and return a Categorical dist.
        """
        # Static difficulty mask
        type_logits = type_logits.masked_fill(self.action_type_mask, -1e9)

        mask = th.zeros_like(type_logits, dtype=th.bool)

        money = denormalize_with_config_tensor(obs["park_vector"][:, 3], "money")
        revenue = denormalize_with_config_tensor(obs["park_vector"][:, 4], "revenue")

        # Check if any entities exist (ride counts + shop counts + staff counts)
        total_rides = denormalize_with_config_tensor(obs["rides_vector"][:, :12].sum(dim=1), "num_rides")
        total_shops = denormalize_with_config_tensor(obs["shops_vector"][:, :12].sum(dim=1), "num_shops")
        num_attractions = total_rides + total_shops

        # Mask move/remove if no entities exist
        mask[:, 1:4] = mask[:, 1:4] | (num_attractions == 0).unsqueeze(1)
        # Mask wait if revenue is 0
        mask[:, 5] = mask[:, 5] | (revenue == 0)

        type_logits = type_logits.masked_fill(mask, -1e9)

        # SAFETY: if an entire row is invalid, fall back to uniform logits (0.0)
        invalid_rows = (type_logits <= -1e8).all(dim=1)
        if invalid_rows.any():
            # clone to avoid unexpected in-place on shared tensors
            type_logits = type_logits.clone()
            type_logits[invalid_rows] = 0.0

        dist = CategoricalDistribution(int(self.nvec[0]))
        return dist.proba_distribution(type_logits)

    def evaluate_actions(self, obs, actions: th.Tensor):
        """
        Evaluate actions with the same masking logic as forward().

        Args:
            obs: batch of observations
            actions: [B, 4] = [action_type, type, subtype, subclass]

        Returns:
            values: [B]
            log_prob: [B]
            entropy: [B]
        """
        features = self.extract_features(obs)
        latent_pi, latent_vf = self.mlp_extractor(features)
        values = self.value_net(latent_vf).flatten()

        action_logits = self.action_net(latent_pi)
        batch_size = action_logits.shape[0]

        type_logits = action_logits[:, : self.nvec[0]]
        param_logits_flat = action_logits[:, self.nvec[0] :]

        action_type = actions[:, 0].long()
        param_actions = actions[:, 1:].long()

        # Keep an original copy of param logits, so we can restore chosen entries if masks overreach
        orig_param_logits_flat = param_logits_flat.clone()
        orig_param_logits_list = list(
            th.split(orig_param_logits_flat, self.nvec[1:].tolist(), dim=1)
        )

        # Type distribution with the same masking as forward
        type_dist = self.get_valid_action_type_dist(type_logits, obs)
        type_log_prob = type_dist.log_prob(action_type)
        type_entropy = type_dist.entropy()

        # Param mask
        param_mask = self.param_mask_by_type[action_type]

        # Existence cache for move/remove/modify
        move_remove_modify = (action_type >= 1) & (action_type <= 3)
        existence_cache = None
        if move_remove_modify.any():
            existence_cache = get_entity_existence_cache_simple(
                obs["rides_vector"],
                obs["shops_vector"],
                obs["staff_vector"],
            )

        # Split param logits
        param_logits_list = list(
            th.split(param_logits_flat, self.nvec[1:].tolist(), dim=1)
        )

        # Dim indices
        type_dim = self.param_to_dim_mapping["type"]
        subtype_dim = self.param_to_dim_mapping["subtype"]
        subclass_dim = self.param_to_dim_mapping["subclass"]

        # Extract selected params
        selected_type = param_actions[:, type_dim]
        selected_subtype = param_actions[:, subtype_dim]
        selected_subclass = param_actions[:, subclass_dim]

        # === TYPE PARAM AFFORDABILITY (mirror forward) ===
        place = action_type == 0
        if place.any():
            money = denormalize_with_config_tensor(
                obs["park_vector"][:, 3], "money"
            )
            money_expanded = money.unsqueeze(1)
            cannot_afford = money_expanded < self.min_type_costs.unsqueeze(0)  # [B,3]
            type_mask = cannot_afford & place.unsqueeze(1)
            param_logits_list[type_dim] = param_logits_list[type_dim].masked_fill(
                type_mask, -1e9
            )

        # === SUBTYPE STATIC MASK & SUBTYPE AFFORDABILITY (mirror forward) ===
        subtype_mask = (
            (selected_type == 0).unsqueeze(1) & ~self.ride_mask.unsqueeze(0)
        ) | (
            (selected_type == 1).unsqueeze(1) & ~self.shop_mask.unsqueeze(0)
        ) | (
            (selected_type == 2).unsqueeze(1) & ~self.staff_mask.unsqueeze(0)
        )
        param_logits_list[subtype_dim] = param_logits_list[subtype_dim].masked_fill(
            subtype_mask, -1e9
        )

        needs_affordability = place & (selected_type < 2)
        if needs_affordability.any():
            money = denormalize_with_config_tensor(
                obs["park_vector"][:, 3], "money"
            )
            money_expanded = money.unsqueeze(1)
            cannot_afford_subtype = (
                money_expanded < self.min_subtype_costs.unsqueeze(0)
            )  # [B,9]
            affordability_mask = (
                cannot_afford_subtype & needs_affordability.unsqueeze(1)
            )
            param_logits_list[subtype_dim] = param_logits_list[subtype_dim].masked_fill(
                affordability_mask, -1e9
            )

        # === EXISTENCE MASKS for type/subtype/subclass (mirror forward) ===
        if move_remove_modify.any() and existence_cache is not None:
            type_existence = existence_cache["type"]       # [B,3]
            subtype_existence = existence_cache["subtype"] # [B,9]
            batch_indices = th.arange(batch_size, device=self.device)
            subclass_existence = existence_cache["subclass"][
                batch_indices, selected_subtype, :
            ]  # [B,4]

            existence_mask_type = ~type_existence & move_remove_modify.unsqueeze(1)
            existence_mask_subtype = ~subtype_existence & move_remove_modify.unsqueeze(1)
            existence_mask_subclass = ~subclass_existence & move_remove_modify.unsqueeze(1)

            param_logits_list[type_dim] = param_logits_list[type_dim].masked_fill(
                existence_mask_type, -1e9
            )
            param_logits_list[subtype_dim] = param_logits_list[subtype_dim].masked_fill(
                existence_mask_subtype, -1e9
            )
            param_logits_list[subclass_dim] = param_logits_list[subclass_dim].masked_fill(
                existence_mask_subclass, -1e9
            )

        # === SUBCLASS AFFORDABILITY (mirror forward) ===
        is_ride = selected_type == 0
        is_shop = selected_type == 1
        needs_affordability = place & (is_ride | is_shop)

        if needs_affordability.any():
            money = denormalize_with_config_tensor(
                obs["park_vector"][:, 3], "money"
            )
            local_subtype = th.where(
                is_shop, selected_subtype - 3, selected_subtype
            ).clamp(0, 2)
            building_costs_rides = self.ride_building_costs[local_subtype, :]
            building_costs_shops = self.shop_building_costs[local_subtype, :]
            building_costs = th.where(
                is_ride.unsqueeze(1), building_costs_rides, building_costs_shops
            )
            money_expanded = money.unsqueeze(1)
            cannot_afford = money_expanded < building_costs  # [B,4]
            affordability_mask = cannot_afford & needs_affordability.unsqueeze(1)
            param_logits_list[subclass_dim] = param_logits_list[subclass_dim].masked_fill(
                affordability_mask, -1e9
            )

        # === SAFETY FIX: ensure the actually-taken actions are never effectively masked out ===
        batch_idx = th.arange(batch_size, device=self.device)

        for dim_idx in range(len(param_logits_list)):
            cur = param_logits_list[dim_idx]       # [B, n_dim], view from split
            orig = orig_param_logits_list[dim_idx] # [B, n_dim], cloned base logits
            chosen = param_actions[:, dim_idx]     # [B]

            # What is the current logit for the chosen index?
            chosen_cur = cur[batch_idx, chosen]

            # Only repair entries that were slammed to a very low value (e.g. -1e9)
            need_fix = chosen_cur <= -1e8
            if not need_fix.any():
                continue

            index_mask = th.zeros_like(cur, dtype=th.bool)
            index_mask[batch_idx[need_fix], chosen[need_fix]] = True

            repaired = th.where(index_mask, orig, cur)
            param_logits_list[dim_idx] = repaired

        # === Final param log_prob + entropy (respecting param_mask) ===
        param_log_prob = th.zeros(batch_size, device=self.device)
        param_entropy = th.zeros(batch_size, device=self.device)

        for dim_idx, (n, logits_dim) in enumerate(zip(self.nvec[1:], param_logits_list)):
            log_probs_dim = F.log_softmax(logits_dim, dim=1)
            probs_dim = log_probs_dim.exp()

            acts_dim = param_actions[:, dim_idx].unsqueeze(-1)
            lp_dim = log_probs_dim.gather(1, acts_dim).squeeze(1)
            ent_dim = -(probs_dim * log_probs_dim).sum(dim=1)

            mask_dim = param_mask[:, dim_idx].float()
            param_log_prob += lp_dim * mask_dim
            param_entropy += ent_dim * mask_dim

        log_prob = type_log_prob + param_log_prob
        entropy = type_entropy + param_entropy

        return values, log_prob, entropy

    def _predict(self, observation, deterministic: bool = False):
        """SB3 hook for prediction."""
        actions, _, _ = self.forward(observation, deterministic=deterministic)
        return actions
