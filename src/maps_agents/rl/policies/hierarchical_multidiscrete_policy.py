"""
Hierarchical Policy for MultiDiscrete Action Spaces.

This policy works with the SB3ActionSpaceWrapper's MultiDiscrete action space
and performs hierarchical sampling to reduce wasted exploration.
"""

from typing import Any, Dict, Tuple
import torch as th
import torch.nn.functional as F
from stable_baselines3.common.policies import MultiInputActorCriticPolicy
from stable_baselines3.common.distributions import CategoricalDistribution
from map_py.observations_and_actions.gym_obs import GRID_CHANNEL_INDICES, NORMALIZATION_CONFIG, MAX_STAFF_PER_TYPE
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
        raise ValueError(f"Normalization max=1.0 for {config_key} is 1.0; cannot invert.")
    if max_value < -1.0:
        raise ValueError(f"Normalization max={max_value} for {config_key} is less than -1.0; cannot invert.")

    max_value_t = th.tensor(max_value, dtype=value.dtype, device=value.device)

    if use_log:
        # y = log1p(x) / log1p(M)  →  log1p(x) = y * log1p(M)  →  x = expm1(...)
        max_value_eff = th.log1p(max_value_t)      # log1p(M)
        unscaled = value * max_value_eff          # log1p(x)
        return th.expm1(unscaled)                 # x
    else:
        # y = x / M → x = y * M
        return value * max_value_t


def get_valid_empty_adjacent_xy_mask(grid) -> Tuple[th.Tensor, th.Tensor]:
    """
    grid: [B, 20, 20, C] tensor
    Returns: [B, 20] bool (valid x) and [B, 20, 20] bool (valid y per x)
    where True indicates a tile that is empty AND adjacent to a path.
    """
    path  = grid[..., GRID_CHANNEL_INDICES['is_path']].bool()
    water = grid[..., GRID_CHANNEL_INDICES['is_water']].bool()
    entr  = grid[..., GRID_CHANNEL_INDICES['is_entrance']].bool()
    exit  = grid[..., GRID_CHANNEL_INDICES['is_exit']].bool()
    yellow = grid[..., GRID_CHANNEL_INDICES['is_yellow']].bool()
    blue = grid[..., GRID_CHANNEL_INDICES['is_blue']].bool()
    green = grid[..., GRID_CHANNEL_INDICES['is_green']].bool()
    red = grid[..., GRID_CHANNEL_INDICES['is_red']].bool()

    empty = ~(path | water | entr | exit | yellow | blue | green | red)

    up    = F.pad(path[:, 1:],    (0, 0, 0, 1))
    down  = F.pad(path[:, :-1],   (0, 0, 1, 0))
    right = F.pad(path[:, :, 1:], (1, 0, 0, 0))
    left  = F.pad(path[:, :, :-1],(0, 1, 0, 0))

    adjacent = up | down | left | right
    valid = empty & adjacent
    valid_xs = valid.any(dim=2)
    valid_ys_per_x = valid
    return valid_xs, valid_ys_per_x


def get_path_xy_mask(grid) -> Tuple[th.Tensor, th.Tensor]:
    """
    grid: [B, 20, 20, C] tensor
    Returns: [B, 20] bool (valid x) and [B, 20, 20] bool (valid y per x)
    where True indicates a tile with a path.
    """
    path = grid[..., GRID_CHANNEL_INDICES['is_path']].bool()
    valid_xs = path.any(dim=2)
    valid_ys_per_x = path
    return valid_xs, valid_ys_per_x


def get_existing_attraction_xy_mask(grid) -> Tuple[th.Tensor, th.Tensor]:
    """
    grid: [B, 20, 20, C] tensor
    Returns: [B, 20] bool (valid x) and [B, 20, 20] bool (valid y per x)
    where True indicates a tile with an attraction.
    """
    yellow = grid[..., GRID_CHANNEL_INDICES['is_yellow']].bool()
    blue = grid[..., GRID_CHANNEL_INDICES['is_blue']].bool()
    green = grid[..., GRID_CHANNEL_INDICES['is_green']].bool()
    red = grid[..., GRID_CHANNEL_INDICES['is_red']].bool()
    valid = yellow | blue | green | red
    valid_xs = valid.any(dim=2)
    valid_ys_per_x = valid
    return valid_xs, valid_ys_per_x


def get_entity_existence_cache(
    grid: th.Tensor,
    staff_vector_summary: th.Tensor,
    ride_channel_indices: th.Tensor,
    shop_channel_indices: th.Tensor,
    subclass_channel_indices: th.Tensor
) -> Dict[str, th.Tensor]:
    """
    Compute all entity existence information at once for efficient reuse.

    Returns:
        Dictionary with:
        - 'type': [B, 3] tensor (ride/shop/staff existence)
        - 'subtype': [B, 9] tensor (all subtype existence)
        - 'subclass': [B, 9, 4] tensor (subtype+color combinations)
    """
    batch_size = grid.shape[0]
    device = grid.device

    # Stack ride channels: carousel, ferris_wheel, roller_coaster → [B, 3, 20, 20]
    ride_matrix = th.stack([grid[:, :, :, idx] for idx in ride_channel_indices], dim=1).bool()

    # Stack shop channels: drink, food, specialty → [B, 3, 20, 20]
    shop_matrix = th.stack([grid[:, :, :, idx] for idx in shop_channel_indices], dim=1).bool()

    # Stack color channels: yellow, blue, green, red → [B, 4, 20, 20]
    color_matrix = th.stack([grid[:, :, :, idx] for idx in subclass_channel_indices], dim=1).bool()

    # staff_vector_summary is [B, 14] where indices 0-11 are counts by type×color
    staff_counts = staff_vector_summary[:, :12]  # [B, 12]
    staff_matrix = staff_counts.view(batch_size, 3, 4)  # [B, 3, 4]

    # TYPE EXISTENCE [B, 3]
    type_exists = th.zeros(batch_size, 3, dtype=th.bool, device=device)
    type_exists[:, 0] = ride_matrix.any(dim=(1, 2, 3))          # rides
    type_exists[:, 1] = shop_matrix.any(dim=(1, 2, 3))          # shops
    type_exists[:, 2] = (staff_counts > 0.0).any(dim=1)         # staff

    # SUBTYPE EXISTENCE [B, 9]
    subtype_exists = th.zeros(batch_size, 9, dtype=th.bool, device=device)
    subtype_exists[:, 0:3] = ride_matrix.any(dim=(2, 3))        # ride subtypes
    subtype_exists[:, 3:6] = shop_matrix.any(dim=(2, 3))        # shop subtypes
    subtype_exists[:, 6:9] = (staff_matrix > 0.0).any(dim=2)    # staff subtypes

    # SUBCLASS EXISTENCE [B, 9, 4]
    subclass_exists = th.zeros(batch_size, 9, 4, dtype=th.bool, device=device)
    ride_color_intersections = ride_matrix.unsqueeze(2) & color_matrix.unsqueeze(1)   # [B,3,4,20,20]
    shop_color_intersections = shop_matrix.unsqueeze(2) & color_matrix.unsqueeze(1)   # [B,3,4,20,20]
    subclass_exists[:, 0:3, :] = ride_color_intersections.any(dim=(3, 4))             # rides
    subclass_exists[:, 3:6, :] = shop_color_intersections.any(dim=(3, 4))             # shops
    subclass_exists[:, 6:9, :] = staff_matrix > 0.0                                    # staff

    return {
        'type': type_exists,        # [B, 3]
        'subtype': subtype_exists,  # [B, 9]
        'subclass': subclass_exists # [B, 9, 4]
    }


def get_filtered_xy_mask(
    grid: th.Tensor,
    selected_type: th.Tensor,
    selected_subtype: th.Tensor,
    selected_subclass: th.Tensor,
    ride_channel_indices: th.Tensor,
    shop_channel_indices: th.Tensor,
    subclass_channel_indices: th.Tensor,
    janitor_vector: th.Tensor,     # [B, 50, 8]
    mechanic_vector: th.Tensor,    # [B, 50, 8]
    specialist_vector: th.Tensor   # [B, 50, 8]
) -> Tuple[th.Tensor, th.Tensor]:
    """
    Get x/y masks for tiles with exact type+subtype+subclass combination.
    Returns [B, 20] and [B, 20, 20] bool tensors for x/y coordinates.
    """
    batch_size = grid.shape[0]
    device = grid.device
    is_ride = (selected_type == 0)
    is_shop = (selected_type == 1)
    is_attraction = is_ride | is_shop

    valid = th.zeros(batch_size, 20, 20, dtype=th.bool, device=device)

    # Attractions: rides/shops
    if is_attraction.any():
        local_subtype = th.where(is_shop, selected_subtype - 3, selected_subtype).clamp(0, 2)
        subtype_channels = th.where(
            is_ride,
            ride_channel_indices[local_subtype],
            shop_channel_indices[local_subtype]
        )  # [B]
        color_channels = subclass_channel_indices[selected_subclass]  # [B]

        batch_indices = th.arange(batch_size, device=device)
        subtype_grids = grid[batch_indices, :, :, subtype_channels].bool()  # [B, 20, 20]
        color_grids = grid[batch_indices, :, :, color_channels].bool()      # [B, 20, 20]
        attraction_valid = subtype_grids & color_grids                      # [B, 20, 20]
        valid[is_attraction] = attraction_valid[is_attraction]

    # Staff entities
    is_staff = (selected_type == 2)
    if is_staff.any():
        all_staff = th.stack([janitor_vector, mechanic_vector, specialist_vector], dim=0)  # [3,B,50,8]
        staff_type_idx = (selected_subtype - 6).clamp(0, 2)                                # [B]
        staff_type_idx_expanded = staff_type_idx.view(1, batch_size, 1, 1).expand(1, batch_size, MAX_STAFF_PER_TYPE, 8)
        selected_staff = th.gather(all_staff, 0, staff_type_idx_expanded).squeeze(0)       # [B,50,8]

        x_coords = selected_staff[:, :, 0]
        y_coords = selected_staff[:, :, 1]
        subclass_ids = selected_staff[:, :, 2]
        salary = selected_staff[:, :, 3]

        x_int = (x_coords * 20.0).round().long().clamp(0, 19)
        y_int = (y_coords * 20.0).round().long().clamp(0, 19)
        subclass_int = (subclass_ids * 4.0).round().long().clamp(0, 3)

        is_valid_staff = (subclass_int == selected_subclass.unsqueeze(1)) & (salary > 0.0)

        valid_staff_flat = th.zeros(batch_size, 400, dtype=th.bool, device=device)
        flat_indices = x_int * 20 + y_int
        batch_indices = th.arange(batch_size, device=device).unsqueeze(1).expand(-1, MAX_STAFF_PER_TYPE)

        batch_flat = batch_indices.flatten()
        indices_flat = flat_indices.flatten()
        mask_flat = is_valid_staff.flatten()

        valid_batch = batch_flat[mask_flat]
        valid_indices = indices_flat[mask_flat]
        valid_staff_flat[valid_batch, valid_indices] = True

        valid_staff_grid = valid_staff_flat.view(batch_size, 20, 20)
        valid[is_staff] = valid_staff_grid[is_staff]

    valid_xs = valid.any(dim=2)
    valid_ys_per_x = valid
    return valid_xs, valid_ys_per_x


class HierarchicalMultiDiscretePolicy(MultiInputActorCriticPolicy):
    """
    Custom policy for hierarchical sampling with MultiDiscrete action spaces.

    Assumes observation_space is Dict, so we inherit directly from
    MultiInputActorCriticPolicy.
    """

    def __init__(self, *args, difficulty: str = "easy", **kwargs):
        super().__init__(*args, **kwargs)

        self.nvec = self.action_space.nvec
        self.difficulty = difficulty

        mask_tensor = th.zeros(int(self.nvec[0]), len(self.nvec) - 1, dtype=th.bool, device=self.device)
        for action_type, mask_list in self.action_space.action_masks.items():
            mask_tensor[action_type, :] = th.tensor(mask_list, dtype=th.bool, device=self.device)
        self.register_buffer("param_mask_by_type", mask_tensor)

        self.staff_subtype_index = 2
        self.dim_to_param_mapping = self.action_space.dim_to_param_mapping
        self.param_to_dim_mapping = self.action_space.param_to_dim_mapping

        if self.difficulty == "easy":
            action_type_mask = [False, False, False, False, True, False, True, True, True, True, False]
        elif self.difficulty == "medium":
            action_type_mask = [False, False, False, False, False, False, True, True, True, True, False]
        else:
            action_type_mask = [False] * int(self.nvec[0])
        self.register_buffer("action_type_mask", th.tensor(action_type_mask, dtype=th.bool))

        self.register_buffer("ride_mask", th.tensor([True, True, True, False, False, False, False, False, False], dtype=th.bool))
        self.register_buffer("shop_mask", th.tensor([False, False, False, True, True, True, False, False, False], dtype=th.bool))
        self.register_buffer("staff_mask", th.tensor([False, False, False, False, False, False, True, True, True], dtype=th.bool))

        # Max price tables
        ride_max_prices = th.zeros(3, 4, dtype=th.long, device=self.device)
        shop_max_prices = th.zeros(3, 4, dtype=th.long, device=self.device)

        ride_subtypes = ["carousel", "ferris_wheel", "roller_coaster"]
        shop_subtypes = ["drink", "food", "specialty"]
        subclasses = ["yellow", "blue", "green", "red"]

        for local_idx, subtype_name in enumerate(ride_subtypes):
            for subclass_idx, subclass_name in enumerate(subclasses):
                max_price = MAP_CONFIG["rides"][subtype_name][subclass_name]["max_ticket_price"]
                ride_max_prices[local_idx, subclass_idx] = max_price
        for local_idx, subtype_name in enumerate(shop_subtypes):
            for subclass_idx, subclass_name in enumerate(subclasses):
                max_price = MAP_CONFIG["shops"][subtype_name][subclass_name]["max_item_price"]
                shop_max_prices[local_idx, subclass_idx] = max_price

        self.register_buffer("ride_max_prices", ride_max_prices)
        self.register_buffer("shop_max_prices", shop_max_prices)

        # Building cost tables
        ride_building_costs = th.zeros(3, 4, dtype=th.long, device=self.device)
        shop_building_costs = th.zeros(3, 4, dtype=th.long, device=self.device)

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

        # Minimum type / subtype costs
        min_ride_cost = ride_building_costs.min()
        min_shop_cost = shop_building_costs.min()
        self.register_buffer("min_type_costs", th.tensor([min_ride_cost, min_shop_cost, 0], dtype=th.long))

        min_ride_subtype_costs = ride_building_costs.min(dim=1)[0]  # [3]
        min_shop_subtype_costs = shop_building_costs.min(dim=1)[0]  # [3]
        staff_costs = th.zeros(3, dtype=th.long, device=self.device)
        self.register_buffer(
            "min_subtype_costs",
            th.cat([min_ride_subtype_costs, min_shop_subtype_costs, staff_costs])
        )

        # Grid channel indices
        self.register_buffer("ride_channel_indices", th.tensor([6, 5, 4], dtype=th.long))
        self.register_buffer("shop_channel_indices", th.tensor([7, 8, 9], dtype=th.long))
        self.register_buffer("subclass_channel_indices", th.tensor([10, 11, 12, 13], dtype=th.long))

    # ---------- Forward: sampling + log_prob ----------

    def forward(self, obs, deterministic: bool = False):
        # Feature extraction
        features = self.extract_features(obs)
        latent_pi, latent_vf = self.mlp_extractor(features)
        values = self.value_net(latent_vf)

        action_logits = self.action_net(latent_pi)  # [B, sum(nvec)]
        batch_size = action_logits.shape[0]
        actions = th.zeros(batch_size, len(self.nvec), dtype=th.long, device=self.device)

        type_logits = action_logits[:, :self.nvec[0]]          # [B, n_type]
        param_logits_flat = action_logits[:, self.nvec[0]:]    # [B, sum(param_nvec)]

        type_dist = self.get_valid_action_type_dist(type_logits, obs)

        action_type = type_dist.mode() if deterministic else type_dist.sample()  # [B]
        actions[:, 0] = action_type
        log_prob = type_dist.log_prob(action_type)                               # [B]

        # Param mask by action type
        param_mask = self.param_mask_by_type[action_type]  # [B, num_param_dims]

        # Mask modify-staff subtype
        mask = (action_type == 3)
        full_mask = th.zeros_like(param_logits_flat, dtype=th.bool)
        full_mask[mask, self.staff_subtype_index] = True
        param_logits_flat = param_logits_flat.masked_fill(full_mask, -1e9)

        # Guest survey affordability
        survey = (action_type == 5)
        if survey.any():
            money = denormalize_with_config_tensor(obs["park_vector"][:, 3], "money")  # [B]
            max_affordable = (money // 500).long()                                     # [B]
            num_guests_offset = sum(self.nvec[1:self.param_to_dim_mapping["num_guests"] + 1].tolist())
            num_guests_values = th.arange(26, device=self.device)                      # [26]
            cannot_afford = num_guests_values.unsqueeze(0) > max_affordable.unsqueeze(1)  # [B,26]
            num_guests_mask = cannot_afford & survey.unsqueeze(1)
            full_mask = th.zeros_like(param_logits_flat, dtype=th.bool)
            full_mask[survey, num_guests_offset:num_guests_offset+26] = num_guests_mask[survey]
            param_logits_flat = param_logits_flat.masked_fill(full_mask, -1e9)

        param_logits_list = list(th.split(param_logits_flat, self.nvec[1:].tolist(), dim=1))

        # Caches
        existence_cache = None
        move_remove_modify_any = ((action_type >= 1) & (action_type <= 3)).any()
        if move_remove_modify_any:
            existence_cache = get_entity_existence_cache(
                obs["grid"],
                obs["staff_vector"],
                self.ride_channel_indices,
                self.shop_channel_indices,
                self.subclass_channel_indices
            )

        grid = obs["grid"]
        path_xs, path_ys_per_x = get_path_xy_mask(grid)
        empty_xs, empty_ys_per_x = get_valid_empty_adjacent_xy_mask(grid)
        existing_attraction_xs, existing_attraction_ys_per_x = get_existing_attraction_xy_mask(grid)
        _ = existing_attraction_xs, existing_attraction_ys_per_x  # currently unused

        for dim_idx, n in enumerate(self.nvec[1:]):
            logits_dim = param_logits_list[dim_idx]

            # TYPE param: type-level affordability + existence
            if dim_idx == self.param_to_dim_mapping["type"]:
                place = (action_type == 0)
                if place.any():
                    money = denormalize_with_config_tensor(obs["park_vector"][:, 3], "money")
                    money_expanded = money.unsqueeze(1)
                    cannot_afford = money_expanded < self.min_type_costs.unsqueeze(0)  # [B,3]
                    type_mask = cannot_afford & place.unsqueeze(1)
                    param_logits_list[dim_idx] = param_logits_list[dim_idx].masked_fill(type_mask, -1e9)
                    logits_dim = param_logits_list[dim_idx]

                move_remove_modify = ((action_type >= 1) & (action_type <= 3))
                if move_remove_modify.any() and existence_cache is not None:
                    type_existence = existence_cache['type']  # [B,3]
                    existence_mask = ~type_existence & move_remove_modify.unsqueeze(1)
                    param_logits_list[dim_idx] = param_logits_list[dim_idx].masked_fill(existence_mask, -1e9)
                    logits_dim = param_logits_list[dim_idx]

            # Sample from this dim
            dim_dist = CategoricalDistribution(n).proba_distribution(logits_dim)
            selected_param = dim_dist.mode() if deterministic else dim_dist.sample()
            actions[:, 1 + dim_idx] = selected_param

            # After sampling TYPE: mask SUBTYPE, subtype-affordability & subtype-existence
            if dim_idx == self.param_to_dim_mapping["type"]:
                selected_type = selected_param

                subtype_mask = (
                    ((selected_type == 0).unsqueeze(1) & ~self.ride_mask.unsqueeze(0)) |
                    ((selected_type == 1).unsqueeze(1) & ~self.shop_mask.unsqueeze(0)) |
                    ((selected_type == 2).unsqueeze(1) & ~self.staff_mask.unsqueeze(0))
                )
                subtype_dim = self.param_to_dim_mapping["subtype"]
                param_logits_list[subtype_dim] = param_logits_list[subtype_dim].masked_fill(subtype_mask, -1e9)

                place = (action_type == 0)
                needs_affordability = place & (selected_type < 2)
                if needs_affordability.any():
                    money = denormalize_with_config_tensor(obs["park_vector"][:, 3], "money")
                    money_expanded = money.unsqueeze(1)
                    cannot_afford_subtype = money_expanded < self.min_subtype_costs.unsqueeze(0)  # [B,9]
                    affordability_mask = cannot_afford_subtype & needs_affordability.unsqueeze(1)
                    param_logits_list[subtype_dim] = param_logits_list[subtype_dim].masked_fill(affordability_mask, -1e9)

                move_remove_modify = ((action_type >= 1) & (action_type <= 3))
                if move_remove_modify.any() and existence_cache is not None:
                    subtype_existence = existence_cache['subtype']  # [B,9]
                    existence_mask = ~subtype_existence & move_remove_modify.unsqueeze(1)
                    param_logits_list[subtype_dim] = param_logits_list[subtype_dim].masked_fill(existence_mask, -1e9)

            # After sampling SUBTYPE: subclass affordability, existence, and X/new_X masks
            if dim_idx == self.param_to_dim_mapping["subtype"]:
                selected_subtype = selected_param

                place = (action_type == 0)
                is_ride = (selected_type == 0)
                is_shop = (selected_type == 1)
                needs_affordability = place & (is_ride | is_shop)

                if needs_affordability.any():
                    money = denormalize_with_config_tensor(obs["park_vector"][:, 3], "money")
                    local_subtype = th.where(is_shop, selected_subtype - 3, selected_subtype).clamp(0, 2)
                    building_costs_rides = self.ride_building_costs[local_subtype, :]
                    building_costs_shops = self.shop_building_costs[local_subtype, :]
                    building_costs = th.where(
                        is_ride.unsqueeze(1), building_costs_rides, building_costs_shops
                    )
                    money_expanded = money.unsqueeze(1)
                    cannot_afford = money_expanded < building_costs  # [B,4]
                    affordability_mask = cannot_afford & needs_affordability.unsqueeze(1)

                    subclass_dim = self.param_to_dim_mapping["subclass"]
                    param_logits_list[subclass_dim] = param_logits_list[subclass_dim].masked_fill(
                        affordability_mask, -1e9
                    )

                move_remove_modify = ((action_type >= 1) & (action_type <= 3))
                if move_remove_modify.any() and existence_cache is not None:
                    batch_indices = th.arange(batch_size, device=self.device)
                    subclass_existence = existence_cache['subclass'][batch_indices, selected_subtype, :]  # [B,4]
                    existence_mask = ~subclass_existence & move_remove_modify.unsqueeze(1)
                    subclass_dim = self.param_to_dim_mapping["subclass"]
                    param_logits_list[subclass_dim] = param_logits_list[subclass_dim].masked_fill(existence_mask, -1e9)

                place = (action_type == 0).unsqueeze(1)
                attractions = (selected_type < 2).unsqueeze(1)
                staff = (selected_type == 2).unsqueeze(1)

                invalid_xs = (
                    (place & attractions & ~empty_xs) |
                    (place & staff & ~path_xs)
                )
                x_dim = self.param_to_dim_mapping["x"]
                param_logits_list[x_dim] = param_logits_list[x_dim].masked_fill(invalid_xs, -1e9)

                move = (action_type == 1).unsqueeze(1)
                invalid_new_xs = (
                    (move & attractions & ~empty_xs) |
                    (move & staff & ~path_xs)
                )
                new_x_dim = self.param_to_dim_mapping["new_x"]
                param_logits_list[new_x_dim] = param_logits_list[new_x_dim].masked_fill(invalid_new_xs, -1e9)

            # After sampling SUBCLASS: filtered_xs/ys, X mask for move/remove/modify, and price mask
            if dim_idx == self.param_to_dim_mapping["subclass"]:
                selected_subclass = selected_param

                filtered_xs, filtered_ys_per_x = get_filtered_xy_mask(
                    obs["grid"],
                    selected_type,
                    selected_subtype,
                    selected_subclass,
                    self.ride_channel_indices,
                    self.shop_channel_indices,
                    self.subclass_channel_indices,
                    obs["janitor_vector"],
                    obs["mechanic_vector"],
                    obs["specialist_vector"]
                )

                move_remove_modify = ((action_type >= 1) & (action_type <= 3))
                if move_remove_modify.any():
                    invalid_xs = ~filtered_xs & move_remove_modify.unsqueeze(1)
                    x_dim = self.param_to_dim_mapping["x"]
                    param_logits_list[x_dim] = param_logits_list[x_dim].masked_fill(invalid_xs, -1e9)

                place_or_modify = ((action_type == 0) | (action_type == 3))
                is_ride = (selected_type == 0)
                is_shop = (selected_type == 1)
                should_mask = place_or_modify & (is_ride | is_shop)

                if should_mask.any():
                    local_subtype = th.where(is_shop, selected_subtype - 3, selected_subtype).clamp(0, 2)
                    max_price_rides = self.ride_max_prices[local_subtype, selected_subclass]
                    max_price_shops = self.shop_max_prices[local_subtype, selected_subclass]
                    max_price = th.where(is_ride, max_price_rides, max_price_shops)
                    price_dim = self.param_to_dim_mapping["price"]
                    price_indices = th.arange(51, device=self.device).unsqueeze(0)  # [1,51]
                    price_mask = price_indices > max_price.unsqueeze(1)            # [B,51]
                    price_mask = price_mask & should_mask.unsqueeze(1)
                    param_logits_list[price_dim] = param_logits_list[price_dim].masked_fill(price_mask, -1e9)

            # After sampling X: mask Y using empties/paths/filtered_ys_per_x
            if dim_idx == self.param_to_dim_mapping["x"]:
                chosen_x = selected_param  # [B]
                batch_indexing = th.arange(batch_size, device=self.device)

                invalid_ys = (
                    (place & attractions & ~empty_ys_per_x[batch_indexing, chosen_x]) |
                    (place & staff & ~path_ys_per_x[batch_indexing, chosen_x]) |
                    (move_remove_modify.unsqueeze(1) & attractions & ~filtered_ys_per_x[batch_indexing, chosen_x]) |
                    (move_remove_modify.unsqueeze(1) & staff & ~filtered_ys_per_x[batch_indexing, chosen_x])
                )
                y_dim = self.param_to_dim_mapping["y"]
                param_logits_list[y_dim] = param_logits_list[y_dim].masked_fill(invalid_ys, -1e9)

            # After sampling new_x: mask new_y similarly
            if dim_idx == self.param_to_dim_mapping["new_x"]:
                selected_new_x = selected_param  # [B]
                invalid_new_ys = (
                    (move & attractions & ~empty_ys_per_x[batch_indexing, selected_new_x]) |
                    (move & staff & ~path_ys_per_x[batch_indexing, selected_new_x])
                )
                new_y_dim = self.param_to_dim_mapping["new_y"]
                param_logits_list[new_y_dim] = param_logits_list[new_y_dim].masked_fill(invalid_new_ys, -1e9)

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

    # ---------- Shared action-type masking ----------

    def get_valid_action_type_dist(self, type_logits: th.Tensor, obs: Dict[str, Any]):
        """
        Apply static + dynamic feasibility masks to type logits and return a Categorical dist.
        """
        # Static difficulty mask
        type_logits = type_logits.masked_fill(self.action_type_mask, -1e9)

        mask = th.zeros_like(type_logits, dtype=th.bool)

        money = denormalize_with_config_tensor(obs["park_vector"][:, 3], "money")
        revenue = denormalize_with_config_tensor(obs["park_vector"][:, 4], "revenue")
        num_attr = (
            denormalize_with_config_tensor(obs["rides_vector"][:, 0], "total_rides") +
            denormalize_with_config_tensor(obs["shops_vector"][:, 0], "total_shops")
        )

        # Mask move/remove/modify if no attractions exist
        mask[:, 1:4] = mask[:, 1:4] | (num_attr == 0).unsqueeze(1)
        # Mask survey if can't afford
        mask[:, 5] = mask[:, 5] | (money < MAP_CONFIG["per_guest_survey_cost"])
        # Mask wait if revenue is 0
        mask[:, 10] = mask[:, 10] | (revenue == 0)

        type_logits = type_logits.masked_fill(mask, -1e9)

        dist = CategoricalDistribution(self.nvec[0])
        return dist.proba_distribution(type_logits)

    # ---------- Evaluate: must match forward's distributions ----------

    def evaluate_actions(self, obs, actions: th.Tensor):
        """
        obs:      batch of observations
        actions:  [B, 1 + num_param_dims]
        Returns:
            values:   [B,]
            log_prob: [B,]
            entropy:  [B,]
        """
        features = self.extract_features(obs)
        latent_pi, latent_vf = self.mlp_extractor(features)
        values = self.value_net(latent_vf).flatten()

        action_logits = self.action_net(latent_pi)
        batch_size = action_logits.shape[0]

        type_logits = action_logits[:, :self.nvec[0]]
        param_logits_flat = action_logits[:, self.nvec[0]:]

        action_type = actions[:, 0].long()
        param_actions = actions[:, 1:].long()

        # Keep an original copy of param logits, so we can restore chosen entries if masks overreach
        orig_param_logits_flat = param_logits_flat.clone()
        orig_param_logits_list = list(th.split(
            orig_param_logits_flat,
            self.nvec[1:].tolist(),
            dim=1,
        ))

        # Type distribution with the same masking as forward
        type_dist = self.get_valid_action_type_dist(type_logits, obs)
        type_log_prob = type_dist.log_prob(action_type)
        type_entropy = type_dist.entropy()

        # Param mask
        param_mask = self.param_mask_by_type[action_type]

        # Mask modify-staff subtype (same as forward)
        staff_mask = (action_type == 3)
        if staff_mask.any():
            full_mask = th.zeros_like(param_logits_flat, dtype=th.bool)
            full_mask[staff_mask, self.staff_subtype_index] = True
            param_logits_flat = param_logits_flat.masked_fill(full_mask, -1e9)

        # Guest survey affordability (same as forward)
        survey = (action_type == 5)
        if survey.any():
            money = denormalize_with_config_tensor(obs["park_vector"][:, 3], "money")
            max_affordable = (money // 50000).long()
            num_guests_offset = sum(self.nvec[1:self.param_to_dim_mapping["num_guests"] + 1].tolist())
            num_guests_values = th.arange(26, device=self.device)
            cannot_afford = num_guests_values.unsqueeze(0) > max_affordable.unsqueeze(1)
            num_guests_mask = cannot_afford & survey.unsqueeze(1)
            full_mask = th.zeros_like(param_logits_flat, dtype=th.bool)
            full_mask[survey, num_guests_offset:num_guests_offset+26] = num_guests_mask[survey]
            param_logits_flat = param_logits_flat.masked_fill(full_mask, -1e9)

        # Existence cache for move/remove/modify
        move_remove_modify = ((action_type >= 1) & (action_type <= 3))
        existence_cache = None
        if move_remove_modify.any():
            existence_cache = get_entity_existence_cache(
                obs["grid"],
                obs["staff_vector"],
                self.ride_channel_indices,
                self.shop_channel_indices,
                self.subclass_channel_indices
            )

        # Coordinate masks
        grid = obs["grid"]
        path_xs, path_ys_per_x = get_path_xy_mask(grid)
        empty_xs, empty_ys_per_x = get_valid_empty_adjacent_xy_mask(grid)

        # Split param logits
        param_logits_list = list(th.split(param_logits_flat, self.nvec[1:].tolist(), dim=1))

        # Dim indices
        type_dim = self.param_to_dim_mapping["type"]
        subtype_dim = self.param_to_dim_mapping["subtype"]
        subclass_dim = self.param_to_dim_mapping["subclass"]
        x_dim = self.param_to_dim_mapping["x"]
        y_dim = self.param_to_dim_mapping["y"]
        new_x_dim = self.param_to_dim_mapping["new_x"]
        new_y_dim = self.param_to_dim_mapping["new_y"]
        price_dim = self.param_to_dim_mapping["price"]

        # Extract selected params
        selected_type = param_actions[:, type_dim]
        selected_subtype = param_actions[:, subtype_dim]
        selected_subclass = param_actions[:, subclass_dim]

        # --- TYPE PARAM AFFORDABILITY (mirror forward) ---
        place = (action_type == 0)
        if place.any():
            money = denormalize_with_config_tensor(obs["park_vector"][:, 3], "money")
            money_expanded = money.unsqueeze(1)
            cannot_afford = money_expanded < self.min_type_costs.unsqueeze(0)  # [B,3]
            type_mask = cannot_afford & place.unsqueeze(1)
            param_logits_list[type_dim] = param_logits_list[type_dim].masked_fill(type_mask, -1e9)

        # --- SUBTYPE STATIC MASK & SUBTYPE AFFORDABILITY (mirror forward) ---
        subtype_mask = (
            ((selected_type == 0).unsqueeze(1) & ~self.ride_mask.unsqueeze(0)) |
            ((selected_type == 1).unsqueeze(1) & ~self.shop_mask.unsqueeze(0)) |
            ((selected_type == 2).unsqueeze(1) & ~self.staff_mask.unsqueeze(0))
        )
        param_logits_list[subtype_dim] = param_logits_list[subtype_dim].masked_fill(subtype_mask, -1e9)

        needs_affordability = place & (selected_type < 2)
        if needs_affordability.any():
            money = denormalize_with_config_tensor(obs["park_vector"][:, 3], "money")
            money_expanded = money.unsqueeze(1)
            cannot_afford_subtype = money_expanded < self.min_subtype_costs.unsqueeze(0)  # [B,9]
            affordability_mask = cannot_afford_subtype & needs_affordability.unsqueeze(1)
            param_logits_list[subtype_dim] = param_logits_list[subtype_dim].masked_fill(affordability_mask, -1e9)

        # --- EXISTENCE MASKS for type/subtype/subclass (mirror forward) ---
        if move_remove_modify.any() and existence_cache is not None:
            type_existence = existence_cache['type']      # [B,3]
            subtype_existence = existence_cache['subtype']# [B,9]
            batch_indices = th.arange(batch_size, device=self.device)
            subclass_existence = existence_cache['subclass'][batch_indices, selected_subtype, :]  # [B,4]

            existence_mask_type = ~type_existence & move_remove_modify.unsqueeze(1)
            existence_mask_subtype = ~subtype_existence & move_remove_modify.unsqueeze(1)
            existence_mask_subclass = ~subclass_existence & move_remove_modify.unsqueeze(1)

            param_logits_list[type_dim] = param_logits_list[type_dim].masked_fill(existence_mask_type, -1e9)
            param_logits_list[subtype_dim] = param_logits_list[subtype_dim].masked_fill(existence_mask_subtype, -1e9)
            param_logits_list[subclass_dim] = param_logits_list[subclass_dim].masked_fill(existence_mask_subclass, -1e9)

        # --- FILTERED X/Y MASKS for move/remove/modify (mirror forward) ---
        filtered_xs, filtered_ys_per_x = get_filtered_xy_mask(
            obs["grid"],
            selected_type,
            selected_subtype,
            selected_subclass,
            self.ride_channel_indices,
            self.shop_channel_indices,
            self.subclass_channel_indices,
            obs["janitor_vector"],
            obs["mechanic_vector"],
            obs["specialist_vector"]
        )

        if move_remove_modify.any():
            invalid_xs = ~filtered_xs & move_remove_modify.unsqueeze(1)
            param_logits_list[x_dim] = param_logits_list[x_dim].masked_fill(invalid_xs, -1e9)

        # --- PLACE X MASK (attractions use empty_xs, staff uses path_xs) ---
        attractions = (selected_type < 2)
        staff = (selected_type == 2)
        invalid_xs_place = (
            (place & attractions).unsqueeze(1) & ~empty_xs |
            (place & staff).unsqueeze(1) & ~path_xs
        )
        param_logits_list[x_dim] = param_logits_list[x_dim].masked_fill(invalid_xs_place, -1e9)

        # --- PLACE Y MASK (using chosen X and empty_ys/path_ys) ---
        if place.any():
            b = th.arange(batch_size, device=self.device)
            chosen_x = param_actions[:, x_dim].clamp(0, empty_ys_per_x.size(1) - 1)
            invalid_ys_place = (
                (place & attractions).unsqueeze(1) & ~empty_ys_per_x[b, chosen_x] |
                (place & staff).unsqueeze(1) & ~path_ys_per_x[b, chosen_x]
            )
            param_logits_list[y_dim] = param_logits_list[y_dim].masked_fill(invalid_ys_place, -1e9)

        # --- MOVE/REMOVE/MODIFY Y MASK (using filtered_ys_per_x) ---
        if move_remove_modify.any():
            b = th.arange(batch_size, device=self.device)
            chosen_x = param_actions[:, x_dim].clamp(0, filtered_ys_per_x.size(1) - 1)
            invalid_ys_mrm = move_remove_modify.unsqueeze(1) & ~filtered_ys_per_x[b, chosen_x]
            param_logits_list[y_dim] = param_logits_list[y_dim].masked_fill(invalid_ys_mrm, -1e9)

        # --- MOVE new_x/new_y MASKS (same as forward) ---
        move = (action_type == 1)
        if move.any():
            invalid_new_xs = (
                (move & attractions).unsqueeze(1) & ~empty_xs |
                (move & staff).unsqueeze(1) & ~path_xs
            )
            param_logits_list[new_x_dim] = param_logits_list[new_x_dim].masked_fill(invalid_new_xs, -1e9)

            b = th.arange(batch_size, device=self.device)
            chosen_new_x = param_actions[:, new_x_dim].clamp(0, empty_ys_per_x.size(1) - 1)
            invalid_new_ys = (
                (move & attractions).unsqueeze(1) & ~empty_ys_per_x[b, chosen_new_x] |
                (move & staff).unsqueeze(1) & ~path_ys_per_x[b, chosen_new_x]
            )
            param_logits_list[new_y_dim] = param_logits_list[new_y_dim].masked_fill(invalid_new_ys, -1e9)

        # --- PRICE MASK for place/modify rides/shops (same as forward) ---
        action_type_eval = action_type
        place_or_modify_eval = ((action_type_eval == 0) | (action_type_eval == 3))
        is_ride_eval = (selected_type == 0)
        is_shop_eval = (selected_type == 1)
        should_mask_eval = place_or_modify_eval & (is_ride_eval | is_shop_eval)

        if should_mask_eval.any():
            local_subtype_eval = th.where(
                is_shop_eval,
                selected_subtype - 3,
                selected_subtype
            ).clamp(0, 2)

            max_price_rides_eval = self.ride_max_prices[
                local_subtype_eval.clamp(0, 2),
                selected_subclass.clamp(0, 3)
            ]
            max_price_shops_eval = self.shop_max_prices[
                local_subtype_eval.clamp(0, 2),
                selected_subclass.clamp(0, 3)
            ]
            max_price_eval = th.where(is_ride_eval, max_price_rides_eval, max_price_shops_eval)

            price_indices_eval = th.arange(51, device=self.device).unsqueeze(0)
            price_mask_eval = price_indices_eval > max_price_eval.unsqueeze(1)
            price_mask_eval = price_mask_eval & should_mask_eval.unsqueeze(1)
            param_logits_list[price_dim] = param_logits_list[price_dim].masked_fill(price_mask_eval, -1e9)


        # --- SAFETY FIX: ensure the actually-taken actions are never effectively masked out ---
        batch_idx = th.arange(batch_size, device=self.device)

        for dim_idx in range(len(param_logits_list)):
            cur = param_logits_list[dim_idx]            # [B, n_dim], view from split
            orig = orig_param_logits_list[dim_idx]      # [B, n_dim], cloned base logits
            chosen = param_actions[:, dim_idx]          # [B]

            # What is the current logit for the chosen index?
            chosen_cur = cur[batch_idx, chosen]

            # Only repair entries that were slammed to a very low value (e.g. -1e9)
            need_fix = chosen_cur <= -1e8
            if not need_fix.any():
                continue

            # Build a boolean mask over [B, n_dim] marking exactly the (b, chosen[b]) positions
            index_mask = th.zeros_like(cur, dtype=th.bool)
            index_mask[batch_idx[need_fix], chosen[need_fix]] = True

            # Out-of-place: for those positions, restore logits from the original unmasked tensor
            repaired = th.where(index_mask, orig, cur)

            # Replace the entry in the list (no in-place on the split view)
            param_logits_list[dim_idx] = repaired

        # --- Final param log_prob + entropy (respecting param_mask) ---
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

    # ---------- SB3 hook ----------

    def _predict(self, observation, deterministic: bool = False):
        actions, _, _ = self.forward(observation, deterministic=deterministic)
        return actions
