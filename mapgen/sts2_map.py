#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""杀戮尖塔 2 地图生成器。

按反编译源码中的算法重建单幕地图：xoshiro256** 随机数、七列网格上的七条路径、
按配额与规则分配节点类型、重复路径段剪枝与修复、以及居中 / 撑开 / 拉直三步后处理。
给定同一种子必得同一张图。

用法：
    python sts2_map.py --seed 12345
    python sts2_map.py 12345 --out my_map.mmd
    python sts2_map.py --seed 12345 --act 2 --print

配置文件默认在脚本同级目录下的 sts2_map.json，不存在时自动生成一份默认配置。
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections import deque

MASK64 = (1 << 64) - 1

# 节点类型的数值编号。剪枝阶段会用它生成路径段的字符键，键的字典序决定剪枝顺序，
# 因此编号必须固定，不能随配置顺序变化。
CANONICAL_TYPE_ORDER = [
    "Unassigned",
    "Unknown",
    "Shop",
    "Treasure",
    "RestSite",
    "Monster",
    "Elite",
    "Boss",
    "Ancient",
]

DEFAULT_CONFIG_NAME = "sts2_map.json"

DEFAULT_CONFIG = {
    "_note": (
        "node_types 里的每个条目描述一种地图节点。count 是这一节点在每张图上的固定个数；"
        "roll 用高斯抽样决定个数，用于还原原作里数量随机的那几类节点；"
        "weight 是权重，给出权重的节点会按权重瓜分前面没有分完的空位。"
        "三者都不写的节点只由 fixed_rows 或 default_type 决定。"
    ),
    "version": 1,
    "seed": 0,
    "map": {
        "columns": 7,
        "rooms": 15,
        "path_count": 7,
        "post_process": True,
        "prune": True,
        "prune_iterations": 3,
        "prune_loop_limit": 50,
        "assignment_passes": 3,
    },
    "rules": {
        "lower_row_threshold": 6,
        "lower_restricted_types": ["RestSite", "Elite"],
        "upper_row_offset": 3,
        "upper_restricted_types": ["RestSite"],
        "no_adjacent_repeat_types": ["Elite", "RestSite", "Treasure", "Shop"],
        "no_sibling_types": ["RestSite", "Monster", "Unknown", "Elite", "Shop"],
        "ignore_rules_types": [],
    },
    "fixed_rows": [
        {"from": "start", "row": 0, "type": "Ancient"},
        {"from": "start", "row": 1, "type": "Monster"},
        {"from": "end", "offset": 7, "type": "Treasure"},
        {"from": "end", "offset": 1, "type": "RestSite"},
    ],
    "start_type": "Ancient",
    "end_type": "Boss",
    "default_type": "Monster",
    "node_types": [
        {"id": "Ancient", "label": "远古", "color": "#c9a227"},
        {"id": "Boss", "label": "首领", "color": "#8b2b2b"},
        {"id": "Monster", "label": "怪物", "color": "#5b6b73"},
        {"id": "Elite", "label": "精英", "color": "#a0453f", "count": 5},
        {
            "id": "RestSite",
            "label": "休息",
            "color": "#3f8f6a",
            "roll": {"kind": "gaussian_int", "mean": 7, "stddev": 1, "min": 6, "max": 7},
        },
        {"id": "Shop", "label": "商店", "color": "#c98b2e", "count": 3},
        {"id": "Treasure", "label": "宝箱", "color": "#c9b02e"},
        {
            "id": "Unknown",
            "label": "未知",
            "color": "#7d8fa8",
            "roll": {"kind": "gaussian_int", "mean": 12, "stddev": 1, "min": 10, "max": 14},
        },
    ],
}


# --------------------------------------------------------------------------
# 随机数：与源码中的 MegaRandom 一致的 xoshiro256**，用 splitmix64 播种
# --------------------------------------------------------------------------


def _rotl(x: int, k: int) -> int:
    return ((x << k) | (x >> (64 - k))) & MASK64


def _splitmix64(state: int) -> tuple[int, int]:
    state = (state + 0x9E3779B97F4A7C15) & MASK64
    z = state
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & MASK64
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & MASK64
    return (z ^ (z >> 31)) & MASK64, state


class MegaRandom:
    """xoshiro256**。"""

    __slots__ = ("s0", "s1", "s2", "s3")

    def __init__(self, seed: int) -> None:
        seed &= MASK64
        seed, self.s0 = _splitmix64(seed)
        seed, self.s1 = _splitmix64(seed)
        seed, self.s2 = _splitmix64(seed)
        seed, self.s3 = _splitmix64(seed)

    def next_uint64(self) -> int:
        s0, s1, s2, s3 = self.s0, self.s1, self.s2, self.s3
        result = (_rotl((s1 * 5) & MASK64, 7) * 9) & MASK64
        t = (s1 << 17) & MASK64
        s2 ^= s0
        s3 ^= s1
        s1 ^= s2
        s0 ^= s3
        s2 ^= t
        s3 = _rotl(s3, 45)
        self.s0, self.s1, self.s2, self.s3 = s0 & MASK64, s1 & MASK64, s2 & MASK64, s3 & MASK64
        return result

    def next_double(self) -> float:
        return (self.next_uint64() >> 11) * (2.0 ** -53)


class Rng:
    """带调用计数的随机数封装，方法语义与源码中的 Rng 一致。"""

    __slots__ = ("_r", "counter")

    def __init__(self, seed: int) -> None:
        self._r = MegaRandom(seed)
        self.counter = 0

    def next_int(self, max_exclusive: int) -> int:
        self.counter += 1
        return int(self._r.next_double() * max_exclusive)

    def next_int_range(self, min_inclusive: int, max_exclusive: int) -> int:
        if min_inclusive >= max_exclusive:
            raise ValueError("minInclusive must be lower than maxExclusive")
        self.counter += 1
        return int(self._r.next_double() * (max_exclusive - min_inclusive)) + min_inclusive

    def next_double(self) -> float:
        self.counter += 1
        return self._r.next_double()

    def next_gaussian_int(self, mean: int, stddev: int, low: int, high: int) -> int:
        if low > high:
            raise ValueError("min must not exceed max")
        while True:
            d = 1.0 - self.next_double()
            u = 1.0 - self.next_double()
            z = math.sqrt(-2.0 * math.log(d)) * math.sin(math.pi * 2.0 * u)
            value = round(mean + stddev * z)
            if low <= value <= high:
                return value

    def shuffle(self, seq: list) -> list:
        """原地 Fisher-Yates，与源码的 UnstableShuffle 一致。"""
        n = len(seq)
        while n > 1:
            n -= 1
            k = self.next_int(n + 1)
            seq[k], seq[n] = seq[n], seq[k]
        return seq


def stable_shuffle(seq: list, rng: Rng, key=None) -> list:
    """先按键排序再洗牌，与源码的 StableShuffle 一致，结果与初始顺序无关。"""
    seq.sort(key=key)
    return rng.shuffle(seq)


# --------------------------------------------------------------------------
# 节点与有序集合
# --------------------------------------------------------------------------


class OrderedSet:
    """保持插入顺序的集合，用来模拟源码里 HashSet 的枚举顺序。"""

    __slots__ = ("_d",)

    def __init__(self, items=()) -> None:
        self._d = {item: None for item in items}

    def add(self, item) -> None:
        self._d[item] = None

    def discard(self, item) -> None:
        self._d.pop(item, None)

    def first(self):
        for item in self._d:
            return item
        return None

    def __contains__(self, item) -> bool:
        return item in self._d

    def __iter__(self):
        return iter(self._d)

    def __len__(self) -> int:
        return len(self._d)


class MapPoint:
    """地图上的一个节点。"""

    __slots__ = ("col", "row", "point_type", "children", "parents", "can_be_modified")

    def __init__(self, col: int, row: int) -> None:
        self.col = col
        self.row = row
        self.point_type = "Unassigned"
        self.children = OrderedSet()
        self.parents = OrderedSet()
        self.can_be_modified = True

    def add_child(self, child: "MapPoint") -> None:
        self.children.add(child)
        child.parents.add(self)

    def remove_child(self, child: "MapPoint") -> None:
        self.children.discard(child)
        child.parents.discard(self)

    def __repr__(self) -> str:  # pragma: no cover - 仅调试用
        return f"MapPoint({self.col},{self.row},{self.point_type})"


# --------------------------------------------------------------------------
# 地图
# --------------------------------------------------------------------------


class MapGenerationError(RuntimeError):
    pass


class ActMap:
    def __init__(self, cfg: dict, seed: int) -> None:
        self.cfg = cfg
        self.seed = seed
        self.rng = Rng(seed)

        m = cfg["map"]
        self.columns = int(m.get("columns", 7))
        self.rooms = int(m.get("rooms", 15))
        self.path_count = int(m.get("path_count", 7))
        self.map_length = self.rooms + 1
        self.do_prune = bool(m.get("prune", True))
        self.do_post = bool(m.get("post_process", True))
        self.prune_iterations = int(m.get("prune_iterations", 3))
        self.prune_loop_limit = int(m.get("prune_loop_limit", 50))
        self.assignment_passes = int(m.get("assignment_passes", 3))

        self.rules = cfg["rules"]
        self.type_order = self._build_type_order(cfg)
        self.rolls = self._roll_counts(cfg)

        self.grid: list[list[MapPoint | None]] = [
            [None] * self.map_length for _ in range(self.columns)
        ]
        self.start_points: OrderedSet = OrderedSet()

        self.start = MapPoint(self.columns // 2, 0)
        self.boss = MapPoint(self.columns // 2, self.map_length)

    # -- 配置 ---------------------------------------------------------------

    @staticmethod
    def _build_type_order(cfg: dict) -> dict[str, int]:
        order = {name: i for i, name in enumerate(CANONICAL_TYPE_ORDER)}
        nxt = len(order)
        for entry in cfg["node_types"]:
            tid = entry["id"]
            if tid not in order:
                order[tid] = nxt
                nxt += 1
        return order

    def _roll_counts(self, cfg: dict) -> dict[str, int]:
        """按 node_types 的出现顺序结算固定个数与随机个数。"""
        counts: dict[str, int] = {}
        for entry in cfg["node_types"]:
            tid = entry["id"]
            if "count" in entry:
                counts[tid] = int(entry["count"])
            elif "roll" in entry:
                spec = entry["roll"]
                if spec.get("kind") != "gaussian_int":
                    raise ValueError(f"未知的 roll 类型：{spec.get('kind')}")
                counts[tid] = self.rng.next_gaussian_int(
                    int(spec["mean"]), int(spec["stddev"]), int(spec["min"]), int(spec["max"])
                )
        return counts

    def weighted_remainder(self, already_demanded: int) -> list[str]:
        """把权重类型的份额换算成具体个数，用于填补固定个数没有占满的空位。

        没有配置任何权重时返回空列表，此时空位全部落到 default_type，
        与原作的行为一致。
        """
        weighted = [
            (entry["id"], float(entry["weight"]))
            for entry in self.cfg["node_types"]
            if "weight" in entry
        ]
        if not weighted:
            return []
        assignable = sum(1 for p in self.all_points() if p.point_type == "Unassigned")
        leftover = assignable - already_demanded
        if leftover <= 0:
            return []
        total = sum(w for _, w in weighted)
        if total <= 0:
            return []
        exact = [(tid, w / total * leftover) for tid, w in weighted]
        counts = {tid: int(math.floor(v)) for tid, v in exact}
        missing = leftover - sum(counts.values())
        order = [tid for tid, _ in weighted]
        ranked = sorted(exact, key=lambda kv: (-(kv[1] - math.floor(kv[1])), order.index(kv[0])))
        for i in range(missing):
            counts[ranked[i % len(ranked)][0]] += 1
        result: list[str] = []
        for tid, _ in weighted:
            result.extend([tid] * counts[tid])
        return result

    # -- 网格访问 -----------------------------------------------------------

    def get_point(self, col: int, row: int) -> MapPoint | None:
        if col == self.boss.col and row == self.boss.row:
            return self.boss
        if col == self.start.col and row == self.start.row:
            return self.start
        if 0 <= col < self.columns and 0 <= row < self.map_length:
            return self.grid[col][row]
        return None

    def get_or_create_point(self, col: int, row: int) -> MapPoint:
        point = self.get_point(col, row)
        if point is not None:
            return point
        point = MapPoint(col, row)
        self.grid[col][row] = point
        return point

    def row_points(self, row: int) -> list[MapPoint]:
        return [self.grid[c][row] for c in range(self.columns) if self.grid[c][row] is not None]

    def all_points(self) -> list[MapPoint]:
        out = []
        for c in range(self.columns):
            for r in range(self.map_length):
                if self.grid[c][r] is not None:
                    out.append(self.grid[c][r])
        return out

    # -- 路径生成 -----------------------------------------------------------

    def has_invalid_crossover(self, current: MapPoint, target_col: int) -> bool:
        delta = target_col - current.col
        if delta == 0:
            return False
        neighbour = self.grid[target_col][current.row]
        if neighbour is None:
            return False
        for child in neighbour.children:
            if child.col - neighbour.col == -delta:
                return True
        return False

    def generate_next_coord(self, current: MapPoint) -> tuple[int, int]:
        col = current.col
        low = max(0, col - 1)
        high = min(col + 1, self.columns - 1)
        directions = [-1, 0, 1]
        stable_shuffle(directions, self.rng)
        for d in directions:
            target = {-1: low, 0: col, 1: high}[d]
            if not self.has_invalid_crossover(current, target):
                return target, current.row + 1
        raise MapGenerationError("找不到可用的下一个节点")

    def path_generate(self, starting: MapPoint) -> None:
        point = starting
        while point.row < self.map_length - 1:
            col, row = self.generate_next_coord(point)
            nxt = self.get_or_create_point(col, row)
            point.add_child(nxt)
            point = nxt

    def generate_map(self) -> None:
        for i in range(self.path_count):
            point = self.get_or_create_point(self.rng.next_int_range(0, self.columns), 1)
            if i == 1:
                while point in self.start_points:
                    point = self.get_or_create_point(self.rng.next_int_range(0, self.columns), 1)
            self.start_points.add(point)
            self.path_generate(point)

        for point in self.row_points(self.map_length - 1):
            point.add_child(self.boss)
        for point in self.row_points(1):
            self.start.add_child(point)

    # -- 节点类型 -----------------------------------------------------------

    def _fixed_row_index(self, spec: dict) -> int:
        if spec["from"] == "start":
            return int(spec["row"])
        if spec["from"] == "end":
            return self.map_length - int(spec["offset"])
        raise ValueError(f"未知的 fixed_rows 来源：{spec['from']}")

    def assign_point_types(self) -> None:
        for spec in self.cfg["fixed_rows"]:
            for point in self.row_points(self._fixed_row_index(spec)):
                point.point_type = spec["type"]
                point.can_be_modified = False

        queue: deque[str] = deque()
        demanded = 0
        for entry in self.cfg["node_types"]:
            tid = entry["id"]
            amount = self.rolls.get(tid, 0)
            demanded += amount
            for _ in range(amount):
                queue.append(tid)
        for tid in self.weighted_remainder(demanded):
            queue.append(tid)
        self._assign_remaining_types(queue)

        default_type = self.cfg["default_type"]
        for point in self.all_points():
            if point.point_type == "Unassigned":
                point.point_type = default_type

        self.boss.point_type = self.cfg["end_type"]
        self.start.point_type = self.cfg["start_type"]

    def _assign_remaining_types(self, queue: deque[str]) -> None:
        for _ in range(self.assignment_passes):
            if not queue:
                break
            candidates = [p for p in self.all_points() if p.point_type == "Unassigned"]
            stable_shuffle(candidates, self.rng, key=lambda p: (p.col, p.row))
            for point in candidates:
                if not queue:
                    break
                point.point_type = self._next_valid_type(queue, point)

    def _next_valid_type(self, queue: deque[str], point: MapPoint) -> str:
        for _ in range(len(queue)):
            tid = queue.popleft()
            if tid in self.rules.get("ignore_rules_types", []):
                return tid
            if self.is_valid_point_type(tid, point):
                return tid
            queue.append(tid)
        return "Unassigned"

    # -- 放置规则 -----------------------------------------------------------

    def is_valid_point_type(self, tid: str, point: MapPoint) -> bool:
        return (
            self._valid_for_lower(tid, point)
            and self._valid_for_upper(tid, point)
            and self._valid_with_parents(tid, point)
            and self._valid_with_children(tid, point)
            and self._valid_with_siblings(tid, point)
        )

    def _valid_for_lower(self, tid: str, point: MapPoint) -> bool:
        if point.row < int(self.rules["lower_row_threshold"]):
            return tid not in self.rules["lower_restricted_types"]
        return True

    def _valid_for_upper(self, tid: str, point: MapPoint) -> bool:
        if point.row >= self.map_length - int(self.rules["upper_row_offset"]):
            return tid not in self.rules["upper_restricted_types"]
        return True

    def _valid_with_parents(self, tid: str, point: MapPoint) -> bool:
        if tid in self.rules["no_adjacent_repeat_types"]:
            for other in list(point.parents) + list(point.children):
                if other is not point and other.point_type == tid:
                    return False
        return True

    def _valid_with_children(self, tid: str, point: MapPoint) -> bool:
        if tid in self.rules["no_adjacent_repeat_types"]:
            for child in point.children:
                if child.point_type == tid:
                    return False
        return True

    def _valid_with_siblings(self, tid: str, point: MapPoint) -> bool:
        if tid in self.rules["no_sibling_types"]:
            for sibling in self.siblings(point):
                if sibling.point_type == tid:
                    return False
        return True

    @staticmethod
    def siblings(point: MapPoint) -> list[MapPoint]:
        out = []
        for parent in point.parents:
            for child in parent.children:
                if child is not point:
                    out.append(child)
        return out


# --------------------------------------------------------------------------
# 剪枝：删除节点类型序列完全相同的重复路径段，再把掉到配额以下的类型补回来
# --------------------------------------------------------------------------


def is_in_grid(map_: ActMap, point: MapPoint) -> bool:
    if point.row < 0 or point.row >= map_.map_length or point.col < 0 or point.col >= map_.columns:
        return point.point_type == map_.cfg["end_type"]
    if map_.grid[point.col][point.row] is None and point.point_type != map_.cfg["start_type"]:
        return point.point_type == map_.cfg["end_type"]
    return True


def is_removed(map_: ActMap, point: MapPoint) -> bool:
    if point.row < 0 or point.row >= map_.map_length or point.col < 0 or point.col >= map_.columns:
        return True
    return map_.grid[point.col][point.row] is None


def remove_point(map_: ActMap, point: MapPoint) -> None:
    map_.grid[point.col][point.row] = None
    map_.start_points.discard(point)
    for child in list(point.children):
        point.remove_child(child)
    for parent in list(point.parents):
        parent.remove_child(point)


def find_all_paths(point: MapPoint, end_type: str) -> list[list[MapPoint]]:
    if point.point_type == end_type:
        return [[point]]
    result: list[list[MapPoint]] = []
    for child in point.children:
        for sub in find_all_paths(child, end_type):
            result.append([point] + sub)
    return result


def segment_key(segment: list[MapPoint], type_order: dict[str, int]) -> str:
    first, last = segment[0], segment[-1]
    if first.row == 0:
        prefix = f"{first.row}-{last.col},{last.row}-"
    else:
        prefix = f"{first.col},{first.row}-{last.col},{last.row}-"
    return prefix + ",".join(str(type_order.get(p.point_type, 0)) for p in segment)


def overlapping(a: list[MapPoint], b: list[MapPoint]) -> bool:
    if len(a) < 3 or len(b) < 3:
        return False
    for i in range(1, len(a) - 1):
        if a[i] is b[i]:
            return True
    return False


def find_matching_segments(map_: ActMap) -> list[list[list[MapPoint]]]:
    paths = find_all_paths(map_.start, map_.cfg["end_type"])
    segments: dict[str, list[list[MapPoint]]] = {}
    for path in paths:
        for i in range(len(path) - 1):
            start_point = path[i]
            if len(start_point.children) <= 1 and start_point.row != 0:
                continue
            for j in range(2, len(path) - i):
                end_point = path[i + j]
                if len(end_point.parents) < 2:
                    continue
                segment = path[i : i + j + 1]
                key = segment_key(segment, map_.type_order)
                bucket = segments.get(key)
                if bucket is None:
                    segments[key] = [segment]
                elif not any(overlapping(existing, segment) for existing in bucket):
                    bucket.append(segment)
    return [segments[k] for k in sorted(segments.keys()) if len(segments[k]) > 1]


def prune_segment(map_: ActMap, segment: list[MapPoint]) -> bool:
    result = False
    for i in range(len(segment) - 1):
        point = segment[i]
        if not is_in_grid(map_, point):
            return True
        if len(point.children) > 1 or len(point.parents) > 1:
            continue
        if any(len(p.children) == 1 and not is_removed(map_, p) for p in point.parents):
            continue
        source = segment[i:]
        if any(len(n.children) > 1 and len(n.parents) == 1 for n in source):
            continue
        if len(segment[-1].parents) == 1:
            return False
        if not any(len(c.parents) == 1 for c in point.children if c not in segment):
            remove_point(map_, point)
            result = True
    return result


def prune_all_but_last(map_: ActMap, matches: list[list[MapPoint]]) -> int:
    count = 0
    for match in matches:
        if count == len(matches) - 1:
            return count
        if prune_segment(map_, match):
            count += 1
    return count


def break_parent_child_in_segment(segment: list[MapPoint]) -> bool:
    result = False
    for i in range(len(segment) - 1):
        point = segment[i]
        if len(point.children) >= 2:
            nxt = segment[i + 1]
            if len(nxt.parents) != 1:
                point.remove_child(nxt)
                result = True
    return result


def prune_paths(map_: ActMap, matches: list[list[list[MapPoint]]]) -> bool:
    for bucket in matches:
        map_.rng.shuffle(bucket)
        if prune_all_but_last(map_, bucket) != 0:
            return True
        if any(break_parent_child_in_segment(m) for m in bucket):
            return True
    return False


def prune_duplicate_segments(map_: ActMap) -> None:
    iterations = 0
    matches = find_matching_segments(map_)
    while prune_paths(map_, matches):
        iterations += 1
        if iterations > map_.prune_loop_limit:
            raise MapGenerationError("剪枝迭代次数超出上限")
        matches = find_matching_segments(map_)


def repair_point_type(map_: ActMap, tid: str, target: int) -> bool:
    current = sum(1 for p in map_.all_points() if p.point_type == tid)
    missing = target - current
    if missing <= 0:
        return False
    candidates = [
        p for p in map_.all_points() if p.point_type == map_.cfg["default_type"] and p.can_be_modified
    ]
    stable_shuffle(candidates, map_.rng, key=lambda p: (p.col, p.row))
    result = False
    for point in candidates:
        if missing == 0:
            break
        if map_.is_valid_point_type(tid, point):
            point.point_type = tid
            missing -= 1
            result = True
    return result


def repair_pruned_point_types(map_: ActMap) -> bool:
    changed = False
    for entry in map_.cfg["node_types"]:
        tid = entry["id"]
        if tid not in map_.rolls:
            continue
        changed |= repair_point_type(map_, tid, map_.rolls[tid])
    return changed


def prune_and_repair(map_: ActMap) -> None:
    for _ in range(map_.prune_iterations):
        prune_duplicate_segments(map_)
        if not repair_pruned_point_types(map_):
            break


# --------------------------------------------------------------------------
# 后处理：居中、撑开、拉直
# --------------------------------------------------------------------------


def column_empty(grid: list[list[MapPoint | None]], col: int) -> bool:
    return all(cell is None for cell in grid[col])


def center_grid(grid: list[list[MapPoint | None]], columns: int, rows: int):
    left_empty = column_empty(grid, 0) and column_empty(grid, 1)
    right_empty = column_empty(grid, columns - 1) and column_empty(grid, columns - 2)
    shift = 0
    if left_empty and not right_empty:
        shift = -1
    elif not left_empty and right_empty:
        shift = 1
    if shift == 0:
        return grid
    for row in range(rows):
        order = range(columns - 1, -1, -1) if shift > 0 else range(columns)
        for col in list(order):
            point = grid[col][row]
            grid[col][row] = None
            target = col + shift
            if 0 <= target < columns:
                grid[target][row] = point
                if point is not None:
                    point.col = target
    return grid


def straighten_paths(grid: list[list[MapPoint | None]], columns: int, rows: int):
    for row in range(rows):
        for col in range(columns):
            point = grid[col][row]
            if point is None or len(point.parents) != 1 or len(point.children) != 1:
                continue
            parent = point.parents.first()
            child = point.children.first()
            bend_right = point.col < child.col and point.col < parent.col
            bend_left = point.col > child.col and point.col > parent.col
            if bend_right and col < columns - 1:
                target = col + 1
                if grid[target][row] is not None:
                    continue
                point.col = target
                grid[col][row] = None
                grid[target][row] = point
            if bend_left and col > 0:
                target = col - 1
                if grid[target][row] is None:
                    point.col = target
                    grid[col][row] = None
                    grid[target][row] = point
    return grid


def neighbor_positions(col: int, columns: int) -> set[int]:
    return {c for c in (col - 1, col, col + 1) if 0 <= c < columns}


def allowed_positions(point: MapPoint, columns: int) -> set[int]:
    allowed = set(range(columns))
    for parent in point.parents:
        allowed &= neighbor_positions(parent.col, columns)
    for child in point.children:
        allowed &= neighbor_positions(child.col, columns)
    return allowed


def compute_gap(candidate_col: int, row_nodes: list[MapPoint], point: MapPoint) -> int:
    best = None
    for other in row_nodes:
        if other is not point:
            gap = abs(candidate_col - other.col)
            if best is None or gap < best:
                best = gap
    return best if best is not None else (1 << 30)


def spread_adjacent_map_points(grid: list[list[MapPoint | None]], columns: int, rows: int):
    for row in range(rows):
        row_nodes = [grid[c][row] for c in range(columns) if grid[c][row] is not None]
        moved = True
        while moved:
            moved = False
            for point in row_nodes:
                col = point.col
                best_col, best_gap = col, compute_gap(col, row_nodes, point)
                for candidate in sorted(allowed_positions(point, columns)):
                    if candidate == col:
                        continue
                    if grid[candidate][row] is None or grid[candidate][row] is point:
                        gap = compute_gap(candidate, row_nodes, point)
                        if gap > best_gap:
                            best_col, best_gap = candidate, gap
                if best_col != col:
                    grid[col][row] = None
                    grid[best_col][row] = point
                    point.col = best_col
                    moved = True
    return grid


# --------------------------------------------------------------------------
# 世界构建与输出
# --------------------------------------------------------------------------


def build_map(cfg: dict, seed: int) -> tuple[ActMap, list[list[MapPoint | None]]]:
    map_ = ActMap(cfg, seed)
    map_.generate_map()
    map_.assign_point_types()
    if map_.do_prune:
        prune_and_repair(map_)
    grid = map_.grid
    if map_.do_post:
        grid = center_grid(grid, map_.columns, map_.map_length)
        grid = spread_adjacent_map_points(grid, map_.columns, map_.map_length)
        grid = straighten_paths(grid, map_.columns, map_.map_length)
    return map_, grid


def collect_output_points(map_: ActMap, grid: list[list[MapPoint | None]]) -> list[MapPoint]:
    """按行优先收集所有节点，起点在最前、终点在最后。"""
    points = [map_.start]
    for row in range(map_.map_length):
        for col in range(map_.columns):
            cell = grid[col][row]
            if cell is not None:
                points.append(cell)
    points.append(map_.boss)
    return points


def to_mermaid(map_: ActMap, grid: list[list[MapPoint | None]], seed: int, title: str) -> str:
    labels = {entry["id"]: entry.get("label", entry["id"]) for entry in map_.cfg["node_types"]}
    colors = {entry["id"]: entry.get("color", "#888888") for entry in map_.cfg["node_types"]}
    lines: list[str] = []
    lines.append("---")
    lines.append(f"title: {title}")
    lines.append("---")
    lines.append("flowchart TD")
    lines.append(f"    %% seed={seed}  rows={map_.map_length}  columns={map_.columns}")

    used = []
    for point in collect_output_points(map_, grid):
        if point.point_type not in used:
            used.append(point.point_type)
    for tid in used:
        color = colors.get(tid, "#888888")
        lines.append(
            f"    classDef cls_{tid.lower()} fill:{color},stroke:#222222,stroke-width:1px,color:#111111"
        )

    def node_id(point: MapPoint) -> str:
        return f"r{point.row}c{point.col}"

    for point in collect_output_points(map_, grid):
        label = labels.get(point.point_type, point.point_type)
        lines.append(f'    {node_id(point)}["{label}"]:::cls_{point.point_type.lower()}')

    for point in collect_output_points(map_, grid):
        for child in sorted(point.children, key=lambda p: (p.col, p.row)):
            lines.append(f"    {node_id(point)} --> {node_id(child)}")

    lines.append("")
    return "\n".join(lines)


def load_config(path: str) -> tuple[dict, bool]:
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle), False
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(DEFAULT_CONFIG, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    return json.loads(json.dumps(DEFAULT_CONFIG)), True


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="按种子生成杀戮尖塔 2 的单幕地图，输出 Mermaid 文件。")
    parser.add_argument("seed_pos", nargs="?", type=int, help="种子，也可以改用 --seed")
    parser.add_argument("--seed", type=int, help="种子")
    parser.add_argument("--act", type=int, default=1, help="幕号，仅用于标题，默认 1")
    parser.add_argument("--config", default=None, help=f"配置文件路径，默认脚本同级的 {DEFAULT_CONFIG_NAME}")
    parser.add_argument("--out", default=None, help="输出文件路径，默认 map_<种子>.mmd")
    parser.add_argument("--print", dest="to_stdout", action="store_true", help="同时打印到标准输出")
    args = parser.parse_args(argv)

    here = os.path.dirname(os.path.abspath(__file__))
    config_path = args.config or os.path.join(here, DEFAULT_CONFIG_NAME)
    cfg, created = load_config(config_path)
    if created:
        print(f"已生成默认配置：{config_path}")

    seed = args.seed if args.seed is not None else args.seed_pos
    if seed is None:
        seed = int(cfg.get("seed", 0))

    out_path = args.out or os.path.join(here, f"map_{seed}.mmd")
    if not os.path.isabs(out_path):
        out_path = os.path.join(os.getcwd(), out_path)

    map_, grid = build_map(cfg, seed)
    title = f"杀戮尖塔 2 第 {args.act} 幕地图（种子 {seed}）"
    text = to_mermaid(map_, grid, seed, title)

    with open(out_path, "w", encoding="utf-8") as handle:
        handle.write(text)

    if args.to_stdout:
        print(text)

    counts: dict[str, int] = {}
    for point in map_.all_points():
        counts[point.point_type] = counts.get(point.point_type, 0) + 1
    summary = "  ".join(f"{k}={v}" for k, v in sorted(counts.items()))
    print(f"地图已写入：{out_path}")
    print(f"节点统计：{summary}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
