"""
tracker.py — Euclidean distance-based multi-object tracker.
Assigns stable IDs to detected objects across frames by matching
each new detection's centre-point to the closest known centre-point.

Based on: https://github.com/Subhadip7/yolov8-multiple-vehicle-detection
"""

import math


class Tracker:
    def __init__(self, max_distance: int = 50):
        # Store the centre positions of tracked objects  {id: (cx, cy)}
        self.center_points: dict[int, tuple[int, int]] = {}
        # Auto-incrementing ID counter
        self.id_count: int = 0
        # Maximum distance (pixels) to consider two detections the same object
        self.max_distance = max_distance

    def update(self, objects_rect: list[list[int]]) -> list[list[int]]:
        """
        Match new detections to existing tracked objects.

        Parameters
        ----------
        objects_rect : list of [x1, y1, x2, y2]
            Bounding boxes for the current frame.

        Returns
        -------
        list of [x1, y1, x2, y2, object_id]
            Each detection tagged with its stable tracking ID.
        """
        objects_bbs_ids: list[list[int]] = []

        for rect in objects_rect:
            x1, y1, x2, y2 = rect
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2

            # Try to match to an existing tracked object
            same_object_detected = False
            for obj_id, pt in self.center_points.items():
                dist = math.hypot(cx - pt[0], cy - pt[1])
                if dist < self.max_distance:
                    self.center_points[obj_id] = (cx, cy)
                    objects_bbs_ids.append([x1, y1, x2, y2, obj_id])
                    same_object_detected = True
                    break

            # New object — assign a fresh ID
            if not same_object_detected:
                self.center_points[self.id_count] = (cx, cy)
                objects_bbs_ids.append([x1, y1, x2, y2, self.id_count])
                self.id_count += 1

        # Prune IDs that were not matched this frame
        new_center_points: dict[int, tuple[int, int]] = {}
        for obj_bb_id in objects_bbs_ids:
            obj_id = obj_bb_id[4]
            new_center_points[obj_id] = self.center_points[obj_id]
        self.center_points = new_center_points

        return objects_bbs_ids
