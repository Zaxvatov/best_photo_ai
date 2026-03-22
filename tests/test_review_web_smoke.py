from __future__ import annotations

import py_compile
import tempfile
import unittest
from pathlib import Path

import pandas as pd
from fastapi.testclient import TestClient


ROOT = Path(__file__).resolve().parents[1]
REVIEW_WEB_APP = ROOT / "Scripts" / "review_web_app.py"


def test_review_web_app_compiles() -> None:
    py_compile.compile(str(REVIEW_WEB_APP), doraise=True)


class ReviewWebSmokeTest(unittest.TestCase):
    def test_review_web_api_smoke(self) -> None:
        import sys

        scripts_dir = ROOT / "Scripts"
        if str(scripts_dir) not in sys.path:
            sys.path.insert(0, str(scripts_dir))

        import review_web_app as app_module

        with tempfile.TemporaryDirectory() as tmp_dir_name:
            tmp_path = Path(tmp_dir_name)
            photo_file = tmp_path / "photo_review.csv"
            video_file = tmp_path / "video_review.csv"

            pd.DataFrame(
                [
                    {
                        "group_id": 1,
                        "scene_group_id": 10,
                        "file_path": str(tmp_path / "IMG_0001.JPG"),
                        "asset_id": "asset-photo-1",
                        "is_best": True,
                        "content_type_file": "people",
                        "content_type_group": "people",
                        "content_type_scene": "people",
                        "file_size": 1200,
                        "width": 100,
                        "height": 50,
                        "final_score": 0.9,
                    },
                    {
                        "group_id": 2,
                        "scene_group_id": 10,
                        "file_path": str(tmp_path / "IMG_0002.JPG"),
                        "asset_id": "asset-photo-2",
                        "is_best": False,
                        "content_type_file": "people",
                        "content_type_group": "people",
                        "content_type_scene": "people",
                        "file_size": 1000,
                        "width": 100,
                        "height": 50,
                        "final_score": 0.8,
                    },
                ]
            ).to_csv(photo_file, index=False, encoding="utf-8-sig")

            pd.DataFrame(
                [
                    {
                        "group_id": 101,
                        "scene_group_id": 101,
                        "video_group_id": 101,
                        "file_path": str(tmp_path / "VID_0001.MP4"),
                        "asset_id": "asset-video-1",
                        "is_best": True,
                        "has_live_photo_pair": True,
                        "video_score": 0.7,
                        "duration_sec_final": 2.1,
                    },
                    {
                        "group_id": 101,
                        "scene_group_id": 101,
                        "video_group_id": 101,
                        "file_path": str(tmp_path / "VID_0002.MP4"),
                        "asset_id": "asset-video-2",
                        "is_best": False,
                        "has_live_photo_pair": False,
                        "video_score": 0.6,
                        "duration_sec_final": 2.0,
                    },
                ]
            ).to_csv(video_file, index=False, encoding="utf-8-sig")

            app_module.PHOTO_DATA = photo_file
            app_module.VIDEO_DATA = video_file

            client = TestClient(app_module.app)

            root = client.get("/")
            self.assertEqual(root.status_code, 200)

            meta = client.get("/api/meta")
            self.assertEqual(meta.status_code, 200)
            self.assertIn("metricLabels", meta.json())

            photo_groups = client.get("/api/groups", params={"media_mode": "photo", "merge_scene_mode": "true"})
            self.assertEqual(photo_groups.status_code, 200)
            photo_payload = photo_groups.json()
            self.assertTrue(photo_payload["groups"])
            self.assertEqual(photo_payload["groups"][0]["commonLabel"], "10")
            self.assertEqual(photo_payload["groups"][0]["privateLabel"], "1, 2")

            photo_group = client.get("/api/group/10", params={"media_mode": "photo", "merge_scene_mode": "true"})
            self.assertEqual(photo_group.status_code, 200)
            photo_group_payload = photo_group.json()
            self.assertTrue(photo_group_payload["rows"])
            self.assertTrue(photo_group_payload["rows"][0]["assetId"])
            self.assertIn("final_score", photo_group_payload["metricOrder"])

            video_groups = client.get("/api/groups", params={"media_mode": "video", "show_single_videos": "true"})
            self.assertEqual(video_groups.status_code, 200)
            video_payload = video_groups.json()
            self.assertTrue(video_payload["groups"])
            self.assertTrue(video_payload["groups"][0]["hasLivePhoto"])
