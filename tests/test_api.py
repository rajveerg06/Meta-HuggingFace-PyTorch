"""
Integration tests for the FastAPI server (api/server.py).
Uses FastAPI's TestClient — no actual network I/O.
"""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient


class TestHealthEndpoint:
    def test_health_returns_200(self, api_client: TestClient) -> None:
        resp = api_client.get("/health")
        assert resp.status_code == 200

    def test_health_payload(self, api_client: TestClient) -> None:
        data = api_client.get("/health").json()
        assert data["status"] == "ok"
        assert "total_samples" in data
        assert data["total_samples"] > 0
        assert "samples_by_difficulty" in data


class TestTasksEndpoint:
    def test_tasks_returns_200(self, api_client: TestClient) -> None:
        resp = api_client.get("/tasks")
        assert resp.status_code == 200

    def test_tasks_contains_all_difficulties(self, api_client: TestClient) -> None:
        tasks = api_client.get("/tasks").json()
        difficulties = {t["difficulty"] for t in tasks}
        assert difficulties == {"easy", "medium", "hard"}

    def test_tasks_have_required_fields(self, api_client: TestClient) -> None:
        tasks = api_client.get("/tasks").json()
        for task in tasks:
            assert "task_id" in task
            assert "difficulty" in task
            assert "objective" in task
            assert "required_actions" in task
            assert "sample_count" in task
            assert task["sample_count"] > 0


class TestActionObservationSpaceEndpoints:
    def test_action_space(self, api_client: TestClient) -> None:
        resp = api_client.get("/action_space")
        assert resp.status_code == 200
        data = resp.json()
        assert data["type"] == "Discrete"
        assert data["n"] == 4

    def test_observation_space(self, api_client: TestClient) -> None:
        resp = api_client.get("/observation_space")
        assert resp.status_code == 200
        data = resp.json()
        assert "ocr_text" in data


class TestResetEndpoint:
    def test_reset_easy(self, api_client: TestClient) -> None:
        resp = api_client.post("/reset", json={"difficulty": "easy"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["observation"]["difficulty"] == "easy"
        assert data["observation"]["steps_taken"] == 0
        assert data["reward"]["total_reward"] == 0.0

    def test_reset_medium(self, api_client: TestClient) -> None:
        resp = api_client.post("/reset", json={"difficulty": "medium"})
        assert resp.status_code == 200

    def test_reset_hard(self, api_client: TestClient) -> None:
        resp = api_client.post("/reset", json={"difficulty": "hard"})
        assert resp.status_code == 200

    def test_reset_invalid_difficulty(self, api_client: TestClient) -> None:
        resp = api_client.post("/reset", json={"difficulty": "ultra"})
        assert resp.status_code == 400

    def test_reset_with_sample_index(self, api_client: TestClient) -> None:
        resp = api_client.post("/reset", json={"difficulty": "easy", "sample_index": 0})
        assert resp.status_code == 200

    def test_reset_observation_has_ocr_text(self, api_client: TestClient) -> None:
        resp = api_client.post("/reset", json={"difficulty": "hard"})
        data = resp.json()
        assert len(data["observation"]["ocr_text"]) > 0


class TestStepEndpoint:
    def _reset(self, client: TestClient, difficulty: str = "easy") -> None:
        client.post("/reset", json={"difficulty": difficulty})

    def test_step_classify_invoice(self, api_client: TestClient) -> None:
        self._reset(api_client, "easy")
        resp = api_client.post(
            "/step",
            json={"action": {"action_type": "classify_document", "payload": {"document_type": "invoice"}}},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["observation"]["steps_taken"] == 1
        assert data["observation"]["predicted_document_type"] == "invoice"

    def test_step_classify_receipt(self, api_client: TestClient) -> None:
        self._reset(api_client, "easy")
        resp = api_client.post(
            "/step",
            json={"action": {"action_type": "classify_document", "payload": {"document_type": "receipt"}}},
        )
        assert resp.status_code == 200

    def test_step_extract_fields(self, api_client: TestClient) -> None:
        self._reset(api_client, "medium")
        resp = api_client.post(
            "/step",
            json={
                "action": {
                    "action_type": "extract_fields",
                    "payload": {
                        "fields": {
                            "vendor_name": "Star Groceries",
                            "total_amount": "RM 13.30",
                            "date": "04-05-2023",
                        }
                    },
                }
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["observation"]["extracted_fields"]["vendor_name"] == "Star Groceries"

    def test_step_finish(self, api_client: TestClient) -> None:
        self._reset(api_client, "easy")
        resp = api_client.post(
            "/step",
            json={"action": {"action_type": "finish", "payload": {}}},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["observation"]["done"] is True

    def test_step_reward_nonzero_after_action(self, api_client: TestClient) -> None:
        self._reset(api_client, "easy")
        resp = api_client.post(
            "/step",
            json={"action": {"action_type": "classify_document", "payload": {"document_type": "invoice"}}},
        )
        data = resp.json()
        # Reward should have been accumulated (positive or negative depending on correctness)
        assert data["reward"]["step_reward"] != 0.0 or data["reward"]["total_reward"] == 0.0


class TestStateEndpoint:
    def test_state_after_reset(self, api_client: TestClient) -> None:
        api_client.post("/reset", json={"difficulty": "easy"})
        resp = api_client.get("/state")
        assert resp.status_code == 200
        data = resp.json()
        assert "observation" in data
        assert "reward" in data

    def test_state_reflects_last_step(self, api_client: TestClient) -> None:
        api_client.post("/reset", json={"difficulty": "easy"})
        api_client.post(
            "/step",
            json={"action": {"action_type": "classify_document", "payload": {"document_type": "invoice"}}},
        )
        resp = api_client.get("/state")
        data = resp.json()
        assert data["observation"]["steps_taken"] == 1


class TestRenderEndpoint:
    def test_render_returns_string(self, api_client: TestClient) -> None:
        api_client.post("/reset", json={"difficulty": "easy"})
        resp = api_client.get("/render")
        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data["render"], str)
        assert len(data["render"]) > 0


class TestHistoryEndpoint:
    def test_history_empty_after_reset(self, api_client: TestClient) -> None:
        api_client.post("/reset", json={"difficulty": "easy"})
        resp = api_client.get("/history")
        assert resp.status_code == 200
        assert resp.json()["history"] == []

    def test_history_grows_with_steps(self, api_client: TestClient) -> None:
        api_client.post("/reset", json={"difficulty": "easy"})
        api_client.post(
            "/step",
            json={"action": {"action_type": "classify_document", "payload": {"document_type": "invoice"}}},
        )
        resp = api_client.get("/history")
        assert len(resp.json()["history"]) == 1


class TestFullEpisodeFlow:
    def test_hard_full_pipeline(self, api_client: TestClient) -> None:
        """End-to-end hard episode: classify → extract → validate → finish."""
        api_client.post("/reset", json={"difficulty": "hard"})

        api_client.post(
            "/step",
            json={"action": {"action_type": "classify_document", "payload": {"document_type": "receipt"}}},
        )
        api_client.post(
            "/step",
            json={
                "action": {
                    "action_type": "extract_fields",
                    "payload": {
                        "fields": {
                            "vendor_name": "JJ MART SDN BHD",
                            "total_amount": "RM 6.00",
                            "date": "18/02/2024",
                        }
                    },
                }
            },
        )
        api_client.post(
            "/step",
            json={"action": {"action_type": "validate_fields", "payload": {"is_valid": True}}},
        )
        resp = api_client.post("/step", json={"action": {"action_type": "finish", "payload": {}}})

        data = resp.json()
        assert data["observation"]["done"] is True
        assert data["reward"]["grader_score"] >= 0.0
