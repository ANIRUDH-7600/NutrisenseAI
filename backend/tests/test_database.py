"""
Tests for MongoDB database integration and fail-soft behavior.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from src.api.database import (
    init_db,
    close_db,
    check_db_health,
    save_screening_record,
    get_screening_records,
    get_screening_record_by_id,
    delete_screening_record,
)


def test_database_health_disconnected():
    """Verify check_db_health reports disconnected gracefully when client is not active."""
    async def run():
        with patch("src.api.database._client", None), patch("src.api.database._is_connected", False):
            health = await check_db_health()
            assert health["status"] == "disconnected"
            assert health["connected"] is False
            assert "stateless" in health["message"]

    asyncio.run(run())


def test_database_health_disabled():
    """Verify check_db_health reports disabled when MONGODB_ENABLED is False."""
    async def run():
        with patch("src.api.database.MONGODB_ENABLED", False):
            health = await check_db_health()
            assert health["status"] == "disabled"
            assert health["connected"] is False

    asyncio.run(run())


def test_save_screening_fail_soft_when_disconnected():
    """Verify save_screening_record returns None without raising exceptions when DB is disconnected."""
    async def run():
        with patch("src.api.database._db", None), patch("src.api.database._is_connected", False):
            record_id = await save_screening_record({"dummy": "value"})
            assert record_id is None

    asyncio.run(run())


def test_get_screenings_returns_empty_when_disconnected():
    """Verify get_screening_records returns empty list when DB is disconnected."""
    async def run():
        with patch("src.api.database._db", None), patch("src.api.database._is_connected", False):
            records = await get_screening_records()
            assert records == []

    asyncio.run(run())


def test_get_screening_by_id_returns_none_when_disconnected():
    """Verify get_screening_record_by_id returns None when DB is disconnected."""
    async def run():
        with patch("src.api.database._db", None), patch("src.api.database._is_connected", False):
            record = await get_screening_record_by_id("nonexistent_id")
            assert record is None

    asyncio.run(run())


def test_delete_screening_returns_false_when_disconnected():
    """Verify delete_screening_record returns False when DB is disconnected."""
    async def run():
        with patch("src.api.database._db", None), patch("src.api.database._is_connected", False):
            deleted = await delete_screening_record("nonexistent_id")
            assert deleted is False

    asyncio.run(run())


def test_save_and_retrieve_with_mocked_db():
    """Verify document insertion and retrieval logic with mocked Motor database."""
    async def run():
        mock_db = MagicMock()
        mock_screenings = MagicMock()

        # Mock insert_one
        mock_insert_result = MagicMock()
        mock_insert_result.inserted_id = "mock_obj_id_123"
        mock_screenings.insert_one = AsyncMock(return_value=mock_insert_result)

        # Mock find_one
        mock_doc = {
            "screening_id": "scr_test123",
            "created_at": "2026-09-27T12:00:00Z",
            "inputs": {"child_age_months": 24},
            "predictions": {"stunting": {"risk_level": "Elevated Risk"}},
        }
        mock_screenings.find_one = AsyncMock(return_value=mock_doc)

        mock_db.screenings = mock_screenings

        with patch("src.api.database._db", mock_db), patch("src.api.database._is_connected", True):
            # Test save
            screening_id = await save_screening_record({
                "screening_id": "scr_test123",
                "inputs": {"child_age_months": 24},
                "predictions": {"stunting": {"risk_level": "Elevated Risk"}},
            })
            assert screening_id == "scr_test123"
            mock_screenings.insert_one.assert_called_once()

            # Test retrieve by ID
            fetched = await get_screening_record_by_id("scr_test123")
            assert fetched is not None
            assert fetched["screening_id"] == "scr_test123"
            assert fetched["inputs"]["child_age_months"] == 24

    asyncio.run(run())
