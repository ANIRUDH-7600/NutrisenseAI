"""
Database management module for NutriSense AI using MongoDB (Motor async driver).

Features:
- Asynchronous client lifecycle (connect on startup, disconnect on shutdown).
- Safe fail-soft architecture: if MongoDB is unreachable or down, screening inference
  continues normally without interruptions, and health check reports the connection status.
- Automatic UUID generation, ISO timestamps, and indexing on session IDs.
"""

import logging
import uuid
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional

from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase
from pymongo.errors import PyMongoError, ServerSelectionTimeoutError, DuplicateKeyError

from src.api.config import (
    MONGODB_URI,
    MONGODB_DB_NAME,
    MONGODB_ENABLED,
    MONGODB_SERVER_SELECTION_TIMEOUT_MS,
)

logger = logging.getLogger("nutrisense.database")

# Global client and database instances
_client: Optional[AsyncIOMotorClient] = None
_db: Optional[AsyncIOMotorDatabase] = None
_is_connected: bool = False


async def init_db() -> None:
    """Initializes the MongoDB client and verifies connectivity."""
    global _client, _db, _is_connected

    if not MONGODB_ENABLED:
        logger.info("[DATABASE] MongoDB is explicitly disabled via MONGODB_ENABLED=false.")
        return

    try:
        logger.info(f"[DATABASE] Connecting to MongoDB at {MONGODB_URI}...")
        _client = AsyncIOMotorClient(
            MONGODB_URI,
            serverSelectionTimeoutMS=MONGODB_SERVER_SELECTION_TIMEOUT_MS,
        )
        _db = _client[MONGODB_DB_NAME]

        # Verify connectivity with an immediate ping
        await _client.admin.command("ping")
        _is_connected = True
        logger.info(f"[DATABASE] Connected successfully to MongoDB database: '{MONGODB_DB_NAME}'.")

        # Create indexes asynchronously
        await _db.screenings.create_index("screening_id", unique=True)
        await _db.screenings.create_index([("timestamp", -1)])
        logger.info("[DATABASE] Database indexes verified on 'screenings' collection.")

        # Create user indexes asynchronously
        await _db.users.create_index("email", unique=True)
        await _db.users.create_index("user_id", unique=True)
        logger.info("[DATABASE] Database indexes verified on 'users' collection.")


    except ServerSelectionTimeoutError as e:
        _is_connected = False
        logger.warning(
            f"[DATABASE] MongoDB server is not currently reachable at {MONGODB_URI} "
            f"(Timeout: {MONGODB_SERVER_SELECTION_TIMEOUT_MS}ms). "
            f"Screening will run in memory-only mode without persistence. Detail: {e}"
        )
    except Exception as e:
        _is_connected = False
        logger.warning(
            f"[DATABASE] Unexpected error connecting to MongoDB: {e}. "
            f"Screening will run in memory-only mode."
        )


async def close_db() -> None:
    """Closes the MongoDB connection pool cleanly."""
    global _client, _db, _is_connected
    if _client is not None:
        logger.info("[DATABASE] Closing MongoDB connection pool...")
        _client.close()
        _client = None
        _db = None
        _is_connected = False
        logger.info("[DATABASE] MongoDB connection closed.")


def get_db() -> Optional[AsyncIOMotorDatabase]:
    """Returns the current database instance if connected."""
    return _db if _is_connected else None


async def check_db_health() -> Dict[str, Any]:
    """Checks the health and round-trip ping latency of MongoDB."""
    if not MONGODB_ENABLED:
        return {
            "status": "disabled",
            "connected": False,
            "message": "MongoDB is disabled via configuration.",
            "database": MONGODB_DB_NAME,
        }

    if _client is None or not _is_connected:
        return {
            "status": "disconnected",
            "connected": False,
            "message": "MongoDB client is not connected. Screening operates in stateless mode.",
            "database": MONGODB_DB_NAME,
            "uri": MONGODB_URI.split("@")[-1] if "@" in MONGODB_URI else MONGODB_URI,
        }

    try:
        start_time = datetime.now(timezone.utc)
        await _client.admin.command("ping")
        latency_ms = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000.0
        return {
            "status": "healthy",
            "connected": True,
            "latency_ms": round(latency_ms, 2),
            "database": MONGODB_DB_NAME,
            "uri": MONGODB_URI.split("@")[-1] if "@" in MONGODB_URI else MONGODB_URI,
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "connected": False,
            "error": str(e),
            "database": MONGODB_DB_NAME,
        }


async def save_screening_record(record_data: Dict[str, Any]) -> Optional[str]:
    """
    Saves a completed screening assessment into the 'screenings' collection.
    
    If MongoDB is disconnected or encounters an error, logs a warning and returns None
    so that user requests NEVER fail due to database downtime.
    """
    if _db is None or not _is_connected:
        logger.debug("[DATABASE] Skipping persistence: MongoDB is not connected.")
        return None

    try:
        # Guarantee unique screening_id and created_at ISO timestamp
        screening_id = record_data.get("screening_id") or f"scr_{uuid.uuid4().hex[:12]}"
        document = {
            "screening_id": screening_id,
            "created_at": datetime.now(timezone.utc).isoformat(),
            **record_data,
        }

        # Prevent duplicate _id conflict if dict had one
        document.pop("_id", None)

        result = await _db.screenings.insert_one(document)
        logger.info(f"[DATABASE] Saved screening record '{screening_id}' (DocID: {result.inserted_id}).")
        return screening_id

    except PyMongoError as e:
        logger.warning(f"[DATABASE] Failed to save screening record to MongoDB: {e}")
        return None
    except Exception as e:
        logger.warning(f"[DATABASE] Unexpected error saving screening record: {e}")
        return None


async def get_screening_records(limit: int = 20, skip: int = 0) -> List[Dict[str, Any]]:
    """Retrieves recent screening records, ordered from newest to oldest."""
    if _db is None or not _is_connected:
        return []

    try:
        cursor = (
            _db.screenings.find({}, {"_id": 0})
            .sort("created_at", -1)
            .skip(max(0, skip))
            .limit(min(100, max(1, limit)))
        )
        return await cursor.to_list(length=limit)
    except Exception as e:
        logger.warning(f"[DATABASE] Failed to retrieve screening records: {e}")
        return []


async def get_screening_record_by_id(screening_id: str) -> Optional[Dict[str, Any]]:
    """Retrieves a single screening assessment by its unique screening_id."""
    if _db is None or not _is_connected:
        return None

    try:
        record = await _db.screenings.find_one({"screening_id": screening_id}, {"_id": 0})
        return record
    except Exception as e:
        logger.warning(f"[DATABASE] Failed to fetch screening record '{screening_id}': {e}")
        return None


async def delete_screening_record(screening_id: str) -> bool:
    """Deletes a screening record by its unique screening_id."""
    if _db is None or not _is_connected:
        return False

    try:
        result = await _db.screenings.delete_one({"screening_id": screening_id})
        return result.deleted_count > 0
    except Exception as e:
        logger.warning(f"[DATABASE] Failed to delete screening record '{screening_id}': {e}")
        return False


# ==============================================================================
# User Account Operations (Phase 2)
# ==============================================================================

async def create_user(user_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Creates a new user record in the 'users' collection.
    Enforces unique email, generates user_id and timestamps.
    Raises DuplicateKeyError if an account with the normalized email already exists.
    Raises RuntimeError if database is offline or disconnected.
    """
    if _db is None or not _is_connected:
        raise RuntimeError("Database is currently disconnected or unavailable.")

    now_iso = datetime.now(timezone.utc).isoformat()
    user_id = user_data.get("user_id") or f"usr_{uuid.uuid4().hex[:12]}"
    normalized_email = user_data["email"].strip().lower()

    document = {
        "user_id": user_id,
        "name": user_data["name"].strip(),
        "email": normalized_email,
        "password_hash": user_data["password_hash"],
        "role": user_data.get("role", "health_worker"),
        "is_active": user_data.get("is_active", True),
        "created_at": user_data.get("created_at") or now_iso,
        "updated_at": user_data.get("updated_at") or now_iso,
    }

    # Ensure no conflicting _id passed
    document.pop("_id", None)

    await _db.users.insert_one(document)
    logger.info(f"[DATABASE] Created user account '{user_id}' with email '{normalized_email}'.")
    return document


async def get_user_by_email(email: str) -> Optional[Dict[str, Any]]:
    """Retrieves a user document by normalized email address from the 'users' collection."""
    if _db is None or not _is_connected:
        return None

    normalized_email = email.strip().lower()
    try:
        user = await _db.users.find_one({"email": normalized_email})
        return user
    except Exception as e:
        logger.warning(f"[DATABASE] Error retrieving user by email '{normalized_email}': {e}")
        return None


async def get_user_by_id(user_id: str) -> Optional[Dict[str, Any]]:
    """Retrieves a user document by user_id from the 'users' collection."""
    if _db is None or not _is_connected:
        return None

    try:
        user = await _db.users.find_one({"user_id": user_id})
        return user
    except Exception as e:
        logger.warning(f"[DATABASE] Error retrieving user by user_id '{user_id}': {e}")
        return None


async def delete_user(user_id: str) -> bool:
    """Deletes a user account by user_id (used for test teardowns and administrative actions)."""
    if _db is None or not _is_connected:
        return False

    try:
        result = await _db.users.delete_one({"user_id": user_id})
        return result.deleted_count > 0
    except Exception as e:
        logger.warning(f"[DATABASE] Error deleting user '{user_id}': {e}")
        return False

