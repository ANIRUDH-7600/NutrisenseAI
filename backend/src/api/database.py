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
    """Initializes the MongoDB client and verifies connectivity with automatic local fallback."""
    global _client, _db, _is_connected

    if not MONGODB_ENABLED:
        logger.info("[DATABASE] MongoDB is explicitly disabled via MONGODB_ENABLED=false.")
        return

    uris_to_try = [MONGODB_URI]
    if MONGODB_URI not in ("mongodb://127.0.0.1:27017", "mongodb://localhost:27017"):
        uris_to_try.append("mongodb://127.0.0.1:27017")

    last_error: Optional[Exception] = None

    for target_uri in uris_to_try:
        try:
            logger.info(f"[DATABASE] Connecting to MongoDB at {target_uri}...")
            client = AsyncIOMotorClient(
                target_uri,
                serverSelectionTimeoutMS=MONGODB_SERVER_SELECTION_TIMEOUT_MS,
            )
            # Verify connectivity with an immediate ping
            await client.admin.command("ping")

            _client = client
            _db = _client[MONGODB_DB_NAME]
            _is_connected = True
            logger.info(f"[DATABASE] Connected successfully to MongoDB database: '{MONGODB_DB_NAME}' via {target_uri}.")

            # Create indexes asynchronously
            await _db.screenings.create_index("screening_id", unique=True)
            await _db.screenings.create_index([("user_id", 1), ("created_at", -1)])
            logger.info("[DATABASE] Database indexes verified on 'screenings' collection (user_id + created_at).")

            # Create user indexes asynchronously
            await _db.users.create_index("email", unique=True)
            await _db.users.create_index("user_id", unique=True)
            logger.info("[DATABASE] Database indexes verified on 'users' collection.")

            # Create token revocation indexes asynchronously (TTL expiration)
            await _db.revoked_tokens.create_index("jti", unique=True)
            await _db.revoked_tokens.create_index("expires_at", expireAfterSeconds=0)
            logger.info("[DATABASE] Database indexes verified on 'revoked_tokens' collection.")

            # Seed demo user if not present
            existing_demo = await _db.users.find_one({"email": "asha.worker@health.gov.in"})
            if not existing_demo:
                from src.api.security import hash_password
                await _db.users.insert_one({
                    "user_id": "usr_demo_asha_001",
                    "name": "Sunita Devi (ASHA)",
                    "email": "asha.worker@health.gov.in",
                    "password_hash": hash_password("HealthWorker#2026"),
                    "role": "health_worker",
                    "is_active": True,
                    "created_at": datetime.now(timezone.utc).isoformat()
                })
                logger.info("[DATABASE] Seeded default health worker demo account.")

            return

        except Exception as e:
            last_error = e
            logger.warning(
                f"[DATABASE] MongoDB connection attempt failed for {target_uri}: {e}."
            )

    _is_connected = False
    logger.warning(
        f"[DATABASE] All MongoDB connection attempts failed. Last error: {last_error}. "
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


async def save_screening_record(
    record_data: Dict[str, Any],
    user_id: Optional[str] = None
) -> Optional[str]:
    """
    Saves a completed screening assessment into the 'screenings' collection.
    
    If user_id is provided, attaches it to the document (overriding any client payload value).
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

        # Enforce server-side authenticated user_id
        if user_id is not None:
            document["user_id"] = str(user_id)

        # Prevent duplicate _id conflict if dict had one
        document.pop("_id", None)

        result = await _db.screenings.insert_one(document)
        logger.info(f"[DATABASE] Saved screening record '{screening_id}' (DocID: {result.inserted_id}) for user '{document.get('user_id')}'.")
        return screening_id

    except PyMongoError as e:
        logger.warning(f"[DATABASE] Failed to save screening record to MongoDB: {e}")
        return None
    except Exception as e:
        logger.warning(f"[DATABASE] Unexpected error saving screening record: {e}")
        return None


async def get_screening_records(
    user_id: Optional[str] = None,
    limit: int = 20,
    skip: int = 0
) -> List[Dict[str, Any]]:
    """Retrieves recent screening records, ordered from newest to oldest, optionally filtered by user_id."""
    if _db is None or not _is_connected:
        return []

    try:
        query: Dict[str, Any] = {}
        if user_id is not None:
            if not isinstance(user_id, str):
                return []
            query["user_id"] = user_id

        cursor = (
            _db.screenings.find(query, {"_id": 0})
            .sort("created_at", -1)
            .skip(max(0, skip))
            .limit(min(100, max(1, limit)))
        )
        return await cursor.to_list(length=limit)
    except Exception as e:
        logger.warning(f"[DATABASE] Failed to retrieve screening records: {e}")
        return []


async def get_screening_record_by_id(
    screening_id: str,
    user_id: Optional[str] = None
) -> Optional[Dict[str, Any]]:
    """Retrieves a single screening assessment by its unique screening_id, scoped by owner user_id if provided."""
    if _db is None or not _is_connected:
        return None

    # Enforce strict string type to defend against NoSQL injection
    if not isinstance(screening_id, str) or (user_id is not None and not isinstance(user_id, str)):
        return None

    try:
        query: Dict[str, Any] = {"screening_id": screening_id}
        if user_id is not None:
            query["user_id"] = user_id

        record = await _db.screenings.find_one(query, {"_id": 0})
        return record
    except Exception as e:
        logger.warning(f"[DATABASE] Failed to fetch screening record '{screening_id}': {e}")
        return None


async def delete_screening_record(
    screening_id: str,
    user_id: Optional[str] = None
) -> bool:
    """Deletes a screening record by its unique screening_id, scoped by owner user_id if provided."""
    if _db is None or not _is_connected:
        return False

    # Enforce strict string type to defend against NoSQL injection
    if not isinstance(screening_id, str) or (user_id is not None and not isinstance(user_id, str)):
        return False

    try:
        query: Dict[str, Any] = {"screening_id": screening_id}
        if user_id is not None:
            query["user_id"] = user_id

        result = await _db.screenings.delete_one(query)
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
    if _db is None or not _is_connected or not isinstance(email, str):
        return None

    normalized_email = email.strip().lower()
    try:
        user = await _db.users.find_one({"email": normalized_email})
        return user
    except PyMongoError as e:
        logger.warning(f"[DATABASE] Error retrieving user by email '{normalized_email}': {e}")
        raise
    except Exception as e:
        logger.warning(f"[DATABASE] Error retrieving user by email '{normalized_email}': {e}")
        return None


async def get_user_by_id(user_id: str) -> Optional[Dict[str, Any]]:
    """Retrieves a user document by user_id from the 'users' collection."""
    if _db is None or not _is_connected or not isinstance(user_id, str):
        return None

    try:
        user = await _db.users.find_one({"user_id": user_id})
        return user
    except PyMongoError as e:
        logger.warning(f"[DATABASE] Error retrieving user by user_id '{user_id}': {e}")
        raise
    except Exception as e:
        logger.warning(f"[DATABASE] Error retrieving user by user_id '{user_id}': {e}")
        return None


async def delete_user(user_id: str) -> bool:
    """Deletes a user account by user_id (used for test teardowns and administrative actions)."""
    if _db is None or not _is_connected or not isinstance(user_id, str):
        return False

    try:
        result = await _db.users.delete_one({"user_id": user_id})
        return result.deleted_count > 0
    except Exception as e:
        logger.warning(f"[DATABASE] Error deleting user '{user_id}': {e}")
        return False


# ==============================================================================
# Token Revocation Operations (Phase 4)
# ==============================================================================

async def revoke_token(jti: str, user_id: str, expires_at: datetime) -> bool:
    """
    Revokes a JWT token by storing its unique jti in the 'revoked_tokens' collection.
    Idempotent: if already revoked, returns True safely without error.
    Raises RuntimeError if database is currently disconnected.
    Raises PyMongoError if query execution fails.
    """
    if _db is None or not _is_connected:
        raise RuntimeError("Database is currently disconnected or unavailable.")

    now_dt = datetime.now(timezone.utc)
    document = {
        "jti": str(jti),
        "user_id": str(user_id),
        "revoked_at": now_dt,
        "expires_at": expires_at,
    }

    try:
        query_res = _db.revoked_tokens.update_one(
            {"jti": str(jti)},
            {"$setOnInsert": document},
            upsert=True
        )
        if hasattr(query_res, "__await__"):
            await query_res
        logger.info(f"[DATABASE] Token jti '{jti}' for user '{user_id}' marked as revoked.")
        return True
    except DuplicateKeyError:
        return True
    except PyMongoError as e:
        logger.warning(f"[DATABASE] Error revoking token jti '{jti}': {e}")
        raise


async def is_token_revoked(jti: str) -> bool:
    """
    Checks if a JWT jti is recorded in the 'revoked_tokens' collection.
    Returns True if revoked, False otherwise.
    Raises RuntimeError if database is disconnected.
    Raises PyMongoError if query execution fails.
    """
    if _db is None or not _is_connected:
        raise RuntimeError("Database is currently disconnected or unavailable.")

    try:
        query_res = _db.revoked_tokens.find_one({"jti": str(jti)})
        if hasattr(query_res, "__await__"):
            record = await query_res
        else:
            record = None
        return record is not None
    except PyMongoError as e:
        logger.warning(f"[DATABASE] Error checking revocation for jti '{jti}': {e}")
        raise


async def clean_expired_revocations() -> int:
    """
    Explicit cleanup helper to remove any expired revocation entries.
    Useful for testing or manual pruning alongside MongoDB's automatic TTL index.
    """
    if _db is None or not _is_connected:
        return 0

    now_dt = datetime.now(timezone.utc)
    try:
        result = await _db.revoked_tokens.delete_many({"expires_at": {"$lte": now_dt}})
        return result.deleted_count
    except Exception as e:
        logger.warning(f"[DATABASE] Error cleaning expired revocations: {e}")
        return 0


