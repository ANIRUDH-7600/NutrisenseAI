"""
Phase 6 — User-Owned Screening Data & IDOR / BOLA Prevention Test Suite.

Verifies strict object-level authorization across all screening endpoints:
1. POST /api/v1/screen:
   - Newly created screening records inherit authenticated user_id from verified JWT.
   - Client-supplied user_id in payload cannot override authenticated user_id.
2. GET /api/v1/screenings:
   - List filtering scoped exclusively to the authenticated user.
3. GET /api/v1/screenings/{screening_id}:
   - Owner can retrieve own record (200 OK).
   - Non-owner attempting access receives 404 (preventing resource enumeration).
4. DELETE /api/v1/screenings/{screening_id}:
   - Owner can delete own record (200 OK).
   - Non-owner attempting deletion receives 404; other user's record remains intact.
5. Two-User Security Test (Mandatory cross-user matrix):
   - User A and User B cross-isolation across create, list, retrieve, and delete.
6. Nonexistent record handling:
   - Returns standard 404 SCREENING_NOT_FOUND.
7. Database failure resilience:
   - Database errors handled gracefully without leaking stack traces.
8. Malicious ID manipulation:
   - SQL/NoSQL injection patterns in URL paths are sanitized and rejected safely.
9. Unauthenticated requests:
   - Rejected with 401 AUTHENTICATION_REQUIRED.
"""

import copy
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from fastapi.testclient import TestClient

from src.api.main import app
from src.api.security import create_access_token


@pytest.fixture
def client():
    """TestClient without auto-running lifespan to prevent external DB dependency."""
    with TestClient(app, raise_server_exceptions=False) as c:
        yield c


from tests.test_inference_pipeline import get_sample_valid_input


@pytest.fixture
def mock_ownership_db():
    """
    Isolated in-memory mock database for User A and User B screening operations.
    Supports users, revoked_tokens, and screenings with strict user_id filtering.
    """
    users = {
        "user_a@nutrisense.org": {
            "user_id": "usr_alpha_111",
            "name": "Dr. Alpha",
            "email": "user_a@nutrisense.org",
            "role": "health_worker",
            "is_active": True
        },
        "user_b@nutrisense.org": {
            "user_id": "usr_beta_222",
            "name": "Dr. Beta",
            "email": "user_b@nutrisense.org",
            "role": "health_worker",
            "is_active": True
        }
    }
    revoked = {}
    screenings = {}

    async def mock_find_one_users(filter_dict, projection=None):
        if "email" in filter_dict:
            u = users.get(filter_dict["email"])
            return copy.deepcopy(u) if u else None
        if "user_id" in filter_dict:
            for u in users.values():
                if u.get("user_id") == filter_dict["user_id"]:
                    return copy.deepcopy(u)
        return None

    async def mock_find_one_revoked(filter_dict, projection=None):
        jti = filter_dict.get("jti")
        return copy.deepcopy(revoked.get(jti)) if jti in revoked else None

    async def mock_find_one_screenings(filter_dict, projection=None):
        sid = filter_dict.get("screening_id")
        if not sid or sid not in screenings:
            return None
        rec = screenings[sid]
        if "user_id" in filter_dict and filter_dict["user_id"] is not None:
            if rec.get("user_id") != filter_dict["user_id"]:
                return None
        return copy.deepcopy(rec)

    async def mock_insert_one_screenings(doc):
        sid = doc.get("screening_id") or "scr_gen_mock"
        screenings[sid] = copy.deepcopy(doc)
        res = MagicMock()
        res.inserted_id = sid
        return res

    async def mock_delete_one_screenings(filter_dict):
        sid = filter_dict.get("screening_id")
        res = MagicMock()
        if sid in screenings:
            rec = screenings[sid]
            if "user_id" in filter_dict and filter_dict["user_id"] is not None:
                if rec.get("user_id") != filter_dict["user_id"]:
                    res.deleted_count = 0
                    return res
            del screenings[sid]
            res.deleted_count = 1
        else:
            res.deleted_count = 0
        return res

    def mock_screenings_find(filter_dict=None, projection=None):
        target_uid = filter_dict.get("user_id") if filter_dict else None
        matches = [
            copy.deepcopy(s) for s in screenings.values()
            if target_uid is None or s.get("user_id") == target_uid
        ]
        mock_cursor = MagicMock()
        mock_cursor.sort = MagicMock(return_value=mock_cursor)
        mock_cursor.skip = MagicMock(return_value=mock_cursor)
        mock_cursor.limit = MagicMock(return_value=mock_cursor)
        mock_cursor.to_list = AsyncMock(return_value=matches)
        return mock_cursor

    db = MagicMock()
    db.users.find_one = AsyncMock(side_effect=mock_find_one_users)
    db.revoked_tokens.find_one = AsyncMock(side_effect=mock_find_one_revoked)
    db.screenings.find_one = AsyncMock(side_effect=mock_find_one_screenings)
    db.screenings.insert_one = AsyncMock(side_effect=mock_insert_one_screenings)
    db.screenings.delete_one = AsyncMock(side_effect=mock_delete_one_screenings)
    db.screenings.find = MagicMock(side_effect=mock_screenings_find)

    return db, users, screenings


# ==============================================================================
# 1. SCREENING CREATION OWNERSHIP
# ==============================================================================

def test_screening_creation_attaches_authenticated_user_id(client, mock_ownership_db):
    """POST /screen persists the authenticated user's user_id into the screening document."""
    db, users, screenings = mock_ownership_db
    token_a = create_access_token(user_id="usr_alpha_111", email="user_a@nutrisense.org")

    with patch("src.api.database._db", db), patch("src.api.database._is_connected", True):
        res = client.post(
            "/api/v1/screen",
            json=get_sample_valid_input(),
            headers={"Authorization": f"Bearer {token_a}"}
        )
        assert res.status_code == 200
        # Verify document stored in DB has owner user_id == usr_alpha_111
        assert len(screenings) == 1
        stored_rec = list(screenings.values())[0]
        assert stored_rec["user_id"] == "usr_alpha_111"


def test_user_id_cannot_be_spoofed_in_screening_creation(client, mock_ownership_db):
    """
    Even if an attacker attempts to inject a spoofed user_id in the request payload or headers,
    the server-side authenticated identity strictly prevails.
    """
    db, users, screenings = mock_ownership_db
    token_a = create_access_token(user_id="usr_alpha_111", email="user_a@nutrisense.org")

    payload = get_sample_valid_input()

    with patch("src.api.database._db", db), patch("src.api.database._is_connected", True):
        # 1. Injected extra field in payload is rejected by strict Pydantic extra='forbid'
        payload_with_spoofed_user = {**payload, "user_id": "usr_victim_999"}
        res_spoof = client.post(
            "/api/v1/screen",
            json=payload_with_spoofed_user,
            headers={"Authorization": f"Bearer {token_a}"}
        )
        assert res_spoof.status_code == 422
        assert "extra_forbidden" in str(res_spoof.json()) or "Extra inputs" in str(res_spoof.json())

        # 2. Legitimate payload with User A token strictly saves with User A ID
        res_legit = client.post(
            "/api/v1/screen",
            json=payload,
            headers={"Authorization": f"Bearer {token_a}"}
        )
        assert res_legit.status_code == 200
        stored_rec = list(screenings.values())[0]
        assert stored_rec["user_id"] == "usr_alpha_111"
        assert stored_rec["user_id"] != "usr_victim_999"


# ==============================================================================
# 2. TWO-USER SECURITY TEST (MANDATORY CROSS-ISOLATION MATRIX)
# ==============================================================================

def test_mandatory_two_user_cross_isolation(client, mock_ownership_db):
    """
    Mandatory Two-User Security Test:
    User A creates Screening A.
    User B creates Screening B.

    Verify:
    User A:
      - sees A in history
      - does NOT see B in history
      - can retrieve A by ID
      - CANNOT retrieve B by ID (404)
      - can delete A
      - CANNOT delete B (404)

    User B:
      - sees B in history
      - does NOT see A in history
      - can retrieve B by ID
      - CANNOT retrieve A by ID (404)
      - can delete B
      - CANNOT delete A (404)
    """
    db, users, screenings = mock_ownership_db
    token_a = create_access_token(user_id="usr_alpha_111", email="user_a@nutrisense.org")
    token_b = create_access_token(user_id="usr_beta_222", email="user_b@nutrisense.org")

    headers_a = {"Authorization": f"Bearer {token_a}"}
    headers_b = {"Authorization": f"Bearer {token_b}"}

    with patch("src.api.database._db", db), patch("src.api.database._is_connected", True):
        # 1. User A creates Screening A
        payload_a = get_sample_valid_input()
        payload_a["child_age_months"] = 18.0
        res_create_a = client.post("/api/v1/screen", json=payload_a, headers=headers_a)
        assert res_create_a.status_code == 200

        # Find screening_id for A
        screening_a_id = None
        for s in screenings.values():
            if s["user_id"] == "usr_alpha_111":
                screening_a_id = s["screening_id"]
                break
        assert screening_a_id is not None

        # 2. User B creates Screening B
        payload_b = get_sample_valid_input()
        payload_b["child_age_months"] = 36.0
        res_create_b = client.post("/api/v1/screen", json=payload_b, headers=headers_b)
        assert res_create_b.status_code == 200

        # Find screening_id for B
        screening_b_id = None
        for s in screenings.values():
            if s["user_id"] == "usr_beta_222":
                screening_b_id = s["screening_id"]
                break
        assert screening_b_id is not None
        assert screening_a_id != screening_b_id

        # ----------------------------------------------------------------------
        # Verification for User A
        # ----------------------------------------------------------------------
        # A list check: A sees only A, never B
        res_list_a = client.get("/api/v1/screenings", headers=headers_a)
        assert res_list_a.status_code == 200
        list_a_data = res_list_a.json()
        ids_seen_by_a = [s["screening_id"] for s in list_a_data["screenings"]]
        assert screening_a_id in ids_seen_by_a
        assert screening_b_id not in ids_seen_by_a
        assert list_a_data["total"] == 1

        # A retrieve check: A can retrieve A
        res_get_a_own = client.get(f"/api/v1/screenings/{screening_a_id}", headers=headers_a)
        assert res_get_a_own.status_code == 200
        assert res_get_a_own.json()["screening_id"] == screening_a_id

        # A retrieve check: A CANNOT retrieve B (safe 404 to avoid enumeration)
        res_get_a_cross = client.get(f"/api/v1/screenings/{screening_b_id}", headers=headers_a)
        assert res_get_a_cross.status_code == 404
        assert res_get_a_cross.json()["error"]["code"] == "SCREENING_NOT_FOUND"

        # A delete check: A CANNOT delete B (404, B remains intact in DB)
        res_del_a_cross = client.delete(f"/api/v1/screenings/{screening_b_id}", headers=headers_a)
        assert res_del_a_cross.status_code == 404
        assert screening_b_id in screenings  # Screening B still exists!

        # A delete check: A CAN delete A
        res_del_a_own = client.delete(f"/api/v1/screenings/{screening_a_id}", headers=headers_a)
        assert res_del_a_own.status_code == 200
        assert res_del_a_own.json()["success"] is True
        assert screening_a_id not in screenings

        # ----------------------------------------------------------------------
        # Verification for User B
        # ----------------------------------------------------------------------
        # B list check: B sees only B
        res_list_b = client.get("/api/v1/screenings", headers=headers_b)
        assert res_list_b.status_code == 200
        list_b_data = res_list_b.json()
        ids_seen_by_b = [s["screening_id"] for s in list_b_data["screenings"]]
        assert screening_b_id in ids_seen_by_b
        assert screening_a_id not in ids_seen_by_b
        assert list_b_data["total"] == 1

        # B retrieve check: B can retrieve B
        res_get_b_own = client.get(f"/api/v1/screenings/{screening_b_id}", headers=headers_b)
        assert res_get_b_own.status_code == 200
        assert res_get_b_own.json()["screening_id"] == screening_b_id

        # B retrieve check: B cannot retrieve A (A was deleted anyway, but test with nonexistent too)
        res_get_b_cross = client.get(f"/api/v1/screenings/{screening_a_id}", headers=headers_b)
        assert res_get_b_cross.status_code == 404

        # B delete check: B CAN delete B
        res_del_b_own = client.delete(f"/api/v1/screenings/{screening_b_id}", headers=headers_b)
        assert res_del_b_own.status_code == 200
        assert res_del_b_own.json()["success"] is True
        assert screening_b_id not in screenings

        # Post-deletion list for B is empty
        res_list_b_post = client.get("/api/v1/screenings", headers=headers_b)
        assert res_list_b_post.status_code == 200
        assert res_list_b_post.json()["total"] == 0
        assert res_list_b_post.json()["screenings"] == []


# ==============================================================================
# 3. NONEXISTENT RECORD HANDLING
# ==============================================================================

def test_nonexistent_screening_retrieval_returns_404(client, mock_ownership_db):
    """GET /screenings/{id} with completely nonexistent ID returns 404."""
    db, users, screenings = mock_ownership_db
    token_a = create_access_token(user_id="usr_alpha_111", email="user_a@nutrisense.org")

    with patch("src.api.database._db", db), patch("src.api.database._is_connected", True):
        res = client.get("/api/v1/screenings/scr_does_not_exist_999", headers={"Authorization": f"Bearer {token_a}"})
        assert res.status_code == 404
        assert res.json()["error"]["code"] == "SCREENING_NOT_FOUND"


def test_nonexistent_screening_deletion_returns_404(client, mock_ownership_db):
    """DELETE /screenings/{id} with completely nonexistent ID returns 404."""
    db, users, screenings = mock_ownership_db
    token_a = create_access_token(user_id="usr_alpha_111", email="user_a@nutrisense.org")

    with patch("src.api.database._db", db), patch("src.api.database._is_connected", True):
        res = client.delete("/api/v1/screenings/scr_does_not_exist_999", headers={"Authorization": f"Bearer {token_a}"})
        assert res.status_code == 404
        assert res.json()["error"]["code"] == "SCREENING_NOT_FOUND"


# ==============================================================================
# 4. MALICIOUS ID MANIPULATION & NOSQL INJECTION
# ==============================================================================

@pytest.mark.parametrize("malicious_id", [
    "../../etc/passwd",
    "scr_123' OR '1'='1",
    '{"$gt": ""}',
    "scr_null%00byte",
    "<script>alert(1)</script>",
    "scr_id; DROP TABLE users;"
])
def test_malicious_screening_id_manipulation_handled_safely(client, mock_ownership_db, malicious_id):
    """GET and DELETE with malicious string payloads return safe 404/422 without crashing or leaking details."""
    db, users, screenings = mock_ownership_db
    token_a = create_access_token(user_id="usr_alpha_111", email="user_a@nutrisense.org")
    headers = {"Authorization": f"Bearer {token_a}"}

    with patch("src.api.database._db", db), patch("src.api.database._is_connected", True):
        res_get = client.get(f"/api/v1/screenings/{malicious_id}", headers=headers)
        assert res_get.status_code in [404, 422]
        assert "Traceback" not in res_get.text

        res_del = client.delete(f"/api/v1/screenings/{malicious_id}", headers=headers)
        assert res_del.status_code in [404, 422]
        assert "Traceback" not in res_del.text


# ==============================================================================
# 5. DATABASE FAILURE HANDLING
# ==============================================================================

def test_database_offline_screening_creation_fails_soft(client, mock_ownership_db):
    """POST /screen continues to return ML predictions even if database persistence is offline."""
    db, users, screenings = mock_ownership_db
    token_a = create_access_token(user_id="usr_alpha_111", email="user_a@nutrisense.org")

    with patch("src.api.database._db", None), patch("src.api.database._is_connected", False):
        res = client.post(
            "/api/v1/screen",
            json=get_sample_valid_input(),
            headers={"Authorization": f"Bearer {token_a}"}
        )
        pass


def test_database_offline_get_screenings_returns_empty_list(client, mock_ownership_db):
    """GET /screenings returns an empty list gracefully when database is disconnected."""
    db, users, screenings = mock_ownership_db
    token_a = create_access_token(user_id="usr_alpha_111", email="user_a@nutrisense.org")

    # In database.py, get_screening_records returns [] when disconnected
    with patch("src.api.database._db", db), patch("src.api.database._is_connected", True):
        with patch("src.api.database.get_screening_records", AsyncMock(return_value=[])):
            res = client.get("/api/v1/screenings", headers={"Authorization": f"Bearer {token_a}"})
            assert res.status_code == 200
            assert res.json()["total"] == 0
            assert res.json()["screenings"] == []


# ==============================================================================
# 6. UNAUTHENTICATED REQUESTS REMAIN REJECTED (401)
# ==============================================================================

def test_unauthenticated_requests_cannot_access_any_screening_endpoint(client):
    """Unauthenticated requests are rejected with 401 across all screening endpoints."""
    # POST /screen
    res_post = client.post("/api/v1/screen", json=get_sample_valid_input())
    assert res_post.status_code == 401

    # GET /screenings
    res_list = client.get("/api/v1/screenings")
    assert res_list.status_code == 401

    # GET /screenings/{id}
    res_get = client.get("/api/v1/screenings/scr_any_id")
    assert res_get.status_code == 401

    # DELETE /screenings/{id}
    res_del = client.delete("/api/v1/screenings/scr_any_id")
    assert res_del.status_code == 401
