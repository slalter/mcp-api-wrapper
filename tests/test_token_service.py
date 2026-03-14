"""Tests for the auth token service."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import jwt
import pytest

from mcp_api_wrapper.auth.token_service import TokenService
from mcp_api_wrapper.config import Settings


class TestTokenIssuance:
    """Test token generation."""

    def test_issue_token_valid_client(self, token_service: TokenService) -> None:
        token, claims = token_service.issue_token("test-client", "test-secret")
        assert token
        assert claims.sub == "test-client"
        assert "read" in claims.scopes
        assert claims.exp > datetime.now(timezone.utc)

    def test_issue_token_invalid_client(self, token_service: TokenService) -> None:
        with pytest.raises(ValueError, match="Invalid client credentials"):
            token_service.issue_token("nonexistent", "bad-secret")

    def test_issue_token_wrong_secret(self, token_service: TokenService) -> None:
        with pytest.raises(ValueError, match="Invalid client credentials"):
            token_service.issue_token("test-client", "wrong-secret")

    def test_scope_intersection(self, token_service: TokenService) -> None:
        """Only requested scopes that are allowed get granted."""
        _, claims = token_service.issue_token(
            "test-client", "test-secret", requested_scopes=["read", "delete"]
        )
        assert claims.scopes == ["read"]  # "delete" not in allowed

    def test_no_valid_scopes_raises(self, token_service: TokenService) -> None:
        with pytest.raises(ValueError, match="None of the requested scopes"):
            token_service.issue_token(
                "test-client", "test-secret", requested_scopes=["nuke"]
            )

    def test_readonly_client_limited_scopes(self, token_service: TokenService) -> None:
        _, claims = token_service.issue_token("readonly-client", "readonly-secret")
        assert claims.scopes == ["read"]

    def test_readonly_client_cannot_get_write(self, token_service: TokenService) -> None:
        with pytest.raises(ValueError, match="None of the requested scopes"):
            token_service.issue_token(
                "readonly-client", "readonly-secret", requested_scopes=["write"]
            )

    def test_token_ttl(self, settings: Settings, token_service: TokenService) -> None:
        _, claims = token_service.issue_token("test-client", "test-secret")
        expected_exp = claims.iat + timedelta(minutes=settings.auth_token_ttl_minutes)
        # Allow 2 second tolerance
        assert abs((claims.exp - expected_exp).total_seconds()) < 2


class TestTokenValidation:
    """Test token validation."""

    def test_validate_valid_token(self, token_service: TokenService) -> None:
        token, original_claims = token_service.issue_token("test-client", "test-secret")
        validated = token_service.validate_token(token)
        assert validated.sub == original_claims.sub
        assert validated.scopes == original_claims.scopes

    def test_validate_expired_token(self, settings: Settings) -> None:
        svc = TokenService(settings)
        svc.register_client("exp-client", "exp-secret", ["read"])
        # Manually create an expired token
        payload = {
            "sub": "exp-client",
            "scopes": ["read"],
            "iss": settings.auth_issuer,
            "iat": datetime.now(timezone.utc) - timedelta(hours=2),
            "exp": datetime.now(timezone.utc) - timedelta(hours=1),
        }
        expired_token = jwt.encode(payload, settings.auth_secret_key, algorithm="HS256")
        with pytest.raises(jwt.ExpiredSignatureError):
            svc.validate_token(expired_token)

    def test_validate_tampered_token(self, token_service: TokenService) -> None:
        token, _ = token_service.issue_token("test-client", "test-secret")
        # Tamper with the token
        tampered = token[:-5] + "XXXXX"
        with pytest.raises(jwt.InvalidSignatureError):
            token_service.validate_token(tampered)

    def test_validate_wrong_issuer(self, settings: Settings) -> None:
        svc = TokenService(settings)
        payload = {
            "sub": "client",
            "scopes": ["read"],
            "iss": "wrong-issuer",
            "iat": datetime.now(timezone.utc),
            "exp": datetime.now(timezone.utc) + timedelta(hours=1),
        }
        token = jwt.encode(payload, settings.auth_secret_key, algorithm="HS256")
        with pytest.raises(jwt.InvalidIssuerError):
            svc.validate_token(token)
