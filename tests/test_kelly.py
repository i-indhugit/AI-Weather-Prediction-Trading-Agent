"""
tests/test_kelly.py
====================
Unit tests for the Kelly Criterion calculator (utils/kelly.py).

Tests cover:
- Correct formula output for known inputs
- Edge capping behaviour
- BUY_NO (kelly_for_no) sizing
- Invalid input validation
- Zero-edge scenarios
"""

from __future__ import annotations

import pytest

from utils.kelly import KellyResult, kelly_criterion, kelly_for_no


# ===========================================================================
# kelly_criterion — happy path
# ===========================================================================

class TestKellyCriterionHappyPath:
    """Tests for the main kelly_criterion function with valid inputs."""

    def test_returns_kelly_result(self) -> None:
        """Result should be a KellyResult dataclass."""
        result = kelly_criterion(
            probability=0.6,
            market_price=0.5,
            capital=10_000.0,
        )
        assert isinstance(result, KellyResult)

    def test_positive_edge_yields_positive_fraction(self) -> None:
        """When model prob > market prob, edge and fraction should be positive."""
        result = kelly_criterion(
            probability=0.7,
            market_price=0.5,
            capital=10_000.0,
        )
        assert result.edge > 0
        assert result.fraction > 0
        assert result.is_positive_edge is True

    def test_known_kelly_formula_output(self) -> None:
        """
        Validate against hand-calculated values.

        Given:
            p=0.6, market_price=0.5  → b = (1-0.5)/0.5 = 1.0
            f* = (0.6*1.0 - 0.4) / 1.0 = 0.2
        """
        result = kelly_criterion(
            probability=0.6,
            market_price=0.5,
            capital=10_000.0,
            max_fraction=0.25,
        )
        assert abs(result.fraction - 0.2) < 1e-6
        assert abs(result.capped_fraction - 0.2) < 1e-6
        assert abs(result.position_size - 2_000.0) < 0.01

    def test_fraction_is_capped_at_max(self) -> None:
        """Fraction must never exceed max_fraction."""
        result = kelly_criterion(
            probability=0.95,
            market_price=0.1,  # Huge odds → huge raw Kelly
            capital=10_000.0,
            max_fraction=0.25,
        )
        assert result.capped_fraction <= 0.25
        assert result.fraction >= result.capped_fraction

    def test_position_size_matches_capital_times_fraction(self) -> None:
        """position_size should equal capped_fraction × capital."""
        capital = 8_000.0
        result = kelly_criterion(
            probability=0.6,
            market_price=0.5,
            capital=capital,
        )
        expected = round(result.capped_fraction * capital, 2)
        assert result.position_size == expected

    def test_negative_edge_returns_zero_fraction(self) -> None:
        """When model prob < market prob, we have negative edge → no bet."""
        result = kelly_criterion(
            probability=0.3,
            market_price=0.5,
            capital=10_000.0,
        )
        assert result.fraction == 0.0
        assert result.capped_fraction == 0.0
        assert result.position_size == 0.0
        assert result.is_positive_edge is False

    def test_high_odds_large_edge(self) -> None:
        """High market odds amplify the Kelly fraction."""
        result_low_odds = kelly_criterion(
            probability=0.6, market_price=0.5, capital=10_000.0
        )
        result_high_odds = kelly_criterion(
            probability=0.6, market_price=0.2, capital=10_000.0
        )
        # Higher odds (market_price=0.2 → b=4.0) should yield higher raw Kelly
        assert result_high_odds.fraction > result_low_odds.fraction


# ===========================================================================
# kelly_for_no
# ===========================================================================

class TestKellyForNo:
    """Tests for the kelly_for_no function."""

    def test_no_position_inverts_probability(self) -> None:
        """
        Buying NO when model says LOW rain probability.

        If model says 30% rain and market says 50%, buying NO has a positive
        edge because we think NO is underpriced.
        """
        result = kelly_for_no(
            probability=0.3,    # 30% chance of rain
            market_price=0.5,   # Market prices YES at 50%
            capital=10_000.0,
        )
        # From NO perspective: win_prob=0.7, NO_price=0.5, b=1.0
        # f* = (0.7*1.0 - 0.3)/1.0 = 0.4 → capped at 0.25
        assert result.is_positive_edge is True
        assert result.capped_fraction <= 0.25

    def test_no_position_when_model_agrees_with_market(self) -> None:
        """If model prob ≈ market prob for NO side, no strong edge."""
        result = kelly_for_no(
            probability=0.5,
            market_price=0.5,
            capital=10_000.0,
        )
        assert result.edge == pytest.approx(0.0, abs=1e-9)


# ===========================================================================
# Input validation
# ===========================================================================

class TestKellyValidation:
    """Tests that invalid inputs raise ValueError."""

    @pytest.mark.parametrize("probability", [0.0, 1.0, -0.1, 1.5])
    def test_invalid_probability_raises(self, probability: float) -> None:
        with pytest.raises(ValueError, match="probability"):
            kelly_criterion(
                probability=probability,
                market_price=0.5,
                capital=10_000.0,
            )

    @pytest.mark.parametrize("market_price", [0.0, 1.0, -0.1, 1.5])
    def test_invalid_market_price_raises(self, market_price: float) -> None:
        with pytest.raises(ValueError, match="market_price"):
            kelly_criterion(
                probability=0.6,
                market_price=market_price,
                capital=10_000.0,
            )

    @pytest.mark.parametrize("capital", [0.0, -100.0])
    def test_invalid_capital_raises(self, capital: float) -> None:
        with pytest.raises(ValueError, match="capital"):
            kelly_criterion(
                probability=0.6,
                market_price=0.5,
                capital=capital,
            )

    @pytest.mark.parametrize("max_fraction", [0.0, 1.1, -0.5])
    def test_invalid_max_fraction_raises(self, max_fraction: float) -> None:
        with pytest.raises(ValueError, match="max_fraction"):
            kelly_criterion(
                probability=0.6,
                market_price=0.5,
                capital=10_000.0,
                max_fraction=max_fraction,
            )
