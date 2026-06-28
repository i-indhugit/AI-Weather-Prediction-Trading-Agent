"""
utils/kelly.py
==============
Kelly Criterion position-sizing calculator.

The Kelly Criterion determines the optimal fraction of capital to wager
given a known edge and odds.  We implement both the full-Kelly and a
fractional-Kelly (capped at max_fraction) to limit variance.

Formula
-------
    f* = (p * b - q) / b

where:
    p = probability of winning (model prediction)
    q = 1 - p  (probability of losing)
    b = net decimal odds (e.g., 1.0 for even-money binary markets)

For Polymarket-style binary prediction markets the payout is approximately
1 USDC per YES share, so b ≈ 1.0 when you buy YES at 0.50, but scales
with the market price.  We compute b from the market price.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class KellyResult:
    """Immutable result of a Kelly calculation."""

    fraction: float
    """Raw Kelly fraction (may exceed max_fraction before clamping)."""

    capped_fraction: float
    """Final fraction after applying max_fraction cap."""

    position_size: float
    """Dollar amount to allocate (capped_fraction × capital)."""

    capital: float
    """Capital used as input."""

    probability: float
    """Win probability used as input."""

    odds: float
    """Net decimal odds used as input."""

    edge: float
    """Expected value of the bet (p * b - q)."""

    is_positive_edge: bool
    """True if the bet has a positive expected value."""


def kelly_criterion(
    probability: float,
    market_price: float,
    capital: float,
    max_fraction: float = 0.25,
) -> KellyResult:
    """
    Compute the Kelly-optimal position size for a binary prediction market.

    Args:
        probability:  Predicted win probability in [0, 1].
        market_price: Current market price of the contract in [0, 1].
                      For a YES contract this is the cost per share.
                      The implied odds b = (1 - market_price) / market_price.
        capital:      Available capital in USD.
        max_fraction: Maximum allowable Kelly fraction (default 0.25 = 25%).

    Returns:
        A :class:`KellyResult` with the recommended position size.

    Raises:
        ValueError: If inputs are outside valid ranges.

    Examples::

        # Model says 70% chance of rain, market says 50%
        result = kelly_criterion(
            probability=0.70,
            market_price=0.50,
            capital=10_000.0,
        )
        print(result.position_size)   # 1_000.0 (10% of capital)
    """
    # ── Input validation ──────────────────────────────────────────────────
    if not 0.0 < probability < 1.0:
        raise ValueError(f"probability must be in (0, 1), got {probability}")
    if not 0.0 < market_price < 1.0:
        raise ValueError(f"market_price must be in (0, 1), got {market_price}")
    if capital <= 0:
        raise ValueError(f"capital must be positive, got {capital}")
    if not 0.0 < max_fraction <= 1.0:
        raise ValueError(f"max_fraction must be in (0, 1], got {max_fraction}")

    # ── Compute odds from market price ───────────────────────────────────
    # In a binary market priced at p_market, buying YES at price p_market
    # gives a net gain of (1 - p_market) per dollar staked if it resolves YES.
    # Net decimal odds b = (1 - p_market) / p_market
    b: float = (1.0 - market_price) / market_price
    p: float = probability
    q: float = 1.0 - p

    # ── Kelly formula ─────────────────────────────────────────────────────
    edge: float = p * b - q
    raw_fraction: float = edge / b if b > 0 else 0.0

    # Negative Kelly → don't bet
    raw_fraction = max(raw_fraction, 0.0)

    # Apply cap
    capped_fraction: float = min(raw_fraction, max_fraction)

    position_size: float = capped_fraction * capital

    return KellyResult(
        fraction=raw_fraction,
        capped_fraction=capped_fraction,
        position_size=round(position_size, 2),
        capital=capital,
        probability=p,
        odds=b,
        edge=round(edge, 6),
        is_positive_edge=edge > 0.0,
    )


def kelly_for_no(
    probability: float,
    market_price: float,
    capital: float,
    max_fraction: float = 0.25,
) -> KellyResult:
    """
    Compute Kelly sizing for a BUY NO decision.

    When buying NO, the model believes the event will NOT happen (1 - probability).
    The NO contract costs (1 - market_price) per share.

    Args:
        probability:  Model's predicted probability that event WILL happen.
        market_price: Current YES price (NO price = 1 - market_price).
        capital:      Available capital.
        max_fraction: Maximum Kelly fraction.

    Returns:
        KellyResult sized for the NO position.
    """
    # Flip perspective: "NO" win prob = 1 - model_prob, NO price = 1 - market_price
    return kelly_criterion(
        probability=1.0 - probability,
        market_price=1.0 - market_price,
        capital=capital,
        max_fraction=max_fraction,
    )
