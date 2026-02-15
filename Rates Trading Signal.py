"""
Rates Market Making Trading Signal Generator
============================================
A comprehensive signal system for rates market making strategies including
spread signals, volatility signals, flow imbalance, and inventory management.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum


class SignalType(Enum):
    """Types of trading signals"""
    WIDEN_SPREAD = "WIDEN_SPREAD"
    TIGHTEN_SPREAD = "TIGHTEN_SPREAD"
    SKEW_BID = "SKEW_BID"
    SKEW_OFFER = "SKEW_OFFER"
    REDUCE_POSITION = "REDUCE_POSITION"
    NEUTRAL = "NEUTRAL"


class SignalStrength(Enum):
    """Signal strength levels"""
    WEAK = 1
    MODERATE = 2
    STRONG = 3


@dataclass
class MarketData:
    """Market data snapshot"""
    timestamp: datetime
    mid_price: float
    bid_price: float
    offer_price: float
    bid_size: float
    offer_size: float
    recent_trades: List[Tuple[float, float, str]]  # (price, size, side)
    volatility: float
    benchmark_spread: float


@dataclass
class PositionData:
    """Current position information"""
    current_position: float
    position_limit: float
    pnl: float
    inventory_age: float  # hours


@dataclass
class TradingSignal:
    """Trading signal output"""
    timestamp: datetime
    signal_type: SignalType
    strength: SignalStrength
    recommended_bid_spread: float  # bps
    recommended_offer_spread: float  # bps
    confidence: float  # 0-1
    reasons: List[str]
    metrics: Dict[str, float]


class RatesMarketMakingSignals:
    """
    Generate trading signals for rates market making based on:
    - Flow imbalance
    - Volatility regime
    - Inventory position
    - Market microstructure
    """
    
    def __init__(
        self,
        base_spread_bps: float = 2.0,
        max_spread_bps: float = 10.0,
        min_spread_bps: float = 0.5,
        inventory_threshold: float = 0.7,
        volatility_lookback: int = 20,
    ):
        """
        Initialize signal generator with parameters
        
        Args:
            base_spread_bps: Normal market making spread in basis points
            max_spread_bps: Maximum allowable spread
            min_spread_bps: Minimum allowable spread
            inventory_threshold: Position % that triggers inventory management (0-1)
            volatility_lookback: Number of periods for volatility calculation
        """
        self.base_spread_bps = base_spread_bps
        self.max_spread_bps = max_spread_bps
        self.min_spread_bps = min_spread_bps
        self.inventory_threshold = inventory_threshold
        self.volatility_lookback = volatility_lookback
        
        # Historical data for calculations
        self.price_history: List[float] = []
        self.trade_history: List[Tuple[datetime, float, float, str]] = []
        
    def generate_signal(
        self,
        market_data: MarketData,
        position_data: PositionData
    ) -> TradingSignal:
        """
        Generate comprehensive trading signal
        
        Args:
            market_data: Current market data snapshot
            position_data: Current position information
            
        Returns:
            TradingSignal with recommendations
        """
        # Update history
        self.price_history.append(market_data.mid_price)
        if len(self.price_history) > 100:
            self.price_history.pop(0)
        
        # Calculate component signals
        flow_signal = self._analyze_flow_imbalance(market_data)
        volatility_signal = self._analyze_volatility(market_data)
        inventory_signal = self._analyze_inventory(position_data)
        microstructure_signal = self._analyze_microstructure(market_data)
        
        # Combine signals
        combined_signal = self._combine_signals(
            flow_signal,
            volatility_signal,
            inventory_signal,
            microstructure_signal,
            position_data
        )
        
        return combined_signal
    
    def _analyze_flow_imbalance(self, market_data: MarketData) -> Dict:
        """
        Analyze order flow imbalance to detect directional pressure
        
        Returns dict with:
        - imbalance_ratio: bid/offer flow ratio
        - signal: direction to skew quotes
        - confidence: 0-1
        """
        if not market_data.recent_trades:
            return {
                'imbalance_ratio': 1.0,
                'signal': 'neutral',
                'confidence': 0.0,
                'weight': 0.3
            }
        
        # Calculate flow by side
        bid_volume = sum(size for _, size, side in market_data.recent_trades if side == 'BUY')
        offer_volume = sum(size for _, size, side in market_data.recent_trades if side == 'SELL')
        
        total_volume = bid_volume + offer_volume
        if total_volume == 0:
            return {
                'imbalance_ratio': 1.0,
                'signal': 'neutral',
                'confidence': 0.0,
                'weight': 0.3
            }
        
        # Imbalance ratio > 1 means more buying pressure
        imbalance_ratio = (bid_volume + 1) / (offer_volume + 1)
        
        # Determine signal
        if imbalance_ratio > 1.5:
            signal = 'skew_offer'  # More buyers, lift offer
            confidence = min((imbalance_ratio - 1.0) / 2.0, 1.0)
        elif imbalance_ratio < 0.67:
            signal = 'skew_bid'  # More sellers, lift bid
            confidence = min((1.0 - imbalance_ratio) / 0.5, 1.0)
        else:
            signal = 'neutral'
            confidence = 0.3
        
        return {
            'imbalance_ratio': imbalance_ratio,
            'signal': signal,
            'confidence': confidence,
            'weight': 0.3
        }
    
    def _analyze_volatility(self, market_data: MarketData) -> Dict:
        """
        Analyze volatility regime to adjust spreads
        
        Returns dict with volatility metrics and spread adjustment
        """
        current_vol = market_data.volatility
        
        # Calculate historical volatility if we have enough data
        if len(self.price_history) >= self.volatility_lookback:
            returns = np.diff(self.price_history[-self.volatility_lookback:])
            historical_vol = np.std(returns) * np.sqrt(252)  # Annualized
        else:
            historical_vol = current_vol
        
        # Volatility ratio
        vol_ratio = current_vol / (historical_vol + 1e-6)
        
        # Determine spread adjustment
        if vol_ratio > 1.5:
            signal = 'widen_spread'
            spread_multiplier = min(vol_ratio, 3.0)
            confidence = min((vol_ratio - 1.0) / 2.0, 1.0)
        elif vol_ratio < 0.7:
            signal = 'tighten_spread'
            spread_multiplier = max(vol_ratio, 0.5)
            confidence = min((1.0 - vol_ratio) / 0.5, 0.8)
        else:
            signal = 'neutral'
            spread_multiplier = 1.0
            confidence = 0.3
        
        return {
            'current_vol': current_vol,
            'historical_vol': historical_vol,
            'vol_ratio': vol_ratio,
            'signal': signal,
            'spread_multiplier': spread_multiplier,
            'confidence': confidence,
            'weight': 0.25
        }
    
    def _analyze_inventory(self, position_data: PositionData) -> Dict:
        """
        Analyze inventory position and recommend adjustments
        
        Returns dict with inventory metrics and skew recommendations
        """
        position_pct = abs(position_data.current_position) / position_data.position_limit
        
        # Inventory urgency increases with position size and age
        age_factor = min(position_data.inventory_age / 24.0, 1.0)  # Normalize to 24 hours
        urgency = position_pct * (1 + age_factor)
        
        # Determine signal based on position
        if position_data.current_position > position_data.position_limit * self.inventory_threshold:
            # Long position - incentivize selling
            signal = 'skew_bid'  # Make bid less attractive, offer more attractive
            skew_magnitude = position_pct
            confidence = min(urgency, 1.0)
        elif position_data.current_position < -position_data.position_limit * self.inventory_threshold:
            # Short position - incentivize buying
            signal = 'skew_offer'  # Make offer less attractive, bid more attractive
            skew_magnitude = position_pct
            confidence = min(urgency, 1.0)
        else:
            signal = 'neutral'
            skew_magnitude = 0.0
            confidence = 0.0
        
        return {
            'position_pct': position_pct,
            'urgency': urgency,
            'signal': signal,
            'skew_magnitude': skew_magnitude,
            'confidence': confidence,
            'weight': 0.35
        }
    
    def _analyze_microstructure(self, market_data: MarketData) -> Dict:
        """
        Analyze market microstructure signals
        
        Returns dict with microstructure metrics
        """
        # Bid-ask spread width relative to benchmark
        current_spread_bps = (market_data.offer_price - market_data.bid_price) / market_data.mid_price * 10000
        spread_ratio = current_spread_bps / (market_data.benchmark_spread + 1e-6)
        
        # Order book imbalance
        book_imbalance = (market_data.bid_size - market_data.offer_size) / (market_data.bid_size + market_data.offer_size + 1e-6)
        
        # Determine signal
        if spread_ratio > 1.3:
            signal = 'tighten_spread'  # Market spread is wide, opportunity to tighten
            confidence = min((spread_ratio - 1.0) / 1.0, 0.7)
        elif spread_ratio < 0.7:
            signal = 'widen_spread'  # Market spread is tight, widen for safety
            confidence = min((1.0 - spread_ratio) / 0.5, 0.7)
        else:
            signal = 'neutral'
            confidence = 0.2
        
        return {
            'current_spread_bps': current_spread_bps,
            'spread_ratio': spread_ratio,
            'book_imbalance': book_imbalance,
            'signal': signal,
            'confidence': confidence,
            'weight': 0.1
        }
    
    def _combine_signals(
        self,
        flow: Dict,
        volatility: Dict,
        inventory: Dict,
        microstructure: Dict,
        position_data: PositionData
    ) -> TradingSignal:
        """
        Combine all signals into final trading recommendation
        """
        # Calculate weighted confidence for each signal type
        signals = [flow, volatility, inventory, microstructure]
        
        # Start with base spread
        bid_spread = self.base_spread_bps
        offer_spread = self.base_spread_bps
        
        # Apply volatility adjustment (affects both sides)
        vol_mult = volatility['spread_multiplier']
        bid_spread *= vol_mult
        offer_spread *= vol_mult
        
        # Apply inventory skew (asymmetric adjustment)
        if inventory['signal'] == 'skew_bid':
            # Make bid less attractive (wider), offer more attractive (tighter)
            bid_spread *= (1 + inventory['skew_magnitude'] * 0.5)
            offer_spread *= (1 - inventory['skew_magnitude'] * 0.3)
        elif inventory['signal'] == 'skew_offer':
            # Make offer less attractive (wider), bid more attractive (tighter)
            offer_spread *= (1 + inventory['skew_magnitude'] * 0.5)
            bid_spread *= (1 - inventory['skew_magnitude'] * 0.3)
        
        # Apply flow imbalance adjustments
        if flow['signal'] == 'skew_bid':
            bid_spread *= (1 + flow['confidence'] * 0.3)
        elif flow['signal'] == 'skew_offer':
            offer_spread *= (1 + flow['confidence'] * 0.3)
        
        # Apply microstructure adjustments
        if microstructure['signal'] == 'widen_spread':
            bid_spread *= (1 + microstructure['confidence'] * 0.2)
            offer_spread *= (1 + microstructure['confidence'] * 0.2)
        elif microstructure['signal'] == 'tighten_spread':
            bid_spread *= (1 - microstructure['confidence'] * 0.15)
            offer_spread *= (1 - microstructure['confidence'] * 0.15)
        
        # Apply bounds
        bid_spread = np.clip(bid_spread, self.min_spread_bps, self.max_spread_bps)
        offer_spread = np.clip(offer_spread, self.min_spread_bps, self.max_spread_bps)
        
        # Determine primary signal type
        signal_type = self._determine_primary_signal(flow, volatility, inventory, microstructure)
        
        # Calculate overall confidence
        total_weight = sum(s['weight'] for s in signals)
        overall_confidence = sum(s['confidence'] * s['weight'] for s in signals) / total_weight
        
        # Determine signal strength
        if overall_confidence > 0.7:
            strength = SignalStrength.STRONG
        elif overall_confidence > 0.4:
            strength = SignalStrength.MODERATE
        else:
            strength = SignalStrength.WEAK
        
        # Build reasons list
        reasons = []
        if inventory['confidence'] > 0.5:
            reasons.append(f"Inventory at {inventory['position_pct']:.1%} of limit - {inventory['signal']}")
        if volatility['confidence'] > 0.5:
            reasons.append(f"Volatility {volatility['vol_ratio']:.2f}x historical - {volatility['signal']}")
        if flow['confidence'] > 0.5:
            reasons.append(f"Flow imbalance {flow['imbalance_ratio']:.2f} - {flow['signal']}")
        if microstructure['confidence'] > 0.4:
            reasons.append(f"Spread {microstructure['spread_ratio']:.2f}x benchmark - {microstructure['signal']}")
        
        if not reasons:
            reasons.append("Normal market making conditions")
        
        # Compile metrics
        metrics = {
            'flow_imbalance': flow['imbalance_ratio'],
            'volatility_ratio': volatility['vol_ratio'],
            'position_pct': inventory['position_pct'],
            'spread_ratio': microstructure['spread_ratio'],
            'inventory_urgency': inventory['urgency']
        }
        
        return TradingSignal(
            timestamp=datetime.now(),
            signal_type=signal_type,
            strength=strength,
            recommended_bid_spread=round(bid_spread, 2),
            recommended_offer_spread=round(offer_spread, 2),
            confidence=round(overall_confidence, 3),
            reasons=reasons,
            metrics=metrics
        )
    
    def _determine_primary_signal(
        self,
        flow: Dict,
        volatility: Dict,
        inventory: Dict,
        microstructure: Dict
    ) -> SignalType:
        """Determine the primary signal type based on strongest component"""
        
        # Inventory takes priority if confidence is high
        if inventory['confidence'] > 0.6:
            if inventory['signal'] == 'skew_bid':
                return SignalType.SKEW_BID
            elif inventory['signal'] == 'skew_offer':
                return SignalType.SKEW_OFFER
        
        # Then volatility
        if volatility['confidence'] > 0.6:
            if volatility['signal'] == 'widen_spread':
                return SignalType.WIDEN_SPREAD
            elif volatility['signal'] == 'tighten_spread':
                return SignalType.TIGHTEN_SPREAD
        
        # Then flow
        if flow['confidence'] > 0.5:
            if flow['signal'] == 'skew_bid':
                return SignalType.SKEW_BID
            elif flow['signal'] == 'skew_offer':
                return SignalType.SKEW_OFFER
        
        return SignalType.NEUTRAL


def example_usage():
    """
    Example of how to use the signal generator
    """
    # Initialize signal generator
    signal_gen = RatesMarketMakingSignals(
        base_spread_bps=2.0,
        max_spread_bps=10.0,
        min_spread_bps=0.5,
        inventory_threshold=0.7
    )
    
    # Create sample market data
    market_data = MarketData(
        timestamp=datetime.now(),
        mid_price=100.0,
        bid_price=99.98,
        offer_price=100.02,
        bid_size=10_000_000,
        offer_size=8_000_000,
        recent_trades=[
            (100.01, 2_000_000, 'BUY'),
            (100.00, 3_000_000, 'BUY'),
            (99.99, 1_000_000, 'SELL'),
            (100.02, 4_000_000, 'BUY'),
        ],
        volatility=0.0015,  # 15 bps
        benchmark_spread=2.5
    )
    
    # Create sample position data
    position_data = PositionData(
        current_position=75_000_000,  # Long 75MM
        position_limit=100_000_000,
        pnl=25_000,
        inventory_age=3.5  # hours
    )
    
    # Generate signal
    signal = signal_gen.generate_signal(market_data, position_data)
    
    # Display results
    print("=" * 80)
    print("RATES MARKET MAKING TRADING SIGNAL")
    print("=" * 80)
    print(f"Timestamp: {signal.timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Signal Type: {signal.signal_type.value}")
    print(f"Strength: {signal.strength.name}")
    print(f"Confidence: {signal.confidence:.1%}")
    print()
    print(f"Recommended Bid Spread: {signal.recommended_bid_spread:.2f} bps")
    print(f"Recommended Offer Spread: {signal.recommended_offer_spread:.2f} bps")
    print()
    print("Reasons:")
    for i, reason in enumerate(signal.reasons, 1):
        print(f"  {i}. {reason}")
    print()
    print("Key Metrics:")
    for metric, value in signal.metrics.items():
        print(f"  {metric}: {value:.4f}")
    print("=" * 80)
    
    return signal


if __name__ == "__main__":
    signal = example_usage()
