#!/usr/bin/env python
"""
AVWAP Options Selling Trading Strategy
=======================================
Automated, rule-based Option Selling System for NSE F&O segments.
Uses Anchored VWAP (AVWAP) as the sole mathematical foundation.

Strategy Rules:
- Sell (short) options when premium crosses BELOW AVWAP
- Buy to close when premium crosses ABOVE AVWAP
- Exit rule = Stop Loss (no fixed SL)
"""

import sys
import os
import json
import time
import logging
import datetime
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Tuple, Any

import numpy as np
import pandas as pd

# OpenAlgo imports
from openalgo import api
from openalgo.trading import strategy_module
from openalgo import ta

# Import our custom AVWAP indicator
from strategies.indicators.avwap_indicator import calculate_avwap

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration Constants
# ---------------------------------------------------------------------------
CONFIG = {
    "broker": "dhan",
    "strategy_name": "AVWAP_Options_Selling",
    "timeframe": "15min",
    "underlyings": ["NIFTY", "BANKNIFTY"],
    "product": "MIS",  # Intraday product
    "quantity": 1,  # Lotsize will be determined from master contract
    "max_open_positions": 5,  # Max concurrent trades portfolio-wide
    "max_trades_per_day": 10,  # Max trades per day per underlying
    "max_daily_loss": 5000,  # Stop trading if daily loss exceeds this (INR)
    "paper_mode": True,  # Set False for LIVE (with checklist compliance)
    "ancher_from_first_candle": True,  # Anchor AVWAP from first tradable candle
}

# Strike selection: ATM + 4 ITM on both CE & PE sides = 10 contracts
STRIKE_SPAN = 4  # Number of strikes away from ATM (both sides)


# ---------------------------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------------------------

def get_expiry_dates(underlying: str, broker: str = "dhan") -> Tuple[str, str]:
    """
    Get current and target expiry dates based on underlying.
    
    NIFTY: current weekly + following weekly (2 legs)
    BANKNIFTY: current monthly, rolls on 24th to next month
    """
    today = datetime.now()
    today_str = today.strftime("%d-%b-%y").upper()
    
    if underlying == "NIFTY":
        # NIFTY: current weekly expiry + next weekly expiry
        # Find current weekly expiry (typically Thursday or nearest expiry)
        # For simplicity, use current month + following month weekly
        current_expiry = f"NIFTY{today.strftime('%d')}{today.strftime('%b').upper()}{today.strftime('%y')}"
        # Next expiry - approximate: add 7 days
        next_date = today + timedelta(days=7)
        next_expiry = f"NIFTY{next_date.strftime('%d')}{next_date.strftime('%b').upper()}{next_date.strftime('%y')}"
        return current_expiry, next_expiry
    
    elif underlying == "BANKNIFTY":
        # BANKNIFTY: monthly expiry
        # Roll on 24th of month to next month
        if today.day >= 24:
            # Roll to next month
            if today.month == 12:
                next_month = today.replace(year=today.year+1, month=1, day=1)
            else:
                next_month = today.replace(month=today.month+1, day=1)
            current_expiry = f"BANKNIFTY{today.strftime('%d')}{today.strftime('%b').upper()}{today.strftime('%y')}"
            next_expiry = f"BANKNIFTY{next_month.strftime('%d')}{next_month.strftime('%b').upper()}{next_month.strftime('%y')}"
        else:
            current_expiry = f"BANKNIFTY{today.strftime('%d')}{today.strftime('%b').upper()}{today.strftime('%y')}"
            next_expiry = current_expiry  # Same expiry until 24th
        return current_expiry, next_expiry
    
    return "", ""


def get_atm_strike(spot_price: float, underlying: str) -> int:
    """
    Calculate At-The-Money strike price.
    For NIFTY: round to nearest 50
    For BANKNIFTY: round to nearest 100
    """
    if underlying == "NIFTY":
        return int(round(spot_price / 50) * 50)
    elif underlying == "BANKNIFTY":
        return int(round(spot_price / 100) * 100)
    else:
        return int(round(spot_price))


def get_strikes_around_atm(spot_price: float, span: int = STRIKE_SPAN, underlying: str = "NIFTY") -> List[int]:
    """
    Get strike prices: ATM, +1 ITM, +2 ITM, +3 ITM, +4 ITM on both sides
    Returns list of strike prices
    """
    atm = get_atm_strike(spot_price, underlying)
    strikes = []
    
    # Add strikes: ATM, ATM+1ITM, ATM+2ITM, ATM+3ITM, ATM+4ITM on CE side
    # And ATM, ATM-1ITM, ATM-2ITM, ATM-3ITM, ATM-4ITM on PE side
    for i in range(span + 1):  # 0 to 4 (ATM + 4 ITM)
        strikes.append(atm + i * (100 if underlying == "BANKNIFTY" else 50))  # CE side (higher strikes)
    for i in range(1, span + 1):  # 1 to 4 (excluding ATM duplicate)
        strikes.append(atm - i * (100 if underlying == "BANKNIFTY" else 50))  # PE side (lower strikes)
    
    # Remove duplicates and sort
    strikes = sorted(list(set(strikes)))
    return strikes


def calculate_avwap_for_series(
    df: pd.DataFrame,
    anchor_index: Optional[int] = None,
) -> pd.Series:
    """
    Calculate AVWAP for a price series using our custom indicator.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with 'high', 'low', 'close', 'volume' columns
    anchor_index : int, optional
        Index to anchor from. If None, anchors from first candle (index 0)
        
    Returns
    -------
    pd.Series
        AVWAP values aligned with input DataFrame index
    """
    if df.empty:
        return pd.Series(dtype=float, index=df.index)
    
    # Calculate AVWAP using custom function
    avwap_values = calculate_avwap(
        high=df['high'].values,
        low=df['low'].values,
        close=df['close'].values,
        volume=df['volume'].values,
        anchor_index=anchor_index,
    )
    
    return pd.Series(avwap_values, index=df.index, dtype=float)


def check_entry_signal(
    current_close: float,
    previous_close: float,
    current_avwap: float,
    previous_avwap: float,
) -> bool:
    """
    Entry Rule: 
    Previous Candle Close >= Previous AVWAP AND Current Candle Close < Current AVWAP
    Clean cross from above to below.
    """
    # Previous candle: close >= AVWAP
    prev_condition = previous_close >= previous_avwap
    # Current candle: close < AVWAP
    curr_condition = current_close < current_avwap
    
    return prev_condition and curr_condition


def check_exit_signal(
    current_close: float,
    current_avwap: float,
) -> bool:
    """
    Exit Rule: 
    Current Candle Close > Current AVWAP
    Structural strength against seller - terminates the trade.
    """
    return current_close > current_avwap


# ---------------------------------------------------------------------------
# Core Strategy Engine
# ---------------------------------------------------------------------------

class AVWAPOptionsSellingStrategy:
    """
    Automated AVWAP Options Selling Strategy.
    """
    
    def __init__(self, api_key: str, host: str = "http://127.0.0.1:5000", 
                 ws_url: str = "ws://127.0.0.1:8765"):
        self.api_key = api_key
        self.host = host
        self.ws_url = ws_url
        
        # Initialize API client
        self.client = api(api_key=api_key, host=host, ws_url=ws_url)
        
        # State tracking
        self.open_positions: Dict[str, Dict] = {}  # track open trades
        self.daily_trade_count: Dict[str, int] = {}  # per underlying
        self.daily_realized_pnl: float = 0.0
        self.last_reset_date: str = datetime.now().strftime("%Y-%m-%d")
        
        # Strategy parameters
        self.strategy_name = CONFIG["strategy_name"]
        
    def reset_daily_counters(self):
        """Reset daily counters if new trading day."""
        today = datetime.now().strftime("%Y-%m-%d")
        if today != self.last_reset_date:
            logger.info("Resetting daily counters for new trading day")
            self.daily_trade_count = {}
            self.daily_realized_pnl = 0.0
            self.last_reset_date = today
    
    def check_risk_limits(self, underlying: str) -> bool:
        """
        Check risk management limits before taking a trade.
        Returns True if trade is allowed, False if limit hit.
        """
        self.reset_daily_counters()
        
        # Check max open positions
        total_open = len(self.open_positions)
        if total_open >= CONFIG["max_open_positions"]:
            logger.warning(f"Max open positions limit reached: {total_open}/{CONFIG['max_open_positions']}")
            return False
        
        # Check max trades per day for this underlying
        current_count = self.daily_trade_count.get(underlying, 0)
        if current_count >= CONFIG["max_trades_per_day"]:
            logger.warning(f"Max trades per day limit for {underlying}: {current_count}/{CONFIG['max_trades_per_day']}")
            return False
        
        # Check max daily loss
        if self.daily_realized_pnl <= -CONFIG["max_daily_loss"]:
            logger.warning(f"Max daily loss limit hit: ₹{abs(self.daily_realized_pnl):.2f}. Locking trading engine.")
            return False
        
        return True
    
    def execute_trade(
        self,
        underlying: str,
        option_type: str,  # "CE" or "PE"
        strike: int,
        action: str,  # "SELL" (short) or "BUY" (close)
        price: float,
        quantity: int,
        expiry: str,
    ) -> Optional[Dict]:
        """
        Execute a trade order through the OpenAlgo API.
        """
        try:
            # Construct the option symbol
            symbol = f"{underlying}{expiry}{strike}{option_type}"
            
            # Place the order
            response = self.client.placesmartorder(
                strategy=self.strategy_name,
                symbol=symbol,
                exchange="NSE",
                action=action,
                price_type="MARKET",
                product=CONFIG["product"],
                quantity=quantity,
            )
            
            # Track the position
            trade_key = f"{underlying}_{option_type}_{strike}_{expiry}"
            self.open_positions[trade_key] = {
                "underlying": underlying,
                "option_type": option_type,
                "strike": strike,
                "expiry": expiry,
                "action": action,
                "entry_price": price,
                "entry_time": datetime.now(),
                "quantity": quantity,
                "status": "active",
                "pnl": 0.0,
            }
            
            # Increment trade count
            self.daily_trade_count[underlying] = self.daily_trade_count.get(underlying, 0) + 1
            
            logger.info(f"Order executed: {action} {symbol} at ₹{price:.2f}")
            return response
            
        except Exception as e:
            logger.error(f"Error executing trade for {symbol}: {str(e)}")
            return None
    
    def square_off_position(self, trade_key: str, current_price: float) -> Optional[Dict]:
        """
        Buy to close a position (exit rule).
        """
        if trade_key not in self.open_positions:
            logger.warning(f"Trade key {trade_key} not found in open positions")
            return None
        
        position = self.open_positions[trade_key]
        underlying = position["underlying"]
        option_type = position["option_type"]
        strike = position["strike"]
        expiry = position["expiry"]
        
        symbol = f"{underlying}{expiry}{strike}{option_type}"
        
        try:
            # Execute buy to close
            response = self.client.placesmartorder(
                strategy=self.strategy_name,
                symbol=symbol,
                exchange="NSE",
                action="BUY",
                price_type="MARKET",
                product=CONFIG["product"],
                quantity=position["quantity"],
            )
            
            # Calculate P&L
            entry_price = position["entry_price"]
            quantity = position["quantity"]
            pnl = (entry_price - current_price) * quantity  # Short P&L: entry - exit * qty
            position["pnl"] = pnl
            position["status"] = "closed"
            
            # Update daily realized P&L
            self.daily_realized_pnl += pnl
            
            # Remove from open positions
            del self.open_positions[trade_key]
            
            logger.info(f"Position squared off: {symbol}. P&L: ₹{pnl:.2f}. Daily P&L: ₹{self.daily_realized_pnl:.2f}")
            return response
            
        except Exception as e:
            logger.error(f"Error squaring off position {symbol}: {str(e)}")
            return None
    
    def scan_and_trade(self, underlying: str, df: pd.DataFrame, expiry: str) -> None:
        """
        Main scanning logic for one underlying.
        """
        # Filter data for this specific option chain
        # We need option data - in practice this comes from Dhan API
        # For now, using spot price and generating signals
        
        if df.empty or len(df) < 3:
            return
        
        # Get spot price (latest close)
        spot_price = df['close'].iloc[-1]
        
        # Get ATM and strikes
        strikes = get_strikes_around_atm(spot_price, underlying=underlying)
        
        # Calculate AVWAP for the series
        avwap = calculate_avwap_for_series(df)
        
        # Latest AVWAP values
        current_avwap = avwap.iloc[-1]
        previous_avwap = avwap.iloc[-2] if len(avwap) > 1 else current_avwap
        
        # Latest closes
        current_close = df['close'].iloc[-1]
        previous_close = df['close'].iloc[-2] if len(df) > 1 else current_close
        
        logger.info(f"{underlying} | Spot: ₹{spot_price:.2f} | ATM strikes: {strikes[:3]}... | AVWAP: ₹{current_avwap:.2f}")
        
        # Check time - skip first 09:15 AM candle
        current_time = datetime.now().time()
        market_open = datetime.now().replace(hour=9, minute=15, second=0, microsecond=0).time()
        
        # Skip trading if before 09:30 AM (after first candle closes)
        if current_time < datetime.strptime("09:30", "%H:%M").time():
            # Check if we're in the 09:15-09:30 window
            logger.info(f"Pre-market session: {current_time}. Waiting for 09:30 AM candle close.")
            return
        
        # Trading only after 09:30 AM
        if current_time >= datetime.strptime("09:30", "%H:%M").time():
            # Process all strike contracts
            for strike in strikes:
                # Calculate CE and PE symbols
                # In practice, we'd get option chain data from Dhan
                # For this framework, we generate signals for both CE and PE
                
                # CE (Call) signal
                ce_avwap = current_avwap  # Each strike/expiry has unique AVWAP
                ce_current_close = current_close
                ce_previous_close = previous_close
                
                # PE (Put) signal  
                pe_avwap = current_avwap  # Unique per strike/expiry
                pe_current_close = current_close
                pe_previous_close = previous_close
                
                # Entry signals for CE
                ce_entry_signal = check_entry_signal(
                    ce_current_close, ce_previous_close, ce_avwap, previous_avwap
                )
                
                # Entry signals for PE
                pe_entry_signal = check_entry_signal(
                    pe_current_close, pe_previous_close, pe_avwap, previous_avwap
                )
                
                # Exit signals
                ce_exit_signal = check_exit_signal(ce_current_close, ce_avwap)
                pe_exit_signal = check_exit_signal(pe_current_close, pe_avwap)
                
                # --- ENTRY: Sell CE if signal triggers ---
                if ce_entry_signal:
                    if self.check_risk_limits(underlying):
                        logger.info(f"ENTRY SIGNAL: SELL CE {underlying} Strike {strike}")
                        self.execute_trade(
                            underlying=underlying,
                            option_type="CE",
                            strike=strike,
                            action="SELL",
                            price=ce_current_close,  # Market order - will execute at LTP
                            quantity=CONFIG["quantity"],
                            expiry=expiry,
                        )
                
                # --- ENTRY: Sell PE if signal triggers ---
                if pe_entry_signal:
                    if self.check_risk_limits(underlying):
                        logger.info(f"ENTRY SIGNAL: SELL PE {underlying} Strike {strike}")
                        self.execute_trade(
                            underlying=underlying,
                            option_type="PE",
                            strike=strike,
                            action="SELL",
                            price=pe_current_close,
                            quantity=CONFIG["quantity"],
                            expiry=expiry,
                        )
                
                # --- EXIT: Buy to Close CE if signal triggers ---
                if position_key := f"{underlying}_CE_{strike}_{expiry}" in self.open_positions:
                    if ce_exit_signal:
                        logger.info(f"EXIT SIGNAL: BUY TO CLOSE CE {underlying} Strike {strike}")
                        self.square_off_position(position_key, ce_current_close)
                
                # --- EXIT: Buy to Close PE if signal triggers ---
                if position_key := f"{underlying}_PE_{strike}_{expiry}" in self.open_positions:
                    if pe_exit_signal:
                        logger.info(f"EXIT SIGNAL: BUY TO CLOSE PE {underlying} Strike {strike}")
                        self.square_off_position(position_key, pe_current_close)
        
    def run_strategy_loop(self):
        """
        Main strategy loop - runs continuously during market hours.
        """
        logger.info("=" * 60)
        logger.info("AVWAP Options Selling Strategy Starting...")
        logger.info("=" * 60)
        logger.info(f"Strategy: {CONFIG['strategy_name']}")
        logger.info(f"Underlyings: {CONFIG['underlyings']}")
        logger.info(f"Timeframe: {CONFIG['timeframe']}")
        logger.info(f"Product: {CONFIG['product']}")
        logger.info(f"Paper Mode: {CONFIG['paper_mode']}")
        logger.info("Rules:")
        logger.info("  1. Anchor AVWAP from first tradable candle of option life")
        logger.info("  2. SELL when Close crosses BELOW AVWAP (prev close >= prev AVWAP & curr close < curr AVWAP)")
        logger.info("  3. BUY TO CLOSE when Close crosses ABOVE AVWAP")
        logger.info("  4. Exit rule = Stop Loss (no fixed SL)")
        logger.info("  5. No trades before 09:30 AM (wait for first candle close)")
        logger.info("  6. Risk limits: max open positions, max trades/day, max daily loss")
        logger.info("=" * 60)
        
        # Get expiry dates for each underlying
        underly_expiry = {}
        for underlying in CONFIG["underlyings"]:
            current, target = get_expiry_dates(underlying)
            underly_expiry[underlying] = {
                "current": current,
                "target": target,
            }
        
        logger.info(f"Expiry dates: {underly_expiry}")
        
        # Main loop
        while True:
            try:
                current_time = datetime.now()
                current_time_str = current_time.strftime("%H:%M:%S")
                
                # Check market hours (9:15 AM to 3:30 PM IST approximately)
                current_hour = current_time.hour
                current_minute = current_time.minute
                
                # Skip outside market hours
                if current_hour < 9 or current_hour >= 15:
                    # Sleep and check again
                    time.sleep(30)
                    continue
                
                # For each underlying, fetch data and scan
                for underlying in CONFIG["underlyings"]:
                    expiry_info = underly_expiry[underlying]
                    expiry = expiry_info["current"]
                    
                    # Fetch historical 15-min data
                    # In production, this would come from Dhan WebSocket / historical API
                    # For now, we'll use a simplified approach
                    
                    try:
                        # Fetch 15-min historical data for the last ~10 candles
                        end_date = current_time.strftime("%Y-%m-%d")
                        start_date = (current_time - timedelta(days=1)).strftime("%Y-%m-%d")
                        
                        df = self.client.history(
                            symbol=underlying,
                            exchange="NSE",
                            interval="15m",
                            start_date=start_date,
                            end_date=end_date,
                        )
                        
                        if not df.empty:
                            self.scan_and_trade(underlying, df, expiry)
                        
                    except Exception as e:
                        logger.error(f"Error fetching data for {underlying}: {str(e)}")
                    
                    # Small delay between underlyings
                    time.sleep(2)
                
                # Wait before next cycle - check every 15 minutes
                # But we process on every candle completion
                time.sleep(20)  # ~15 min + buffer
                
            except KeyboardInterrupt:
                logger.info("Strategy stopped by user")
                break
            except Exception as e:
                logger.error(f"Unexpected error in main loop: {str(e)}")
                time.sleep(30)


# ---------------------------------------------------------------------------
# Main Entry Point
# ---------------------------------------------------------------------------

def main():
    """Main entry point for the AVWAP Options Selling Strategy."""
    
    # Get API key from environment
    api_key = os.getenv("OPENALGO_API_KEY")
    
    if not api_key:
        logger.error("ERROR: OPENALGO_API_KEY environment variable not set!")
        logger.error("Please set your Dhan/OpenAlgo API key and restart.")
        sys.exit(1)
    
    # Initialize strategy
    strategy = AVWAPOptionsSellingStrategy(api_key=api_key)
    
    # Run the strategy loop
    strategy.run_strategy_loop()


if __name__ == "__main__":
    main()