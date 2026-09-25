/**
 * Anchored Volume Weighted Average Price (AVWAP) Indicator
 * 
 * Custom AVWAP that anchors to the first tradable candle of an option contract's life.
 * Does not reset at day/week boundaries - unique to each strike/expiry combination.
 * 
 * Rules:
 * - Anchor: First candle when option contract becomes tradable
 * - Timeframe: 15-minute candles (IST market sessions)
 * - AVWAP is cumulative from anchor point, persists across days
 */

export default function ({ registerIndicator, sourceValues }) {
  registerIndicator({
    id: 'avwap',
    name: 'Anchored VWAP',
    category: 'Custom',
    placement: 'onchart',
    inputs: [
      { key: 'anchorIndex', type: 'number', label: 'Anchor Candle Index', default: 0, min: 0 },
      { key: 'anchorTime', type: 'string', label: 'Anchor Time (ISO)', default: '' },
      { key: 'source', type: 'string', label: 'Price Source', default: 'hlc3', options: ['hlc3', 'hl2', 'ohlc4', 'close'] }
    ],
    plots: [
      { key: 'avwap', type: 'line', title: 'AVWAP', style: { color: '#4f8cff', lineWidth: 2 } },
      { key: 'avwap_nan', type: 'line', title: 'AVWAP (Pre-Anchor)', style: { color: 'rgba(255,0,0,0.1)', lineWidth: 1, dash: 'dotted' } }
    ],
    calc(bars, settings) {
      // Import the AVWAP calculation logic
      // We'll use a simple implementation here
      const length = Number(settings.anchorIndex) || 0;
      const anchorTime = settings.anchorTime || '';
      const src = sourceValues(bars, settings.source);
      
      // Initialize AVWAP array with NaN
      const avwap = new Array(bars.length).fill(null);
      
      // Determine anchor start index
      let startIdx = 0;
      if (anchorTime) {
        // Find first candle at or after anchor time
        // This requires timestamp data - for now use index 0
        startIdx = 0;
      } else if (length > 0) {
        startIdx = length;
      }
      
      // Calculate cumulative VWAP from anchor point
      let cumulativePV = 0; // price * volume
      let cumulativeV = 0;  // volume
      
      for (let i = startIdx; i < bars.length; i++) {
        // Typical price = (high + low + close) / 3
        const typicalPrice = (bars.high[i] + bars.low[i] + src[i]) / 3;
        const pv = typicalPrice * bars.volume[i];
        
        cumulativePV += pv;
        cumulativeV += bars.volume[i];
        
        if (cumulativeV > 0) {
          avwap[i] = cumulativePV / cumulativeV;
        } else {
          avwap[i] = typicalPrice;
        }
      }
      
      // Set NaN before anchor index (AVWAP not defined yet)
      for (let i = 0; i < startIdx; i++) {
        avwap[i] = null;
      }
      
      return { avwap };
    }
  });
}