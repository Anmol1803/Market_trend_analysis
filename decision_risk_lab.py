# ===========================================
# FILE: decision_risk_lab.py (UPDATED VERSION WITH COMPATIBILITY)
# ===========================================

import streamlit as st
import plotly.graph_objects as go
import numpy as np
from typing import Dict, Any

class DecisionRiskLab:
    """Fourth Tab: Consumes data from Scoring and Forecasting tabs"""
    
    def __init__(self, scoring_data: Dict[str, Any], forecast_data: Dict[str, Any]):
        """
        Initialize with data from previous tabs
        
        Args:
            scoring_data: All data from scoring.py tab
            forecast_data: All data from forecast.py tab
        """
        self.scoring = scoring_data if scoring_data else {}
        self.forecast = self._ensure_forecast_compatibility(forecast_data) if forecast_data else {}
    
    def _ensure_forecast_compatibility(self, forecast_data: Dict) -> Dict:
        """
        Ensure forecast data has expected keys for the lab
        Handles compatibility between different forecast.py versions
        """
        data = forecast_data.copy()
        
        # Key name mappings for compatibility
        key_mappings = {
            'forecast_days': 'forecast_horizon',
            'max_drawdown_median': 'max_drawdown',
            'drawdown_95pct': 'max_drawdown_95pct',
            'paths': 'paths'
        }
        
        # Apply mappings
        for new_key, old_key in key_mappings.items():
            if new_key in data and old_key not in data:
                data[old_key] = data[new_key]
        
        # Create forecast_bands if not present (for older compatibility)
        if 'forecast_bands' not in data and 'quantiles' in data:
            quantiles = data['quantiles']
            if isinstance(quantiles, dict) and len(quantiles) > 0:
                # Create simple forecast bands from quantiles
                if '5%' in quantiles and '95%' in quantiles:
                    upper = quantiles['95%']
                    lower = quantiles['5%']
                    if len(upper) > 0 and len(lower) > 0:
                        current_price = data.get('start_price', 100)
                        band_width_pct = ((upper[-1] - lower[-1]) / current_price) * 100
                        data['forecast_bands'] = {
                            'width_pct': f"{band_width_pct:.1f}%",
                            'upper': upper,
                            'lower': lower
                        }
        
        # Ensure all required keys exist (with defaults)
        required_keys = ['forecast_horizon', 'start_price', 'expected_price', 
                        'positive_return_prob', 'var_95', 'max_drawdown']
        
        for key in required_keys:
            if key not in data:
                if key == 'forecast_horizon':
                    data[key] = data.get('forecast_days', 90)
                elif key == 'max_drawdown':
                    data[key] = data.get('drawdown_95pct', 0.15)
                elif key == 'var_95':
                    data[key] = -0.08  # Default 8% VaR
                else:
                    data[key] = None
        
        return data
    
    def render(self):
        """Main render function for the tab"""
        st.header("🧠 Decision & Risk Lab")
        st.caption("Step 4: Advanced analysis using data from Scoring & Forecasting")
        
        # Show data source indicators
        self._show_data_sources()
        
        # Check if we have enough data
        if not self.scoring or not self.forecast:
            st.warning("⚠️ Need both scoring and forecast data for complete analysis")
            if not self.scoring:
                st.info("Go to 🧠 Scoring tab first")
            if not self.forecast:
                st.info("Go to 🔮 Forecasting tab first")
            return
        
        # Horizontal separator
        st.markdown("---")
        
        # SECTION 1: DECISION LENS
        self._render_decision_lens()
        
        # SECTION 2: STRESS TEST (using forecast data)
        self._render_stress_test()
        
        # SECTION 3: MODEL DIAGNOSTICS
        self._render_model_diagnostics()
    
    def _show_data_sources(self):
        """Show where data is coming from"""
        has_scoring = bool(self.scoring)
        has_forecast = bool(self.forecast)
        
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            if has_scoring and has_forecast:
                st.success("✅ Connected to Scoring & Forecasting tabs")
            elif has_scoring:
                st.warning("⚠️ Connected to Scoring only")
            elif has_forecast:
                st.warning("⚠️ Connected to Forecasting only")
            else:
                st.error("❌ No data from previous tabs")
        
        with col2:
            if has_scoring:
                # Handle different key names for score
                base_score = self.scoring.get('total_score', self.scoring.get('base_score', 'N/A'))
                adj_score = self.scoring.get('regime_adjusted_score', 'N/A')
                st.metric("Score", 
                         f"{base_score}",
                         delta=f"Regime: {adj_score}")
        
        with col3:
            if has_forecast:
                # Use the compatible key name
                horizon = self.forecast.get('forecast_horizon', 'N/A')
                if horizon is None:
                    horizon = self.forecast.get('forecast_days', 'N/A')
                st.metric("Horizon", f"{horizon} days")
    
    def _render_decision_lens(self):
        """Section 1: Decision Interpretation"""
        st.subheader("🧠 Decision Lens")
        st.markdown("*Using scores from Step 2 and forecasts from Step 3*")
        
        with st.container():
            # Row 1: Key Metrics from Scoring
            col1, col2, col3 = st.columns(3)
            
            with col1:
                self._render_market_snapshot()
            
            with col2:
                self._render_readiness_matrix()
            
            with col3:
                self._render_risk_metrics()
            
            # Row 2: Interpretation
            st.markdown("#### 📝 Interpretation")
            interpretation = self._generate_interpretation()
            
            # Color code based on confidence
            confidence = self.scoring.get('ml_confidence', 'medium')
            if confidence == 'low':
                st.error(interpretation)
            elif confidence == 'medium':
                st.warning(interpretation)
            else:
                st.info(interpretation)
    
    def _render_market_snapshot(self):
        """Market snapshot using scoring data"""
        st.markdown("#### 📊 Market Snapshot")
        
        # Extract data from scoring with defaults
        regime = self.scoring.get('market_regime', 'Unknown')
        stability = self.scoring.get('forecast_stability', 'Unknown') 
        confidence = self.scoring.get('ml_confidence', 'medium')
        
        # Get colors for indicators
        regime_color = self._get_regime_color(regime)
        stability_color = self._get_stability_color(stability)
        confidence_color = self._get_confidence_color(confidence)
        
        st.markdown(f"""
        **Regime**: <span style='color:{regime_color};font-weight:bold'>{regime}</span>  
        **Stability**: <span style='color:{stability_color};font-weight:bold'>{stability}</span>  
        **ML Confidence**: <span style='color:{confidence_color};font-weight:bold'>{confidence}</span>
        """, unsafe_allow_html=True)
    
    def _render_readiness_matrix(self):
        """Stock readiness matrix using scoring data"""
        st.markdown("#### 📈 Stock Readiness Matrix")
        
        # Extract scores with defaults
        base_score = self.scoring.get('total_score', self.scoring.get('base_score', 50))
        regime_score = self.scoring.get('regime_adjusted_score', 50)
        risk_score = self.scoring.get('risk_score', 50)
        
        # Determine labels
        quality_label = self._score_to_label(base_score)
        timing_label = self._score_to_label(regime_score)
        risk_label = self._score_to_label(risk_score, invert=True)  # Higher risk score = worse
        
        # Get emojis
        quality_emoji = self._label_to_emoji(quality_label)
        timing_emoji = self._label_to_emoji(timing_label)
        risk_emoji = self._label_to_emoji(risk_label, invert=True)
        
        st.markdown(f"""
        **Quality**: {quality_emoji} {quality_label} ({base_score:.0f}/100)  
        **Timing**: {timing_emoji} {timing_label} ({regime_score:.0f}/100)  
        **Risk**: {risk_emoji} {risk_label} ({risk_score:.0f}/100)
        """)
    
    def _render_risk_metrics(self):
        """Risk metrics from scoring"""
        st.markdown("#### ⚠️ Risk Metrics")
        
        # Extract risk data with safe parsing
        var_95 = self.scoring.get('var_95', '5%')
        max_dd = self.scoring.get('max_drawdown', '10%')
        sharpe = self.scoring.get('sharpe_ratio', 'N/A')
        
        # Parse percentages for display
        try:
            var_display = self._parse_percentage(var_95)
            dd_display = self._parse_percentage(max_dd)
        except:
            var_display = 0.05
            dd_display = 0.10
        
        st.markdown(f"""
        **VaR (95%)**: `{var_display:.1%}`  
        **Max Drawdown**: `{dd_display:.1%}`  
        **Sharpe Ratio**: `{sharpe}`
        """)
    
    def _generate_interpretation(self):
        """Generate interpretation based on scoring data"""
        # Extract all needed data with defaults
        base_score = self.scoring.get('total_score', self.scoring.get('base_score', 50))
        regime_score = self.scoring.get('regime_adjusted_score', 50)
        risk_score = self.scoring.get('risk_score', 50)
        regime = self.scoring.get('market_regime', 'Neutral')
        stability = self.scoring.get('forecast_stability', 'Medium')
        confidence = self.scoring.get('ml_confidence', 'medium')
        
        # Build interpretation
        parts = []
        
        # Quality assessment
        if base_score >= 70:
            parts.append("Strong fundamentals")
        elif base_score >= 40:
            parts.append("Moderate fundamentals")
        else:
            parts.append("Weak fundamentals")
        
        # Timing assessment
        timing_gap = regime_score - base_score
        if timing_gap > 10:
            parts.append("with improved timing")
        elif timing_gap < -10:
            parts.append("with deteriorated timing")
        else:
            parts.append("with stable timing")
        
        # Regime context
        if 'risk' in regime.lower() or 'bear' in regime.lower():
            parts.append("in a defensive market regime.")
        elif 'bull' in regime.lower() or 'growth' in regime.lower():
            parts.append("in a favorable market regime.")
        else:
            parts.append("in a neutral market regime.")
        
        # Risk warning
        if risk_score > 70:
            parts.append("⚠️ Elevated risk requires caution.")
        elif risk_score > 50:
            parts.append("Moderate risk profile.")
        
        # Confidence note
        if confidence == 'low':
            parts.append("Low model confidence suggests verification needed.")
        elif confidence == 'medium':
            parts.append("Moderate confidence warrants careful position sizing.")
        
        # Forecast stability note
        if 'fragile' in stability.lower():
            parts.append("Forecast stability is fragile - high uncertainty.")
        
        return " ".join(parts)
    
    def _render_stress_test(self):
        """Section 2: Stress test using forecast data"""
        st.markdown("---")
        st.subheader("🧪 Stress Test / What-If Analysis")
        st.markdown("*Using Monte Carlo simulations from Forecasting tab*")
        
        with st.container():
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.markdown("#### 🔧 Stress Parameters")
                
                # Use forecast horizon from forecasting tab
                base_horizon = self.forecast.get('forecast_horizon', 90)
                if base_horizon is None:
                    base_horizon = 90
                
                # Get current volatility from scoring or forecast
                current_vol = self.scoring.get('current_volatility', 0.2)
                
                # Volatility adjustment
                vol_mult = st.slider(
                    "Volatility Multiplier",
                    min_value=1.0,
                    max_value=3.0,
                    value=1.5,
                    step=0.25,
                    help=f"Base volatility: {current_vol:.1%}"
                )
                st.caption(f"Current: {vol_mult}× → Effective volatility: {current_vol*vol_mult:.1%}")
                
                # Regime override
                current_regime = self.scoring.get('market_regime', 'Neutral')
                regime_options = ['Current', 'Risk-Off', 'Volatile', 'Crash']
                regime_override = st.selectbox(
                    "Stress Regime",
                    options=regime_options,
                    index=0
                )
                
                # Horizon adjustment
                horizon_mult = st.slider(
                    "Horizon Extension",
                    min_value=0.5,
                    max_value=3.0,
                    value=1.0,
                    step=0.25,
                    help=f"Base horizon: {base_horizon} days"
                )
                
                # Show the calculated horizon
                new_horizon = int(base_horizon * horizon_mult)
                st.caption(f"Adjusted horizon: {new_horizon} days")
                
                if st.button("🚀 Apply Stress Test", type="secondary"):
                    st.session_state['stress_params'] = {
                        'vol_mult': vol_mult,
                        'regime': regime_override,
                        'horizon_mult': horizon_mult,
                        'new_horizon': new_horizon
                    }
            
            with col2:
                st.markdown("#### 📊 Stress Impact")
                
                if 'stress_params' in st.session_state:
                    self._show_stress_results(st.session_state['stress_params'])
                else:
                    st.info("Configure stress parameters and click 'Apply Stress Test'")
        
        st.caption("⚠️ *Stress scenarios exaggerate risk. Not predictions.*")
    
    def _show_stress_results(self, params):
        """Show stress test results"""
        # Get base metrics with safe parsing
        base_var_str = self.scoring.get('var_95', '5%')
        base_dd_str = self.scoring.get('max_drawdown', '10%')
        
        base_var = self._parse_percentage(base_var_str)
        base_dd = self._parse_percentage(base_dd_str)
        base_score = self.scoring.get('total_score', self.scoring.get('base_score', 50))
        
        # Apply stress multipliers
        if params['regime'] == 'Risk-Off':
            regime_mult = 1.8
        elif params['regime'] == 'Volatile':
            regime_mult = 2.2
        elif params['regime'] == 'Crash':
            regime_mult = 3.0
        else:
            regime_mult = 1.0
        
        # Calculate stressed metrics
        stressed_var = base_var * params['vol_mult'] * regime_mult * 0.5
        stressed_dd = base_dd * params['vol_mult'] * regime_mult
        stressed_score = max(0, base_score - (params['vol_mult'] - 1) * 15)
        
        # Display
        st.error(f"""
        **STRESS SCENARIO**: {params['vol_mult']}× Vol | {params['regime']} Regime
        
        **VaR (95%)**: `{base_var:.1%}` → `{stressed_var:.1%}`  
        **Max DD**: `{base_dd:.1%}` → `{stressed_dd:.1%}`  
        **Score Impact**: `{base_score:.0f}` → `{stressed_score:.0f}`
        
        **Capital-at-Risk**: `{min(100, stressed_dd*2):.1%}`
        """)
        
        # Visual comparison
        self._render_stress_comparison(base_var, stressed_var, base_dd, stressed_dd)
    
    def _render_stress_comparison(self, base_var, stress_var, base_dd, stress_dd):
        """Simple bar comparison"""
        fig = go.Figure(data=[
            go.Bar(name='Baseline', x=['VaR', 'Max DD'], 
                   y=[base_var*100, base_dd*100], marker_color='blue'),
            go.Bar(name='Stress', x=['VaR', 'Max DD'], 
                   y=[stress_var*100, stress_dd*100], marker_color='red')
        ])
        
        fig.update_layout(
            title="Risk Metric Comparison",
            yaxis_title="Percentage (%)",
            barmode='group',
            height=250,
            showlegend=True
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def _render_model_diagnostics(self):
        """Section 3: Model diagnostics using both scoring and forecast data"""
        st.markdown("---")
        st.subheader("📊 Model Diagnostics & Trust")
        st.markdown("*Using confidence metrics from previous steps*")
        
        with st.container():
            # Three diagnostic panels
            diag1, diag2, diag3 = st.columns(3)
            
            with diag1:
                self._render_forecast_diagnostics()
            
            with diag2:
                self._render_regime_diagnostics()
            
            with diag3:
                self._render_confidence_diagnostics()
            
            # Warning checklist
            with st.expander("⚠️ When NOT to trust this analysis", expanded=False):
                st.markdown(self._generate_warnings())
    
    def _render_forecast_diagnostics(self):
        """Forecast quality diagnostics"""
        st.markdown("#### 🔍 Forecast Quality")
        
        # Get forecast metrics
        stability = self.scoring.get('forecast_stability', 'Medium')
        confidence = self.scoring.get('ml_confidence', 'medium')
        
        # Get forecast bands from forecasting tab if available
        band_width = "N/A"
        if 'forecast_bands' in self.forecast:
            bands = self.forecast['forecast_bands']
            band_width = bands.get('width_pct', 'N/A')
        elif 'quantiles' in self.forecast:
            # Calculate band width from quantiles
            quantiles = self.forecast['quantiles']
            if isinstance(quantiles, dict) and '5%' in quantiles and '95%' in quantiles:
                current_price = self.forecast.get('start_price', 100)
                upper = quantiles['95%'][-1] if len(quantiles['95%']) > 0 else current_price
                lower = quantiles['5%'][-1] if len(quantiles['5%']) > 0 else current_price
                band_width_pct = ((upper - lower) / current_price) * 100
                band_width = f"{band_width_pct:.1f}%"
        
        if band_width != "N/A":
            st.metric("Band Width", band_width)
        
        # Stability indicator
        if stability == 'Fragile':
            st.error(f"Stability: {stability}")
        elif stability == 'Stable':
            st.success(f"Stability: {stability}")
        else:
            st.warning(f"Stability: {stability}")
        
        st.metric("ML Confidence", confidence.capitalize())
    
    def _render_regime_diagnostics(self):
        """Regime diagnostics"""
        st.markdown("#### 🎛️ Regime Analysis")
        
        regime = self.scoring.get('market_regime', 'Unknown')
        regime_prob = self.scoring.get('regime_probability', 0.5)
        
        st.metric("Current Regime", regime, delta=f"{regime_prob:.0%} confidence")
        
        # Show regime transition if available
        if 'prev_regime' in self.scoring:
            prev = self.scoring['prev_regime']
            if prev != regime:
                st.caption(f"Changed from {prev}")
        
        # Regime strength
        if regime_prob > 0.7:
            st.success("Strong regime signal")
        elif regime_prob > 0.5:
            st.warning("Moderate regime signal")
        else:
            st.error("Weak regime signal")
    
    def _render_confidence_diagnostics(self):
        """Confidence breakdown"""
        st.markdown("#### 🤖 Confidence Drivers")
        
        # Get confidence factors if available
        factors = self.scoring.get('confidence_factors', {})
        
        if factors:
            for factor, value in factors.items():
                st.progress(value/100, text=f"{factor}: {value}%")
        else:
            # Fallback to simple indicators
            indicators = {
                'Data Quality': self.scoring.get('data_quality', 75),
                'Model Fit': self.scoring.get('model_fit', 65),
                'Signal Strength': self.scoring.get('signal_strength', 60)
            }
            
            for factor, value in indicators.items():
                st.progress(value/100, text=f"{factor}: {value}%")
    
    def _generate_warnings(self):
        """Generate warnings based on data"""
        warnings = []
        
        # Check various warning conditions
        if self.scoring.get('ml_confidence') == 'low':
            warnings.append("- **Low ML Confidence**: Model uncertainty is high")
        
        if self.scoring.get('forecast_stability') == 'Fragile':
            warnings.append("- **Fragile Forecast**: High sensitivity to inputs")
        
        regime = self.scoring.get('market_regime', '').lower()
        if 'volatile' in regime or 'risk' in regime:
            warnings.append("- **Unfavorable Regime**: Market conditions are challenging")
        
        try:
            var_95 = self._parse_percentage(self.scoring.get('var_95', '5%'))
            if var_95 > 0.08:
                warnings.append("- **High VaR**: Significant downside risk")
        except:
            pass
        
        if len(warnings) == 0:
            warnings.append("- No major warnings detected")
        
        return "\n".join(warnings)
    
    # ===========================================
    # HELPER METHODS
    # ===========================================
    
    def _get_regime_color(self, regime):
        """Get color for regime"""
        regime_lower = regime.lower()
        if 'risk' in regime_lower or 'bear' in regime_lower:
            return '#dc3545'  # Red
        elif 'neutral' in regime_lower or 'sideways' in regime_lower:
            return '#f39c12'  # Orange
        elif 'bull' in regime_lower or 'growth' in regime_lower:
            return '#27ae60'  # Green
        else:
            return '#6c757d'  # Default gray
    
    def _get_stability_color(self, stability):
        """Get color for stability"""
        stability_lower = stability.lower()
        if 'fragile' in stability_lower:
            return '#e74c3c'   # Red
        elif 'stable' in stability_lower:
            return '#27ae60'   # Green
        else:
            return '#f39c12'   # Orange (default for medium)
    
    def _get_confidence_color(self, confidence):
        """Get color for confidence"""
        if confidence == 'low':
            return '#e74c3c'      # Red
        elif confidence == 'medium':
            return '#f39c12'      # Orange
        elif confidence == 'high':
            return '#27ae60'      # Green
        else:
            return '#6c757d'      # Gray
    
    def _score_to_label(self, score, invert=False):
        """Convert score to label"""
        if invert:
            score = 100 - score
        
        if score >= 70:
            return 'High'
        elif score >= 40:
            return 'Medium'
        else:
            return 'Low'
    
    def _label_to_emoji(self, label, invert=False):
        """Convert label to emoji"""
        if invert:
            if label == 'High':
                return '🔴'  # Red for high risk
            elif label == 'Medium':
                return '🟡'  # Yellow for medium risk
            else:
                return '🟢'  # Green for low risk
        else:
            if label == 'High':
                return '🟢'
            elif label == 'Medium':
                return '🟡'
            else:
                return '🔴'
    
    def _parse_percentage(self, value):
        """Parse percentage string to float"""
        if isinstance(value, str):
            if '%' in value:
                return float(value.replace('%', '')) / 100
            else:
                try:
                    return float(value)
                except:
                    return 0.05  # Default 5%
        elif isinstance(value, (int, float)):
            # If value > 1, assume it's percentage (e.g., 5 for 5%)
            return float(value) / 100 if value > 1 else float(value)
        else:
            return 0.05  # Default 5%