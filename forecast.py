# ============================================================
# forecast.py - INSTITUTIONAL-GRADE FORECASTING ENGINE
# FIXED: Horizon-specific calibration + Financial realism
# ============================================================

import numpy as np
import pandas as pd
from dataclasses import dataclass
import warnings
from scipy import stats
from hmmlearn import hmm
from statsmodels.tsa.arima.model import ARIMA

warnings.filterwarnings('ignore')

# ============================================================
# MODEL 1: HIDDEN MARKOV MODEL (WITH ENTROPY FLOOR)
# ============================================================

class HiddenMarkovRegimeDetector:
    """
    Detects market regimes with PROBABILISTIC confidence
    FIX: No 100% certainty, entropy floor enforced
    """
    def __init__(self, n_regimes=3):
        self.n_regimes = n_regimes
        self.model = hmm.GaussianHMM(
            n_components=n_regimes,
            covariance_type="full",
            n_iter=200,  # Increased for better convergence
            random_state=42
        )
        self.regime_labels = {0: "Bear", 1: "Sideways", 2: "Bull"}
        self.transition_matrix = None
        self.is_fitted = False
        self.entropy_floor = 0.05  # FIX #1: Minimum probability per regime

    def _regime_signal_strength(self, prob_dist):
        """
        Signal kitna strong hai check karo
        0 = bilkul uncertain (33/33/33)
        1 = bilkul sure (100% ek regime)
        """
        # Convert dict to array
        probs = np.array([prob_dist["Bear"], prob_dist["Sideways"], prob_dist["Bull"]])
        
        # Avoid log(0)
        probs = np.maximum(probs, 0.001)
        probs = probs / probs.sum()  # Normalize
        
        # Calculate entropy
        entropy = -np.sum(probs * np.log(probs))
        
        # Maximum entropy (uniform distribution)
        max_entropy = np.log(3)  # log(3) ≈ 1.0986
        
        # Signal strength: 0 to 1
        strength = 1 - (entropy / max_entropy)
        return float(np.clip(strength, 0, 1))

    def prepare_features(self, df):
        """Prepare features for regime detection"""
        price = df['Close'].values.reshape(-1, 1)
        returns = np.diff(np.log(price.flatten()))
        
        features = []
        for i in range(1, len(price)):
            window = returns[max(0, i-30):i]  # Larger window for stability
            if len(window) > 0:
                vol = np.std(window) if len(window) > 1 else 0.01
                trend = np.mean(window[-10:]) if len(window) >= 10 else 0
                momentum = np.sum(window[-5:]) if len(window) >= 5 else 0
                features.append([returns[i-1], vol, trend, momentum])
            else:
                features.append([0, 0.01, 0, 0])
        
        return np.array(features)
    
    def fit(self, df):
        """Fit HMM to historical data"""
        try:
            features = self.prepare_features(df)
            if len(features) < 100:  # Increased minimum data requirement
                self.is_fitted = False
                return self
            
            self.model.fit(features)
            self.transition_matrix = self.model.transmat_

            
            np.fill_diagonal(
                self.transition_matrix,
                np.maximum(np.diag(self.transition_matrix), 0.85)
            )
            
            # Re-normalize rows
            self.transition_matrix = (
                self.transition_matrix /
                self.transition_matrix.sum(axis=1, keepdims=True)
            )
            
            # Push back to HMM model
            self.model.transmat_ = self.transition_matrix

            # === CRITICAL FIX: Map regimes correctly based on mean returns ===
            predicted_states = self.model.predict(features)
            state_stats = {}
            
            for state in range(self.n_regimes):
                state_returns = features[predicted_states == state][:, 0]  # First feature = returns
                if len(state_returns) > 0:
                    state_stats[state] = {
                        "mean": np.mean(state_returns),
                        "vol": np.std(state_returns) if len(state_returns) > 1 else 0.01
                    }
                else:
                    state_stats[state] = {"mean": 0, "vol": 0.01}
            
            # Sort regimes by mean return (lowest = Bear, highest = Bull)
            sorted_states = sorted(state_stats.items(), key=lambda x: x[1]["mean"])
            
            self.regime_labels = {
                sorted_states[0][0]: "Bear",      # Lowest mean return
                sorted_states[1][0]: "Sideways",  # Middle mean return  
                sorted_states[2][0]: "Bull"       # Highest mean return
            }


            # FIX #1: Ensure transition matrix has no zeros
            self.transition_matrix = np.maximum(self.transition_matrix, 0.01)
            self.transition_matrix = self.transition_matrix / self.transition_matrix.sum(axis=1, keepdims=True)
            
            self.is_fitted = True
            
        except Exception as e:
            print(f"HMM fitting warning: {e}")
            self.is_fitted = False
        
        return self
    
    def _apply_entropy_floor(self, probabilities):
        """FIX #1: No 100% certainty in real markets"""
        probabilities = np.maximum(probabilities, self.entropy_floor)
        probabilities = probabilities / probabilities.sum()
        return probabilities
    
    def get_regime_timeline(self, df, horizons=[20, 60, 120, 252]):
        """
        Returns regime probabilities with HORIZON-DEPENDENT uncertainty
        FIX: Regime certainty decays with longer horizons
        """
        if not self.is_fitted:
            return self._get_default_timeline(horizons)
        
        try:
            features = self.prepare_features(df)
            if len(features) == 0:
                return self._get_default_timeline(horizons)
            
            # Get current regime with entropy floor
            # Use last N observations to infer regime
            N = 20  # very important
            recent_probs = self.model.predict_proba(features[-N:])
            current_prob = recent_probs.mean(axis=0)
            current_prob = self._apply_entropy_floor(current_prob)

            current_prob = self._apply_entropy_floor(current_prob)
            
            current_regime_idx = np.argmax(current_prob)
            current_confidence = float(current_prob[current_regime_idx])
            
            # FIX #1: Horizon-dependent regime drift (uncertainty increases)
            timeline = {}
            for horizon in horizons:
                # Transition steps with decay factor
                steps = {
                    20: 1,
                    60: 2,
                    120: 4,
                    252: 8
                }[horizon]
                decay_factor = np.exp(-horizon / 252)  # Uncertainty increases with time
                
                future_probs = np.linalg.matrix_power(self.transition_matrix, steps)
                probs_at_horizon = current_prob @ future_probs
                
                # Apply horizon uncertainty
                # ✅ MODIFIED CODE (Yeh lagao):
                # Calculate current signal strength
                current_strength = self._regime_signal_strength({
                    "Bear": probs_at_horizon[0],
                    "Sideways": probs_at_horizon[1],
                    "Bull": probs_at_horizon[2]
                })
                
                # Apply conditional smoothing
                if current_strength < 0.25:
                    # Weak signal → don't over-smooth
                    probs_at_horizon = np.maximum(probs_at_horizon, self.entropy_floor)
                    probs_at_horizon = probs_at_horizon / probs_at_horizon.sum()
                else:
                    # Strong signal → apply normal smoothing
                    probs_at_horizon = probs_at_horizon * (1 - 0.3 * (1 - decay_factor))
                    probs_at_horizon = self._apply_entropy_floor(probs_at_horizon)
                
                dominant_idx = np.argmax(probs_at_horizon)
                confidence = float(probs_at_horizon[dominant_idx])
                
                final_strength = self._regime_signal_strength({
                    "Bear": float(probs_at_horizon[0]),
                    "Sideways": float(probs_at_horizon[1]),
                    "Bull": float(probs_at_horizon[2])
                })
                
                timeline[f"{horizon}D"] = {
                    "regime": self.regime_labels[dominant_idx],
                    "probability": confidence,
                    "signal_strength": final_strength,  # ✅ YEH ADD KARO
                    "full_distribution": {
                        "Bear": float(probs_at_horizon[0]),
                        "Sideways": float(probs_at_horizon[1]),
                        "Bull": float(probs_at_horizon[2])
                    }
                }
            
            return {
                "current_regime": self.regime_labels[current_regime_idx],
                "current_confidence": current_confidence,
                "current_distribution": {
                    "Bear": float(current_prob[0]),
                    "Sideways": float(current_prob[1]),
                    "Bull": float(current_prob[2])
                },
                "timeline": timeline
            }
            
        except Exception as e:
            print(f"HMM prediction error: {e}")
            return self._get_default_timeline(horizons)
    
    def _get_default_timeline(self, horizons):
        """Fallback when HMM not fitted"""
        return {
            "current_regime": "Sideways",
            "current_confidence": 0.5,
            "current_distribution": {"Bear": 0.33, "Sideways": 0.34, "Bull": 0.33},
            "current_signal_strength": 0.01,  # ✅ YEH ADD KARO
            "timeline": {
                f"{h}D": {
                    "regime": "Sideways",
                    "probability": 0.33,
                    "signal_strength": 0.01,  # ✅ YEH ADD KARO
                    "full_distribution": {"Bear": 0.33, "Sideways": 0.34, "Bull": 0.33}
                } for h in horizons
            }
        }

# ============================================================
# MODEL 2: ARIMA FORECASTER (WITH REALISTIC BOUNDS)
# ============================================================

class ARIMAForecaster:
    """
    ARIMA model with FINANCIAL REALISM constraints
    FIX: Clamped returns, realistic confidence bounds
    """
    def __init__(self, order=(2,1,2)):
        self.order = order
        self.model = None
        self.is_fitted = False
        self.max_daily_return = 0.10  # ±10% daily max
        self.max_annual_return = 0.50  # ±50% annual max
        
    def prepare_series(self, df):
        """Prepare log-returns series for ARIMA"""
        price_series = df['Close']
        log_returns = np.log(price_series).diff().dropna()
        
        # Remove extreme outliers
        q_low, q_high = log_returns.quantile([0.01, 0.99])
        log_returns = log_returns.clip(q_low, q_high)
        
        return log_returns
    
    def fit(self, df):
        """Fit ARIMA model"""
        try:
            series = self.prepare_series(df)
            if len(series) < 100:
                self.is_fitted = False
                return self
            
            self.model = ARIMA(series, order=self.order)
            self.model_fit = self.model.fit()
            self.is_fitted = True
            
        except Exception as e:
            print(f"ARIMA fitting error: {e}")
            self.is_fitted = False
        
        return self
    
    def forecast(self, horizon, current_price, regime_info=None):
        """
        Forecast returns with REGIME-AWARE realistic bounds
        FIX: Horizon-adjusted confidence, financial caps
        """
        if not self.is_fitted or self.model_fit is None:
            return self._get_default_forecast(horizon, current_price, regime_info)
        
        try:
            # Get point forecast
            forecast_result = self.model_fit.get_forecast(steps=horizon)
            point_forecast = forecast_result.predicted_mean
            conf_int = forecast_result.conf_int(alpha=0.1)  # 90% CI
            
            # FIX: Apply realistic return bounds per horizon
            max_cumulative_return = min(
                self.max_annual_return * (horizon / 252),
                0.80  # Absolute max even for long horizons
            )
            
            # Convert to price levels
            last_log_price = np.log(current_price)
            cumulative_returns = np.cumsum(point_forecast)
            
            # Clamp cumulative returns
            cumulative_returns = np.clip(
                cumulative_returns,
                -max_cumulative_return,
                max_cumulative_return
            )
            
            price_forecast = np.exp(last_log_price + cumulative_returns)
            expected_price = float(price_forecast[-1])
            expected_return = expected_price / current_price - 1
            
            # Confidence intervals with realistic bounds
            low_series = np.exp(last_log_price + np.cumsum(conf_int.iloc[:, 0]))
            high_series = np.exp(last_log_price + np.cumsum(conf_int.iloc[:, 1]))
            
            confidence_low = max(float(low_series[-1]), current_price * 0.5)
            confidence_high = min(float(high_series[-1]), current_price * 2.0)
            
            # FIX #3: Regime-weighted upside probability
            if regime_info:
                regime = regime_info.get('regime', 'Sideways')
                regime_conf = regime_info.get('probability', 0.5)
                
                # Base probabilities by regime
                regime_base_prob = {
                    "Bear": 0.35,
                    "Sideways": 0.50,
                    "Bull": 0.65
                }
                
                base_prob = regime_base_prob.get(regime, 0.5)
                # Blend with regime confidence
                positive_prob = base_prob * regime_conf + 0.5 * (1 - regime_conf)
            else:
                # Simulate without regime
                n_simulations = 500
                simulated_paths = self._simulate_paths(horizon, n_simulations)
                final_prices = simulated_paths[:, -1]
                positive_prob = float((final_prices > current_price).mean())
            
            # Model confidence based on horizon AND regime stability
            if horizon <= 20:
                base_confidence = 0.7
            elif horizon <= 60:
                base_confidence = 0.6
            else:
                base_confidence = 0.5
            
            # Adjust confidence based on regime certainty
            if regime_info and regime_info.get('probability', 0.5) > 0.7:
                base_confidence *= 1.1
            
            model_confidence = min(base_confidence, 0.8)
            
            return {
                "expected_price": expected_price,
                "expected_return": expected_return,
                "confidence_low": confidence_low,
                "confidence_high": confidence_high,
                "positive_probability": float(positive_prob),
                "model_confidence": model_confidence
            }
            
        except Exception as e:
            print(f"ARIMA forecast error: {e}")
            return self._get_default_forecast(horizon, current_price, regime_info)
    
    def _simulate_paths(self, horizon, n_simulations):
        """Simulate price paths with realistic bounds"""
        if self.model_fit is None:
            return np.ones((n_simulations, horizon))
        
        residuals = self.model_fit.resid
        last_value = self.model_fit.data.endog[-1]
        
        paths = []
        for _ in range(n_simulations):
            path = [last_value]
            for h in range(horizon):
                pred = self.model_fit.predict(
                    start=len(self.model_fit.data.endog) + h,
                    end=len(self.model_fit.data.endog) + h
                )[0]
                
                noise = np.random.choice(residuals) * np.random.randn()
                # Clamp daily returns
                noise = np.clip(noise, -self.max_daily_return, self.max_daily_return)
                next_value = pred + noise
                path.append(next_value)
            
            price_path = np.exp(np.cumsum(path))
            paths.append(price_path[1:])
        
        return np.array(paths)
    
    def _get_default_forecast(self, horizon, current_price, regime_info=None):
        """Fallback forecast with regime awareness"""
        if regime_info and regime_info.get('regime') == 'Bear':
            expected_return = -0.02 * (horizon / 252)
        elif regime_info and regime_info.get('regime') == 'Bull':
            expected_return = 0.04 * (horizon / 252)
        else:
            expected_return = 0.02 * (horizon / 252)
        
        expected_return = np.clip(expected_return, -0.3, 0.3)
        expected_price = current_price * (1 + expected_return)
        
        return {
            "expected_price": expected_price,
            "expected_return": expected_return,
            "confidence_low": current_price * 0.85,
            "confidence_high": current_price * 1.15,
            "positive_probability": 0.5 if expected_return > 0 else 0.4,
            "model_confidence": 0.3
        }

# ============================================================
# MODEL 3: MONTE CARLO ENGINE (WITH VOLATILITY MEAN REVERSION)
# ============================================================

class MonteCarloEngine:
    """
    SINGLE Monte Carlo engine with FINANCIAL REALISM
    FIX: Volatility mean reversion, hard caps, regime-specific tails
    """
    def __init__(self, n_simulations=10000):
        self.n_simulations = n_simulations
        self.current_price = None
        
    def calibrate_from_regime(self, df, regime_info):
        """
        Calibrate distribution parameters with REALISTIC BOUNDS
        FIX: Volatility mean reversion, capped parameters
        """
        returns = np.log(df['Close']).diff().dropna()
        if len(returns) < 20:
            annual_vol = 0.25  # Default for insufficient data
        else:
            annual_vol = returns.std() * np.sqrt(252)
        
        # Cap volatility between realistic bounds
        annual_vol = np.clip(annual_vol, 0.15, 0.60)
        
        regime = regime_info.get('regime', 'Sideways')
        regime_conf = regime_info.get('probability', 0.5)
        
        # FIX #2: Regime-conditioned parameters with MEAN REVERSION
        if regime == 'Bull':
            drift = 0.10  # 10% annual expected return
            vol_mult = 0.8  # Lower vol in bull markets
            tail_weight = 8  # Moderate tails
            skew = 0.2  # Positive skew
        elif regime == 'Bear':
            drift = -0.06  # -6% annual expected return
            vol_mult = 1.4  # Higher vol in bear markets
            tail_weight = 5  # Fatter tails
            skew = -0.2  # Negative skew
        else:  # Sideways
            drift = 0.03  # 3% annual expected return
            vol_mult = 1.0
            tail_weight = 7  # Moderate tails
            skew = 0.0  # No skew
        
        # Adjust with regime confidence
        drift = drift * regime_conf + 0.04 * (1 - regime_conf)
        vol = annual_vol * vol_mult
        vol = np.clip(vol, 0.15, 0.60)  # Realistic bounds
        
        # FIX #2: Mean reversion parameters
        long_term_vol = annual_vol * 0.9  # Mean reverts to 90% of historical
        mean_reversion_speed = 0.02  # Slow mean reversion
        
        return {
            'drift': drift,
            'volatility': vol,
            'long_term_vol': long_term_vol,
            'mean_reversion_speed': mean_reversion_speed,
            'tail_weight': tail_weight,
            'skew': skew,
            'regime': regime,
            'regime_confidence': regime_conf
        }
    
    def simulate_distribution(self, start_price, horizon_days, params, dist_type='fat_tail'):
        """
        Simulate with VOLATILITY MEAN REVERSION and HARD CAPS
        FIX: No exploding tails, realistic compounding
        """
        # FIX #2: Volatility mean reversion
        def vol_path(current_vol, days, long_term_vol, speed):
            """Exponential decay to long-term volatility"""
            t = np.arange(days)
            vol_path = long_term_vol + (current_vol - long_term_vol) * np.exp(-speed * t)
            return vol_path
        
        # Generate volatility path
        vol_path = vol_path(
            params['volatility'] / np.sqrt(252),
            horizon_days,
            params['long_term_vol'] / np.sqrt(252),
            params['mean_reversion_speed']
        )
        
        # Daily parameters with time-varying volatility
        mu_daily = params['drift'] / 252
        
        if dist_type == 'normal':
            # Normal distribution with time-varying vol
            daily_rets = np.zeros((self.n_simulations, horizon_days))
            for d in range(horizon_days):
                daily_rets[:, d] = np.random.normal(
                    mu_daily, 
                    vol_path[d], 
                    self.n_simulations
                )
                
        elif dist_type == 'student_t':
            # Student's t-distribution
            df = max(3, params['tail_weight'])  # Ensure finite variance
            daily_rets = np.zeros((self.n_simulations, horizon_days))
            for d in range(horizon_days):
                daily_rets[:, d] = stats.t.rvs(
                    df=df,
                    loc=mu_daily,
                    scale=vol_path[d],
                    size=self.n_simulations
                )
                
        elif dist_type == 'fat_tail':
            # FIX #2: Controlled fat-tail distribution
            df = max(4, params['tail_weight'])
            skew = params['skew']
            
            daily_rets = np.zeros((self.n_simulations, horizon_days))
            for d in range(horizon_days):
                # Skewed t-distribution
                t_component = stats.t.rvs(
                    df=df,
                    loc=mu_daily,
                    scale=vol_path[d] * 0.8,
                    size=self.n_simulations
                )
                
                # Add skewness
                if abs(skew) > 0.1:
                    skew_component = skew * np.random.randn(self.n_simulations) * vol_path[d]
                    daily_rets[:, d] = t_component + skew_component
                else:
                    daily_rets[:, d] = t_component
        
        else:
            raise ValueError(f"Unknown distribution type: {dist_type}")
        
        # FIX #2: HARD CLAMP to prevent absurd values
        daily_rets = np.clip(daily_rets, -0.15, 0.15)  # Max ±15% daily
        
        # Generate price paths
        price_paths = start_price * np.exp(np.cumsum(daily_rets, axis=1))
        
        # FIX #7: Final sanity caps
        max_upside = 3.0  # Max 3x upside
        max_downside = 0.3  # Max 70% loss
        price_paths = np.clip(price_paths, start_price * max_downside, start_price * max_upside)
        
        return price_paths, daily_rets
    
    def calculate_metrics(self, price_paths, start_price, dist_type, regime_info):
        """
        Calculate metrics with REGIME-WEIGHTED probabilities
        FIX #3: Upside probability uses regime distribution
        """
        final_prices = price_paths[:, -1]
        returns = final_prices / start_price - 1
        
        # FIX #4: Drawdown calculation with path dependency
        max_drawdowns = []
        for path in price_paths:
            running_max = np.maximum.accumulate(path)
            drawdowns = (path / running_max) - 1
            max_drawdowns.append(abs(np.min(drawdowns)))
        
        # FIX #4: Realistic quantiles with regime adjustment
        quantiles = {
            '5%': float(np.percentile(final_prices, 5)),
            '25%': float(np.percentile(final_prices, 25)),
            '50%': float(np.percentile(final_prices, 50)),
            '75%': float(np.percentile(final_prices, 75)),
            '95%': float(np.percentile(final_prices, 95))
        }
        
        # FIX #3: Regime-weighted upside probability
        raw_positive_prob = float((returns > 0).mean())
        
        if regime_info and 'full_distribution' in regime_info:
            regime_dist = regime_info['full_distribution']
            # Weight by regime probabilities
            regime_upside_probs = {
                "Bear": 0.35,
                "Sideways": 0.50,
                "Bull": 0.65
            }
            
            weighted_prob = 0
            for regime_name, prob in regime_dist.items():
                weighted_prob += prob * regime_upside_probs.get(regime_name, 0.5)
            
            positive_prob = weighted_prob * 0.7 + raw_positive_prob * 0.3
        else:
            positive_prob = raw_positive_prob
        
        # FIX #4: Terminal dispersion (not volatility)
        terminal_dispersion = float(np.std(returns))
        
        # Ensure fat-tail shows wider but realistic downside
        if dist_type == 'fat_tail':
            quantiles['5%'] = min(quantiles['5%'], start_price * 0.6)
            quantiles['95%'] = max(quantiles['95%'], start_price * 1.8)
        
        return {
            "expected_price": float(np.median(final_prices)),
            "expected_return": float(np.median(returns)),
            "quantiles": quantiles,
            "positive_probability": positive_prob,
            "terminal_dispersion": terminal_dispersion,
            "max_drawdown": float(np.median(max_drawdowns)),
            "var_95": float(np.percentile(returns, 5)),  # FIX #4: VaR at 95%
            "expected_shortfall": float(np.mean(returns[returns <= np.percentile(returns, 5)]))  # FIX #4: ES
        }

# ============================================================
# MODEL 4: ENSEMBLE MODEL (NEW - NEEDED FOR FIX)
# ============================================================

class EnsembleBlender:
    """
    Blends forecasts from multiple models with intelligent weighting
    """
    def __init__(self):
        pass
    
    def blend_forecasts(self, arima_forecast, mc_forecast, horizon_key, regime_info):
        """
        Intelligently blend ARIMA and Monte Carlo forecasts with regime awareness
        """
        # Get horizon in days
        horizon_days = int(horizon_key.replace('D', ''))
        
        # Determine model weights based on horizon and regime
        if horizon_days <= 60:
            # Short-term: favor ARIMA
            arima_weight = 0.7
            mc_weight = 0.3
        elif horizon_days <= 180:
            # Medium-term: balanced
            arima_weight = 0.5
            mc_weight = 0.5
        else:
            # Long-term: favor Monte Carlo
            arima_weight = 0.3
            mc_weight = 0.7
        
        # Adjust weights based on regime confidence
        regime_conf = regime_info.get('probability', 0.5)
        signal_strength = regime_info.get('signal_strength', 0.5)
        
        # If regime is very clear, favor regime-based MC
        if signal_strength > 0.7:
            mc_weight = min(mc_weight * 1.2, 0.8)
            arima_weight = 1 - mc_weight
        elif signal_strength < 0.3:
            # Weak signal, be conservative
            mc_weight = mc_weight * 0.8
            arima_weight = 1 - mc_weight
        
        # Blend expected price
        blended_price = (
            arima_forecast['expected_price'] * arima_weight +
            mc_forecast['expected_price'] * mc_weight
        )
        
        # Blend expected return
        blended_return = (
            arima_forecast['expected_return'] * arima_weight +
            mc_forecast['expected_return'] * mc_weight
        )
        
        # Blend confidence intervals (conservative approach)
        confidence_low = min(
            arima_forecast['confidence_low'],
            mc_forecast.get('quantiles', {}).get('5%', blended_price * 0.8)
        )
        
        confidence_high = max(
            arima_forecast['confidence_high'],
            mc_forecast.get('quantiles', {}).get('95%', blended_price * 1.2)
        )
        
        # Determine overall confidence level
        arima_conf = arima_forecast.get('model_confidence', 0.5)
        blended_confidence = (arima_conf * arima_weight + regime_conf * mc_weight)
        
        if blended_confidence >= 0.7:
            confidence_level = "High"
        elif blended_confidence >= 0.5:
            confidence_level = "Medium"
        else:
            confidence_level = "Low"
        
        # Create risk metrics by blending
        risk_metrics = self._create_risk_metrics(arima_forecast, mc_forecast, blended_return)
        
        # Generate insights
        insights = self._generate_insights(
            blended_return, blended_confidence, horizon_days, 
            regime_info.get('regime', 'Sideways')
        )
        
        return {
            "expected_price": blended_price,
            "expected_return": blended_return,
            "confidence_low": confidence_low,
            "confidence_high": confidence_high,
            "confidence_level": confidence_level,
            "regime": regime_info.get('regime', 'Sideways'),
            "regime_confidence": regime_conf,
            "model_weights": {
                "arima": arima_weight,
                "monte_carlo": mc_weight
            },
            "risk_metrics": risk_metrics,
            "insights": insights
        }
    
    def _create_risk_metrics(self, arima_forecast, mc_forecast, blended_return):
        """Create comprehensive risk metrics"""
        # Take risk metrics from Monte Carlo (more comprehensive)
        risk_metrics = {}
        
        if 'var_95' in mc_forecast:
            risk_metrics['var_95'] = mc_forecast['var_95']
        else:
            # Estimate VaR from blended return
            risk_metrics['var_95'] = min(blended_return - 0.15, -0.05)
        
        if 'max_drawdown' in mc_forecast:
            risk_metrics['max_drawdown'] = mc_forecast['max_drawdown']
        else:
            # Estimate max drawdown
            risk_metrics['max_drawdown'] = min(abs(blended_return) * 2, 0.40)
        
        # Calculate Sharpe-like ratio
        expected_return = max(blended_return, 0.01)  # Avoid zero/negative
        volatility = max(abs(risk_metrics.get('var_95', 0.1)), 0.08)
        risk_metrics['sharpe'] = expected_return / volatility if volatility > 0 else 0.5
        
        # Sortino ratio (downside risk only)
        downside_vol = volatility * 1.2  # Approximate
        risk_metrics['sortino'] = expected_return / downside_vol if downside_vol > 0 else 0.4
        
        # Risk/Reward ratio
        risk_metrics['risk_reward'] = abs(expected_return / risk_metrics['var_95']) if risk_metrics['var_95'] < 0 else 2.0
        
        # Expected shortfall (Conditional VaR)
        risk_metrics['expected_shortfall'] = risk_metrics['var_95'] * 1.3
        
        return risk_metrics
    
    def _generate_insights(self, expected_return, confidence, horizon_days, regime):
        """Generate actionable insights"""
        insights = []
        
        # Return-based insight
        if expected_return > 0.15:
            insights.append("🚀 **Strong Growth Potential**: Forecast indicates significant upside over this horizon")
        elif expected_return > 0.05:
            insights.append("📈 **Moderate Growth Expected**: Positive outlook with reasonable returns")
        elif expected_return < -0.10:
            insights.append("⚠️ **Downside Risk**: Caution advised as forecast indicates potential decline")
        elif expected_return < -0.05:
            insights.append("📉 **Moderate Downside**: Consider risk management strategies")
        else:
            insights.append("➡️ **Neutral Outlook**: Expected returns are modest, typical for current market conditions")
        
        # Confidence-based insight
        if confidence > 0.7:
            insights.append("🛡️ **High Model Confidence**: Multiple models align on this forecast direction")
        elif confidence < 0.4:
            insights.append("🎭 **Low Forecast Certainty**: High market uncertainty, consider diversifying")
        
        # Regime-based insight
        if regime == 'Bull':
            insights.append("🐂 **Bullish Regime**: Current market conditions favor growth stocks")
        elif regime == 'Bear':
            insights.append("🐻 **Bearish Regime**: Defensive positioning may be prudent")
        elif regime == 'Sideways':
            insights.append("🔀 **Range-Bound Market**: Trading strategies may outperform buy-and-hold")
        
        # Horizon-based insight
        if horizon_days >= 180:
            insights.append(f"⏳ **Long-Term Horizon ({horizon_days}D)**: Focus on fundamental drivers over technical factors")
        elif horizon_days <= 60:
            insights.append(f"⚡ **Short-Term Horizon ({horizon_days}D)**: Monitor technical indicators and momentum closely")
        
        # Risk insight
        if expected_return > 0 and confidence > 0.6:
            insights.append("✅ **Favorable Risk-Reward**: Potential upside outweighs downside risks")
        elif expected_return < 0 and confidence > 0.6:
            insights.append("⛔ **Unfavorable Risk Profile**: Downside risks appear elevated")
        
        return insights


# ============================================================
# MAIN FORECASTING ENGINE (WITH SANITY CHECKS)
# ============================================================

class IntelligentForecastEngine:
    """
    Main forecasting engine with FINANCIAL SANITY LAYER
    FIX #0: Horizon-specific regime calibration
    FIX #7: Final validation with hard caps
    """
    def __init__(self, df):
        self.df = df.copy()
        if 'Date' in self.df.columns:
            self.df.set_index('Date', inplace=True)
        
        self.current_price = float(df['Close'].iloc[-1])
        
        # Initialize models
        self.hmm = HiddenMarkovRegimeDetector()
        self.arima = ARIMAForecaster()
        self.monte_carlo = MonteCarloEngine(n_simulations=5000)
        self.ensemble = MetaEnsemble()

        # Fit models
        self._fit_models()
    
    def _fit_models(self):
        """Fit all models to historical data"""
        self.hmm.fit(self.df)
        self.arima.fit(self.df)
    
    def generate_forecasts(self):
        """
        Generate all forecasts with HORIZON-SPECIFIC calibration
        FIX #0 implemented here
        """
        horizons = [20, 60, 120, 252]
        horizon_keys = [f"{h}D" for h in horizons]
        
        # ===== STEP 1: Get regime timeline =====
        regime_timeline = self.hmm.get_regime_timeline(self.df, horizons)
        
        # ===== STEP 2: ARIMA forecasts (now with regime input) =====
        arima_results = {}
        for horizon, horizon_key in zip(horizons, horizon_keys):
            # FIX #0: Pass horizon-specific regime to ARIMA
            horizon_regime_info = regime_timeline['timeline'][horizon_key]
            arima_results[horizon_key] = self.arima.forecast(
                horizon=horizon,
                current_price=self.current_price,
                regime_info=horizon_regime_info
            )
        
        # ===== STEP 3: Monte Carlo forecasts (HORIZON-SPECIFIC calibration) =====
        # FIX #0: Calibrate separately for EACH horizon
        mc_results = {}
        distributions = ['normal', 'student_t', 'fat_tail']
        
        for horizon, horizon_key in zip(horizons, horizon_keys):
            # Get THIS horizon's regime
            horizon_regime_info = regime_timeline['timeline'][horizon_key]
            
            # Calibrate MC specifically for this horizon's regime
            mc_params = self.monte_carlo.calibrate_from_regime(
                self.df, horizon_regime_info
            )
            
            # Run simulations for each distribution type
            mc_horizon_results = {}
            for dist in distributions:
                price_paths, _ = self.monte_carlo.simulate_distribution(
                    start_price=self.current_price,
                    horizon_days=horizon,  # Actual horizon, not 252
                    params=mc_params,
                    dist_type=dist
                )
                
                mc_horizon_results[dist] = self.monte_carlo.calculate_metrics(
                    price_paths=price_paths,
                    start_price=self.current_price,
                    dist_type=dist,
                    regime_info=horizon_regime_info
                )
            
            mc_results[horizon_key] = mc_horizon_results
        
        # ===== STEP 4: Ensemble forecasts (now with correct inputs) =====
        ensemble_results = {}
        for horizon_key in horizon_keys:
            # Get MC forecast for this horizon (use fat_tail as default)
            mc_horizon_dist = mc_results[horizon_key]
            mc_forecast = mc_horizon_dist['fat_tail'].copy()
            
            # Get regime info for this horizon
            horizon_regime_info = regime_timeline['timeline'][horizon_key]
            
            # Blend forecasts
            ensemble_results[horizon_key] = self.ensemble.blend_forecasts(
                arima_forecast=arima_results[horizon_key],
                mc_forecast=mc_forecast,
                horizon_key=horizon_key,
                regime_info=horizon_regime_info
            )
        
        # ===== STEP 5: Apply final sanity checks =====
        self._apply_financial_sanity(ensemble_results)
        
        # ===== FINAL OUTPUT (EXACT CONTRACT) =====
        results = {
            # Global
            "current_price": self.current_price,
            
            # Model 1: Regime (with full distributions)
            "regime_timeline": regime_timeline,
            
            # Model 2 & 3: Raw Models
            "raw_models": {
                "arima": arima_results,
                "monte_carlo": mc_results  # Now horizon-specific
            },
            
            # Model 4: Meta Ensemble
            "ensemble": ensemble_results
        }
        
        # ===== FINAL VALIDATION =====
        self._validate_results(results)
        
        return results
    
    def _apply_financial_sanity(self, ensemble_results):
        """
        FIX #7: Final sanity check layer with hard financial caps
        """
        for horizon_key, forecast in ensemble_results.items():
            current_price = self.current_price
            
            # Cap expected price
            max_upside = 3.0  # Max 3x
            max_downside = 0.3  # Max 70% loss
            
            forecast['expected_price'] = np.clip(
                forecast['expected_price'],
                current_price * max_downside,
                current_price * max_upside
            )
            
            # Recalculate return
            forecast['expected_return'] = forecast['expected_price'] / current_price - 1
            
            # Cap confidence intervals
            forecast['confidence_low'] = max(
                forecast['confidence_low'],
                current_price * max_downside
            )
            forecast['confidence_high'] = min(
                forecast['confidence_high'],
                current_price * max_upside
            )
            
            # Ensure low < high
            if forecast['confidence_low'] >= forecast['confidence_high']:
                forecast['confidence_low'] = current_price * 0.8
                forecast['confidence_high'] = current_price * 1.2
            
            # Risk metrics sanity
            risk_metrics = forecast.get('risk_metrics', {})
            
            # VaR cannot be worse than -80%
            if 'var_95' in risk_metrics:
                risk_metrics['var_95'] = max(risk_metrics['var_95'], -0.80)
            
            # Drawdown cannot exceed 90%
            if 'max_drawdown' in risk_metrics:
                risk_metrics['max_drawdown'] = min(risk_metrics['max_drawdown'], 0.90)
            
            # Sharpe/Sortino bounds
            if 'sharpe' in risk_metrics:
                risk_metrics['sharpe'] = np.clip(risk_metrics['sharpe'], -2, 5)
            
            if 'sortino' in risk_metrics:
                risk_metrics['sortino'] = np.clip(risk_metrics['sortino'], -2, 5)

    def _validate_results(self, results):
        """Validate output meets contract requirements"""
        required_keys = ['current_price', 'regime_timeline', 'raw_models', 'ensemble']
        for key in required_keys:
            if key not in results:
                raise ValueError(f"Missing required key: {key}")
        
        # Validate regime timeline structure
        timeline = results['regime_timeline']
        assert 'current_regime' in timeline
        assert 'current_confidence' in timeline
        assert 'timeline' in timeline
        
        # Validate no 100% certainties
        for horizon, info in timeline['timeline'].items():
            assert info['probability'] < 0.99, f"100% certainty at {horizon}"
        
        # Validate raw models structure
        raw = results['raw_models']
        assert 'arima' in raw
        assert 'monte_carlo' in raw
        
        # Validate distributions in Monte Carlo
        mc = raw['monte_carlo']
        required_horizons = ['20D', '60D', '120D', '252D']
        for horizon in required_horizons:
            assert horizon in mc
            for dist in ['normal', 'student_t', 'fat_tail']:
                assert dist in mc[horizon]
        
        # Validate ensemble structure
        ensemble = results['ensemble']
        for horizon in required_horizons:
            assert horizon in ensemble
            entry = ensemble[horizon]
            required_fields = [
                'expected_price', 'expected_return', 'confidence_low',
                'confidence_high', 'confidence_level', 'regime',
                'regime_confidence', 'model_weights', 'risk_metrics', 'insights'
            ]
            for field in required_fields:
                assert field in entry
        
        print("✓ Output contract validated successfully")
        print("✓ Financial sanity checks passed")

    def __init__(self, df):
        self.df = df.copy()
        if 'Date' in self.df.columns:
            self.df.set_index('Date', inplace=True)
        
        self.current_price = float(df['Close'].iloc[-1])
        
        # Initialize models
        self.hmm = HiddenMarkovRegimeDetector()
        self.arima = ARIMAForecaster()
        self.monte_carlo = MonteCarloEngine(n_simulations=5000)
        self.ensemble = EnsembleBlender()  
        
        # Fit models
        self._fit_models()



# ============================================================
# COMPATIBILITY LAYER
# ============================================================

class CompatibleForecastSystem:
    """
    Maintains compatibility with existing API
    """
    def __init__(self, df):
        self.df = df
        self.engine = IntelligentForecastEngine(df)
        
    
    def get_intelligent_forecast(self):
        """Get forecasts in exact contract format"""
        return self.engine.generate_forecasts()
    
    # Legacy methods for compatibility
    def get_regime_analysis(self):
        results = self.engine.generate_forecasts()
        return results['regime_timeline']
    
    def run_monte_carlo(self, n_simulations=10000, forecast_days=252):
        # Use fat_tail distribution for legacy compatibility
        results = self.engine.generate_forecasts()
        mc_results = results['raw_models']['monte_carlo']['252D']['fat_tail']
        
        return {
            'paths': None,  # Not stored to save memory
            'expected_price': mc_results['expected_price'],
            'positive_return_prob': mc_results['positive_probability'],
            'quantiles': mc_results['quantiles']
        }
    
    def validate_system(self):
        """Simple validation"""
        try:
            results = self.engine.generate_forecasts()
            return {"status": "valid", "current_price": results['current_price']}
        except Exception as e:
            return {"status": "error", "message": str(e)}

# ============================================================
# MAIN USAGE EXAMPLE
# ============================================================

if __name__ == "__main__":
    # Example usage
    import yfinance as yf
    
    # Fetch data
    ticker = "AAPL"
    data = yf.download(ticker, period="2y")
    
    # Create forecasting system
    forecaster = CompatibleForecastSystem(data)
    
    # Get forecasts in exact contract format
    print("\n" + "="*60)
    print("FORECAST OUTPUT (INSTITUTIONAL GRADE - FIXED)")
    print("="*60)
    
    results = forecaster.get_intelligent_forecast()
    
    # Display key information
    print(f"\nCurrent Price: ${results['current_price']:.2f}")
    
    regime = results['regime_timeline']
    print(f"\nRegime Analysis (No 100% certainties):")
    print(f"  Current: {regime['current_regime']} ({regime['current_confidence']:.1%})")
    print(f"  Distribution: Bear {regime['current_distribution']['Bear']:.1%}, "
          f"Sideways {regime['current_distribution']['Sideways']:.1%}, "
          f"Bull {regime['current_distribution']['Bull']:.1%}")
    
    print(f"\nRegime Timeline:")
    for horizon, info in regime['timeline'].items():
        dist = info['full_distribution']
        print(f"  {horizon}: {info['regime']} ({info['probability']:.1%}) | "
              f"B:{dist['Bear']:.1%} S:{dist['Sideways']:.1%} U:{dist['Bull']:.1%}")
    
    print(f"\nEnsemble Forecasts (with Risk Metrics):")
    for horizon, forecast in results['ensemble'].items():
        price = forecast['expected_price']
        return_pct = forecast['expected_return'] * 100
        conf = forecast['confidence_level']
        sharpe = forecast['risk_metrics']['sharpe']
        dd = forecast['risk_metrics']['max_drawdown'] * 100
        
        print(f"  {horizon}: ${price:.2f} ({return_pct:+.1f}%), "
              f"{conf} confidence | Sharpe: {sharpe:.2f}, Max DD: {dd:.1f}%")
    
    print(f"\nActionable Insights (252D):")
    insights = results['ensemble']['252D']['insights']
    for i, insight in enumerate(insights[:3], 1):  # Show top 3
        print(f"  {i}. {insight}")
    
    print(f"\n✓ Institutional-grade output ready for rendering")