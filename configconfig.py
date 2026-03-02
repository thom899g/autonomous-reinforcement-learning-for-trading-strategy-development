"""
Configuration management for the autonomous trading RL system.
Centralized configuration ensures consistency across all components.
"""
import os
from typing import Dict, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime
import json
from pathlib import Path
import logging

@dataclass
class RLConfig:
    """Reinforcement Learning configuration parameters"""
    # PPO Algorithm Parameters
    learning_rate: float = 3e-4
    batch_size: int = 64
    gamma: float = 0.99  # discount factor
    gae_lambda: float = 0.95  # GAE parameter
    clip_range: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    n_steps: int = 2048
    n_epochs: int = 10
    
    # Network Architecture
    policy_net_hidden_layers: tuple = (256, 128, 64)
    value_net_hidden_layers: tuple = (256, 128)
    activation_fn: str = "relu"
    use_lstm: bool = True
    lstm_size: int = 128
    
    # Training
    total_timesteps: int = 1_000_000
    eval_frequency: int = 10_000
    save_frequency: int = 50_000
    early_stopping_patience: int = 20

@dataclass
class TradingConfig:
    """Trading-specific configuration parameters"""
    # Market Parameters
    symbols: tuple = ("BTC/USDT", "ETH/USDT", "SOL/USDT")
    timeframe: str = "1h"
    initial_balance: float = 10000.0
    max_position_size: float = 0.1  # 10% of portfolio per trade
    transaction_cost: float = 0.001  # 0.1% per trade
    
    # Risk Management
    max_drawdown_limit: float = 0.2  # 20%
    max_consecutive_losses: int = 5
    stop_loss_pct: float = 0.02  # 2%
    take_profit_pct: float = 0.05  # 5%
    
    # Feature Engineering
    technical_indicators: tuple = (
        "rsi", "macd", "bbands", "ema_20", "ema_50",
        "atr", "obv", "volume_ratio"
    )
    lookback_window: int = 100
    normalize_features: bool = True

@dataclass
class SystemConfig:
    """System-level configuration"""
    # Paths
    data_dir: str = "./data"
    model_dir: str = "./models"
    log_dir: str = "./logs"
    
    # Firebase Configuration
    firebase_project_id: Optional[str] = None
    firestore_collection: str = "trading_strategies"
    realtime_db_url: Optional[str] = None
    
    # Logging
    log_level: str = "INFO"
    enable_file_logging: bool = True
    enable_firebase_logging: bool = False
    
    # Performance
    num_workers: int = 4
    use_gpu: bool = True
    seed: int = 42

class ConfigManager:
    """Centralized configuration manager with validation"""
    
    def __init__(self, config_path: Optional[str] = None):
        self.logger = logging.getLogger(__name__)
        self.rl_config = RLConfig()
        self.trading_config = TradingConfig()
        self.system_config = SystemConfig()
        
        if config_path and Path(config_path).exists():
            self.load_from_file(config_path)
        
        self._validate_config()
        self._setup_directories()
    
    def _validate_config(self) -> None:
        """Validate configuration parameters"""
        assert 0 < self.rl_config.learning_rate <= 1, "Learning rate must be between 0 and 1"
        assert self.rl_config.batch_size > 0, "Batch size must be positive"
        assert 0 <= self.rl_config.gamma <= 1, "Gamma must be between 0 and 1"
        assert self.trading_config.initial_balance > 0, "Initial balance must be positive"
        assert 0 <= self.trading_config.transaction_cost <= 1, "Transaction cost must be between 0 and 1"
        
        self.logger.info("Configuration validation passed")
    
    def _setup_directories(self) -> None:
        """Create necessary directories"""
        for dir_path in [
            self.system_config.data_dir,
            self.system_config.model_dir,
            self.system_config.log_dir
        ]:
            Path(dir_path).mkdir(parents=True, exist_ok=True)
        
        self.log