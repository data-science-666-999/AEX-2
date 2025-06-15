import numpy as np
import pandas as pd
import yfinance as yf
import ta
import logging # Keep this for now, will replace with logger from logger_setup
from typing import Tuple, Dict
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from config import Config
from logger_setup import logger # Import the logger

class AEXDataProcessor:
    """Professional data processing pipeline for AEX index forecasting."""

    def __init__(self, config: Config):
        self.config = config
        self.scaler = None
        self.feature_columns = None
        self.raw_data = None
        self.processed_data = None

    def download_data(self) -> pd.DataFrame:
        """Download AEX index data with comprehensive error handling."""
        try:
            logger.info(f"Downloading AEX data from {self.config.data_config['start_date']} to {self.config.data_config['end_date']}")

            ticker = yf.Ticker(self.config.data_config['symbol'])
            data = ticker.history(
                start=self.config.data_config['start_date'],
                end=self.config.data_config['end_date'],
                interval='1d'
            )

            if data.empty:
                raise ValueError("No data retrieved from Yahoo Finance")

            logger.info(f"Successfully downloaded {len(data)} trading days of data")
            self.raw_data = data

            return data

        except Exception as e:
            logger.error(f"Error downloading data: {str(e)}")
            raise

    def add_technical_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add comprehensive technical indicators to the dataset."""
        try:
            logger.info("Computing technical indicators...")

            # Trend Indicators
            data['SMA_5'] = ta.trend.sma_indicator(data['Close'], window=5)
            data['SMA_10'] = ta.trend.sma_indicator(data['Close'], window=10)
            data['SMA_20'] = ta.trend.sma_indicator(data['Close'], window=20)
            data['SMA_50'] = ta.trend.sma_indicator(data['Close'], window=50)

            data['EMA_5'] = ta.trend.ema_indicator(data['Close'], window=5)
            data['EMA_10'] = ta.trend.ema_indicator(data['Close'], window=10)
            data['EMA_20'] = ta.trend.ema_indicator(data['Close'], window=20)

            # MACD
            macd = ta.trend.MACD(data['Close'])
            data['MACD'] = macd.macd()
            data['MACD_signal'] = macd.macd_signal()
            data['MACD_histogram'] = macd.macd_diff()

            # Momentum Indicators
            data['RSI'] = ta.momentum.rsi(data['Close'], window=14)
            data['Stochastic_K'] = ta.momentum.stoch(data['High'], data['Low'], data['Close'])
            data['Stochastic_D'] = ta.momentum.stoch_signal(data['High'], data['Low'], data['Close'])

            # Volatility Indicators
            bollinger = ta.volatility.BollingerBands(data['Close'])
            data['BB_upper'] = bollinger.bollinger_hband()
            data['BB_middle'] = bollinger.bollinger_mavg()
            data['BB_lower'] = bollinger.bollinger_lband()
            data['BB_width'] = (data['BB_upper'] - data['BB_lower']) / data['BB_middle']

            data['ATR'] = ta.volatility.average_true_range(data['High'], data['Low'], data['Close'])

            # Volume Indicators
            data['Volume_SMA'] = ta.volume.volume_sma(data['Close'], data['Volume']) # Changed from ta.volume.VolumeSMA to ta.volume.volume_sma
            data['OBV'] = ta.volume.on_balance_volume(data['Close'], data['Volume'])

            # Price-based features
            data['High_Low_Ratio'] = data['High'] / data['Low']
            data['Close_Open_Ratio'] = data['Close'] / data['Open']

            # Returns
            data['Returns'] = data['Close'].pct_change()
            data['Log_Returns'] = np.log(data['Close'] / data['Close'].shift(1))

            # Volatility (rolling standard deviation of returns)
            data['Volatility_5'] = data['Returns'].rolling(window=5).std()
            data['Volatility_20'] = data['Returns'].rolling(window=20).std()

            logger.info(f"Added {len(data.columns) - 5} technical indicators") # Assuming 5 base columns before adding TIs

            return data

        except Exception as e:
            logger.error(f"Error computing technical indicators: {str(e)}")
            raise

    def clean_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Comprehensive data cleaning and quality assurance."""
        try:
            logger.info("Starting data cleaning process...")

            initial_rows = len(data)

            # Remove rows with all NaN values
            data = data.dropna(how='all')

            # Forward fill missing values for financial data
            data = data.fillna(method='ffill')

            # Backward fill any remaining NaN values at the beginning
            data = data.fillna(method='bfill')

            # Remove any remaining rows with NaN values
            data = data.dropna()

            # Remove outliers using IQR method for returns
            if 'Returns' in data.columns:
                Q1 = data['Returns'].quantile(0.01)
                Q3 = data['Returns'].quantile(0.99)
                data = data[(data['Returns'] >= Q1) & (data['Returns'] <= Q3)]

            # Ensure positive values for volume
            if 'Volume' in data.columns:
                data = data[data['Volume'] > 0]

            final_rows = len(data)
            logger.info(f"Data cleaning completed: {initial_rows} -> {final_rows} rows ({final_rows/initial_rows:.2%} retained)")

            return data

        except Exception as e:
            logger.error(f"Error during data cleaning: {str(e)}")
            raise

    def scale_features(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, MinMaxScaler]: # Adjusted return type hint
        """Scale features using specified scaling method."""
        try:
            logger.info(f"Scaling features using {self.config.data_config['scaling_method']} method...")

            # Select numeric columns only
            numeric_columns = data.select_dtypes(include=[np.number]).columns.tolist()

            if self.config.data_config['scaling_method'] == 'minmax':
                scaler = MinMaxScaler(feature_range=(0, 1))
            elif self.config.data_config['scaling_method'] == 'standard':
                scaler = StandardScaler()
            else:
                raise ValueError(f"Unsupported scaling method: {self.config.data_config['scaling_method']}")

            # Fit scaler on training data only (first 70% of data)
            train_size = int(len(data) * self.config.data_config['train_ratio'])
            train_data = data.iloc[:train_size]

            scaler.fit(train_data[numeric_columns])

            # Transform all data
            scaled_values = scaler.transform(data[numeric_columns])
            scaled_data = pd.DataFrame(scaled_values, columns=numeric_columns, index=data.index)

            # Add back non-numeric columns if any
            non_numeric_columns = data.select_dtypes(exclude=[np.number]).columns.tolist()
            for col in non_numeric_columns:
                scaled_data[col] = data[col].values

            self.scaler = scaler
            self.feature_columns = numeric_columns

            logger.info(f"Successfully scaled {len(numeric_columns)} features")

            return scaled_data, self.scaler # Return self.scaler which is the fitted scaler

        except Exception as e:
            logger.error(f"Error during feature scaling: {str(e)}")
            raise

    def create_sequences(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Create sequences for LSTM training with comprehensive validation."""
        try:
            logger.info("Creating sequences for LSTM training...")

            sequence_length = self.config.data_config['sequence_length']

            # Select features for modeling
            if self.config.data_config['technical_indicators']:
                # Use all available features (that are numeric and not the target)
                feature_cols = [col for col in data.columns if col != 'Close' and pd.api.types.is_numeric_dtype(data[col])]
                target_col = 'Close'
            else:
                # Use only OHLCV data (ensure these are present and numeric)
                base_features = ['Open', 'High', 'Low', 'Volume'] # 'Close' is target
                feature_cols = [col for col in base_features if col in data.columns and pd.api.types.is_numeric_dtype(data[col])]
                target_col = 'Close'

            if target_col not in data.columns:
                logger.error(f"Target column '{target_col}' not found in data.")
                raise ValueError(f"Target column '{target_col}' not found.")

            # Ensure all feature columns exist and are numeric
            available_cols = [col for col in feature_cols if col in data.columns and pd.api.types.is_numeric_dtype(data[col])]
            if len(available_cols) != len(feature_cols):
                missing_cols = set(feature_cols) - set(available_cols)
                logger.warning(f"Missing or non-numeric columns: {missing_cols}. Using available numeric columns: {available_cols}")
                feature_cols = available_cols

            if not feature_cols:
                logger.error("No valid feature columns found for sequence creation.")
                raise ValueError("No valid feature columns available.")

            features = data[feature_cols].values
            targets = data[target_col].values

            X, y = [], []

            for i in range(sequence_length, len(data)):
                X.append(features[i-sequence_length:i])
                y.append(targets[i])

            X = np.array(X)
            y = np.array(y)

            if X.shape[0] == 0:
                logger.warning("No sequences were created. Check data length and sequence length.")
            else:
                logger.info(f"Created {len(X)} sequences with shape {X.shape}")
                logger.info(f"Features used for sequences: {feature_cols}")

            return X, y

        except Exception as e:
            logger.error(f"Error creating sequences: {str(e)}")
            raise

    def split_data(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, ...]:
        """Split data into train, validation, and test sets with temporal ordering."""
        try:
            logger.info("Splitting data into train/validation/test sets...")

            total_samples = len(X)
            train_size = int(total_samples * self.config.data_config['train_ratio'])
            val_size = int(total_samples * self.config.data_config['val_ratio'])

            X_train = X[:train_size]
            y_train = y[:train_size]

            X_val = X[train_size:train_size + val_size]
            y_val = y[train_size:train_size + val_size]

            X_test = X[train_size + val_size:]
            y_test = y[train_size + val_size:]

            logger.info(f"Data split - Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

            return X_train, X_val, X_test, y_train, y_val, y_test

        except Exception as e:
            logger.error(f"Error splitting data: {str(e)}")
            raise

    def process_all(self) -> Dict[str, np.ndarray]:
        """Execute complete data processing pipeline."""
        try:
            logger.info("Starting complete data processing pipeline...")

            # Download data
            data = self.download_data()

            # Add technical indicators if requested
            if self.config.data_config['technical_indicators']:
                data = self.add_technical_indicators(data)

            # Clean data
            data = self.clean_data(data) # This data is now cleaned, before scaling

            # Scale features
            scaled_data, scaler = self.scale_features(data.copy()) # Pass a copy to avoid changing 'data' which might be used later

            # Create sequences
            X, y = self.create_sequences(scaled_data)

            # Split data
            X_train, X_val, X_test, y_train, y_val, y_test = self.split_data(X, y)

            # Store processed data
            self.processed_data = {
                'X_train': X_train,
                'X_val': X_val,
                'X_test': X_test,
                'y_train': y_train,
                'y_val': y_val,
                'y_test': y_test,
                'scaler': scaler, # Storing the fitted scaler
                'feature_columns': self.feature_columns,
                'raw_data': self.raw_data, # Storing raw_data downloaded
                'cleaned_data': data, # Storing data after cleaning and TI addition
                'scaled_data': scaled_data # Storing data after scaling
            }

            logger.info("Data processing pipeline completed successfully")

            return self.processed_data

        except Exception as e:
            logger.error(f"Error in data processing pipeline: {str(e)}")
            raise
