import numpy as np
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder

class DatasetBuilder:
    def __init__(self, n_samples=100, random_seed=42):
        self.n_samples = n_samples
        np.random.seed(random_seed)
        
        # Initialize empty dictionary to hold features
        self.features = {}
        self.coefficients = {}
        self.noise = {}
        self.ordinal_encoders = {}

    def add_feature(self, name, generator, coefficient=1, noise_level=0, depends_on=None, correlation_factor=0):
        """
        Add a feature to the dataset.

        Parameters:
        name: str
            The name of the feature (e.g., 'level', 'hours_played')
        generator: function
            A function that generates the values for this feature
        coefficient: float
            The coefficient to apply for this feature in the linear relationship, higher values mean more impact on the target
        noise_level: float
            Standard deviation of noise to add to this feature (default is 0), higher values mean more noise
        depends_on: str
            The name of another feature this feature depends on (optional)
        correlation_factor: float
            The factor by which this feature correlates with the `depends_on` feature (default is 0, no correlation)
        """
        if depends_on and depends_on in self.features:
            # Make this feature dependent on another feature with some random noise
            values = correlation_factor * self.features[depends_on] + generator(self.n_samples) * (1 - correlation_factor)
        else:
            values = generator(self.n_samples)

        # check if the feature is categorical
        if isinstance(values[0], str):
            # encode the values
            if name not in self.ordinal_encoders:
                self.ordinal_encoders[name] = OrdinalEncoder()
                values = self.ordinal_encoders[name].fit_transform(np.array(values).reshape(-1, 1)).reshape(-1)
            else:
                encoder = self.ordinal_encoders[name]
                values = encoder.transform(np.array(values).reshape(-1, 1)).reshape(-1)
        
        self.features[name] = values
        self.coefficients[name] = coefficient
        self.noise[name] = noise_level

    def build_target(self, target_name='target', bias=0, normalize=True, noise_std=10, target_range=(1, 10)):
      """
      Build a target variable based on the added features.

      Parameters:
      target_name: str
          The name of the target variable (default 'target')
      bias: float
          Constant bias to add to the linear combination
      normalize: bool
          Whether to normalize the target to a specified range
      noise_std: float
          Standard deviation of additional noise to add to the target
      target_range: tuple
          Min and max range for the target normalization

      Returns:
      DataFrame
          A DataFrame containing the features and the target variable
      """
      # Start with bias
      target = np.full(self.n_samples, bias, dtype=np.float64)
      
      # Add the weighted features and noise
      for feature in self.features:
          self.features[feature] = self.features[feature].astype(np.float64)  # Ensure feature is float64
          target += self.coefficients[feature] * self.features[feature]
          target += np.random.normal(0, self.noise[feature], self.n_samples).astype(np.float64)
      
      # Add some final noise to the target
      target += np.random.normal(0, noise_std, self.n_samples).astype(np.float64)
      
      # Optionally normalize the target
      if normalize:
          min_val, max_val = target_range
          target = np.clip(target, min_val, max_val)
      
      # Add target to features
      self.features[target_name] = target

      # replace all the encoded values with the original values
      for feature in self.ordinal_encoders:
          encoder = self.ordinal_encoders[feature]
          self.features[feature] = encoder.inverse_transform(np.array(self.features[feature]).reshape(-1, 1)).reshape(-1)
      
      # Return as a DataFrame
      return pd.DataFrame(self.features)
