'''
Abstract base class for data loading and preprocessing
'''
from abc import ABC, abstractmethod
import pandas as pd

class BaseDataLoader(ABC):
    '''
    Abstract base class for loading and preprocessing training data using nfelodcm
    '''
    def load_data(self) -> pd.DataFrame:
        '''
        Load data using nfelodcm
        
        Returns:
        * pd.DataFrame: loaded and processed data
        '''
        ##  Fetch raw data using nfelodcm
        raw_data = self._fetch_from_source()
        ##  Apply transformations
        transformed = self.transform(raw_data)
        ##  Create features
        featured = self.create_features(transformed)
        return featured
    
    @abstractmethod
    def _fetch_from_source(self) -> pd.DataFrame:
        '''
        Fetch data from nfelodcm (implemented by subclasses)
        
        Returns:
        * pd.DataFrame: raw data
        '''
        raise NotImplementedError
    
    @abstractmethod
    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        '''
        Apply model-specific transformations to raw data
        
        Params:
        * data: pd.DataFrame - raw data
        
        Returns:
        * pd.DataFrame: transformed data
        '''
        raise NotImplementedError
    
    @abstractmethod
    def create_features(self, data: pd.DataFrame) -> pd.DataFrame:
        '''
        Create model-specific features
        
        Params:
        * data: pd.DataFrame - transformed data
        
        Returns:
        * pd.DataFrame: data with features
        '''
        raise NotImplementedError

