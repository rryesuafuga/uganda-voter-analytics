import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Mock Streamlit to avoid import errors during testing
sys.modules['streamlit'] = MagicMock()

# Now we can import our functions
from app import (
    generate_synthetic_voters,
    initialize_causal_model,
    simulate_multi_agent_voting,
    generate_real_time_sentiment,
    train_voter_turnout_model
)

class TestVoterGeneration:
    """Test synthetic voter generation"""
    
    def test_generate_synthetic_voters_count(self):
        """Test that correct number of voters is generated"""
        n_voters = 100
        voters_df = generate_synthetic_voters(n_voters)
        assert len(voters_df) == n_voters
    
    def test_generate_synthetic_voters_columns(self):
        """Test that all required columns are present"""
        voters_df = generate_synthetic_voters(100)
        required_columns = [
            'voter_id', 'age', 'district', 'gender', 'education',
            'employment', 'mobile_user', 'social_media_user',
            'registered_2021', 'voted_2021', 'turnout_probability'
        ]
        for col in required_columns:
            assert col in voters_df.columns
    
    def test_age_distribution(self):
        """Test that 78% of voters are under 30"""
        voters_df = generate_synthetic_voters(10000)
        youth_percentage = (voters_df['age'] < 30).sum() / len(voters_df)
        assert 0.75 <= youth_percentage <= 0.81  # Allow some variance
    
    def test_turnout_probability_bounds(self):
        """Test that turnout probability is between 0 and 1"""
        voters_df = generate_synthetic_voters(1000)
        assert voters_df['turnout_probability'].min() >= 0
        assert voters_df['turnout_probability'].max() <= 1
    
    def test_district_distribution(self):
        """Test that all districts are represented"""
        voters_df = generate_synthetic_voters(5000)
        expected_districts = [
            'Kampala', 'Wakiso', 'Mukono', 'Jinja', 'Gulu', 'Lira',
            'Mbale', 'Mbarara', 'Kabale', 'Fort Portal', 'Arua', 'Soroti', 'Tororo'
        ]
        assert set(expected_districts).issubset(set(voters_df['district'].unique()))

class TestCausalModel:
    """Test causal inference model"""
    
    def test_causal_model_initialization(self):
        """Test that causal model initializes correctly"""
        model = initialize_causal_model()
        assert hasattr(model, 'treatment_effects')
        assert hasattr(model, 'estimate_effect')
    
    def test_treatment_effects(self):
        """Test that all treatment effects are defined"""
        model = initialize_causal_model()
        expected_treatments = [
            'social_media_campaign', 'youth_mobilization',
            'economic_messaging', 'peer_influence'
        ]
        for treatment in expected_treatments:
            assert treatment in model.treatment_effects
    
    def test_effect_estimation_youth(self):
        """Test that youth have higher treatment effects"""
        model = initialize_causal_model()
        
        # Test with youth
        youth_effect = model.estimate_effect('social_media_campaign', {'age': 25})
        adult_effect = model.estimate_effect('social_media_campaign', {'age': 45})
        
        assert youth_effect > adult_effect
    
    def test_effect_estimation_urban(self):
        """Test that urban areas have higher effects"""
        model = initialize_causal_model()
        
        urban_effect = model.estimate_effect('social_media_campaign', {'urban': True})
        rural_effect = model.estimate_effect('social_media_campaign', {'urban': False})
        
        assert urban_effect > rural_effect

class TestMultiAgentSimulation:
    """Test multi-agent voting simulation"""
    
    def test_simulation_output_shape(self):
        """Test that simulation returns correct output shape"""
        voters_df = generate_synthetic_voters(100)
        history, G = simulate_multi_agent_voting(voters_df, n_iterations=3)
        
        assert len(history) == 4  # Initial + 3 iterations
        assert len(history[0]) == len(voters_df)
    
    def test_network_properties(self):
        """Test that network has expected properties"""
        voters_df = generate_synthetic_voters(100)
        history, G = simulate_multi_agent_voting(voters_df, n_iterations=3)
        
        assert G.number_of_nodes() == len(voters_df)
        assert G.number_of_edges() > 0
        
        # Test that it's a connected graph (or mostly connected)
        largest_cc = max(nx.connected_components(G), key=len)
        assert len(largest_cc) > 0.8 * len(voters_df)
    
    def test_voting_intentions_binary(self):
        """Test that voting intentions are binary (0 or 1)"""
        voters_df = generate_synthetic_voters(100)
        history, G = simulate_multi_agent_voting(voters_df, n_iterations=3)
        
        for iteration in history:
            assert set(iteration).issubset({0, 1})

class TestSentimentGeneration:
    """Test real-time sentiment generation"""
    
    def test_sentiment_data_shape(self):
        """Test that sentiment data has correct shape"""
        sentiment_data = generate_real_time_sentiment()
        
        assert len(sentiment_data) == 8760  # Hours in a year
        assert set(sentiment_data.columns) == {'timestamp', 'positive', 'negative', 'neutral'}
    
    def test_sentiment_normalization(self):
        """Test that sentiments sum to 1"""
        sentiment_data = generate_real_time_sentiment()
        
        sums = sentiment_data[['positive', 'negative', 'neutral']].sum(axis=1)
        np.testing.assert_array_almost_equal(sums, 1.0, decimal=5)
    
    def test_sentiment_bounds(self):
        """Test that all sentiments are between 0 and 1"""
        sentiment_data = generate_real_time_sentiment()
        
        for col in ['positive', 'negative', 'neutral']:
            assert sentiment_data[col].min() >= 0
            assert sentiment_data[col].max() <= 1

class TestModelTraining:
    """Test ML model training"""
    
    def test_model_training_success(self):
        """Test that model trains successfully"""
        voters_df = generate_synthetic_voters(1000)
        model, explainer, X_test, shap_values = train_voter_turnout_model(voters_df)
        
        assert model is not None
        assert explainer is not None
        assert len(X_test) > 0
        assert shap_values is not None
    
    def test_model_accuracy(self):
        """Test that model achieves reasonable accuracy"""
        voters_df = generate_synthetic_voters(5000)
        model, explainer, X_test, shap_values = train_voter_turnout_model(voters_df)
        
        # Make predictions
        y_test = (voters_df.iloc[X_test.index]['turnout_probability'] > 0.5).astype(int)
        y_pred = model.predict(X_test)
        
        accuracy = (y_pred == y_test).mean()
        assert accuracy > 0.7  # Should achieve at least 70% accuracy

class TestEdgeCases:
    """Test edge cases and error handling"""
    
    def test_empty_voter_generation(self):
        """Test handling of zero voters"""
        with pytest.raises(ValueError):
            generate_synthetic_voters(0)
    
    def test_negative_voters(self):
        """Test handling of negative voter count"""
        with pytest.raises(ValueError):
            generate_synthetic_voters(-10)
    
    def test_single_voter_simulation(self):
        """Test simulation with single voter"""
        voters_df = generate_synthetic_voters(1)
        history, G = simulate_multi_agent_voting(voters_df, n_iterations=1)
        
        assert len(history) == 2
        assert G.number_of_nodes() == 1

# Performance tests
class TestPerformance:
    """Test performance requirements"""
    
    def test_voter_generation_speed(self):
        """Test that voter generation is fast enough"""
        import time
        
        start = time.time()
        generate_synthetic_voters(10000)
        end = time.time()
        
        assert end - start < 1.0  # Should take less than 1 second
    
    def test_model_training_speed(self):
        """Test that model training is fast enough"""
        import time
        
        voters_df = generate_synthetic_voters(5000)
        
        start = time.time()
        train_voter_turnout_model(voters_df)
        end = time.time()
        
        assert end - start < 5.0  # Should take less than 5 seconds

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
