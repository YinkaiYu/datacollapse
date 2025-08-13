import numpy as np
import pytest
from datacollapse import fit_data_collapse, fit_data_collapse_fse, fit_data_collapse_fse_robust, collapse_transform

def generate_test_data():
    """Generate synthetic test data for collapse analysis."""
    np.random.seed(42)
    L_values = [7, 9, 11, 13]
    U_range = np.linspace(8.3, 9.0, 20)
    U_c_true = 8.65
    a_true = 1.1
    
    all_data = []
    for L in L_values:
        x_scaled = (U_range - U_c_true) * (L ** a_true)
        Y_universal = 0.6 + 0.12 * x_scaled + 0.08 * (x_scaled**2)
        Y_fse = Y_universal * (1 + 0.4 * L**(-0.8))
        noise = 0.02 * np.random.randn(len(U_range))
        Y_noisy = Y_fse + noise
        
        for U, Y in zip(U_range, Y_noisy):
            all_data.append([L, U, Y])
    
    data = np.array(all_data)
    err = 0.03 * np.ones(len(data))
    return data, err

def test_fit_and_transform_runs():
    """Test that basic fitting and transform execute without errors."""
    data, err = generate_test_data()
    
    # Test basic fitting
    (params, errs) = fit_data_collapse(data, err, U_c_0=8.6, a_0=1.0,
                                      n_knots=8, lam=1e-3, n_boot=3,
                                      bounds=((8.3, 8.9), (0.5, 2.0)))
    
    assert len(params) == 2
    assert len(errs) == 2
    assert all(np.isfinite(params))
    assert all(np.isfinite(errs))
    
    # Test transform
    x, Yc = collapse_transform(data, params)
    assert len(x) == len(data)
    assert len(Yc) == len(data)
    assert all(np.isfinite(x))
    assert all(np.isfinite(Yc))

def test_fse_fitting():
    """Test finite-size correction fitting."""
    data, err = generate_test_data()
    
    (params, errs) = fit_data_collapse_fse(data, err, 
                                           U_c_0=8.6, a_0=1.0, b_0=0.5, c_0=-0.8,
                                           n_knots=8, lam=1e-3, n_boot=3,
                                           bounds=((8.3, 8.9), (0.5, 2.0), (0.1, 1.0), (-1.5, -0.1)))
    
    assert len(params) == 4
    assert len(errs) == 4
    assert all(np.isfinite(params))
    assert all(np.isfinite(errs))
    # Check c < 0 constraint is satisfied
    assert params[3] < 0

def test_fse_robust():
    """Test robust finite-size correction fitting."""
    data, err = generate_test_data()
    
    b_grid = np.linspace(0.2, 0.8, 4)
    c_grid = np.linspace(-1.2, -0.6, 4)
    
    (params, errs) = fit_data_collapse_fse_robust(data, err,
                                                  U_c_0=8.6, a_0=1.0,
                                                  b_grid=b_grid, c_grid=c_grid,
                                                  n_knots=8, lam=1e-3, n_boot=3,
                                                  bounds_Ua=((8.3, 8.9), (0.5, 2.0)))
    
    assert len(params) == 4
    assert len(errs) == 4
    assert all(np.isfinite(params))
    assert all(np.isfinite(errs))
    assert params[3] < 0  # c should be negative

def test_random_restarts():
    """Test multi-start random restarts functionality."""
    data, err = generate_test_data()
    
    # Fit with random restarts
    (params1, errs1) = fit_data_collapse(data, err, U_c_0=8.6, a_0=1.0,
                                        n_knots=8, lam=1e-3, n_boot=3,
                                        random_restarts=3, random_state=42)
    
    # Fit without random restarts
    (params2, errs2) = fit_data_collapse(data, err, U_c_0=8.6, a_0=1.0,
                                        n_knots=8, lam=1e-3, n_boot=3,
                                        random_restarts=0, random_state=42)
    
    # Both should give finite results
    assert all(np.isfinite(params1))
    assert all(np.isfinite(params2))

def test_normalization():
    """Test finite-size correction with normalization."""
    data, err = generate_test_data()
    
    (params, errs) = fit_data_collapse_fse(data, err, 
                                           U_c_0=8.6, a_0=1.0, b_0=0.5, c_0=-0.8,
                                           n_knots=8, lam=1e-3, n_boot=3,
                                           normalize=True, L_ref="geom")
    
    # Test transform with normalization
    x, Yc = collapse_transform(data, params, normalize=True, L_ref="geom")
    
    assert all(np.isfinite(x))
    assert all(np.isfinite(Yc))

def test_different_optimizers():
    """Test different optimizer options."""
    data, err = generate_test_data()
    
    optimizers = ["NM", "Powell", "NM_then_Powell"]
    
    for opt in optimizers:
        (params, errs) = fit_data_collapse(data, err, U_c_0=8.6, a_0=1.0,
                                          n_knots=8, lam=1e-3, n_boot=3,
                                          optimizer=opt, maxiter=1000)
        
        assert all(np.isfinite(params))
        assert all(np.isfinite(errs))

def test_error_handling():
    """Test error handling for invalid inputs."""
    data, err = generate_test_data()
    
    # Test with invalid bounds (should not crash)
    try:
        (params, errs) = fit_data_collapse(data, err, U_c_0=8.6, a_0=1.0,
                                          n_knots=8, lam=1e-3, n_boot=3,
                                          bounds=((8.9, 8.3), (0.5, 2.0)))  # Invalid: low > high
        # Should either work or fail gracefully
    except Exception:
        pass  # Expected for invalid bounds
    
    # Test with zero errors (should handle gracefully)
    zero_err = np.zeros_like(err)
    (params, errs) = fit_data_collapse(data, zero_err, U_c_0=8.6, a_0=1.0,
                                      n_knots=8, lam=1e-3, n_boot=3)
    assert all(np.isfinite(params))

def test_reproducibility():
    """Test that results are reproducible with same random_state."""
    data, err = generate_test_data()
    
    # Run twice with same random state
    (params1, errs1) = fit_data_collapse(data, err, U_c_0=8.6, a_0=1.0,
                                        n_knots=8, lam=1e-3, n_boot=5,
                                        random_state=123)
    
    (params2, errs2) = fit_data_collapse(data, err, U_c_0=8.6, a_0=1.0,
                                        n_knots=8, lam=1e-3, n_boot=5,
                                        random_state=123)
    
    # Results should be identical
    np.testing.assert_array_almost_equal(params1, params2, decimal=10)
    np.testing.assert_array_almost_equal(errs1, errs2, decimal=10)

def test_data_formats():
    """Test different input data formats."""
    data, err = generate_test_data()
    
    # Test with None error
    (params1, errs1) = fit_data_collapse(data, None, U_c_0=8.6, a_0=1.0,
                                        n_knots=8, lam=1e-3, n_boot=3)
    
    # Test with list inputs
    data_list = data.tolist()
    err_list = err.tolist()
    (params2, errs2) = fit_data_collapse(data_list, err_list, U_c_0=8.6, a_0=1.0,
                                        n_knots=8, lam=1e-3, n_boot=3)
    
    assert all(np.isfinite(params1))
    assert all(np.isfinite(params2))

def test_edge_cases():
    """Test edge cases and boundary conditions."""
    data, err = generate_test_data()
    
    # Test with minimal knots
    (params, errs) = fit_data_collapse(data, err, U_c_0=8.6, a_0=1.0,
                                      n_knots=3, lam=1e-3, n_boot=3)
    assert all(np.isfinite(params))
    
    # Test with high smoothing
    (params, errs) = fit_data_collapse(data, err, U_c_0=8.6, a_0=1.0,
                                      n_knots=8, lam=1e-1, n_boot=3)
    assert all(np.isfinite(params))
    
    # Test with no bootstrap
    (params, errs) = fit_data_collapse(data, err, U_c_0=8.6, a_0=1.0,
                                      n_knots=8, lam=1e-3, n_boot=0)
    assert all(np.isfinite(params))
    # Errors should be NaN when no bootstrap
    assert all(np.isnan(errs))

if __name__ == "__main__":
    pytest.main([__file__]) 