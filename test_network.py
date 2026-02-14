from network import sigmoid, sigmoid_deriv

def test_math_functions():
    # sigmoid test function
    assert sigmoid(0) == 0.5
    assert sigmoid(10) > 0.99
    assert sigmoid(-10) < 0.01
    
    # sigmoid derivation test function
    assert sigmoid_deriv(0.0) == 0.25
    # sigmoid derivation shape test
    assert sigmoid_deriv(10.0) < sigmoid_deriv(0.0)
    assert sigmoid_deriv(-10.0) < sigmoid_deriv(0.0)