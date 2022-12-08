import functions
import numpy as np
import os
import random
import pytest
import os.path


def test_generate_patterns():
    # Catching exceptions
    with pytest.raises(ValueError, match="num_patterns must be >= 0"):
        functions.generate_patterns(-2, 5)
    with pytest.raises(ValueError, match="pattern_size must be >= 0"):
        functions.generate_patterns(2, -5)
    with pytest.raises(ValueError, match="num_patterns must be an integer"):
        functions.generate_patterns(1.5, 5)
    with pytest.raises(ValueError, match="pattern_size must be an integer"):
        functions.generate_patterns(2, 1.5)

    P = functions.generate_patterns(2, 3)
    # Testing for correct shape
    assert np.shape(P) == (2, 3)
    for ele in P.flatten():
        assert (ele == -1 or ele == 1)    

def test_perturb_pattern():
    with pytest.raises(TypeError, match="pattern must be an np.array"):
        functions.perturb_pattern([1, -1, -1,  1,  1], 3)
    with pytest.raises(ValueError, match="num_perturb must be >= 0"):
        functions.perturb_pattern(np.array([1, -1, -1,  1,  1]), -1)
    with pytest.raises(ValueError, match="num_perturb must be an integer"):
        functions.perturb_pattern(np.array([1, -1, -1,  1,  1]), 1.5)

    # Testing for perturbation
    p = np.array([1, -1, -1])
    perturbed_pattern = functions.perturb_pattern(p, 1)
    assert not (np.allclose(p, perturbed_pattern))
    # Testing for correct shape
    assert np.shape(p) == (3,)


def test_pattern_match():
    with pytest.raises(TypeError, match="memorized_patterns must be an np.array"):
        functions.pattern_match([1, -1, 1, -1], np.array([1, -1]))
    with pytest.raises(TypeError, match="pattern must be an np.array"):
        functions.pattern_match(np.array([[1, -1], [1, -1]]), [1, -1])

    mem = np.array([[1, -1], [1, -1]])
    p = np.array([1, -1])
    assert functions.pattern_match(mem, p) == 0


def test_hebbian_weights():
    with pytest.raises(TypeError, match="patterns must be an np.array"):
        functions.hebbian_weights([ 1, -1, -1,  1,  1, 1,  1, -1,  1, -1])

    # Testing for correct size
    W = functions.hebbian_weights(np.array([[ 1, -1, -1,  1,  1,], [ 1,  1, -1,  1, -1]]))
    assert np.shape(W) == (5, 5)
    # Testing for correct range
    for ele in W.flatten():
        assert not (ele < -1 or ele > 1)
    # Testing for symmetry
    assert np.allclose(W.flatten(), W.T.flatten())
    # Testing for zeros in diagonal
    assert np.allclose(W.diagonal(), np.zeros((1, 5)))


def test_storkey_weights():
    with pytest.raises(TypeError, match="patterns must be an np.array"):
        functions.storkey_weights([1, -1, -1, 1])

    # Testing for correct size
    W = functions.storkey_weights(np.array([[ 1, -1, -1,  1,  1,], [ 1,  1, -1,  1, -1]]))
    assert np.shape(W) == (5, 5)
    # Testing for symmetry
    assert np.allclose(W.flatten(), W.T.flatten())
     

def test_weights():
    # Testing for Hebbian weights
    P = np.array([[1, 1, -1, -1], [1, 1, -1, 1], [-1, 1, -1, 1]])
    expected_W = np.array([[0, 0.33333333, -0.33333333, -0.33333333],
                           [0.33333333, 0, -1, 0.33333333],
                           [-0.33333333, -1, 0, -0.33333333],
                           [-0.33333333, 0.33333333, -0.33333333, 0]])
    assert np.allclose(functions.hebbian_weights(P), expected_W)
    # Testing for Storkey weights
    P = np.array([[1, 1, -1, -1], [1, 1, -1, 1], [-1, 1, -1, 1]])
    expected_W = np.array([[1.125, 0.25, -0.25, -0.5],
                           [0.25, 0.625, -1, 0.25],
                           [-0.25, -1, 0.625, -0.25],
                           [-0.5, 0.25, -0.25, 1.125]])
    assert np.allclose(functions.storkey_weights(P), expected_W)


def test_update():
    with pytest.raises(TypeError, match="state must be an np.array"):
        functions.update([1, -1], np.array([[0, 0.5], [0.5, 0]]))
    with pytest.raises(TypeError, match="weights must be an np.array"):
        functions.update(np.array([1, -1]), [[0, 0.5], [0.5, 0]])

    # Testing for perturbation
    p = np.array([1, 1, -1, -1])
    W = np.array([[0, 0.33333333, -0.33333333, -0.33333333],
                  [0.33333333, 0, -1, 0.33333333],
                  [-0.33333333, -1, 0, -0.33333333],
                  [-0.33333333, 0.33333333, -0.33333333, 0]])
    updated_p = functions.update(p, W)
    assert not (np.allclose(p, updated_p))
    # Testing each value is either 1 or -1
    for ele in updated_p:
        assert (ele == -1 or ele == 1)


def test_update_async():
    with pytest.raises(TypeError, match="state must be an np.array"):
        functions.update_async([1, -1], np.array([[0, 0.5], [0.5, 0]]))
    with pytest.raises(TypeError, match="weights must be an np.array"):
        functions.update_async(np.array([1, -1]), [0, 0.5, 0.5, 0])

    # Testing for perturbation
    p = np.array([1, 1, -1, -1])
    W = np.array([[0, 0.33333333, -0.33333333, -0.33333333],
                  [0.33333333, 0, -1, 0.33333333],
                  [-0.33333333, -1, 0, -0.33333333],
                  [-0.33333333, 0.33333333, -0.33333333, 0]])
    updated_p = functions.update(p, W)
    assert not (np.allclose(p, updated_p))
    # Testing each value is either 1 or -1
    for ele in updated_p:
        assert (ele == -1 or ele == 1)
    

def test_dynamics():
    with pytest.raises(TypeError, match="state must be an np.array"):
        functions.dynamics([1, -1], np.array([[0, 0.5], [0.5, 0]]), 3)
    with pytest.raises(TypeError, match="weights must be an np.array"):
        functions.dynamics(np.array([1, -1]), [0, 0.5, 0.5, 0], 3)
    with pytest.raises(ValueError, match="max_iter must be >= 0"):
        functions.dynamics(np.array([1, -1]), np.array([[0, 0.5], [0.5, 0]]), -1)
    with pytest.raises(ValueError, match="max_iter must be an integer"):
        functions.dynamics(np.array([1, -1]), np.array([[0, 0.5], [0.5, 0]]), 1.5)

    # Testing for updates
    state = np.array([1, 1, -1, -1])
    W = np.array([[0, 0.33333333, -0.33333333, -0.33333333],
                  [0.33333333, 0, -1, 0.33333333],
                  [-0.33333333, -1, 0, -0.33333333],
                  [-0.33333333, 0.33333333, -0.33333333, 0]])
    history = functions.dynamics(state, W, 20)
    assert not (np.allclose(history[len(history) - 1], state))


def test_dynamics_async():
    with pytest.raises(TypeError, match="state must be an np.array"):
        functions.dynamics_async([1, -1], np.array([[0, 0.5], [0.5, 0]]), 3, 2)
    with pytest.raises(TypeError, match="weights must be an np.array"):
        functions.dynamics_async(np.array([1, -1]), [0, 0.5, 0.5, 0], 3, 2)
    with pytest.raises(ValueError, match="max_iter must be >= 0"):
        functions.dynamics_async(np.array([1, -1]), np.array([[0, 0.5], [0.5, 0]]), -1, 2)
    with pytest.raises(ValueError, match="max_iter must be an integer"):
        functions.dynamics_async(np.array([1, -1]), np.array([[0, 0.5], [0.5, 0]]), 1.5, 2)
    with pytest.raises(ValueError, match="convergence_num_iter must be >= 0"):
        functions.dynamics_async(np.array([1, -1]), np.array([[0, 0.5], [0.5, 0]]), 3, -1)
    with pytest.raises(ValueError, match="convergence_num_iter must be an integer"):
        functions.dynamics_async(np.array([1, -1]), np.array([[0, 0.5], [0.5, 0]]), 3, 1.5)

    # Testing for updates
    state = np.array([1, 1, -1, -1])
    perturbed_state = functions.perturb_pattern(state, 1)
    W = np.array([[0, 0.33333333, -0.33333333, -0.33333333],
                  [0.33333333, 0, -1, 0.33333333],
                  [-0.33333333, -1, 0, -0.33333333],
                  [-0.33333333, 0.33333333, -0.33333333, 0]])
    history = functions.dynamics_async(state, W, 20, 10)
    assert len(history) != 0


def test_evolution():
    # Testing for convergence
    P = np.array([[1, 1, -1, -1], [1, 1, -1, 1], [-1, 1, -1, 1]])
    W = np.array([[0, 0.33333333, -0.33333333, -0.33333333],
                  [0.33333333, 0, -1, 0.33333333],
                  [-0.33333333, -1, 0, -0.33333333],
                  [-0.33333333, 0.33333333, -0.33333333, 0]])
    history = functions.dynamics(P[0], W, 20)
    assert functions.pattern_match(P, history[len(history) - 1])

    # Testing that function is non-increasing (negative difference between two successive values)
    energies = functions.compute_energies(history, W)[0]
    diff = (np.diff(energies) <= 0)
    assert all(diff)


def test_energy():
    with pytest.raises(TypeError, match="state must be an np.array"):
        functions.energy([1, -1], np.array([[0., 0.5], [0.5, 0.]]))
    with pytest.raises(TypeError, match="weights must be an np.array"):
        functions.energy(np.array([1, -1]), [0., 0.5, 0.5, 0.])

    E = functions.energy(np.array([1, -1]), np.array([[0., 0.5], [0.5, 0.]]))
    assert type(E) == np.float64


def test_compute_energies():
    with pytest.raises(TypeError, match="history_of_states must be a list"):
        functions.compute_energies(1, np.array([[0., 0.5], [0.5, 0.]]))
    with pytest.raises(TypeError, match="weights must be an np.array"):
        functions.compute_energies([[1, -1], [-1 ,1]], [0., 0.5, 0.5, 0.])

    history = [np.array([1, -1, -1, 1]), np.array([1, 1, -1, 1]), np.array([1, 1, 1, 1])]
    W = np.array([[0, 0.33333333, -0.33333333, -0.33333333],
                  [0.33333333, 0, -1, 0.33333333],
                  [-0.33333333, -1, 0, -0.33333333],
                  [-0.33333333, 0.33333333, -0.33333333, 0]])
    energies = functions.compute_energies(history, W)[0]
    assert np.shape(energies) == (3,)


    history = [np.array([1, -1, -1, 1]), np.array([1, 1, -1, 1]), np.array([1, 1, 1, 1]),
               np.array([1, -1, -1, 1]), np.array([1, 1, -1, 1]), np.array([1, 1, 1, 1]),
               np.array([1, -1, -1, 1]), np.array([1, 1, -1, 1]), np.array([1, 1, 1, 1]),
               np.array([1, -1, -1, 1]), np.array([1, 1, -1, 1]), np.array([1, 1, 1, 1])]
    W = np.array([[0, 0.33333333, -0.33333333, -0.33333333],
                  [0.33333333, 0, -1, 0.33333333],
                  [-0.33333333, -1, 0, -0.33333333],
                  [-0.33333333, 0.33333333, -0.33333333, 0]])
    energies = functions.compute_energies(history, W)[0]
    assert np.shape(energies) == (12,)


def test_generate_checkerboard():
    # Testing for correct size
    checkerboard = functions.generate_checkerboard()
    assert np.shape(checkerboard) == (50, 50)
    # Testing for correct values
    for ele in checkerboard.flatten():
        assert (ele == -1 or ele == 1)


def test_save_video(): 
    with pytest.raises(ValueError, match="state_list cannot be empty"):
        functions.save_video([], os.getcwd() + "/checkerboard_async.mp4")
    with pytest.raises(ValueError, match="state_list must be a list"):
        functions.save_video(np.array([[1, -1], [-1, 1]]), os.getcwd() + "/checkerboard_async.mp4")
    with pytest.raises(ValueError, match="out_path must be a string"):
        functions.save_video([np.array([[1, -1], [-1, 1]])], 1)

    video = [[[1, -1], [1, 1]], [[1, -1], [-1, -1]]]
    functions.save_video(video, os.getcwd() + "/test_save_video.mp4")
    assert os.path.isfile(os.getcwd() + "/test_save_video.mp4")

    video2 = []
    for i in range(1001): 
        video2.append([[1, -1], [1, 1]])
    functions.save_video(video2, os.getcwd() + "/test_save_video2.mp4")
    assert os.path.isfile(os.getcwd() + "/test_save_video2.mp4")


def test_reshape_history():
    memorized_patterns = functions.generate_patterns(50, 2500)
    checkerboard_pattern = functions.generate_checkerboard()
    flattened_pattern = checkerboard_pattern.flatten()
    # Replace one of the patterns of memorized_patterns with flattened checkerboard pattern
    i = random.randint(0, 49)
    memorized_patterns[i] = flattened_pattern
    # Generate Hebbian weights matrix with memorized patterns containing the checkerboard pattern
    chess_hebbian = functions.hebbian_weights(memorized_patterns)
    # Perturbation of 1000 elements in checkerboard pattern
    perturbed_checkerboard = functions.perturb_pattern(flattened_pattern, 1000)
    history = functions.dynamics(perturbed_checkerboard, chess_hebbian, 20)
    functions.reshape_history(history)
    for ele in history:
        assert np.shape(ele) == (50, 50)
