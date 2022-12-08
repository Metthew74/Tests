import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib.animation as pltanim
import math
import os


def generate_patterns(num_patterns, pattern_size):
    """ Generates the patterns to be memorized, returns a 2-dimensional array of num_patterns random binary patterns of size pattern_size.
    
    Parameters
    ----------
    num_patterns (int) : number of patterns to be memorized
    pattern_size (int) : size of each pattern

    Returns
    ----------
    2D array of num_patterns random binary patterns each of size pattern_size
        P
    
    Example
    ---------
    >>> generate_patterns(2, 5)
    array([[-1, -1, -1, -1, 1],
           [-1, -1, 1, 1, -1]])
    """

    # Testing input values
    if not num_patterns >= 0:
        raise ValueError("num_patterns must be >= 0")
    if not pattern_size >= 0:
        raise ValueError("pattern_size must be >= 0")
    if math.floor(num_patterns) != num_patterns:
        raise ValueError("num_patterns must be an integer")
    if math.floor(pattern_size) != pattern_size:
        raise ValueError("pattern_size must be an integer")

    P = 2 * np.random.binomial(1, 0.5, (num_patterns, pattern_size)) - 1

    return P


def perturb_pattern(pattern, num_perturb):
    """ Perturbs a given pattern by sampling num_perturb elements of input pattern at random and changing their sign. Returns the perturbed pattern.
    
    Parameters
    ----------
    pattern (1D array) : pattern of firing states of which num_perturb elements will be modified

    Returns
    ----------
    1D array, perturbed pattern
        new_pattern
    
    Example
    ---------
    >>> perturb_pattern(np.array([1, -1, -1,  1,  1]), 3)
    np.array([-1, -1, 1,  1,  -1])
    """
    # Testing for correct input
    if type(pattern) != np.ndarray:
        raise TypeError("pattern must be an np.array")
    if not num_perturb >= 0:
        raise ValueError("num_perturb must be >= 0")
    if math.floor(num_perturb) != num_perturb:
        raise ValueError("num_perturb must be an integer")

    new_pattern = pattern.copy()
    indexes = set()
    # Assures that we modify num_perturb different elements 
    while len(indexes) < num_perturb :
        indexes.add(random.randint(0, len(pattern) - 1))
    for idx in indexes : 
        new_pattern[idx] = - new_pattern[idx] 

    return new_pattern


def pattern_match(memorized_patterns, pattern):
    """ Matches a given pattern with the corresponding memorized one. Returns index of row of matching pattern if a match is found, otherwise returns None.
    
    Parameters
    ----------
    memorized_patterns (2D array) : memorized patterns of firing states
    pattern (1D array) : pattern of firing stated to be compared with each row of memorized_patterns

    Returns
    ----------
    int, index
        i

    Example
    ---------
    >>> pattern_match(np.array([[1, -1], [1, -1]]), np.array([1, -1]))
    0
    """
    # Testing for correct input
    if type(memorized_patterns) != np.ndarray:
        raise TypeError("memorized_patterns must be an np.array")
    if type(pattern) != np.ndarray:
        raise TypeError("pattern must be an np.array")

    for i in range(np.shape(memorized_patterns)[0]):
        if np.allclose(memorized_patterns[i], pattern):
            return i


def hebbian_weights(patterns):
    """ Applies the hebbian learning rule to given patterns to create and return the weights matrix.

    Parameters
    ----------
    patterns (2D array) : patterns used to compute the hebbian weights matrix

    Returns
    ----------
    2D array, hebbian weights matrix
        weights_matrix
    
    Example
    ---------
    >>> hebbian_weights(np.array([[ 1, -1, -1,  1,  1,], [ 1,  1, -1,  1, -1]]))
    array([[ 0.,  0., -1.,  1.,  0.],
           [ 0.,  0.,  0.,  0., -1.],
           [-1.,  0.,  0., -1.,  0.],
           [ 1.,  0., -1.,  0.,  0.],
           [ 0., -1.,  0.,  0.,  0.]])
    """
    # Testing for correct type of input
    if type(patterns) != np.ndarray:
        raise TypeError("patterns must be an np.array")

    # Number of lines of patterns
    num_patterns = np.shape(patterns)[0]
    # Number of columns of patterns
    pattern_size = np.shape(patterns)[1]

    # Compute the hebbian weights matrix
    weights_matrix = np.zeros((pattern_size, pattern_size))
    pattern_sum = 0
    for mu in range (num_patterns) : 
        pattern_sum += np.outer(patterns[mu], patterns[mu])
    weights_matrix = 1/num_patterns * pattern_sum

    # Fill diagonal of weights matrix with zeros, as there are no self-connections
    np.fill_diagonal(weights_matrix, 0)

    return weights_matrix


def storkey_weights(patterns):
    """ Applies the Storkey learning rule on some given patterns to create and return the Storkey weights matrix.

    Parameters
    ----------
    patterns (2D array) : patterns of firing states used to create the Storkey weights matrix

    Returns
    ----------
    2D array, Storkey weights matrix
        W

    Example
    ---------
    >>> functions.storkey_weights(np.array([[1, -1], [-1, 1]]))
    array([[ 0.5, -1. ],
           [-1. ,  0.5]])
    """
    # Testing for correct input
    if type(patterns) != np.ndarray:
        raise TypeError("patterns must be an np.array")

    # Number of lines of patterns
    num_patterns = np.shape(patterns)[0]
    # Number of columns of patterns
    pattern_size = np.shape(patterns)[1]
    
    # Initialize the Storkey weights matrix
    W = np.zeros((pattern_size, pattern_size))
    H = np.zeros((pattern_size, pattern_size))
    W_prev = W

    # Iterate over each state in patterns
    for mu in range(num_patterns):
        # Compute the H matrix
        H = np.reshape(np.dot(W_prev, patterns[mu]), (-1, 1))
        # First case to be substracted : when i = k 
        case_i_equal_k = np.reshape(np.diagonal(W_prev) * patterns[mu], (-1, 1))
        # Second case to be substracted : when j = k 
        case_j_equal_k = W_prev * patterns[mu]
        np.fill_diagonal(case_j_equal_k, 0)
        H = H - case_i_equal_k - case_j_equal_k  

        # Compute the weights matrix
        p_ij = np.outer(patterns[mu], patterns[mu])
        ph_ij = H * patterns[mu]
        ph_ji = ph_ij.T
        W = W_prev + 1/pattern_size * (p_ij - ph_ji - ph_ij)
        W_prev = W

    return W


def update(state, weights):
    """ Applies the update rule to given state (pattern) using weights matrix and returns new state.

    Parameters
    ----------
    state (1D array) : pattern of firing states at time t
    weights (2D array) : weights matrix used to compute the new state at time t + 1

    Returns
    ----------
    1D array, updated state
        new_state
    
    Example
    ---------
    >>> update(np.array([1, -1]), np.array([[0, 0.5], [0.5, 0]]))
    array([-1,  1])
    """
    # Testing for correct input
    if type(state) != np.ndarray:
        raise TypeError("state must be an np.array")
    if type(weights) != np.ndarray:
        raise TypeError("weights must be an np.array")

    # Compute dot product
    new_state = state.copy()
    Wp = np.dot(weights, new_state)

    # Apply sigma function
    for k in range(len(new_state)): 
        if Wp[k] < 0:
            new_state[k] = -1
        else:
            new_state[k] = 1

    return new_state
 

def update_async(state, weights):
    """ Applies the asynchronous update rule to a state pattern, i.e. updates the randomly chosen i-th component of given state.
    
    Parameters
    ----------
    state (1D array) : pattern of firing states at time t
    weights (2D array) : weights matrix used to compute the new state at time t + 1

    Returns
    ----------
    1D array, updated state
        new_state
    
    Example
    ---------
    >>> update_async(np.array([1, -1]), np.array([[0, 0.5], [0.5, 0]]))
    array([1, -1])
    """
    # Testing for correct input
    if type(state) != np.ndarray:
        raise TypeError("state must be an np.array")
    if type(weights) != np.ndarray:
        raise TypeError("weights must be an np.array")

    # Compute dot product
    new_state = state.copy()
    i = np.random.randint(0, len(new_state))

    # Apply sigma function
    if np.inner(new_state,weights[i,:]) < 0:
        new_state[i] = -1
    else:
        new_state[i] = 1

    return new_state


def dynamics(state, weights, max_iter):
    """ Runs the dynamical system from an initial state until convergence or until maximum number of steps is reached.
    Convergence is defined as two consecutive updates returning the same state.
    
    Parameters
    ----------
    state (1D array) : pattern of firing states
    weights (2D array) : weights matrix used to compute the new state using update function
    max_iter (int) : maximum number of iterations before convergence is reached

    Returns
    ----------
    1D array, history of all the states during the execution of the dynamical system
        history_of_states
    
    Example
    ---------
    >>> dynamics(np.array([1, -1]), np.array([[0, 0.5], [0.5, 0]]), 3)
    [array([-1,  1]), array([ 1, -1]), array([-1,  1])]
    """
    # Testing for correct input
    if type(state) != np.ndarray:
        raise TypeError("state must be an np.array")
    if type(weights) != np.ndarray:
        raise TypeError("weights must be an np.array")
    if not max_iter >= 0:
        raise ValueError("max_iter must be >= 0")
    if math.floor(max_iter) != max_iter:
        raise ValueError("max_iter must be an integer")

    history_of_states = []
    previous_update = state
    

    # Call update function on given state max_iter times or until convergence 
    for i in range(max_iter):
        current_update = update(previous_update, weights)
        history_of_states.append(current_update)
        # If two consecutive updates return the same state, convergence has been reached
        if np.allclose(previous_update, current_update):
            print(f"Convergence has occurred in {i} iterations")
            break
        previous_update = current_update

    return history_of_states


def dynamics_async(state, weights, max_iter, convergence_num_iter):
    """ Runs the dynamical system from an initial state until convergence or until a maximum number of steps is reached.
    Convergence is defined as convergence_num_iter consecutive updates returning the same state.
    
    Parameters
    ----------
    state (1D array) : pattern of firing states
    weights (2D array) : weights matrix used to compute the new state using update_async function
    max_iter (int) : maximum number of iterations before convergence is reached
    convergence_num_iter (int) : number of consecutive updates required to return the same state before convergence is reached

    Returns
    ----------
    1D array, history of all the states during the execution of the dynamical system
        history_of_states
    
    Example
    ---------
    >>> dynamics_async(np.array([1, -1]), np.array([[0, 0.5], [0.5, 0]]), 3, 2)
    [array([1, -1]), array([1, 1]), array([1, 1])]
    """
    # Testing for correct input
    if type(state) != np.ndarray:
        raise TypeError("state must be an np.array")
    if type(weights) != np.ndarray:
        raise TypeError("weights must be an np.array")
    if not max_iter >= 0:
        raise ValueError("max_iter must be >= 0")
    if math.floor(max_iter) != max_iter:
        raise ValueError("max_iter must be an integer")
    if not convergence_num_iter >= 0:
        raise ValueError("convergence_num_iter must be >= 0")
    if math.floor(convergence_num_iter) != convergence_num_iter:
        raise ValueError("convergence_num_iter must be an integer")

    history_of_states = [state]
    previous_update = state
    counter = 0

    final_state = state

    # Call update_async function on given state max_iter times or until convergence 
    for i in range(max_iter):
        current_update = update_async(previous_update, weights)
        history_of_states.append(current_update)
        # If convergence_num_iter consecutive updates return the same state, then convergence has been reached
        if np.allclose(previous_update, current_update):
            counter += 1
        else:
            counter = 0
        if counter >= convergence_num_iter:
            print(f"Convergence has occurred in {i} iterations")
            break
        previous_update = current_update
        final_state = current_update

    history_of_states.append(final_state)

    return history_of_states


def energy(state, weights):
    """ Returns the energy associated to the given state, using the weights matrix.

    Parameters
    ----------
    state (1D array) : pattern of firing states
    weights (2D array) : weights matrix used to calculate the energy

    Returns
    ----------
    float, energy corresponding to the state
        E

    Example
    ---------
    >>> energy(np.array([1, -1]), np.array([[0., 0.5], [0.5, 0.]]))
    0.5
    """
    # Testing for correct input
    if type(state) != np.ndarray:
        raise TypeError("state must be an np.array")
    if type(weights) != np.ndarray:
        raise TypeError("weights must be an np.array")

    E = -1/2 * np.sum(weights * np.outer(state, state))

    return E


def compute_energies(history_of_states, weights):
    """ Computes the energies of every state in history_of_states

    Parameters
    ----------
    history_of_states (1D array) : history of all the states during the execution of the dynamical system
    weights (2D array) : weights matrix used to compute the energies at each state in history_of_states

    Returns
    ---------
    1D array, containing sequence of energies of each of the consecutive states in history_of_states
        energies
    
    Example
    ---------
    >>> compute_energies([np.array([1, -1]), np.array([1, -1]), np.array([1, -1]), np.array([-1, 1])], np.array([np.array([0., 0.5]), np.array([0.5, 0.])]))
    (array([0.5, 0.5, 0.5, 0.5]), {0: 0.5, 1: 0.5, 2: 0.5, 3: 0.5})

    """
    # Testing for correct type of input
    if type(history_of_states) != list:
        raise TypeError("history_of_states must be a list")
    if type(weights) != np.ndarray:
        raise TypeError("weights must be an np.array")

    time_vs_energy = {}

    energies = np.zeros(len(history_of_states))
    # If history_of_states is long enough to be async, store every 1000th energy value
    if len(history_of_states) >= 10:
        for i in range(len(history_of_states)):
            if i % 1000 == 0:
                energies[i] = energy(history_of_states[i], weights)
                time_vs_energy[i] = energies[i]
    # Else, store every energy value
    else:
        for i in range(len(history_of_states)):
            energies[i] = energy(history_of_states[i], weights)
            time_vs_energy[i] = energies[i]
    return energies, time_vs_energy

def generate_checkerboard() :
    """ Generates a checkerboard pattern, where 1 is white and -1 is black, starting with a white checker in the top left corner. 

    Returns
    ----------
    2D array, checkerboard pattern
        checkerboard_pattern
    
    Example
    ---------
    >>> generate_checkerboard()
    array([[ 1.,  1.,  1., ..., -1., -1., -1.],
           [ 1.,  1.,  1., ..., -1., -1., -1.],
           [ 1.,  1.,  1., ..., -1., -1., -1.],
           ...,
           [-1., -1., -1., ...,  1.,  1.,  1.],
           [-1., -1., -1., ...,  1.,  1.,  1.],
           [-1., -1., -1., ...,  1.,  1.,  1.]])
    """
    # Create first line of white and black checkers
    white = np.ones((1, 5))
    black = - (white.copy())

    # Create first row of 10 checkers, starting with white and black respectively
    full_row_w = np.tile(np.concatenate((white, black), axis=1), 25)
    full_row_b = np.tile(np.concatenate((black, white), axis=1), 25)

    # Create the first two row of checkers
    two_rows = np.concatenate((full_row_w, full_row_b), axis=1)

    # Create and reshape the whole checkerboard 
    checkerboard_pattern = np.reshape(np.tile(two_rows, 5), (50, 50))

    return checkerboard_pattern


def save_video(state_list, out_path):
    """ Creates a video of the evolution of states contained in state_list, and saves it to out_path

    Parameters
    ----------
    state_list (1D array) : sequence of states that will be turned into frames
    out_path (path) : path where video will be saved

    Example
    ---------
    >>> save_video([np.array([[1, -1], [-1, 1]])], os.getcwd() + "/checkerboard_async.mp4")
    """
    # Testing for correct input
    if len(state_list) == 0:
        raise ValueError("state_list cannot be empty")
    if type(state_list) != list:
        raise ValueError("state_list must be a list")
    if type(out_path) != str:
        raise ValueError("out_path must be a string")

    # Create figure and list of frames
    anim = plt.figure()
    frames = []

    # Turn each state into a frame
    for i in range(len(state_list)):
        if(len(state_list) > 10):
            if i % 1000 == 0:
                new_frame = state_list[i]
                # Add new frame to list of frames
                frames.append([plt.imshow(new_frame, cmap='gray')])
        else:
            new_frame = state_list[i]
            # Add new frame to list of frames
            frames.append([plt.imshow(new_frame, cmap='gray')])
    chess_anim = pltanim.ArtistAnimation(anim, frames)

    # Save video
    writervideo = pltanim.FFMpegWriter(fps=2)
    chess_anim.save(out_path, writer=writervideo)

    return out_path

def reshape_history(history) : 
    """ Reshapes elements in array history to 50x50 matrices

    Parameters
    ----------
    history (1D array) : history of states leading to convergence

    Return
    --------
    2D array, history of states with each state being a 50x50 matrix
        history
    """

    # Reshape the stored states to 50*50 matrices
    for i in range(np.shape(history)[0]) : 
        history[i] = np.reshape(history[i], (50, 50))
    
    return history