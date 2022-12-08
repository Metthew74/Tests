from functions import *

#********************** PATTERNS **********************#

M = 50
N = 2500 

memorized_patterns = generate_patterns(M, N)

# Generate relevant weights matrix by uncommenting either line
weightsMatrix_hebbian = hebbian_weights(memorized_patterns)
weightsMatrix_storkey = storkey_weights(memorized_patterns)

# Arbitrarily choose a pattern and perturb it
p0 = memorized_patterns[0]
perturbed_p0 = perturb_pattern(p0, 1000)

# Run dynamical system
history_sync_h = dynamics(perturbed_p0, weightsMatrix_hebbian, 20)
history_sync_s = dynamics(perturbed_p0, weightsMatrix_storkey, 20)

# Run dynamical asynchronous system
history_async_h = dynamics_async(perturbed_p0, weightsMatrix_hebbian, 30000, 10000)
history_async_s = dynamics_async(perturbed_p0, weightsMatrix_storkey, 30000, 10000)

# Compute energies for 4 histories
time_vs_energy_sync_h = compute_energies(history_sync_h, weightsMatrix_hebbian)[1]
time_vs_energy_sync_s = compute_energies(history_sync_s, weightsMatrix_storkey)[1]
time_vs_energy_async_h = compute_energies(history_async_h, weightsMatrix_hebbian)[1]
time_vs_energy_async_s = compute_energies(history_async_s, weightsMatrix_storkey)[1]

# Create time-energy plot with four histories

fig, axs = plt.subplots(2, 2)
fig.suptitle('Time-energy plots', size=28)

# Create time-energy pairs 
list_of_pairs_sync_h = sorted(time_vs_energy_sync_h.items())
x_sync_h, y_sync_h = zip(*list_of_pairs_sync_h)

list_of_pairs_sync_s = sorted(time_vs_energy_sync_s.items())
x_sync_s, y_sync_s = zip(*list_of_pairs_sync_s)

list_of_pairs_async_h = sorted(time_vs_energy_async_h.items())
x_async_h, y_async_h = zip(*list_of_pairs_async_h)

list_of_pairs_async_s = sorted(time_vs_energy_async_s.items())
x_async_s, y_async_s = zip(*list_of_pairs_async_s)

# Sync h
axs[0, 0].plot(x_sync_h, y_sync_h)
axs[0, 0].set_title('Sync Hebbian', fontweight="bold")

# Sync s
axs[0, 1].plot(x_sync_s, y_sync_s, 'tab:red')
axs[0, 1].set_title('Sync Storkey', fontweight="bold")

# Async h
axs[1, 0].plot(x_async_h, y_async_h, 'tab:orange')
axs[1, 0].set_title('Async Hebbian', fontweight="bold")

# Async s
axs[1, 1].plot(x_async_s, y_async_s, 'tab:green')
axs[1, 1].set_title('Async Storkey', fontweight="bold")

for ax in axs.flat:
    ax.set(xlabel='time', ylabel='energy')

fig.tight_layout()

plt.show()

#********************** CHECKERBOARD **********************#

# Generate checkerboard and flatten it
checkerboard_pattern = generate_checkerboard()
flattened_pattern = checkerboard_pattern.flatten()

# Replace one of the patterns of memorized_patterns with flattened checkerboard pattern
i = random.randint(0, 49)
memorized_patterns[i] = flattened_pattern

# Generate Hebbian weights matrix with memorized patterns containing the checkerboard pattern
chess_hebbian = hebbian_weights(memorized_patterns)

# Perturbation of 1000 elements in checkerboard pattern
perturbed_checkerboard = perturb_pattern(flattened_pattern, 1000)

# Run dynamical synchronous system and save video representing evolution of states until convergence
history_chess_sync = dynamics(perturbed_checkerboard, chess_hebbian, 20)
history_chess_sync = reshape_history(history_chess_sync)
destination_path_sync = os.getcwd() + "/checkerboard_sync.mp4"
save_video(history_chess_sync, destination_path_sync)

# Run dynamical asynchronous system and save video representing evolution of states until convergence
history_chess_async = dynamics_async(perturbed_checkerboard, chess_hebbian, 20000, 3000)
history_chess_async = reshape_history(history_chess_async)
destination_path_async = os.getcwd() + "/checkerboard_async.mp4"
save_video(history_chess_async, destination_path_async)  