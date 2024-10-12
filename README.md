# Coupled Oscillators Simulation

## Description

This project simulates a system of coupled oscillators using various numerical methods, including Euler Forward, Euler Backward, and Crank-Nicholson. The simulation visualizes the dynamics of the oscillators and allows for exploration of different representation methods, such as adjacency matrix, adjacency list, and incidence list.

## Features

- **Multiple Representation Methods**: Supports adjacency matrix, adjacency list, and incidence list for defining the coupled oscillators system.
- **Numerical Methods**: Implements Euler Forward, Euler Backward, and Crank-Nicholson methods for simulating oscillator dynamics.
- **Dynamic Visualization**: Generates an animation to visualize the motion of the oscillators over time.
- **Customizable Parameters**: Users can define masses, initial positions, initial velocities, and connection constants of the oscillators.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/coupled-oscillators.git
   ```
2. Change to the project directory:
  ```bash
  cd coupled-oscillators
  ```
3. Install the required packages:
  ```bash
  pip install numpy matplotlib
  ```

## Usage
Run the main simulation script:
```bash
python simulation.py
```

Modify the initial parameters in the script to observe different behaviors of the oscillators.

## Example
The following parameters can be modified in the simulation.py file:

- Masses: Define the masses of the oscillators.
- Initial Positions: Set the initial coordinates of each oscillator.
- Initial Velocities: Assign initial velocities to the oscillators.
- Connection Constants: Specify the constants that dictate the strength of connections between the oscillators.

## Contributing
Contributions are welcome! If you have suggestions for improvements or want to add new features, please fork the repository and create a pull request.
