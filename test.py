import matplotlib.pyplot as plt
from math import fabs

# Constants for the golden ratio
R = 0.61803399  # The golden ratio
C = 1.0 - R     # Complement of the golden ratio

def golden(ax, bx, cx, f, tol):
    """
    Perform a golden section search to find the minimum of the function f,
    with plots of each iteration showing the function and abscissas.
    """
    x0, x3 = ax, cx  # Initial points
    if fabs(cx - bx) > fabs(bx - ax):
        x1 = bx
        x2 = bx + C * (cx - bx)  # x0 to x1 is the smaller segment
    else:
        x2 = bx
        x1 = bx - C * (bx - ax)

    # Initial function evaluations
    f1 = f(x1)
    f2 = f(x2)

    iteration = 0  # Track iteration count

    # Iteratively refine the search
    while fabs(x3 - x0) > tol * (fabs(x1) + fabs(x2)):
        # Plot the function and highlight points
        if iteration == 0:
            plot_iteration(f, x0, x1, x2, x3, iteration)
        iteration += 1

        if f2 < f1:
            # Update points and function evaluations for this case
            x0, x1, x2 = x1, x2, R * x1 + C * x3
            f1, f2 = f2, f(x2)
        else:
            # Update points and function evaluations for the other case
            x3, x2, x1 = x2, x1, R * x2 + C * x0
            f2, f1 = f1, f(x1)

    # Final plot
    print(f"{iteration} iterations")
    # plot_iteration(f, x0, x1, x2, x3, iteration, final=True)

    # Determine the final result
    if f1 < f2:
        return f1, x1  # Return the minimum value and its abscissa
    else:
        return f2, x2

def plot_iteration(f, x0, x1, x2, x3, iteration, final=False):
    """
    Plot the function and highlight the abscissas for the current iteration.
    """
    x = [i * 0.01 for i in range(int(x0 * 100), int(x3 * 100) + 1)]
    y = [f(xi) for xi in x]

    plt.figure()
    plt.plot(x, y, label='f(x)')
    plt.scatter([x0, x1, x2, x3], [f(x0), f(x1), f(x2), f(x3)], color='red', zorder=5)
    plt.axvline(x=x1, color='blue', linestyle='--', label='x1')
    plt.axvline(x=x2, color='green', linestyle='--', label='x2')

    plt.title(f"Iteration {iteration}" + (" (Final)" if final else ""))
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.legend()
    plt.show()

# Example usage
if __name__ == "__main__":
    # Define a test function
    def test_function(x):
        return (x - 2) ** 2 + 3*x -5

    # Parameters
    ax, bx, cx = -20.0, -10.0, 20.0
    print(ax, bx, cx)
    tol = 1e-5

    # Perform the golden section search with plots
    min_val, min_x = golden(ax, bx, cx, test_function, tol)
    print(f"Minimum value: {min_val}")
    print(f"Minimum occurs at: {min_x}")
