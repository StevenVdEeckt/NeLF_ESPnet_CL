from decimal import Decimal

def average_eta(A: float, updates: int) -> float:
    """
    Compute the average eta given:
      A       = final weight of the initial model after all updates
      updates = number of updates applied

    Returns:
      average eta value
    """
    if not (0.0 < A <= 1.0):
        raise ValueError("A must be in (0, 1].")
    if updates <= 0:
        raise ValueError("Number of updates must be positive.")

    avg_eta = 1.0 - A ** (1.0 / updates)
    return avg_eta

updates = 800

# AOS
A = 0.905617
print("AOS -- Average eta: %.2E" % Decimal(average_eta(A, updates)))

# AOS+
A = 0.695693        # initial model still has 50% weight
print("AOS+ [default] -- Average eta: %.2E" %  Decimal(average_eta(A, updates)))

# AOS
A = 0.672101        # initial model still has 50% weight
print("AOS+ [target_eta] -- Average eta: %.2E" % Decimal(average_eta(A, updates)))

# AOS
A = 0.625178        # initial model still has 50% weight
print("AOS [target_eta, optimal_tau] -- Average eta: %.2E" % Decimal(average_eta(A, updates)))

