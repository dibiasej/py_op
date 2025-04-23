"""
This module will hold any models that need quadratured to price

We might have to get rid of this module and move these functions to a fft/characteristic fucntion module
"""
def m7_stoch_jump_call_func(S0, K, T, r, volatility, lamb, mu, delta):

    def integer_function(u, S0, K, T, r, volatility, lamb, mu, delta):
        omega = r - 0.5 * volatility ** 2 - lamb * (np.exp(mu + 0.5 * delta ** 2) - 1)
        char_func_value = np.exp((1j * u * omega - 0.5 * u ** 2 * volatility ** 2 + lamb * (np.exp(1j * u * mu - u ** 2 * delta ** 2 * 0.5) - 1)) * T)
        int_func_value = 1 / (u ** 2 + 0.25) * (np.exp(1j * u * np.log(S0 / K)) * char_func_value).real
        return int_func_value

    int_value = quad(lambda u:
                     integer_function(
                         u, S0, K, T, r, volatility, lamb, mu, delta),
                     0, np.inf, limit=250)[0]
    call_value = max(0, S0 - np.exp(-r * T) * np.sqrt(S0 * K) /
                     np.pi * int_value)
    return call_value


def m7_stoch_jump_put_func(S0, K, T, r, volatility, lamb, mu, delta):

    def integer_function(u, S0, K, T, r, volatility, lamb, mu, delta):
        omega = r - 0.5 * volatility ** 2 - lamb * (np.exp(mu + 0.5 * delta ** 2) - 1)
        char_func_value = np.exp((1j * u * omega - 0.5 * u ** 2 * volatility ** 2 + lamb * (np.exp(1j * u * mu - u ** 2 * delta ** 2 * 0.5) - 1)) * T)
        int_func_value = 1 / (u ** 2 + 0.25) * (np.exp(1j * u * np.log(S0 / K)) * char_func_value).real
        return int_func_value

    int_value = quad(lambda u:
                     integer_function(
                         u, S0, K, T, r, volatility, lamb, mu, delta),
                     0, np.inf, limit=250)[0]
    call_value = max(0, S0 - np.exp(-r * T) * np.sqrt(S0 * K) /
                     np.pi * int_value)
    put_value = call_value + K * np.exp(-r * T) - S0
    return put_value