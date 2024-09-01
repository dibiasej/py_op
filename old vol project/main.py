from payoff import Diagram
from option import Option
import numpy as np
if __name__ == "__main__":
    spot: np.ndarray = np.linspace(80, 120, 100)
    call = Option(100, 100, 1/12, .14)
    call.setPut()
    print(call.getDelta())
    print(call.d1)
    print(call.probabilityITM())