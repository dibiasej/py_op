import analytical_solutions as an
import simulation as sim

class OptionPriceFactory:
    @staticmethod
    def create_model(model_name: str, calculation_type: str):
        """
        calculation_type can be simulation or analytical so far, we will add more eg fft
        """
        if calculation_type.lower() == 'analytical':
            match model_name.lower():
                case 'bachelier':
                    return an.BachelierAnalytical
                case 'blackscholesmerton' | "blackscholes" | 'bsm' | 'bs' | 'black scholes merton' | "black scholes" | 'bsm' | 'bs':
                    return an.BlackScholesMertonAnalytical
                case 'constant elasticity variance' | "cev":
                    return an.CEVAnalytical
                case _:
                    raise ValueError(f"Unknown model name: {model_name}")
                
        elif calculation_type.lower() == 'simulation':
            match model_name.lower():
                case 'blackscholesmerton' | "blackscholes" | 'bsm' | 'bs' | 'black scholes merton' | "black scholes" | 'bsm' | 'bs':
                    model = sim.BlackScholesSimulation()