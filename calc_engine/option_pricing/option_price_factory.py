from calc_engine.option_pricing import simulation as sim
from calc_engine.option_pricing import analytical_solutions as an

class OptionPriceFactory:
    @staticmethod
    def create_model(model_name: str, calculation_type: str):
        """
        calculation_type can be simulation or analytical so far, we will add more eg fft
        """
        if calculation_type.lower() == 'analytical':
            match model_name.lower():
                case 'bachelier' | "bach":
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
                    return sim.BlackScholesSimulation()
                case 'bachelier' | "bach":
                    return sim.BachelierSimulation()
                case "heston":
                    return sim.HestonSimulation()
                case "variance gamma" | "vg" | "variancegamma":
                    return sim.VarianceGammaSimulation()
                case "sabr":
                    return sim.SABRSimulation()
                
        elif calculation_type.lower() == 'fft':
            raise NotImplementedError("FFT not implemented yet")
        
        elif calculation_type.lower() == 'fourier inversion':
            pass