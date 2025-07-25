from calc_engine.option_pricing import simulation as sim
from calc_engine.option_pricing import analytical_solutions as an
from calc_engine.option_pricing import FFT
from calc_engine.option_pricing import foruier_inversion as fi

class OptionPriceFactory:
    @staticmethod
    def create_model(model_name: str, calculation_type: str):
        """
        calculation_type can be simulation or analytical so far, we will add more eg fft
        """
        if calculation_type.lower() == 'analytical':
            match model_name.lower():
                case 'bachelier' | "bach":
                    return an.BachelierAnalytical()
                case 'blackscholesmerton' | "blackscholes" | 'bsm' | 'bs' | 'black scholes merton' | "black scholes" | 'bsm' | 'bs':
                    return an.BlackScholesMertonAnalytical()
                case 'constant elasticity variance' | "cev":
                    return an.CEVAnalytical()
                case 'variance gamma' | "vg" | "variancegamma":
                    return an.VarianceGammaAnalytical()
                case 'sabr':
                    return an.SABRAnalytical()
                case _:
                    raise ValueError(f"Unknown model name: {model_name}")
                
        elif calculation_type.lower() in ('simulation', 'sim'):
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
                case "rbergomi" | "rough bergomi":
                    return sim.rBergomiSimulation()
                
        elif calculation_type.lower() in ('fft', 'fast fourier transform'):
            match model_name.lower():
                case "heston":
                    return FFT.HestonFFT()
                case "merton jump" | "merton jump diffusion":
                    return FFT.MertonJumpDiffusionFFT()
                case "variance gamma" | "vg" | "variancegamma":
                    return FFT.VarianceGammaFFT()
        
        elif calculation_type.lower() == 'fourier inversion':
            match model_name.lower():
                case "heston":
                    return fi.HestonFourierInversion()
                case "merton jump" | "merton jump diffusion":
                    return fi.MertonJumpDiffusionFourierInversion()
                case "variance gamma" | "vg" | "variancegamma":
                    return fi.VarianceGammaFourierInversion()