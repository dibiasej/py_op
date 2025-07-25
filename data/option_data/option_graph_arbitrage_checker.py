#from process_option_chain import OptionFactory

class ButterflyArbitrage:

    def __init__(self, call_graph, put_graph) -> None:
        self.call_graph = call_graph
        self.put_graph = put_graph

    def check_calls(self):
        exps = list(self.call_graph.get_expirations())
        for exp in exps:
            skew = self.call_graph.get_skew(exp)
            call_strikes = skew.strikes()
            call_prices = skew.prices()
            for i in range(1, len(call_strikes) - 1):
                butterfly_price = call_prices[i - 1] - 2 * call_prices[i] + call_prices[i + 1]