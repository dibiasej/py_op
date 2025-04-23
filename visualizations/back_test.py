class BackTest:
    """
    We need to delete this and turn it into a seperate .py file
    """

    def __init__(self):
        if "ticker" not in st.session_state:
            st.session_state["ticker"] = ""
        if "strike" not in st.session_state:
            st.session_state["strike"] = ""
        if "expiration" not in st.session_state:
            st.session_state["expiration"] = ""
        if "option_start_date" not in st.session_state:
            st.session_state["option_start_date"] = ""
        if "option_end_date" not in st.session_state:
            st.session_state["option_end_date"] = ""

    def create_display(self):
        st.session_state["ticker"] = st.text_input("Enter a Ticker", st.session_state["ticker"])
        st.session_state["strike"] = st.text_input("Enter a Strike", st.session_state["strike"])
        st.session_state["expiration"] = st.text_input("Enter the Option Expiration ('YYYY-MM-DD')", st.session_state["expiration"])
        st.session_state["option_start_date"] = st.text_input("Enter the Option Start Date ('YYYY-MM-DD')", st.session_state["option_start_date"])
        st.session_state["option_end_date"] = st.text_input("Enter the Option End Date ('YYYY-MM-DD')", st.session_state["option_end_date"])

    def plot_display(self):
        option_prices, dates = bt.historical_option_price(str(st.session_state["ticker"]), int(st.session_state["strike"]), str(st.session_state["option_start_date"]), str(st.session_state["option_end_date"]), str(st.session_state["expiration"]))

        fig, ax = plt.subplots()
        ax.plot(dates, option_prices)

        ax.set_xlabel("Strike Prices")
        ax.set_ylabel("Implied Volatilities")
        ax.set_title(f"Performance")

        plt.xticks(rotation=45)

        return st.pyplot(fig)
    
def main():
    pass