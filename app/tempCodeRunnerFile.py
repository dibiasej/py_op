    def plot_display(self):
        # Check if all the necessary fields are filled
        if not st.session_state["ticker"] or not st.session_state["strike"] or not st.session_state["expiration"] or not st.session_state["option_start_date"] or not st.session_state["option_end_date"]:
            st.write("Please fill in all the fields.")
            return

        try:
            # Convert strike to integer only if the input is valid
            strike = int(st.session_state["strike"])
            
            # Fetch option prices and dates
            option_prices, dates = bt.historical_option_price(
                str(st.session_state["ticker"]),
                strike,
                str(st.session_state["option_start_date"]),
                str(st.session_state["option_end_date"]),
                str(st.session_state["expiration"])
            )

            # Plot the results
            fig, ax = plt.subplots()
            ax.plot(dates, option_prices)
            ax.set_xlabel("Dates")
            ax.set_ylabel("Option Prices")
            ax.set_title(f"Option Price Performance for {st.session_state['ticker']}")

            plt.xticks(rotation=45)
            st.pyplot(fig)

        except ValueError:
            st.write("Invalid input. Please enter a valid number for the strike price.")
