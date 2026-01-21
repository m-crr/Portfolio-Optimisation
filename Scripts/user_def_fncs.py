import numpy as np
import pandas as pd
import yfinance as yf

# To donwload data from YFinance API


def download_data(tickers, start_date, end_date, progress=True):
    print("Downloading stock data from Yahoo Finance...")
    print(f"Tickers: {', '.join(tickers)}")

    try:
        data = yf.download(
            tickers=tickers,
            start=start_date,
            end=end_date,
            progress=progress,
            group_by="tickers",
            auto_adjust=True,
        )
        data

        if data.empty:
            raise ValueError(
                "No data downloaded. Please check the tickers and date range."
            )
        return data

    except Exception as e:
        print(f"Error downloading data: {str(e)}\n")

        return None


# =========================================================================================================


# To convert daily returns, adj. close price, and volume to dataframes.


def process_raw_data(raw_data, tickers):
    if len(tickers) == 1:
        prices = raw_data["Close"].to_frame()
        prices.columns = tickers
        volume = raw_data["Volume"].to_frame()
        volume.columns = tickers
    else:
        prices = pd.DataFrame()
        volume = pd.DataFrame()

        for ticker in tickers:
            try:
                if isinstance(raw_data.columns, pd.MultiIndex):
                    prices[ticker] = raw_data[ticker]["Close"]
                    volume[ticker] = raw_data[ticker]["Volume"]
                else:
                    prices[ticker] = raw_data[ticker]
                    volume[ticker] = raw_data[ticker]
            except KeyError:
                print(f"Warning: No data found for ticker {ticker}. Skipping.\n")

    returns = prices.pct_change()
    returns = returns.dropna()

    print("Processed data:\n")
    print(f"Prices shape: {prices.shape}\n")
    print(f"Returns shape: {returns.shape}\n")
    print(f"Missing values in prices: {prices.isna().sum().sum()}\n")
    print(f"Missing values in returns: {returns.isna().sum().sum()}\n")

    return {"prices": prices, "returns": returns, "volume": volume}


# =========================================================================================================


# To obtain stats like annulized returns and volitility, Sharpe ratio, max drawdown, and total returns.


def summary_stats(returns, prices, risk_free_rate, start_date, end_date):
    stats = pd.DataFrame(index=returns.columns)

    number_trading_days = 252

    # Annualized Return
    return_for_cagr = (returns + 1).prod()
    stats["annualized_return"] = (
        (return_for_cagr) ** (number_trading_days / len(returns))
    ) - 1

    # Annualized Volatility
    daily_volatility = returns.std()
    stats["annualized_volatility"] = daily_volatility * np.sqrt(252)

    # Sharpe Ratio
    stats["sharpe_ratio"] = (
        stats["annualized_return"] - (risk_free_rate / 100)
    ) / stats["annualized_volatility"]

    # Max Drawdown
    mdd = []

    for ticker in prices.columns:
        running_peak = prices[ticker].cummax()
        drawdown = (prices[ticker] - running_peak) / running_peak
        mdd.append(drawdown.min())

    stats["max_drawdown"] = mdd

    stats[f"current_price_{end_date}"] = prices.iloc[-1]
    stats[f"start_price_{start_date}"] = prices.iloc[0]
    stats["total_returns"] = (prices.iloc[-1] - prices.iloc[0]) / prices.iloc[0]

    for col in [
        "annualized_return",
        "annualized_volatility",
        "max_drawdown",
        "total_returns",
    ]:
        stats[col] = stats[col].apply(lambda x: f"{x * 100:.2f}%")

    stats["sharpe_ratio"] = stats["sharpe_ratio"].apply(lambda x: f"{x:.3f}")
    stats[f"current_price_{end_date}"] = stats[f"current_price_{end_date}"].apply(
        lambda x: f"{x:.2f}"
    )
    stats[f"start_price_{start_date}"] = stats[f"start_price_{start_date}"].apply(
        lambda x: f"{x:.2f}"
    )

    return stats


# =========================================================================================================


# To analyze correlation between different assets within a portfolio.


def correlation_analysis(returns):
    corr_matrix = returns.corr()

    print("Correlation Heatmap")

    high_corr = []

    for i in range(len(corr_matrix.columns)):
        for j in range(i + 1, len(corr_matrix.columns)):
            if corr_matrix.iloc[i, j] > 0.7:
                high_corr.append(
                    {
                        "stock 1": corr_matrix.columns[i],
                        "stock 2": corr_matrix.columns[j],
                        "correlation": corr_matrix.iloc[i, j],
                    }
                )

    if high_corr:
        high_corr_df = pd.DataFrame(high_corr).sort_values(
            "correlation", ascending=False
        )
        print(high_corr_df.to_string(index=False))
    else:
        print("There are no pairs with correlation > 0.7")

    low_corr = []

    for i in range(len(corr_matrix.columns)):
        for j in range(i + 1, len(corr_matrix.columns)):
            if corr_matrix.iloc[i, j] < 0.3:
                low_corr.append(
                    {
                        "stock 1": corr_matrix.columns[i],
                        "stock 2": corr_matrix.columns[j],
                        "correlation": corr_matrix.iloc[i, j],
                    }
                )

    if low_corr:
        low_corr_df = pd.DataFrame(low_corr).sort_values("correlation")
        print(low_corr_df.to_string(index=False))
    else:
        print("There are no pairs with correlation < 0.3\n")

    return corr_matrix


# =========================================================================================================


# To validate missing values and outliers.


def data_validation(returns, prices):
    all_checks_passed = True

    missing_returns = returns.isna().sum()
    missing_prices = prices.isna().sum()

    if missing_returns.sum() > 0:
        print(f"Warning - Found {missing_returns.sum()} missing values in returns")
        print(f"{missing_returns[missing_returns > 0]}\n")
        all_checks_passed = False
    else:
        print("No missing values in returns \n")

    if missing_prices.sum() > 0:
        print(f"Warning - Found {missing_prices.sum()} missing values in prices")
        print(f"{missing_prices[missing_prices > 0]}\n")
        all_checks_passed = False
    else:
        print("No missing values in prices \n")

    extreme_returns = (returns.abs() > 0.5).sum()
    if extreme_returns.sum() > 0:
        print(f"Warning - Found {extreme_returns.sum()} extreme return")
        print(f"{extreme_returns[extreme_returns > 0]}\n")
    else:
        print("No extreme returns found \n")

    return all_checks_passed


# =========================================================================================================


# To save data to .csv and .xlsx files.


def data_write(
    prices_data, returns_data, volume_data, summary_stats_data, folder_name, date_format
):
    import os

    prices_data = prices_data.reset_index()
    returns_data = returns_data.reset_index()
    volume_data = volume_data.reset_index()
    summary_stats_data = summary_stats_data.reset_index()

    parent_dir = os.path.dirname(os.getcwd())
    new_folder_path = os.path.join(parent_dir, folder_name)
    os.makedirs(new_folder_path, exist_ok=True)

    if not os.path.exists(f"{new_folder_path}/prices.csv"):
        prices_data.to_csv(
            f"{new_folder_path}/prices.csv", index=False, date_format=date_format
        )
        print(f"Prices dataframe saved to {new_folder_path} as a .csv file \n")
    else:
        print("Error: A .csv file for prices already exists \n")

    if not os.path.exists(f"{new_folder_path}/returns.csv"):
        returns_data.to_csv(
            f"{new_folder_path}/returns.csv", index=False, date_format=date_format
        )
        print(f"Returns dataframe saved to {new_folder_path} as a .csv file \n")
    else:
        print("Error: A .csv file for returns already exists \n")

    if not os.path.exists(f"{new_folder_path}/volume.csv"):
        volume_data.to_csv(
            f"{new_folder_path}/volume.csv", index=False, date_format=date_format
        )
        print(f"Volume dataframe saved to {new_folder_path} as a .csv file \n")
    else:
        print("Error: A .csv file for volumes already exists \n")

    if not os.path.exists(f"{new_folder_path}/summary_stats.csv"):
        summary_stats_data.to_csv(f"{new_folder_path}/summary_stats.csv")
        print(f"Summary dataframe saved to {new_folder_path} as a .csv file \n")
    else:
        print("Error: A .csv file for summary stats already exists \n")

    df_to_write = {
        "Daily Prices": prices_data,
        "Daily Returns": returns_data,
        "Daily Volumes": volume_data,
        "Summary Stats": summary_stats_data,
    }

    excel_file_name = "data_excel.xlsx"
    excel_output = os.path.join(new_folder_path, excel_file_name)

    if not os.path.exists(excel_output):
        try:
            with pd.ExcelWriter(
                excel_output, engine="xlsxwriter", date_format=date_format
            ) as writer:
                for sheet_name, df in df_to_write.items():
                    df.to_excel(writer, sheet_name=sheet_name, index=False)
            print(f"Dataframes saved to {new_folder_path} as an excel file \n")

        except Exception as e:
            print(
                f"An error occurred while writing data frames to an excel file {e} \n"
            )
    else:
        print("Error: A combined excel file already exists \n")
