# =========================================================================================================

# Required modules

import os
from datetime import datetime

import numpy as np
import pandas as pd
import yfinance as yf
from dateutil.relativedelta import relativedelta
from sklearn.linear_model import LinearRegression

# =========================================================================================================


# *********************************************************************************************************

# Notebook: collection.ipynb

# *********************************************************************************************************


# =========================================================================================================


# To donwload data from YFinance API


def download_data(tickers, lookback_window, progress=True):
    print("Downloading data from Yahoo Finance...\n")
    print(f"Tickers: {', '.join(tickers)}\n")

    current_date = datetime.now()

    try:
        data = yf.download(
            tickers=tickers,
            start=(current_date - relativedelta(years=lookback_window)).strftime(
                "%Y-%m-%d"
            ),
            end=current_date.strftime("%Y-%m-%d"),
            progress=progress,
            group_by="tickers",
            auto_adjust=True,
        )
        data

        if data.empty:
            raise ValueError(
                "No data downloaded. Please check the tickers and date range.\n"
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

    return {"prices": prices, "returns": returns, "volume": volume}


# =========================================================================================================


# To validate missing values and outliers.


def data_validation(returns, prices):
    all_checks_passed = True

    missing_returns = returns.isna().sum()
    missing_prices = prices.isna().sum()

    if missing_returns.sum() > 0:
        print(f"Warning: Found {missing_returns.sum()} missing values in returns - ")
        print(f"{missing_returns[missing_returns > 0]} \n")
        all_checks_passed = False
    else:
        print("No missing values in returns \n")

    if missing_prices.sum() > 0:
        print(f"Warning: Found {missing_prices.sum()} missing values in prices - ")
        print(f"{missing_prices[missing_prices > 0]} \n")
        all_checks_passed = False
    else:
        print("No missing values in prices \n")

    extreme_returns = (returns.abs() > 0.5).sum()
    if extreme_returns.sum() > 0:
        print(f"Warning - Found {extreme_returns.sum()} extreme return")
        print(f"{extreme_returns[extreme_returns > 0]} \n")
    else:
        print("No extreme returns found \n")

    return all_checks_passed


# =========================================================================================================


# To obtain stats like annulized returns and volitility, Sharpe ratio, max drawdown, and total returns.


def summary_stats(returns, prices, risk_free_rate, lookback_window):
    current_date = datetime.now()
    start_date = (current_date - relativedelta(years=lookback_window)).strftime(
        "%Y-%m-%d"
    )
    end_date = current_date.strftime("%Y-%m-%d")

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


# To save data to .csv files.


def data_write_csv(data_files, file_names, folder_name, date_format):
    parent_dir_name = os.path.dirname(os.getcwd())
    folder_path = os.path.join(parent_dir_name, folder_name)
    os.makedirs(folder_path, exist_ok=True)

    for data_file, file_name in zip(data_files, file_names):
        dataframe = data_file.reset_index()
        dataframe_name = file_name.split(".")[0]
        file_path = f"{folder_path}/{file_name}"
        try:
            if not os.path.exists(f"{folder_name}/{file_name}"):
                dataframe.to_csv(file_path, index=False, date_format=date_format)
                print(
                    f"The {dataframe_name} data has been saved to {folder_path} as {file_name}\n"
                )
            else:
                print(f"Warning: {file_name} already exists\n")
        except Exception:
            print(f"Error: Could not write the {dataframe_name} data to a csv file\n")


# =========================================================================================================


# To save data to .xlsx file.


def data_write_excel(worksheet_names, dataframes, folder_name, file_name, date_format):
    sheet_names = worksheet_names
    data_files = dataframes

    parent_directory = os.path.dirname(os.getcwd())
    file_path = os.path.join(parent_directory, folder_name, file_name)

    try:
        if not os.path.exists(file_path):
            with pd.ExcelWriter(
                file_path, engine="openpyxl", date_format=date_format
            ) as writer:
                for sheet_name, dataframe in zip(sheet_names, data_files):
                    dataframe = dataframe.reset_index()
                    if dataframe.columns[0] == "index":
                        dataframe = dataframe.rename(columns={"index": "Ticker"})
                        dataframe.to_excel(writer, sheet_name=sheet_name, index=False)
                    else:
                        dataframe.to_excel(writer, sheet_name=sheet_name, index=False)
                print(
                    f"The following data - {[sheet_name for sheet_name in sheet_names]} has been combined and saved to {os.path.join(parent_directory, folder_name)} as {file_name}\n"
                )

        else:
            with pd.ExcelWriter(
                file_path, engine="openpyxl", mode="a", date_format=date_format
            ) as writer2:
                for sheet_name, dataframe in zip(sheet_names, data_files):
                    dataframe = dataframe.reset_index()
                    if dataframe.columns[0] == "index":
                        dataframe = dataframe.rename(columns={"index": "Ticker"})
                        dataframe.to_excel(writer2, sheet_name=sheet_name, index=False)
                    else:
                        dataframe.to_excel(writer2, sheet_name=sheet_name, index=False)
                print(
                    f"The following data - {[sheet_name for sheet_name in sheet_names]} has been appended and saved to {file_name}\n"
                )

    except Exception:
        print(f"Error: Could not create/append to {file_name} \n")


# =========================================================================================================


# To analyze correlation between different assets within a portfolio.


def correlation_analysis(returns, correlation_thres):
    corr_matrix = returns.corr()

    print("Correlation Analysis - \n")

    high_corr = []

    for i in range(len(corr_matrix.columns)):
        for j in range(i + 1, len(corr_matrix.columns)):
            if corr_matrix.iloc[i, j] > correlation_thres:
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
        print(f"Asset pairs with correlation greater than {correlation_thres}: \n")
        print(f"{high_corr_df.to_string(index=False)}\n")
    else:
        print(f"There are no pairs with correlation > {correlation_thres} \n")

    low_corr = []

    for i in range(len(corr_matrix.columns)):
        for j in range(i + 1, len(corr_matrix.columns)):
            if corr_matrix.iloc[i, j] < (1.0 - correlation_thres):
                low_corr.append(
                    {
                        "stock 1": corr_matrix.columns[i],
                        "stock 2": corr_matrix.columns[j],
                        "correlation": corr_matrix.iloc[i, j],
                    }
                )

    # if low_corr:
    #     low_corr_df = pd.DataFrame(low_corr).sort_values("correlation")
    #     print(low_corr_df.to_string(index=False))
    # else:
    #     print(f"There are no pairs with correlation < {(1.0 - correlation_thres)}\n")

    corr_values = corr_matrix.values
    average_corr = (corr_values.sum() - corr_values.trace()) / (
        corr_values.size - len(corr_values)
    )
    print(
        f"The average pairwise correlation of assets within the portfolio: {average_corr:.3f} \n"
    )

    if average_corr < 0.3:
        print("Summary - Good diversification with low average correlation \n")
    elif average_corr > 0.3 and average_corr < 0.6:
        print("Summary - Decent diversification with moderate correlation \n")
    else:
        print("Summary - Limited diversification with high correlation \n")

    return corr_matrix


# =========================================================================================================


# *********************************************************************************************************

# Notebook: optimization_1.ipynb

# *********************************************************************************************************


# =========================================================================================================


# To retrieve beta values.


def collect_betas(returns, current_rf):
    beta_values = {}

    risk_free_rate = current_rf

    for ticker in returns.columns:
        stock = yf.Ticker(ticker)
        try:
            beta = stock.info["beta"]
            beta_values[ticker] = beta
        except KeyError:
            if ticker != "^GSPC" and ticker != "SPY":
                x = (returns["^GSPC"].values - risk_free_rate).reshape(-1, 1)
                y = (returns[ticker].values - risk_free_rate).reshape(-1, 1)
                regression_model = LinearRegression()
                regression_model.fit(x, y)
                beta = np.round(regression_model.coef_[0][0], 3).item()
                beta_values[ticker] = beta

    return beta_values


# =========================================================================================================


# To calculate expected returns for stocks in portfolio and conver to a dataframe.


def calc_capm(returns, market_ticker, lookback_window, current_risk_free_rate):
    capm_values = {}

    annualized_market_return = (
        ((1 + returns[market_ticker]).prod()) ** (1 / lookback_window)
    ) - 1

    beta_dict = collect_betas(returns, current_risk_free_rate)

    for ticker, beta in beta_dict.items():
        beta_value = beta
        capm = current_risk_free_rate + (
            beta_value * (annualized_market_return - current_risk_free_rate)
        )
        capm_values[ticker] = (np.round(capm, 4)) * 100

    beta_capm_df = pd.DataFrame([beta_dict, capm_values]).T
    beta_capm_df.columns = ["beta_values", "expected_returns"]

    return beta_capm_df


# =========================================================================================================
