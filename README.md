Loading Pipeline:
Production Mode:

Batch File: Loading_Pipeline.bat

Executes Stage-2 loading classification automatically for the previous 1 hour of data for all clients in client_config.

Runs the pipeline in production mode by default.
--------------------------------------------------------------------------------------------------------------------------

Backtest Mode:

Can be enabled by setting BACKTEST_MODE = True ,in **main.py**.

Provides flexible testing for different clients and date ranges.
-------------------------------------------------------------------------------------------------------------------------

Configuration Options:
| Option                    | Description                                                        |
| ------------------------- | ------------------------------------------------------------------ |
| `SINGLE_CLIENT`           | `True` = run for a single client, `False` = all clients            |
| `SINGLE_DATE`             | `True` = run for a single date, `False` = date range               |
| `CLIENT_ID`               | Client ID used when `SINGLE_CLIENT = True`                         |
| `DATE`                    | Single date to run backtest (format: `YYYY-MM-DD`)                 |
| `START_DATE` / `END_DATE` | Start and end dates for multi-date backtest (format: `YYYY-MM-DD`) |
| `SAVE_CSV`                | `True` to save results as CSV                                      |
| `CSV_OUTPUT_DIR`          | Directory to save CSV output                                       |

-------------------------------------------------------------------------------------------------------------------------
Backtest Modes:

| Mode | Description                   |
| ---- | ----------------------------- |
| 1    | Single client, single date    |
| 2    | Single client, multiple dates |
| 3    | All clients, single date      |
| 4    | All clients, multiple dates   |


The backtest imports and executes functions from backtest.py based on the selected configuration.
-------------------------------------------------------------------------------------------------------------------------

Notes:

Production mode ignores the above backtest configuration and always processes the previous hour of data for all clients.

All changes to backtest configuration must be made inside main.py before execution
