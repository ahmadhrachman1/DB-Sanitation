import pandas as pd
import openpyxl


def read_excel_sheet(spreadsheet: str, sheet: str, header: int = 0):
    sheet_dataframe = pd.read_excel(io=spreadsheet, sheet_name=sheet, header=header)
    return sheet_dataframe


def reshape_dataframe(dataframe: pd.DataFrame, rows: list = [], columns: list = []):
    if not rows and not columns:
        dataframe_adjusted = dataframe
    elif not columns:
        dataframe_adjusted = dataframe.iloc[rows[0]:rows[1], :]
    elif not rows:
        dataframe_adjusted = dataframe.iloc[:, columns[0]:columns[1]]
    else:
        dataframe_adjusted = dataframe.iloc[rows[0]:rows[1], columns[0]:columns[1]]

    return dataframe_adjusted


if __name__ == "__main__":
    start_module_dataframe = read_excel_sheet(spreadsheet="erp/erp_3.1.xlsx", sheet="Copy of ST", header=2)
    start_module_dataframe_adjusted = reshape_dataframe(start_module_dataframe, columns=[1, 12])
    print(start_module_dataframe)
