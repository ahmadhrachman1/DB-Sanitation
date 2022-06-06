import pandas as pd
import openpyxl


def read_excel_sheet(spreadsheet: str, sheet: str, header: int = 0):
    sheet_dataframe = pd.read_excel(io=spreadsheet, sheet_name=sheet, header=header)
    return sheet_dataframe


if __name__ == "__main__":
    start_module_dataframe = read_excel_sheet(spreadsheet="erp/erp_3.1.xlsx", sheet="Copy of ST", header=2)

    print(start_module_dataframe)
