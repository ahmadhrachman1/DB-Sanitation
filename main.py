import pandas as pd
import openpyxl


def read_excel_sheet(spreadsheet: str, sheet: str, header: int = 0):
    sheet_dataframe = pd.read_excel(io=spreadsheet, sheet_name=sheet, header=header)
    return sheet_dataframe


def reshape_dataframe(dataframe: pd.DataFrame, rows: list = [], columns: list = [], drop_rows: list = [],
                      drop_columns: list = []):
    if not rows and not columns:
        dataframe_adjusted = dataframe
    elif not columns:
        dataframe_adjusted = dataframe.iloc[rows[0]:rows[1], :]
    elif not rows:
        dataframe_adjusted = dataframe.iloc[:, columns[0]:columns[1]]
    else:
        dataframe_adjusted = dataframe.iloc[rows[0]:rows[1], columns[0]:columns[1]]

    if drop_rows:
        dataframe_adjusted = dataframe_adjusted.drop(drop_rows)
    if drop_columns:
        dataframe_adjusted = dataframe_adjusted.drop(drop_columns, axis=1)

    return dataframe_adjusted


def change_drop_id_form(dataframe: pd.DataFrame, column: str, fill_none: bool = False, to_datatype=None):
    if fill_none:
        dataframe[column] = dataframe[column].fillna(0)
    if to_datatype:
        dataframe[column] = dataframe[column].astype(to_datatype)
    return dataframe


if __name__ == "__main__":
    start_module_dataframe = read_excel_sheet(spreadsheet="erp/erp_3.1.xlsx", sheet="Copy of ST", header=2)
    start_module_dataframe_reshaped = reshape_dataframe(start_module_dataframe, columns=[1, 12], drop_rows=[0],
                                                        drop_columns=["Hardware Version & Type",
                                                                      "Start Module Pre-Production Status",
                                                                      "Assigned to Customer ID",
                                                                      "Assigned to Order ID"])
    start_module_dataframe_reshaped = change_drop_id_form(start_module_dataframe_reshaped,
                                                          column="Drop App Code",
                                                          fill_none=True, to_datatype=int)
    print(start_module_dataframe_reshaped)
