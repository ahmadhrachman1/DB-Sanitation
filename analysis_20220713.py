import pandas as pd
from lib import erp_sccconfig_comparison as lib

if __name__ == '__main__':
    # Preparing ERP 20220713 input
    start_module_erp_three_20220713 = pd.read_excel(
        io="erp/erp_3.1_20220713.xlsx",
        sheet_name="ST",
        header=2
    )

    extension_module_erp_three_20220713 = pd.read_excel(
        io="erp/erp_3.1_20220713.xlsx",
        sheet_name="EXT",
        header=2
    )

    erp_three_module_dataframe_20220713 = lib.setup_erp_three_module_dataframe(
        start_module=start_module_erp_three_20220713,
        extension_module=extension_module_erp_three_20220713
    ).dropna()

    # Preparing DB 20220713 input
    sccconfig_dataframe_20220713 = pd.read_csv(filepath_or_buffer="db/sccconfig_20220713")
    sccconfig_dataframe_20220713 = lib.setup_sccconfig_dataframe(sccconfig=sccconfig_dataframe_20220713)

    # Creating delta between ERP and DB 20220713
    log_dataframe_20220713 = pd.concat(
        [erp_three_module_dataframe_20220713[
             erp_three_module_dataframe_20220713['Creation date'] > '2022-06-02'],
         sccconfig_dataframe_20220713[
             sccconfig_dataframe_20220713['Creation date'] > '2022-06-02']]
    ).drop_duplicates(keep='first')
    log_delta_dict_20220713 = lib.generate_delta_dictionary(
        concatenated_dataframe=log_dataframe_20220713,
        midpoint=erp_three_module_dataframe_20220713.shape[0]
    )
    delta_table_20220713 = lib.format_delta_table(log_dataframe_20220713, log_delta_dict_20220713)

    # Parsing delta
    log_dataframe_20220713_parsed = lib.parse_delta_dict(
        concatenated_dataframe=log_dataframe_20220713,
        delta_dict_prev=log_delta_dict_20220713
    )

    log_delta_dict_20220713_parsed = lib.generate_delta_dictionary(
        concatenated_dataframe=log_dataframe_20220713_parsed,
        midpoint=erp_three_module_dataframe_20220713.shape[0]
    )
    delta_table_20220713_parsed = lib.format_delta_table(
        dataframe=log_dataframe_20220713_parsed,
        delta_code_dict=log_delta_dict_20220713_parsed
    )
