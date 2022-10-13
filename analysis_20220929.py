import pandas as pd
from lib import erp_sccconfig_comparison as lib

if __name__ == "__main__":
    # Preparing ERP input
    # ERP V2
    start_module_erp_two = pd.read_excel(
        io="erp/CX_ERP_V2.xlsx",
        sheet_name="ST",
        header=2
    )
    extension_module_erp_two = pd.read_excel(
        io="erp/CX_ERP_V2.xlsx",
        sheet_name="EXT",
        header=2
    )
    erp_two_module_dataframe = lib.setup_erp_two_dataframe(
        start_module=start_module_erp_two,
        extension_module=extension_module_erp_two
    )

    # Preparing ERP 20220929 input
    start_module_erp_three_20220929 = pd.read_excel(
        io="erp/ERP_3.1_20220929.xlsx",
        sheet_name="ST",
        header=2
    )

    extension_module_erp_three_20220929 = pd.read_excel(
        io="erp/ERP_3.1_20220929.xlsx",
        sheet_name="EXT",
        header=2
    )

    erp_three_module_dataframe_20220929 = lib.setup_erp_three_module_dataframe(
        start_module=start_module_erp_three_20220929,
        extension_module=extension_module_erp_three_20220929
    )

    # Joining ERP V2 & V3
    erp_concatenated_dataframe = pd.concat([
        erp_two_module_dataframe,
        erp_three_module_dataframe_20220929
    ]).drop_duplicates(keep='first')

    delta_dict_erp_20220929 = lib.generate_delta_dictionary(
        concatenated_dataframe=erp_concatenated_dataframe,
        midpoint=erp_two_module_dataframe.shape[0]
    )

    erp_module_dataframe_20220929 = lib.setup_filtered_list(
        erp_concatenated_dataframe,
        delta_dict_erp_20220929
    )

    # Preparing DB 20220929 input
    sccconfig_dataframe_20220929 = pd.read_csv(filepath_or_buffer="db/sccconfig_20220929.csv")
    sccconfig_dataframe_20220929 = lib.setup_sccconfig_dataframe(sccconfig=sccconfig_dataframe_20220929)

    # Creating delta between ERP and DB
    pd.set_option('display.max_rows', 10000)
    log_dataframe_20220929 = pd.concat(
        [erp_module_dataframe_20220929,
         sccconfig_dataframe_20220929],
        ignore_index=True)

    log_delta_dict_20220929 = lib.generate_delta_dictionary(
        log_dataframe_20220929,
        midpoint=erp_three_module_dataframe_20220929.shape[0]
    )
    delta_table_20220929 = lib.format_delta_table(log_dataframe_20220929, log_delta_dict_20220929)

    # lib.format_delta_table_excel(
    #     concatenated_dataframe=log_dataframe_20220929,
    #     delta_code_dict=log_delta_dict_20220929,
    #     file_path='test/delta_table_erp_sccconfig_20220929.xlsx'
    # )

    print(erp_module_dataframe_20220929.shape[0]
          )

    # # Write results
    # with open('log/delta_entries_erp_and_sccconfig_20220929.txt', 'w+') as f:
    #     f.write(delta_table_20220929)
    #
    # import json
    #
    # with open('json/delta_dict_erp_and_sccconfig_20220929.json', 'w+') as f:
    #     json.dump(log_delta_dict_20220929, f, indent=2)
