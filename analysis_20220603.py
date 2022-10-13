import pandas as pd
from lib import erp_sccconfig_comparison as lib

if __name__ == '__main__':
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

    # ERP V3
    start_module_erp_three = pd.read_excel(
        io="erp/erp_3.1.xlsx",
        sheet_name="Copy of ST",
        header=2
    )
    extension_module_erp_three = pd.read_excel(
        io="erp/erp_3.1.xlsx",
        sheet_name="Copy of EXT",
        header=2
    )

    erp_three_module_dataframe = lib.setup_erp_three_module_dataframe(
        start_module=start_module_erp_three,
        extension_module=extension_module_erp_three
    )

    # DB Sheet from ERP V3
    erp_three_db_module_dataframe = pd.read_excel(
        io="erp/erp_3.1.xlsx",
        sheet_name='DB',
        header=1
    )
    erp_three_db_module_dataframe = lib.setup_erp_db_dataframe(erp_db=erp_three_db_module_dataframe)

    # Joining ERP V2 & V3
    erp_concatenated_dataframe = pd.concat([
        erp_two_module_dataframe,
        erp_three_module_dataframe
    ]).drop_duplicates(keep='first')
    delta_dict_ = lib.generate_delta_dictionary(
        concatenated_dataframe=erp_concatenated_dataframe,
        midpoint=erp_two_module_dataframe.shape[0]
    )
    erp_module_dataframe = lib.setup_filtered_list(
        erp_concatenated_dataframe,
        delta_dict_
    )

    # Preparing DB input
    sccconfig_dataframe = pd.read_csv(filepath_or_buffer="db/sccconfig")
    sccconfig_dataframe = lib.setup_sccconfig_dataframe(sccconfig=sccconfig_dataframe)
    print(sccconfig_dataframe[sccconfig_dataframe['UUID'] == '4NBVR8EPJ5TQ4H85'])

    # Creating delta between ERP and DB
    log_dataframe = pd.concat([
        erp_module_dataframe,
        sccconfig_dataframe
    ]).drop_duplicates(keep='first')

    # Generating delta codes and saving result
    log_delta_dict = lib.generate_delta_dictionary(
        concatenated_dataframe=log_dataframe,
        midpoint=erp_two_module_dataframe.shape[0]
    )
    delta_table = lib.format_delta_table(log_dataframe, log_delta_dict)
    # with open('log/delta_entries_erp_and_sccconfig_delta_20220712.txt', 'w') as f:
    #     f.write(delta_table)
