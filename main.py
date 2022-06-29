import numpy as np
import pandas as pd
import openpyxl
import datetime
import sys

pd.options.mode.chained_assignment = None


def reshape_dataframe(dataframe: pd.DataFrame, select_rows: list = [], select_columns: list = [], drop_rows: list = [],
                      drop_columns: list = []):
    if not select_rows and not select_columns:
        dataframe_adjusted = dataframe
    elif not select_columns:
        dataframe_adjusted = dataframe.iloc[select_rows[0]:select_rows[1], :]
    elif not select_rows:
        dataframe_adjusted = dataframe.iloc[:, select_columns[0]:select_columns[1]]
    else:
        dataframe_adjusted = dataframe.iloc[select_rows[0]:select_rows[1], select_columns[0]:select_columns[1]]

    if drop_rows:
        dataframe_adjusted = dataframe_adjusted.drop(drop_rows)
    if drop_columns:
        dataframe_adjusted = dataframe_adjusted.drop(drop_columns, axis=1)

    return dataframe_adjusted


def convert_column_value_to(dataframe: pd.DataFrame, column: str, fill_none: bool = False, to_type=None):
    if fill_none:
        dataframe[column] = dataframe[column].fillna(0)
    if to_type:
        dataframe[column] = dataframe[column].astype(to_type)
    return dataframe


def remove_quotes(dataframe: pd.DataFrame, column: str):
    dataframe[column] = dataframe[column].str.replace('"', '')
    dataframe[column] = dataframe[column].replace({'N/A': '0', np.NaN: '0'})
    return dataframe


def add_is_start_module_column(dataframe: pd.DataFrame, is_start_module: bool):
    value = []
    column_length = dataframe.shape[0]

    for index in range(column_length):
        value.append(is_start_module)

    return value


def setup_erp_two_dataframe(start_module: pd.DataFrame, extension_module: pd.DataFrame):
    start_module = reshape_dataframe(dataframe=start_module, select_rows=[1, 95])
    start_module = start_module.rename({'Drop APP Code': 'Drop App Code',
                                        'Unnamed: 12': 'Hardware Version'})
    is_start_module_true = add_is_start_module_column(dataframe=start_module, is_start_module=True)
    start_module.loc[:, 'Start Module'] = is_start_module_true

    extension_module = reshape_dataframe(dataframe=extension_module, select_rows=[1, 246])
    is_start_module_false = add_is_start_module_column(dataframe=extension_module, is_start_module=False)
    extension_module.loc[:, 'Start Module'] = is_start_module_false

    erp_two_module = pd.concat([start_module,
                                extension_module])
    erp_two_module = remove_quotes(dataframe=erp_two_module, column='Hardware Version')
    erp_two_module = remove_quotes(dataframe=erp_two_module, column='SCC\nRipples')
    erp_two_module = remove_quotes(dataframe=erp_two_module, column='SCC\nmpyCross')

    erp_two_module = convert_column_value_to(dataframe=erp_two_module, column='Drop App Code',
                                             to_type=int, fill_none=True)
    erp_two_module = convert_column_value_to(dataframe=erp_two_module, column='Drop App Code',
                                             to_type=str)

    erp_two_module['Creation date'] = pd.to_datetime(erp_two_module['Creation date']).dt.date
    erp_two_module['Creation date'] = erp_two_module['Creation date'].astype('datetime64[ns]')

    erp_two_module = erp_two_module.rename(columns={'SCC\nRipples': 'SCC Ripples',
                                                    'SCC\nmpyCross': 'SCC mpyCross'})
    erp_two_module = erp_two_module[['UUID', 'Drop App Code', 'Type', 'Hardware Version',
                                     'Creation date', 'SCC Ripples', 'SCC mpyCross',
                                     'Start Module']]
    erp_two_module = erp_two_module.reset_index(drop=True)

    return erp_two_module


def setup_erp_three_module_dataframe(start_module: pd.DataFrame, extension_module: pd.DataFrame):
    start_module = reshape_dataframe(start_module, select_columns=[1, 12], drop_rows=[0],
                                     drop_columns=["Hardware Version & Type",
                                                   "Start Module Pre-Production Status",
                                                   "Assigned to Customer ID",
                                                   "Assigned to Order ID"])
    start_module = convert_column_value_to(start_module,
                                           column="Drop App Code",
                                           fill_none=True, to_type=int)
    start_module = convert_column_value_to(start_module,
                                           column="Hardware Version",
                                           fill_none=True)
    start_module = remove_quotes(start_module, column="SCC\nRipples")
    start_module = remove_quotes(start_module, column="SCC\nmpyCross")

    is_start_module_true = add_is_start_module_column(start_module, True)
    start_module['Start Module'] = is_start_module_true

    extension_module = reshape_dataframe(extension_module, select_columns=[1, 12],
                                         drop_columns=["Hardware Version & Type",
                                                       "Extension Module Pre-Production Status",
                                                       "Assigned to Customer ID",
                                                       "Assigned to Order ID"])
    extension_module = convert_column_value_to(extension_module,
                                               column="Drop App Code",
                                               fill_none=True, to_type=str)
    extension_module = convert_column_value_to(extension_module,
                                               column="Hardware Version",
                                               fill_none=True)
    extension_module = remove_quotes(extension_module, column="SCC\nRipples")
    extension_module = remove_quotes(extension_module, column="SCC\nmpyCross")

    is_start_module_false = add_is_start_module_column(extension_module, False)
    extension_module['Start Module'] = is_start_module_false

    erp_module = pd.concat([start_module, extension_module])
    erp_module = erp_module.reset_index()
    erp_module = reshape_dataframe(dataframe=erp_module, drop_columns=['index'])
    erp_module = convert_column_value_to(erp_module,
                                         column="Drop App Code",
                                         fill_none=True, to_type=str)
    erp_module['SCC\nRipples'] = erp_module['SCC\nRipples'].replace({'5.2.4': '0'})
    erp_module['SCC\nmpyCross'] = erp_module['SCC\nmpyCross'].replace({'1.0.1': '0'})

    erp_module = erp_module.rename(columns={'SCC\nRipples': 'SCC Ripples', 'SCC\nmpyCross': 'SCC mpyCross'})

    return erp_module


def setup_erp_db_dataframe(erp_db_module: pd.DataFrame):
    for index in range(1, 10):
        column = 'UUID Start Module ' + str(index)
        if index == 1:
            column = 'UUID Start Module'

        erp_db_module[column] = \
            erp_db_module[column].replace({'N/A': '0', np.NaN: '0',
                                           'modules reused after fair': '0'})

    for first_index in range(1, 10):
        for second_index in range(1, 10):
            column = 'UUID Extension Module ' + str(second_index) + '.' + str(first_index)

            if first_index == 1:
                column = 'UUID Extension Module ' + str(second_index)

            erp_db_module[column] = erp_db_module[column].replace({'N/A': '0', np.NaN: '0',
                                                                   'modules reused after fair': '0',
                                                                   'Not Found!!\nUpdate corresponding cell AR (OK) to match the formula': '0'
                                                                   })

    return erp_db_module


def setup_sccconfig_dataframe(sccconfig: pd.DataFrame):
    column_value_to_convert = ['SCC mpyCross', 'Hardware Version', 'SCC Ripples', 'Drop App Code', 'J', 'K']
    hardware_version_dict = {'1': 'V1.0',
                             '2': 'V1.3',
                             '3': 'V2.3',
                             '4': '0',
                             '5': '0'}

    scc_mpycross_dict = {'1': '1.9.4',
                         '2': '1.10',
                         '3': '1.11',
                         '4': '1.12',
                         '5': '1.13',
                         '6': '1.14',
                         '7': '1.15',
                         '8': '1.16',
                         '9': '0'}

    scc_ripples_dict = {'1': '0.13.0',
                        '2': '0.14.0',
                        '3': '0.15.0',
                        '4': '0.11.0',
                        '5': '0.12.0',
                        '6': '0.11.1',
                        '7': '0',
                        '8': '0.9.1',
                        '9': '1.0.0',
                        '10': '1.1.0',
                        '11': '1.1.1',
                        '12': '1.2.0'}

    column_dict_to_be_used = [scc_mpycross_dict, hardware_version_dict, scc_ripples_dict]

    for column in column_value_to_convert:
        sccconfig = convert_column_value_to(dataframe=sccconfig,
                                            column=column, fill_none=True, to_type=int)
        sccconfig = convert_column_value_to(dataframe=sccconfig,
                                            column=column, fill_none=True, to_type=str)

    for index in range(3):
        current_column = column_value_to_convert[index]
        current_column_dict = column_dict_to_be_used[index]

        sccconfig[current_column] = sccconfig[current_column].replace(current_column_dict)

    sccconfig['Start Module'] = sccconfig['Start Module'].replace({'t': True, 'f': False})
    sccconfig['Type'] = sccconfig['Type'].replace({'t': 'A.C.', 'f': 'ST.'})
    sccconfig['Creation date'] = pd.to_datetime(sccconfig['Creation date']).dt.date
    sccconfig['Creation date'] = sccconfig['Creation date'].astype('datetime64[ns]')
    sccconfig = sccconfig.drop(['H', 'J', 'K'], axis=1)

    columns_reordered = ['UUID', 'Drop App Code', 'Type', 'Hardware Version',
                         'Creation date', 'SCC Ripples', 'SCC mpyCross', 'Start Module']
    sccconfig = sccconfig[columns_reordered]
    sccconfig = convert_column_value_to(sccconfig,
                                        column="Drop App Code",
                                        fill_none=True, to_type=str)
    return sccconfig


def collect_uuids_from_erp_db(dataframe: pd.DataFrame):
    db_uuid_list = []

    for index in range(1, 10):
        column = 'UUID Start Module ' + str(index)

        if index == 1:
            column = 'UUID Start Module'

        to_append = dataframe[dataframe[column] != '0'][column].values.tolist()
        db_uuid_list = db_uuid_list + to_append

    for first_index in range(1, 10):
        for second_index in range(1, 10):
            column = 'UUID Extension Module ' + str(second_index) + '.' + str(first_index)

            if first_index == 1:
                column = 'UUID Extension Module ' + str(second_index)

            to_append = dataframe[dataframe[column] != '0'][column].values.tolist()
            db_uuid_list = db_uuid_list + to_append

    return db_uuid_list


def generate_code(entry_one: pd.DataFrame, entry_two: pd.DataFrame, column: str):
    code_str = '0'

    special_parameters = ['Type', 'Creation date', 'Start Module']

    entry_one = entry_one[column]
    entry_two = entry_two[column]
    entries_different = entry_one != entry_two
    special_column = column in special_parameters

    entry_one_zero = entry_one == '0'
    entry_two_zero = entry_two == '0'

    if entries_different and not special_column:
        if entry_one_zero:
            code_str = 'a'
        elif entry_two_zero:
            code_str = 'b'
        else:
            code_str = '1'

    if entries_different and special_column:
        if column == 'Creation date':
            entry_one_zero = entry_one is pd.NaT
            entry_two_zero = entry_two is pd.NaT

            if not entry_one_zero and not entry_two_zero:
                entry_with_earliest_date = entry_one < entry_two
                if entry_with_earliest_date:
                    code_str = 'c'
                else:
                    code_str = 'd'

            elif entry_one_zero:
                code_str = 'e'

            elif entry_two_zero:
                code_str = 'f'

            elif entry_one_zero and entry_two_zero:
                code_str = 'n'

        else:
            code_str = '1'

    else:
        if entry_one_zero and entry_two_zero:
            code_str = 'n'

    return code_str


def create_delta_dictionary(dataframe_concatenated: pd.DataFrame):
    unique_uuids = dataframe_concatenated['UUID'].unique()
    columns = ['UUID', 'Drop App Code', 'Type', 'Hardware Version', 'Creation date',
               'SCC Ripples', 'SCC mpyCross', 'Start Module']
    differences_dict = {}

    for uuid in unique_uuids:

        init = True
        code_str = 0

        for column in columns:
            entry_uuid = dataframe_concatenated[dataframe_concatenated['UUID'] == uuid]
            entry_df_one = entry_uuid.iloc[0]
            entry_df_two = entry_uuid.iloc[1]

            if init:
                init = False
                code_str = generate_code(entry_df_one, entry_df_two, column)
                continue

            code_str += generate_code(entry_df_one, entry_df_two, column)

        dict_keys = differences_dict.keys()

        if code_str not in dict_keys:
            differences_dict.update({code_str: []})

        differences_dict[code_str].append(uuid)

    return differences_dict


def decode_code_key(code_key: str):
    columns = ['UUID', 'Drop App Code', 'Type', 'Hardware Version', 'Creation date',
               'SCC Ripples', 'SCC mpyCross', 'Start Module']
    special_parameters = ['Type', 'Creation date', 'Start Module']
    print(f'UUID(s) flagged with code {code_key} indicating:')
    for code, column in zip(code_key, columns):

        if code == 'n':
            print(f'\t Missing ({column}) entries in DataFrames ')
        if code == '1':
            print(f'\t Not matching ({column}) entries between DataFrames')

        if column not in special_parameters:
            if code == 'a':
                print(f'\t Missing ({column}) entry in first DataFrame ')
            if code == 'b':
                print(f'\t Missing ({column}) entry in second DataFrame ')

        else:
            if column == 'Creation date':
                if code == 'c':
                    print(f'\t Earliest ({column}) entry in fist DataFrame')
                if code == 'd':
                    print(f'\t Earliest ({column}) entry in second DataFrame')
                if code == 'e':
                    print(f'\t Missing ({column}) entry in first DataFrame')
                if code == 'f':
                    print(f'\t Missing ({column}) entry in second DataFrame')

    print(f'\n')


def find_uuids_only_in_erp_db(dataframe1: pd.DataFrame, dataframe2: pd.DataFrame, uuids_list: list):
    uuids_only_in_erp_db = []
    for uuid in uuids_list:
        it_exists_in_dataframe1 = dataframe1[dataframe1['UUID'] == uuid]['UUID'].any()
        it_exists_in_dataframe2 = dataframe2[dataframe2['UUID'] == uuid]['UUID'].any()

        if not it_exists_in_dataframe1 and not it_exists_in_dataframe2:
            uuids_only_in_erp_db.append(uuid)
    return uuids_only_in_erp_db


def find_duplicates(dataframe: pd.DataFrame, column: str):
    # print(f'Checking {column_name} duplicates in ERP:')
    # print(f'------------------------------------')

    index = 0
    duplicates_dataframe = 0
    for duplicates_total, column_value in zip(dataframe[column].value_counts(),
                                              dataframe[column].value_counts().index):
        if duplicates_total > 1:
            # print(f'{column_value} has {duplicates_total} duplicates')
            # print(f'Table: \n {dataframe[dataframe[column_name] == column_value]}')
            # print(f'\n------------------------------------')
            if index == 0:
                duplicates_dataframe = dataframe[dataframe[column] == column_value]
            else:
                duplicates_dataframe = pd.concat([duplicates_dataframe,
                                                  dataframe[dataframe[column] == column_value]])
        index += 1
    return duplicates_dataframe


def find_missing_values(dataframe1: pd.DataFrame, dataframe2: pd.DataFrame, column: str):
    missing_values = 0
    init = True
    for value in dataframe1[column]:
        if value not in dataframe2[column].values:
            if init:
                missing_values = dataframe1[dataframe1[column] == value]
                init = False
            missing_values = pd.concat([missing_values,
                                        dataframe1[dataframe1[column] == value]])

    missing_values = missing_values.iloc[1:, :]
    missing_values = missing_values.reset_index()
    missing_values = missing_values.drop(columns='index')
    return missing_values

    # pd.set_option('display.max_rows', missing_values.shape[0] + 1)
    # print(f'{missing_values}\n')
    # print(f'Total: {total}')


def find_common_entries(dataframe1: pd.DataFrame, dataframe2: pd.DataFrame):
    common_entries = dataframe1.merge(dataframe2, how='inner', indicator=False)
    return common_entries

    # pd.set_option('display.max_rows', common_entries.shape[0] + 1)
    # print(f'{common_entries}')


def find_different_entries(dataframe1: pd.DataFrame, dataframe2: pd.DataFrame, column: str):
    merged_dataframe = pd.concat([dataframe1, dataframe2]).drop_duplicates(keep=False)

    concatenated_uuids = 0
    init = True

    try:
        for uuid in merged_dataframe['UUID'].unique():
            if merged_dataframe[merged_dataframe['UUID'] == uuid].shape[0] == 2:
                uuid_in_erp = merged_dataframe[merged_dataframe['UUID'] == uuid].iloc[0]
                uuid_in_sccconfig = merged_dataframe[merged_dataframe['UUID'] == uuid].iloc[1]

                if pd.isnull(uuid_in_erp[column]) or pd.isnull(uuid_in_sccconfig[column]):
                    if init:
                        concatenated_uuids = pd.DataFrame(
                            merged_dataframe[merged_dataframe['UUID'] == uuid])
                        init = False
                    else:
                        concatenated_uuids = pd.concat([concatenated_uuids,
                                                        merged_dataframe[
                                                            merged_dataframe['UUID'] == uuid]])

                elif uuid_in_erp[column] != uuid_in_sccconfig[column]:
                    if init:
                        concatenated_uuids = pd.DataFrame(
                            merged_dataframe[merged_dataframe['UUID'] == uuid])
                        init = False
                    else:
                        concatenated_uuids = pd.concat([concatenated_uuids,
                                                        merged_dataframe[
                                                            merged_dataframe['UUID'] == uuid]])

        concatenated_uuids = concatenated_uuids.reset_index(drop=True)
    except AttributeError:
        concatenated_uuids = f'It appears that both dataframes share the same entries for the column: {column}'

    return concatenated_uuids


def find_start_module_erp_db(erp_db: pd.DataFrame, ext_uuid: str):
    max_length = erp_db.shape[0]
    start_uuid = 0

    for row_index in range(0, max_length):

        for system_index in range(1, 11):
            column_start = f'UUID Start Module {system_index}'
            column_end = f'UUID Extension Module 9.{system_index}'

            if system_index == 1:
                column_start = 'UUID Start Module'
                column_end = 'UUID Extension Module 9'

            row_to_inspect = erp_db.loc[row_index, column_start: column_end]
            elements_to_inspect = row_to_inspect.tolist()

            if ext_uuid in elements_to_inspect:
                start_uuid = row_to_inspect[column_start]

    return start_uuid


def temp_print_key_table(dataframe: pd.DataFrame, delta_dict: dict):
    for key in list(delta_dict.keys()):

        decode_code_key(code_key=key)
        # print(f'UUID(s) flagged with code: {key}\n')
        uuid_list = delta_dict[key]
        init = True
        key_dataframe = 0

        for uuid in uuid_list:
            to_append = dataframe[dataframe['UUID'] == uuid]

            if init:
                key_dataframe = pd.DataFrame(to_append)
                init = False
                continue

            key_dataframe = pd.concat([key_dataframe, to_append])

        try:
            key_dataframe = key_dataframe.reset_index(drop=True)
            pd.set_option('display.max_rows', key_dataframe.shape[0] + 1)
            print(key_dataframe)
            print(
                f'\n---------------------------------------------------------------------------------------------------'
                f'----------- \n')
        except AttributeError:
            print('Error: key_dataframe is empty!')


if __name__ == "__main__":
    start_module_dataframe = pd.read_excel(io="erp/erp_3.1.xlsx", sheet_name="Copy of ST", header=2)
    extension_module_dataframe = pd.read_excel(io="erp/erp_3.1.xlsx", sheet_name="Copy of EXT", header=2)
    erp_three_module_dataframe = setup_erp_three_module_dataframe(start_module=start_module_dataframe,
                                                                  extension_module=extension_module_dataframe)

    erp_db_module_dataframe = pd.read_excel(io="erp/erp_3.1.xlsx", sheet_name='DB', header=1)
    erp_db_module_dataframe = setup_erp_db_dataframe(erp_db_module=erp_db_module_dataframe)

    sccconfig_dataframe = pd.read_csv(filepath_or_buffer="db/sccconfig")
    sccconfig_dataframe = setup_sccconfig_dataframe(sccconfig=sccconfig_dataframe)

    start_module_erp_two = pd.read_excel(io="erp/CX_ERP_V2.xlsx", sheet_name="ST", header=2)
    extension_module_erp_two = pd.read_excel(io="erp/CX_ERP_V2.xlsx", sheet_name="EXT", header=2)
    erp_two_module_dataframe = setup_erp_two_dataframe(start_module=start_module_erp_two,
                                                       extension_module=extension_module_erp_two)

    # pd.set_option('display.max_rows', duplicates_dataframe_one.shape[0] + 1)
    compiled_dataframes = pd.concat([erp_two_module_dataframe,
                                     erp_three_module_dataframe]).drop_duplicates(keep=False)
    different_entries = find_duplicates(dataframe=compiled_dataframes, column='UUID')
    different_entries = different_entries.reset_index(drop=True)

    result = create_delta_dictionary(dataframe_concatenated=different_entries)
    temp_print_key_table(dataframe=compiled_dataframes, delta_dict=result)

    try:
        sys.stdout = open('log/delta_entries_erp_2_and_erp_3_delta_code.txt', 'w')
        pass

    except Exception as e:
        print(f'An error has occurred: {e.__class__.__name__}, {e}')

    finally:
        pass
