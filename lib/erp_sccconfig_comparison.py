import numpy as np
import pandas as pd
from typing import Union

COLUMNS = [
    'UUID', 'Drop App Code', 'Type', 'Hardware Version',
    'Creation date', 'SCC Ripples', 'SCC mpyCross', 'Start Module'
]

CODE_DESCR = {
    'a': 'Missing ({column}) entry in ERP',
    'b': 'Missing ({column}) entry in SCCConfig',
    'c': 'Earliest ({column}) entry in ERP',
    'd': 'Earliest ({column}) entry in SCCConfig',
    'f': 'Not matching ({column}) entries between both ERP and SCCConfig',
    'n': 'Missing ({column}) entries in both ERP and SCCConfig',
    '1': '({column}) entry only found in ERP',
    '2': '({column}) entry only found in SCCConfig'
}

pd.options.mode.chained_assignment = None


def adjust_column_values(
        dataframe: pd.DataFrame,
        column: str,
        fill_none: bool = False,
        to_type: object = None
) -> pd.DataFrame:
    """
    Converts data type of values in a column, as well as filling 0 in empty space if prompted
    Args:
        dataframe: Dataframe
        column: (String) Column name
        fill_none: (Boolean) Prompt for function to fill N/A
        to_type: (Object) Data type

    Returns:
        dataframe: Dataframe after modification
    """
    if fill_none:
        dataframe[column] = dataframe[column].fillna(0)
    if to_type:
        try:
            dataframe[column] = dataframe[column].astype(to_type)
        except ValueError:
            dataframe[column] = dataframe[column].replace({"#VALUE!": '0'})
    return dataframe


def remove_quotes_na(
        dataframe: pd.DataFrame,
        column: str
) -> pd.DataFrame:
    """
    Removes quotation marks and N/A from dataframe. Quotation marks are replaced
    with empty spaces, whereas N/A are replaced with 0
    Args:
        dataframe: Dataframe
        column: (String) Column name

    Returns:
        dataframe: Dataframe after modification
    """
    dataframe[column] = dataframe[column].str.replace('"', '')
    dataframe[column] = dataframe[column].replace({
        'N/A': '0',
        np.NaN: '0'
    })
    return dataframe


def create_start_module_column(
        dataframe: pd.DataFrame,
        is_start_module: bool
) -> list:
    """
    Appends 'Start Module' column to ERP dataframe to match with Monsoon DB dataframe
    Args:
        dataframe: Dataframe
        is_start_module: (Bool)
            if True: module is a start module
            if False: module is an extension module

    Returns:
        value: (List) is_start_module duplicated to the number of modules in dataframe

    """

    row_length = dataframe.shape[0]
    value = [is_start_module] * row_length

    return value


def setup_erp_two_dataframe(
        start_module: pd.DataFrame,
        extension_module: pd.DataFrame
) -> pd.DataFrame:
    """
    Modifies ERP V2 dataframe to have the same framework as Monsoon DB dataframe. Comments are added
    to each section as a description of their purpose
    Args:
        start_module: (Dataframe) start modules
        extension_module: (Dataframe) extension modules

    Returns:
        erp_two_modules: (Dataframe) all modules from ERP V2
    """
    # Setting up start module sheet (ST)
    start_module = start_module.loc[1:94, :]
    start_module = start_module.rename(
        {'Drop APP Code': 'Drop App Code',
         'Unnamed: 12': 'Hardware Version'
         }
    )

    is_start_module_true = create_start_module_column(
        dataframe=start_module,
        is_start_module=True
    )
    start_module.loc[:, 'Start Module'] = is_start_module_true

    # Setting up extension module sheet (EXT)
    extension_module = extension_module.loc[1:245, :]
    is_start_module_false = create_start_module_column(
        dataframe=extension_module,
        is_start_module=False
    )
    extension_module.loc[:, 'Start Module'] = is_start_module_false

    # Concatenation ERP V2
    erp_two_modules = pd.concat(
        [start_module,
         extension_module
         ]
    )

    # Removing quotation marks
    erp_two_modules = remove_quotes_na(dataframe=erp_two_modules, column='Hardware Version')
    erp_two_modules = remove_quotes_na(dataframe=erp_two_modules, column='SCC\nRipples')
    erp_two_modules = remove_quotes_na(dataframe=erp_two_modules, column='SCC\nmpyCross')

    # Converting Drop ID data type
    erp_two_modules = adjust_column_values(
        dataframe=erp_two_modules,
        column='Drop App Code',
        to_type=int,
        fill_none=True
    )

    erp_two_modules = adjust_column_values(
        dataframe=erp_two_modules,
        column='Drop App Code',
        to_type=str
    )

    # Converting creation date template
    erp_two_modules['Creation date'] = pd.to_datetime(erp_two_modules['Creation date']).dt.date
    erp_two_modules['Creation date'] = erp_two_modules['Creation date'].astype('datetime64[ns]')

    # Final adjustments to ERP V2
    erp_two_modules = erp_two_modules.rename(columns={'SCC\nRipples': 'SCC Ripples',
                                                      'SCC\nmpyCross': 'SCC mpyCross'
                                                      })
    erp_two_modules = erp_two_modules[COLUMNS]
    erp_two_modules = erp_two_modules.reset_index(drop=True)

    return erp_two_modules


def setup_erp_three_module_dataframe(
        start_module: pd.DataFrame,
        extension_module: pd.DataFrame
) -> pd.DataFrame:
    """
    Modifies ERP V3 dataframe to have the same framework as Monsoon DB dataframe. Comments are added
    to each section as a description of their purpose
    Args:
        start_module: (Dataframe) start modules
        extension_module: (Dataframe) extension modules

    Returns:
        erp_three_modules: (Dataframe) all modules from ERP V3
    """
    drop_columns_start = [
        "Hardware Version & Type",
        "Start Module Pre-Production Status",
        "Assigned to Customer ID",
        "Assigned to Order ID"
    ]

    drop_columns_extension = [
        "Hardware Version & Type",
        "Extension Module Pre-Production Status",
        "Assigned to Customer",
        "Assigned to Order ID"
    ]

    # Start module
    # Selecting relevant rows and columns
    start_module = start_module.loc[1:, 'UUID':'SCC\nmpyCross']
    start_module = start_module.drop(drop_columns_start, axis=1)

    # Removing quotation marks and NA
    start_module = remove_quotes_na(start_module, column="SCC\nRipples")
    start_module = remove_quotes_na(start_module, column="SCC\nmpyCross")

    # Append 'Start Module' column
    is_start_module_true = create_start_module_column(start_module, True)
    start_module['Start Module'] = is_start_module_true

    # Remove decimal places from Drop App Code
    start_module = adjust_column_values(
        start_module,
        column="Drop App Code",
        fill_none=True,
        to_type=int
    )

    start_module = adjust_column_values(
        start_module,
        column="Drop App Code",
        fill_none=True,
        to_type=str
    )

    # Extension module
    # Selecting relevant rows and columns
    extension_module = extension_module.loc[:, 'UUID':'SCC\nmpyCross']
    extension_module = extension_module.drop(drop_columns_extension, axis=1)

    # Removing quotation marks and NA
    extension_module = remove_quotes_na(extension_module, column="SCC\nRipples")
    extension_module = remove_quotes_na(extension_module, column="SCC\nmpyCross")

    # Append 'Start Module' column
    is_start_module_false = create_start_module_column(extension_module, False)
    extension_module['Start Module'] = is_start_module_false

    # Concatenation
    # Grouping start modules with extension modules
    erp_three_modules = pd.concat([start_module, extension_module])
    erp_three_modules = erp_three_modules.reset_index()
    erp_three_modules = erp_three_modules.rename(columns={
        'SCC\nRipples': 'SCC Ripples',
        'SCC\nmpyCross': 'SCC mpyCross'
    })
    erp_three_modules = erp_three_modules[COLUMNS]

    # Adjust entries and data types
    erp_three_modules = adjust_column_values(
        erp_three_modules,
        column="Hardware Version",
        fill_none=True
    )

    erp_three_modules = adjust_column_values(
        erp_three_modules,
        column="Drop App Code",
        fill_none=True,
        to_type=int
    )

    erp_three_modules = adjust_column_values(
        erp_three_modules,
        column="Drop App Code",
        fill_none=True,
        to_type=str
    )

    # Replacing specific values
    erp_three_modules['Hardware Version'] = erp_three_modules['Hardware Version'].replace(
        {'V2.3 Pro L': 'V2.3',
         'V2.3 Pro': 'V2.3'
         }
    )

    erp_three_modules['SCC Ripples'] = erp_three_modules['SCC Ripples'].replace(
        {'5.2.4': '0',
         'VW GMD MAX3': '0'
         }
    )

    erp_three_modules['SCC mpyCross'] = erp_three_modules['SCC mpyCross'].replace(
        {'1.0.1': '0',
         'VW GMD MAX3': '0'
         }
    )

    return erp_three_modules


def setup_erp_module_dataframe(
        erp_two: pd.DataFrame,
        erp_three: pd.DataFrame
) -> pd.DataFrame:
    """
    Creates an ERP dataframe with all modules from ERP V2 and ERP V3. Comments are added
    to each section as a description of their purpose
    Args:
        erp_two: (Dataframe) ERP V2
        erp_three: (Dataframe) ERP V3

    Returns:
        erp_dataframe: (Dataframe) Concatenation of ERP V2 and ERP V3
    """
    # Concatenation of dataframes with possible duplicate entries
    concatenated_dataframe = pd.concat(
        [erp_two,
         erp_three]
    ).drop_duplicates(keep=False)

    # Delta dictionary created for filter process
    delta_dict = generate_delta_dictionary(
        concatenated_dataframe=concatenated_dataframe,
        midpoint=erp_two.shape[0])

    # Defining empty dataframe
    erp_dataframe = pd.DataFrame()

    # Filtering
    # Unique modules added directly to erp_dataframe
    unique_uuids = delta_dict['00000000']

    for uuid in unique_uuids:
        erp_dataframe = pd.concat([
            erp_dataframe,
            concatenated_dataframe[concatenated_dataframe['UUID'] == uuid]
        ])

    # Case one: Function takes modules with present Drop ID entry
    delta_key_case_one = ['0a0a0000', '0a0a0100', '0b000000', '0n0a0000']
    for delta_key in delta_key_case_one:
        index = 1
        if delta_key == '0b000000':
            index = 0

        uuids_from_delta_key = delta_dict[delta_key]
        for uuid in uuids_from_delta_key:
            uuid_section = concatenated_dataframe[concatenated_dataframe['UUID'] == uuid]
            to_append = uuid_section.iloc[[index]]
            erp_dataframe = pd.concat([erp_dataframe, to_append], axis=0)

    # Case two: Function fills empty module information on SCC Ripples and SCC mpyCross
    # Note: Comment not completely detailed, needs information as to why it works
    # even without factoring date
    delta_key_case_two = ['0n0a0bb0', '0n0adbb0', '0a0adbb0']
    for delta_key in delta_key_case_two:
        uuids_from_delta_key = delta_dict[delta_key]
        for uuid in uuids_from_delta_key:
            uuid_section = concatenated_dataframe[concatenated_dataframe['UUID'] == uuid]

            replace_with_value = uuid_section.iloc[0, uuid_section.columns.get_loc('SCC Ripples')]
            uuid_section.iloc[1, uuid_section.columns.get_loc('SCC Ripples')] = replace_with_value

            replace_with_value = uuid_section.iloc[0, uuid_section.columns.get_loc('SCC mpyCross')]
            uuid_section.iloc[1, uuid_section.columns.get_loc('SCC mpyCross')] = replace_with_value

            to_append = uuid_section.iloc[[1]]
            erp_dataframe = pd.concat([erp_dataframe, to_append], axis=0)

    pd.set_option('display.max_rows', erp_dataframe.shape[0] + 1)
    erp_dataframe = erp_dataframe.reset_index(drop=True)

    return erp_dataframe


def setup_erp_db_dataframe(erp_db: pd.DataFrame) -> pd.DataFrame:
    """
    Converts unusable entries to 0 from the ERP 'DB' sheet
    Args:
        erp_db: (Dataframe) Entries from ERP 'DB' sheet

    Returns:
        erp_db: (Dataframe) Modified entries from ERP 'DB' sheet

    """
    for index in range(1, 10):
        column = 'UUID Start Module ' + str(index)
        if index == 1:
            column = 'UUID Start Module'

        erp_db[column] = erp_db[column].replace(
            {'N/A': '0',
             np.NaN: '0',
             'modules reused after fair': '0'
             }
        )

    for first_index in range(1, 10):
        for second_index in range(1, 10):
            column = 'UUID Extension Module ' + str(second_index) + '.' + str(first_index)

            if first_index == 1:
                column = 'UUID Extension Module ' + str(second_index)

            erp_db[column] = \
                erp_db[column].replace({
                    'N/A': '0', np.NaN: '0',
                    'modules reused after fair': '0',
                    'Not Found!!\nUpdate corresponding cell AR (OK) to match the formula': '0'
                })

    return erp_db


def setup_sccconfig_dataframe(sccconfig: pd.DataFrame) -> pd.DataFrame:
    """
    Configures entries from SCC config by converting them to values
    that corresponds to entries in the ERP
    Args:
        sccconfig: (Dataframe) Entries from SCC config

    Returns:
        sccconfig: (Dataframe) Modified entries from SCC config
    """
    scc_hardware_version_dict = {'1': 'V1.0',
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

    column_values_to_convert = ['SCC mpyCross',
                                'Hardware Version',
                                'SCC Ripples',
                                'Drop App Code']

    column_dict_reference = {'SCC mpyCross': scc_mpycross_dict,
                             'Hardware Version': scc_hardware_version_dict,
                             'SCC Ripples': scc_ripples_dict}

    for column in column_values_to_convert:
        sccconfig = adjust_column_values(
            dataframe=sccconfig,
            column=column,
            fill_none=True,
            to_type=int)

        sccconfig = adjust_column_values(
            dataframe=sccconfig,
            column=column,
            fill_none=True,
            to_type=str)

        if column != 'Drop App Code':
            current_column_dict = column_dict_reference[column]
            sccconfig[column] = sccconfig[column].replace(current_column_dict)

    sccconfig['Start Module'] = sccconfig['Start Module'].replace({
        't': True,
        'f': False}
    )
    sccconfig['Type'] = sccconfig['Type'].replace({
        't': 'A.C.',
        'f': 'ST.'}
    )
    sccconfig['Creation date'] = pd.to_datetime(sccconfig['Creation date']).dt.date
    sccconfig['Creation date'] = sccconfig['Creation date'].astype('datetime64[ns]')

    sccconfig = sccconfig[COLUMNS]
    sccconfig = adjust_column_values(
        sccconfig,
        column="Drop App Code",
        fill_none=True,
        to_type=str
    )

    return sccconfig


def collect_uuids_from_erp_db(erp_db: pd.DataFrame) -> list:
    """
    Collects all the UUIDs from the ERP 'DB' sheet
    Args:
        erp_db: (Dataframe) Entries from ERP 'DB' sheet

    Returns:
        erp_db: (Dataframe) Modified entries from ERP 'DB' sheet
    """
    db_uuid_list = []

    for index in range(1, 10):
        column = 'UUID Start Module ' + str(index)

        if index == 1:
            column = 'UUID Start Module'

        to_append = erp_db[erp_db[column] != '0'][column].values.tolist()
        db_uuid_list = db_uuid_list + to_append

    for first_index in range(1, 10):
        for second_index in range(1, 10):
            column = 'UUID Extension Module ' + str(second_index) + '.' + str(first_index)

            if first_index == 1:
                column = 'UUID Extension Module ' + str(second_index)

            to_append = erp_db[erp_db[column] != '0'][column].values.tolist()
            db_uuid_list = db_uuid_list + to_append

    return db_uuid_list


def generate_code(
        erp: Union[pd.DataFrame, None],
        sccconfig: Union[pd.DataFrame, None],
        column: str,
        midpoint: int,
        is_duplicate: bool = False
) -> str:
    """
    Generates the delta code by taking each property for each dataframe and compares
    the entries of a dataframe with the other. At the end of the function, a delta code
    is generated

    Important information:
        - A fully completed delta code should contain 8 characters,
          with each character representing the 8 properties of a module
          in the following order:

            1. 'UUID'
            2. 'Drop App Code'
            3. 'Type'
            4. 'Hardware Version'
            5. 'Creation date'
            6. 'SCC Ripples'
            7. 'SCC mpyCross'
            8. 'Start Module'

        - Possible codes:

            'a': 'Missing ({column}) entry in ERP',
            'b': 'Missing ({column}) entry in SCCConfig',
            'c': 'Earliest ({column}) entry in ERP',
            'd': 'Earliest ({column}) entry in SCCConfig',
            'f': 'Not matching ({column}) entries between both ERP and SCCConfig',
            'n': 'Missing ({column}) entries in both ERP and SCCConfig',
            '1': '({column}) entry only found in ERP',
            '2': '({column}) entry only found in SCCConfig'

    Args:
        erp: (Dataframe) Entries from ERP
        sccconfig: (Dataframe) Entries from SCC config
        column: (String) Column name
        midpoint: (Integer) Index where ERP and SCC are joined
        is_duplicate: (Boolean) Indicates if erp == sccconfig. It
                                assumes that erp and sccconfig are not
                                the same
    Returns:
        delta_code: (String) one possible code
    """

    erp_value = erp[column]
    sccconfig_value = sccconfig[column]

    entry_one_null = erp_value == '0' or erp_value is pd.NaT
    entry_two_null = sccconfig_value == '0' or sccconfig_value is pd.NaT

    if entry_one_null and entry_two_null:
        return 'n'

    if is_duplicate:
        if erp.name < midpoint:
            return '1'
        elif sccconfig.name >= midpoint:
            return '2'

    if entry_one_null:
        return 'a'
    if entry_two_null:
        return 'b'
    if erp_value == sccconfig_value:
        return '0'

    delta_code = 'f'
    # If both entries exist and are not the same, there is a discrepancy

    # Potentially specify discrepancy further
    if column == 'Creation date':
        entry_with_earliest_date = erp_value < sccconfig_value
        if entry_with_earliest_date:
            delta_code = 'c'
        else:
            delta_code = 'd'

    return delta_code


def generate_delta_dictionary(
        concatenated_dataframe: pd.DataFrame,
        midpoint: int
) -> dict:
    """
    Generates a dictionary containing delta codes, a code system for indicating
    discrepancies of entries from a module (UUID) found in both the ERP and
    SCC config.


    Args:
        concatenated_dataframe: (Dataframe) ERP and SCC config grouped
        midpoint: (Integer) Index where ERP and SCC are joined
    Returns:
        delta_codes_dict: (Dictionary) Collection of delta codes, along with
                          the corresponding UUIDs
            - Key: Delta code
            - Value: UUIDs
    """
    unique_uuids = concatenated_dataframe['UUID'].unique()
    total_duplicates = concatenated_dataframe['UUID'].value_counts()
    delta_codes_dict = {}

    for uuid in unique_uuids:
        delta_code = ""

        entry_uuid = concatenated_dataframe[concatenated_dataframe['UUID'] == uuid]
        entry_erp = entry_uuid.iloc[0]
        entry_sccconfig = entry_erp
        is_duplicate = True

        if total_duplicates[uuid] > 1:
            entry_sccconfig = entry_uuid.iloc[1]
            is_duplicate = False

        for column in COLUMNS:
            delta_code += generate_code(
                entry_erp,
                entry_sccconfig,
                column,
                midpoint,
                is_duplicate
            )

        dict_keys = delta_codes_dict.keys()

        if delta_code not in dict_keys:
            delta_codes_dict.update({delta_code: []})

        delta_codes_dict[delta_code].append(uuid)

    return delta_codes_dict


def decode_code_key(delta_code: str) -> str:
    """
    Converts a delta code to a human-readable form.
    Args:
        delta_code: (String) Delta code

    Returns:
        string: Message on the delta code meaning
    """
    delta_code_parts = [f'UUID(s) flagged with delta code {delta_code} indicating:']

    if delta_code == '00000000':
        delta_code_parts.append('\tEntries from both ERP and SCCConfig match')

    for code, column in zip(delta_code, COLUMNS):
        try:
            delta_code_parts.append("\t" + CODE_DESCR[code].format(column=column))
        except KeyError:
            pass

    delta_code_parts.append("")

    return "\n".join(delta_code_parts)


def number_of_unique_uuids(
        delta_code_dict_key: str,
        delta_code_dict: dict
) -> str:
    """
    Returns the number of unique UUIDs found from a delta code
    Args:
        delta_code_dict_key: (String) Delta code
        delta_code_dict: (Dictionary) Collection of delta codes

    Returns:
        total_str: (String) Sentence expressing the total number of unique
                   UUIDs
    """
    total_int = len(delta_code_dict[delta_code_dict_key])
    total_str = f'Number of unique UUID(s) found: {total_int}\n\n'
    return total_str


def find_uuids_only_in_erp_db(
        dataframe1: pd.DataFrame,
        dataframe2: pd.DataFrame,
        uuids_list: list
) -> list:
    uuids_only_in_erp_db = []
    for uuid in uuids_list:
        it_exists_in_dataframe1 = dataframe1[dataframe1['UUID'] == uuid]['UUID'].any()
        it_exists_in_dataframe2 = dataframe2[dataframe2['UUID'] == uuid]['UUID'].any()

        if not it_exists_in_dataframe1 and not it_exists_in_dataframe2:
            uuids_only_in_erp_db.append(uuid)
    return uuids_only_in_erp_db


def find_duplicates(
        dataframe: pd.DataFrame,
        column: str
) -> pd.DataFrame:
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


def find_missing_values(
        dataframe1: pd.DataFrame,
        dataframe2: pd.DataFrame,
        column: str
) -> pd.DataFrame:
    missing_values_dataframe = pd.DataFrame()

    for value in dataframe1[column]:
        if value not in dataframe2[column].values:
            missing_values_dataframe = pd.concat([
                missing_values_dataframe,
                dataframe1[dataframe1[column] == value]
            ])

    missing_values_dataframe = missing_values_dataframe.iloc[1:, :]
    missing_values_dataframe = missing_values_dataframe.reset_index()
    missing_values_dataframe = missing_values_dataframe.drop(columns='index')

    return missing_values_dataframe

    # pd.set_option('display.max_rows', missing_values_dataframe.shape[0] + 1)
    # print(f'{missing_values_dataframe}\n')
    # print(f'Total: {total}')


def find_common_entries(
        dataframe1: pd.DataFrame,
        dataframe2: pd.DataFrame
) -> pd.DataFrame:
    common_entries_dataframe = dataframe1.merge(
        dataframe2,
        how='inner',
        indicator=False
    )

    return common_entries_dataframe

    # pd.set_option('display.max_rows', common_entries_dataframe.shape[0] + 1)
    # print(f'{common_entries_dataframe}')


def find_different_entries(
        dataframe1: pd.DataFrame,
        dataframe2: pd.DataFrame,
        column: str
) -> Union[pd.DataFrame, str]:
    concatenated_dataframe = pd.concat([dataframe1, dataframe2]).drop_duplicates(keep=False)
    concatenated_uuids = pd.DataFrame()

    try:
        for uuid in concatenated_dataframe['UUID'].unique():
            if concatenated_dataframe[concatenated_dataframe['UUID'] == uuid].shape[0] == 2:
                uuid_in_erp = concatenated_dataframe[concatenated_dataframe['UUID'] == uuid].iloc[0]
                uuid_in_sccconfig = concatenated_dataframe[concatenated_dataframe['UUID'] == uuid].iloc[1]

                if pd.isnull(uuid_in_erp[column]) or pd.isnull(uuid_in_sccconfig[column]):
                    concatenated_uuids = pd.concat([
                        concatenated_uuids,
                        concatenated_dataframe[concatenated_dataframe['UUID'] == uuid]
                    ])

                elif uuid_in_erp[column] != uuid_in_sccconfig[column]:
                    concatenated_uuids = pd.concat([
                        concatenated_uuids,
                        concatenated_dataframe[concatenated_dataframe['UUID'] == uuid]
                    ])

        concatenated_uuids = concatenated_uuids.reset_index(drop=True)

    except AttributeError:
        concatenated_uuids = f'It appears that both dataframes share the same entries for the column: {column}'

    return concatenated_uuids


def find_start_module_erp_db(
        erp_db: pd.DataFrame,
        ext_uuid: str
) -> str:
    max_length = erp_db.shape[0]
    start_uuid = ''

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
                break

    return start_uuid


def format_delta_table(
        dataframe: pd.DataFrame,
        delta_code_dict: dict
) -> str:
    """
    Generates a human-readable message
    Args:
        dataframe: (Dataframe) Entries of a database
        delta_code_dict: (Dict) Delta code dictionary corresponding
                         to the database

    Returns:
        string: Message
    """
    formatted_string_parts = []
    for key in list(delta_code_dict.keys()):
        formatted_string_parts.append(
            decode_code_key(delta_code=key)
        )
        formatted_string_parts.append(
            number_of_unique_uuids(
                delta_code_dict_key=key,
                delta_code_dict=delta_code_dict
            )
        )
        uuid_list = delta_code_dict[key]
        key_dataframe = pd.DataFrame()

        for uuid in uuid_list:
            to_append = dataframe[dataframe['UUID'] == uuid]
            key_dataframe = pd.concat([key_dataframe, to_append])

        try:
            key_dataframe = key_dataframe.reset_index(drop=True)
            pd.set_option('display.max_rows', key_dataframe.shape[0] + 1)
            formatted_string_parts.append(key_dataframe.to_string())
            formatted_string_parts.append(
                '\n-----------------------------------------------------------'
                '---------------------------------------------------\n'
            )
        except AttributeError:
            formatted_string_parts.append('Error: key_dataframe is empty!')

    return "\n".join(formatted_string_parts)


def format_delta_table_excel(
        concatenated_dataframe: pd.DataFrame,
        delta_code_dict: dict,
        file_path: str
):
    for delta_code, uuids in delta_code_dict.items():
        curr_dataframe = pd.DataFrame()
        try:
            for uuid in uuids:
                to_append = concatenated_dataframe[concatenated_dataframe['UUID'] == uuid]
                curr_dataframe = pd.concat(
                    [curr_dataframe,
                     to_append],
                    ignore_index=True)

            with pd.ExcelWriter(
                    file_path,
                    mode='a',
                    if_sheet_exists='replace',
                    engine='openpyxl'
            ) as writer:

                curr_dataframe.style.set_properties(**{'text-align': 'center'}).to_excel(
                    writer,
                    sheet_name=delta_code,
                    index=False
                )

                ws = writer.sheets[delta_code]
                dims = {}
                for row in ws.rows:
                    for cell in row:
                        if cell.value:
                            dims[cell.column_letter] = max((dims.get(cell.column_letter, 0), len(str(cell.value))))
                for col, value in dims.items():
                    ws.column_dimensions[col].width = value + 10

        except FileNotFoundError:
            with pd.ExcelWriter(
                    file_path,
                    mode='w',
                    engine='openpyxl'
            ) as writer:

                curr_dataframe.style.set_properties(**{'text-align': 'center'}).to_excel(
                    writer,
                    sheet_name=delta_code,
                    index=False
                )

                ws = writer.sheets[delta_code]
                dims = {}
                for row in ws.rows:
                    for cell in row:
                        if cell.value:
                            dims[cell.column_letter] = max((dims.get(cell.column_letter, 0), len(str(cell.value))))
                for col, value in dims.items():
                    ws.column_dimensions[col].width = value + 10


def setup_filtered_list(concatenated_dataframe: pd.DataFrame, delta_dict: dict,
                        take_first_value: bool = False) -> pd.DataFrame:
    result = 0
    delta_dict_keys = list(delta_dict.keys())
    init = True

    for delta_key in delta_dict_keys:
        uuid_sample_list = delta_dict[delta_key]

        for uuid_sample in uuid_sample_list:
            dataframe_section = concatenated_dataframe[concatenated_dataframe['UUID'] == uuid_sample]
            index = 1

            if dataframe_section.shape[0] == 1:
                index = 0

            to_append_row = dataframe_section.iloc[[index]]

            for delta, column in zip(delta_key, COLUMNS):
                column_index = dataframe_section.columns.get_loc(column)

                if delta == 'a' or delta == 'd' or delta == 'n' or delta == '0':
                    continue

                if delta == 'b' or delta == 'c':
                    value_to_add = dataframe_section.iloc[0, column_index]
                    to_append_row[column] = value_to_add

                if delta == 'f':
                    first_value = dataframe_section.iloc[0, column_index]

                    if take_first_value and column != 'SCC Ripples':
                        value_to_add = first_value
                        to_append_row[column] = value_to_add

            if init:
                init = False
                result = pd.DataFrame(to_append_row)
            else:
                result = pd.concat([result, to_append_row])

    result = result.sort_values(by='Creation date')
    result = result.reset_index(drop=True)

    return result


def parse_delta_dict(
        concatenated_dataframe: pd.DataFrame,
        delta_dict_prev: dict
) -> pd.DataFrame:
    delta_dict_prev_keys = delta_dict_prev.keys()

    for delta_code in delta_dict_prev_keys:
        delta_code_uuid = delta_dict_prev[delta_code]
        if 'c' in delta_code:
            for uuid in delta_code_uuid:
                row_indexes = concatenated_dataframe.index[concatenated_dataframe['UUID'] == uuid].tolist()
                earliest_date = concatenated_dataframe.loc[row_indexes[0], 'Creation date']
                concatenated_dataframe.loc[row_indexes[1], 'Creation date'] = earliest_date
        if 'd' in delta_code:
            for uuid in delta_code_uuid:
                row_indexes = concatenated_dataframe.index[concatenated_dataframe['UUID'] == uuid].tolist()
                earliest_date = concatenated_dataframe.loc[row_indexes[1], 'Creation date']
                concatenated_dataframe.loc[row_indexes[0], 'Creation date'] = earliest_date

    return concatenated_dataframe


def get_delta_code_group(column: str, delta_dict: dict) -> list:
    delta_dict_codes = delta_dict.keys()
    delta_code_index = COLUMNS.index(column)
    result_delta_codes = []

    for delta_code in delta_dict_codes:
        if delta_code[delta_code_index] != '0':
            print(delta_code[delta_code_index])
            result_delta_codes.append(delta_code)

    return result_delta_codes
