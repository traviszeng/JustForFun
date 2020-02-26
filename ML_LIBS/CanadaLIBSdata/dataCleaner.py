
#转换时间戳
def convert_str_datetime(df):
    '''
    AIM    -> Convert datetime(String) to datetime(format we want)

    INPUT  -> df

    OUTPUT -> updated df with new datetime format
    ------
    '''
    df.insert(loc=2, column='timestamp', value=pd.to_datetime(df.transdate, format='%Y-%m-%d %H:%M:%S.%f'))

#删除列中的空格
def remove_col_white_space(df):
    # remove white space at the beginning of string
    df['col'] = df['col'].str.lstrip()

#将两列字符串数据（在一定条件下）拼接起来


def concat_col_str_condition(df):
    # concat 2 columns with strings if the last 3 letters of the first column are 'pil'
    mask = df['col_1'].str.endswith('pil', na=False)
    col_new = df[mask]['col_1'] + df[mask]['col_2']
    col_new.replace('pil', ' ', regex=True, inplace=True)  # replace the 'pil' with emtpy space

"""
删除列中的字符串

有时你可能会看到一行新的字符，或在字符串列中看到一些奇怪的符号。
你可以很容易地使用 df['col_1'].replace 来处理该问题，其中「col_1」是数据帧 df 中的一列。
"""
def remove_col_str(df):
    # remove a portion of string in a dataframe column - col_1
    df['col_1'].replace('\n', '', regex=True, inplace=True)

    # remove all the characters after &# (including &#) for column - col_1
    df['col_1'].replace(' &#.*', '', regex=True, inplace=True)

"""
检查缺失的数据

如果你想要检查每一列中有多少缺失的数据，这可能是最快的方法。
这种方法可以让你更清楚地知道哪些列有更多的缺失数据，帮助你决定接下来在数据清洗和数据分析工作中应该采取怎样的行动。
"""
def check_missing_data(df):
    # check for any missing data in the df (display in descending order)
    return df.isnull().sum().sort_values(ascending=False)

"""
将分类变量转换为数值变量

有一些机器学习模型要求变量是以数值形式存在的。
这时，我们就需要将分类变量转换成数值变量然后再将它们作为模型的输入。
对于数据可视化任务来说，我建议大家保留分类变量，从而让可视化结果有更明确的解释，便于理解。

"""
def convert_cat2num(df):
    # Convert categorical variable to numerical variable
    num_encode = {'col_1' : {'YES':1, 'NO':0},
                  'col_2'  : {'WON':1, 'LOSE':0, 'DRAW':0}}
    df.replace(num_encode, inplace=True)

#转换 Dtypes
def change_dtypes(col_int, col_float, df):
    '''
    AIM    -> Changing dtypes to save memory

    INPUT  -> List of column names (int, float), df

    OUTPUT -> updated df with smaller memory
    ------
    '''
    df[col_int] = df[col_int].astype('int32')
    df[col_float] = df[col_float].astype('float32')

# 删除多列数据
def drop_multiple_col(col_names_list, df):
    '''
    AIM    -> Drop multiple columns based on their column names

    INPUT  -> List of column names, df

    OUTPUT -> updated df with dropped columns
    ------
    '''
    df.drop(col_names_list, axis=1, inplace=True)
    return df